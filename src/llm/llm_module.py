# Local LLM + RAG integration for DialogueManager (no external API calls)
# Requires: pydantic, sentence-transformers, faiss-cpu (or faiss-gpu), and either
#   - llama-cpp-python + a ggml model file  OR
#   - a locally-mounted HF chat model compatible with transformers
#
# This file provides:
#  - Action Pydantic schema
#  - LocalMemory (embeddings + FAISS)
#  - LocalLLM wrapper (pluggable backend)
#  - DialogueManagerLLM.process replacement helpers to integrate in your repo

from typing import Optional, Dict, Any, List
import json
import time
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError
from pathlib import Path

from dialogue_manager import (
    _format_task_list,
    _format_complete_prompt,
    _task_completed_response,
)

# Optional imports (guarded)
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except Exception:
    SentenceTransformer = None
    faiss = None

# Try llama-cpp-python backend first (ggml), fallback to transformers
try:
    from llama_cpp import Llama
    _HAS_LLAMA_CPP = True
except Exception:
    _HAS_LLAMA_CPP = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


# ----------------------------
# Action schema (pydantic)
# ----------------------------
class ActionModel(BaseModel):
    action: str = Field(..., description="One of add_task, list_tasks, complete_task, delete_task, ask_user, summarize, noop")
    slots: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    explanation: Optional[str] = None

    def is_destructive(self) -> bool:
        return self.action in {"delete_task"}


# ----------------------------
# Local memory (RAG) helper
# ----------------------------
class LocalMemory:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None or faiss is None:
            raise RuntimeError("Install sentence-transformers and faiss to use LocalMemory.")
        self._emb = SentenceTransformer(model_name)
        self._index = None
        self._texts = []
        self._dim = self._emb.get_sentence_embedding_dimension()

    def add(self, text: str):
        vec = self._emb.encode([text])
        if self._index is None:
            self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(vec)
        self._texts.append(text)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        if self._index is None or len(self._texts) == 0:
            return []
        qv = self._emb.encode([query])
        scores, ids = self._index.search(qv, min(k, len(self._texts)))
        out = []
        for idx in ids[0]:
            if idx < len(self._texts):
                out.append(self._texts[idx])
        return out


# ----------------------------
# Local LLM wrapper
# ----------------------------
class LocalLLM:
    def __init__(self, backend: str = "llama", model_path: Optional[str] = None, device: str = "cpu"):
        self.backend = backend
        self.device = device
        self.model_path = model_path

        if backend == "llama":
            if not _HAS_LLAMA_CPP:
                raise RuntimeError("llama-cpp-python not available. Install it or use backend='transformers'.")
            if not model_path or not Path(model_path).exists():
                raise RuntimeError("Provide a valid ggml model file path for llama backend.")
            self.client = Llama(model_path=model_path, n_ctx=2048)
        elif backend == "transformers":
            if not _HAS_TRANSFORMERS:
                raise RuntimeError("transformers not available. Install it or use backend='llama'.")
            if not model_path:
                raise RuntimeError("Provide a local HF model name/path for transformers backend.")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
            self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0 if device == "cuda" else -1)
        else:
            raise ValueError("Unsupported backend")

    def generate_action(self, system_prompt: str, user_prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        prompt = system_prompt + "\n\n" + user_prompt
        if self.backend == "llama":
            res = self.client.create(prompt=prompt, max_tokens=max_tokens, temperature=temperature, stop=["\n\n"])
            # llama-cpp returns text in res["choices"][0]["text"]
            return res["choices"][0]["text"].strip()
        else:
            out = self.pipeline(prompt, max_new_tokens=max_tokens, do_sample=(temperature>0.0), temperature=temperature)[0]["generated_text"]
            # remove the input prompt (very simple strategy)
            return out[len(prompt):].strip()


# ----------------------------
# Prompt template helpers
# ----------------------------
SYSTEM_INSTRUCTIONS = """
You are Aerial, a personal task assistant running locally. You must output exactly one JSON object (and nothing else).
Allowed actions: add_task, list_tasks, complete_task, delete_task, ask_user, summarize, noop.

Action JSON schema:
{
  "action": "<action-name>",
  "slots": { ... },
  "confidence": 0.0-1.0,
  "explanation": "brief text why you chose this"
}

Rules:
- If you are uncertain about required slots, return action 'ask_user' with slots: {"prompt": "<clarifying question>"}.
- For delete_task, do NOT set confirmation; instead return ask_user to confirm, or set slots.confirm=true only if user explicitly asked "delete X, yes".
- Date format must be YYYY-MM-DD when known.
- Keep confidence low (<0.7) if you're not sure.
- output must be valid JSON parsable with utf-8.
"""

def build_user_prompt(user_text: str, history: List[Dict[str,str]], memories: List[str]) -> str:
    # Keep history short and include retrieved memory
    last_turns = "\n".join(f"{h['role']}: {h['text']}" for h in history[-8:])
    mem_text = "\n".join(f"- {m}" for m in memories) if memories else ""
    return (
        f"Conversation:\n{last_turns}\n\n"
        f"Relevant memory:\n{mem_text}\n\n"
        f"User: {user_text}\n\n"
        f"Return the action JSON now."
    )


# ----------------------------
# Integrate into DialogueManager
# ----------------------------
class DialogueManagerLLM:
    def __init__(self, task_fns: dict, llm: LocalLLM, memory: Optional[LocalMemory] = None, user_name: str = ""):
        self.fns = task_fns
        self.llm = llm
        self.memory = memory
        self.state = None  # expected to be an instance of your DialogueState (imported externally)
        self.user_name = user_name

    def set_state(self, state):
        self.state = state

    def _validate_action(self, raw_text: str) -> Optional[ActionModel]:
        try:
            parsed = json.loads(raw_text)
        except Exception:
            # maybe the model appended text: try to extract a JSON substring
            s = raw_text.find("{")
            e = raw_text.rfind("}")
            if s == -1 or e == -1:
                return None
            try:
                parsed = json.loads(raw_text[s:e+1])
            except Exception:
                return None
        try:
            act = ActionModel(**parsed)
            return act
        except ValidationError:
            return None

    def _safe_execute(self, act: ActionModel, user_text: str) -> str:
        # safety guard: require explicit confirmation for deletions
        if act.action == "delete_task":
            if not act.slots.get("confirm"):
                # ask to confirm
                self.state.awaiting_slot = "confirm_delete"
                # stash task_id/text into slots for later
                self.state.slots["task_id"] = act.slots.get("task_id")
                return f"Just to confirm — delete task {act.slots.get('task_id') or act.slots.get('task_title_fragment')}? (yes / no)"
        # map actions to functions
        if act.action == "add_task":
            title = act.slots.get("title", "Untitled task")
            due   = act.slots.get("due_date")
            cat   = act.slots.get("category", "personal")
            if self.fns.get("normalize_date") and due:
                try:
                    due = self.fns["normalize_date"](due)
                except Exception:
                    pass
            self.fns["add_task"](title, cat, due)
            # bookkeeping similar to your original code
            if self.fns.get("update_memory_from_tasks"):
                self.fns["update_memory_from_tasks"]()
            self.state.last_action = f"added:{title}"
            self.state.last_action_time = datetime.now()
            return f'Added "{title}"' + (f" due {due}." if due else ".")
        if act.action == "list_tasks":
            tasks = self.fns.get("list_tasks", lambda: [])()
            return _format_task_list(tasks)
        if act.action == "complete_task":
            tasks = self.fns.get("list_tasks", lambda: [])()
            task_ref = act.slots.get("task_id") or act.slots.get("task_title_fragment")
            task_id = self._resolve_task_id(task_ref, tasks)
            if task_id is None:
                # ask which
                self.state.awaiting_slot = "task_id"
                return _format_complete_prompt(tasks, action="complete")
            self.fns["complete_task"](int(task_id))
            return self._task_completed_response(int(task_id), tasks)
        if act.action == "ask_user":
            return act.slots.get("prompt", "Can you clarify?")
        if act.action == "noop":
            return act.explanation or "Okay."
        if act.action == "summarize":
            scope = act.slots.get("scope", "today")
            gen = self.fns.get("generate_briefing")
            if gen:
                return gen(self.fns.get("get_recommended_tasks", lambda: [])(), self.fns.get("get_task_score"))
            return "No briefing available."
        return "I couldn't map that action to a function."

    # Reuse helpers from your old manager (adapt/inline these if needed):
    # for brevity, expect _format_task_list, _format_complete_prompt, _task_completed_response,
    # and _resolve_task_id are imported from your existing module or reimplemented here.

    def process(self, user_input: str) -> str:
        # build retrieval context
        memories = []
        if self.memory:
            memories = self.memory.retrieve(user_input, k=3)
        history = self.state.history if self.state else []
        user_prompt = build_user_prompt(user_input, history, memories)
        raw = self.llm.generate_action(SYSTEM_INSTRUCTIONS, user_prompt, max_tokens=512, temperature=0.0)
        act = self._validate_action(raw)
        if not act:
            # model failed to produce valid JSON, fallback to ask_user
            return "Sorry — I couldn't understand that. Can you rephrase?"
        # safety: if confidence low, ask user
        if act.confidence < 0.7 and act.action not in {"ask_user", "noop", "list_tasks"}:
            # ask for clarification
            self.state.awaiting_slot = "confirm_low_confidence"
            self.state.slots["proposed_action"] = act.dict()
            return f"I'm a bit unsure about this: {act.explanation or act.action}. Do you want me to {act.action.replace('_',' ')}? (yes / no)"
        # execute action (with internal safety checks)
        return self._safe_execute(act, user_input)