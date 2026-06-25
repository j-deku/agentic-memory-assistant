# ollama_responder.py

import json
import requests

class OllamaResponder:
    """
    Drop-in natural language layer for your task assistant.
    It does NOT control logic — only converts structured data → human response.
    """

    def __init__(self, model="llama3"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, data: dict, style: str = "siri") -> str:
        """
        data: structured output from your DialogueManager
        style: response personality mode
        """

        prompt = self._build_prompt(data, style)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        try:
            res = requests.post(self.url, json=payload, timeout=10)
            res.raise_for_status()
            return res.json()["response"].strip()

        except Exception:
            # SAFE fallback (never break your system)
            return self._fallback(data)

    def _build_prompt(self, data: dict, style: str) -> str:
        return f"""
You are a highly intelligent voice assistant like Siri.

Your job:
Convert structured task system output into a natural spoken response.

RULES:
- Do NOT change meaning
- Do NOT add new information
- Keep it short (1–2 sentences)
- Be friendly and natural
- Do NOT mention JSON or structure

STYLE: {style}

STRUCTURED DATA:
{json.dumps(data, indent=2)}

OUTPUT:
"""

    def _fallback(self, data: dict) -> str:
        intent = data.get("intent")

        if intent == "add_task":
            return f"Got it — I’ve added '{data.get('title', 'your task')}'."

        if intent == "complete_task":
            return "Nice work — task marked as complete."

        if intent == "view_tasks":
            return "Here are your current tasks."

        return "Done."