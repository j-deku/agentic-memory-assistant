"""
conversation_ai.py  —  Aerial's conversational brain (v2)

Improvements over v1:
  - Richer name-recognition patterns ("how do they call me", "what do people call me")
  - Temporal awareness: "when did you do X?" handled gracefully
  - View-tasks patterns expanded ("show my full tasks", "full list", "all tasks", etc.)
  - Smarter fallback: offers specific next steps based on context
  - Tone detection extended with sarcasm heuristic
  - Pending-action state cleared more reliably
  - _handle_delete_query handles "when did you delete" without re-triggering deletion
  - Duplicate handler removed (was causing double-response edge cases)
"""

from dataclasses import dataclass, field
import re
import random
from datetime import datetime
from typing import Optional, Callable


# ─────────────────────────────────────────────
# CONTEXT BUILDER
# ─────────────────────────────────────────────

@dataclass
class ConversationContext:
    last_topic: str = ""
    last_question: str = ""
    last_referenced_task: str = ""
    pending_action: str = ""
    pending_entity: str = ""
    awaiting_followup: str = ""
    followup_data: dict = field(default_factory=dict)
    # New: track recent completed/deleted actions with timestamps
    recent_actions: list = field(default_factory=list)   # [{action, title, time}]

    def record_action(self, action: str, title: str):
        self.recent_actions.append({
            "action": action,
            "title": title,
            "time": datetime.now(),
        })
        # Keep only the last 10
        self.recent_actions = self.recent_actions[-10:]

    def last_action_of(self, action_type: str) -> Optional[dict]:
        for entry in reversed(self.recent_actions):
            if entry["action"] == action_type:
                return entry
        return None


def _build_context(get_memory_fn, list_tasks_fn) -> dict:
    ctx = {
        "most_used_category": "personal",
        "completion_rate":    "unknown",
        "total_tasks":        0,
        "pending_count":      0,
        "overdue_count":      0,
        "top_tasks":          [],
        "due_today":          [],
    }

    if get_memory_fn:
        try:
            ctx["most_used_category"] = get_memory_fn("most_used_category") or "personal"
            ctx["completion_rate"]    = get_memory_fn("completion_rate") or "unknown"
            ctx["total_tasks"]        = int(get_memory_fn("total_tasks") or 0)
        except Exception:
            pass

    if list_tasks_fn:
        try:
            tasks   = list_tasks_fn()
            today   = datetime.now().date()
            pending = []
            overdue = 0
            due_today = []
            titles  = []

            for t in tasks:
                done    = t[4] if isinstance(t, tuple) else t.get("completed", 0)
                title   = t[1] if isinstance(t, tuple) else t.get("title", "")
                due_str = t[3] if isinstance(t, tuple) else t.get("due_date")

                if done:
                    continue

                pending.append(t)
                if title:
                    titles.append(title)

                if due_str:
                    try:
                        from datetime import datetime as dt
                        due = dt.strptime(due_str, "%Y-%m-%d").date()
                        if due < today:
                            overdue += 1
                        elif due == today:
                            due_today.append(title)
                    except ValueError:
                        pass

            ctx["pending_count"] = len(pending)
            ctx["overdue_count"] = overdue
            ctx["top_tasks"]     = titles[:5]
            ctx["due_today"]     = due_today[:3]
        except Exception:
            pass

    return ctx


# ─────────────────────────────────────────────
# TONE DETECTOR
# ─────────────────────────────────────────────

def _detect_tone(text: str) -> str:
    t = text.lower()

    if re.search(r"\b(ugh|argh|damn|dammit|frustrated|annoying|hate|worst|"
                 r"stupid|useless|terrible|awful)\b", t):
        return "frustrated"

    if re.search(r"\b(confused|confusing|don'?t understand|makes no sense|"
                 r"what\?+|huh|lost|unclear)\b", t):
        return "confused"

    if re.search(r"\b(amazing|awesome|excited|can'?t wait|love it|great news|"
                 r"fantastic|wonderful|so happy)\b", t):
        return "excited"

    if re.search(r"\b(sad|tired|exhausted|stressed|overwhelmed|anxious|worried|"
                 r"depressed|burnt out|can'?t do this)\b", t):
        return "sad"

    if re.search(r"\b(please|could you|would you|kindly|i would like|"
                 r"i was wondering)\b", t):
        return "formal"

    # Sarcasm heuristic: short sentence ending in "..." or heavy punctuation with a positive word
    if re.search(r"\b(oh great|wow thanks|sure thing|right\.\.\.|yeah right)\b", t):
        return "sarcastic"

    return "casual"


# ─────────────────────────────────────────────
# RESPONSE ENGINE
# ─────────────────────────────────────────────

class ConversationAI:
    """
    Aerial's conversational layer.
    Scalable rule-based engine with tone awareness and full task context.
    """

    def __init__(
        self,
        user_name:     str = "",
        get_memory_fn: Optional[Callable] = None,
        list_tasks_fn: Optional[Callable] = None,
        api_key:       str = "",
    ):
        self.user_name     = user_name
        self.get_memory_fn = get_memory_fn
        self.list_tasks_fn = list_tasks_fn
        self._cache:  Optional[dict] = None
        self.context = ConversationContext()
        self.context.pending_action = "show_priorities"
        self._last_topic: str = ""

    # ── Context ─────────────────────────────────────────────

    def _ctx(self) -> dict:
        if self._cache is None:
            self._cache = _build_context(self.get_memory_fn, self.list_tasks_fn)
        return self._cache

    def invalidate_cache(self):
        self._cache = None

    # External hook: DialogueManager calls this after completing/deleting tasks
    def record_action(self, action: str, title: str):
        self.context.record_action(action, title)

    # ── Public entry ────────────────────────────────────────

    def chat(self, user_text: str, history: Optional[list] = None) -> str:
        if self.context.awaiting_followup:
            response = self._handle_followup_state(user_text)
            if response:
                return response

        response = self._route(user_text, history or [])
        return response

    # ── Follow-up state handler ──────────────────────────────

    def _handle_followup_state(self, text: str) -> Optional[str]:
        self.context.awaiting_followup = ""
        return None

    # ── Router ──────────────────────────────────────────────

    def _route(self, text: str, history: list) -> str:
        t    = text.lower().strip()
        tone = _detect_tone(text)
        name = self.user_name
        ctx  = self._ctx()

        handlers = [
            self._handle_temporal_query,      # "when did you delete/complete X?" — FIRST
            self._handle_identity,
            self._handle_user_name,
            self._handle_capability,
            self._handle_greeting,
            self._handle_food,
            self._handle_name_intro,
            self._handle_how_are_you,
            self._handle_view_tasks,          # catches "show my full tasks" etc.
            self._handle_task_question,
            self._handle_productivity,
            self._handle_emotional,
            self._handle_motivation,
            self._handle_time,
            self._handle_compliment,
            self._handle_correction,
            self._handle_vague_intent,
            self._handle_acknowledgement,
            self._handle_yes_no,
            self._handle_small_talk,
            self._handle_task_reference,
        ]

        for handler in handlers:
            result = handler(t, text, tone, name, ctx, history)
            if result:
                return result

        return self._fallback(t, tone, name, ctx)

    # ── HANDLERS ────────────────────────────────────────────

    def _handle_temporal_query(self, t, raw, tone, name, ctx, history):
        """Handles 'when did you delete/complete/add X?' without re-triggering actions."""
        if not re.search(
            r"\b(when (did you|was it)|what time (did you|was)|how long ago)\b", t
        ):
            return None

        # Look for the most recent matching action
        for entry in reversed(self.context.recent_actions):
            action = entry["action"]
            title  = entry["title"]
            ts     = entry["time"]
            time_str = ts.strftime("%I:%M %p").lstrip("0")
            date_str = ts.strftime("%B %d")

            # Check if the user is asking about this specific task
            if title and title.lower() in t:
                return f"I {action} \"{title}\" at {time_str} on {date_str}."

            # Generic "when did you delete it / when did that happen"
            if re.search(r"\b(it|that|this)\b", t):
                return f"I {action} \"{title}\" at {time_str} on {date_str}."

        # No action recorded yet
        if re.search(r"\b(delete|remove)\b", t):
            return (f"I don't have a timestamp for that deletion in this session, {name}. "
                    f"Task deletions are permanent — it's gone from your list.")
        if re.search(r"\b(complete|finish|done)\b", t):
            return (f"I don't have a record of when that was completed in this session, {name}.")

        return None

    def _handle_user_name(self, t, raw, tone, name, ctx, history):
        """Handles all ways of asking 'what is my name?'"""
        if re.search(
            r"\b(what('?s| is)|wh?[au]t'?s?)\s+(my\s+)?name\b"
            r"|\b(who am i|do you know my name)\b"
            r"|\b(how do (they|people|everyone|you) call me)\b"
            r"|\bwhat (do|should) (they|people|you|everyone) call me\b"
            r"|\bwhat am i called\b"
            r"|\bwhat'?s my name\b",
            t
        ):
            return f"You go by {name}." if name else "I don't have your name on record yet."
        return None

    def _handle_identity(self, t, raw, tone, name, ctx, history):
        if not re.search(
            r"\b(who are you|what are you|your name|who('?re| is) (this|aerial|"
            r"you)|what('?s| is) your name|introduce yourself|tell me about yourself)\b", t
        ):
            return None

        pending = ctx["pending_count"]
        overdue = ctx["overdue_count"]

        status = ""
        if overdue:
            status = f" Right now, {name} has {overdue} overdue tasks that need attention."
        elif pending:
            status = f" {name} currently has {pending} tasks in the queue."

        return random.choice([
            f"I'm Aerial — {name}'s personal AI task assistant. I help you organise your day, "
            f"track priorities, manage deadlines, and stay productive.{status} What can I do for you?",
            f"The name's Aerial. I'm a smart task assistant built specifically for {name}. "
            f"I track your to-dos, flag what's urgent, and keep you moving forward.{status}",
            f"I'm Aerial, your productivity companion. Think of me as a personal assistant "
            f"that knows your schedule, remembers your tasks, and helps you focus on what matters most.",
        ])

    def _handle_capability(self, t, raw, tone, name, ctx, history):
        if not re.search(
            r"\b(what can you do|what do you do|how (can|do) you help|"
            r"your (features|capabilities|functions)|what are you (capable|able) of|"
            r"how does this work|what('?s| is) your purpose)\b", t
        ):
            return None

        cat  = ctx["most_used_category"]
        rate = ctx["completion_rate"]

        try:
            rate_str = f"{float(str(rate)):.0f}%"
        except Exception:
            rate_str = str(rate)

        return (
            f"Here's what I can do for you, {name}:\n\n"
            f"  📌 Task management — add, complete, delete, and organise tasks\n"
            f"     e.g. \"Remind me to call John tomorrow\" or \"Delete task 3\"\n\n"
            f"  🔥 Smart priorities — I rank tasks by urgency and category\n"
            f"     e.g. \"What should I do today?\"\n\n"
            f"  📊 Productivity tracking — score, habits, completion trends\n"
            f"     e.g. \"What's my productivity score?\"\n\n"
            f"  🧠 Memory — I learn your patterns over time\n"
            f"     Your most active category: {cat} | Completion rate: {rate_str}\n\n"
            f"  🗓 Daily briefing — a morning summary of priorities\n"
            f"     e.g. \"Give me a morning briefing\"\n\n"
            f"  💬 Conversation — I'm always here to chat, remind, and motivate\n\n"
            f"What would you like to tackle first?"
        )

    def _handle_greeting(self, t, raw, tone, name, ctx, history):
        if not re.match(
            r"^(hi+[\s!]*|hey+[\s!]*|hello+[\s!]*|good (morning|afternoon|evening|night)|"
            r"what'?s up|howdy|yo\b|sup\b|greetings|morning[\s!]*|evening[\s!]*)", t
        ):
            return None

        hour      = datetime.now().hour
        pending   = ctx["pending_count"]
        overdue   = ctx["overdue_count"]
        top       = ctx["top_tasks"]
        due_today = ctx["due_today"]

        if hour < 12:
            time_str = "Good morning"
        elif hour < 17:
            time_str = "Good afternoon"
        else:
            time_str = "Good evening"

        if overdue:
            followup = (f"Heads up — you've got {overdue} overdue task(s) that need attention. "
                        f"Want to deal with those first?")
        elif due_today:
            followup = f"You have {len(due_today)} task(s) due today: {', '.join(due_today)}. Ready to get started?"
        elif top:
            followup = f"Your top priority is \"{top[0]}\". Want to jump in?"
        elif pending:
            followup = f"You've got {pending} tasks waiting. Want to see your priorities?"
        else:
            followup = "Your task list is clear. Want to add something or just chat?"

        return random.choice([
            f"{time_str}, {name}! {followup}",
            f"Hey {name}! {followup}",
            f"Hi {name}! Great to hear from you. {followup}",
        ])

    def _handle_name_intro(self, t, raw, tone, name, ctx, history):
        m = re.search(r"\bi'?m\s+([A-Za-z]+)", raw, re.IGNORECASE)
        if not m:
            return None

        NON_NAME_WORDS = {
            "done", "finished", "trying", "going", "thinking",
            "wondering", "planning", "considering", "looking",
            "not", "just", "so", "here", "back",
            "fine", "good", "okay", "well", "great",
            "sure", "ready", "confused", "tired",
            "stressed", "happy", "sad", "excited", "bored",
            "serious", "working", "hoping", "expecting",
            "hungry", "full", "thirsty", "sick", "busy", "free",
            "late", "early", "lost", "stuck", "overwhelmed",
            "anxious", "worried", "focused", "distracted",
            "home", "out", "away", "back", "here", "there",
            "aware", "unsure", "unclear", "certain", "not",
        }

        candidate = m.group(1).lower()
        if candidate in NON_NAME_WORDS:
            return None

        introduced = m.group(1).capitalize()
        pending = ctx["pending_count"]
        return random.choice([
            f"Great to meet you, {introduced}! I'm Aerial, your task assistant. "
            f"You've got {pending} tasks in the system — want to see your priorities?",
            f"Hey {introduced}! I'm Aerial. I'm here to help you stay organised and productive. "
            f"What would you like to work on?",
            f"Nice to meet you, {introduced}. I'll remember that. "
            f"What's on your plate today?",
        ])

    def _handle_how_are_you(self, t, raw, tone, name, ctx, history):
        if not re.search(
            r"\b(how are you|how('?re| are) you doing|you good|you okay|"
            r"how'?s it going|you alright|how'?s your day|are you (okay|good|well))\b", t
        ):
            return None

        pending = ctx["pending_count"]
        overdue = ctx["overdue_count"]
        cat     = ctx["most_used_category"]

        if overdue:
            return (f"Doing well, thank you! Though I'm a little focused on the fact that "
                    f"you've got {overdue} overdue task(s), {name}. Let's fix that — want to go through them?")
        elif pending == 0:
            return (f"Doing great, {name}! And honestly, so are you — your task list is completely clear. "
                    f"Want to plan ahead and add something new?")
        else:
            return (f"Running smoothly, thanks for asking! You've got {pending} tasks waiting, "
                    f"mostly in {cat}. Want to get started on them?")

    def _handle_food(self, t, raw, tone, name, ctx, history):
        if re.search(
            r"\b(what should i eat|what can i eat|food ideas|what'?s for (breakfast|lunch|dinner)|"
            r"i'?m hungry|food recommendation)\b",
            t
        ):
            return (
                f"Food choices are a bit outside my expertise, {name}! "
                f"I'm best with tasks and productivity. But if you need to remember "
                f"to prep a meal or go grocery shopping, just say something like "
                f"\"remind me to cook dinner tonight\" and I'll add it to your list."
            )
        return None

    def _handle_view_tasks(self, t, raw, tone, name, ctx, history):
        """
        Catches all variations of 'show my full/all tasks' that the intent engine
        might miss when extra adjectives are present.
        """
        if re.search(
            r"\b(show|see|view|display|list|get|give me|pull up)\b.{0,20}"
            r"\b(all|full|entire|complete|every|my)?\b.{0,10}"
            r"\b(tasks?|list|to.?dos?|queue|items?)\b",
            t
        ) or re.search(r"\bfull (task )?list\b|\ball (my )?tasks?\b", t):
            # Signal the DialogueManager to run view_tasks — return None so it bubbles
            # up properly.  This handler gives a helpful redirect instead.
            top  = ctx["top_tasks"]
            pend = ctx["pending_count"]
            if pend == 0:
                return f"Your task list is empty, {name}! Want to add something?"
            if top:
                sample = ", ".join(f'"{t}"' for t in top[:3])
                return (
                    f"You have {pend} tasks. Top ones: {sample}"
                    + (f", and {pend - 3} more." if pend > 3 else ".")
                    + "\n\nSay \"show my tasks\" for the full numbered list."
                )
            return f"You have {pend} tasks. Say \"show my tasks\" to see them all."
        return None

    def _handle_task_question(self, t, raw, tone, name, ctx, history):
        if not re.search(
            r"\b(what do you have (for me)?|what'?s (on my list|waiting|next|pending)|"
            r"anything (for me|pending|urgent)|what should i (know|be aware of)|"
            r"give me an? (update|overview|summary)|catch me up|"
            r"is there (anything|something) (specific|i should|urgent|important))\b", t
        ):
            return None

        pending   = ctx["pending_count"]
        overdue   = ctx["overdue_count"]
        top       = ctx["top_tasks"]
        due_today = ctx["due_today"]

        if pending == 0:
            return (f"You're all caught up, {name}! No pending tasks right now. "
                    f"This is a great time to plan ahead — want to add something?")

        parts = [f"Here's your current status, {name}:\n"]

        if overdue:
            parts.append(f"  ⚠️  {overdue} overdue task(s) need immediate attention")

        if due_today:
            parts.append(f"  📅 Due today: {', '.join(due_today)}")

        if top:
            remaining = [task for task in top if task not in due_today]
            if remaining:
                parts.append(f"  📌 Coming up: {', '.join(remaining[:3])}")

        parts.append(f"\n  Total pending: {pending} tasks")
        parts.append("\nSay \"what should I do today?\" for smart priorities.")

        return "\n".join(parts)

    def _handle_productivity(self, t, raw, tone, name, ctx, history):
        if not re.search(
            r"\b(how (am i|have i been) doing|my progress|am i (doing okay|productive|on track)|"
            r"how productive (am i|have i been)|am i behind|my (stats?|numbers?))\b", t
        ):
            return None

        rate    = ctx["completion_rate"]
        pending = ctx["pending_count"]
        overdue = ctx["overdue_count"]

        try:
            rate_num = float(str(rate).replace("%", ""))
            if rate_num >= 75:
                verdict = f"You're doing excellently, {name} — a {rate_num:.0f}% completion rate is impressive"
                tip = "Keep the momentum going"
            elif rate_num >= 50:
                verdict = f"You're making solid progress, {name} — {rate_num:.0f}% completion rate"
                tip = "Tackling your overdue tasks would push that higher" if overdue else "Staying consistent will get you to the next level"
            else:
                verdict = f"There's room to grow, {name} — your completion rate is {rate_num:.0f}%"
                tip = "Start with your smallest task and build from there"

            extras = []
            if overdue:
                extras.append(f"{overdue} overdue tasks")
            if pending:
                extras.append(f"{pending} tasks still pending")

            summary = f". Right now: {', '.join(extras)}." if extras else "."
            return f"{verdict}{summary} {tip}. Say \"productivity score\" for the full breakdown."

        except Exception:
            return (f"Say \"productivity score\" for your full stats, {name}. "
                    f"You currently have {pending} pending tasks.")

    def _handle_emotional(self, t, raw, tone, name, ctx, history):
        if tone not in ("frustrated", "sad"):
            return None

        pending = ctx["pending_count"]

        if tone == "frustrated":
            return random.choice([
                f"I hear you, {name} — that sounds frustrating. Let's simplify things. "
                f"You've got {pending} tasks. Want me to tell you just the one most important thing to do right now?",
                f"Totally understandable. When everything feels overwhelming, it helps to focus on just one thing. "
                f"Want me to pick your highest-priority task?",
                f"Let's take a breath and reset. What's actually bothering you — is it the workload, "
                f"or something specific? I can help you organise it.",
            ])

        if tone == "sad":
            return random.choice([
                f"I'm sorry you're feeling that way, {name}. That's completely valid. "
                f"Even small progress counts — want to start with just one easy task?",
                f"Rough day? That's okay. You don't have to do everything at once. "
                f"I'm here to help you take it one step at a time.",
                f"Take it easy, {name}. Your tasks aren't going anywhere. "
                f"When you're ready, I'll help you figure out where to start.",
            ])

        return None

    def _handle_motivation(self, t, raw, tone, name, ctx, history):
        if not re.search(
            r"\b(motivate me|i need (motivation|a push|help getting started)|"
            r"i'?m (lazy|procrastinat|stuck|not feeling it|struggling)|"
            r"i don'?t (want to|feel like)|i can'?t (do this|get started|focus)|"
            r"help me (get started|focus|stay on track))\b", t
        ):
            return None

        top     = ctx["top_tasks"]
        pending = ctx["pending_count"]

        opener = random.choice([
            f"Here's the thing, {name}:",
            f"Let me be real with you, {name}:",
            f"You've got this, {name}.",
        ])

        if top:
            task_tip = f" Start with \"{top[0]}\" — it's your top priority and completing it will give you real momentum."
        elif pending:
            task_tip = f" You've got {pending} tasks waiting. Pick the smallest one and do just that."
        else:
            task_tip = " Your list is actually clear — use this energy to plan ahead."

        quotes = [
            "Progress, not perfection, is the goal.",
            "Done is better than perfect.",
            "The hardest part is always starting.",
            "Every task you complete is a win — no matter how small.",
        ]

        return f"{opener}{task_tip} Remember: {random.choice(quotes)}"

    def _handle_time(self, t, raw, tone, name, ctx, history):
        if not re.search(
            r"\b(what (time|day|date) is it|today'?s (date|day)|what'?s today|"
            r"what (day|time) (is it|are we)|current (time|date))\b", t
        ):
            return None

        now     = datetime.now()
        pending = ctx["pending_count"]
        overdue = ctx["overdue_count"]

        time_str = now.strftime("%I:%M %p").lstrip("0")
        date_str = now.strftime("%A, %B %d, %Y")

        status = f" You've got {pending} tasks pending" + (f", including {overdue} overdue." if overdue else ".")

        return f"It's {time_str} on {date_str}.{status}"

    def _handle_compliment(self, t, raw, tone, name, ctx, history):
        if not re.search(
            r"\b(good (job|work|one|going)|nice (work|one|job)|well done|"
            r"you'?re (great|awesome|amazing|helpful|good|the best)|"
            r"i (love|like) (you|this|it|how you work)|you('?re| are) (so )?helpful|"
            r"thanks?|thank you|appreciate (it|you|this))\b", t
        ):
            return None

        pending = ctx["pending_count"]

        return random.choice([
            f"Thank you, {name} — that means a lot! Now let's keep that energy going. You've still got {pending} tasks to tackle.",
            f"Appreciate it! I'm here to make your life easier. What's next?",
            f"Glad I could help, {name}! That's what I'm here for. What else do you need?",
            f"You're too kind! Now — back to business. What would you like to work on next?",
        ])

    def _handle_correction(self, t, raw, tone, name, ctx, history):
        if not re.search(
            r"\b(that'?s (wrong|not right|incorrect|not what i (meant|said|asked))|"
            r"i didn'?t (mean|say|ask)|no[,\s]+(that'?s|you|i)|wait[,\s]+no|"
            r"actually[,\s]+(no|that|i)|you misunderstood|"
            r"what\?+|huh\?+|i'?m confused|that'?s confusing|makes no sense)\b", t
        ):
            return None

        return random.choice([
            f"My apologies, {name} — let me try again. What did you mean?",
            f"Sorry about that! I want to get this right. Could you rephrase what you're looking for?",
            f"Got it — I misread that. What were you actually trying to do?",
            f"Let's reset. What exactly do you need from me?",
        ])

    def _handle_acknowledgement(self, t, raw, tone, name, ctx, history):
        t_clean = re.sub(r"[!?.]+$", "", t).strip()

        ack = {
            "yeah", "yep", "yup", "ok", "okay", "cool", "great", "nice", "thanks",
            "thank you", "cheers", "alright", "got it", "sure", "sounds good",
            "perfect", "awesome", "good", "noted", "k", "👍", "np", "no problem",
            "fine", "right", "understood", "i see", "makes sense", "hmm", "hmmm",
            "interesting", "oh", "oh okay", "oh ok", "ah", "ah okay",
            "lol", "haha", "ha", "nice one", "fair enough", "fair", "true",
            "indeed", "of course", "absolutely", "definitely", "certainly",
        }
        words = t_clean.split()
        is_ack = (
            t_clean in ack
            or t in ack
            or (len(words) <= 3 and all(w in ack for w in words))
            or (len(words) <= 3 and any(w in ack for w in words) and len(t_clean) < 20)
        )
        if not is_ack:
            return None

        pending = ctx["pending_count"]
        top     = ctx["top_tasks"]

        generic_responses = [
            f"Got it — what's next, {name}?",
            f"Anything else on your mind?",
            f"What else can I help with?",
            f"Sure thing. I'm here whenever you need me.",
            f"Makes sense. What would you like to do?",
            f"No problem — what else?",
            f"Understood. Anything else?",
            f"Okay! What can I do for you?",
            f"Cool. Let me know if you need anything.",
            f"Sounds good, {name}. What's next?",
        ]

        task_responses = [
            f"Got it! Your top task is \"{top[0]}\" when you're ready.",
            f"Sure thing. \"{top[0]}\" is still waiting — no rush.",
            f"Noted, {name}. Whenever you're ready, \"{top[0]}\" is up first.",
            f"Okay! Just say the word and I'll pull up your full list.",
            f"Understood. You've got {pending} tasks — \"{top[0]}\" is the top priority.",
        ] if top and pending else generic_responses

        if top and pending and len(history) % 4 != 0:
            return random.choice(task_responses)

        return random.choice(generic_responses)

    def _handle_yes_no(self, t, raw, tone, name, ctx, history):
        if t not in {"yes", "yes!", "yep", "yeah", "no", "nope", "nah", "not really", "maybe", "perhaps"}:
            return None

        top     = ctx["top_tasks"]
        pending = ctx["pending_count"]

        if t in {"yes", "yes!", "yep", "yeah"}:
            if self.context.pending_action == "show_priorities":
                self.context.pending_action = ""
                if not top:
                    return "You don't have any pending tasks right now."
                lines = ["Here are your current priorities:\n"]
                for i, task in enumerate(top, start=1):
                    lines.append(f"  {i}. {task}")
                return "\n".join(lines)

            if top:
                return f"Great! Starting with \"{top[0]}\" — say \"show my tasks\" for the full list."
            return f"Perfect — what would you like to do, {name}?"

        if t in {"no", "nope", "nah", "not really"}:
            return random.choice([
                f"No worries, {name}. Is there something else I can help you with?",
                f"No problem, {name}. Is there something else I can help you with? \n I'm here for you",
                f"Got it. What would you like to do instead?",
                f"Fair enough! What's on your mind?",
                f"Alright!, {name}. What's on your mind?",
            ])

        return f"No problem either way, {name}. Just let me know what you need."

    def _handle_vague_intent(self, t, raw, tone, name, ctx, history):
        if re.search(
            r"\bi'?m\s+(thinking|wondering|considering|planning|deciding|trying to figure)\b",
            t
        ):
            top = ctx["top_tasks"]
            if top:
                return (f"Take your time, {name}. When you're ready, your top priority is "
                        f"\"{top[0]}\". Say \"show my tasks\" if you want the full list.")
            return f"No rush, {name}. Let me know what you decide and I'll help you get it done."

        if re.search(r"\bi'?m\s+(not sure|unsure|lost|confused)\b", t):
            return (f"That's okay, {name}! Start by saying \"show my tasks\" to see what's on "
                    f"your plate, or \"what should I do today?\" and I'll pick your top priority.")

        return None

    def _handle_task_reference(self, t, raw, tone, name, ctx, history):
        top = ctx["top_tasks"]
        for task_title in top:
            if task_title.lower() in t:
                self.context.last_referenced_task = task_title
                break

        if re.search(r"\b(delete|remove|cancel)\s+(it|that|this)\b", t):
            task = self.context.last_referenced_task
            if not task:
                return "Which task would you like me to delete? You can say the name or number."
            return f"To delete \"{task}\", say \"delete task\" and I'll take care of it."

        return None

    def _handle_small_talk(self, t, raw, tone, name, ctx, history):
        if re.search(r"\b(weather|rain|sunny|cold|hot|temperature|forecast)\b", t):
            return (f"I don't have access to weather data, {name}, but I can help you plan around it! "
                    f"Want to schedule tasks for today based on your priorities?")

        if re.search(r"\b(news|what'?s happening|current events|latest)\b", t):
            return (f"I'm focused on your personal productivity, {name}, so news is outside my expertise. "
                    f"But I can tell you what's happening in your task list — want a briefing?")

        if re.search(r"\b(tell me a joke|say something funny|make me laugh|joke)\b", t):
            jokes = [
                f"Why did the task manager go to therapy? Too many unresolved issues. 😄 Speaking of which — you've got {ctx['pending_count']} tasks pending, {name}!",
                f"I told my to-do list a joke. It said it couldn't process it right now. 😅 Sound familiar?",
                f"Why do programmers prefer dark mode? Because light attracts bugs — just like procrastination attracts overdue tasks. You've got {ctx['overdue_count']} of those, {name}! 😬",
            ]
            return random.choice(jokes)

        if re.search(r"\b(bye|goodbye|see you|take care|later|gotta go|signing off|"
                     r"talk later|have a good (day|night|one))\b", t):
            pending = ctx["pending_count"]
            return random.choice([
                f"Take care, {name}! You've got {pending} tasks saved — I'll be here when you're back.",
                f"Goodbye, {name}! Don't forget about your priorities. See you soon.",
                f"Have a great one, {name}! Your tasks will be right here waiting for you.",
            ])

        if re.search(r"\b(i'?m (bored|free|not busy)|nothing to do|what'?s (fun|interesting)|"
                     r"entertain me)\b", t):
            top = ctx["top_tasks"]
            if top:
                return (f"Bored? Perfect time to knock out \"{top[0]}\"! "
                        f"Future you will be grateful. Want me to pull up your full list?")
            return f"If you've got free time, {name}, this is a great moment to plan ahead. Want to add some goals?"

        return None

    # ── Fallback ─────────────────────────────────────────────

    def _fallback(self, t, tone, name, ctx) -> str:
        pending = ctx["pending_count"]
        top     = ctx["top_tasks"]

        if tone == "confused":
            return (f"I want to make sure I understand you correctly, {name}. "
                    f"Could you rephrase that? I'm best with task-related requests, "
                    f"but I'm happy to chat too.")

        if tone == "frustrated":
            return (f"I can hear that you're frustrated, {name}. Let's simplify — "
                    f"tell me what you're trying to do and I'll do my best to help.")

        if tone == "sarcastic":
            return (f"Ha — fair enough, {name}. What can I actually help you with?")

        fallbacks_with_task = [
            f"I didn't quite catch that, {name}. Your top priority is \"{top[0]}\" — want to work on that?",
            f"Not sure I followed. Say \"show my tasks\" or \"what should I do today?\" to get going.",
            f"Hmm — try phrasing it differently. Like \"remind me to...\" or \"delete task 3\".",
            f"I'm still learning! Try something like \"show my list\" or tell me what you need.",
        ] if top else [
            f"I didn't quite catch that, {name}. Try \"remind me to...\" or \"what should I do today?\"",
            f"Not sure I followed that. You can say things like \"show my tasks\" or \"add a task\".",
            f"Hmm, I'm not sure what you meant. Try rephrasing — I'm best with task commands.",
        ]

        return random.choice(fallbacks_with_task)
    