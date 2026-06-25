"""
nlp/components/slots.py

SlotFiller — spaCy pipeline component.

Replaces SlotExtractor's regex-strip approach with dependency-tree
traversal.  Sets:

    doc._.title      str | None   — the task title
    doc._.due_date   str | None   — ISO date string or relative keyword
    doc._.task_ref   str | None   — bare numeric task ID

Why DEP-tree over regex strip
──────────────────────────────
Regex strip works by removing noise words and hoping what remains is
the title.  That fails for:
    "add a task to buy groceries by tomorrow"
    → strip "add a task to" → "buy groceries by tomorrow"
    → strip date suffix   → "buy groceries"  ✓ (works by luck)

    "remind me to schedule a meeting with the team next Monday"
    → strip "remind me to" → "schedule a meeting with the team next Monday"
    → strip date suffix   → "schedule a meeting with the team"  ✓ (works)

    "i need to" triggers ADD but "i need to sleep" also would
    → regex can't distinguish; DEP tree can (xcomp verb = sleep → no task object)

DEP approach: find the content verb/noun-phrase that is the *object*
of the intent verb, then collect its full subtree (minus date tokens).

Title extraction strategy
──────────────────────────
For each intent verb (add, remind, create, …) we look for:
  1. A direct `dobj` (direct object): "add [task]", "create [meeting]"
  2. A `xcomp` / `advcl` (clausal complement): "remind me [to call dentist]"
  3. A `prep` → `pobj` chain: "task [for the team meeting]"

We then walk that subtree, exclude DATE-entity tokens and the head
intent-verb tokens, and capitalize the result.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token

# ── Custom attribute registration ─────────────────────────────────────────────

for _ext in ("title", "due_date", "task_ref"):
    if not Doc.has_extension(_ext):
        Doc.set_extension(_ext, default=None)

# ── Constants ─────────────────────────────────────────────────────────────────

# Lemmas of verbs that introduce the task (we skip these in title extraction)
_INTENT_VERB_LEMMAS = frozenset({
    "add", "create", "make", "set", "schedule",
    "remind", "note", "record", "put", "track",
    "delete", "remove", "cancel", "erase",
    "finish", "complete", "mark", "done",
    "show", "list", "display", "view", "see",
    "need", "have", "want", "like", "would",
})

# Stop-words that should NOT appear at the very start of an extracted title
_LEADING_STOP = re.compile(r"^(a|an|the|to|me|my|i|please)\s+", re.IGNORECASE)

# Relative date keywords → ISO resolution
_TODAY    = "today"
_TOMORROW = "tomorrow"
_NEXTWEEK = "next week"

# spaCy NER label for dates/times
_DATE_LABELS = {"DATE", "TIME"}

# Dep relations that can carry the task content
_CONTENT_DEPS = {"dobj", "attr", "nsubj", "oprd"}
_CLAUSE_DEPS  = {"xcomp", "advcl", "ccomp", "relcl"}

# ISO date pattern e.g. 2025-08-20
_ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


# ── Component factory ─────────────────────────────────────────────────────────

@Language.factory("slot_filler")
def create_slot_filler(nlp: Language, name: str):
    return SlotFiller(nlp, name)


class SlotFiller:
    """
    spaCy v3 pipeline component.
    Sets doc._.title, doc._.due_date, doc._.task_ref.
    Runs *after* IntentClassifier so it can read doc._.intent.
    """

    def __init__(self, nlp: Language, name: str):
        self.name = name

    def __call__(self, doc: Doc) -> Doc:
        doc._.due_date = self._extract_date(doc)
        doc._.task_ref = self._extract_task_ref(doc)
        doc._.title    = self._extract_title(doc)
        return doc

    # ── Date extraction ───────────────────────────────────────────────────────

    def _extract_date(self, doc: Doc) -> str | None:
        """
        Priority:
          1. spaCy DATE entity text → resolve to ISO / relative keyword
          2. ISO date literal regex (catches "2025-08-20" that NER might miss)
          3. Manual relative keyword scan (today / tomorrow / next week)
        """
        today = datetime.today()

        # 1. NER entities
        for ent in doc.ents:
            if ent.label_ in _DATE_LABELS:
                resolved = self._resolve_date_text(ent.text.lower(), today)
                if resolved:
                    return resolved

        # 2. ISO literal
        iso_match = _ISO_DATE_RE.search(doc.text)
        if iso_match:
            return iso_match.group(1)

        # 3. Manual relative keywords (NER sometimes misses these in short texts)
        t = doc.text.lower()
        if "next week" in t:
            return (today + timedelta(days=7)).strftime("%Y-%m-%d")
        if "tomorrow" in t:
            return (today + timedelta(days=1)).strftime("%Y-%m-%d")
        if "today" in t:
            return today.strftime("%Y-%m-%d")

        return None

    def _resolve_date_text(self, text: str, today: datetime) -> str | None:
        if "today" in text:
            return today.strftime("%Y-%m-%d")
        if "tomorrow" in text:
            return (today + timedelta(days=1)).strftime("%Y-%m-%d")
        if "next week" in text:
            return (today + timedelta(days=7)).strftime("%Y-%m-%d")
        # spaCy may give us something like "friday" or "august 20" —
        # return the raw text for DialogueManager to handle or prompt about
        iso = _ISO_DATE_RE.search(text)
        if iso:
            return iso.group(1)
        return text if text.strip() else None

    # ── Task reference (numeric ID) ───────────────────────────────────────────

    def _extract_task_ref(self, doc: Doc) -> str | None:
        """Return the first bare number in the text (e.g. 'delete task 3' → '3')."""
        for token in doc:
            if token.like_num and token.text.isdigit():
                return token.text
        return None

    # ── Title extraction ──────────────────────────────────────────────────────

    def _extract_title(self, doc: Doc) -> str | None:
        """
        Walk the DEP tree to find the task content phrase.

        Strategy:
          A) Find the intent verb (root or xcomp of modal).
          B) From that verb, collect the best content subtree:
               - direct object subtree  (dobj)
               - clausal complement     (xcomp / advcl)
               - prepositional object   (prep → pobj)
          C) Strip date tokens from the collected span.
          D) Clean and capitalize.
        """
        # Collect tokens that belong to DATE entities (exclude from title)
        date_token_indices = {
            tok.i for ent in doc.ents if ent.label_ in _DATE_LABELS
            for tok in ent
        }
        # Also exclude tokens that are the resolved relative date words
        t = doc.text.lower()
        if "next week" in t or "tomorrow" in t or "today" in t:
            for tok in doc:
                if tok.lemma_.lower() in {"today", "tomorrow", "week", "next"}:
                    date_token_indices.add(tok.i)

        # Find the content anchor token
        anchor = self._find_content_anchor(doc)
        if anchor is None:
            return None

        # Collect subtree of anchor, minus date tokens and intent-verb tokens
        title_tokens = self._collect_subtree(anchor, date_token_indices, doc)
        if not title_tokens:
            return None

        title = " ".join(t.text for t in title_tokens).strip()
        title = _LEADING_STOP.sub("", title).strip()

        # Minimum length guard
        if len(title) < 2:
            return None

        return title.capitalize()

    def _find_content_anchor(self, doc: Doc) -> Token | None:
        """
        Return the token that heads the task-content phrase.

        For "add a task to buy groceries":
            root = "add" → dobj = "task" → but "task" is meta, not content
            so we look further: xcomp/prep after "task" → "buy" (xcomp of task)
            anchor = "buy"

        For "remind me to call the dentist":
            root = "remind" → xcomp = "call"
            anchor = "call"

        For "i need to submit the report":
            root = "need" (modal) → xcomp = "submit"
            anchor = "submit"
        """
        root = None
        for token in doc:
            if token.dep_ == "ROOT":
                root = token
                break

        if root is None:
            return None

        root_lemma = root.lemma_.lower()

        # Modal verbs: real intent is in xcomp
        if root_lemma in {"need", "have", "want", "would", "like", "plan"}:
            for child in root.children:
                if child.dep_ in _CLAUSE_DEPS and child.pos_ == "VERB":
                    return child
            # No xcomp found — might be "I need sleep" (noun as obj)
            for child in root.children:
                if child.dep_ == "dobj":
                    return child
            return None

        # Intent verb at root — look for content in children
        if root_lemma in _INTENT_VERB_LEMMAS:
            _META_NOUN_DOBJS = {"task", "reminder", "todo", "to-do", "list", "thing"}
            _DATE_PREPS = {"by", "on", "before", "due", "until", "after"}

            # Step 1: Direct object — skip if it's a pronoun (pos_==PRON) OR
            # a meta task noun. Pronouns must be caught by POS, NOT lemma:
            # spaCy lemmatises "me" → "I", "them" → "they", etc., so string
            # matching against "me"/"them" never fires.
            for child in root.children:
                if child.dep_ == "dobj":
                    if child.pos_ == "PRON" or child.lemma_.lower() in _META_NOUN_DOBJS:
                        continue
                    return child

            # Step 2: xcomp / advcl directly off root.
            # "remind me to call dentist" → call is xcomp of remind (root).
            # This fires AFTER the dobj loop so "me" (pronoun dobj, skipped)
            # doesn't block us from reaching "call" (xcomp of root).
            for child in root.children:
                if child.dep_ in _CLAUSE_DEPS and child.pos_ == "VERB":
                    return child

            # Step 3: dobj is a meta noun — check its own clause children.
            # "add a task to buy groceries": task(dobj) → buy(xcomp of task)
            for child in root.children:
                if child.dep_ == "dobj" and child.lemma_.lower() in _META_NOUN_DOBJS:
                    for grandchild in child.children:
                        if grandchild.dep_ in _CLAUSE_DEPS and grandchild.pos_ == "VERB":
                            return grandchild

            # Step 4: prep hanging off dobj meta noun (NOT off root).
            # "create a task for the team meeting":
            #   create(root) → task(dobj) → for(prep) → meeting(pobj)
            # The tree shows "for" is a child of "task", not "create".
            for child in root.children:
                if child.dep_ == "dobj" and child.lemma_.lower() in _META_NOUN_DOBJS:
                    for grandchild in child.children:
                        if grandchild.dep_ == "prep" and grandchild.lower_ not in _DATE_PREPS:
                            for ggchild in grandchild.children:
                                if ggchild.dep_ == "pobj":
                                    return ggchild

            # Step 5: prep hanging directly off root (less common but valid).
            # "schedule a meeting for friday" → meeting is dobj, but
            # "schedule for the team" → for is prep of root with no dobj.
            for child in root.children:
                if child.dep_ == "prep" and child.lower_ not in _DATE_PREPS:
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            return grandchild

        return None

    def _collect_subtree(
        self,
        anchor: Token,
        exclude_indices: set[int],
        doc: Doc,
    ) -> list[Token]:
        """
        Collect the anchor token and all its descendants in doc order,
        excluding:
          - tokens whose index is in exclude_indices (date tokens)
          - punctuation
          - intent-verb lemmas (avoid re-including "add", "remind" etc.)
          - stopword-only particles like bare "to" as the first token
        """
        subtree_tokens = sorted(
            [tok for tok in anchor.subtree],
            key=lambda tok: tok.i,
        )

        result = []
        for tok in subtree_tokens:
            if tok.i in exclude_indices:
                continue
            if tok.is_punct:
                continue
            if tok.lemma_.lower() in _INTENT_VERB_LEMMAS and tok.dep_ in {
                "ROOT", "xcomp", "advcl"
            } and tok.i != anchor.i:
                continue
            result.append(tok)

        # Drop a leading bare "to" (infinitive marker)
        while result and result[0].lower_ == "to":
            result = result[1:]

        # Drop trailing bare prepositions left behind when date tokens were
        # excluded — e.g. "submit the report by [friday]" → "submit the report by"
        # The "by" is a prep whose only child was the excluded date token.
        _TRAILING_PREPS = {"by", "on", "before", "due", "until", "after", "at"}
        while result and result[-1].lower_ in _TRAILING_PREPS:
            result = result[:-1]

        return result