"""
nlp/components/category.py

CategoryDetector — spaCy pipeline component.

Replaces SlotExtractor.extract_category()'s "any keyword in string"
approach with token-context awareness.

The old approach had two failure modes:
  1. "my doctor said I need to work more" → matched both "health" and
     "work" keyword lists; picked whichever came first in the dict.
  2. Category was inferred from the *full* sentence including intent words,
     not just the task-content part.

This component:
  a) Operates only on the title span (if already extracted) OR the full doc.
  b) Uses a scored approach: each keyword match contributes a weighted vote.
     Category with the highest score wins.
  c) Considers POS: a noun/adjective keyword in direct object position
     scores higher than one in a prepositional phrase.

Sets doc._.category (str | None).  None means "not detected, use default".
"""
from __future__ import annotations

import spacy
from spacy.language import Language
from spacy.tokens import Doc

# ── Custom attribute registration ─────────────────────────────────────────────

if not Doc.has_extension("category"):
    Doc.set_extension("category", default=None)


# ── Category keyword definitions ──────────────────────────────────────────────
#
# Each entry is (keyword, weight).
# Higher weight = stronger signal for that category.
# Deliberately no overlap between lists.

_CATEGORY_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "work": [
        ("work",         1.0),
        ("job",          0.9),
        ("office",       0.9),
        ("meeting",      0.8),
        ("project",      0.8),
        ("client",       0.8),
        ("deadline",     0.8),
        ("report",       0.7),
        ("presentation", 0.7),
        ("email",        0.6),
        ("colleague",    0.7),
        ("boss",         0.7),
        ("manager",      0.7),
        ("team",         0.6),
        ("sprint",       0.9),
        ("standup",      0.9),
        ("deploy",       0.8),
        ("review",       0.6),
        ("invoice",      0.8),
        ("contract",     0.8),
    ],
    "health": [
        ("gym",          1.0),
        ("health",       1.0),
        ("doctor",       0.9),
        ("dentist",      0.9),
        ("hospital",     0.9),
        ("exercise",     0.9),
        ("workout",      0.9),
        ("run",          0.7),
        ("jog",          0.8),
        ("walk",         0.5),   # low — too generic
        ("medication",   1.0),
        ("medicine",     0.9),
        ("therapy",      0.9),
        ("therapist",    0.9),
        ("sleep",        0.6),
        ("diet",         0.8),
        ("vitamin",      0.8),
        ("appointment",  0.6),
        ("checkup",      0.9),
        ("prescription", 1.0),
    ],
    "school": [
        ("school",       1.0),
        ("study",        0.9),
        ("homework",     1.0),
        ("assignment",   1.0),
        ("class",        0.8),
        ("lecture",      0.9),
        ("exam",         1.0),
        ("test",         0.7),
        ("quiz",         0.9),
        ("essay",        0.9),
        ("thesis",       1.0),
        ("professor",    0.9),
        ("campus",       0.9),
        ("course",       0.8),
        ("grade",        0.8),
        ("tutor",        0.9),
        ("library",      0.7),
        ("semester",     0.9),
        ("university",   0.9),
        ("college",      0.9),
    ],
    "personal": [
        ("home",         0.8),
        ("family",       0.9),
        ("friend",       0.8),
        ("errand",       0.9),
        ("grocery",      0.9),
        ("groceries",    0.9),
        ("cook",         0.7),
        ("clean",        0.7),
        ("laundry",      0.9),
        ("shopping",     0.7),
        ("birthday",     0.9),
        ("call",         0.4),   # low — could be work too
        ("hobby",        0.8),
        ("vacation",     0.8),
        ("travel",       0.7),
        ("bill",         0.8),
        ("bank",         0.7),
        ("rent",         0.8),
    ],
}

# Build a flat lookup: lemma → {category: weight}
_KEYWORD_MAP: dict[str, dict[str, float]] = {}
for _cat, _entries in _CATEGORY_KEYWORDS.items():
    for _kw, _wt in _entries:
        if _kw not in _KEYWORD_MAP:
            _KEYWORD_MAP[_kw] = {}
        _KEYWORD_MAP[_kw][_cat] = _wt


# ── Component factory ─────────────────────────────────────────────────────────

@Language.factory("category_detector")
def create_category_detector(nlp: Language, name: str):
    return CategoryDetector(nlp, name)


class CategoryDetector:
    """
    spaCy v3 pipeline component.
    Sets doc._.category (str | None).
    """

    # Minimum total score for a category to be returned at all.
    # Below this threshold we return None (DialogueManager will ask or default).
    _MIN_SCORE = 0.6

    def __init__(self, nlp: Language, name: str):
        self.name = name

    def __call__(self, doc: Doc) -> Doc:
        doc._.category = self._detect(doc)
        return doc

    def _detect(self, doc: Doc) -> str | None:
        scores: dict[str, float] = {}

        for token in doc:
            lemma = token.lemma_.lower()
            if lemma not in _KEYWORD_MAP:
                continue

            # Position bonus: token in object/complement position → stronger signal
            position_mult = 1.2 if token.dep_ in {
                "dobj", "nsubj", "attr", "oprd", "pobj"
            } else 1.0

            for cat, base_weight in _KEYWORD_MAP[lemma].items():
                scores[cat] = scores.get(cat, 0.0) + base_weight * position_mult

        if not scores:
            return None

        best_cat  = max(scores, key=lambda c: scores[c])
        best_score = scores[best_cat]

        return best_cat if best_score >= self._MIN_SCORE else None