"""
Run: python debug_tree.py
Shows the exact spaCy dependency tree for the two failing sentences.
"""
import spacy

nlp = spacy.load("en_core_web_sm")

sentences = [
    "remind me to call the dentist tomorrow",
    "create a task for the team meeting",
]

for text in sentences:
    doc = nlp(text)
    print(f"\n{'='*60}")
    print(f"TEXT: {text!r}")
    print(f"{'='*60}")
    print(f"  {'TOKEN':<14} {'DEP':<12} {'POS':<8} {'TAG':<8} HEAD")
    print(f"  {'-'*14} {'-'*12} {'-'*8} {'-'*8} ----")
    for tok in doc:
        marker = " ← ROOT" if tok.dep_ == "ROOT" else ""
        print(f"  {tok.text:<14} {tok.dep_:<12} {tok.pos_:<8} {tok.tag_:<8} {tok.head.text}{marker}")
    print(f"\n  ENTs: {[(e.text, e.label_) for e in doc.ents]}")