"""Document type markers: refuse an upload that is plainly another type before Gemini is asked to
draft it, and learn markers from the documents that are read.

A doc type's markers are phrases printed on every document of the type; its patterns are regular
expressions of its numbers (doc_types table). An upload sent as a type is refused only when the
page shows none of that type's markers or patterns, and a *trusted* marker of another type.
Anything less clear-cut goes to Gemini, which judges the type itself. Patterns only ever count for
the requested type: numbers like a PAN or a 12-digit number are printed on many other documents.

Markers come from two sources:

- seed (doc_types.json): checked by hand on sample documents, trusted from the start, never
  removed automatically;
- gemini: proposed while drafting a template. Such a marker first only counts *for* its type (it
  can let a document through, never refuse one). It is trusted once it was read on
  TRUST_SEEN documents of its type and TRUST_OTHERS documents of other types went without it.
  When a document of another type shows it, it is shared between types and is removed.

"Documents of a type" are confirmed ones: read with a stored template of the type, or drafted as
the type.
"""
from __future__ import annotations

import re
from typing import Any

from configs import app_logger as logger
from templates import store
from templates.drift import Box, _norm, _static_runs

# Similarity a run of OCR words needs to a marker phrase (as for short static labels)
MARKER_MIN_SIMILARITY = 0.85
# A proposed marker needs this many letters/digits, or several words: short text is too common
MARKER_MIN_CHARS = 8
# A Gemini marker is trusted after it was read on this many documents of its type, while this many
# documents of other types went without it
TRUST_SEEN = 5
TRUST_OTHERS = 3


class _Page:
    """A page's OCR words prepared for looking up marker phrases."""

    def __init__(self, words: list[Box]):
        ordered = sorted(words, key=lambda w: (w.line, w.rect[0]))
        self.raw = " ".join(w.text for w in ordered)
        self.text = _norm(self.raw)
        self.normalized = [_norm(w.text) for w in ordered]
        self.pairs = list(enumerate(ordered))

    def shows(self, phrase: str) -> bool:
        target = _norm(phrase)
        if not target:
            return False
        # Exactly, ignoring spaces and punctuation (also when glued to other text), else similar
        # to a run of words of a line
        return target in self.text or bool(
            _static_runs(self.pairs, target, MARKER_MIN_SIMILARITY, len(phrase.split()) + 2, self.normalized)
        )


def other_type(requested: str, words: list[Box]) -> tuple[str, str] | None:
    """(doc_type, marker found) when the page is plainly another type than ``requested``; None
    when it may be ``requested`` or the page cannot tell."""
    types = store.list_doc_types()
    spec = types.get(requested)
    if not spec or not (spec["markers"] or spec["patterns"]) or not words:
        return None
    page = _Page(words)
    if any(re.search(p, page.raw) for p in spec["patterns"]) or any(page.shows(m["text"]) for m in spec["markers"]):
        return None
    for doc_type, other in types.items():
        if doc_type != requested:
            for marker in other["markers"]:
                if marker.get("trusted") and page.shows(marker["text"]):
                    return doc_type, marker["text"]
    return None


def observe(doc_type: str, words: list[Box]) -> None:
    """Learn from a document confirmed to be ``doc_type``: its own Gemini markers read on it count
    towards trust; other types' Gemini markers read on it are shared, so they are removed."""
    page = _Page(words)
    updates: dict[str, list[dict[str, Any]]] = {}
    for name, spec in store.list_doc_types().items():
        kept, learned = [], False
        for marker in spec["markers"]:
            found = page.shows(marker["text"])
            if marker.get("source") == "seed":
                if found and name != doc_type:
                    logger.warning("marker %r of %s is printed on a %s too; review doc_types.json", marker["text"], name, doc_type)
                kept.append(marker)
                continue
            marker, learned = dict(marker), True
            if name == doc_type:
                marker["seen"] = marker.get("seen", 0) + int(found)
            elif found:
                logger.info("marker %r of %s removed: a %s shows it too", marker["text"], name, doc_type)
                continue
            else:
                marker["others_without"] = marker.get("others_without", 0) + 1
            if not marker.get("trusted") and marker.get("seen", 0) >= TRUST_SEEN and marker.get("others_without", 0) >= TRUST_OTHERS:
                marker["trusted"] = True
                logger.info("marker %r of %s is trusted now", marker["text"], name)
            kept.append(marker)
        if learned:
            updates[name] = kept
    if updates:
        store.save_markers(updates)


def propose(doc_type: str, phrases: list[str], words: list[Box]) -> tuple[list[str], list[str]]:
    """Add markers Gemini proposed for ``doc_type`` after checking them: read on this page, long
    enough, and not text of another type (its markers, or the static text of its templates).
    Returns (added, notes on the ones left out)."""
    types = store.list_doc_types()
    if doc_type not in types:
        return [], []
    page = _Page(words)
    other_markers = [(name, m["text"]) for name, spec in types.items() if name != doc_type for m in spec["markers"]]
    other_statics = [
        _Page([Box(text, tuple(s["bbox"]), 1.0, n) for n, s in enumerate(t["static_elements"]) for text in s["text"].split()])
        for t in store.list_templates() if t["doc_type"] != doc_type
    ]
    own = {_norm(m["text"]) for m in types[doc_type]["markers"]}
    added, notes = [], []
    for phrase in dict.fromkeys(p.strip() for p in phrases if p.strip()):
        target = _norm(phrase)
        if target in own:
            notes.append(f"marker {phrase!r}: already a marker of {doc_type}")
        elif len(target) < MARKER_MIN_CHARS and len(phrase.split()) < 2:
            notes.append(f"marker {phrase!r} left out: too short to tell document types apart")
        elif not page.shows(phrase):
            notes.append(f"marker {phrase!r} left out: not read on this page")
        elif clash := next((name for name, text in other_markers if _norm(text) in target or target in _norm(text)), None):
            notes.append(f"marker {phrase!r} left out: it is also a marker of {clash}")
        elif any(statics.shows(phrase) for statics in other_statics):
            notes.append(f"marker {phrase!r} left out: printed on a template of another type")
        else:
            added.append(phrase)
            own.add(target)
    if added:
        markers = types[doc_type]["markers"] + [
            {"text": p, "source": "gemini", "trusted": False, "seen": 1, "others_without": 0} for p in added
        ]
        store.save_markers({doc_type: markers})
        logger.info("markers proposed for %s: %s", doc_type, added)
    return added, notes
