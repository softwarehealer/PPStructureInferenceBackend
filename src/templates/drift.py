"""Template field extraction that follows drift.

All coordinates are in the template's reference page: an upload is aligned to it first, which
removes whole-page drift (margins, tilt, scale). What remains is drift *inside* the page, e.g. a
missing line pushing the fields below it up. Each field therefore hangs from an anchor:

- ``page``: a fixed print position (Aadhaar number, QR code). Searched at its own box only.
- ``static:<id>``: a static element, i.e. printed text found on every document of the template
  (a label such as "DOB"). Static elements are located by their text, which measures drift.
- ``field:<key>``: another field, for text that flows after variable-length content.

A field is searched for at its template box shifted by its anchor's measured shift, and its own
measured shift is passed on to the fields anchored to it, so drift only travels along anchor links.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field as dataclass_field
from difflib import SequenceMatcher

Rect = tuple[float, float, float, float]

# A static element's text must be this similar to OCR text to match (stricter for short texts,
# where one wrong letter changes the word: "Male" is not "Female").
STATIC_MIN_SIMILARITY = 0.75
STATIC_MIN_SIMILARITY_SHORT = 0.85
SHORT_TEXT = 8
# OCR can glue a label to the unreadable text after it ("DOB/" + Hindi -> "DOB/GH"): a static
# element of at least this many letters/digits also matches the start of a word, scored just below
# a real match so that a clean reading elsewhere wins.
PREFIX_MIN_CHARS = 3
PREFIX_SIMILARITY = 0.88
# Static elements are only searched for within this share of the page from their template position.
STATIC_MAX_SHIFT = 0.15
# Words whose area lies at least this much inside a field's (shifted) box belong to it.
MIN_OVERLAP = 0.5
# When nothing is inside the box, rows this far around it (in box heights / page widths) are tried.
NEARBY_ROWS = 0.75
NEARBY_WIDTH = 0.03
# Multi-line fields grow downwards by at most this many rows past their box.
MAX_EXTRA_ROWS = 4

# Formats a field can be checked against; the value is cut down to the part that matches.
PATTERNS = {
    "aadhaar": re.compile(r"(?<!\d)[2-9]\d{11}(?!\d)"),
    "pan": re.compile(r"(?<![A-Z])[A-Z]{5}\d{4}[A-Z](?![A-Z])"),
    "date": re.compile(r"\d{1,2}\s*[/\-.]\s*\d{1,2}\s*[/\-.]\s*\d{2,4}|\d{1,2}[\s/\-.]+[A-Za-z]{3,9}[\s/\-.]+\d{2,4}"),
}


@dataclass
class Box:
    """An OCR word (with the OCR line it belongs to)."""
    text: str
    rect: Rect
    confidence: float = 1.0
    line: int = 0


@dataclass
class Shift:
    """How far something was found from its template position, per edge."""
    dx_left: float = 0.0
    dx_right: float = 0.0
    dy_top: float = 0.0
    dy_bottom: float = 0.0

    @classmethod
    def between(cls, found: Rect, expected: Rect) -> "Shift":
        return cls(found[0] - expected[0], found[2] - expected[2], found[1] - expected[1], found[3] - expected[3])

    @classmethod
    def uniform(cls, dx: float, dy: float) -> "Shift":
        return cls(dx, dx, dy, dy)


@dataclass
class StaticMatch:
    id: str
    found: Rect | None
    shift: Shift | None
    # Indices of the words it was found in; they are label text, never part of a field value
    words: list[int] = dataclass_field(default_factory=list)


@dataclass
class FieldResult:
    key: str
    value: str | None
    # found: inside its (shifted) box; nearby: in a row next to it; corrected: another row that
    # passed the field's check; invalid: found but failed its check; not_found: nothing there
    status: str
    anchor: str
    inherited: tuple[float, float]   # (dx, dy) taken from the anchor
    found: Rect | None
    shift: Shift                     # own shift vs the template box, passed on to anchored fields
    valid: bool | None


def check_value(check: str, text: str) -> str | None:
    """The part of ``text`` in the field's format, or None when it has none."""
    if check == "number":
        return text if re.search(r"\d", text) else None
    if check == "aadhaar":
        match = PATTERNS["aadhaar"].search(re.sub(r"\s", "", text))
        return f"{match.group()[:4]} {match.group()[4:8]} {match.group()[8:]}" if match else None
    if check == "pan":
        match = PATTERNS["pan"].search(re.sub(r"\s", "", text.upper()))
        return match.group() if match else None
    match = PATTERNS[check].search(text)
    return match.group().strip() if match else None


# ---------------------------------------------------------------------------
# Template creation helpers
# ---------------------------------------------------------------------------

def derive_statics(words: list[Box], variable: list[Rect]) -> list[dict]:
    """Static elements of a sample page: runs of OCR words outside every field box and masked region.

    A line that runs into a field (e.g. "DOB / 27/01/2003") contributes only its words outside
    the field ("DOB /"), so labels are kept even when OCR joins them with their value.
    """
    statics = []
    for line_words in _by_line(words).values():
        run: list[Box] = []
        for word in line_words + [None]:
            if word is not None and not any(_overlap(word.rect, r) >= 0.3 for r in variable):
                run.append(word)
                continue
            if run:
                text = " ".join(w.text for w in run).strip()
                if len(_norm(text)) >= 3 and min(w.confidence for w in run) >= 0.8:
                    statics.append({"text": text, "bbox": [round(v) for v in _union([w.rect for w in run])]})
                run = []
    statics.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
    return [{"id": f"s{index}", "text": s["text"], "bbox": s["bbox"]} for index, s in enumerate(statics, start=1)]


def resolve_auto_anchor(field: dict, fields: list[dict], statics: list[dict]) -> str:
    """Anchor for ``auto``: a label on the same row to the left, else the nearest element above.

    Candidates are static elements and fields that start above this one, so anchors never loop.
    """
    rect = tuple(field["bbox"])
    height = rect[3] - rect[1]
    left, above = [], []
    candidates = [(f"static:{s['id']}", tuple(s["bbox"])) for s in statics]
    candidates += [(f"field:{f['key']}", tuple(f["bbox"])) for f in fields if f["key"] != field["key"] and f["bbox"][1] < rect[1]]
    for name, other in candidates:
        vertical = min(rect[3], other[3]) - max(rect[1], other[1])
        if vertical >= 0.5 * min(height, other[3] - other[1]) and other[2] <= rect[0] + 0.1 * (rect[2] - rect[0]):
            left.append((rect[0] - other[2], name))
        elif other[3] <= rect[1] + 0.25 * height and min(rect[2], other[2]) > max(rect[0], other[0]):
            above.append((rect[1] - other[3], name))
    if left:
        return min(left)[1]
    if above:
        return min(above)[1]
    return "page"


def ordered_by_anchor(fields: list[dict]) -> list[dict]:
    """Fields with each field after the field it is anchored to. Raises on unknown anchors or loops."""
    keys = {f["key"] for f in fields}
    done: set[str] = set()
    order, pending = [], list(fields)
    while pending:
        progressed = False
        for field in list(pending):
            anchor = field["anchor"]
            if anchor.startswith("field:") and anchor[6:] not in keys:
                raise ValueError(f"Field '{field['key']}' is anchored to unknown field '{anchor[6:]}'.")
            if not anchor.startswith("field:") or anchor[6:] in done:
                order.append(field)
                done.add(field["key"])
                pending.remove(field)
                progressed = True
        if not progressed:
            raise ValueError("Field anchors form a loop: " + ", ".join(f["key"] for f in pending))
    return order


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def locate_statics(statics: list[dict], words: list[Box], page_size: tuple[int, int]) -> dict[str, StaticMatch]:
    """Find each static element by its text near its template position, measuring its shift.

    Its text is compared with runs of consecutive words of a line, ignoring case, spaces and
    punctuation. The shift is taken from its left edge and vertical centre.
    """
    width, height = page_size
    indexed = list(enumerate(words))
    normalized = [_norm(w.text) for w in words]
    matches = {}
    for static in statics:
        expected = tuple(static["bbox"])
        target = _norm(static["text"])
        threshold = STATIC_MIN_SIMILARITY_SHORT if len(target) < SHORT_TEXT else STATIC_MIN_SIMILARITY
        window = _expand(expected, STATIC_MAX_SHIFT * width, STATIC_MAX_SHIFT * height)
        nearby = [(i, w) for i, w in indexed if _contains(window, _center(w.rect))]
        # Most static elements are where the template has them; search the window only when not
        spot = [(i, w) for i, w in nearby if _overlap(w.rect, expected) >= MIN_OVERLAP]
        max_run = len(static["text"].split()) + 2
        candidates = _static_runs(spot, target, threshold, max_run, normalized, prefix=True)
        if not candidates or max(c[0] for c in candidates) < 0.9:
            candidates = _static_runs(nearby, target, threshold, max_run, normalized, prefix=True)
        first = _norm(static["text"].split()[0]) if static["text"].split() else ""
        if not candidates and first != target and len(first) >= PREFIX_MIN_CHARS:
            # A label read with unreadable words ("DOB/- 4fAf :", OCR of "DOB / जन्म तिथि :") is
            # found by its first word, at its own spot only: the rest reads differently each time
            candidates = _static_runs(spot, first, STATIC_MIN_SIMILARITY_SHORT, 2, normalized, prefix=True)
        if not candidates:
            matches[static["id"]] = StaticMatch(static["id"], None, None)
            continue
        best = max(c[0] for c in candidates)
        scored = []
        for similarity, rect, indices in candidates:
            dx, dy = rect[0] - expected[0], _center(rect)[1] - _center(expected)[1]
            scored.append((abs(dx) / width + abs(dy) / height, similarity, rect, indices, dx, dy))
        _, _, rect, indices, dx, dy = min((c for c in scored if c[1] >= best - 0.1), key=lambda c: c[0])
        matches[static["id"]] = StaticMatch(static["id"], rect, Shift.uniform(dx, dy), indices)
    return matches


def _static_runs(
    pairs: list[tuple[int, Box]], target: str, threshold: float, max_run: int, normalized: list[str],
    *, prefix: bool = False,
) -> list[tuple[float, Rect, list[int]]]:
    """Runs of consecutive words of a line whose text is similar to ``target``:
    (similarity, rect, word indices).

    With ``prefix``, a run whose last word continues past ``target`` ("DOB/GH" for "DOB/") matches
    too, scored PREFIX_SIMILARITY; that last word is not among the indices, so it is not claimed.
    """
    # Texts can only be this similar when their lengths are comparable
    shortest, longest = threshold * len(target) / (2 - threshold), len(target) * (2 - threshold) / threshold
    matcher = SequenceMatcher(None)
    matcher.set_seq2(target)
    runs = []
    for line_words in _by_line([w for _, w in pairs], [i for i, _ in pairs]).values():
        for start in range(len(line_words)):
            text = ""
            for end in range(start, min(start + max_run, len(line_words))):
                before = text
                text += normalized[line_words[end][0]]
                if prefix and len(target) >= PREFIX_MIN_CHARS and len(before) < len(target) < len(text) and text.startswith(target):
                    run = line_words[start:end + 1]
                    runs.append((PREFIX_SIMILARITY, _union([w.rect for _, w in run]), [i for i, _ in run[:-1]]))
                    break
                if len(text) > longest:
                    break
                if len(text) < shortest:
                    continue
                matcher.set_seq1(text)
                if matcher.quick_ratio() < threshold:
                    continue
                similarity = matcher.ratio()
                if similarity >= threshold:
                    run = line_words[start:end + 1]
                    runs.append((similarity, _union([w.rect for _, w in run]), [i for i, _ in run]))
    return runs


def extract_fields(
    fields: list[dict],
    statics: list[dict],
    static_matches: dict[str, StaticMatch],
    words: list[Box],
    page_size: tuple[int, int],
    *,
    unstable_statics: set[str] = frozenset(),
) -> list[FieldResult]:
    """Extract every field, following its anchor's drift. Results are in template field order."""
    template_rects = {f["key"]: tuple(f["bbox"]) for f in fields}
    static_rects = {s["id"]: tuple(s["bbox"]) for s in statics}
    located = [m for m in static_matches.values() if m.found is not None and m.id not in unstable_statics]
    # Multi-line fields stop growing at located static elements and at fixed fields' boxes.
    blockers = [m.found for m in located] + [tuple(f["bbox"]) for f in fields if f["anchor"] == "page"]
    results: dict[str, FieldResult] = {}
    claimed: set[int] = {i for m in located for i in m.words}

    for field in ordered_by_anchor(fields):
        expected = tuple(field["bbox"])
        anchor = field["anchor"]
        dx = dy = 0.0
        if anchor.startswith("static:"):
            match = static_matches.get(anchor[7:])
            if match and match.found is not None and anchor[7:] not in unstable_statics:
                dx, dy = _inherit(expected, static_rects[anchor[7:]], match.shift)
        elif anchor.startswith("field:"):
            dx, dy = _inherit(expected, template_rects[anchor[6:]], results[anchor[6:]].shift)
        box = (expected[0] + dx, expected[1] + dy, expected[2] + dx, expected[3] + dy)

        available = [w for i, w in enumerate(words) if i not in claimed]
        chosen = [w for w in available if _overlap(w.rect, box) >= MIN_OVERLAP]
        status = "found"
        if not chosen:
            chosen = _nearest_row(available, box, page_size)
            status = "nearby" if chosen else "not_found"
        if chosen and field.get("multiline"):
            chosen = _grow_down(chosen, available, box, [b for b in blockers if b != expected])

        value = _join(chosen) if chosen else None
        valid = None
        if chosen and field.get("check"):
            checked = check_value(field["check"], value)
            if checked is None:
                alternative = _valid_row(available, box, page_size, field["check"], exclude=chosen)
                if alternative:
                    chosen, status = alternative, "corrected"
                    checked = check_value(field["check"], _join(chosen))
                else:
                    status = "invalid"
            valid = checked is not None
            value = checked if checked is not None else value

        found = _union([w.rect for w in chosen]) if chosen else None
        claimed.update(i for i, w in enumerate(words) if any(w is c for c in chosen))
        results[field["key"]] = FieldResult(
            key=field["key"],
            value=value,
            status=status,
            anchor=anchor,
            inherited=(round(dx, 1), round(dy, 1)),
            found=found,
            shift=Shift.between(found, expected) if found else Shift.uniform(dx, dy),
            valid=valid,
        )
    return [results[f["key"]] for f in fields]


def _inherit(field_rect: Rect, anchor_rect: Rect, shift: Shift) -> tuple[float, float]:
    """The part of an anchor's shift a field takes: bottom edge if the field is below the anchor,
    right edge if it follows on the same row, else the top/left edges."""
    anchor_height = anchor_rect[3] - anchor_rect[1]
    below = field_rect[1] >= anchor_rect[3] - 0.25 * anchor_height
    right = not below and field_rect[0] >= anchor_rect[2] - 0.1 * (anchor_rect[2] - anchor_rect[0])
    return (shift.dx_right if right else shift.dx_left), (shift.dy_bottom if below else shift.dy_top)


def _nearest_row(words: list[Box], box: Rect, page_size: tuple[int, int]) -> list[Box]:
    """The row closest to the box, within a small window around it (for jitter past the box)."""
    width, _ = page_size
    window = _expand(box, NEARBY_WIDTH * width, NEARBY_ROWS * (box[3] - box[1]))
    rows = _rows([w for w in words if _contains(window, _center(w.rect))])
    if not rows:
        return []
    target = _center(box)[1]
    return min(rows, key=lambda row: abs(_center(_union([w.rect for w in row]))[1] - target))


def _valid_row(words: list[Box], box: Rect, page_size: tuple[int, int], check: str, *, exclude: list[Box]) -> list[Box]:
    """The closest row around the box whose text passes the field's check."""
    width, _ = page_size
    window = _expand(box, 2 * NEARBY_WIDTH * width, 2 * NEARBY_ROWS * (box[3] - box[1]))
    candidates = [w for w in words if _contains(window, _center(w.rect)) and not any(w is e for e in exclude)]
    target = _center(box)[1]
    for row in sorted(_rows(candidates), key=lambda r: abs(_center(_union([w.rect for w in r]))[1] - target)):
        if check_value(check, _join(row)) is not None:
            return row
    return []


def _grow_down(chosen: list[Box], words: list[Box], box: Rect, blockers: list[Rect]) -> list[Box]:
    """Add rows directly below a multi-line field while they continue its column."""
    chosen = list(chosen)
    line_height = statistics.median(w.rect[3] - w.rect[1] for w in chosen)
    for _ in range(MAX_EXTRA_ROWS):
        bottom = max(w.rect[3] for w in chosen)
        below = [
            w for w in words
            if not any(w is c for c in chosen)
            # text written sideways (a number up the margin) is no row of the value
            and w.rect[3] - w.rect[1] <= 2 * line_height
            and bottom - 0.3 * line_height <= w.rect[1] <= bottom + 0.8 * line_height
            and box[0] - line_height <= _center(w.rect)[0] <= box[2]
        ]
        rows = _rows(below)
        if not rows:
            break
        row = min(rows, key=lambda r: min(w.rect[1] for w in r))
        if any(_overlap(w.rect, b) >= 0.3 for w in row for b in blockers):
            break
        # a label on the row's line starts another item ("Mobile: 98..." after an address)
        top, row_bottom = min(w.rect[1] for w in row), max(w.rect[3] for w in row)
        if any(
            min(row_bottom, b[3]) - max(top, b[1]) >= 0.5 * (row_bottom - top)
            and box[0] - line_height <= _center(b)[0] <= box[2]
            for b in blockers
        ):
            break
        chosen.extend(row)
    return chosen


# ---------------------------------------------------------------------------
# Geometry and text helpers
# ---------------------------------------------------------------------------

def _by_line(words: list, keys: list | None = None) -> dict[int, list]:
    """Words grouped by OCR line, left to right; with ``keys``, (key, word) pairs instead."""
    lines: dict[int, list] = {}
    for index, word in enumerate(words):
        lines.setdefault(word.line, []).append(word if keys is None else (keys[index], word))
    position = (lambda w: w.rect[0]) if keys is None else (lambda pair: pair[1].rect[0])
    return {line: sorted(items, key=position) for line, items in lines.items()}


def _rows(words: list[Box]) -> list[list[Box]]:
    """Group words into rows (a word joins a row when its centre is within the row's height)."""
    rows: list[list[Box]] = []
    for word in sorted(words, key=lambda w: _center(w.rect)[1]):
        cy = _center(word.rect)[1]
        if rows:
            last = rows[-1][-1].rect
            if last[1] <= cy <= last[3]:
                rows[-1].append(word)
                continue
        rows.append([word])
    return [sorted(row, key=lambda w: w.rect[0]) for row in rows]


def _join(words: list[Box]) -> str:
    return "\n".join(" ".join(w.text.strip() for w in row) for row in _rows(words))


def _norm(text: str) -> str:
    """Letters and digits only, lower case: OCR often adds or drops spaces and punctuation."""
    return re.sub(r"[\W_]+", "", text).lower()


def _center(rect: Rect) -> tuple[float, float]:
    return (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2


def _contains(rect: Rect, point: tuple[float, float]) -> bool:
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


def _expand(rect: Rect, dx: float, dy: float) -> Rect:
    return rect[0] - dx, rect[1] - dy, rect[2] + dx, rect[3] + dy


def _union(rects: list[Rect]) -> Rect:
    return min(r[0] for r in rects), min(r[1] for r in rects), max(r[2] for r in rects), max(r[3] for r in rects)


def _overlap(inner: Rect, outer: Rect) -> float:
    """Share of ``inner``'s area that lies inside ``outer``."""
    width = min(inner[2], outer[2]) - max(inner[0], outer[0])
    height = min(inner[3], outer[3]) - max(inner[1], outer[1])
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    if width <= 0 or height <= 0 or area <= 0:
        return 0.0
    return width * height / area


def rect_of(region: list[list[float]]) -> Rect:
    xs = [p[0] for p in region]
    ys = [p[1] for p in region]
    return min(xs), min(ys), max(xs), max(ys)
