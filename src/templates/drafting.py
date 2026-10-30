"""Gemini drafts a template for a document that no stored template matches.

Gemini gets the page image and the page's OCR words, numbered, and answers in word numbers:
which words are each field's value and label, which text is printed on every document of the
layout (static elements) and which is personal data (masked). Everything it says is checked
against the OCR before it becomes a template, and boxes always come from OCR coordinates.
"""
from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Literal

import cv2
import numpy as np
import pydantic
from pydantic import BaseModel, Field

from configs import app_logger as logger
from configs.settings import settings
from parser_client import gemini_client
from templates import precheck, store
from templates.drift import (
    SHORT_TEXT,
    STATIC_MIN_SIMILARITY,
    STATIC_MIN_SIMILARITY_SHORT,
    Box,
    Rect,
    _by_line,
    _center,
    _contains,
    _norm,
    _static_runs,
    _union,
    check_value,
)

# Gemini's reading of a value and the OCR words it points at must be this similar.
VALUE_MATCH = 0.6
# Value boxes are widened into the free space beside the value, so longer values of other
# documents still fit: up to the neighbouring text on the line (less this share of the line
# height), else by this many line heights (left) or to the page edge (right); never over another
# field's value.
NEIGHBOUR_GAP = 0.3
NO_NEIGHBOUR_LEFT = 3
LINE_PAD = 0.15
# Image regions are masked this much (share of the page) past their box.
IMAGE_REGION_PAD = 0.01
# An image region larger than this share of the page is printed design or a misplaced box rather
# than a photo or code: it is not masked.
MAX_IMAGE_REGION_AREA = 0.25
# Static elements need this many letters/digits to be found again reliably (labels: fewer).
MIN_STATIC_CHARS = 3
MIN_LABEL_CHARS = 2


class DraftingError(Exception):
    """Gemini's answer could not be turned into a template."""


# ---------------------------------------------------------------------------
# Gemini's answer
# ---------------------------------------------------------------------------

class DraftField(BaseModel):
    key: str = Field(description="Field key (snake_case)")
    value: str = Field(description="The value as printed, read from the image")
    value_words: list[int] = Field(description="Numbers of exactly the OCR words that make up the value, in reading order")
    label_words: list[int] = Field(description="Numbers of the OCR words of this value's own printed label; empty when it has none")
    fixed_position: bool = Field(description="Printed at the same place on every document of this layout")
    multiline: bool = Field(description="The value can run over several lines")
    check: Literal["date", "pan", "aadhaar", "number", "none"] = Field(description="Format the value always has")


class ImageRegion(BaseModel):
    kind: Literal["photo", "qr_code", "barcode", "signature", "other"]
    box_2d: list[int] = Field(description="[ymin, xmin, ymax, xmax], each scaled 0-1000 to the image")


class TemplateDraft(BaseModel):
    doc_type: str
    layout_name: str = Field(description="Issuer and document, e.g. 'Infosys salary slip'; never a person's name")
    fields: list[DraftField]
    static_words: list[int] = Field(description="Numbers of the words printed identically on every document of this layout")
    personal_words: list[int] = Field(description="Numbers of the words with personal or transaction data that is not a requested field")
    image_regions: list[ImageRegion] = Field(description="Photos, QR codes, barcodes and signatures; never logos or emblems")
    markers: list[list[int]] = Field(description="2 to 4 phrases, each the numbers of consecutive words of one line, printed on every document of this doc_type and seldom on other documents")


@dataclass
class Draft:
    """A validated draft, ready to be saved as a template."""
    doc_type: str
    name: str
    fields: list[dict]                 # template fields; anchor "static:<id>", "page" or "auto"
    static_elements: list[dict]
    personal: list[Rect]               # boxes of the value words and other personal words
    image_regions: list[Rect]          # photos, QR codes, barcodes, signatures
    values: dict[str, str]             # each field's value as Gemini read it
    markers: list[str] = field(default_factory=list)  # phrases proposed as the doc type's markers
    warnings: list[str] = field(default_factory=list)
    llm_elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Drafting
# ---------------------------------------------------------------------------

def known_doc_types() -> dict[str, dict]:
    """Document types with the keys to read, from the doc_types table."""
    return store.list_doc_types()


def record_doc_type(doc_type: str, keys: list[str]) -> None:
    """Remember the keys of a doc_type Gemini named while drafting, so later layouts of it are asked
    for the same keys (plus any it adds)."""
    types = store.list_doc_types()
    existing = types.get(doc_type)
    if existing is not None and existing["source"] == "seed":
        return
    fields = list(dict.fromkeys([*(existing["fields"] if existing else []), *keys]))
    store.save_doc_type(doc_type=doc_type, name=doc_type.replace("_", " "), fields=fields, source="gemini")


async def draft_template(image: np.ndarray, words: list[Box], doc_type: str | None) -> Draft:
    """Ask Gemini to map this page to a template, and validate its answer against the OCR."""
    if not words:
        raise DraftingError("No text was found on the page.")
    if doc_type and (other := precheck.other_type(doc_type, words)):
        logger.info("template not drafted: requested doc_type=%s, page shows %r of %s", doc_type, other[1], other[0])
        raise DraftingError(
            f"The document looks like a '{other[0]}' (it reads \"{other[1]}\"), not a '{doc_type}'; no template was created."
        )
    types = known_doc_types()
    height, width = image.shape[:2]
    prompt = build_prompt(words, (width, height), doc_type, types)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise DraftingError("Could not encode the page for Gemini.")

    started_at = time.perf_counter()
    try:
        result = await gemini_client.gemini_request(
            settings.GEMINI_MODEL_NAME, prompt, encoded.tobytes(), "page.jpg",
            response_schema=TemplateDraft, max_side=2048,
        )
    except gemini_client.GeminiError:
        raise
    except Exception as exc:
        raise gemini_client.GeminiError(f"Gemini request failed: {exc}") from exc
    llm_elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    try:
        answer = TemplateDraft.model_validate_json(result["response"])
    except pydantic.ValidationError as exc:
        raise DraftingError(f"Gemini's answer is not a valid template draft: {exc.errors()[:3]}") from exc

    draft = validate_draft(answer, words, (width, height), doc_type, types)
    draft.llm_elapsed_ms = llm_elapsed_ms
    logger.info(
        "template drafted doc_type=%s name=%s fields=%s statics=%s personal_words=%s image_regions=%s markers=%s warnings=%s llm_ms=%s",
        draft.doc_type, draft.name, len(draft.fields), len(draft.static_elements), len(draft.personal),
        len(draft.image_regions), draft.markers, len(draft.warnings), llm_elapsed_ms,
    )
    return draft


def build_prompt(words: list[Box], page_size: tuple[int, int], doc_type: str | None, types: dict[str, dict]) -> str:
    width, height = page_size
    if doc_type and types.get(doc_type, {}).get("fields"):
        spec = types[doc_type]
        document = (
            f'This document is a "{doc_type}" ({spec["name"]}). Requested fields:\n'
            + "\n".join(f"- {key}" for key in spec["fields"])
            + (f"\nNote: {spec['notes']}" if spec.get("notes") else "")
            + f'\nReturn doc_type "{doc_type}" exactly. Only if the page is clearly a different kind of document, '
            "return a short snake_case doc_type for what it really is (then no template is made)."
        )
    elif doc_type:
        document = (
            f'This document is a "{doc_type}". No fields are defined for it yet: request the information a '
            "lender needs from it (names, identifiers, dates, amounts, addresses), with short snake_case keys.\n"
            f'Return doc_type "{doc_type}" exactly. Only if the page is clearly a different kind of document, '
            "return a short snake_case doc_type for what it really is (then no template is made)."
        )
    else:
        listing = "\n".join(
            f"- {key} ({spec['name']}): {', '.join(spec['fields'])}" + (f". {spec['notes']}" if spec.get("notes") else "")
            for key, spec in types.items()
        )
        document = (
            "Known document types and their fields:\n" + listing + "\n"
            "If this document is one of these types, return that doc_type and use its fields. Otherwise return a "
            "new short snake_case doc_type and request the information a lender needs from the document (names, "
            "identifiers, dates, amounts, addresses), with short snake_case keys."
        )

    return f"""You prepare a reusable extraction template for one document layout. Later documents of the same layout (same issuer and design, other people and values) will be read with it automatically, so decide what holds for every document of this layout, not just this one.

You get the page image and the page's OCR words. Each OCR word has a number in square brackets. Words are listed per printed line, top to bottom: each line shows its vertical pixel range, each word its horizontal pixel range, in an image of {width} x {height} pixels.

{document}

Return:
1. doc_type, as described above.
2. layout_name: a short name for this layout, issuer and document, e.g. "Infosys salary slip", "HDFC Bank account statement", "UIDAI e-Aadhaar". Never a person's name.
3. fields: one entry per requested field that is printed on this page. Leave out fields that are not printed; never guess.
   - value: the value as printed, read from the image.
   - value_words: the numbers of exactly the OCR words that make up the value, in reading order. Only the value: no label words, no separators such as ":". If one OCR word joins label and value (e.g. "DOB/27/01/2003"), use that word.
   - label_words: the numbers of the words of this value's own printed label (e.g. "Date of Birth", "Net Pay :"), just before the value on the same line or directly above it. If the same label text is printed more than once on the page, include enough neighbouring words of the label to make it unique (e.g. "Date of Issue" rather than "Date"). Empty when the value has no label of its own (e.g. the holder's name on an ID card); never use a title, logo or heading elsewhere on the page as a label.
   - fixed_position: true when the value is printed at the same place on every document of this layout whatever the other content (e.g. the number on an ID card); false when content above or before it can push it (e.g. lines after a multi-line address, rows of a table).
   - multiline: true when the value can run over several lines (e.g. an address).
   - check: "date", "pan", "aadhaar" or "number" when the value always has that format, else "none".
4. static_words: the numbers of every word printed identically on all documents of this layout: titles, headings, labels (including those of step 3), column headers, fixed notes and footers. Never values, never anything about this person or transaction.

For label_words and static_words use only words whose OCR text is a correct reading of what is printed: compare with the image and skip words the OCR misread, e.g. text in another script (Hindi) read as random Latin letters. They are found again on other documents by their text.
5. personal_words: the numbers of the words with data about this person or transaction that are not a requested field's value (other names, numbers, dates, amounts, addresses), also when the OCR misread them (e.g. the name in another script). They are erased from the stored layout. Words printed on every document (headings, notes, text in another script) are never personal, even when the OCR misread them.
6. image_regions: photos, QR codes, barcodes and signatures, with box_2d as [ymin, xmin, ymax, xmax] scaled 0-1000. They are erased from the stored layout. Never logos, emblems, seals, flags or other artwork printed on every document: those identify the layout.
7. markers: 2 to 4 phrases that tell this document type apart from other documents: printed on every document of this doc_type whatever its issuer or layout, and seldom on other documents. Each phrase is the numbers of consecutive, correctly read OCR words of one line. Good: "Permanent Account Number" on a PAN card, "Unique Identification Authority" on an Aadhaar, "Registration Certificate" on a vehicle RC. Never generic text that other documents print too, such as "Government of India", "Name", "Date of Birth", "Address", "Signature", or a bank's or company's name.

Use only word numbers from this list.

OCR words:
{format_words(words)}
"""


def format_words(words: list[Box]) -> str:
    lines = _by_line(words, list(range(len(words))))
    out = []
    for number, line in enumerate(sorted(lines.values(), key=lambda items: min(w.rect[1] for _, w in items)), start=1):
        top = round(min(w.rect[1] for _, w in line))
        bottom = round(max(w.rect[3] for _, w in line))
        cells = " | ".join(f"[{i}] {json.dumps(w.text, ensure_ascii=False)} x{round(w.rect[0])}-{round(w.rect[2])}" for i, w in line)
        out.append(f"line {number} (y {top}-{bottom}): {cells}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_draft(
    answer: TemplateDraft,
    words: list[Box],
    page_size: tuple[int, int],
    requested_doc_type: str | None,
    types: dict[str, dict],
) -> Draft:
    """Check Gemini's answer against the OCR and turn it into template fields, static elements
    and masks. Problems are fixed or the item is left out, each with a warning."""
    warnings: list[str] = []
    width, height = page_size

    def known(ids: list[int], what: str) -> list[int]:
        good = [i for i in dict.fromkeys(ids) if 0 <= i < len(words)]
        if len(good) < len(set(ids)):
            warnings.append(f"{what}: ignored word numbers that are not on the page")
        return good

    answered = _snake_case(answer.doc_type)
    if requested_doc_type and answered and answered != requested_doc_type:
        raise DraftingError(
            f"The document looks like a '{answered}', not a '{requested_doc_type}'; no template was created."
        )
    doc_type = requested_doc_type or answered or "unknown_document"
    spec = types.get(doc_type)
    # Keys of seeded types are fixed; types Gemini created may gain keys
    allowed = set(spec["fields"]) if spec and spec.get("fields") and spec.get("source") == "seed" else None

    # Fields: value words must exist, belong to one field only, and read like the value
    kept: list[tuple[DraftField, list[int], str | None]] = []
    value_words: set[int] = set()
    values: dict[str, str] = {}
    for item in answer.fields:
        key = item.key.strip()
        if key in values:
            warnings.append(f"{key}: listed twice; kept the first")
            continue
        if allowed is not None and key not in allowed and not _group_key(key, spec):
            warnings.append(f"{key}: not a field of {doc_type}; left out")
            continue
        ids = known(item.value_words, key)
        if not ids:
            warnings.append(f"{key}: no value words; left out")
            continue
        if value_words & set(ids):
            warnings.append(f"{key}: its words are already another field's value; left out")
            continue
        ids = _trim_value_words(ids, words, item.value)
        ocr_text = " ".join(words[i].text for i in ids)
        if not same_text(ocr_text, item.value):
            warnings.append(f"{key}: the words given ({ocr_text!r}) do not read as the value {item.value!r}; left out")
            continue
        check = None if item.check == "none" else item.check
        if check and check_value(check, ocr_text) is None:
            warnings.append(f"{key}: {ocr_text!r} is not a {check}; check dropped")
            check = None
        value_words.update(ids)
        values[key] = item.value
        kept.append((item, ids, check))
    if not kept:
        raise DraftingError("Gemini's draft has no usable fields: " + "; ".join(warnings or ["none were returned"]))

    # A multi-line value whose first line is another field's value ("To / <name> / <address>")
    # starts after it; the line's words stay value words, so they are still erased
    for n, (item, ids, check) in enumerate(kept):
        first = [i for i in ids if words[i].line == words[ids[0]].line]
        rest = [i for i in ids if i not in first]
        text = " ".join(words[i].text for i in first)
        if item.multiline and rest and any(other is not item and same_text(text, other.value) for other, _, _ in kept):
            kept[n] = (item, rest, check)
            values[item.key.strip()] = "\n".join(item.value.splitlines()[1:]) or item.value
            warnings.append(f"{item.key}: starts with another field's value {text!r}; that line left out")

    # Labels: one static element per label, on the label line nearest the value
    statics: list[dict] = []
    label_of: dict[str, int] = {}
    used: set[int] = set(value_words)
    static_ids = set(known(answer.static_words, "static_words")) - value_words
    # A word ending in ":" outside every value is a printed label ("Mobile:"), also when Gemini left
    # it out: it is kept on the reference and stops multi-line values above it. Inside a value's area
    # it is part of the value ("VTC:", "District:" in an address).
    value_areas = [_union([words[i].rect for i in ids]) for _, ids, _ in kept]
    static_ids |= {
        i for i, w in enumerate(words)
        if i not in value_words and w.text.rstrip().endswith(":") and len(_norm(w.text)) >= MIN_STATIC_CHARS
        and not any(_contains(area, _center(w.rect)) for area in value_areas)
    }
    normalized = [_norm(w.text) for w in words]
    for item, ids, _ in kept:
        label = [i for i in known(item.label_words, f"{item.key} label") if i not in value_words]
        if not label:
            continue
        lines: dict[int, list[int]] = {}
        for i in label:
            lines.setdefault(words[i].line, []).append(i)
        value_rect = _union([words[i].rect for i in ids])
        label = min(lines.values(), key=lambda line: abs(_center(_union([words[i].rect for i in line]))[1] - _center(value_rect)[1]))
        label = sorted(label, key=lambda i: words[i].rect[0])
        if not _is_label_of(_union([words[i].rect for i in label]), value_rect):
            warnings.append(f"{item.key}: label {' '.join(words[i].text for i in label)!r} is not before or above the value; not used as anchor")
            continue
        existing = next((n for n, s in enumerate(statics) if s["words"] == label), None)
        if existing is not None:
            label_of[item.key] = existing
            continue
        label = _unique_label(label, words, normalized, static_ids | set(label), item.key, warnings)
        text = " ".join(words[i].text for i in label)
        if len(_norm(text)) < MIN_LABEL_CHARS:
            warnings.append(f"{item.key}: label {text!r} is too short to find again; not used as anchor")
            continue
        statics.append({"text": text, "bbox": _union([words[i].rect for i in label]), "words": label})
        label_of[item.key] = len(statics) - 1
        used.update(label)

    # Other static elements: runs of static words on a line
    for line in _by_line(words, list(range(len(words)))).values():
        run: list[int] = []
        for i, _ in line + [(None, None)]:
            if i is not None and i in static_ids and i not in used:
                run.append(i)
                continue
            text = " ".join(words[j].text for j in run)
            if run and len(_norm(text)) >= MIN_STATIC_CHARS:
                statics.append({"text": text, "bbox": _union([words[j].rect for j in run]), "words": run})
                used.update(run)
            run = []

    order = sorted(range(len(statics)), key=lambda n: (statics[n]["bbox"][1], statics[n]["bbox"][0]))
    static_id = {n: f"s{rank}" for rank, n in enumerate(order, start=1)}
    static_elements = [
        {"id": static_id[n], "text": statics[n]["text"], "bbox": [round(v) for v in statics[n]["bbox"]]} for n in order
    ]

    value_rects = {item.key: _union([words[i].rect for i in ids]) for item, ids, _ in kept}
    fields = [
        {
            "key": item.key.strip(),
            "bbox": _value_box(ids, words, page_size, [r for k, r in value_rects.items() if k != item.key]),
            "anchor": f"static:{static_id[label_of[item.key]]}" if item.key in label_of else "page" if item.fixed_position else "auto",
            "multiline": item.multiline,
            "check": check,
        }
        for item, ids, check in kept
    ]

    # Erased from the stored reference: field values and personal words, photos and codes. Words
    # that are neither (printed text the OCR misread) stay: they are part of the layout.
    personal_ids = value_words | (set(known(answer.personal_words, "personal_words")) - used)
    # Consecutive personal words of a line as one box, so the gaps between them are covered too
    personal: list[Rect] = []
    for line in _by_line(words, list(range(len(words)))).values():
        run: list[int] = []
        for i, _ in line + [(None, None)]:
            if i is not None and i in personal_ids:
                run.append(i)
                continue
            if run:
                personal.append(_union([words[j].rect for j in run]))
            run = []
    # Photos and codes are erased; static text they overlap stays readable (see _masked_reference)
    image_regions: list[Rect] = []
    for region in answer.image_regions:
        if len(region.box_2d) != 4 or not (0 <= region.box_2d[0] < region.box_2d[2] <= 1000 and 0 <= region.box_2d[1] < region.box_2d[3] <= 1000):
            warnings.append(f"{region.kind}: box {region.box_2d} is not [ymin, xmin, ymax, xmax] within 0-1000; ignored")
            continue
        ymin, xmin, ymax, xmax = region.box_2d
        pad_x, pad_y = IMAGE_REGION_PAD * width, IMAGE_REGION_PAD * height
        rect = (xmin / 1000 * width - pad_x, ymin / 1000 * height - pad_y, xmax / 1000 * width + pad_x, ymax / 1000 * height + pad_y)
        if (rect[2] - rect[0]) * (rect[3] - rect[1]) > MAX_IMAGE_REGION_AREA * width * height:
            warnings.append(f"{region.kind}: box {region.box_2d} covers over {MAX_IMAGE_REGION_AREA:.0%} of the page; not erased")
            continue
        image_regions.append(rect)

    # Markers: printed text of one line, never a value or personal data (their checks against
    # other doc types follow in precheck.propose)
    markers: list[str] = []
    for ids in answer.markers:
        ids = known(ids, "markers")
        if not ids:
            continue
        text = " ".join(words[i].text for i in sorted(ids, key=lambda i: words[i].rect[0])).strip(" ,.;:/-")
        if personal_ids & set(ids):
            warnings.append(f"marker {text!r} left out: it is a value or personal data")
        elif len({words[i].line for i in ids}) > 1:
            warnings.append(f"marker {text!r} left out: its words are on different lines")
        else:
            markers.append(text)

    name = re.sub(r"\s+", " ", answer.layout_name).strip()[:80] or f"{doc_type} layout"
    return Draft(
        doc_type=doc_type, name=name, fields=fields, static_elements=static_elements,
        personal=personal, image_regions=image_regions, values=values, warnings=warnings, markers=markers,
    )


def _unique_label(label: list[int], words: list[Box], normalized: list[str], static_ids: set[int], key: str, warnings: list[str]) -> list[int]:
    """The label words, extended with neighbouring static words of its line while its text is
    printed elsewhere on the page too."""
    line = sorted((i for i, w in enumerate(words) if w.line == words[label[0]].line), key=lambda i: words[i].rect[0])
    for _ in range(3):
        count = _occurrences(" ".join(words[i].text for i in label), words, normalized)
        if count <= 1:
            return label
        start, end = line.index(label[0]), line.index(label[-1])
        before = line[start - 1] if start > 0 else None
        after = line[end + 1] if end + 1 < len(line) else None
        if before is not None and before in static_ids:
            label = [before] + label
        elif after is not None and after in static_ids:
            label = label + [after]
        else:
            break
    warnings.append(f"{key}: label {' '.join(words[i].text for i in label)!r} is printed {count} times; it is told apart by position")
    return label


def _is_label_of(label: Rect, value: Rect) -> bool:
    """A label sits before the value on its (first) line, or just above it, near its left edge or
    over it."""
    label_height = label[3] - label[1]
    same_line = (
        min(label[3], value[3]) - max(label[1], value[1]) >= 0.5 * label_height
        and label[2] <= value[0] + label_height
    )
    above = (
        label[3] <= value[1] + 0.5 * label_height
        and value[1] - label[3] <= 3 * label_height
        and (min(label[2], value[2]) > max(label[0], value[0]) or abs(label[0] - value[0]) <= 3 * label_height)
    )
    return same_line or above


def _occurrences(text: str, words: list[Box], normalized: list[str]) -> int:
    """Places on the page that read like ``text`` (overlapping matches count once)."""
    target = _norm(text)
    if not target:
        return 0
    threshold = STATIC_MIN_SIMILARITY_SHORT if len(target) < SHORT_TEXT else STATIC_MIN_SIMILARITY
    runs = _static_runs(list(enumerate(words)), target, threshold, len(text.split()) + 2, normalized)
    groups: list[set[int]] = []
    for _, _, indices in runs:
        group = set(indices)
        for other in [g for g in groups if g & group]:
            group |= other
            groups.remove(other)
        groups.append(group)
    return len(groups)


def _value_box(ids: list[int], words: list[Box], page_size: tuple[int, int], other_values: list[Rect]) -> list[float]:
    """The value's words, widened into the free space beside them on their lines, stopping at
    other text on those lines and at any other field's value."""
    width, height = page_size
    rects = [words[i].rect for i in ids]
    x1, y1, x2, y2 = _union(rects)
    line_height = statistics.median(r[3] - r[1] for r in rects)
    members = set(ids)
    beside = [
        w.rect for i, w in enumerate(words)
        if i not in members and min(y2, w.rect[3]) - max(y1, w.rect[1]) >= 0.5 * min(line_height, w.rect[3] - w.rect[1])
    ] + [r for r in other_values if min(y2, r[3]) > max(y1, r[1])]
    tolerance = 0.2 * line_height
    lefts = [r[2] for r in beside if r[2] <= x1 + tolerance]
    rights = [r[0] for r in beside if r[0] >= x2 - tolerance]
    gap = NEIGHBOUR_GAP * line_height
    left = max(lefts) + gap if lefts else x1 - NO_NEIGHBOUR_LEFT * line_height
    right = min(rights) - gap if rights else width * 0.99
    left, right = max(0.0, min(left, x1)), min(float(width), max(right, x2))
    pad = LINE_PAD * line_height
    return [round(left, 1), round(max(0.0, y1 - pad), 1), round(right, 1), round(min(float(height), y2 + pad), 1)]


def _trim_value_words(ids: list[int], words: list[Box], value: str) -> list[int]:
    """Drop words at either end while the rest reads more like the value Gemini read."""
    target = _norm(value)
    likeness = lambda part: SequenceMatcher(None, _norm(" ".join(words[i].text for i in part)), target).ratio()
    while len(ids) > 1:
        best = max((ids[1:], ids[:-1]), key=likeness)
        if likeness(best) <= likeness(ids):
            break
        ids = best
    return ids


def same_text(ocr_text: str, value: str) -> bool:
    a, b = _norm(ocr_text), _norm(value)
    if not a or not b:
        return False
    return a in b or b in a or SequenceMatcher(None, a, b).ratio() >= VALUE_MATCH


def _group_key(key: str, spec: dict) -> bool:
    """Keys like "allowances.HRA", when the doc type's notes describe the "allowances." group."""
    prefix, dot, item = key.partition(".")
    return bool(dot and item and f'"{prefix}.' in spec.get("notes", ""))


def _snake_case(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
