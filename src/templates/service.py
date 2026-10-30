from __future__ import annotations

import copy
import math
import re
import statistics
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pydantic
from fastapi import UploadFile

from alignment import service as alignment_service
from configs import app_logger as logger
from normalize import service as normalize_service
from parser_client import gemini_client
from pipeline import service as pipeline_service
from pipeline.models import DocumentProcessingError, ValidationError
from templates import drafting, precheck, store
from templates.drift import (
    Box,
    derive_statics,
    extract_fields,
    locate_statics,
    ordered_by_anchor,
    rect_of,
    resolve_auto_anchor,
)
from templates.models import (
    DocTypeInfo,
    DocTypeMarker,
    FieldLearning,
    SampleValue,
    StaticLearning,
    TemplateCandidate,
    TemplateCreated,
    TemplateDetail,
    TemplateExtractionResponse,
    TemplateFieldIn,
    TemplateFieldResult,
    TemplateLearning,
    TemplateSummary,
)

# A template matches only with at least this many keypoint matches agreeing with the alignment.
# In tests, uploads of a template's layout scored 395+ (a dim, blurry photo) and other layouts under 100.
TEMPLATE_MIN_INLIERS = 200
# A best match between this and TEMPLATE_MIN_INLIERS is probably a known layout captured badly, not
# a new layout: no template is drafted for it, so the store doesn't fill up with duplicates.
NEAR_MATCH_INLIERS = 120
# Learning needs this many matched documents before it marks static elements as variable or
# suggests a different anchor.
LEARNING_MIN_MATCHES = 3
# Field values are blanked this far past their box in the stored reference image.
MASK_PADDING = 8
# Personal words of a drafted template are blanked this share of their line height past their
# OCR box (glyphs, accents and vowel signs reach past it)
PERSONAL_MASK_PADDING = 0.3

_fields_adapter = pydantic.TypeAdapter(list[TemplateFieldIn])
_regions_adapter = pydantic.TypeAdapter(list[tuple[float, float, float, float]])
# SIFT features of each template's reference image, computed on first use (references never change)
_reference_features: dict[int, tuple[tuple[int, int], Any]] = {}


async def create_template(
    *,
    doc_type: str,
    name: str | None,
    sample: UploadFile,
    fields_json: str,
    mask_regions_json: str | None,
) -> TemplateCreated:
    """Create a template from a sample document (page 1) and the field boxes picked on it."""
    doc_type = doc_type.strip()
    if not doc_type:
        raise ValidationError("doc_type is required.")
    fields_in = _parse(_fields_adapter, fields_json, "fields")
    if not fields_in:
        raise ValidationError("Provide at least one field.")
    if len({f.key for f in fields_in}) != len(fields_in):
        raise ValidationError("Field keys must be unique.")
    masks = _parse(_regions_adapter, mask_regions_json, "mask_regions") if mask_regions_json else []

    request_id = uuid.uuid4().hex[:12]
    try:
        with tempfile.TemporaryDirectory(prefix="template_") as tmp:
            page_paths, _, filename = alignment_service._prepare([sample], Path(tmp) / "sample", request_id)
            image = alignment_service._read_image(page_paths[0])
            words = _ocr(page_paths[0])
    finally:
        await sample.close()

    fields = [
        {"key": f.key, "bbox": [round(v, 1) for v in f.bbox], "anchor": f.anchor, "multiline": f.multiline, "check": f.check}
        for f in fields_in
    ]
    statics = derive_statics(words, [tuple(f["bbox"]) for f in fields] + [tuple(m) for m in masks])
    template, sample_results, _ = _save_template(
        doc_type=doc_type,
        name=(name or "").strip() or f"{doc_type} ({filename})",
        image=image,
        words=words,
        fields=fields,
        statics=statics,
        # Field boxes are picked by hand around the values: blanked whole, as the mask regions
        reference_png=_masked_reference(image, [tuple(f["bbox"]) for f in fields] + [tuple(m) for m in masks]),
    )
    detail = _detail(template)
    return TemplateCreated(
        **detail.model_dump(),
        sample_values=[SampleValue(key=r.key, value=r.value) for r in sample_results],
    )


def list_doc_types() -> list[DocTypeInfo]:
    return [
        DocTypeInfo(
            doc_type=doc_type,
            name=info["name"],
            fields=info["fields"],
            source=info["source"],
            markers=[DocTypeMarker(**{"trusted": False, **m}) for m in info["markers"]],
        )
        for doc_type, info in store.list_doc_types().items()
    ]


def list_templates(doc_type: str | None) -> list[TemplateSummary]:
    return [_summary(t) for t in store.list_templates(doc_type)]


def get_template(template_id: int) -> TemplateDetail | None:
    template = store.get(template_id)
    return _detail(template) if template else None


def template_reference(template_id: int) -> bytes | None:
    """The stored reference page (PNG, field values and masked regions blanked)."""
    template = store.get(template_id, with_reference=True)
    return template["reference_png"] if template else None


def delete_template(template_id: int) -> bool:
    _reference_features.pop(template_id, None)
    return store.delete(template_id)


async def extract_with_templates(
    files: list[UploadFile],
    *,
    doc_type: str | None,
    template_id: int | None,
    learn: bool = True,
) -> TemplateExtractionResponse:
    """Read the fields of page 1 of the upload with the best matching stored template.

    Templates of ``doc_type`` are tried (all templates without one, only ``template_id`` with one).
    When none matches and ``learn`` is on, the page is normalized and OCR'd, Gemini drafts a
    template from it, and that template is saved and used to read the fields: one OCR either way.
    """
    started_at = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]
    uploads = [upload for upload in files if upload.filename]
    if not uploads:
        raise ValidationError("Upload at least one file.")
    doc_type = (doc_type or "").strip() or None
    if template_id is not None:
        template = store.get(template_id, with_reference=True)
        if template is None:
            raise ValidationError(f"Template {template_id} does not exist.")
        templates = [template]
    else:
        templates = store.list_templates(doc_type, with_reference=True)
    can_learn = learn and template_id is None and gemini_client.is_configured()

    candidates: list[TemplateCandidate] = []
    best: tuple[dict, int, np.ndarray] | None = None
    normalized: normalize_service.Normalized | None = None
    near_match = False
    try:
        with tempfile.TemporaryDirectory(prefix="template_extract_") as tmp:
            page_paths, _, filename = alignment_service._prepare(uploads, Path(tmp) / "document", request_id)
            image = alignment_service._read_image(page_paths[0])
            page_features = alignment_service._features(image)
            for template in templates:
                shape, features = _features_of(template)
                warped, alignment = alignment_service._align_page(image, shape, features, index=0, reference_index=0, page_features=page_features)
                candidates.append(TemplateCandidate(template_id=template["id"], name=template["name"], aligned=alignment.aligned, inliers=alignment.inliers))
                confident = alignment.aligned and alignment.inliers >= TEMPLATE_MIN_INLIERS
                if confident and (best is None or alignment.inliers > best[1]):
                    best = (template, alignment.inliers, warped)
            near_match = best is None and any(c.inliers >= NEAR_MATCH_INLIERS for c in candidates)
            if best is not None:
                aligned_path = Path(tmp) / "aligned.png"
                cv2.imwrite(str(aligned_path), best[2])
                words = _ocr(aligned_path)
            elif can_learn and not near_match:
                normalized = normalize_service.normalize_image(image)
                normalized_path = Path(tmp) / "normalized.png"
                cv2.imwrite(str(normalized_path), normalized.image)
                words = _ocr(normalized_path)
    finally:
        for upload in uploads:
            await upload.close()

    summary = {"request_id": request_id, "filename": filename, "page_count": len(page_paths), "candidates": candidates}
    if best is None and normalized is not None:
        return await _create_from_document(normalized, words, doc_type, summary, started_at, request_id)
    if best is None:
        if near_match:
            reason = "a stored template nearly matched (the capture may be poor), so no new template was drafted"
        elif not learn or template_id is not None:
            reason = "learning is off"
        elif not gemini_client.is_configured():
            reason = "GEMINI_API_KEY is not set, so no template could be drafted"
        else:
            reason = "no template could be drafted"
        logger.info("request_id=%s no template matched doc_type=%s candidates=%s", request_id, doc_type, [(c.template_id, c.inliers) for c in candidates])
        return TemplateExtractionResponse(
            **summary,
            matched=False,
            doc_type=doc_type,
            fields=[],
            message="No template matched" + ("" if templates else " (none stored for this doc_type)") + f"; {reason}.",
            elapsed_ms=_elapsed(started_at),
        )

    template = best[0]
    page_size = (template["page_width"], template["page_height"])
    statics = template["static_elements"]
    unstable = _unstable_statics(template)
    static_matches = locate_statics(statics, words, page_size)
    results = extract_fields(template["fields"], statics, static_matches, words, page_size, unstable_statics=unstable)
    store.record_match(template["id"], _learn(template, static_matches, results))
    # A matched document is confirmed to be of the template's type: its markers learn from it
    precheck.observe(template["doc_type"], words)

    stable = [s for s in statics if s["id"] not in unstable]
    found = sum(1 for s in stable if static_matches[s["id"]].found is not None)
    logger.info("request_id=%s matched template=%s inliers=%s statics=%s/%s", request_id, template["id"], best[1], found, len(stable))
    return TemplateExtractionResponse(
        **summary,
        matched=True,
        doc_type=template["doc_type"],
        template_id=template["id"],
        template_name=template["name"],
        static_found=f"{found}/{len(stable)}",
        fields=_field_results(results),
        elapsed_ms=_elapsed(started_at),
    )


async def _create_from_document(
    normalized: normalize_service.Normalized,
    words: list[Box],
    doc_type: str | None,
    summary: dict[str, Any],
    started_at: float,
    request_id: str,
) -> TemplateExtractionResponse:
    """Gemini drafts a template from the normalized page; it is saved and its fields read from the
    same OCR (the page is the template's own reference, so no second OCR is needed)."""
    draft = await drafting.draft_template(normalized.image, words, doc_type)
    template, results, found = _save_template(
        doc_type=draft.doc_type,
        name=_unique_name(draft.doc_type, draft.name),
        image=normalized.image,
        words=words,
        fields=draft.fields,
        statics=draft.static_elements,
        # Value and personal words with a margin, photos and codes, but labels next to them stay
        # readable; the personal words themselves are always blanked
        reference_png=_masked_reference(
            normalized.image,
            draft.personal,
            regions=draft.image_regions,
            keep=[tuple(s["bbox"]) for s in draft.static_elements],
            exact=draft.personal,
            line_padding=PERSONAL_MASK_PADDING,
        ),
    )
    drafting.record_doc_type(template["doc_type"], [f["key"] for f in template["fields"]])
    # The drafted document is confirmed to be of its type: other types' markers learn from it,
    # then the markers Gemini proposed for the type are added
    precheck.observe(template["doc_type"], words)
    markers, marker_notes = precheck.propose(template["doc_type"], draft.markers, words)
    warnings = [*normalized.warnings, *draft.warnings, *marker_notes]
    if markers:
        warnings.append(f"markers learned for {template['doc_type']} (not trusted yet): {', '.join(markers)}")
    for result in results:
        read = draft.values.get(result.key)
        if read is not None and not drafting.same_text(result.value or "", read):
            warnings.append(f"{result.key}: the new template reads {result.value!r}, Gemini read {read!r}")
    logger.info("request_id=%s created template=%s doc_type=%s fields=%s", request_id, template["id"], template["doc_type"], len(results))
    return TemplateExtractionResponse(
        **summary,
        matched=False,
        created=True,
        doc_type=template["doc_type"],
        template_id=template["id"],
        template_name=template["name"],
        static_found=f"{found}/{len(template['static_elements'])}",
        fields=_field_results(results),
        message="No stored template matched; Gemini drafted a template from this document, it was saved and used.",
        warnings=warnings,
        llm_elapsed_ms=draft.llm_elapsed_ms,
        elapsed_ms=_elapsed(started_at),
    )


def _save_template(
    *,
    doc_type: str,
    name: str,
    image: np.ndarray,
    words: list[Box],
    fields: list[dict],
    statics: list[dict],
    reference_png: bytes,
) -> tuple[dict, list, int]:
    """Resolve "auto" anchors, read the fields from the sample itself and store the template with
    its masked reference page. Returns the stored template, the sample's field results and how
    many statics were found."""
    height, width = image.shape[:2]
    for field in fields:
        if field["anchor"] == "auto":
            field["anchor"] = resolve_auto_anchor(field, fields, statics)
    try:
        ordered_by_anchor(fields)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    static_matches = locate_statics(statics, words, (width, height))
    sample_results = extract_fields(fields, statics, static_matches, words, (width, height))
    template_id = store.create(
        doc_type=doc_type,
        name=name,
        page_width=width,
        page_height=height,
        static_elements=statics,
        fields=fields,
        reference_png=reference_png,
    )
    logger.info("template created id=%s doc_type=%s statics=%s fields=%s", template_id, doc_type, len(statics), len(fields))
    found = sum(1 for m in static_matches.values() if m.found is not None)
    return store.get(template_id), sample_results, found


def _unique_name(doc_type: str, name: str) -> str:
    taken = {t["name"] for t in store.list_templates(doc_type)}
    candidate, number = name, 2
    while candidate in taken:
        candidate, number = f"{name} ({number})", number + 1
    return candidate


def _field_results(results: list) -> list[TemplateFieldResult]:
    return [
        TemplateFieldResult(
            key=r.key,
            value=r.value,
            status=r.status,
            valid=r.valid,
            anchor=r.anchor,
            drift=list(r.inherited),
            box=[round(v, 1) for v in r.found] if r.found else None,
        )
        for r in results
    ]


def _ocr(image_path: Path) -> list[Box]:
    """OCR one page, returning its words (each knowing its line)."""
    try:
        raw = pipeline_service.get_engine().predict_file(image_path, word_boxes=True)
    except Exception as exc:
        raise DocumentProcessingError(f"OCR failed: {type(exc).__name__}: {exc}") from exc
    words: list[Box] = []
    for line_id, line in enumerate(_ocr_lines(raw)):
        words.extend(_line_words(line, line_id))
    return words


def _ocr_lines(value: Any) -> list[dict]:
    if isinstance(value, dict):
        if "text" in value and "text_region" in value:
            return [value]
        value = list(value.values())
    if isinstance(value, list):
        return [line for child in value for line in _ocr_lines(child)]
    return []


def _line_words(line: dict, line_id: int) -> list[Box]:
    """Words of one OCR line, split at spaces and at wide gaps.

    OCR can join text that lies far apart into one token when what was between is unreadable to
    it ("DOB / जन्म तिथि : 27/01/2003" -> "DOB/27/01/2003"). The recogniser's own word pieces show
    the gap, so such a token is split there; characters between the parts go to the left part.
    """
    text = str(line.get("text") or "")
    confidence = pipeline_service._as_float(line.get("confidence"), default=1.0)
    left, top, right, bottom = rect_of(line["text_region"])
    # Character span and x-range of each recogniser piece in the line text
    pieces: list[tuple[int, int, float, float]] = []
    cursor = 0
    for piece, region in zip(line.get("text_word") or [], line.get("text_word_region") or []):
        start = text.find(piece, cursor)
        if start < 0 or not piece:
            continue
        x1, _, x2, _ = rect_of(region)
        pieces.append((start, start + len(piece), x1, x2))
        cursor = start + len(piece)
    char_width = (
        statistics.median((x2 - x1) / (end - start) for start, end, x1, x2 in pieces)
        if pieces else (right - left) / max(len(text), 1)
    )

    words: list[Box] = []
    for token in re.finditer(r"\S+", text):
        inside = [p for p in pieces if p[0] >= token.start() and p[1] <= token.end()]
        if not inside:
            x1 = left + (right - left) * token.start() / len(text)
            x2 = left + (right - left) * token.end() / len(text)
            words.append(Box(token.group(), (x1, top, x2, bottom), confidence, line_id))
            continue
        groups = [[inside[0]]]
        for previous, piece in zip(inside, inside[1:]):
            if piece[2] - previous[3] > (piece[0] - previous[1] + 2) * char_width:
                groups.append([piece])
            else:
                groups[-1].append(piece)
        for index, group in enumerate(groups):
            start = token.start() if index == 0 else group[0][0]
            end = groups[index + 1][0][0] if index + 1 < len(groups) else token.end()
            # Characters before the first / after the last piece (punctuation) widen the box
            x1 = max(left, group[0][2] - (group[0][0] - start) * char_width)
            x2 = min(right, group[-1][3] + (end - group[-1][1]) * char_width)
            words.append(Box(text[start:end], (x1, top, x2, bottom), confidence, line_id))
    return words


def _masked_reference(
    image: np.ndarray,
    padded: list[tuple],
    *,
    regions: list[tuple] = (),
    keep: list[tuple] = (),
    exact: list[tuple] = (),
    line_padding: float | None = None,
) -> bytes:
    """The sample page with its personal data blanked: only the fixed design is kept, which is
    what matching needs and keeps personal data out of the store.

    ``padded`` boxes are blanked with a margin (MASK_PADDING pixels, or ``line_padding`` times
    each box's height for boxes of one text line: OCR boxes are tighter than the ink) and
    ``regions`` (photos, codes) as they are, except over ``keep`` boxes (static elements, which
    must stay readable); ``exact`` boxes are then blanked as they are, also over static elements.
    """
    height, width = image.shape[:2]
    blank = np.zeros((height, width), dtype=bool)

    def box(x1: float, y1: float, x2: float, y2: float, pad: int) -> tuple[slice, slice]:
        return (
            slice(max(0, int(y1) - pad), min(height, int(np.ceil(y2)) + pad)),
            slice(max(0, int(x1) - pad), min(width, int(np.ceil(x2)) + pad)),
        )

    for rect in padded:
        pad = MASK_PADDING if line_padding is None else max(MASK_PADDING, round(line_padding * (rect[3] - rect[1])))
        blank[box(*rect, pad)] = True
    for rect in regions:
        blank[box(*rect, 0)] = True
    for rect in keep:
        blank[box(*rect, 0)] = False
    for rect in exact:
        blank[box(*rect, 0)] = True
    masked = image.copy()
    masked[blank] = 255
    ok, encoded = cv2.imencode(".png", masked, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise DocumentProcessingError("Could not encode the reference image.")
    return encoded.tobytes()


def _features_of(template: dict) -> tuple[tuple[int, int], Any]:
    cached = _reference_features.get(template["id"])
    if cached is None:
        image = cv2.imdecode(np.frombuffer(template["reference_png"], np.uint8), cv2.IMREAD_COLOR)
        cached = (image.shape[:2], alignment_service._features(image))
        _reference_features[template["id"]] = cached
    return cached


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------

def _learn(template: dict, static_matches: dict, results: list) -> dict[str, Any]:
    """Record which static elements were found and where each field was found, so repeated matches
    reveal static elements that are really variable and fields anchored to the wrong thing."""
    stats = copy.deepcopy(template["stats"]) or {}
    for static_id, match in static_matches.items():
        entry = stats.setdefault("statics", {}).setdefault(static_id, {"seen": 0, "found": 0})
        entry["seen"] += 1
        entry["found"] += match.found is not None
    for result in results:
        if result.found is None:
            continue
        entry = stats.setdefault("fields", {}).setdefault(result.key, {"n": 0, "page": [0.0] * 4, "anchor": [0.0] * 4})
        page = (result.shift.dx_left, result.shift.dy_top)
        relative = (page[0] - result.inherited[0], page[1] - result.inherited[1])
        entry["n"] += 1
        for name, (x, y) in (("page", page), ("anchor", relative)):
            acc = entry[name]
            acc[0] += x
            acc[1] += y
            acc[2] += x * x
            acc[3] += y * y
    return stats


def _unstable_statics(template: dict) -> set[str]:
    statics = (template.get("stats") or {}).get("statics", {})
    return {
        static_id
        for static_id, entry in statics.items()
        if entry["seen"] >= LEARNING_MIN_MATCHES and entry["found"] / entry["seen"] < 0.5
    }


def _spread(acc: list[float], n: int) -> float:
    var_x = max(0.0, acc[2] / n - (acc[0] / n) ** 2)
    var_y = max(0.0, acc[3] / n - (acc[1] / n) ** 2)
    return round(math.sqrt(var_x + var_y), 1)


def _learning(template: dict) -> TemplateLearning:
    stats = template.get("stats") or {}
    unstable = _unstable_statics(template)
    statics = [
        StaticLearning(id=s["id"], seen=e["seen"], found=e["found"], stable=s["id"] not in unstable)
        for s in template["static_elements"]
        if (e := stats.get("statics", {}).get(s["id"]))
    ]
    fields = []
    for field in template["fields"]:
        entry = stats.get("fields", {}).get(field["key"])
        if not entry:
            fields.append(FieldLearning(key=field["key"], observations=0))
            continue
        page, relative = _spread(entry["page"], entry["n"]), _spread(entry["anchor"], entry["n"])
        suggestion = None
        if entry["n"] >= LEARNING_MIN_MATCHES:
            if field["anchor"] == "page" and page > 15:
                suggestion = "auto"   # it moves around the page: anchor it to its label or the text above
            elif field["anchor"] != "page" and page + 5 < relative:
                suggestion = "page"   # steadier on the page than relative to its anchor
        fields.append(FieldLearning(key=field["key"], observations=entry["n"], page_spread_px=page, anchor_spread_px=relative, suggested_anchor=suggestion))
    return TemplateLearning(statics=statics, fields=fields)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summary(template: dict) -> TemplateSummary:
    return TemplateSummary(
        id=template["id"],
        doc_type=template["doc_type"],
        name=template["name"],
        page_width=template["page_width"],
        page_height=template["page_height"],
        static_count=len(template["static_elements"]),
        field_count=len(template["fields"]),
        match_count=template["match_count"],
        created_at=template["created_at"],
        updated_at=template["updated_at"],
    )


def _detail(template: dict) -> TemplateDetail:
    return TemplateDetail(
        **_summary(template).model_dump(),
        static_elements=template["static_elements"],
        fields=template["fields"],
        learning=_learning(template),
    )


def _parse(adapter: pydantic.TypeAdapter, raw: str, name: str) -> Any:
    try:
        return adapter.validate_json(raw)
    except pydantic.ValidationError as exc:
        problems = []
        for error in exc.errors():
            where = name + "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in error["loc"])
            problems.append(f"{where}: {error['msg'].removeprefix('Value error, ')}")
        raise ValidationError("Invalid " + "; ".join(problems)) from exc


def _elapsed(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)
