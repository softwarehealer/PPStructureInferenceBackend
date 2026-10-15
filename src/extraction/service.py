from __future__ import annotations

import pydantic
from fastapi import UploadFile

from configs.settings import settings
from extraction.models import ExtractedField, ExtractionResponse, FieldBox
from pipeline import service as pipeline_service
from pipeline.models import Granularity, OcrLine, ParsedPage, ValidationError

Rect = tuple[float, float, float, float]

_field_boxes = pydantic.TypeAdapter(list[FieldBox])


def parse_field_boxes(raw: str) -> list[FieldBox]:
    """Validate the JSON list of field boxes sent with the request."""
    try:
        field_boxes = _field_boxes.validate_json(raw)
    except pydantic.ValidationError as exc:
        raise ValidationError(f"Invalid fields: {exc}") from exc
    if not field_boxes:
        raise ValidationError("Provide at least one field.")
    return field_boxes


async def extract_by_coordinates(
    files: list[UploadFile],
    field_boxes: list[FieldBox],
    granularity: Granularity = "lines",
) -> ExtractionResponse:
    """Parse the document and return, for each field, the text inside its box.

    With "words", a box matches individual words, so it can pick a value out of a longer line.
    """
    parsed = await pipeline_service.parse_uploads(files, granularity)
    lines_by_page = [_page_lines(page) for page in parsed.pages]
    return ExtractionResponse(
        **pipeline_service.summarize(parsed).model_dump(),
        fields=[
            ExtractedField(
                key=field.key,
                value=_text_in_box(
                    lines_by_page[field.page],
                    field.bbox,
                    min_overlap=settings.EXTRACTION_MIN_OVERLAP,
                )
                if field.page < len(lines_by_page)
                else None,
            )
            for field in field_boxes
        ],
    )


def _page_lines(page: ParsedPage) -> list[OcrLine]:
    """OCR lines of one page, empty if the page failed."""
    if not page.success:
        return []
    return [line for line in page.result.res if line.text]


def _region_rect(text_region: list[list[float]]) -> Rect:
    xs = [point[0] for point in text_region]
    ys = [point[1] for point in text_region]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap_ratio(line: Rect, box: Rect) -> float:
    """Fraction of the line's area that lies inside the box."""
    width = min(line[2], box[2]) - max(line[0], box[0])
    height = min(line[3], box[3]) - max(line[1], box[1])
    if width <= 0 or height <= 0:
        return 0.0
    line_area = (line[2] - line[0]) * (line[3] - line[1])
    if line_area <= 0:
        return 0.0
    return (width * height) / line_area


def _text_in_box(lines: list[OcrLine], box: Rect, *, min_overlap: float) -> str | None:
    """Join the text of lines that fall inside the box, in reading order.

    Lines on the same row are joined with a space, rows with a newline.
    Returns None when no line matches.
    """
    matched = [
        (_region_rect(line.text_region), line.text.strip())
        for line in lines
        if _overlap_ratio(_region_rect(line.text_region), box) >= min_overlap
    ]
    if not matched:
        return None

    matched.sort(key=lambda item: (item[0][1] + item[0][3]) / 2)
    rows: list[list[tuple[Rect, str]]] = []
    for rect, text in matched:
        center_y = (rect[1] + rect[3]) / 2
        if rows:
            last = rows[-1][-1][0]
            # Same row when this line's center sits within the previous line's vertical span.
            if last[1] <= center_y <= last[3]:
                rows[-1].append((rect, text))
                continue
        rows.append([(rect, text)])

    return "\n".join(
        " ".join(text for _, text in sorted(row, key=lambda item: item[0][0])) for row in rows
    )
