from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from pipeline.models import PageResult


class ViewerPage(BaseModel):
    page_index: int
    image_name: str
    image_mime: str
    # The page image the coordinates refer to, so the viewer draws boxes on exactly what was OCR'd
    image_base64: str
    success: bool
    error: str | None = None
    viewer_result: PageResult
    # PPStructure output as-is (layout regions, table html, ...), empty when the page failed
    raw_result: list[dict[str, Any]]
    elapsed_ms: float


class ProcessResponse(BaseModel):
    request_id: str
    filename: str
    input_type: str
    status: str
    page_count: int
    successful_pages: int
    failed_pages: int
    elapsed_ms: float
    pages: list[ViewerPage]
