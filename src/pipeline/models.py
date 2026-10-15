from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# Unit of each OCR entry: a detected text line, or a space-separated word within a line.
Granularity = Literal["lines", "words"]


class OcrLine(BaseModel):
    """One OCR entry: a line, or a word when granularity is "words"."""
    text: str
    # Words carry the confidence of the line they belong to
    confidence: float
    # Four [x, y] corner points in pixels of the page image
    text_region: list[list[float]]


class PageResult(BaseModel):
    """OCR result of one page in the viewer format."""
    type: str
    bbox: list[int | float]
    res: list[OcrLine]
    img_idx: int
    score: float


class ParsedPage(BaseModel):
    page_index: int
    image_name: str
    image_mime: str
    success: bool
    error: str | None = None
    result: PageResult
    elapsed_ms: float


class ParseResponse(BaseModel):
    request_id: str
    filename: str
    input_type: str
    status: str
    page_count: int
    successful_pages: int
    failed_pages: int
    elapsed_ms: float
    pages: list[ParsedPage]


class ParseSummary(BaseModel):
    """Request-level fields of a parse, shared by responses that reshape the pages."""
    request_id: str
    filename: str
    input_type: str
    status: str
    page_count: int
    elapsed_ms: float


class PageText(BaseModel):
    page_index: int
    success: bool
    error: str | None = None
    texts: list[str]


class ParseTextResponse(ParseSummary):
    pages: list[PageText]


class ValidationError(Exception):
    """raise if the request is invalid (400)"""


class DocumentProcessingError(Exception):
    """raise if the uploaded document could not be turned into pages (422)"""
