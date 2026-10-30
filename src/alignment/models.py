from __future__ import annotations

from pydantic import BaseModel

from extraction.models import ExtractionResponse
from pipeline.models import ParseResponse, ParseTextResponse
from viewer.models import ProcessResponse


class PageAlignment(BaseModel):
    page_index: int
    # Page of the reference document this page was aligned to
    reference_page: int
    # False when no confident match was found; the page is then processed as uploaded
    aligned: bool
    # Keypoint matches that passed the ratio test, and how many of them agree with the final warp
    matches: int
    inliers: int
    elapsed_ms: float
    error: str | None = None


# Each response is the document's result (pages aligned to the reference) plus the same
# result for the reference itself, so both can be compared side by side.

class ParseWithReferenceResponse(ParseResponse):
    alignment: list[PageAlignment]
    reference: ParseResponse


class ParseTextWithReferenceResponse(ParseTextResponse):
    alignment: list[PageAlignment]
    reference: ParseTextResponse


class ExtractionWithReferenceResponse(ExtractionResponse):
    alignment: list[PageAlignment]
    reference: ExtractionResponse


class ProcessWithReferenceResponse(ProcessResponse):
    alignment: list[PageAlignment]
    reference: ProcessResponse


class PageImage(BaseModel):
    page_index: int
    image_name: str
    image_mime: str = "image/jpeg"
    image_base64: str
    width: int
    height: int


class PageImages(BaseModel):
    filename: str
    input_type: str
    page_count: int
    pages: list[PageImage]


class NormalizeWithReferenceResponse(PageImages):
    """Each page aligned to the reference (no OCR), and the reference's own pages."""
    alignment: list[PageAlignment]
    reference: PageImages
    elapsed_ms: float
