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


class ParseWithReferenceResponse(ParseResponse):
    alignment: list[PageAlignment]


class ParseTextWithReferenceResponse(ParseTextResponse):
    alignment: list[PageAlignment]


class ExtractionWithReferenceResponse(ExtractionResponse):
    alignment: list[PageAlignment]


class ProcessWithReferenceResponse(ProcessResponse):
    alignment: list[PageAlignment]
