from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Check = Literal["aadhaar", "pan", "date", "number"]


class TemplateFieldIn(BaseModel):
    key: str = Field(min_length=1)
    # [x1, y1, x2, y2] in pixels of the sample page (as shown by /process)
    bbox: tuple[float, float, float, float]
    # "page" (fixed print position, the default), "auto" (the label to its left or the nearest
    # element above), or "field:<key>" (moves with that field)
    anchor: str = "page"
    # Value may run over several lines; it grows downwards past its box while lines continue
    multiline: bool = False
    # Format the value must have: the value is cut to the matching text, and another nearby row is
    # tried when it has none
    check: Check | None = None

    @field_validator("anchor")
    @classmethod
    def _check_anchor(cls, value: str) -> str:
        if not re.fullmatch(r"page|auto|field:.+", value):
            raise ValueError('anchor must be "page", "auto" or "field:<key>"')
        return value

    @model_validator(mode="after")
    def _check_bbox(self) -> "TemplateFieldIn":
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"bbox of field '{self.key}' must be [x1, y1, x2, y2] with x2 > x1 and y2 > y1")
        return self


class TemplateField(BaseModel):
    key: str
    bbox: list[float]
    # Resolved anchor: "page", "static:<id>" or "field:<key>"
    anchor: str
    multiline: bool = False
    check: Check | None = None


class StaticElement(BaseModel):
    id: str
    text: str
    bbox: list[float]


class StaticLearning(BaseModel):
    id: str
    seen: int
    found: int
    # False once a static element is missing on most matched documents (it is variable after all);
    # it is then no longer used as an anchor
    stable: bool


class FieldLearning(BaseModel):
    key: str
    observations: int
    # How much the field's position varies across matched documents, measured against the page and
    # against its anchor; the steadier one suggests what the field really moves with
    page_spread_px: float | None = None
    anchor_spread_px: float | None = None
    suggested_anchor: str | None = None


class TemplateLearning(BaseModel):
    statics: list[StaticLearning]
    fields: list[FieldLearning]


class DocTypeMarker(BaseModel):
    text: str
    # "seed" (doc_types.json, checked by hand) or "gemini" (proposed while drafting a template)
    source: str
    # Refuses uploads sent as another type; untrusted markers only let documents of their own
    # type through
    trusted: bool
    # Gemini markers: documents of this type that showed it, and of other types that did not
    seen: int | None = None
    others_without: int | None = None


class DocTypeInfo(BaseModel):
    doc_type: str
    name: str
    # Keys read from documents of this type
    fields: list[str]
    # "seed" (doc_types.json) or "gemini" (named while drafting a template)
    source: str
    # Phrases printed on every document of this type, used to refuse an upload of another type
    # before Gemini is asked
    markers: list[DocTypeMarker] = []


class TemplateSummary(BaseModel):
    id: int
    doc_type: str
    name: str
    page_width: int
    page_height: int
    static_count: int
    field_count: int
    match_count: int
    created_at: str
    updated_at: str


class TemplateDetail(TemplateSummary):
    static_elements: list[StaticElement]
    fields: list[TemplateField]
    learning: TemplateLearning


class SampleValue(BaseModel):
    key: str
    value: str | None


class TemplateCreated(TemplateDetail):
    # Each field read from the sample itself, to confirm the boxes; not stored
    sample_values: list[SampleValue]


class TemplateFieldResult(BaseModel):
    key: str
    value: str | None
    status: Literal["found", "nearby", "corrected", "invalid", "not_found"]
    valid: bool | None = None
    anchor: str
    # (dx, dy) taken from the anchor's measured drift
    drift: list[float]
    # where the value was found, in template page pixels
    box: list[float] | None = None


class TemplateCandidate(BaseModel):
    template_id: int
    name: str
    aligned: bool
    inliers: int


class TemplateExtractionResponse(BaseModel):
    request_id: str
    filename: str
    page_count: int
    # A stored template matched the upload
    matched: bool
    # No template matched: Gemini drafted a new one from this document, it was saved and the
    # fields were read with it
    created: bool = False
    doc_type: str | None = None
    template_id: int | None = None
    template_name: str | None = None
    # Every template of the doc_type that was tried, with how well it matched
    candidates: list[TemplateCandidate]
    # Static elements found on the upload, e.g. "14/16"
    static_found: str | None = None
    fields: list[TemplateFieldResult]
    message: str | None = None
    # What was fixed or left out while turning Gemini's draft into a template
    warnings: list[str] = []
    llm_elapsed_ms: float | None = None
    elapsed_ms: float
