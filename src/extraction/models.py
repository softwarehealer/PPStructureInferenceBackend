from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from pipeline.models import ParseSummary


class FieldBox(BaseModel):
    key: str = Field(min_length=1)
    # [x1, y1, x2, y2] in pixels of the page image, same space as /parse text_region
    bbox: tuple[float, float, float, float]
    page: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_bbox(self) -> "FieldBox":
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"bbox of field '{self.key}' must be [x1, y1, x2, y2] with x2 > x1 and y2 > y1")
        return self


class ExtractedField(BaseModel):
    key: str
    # None when no text falls inside the box or the page does not exist
    value: str | None = None


class ExtractionResponse(ParseSummary):
    fields: list[ExtractedField]
