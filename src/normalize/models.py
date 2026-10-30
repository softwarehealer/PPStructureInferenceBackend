from __future__ import annotations

from pydantic import BaseModel


class NormalizedPage(BaseModel):
    page_index: int
    image_name: str
    # The page as uploaded (PDF pages rendered at 300 DPI, EXIF rotation applied), as a JPEG preview
    # at most 2000 px wide or high; original_width/height and corners are in its full-size pixels
    original_mime: str = "image/jpeg"
    original_base64: str
    original_width: int
    original_height: int
    # Corners of the document found in the original: [[x, y]] top-left, top-right, bottom-right,
    # bottom-left as seen in the original; null when no document outline was found
    corners: list[list[float]] | None = None
    # The page was cut out along the corners and flattened
    cropped: bool
    # Degrees the page was turned counter-clockwise to make it upright: 0, 90, 180 or 270
    rotation: int
    orientation_confidence: float | None = None
    # The normalized page (JPEG, quality 92)
    image_mime: str = "image/jpeg"
    image_base64: str
    width: int
    height: int
    warnings: list[str]
    elapsed_ms: float


class NormalizeResponse(BaseModel):
    request_id: str
    filename: str
    input_type: str
    page_count: int
    pages: list[NormalizedPage]
    elapsed_ms: float
