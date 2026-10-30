"""Normalize a document photo or scan: cut the document out of its background, flatten its
perspective and turn it upright, so it can be stored as a template sample.

1. Outline: the document's four corners, from the bright region it forms against a darker
   background or from its edges. A shape only counts as the document when what surrounds it is
   darker (a photo's background) or empty (blank paper around a scanned card), so a table or box
   inside a page is never cut out of it.
2. Flatten: the outline is warped to a rectangle.
3. Orientation: PP-LCNet_x1_0_doc_ori classifies the page as turned 0, 90, 180 or 270 degrees.
"""
from __future__ import annotations

import base64
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import UploadFile
from paddle import inference

from alignment import service as alignment_service
from configs import app_logger as logger
from configs.settings import settings
from normalize.models import NormalizedPage, NormalizeResponse
from pipeline.models import DocumentProcessingError

ORIENTATION_MODEL_DIR = Path(settings.INFERENCE_BASE_PATH) / "PP-LCNet_x1_0_doc_ori_infer"
ORIENTATION_ANGLES = (0, 90, 180, 270)
# Below this confidence the page is left as it is
ORIENTATION_MIN_CONFIDENCE = 0.5
# The outline is searched for on a copy this large (longest side)
OUTLINE_MAX_SIDE = 1000
# A document covers between these shares of the image (all of it means there is no outline)
MIN_DOCUMENT_AREA = 0.15
MAX_DOCUMENT_AREA = 0.97
# Around a document the image must be at least this much darker (0-255) ...
MIN_BACKGROUND_CONTRAST = 20
# ... or have edges on at most this share of its pixels (blank paper)
MAX_BACKGROUND_EDGES = 0.01
# A corner this close to the image's edge (share of its size) suggests the document is cut off
BORDER_MARGIN = 0.01
PREVIEW_MAX_SIDE = 2000

_orientation_predictor: Any = None


@dataclass
class Normalized:
    image: np.ndarray
    corners: np.ndarray | None = None
    rotation: int = 0
    orientation_confidence: float | None = None
    warnings: list[str] = field(default_factory=list)


async def normalize_uploads(files: list[UploadFile], *, crop: bool, orient: bool) -> NormalizeResponse:
    started_at = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]
    pages: list[NormalizedPage] = []
    try:
        with tempfile.TemporaryDirectory(prefix="normalize_") as tmp:
            page_paths, input_type, filename = alignment_service._prepare(files, Path(tmp), request_id)
            for index, path in enumerate(page_paths):
                page_started_at = time.perf_counter()
                image = alignment_service._read_image(path)
                result = normalize_image(image, crop=crop, orient=orient)
                pages.append(_to_page(index, path.name, image, result, page_started_at))
    finally:
        for upload in files:
            await upload.close()
    logger.info(
        "request_id=%s normalized pages=%s cropped=%s rotations=%s",
        request_id, len(pages), sum(p.cropped for p in pages), [p.rotation for p in pages],
    )
    return NormalizeResponse(
        request_id=request_id,
        filename=filename,
        input_type=input_type,
        page_count=len(pages),
        pages=pages,
        elapsed_ms=_elapsed(started_at),
    )


def normalize_image(image: np.ndarray, *, crop: bool = True, orient: bool = True) -> Normalized:
    result = Normalized(image=image)
    if crop:
        corners = find_outline(image)
        if corners is None:
            result.warnings.append("No document edges found against a background; the whole page is used as it is.")
        else:
            result.corners = corners
            result.image = _flatten(image, corners)
            if _touches_border(corners, image.shape[:2]):
                result.warnings.append("The document touches the edge of the image; part of it may be cut off.")
    if orient:
        angle, confidence = orientation(result.image)
        result.orientation_confidence = round(confidence, 3)
        if confidence < ORIENTATION_MIN_CONFIDENCE:
            result.warnings.append(f"Orientation unclear ({confidence:.0%} sure it is turned {angle} degrees); not turned.")
        elif angle:
            result.rotation = angle
            result.image = np.ascontiguousarray(np.rot90(result.image, angle // 90))
    return result


def find_outline(image: np.ndarray) -> np.ndarray | None:
    """The document's corners (top-left, top-right, bottom-right, bottom-left) in image pixels."""
    scale = min(1.0, OUTLINE_MAX_SIDE / max(image.shape[:2]))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    area = gray.shape[0] * gray.shape[1]

    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    edges = cv2.dilate(cv2.Canny(gray, 30, 90), np.ones((5, 5), np.uint8))
    candidates = [
        quad for mask in (bright, edges) for quad in _quads(mask)
        if MIN_DOCUMENT_AREA * area <= cv2.contourArea(quad) <= MAX_DOCUMENT_AREA * area
        and _stands_out(gray, quad)
    ]
    if not candidates:
        return None
    return _order(max(candidates, key=cv2.contourArea) / scale)


def orientation(image: np.ndarray) -> tuple[int, float]:
    """Degrees the page is turned (it is made upright by turning it back counter-clockwise) and the
    classifier's confidence. Preprocessing as in the model's inference.yml."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    scale = 256 / min(height, width)
    rgb = cv2.resize(rgb, (max(224, round(width * scale)), max(224, round(height * scale))))
    height, width = rgb.shape[:2]
    top, left = (height - 224) // 2, (width - 224) // 2
    crop = rgb[top:top + 224, left:left + 224].astype(np.float32) / 255
    crop = (crop - np.float32([0.485, 0.456, 0.406])) / np.float32([0.229, 0.224, 0.225])
    tensor = np.ascontiguousarray(crop.transpose(2, 0, 1)[None], dtype=np.float32)

    predictor = _get_orientation_predictor()
    handle = predictor.get_input_handle(predictor.get_input_names()[0])
    handle.reshape(tensor.shape)
    handle.copy_from_cpu(tensor)
    predictor.run()
    scores = predictor.get_output_handle(predictor.get_output_names()[0]).copy_to_cpu()[0]
    best = int(np.argmax(scores))
    return ORIENTATION_ANGLES[best], float(scores[best])


def _get_orientation_predictor() -> Any:
    global _orientation_predictor
    if _orientation_predictor is None:
        model = ORIENTATION_MODEL_DIR / "inference.json"
        if not model.is_file():
            raise DocumentProcessingError(f"Orientation model not found at {ORIENTATION_MODEL_DIR}")
        config = inference.Config(str(model), str(ORIENTATION_MODEL_DIR / "inference.pdiparams"))
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(4)
        config.disable_glog_info()
        _orientation_predictor = inference.create_predictor(config)
    return _orientation_predictor


def _quads(mask: np.ndarray) -> list[np.ndarray]:
    """Four-cornered convex outlines of the mask's largest shapes."""
    quads = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        # Rounded corners (ID cards) need a looser fit
        for epsilon in (0.02, 0.03, 0.04, 0.05):
            quad = cv2.approxPolyDP(hull, epsilon * perimeter, True)
            if len(quad) == 4 and cv2.isContourConvex(quad):
                quads.append(quad.reshape(4, 2).astype(np.float32))
                break
    return quads


def _stands_out(gray: np.ndarray, quad: np.ndarray) -> bool:
    """Whether the shape lies on something else (darker or empty surroundings), rather than being a
    part of the page such as a table."""
    inside = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(inside, quad.astype(np.int32), 255)
    # Skip a band along the outline, where shadows and the document's own border are
    outside = cv2.dilate(inside, np.ones((25, 25), np.uint8)) == 0
    if outside.mean() < 0.03:
        return True
    darker = gray[inside > 0].mean() - gray[outside].mean() >= MIN_BACKGROUND_CONTRAST
    empty = (cv2.Canny(gray, 50, 150)[outside] > 0).mean() <= MAX_BACKGROUND_EDGES
    return bool(darker or empty)


def _order(points: np.ndarray) -> np.ndarray:
    total, diff = points.sum(axis=1), np.diff(points, axis=1).ravel()
    return np.float32([points[total.argmin()], points[diff.argmin()], points[total.argmax()], points[diff.argmax()]])


def _flatten(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = corners
    width = int(round(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))))
    height = int(round(max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))))
    target = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(corners, target)
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _touches_border(corners: np.ndarray, shape: tuple[int, int]) -> bool:
    height, width = shape
    margin_x, margin_y = BORDER_MARGIN * width, BORDER_MARGIN * height
    return bool(np.any(
        (corners[:, 0] <= margin_x) | (corners[:, 0] >= width - margin_x)
        | (corners[:, 1] <= margin_y) | (corners[:, 1] >= height - margin_y)
    ))


def _to_page(index: int, name: str, original: np.ndarray, result: Normalized, started_at: float) -> NormalizedPage:
    height, width = original.shape[:2]
    scale = min(1.0, PREVIEW_MAX_SIDE / max(height, width))
    preview = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else original
    out_height, out_width = result.image.shape[:2]
    return NormalizedPage(
        page_index=index,
        image_name=name,
        original_base64=_encode(preview, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 85]),
        original_width=width,
        original_height=height,
        corners=[[round(float(x), 1), round(float(y), 1)] for x, y in result.corners] if result.corners is not None else None,
        cropped=result.corners is not None,
        rotation=result.rotation,
        orientation_confidence=result.orientation_confidence,
        image_base64=_encode(result.image, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 92]),
        width=out_width,
        height=out_height,
        warnings=result.warnings,
        elapsed_ms=_elapsed(started_at),
    )


def _encode(image: np.ndarray, extension: str, params: list[int]) -> str:
    ok, encoded = cv2.imencode(extension, image, params)
    if not ok:
        raise DocumentProcessingError(f"Could not encode the page as {extension}")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _elapsed(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)
