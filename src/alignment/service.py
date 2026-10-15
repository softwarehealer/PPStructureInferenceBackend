from __future__ import annotations

import io
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import UploadFile

from alignment.models import (
    ExtractionWithReferenceResponse,
    PageAlignment,
    ParseTextWithReferenceResponse,
    ParseWithReferenceResponse,
    ProcessWithReferenceResponse,
)
from configs import app_logger as logger
from extraction import service as extraction_service
from extraction.models import FieldBox
from pipeline import service as pipeline_service
from pipeline.models import DocumentProcessingError, Granularity, ValidationError
from viewer import service as viewer_service

# Keypoints are detected on a copy downscaled to this longest side: plenty for alignment, much faster.
FEATURE_MAX_SIDE = 1600
SIFT_FEATURES = 4000
# Lowe's ratio test: keep a match only if clearly better than the second-best candidate.
RATIO_TEST = 0.75
# Max distance (reference pixels) for a match to count as agreeing with the warp.
RANSAC_REPROJ_PX = 5.0
# A warp is trusted only with enough agreeing matches.
MIN_INLIERS = 40
MIN_INLIER_RATIO = 0.25
# The reference outline found in the upload must cover at least this share of the upload.
MIN_REFERENCE_AREA_RATIO = 0.02

_sift: Any = None


@dataclass
class AlignedUploads:
    uploads: list[UploadFile]
    filename: str
    input_type: str
    alignment: list[PageAlignment]
    elapsed_ms: float


async def parse_with_reference(
    files: list[UploadFile],
    reference: UploadFile,
    granularity: Granularity = "lines",
) -> ParseWithReferenceResponse:
    aligned = await align_uploads(files, reference)
    parsed = await pipeline_service.parse_uploads(aligned.uploads, granularity)
    return ParseWithReferenceResponse(**_with_alignment(parsed.model_dump(), aligned))


async def parse_text_with_reference(
    files: list[UploadFile],
    reference: UploadFile,
    granularity: Granularity = "lines",
) -> ParseTextWithReferenceResponse:
    aligned = await align_uploads(files, reference)
    parsed = await pipeline_service.parse_text(aligned.uploads, granularity)
    return ParseTextWithReferenceResponse(**_with_alignment(parsed.model_dump(), aligned))


async def extract_by_coordinates_with_reference(
    files: list[UploadFile],
    reference: UploadFile,
    field_boxes: list[FieldBox],
    granularity: Granularity = "lines",
) -> ExtractionWithReferenceResponse:
    aligned = await align_uploads(files, reference)
    extracted = await extraction_service.extract_by_coordinates(aligned.uploads, field_boxes, granularity)
    return ExtractionWithReferenceResponse(**_with_alignment(extracted.model_dump(), aligned))


async def process_with_reference(
    files: list[UploadFile],
    reference: UploadFile,
    granularity: Granularity = "lines",
) -> ProcessWithReferenceResponse:
    aligned = await align_uploads(files, reference)
    processed = await viewer_service.process_uploads(aligned.uploads, granularity)
    return ProcessWithReferenceResponse(**_with_alignment(processed.model_dump(), aligned))


async def align_uploads(files: list[UploadFile], reference: UploadFile | None) -> AlignedUploads:
    """Warp every uploaded page into the pixel space of the matching reference page.

    The result is one PNG upload per page, ready for the regular parse/extract/process services, so
    boxes picked on the reference apply directly. A reference with one page is used for every
    uploaded page; a multi-page reference is matched page by page.
    """
    request_id = uuid.uuid4().hex[:12]
    started_at = time.perf_counter()
    uploads = [upload for upload in files if upload.filename]
    if not uploads:
        raise ValidationError("Upload at least one file.")
    if reference is None or not reference.filename:
        raise ValidationError("Upload a reference document.")

    try:
        with tempfile.TemporaryDirectory(prefix="align_") as tmp:
            work_dir = Path(tmp)
            page_paths, input_type, filename = _prepare(uploads, work_dir / "document", request_id)
            reference_paths, _, _ = _prepare([reference], work_dir / "reference", request_id)
            if len(reference_paths) not in (1, len(page_paths)):
                raise ValidationError(
                    f"Reference has {len(reference_paths)} pages; use a one-page reference "
                    f"or one with the same number of pages as the document ({len(page_paths)})."
                )

            reference_cache: dict[int, tuple[np.ndarray, tuple[np.ndarray, Any]]] = {}
            aligned_uploads: list[UploadFile] = []
            alignment: list[PageAlignment] = []
            for index, page_path in enumerate(page_paths):
                reference_index = index if len(reference_paths) > 1 else 0
                if reference_index not in reference_cache:
                    reference_image = _read_image(reference_paths[reference_index])
                    reference_cache[reference_index] = (reference_image, _features(reference_image))
                reference_image, reference_features = reference_cache[reference_index]

                page_image, page_alignment = _align_page(
                    _read_image(page_path),
                    reference_image.shape[:2],
                    reference_features,
                    index=index,
                    reference_index=reference_index,
                )
                logger.info(
                    "request_id=%s page=%s aligned=%s matches=%s inliers=%s elapsed_ms=%s",
                    request_id,
                    index,
                    page_alignment.aligned,
                    page_alignment.matches,
                    page_alignment.inliers,
                    page_alignment.elapsed_ms,
                )
                alignment.append(page_alignment)
                aligned_uploads.append(_png_upload(page_image, f"page_{index + 1:04d}.png"))
    finally:
        for upload in [*uploads, reference]:
            await upload.close()

    return AlignedUploads(
        uploads=aligned_uploads,
        filename=filename,
        input_type=input_type,
        alignment=alignment,
        elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
    )


def _prepare(uploads: list[UploadFile], work_dir: Path, request_id: str) -> tuple[list[Path], str, str]:
    """Save and turn uploads into page PNGs exactly like the parse flow (same DPI, EXIF handling)."""
    upload_dir = work_dir / "uploads"
    pages_dir = work_dir / "pages"
    upload_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)
    saved = pipeline_service._save_uploads(uploads, upload_dir, request_id=request_id)
    page_paths, input_type, filename = pipeline_service._prepare_pages(saved, pages_dir=pages_dir, request_id=request_id)
    if not page_paths:
        raise DocumentProcessingError("No pages could be processed.")
    return page_paths, input_type, filename


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise DocumentProcessingError(f"Could not read page image {path.name}")
    return image


def _features(image: np.ndarray) -> tuple[np.ndarray, Any]:
    """SIFT keypoints (in full-resolution pixel coordinates) and descriptors."""
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, FEATURE_MAX_SIDE / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    keypoints, descriptors = _sift.detectAndCompute(gray, None)
    points = np.float32([keypoint.pt for keypoint in keypoints]).reshape(-1, 2) / scale
    return points, descriptors


def _align_page(
    image: np.ndarray,
    reference_shape: tuple[int, int],
    reference_features: tuple[np.ndarray, Any],
    *,
    index: int,
    reference_index: int,
) -> tuple[np.ndarray, PageAlignment]:
    """Warp one page onto the reference. Returns the page unchanged when the match is not confident."""
    started_at = time.perf_counter()
    matches = inliers = 0

    def result(page: np.ndarray, aligned: bool, error: str | None = None) -> tuple[np.ndarray, PageAlignment]:
        return page, PageAlignment(
            page_index=index,
            reference_page=reference_index,
            aligned=aligned,
            matches=matches,
            inliers=inliers,
            elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
            error=error,
        )

    reference_points, reference_descriptors = reference_features
    page_points, page_descriptors = _features(image)
    if page_descriptors is None or reference_descriptors is None or len(page_points) < 2 or len(reference_points) < 2:
        return result(image, False, "Not enough detail to match against the reference.")

    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(page_descriptors, reference_descriptors, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < RATIO_TEST * pair[1].distance]
    matches = len(good)
    if matches < MIN_INLIERS:
        return result(image, False, f"Only {matches} matching points with the reference (need {MIN_INLIERS}).")

    source = page_points[[match.queryIdx for match in good]]
    target = reference_points[[match.trainIdx for match in good]]
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, RANSAC_REPROJ_PX)
    inliers = int(mask.sum()) if mask is not None else 0
    if homography is None or inliers < MIN_INLIERS or inliers / matches < MIN_INLIER_RATIO:
        return result(image, False, f"Matches with the reference are inconsistent ({inliers}/{matches} agree).")

    reference_height, reference_width = reference_shape
    if not _is_plausible(homography, image.shape[:2], reference_width, reference_height):
        return result(image, False, "Reference outline found in the page is implausible (too small or twisted).")

    warped = cv2.warpPerspective(
        image,
        homography,
        (reference_width, reference_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return result(warped, True)


def _is_plausible(homography: np.ndarray, image_shape: tuple[int, int], reference_width: int, reference_height: int) -> bool:
    """The reference page's outline, projected into the upload, must be a convex shape of reasonable size."""
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return False
    corners = np.float32([[0, 0], [reference_width, 0], [reference_width, reference_height], [0, reference_height]])
    outline = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), inverse).reshape(-1, 2)
    if not np.all(np.isfinite(outline)) or not cv2.isContourConvex(outline.astype(np.float32)):
        return False
    image_height, image_width = image_shape
    return cv2.contourArea(outline) >= MIN_REFERENCE_AREA_RATIO * image_width * image_height


def _png_upload(image: np.ndarray, filename: str) -> UploadFile:
    # Low compression: this PNG only lives for the rest of the request.
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise DocumentProcessingError(f"Could not encode aligned page {filename}")
    return UploadFile(io.BytesIO(encoded.tobytes()), filename=filename)


def _with_alignment(response: dict[str, Any], aligned: AlignedUploads) -> dict[str, Any]:
    """Report the original upload's name/type and include alignment time in the total."""
    return {
        **response,
        "filename": aligned.filename,
        "input_type": aligned.input_type,
        "elapsed_ms": round(response["elapsed_ms"] + aligned.elapsed_ms, 2),
        "alignment": aligned.alignment,
    }
