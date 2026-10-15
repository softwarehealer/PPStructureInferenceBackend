from __future__ import annotations

import base64
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import UploadFile

from configs import app_logger as logger
from pipeline import service as pipeline_service
from pipeline.models import DocumentProcessingError, Granularity, ValidationError
from viewer.models import ProcessResponse, ViewerPage


async def process_uploads(files: list[UploadFile], granularity: Granularity = "lines") -> ProcessResponse:
    """Parse like /parse, and also return each page image and the raw PPStructure output for the viewer.

    Upload handling, page preparation, the engine and the viewer format all come from the pipeline
    service, so the coordinates match /parse and /extract-by-coordinates exactly.
    """
    request_id = uuid.uuid4().hex[:12]
    started_at = time.perf_counter()
    uploads = [upload for upload in files if upload.filename]
    if not uploads:
        raise ValidationError("Upload at least one file.")

    logger.info(
        "request_id=%s upload received count=%s filenames=%s",
        request_id,
        len(uploads),
        [upload.filename for upload in uploads],
    )

    try:
        with tempfile.TemporaryDirectory(prefix="viewer_") as tmp:
            work_dir = Path(tmp)
            upload_dir = work_dir / "uploads"
            pages_dir = work_dir / "pages"
            upload_dir.mkdir(parents=True, exist_ok=True)
            pages_dir.mkdir(parents=True, exist_ok=True)

            saved_uploads = pipeline_service._save_uploads(uploads, upload_dir, request_id=request_id)
            image_paths, input_type, filename = pipeline_service._prepare_pages(
                saved_uploads,
                pages_dir=pages_dir,
                request_id=request_id,
            )

            if not image_paths:
                logger.warning("request_id=%s no pages produced", request_id)
                raise DocumentProcessingError("No pages could be processed.")

            pipeline_service.get_engine()
            pages = [
                _process_page(
                    image_path=image_path,
                    index=index,
                    granularity=granularity,
                    request_id=request_id,
                )
                for index, image_path in enumerate(image_paths)
            ]

            successful_pages = sum(1 for page in pages if page.success)
            failed_pages = len(pages) - successful_pages
            status = (
                "completed"
                if failed_pages == 0
                else "failed"
                if successful_pages == 0
                else "partial_failed"
            )
            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
            logger.info(
                "request_id=%s finished status=%s pages=%s successful=%s failed=%s elapsed_ms=%s",
                request_id,
                status,
                len(pages),
                successful_pages,
                failed_pages,
                elapsed_ms,
            )

            return ProcessResponse(
                request_id=request_id,
                filename=filename,
                input_type=input_type,
                status=status,
                page_count=len(pages),
                successful_pages=successful_pages,
                failed_pages=failed_pages,
                elapsed_ms=elapsed_ms,
                pages=pages,
            )
    finally:
        for upload in uploads:
            await upload.close()


def _process_page(*, image_path: Path, index: int, granularity: Granularity, request_id: str) -> ViewerPage:
    page_started_at = time.perf_counter()
    width, height = pipeline_service._safe_image_size(image_path, request_id=request_id, index=index)
    success = False
    error: str | None = None
    viewer_result = pipeline_service._empty_page_result(width=width, height=height, img_idx=index)
    raw_result: list[dict] = []

    try:
        logger.info(
            "request_id=%s page=%s ocr start image=%s size=%sx%s",
            request_id,
            index,
            image_path.name,
            width,
            height,
        )
        raw_result = pipeline_service.get_engine().predict_file(image_path, word_boxes=granularity == "words")
        viewer_result = pipeline_service._to_page_result(
            raw_result,
            width=width,
            height=height,
            img_idx=index,
            granularity=granularity,
        )
        success = True
        logger.info(
            "request_id=%s page=%s ocr success regions=%s",
            request_id,
            index,
            len(raw_result),
        )
    except Exception as exc:
        raw_result = []
        error = f"{type(exc).__name__}: {exc}"
        logger.exception(
            "request_id=%s page=%s ocr failed image=%s",
            request_id,
            index,
            image_path.name,
        )

    return ViewerPage(
        page_index=index,
        image_name=image_path.name,
        image_mime="image/png",
        image_base64=_safe_base64_file(image_path, request_id=request_id, index=index),
        success=success,
        error=error,
        viewer_result=viewer_result,
        raw_result=raw_result,
        elapsed_ms=round((time.perf_counter() - page_started_at) * 1000, 2),
    )


def _safe_base64_file(path: Path, *, request_id: str, index: int) -> str:
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        logger.exception(
            "request_id=%s page=%s failed to encode image path=%s",
            request_id,
            index,
            path,
        )
        return ""
