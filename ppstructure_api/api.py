from __future__ import annotations

import base64
import logging
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

from ppstructure_api.engine import PPStructureEngine
from ppstructure_api.normalizer import build_empty_viewer_result, build_viewer_result
from ppstructure_api.pdf_splitter import PDFPageSplitter


ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}

logger = logging.getLogger("ppstructure_api")

app = FastAPI(title="PPStructure API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine: PPStructureEngine | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process")
async def process_upload(
    file: UploadFile | None = File(default=None),
    files: list[UploadFile] | None = File(default=None),
) -> dict[str, Any]:
    request_id = uuid.uuid4().hex[:12]
    started_at = time.perf_counter()
    uploads = _collect_uploads(file=file, files=files)
    if not uploads:
        raise HTTPException(status_code=400, detail="Upload at least one file.")

    logger.info(
        "request_id=%s upload received count=%s filenames=%s",
        request_id,
        len(uploads),
        [upload.filename for upload in uploads],
    )

    try:
        with tempfile.TemporaryDirectory(prefix="ppstructure_api_") as tmp:
            work_dir = Path(tmp)
            upload_dir = work_dir / "uploads"
            pages_dir = work_dir / "pages"
            upload_dir.mkdir(parents=True, exist_ok=True)
            pages_dir.mkdir(parents=True, exist_ok=True)

            saved_uploads = _save_uploads(uploads, upload_dir, request_id=request_id)
            image_paths, input_type, filename = _prepare_uploaded_pages(
                saved_uploads,
                pages_dir=pages_dir,
                request_id=request_id,
            )

            if not image_paths:
                logger.warning("request_id=%s no pages produced", request_id)
                raise HTTPException(status_code=422, detail="No pages could be processed.")

            _get_engine()
            pages = [
                _process_page(
                    image_path=image_path,
                    index=index,
                    request_id=request_id,
                )
                for index, image_path in enumerate(image_paths)
            ]

            successful_pages = sum(1 for page in pages if page["success"])
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

            return {
                "request_id": request_id,
                "filename": filename,
                "input_type": input_type,
                "status": status,
                "page_count": len(pages),
                "successful_pages": successful_pages,
                "failed_pages": failed_pages,
                "elapsed_ms": elapsed_ms,
                "pages": pages,
            }
    finally:
        for upload in uploads:
            await upload.close()


def _collect_uploads(
    *,
    file: UploadFile | None,
    files: list[UploadFile] | None,
) -> list[UploadFile]:
    uploads: list[UploadFile] = []
    if files:
        uploads.extend(upload for upload in files if upload.filename)
    if file is not None and file.filename:
        uploads.append(file)
    return uploads


def _save_uploads(
    uploads: list[UploadFile],
    upload_dir: Path,
    *,
    request_id: str,
) -> list[tuple[Path, str]]:
    saved: list[tuple[Path, str]] = []
    for index, upload in enumerate(uploads):
        filename = upload.filename or f"upload_{index + 1}"
        extension = Path(filename).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            logger.warning(
                "request_id=%s unsupported file type filename=%s extension=%s",
                request_id,
                filename,
                extension,
            )
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Upload PDF, PNG, JPG, JPEG, or WEBP files.",
            )

        upload_path = upload_dir / f"{index + 1:04d}_{_safe_filename(filename)}"
        try:
            with upload_path.open("wb") as out_file:
                shutil.copyfileobj(upload.file, out_file)
        except Exception as exc:
            logger.exception(
                "request_id=%s failed to save upload filename=%s",
                request_id,
                filename,
            )
            raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {filename}") from exc
        saved.append((upload_path, filename))
    return saved


def _prepare_uploaded_pages(
    saved_uploads: list[tuple[Path, str]],
    *,
    pages_dir: Path,
    request_id: str,
) -> tuple[list[Path], str, str]:
    pages_dir.mkdir(parents=True, exist_ok=True)
    pdf_uploads = [item for item in saved_uploads if item[0].suffix.lower() == ".pdf"]
    if pdf_uploads and len(saved_uploads) > 1:
        raise HTTPException(
            status_code=400,
            detail="Upload either one PDF or one/more image files, not a mix.",
        )

    if pdf_uploads:
        upload_path, filename = pdf_uploads[0]
        logger.info("request_id=%s splitting pdf path=%s", request_id, upload_path.name)
        try:
            image_paths = PDFPageSplitter(upload_path, pages_dir).split()
        except Exception as exc:
            logger.exception("request_id=%s failed to split pdf", request_id)
            raise HTTPException(status_code=422, detail=f"Could not split PDF: {exc}") from exc
        logger.info("request_id=%s pdf split complete pages=%s", request_id, len(image_paths))
        return image_paths, "pdf", filename

    image_paths: list[Path] = []
    for index, (upload_path, filename) in enumerate(saved_uploads):
        try:
            image_path = _normalize_image(
                upload_path,
                pages_dir,
                output_name=f"page_{index + 1:04d}.png",
            )
        except Exception as exc:
            logger.exception(
                "request_id=%s failed to normalize image filename=%s",
                request_id,
                filename,
            )
            raise HTTPException(status_code=422, detail=f"Could not prepare image {filename}: {exc}") from exc
        logger.info("request_id=%s image normalized path=%s", request_id, image_path.name)
        image_paths.append(image_path)

    input_type = "multi_image" if len(image_paths) > 1 else "image"
    filename = saved_uploads[0][1] if len(saved_uploads) == 1 else f"{len(saved_uploads)} images"
    return image_paths, input_type, filename


def _prepare_input_pages(
    upload_path: Path,
    *,
    extension: str,
    work_dir: Path,
    request_id: str,
) -> tuple[list[Path], str]:
    try:
        if extension == ".pdf":
            logger.info("request_id=%s splitting pdf path=%s", request_id, upload_path.name)
            image_paths = PDFPageSplitter(upload_path, work_dir / "pages").split()
            logger.info(
                "request_id=%s pdf split complete pages=%s",
                request_id,
                len(image_paths),
            )
            return image_paths, "pdf"

        image_path = _normalize_image(upload_path, work_dir)
        logger.info("request_id=%s image normalized path=%s", request_id, image_path.name)
        return [image_path], "image"
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("request_id=%s failed to prepare upload", request_id)
        raise HTTPException(
            status_code=422,
            detail=f"Could not prepare uploaded file for OCR: {exc}",
        ) from exc


def _process_page(*, image_path: Path, index: int, request_id: str) -> dict[str, Any]:
    page_started_at = time.perf_counter()
    width, height = _safe_image_size(image_path, request_id=request_id, index=index)
    page_payload = {
        "page_index": index,
        "image_name": image_path.name,
        "image_mime": "image/png",
        "image_base64": _safe_base64_file(image_path, request_id=request_id, index=index),
        "success": False,
        "error": None,
        "viewer_result": build_empty_viewer_result(width=width, height=height, img_idx=index),
        "raw_result": [],
        "elapsed_ms": 0.0,
    }

    try:
        logger.info(
            "request_id=%s page=%s ocr start image=%s size=%sx%s",
            request_id,
            index,
            image_path.name,
            width,
            height,
        )
        raw_result = _get_engine().predict_file(image_path)
        page_payload.update(
            {
                "success": True,
                "viewer_result": build_viewer_result(
                    raw_result,
                    width=width,
                    height=height,
                    img_idx=index,
                ),
                "raw_result": raw_result,
            }
        )
        logger.info(
            "request_id=%s page=%s ocr success regions=%s",
            request_id,
            index,
            len(raw_result),
        )
    except Exception as exc:
        page_payload["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception(
            "request_id=%s page=%s ocr failed image=%s",
            request_id,
            index,
            image_path.name,
        )
    finally:
        page_payload["elapsed_ms"] = round((time.perf_counter() - page_started_at) * 1000, 2)

    return page_payload


def _get_engine() -> PPStructureEngine:
    global _engine
    if _engine is None:
        logger.info("initializing PPStructure engine")
        _engine = PPStructureEngine()
        logger.info("PPStructure engine initialized")
    return _engine


def _safe_filename(filename: str) -> str:
    name = Path(filename).name.replace("\x00", "")
    return name or "upload"


def _normalize_image(path: Path, work_dir: Path, *, output_name: str | None = None) -> Path:
    output_path = work_dir / (output_name or f"{path.stem}.png")
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        image.save(output_path, "PNG")
    return output_path


def _safe_image_size(path: Path, *, request_id: str, index: int) -> tuple[int, int]:
    try:
        return _image_size(path)
    except Exception:
        logger.exception(
            "request_id=%s page=%s failed to read image size path=%s",
            request_id,
            index,
            path,
        )
        return 1000, 1400


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _safe_base64_file(path: Path, *, request_id: str, index: int) -> str:
    try:
        return _base64_file(path)
    except Exception:
        logger.exception(
            "request_id=%s page=%s failed to encode image path=%s",
            request_id,
            index,
            path,
        )
        return ""


def _base64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")
