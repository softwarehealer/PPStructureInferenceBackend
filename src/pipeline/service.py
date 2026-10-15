from __future__ import annotations

import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import UploadFile
from pdf2image import convert_from_path
from PIL import Image, ImageOps

from configs import app_logger as logger
from pipeline.engine import PPStructureEngine
from pipeline.models import (
    DocumentProcessingError,
    Granularity,
    OcrLine,
    PageResult,
    PageText,
    ParsedPage,
    ParseResponse,
    ParseSummary,
    ParseTextResponse,
    ValidationError,
)

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
PDF_RENDER_DPI = 300

_engine: PPStructureEngine | None = None


async def parse_uploads(files: list[UploadFile], granularity: Granularity = "lines") -> ParseResponse:
    """Run OCR on uploaded files: one PDF, or one or more images, each image being one page.

    granularity decides whether each OCR entry is a detected line or a space-separated word.

    Shared by the parse and extraction APIs so they parse in-process instead of over HTTP.
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
        with tempfile.TemporaryDirectory(prefix="pipeline_") as tmp:
            work_dir = Path(tmp)
            upload_dir = work_dir / "uploads"
            pages_dir = work_dir / "pages"
            upload_dir.mkdir(parents=True, exist_ok=True)
            pages_dir.mkdir(parents=True, exist_ok=True)

            saved_uploads = _save_uploads(uploads, upload_dir, request_id=request_id)
            image_paths, input_type, filename = _prepare_pages(
                saved_uploads,
                pages_dir=pages_dir,
                request_id=request_id,
            )

            if not image_paths:
                logger.warning("request_id=%s no pages produced", request_id)
                raise DocumentProcessingError("No pages could be processed.")

            get_engine()
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

            return ParseResponse(
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


async def parse_text(files: list[UploadFile], granularity: Granularity = "lines") -> ParseTextResponse:
    """Parse, then reduce each page to its list of texts (lines or words)."""
    parsed = await parse_uploads(files, granularity)
    return ParseTextResponse(
        **summarize(parsed).model_dump(),
        pages=[
            PageText(
                page_index=page.page_index,
                success=page.success,
                error=page.error,
                texts=[line.text for line in page.result.res if line.text] if page.success else [],
            )
            for page in parsed.pages
        ],
    )


def summarize(parsed: ParseResponse) -> ParseSummary:
    return ParseSummary(**parsed.model_dump(include=set(ParseSummary.model_fields)))


def get_engine() -> PPStructureEngine:
    global _engine
    if _engine is None:
        logger.info("initializing PPStructure engine")
        _engine = PPStructureEngine()
        logger.info("PPStructure engine initialized")
    return _engine


# ---------------------------------------------------------------------------
# Upload handling
# ---------------------------------------------------------------------------

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
            raise ValidationError("Unsupported file type. Upload PDF, PNG, JPG, JPEG, or WEBP files.")

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
            raise ValidationError(f"Could not read uploaded file: {filename}") from exc
        saved.append((upload_path, filename))
    return saved


def _prepare_pages(
    saved_uploads: list[tuple[Path, str]],
    *,
    pages_dir: Path,
    request_id: str,
) -> tuple[list[Path], str, str]:
    """Turn saved uploads into one PNG per page. Returns (page paths, input type, display filename)."""
    pdf_uploads = [item for item in saved_uploads if item[0].suffix.lower() == ".pdf"]
    if pdf_uploads and len(saved_uploads) > 1:
        raise ValidationError("Upload either one PDF or one/more image files, not a mix.")

    if pdf_uploads:
        upload_path, filename = pdf_uploads[0]
        logger.info("request_id=%s splitting pdf path=%s", request_id, upload_path.name)
        try:
            image_paths = _split_pdf(upload_path, pages_dir)
        except Exception as exc:
            logger.exception("request_id=%s failed to split pdf", request_id)
            raise DocumentProcessingError(f"Could not split PDF: {exc}") from exc
        logger.info("request_id=%s pdf split complete pages=%s", request_id, len(image_paths))
        return image_paths, "pdf", filename

    image_paths: list[Path] = []
    for index, (upload_path, filename) in enumerate(saved_uploads):
        try:
            image_path = _normalize_image(upload_path, pages_dir / f"page_{index + 1:04d}.png")
        except Exception as exc:
            logger.exception(
                "request_id=%s failed to normalize image filename=%s",
                request_id,
                filename,
            )
            raise DocumentProcessingError(f"Could not prepare image {filename}: {exc}") from exc
        logger.info("request_id=%s image normalized path=%s", request_id, image_path.name)
        image_paths.append(image_path)

    input_type = "multi_image" if len(image_paths) > 1 else "image"
    filename = saved_uploads[0][1] if len(saved_uploads) == 1 else f"{len(saved_uploads)} images"
    return image_paths, input_type, filename


def _split_pdf(pdf_path: Path, output_dir: Path) -> list[Path]:
    pages = convert_from_path(
        str(pdf_path),
        dpi=PDF_RENDER_DPI,
        use_cropbox=False,
        strict=False,
    )

    saved_paths: list[Path] = []
    for index, page in enumerate(pages):
        page = ImageOps.exif_transpose(page)
        output_path = output_dir / f"{pdf_path.stem}_page{index + 1}.png"
        page.save(output_path, "PNG")
        saved_paths.append(output_path)
    return saved_paths


def _normalize_image(path: Path, output_path: Path) -> Path:
    """Save the image as PNG with EXIF rotation applied, so page pixels match what the user sees."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        image.save(output_path, "PNG")
    return output_path


def _safe_filename(filename: str) -> str:
    name = Path(filename).name.replace("\x00", "")
    return name or "upload"


# ---------------------------------------------------------------------------
# OCR per page
# ---------------------------------------------------------------------------

def _process_page(*, image_path: Path, index: int, granularity: Granularity, request_id: str) -> ParsedPage:
    page_started_at = time.perf_counter()
    width, height = _safe_image_size(image_path, request_id=request_id, index=index)
    success = False
    error: str | None = None
    result = _empty_page_result(width=width, height=height, img_idx=index)

    try:
        logger.info(
            "request_id=%s page=%s ocr start image=%s size=%sx%s",
            request_id,
            index,
            image_path.name,
            width,
            height,
        )
        raw_result = get_engine().predict_file(image_path, word_boxes=granularity == "words")
        result = _to_page_result(
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
        error = f"{type(exc).__name__}: {exc}"
        logger.exception(
            "request_id=%s page=%s ocr failed image=%s",
            request_id,
            index,
            image_path.name,
        )

    return ParsedPage(
        page_index=index,
        image_name=image_path.name,
        image_mime="image/png",
        success=success,
        error=error,
        result=result,
        elapsed_ms=round((time.perf_counter() - page_started_at) * 1000, 2),
    )


def _safe_image_size(path: Path, *, request_id: str, index: int) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        logger.exception(
            "request_id=%s page=%s failed to read image size path=%s",
            request_id,
            index,
            path,
        )
        return 1000, 1400


# ---------------------------------------------------------------------------
# Raw PPStructure output -> viewer format
# ---------------------------------------------------------------------------

def _empty_page_result(*, width: int, height: int, img_idx: int) -> PageResult:
    return PageResult(type="figure", bbox=[0, 0, width, height], res=[], img_idx=img_idx, score=0.0)


def _to_page_result(
    raw_result: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    img_idx: int,
    granularity: Granularity,
) -> PageResult:
    """Flatten every OCR line found in the layout regions into one list.

    If no region produced text, fall back to one entry per layout region, labelled with its type.
    """
    lines: list[OcrLine] = []
    for region in raw_result:
        _collect_ocr_lines(region.get("res"), lines, granularity)

    if not lines:
        for region in raw_result:
            text_region = _bbox_to_text_region(region.get("bbox"))
            if text_region is None:
                continue
            lines.append(
                OcrLine(
                    text=str(region.get("type") or "region"),
                    confidence=_as_float(region.get("score"), default=1.0),
                    text_region=text_region,
                )
            )

    return PageResult(type="figure", bbox=[0, 0, width, height], res=lines, img_idx=img_idx, score=1.0)


def _collect_ocr_lines(value: Any, lines: list[OcrLine], granularity: Granularity) -> None:
    if isinstance(value, dict):
        if "text" in value and "text_region" in value:
            text = str(value.get("text") or "")
            confidence = _as_float(value.get("confidence"), default=1.0)
            if granularity == "words" and "text_word" in value:
                lines.extend(
                    _split_words(
                        text,
                        text_region=value["text_region"],
                        confidence=confidence,
                        paddle_words=value["text_word"],
                        paddle_word_regions=value["text_word_region"],
                    )
                )
            else:
                lines.append(OcrLine(text=text, confidence=confidence, text_region=value["text_region"]))
            return

        for child in value.values():
            _collect_ocr_lines(child, lines, granularity)
        return

    if isinstance(value, list):
        for child in value:
            _collect_ocr_lines(child, lines, granularity)


def _split_words(
    text: str,
    *,
    text_region: list[list[float]],
    confidence: float,
    paddle_words: list[str],
    paddle_word_regions: list[list[list[float]]],
) -> list[OcrLine]:
    """Split one OCR line into space-separated words, each with its own box.

    Paddle's own words drop punctuation ('1,234.56' -> '1', '234.56'), so they are used only for
    positions: a word's box spans the Paddle words inside it. A word with no Paddle word inside
    (e.g. a lone '&') gets a box estimated from its character position in the line.
    """
    # Character span of each Paddle word within the line text
    spans: list[tuple[int, int, list[list[float]]]] = []
    cursor = 0
    for paddle_word, region in zip(paddle_words, paddle_word_regions):
        start = text.find(paddle_word, cursor)
        if start < 0:
            continue
        spans.append((start, start + len(paddle_word), region))
        cursor = start + len(paddle_word)

    xs = [point[0] for point in text_region]
    ys = [point[1] for point in text_region]
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)

    words: list[OcrLine] = []
    for match in re.finditer(r"\S+", text):
        inside = [region for start, end, region in spans if start >= match.start() and end <= match.end()]
        if inside:
            word_xs = [point[0] for region in inside for point in region]
            x1, x2 = min(word_xs), max(word_xs)
        else:
            x1 = left + (right - left) * match.start() / len(text)
            x2 = left + (right - left) * match.end() / len(text)
        words.append(
            OcrLine(
                text=match.group(),
                confidence=confidence,
                text_region=[[x1, top], [x2, top], [x2, bottom], [x1, bottom]],
            )
        )
    return words


def _bbox_to_text_region(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    if not all(isinstance(item, (int, float)) for item in value[:4]):
        return None

    x1, y1, x2, y2 = [float(item) for item in value[:4]]
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def _as_float(value: Any, *, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default
