from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette import status

from configs.settings import settings
from configs import app_logger as logger
from pipeline import service
from pipeline.models import (
    DocumentProcessingError,
    Granularity,
    ParseResponse,
    ParseTextResponse,
    ValidationError,
)

applicable_path_prefix = settings.PPS_PIPELINE_API_PATH_PREFIX
if applicable_path_prefix == "/" :
    applicable_path_prefix = ""

router = APIRouter(prefix=applicable_path_prefix, tags=["PPStructure pipeline"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Endpoints stay async: the Paddle predictors only work on the thread that created them,
# and async routes always run on the event loop thread.

@router.post("/parse")
async def process_upload(
    files: Annotated[list[UploadFile], File()],
    granularity: Annotated[Granularity, Form(description='"lines" (default) or "words": unit of each OCR entry')] = "lines",
) -> ParseResponse:
    """OCR one PDF or one/more images and return every text line with its region per page."""
    logger.info("parse api called ...")
    try:
        return await service.parse_uploads(files, granularity)
    except ValidationError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except DocumentProcessingError as de:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(de))


@router.post("/parse-text")
async def parse_text(
    files: Annotated[list[UploadFile], File()],
    granularity: Annotated[Granularity, Form(description='"lines" (default) or "words": unit of each OCR entry')] = "lines",
) -> ParseTextResponse:
    """Same as /parse, but each page is reduced to its list of texts."""
    logger.info("parse-text api called ...")
    try:
        return await service.parse_text(files, granularity)
    except ValidationError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except DocumentProcessingError as de:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(de))
