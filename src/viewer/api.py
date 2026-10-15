from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette import status

from configs.settings import settings
from configs import app_logger as logger
from pipeline.models import DocumentProcessingError, Granularity, ValidationError
from viewer import service
from viewer.models import ProcessResponse

applicable_path_prefix = settings.PPS_PIPELINE_API_PATH_PREFIX
if applicable_path_prefix == "/" :
    applicable_path_prefix = ""

router = APIRouter(prefix=applicable_path_prefix, tags=["BBox viewer"])


@router.post("/process")
async def process_upload(
    file: Annotated[UploadFile | None, File(description="Single upload (kept for older viewer builds)")] = None,
    files: Annotated[list[UploadFile] | None, File(description="One PDF, or one or more images")] = None,
    granularity: Annotated[Granularity, Form(description='"lines" (default) or "words": unit of each OCR entry')] = "lines",
) -> ProcessResponse:
    """/parse plus each page image and the raw PPStructure output, for the BBox viewer."""
    logger.info("process api called ...")
    uploads = [*(files or []), *([file] if file is not None else [])]
    try:
        return await service.process_uploads(uploads, granularity)
    except ValidationError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except DocumentProcessingError as de:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(de))
