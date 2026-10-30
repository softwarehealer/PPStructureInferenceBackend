from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette import status

from configs.settings import settings
from configs import app_logger as logger
from normalize import service
from normalize.models import NormalizeResponse
from pipeline.models import DocumentProcessingError, ValidationError

applicable_path_prefix = settings.PPS_PIPELINE_API_PATH_PREFIX
if applicable_path_prefix == "/" :
    applicable_path_prefix = ""

router = APIRouter(prefix=applicable_path_prefix, tags=["Normalize"])


@router.post("/normalize")
async def normalize(
    files: Annotated[list[UploadFile], File()],
    crop: Annotated[bool, Form(description="Cut the document out of its background and flatten its perspective")] = True,
    orient: Annotated[bool, Form(description="Turn the page upright (0, 90, 180 or 270 degrees)")] = True,
) -> NormalizeResponse:
    """Each page cropped to the document, flattened and turned upright, with the original and the
    corners found on it."""
    logger.info("normalize api called ...")
    try:
        return await service.normalize_uploads(files, crop=crop, orient=orient)
    except ValidationError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except DocumentProcessingError as de:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(de))
