from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette import status

from configs.settings import settings
from configs import app_logger as logger
from extraction import service
from extraction.models import ExtractionResponse
from pipeline.models import DocumentProcessingError, Granularity, ValidationError

applicable_path_prefix = settings.PPS_PIPELINE_API_PATH_PREFIX
if applicable_path_prefix == "/" :
    applicable_path_prefix = ""

router = APIRouter(prefix=applicable_path_prefix, tags=["Extraction"])


@router.post("/extract-by-coordinates")
async def extract_by_coordinates(
    files: Annotated[list[UploadFile], File()],
    fields: Annotated[str, Form(description='JSON list: [{"key": "surname", "bbox": [x1, y1, x2, y2], "page": 0}]')],
    granularity: Annotated[Granularity, Form(description='"lines" (default) or "words": unit of each OCR entry')] = "lines",
) -> ExtractionResponse:
    """Return the text inside each requested box, as key/value pairs."""
    logger.info("extract-by-coordinates api called ...")
    try:
        # Validated before parsing so a bad request doesn't pay for OCR.
        field_boxes = service.parse_field_boxes(fields)
        return await service.extract_by_coordinates(files, field_boxes, granularity)
    except ValidationError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except DocumentProcessingError as de:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(de))
