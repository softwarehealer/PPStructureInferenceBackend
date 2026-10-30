from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette import status

from alignment import service
from alignment.models import (
    ExtractionWithReferenceResponse,
    NormalizeWithReferenceResponse,
    ParseTextWithReferenceResponse,
    ParseWithReferenceResponse,
    ProcessWithReferenceResponse,
)
from configs.settings import settings
from configs import app_logger as logger
from extraction import service as extraction_service
from pipeline.models import DocumentProcessingError, Granularity, ValidationError

applicable_path_prefix = settings.PPS_PIPELINE_API_PATH_PREFIX
if applicable_path_prefix == "/" :
    applicable_path_prefix = ""

router = APIRouter(prefix=applicable_path_prefix, tags=["With reference"])

ReferenceFile = Annotated[
    UploadFile,
    File(description="Reference document (image or PDF) the uploads are aligned to before OCR"),
]
GranularityForm = Annotated[Granularity, Form(description='"lines" (default) or "words": unit of each OCR entry')]


@router.post("/parse-with-reference")
async def parse_with_reference(
    files: Annotated[list[UploadFile], File()],
    reference: ReferenceFile,
    granularity: GranularityForm = "lines",
) -> ParseWithReferenceResponse:
    """/parse after aligning each page to the reference, so coordinates are in the reference's pixel space."""
    logger.info("parse-with-reference api called ...")
    try:
        return await service.parse_with_reference(files, reference, granularity)
    except ValidationError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except DocumentProcessingError as de:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(de))


@router.post("/parse-text-with-reference")
async def parse_text_with_reference(
    files: Annotated[list[UploadFile], File()],
    reference: ReferenceFile,
    granularity: GranularityForm = "lines",
) -> ParseTextWithReferenceResponse:
    """/parse-text after aligning each page to the reference."""
    logger.info("parse-text-with-reference api called ...")
    try:
        return await service.parse_text_with_reference(files, reference, granularity)
    except ValidationError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except DocumentProcessingError as de:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(de))


@router.post("/extract-by-coordinates-with-reference")
async def extract_by_coordinates_with_reference(
    files: Annotated[list[UploadFile], File()],
    reference: ReferenceFile,
    fields: Annotated[str, Form(description='JSON list: [{"key": "surname", "bbox": [x1, y1, x2, y2], "page": 0}]')],
    granularity: GranularityForm = "lines",
) -> ExtractionWithReferenceResponse:
    """/extract-by-coordinates after aligning each page to the reference; boxes are picked on the reference."""
    logger.info("extract-by-coordinates-with-reference api called ...")
    try:
        # Validated before alignment and OCR so a bad request costs nothing.
        field_boxes = extraction_service.parse_field_boxes(fields)
        return await service.extract_by_coordinates_with_reference(files, reference, field_boxes, granularity)
    except ValidationError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except DocumentProcessingError as de:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(de))


@router.post("/process-with-reference")
async def process_with_reference(
    files: Annotated[list[UploadFile], File()],
    reference: ReferenceFile,
    granularity: GranularityForm = "lines",
) -> ProcessWithReferenceResponse:
    """/process after aligning each page to the reference; image_base64 is the aligned page."""
    logger.info("process-with-reference api called ...")
    try:
        return await service.process_with_reference(files, reference, granularity)
    except ValidationError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except DocumentProcessingError as de:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(de))


@router.post("/normalize-with-reference")
async def normalize_with_reference(
    files: Annotated[list[UploadFile], File()],
    reference: ReferenceFile,
) -> NormalizeWithReferenceResponse:
    """Each page aligned to the reference, as images only (no OCR): the alignment step of
    /process-with-reference, to check it quickly."""
    logger.info("normalize-with-reference api called ...")
    try:
        return await service.normalize_with_reference(files, reference)
    except ValidationError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except DocumentProcessingError as de:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(de))
