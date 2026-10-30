from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from starlette import status

from configs.settings import settings
from configs import app_logger as logger
from parser_client.gemini_client import GeminiError
from pipeline.models import DocumentProcessingError, ValidationError
from templates import service
from templates.drafting import DraftingError
from templates.models import DocTypeInfo, TemplateCreated, TemplateDetail, TemplateExtractionResponse, TemplateSummary

applicable_path_prefix = settings.PPS_PIPELINE_API_PATH_PREFIX
if applicable_path_prefix == "/" :
    applicable_path_prefix = ""

router = APIRouter(prefix=applicable_path_prefix, tags=["Templates"])

FIELDS_DESCRIPTION = (
    'JSON list: [{"key": "dob", "bbox": [x1, y1, x2, y2], "anchor": "auto", "multiline": false, "check": "date"}]. '
    'anchor: "page" (fixed position, default), "auto" (label to its left or nearest text above) or "field:<key>". '
    'check: "aadhaar", "pan", "date" or "number".'
)


@router.post("/templates")
async def create_template(
    doc_type: Annotated[str, Form()],
    sample: Annotated[UploadFile, File(description="Sample document; its first page becomes the template")],
    fields: Annotated[str, Form(description=FIELDS_DESCRIPTION)],
    name: Annotated[str | None, Form()] = None,
    mask_regions: Annotated[str | None, Form(description="JSON list of [x1, y1, x2, y2]: regions that differ per document but are no field (photo, QR code)")] = None,
) -> TemplateCreated:
    """Create a template from a sample document and the field boxes picked on it."""
    logger.info("create template api called ...")
    try:
        return await service.create_template(doc_type=doc_type, name=name, sample=sample, fields_json=fields, mask_regions_json=mask_regions)
    except ValidationError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except DocumentProcessingError as de:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(de))


@router.post("/templates/extract")
async def extract_with_templates(
    files: Annotated[list[UploadFile], File()],
    doc_type: Annotated[str | None, Form(description="Try the templates of this document type (all templates when omitted); a template Gemini drafts gets this doc_type")] = None,
    template_id: Annotated[int | None, Form(description="Try only this template (no drafting)")] = None,
    learn: Annotated[bool, Form(description="When no template matches, let Gemini draft one from this document, save it and use it")] = True,
) -> TemplateExtractionResponse:
    """Read the document's fields with the best matching stored template.

    With no match, Gemini drafts a template from the document (`created` is true) and the fields are
    read with it; `matched` and `created` both false means no fields could be read.
    """
    logger.info("extract with templates api called ...")
    try:
        return await service.extract_with_templates(files, doc_type=doc_type, template_id=template_id, learn=learn)
    except ValidationError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except (DocumentProcessingError, DraftingError) as de:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(de))
    except GeminiError as ge:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(ge))


@router.get("/templates")
async def list_templates(doc_type: str | None = None) -> list[TemplateSummary]:
    return service.list_templates(doc_type)


@router.get("/templates/doc-types")
async def list_doc_types() -> list[DocTypeInfo]:
    """Document types and the keys read from them (declared before /templates/{template_id})."""
    return service.list_doc_types()


@router.get("/templates/{template_id}")
async def get_template(template_id: int) -> TemplateDetail:
    template = service.get_template(template_id)
    if template is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Template {template_id} does not exist.")
    return template


@router.get("/templates/{template_id}/reference", response_class=Response, responses={200: {"content": {"image/png": {}}}})
async def get_template_reference(template_id: int) -> Response:
    """The template's reference page as PNG: the sample with its field values and masked regions
    (photo, QR code) blanked; field and static boxes are in its pixels."""
    png = service.template_reference(template_id)
    if png is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Template {template_id} does not exist.")
    return Response(content=png, media_type="image/png")


@router.delete("/templates/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_template(template_id: int) -> None:
    if not service.delete_template(template_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Template {template_id} does not exist.")
