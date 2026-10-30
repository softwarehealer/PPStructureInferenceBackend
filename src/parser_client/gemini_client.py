"""Gemini client, ported from alos (alos/parser_client/gemini_client.py).

Added here: an optional response schema, so Gemini answers with JSON in that shape, and
GeminiError, raised when every attempt failed.
"""
import asyncio
import io
import time
from pathlib import Path
from typing import Any, Dict

from configs import app_logger as logger
from configs.settings import settings

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

_client = None


class GeminiError(Exception):
    """Gemini could not be reached or did not answer."""


def is_configured() -> bool:
    return genai is not None and bool(settings.GEMINI_API_KEY.get_secret_value())


def _get_client():
    global _client
    if _client is None:
        if genai is None:
            raise GeminiError("google-genai package is not installed. Run: pip install google-genai")
        _client = genai.Client(api_key=settings.GEMINI_API_KEY.get_secret_value())
    return _client


def _preprocess_image(image_data: bytes, max_side: int) -> bytes:
    """Resize and convert to JPEG to reduce payload size."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_data))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        return buf.getvalue()
    except Exception as exc:
        logger.warning("gemini_client: image preprocess failed, using raw bytes: %s", exc)
        return image_data


def _sync_request(model: str, prompt: str, img_bytes: bytes, mime_type: str, response_schema: Any) -> str:
    client = _get_client()
    image_part = genai_types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
    config = genai_types.GenerateContentConfig(
        http_options=genai_types.HttpOptions(timeout=settings.GEMINI_TIMEOUT * 1000),
    )
    if response_schema is not None:
        config.response_mime_type = "application/json"
        config.response_schema = response_schema
    for attempt in range(settings.GEMINI_MAX_RETRIES):
        try:
            response = client.models.generate_content(model=model, contents=[image_part, prompt], config=config)
            return response.text or ""
        except Exception as exc:
            if attempt < settings.GEMINI_MAX_RETRIES - 1:
                logger.warning("gemini_client: attempt %d failed: %s — retrying", attempt + 1, exc)
            else:
                raise GeminiError(f"Gemini request failed after {settings.GEMINI_MAX_RETRIES} attempts: {exc}") from exc
    return ""


async def gemini_request(
    model: str,
    prompt: str,
    image_bstream: bytes,
    file_name: str,
    *,
    response_schema: Any = None,
    max_side: int = 1600,
) -> Dict:
    """
    Send an image + prompt to Gemini and return a dict with a 'response' key,
    matching the shape returned by llama_cpp_request in alos. With ``response_schema``
    (a pydantic model), the response text is JSON in that shape.
    """
    logger.info("gemini_request: file=%s model=%s", file_name, model)
    t0 = time.perf_counter()

    suffix = Path(file_name or "").suffix.lower()
    mime_type = MIME_TYPES.get(suffix, "image/jpeg")
    processed = _preprocess_image(image_bstream, max_side)
    if processed is not image_bstream:
        mime_type = "image/jpeg"

    loop = asyncio.get_running_loop()
    try:
        response_text = await loop.run_in_executor(
            None,
            lambda: _sync_request(model, prompt, processed, mime_type, response_schema),
        )
        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info("gemini_request: done file=%s duration_ms=%d", file_name, duration_ms)
        return {
            "model": model,
            "response": response_text,
            "total_duration": duration_ms,
        }
    except Exception as exc:
        logger.error("gemini_request: error file=%s: %s", file_name, repr(exc))
        raise
