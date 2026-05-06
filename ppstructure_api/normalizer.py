from __future__ import annotations

from typing import Any


def build_empty_viewer_result(*, width: int, height: int, img_idx: int) -> dict[str, Any]:
    return {
        "type": "figure",
        "bbox": [0, 0, width, height],
        "res": [],
        "img_idx": img_idx,
        "score": 0.0,
    }


def build_viewer_result(
    raw_result: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    img_idx: int,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for region in raw_result:
        _collect_ocr_items(region.get("res"), items)

    if not items:
        for region in raw_result:
            bbox = _bbox_to_text_region(region.get("bbox"))
            if bbox is None:
                continue
            items.append(
                {
                    "text": str(region.get("type") or "region"),
                    "confidence": _as_float(region.get("score"), default=1.0),
                    "text_region": bbox,
                }
            )

    result = build_empty_viewer_result(width=width, height=height, img_idx=img_idx)
    result.update({"res": items, "score": 1.0})
    return result


def _collect_ocr_items(value: Any, items: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        if "text" in value and "text_region" in value:
            item = {
                "text": str(value.get("text") or ""),
                "confidence": _as_float(value.get("confidence"), default=1.0),
                "text_region": value["text_region"],
            }
            items.append(item)
            return

        for child in value.values():
            _collect_ocr_items(child, items)
        return

    if isinstance(value, list):
        for child in value:
            _collect_ocr_items(child, items)


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
