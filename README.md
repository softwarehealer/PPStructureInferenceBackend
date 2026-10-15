# PPStructure API

PPStructure inference API for the BBox viewer. It runs the same inference flow as `pps-pipeline`
(same vendored PaddleOCR, engine, preprocessing, settings and models) and adds the viewer's own
`/process` endpoint, so coordinates picked in the viewer match what `pps-pipeline` extracts.

## Layout

```
src/
  main.py       FastAPI app
  configs/      settings (.env), logging
  PaddleOCR/    vendored PaddleOCR inference code
  pipeline/     /health, /parse, /parse-text  + engine        (same as pps-pipeline)
  extraction/   /extract-by-coordinates                       (same as pps-pipeline)
  viewer/       /process for the BBox viewer                  (this project only)
inference/      OCR, layout and table models
```

`configs/`, `PaddleOCR/`, `pipeline/` and `extraction/` are copies of `pps-pipeline/src/pps_pipeline/*`
with the `pps_pipeline.` import prefix removed. Keep them in sync by copying them over again, so both
projects never produce different results for the same document.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env      # then set SECRET_KEY
./run.sh
```

The server starts on `localhost:8000`, which is what the viewer calls. `run.sh` uses `$PYTHON` if set,
else `.venv/bin/python`, else `python` on PATH; `HOST` and `PORT` override the defaults.

PDF splitting uses `pdf2image`, which needs Poppler (`poppler-utils` on Debian/Ubuntu, `poppler-utils` on Fedora).

## Endpoints

All uploads are multipart: one PDF, or one or more images (`pdf`, `png`, `jpg`, `jpeg`, `webp`).
Every endpoint takes an optional `granularity` form field: `lines` (default) or `words`.

| Endpoint | Returns |
|---|---|
| `GET /health` | `{"status": "ok"}` |
| `POST /parse` | Per page: every OCR line (or word) with `text`, `confidence`, `text_region` |
| `POST /parse-text` | Per page: just the list of texts |
| `POST /extract-by-coordinates` | `fields` form field `[{"key": "dob", "bbox": [x1, y1, x2, y2], "page": 0}]` returns `[{"key": "dob", "value": "..."}]` |
| `POST /process` | Like `/parse`, plus per page `image_base64` and the raw PPStructure output (`raw_result`), with the OCR result under `viewer_result`. Accepts `files` or a single `file`. |

Coordinates are pixels of the page image: images as uploaded (EXIF rotation applied), PDFs rendered at 300 DPI.
