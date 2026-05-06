# PPStructure API

Minimal inference-only extraction of the local PaddleOCR PPStructure flow.

## Python API

```python
from ppstructure_api import PPStructureEngine

engine = PPStructureEngine()
result = engine.predict_file("/path/to/image.png")
```

The result is JSON-safe: raw cropped image arrays are removed and NumPy values are converted to normal Python values.

## Smoke CLI

```bash
python3 scripts/ppstructure_smoke.py /path/to/image.png --output-json output/result.json
```

## HTTP API

Install dependencies in your PaddleOCR venv, then start the API:

```bash
pip install -r requirements.txt
python3 scripts/serve_api.py --host 127.0.0.1 --port 8000 --log-level info
```

Endpoints:

- `GET /health`
- `POST /process` with multipart field `file` for backward-compatible single upload
- `POST /process` with repeated multipart field `files` for one PDF or multiple image uploads

Supported uploads: `pdf`, `png`, `jpg`, `jpeg`, `webp`. A request may contain either one PDF or one/more image files.

The response contains `pages`; each page includes:

- `success`: whether OCR completed for that page.
- `error`: a page-level error message when OCR failed.
- `image_base64`: the page image, returned even if OCR failed when possible.
- `viewer_result`: Flutter viewer-compatible JSON. Failed pages return an empty result with the correct page bbox.
- `raw_result`: original PPStructure JSON-safe result list when OCR succeeds.

For PDFs, pages are split to PNGs in a temporary request folder and processed independently. For multi-image uploads, images are saved and normalized into the same temporary page folder before processing. If OCR fails on one page, the failure is logged and returned for that page while remaining pages continue.

PDF splitting uses `pdf2image`, which requires Poppler installed on the system. On Ubuntu/Debian:

```bash
sudo apt-get install poppler-utils
```

## Included Runtime Assets

- `inference/en_PP-OCRv3_det_infer`
- `inference/en_PP-OCRv4_rec_infer`
- `inference/en_ppstructure_mobile_v2.0_SLANet_infer`
- `inference/picodet_lcnet_x1_0_fgd_layout_infer`
- English OCR, table-structure, and PubLayNet layout dictionaries under `ppocr/utils`.

Training, KIE, PDF recovery, generated outputs, and old `src` scripts are intentionally excluded.
