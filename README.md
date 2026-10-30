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
  alignment/    *-with-reference: align uploads to a reference (this project only)
  templates/    document templates, experimental              (this project only)
  normalize/    /normalize: crop, flatten and turn pages upright (this project only)
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
| `POST /normalize-with-reference` | `files` plus `reference`: each page aligned to the reference (the alignment step of the `-with-reference` endpoints, no OCR), as `pages[].image_base64`, with `alignment` per page and the reference's own pages under `reference`. |
| `POST /normalize` | Per page: the document cut out of its background (photo or scan margins), its perspective flattened and turned upright (0/90/180/270), as `image_base64`; plus the upload (`original_base64`), the `corners` found on it, the `rotation` applied and `warnings`. `crop` and `orient` form fields (default `true`) switch the steps. |

Coordinates are pixels of the page image: images as uploaded (EXIF rotation applied), PDFs rendered at 300 DPI.

## Templates (experimental)

A template stores a document layout once, so later uploads of that layout are read without picking
boxes again. It holds the field boxes, the sample page with field values and `mask_regions` blanked
(used to recognise and align uploads), and its static elements: printed text outside every field,
such as labels, found automatically on the sample.

`POST /templates/extract` aligns page 1 of the upload to each template of the `doc_type` (all
templates when no `doc_type` is given) and uses the best one with at least 200 matching keypoints.
The aligned page is OCR'd and the static elements are found by their text, which measures how far
parts of the page moved (a missing line moves the fields below it up). Each field is read at its box
moved by its anchor.

When no template matches, Gemini drafts one (`learn`, on by default; needs `GEMINI_API_KEY`):

1. the page is normalized as by `/normalize` (cut out, flattened, turned upright) and OCR'd once;
2. Gemini gets the page image, the OCR words numbered and the fields of the `doc_type` from
   `src/templates/doc_types.json` (built from `LOAN_DOCUMENT_EXTRACTION_FIELDS.md`), and answers in
   word numbers: each field's value and label, the text printed on every document of the layout, the
   personal data, and photo/QR boxes. Without a `doc_type` it also says what the document is;
3. its answer is checked against the OCR (the words must exist, read as the value Gemini saw, belong
   to one field only; labels must sit before or above their value and be unique on the page) and boxes
   are taken from the OCR words, widened into the free space beside each value;
4. the template is saved and the fields are read with it from the same OCR (`created` is true).

A document Gemini says is not of the requested `doc_type`, or one that nearly matches a stored
template (likely a poor capture of a known layout), gets no new template.

Before Gemini is asked, a keyword check refuses an upload that is plainly another type (422, no LLM
call): a `doc_type` with `markers` (phrases printed on every such document) or `patterns` (its
numbers) is refused only when the page shows none of them but a trusted marker of another type, e.g.
a PAN card ("Income Tax Department") sent as `aadhaar_card`. Anything less clear goes to Gemini.

Markers live in the `doc_types` table (`GET /templates/doc-types` lists them):

- seed markers come from `doc_types.json` (PAN, Aadhaar, salary slip, bank statement, RC, vehicle
  insurance, sale receipt), were checked by hand on sample documents and are trusted: on the samples
  no upload of the right type was refused, also with up to 70% of the OCR words dropped;
- Gemini proposes 2-4 markers whenever it drafts a template. One is kept only when the OCR read it
  on the page, it is long enough, and it is neither another type's marker nor printed on another
  type's template. It first only counts for its own type; it is trusted after it was read on 5
  documents of its type while 3 documents of other types went without it, and it is removed as soon
  as a document of another type shows it. Documents count when a template matched them or Gemini
  drafted them.

Databases created before markers existed get the `markers` and `patterns` columns, and their seeded
types the seed markers, when the store is next opened.

| `anchor` | Field moves with |
|---|---|
| `page` (default) | nothing: fixed print position (Aadhaar number) |
| `auto` | the label on its left, else the nearest text above (resolved when the template is created) |
| `field:<key>` | another field, for text that follows variable-length content |

A field can also be `multiline` (grows down while lines continue) and have a `check`
(`aadhaar`, `pan`, `date`, `number`): its value is cut to the matching text, or another row nearby is
tried. Every match is recorded: `GET /templates/{id}` shows under `learning` which static elements
are really variable (they stop counting) and which fields would be better anchored differently.

| Endpoint | Does |
|---|---|
| `POST /templates` | `doc_type`, `sample` (file), `fields` `[{"key", "bbox", "anchor", "multiline", "check"}]`, optional `name`, `mask_regions` `[[x1, y1, x2, y2]]` (photo, QR code). Returns the template and the values read from the sample. |
| `POST /templates/extract` | `files`, optional `doc_type` (tries its templates) or `template_id` (only that one), `learn` (default `true`: draft a template with Gemini when none matches) |
| `GET /templates`, `GET /templates/{id}`, `DELETE /templates/{id}` | list (`?doc_type=`), show, delete |
| `GET /templates/{id}/reference` | the template's reference page (PNG) its boxes are in |

Templates live in SQLite at `TEMPLATES_DB_PATH` (default `data/templates.sqlite`), a temporary store
while the flow is evaluated. Only page 1 of a document is used.

Gemini settings in `.env`: `GEMINI_API_KEY`, `GEMINI_MODEL_NAME` (default `gemini-3-flash-preview`),
`GEMINI_TIMEOUT` (seconds, default 180) and `GEMINI_MAX_RETRIES` (default 3). The client is
`src/parser_client/gemini_client.py`, ported from alos.
