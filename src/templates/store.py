"""SQLite store for document templates (a temporary store while the template flow is evaluated)."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS templates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_type        TEXT    NOT NULL,
    name            TEXT    NOT NULL,
    page_width      INTEGER NOT NULL,
    page_height     INTEGER NOT NULL,
    static_elements TEXT    NOT NULL,   -- JSON [{"id", "text", "bbox"}]
    fields          TEXT    NOT NULL,   -- JSON [{"key", "bbox", "anchor", "multiline", "check"}]
    stats           TEXT    NOT NULL DEFAULT '{}',  -- JSON learning statistics
    match_count     INTEGER NOT NULL DEFAULT 0,
    reference_png   BLOB    NOT NULL,   -- sample page with field values and masked regions blanked
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_templates_doc_type ON templates (doc_type);

CREATE TABLE IF NOT EXISTS doc_types (
    doc_type    TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    field_list  TEXT NOT NULL,              -- JSON ["full_name", ...]: keys to read from this type
    notes       TEXT NOT NULL DEFAULT '',   -- extra instructions for drafting, e.g. grouped keys
    source      TEXT NOT NULL,              -- 'seed' (doc_types.json) or 'gemini' (created while drafting)
    updated_at  TEXT NOT NULL,
    markers     TEXT NOT NULL DEFAULT '[]', -- JSON [{"text", "source", "trusted", "seen", "others_without"}]: phrases printed on every document of the type
    patterns    TEXT NOT NULL DEFAULT '[]'  -- JSON regular expressions of the type's numbers
);
"""

# Seed for the doc_types table, read once when the table is empty
DOC_TYPES_SEED = Path(__file__).with_name("doc_types.json")

_JSON_COLUMNS = ("static_elements", "fields", "stats")


def db_path() -> Path:
    return Path(os.getenv("TEMPLATES_DB_PATH", "data/templates.sqlite"))


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    # Databases created before markers existed get the columns, and their seeded types the markers
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(doc_types)")}
    added = [c for c in ("markers", "patterns") if c not in columns]
    for column in added:
        conn.execute(f"ALTER TABLE doc_types ADD COLUMN {column} TEXT NOT NULL DEFAULT '[]'")
    if conn.execute("SELECT COUNT(*) FROM doc_types").fetchone()[0] == 0 and DOC_TYPES_SEED.is_file():
        conn.executemany(
            """INSERT INTO doc_types (doc_type, name, field_list, notes, source, updated_at, markers, patterns)
               VALUES (?, ?, ?, ?, 'seed', ?, ?, ?)""",
            [
                (k, v["name"], json.dumps(v["fields"]), v.get("notes", ""), _now(), *_seed_markers(v))
                for k, v in _seed().items()
            ],
        )
    elif added:
        conn.executemany(
            "UPDATE doc_types SET markers = ?, patterns = ? WHERE doc_type = ? AND source = 'seed'",
            [(*_seed_markers(v), k) for k, v in _seed().items()],
        )
    conn.commit()
    return conn


def _seed() -> dict[str, dict[str, Any]]:
    if not DOC_TYPES_SEED.is_file():
        return {}
    return {k: v for k, v in json.loads(DOC_TYPES_SEED.read_text()).items() if not k.startswith("_")}


def _seed_markers(spec: dict[str, Any]) -> tuple[str, str]:
    """markers and patterns columns of a seeded type: its hand-checked markers are trusted."""
    markers = [{"text": text, "source": "seed", "trusted": True} for text in spec.get("markers", [])]
    return json.dumps(markers), json.dumps(spec.get("patterns", []))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row(row: sqlite3.Row | None, *, with_reference: bool) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    for column in _JSON_COLUMNS:
        data[column] = json.loads(data[column])
    if not with_reference:
        data.pop("reference_png", None)
    return data


def create(*, doc_type: str, name: str, page_width: int, page_height: int, static_elements: list, fields: list, reference_png: bytes) -> int:
    now = _now()
    with _connect() as conn:
        cursor = conn.execute(
            """INSERT INTO templates (doc_type, name, page_width, page_height, static_elements, fields, stats,
                                      reference_png, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, '{}', ?, ?, ?)""",
            (doc_type, name, page_width, page_height, json.dumps(static_elements), json.dumps(fields), reference_png, now, now),
        )
        return int(cursor.lastrowid)


def get(template_id: int, *, with_reference: bool = False) -> dict[str, Any] | None:
    with _connect() as conn:
        return _row(conn.execute("SELECT * FROM templates WHERE id = ?", (template_id,)).fetchone(), with_reference=with_reference)


def list_templates(doc_type: str | None = None, *, with_reference: bool = False) -> list[dict[str, Any]]:
    with _connect() as conn:
        if doc_type:
            rows = conn.execute("SELECT * FROM templates WHERE doc_type = ? ORDER BY id", (doc_type,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM templates ORDER BY id").fetchall()
        return [_row(r, with_reference=with_reference) for r in rows]


def delete(template_id: int) -> bool:
    with _connect() as conn:
        return conn.execute("DELETE FROM templates WHERE id = ?", (template_id,)).rowcount > 0


def record_match(template_id: int, stats: dict[str, Any]) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE templates SET stats = ?, match_count = match_count + 1, updated_at = ? WHERE id = ?",
            (json.dumps(stats), _now(), template_id),
        )


def list_doc_types() -> dict[str, dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM doc_types ORDER BY doc_type").fetchall()
    return {
        row["doc_type"]: {
            "name": row["name"],
            "fields": json.loads(row["field_list"]),
            "notes": row["notes"],
            "source": row["source"],
            "markers": json.loads(row["markers"]),
            "patterns": json.loads(row["patterns"]),
        }
        for row in rows
    }


def save_markers(markers_by_type: dict[str, list[dict[str, Any]]]) -> None:
    """Replace the markers of each given doc type."""
    with _connect() as conn:
        conn.executemany(
            "UPDATE doc_types SET markers = ?, updated_at = ? WHERE doc_type = ?",
            [(json.dumps(markers), _now(), doc_type) for doc_type, markers in markers_by_type.items()],
        )


def save_doc_type(*, doc_type: str, name: str, fields: list[str], notes: str = "", source: str) -> None:
    with _connect() as conn:
        conn.execute(
            """INSERT INTO doc_types (doc_type, name, field_list, notes, source, updated_at) VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT (doc_type) DO UPDATE SET field_list = excluded.field_list, updated_at = excluded.updated_at""",
            (doc_type, name, json.dumps(fields), notes, source, _now()),
        )
