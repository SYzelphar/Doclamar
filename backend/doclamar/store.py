"""Persistent chunk index: one SQLite file with files, chunks (+ embeddings) and an FTS5 keyword index.

Indexing happens once per file version (path + size + mtime). A query only
embeds the question and searches this index, so query cost no longer grows with
the number of files the way the old parse-everything-per-query pipeline did.
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .chunking import Chunk

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL,
    path_key    TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    ext         TEXT NOT NULL,
    size        INTEGER NOT NULL,
    mtime       REAL NOT NULL,
    status      TEXT NOT NULL,   -- indexed | no_text | error | too_large
    error       TEXT,
    num_pages   INTEGER,
    num_chunks  INTEGER NOT NULL DEFAULT 0,
    ocr_pages   INTEGER NOT NULL DEFAULT 0,   -- pages/images read with OCR
    indexed_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    section     TEXT,
    page_start  INTEGER,
    page_end    INTEGER,
    ocr         INTEGER NOT NULL DEFAULT 0,
    embedding   BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, section, name, tokenize = 'porter unicode61'
);
CREATE TABLE IF NOT EXISTS folders (
    path_key    TEXT PRIMARY KEY,
    path        TEXT NOT NULL,
    synced_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL
);
"""

_STOPWORDS = set("""
a about above after again all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from
further had has have having he her here hers him his how i if in into is it its itself
just me more most my no nor not now of off on once only or other our out over own same
she should so some such than that the their them then there these they this those
through to too under until up very was we were what when where which while who whom
why will with would you your tell explain describe give show find please document
documents file files paper
""".split())


def path_key(path: str) -> str:
    """Canonical key: absolute, and case-folded on Windows."""
    return os.path.normcase(os.path.abspath(path))


def folder_prefix(folder: str) -> str:
    key = path_key(folder)
    return key if key.endswith(os.sep) else key + os.sep


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fts_query(text: str) -> Optional[str]:
    tokens = []
    for tok in re.findall(r"\w+", text.lower()):
        if len(tok) > 1 and tok not in _STOPWORDS and tok not in tokens:
            tokens.append(tok)
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens[:32])


class Scope:
    """Which files a search may look at: everything under a folder, or specific files."""

    def __init__(self, folder: Optional[str] = None, files: Optional[Sequence[str]] = None):
        if not folder and not files:
            raise ValueError("Scope needs a folder or a list of files")
        self.folder = folder
        self.files = list(files) if files else None

    def sql(self, alias: str = "f") -> Tuple[str, list]:
        if self.files:
            keys = [path_key(p) for p in self.files]
            return f"{alias}.path_key IN ({','.join('?' * len(keys))})", keys
        prefix = folder_prefix(self.folder)
        return f"substr({alias}.path_key, 1, ?) = ?", [len(prefix), prefix]


class IndexStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._version = 0
        self._cache_lock = threading.Lock()
        self._cache_version = -1
        self._cache: Tuple[np.ndarray, np.ndarray, np.ndarray] = (
            np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 0), np.float32)
        )
        with self._db() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            # v1 -> v2: OCR bookkeeping. Additive, so existing indexes are kept.
            for table, column in (("files", "ocr_pages"), ("chunks", "ocr")):
                if column not in {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0")
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        """One short-lived connection per operation; commits on success, always closes."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------- meta

    def get_meta(self, key: str) -> Optional[str]:
        with self._db() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._write_lock, self._db() as conn:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value),
            )

    def clear(self) -> None:
        """Drop every indexed file (e.g. after switching embedding model)."""
        with self._write_lock, self._db() as conn:
            conn.execute("DELETE FROM chunks_fts")
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM files")
            conn.execute("DELETE FROM folders")
            self._version += 1

    # ------------------------------------------------------------------ files

    def files_in(self, scope: Scope) -> Dict[str, sqlite3.Row]:
        where, params = scope.sql()
        with self._db() as conn:
            rows = conn.execute(
                f"SELECT id, path, path_key, size, mtime, status, ocr_pages FROM files f WHERE {where}", params
            ).fetchall()
        return {r["path_key"]: r for r in rows}

    def get_file(self, path: str) -> Optional[sqlite3.Row]:
        with self._db() as conn:
            return conn.execute("SELECT * FROM files WHERE path_key = ?", (path_key(path),)).fetchone()

    def save_file(
        self,
        path: str,
        size: int,
        mtime: float,
        status: str,
        chunks: Sequence[Chunk] = (),
        embeddings: Optional[np.ndarray] = None,
        num_pages: Optional[int] = None,
        error: Optional[str] = None,
        ocr_pages: int = 0,
    ) -> int:
        """Insert or replace a file and all of its chunks atomically."""
        name = os.path.basename(path)
        ext = Path(path).suffix.lstrip(".").lower()
        key = path_key(path)
        if chunks and (embeddings is None or len(embeddings) != len(chunks)):
            raise ValueError("need one embedding per chunk")

        with self._write_lock, self._db() as conn:
            row = conn.execute("SELECT id FROM files WHERE path_key = ?", (key,)).fetchone()
            values = (path, name, ext, size, mtime, status, error, num_pages, len(chunks), ocr_pages, _now())
            if row:
                file_id = row["id"]
                self._delete_chunks(conn, [file_id])
                conn.execute(
                    "UPDATE files SET path=?, name=?, ext=?, size=?, mtime=?, status=?, error=?,"
                    " num_pages=?, num_chunks=?, ocr_pages=?, indexed_at=? WHERE id=?",
                    values + (file_id,),
                )
            else:
                file_id = conn.execute(
                    "INSERT INTO files (path, name, ext, size, mtime, status, error, num_pages,"
                    " num_chunks, ocr_pages, indexed_at, path_key) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    values + (key,),
                ).lastrowid

            for chunk, vector in zip(chunks, embeddings if embeddings is not None else []):
                chunk_id = conn.execute(
                    "INSERT INTO chunks (file_id, ordinal, text, section, page_start, page_end, ocr, embedding)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (file_id, chunk.ordinal, chunk.text, chunk.section, chunk.page_start,
                     chunk.page_end, int(chunk.ocr), np.asarray(vector, dtype=np.float32).tobytes()),
                ).lastrowid
                conn.execute(
                    "INSERT INTO chunks_fts (rowid, text, section, name) VALUES (?,?,?,?)",
                    (chunk_id, chunk.text, chunk.section or "", name),
                )
            self._version += 1
        return file_id

    def delete_files(self, file_ids: Iterable[int]) -> None:
        ids = list(file_ids)
        if not ids:
            return
        with self._write_lock, self._db() as conn:
            self._delete_chunks(conn, ids)
            conn.execute(f"DELETE FROM files WHERE id IN ({','.join('?' * len(ids))})", ids)
            self._version += 1

    @staticmethod
    def _delete_chunks(conn: sqlite3.Connection, file_ids: List[int]) -> None:
        marks = ",".join("?" * len(file_ids))
        conn.execute(
            f"DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE file_id IN ({marks}))",
            file_ids,
        )
        conn.execute(f"DELETE FROM chunks WHERE file_id IN ({marks})", file_ids)

    # ---------------------------------------------------------------- folders

    def mark_folder_synced(self, folder: str) -> None:
        with self._write_lock, self._db() as conn:
            conn.execute(
                "INSERT INTO folders (path_key, path, synced_at) VALUES (?,?,?)"
                " ON CONFLICT(path_key) DO UPDATE SET path=excluded.path, synced_at=excluded.synced_at",
                (path_key(folder), os.path.abspath(folder), _now()),
            )

    def folder_summary(self, folder: str) -> dict:
        scope = Scope(folder=folder)
        where, params = scope.sql()
        with self._db() as conn:
            synced = conn.execute(
                "SELECT synced_at FROM folders WHERE path_key = ?", (path_key(folder),)
            ).fetchone()
            counts = dict(conn.execute(
                f"SELECT status, COUNT(*) FROM files f WHERE {where} GROUP BY status", params
            ).fetchall())
            chunks, ocr_files = conn.execute(
                f"SELECT COALESCE(SUM(num_chunks), 0), COUNT(CASE WHEN status = 'indexed' AND ocr_pages > 0"
                f" THEN 1 END) FROM files f WHERE {where}", params
            ).fetchone()
            issues = conn.execute(
                f"SELECT path, status, error FROM files f WHERE {where} AND status != 'indexed'"
                " ORDER BY name LIMIT 50", params
            ).fetchall()
        return {
            "synced_at": synced["synced_at"] if synced else None,
            "files_indexed": counts.get("indexed", 0),
            "files_total": sum(counts.values()),
            "chunks": chunks,
            "files_ocr": ocr_files,
            "issues": [dict(r) for r in issues],
        }

    # ----------------------------------------------------------------- search

    def _vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All chunk embeddings as one matrix, reloaded only after the index changes."""
        with self._cache_lock:
            if self._cache_version != self._version:
                version = self._version
                with self._db() as conn:
                    rows = conn.execute("SELECT id, file_id, embedding FROM chunks").fetchall()
                if rows:
                    ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
                    file_ids = np.fromiter((r[1] for r in rows), dtype=np.int64, count=len(rows))
                    matrix = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
                else:
                    ids = file_ids = np.zeros(0, np.int64)
                    matrix = np.zeros((0, 0), np.float32)
                self._cache = (ids, file_ids, matrix)
                self._cache_version = version
            return self._cache

    def scope_file_ids(self, scope: Scope) -> np.ndarray:
        where, params = scope.sql()
        with self._db() as conn:
            rows = conn.execute(f"SELECT id FROM files f WHERE {where}", params).fetchall()
        return np.array([r[0] for r in rows], dtype=np.int64)

    def vector_search(self, query_vec: np.ndarray, scope: Scope, limit: int) -> List[Tuple[int, float]]:
        ids, file_ids, matrix = self._vectors()
        if not len(ids):
            return []
        mask = np.isin(file_ids, self.scope_file_ids(scope))
        if not mask.any():
            return []
        scores = matrix[mask] @ query_vec.astype(np.float32)
        candidate_ids = ids[mask]
        k = min(limit, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(candidate_ids[i]), float(scores[i])) for i in top]

    def keyword_search(self, query: str, scope: Scope, limit: int) -> List[Tuple[int, float]]:
        match = fts_query(query)
        if not match:
            return []
        where, params = scope.sql()
        sql = (
            "SELECT c.id, bm25(chunks_fts, 1.0, 0.6, 0.3) AS score FROM chunks_fts"
            " JOIN chunks c ON c.id = chunks_fts.rowid JOIN files f ON f.id = c.file_id"
            f" WHERE chunks_fts MATCH ? AND {where} ORDER BY score LIMIT ?"
        )
        with self._db() as conn:
            rows = conn.execute(sql, [match, *params, limit]).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]

    def get_chunks(self, chunk_ids: Sequence[int]) -> Dict[int, dict]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        with self._db() as conn:
            rows = conn.execute(
                "SELECT c.id, c.file_id, c.ordinal, c.text, c.section, c.page_start, c.page_end, c.ocr,"
                f" f.path, f.name FROM chunks c JOIN files f ON f.id = c.file_id WHERE c.id IN ({marks})",
                list(chunk_ids),
            ).fetchall()
        return {r["id"]: dict(r) for r in rows}
