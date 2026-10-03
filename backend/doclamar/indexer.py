"""Incremental folder indexing.

A sync walks the folder, compares (size, mtime) with what is stored, parses +
embeds only new or changed files, and drops files that disappeared. Jobs run one
at a time on a background worker so the API stays responsive.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from .chunking import Chunk, chunk_document
from .config import Settings
from .embeddings import Embedder, ModelUnavailable
from .parsing import SUPPORTED_EXTENSIONS, ParseError, parse_file
from .store import IndexStore, Scope, path_key

logger = logging.getLogger(__name__)

SKIP_DIRS = {
    "node_modules", "__pycache__", "venv", ".venv", "site-packages", "$recycle.bin",
    "system volume information", "appdata",
}

ACTIVE_STATES = ("queued", "scanning", "indexing")


@dataclass
class IndexJob:
    folder: str
    state: str = "queued"  # queued | scanning | indexing | done | error
    files_found: int = 0
    to_index: int = 0
    processed: int = 0
    failed: int = 0
    removed: int = 0
    current_file: Optional[str] = None
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES

    def to_dict(self) -> dict:
        return asdict(self)


def embedding_text(path: str, chunk: Chunk) -> str:
    """What gets embedded: the chunk plus where it lives, so short chunks keep their context."""
    title = Path(path).stem.replace("_", " ").replace("-", " ")
    header = f"{title} — {chunk.section}" if chunk.section else title
    return f"{header}\n{chunk.text}"


# Bump when parsing/chunking changes in a way that should rebuild existing indexes.
PIPELINE_VERSION = "1"


def index_fingerprint(settings: Settings) -> str:
    return "|".join(map(str, (
        PIPELINE_VERSION, settings.embed_model, settings.chunk_chars,
        settings.chunk_overlap_chars, settings.min_chunk_chars,
    )))


class Indexer:
    def __init__(self, store: IndexStore, embedder: Embedder, settings: Settings):
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="indexer")
        self._jobs: Dict[str, IndexJob] = {}
        self._lock = threading.Lock()

        fingerprint = index_fingerprint(settings)
        if store.get_meta("fingerprint") != fingerprint:
            # Old chunks/embeddings are incompatible; folders re-index on their next sync.
            if store.get_meta("fingerprint") is not None:
                logger.info("Index format changed; clearing the index so it is rebuilt.")
            store.clear()
            store.set_meta("fingerprint", fingerprint)

    # ----------------------------------------------------------- job control

    def start(self, folder: str) -> IndexJob:
        """Queue a background sync of `folder` (no-op if one is already queued/running)."""
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
        key = path_key(folder)
        with self._lock:
            job = self._jobs.get(key)
            if job and job.active:
                return job
            job = IndexJob(folder=os.path.abspath(folder))
            self._jobs[key] = job
        self._executor.submit(self._run_safely, job)
        return job

    def job(self, folder: str) -> Optional[IndexJob]:
        return self._jobs.get(path_key(folder))

    def is_busy(self, folder: str) -> bool:
        job = self.job(folder)
        return bool(job and job.active)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run_safely(self, job: IndexJob) -> None:
        try:
            self.sync_folder(job.folder, job)
        except Exception as e:
            logger.exception("Indexing failed for %s", job.folder)
            job.state, job.error = "error", str(e)
        finally:
            job.current_file = None
            job.finished_at = time.time()

    # --------------------------------------------------------------- syncing

    def sync_folder(self, folder: str, job: Optional[IndexJob] = None) -> IndexJob:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
        job = job or IndexJob(folder=os.path.abspath(folder))
        job.state = "scanning"
        found = self._scan(folder)
        job.files_found = len(found)

        known = self.store.files_in(Scope(folder=folder))
        gone = [row["id"] for key, row in known.items() if key not in found]
        self.store.delete_files(gone)
        job.removed = len(gone)

        todo = [
            (path, size, mtime)
            for key, (path, size, mtime) in found.items()
            if key not in known
            or known[key]["size"] != size
            or abs(known[key]["mtime"] - mtime) > 1e-3
        ]
        job.to_index = len(todo)
        job.state = "indexing"
        logger.info("Sync %s: %d files, %d to index, %d removed", folder, len(found), len(todo), len(gone))

        for path, size, mtime in todo:
            job.current_file = path
            try:
                status = self._index_one(path, size, mtime)
                if status == "error":
                    job.failed += 1
            except ModelUnavailable:
                raise  # nothing else will work either; fail the whole job with a clear message
            except Exception:
                logger.exception("Failed to index %s", path)
                job.failed += 1
            job.processed += 1

        self.store.mark_folder_synced(folder)
        job.state = "done"
        return job

    def index_file(self, path: str) -> dict:
        """Index one file now (used when chatting with a specific document)."""
        stat = os.stat(path)
        row = self.store.get_file(path)
        if not (row and row["size"] == stat.st_size and abs(row["mtime"] - stat.st_mtime) <= 1e-3):
            self._index_one(os.path.abspath(path), stat.st_size, stat.st_mtime)
            row = self.store.get_file(path)
        return dict(row)

    def _scan(self, folder: str) -> Dict[str, Tuple[str, int, float]]:
        found: Dict[str, Tuple[str, int, float]] = {}
        for dirpath, dirnames, filenames in os.walk(folder, onerror=lambda e: None):
            dirnames[:] = [
                d for d in dirnames if not d.startswith(".") and d.lower() not in SKIP_DIRS
            ]
            for name in filenames:
                if name.startswith((".", "~$")):
                    continue
                if Path(name).suffix.lstrip(".").lower() not in SUPPORTED_EXTENSIONS:
                    continue
                path = os.path.abspath(os.path.join(dirpath, name))
                try:
                    stat = os.stat(path)
                except OSError:
                    continue
                found[path_key(path)] = (path, stat.st_size, stat.st_mtime)
        return found

    def _index_one(self, path: str, size: int, mtime: float) -> str:
        if size > self.settings.max_file_mb * 1024 * 1024:
            self.store.save_file(path, size, mtime, "too_large",
                                 error=f"Larger than {self.settings.max_file_mb} MB")
            return "too_large"
        try:
            doc = parse_file(path)
        except ParseError as e:
            self.store.save_file(path, size, mtime, "error", error=str(e))
            return "error"
        except Exception as e:  # corrupt files can make any parser raise anything
            logger.warning("Could not parse %s: %s", path, e)
            self.store.save_file(path, size, mtime, "error", error=f"Could not read file: {e}")
            return "error"

        s = self.settings
        chunks = chunk_document(doc, s.chunk_chars, s.chunk_overlap_chars, s.min_chunk_chars)
        if not chunks:
            hint = " (scanned PDF? it needs OCR)" if path.lower().endswith(".pdf") else ""
            self.store.save_file(path, size, mtime, "no_text", num_pages=doc.num_pages,
                                 error=f"No extractable text{hint}")
            return "no_text"

        vectors = self.embedder.embed_documents([embedding_text(path, c) for c in chunks])
        self.store.save_file(path, size, mtime, "indexed", chunks, vectors, num_pages=doc.num_pages)
        return "indexed"
