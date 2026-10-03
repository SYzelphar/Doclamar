from __future__ import annotations
import os
import json
import pickle
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss

from embeddings.embedder import embed_texts

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {"pdf", "txt", "docx", "md", "csv", "json"}
CONTENT_PREVIEW_BYTES = 512
INDEX_DIR = ".doclamar_index"
INDEX_FILE = "fileindex.faiss"
META_FILE = "fileindex_meta.pkl"
MANIFEST_FILE = "fileindex_manifest.json"


def _get_index_dir() -> str:
    home = Path.home()
    return str(home / INDEX_DIR)


def _build_document_text(file_path: str, filename: str) -> str:
    parts = []

    stem = Path(filename).stem.replace("_", " ").replace("-", " ")
    parts.append(stem)
    parts.append(filename)

    path_parts = Path(file_path).parts
    parts.extend(path_parts[-4:])

    try:
        with open(file_path, "rb") as fh:
            raw = fh.read(CONTENT_PREVIEW_BYTES)
            preview = raw.decode(errors="ignore").strip()
            if preview:
                parts.append(preview[:300])
    except OSError:
        pass

    return " | ".join(str(p) for p in parts if p)


def _file_hash(file_path: str) -> str:
    try:
        stat = os.stat(file_path)
        key = f"{file_path}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.md5(key.encode()).hexdigest()
    except OSError:
        return ""


def build_file_index(
    root_paths: List[str],
    on_progress: Optional[callable] = None,
) -> Dict:
    index_dir = _get_index_dir()
    os.makedirs(index_dir, exist_ok=True)

    all_files = []
    seen_paths = set()

    for root_path in root_paths:
        if not os.path.isdir(root_path):
            logger.warning(f"[FileIndex] Skipping invalid path: {root_path}")
            continue

        logger.info(f"[FileIndex] Scanning: {root_path}")
        for dirpath, _, filenames in os.walk(root_path):
            for filename in filenames:
                ext = Path(filename).suffix.lstrip(".").lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue
                full_path = os.path.join(dirpath, filename)
                if full_path in seen_paths:
                    continue
                seen_paths.add(full_path)
                all_files.append({
                    "file_path": full_path,
                    "file_name": filename,
                    "file_type": ext,
                    "size_kb": round(os.path.getsize(full_path) / 1024, 2),
                    "directory": dirpath,
                })

    if not all_files:
        return {"status": "empty", "files_indexed": 0}

    logger.info(f"[FileIndex] Building embeddings for {len(all_files)} files...")

    texts = []
    for i, f in enumerate(all_files):
        text = _build_document_text(f["file_path"], f["file_name"])
        texts.append(text)
        if on_progress and i % 50 == 0:
            on_progress(i, len(all_files))

    embeddings = embed_texts(texts)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = os.path.join(index_dir, INDEX_FILE)
    meta_path = os.path.join(index_dir, META_FILE)
    manifest_path = os.path.join(index_dir, MANIFEST_FILE)

    faiss.write_index(index, index_path)

    with open(meta_path, "wb") as fh:
        pickle.dump(all_files, fh)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "root_paths": root_paths,
        "files_indexed": len(all_files),
        "extensions_included": list(SUPPORTED_EXTENSIONS),
        "file_hashes": {
            f["file_path"]: _file_hash(f["file_path"]) for f in all_files
        },
    }

    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    logger.info(f"[FileIndex] Index built: {len(all_files)} files at {index_dir}")

    return {
        "status": "built",
        "files_indexed": len(all_files),
        "index_dir": index_dir,
        "root_paths": root_paths,
    }


def index_exists() -> bool:
    index_dir = _get_index_dir()
    return (
        os.path.exists(os.path.join(index_dir, INDEX_FILE)) and
        os.path.exists(os.path.join(index_dir, META_FILE)) and
        os.path.exists(os.path.join(index_dir, MANIFEST_FILE))
    )


def load_manifest() -> Optional[Dict]:
    manifest_path = os.path.join(_get_index_dir(), MANIFEST_FILE)
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path) as fh:
        return json.load(fh)


def check_staleness(threshold_changed_pct: float = 0.1) -> Dict:
    manifest = load_manifest()
    if not manifest:
        return {"stale": True, "reason": "No manifest found"}

    hashes = manifest.get("file_hashes", {})
    if not hashes:
        return {"stale": False, "changed": 0, "total": 0}

    changed = 0
    missing = 0
    for path, old_hash in hashes.items():
        if not os.path.exists(path):
            missing += 1
        elif _file_hash(path) != old_hash:
            changed += 1

    total = len(hashes)
    changed_pct = (changed + missing) / max(total, 1)

    return {
        "stale": changed_pct >= threshold_changed_pct,
        "changed": changed,
        "missing": missing,
        "total": total,
        "changed_pct": round(changed_pct * 100, 1),
        "created_at": manifest.get("created_at", "unknown"),
        "root_paths": manifest.get("root_paths", []),
        "files_indexed": manifest.get("files_indexed", 0),
    }


def delete_index():
    index_dir = _get_index_dir()
    for fname in [INDEX_FILE, META_FILE, MANIFEST_FILE]:
        fpath = os.path.join(index_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
    logger.info("[FileIndex] Index deleted.")
