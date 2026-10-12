"""Answer "what files are in this folder?" from the folder itself.

The language model only ever sees passages from inside documents, so it cannot
list a folder. The indexer already knows every document and its status; this
renders that, plus subfolders and files DocLamar can't read, as a Markdown tree.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

from .indexer import SKIP_DIRS
from .parsing import SUPPORTED_EXTENSIONS
from .store import IndexStore, Scope, path_key

MAX_LISTED = 80        # documents shown before "...and N more"
MAX_WALK = 20_000      # stop walking enormous trees


def _md(name: str) -> str:
    return re.sub(r"([\\`*_\[\]<>#|])", r"\\\1", name)


def _describe(row) -> str:
    if row is None:
        return "not indexed yet"
    status = row["status"]
    if status == "indexed":
        parts = []
        if row["num_pages"]:
            parts.append(f"{row['num_pages']} page{'s' if row['num_pages'] != 1 else ''}")
        if row["ocr_pages"]:
            parts.append("read with OCR" if row["ocr_pages"] == row["num_pages"] else "partly read with OCR")
        return ", ".join(parts)
    return {
        "needs_ocr": "scanned, waiting for OCR",
        "no_text": "no readable text",
        "error": "couldn't be read",
        "too_large": "too large, skipped",
    }.get(status, status)


def folder_inventory(folder: str, store: IndexStore) -> str:
    root = os.path.abspath(folder)
    known = store.files_in(Scope(folder=root))
    tree: Dict = {"dirs": {}, "docs": []}
    others: Counter = Counter()
    seen = 0

    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d.lower() not in SKIP_DIRS)
        rel = Path(dirpath).relative_to(root).parts
        node = tree
        for part in rel:
            node = node["dirs"].setdefault(part, {"dirs": {}, "docs": []})
        for d in dirnames:
            node["dirs"].setdefault(d, {"dirs": {}, "docs": []})
        for name in filenames:
            if name.startswith((".", "~$")):
                continue
            ext = Path(name).suffix.lstrip(".").lower()
            if ext in SUPPORTED_EXTENSIONS:
                node["docs"].append((name, os.path.join(dirpath, name)))
            else:
                others[f".{ext}" if ext else "no extension"] += 1
        seen += len(filenames) + len(dirnames)
        if seen > MAX_WALK:
            break

    def count(node) -> int:
        return len(node["docs"]) + sum(count(child) for child in node["dirs"].values())

    def count_dirs(node) -> int:
        return len(node["dirs"]) + sum(count_dirs(child) for child in node["dirs"].values())

    lines = []
    listed = 0

    def render(node, depth: int) -> None:
        nonlocal listed
        indent = "  " * depth
        for name in sorted(node["dirs"], key=str.lower):
            child = node["dirs"][name]
            n = count(child)
            lines.append(f"{indent}- 📁 **{_md(name)}/** ({n} document{'s' if n != 1 else ''})")
            render(child, depth + 1)
        for name, path in sorted(node["docs"], key=lambda d: d[0].lower()):
            if listed >= MAX_LISTED:
                return
            listed += 1
            detail = _describe(known.get(path_key(path)))
            lines.append(f"{indent}- {_md(name)}" + (f" — {detail}" if detail else ""))

    total, folders = count(tree), count_dirs(tree)
    title = _md(os.path.basename(root.rstrip("\\/")) or root)
    if total == 0 and not folders:
        return f"**{title}** has no files DocLamar can read (PDF, Word, text, Markdown or images)."

    head = f"**{title}** contains {total} document{'s' if total != 1 else ''}"
    if folders:
        head += f" across {folders + 1} folders"
    out = [head + ":", ""]
    render(tree, 0)
    out += lines
    if total > listed:
        out.append(f"- …and {total - listed} more")
    if others:
        kinds = ", ".join(f"{n} {ext}" for ext, n in others.most_common(6))
        n_other = sum(others.values())
        out += ["", f"Also {n_other} other file{'s' if n_other != 1 else ''} I can't search ({kinds})."]
    return "\n".join(out)


def file_summary(path: str, store: IndexStore) -> Optional[str]:
    row = store.get_file(path)
    if row is None:
        return None
    detail = _describe(row)
    where = os.path.dirname(row["path"])
    return (f"This chat is about one file: **{_md(row['name'])}**" + (f" ({detail})" if detail else "")
            + f", in `{where}`. Use **← Back to folder search** to ask about the whole folder.")
