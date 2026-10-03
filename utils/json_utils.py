from __future__ import annotations
import json
import re
from typing import Any, Dict


def extract_json(text: str) -> Dict[str, Any]:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    candidates = _balanced_braces(text)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    raise ValueError(f"No valid JSON found in output:\n{text[:400]}")


def _balanced_braces(text: str) -> list[str]:
    results = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                results.append(text[start: i + 1])
                start = -1
    return results
