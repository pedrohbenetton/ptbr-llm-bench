from __future__ import annotations

import json
import re
import unicodedata
from typing import Any


def normalize(text: str) -> str:
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip().casefold()


def label_scorer(pred: str, gold: str) -> float:
    lines = pred.strip().splitlines()
    first = lines[0] if lines else ""
    first = re.sub(r"[^\w\s]", "", first)
    return float(normalize(first) == normalize(gold))


def extract_json(text: str) -> dict | None:
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                    except ValueError:
                        break
                    return obj if isinstance(obj, dict) else None
        start = text.find("{", start + 1)
    return None


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return normalize(a) == normalize(b)
    return a == b


def json_fields_scorer(pred: str, gold: dict) -> float:
    obj = extract_json(pred)
    if obj is None or not gold:
        return 0.0
    hits = sum(1 for k, v in gold.items() if k in obj and _same(obj[k], v))
    return hits / len(gold)
