from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def find_by_id(rows: List[Dict[str, Any]], review_id: int) -> Optional[Dict[str, Any]]:
    for r in rows:
        try:
            if int(r.get("id")) == int(review_id):
                return r
        except Exception:
            continue
    return None
