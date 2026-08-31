from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.serialization import to_plain_data


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> Path:
    ensure_directory(path.parent)
    path.write_text(json.dumps(to_plain_data(data), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_text(path: Path, text: str) -> Path:
    ensure_directory(path.parent)
    path.write_text(text, encoding="utf-8")
    return path
