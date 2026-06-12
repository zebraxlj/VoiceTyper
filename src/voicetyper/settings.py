"""Tiny JSON-backed settings store at ``~/.voicetyper/settings.json``.

UI-agnostic and cross-platform. Used by the device picker (and reusable by the
formal app) to persist small preferences. Reads never raise — a missing or
corrupt file yields ``{}``.
"""

import json
from pathlib import Path
from typing import Any

SETTINGS_PATH = Path.home() / ".voicetyper" / "settings.json"


def load() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save(data: dict[str, Any]) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
