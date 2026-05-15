from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


def is_world_model(model_path: str | Path) -> bool:
    return "-world" in Path(model_path).stem


def parse_csv(csv: str) -> list[str]:
    return [c.strip() for c in (csv or "").split(",") if c.strip()]


@lru_cache(maxsize=32)
def get_model_class_names(model_path: str) -> list[str]:
    """Resolve class names for a YOLO model.

    Resolution order:
      1. Sidecar JSON file next to the .pt (list, {"names": [...]}, or {"0": "...", ...}).
      2. model.names from a transient YOLO() load.
      3. [] if unknown (YOLO-World, MediaPipe, missing file, or load failure).
    """
    p = Path(model_path)
    if is_world_model(p) or not p.exists() or p.suffix != ".pt":
        return []

    sidecar = p.with_suffix(".json")
    if sidecar.is_file():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(x) for x in data]
            if isinstance(data, dict):
                if "names" in data and isinstance(data["names"], list):
                    return [str(x) for x in data["names"]]
                keys = sorted(int(k) for k in data if str(k).lstrip("-").isdigit())
                return [str(data[str(i)]) for i in keys if str(i) in data]
        except (json.JSONDecodeError, OSError, KeyError, ValueError):
            pass

    try:
        from ultralytics import YOLO

        names = YOLO(str(p)).names
        if isinstance(names, dict):
            return [str(names[i]) for i in sorted(names)]
        return [str(n) for n in (names or [])]
    except Exception:
        return []


def resolve_class_ids(model_path: str, requested: list[str]) -> list[int]:
    """Convert user-provided class names (or numeric ids as strings) to int ids.
    Unknown entries are silently dropped — matches uddetailer's behavior.
    """
    names = get_model_class_names(model_path)
    out: list[int] = []
    for token in requested:
        if token.isdigit():
            i = int(token)
            if 0 <= i < max(1, len(names) or 10_000):
                out.append(i)
            continue
        if token in names:
            out.append(names.index(token))
    return out
