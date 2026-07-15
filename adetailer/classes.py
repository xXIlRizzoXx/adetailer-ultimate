from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


def is_world_model(model_path: str | Path) -> bool:
    return "-world" in Path(model_path).stem


def parse_csv(csv: str) -> list[str]:
    return [c.strip() for c in (csv or "").split(",") if c.strip()]


# The multi-class MediaPipe "face features" pseudo-detector. Its class list is
# the single source of truth shared by: the CLASSES dropdown, the Auto
# class-guard (build_class_guard), per-class prompts, and the detector itself
# (adetailer/mediapipe.py imports these). Lives here in the stdlib-only leaf
# module so mediapipe.py / args.py / ui.py can import it without a cycle.
# Order = dropdown order = sequential-pass order; keep STABLE (preset/PNG-info).
MEDIAPIPE_FACE_FEATURES_MODEL = "mediapipe_face_features"
FACE_FEATURE_CLASSES = ["eyes", "mouth", "nose", "eyebrows", "face"]


def _names_from_json(data: Any) -> list[str]:
    """Try to extract a class-names list from a parsed JSON blob.

    Returns [] when the blob is *not* a recognized class-names format.
    The caller is expected to fall through to another resolution path on [].
    Recognized formats:
      - ["face", "hand", ...]                          (list of names)
      - {"names": ["face", "hand", ...]}               (Ultralytics-export-style)
      - {"names": {"0": "face", "1": "hand", ...}}     (Ultralytics-dict-style)
      - {"0": "face", "1": "hand", ...}                (bare integer-keyed map)
    Anything else (e.g. civitai_helper sidecar JSONs) returns [].
    """
    if isinstance(data, list):
        return [str(x) for x in data if isinstance(x, (str, int, float))]

    if not isinstance(data, dict):
        return []

    if "names" in data:
        inner = data["names"]
        if isinstance(inner, list):
            return [str(x) for x in inner if isinstance(x, (str, int, float))]
        if isinstance(inner, dict):
            try:
                keys = sorted(int(k) for k in inner)
            except (TypeError, ValueError):
                return []
            return [
                str(inner[str(i)])
                for i in keys
                if str(i) in inner and isinstance(inner[str(i)], (str, int, float))
            ]

    # Bare {"0": "face", "1": "hand"}. Require ALL top-level keys to be ints
    # AND all values to be scalars — otherwise treat as unrelated metadata.
    try:
        int_keys = [int(k) for k in data]
    except (TypeError, ValueError):
        return []
    if not int_keys or len(int_keys) != len(data):
        return []
    keys = sorted(int_keys)
    result: list[str] = []
    for i in keys:
        v = data.get(str(i))
        if not isinstance(v, (str, int, float)):
            return []
        result.append(str(v))
    return result


@lru_cache(maxsize=32)
def get_model_class_names(model_path: str) -> list[str]:
    """Resolve class names for a YOLO model.

    Resolution order:
      1. A sidecar JSON next to the .pt, only if it parses into a recognized
         class-names format. Two names are tried, in order:
           a. <model>.names.json  — a DEDICATED file that never collides with
              civitai_helper / Stability Matrix metadata (which claims the plain
              <model>.json). This is the escape hatch for models whose .pt a
              WebUI's safe-unpickle refuses to read (e.g. non-standard
              segmentation models -> otherwise-empty class dropdown).
           b. <model>.json        — legacy/plain sidecar; unrelated JSONs (e.g.
              civitai_helper metadata) don't match the format and are ignored.
      2. model.names from a transient YOLO() load.
      3. [] if unknown (YOLO-World, MediaPipe, missing file, or load failure).

    Special case: the "mediapipe_face_features" pseudo-detector has a fixed,
    code-defined class list (facial parts). It is matched FIRST, before the
    .pt-path checks below (a bare mediapipe name has no .pt on disk), so the
    CLASSES dropdown, Auto class-guard and per-class prompts all light up.
    """
    if (
        model_path == MEDIAPIPE_FACE_FEATURES_MODEL
        or Path(model_path).name == MEDIAPIPE_FACE_FEATURES_MODEL
    ):
        return list(FACE_FEATURE_CLASSES)

    p = Path(model_path)
    if is_world_model(p) or not p.exists() or p.suffix != ".pt":
        return []

    for sidecar in (p.with_name(p.stem + ".names.json"), p.with_suffix(".json")):
        if not sidecar.is_file():
            continue
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = None
        if data is not None:
            names = _names_from_json(data)
            if names:
                return names

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


def _strip_weight(token: str) -> str:
    """Reduce an attention-weighted token like '(face:1.20)' to 'face'."""
    t = token.strip()
    m = re.fullmatch(r"\(\s*(.*?)\s*:\s*[0-9.]+\s*\)", t)
    if m:
        return m.group(1).strip()
    if len(t) >= 2 and t[0] == "(" and t[-1] == ")":
        return t[1:-1].strip()
    return t


def _has_token(prompt: str, tok: str) -> bool:
    """True if ``tok`` appears as a whole comma-separated token in ``prompt``,
    case-insensitive and ignoring any surrounding attention-weight wrapper."""
    if not prompt or not tok:
        return False
    target = tok.strip().casefold()
    return any(_strip_weight(part).casefold() == target for part in prompt.split(","))


def _sanitize_token(name: str) -> str:
    """Strip characters that would break prompt / attention-weight syntax
    ('(', ')', ':', ',') so a class label can never inject a malformed token."""
    return (
        name.replace("(", "")
        .replace(")", "")
        .replace(":", "")
        .replace(",", "")
        .strip()
    )


def build_class_guard(
    detected: str, full_list: list[str], weight: float
) -> tuple[str, str]:
    """Build the (positive_token, negative_tokens) pair for the Auto class-guard.

    positive: the detected class name — bare at weight 1.0, else '(name:weight)'.
    negative: every OTHER class of the model, comma-joined ('' for a
              single-class model).
    All tokens are sanitized of prompt-syntax characters. Returns ('', '') when
    ``detected`` is empty or is not one of ``full_list`` (e.g. a numeric-id
    fallback), so the caller degrades to a no-op.
    """
    if not detected or not full_list:
        return "", ""
    if detected.casefold() not in {c.casefold() for c in full_list}:
        return "", ""
    try:
        w = float(weight)
    except (TypeError, ValueError):
        w = 1.0
    safe = _sanitize_token(detected)
    if not safe:
        return "", ""
    if abs(w - 1.0) < 1e-3:
        pos = safe
    else:
        w = max(0.5, min(2.0, w))
        pos = f"({safe}:{w:.2f})"
    others = [
        s
        for c in full_list
        if c.casefold() != detected.casefold()
        for s in (_sanitize_token(c),)
        if s
    ]
    return pos, ", ".join(others)
