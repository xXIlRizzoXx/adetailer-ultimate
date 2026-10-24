from __future__ import annotations

import json
import re
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
# "iris" is opt-in: detected only when picked by name (see mediapipe.py).
MEDIAPIPE_FACE_FEATURES_MODEL = "mediapipe_face_features"
FACE_FEATURE_CLASSES = ["eyes", "mouth", "nose", "eyebrows", "face", "iris"]


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


def _host_refuses_unpickle() -> bool:
    """True when the WebUI would refuse to read the detector file at this
    point; the class names then come only from a sidecar JSON. False
    otherwise, and outside a WebUI."""
    try:
        from modules import shared

        if getattr(shared.cmd_opts, "disable_safe_unpickle", True) is not False:
            return False
        import torch
        from modules import safe

        return torch.load is getattr(safe, "load", None)
    except Exception:  # noqa: BLE001
        return False


# Class names read successfully, per model path, kept for the session.
_RESOLVED_NAMES: dict[str, list[str]] = {}


def get_model_class_names(model_path: str) -> list[str]:
    """Resolve class names for a YOLO model (see _read_class_names).

    Only names that were found are cached. Where the WebUI does not let the UI
    read the detector file, the UI's lookup of a .pt without a sidecar reads
    only the checkpoint's metadata and can come back empty; detection's own
    lookup must still read the real names instead of reusing that empty result
    (which would silently ignore a class filter). Names found once are then
    reused everywhere.
    """
    names = _RESOLVED_NAMES.get(model_path)
    if names:
        return names
    # Where the WebUI would refuse the .pt here (and print an error report for
    # it), only a sidecar or the checkpoint's metadata is read.
    names = _read_class_names(model_path, load_pt=not _host_refuses_unpickle())
    if names:
        _RESOLVED_NAMES[model_path] = names
    return names


def _clear_class_name_cache() -> None:
    _RESOLVED_NAMES.clear()


# Keeps the reset hook of the lru_cache this replaced.
get_model_class_names.cache_clear = _clear_class_name_cache  # type: ignore[attr-defined]


def _read_class_names(model_path: str, load_pt: bool = True) -> list[str]:
    """Resolve class names for a YOLO model.

    Resolution order:
      1. A sidecar JSON next to the .pt, only if it parses into a recognized
         class-names format. Two names are tried, in order:
           a. <model>.names.json  — a DEDICATED file that never collides with
              civitai_helper / Stability Matrix metadata (which claims the plain
              <model>.json). It is the first file tried.
           b. <model>.json        — legacy/plain sidecar; unrelated JSONs (e.g.
              civitai_helper metadata) don't match the format and are ignored.
      2. model.names from a transient YOLO() load; where the WebUI does not
         let the panel load the .pt (load_pt=False), the names stored in the
         checkpoint, read without loading it (_names_from_checkpoint).
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
            # Bytes let json detect UTF-8 (with or without a byte-order mark)
            # and UTF-16/32, as Windows editors and PowerShell write them. An
            # undecodable file falls through like a malformed one.
            data = json.loads(sidecar.read_bytes())
        except (ValueError, OSError):
            data = None
        if data is not None:
            names = _names_from_json(data)
            if names:
                return names

    if not load_pt:
        # The WebUI would refuse (and report) a torch.load here: read the
        # names from the checkpoint's metadata without loading it.
        return _names_from_checkpoint(p)

    try:
        from ultralytics import YOLO

        names = YOLO(str(p)).names
        if isinstance(names, dict):
            return [str(names[i]) for i in sorted(names)]
        return [str(n) for n in (names or [])]
    except Exception:
        return []


# Largest pickle stream read from a checkpoint for its class names. The data.pkl
# of a YOLO detector is well under 1 MB (its weights are separate entries).
_MAX_CHECKPOINT_PICKLE = 64 * 1024 * 1024
_DATA_PKL = re.compile(r"^[^/]+/data\.pkl$")


class _Inert:
    """Stands in for every class or function named in a checkpoint's pickle.

    Nothing is imported and nothing named in the file is ever called: the
    object only records what the pickle hands it, so its state (where a
    detector keeps `names`) can be read."""

    __slots__ = ("args", "state", "items")

    # __new__, not __init__: the pickle may create the object without calling
    # it (NEWOBJ), as it does for a model.
    def __new__(cls, *args: Any, **kwargs: Any) -> "_Inert":
        obj = super().__new__(cls)
        obj.args = args
        obj.state = None
        obj.items = []
        return obj

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        self.state = state

    def __setitem__(self, key: Any, value: Any) -> None:
        self.items.append((key, value))

    def append(self, value: Any) -> None:
        self.items.append(value)

    def extend(self, values: Any) -> None:
        self.items.extend(values)

    def add(self, value: Any) -> None:
        self.items.append(value)


def _names_from_checkpoint(p: Path) -> list[str]:
    """Class names of a YOLO .pt read from its pickle without loading it.

    torch.load runs whatever the pickle names (which is why AUTOMATIC1111
    checks the file first and refuses a detector). Here every class or
    function the pickle names is replaced by an inert placeholder and no
    tensor data is read, so nothing in the file is imported or run. The names
    are taken where Ultralytics takes them (the "ema" model, else "model") and
    normalized as Ultralytics does. [] for anything else, e.g. a file in the
    old non-zip format."""
    import pickle
    import zipfile

    class _Reader(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            return _Inert

        def persistent_load(self, pid: Any) -> Any:
            return None

    try:
        with zipfile.ZipFile(p) as z:
            entries = [i for i in z.infolist() if _DATA_PKL.match(i.filename)]
            if len(entries) != 1 or entries[0].file_size > _MAX_CHECKPOINT_PICKLE:
                return []
            with z.open(entries[0]) as f:
                ckpt = _Reader(f).load()
        if not isinstance(ckpt, dict):
            return []
        model = ckpt.get("ema") or ckpt.get("model")
        state = getattr(model, "state", None) if isinstance(model, _Inert) else None
        if isinstance(state, tuple) and state and isinstance(state[0], dict):
            state = state[0]  # (dict state, slot state)
        names = state.get("names") if isinstance(state, dict) else None
        if isinstance(names, (list, tuple)):
            names = dict(enumerate(names))
        if not isinstance(names, dict) or not names:
            return []
        if not all(isinstance(v, (str, int, float)) for v in names.values()):
            return []
        by_id = {int(k): str(v) for k, v in names.items()}
        # Ultralytics refuses ids outside 0..n-1, and maps ImageNet codes to
        # other names: leave both to detection.
        if sorted(by_id) != list(range(len(by_id))) or by_id[0].startswith("n0"):
            return []
        return [by_id[i] for i in range(len(by_id))]
    except Exception:  # noqa: BLE001 - unreadable or unexpected: no names
        return []


def resolve_class_ids(model_path: str, requested: list[str]) -> list[int]:
    """Convert user-provided class names (or numeric ids as strings) to int ids.
    Without an exact match a name matches case-insensitively, as an API call or
    an old preset may send "Face" for "face". Unknown entries are dropped —
    matches uddetailer's behavior — and named in the console when the model's
    class names are known.
    """
    names = get_model_class_names(model_path)
    folded = [n.casefold() for n in names]
    out: list[int] = []
    unknown: list[str] = []
    for token in requested:
        # Not isdigit(): it is also true for "²" or "①", which int() rejects.
        if token.isdecimal():
            try:
                i = int(token)
            except ValueError:  # not a usable number
                i = -1
            if 0 <= i < max(1, len(names) or 10_000):
                out.append(i)
            else:
                unknown.append(token)
            continue
        if token in names:
            out.append(names.index(token))
        elif folded.count(token.casefold()) == 1:
            out.append(folded.index(token.casefold()))
        else:
            unknown.append(token)
    if unknown and names:
        msg = (
            f"[-] ADetailer: class not found in {Path(model_path).name}, ignored:"
            f" {', '.join(unknown)}"
        )
        # ASCII only (other characters escaped), so a console pipe in any
        # legacy code page can print it instead of raising and stopping
        # ADetailer for the image.
        print(msg.encode("ascii", "backslashreplace").decode("ascii"))
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
