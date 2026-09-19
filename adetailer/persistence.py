"""Per-tab state persistence across WebUI restarts.

The state file is a plain JSON keyed by "<mode>:<tab_index>" strings —
e.g. `"txt2img:0"`, `"img2img:2"`. The `mode` prefix scopes the state by
processing pipeline so that txt2img and img2img can have independent
last-used configurations on the same tab number.

For backwards compatibility a legacy key form (just `"0"`, `"1"`, …
without a mode prefix) is still loadable: those entries are returned for
BOTH modes so users upgrading from the pre-scoped layout don't lose
their last-used values. The next `save_tab_state` call from either mode
writes the new scoped form; the legacy entry stays harmlessly until the
JSON file is overwritten.

Storage location: `<extension_root>/user_state.json` (gitignored). We pick
the extension root by walking up from this file — works regardless of where
the extension is installed (Stability Matrix shared dir, A1111
extensions/, etc.).

Failures are swallowed. The plugin must never crash because we couldn't
read/write the cache file; in the worst case the user just doesn't get
their settings restored.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from typing import Any

from adetailer.json_file import read_json_object, set_aside

# extension_root = parent of the `adetailer/` package this file lives in.
_EXT_ROOT = Path(__file__).resolve().parent.parent
_STATE_FILE = _EXT_ROOT / "user_state.json"
# Generate dispatches one unqueued callback per tab. Protect the entire
# read/modify/replace transaction so those callbacks cannot drop each other.
_STATE_LOCK = RLock()


def _enabled() -> bool:
    """User-facing toggle exposed in Settings > ADetailer.

    Default = True so the feature is opt-OUT, not opt-in. Falls back to
    True when modules.shared isn't importable (standalone preview / test).
    """
    try:
        from modules.shared import opts
    except ImportError:
        return True
    return bool(opts.data.get("ad_remember_last_settings", True))


def _read_raw() -> tuple[dict[str, Any] | None, bool]:
    return read_json_object(_STATE_FILE)


def _load_raw() -> dict[str, Any]:
    data, _damaged = _read_raw()
    return {} if data is None else data


def _state_key(mode: str, tab_index: int) -> str:
    """Build the JSON key for a (mode, tab_index) pair."""
    return f"{mode}:{tab_index}"


def load_state(mode: str = "txt2img") -> dict[str, dict[str, Any]]:
    """Return per-tab saved state for the given mode (`"txt2img"` or
    `"img2img"`). Outer key = str(tab_index).

    Reads two key forms from disk:
      1. New scoped form `"<mode>:<tab_index>"` → loaded when mode matches.
      2. Legacy unscoped form `"0"`, `"1"`, … → loaded for any mode (so
         users upgrading from the pre-scoped layout keep their last-used
         values until the next Generate click writes the scoped form).
    Scoped entries take precedence over legacy ones for the same tab.

    Returns an empty dict (no restore) if the user disabled the feature
    in Settings > ADetailer > Remember last-used settings.
    """
    if not _enabled():
        return {}
    raw = _load_raw()
    out: dict[str, dict[str, Any]] = {}
    prefix = f"{mode}:"
    # First pass: legacy unscoped keys (lower priority).
    for k, v in raw.items():
        if not isinstance(v, dict):
            continue
        key = str(k)
        if ":" in key:
            continue
        out[key] = v
    # Second pass: scoped keys for THIS mode (higher priority, overrides
    # any legacy entry for the same tab number).
    for k, v in raw.items():
        if not isinstance(v, dict):
            continue
        key = str(k)
        if key.startswith(prefix):
            out[key[len(prefix):]] = v
    return out


def save_tab_state(
    mode: str, tab_index: int, state: dict[str, Any]
) -> None:
    """Persist a single tab's state to disk under the scoped key
    `"<mode>:<tab_index>"`.

    No-op if the user disabled the feature in Settings > ADetailer.
    Writes the full file atomically: read existing -> mutate -> tmp file ->
    rename. Drops `is_api` (it's transient, tuple-vs-bool serialization
    causes infotext quirks).
    """
    if not _enabled():
        return
    with _STATE_LOCK:
        try:
            current, damaged = _read_raw()
            if current is None:
                # Unreadable right now: skip this save. Damaged: keep the old
                # file under a new name rather than overwrite every other tab.
                if not damaged or set_aside(_STATE_FILE) is None:
                    return
                current = {}
            cleaned = {k: v for k, v in state.items() if k != "is_api"}
            current[_state_key(mode, tab_index)] = cleaned

            tmp = _STATE_FILE.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                f.write(json.dumps(current, indent=2, default=str))
                # On disk before the rename: after a power cut the file is
                # then the old or the new one, never a file of zero bytes.
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass  # best effort: some file systems cannot sync
            os.replace(tmp, _STATE_FILE)
        except OSError:
            # Disk full / permission denied / network drive flaked / ... — we
            # don't want a save failure to break the user's generation.
            pass
