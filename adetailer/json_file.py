"""Safe reading of the extension's small JSON data files.

`user_presets.json` and `user_state.json` are rewritten as a whole on every
save. If a file exists but cannot be read, treating it as empty would let the
next save erase everything in it, so writers set such a file aside first.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def read_json_object(path: Path) -> tuple[dict[str, Any] | None, bool]:
    """Return ``(data, damaged)`` for a file holding one JSON object.

    A missing file is an empty object. A file that cannot be decoded, or that
    holds anything other than an object, is damaged: ``(None, True)``. A file
    that exists but cannot be read right now gives ``(None, False)``.
    """
    if not path.is_file():
        return {}, False
    try:
        # utf-8-sig also accepts the byte-order mark some Windows editors add.
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError):
        return None, True
    except OSError:
        return None, False
    if not isinstance(data, dict):
        return None, True
    return data, False


def set_aside(path: Path) -> Path | None:
    """Rename a damaged file so the next save cannot overwrite it.

    Returns the new path, or None if the file could not be renamed.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.stem}.unreadable-{stamp}{path.suffix}")
    n = 1
    while backup.exists():
        n += 1
        backup = path.with_name(f"{path.stem}.unreadable-{stamp}-{n}{path.suffix}")
    try:
        os.replace(path, backup)
    except OSError:
        return None
    print(
        f"[ADetailer] {path.name} could not be read; kept it as {backup.name}.",
        file=sys.stderr,
    )
    return backup
