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
    try:
        if not path.is_file():
            return {}, False
        raw = path.read_bytes()
    except OSError:
        return None, False
    try:
        # Decoded as json decodes bytes: UTF-8 with or without the byte-order
        # mark some Windows editors add, or UTF-16/32 as Windows PowerShell 5.1
        # and Notepad's "Unicode" write it (strict, so other encodings fail).
        data = json.loads(raw.decode(json.detect_encoding(raw)))
    except (ValueError, RecursionError):  # UnicodeDecodeError is a ValueError
        return None, True
    if not isinstance(data, dict):
        return None, True
    return data, False


def set_aside(path: Path) -> Path | None:
    """Rename a damaged file so the next save cannot overwrite it.

    Returns the new path, or None if the file could not be renamed.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for n in range(1, 1000):
        tail = "" if n == 1 else f"-{n}"
        backup = path.with_name(f"{path.stem}.unreadable-{stamp}{tail}{path.suffix}")
        # Claim the name first: os.replace would silently overwrite a backup
        # another WebUI process made in the same second.
        try:
            os.close(os.open(backup, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        except OSError:
            return None
        try:
            os.replace(path, backup)
        except OSError:
            try:
                os.unlink(backup)
            except OSError:
                pass
            return None
        print(
            f"[ADetailer] {path.name} could not be read; kept it as {backup.name}.",
            file=sys.stderr,
        )
        return backup
    return None
