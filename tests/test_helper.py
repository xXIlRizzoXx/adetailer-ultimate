"""aaaaaa.helper regressions, loaded through AST without torch or a WebUI."""

from __future__ import annotations

import ast
import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_HELPER_PATH = Path(__file__).resolve().parents[1] / "aaaaaa" / "helper.py"


def _pause_total_tqdm(data: dict):
    tree = ast.parse(_HELPER_PATH.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "pause_total_tqdm")
        or (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", "") == "_AD_OVERRIDE_KEYS" for t in node.targets)
        )
    ]
    namespace = {
        "contextmanager": contextmanager,
        "patch": patch,
        "opts": SimpleNamespace(data=data),
    }
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(_HELPER_PATH), "exec"), namespace)
    return namespace["pause_total_tqdm"]


@pytest.mark.parametrize("fails", [False, True])
def test_pause_total_tqdm_keeps_settings_changed_during_the_pass(fails):
    # The whole opts.data was put back after every ADetailer pass, so a
    # setting changed from the UI meanwhile (Clip skip, the checkpoint, any
    # other option) was silently undone.
    data = {
        "multiple_tqdm": True,
        "CLIP_stop_at_last_layers": 1,
        "sd_model_checkpoint": "base.safetensors",
    }
    pause = _pause_total_tqdm(data)

    def change_settings():
        data["CLIP_stop_at_last_layers"] = 2
        data["sd_model_checkpoint"] = "other.safetensors"
        data["samples_format"] = "jpg"

    with pytest.raises(RuntimeError) if fails else nullcontext(), pause():
        assert data["multiple_tqdm"] is False
        thread = threading.Thread(target=change_settings)
        thread.start()
        thread.join()
        if fails:
            msg = "simulated pass failure"
            raise RuntimeError(msg)

    assert data == {
        "multiple_tqdm": True,
        "CLIP_stop_at_last_layers": 2,
        "sd_model_checkpoint": "other.safetensors",
        "samples_format": "jpg",
    }


def test_pause_total_tqdm_removes_a_missing_option_again():
    data = {"CLIP_stop_at_last_layers": 1}
    with _pause_total_tqdm(data)():
        assert data["multiple_tqdm"] is False
    assert data == {"CLIP_stop_at_last_layers": 1}


def test_pause_total_tqdm_drops_override_options_the_host_does_not_put_back():
    # The host restores an override option only if it was already in
    # opts.data; one it added for the detailer pass must not stay behind.
    data = {"multiple_tqdm": True, "sd_model_checkpoint": "base.safetensors"}
    with _pause_total_tqdm(data)():
        data["CLIP_stop_at_last_layers"] = 3
        data["forge_additional_modules"] = ["detail-vae.safetensors"]
    assert data == {"multiple_tqdm": True, "sd_model_checkpoint": "base.safetensors"}
