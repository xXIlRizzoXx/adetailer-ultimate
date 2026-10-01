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


def _pause_total_tqdm(data: dict, defaults: dict | None = None):
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
    opts = SimpleNamespace(data=data)
    if defaults is not None:
        opts.get_default = defaults.get
    namespace = {
        "contextmanager": contextmanager,
        "patch": patch,
        "opts": opts,
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


_DEFAULTS = {
    "CLIP_stop_at_last_layers": 1,
    "sd_vae": "Automatic",
    "sd_model_checkpoint": "base.safetensors",
    "forge_additional_modules": [],
}
# AUTOMATIC1111 does not register Forge's module list: no default for it.
_A1111_DEFAULTS = {k: v for k, v in _DEFAULTS.items() if k != "forge_additional_modules"}


class _Host:
    """AUTOMATIC1111's process_images: an override is put back afterwards only
    if the option was already in opts.data, and applying or putting back
    sd_vae reloads the VAE."""

    def __init__(self, data):
        self.data = data
        self.loaded_vae = "base.vae"

    def setting(self, key):
        return self.data.get(key, _DEFAULTS[key])

    def reload_vae(self):
        vae = self.setting("sd_vae")
        self.loaded_vae = "base.vae" if vae == "Automatic" else vae

    def process(self, override):
        stored = {k: self.data[k] for k in override if k in self.data}
        for key, value in override.items():
            self.data[key] = value
            if key == "sd_vae":
                self.reload_vae()
        seen = (self.setting("CLIP_stop_at_last_layers"), self.loaded_vae)
        for key, value in stored.items():
            self.data[key] = value
            if key == "sd_vae":
                self.reload_vae()
        return seen


def test_pause_total_tqdm_lets_the_host_put_tab_overrides_back():
    # With Clip skip and the VAE never saved in config.json (the case until
    # Settings > Apply is clicked), the host kept a tab's own values after its
    # pass: the next tabs ran with them, and the detailer VAE stayed loaded
    # for the next images and later generations.
    data = {"multiple_tqdm": True}
    host = _Host(data)
    with _pause_total_tqdm(data, _A1111_DEFAULTS)():
        # The checkpoint is not filled in, nor Forge's module list where the
        # option does not exist: whether it is in opts.data tells ADetailer
        # which WebUI it runs on.
        assert "sd_model_checkpoint" not in data
        assert "forge_additional_modules" not in data
        first = host.process({"CLIP_stop_at_last_layers": 2, "sd_vae": "detail.vae"})
        second = host.process({})
        third = host.process({"CLIP_stop_at_last_layers": 3})
    assert first == (2, "detail.vae")
    assert second == (1, "base.vae")
    assert third == (3, "base.vae")
    assert host.loaded_vae == "base.vae"
    assert data == {"multiple_tqdm": True}


def test_pause_total_tqdm_keeps_a_first_change_of_a_missing_option():
    # A missing option changed for the first time from the UI during the pass
    # (Clip skip in the quicksettings) was removed again when the pass ended.
    data = {"multiple_tqdm": True}

    def change_settings():
        data["CLIP_stop_at_last_layers"] = 3

    with _pause_total_tqdm(data, _DEFAULTS)():
        thread = threading.Thread(target=change_settings)
        thread.start()
        thread.join()
    assert data == {"multiple_tqdm": True, "CLIP_stop_at_last_layers": 3}


class _ForgeHost:
    """Forge Neo's process_images: an override is put back afterwards only if
    the option was already in opts.data, and a new module list sets the
    modules the model is loaded with until the list changes again."""

    def __init__(self, data):
        self.data = data
        self.loaded = []

    def _set(self, key, value):
        self.data[key] = value
        if key == "forge_additional_modules":
            self.loaded = list(value)

    def process(self, override):
        stored = {k: self.data[k] for k in override if k in self.data}
        for key, value in override.items():
            self._set(key, value)
        seen = list(self.loaded)
        for key, value in stored.items():
            self._set(key, value)
        return seen


def test_pause_total_tqdm_lets_forge_put_the_module_list_back():
    # Forge writes its VAE / text-encoder list to opts.data only once a
    # module is picked at the top of the page. Until then the host kept a
    # tab's separate text encoder or VAE loaded after its pass, for the next
    # tabs and later generations, and ADetailer, not seeing the list, sent a
    # separate VAE as sd_vae, which Forge Neo cannot load.
    data = {"multiple_tqdm": True}
    host = _ForgeHost(data)
    detail = ["/models/text_encoder/detail-te.safetensors"]
    with _pause_total_tqdm(data, _DEFAULTS)():
        assert data.get("forge_additional_modules") == []
        # A copy: the option's own default list is never handed out.
        assert data["forge_additional_modules"] is not _DEFAULTS["forge_additional_modules"]
        first = host.process({"forge_additional_modules": detail})
        second = host.process({})
    assert first == detail
    assert second == []
    assert host.loaded == []
    assert data == {"multiple_tqdm": True}
    assert _DEFAULTS["forge_additional_modules"] == []

    # Unchanged: a list saved before is left to the host, and one changed
    # from the UI during the pass is kept.
    data = {"multiple_tqdm": True, "forge_additional_modules": ["/models/VAE/base.vae"]}
    with _pause_total_tqdm(data, _DEFAULTS)():
        pass
    assert data == {"multiple_tqdm": True, "forge_additional_modules": ["/models/VAE/base.vae"]}
    data = {"multiple_tqdm": True}
    with _pause_total_tqdm(data, _DEFAULTS)():
        data["forge_additional_modules"] = ["/models/VAE/picked.vae"]
    assert data == {"multiple_tqdm": True, "forge_additional_modules": ["/models/VAE/picked.vae"]}
