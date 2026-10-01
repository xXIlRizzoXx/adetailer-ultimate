"""The standard (sd-webui-controlnet) backend picks the preprocessor a model needs.

controlnet_ext/controlnet_ext.py imports the WebUI's modules, so they are
replaced by small stand-ins here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_CN_DIR = Path(__file__).resolve().parents[1] / "controlnet_ext"
_PACKAGE = "_adetailer_cn_standard_test"


def _load_backend(monkeypatch):
    modules = ModuleType("modules")
    modules.extensions = SimpleNamespace(active=lambda: [])
    modules.sd_models = SimpleNamespace(model_hash=lambda _path: "0000")
    modules.shared = SimpleNamespace(opts=SimpleNamespace(data={}), cmd_opts=SimpleNamespace())
    paths = ModuleType("modules.paths")
    paths.extensions_builtin_dir = paths.extensions_dir = paths.models_path = "."
    package = ModuleType(_PACKAGE)
    package.__path__ = [str(_CN_DIR)]
    for name, module in {
        "modules": modules, "modules.paths": paths, _PACKAGE: package
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    loaded = {}
    for stem in ("common", "controlnet_ext"):
        spec = importlib.util.spec_from_file_location(
            f"{_PACKAGE}.{stem}", _CN_DIR / f"{stem}.py"
        )
        loaded[stem] = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, loaded[stem])
        spec.loader.exec_module(loaded[stem])
    return loaded["controlnet_ext"]


def _module_sent(monkeypatch, model: str, module: str | None) -> str | None:
    backend = _load_backend(monkeypatch)
    ext = backend.ControlNetExt()
    ext.cn_available = True
    sent = []
    ext.external_cn = SimpleNamespace(
        ControlNetUnit=lambda **kwargs: kwargs,
        ControlMode=SimpleNamespace(BALANCED="Balanced"),
        update_cn_script_in_processing=lambda _p, units: sent.extend(units),
    )
    ext.update_scripts_args(SimpleNamespace(), model, module, 1.0, 0.0, 1.0)
    (unit,) = sent
    assert unit["model"] == model  # the name itself is passed on unchanged
    return unit["module"]


@pytest.mark.parametrize(
    ("model", "module", "expected"),
    [
        # Listed by cn_model_regex in any case, so matched in any case too:
        # these got module None, so no preprocessor ran.
        ("OpenPoseXL2 [9a0c1e]", "None", "openpose_full"),
        ("Controlnet_Tile_Realistic_v2_fp16 [abcd]", None, "tile_resample"),
        ("controlnetxlCNXL_sdxlDepth [1234]", "None", "depth_midas"),
        # As before.
        ("control_v11p_sd15_openpose [cab727d4]", "None", "openpose_full"),
        ("control_v11p_sd15_inpaint [ebff9138]", None, "inpaint_global_harmonious"),
        ("OpenPoseXL2 [9a0c1e]", "dw_openpose_full", "dw_openpose_full"),
        ("controlnet_union_sdxl [5678]", "None", None),
    ],
)
def test_standard_backend_picks_the_preprocessor_in_any_case(
    monkeypatch, model, module, expected
):
    assert _module_sent(monkeypatch, model, module) == expected
