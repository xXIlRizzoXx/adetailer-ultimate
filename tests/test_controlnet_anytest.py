"""Anytest ControlNet models (any-test, any_test, AnyTest) are listed and start
with no preprocessor, on Forge and on AUTOMATIC1111; other models as before."""

from __future__ import annotations

import ast
import importlib.util
import sys
from itertools import chain
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.test_controlnet_forge import _Gradio4Unit
from tests.test_controlnet_forge import _load_backend as _load_forge
from tests.test_controlnet_standard import _module_sent

_ROOT = Path(__file__).resolve().parents[1]

ANYTEST_NAMES = [
    "anima-lllite-any-test-like-v2 [1a2b3c4d]",
    "anima-lllite-any-test-like-v2.safetensors",
    "CN-anytest_v4 [abcd]",
    "sdxl_any_test_v3",
    "AnyTest-XL",
]


def _common():
    path = _ROOT / "controlnet_ext" / "common.py"
    spec = importlib.util.spec_from_file_location("_adetailer_cn_common_any", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Forge Neo's lists: "None" first, then by name.
_FORGE_LISTS = {
    "Inpaint": ["None", "inpaint_global_harmonious", "inpaint_only"],
    "OpenPose": ["None", "dw_openpose_full", "openpose_full"],
    "Depth": ["None", "depth_anything", "depth_midas"],
    "Lineart": ["None", "lineart_anime", "lineart_coarse"],
}


def _ui(monkeypatch, controlnet_type):
    """ui.py's module dropdown lists and on_cn_model_update for one WebUI."""
    global_state = ModuleType("lib_controlnet.global_state")
    global_state.get_filtered_preprocessors = lambda tag: {
        n: None for n in _FORGE_LISTS.get(tag, ["None"])
    }
    lib = ModuleType("lib_controlnet")
    lib.global_state = global_state
    monkeypatch.setitem(sys.modules, "lib_controlnet", lib)
    monkeypatch.setitem(sys.modules, "lib_controlnet.global_state", global_state)

    path = _ROOT / "aaaaaa" / "ui.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    def wanted(node):
        if isinstance(node, ast.FunctionDef):
            return node.name == "on_cn_model_update"
        if isinstance(node, ast.If):
            return "controlnet_type" in ast.unparse(node.test)
        if isinstance(node, ast.Assign):
            text = ast.unparse(node.targets[0])
            return text == "union" or text.startswith("cn_module_choices[")
        return False

    body = [n for n in tree.body if wanted(n)]
    common = _common()
    namespace = {
        "controlnet_type": controlnet_type,
        "chain": chain,
        "gr": SimpleNamespace(update=lambda **kwargs: kwargs),
        "default_cn_module": common.default_cn_module,
        "anytest_choices": common.anytest_choices,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("name", ANYTEST_NAMES)
def test_anytest_models_are_listed(name):
    assert _common().cn_model_regex.search(name)


@pytest.mark.parametrize("name", ["control_v11p_sd15_canny [d14c016b]", "diffusers_xl_canny_mid"])
def test_canny_models_stay_out(name):
    assert not _common().cn_model_regex.search(name)


@pytest.mark.parametrize(
    ("controlnet_type", "none"), [("forge", "None"), ("standard", "none")]
)
@pytest.mark.parametrize("name", ANYTEST_NAMES)
def test_an_anytest_model_starts_with_no_preprocessor(
    monkeypatch, controlnet_type, none, name
):
    ui = _ui(monkeypatch, controlnet_type)
    update = ui["on_cn_model_update"](name, "None")
    assert update["visible"] is True
    assert update["value"] == none
    # Only "no preprocessor": the crop as it is.
    assert update["choices"] == [none]


@pytest.mark.parametrize("controlnet_type", ["forge", "standard"])
def test_the_other_models_keep_their_list_and_default(monkeypatch, controlnet_type):
    ui = _ui(monkeypatch, controlnet_type)
    choices = ui["cn_module_choices"]
    # The union list does not take the Anytest entry in.
    assert choices["union"] == list(
        chain.from_iterable(v for k, v in choices.items() if k not in ("union", "anytest"))
    )
    forge = controlnet_type == "forge"
    for model, expected, listed in [
        ("control_v11f1p_sd15_depth [cfd03158]", "depth_midas", "depth"),
        ("control_v11p_sd15_openpose [cab727d4]", "openpose_full", "openpose"),
        # A name with a type word too is of that type.
        ("anytest_depth_v1", "depth_midas", "depth"),
        ("any-test-openpose", "openpose_full", "openpose"),
    ]:
        update = ui["on_cn_model_update"](model, "None")
        assert update["choices"] == choices[listed]
        assert update["value"] == expected, (model, forge)


@pytest.mark.parametrize(
    ("controlnet_type", "none"), [("forge", "None"), ("standard", "none")]
)
@pytest.mark.parametrize(
    "previous", ["inpaint_global_harmonious", "lineart_anime", "depth_midas"]
)
def test_an_anytest_model_never_keeps_another_preprocessor(
    monkeypatch, controlnet_type, none, previous
):
    # From an inpaint model to an Anytest one, or a preset / pasted
    # parameters with an Anytest model and another preprocessor: the module
    # of the model before was kept, so the crop went through inpaint.
    ui = _ui(monkeypatch, controlnet_type)
    model = "anima-lllite-any-test-like-v2 [1a2b3c4d]"
    for keep_none in (False, True):
        update = ui["on_cn_model_update"](model, previous, keep_none=keep_none)
        assert (update["visible"], update["value"]) == (True, none)
    # Its own module stays.
    assert ui["on_cn_model_update"](model, none, keep_none=True)["value"] == none


@pytest.mark.parametrize(
    ("model", "module", "expected"),
    [
        ("anima-lllite-any-test-like-v2", None, "None"),
        ("AnyTest_v4", None, "None"),
        ("anytest_depth_v1", None, "depth_midas"),
        ("control_v11p_sd15_openpose", None, "openpose_full"),
        ("anima-lllite-any-test-like-v2", "lineart_anime", "lineart_anime"),
    ],
)
def test_a_forge_request_without_a_preprocessor(monkeypatch, model, module, expected):
    backend = _load_forge(monkeypatch, _Gradio4Unit)
    assert backend.get_cn_models  # loaded
    if module is None:
        assert backend.default_module(model) == expected
    ext = backend.ControlNetExt()
    ext.init_controlnet()
    p = SimpleNamespace(init_images=[], width=512, height=512, resize_mode=0)
    ext.update_scripts_args(p, model, module, 1.0, 0.0, 1.0)
    (unit,) = p.script_args_value
    assert unit.module == expected


@pytest.mark.parametrize(
    ("model", "module", "expected"),
    [
        ("anima-lllite-any-test-like-v2 [1a2b]", None, "none"),
        ("CN-AnyTest_v4 [abcd]", "None", "none"),
        ("sdxl_any_test_v3 [abcd]", "None", "none"),
        ("anytest_depth_v1 [abcd]", "None", "depth_midas"),
        # A union model that also holds "anytest" stays a union model, as in
        # the UI: no module, as for any union model.
        ("anytest_union_v1 [abcd]", "None", None),
        ("control_v11f1p_sd15_depth [cfd03158]", "None", "depth_midas"),
        ("control_v11p_sd15_openpose [cab727d4]", None, "openpose_full"),
        ("anima-lllite-any-test-like-v2 [1a2b]", "lineart_anime", "lineart_anime"),
    ],
)
def test_an_automatic1111_anytest_model_runs_without_a_preprocessor(
    monkeypatch, model, module, expected
):
    assert _module_sent(monkeypatch, model, module) == expected
