"""The Forge ControlNet backend builds the unit each Forge variant can read.

controlnet_ext/controlnet_ext_forge.py imports the Forge-only lib_controlnet
and the WebUI's modules, so both are replaced by small stand-ins here.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

_CN_DIR = Path(__file__).resolve().parents[1] / "controlnet_ext"
_PACKAGE = "_adetailer_cn_forge_test"


@dataclass
class _Gradio3Unit:
    """ControlNetUnit of older Forge and reForge's main branch: it reads the
    image and the mask from the {"image", "mask"} dict in `image`."""

    enabled: bool = True
    module: str = "None"
    model: str = "None"
    weight: float = 1.0
    image: object = None
    mask_image: object = None
    processor_res: int = -1
    guidance_start: float = 0.0
    guidance_end: float = 1.0


@dataclass
class _Gradio4Unit(_Gradio3Unit):
    """ControlNetUnit of classic Forge since its Gradio 4 update (and Forge
    Neo): separate image / mask_image arrays, with the image_fg fields."""

    image_fg: object = None
    mask_image_fg: object = None


class _PlainUnit:
    """A ControlNetUnit that is not a dataclass."""

    def __init__(self, **kwargs):
        self.image = self.mask_image = None
        self.__dict__.update(kwargs)


# Forge Neo's lists: "None" first, then by name.
_PREPROCESSORS = {
    "Inpaint": ["None", "inpaint_global_harmonious", "inpaint_only", "inpaint_only+lama"],
    "OpenPose": ["None", "dw_openpose_full", "openpose", "openpose_full"],
    "Scribble": ["None", "scribble_hed", "scribble_pidinet"],
    "Depth": ["None", "depth_anything", "depth_midas"],
}


def _load_backend(monkeypatch, unit_class):
    external_code = ModuleType("lib_controlnet.external_code")
    external_code.ControlNetUnit = unit_class
    external_code.pixel_perfect_resolution = lambda *_args, **_kwargs: 512
    external_code.resize_mode_from_value = lambda value: value
    global_state = ModuleType("lib_controlnet.global_state")
    global_state.get_all_controlnet_names = lambda: []
    global_state.get_filtered_preprocessors = lambda tag: {
        name: None for name in _PREPROCESSORS.get(tag, ["None"])
    }
    lib = ModuleType("lib_controlnet")
    lib.external_code, lib.global_state = external_code, global_state

    class ControlNetScript:
        @staticmethod
        def title():
            return "ControlNet"

    scripts = ModuleType("modules.scripts")
    scripts.scripts_img2img = SimpleNamespace(
        scripts=[ControlNetScript()], alwayson_scripts=[]
    )
    processing = ModuleType("modules.processing")
    processing.StableDiffusionProcessing = object
    modules = ModuleType("modules")
    modules.scripts, modules.processing = scripts, processing
    package = ModuleType(_PACKAGE)
    package.__path__ = [str(_CN_DIR)]
    stand_ins = {
        "lib_controlnet": lib,
        "lib_controlnet.external_code": external_code,
        "lib_controlnet.global_state": global_state,
        "modules": modules,
        "modules.scripts": scripts,
        "modules.processing": processing,
        _PACKAGE: package,
    }
    for name, module in stand_ins.items():
        monkeypatch.setitem(sys.modules, name, module)
    loaded = {}
    for stem in ("common", "controlnet_ext_forge"):
        spec = importlib.util.spec_from_file_location(
            f"{_PACKAGE}.{stem}", _CN_DIR / f"{stem}.py"
        )
        loaded[stem] = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, loaded[stem])
        spec.loader.exec_module(loaded[stem])
    return loaded["controlnet_ext_forge"]


def _unit(monkeypatch, unit_class):
    backend = _load_backend(monkeypatch, unit_class)
    ext = backend.ControlNetExt()
    ext.init_controlnet()
    p = SimpleNamespace(
        init_images=[Image.new("RGB", (64, 48), (200, 120, 40))],
        width=512, height=512, resize_mode=0,
    )
    ext.update_scripts_args(p, "control_v11p_sd15_inpaint", "inpaint_only", 0.8, 0.1, 0.9)
    (script,) = p.scripts.alwayson_scripts
    assert (script.title(), script.args_from, script.args_to) == ("ControlNet", 0, 1)
    (unit,) = p.script_args_value
    assert (unit.model, unit.module, unit.weight) == ("control_v11p_sd15_inpaint", "inpaint_only", 0.8)
    assert (unit.guidance_start, unit.guidance_end, unit.processor_res) == (0.1, 0.9, 512)
    return np.asarray(p.init_images[0]), unit


def test_a_gradio_4_forge_controlnet_gets_the_image_and_mask_as_arrays(monkeypatch):
    # Classic Forge since its Gradio 4 update compares unit.image as an array:
    # the {"image", "mask"} dict raised a TypeError there, so the ADetailer
    # ControlNet model did nothing on every region.
    image, unit = _unit(monkeypatch, _Gradio4Unit)

    assert isinstance(unit.image, np.ndarray)
    assert np.array_equal(unit.image, image)
    assert isinstance(unit.mask_image, np.ndarray)
    assert unit.mask_image.shape == image.shape
    assert (unit.mask_image == 255).all()
    # The check classic Forge runs on it, and its choice of mask.
    assert not (unit.image < 5).all()
    assert (unit.mask_image > 5).any()


@pytest.mark.parametrize("unit_class", [_Gradio3Unit, _PlainUnit])
def test_an_older_forge_controlnet_still_gets_the_image_dict(monkeypatch, unit_class):
    # Older Forge and reForge's main branch read only unit.image["image"] and
    # unit.image["mask"]; an unknown unit keeps that format too.
    image, unit = _unit(monkeypatch, unit_class)

    assert isinstance(unit.image, dict)
    assert set(unit.image) == {"image", "mask"}
    assert np.array_equal(unit.image["image"], image)
    assert (unit.image["mask"] == 255).all()
    assert unit.mask_image is None


@pytest.mark.parametrize(
    ("model", "module", "expected"),
    [
        # An API request without a preprocessor: the module dropdown's pick.
        ("control_v11p_sd15_openpose", None, "openpose_full"),
        ("control_v11f1p_sd15_depth", None, "depth_midas"),
        ("control_v11p_sd15_inpaint", None, "inpaint_global_harmonious"),
        ("control_sd15_inpaint_depth_hand_fp16", None, "depth_midas"),
        # No usual one in the list: the first real preprocessor.
        ("control_v11p_sd15_scribble", None, "scribble_hed"),
        ("controlnet_union_sdxl", None, "inpaint_global_harmonious"),
        # A type this Forge lists nothing for: no preprocessor, as before.
        ("control_v11p_sd15_lineart", None, "None"),
        # Chosen on purpose: kept.
        ("control_v11p_sd15_openpose", "None", "None"),
        ("control_v11p_sd15_openpose", "openpose", "openpose"),
    ],
)
def test_a_forge_unit_without_a_preprocessor_gets_the_models_default(
    monkeypatch, model, module, expected
):
    # Forge reads a missing preprocessor as "None", the picture as it is.
    backend = _load_backend(monkeypatch, _Gradio4Unit)
    ext = backend.ControlNetExt()
    ext.init_controlnet()
    p = SimpleNamespace(
        init_images=[Image.new("RGB", (64, 48))], width=512, height=512, resize_mode=0,
    )

    ext.update_scripts_args(p, model, module, 1.0, 0.0, 1.0)

    (unit,) = p.script_args_value
    assert unit.module == expected


def test_a_forge_without_the_preprocessor_lists_runs_as_before(monkeypatch):
    backend = _load_backend(monkeypatch, _Gradio4Unit)
    del sys.modules["lib_controlnet.global_state"].get_filtered_preprocessors

    assert backend.default_module("control_v11p_sd15_openpose") == "None"


class _Runner:
    """ScriptRunner of Forge and Forge Neo: `process` goes through a cache that
    compares only the number of scripts, `process_before_every_sampling`
    through alwayson_scripts, without the cache."""

    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.alwayson_scripts = list(scripts)
        self.callback_map = {}

    def process(self, p):
        length, callbacks = self.callback_map.get("script_process", (-1, None))
        if callbacks is None or length != len(self.alwayson_scripts):
            callbacks = list(self.alwayson_scripts)
            self.callback_map["script_process"] = len(callbacks), callbacks
        for script in callbacks:
            script.process(p)

    def process_before_every_sampling(self, p):
        for script in self.alwayson_scripts:
            script.process_before_every_sampling(p)


class _ControlNet:
    """Like Forge's script, it has current_params only after its own process."""

    @staticmethod
    def title():
        return "ControlNet"

    def process(self, p):
        p.ran.append("ControlNet prepared")
        self.current_params = {}

    def process_before_every_sampling(self, p):
        self.current_params.clear()
        p.ran.append("ControlNet applied")


class _SoftInpainting:
    @staticmethod
    def title():
        return "Soft Inpainting"

    def process(self, p):
        p.ran.append("Soft Inpainting")

    def process_before_every_sampling(self, p):
        pass


def _forge_host(monkeypatch):
    backend = _load_backend(monkeypatch, _Gradio4Unit)
    runner = _Runner(_SoftInpainting(), _ControlNet())
    backend.scripts.scripts_img2img = runner
    ext = backend.ControlNetExt()
    ext.init_controlnet()
    return runner, ext


def _pass(runner, ext=None):
    """One detailer pass: with `ext` the tab has a ControlNet model. Without,
    the pass keeps the runner of "Apply only selected scripts", a copy of the
    main one with fewer scripts."""
    p = SimpleNamespace(
        init_images=[Image.new("RGB", (64, 48))],
        width=512, height=512, resize_mode=0, ran=[],
    )
    p.scripts = copy.copy(runner)
    p.scripts.alwayson_scripts = runner.alwayson_scripts[:1]
    if ext:
        ext.update_scripts_args(p, "control_v11p_sd15_inpaint", "inpaint_only", 1.0, 0.0, 1.0)
    p.scripts.process(p)
    p.scripts.process_before_every_sampling(p)
    return p.ran


def test_every_forge_controlnet_pass_runs_its_own_script(monkeypatch):
    # The next image, tab or X/Y/Z cell got a new copy of the script, but the
    # WebUI's cache kept the first copy for `process`: the new one had no
    # current_params, and the region was inpainted without ControlNet.
    runner, ext = _forge_host(monkeypatch)
    expected = ["ControlNet prepared", "ControlNet applied"]

    assert [_pass(runner, ext) for _ in range(3)] == [expected] * 3


@pytest.mark.parametrize("controlnet_first", [False, True])
def test_two_forge_tabs_keep_their_own_scripts(monkeypatch, controlnet_first):
    # Both runners have one script: with the cache shared, the second tab ran
    # the script of the first one with the wrong arguments.
    runner, ext = _forge_host(monkeypatch)
    with_cn = ["ControlNet prepared", "ControlNet applied"]
    without = ["Soft Inpainting"]

    if controlnet_first:
        assert [_pass(runner, ext), _pass(runner)] == [with_cn, without]
    else:
        assert [_pass(runner), _pass(runner, ext)] == [without, with_cn]


def test_a_forge_controlnet_pass_leaves_the_cache_of_the_img2img_runner(monkeypatch):
    runner, ext = _forge_host(monkeypatch)
    cache = runner.callback_map
    runner.process(SimpleNamespace(ran=[]))
    before = dict(cache)

    _pass(runner, ext)

    assert runner.callback_map is cache
    assert cache == before
