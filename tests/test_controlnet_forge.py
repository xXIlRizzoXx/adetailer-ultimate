"""The Forge ControlNet backend builds the unit each Forge variant can read.

controlnet_ext/controlnet_ext_forge.py imports the Forge-only lib_controlnet
and the WebUI's modules, so both are replaced by small stand-ins here.
"""

from __future__ import annotations

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


def _load_backend(monkeypatch, unit_class):
    external_code = ModuleType("lib_controlnet.external_code")
    external_code.ControlNetUnit = unit_class
    external_code.pixel_perfect_resolution = lambda *_args, **_kwargs: 512
    external_code.resize_mode_from_value = lambda value: value
    global_state = ModuleType("lib_controlnet.global_state")
    global_state.get_all_controlnet_names = lambda: []
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
