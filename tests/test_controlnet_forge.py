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
    pixel_perfect: bool = False


@dataclass
class _Gradio4Unit(_Gradio3Unit):
    """ControlNetUnit of classic Forge since its Gradio 4 update (and Forge
    Neo): separate image / mask_image arrays, with the image_fg fields."""

    image_fg: object = None
    mask_image_fg: object = None


@dataclass
class _NoPixelPerfectUnit:
    """A ControlNetUnit without the Pixel Perfect option (none known)."""

    enabled: bool = True
    module: str = "None"
    model: str = "None"
    weight: float = 1.0
    image: object = None
    mask_image: object = None
    processor_res: int = -1
    guidance_start: float = 0.0
    guidance_end: float = 1.0


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
    assert (unit.guidance_start, unit.guidance_end) == (0.1, 0.9)
    return np.asarray(p.init_images[0]), unit


def _gradio4_input(p, unit):
    """The image and mask classic Forge (Gradio 4) and Forge Neo pick for a
    unit (scripts/controlnet.py, get_input_data)."""
    unit_image = unit.image
    unit_image_fg = unit.image_fg[:, :, 3] if unit.image_fg is not None else None
    if unit.image is None:
        image, mask = np.asarray(p.init_images[0]), np.asarray(p.image_mask)
    else:
        if (unit_image < 5).all() and (unit_image_fg > 5).any():
            image = unit_image_fg
        else:
            image = unit_image
        mask = unit.mask_image if (unit.mask_image > 5).any() else None
    return image, mask


def _dict_input(p, unit):
    """The same for older Forge and reForge, which read the image dict."""
    if unit.image is None:
        image, mask = np.asarray(p.init_images[0]), np.asarray(p.image_mask)
    else:
        if (unit.image["image"] < 5).all() and (unit.image["mask"] > 5).any():
            image = unit.image["mask"]
        else:
            image = unit.image["image"]
        mask = unit.image["mask"] if (unit.image["mask"] > 5).any() else None
    return image, mask


def _region_mask(box):
    mask = Image.new("L", (64, 48), 0)
    mask.paste(255, box)
    return mask


@pytest.mark.parametrize(
    ("unit_class", "host_input"),
    [
        (_Gradio4Unit, _gradio4_input),
        (_Gradio3Unit, _dict_input),
        (_PlainUnit, _dict_input),
    ],
)
def test_every_forge_controlnet_reads_each_regions_own_picture_and_mask(
    monkeypatch, unit_class, host_input
):
    # As on AUTOMATIC1111. The unit carried the picture of the tab's start
    # with an all-white mask: ControlNet covered the whole crop, and the 2nd
    # region did not see the 1st one already inpainted.
    backend = _load_backend(monkeypatch, unit_class)
    ext = backend.ControlNetExt()
    ext.init_controlnet()
    i2i = SimpleNamespace(
        init_images=[Image.new("RGB", (64, 48), (200, 120, 40))],
        width=512, height=512, resize_mode=0,
    )
    ext.update_scripts_args(i2i, "control_v11p_sd15_inpaint", "inpaint_only", 1.0, 0.0, 1.0)
    (unit,) = i2i.script_args_value
    assert unit.image is None
    assert unit.mask_image is None

    # The detailer loop: p2 = copy(i2i) for each region, with its own mask,
    # and from the 2nd region on the picture the 1st one returned.
    first = copy.copy(i2i)
    first.image_mask = _region_mask((4, 4, 20, 20))
    second = copy.copy(i2i)
    second.init_images = [Image.new("RGB", (64, 48), (10, 220, 90))]
    second.image_mask = _region_mask((40, 20, 60, 44))

    for p in (first, second):
        image, mask = host_input(p, unit)
        assert np.array_equal(image, np.asarray(p.init_images[0]))
        assert np.array_equal(mask, np.asarray(p.image_mask))


@pytest.mark.parametrize("unit_class", [_Gradio4Unit, _Gradio3Unit, _PlainUnit])
def test_a_forge_unit_carries_the_tabs_controlnet_settings(monkeypatch, unit_class):
    _image, unit = _unit(monkeypatch, unit_class)

    assert unit.enabled is True


@pytest.mark.parametrize("unit_class", [_Gradio4Unit, _Gradio3Unit, _PlainUnit])
def test_a_forge_preprocessor_fits_each_regions_canvas(monkeypatch, unit_class):
    # As on AUTOMATIC1111 (Pixel Perfect). One resolution was worked out from
    # the whole picture and the size the pass had before each region's own.
    backend = _load_backend(monkeypatch, unit_class)
    ext = backend.ControlNetExt()
    ext.init_controlnet()
    i2i = SimpleNamespace(
        init_images=[Image.new("RGB", (64, 48))],
        width=1024, height=1024, resize_mode=0,
    )
    ext.update_scripts_args(i2i, "control_v11f1p_sd15_depth", "depth_midas", 1.0, 0.0, 1.0)
    (unit,) = i2i.script_args_value
    assert unit.pixel_perfect is True

    used = []
    for width, height in ((512, 640), (768, 768)):
        p2 = copy.copy(i2i)
        p2.width, p2.height = width, height  # what fix_p2 sets per region
        # Every Forge, when the region runs: the preprocessor's resolution
        # follows the region's canvas (here, its short side).
        if unit.pixel_perfect:
            unit.processor_res = min(p2.width, p2.height)
        used.append(unit.processor_res)

    assert used == [512, 768]


def test_a_forge_unit_without_pixel_perfect_keeps_the_default(monkeypatch):
    _image, unit = _unit(monkeypatch, _NoPixelPerfectUnit)

    assert unit.processor_res == -1


def test_a_forge_controlnet_reads_an_almost_black_picture(monkeypatch):
    # Classic Forge and Forge Neo compare a near-black unit image with its
    # missing image_fg, which raised a TypeError: no ControlNet there.
    backend = _load_backend(monkeypatch, _Gradio4Unit)
    ext = backend.ControlNetExt()
    ext.init_controlnet()
    p = SimpleNamespace(
        init_images=[Image.new("RGB", (64, 48), (2, 2, 2))],
        image_mask=_region_mask((4, 4, 20, 20)),
        width=512, height=512, resize_mode=0,
    )
    ext.update_scripts_args(p, "control_v11f1p_sd15_depth", "depth_midas", 1.0, 0.0, 1.0)
    (unit,) = p.script_args_value

    image, _mask = _gradio4_input(p, unit)

    assert (image == 2).all()


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


class _BatchControlNet(_ControlNet):
    """Forge Neo's script reads the batch folder while `process` reads the image."""

    def __init__(self, group):
        self.group = group

    def process(self, p):
        p.ran.append(f"folder={self.group.GLOBAL_CONTROLNET_BATCH_DIR!r}")
        super().process(p)
        if getattr(p, "fail", False):
            msg = "preprocessor failed"
            raise RuntimeError(msg)


def _forge_neo_host(monkeypatch, folder="batch_input"):
    backend = _load_backend(monkeypatch, _Gradio4Unit)
    group = type("ControlNetUiGroup", (), {"GLOBAL_CONTROLNET_BATCH_DIR": folder})
    ui = ModuleType("lib_controlnet.controlnet_ui")
    ui_group = ModuleType("lib_controlnet.controlnet_ui.controlnet_ui_group")
    ui_group.ControlNetUiGroup = group
    ui.controlnet_ui_group = ui_group
    sys.modules["lib_controlnet"].controlnet_ui = ui
    monkeypatch.setitem(sys.modules, "lib_controlnet.controlnet_ui", ui)
    monkeypatch.setitem(sys.modules, ui_group.__name__, ui_group)
    runner = _Runner(_SoftInpainting(), _BatchControlNet(group))
    backend.scripts.scripts_img2img = runner
    ext = backend.ControlNetExt()
    ext.init_controlnet()
    return runner, ext, group


def test_a_forge_neo_controlnet_pass_ignores_the_batch_folder(monkeypatch):
    # A folder typed in ControlNet's or img2img Batch's input directory made
    # Forge Neo read every unit's image from it, ADetailer's too: the region
    # got the control map of another picture.
    runner, ext, group = _forge_neo_host(monkeypatch)

    ran = _pass(runner, ext)

    assert ran == ["folder=''", "ControlNet prepared", "ControlNet applied"]
    # The main ControlNet keeps the folder.
    assert group.GLOBAL_CONTROLNET_BATCH_DIR == "batch_input"


def test_the_batch_folder_comes_back_after_a_failed_controlnet_pass(monkeypatch):
    runner, ext, group = _forge_neo_host(monkeypatch)
    p = SimpleNamespace(
        init_images=[Image.new("RGB", (64, 48))],
        width=512, height=512, resize_mode=0, ran=[], fail=True,
    )
    p.scripts = copy.copy(runner)
    ext.update_scripts_args(p, "control_v11p_sd15_depth", "depth_midas", 1.0, 0.0, 1.0)

    with pytest.raises(RuntimeError, match="preprocessor failed"):
        p.scripts.process(p)

    assert p.ran == ["folder=''", "ControlNet prepared"]
    assert group.GLOBAL_CONTROLNET_BATCH_DIR == "batch_input"


def test_a_batch_folder_typed_during_the_pass_is_kept(monkeypatch):
    runner, ext, group = _forge_neo_host(monkeypatch)
    script = runner.scripts[1]
    process = type(script).process

    def typed_meanwhile(self, p):
        process(self, p)
        group.GLOBAL_CONTROLNET_BATCH_DIR = "new_folder"

    monkeypatch.setattr(_BatchControlNet, "process", typed_meanwhile)

    _pass(runner, ext)

    assert group.GLOBAL_CONTROLNET_BATCH_DIR == "new_folder"


def test_a_batch_folder_emptied_during_the_pass_stays_empty(monkeypatch):
    # Forge Neo updates the folder as soon as the box loses focus, also while
    # a pass runs: the emptied box was taken for the pass's own clearing and
    # the old folder came back, active with an empty box on screen.
    runner, ext, group = _forge_neo_host(monkeypatch)
    script = runner.scripts[1]
    process = type(script).process

    def emptied_meanwhile(self, p):
        process(self, p)
        group.GLOBAL_CONTROLNET_BATCH_DIR = ""

    monkeypatch.setattr(_BatchControlNet, "process", emptied_meanwhile)

    ran = _pass(runner, ext)

    assert ran[0] == "folder=''"
    assert group.GLOBAL_CONTROLNET_BATCH_DIR == ""


def test_the_cleared_batch_folder_reads_as_no_folder(monkeypatch):
    # Forge Neo reads the value with .strip() and os.path.isdir().
    runner, ext, group = _forge_neo_host(monkeypatch)
    script = runner.scripts[1]
    process = type(script).process
    seen = []

    def read_as_forge_neo(self, p):
        seen.append(group.GLOBAL_CONTROLNET_BATCH_DIR.strip())
        process(self, p)

    monkeypatch.setattr(_BatchControlNet, "process", read_as_forge_neo)

    _pass(runner, ext)

    assert seen == [""]
    assert type(seen[0]) is str
    assert group.GLOBAL_CONTROLNET_BATCH_DIR == "batch_input"
    assert type(group.GLOBAL_CONTROLNET_BATCH_DIR) is str


@pytest.mark.parametrize("folder", ["", "   "])
def test_a_forge_without_a_batch_folder_runs_controlnet_as_before(monkeypatch, folder):
    runner, ext, group = _forge_neo_host(monkeypatch, folder)

    ran = _pass(runner, ext)

    assert ran == [f"folder={folder!r}", "ControlNet prepared", "ControlNet applied"]
    assert group.GLOBAL_CONTROLNET_BATCH_DIR == folder
