from __future__ import annotations

import copy
from itertools import chain

from lib_controlnet import external_code, global_state
from lib_controlnet.external_code import ControlNetUnit

from modules import scripts
from modules.processing import StableDiffusionProcessing

from .common import anytest_choices, anytest_name, cn_model_regex, default_cn_module

controlnet_exists = True
controlnet_type = "forge"


def find_script(p: StableDiffusionProcessing, script_title: str) -> scripts.Script:
    script = next((s for s in p.scripts.scripts if s.title() == script_title), None)
    if not script:
        msg = f"Script not found: {script_title!r}"
        raise RuntimeError(msg)
    return script


class _ClearedBatchDir(str):
    """The empty batch folder a pass sets while it reads its image."""

    __slots__ = ()


def _without_batch_dir(process):
    """Run the pass's ControlNet `process` without Forge Neo's batch folder.

    On Forge Neo, a folder typed in ControlNet's batch input directory or in
    the img2img Batch tab's input directory replaces the image of every unit,
    the ADetailer one included, with a file of that folder: the region got
    the control map of another picture. The folder is cleared only while this
    pass reads its image (as in the IOSakaki/adetailer fork, commit 649ef43).
    Other Forge versions have no such folder and run as before.
    """

    def run(*args, **kwargs):
        try:
            from lib_controlnet.controlnet_ui import controlnet_ui_group

            group = controlnet_ui_group.ControlNetUiGroup
        except Exception:  # noqa: BLE001 - no batch folder in this Forge
            group = None
        folder = getattr(group, "GLOBAL_CONTROLNET_BATCH_DIR", None)
        if not isinstance(folder, str) or not folder.strip():
            return process(*args, **kwargs)
        # An empty text of its own, which Forge Neo reads as no folder
        # (.strip(), then os.path.isdir) and which is told apart from the ""
        # of a box the user emptied in the meantime.
        cleared = _ClearedBatchDir()
        group.GLOBAL_CONTROLNET_BATCH_DIR = cleared
        try:
            return process(*args, **kwargs)
        finally:
            # Not over a folder the user typed, or emptied, in the meantime.
            if group.GLOBAL_CONTROLNET_BATCH_DIR is cleared:
                group.GLOBAL_CONTROLNET_BATCH_DIR = folder

    return run


def add_forge_script_to_adetailer_run(
    p: StableDiffusionProcessing, script_title: str, script_args: list
):
    p.scripts = copy.copy(scripts.scripts_img2img)
    # The host's cache of ordered scripts compares only their number, and this
    # runner always has one: shared, the next pass ran the script of the last.
    p.scripts.callback_map = {}
    p.scripts.alwayson_scripts = []
    p.script_args_value = []

    script = copy.copy(find_script(p, script_title))
    if callable(getattr(script, "process", None)):
        script.process = _without_batch_dir(script.process)
    script.args_from = len(p.script_args_value)
    script.args_to = len(p.script_args_value) + len(script_args)
    p.scripts.alwayson_scripts.append(script)
    p.script_args_value.extend(script_args)


# The preprocessor tags the ADetailer module dropdown lists for each model type.
_CN_TYPE_TAGS = {
    "inpaint": "Inpaint",
    "lineart": "Lineart",
    "openpose": "OpenPose",
    "tile": "Tile",
    "scribble": "Scribble",
    "depth": "Depth",
}


def default_module(model: str) -> str:
    """The preprocessor the module dropdown would pick for `model`.

    For a request (API) that names no preprocessor: Forge reads a missing one
    as "None" and ran the model on the raw crop. "None" when unknown.
    """
    try:
        by_type = {
            t: list(global_state.get_filtered_preprocessors(tag))
            for t, tag in _CN_TYPE_TAGS.items()
        }
        by_type["union"] = list(chain.from_iterable(by_type.values()))
        by_type["anytest"] = anytest_choices(by_type["union"], "None")
        name = anytest_name(model).replace("inpaint_depth", "depth")
        for t, choices in by_type.items():
            if t in name:
                return default_cn_module(t, choices)
    except Exception:  # noqa: BLE001 - a different Forge: run as before
        pass
    return "None"


class ControlNetExt:
    def __init__(self):
        self.cn_available = False
        self.external_cn = external_code

    def init_controlnet(self):
        self.cn_available = True

    def update_scripts_args(  # noqa: PLR0913
        self,
        p,
        model: str,
        module: str | None,
        weight: float,
        guidance_start: float,
        guidance_end: float,
    ):
        if (not self.cn_available) or model == "None":
            return

        if module is None:
            module = default_module(model)

        # No image in the unit, as on AUTOMATIC1111: every Forge (classic,
        # Forge Neo, reForge, old and Gradio 4) then reads the pass's own
        # picture and mask (p.init_images[0], p.image_mask) when each region
        # runs. A fixed picture with an all-white mask applied ControlNet to
        # the whole crop, and later regions did not see the earlier ones.
        # Pixel Perfect, also as on AUTOMATIC1111: ControlNet sizes the
        # preprocessor to each region's crop and canvas. A resolution worked
        # out here came from the whole picture, before the region's size was
        # set. A unit without the option keeps ControlNet's default.
        fields = getattr(ControlNetUnit, "__dataclass_fields__", None)
        resolution = {"pixel_perfect": True}
        if fields is not None and "pixel_perfect" not in fields:
            resolution = {}

        add_forge_script_to_adetailer_run(
            p,
            "ControlNet",
            [
                ControlNetUnit(
                    enabled=True,
                    model=model,
                    module=module,
                    weight=weight,
                    guidance_start=guidance_start,
                    guidance_end=guidance_end,
                    **resolution,
                )
            ],
        )


def get_cn_models() -> list[str]:
    models = global_state.get_all_controlnet_names()
    return [m for m in models if cn_model_regex.search(m)]
