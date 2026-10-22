from __future__ import annotations

import copy
from itertools import chain

import numpy as np
from lib_controlnet import external_code, global_state
from lib_controlnet.external_code import ControlNetUnit

from modules import scripts
from modules.processing import StableDiffusionProcessing

from .common import cn_model_regex, default_cn_module

controlnet_exists = True
controlnet_type = "forge"


def find_script(p: StableDiffusionProcessing, script_title: str) -> scripts.Script:
    script = next((s for s in p.scripts.scripts if s.title() == script_title), None)
    if not script:
        msg = f"Script not found: {script_title!r}"
        raise RuntimeError(msg)
    return script


def add_forge_script_to_adetailer_run(
    p: StableDiffusionProcessing, script_title: str, script_args: list
):
    p.scripts = copy.copy(scripts.scripts_img2img)
    p.scripts.alwayson_scripts = []
    p.script_args_value = []

    script = copy.copy(find_script(p, script_title))
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
        name = model.lower().replace("inpaint_depth", "depth")
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

        image = np.asarray(p.init_images[0])
        mask = np.full_like(image, fill_value=255)

        # The Gradio 4 ControlNet (classic Forge since mid-2024, reForge's
        # Gradio 4 branches, Forge Neo; its unit has image_fg) reads image and
        # mask_image as separate arrays; the older one (older Forge, reForge's
        # main branch) reads only the {"image", "mask"} dict.
        if "image_fg" in getattr(ControlNetUnit, "__dataclass_fields__", {}):
            image_kwargs = {"image": image, "mask_image": mask}
        else:
            image_kwargs = {"image": {"image": image, "mask": mask}}

        pres = external_code.pixel_perfect_resolution(
            image,
            target_H=p.height,
            target_W=p.width,
            resize_mode=external_code.resize_mode_from_value(p.resize_mode),
        )

        add_forge_script_to_adetailer_run(
            p,
            "ControlNet",
            [
                ControlNetUnit(
                    enabled=True,
                    **image_kwargs,
                    model=model,
                    module=module,
                    weight=weight,
                    guidance_start=guidance_start,
                    guidance_end=guidance_end,
                    processor_res=pres,
                )
            ],
        )


def get_cn_models() -> list[str]:
    models = global_state.get_all_controlnet_names()
    return [m for m in models if cn_model_regex.search(m)]
