from __future__ import annotations

import os
from contextlib import contextmanager
from copy import copy
from typing import TYPE_CHECKING, Any, Union
from unittest.mock import patch

import torch
from PIL import Image
from typing_extensions import Protocol

try:
    from modules import safe
except Exception:  # variant without modules.safe (Forge Neo is moving off it)
    safe = None
from modules.shared import cmd_opts, opts

if TYPE_CHECKING:
    # 타입 체커가 빨간 줄을 긋지 않게 하는 편법
    from types import SimpleNamespace

    StableDiffusionProcessingTxt2Img = SimpleNamespace
    StableDiffusionProcessingImg2Img = SimpleNamespace
else:
    from modules.processing import (
        StableDiffusionProcessingImg2Img,
        StableDiffusionProcessingTxt2Img,
    )

PT = Union[StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img]


@contextmanager
def change_torch_load():
    orig = torch.load
    try:
        if safe is not None and hasattr(safe, "unsafe_torch_load"):
            torch.load = safe.unsafe_torch_load
        yield
    finally:
        torch.load = orig


@contextmanager
def disable_safe_unpickle():
    # Forge Neo (>= neo-2.x) dropped the `disable_safe_unpickle` attribute from
    # cmd_opts. patch.object(..., create=True) makes the patch resilient: it
    # creates the attribute if missing, restores it (or deletes it) afterward.
    with (
        patch.dict(os.environ, {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}, clear=False),
        patch.object(cmd_opts, "disable_safe_unpickle", True, create=True),
    ):
        yield


# Options ADetailer's inner pass can set through override_settings (see
# get_override_settings in scripts/!adetailer.py). The host puts such an
# option back afterwards only if it was already in opts.data.
_AD_OVERRIDE_KEYS = (
    "CLIP_stop_at_last_layers",
    "sd_model_checkpoint",
    "sd_vae",
    "forge_additional_modules",
)


@contextmanager
def pause_total_tqdm():
    # Undo only the pass's own changes: restoring a snapshot of the whole
    # opts.data also undid every setting changed from the UI meanwhile.
    data = opts.data
    had_key = "multiple_tqdm" in data
    orig = data.get("multiple_tqdm")
    absent = [key for key in _AD_OVERRIDE_KEYS if key not in data]
    data["multiple_tqdm"] = False
    try:
        yield
    finally:
        if had_key:
            data["multiple_tqdm"] = orig
        else:
            data.pop("multiple_tqdm", None)
        for key in absent:
            data.pop(key, None)


@contextmanager
def preserve_prompts(p: PT):
    all_pt = copy(p.all_prompts)
    all_ng = copy(p.all_negative_prompts)
    try:
        yield
    finally:
        p.all_prompts = all_pt
        p.all_negative_prompts = all_ng


def copy_extra_params(extra_params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in extra_params.items() if not callable(v)}


class PPImage(Protocol):
    image: Image.Image
