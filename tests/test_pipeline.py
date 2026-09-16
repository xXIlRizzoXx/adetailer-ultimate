"""Isolated runtime regressions without starting a WebUI or loading a model.

The host script imports WebUI-only modules and downloads detectors at import
time. Extract its actual function bodies through AST and provide the small host
surface needed by each test. These tests cover control flow and override values;
they do not replace an end-to-end generation test in A1111 / Forge.
"""

from __future__ import annotations

import ast
import re
import sys
import time
from contextlib import nullcontext
from copy import copy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "!adetailer.py"


def _load_runtime(**host_globals):
    tree = ast.parse(_SCRIPT_PATH.read_text(encoding="utf-8"))
    selected = ast.parse("from __future__ import annotations").body
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_forge_wanted_modules", "_parse_class_prompts"
        }:
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "AfterDetailerScript":
            node.bases = []
            node.body = [
                method
                for method in node.body
                if isinstance(method, ast.FunctionDef)
                and method.name in {
                    "process", "set_skip_img2img", "_postprocess_image_inner"
                }
            ]
            for method in node.body:
                method.decorator_list = []
            selected.append(node)
    namespace = {"Path": Path, "sys": sys, **host_globals}
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_SCRIPT_PATH), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("manual_mode", [True, False])
def test_manual_mode_preserves_generation_with_skip_img2img(manual_mode):
    runtime = _load_runtime(
        opts=SimpleNamespace(data={"ad_manual_mode": manual_mode}),
        is_img2img_inpaint=lambda _p: False,
        SkipImg2ImgOrig=SimpleNamespace,
    )
    script = runtime.AfterDetailerScript()
    script.is_ad_enabled = lambda *_args: True
    script.get_args = lambda *_args: []
    script.extra_params = lambda _args: {}
    p = SimpleNamespace(
        init_images=[object()],
        width=1024,
        height=768,
        steps=30,
        sampler_name="DPM++ 2M",
        extra_generation_params={},
    )

    script.process(p, True, True, {})

    if manual_mode:
        assert (p.width, p.height, p.steps, p.sampler_name) == (
            1024, 768, 30, "DPM++ 2M"
        )
        assert p._ad_disabled
        assert not hasattr(p, "_ad_skip_img2img")
    else:
        assert (p.width, p.height, p.steps, p.sampler_name) == (128, 128, 1, "Euler")
        assert p._ad_skip_img2img
        assert (p._ad_orig.width, p._ad_orig.height, p._ad_orig.steps) == (1024, 768, 30)


@pytest.fixture
def forge_runtime(monkeypatch):
    forge = ModuleType("modules_forge")
    entry = ModuleType("modules_forge.main_entry")
    entry.module_list = {
        "new-te.safetensors": "/models/text_encoder/new-te.safetensors",
        "new-vae.safetensors": "/models/VAE/new-vae.safetensors",
    }
    forge.main_entry = entry
    monkeypatch.setitem(sys.modules, "modules_forge", forge)
    monkeypatch.setitem(sys.modules, "modules_forge.main_entry", entry)
    base = [
        "/models/text_encoder/base-te.safetensors",
        "/models/VAE/base-vae.safetensors",
    ]
    runtime = _load_runtime(
        shared=SimpleNamespace(opts=SimpleNamespace(forge_additional_modules=base))
    )
    return runtime, base


@pytest.mark.parametrize(
    ("text_encoder", "vae", "expected"),
    [
        ("missing-te.safetensors", None, None),
        (None, "missing-vae.safetensors", None),
        ("missing-te.safetensors", "missing-vae.safetensors", None),
        ("/models/text_encoder/missing-te.safetensors", None, None),
        (None, "/models/VAE/missing-vae.safetensors", None),
        (
            "/models/text_encoder/new-te.safetensors",
            None,
            ["/models/VAE/base-vae.safetensors", "/models/text_encoder/new-te.safetensors"],
        ),
        (
            None,
            "/models/VAE/new-vae.safetensors",
            ["/models/text_encoder/base-te.safetensors", "/models/VAE/new-vae.safetensors"],
        ),
        (
            "new-te.safetensors",
            "missing-vae.safetensors",
            ["/models/VAE/base-vae.safetensors", "/models/text_encoder/new-te.safetensors"],
        ),
        (
            "missing-te.safetensors",
            "new-vae.safetensors",
            ["/models/text_encoder/base-te.safetensors", "/models/VAE/new-vae.safetensors"],
        ),
        (
            "new-te.safetensors",
            "new-vae.safetensors",
            ["/models/text_encoder/new-te.safetensors", "/models/VAE/new-vae.safetensors"],
        ),
        ("None (remove text encoder)", None, ["/models/VAE/base-vae.safetensors"]),
        ("Use same text encoder", "Use same VAE", None),
    ],
)
def test_forge_preserves_modules_when_preset_choice_is_missing(
    forge_runtime, text_encoder, vae, expected
):
    runtime, base = forge_runtime
    before = list(base)
    args = SimpleNamespace(
        ad_use_text_encoder=text_encoder is not None,
        ad_text_encoder=text_encoder,
        ad_use_vae=vae is not None,
        ad_vae=vae,
    )

    result = runtime._forge_wanted_modules(args)

    assert result == expected
    assert base == before


@pytest.mark.parametrize("cancel_flag", ["skipped", "interrupted"])
def test_sequential_cancel_stops_next_mask_before_host_can_clear_skip(cancel_flag):
    state = SimpleNamespace(
        interrupted=False, skipped=False, job_count=0,
        assign_current_image=lambda _image: None,
    )
    before = Image.new("RGB", (8, 8), "white")
    partial_result = Image.new("RGB", (8, 8), "red")
    pp = SimpleNamespace(image=before)
    calls = []

    def process_images(_p):
        # Both hosts clear Skip when a fresh process_images iteration starts.
        # Calling this again would therefore erase the user's cancellation.
        state.skipped = False
        calls.append(_p)
        if len(calls) == 1:
            setattr(state, cancel_flag, True)
        return SimpleNamespace(images=[partial_result])

    pred = SimpleNamespace(preview=before)
    runtime = _load_runtime(
        state=state, shared=SimpleNamespace(state=state),
        re=re, time=time, copy=copy, get_i=lambda _p: 0,
        parse_csv=lambda text: text.split(","),
        is_skip_img2img=lambda _p: False,
        _ad_verbose=lambda: False,
        _verbose_pass_header=lambda *_args: None,
        _verbose_detection=lambda *_args: None,
        disable_safe_unpickle=nullcontext,
        ultralytics_predict=lambda *_args, **_kwargs: pred,
        ensure_pil_image=lambda image, _mode: image,
        process_images=process_images,
        NansException=type("NansException", (Exception,), {}),
    )
    script = runtime.AfterDetailerScript()
    script.ultralytics_device = "cpu"
    script.get_i2i_p = lambda *_args: SimpleNamespace(
        init_images=[pp.image], prompt="face", close=lambda: None
    )
    script.get_prompt = lambda *_args: (["face"], [""])
    script.get_ad_model = lambda _name: "model.pt"
    script.pred_preprocessing = lambda *_args: [before, before]
    script.save_image = lambda *_args, **_kwargs: None
    script.i2i_prompts_replace = lambda *_args: None
    script._apply_inline_class_prompts = lambda *_args: None
    script._apply_auto_class_guard = lambda *_args: None
    script.fix_p2 = lambda *_args: None
    script.compare_prompt = lambda *_args, **_kwargs: None

    class Args(SimpleNamespace):
        def copy(self, update):
            return Args(**{**vars(self), **update})

    args = Args(
        ad_classes_sequential=True, ad_model_classes="face,hand",
        ad_model_classes_exclude=False, ad_class_prompts="",
        ad_model="model.pt", ad_confidence=0.3, ad_use_bbox_mask=False,
        ad_detection_resolution=0, is_mediapipe=lambda: False,
    )

    processed = script._postprocess_image_inner(
        SimpleNamespace(extra_generation_params={}), pp, args
    )

    assert len(calls) == 1
    assert processed is False
    assert pp.image.tobytes() == before.tobytes()
    assert getattr(state, cancel_flag)


@pytest.mark.parametrize("cancel_flag", ["skipped", "interrupted"])
@pytest.mark.parametrize(
    ("n_masks", "cancel_at"),
    [(1, 0), (2, 0), (2, 1)],
    ids=["only-region", "first-of-two", "last-of-two"],
)
def test_cancel_in_ordinary_flow_discards_the_half_finished_pass(
    cancel_flag, n_masks, cancel_at
):
    # One tab, not sequential. The user cancels while a region is being
    # inpainted: the host still hands back its half-denoised image. That must
    # not reach the final picture (nor a standalone/folder save) whichever
    # region was cancelled, including the last or only one.
    state = SimpleNamespace(
        interrupted=False, skipped=False, job_count=0,
        assign_current_image=lambda _image: None,
    )
    before = Image.new("RGB", (8, 8), "white")
    partial_result = Image.new("RGB", (8, 8), "red")
    pp = SimpleNamespace(image=before)
    calls = []

    def process_images(_p):
        state.skipped = False
        calls.append(_p)
        if len(calls) - 1 == cancel_at:
            setattr(state, cancel_flag, True)  # cancelled during this region
        return SimpleNamespace(images=[partial_result])

    pred = SimpleNamespace(preview=before)
    runtime = _load_runtime(
        state=state, shared=SimpleNamespace(state=state),
        re=re, time=time, copy=copy, get_i=lambda _p: 0,
        parse_csv=lambda text: text.split(","),
        is_skip_img2img=lambda _p: False,
        _ad_verbose=lambda: False,
        _verbose_pass_header=lambda *_args: None,
        _verbose_detection=lambda *_args: None,
        disable_safe_unpickle=nullcontext,
        ultralytics_predict=lambda *_args, **_kwargs: pred,
        ensure_pil_image=lambda image, _mode: image,
        process_images=process_images,
        NansException=type("NansException", (Exception,), {}),
    )
    script = runtime.AfterDetailerScript()
    script.ultralytics_device = "cpu"
    script.get_i2i_p = lambda *_args: SimpleNamespace(
        init_images=[pp.image], prompt="face", close=lambda: None
    )
    script.get_prompt = lambda *_args: (["face"], [""])
    script.get_ad_model = lambda _name: "model.pt"
    script.pred_preprocessing = lambda *_args: [before] * n_masks
    script.save_image = lambda *_args, **_kwargs: None
    script.i2i_prompts_replace = lambda *_args: None
    script._apply_inline_class_prompts = lambda *_args: None
    script._apply_auto_class_guard = lambda *_args: None
    script.fix_p2 = lambda *_args: None
    script.compare_prompt = lambda *_args, **_kwargs: None

    class Args(SimpleNamespace):
        def copy(self, update):
            return Args(**{**vars(self), **update})

    args = Args(
        ad_classes_sequential=False, ad_model_classes="",
        ad_model_classes_exclude=False, ad_class_prompts="",
        ad_model="model.pt", ad_confidence=0.3, ad_use_bbox_mask=False,
        ad_detection_resolution=0, is_mediapipe=lambda: False,
    )

    processed = script._postprocess_image_inner(
        SimpleNamespace(extra_generation_params={}), pp, args
    )

    assert len(calls) == cancel_at + 1
    assert processed is False
    assert pp.image.tobytes() == before.tobytes()
    assert getattr(state, cancel_flag)
