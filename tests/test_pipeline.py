"""Isolated runtime regressions without starting a WebUI or loading a model.

The host script imports WebUI-only modules and downloads detectors at import
time. Extract its actual function bodies through AST and provide the small host
surface needed by each test. These tests cover control flow and override values;
they do not replace an end-to-end generation test in A1111 / Forge.
"""

from __future__ import annotations

import ast
import io
import random
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
            "_forge_wanted_modules", "_parse_class_prompts", "_class_prompt_for"
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
        # "Automatic" is a dropdown choice, not a missing file: the detailer
        # checkpoint uses its own VAE, so the base VAE is dropped.
        (None, "Automatic", ["/models/text_encoder/base-te.safetensors"]),
        ("None (remove text encoder)", "Automatic", []),
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


@pytest.mark.parametrize("cancel_flag", ["skipped", "interrupted"])
def test_cancel_with_nan_on_the_last_region_discards_the_pass(cancel_flag):
    # A NansException makes the loop `continue` past the per-region cancel
    # check. If that happens on the last region after the user cancelled, the
    # cancelled pass must still be discarded rather than kept as complete.
    state = SimpleNamespace(
        interrupted=False, skipped=False, job_count=0,
        assign_current_image=lambda _image: None,
    )
    before = Image.new("RGB", (8, 8), "white")
    first_result = Image.new("RGB", (8, 8), "red")
    pp = SimpleNamespace(image=before)
    nans = type("NansException", (Exception,), {})
    calls = []

    def process_images(_p):
        state.skipped = False
        calls.append(_p)
        if len(calls) == 2:
            setattr(state, cancel_flag, True)
            raise nans("NaN in the last region")
        return SimpleNamespace(images=[first_result])

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
        NansException=nans,
        ordinal=str,
    )
    script = runtime.AfterDetailerScript()
    script.ultralytics_device = "cpu"
    script.get_i2i_p = lambda *_args: SimpleNamespace(
        init_images=[pp.image], prompt="face", close=lambda: None,
        denoising_strength=0.4, width=512, height=512,
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
        ad_classes_sequential=False, ad_model_classes="",
        ad_model_classes_exclude=False, ad_class_prompts="",
        ad_model="model.pt", ad_confidence=0.3, ad_use_bbox_mask=False,
        ad_detection_resolution=0, is_mediapipe=lambda: False,
    )

    processed = script._postprocess_image_inner(
        SimpleNamespace(extra_generation_params={}), pp, args
    )

    assert len(calls) == 2
    assert processed is False
    assert pp.image.tobytes() == before.tobytes()


def _detailer(state, process_images, n_masks, **host_globals):
    before = Image.new("RGB", (8, 8), "white")
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
        **host_globals,
    )
    script = runtime.AfterDetailerScript()
    pp = SimpleNamespace(image=before)
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

    def run(classes="", sequential=False):
        args = Args(
            ad_classes_sequential=sequential, ad_model_classes=classes,
            ad_model_classes_exclude=False, ad_class_prompts="",
            ad_model="model.pt", ad_confidence=0.3, ad_use_bbox_mask=False,
            ad_detection_resolution=0, is_mediapipe=lambda: False,
        )
        return script._postprocess_image_inner(
            SimpleNamespace(extra_generation_params={}), pp, args
        )

    return run, pp, before


@pytest.mark.parametrize(
    ("stop_at", "n_calls", "kept"),
    [(1, 2, False), (2, 2, False), (3, 4, False), (4, 4, True)],
    ids=[
        "first-class-first-region", "first-class-last-region",
        "last-class-first-region", "last-class-last-region",
    ],
)
def test_sequential_pass_honours_stop_after_current_image(stop_at, n_calls, kept):
    # AUTOMATIC1111's default Interrupt only sets stopping_generation. The host
    # finishes the region in progress, then returns no images for any later
    # region. A sequential tab cut short must roll back instead of keeping the
    # classes done so far as a complete result; one that finished is kept.
    state = SimpleNamespace(
        interrupted=False, skipped=False, stopping_generation=False,
        job_count=0, assign_current_image=lambda _image: None,
    )
    result = Image.new("RGB", (8, 8), "red")
    calls = []

    def process_images(_p):
        calls.append(_p)
        if state.stopping_generation:
            return SimpleNamespace(images=[])
        if len(calls) == stop_at:
            state.stopping_generation = True
        return SimpleNamespace(images=[result])

    run, pp, before = _detailer(state, process_images, n_masks=2)
    processed = run(classes="face,hand", sequential=True)

    assert len(calls) == n_calls
    assert processed is kept
    expected = result if kept else before
    assert pp.image.tobytes() == expected.tobytes()


@pytest.mark.parametrize("cancel_flag", ["skipped", "interrupted", None])
def test_host_error_after_a_cancel_counts_as_the_cancel(cancel_flag):
    # Forge Neo hands back no latent when a cancel lands before the first
    # sampling step, and then fails on it. That is the user's cancel, not a
    # failure; any other error must still reach the caller.
    state = SimpleNamespace(
        interrupted=False, skipped=False, job_count=0,
        assign_current_image=lambda _image: None,
    )

    def process_images(_p):
        if cancel_flag:
            setattr(state, cancel_flag, True)
        msg = "unsupported operand type(s) for *: 'NoneType' and 'Tensor'"
        raise TypeError(msg)

    run, pp, before = _detailer(state, process_images, n_masks=2)
    if cancel_flag is None:
        with pytest.raises(TypeError):
            run()
        return
    assert run() is False
    assert pp.image.tobytes() == before.tobytes()


@pytest.mark.parametrize(("start", "expected"), [(-1, 2), (0, 2), (3, 5)])
def test_region_count_is_added_to_the_host_job_count(start, expected):
    # A standalone job begins at the host's "not counted yet" value, -1.
    state = SimpleNamespace(
        interrupted=False, skipped=False, job_count=start,
        assign_current_image=lambda _image: None,
    )
    result = Image.new("RGB", (8, 8), "red")
    run, _pp, _before = _detailer(
        state, lambda _p: SimpleNamespace(images=[result]), n_masks=2
    )
    assert run() is True
    assert state.job_count == expected


# Final debugging pass: generation pipeline, prompts and docs.


def _load_script(methods=(), functions=(), assigns=(), **host_globals):
    """Like _load_runtime, for any methods and module-level names.

    Keeps ``@staticmethod`` (the methods call each other through ``self``)
    and drops the other decorators.
    """
    tree = ast.parse(_SCRIPT_PATH.read_text(encoding="utf-8"))
    selected = ast.parse("from __future__ import annotations").body
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            getattr(target, "id", "") in assigns for target in node.targets
        ):
            selected.append(node)
        elif (
            isinstance(node, ast.ClassDef)
            and node.name == "AfterDetailerScript"
            and methods
        ):
            node.bases = []
            node.body = [
                method
                for method in node.body
                if isinstance(method, ast.FunctionDef) and method.name in methods
            ]
            for method in node.body:
                method.decorator_list = [
                    d for d in method.decorator_list
                    if getattr(d, "id", "") == "staticmethod"
                ]
            selected.append(node)
    namespace = {"Path": Path, "sys": sys, "re": re, **host_globals}
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_SCRIPT_PATH), "exec"), namespace)
    return SimpleNamespace(**namespace)


def test_forge_automatic_vae_prints_no_missing_module_warning(forge_runtime, capsys):
    runtime, _base = forge_runtime
    args = SimpleNamespace(
        ad_use_text_encoder=False, ad_text_encoder=None,
        ad_use_vae=True, ad_vae="Automatic",
    )

    runtime._forge_wanted_modules(args)

    assert "not found" not in capsys.readouterr().err


def test_region_after_a_nan_error_gets_its_own_dynamic_denoise():
    # A region that raises NansException leaves p2 in place for the next
    # region. fix_p2 derives the denoising strength and the inpaint size from
    # p2's own values, so the next region must not start from the failed
    # region's adjusted ones.
    from adetailer.args import ADetailerArgs, InpaintBBoxMatchMode
    from adetailer.opts import dynamic_denoise_strength, optimal_crop_size

    state = SimpleNamespace(
        interrupted=False, skipped=False, job_count=0,
        assign_current_image=lambda _image: None,
    )
    nans = type("NansException", (Exception,), {})
    image = Image.new("RGB", (1000, 1000), "white")
    mask = Image.new("L", (1000, 1000), 0)
    pred = SimpleNamespace(
        preview=image,
        bboxes=[[0, 0, 316, 316], [500, 500, 816, 816]],
        class_names=[],
    )
    seen = []

    def process_images(p2):
        seen.append((round(p2.denoising_strength, 4), p2.width, p2.height))
        if len(seen) == 1:
            raise nans("NaN in the first region")
        return SimpleNamespace(images=[image])

    runtime = _load_script(
        methods={
            "_postprocess_image_inner", "fix_p2", "get_dynamic_denoise_strength",
            "get_optimal_crop_image_size", "get_seed", "get_each_tab_seed",
        },
        state=state,
        shared=SimpleNamespace(
            state=state, opts=SimpleNamespace(data={}), sd_model=None
        ),
        opts=SimpleNamespace(
            data={
                "ad_dynamic_denoise_power": 2,
                "ad_match_inpaint_bbox_size": InpaintBBoxMatchMode.OFF.value,
            }
        ),
        time=time, copy=copy, get_i=lambda _p: 0,
        parse_csv=lambda text: text.split(","),
        is_skip_img2img=lambda _p: False,
        _ad_verbose=lambda: False,
        _verbose_pass_header=lambda *_args: None,
        _verbose_detection=lambda *_args: None,
        disable_safe_unpickle=nullcontext,
        ultralytics_predict=lambda *_args, **_kwargs: pred,
        ensure_pil_image=lambda im, _mode: im,
        process_images=process_images,
        NansException=nans,
        ordinal=str,
        InpaintBBoxMatchMode=InpaintBBoxMatchMode,
        dynamic_denoise_strength=dynamic_denoise_strength,
        optimal_crop_size=optimal_crop_size,
    )
    script = runtime.AfterDetailerScript()
    script.ultralytics_device = "cpu"
    script.get_i2i_p = lambda *_args: SimpleNamespace(
        init_images=[image], prompt="face", negative_prompt="",
        close=lambda: None, denoising_strength=0.4, width=512, height=512,
    )
    script.get_prompt = lambda *_args: (["face"], [""])
    script.get_ad_model = lambda _name: "model.pt"
    script.pred_preprocessing = lambda *_args: [mask, mask]
    script.save_image = lambda *_args, **_kwargs: None
    script.i2i_prompts_replace = lambda *_args: None
    script._apply_inline_class_prompts = lambda *_args: None
    script._apply_auto_class_guard = lambda *_args: None
    script.compare_prompt = lambda *_args, **_kwargs: None
    p = SimpleNamespace(
        extra_generation_params={}, seed=1, subseed=1, all_seeds=[1], all_subseeds=[1]
    )

    processed = script._postprocess_image_inner(
        p, SimpleNamespace(image=image), ADetailerArgs(ad_model="face_yolov8n.pt")
    )

    assert processed is True
    assert seen == [(0.3241, 512, 512), (0.3241, 512, 512)]


def test_skip_img2img_keeps_the_users_sampler_and_infotext():
    from adetailer.args import ADetailerArgs, SkipImg2ImgOrig

    runtime = _load_script(
        methods={"process", "set_skip_img2img", "get_sampler"},
        opts=SimpleNamespace(data={}),
        is_img2img_inpaint=lambda _p: False,
        SkipImg2ImgOrig=SkipImg2ImgOrig,
    )
    script = runtime.AfterDetailerScript()
    script.is_ad_enabled = lambda *_args: True
    script.get_args = lambda *_args: []
    script.extra_params = lambda _args: {}
    p = SimpleNamespace(
        init_images=[object()], width=1024, height=768, steps=30,
        sampler_name="DPM++ 2M", extra_generation_params={},
    )

    script.process(p, True, True, {})

    same = ADetailerArgs(ad_use_sampler=True, ad_sampler="Use same sampler")
    assert script.get_sampler(p, same) == "DPM++ 2M"
    assert script.get_sampler(SimpleNamespace(sampler_name="Euler a"), same) == "Euler a"
    # The host builds the saved image's infotext from p and merges its extra
    # params after its own keys.
    infotext = {
        "Steps": p.steps,
        "Sampler": p.sampler_name,
        "Size": f"{p.width}x{p.height}",
        **p.extra_generation_params,
    }
    assert infotext == {"Steps": 30, "Sampler": "DPM++ 2M", "Size": "1024x768"}
    # The throwaway pass, and later batch iterations, keep their values.
    assert (p.steps, p.sampler_name, p.width, p.height) == (1, "Euler", 128, 128)


@pytest.mark.parametrize(
    ("second_file", "expected"),
    [
        # img2img Batch tab, "Resize by": the host sets each file's own size.
        ({"width": 1216, "height": 832}, (30, "DPM++ 2M", 1216, 832)),
        # "Append png info": also that file's own steps and sampler.
        (
            {"width": 1216, "height": 832, "steps": 40, "sampler_name": "DDIM"},
            (40, "DDIM", 1216, 832),
        ),
        # A file of its own made with Euler, or with one step, keeps it: the
        # host puts back steps and sampler together.
        ({"steps": 25, "sampler_name": "Euler"}, (25, "Euler", 832, 1216)),
        ({"steps": 1, "sampler_name": "Euler a"}, (1, "Euler a", 832, 1216)),
        # Batch count, loopback, "Resize to": p keeps the throwaway values.
        ({}, (30, "DPM++ 2M", 832, 1216)),
    ],
)
def test_skip_img2img_records_each_batch_files_own_settings(second_file, expected):
    # The host reuses p for every file of the Batch tab; the first file's
    # size, steps and sampler were recorded for all of them, and the later
    # files' throwaway pass ran at full size.
    from adetailer.args import ADetailerArgs, SkipImg2ImgOrig

    runtime = _load_script(
        methods={"process", "set_skip_img2img", "get_width_height"},
        opts=SimpleNamespace(data={}),
        is_img2img_inpaint=lambda _p: False,
        SkipImg2ImgOrig=SkipImg2ImgOrig,
    )
    script = runtime.AfterDetailerScript()
    script.is_ad_enabled = lambda *_args: True
    script.get_args = lambda *_args: []
    script.extra_params = lambda _args: {}
    p = SimpleNamespace(
        init_images=[object()], width=832, height=1216, steps=30,
        sampler_name="DPM++ 2M", extra_generation_params={},
    )
    script.process(p, True, True, {})

    p.init_images = [object()]
    for key, value in second_file.items():
        setattr(p, key, value)
    script.process(p, True, True, {})

    steps, sampler, width, height = expected
    infotext = {
        "Steps": p.steps,
        "Sampler": p.sampler_name,
        "Size": f"{p.width}x{p.height}",
        **p.extra_generation_params,
    }
    assert infotext == {"Steps": steps, "Sampler": sampler, "Size": f"{width}x{height}"}
    assert (p.steps, p.sampler_name, p.width, p.height) == (1, "Euler", 128, 128)
    whole = ADetailerArgs(ad_inpaint_only_masked=False)
    assert script.get_width_height(p, whole) == (width, height)


def _standalone_runtime(tmp_path, **host_globals):
    state = SimpleNamespace(
        interrupted=False, skipped=False, stopping_generation=False
    )
    host_globals.setdefault("pause_total_tqdm", nullcontext)
    return _load_script(
        methods={
            "run_detailer_on_image", "read_params_txt", "write_params_txt",
            "get_seed", "get_each_tab_seed",
        },
        paths=SimpleNamespace(data_path=str(tmp_path)),
        PARAMS_TXT="params.txt",
        shared=SimpleNamespace(
            sd_model=None,
            opts=SimpleNamespace(data={"ad_same_seed_for_each_tab": False}),
        ),
        state=state,
        opts=SimpleNamespace(samples_format="png"),
        all_samplers=[],
        ensure_pil_image=lambda image, _mode: image,
        images=None,
        AD_APPLY_SUBDIR="ADetailer-Inpaint",
        get_i=lambda _p: 0,
        **host_globals,
    )


@pytest.mark.parametrize("outcome", ["detailed", "nothing", "error"])
def test_standalone_run_keeps_the_last_generation_in_params_txt(tmp_path, outcome):
    # With an empty prompt the paste button reads params.txt, which the host
    # rewrites for the inner detailer pass.
    params_txt = tmp_path / "params.txt"
    params_txt.write_text("user generation", encoding="utf-8")
    script = _standalone_runtime(tmp_path).AfterDetailerScript()

    def inner(_p, _pp, _args, n=0):
        params_txt.write_text("inner pass", encoding="utf-8")
        if outcome == "error":
            raise RuntimeError("CUDA out of memory")
        return outcome == "detailed"

    script._postprocess_image_inner = inner

    image, status = script.run_detailer_on_image(
        Image.new("RGB", (64, 64)), SimpleNamespace(), save=False
    )

    assert params_txt.read_text(encoding="utf-8") == "user generation"
    if outcome == "error":
        assert image is None
        assert "failed" in status
    else:
        assert image is not None


@pytest.mark.parametrize("tab", [0, 1, 3])
def test_standalone_run_names_the_tab_it_was_started_from(tmp_path, tab):
    # "Run ADetailer on an image" and folder runs from the 2nd tab or a later
    # one ran their inner pass as the 1st tab (n=0): its console lines and
    # its -ad-preview / -ad-step files named the 1st tab.
    script = _standalone_runtime(tmp_path).AfterDetailerScript()
    tabs = []

    def inner(_p, _pp, _args, n=0):
        tabs.append(n)
        return True

    script._postprocess_image_inner = inner

    _image, status = script.run_detailer_on_image(
        Image.new("RGB", (64, 64)), SimpleNamespace(), save=False, tab=tab
    )

    assert status == "✅ ADetailer pass complete."
    assert tabs == [tab]


def test_standalone_run_gives_every_region_a_random_seed(tmp_path, monkeypatch):
    values = iter([1000, 2000, 3000, 4000])
    monkeypatch.setattr(random, "randrange", lambda _stop: next(values))
    script = _standalone_runtime(tmp_path).AfterDetailerScript()
    shells = []

    def inner(p, _pp, _args, n=0):
        shells.append(p)
        return False

    script._postprocess_image_inner = inner

    for _ in range(2):
        script.run_detailer_on_image(Image.new("RGB", (64, 64)), SimpleNamespace())

    first = shells[0]
    assert (first.seed, first.all_seeds) == (1000, [1000])
    assert (first.subseed, first.all_subseeds) == (2000, [2000])
    # fix_p2 inpaints region j with seed + j.
    region_seeds = [
        [script.get_each_tab_seed(script.get_seed(p)[0], j) for j in range(3)]
        for p in shells
    ]
    assert region_seeds == [[1000, 1001, 1002], [3000, 3001, 3002]]


_FACE_BOXES = [(10, 10, 110, 110), (150, 10, 250, 110)]
_HAND_BOX = (300, 300, 400, 400)
_CLASS_PROMPT = "detailed, [CLASS=face]smile[/CLASS], [CLASS=hand]five fingers[/CLASS]"


def _class_prompt_script():
    from adetailer.args import BBOX_SORTBY
    from adetailer.classes import _has_token, build_class_guard
    from adetailer.mask import (
        filter_by_indices,
        filter_by_ratio,
        filter_k_by,
        mask_preprocess,
        parse_indices,
        sort_bboxes,
    )

    runtime = _load_script(
        methods={
            "pred_preprocessing", "_reindex_pred", "sort_bboxes",
            "_apply_auto_class_guard", "_apply_inline_class_prompts",
        },
        functions={
            "_parse_class_prompts", "_class_prompt_for", "_resolve_inline_class_prompt"
        },
        assigns={"_INLINE_CLASS_RE"},
        opts=SimpleNamespace(data={}),
        BBOX_SORTBY=BBOX_SORTBY,
        filter_by_indices=filter_by_indices,
        filter_by_ratio=filter_by_ratio,
        filter_k_by=filter_k_by,
        mask_preprocess=mask_preprocess,
        parse_indices=parse_indices,
        sort_bboxes=sort_bboxes,
        is_img2img_inpaint=lambda _p: False,
        is_inpaint_only_masked=lambda _p: True,
        get_model_class_names=lambda _path: ["face", "hand"],
        build_class_guard=build_class_guard,
        _has_token=_has_token,
    )
    script = runtime.AfterDetailerScript()
    script.get_ad_model = lambda name: name
    return script


def _detections(classes, size=(512, 512)):
    from adetailer.common import PredictOutput

    faces = iter(_FACE_BOXES)
    bboxes, masks = [], []
    for name in classes:
        box = next(faces) if name == "face" else _HAND_BOX
        mask = Image.new("L", size, 0)
        mask.paste(255, box)
        bboxes.append(list(box))
        masks.append(mask)
    return PredictOutput(
        bboxes=bboxes,
        masks=masks,
        confidences=[0.9] * len(classes),
        preview=Image.new("RGB", size),
        class_names=list(classes),
    )


@pytest.mark.parametrize(
    ("mode", "classes", "prompt", "negative"),
    [
        # One merged mask holds a face and a hand: no single class applies.
        ("Merge", ["face", "hand"], "detailed, smile, five fingers", "blurry"),
        # The inverted mask is the background: no detected class applies.
        ("Merge and Invert", ["face"], "detailed", "blurry"),
        ("Merge and Invert", ["face", "hand"], "detailed", "blurry"),
        # Unchanged: two merged faces are still a face, and so is one region.
        ("Merge", ["face", "face"], "face, detailed, smile", "blurry, hand"),
        ("None", ["face"], "face, detailed, smile", "blurry, hand"),
    ],
)
def test_merged_and_inverted_masks_get_the_right_class_prompts(
    mode, classes, prompt, negative
):
    from adetailer.args import ADetailerArgs

    script = _class_prompt_script()
    pred = _detections(classes)
    args = ADetailerArgs(
        ad_model="faces.pt", ad_class_guard=True, ad_mask_merge_invert=mode
    )

    masks = script.pred_preprocessing(SimpleNamespace(), pred, args)
    p2 = SimpleNamespace(prompt=_CLASS_PROMPT, negative_prompt="blurry")
    script._apply_inline_class_prompts(
        p2, pred, 0, len(masks), mode == "Merge and Invert"
    )
    script._apply_auto_class_guard(p2, args, pred, 0, len(masks))

    assert (p2.prompt, p2.negative_prompt) == (prompt, negative)


@pytest.mark.parametrize(
    ("spec", "kept", "warned"),
    [
        # The label drawn on the Detection preview keeps only that detection.
        ("#2", [list(_FACE_BOXES[1])], False),
        # So does a full-width comma typed with a Chinese or Japanese IME.
        ("1\uff0c3", [list(_FACE_BOXES[0]), list(_HAND_BOX)], False),
        # No readable number keeps every detection, and the console says so.
        ("x", [list(_FACE_BOXES[0]), list(_FACE_BOXES[1]), list(_HAND_BOX)], True),
        # Also for non-Latin text, echoed as ASCII so no console code page
        # can fail to print it.
        ("\u4e8c", [list(_FACE_BOXES[0]), list(_FACE_BOXES[1]), list(_HAND_BOX)], True),
        # Numbers that match no detection keep none; that warning is ASCII
        # too, also next to non-Latin text.
        ("5", [], False),
        ("\u4e8c 5", [], False),
    ],
)
def test_inpaint_indices_accept_preview_labels_and_warn_when_unreadable(
    spec, kept, warned, capsys
):
    from adetailer.args import ADetailerArgs

    script = _class_prompt_script()
    pred = _detections(["face", "face", "hand"])
    args = ADetailerArgs(ad_model="faces.pt", ad_inpaint_indices=spec)

    masks = script.pred_preprocessing(SimpleNamespace(), pred, args)

    assert pred.bboxes == kept
    assert len(masks) == len(kept)
    out = capsys.readouterr().out
    assert ("has no usable number" in out) is warned
    assert ("matched none" in out) is (kept == [])
    assert out.isascii()


def test_console_messages_are_ascii():
    # A console in a legacy code page (output redirected to a file on a
    # Japanese or Korean system) cannot print a dash those code pages lack:
    # the print raised, and the tab-restore line then stopped the ADetailer
    # panel from being built at every start.
    root = Path(__file__).resolve().parents[1]
    paths = [
        *sorted(root.glob("aaaaaa/*.py")),
        *sorted(root.glob("adetailer/*.py")),
        *sorted(root.glob("controlnet_ext/*.py")),
        *sorted(root.glob("scripts/*.py")),
        root / "install.py",
        root / "preload.py",
    ]
    found = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = ast.unparse(node.func)
            if func not in ("print", "sys.stdout.write", "sys.stderr.write"):
                continue
            found.extend(
                f"{path.name}:{node.lineno}"
                for sub in ast.walk(node)
                if isinstance(sub, ast.Constant)
                and isinstance(sub.value, str)
                and not sub.value.isascii()
            )
    assert "!adetailer.py" in {p.name for p in paths}
    assert found == []


def test_ui_console_lines_print_names_as_ascii():
    # repr() keeps a non-Latin detector or class name as it is. On a console
    # in a legacy code page the saved-tab line then failed on every Generate,
    # and the restored-tab and missing-detector lines, which run while the
    # panel is built, kept the ADetailer panel from being built at every start.
    ui_path = _SCRIPT_PATH.parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(ui_path.read_text(encoding="utf-8"))
    found = [
        sub.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "print"
        for sub in ast.walk(node)
        if isinstance(sub, ast.FormattedValue) and sub.conversion == ord("r")
    ]
    assert found == []


@pytest.mark.parametrize(
    ("model", "classes", "printed"),
    [
        ("face_yolov8n.pt", "顔", b"classes[include]='\\u9854'"),
        ("顔.pt", "face", b"detector='\\u9854.pt'"),
        # ASCII names print exactly as before.
        ("face_yolov8n.pt", "face", b"detector='face_yolov8n.pt', classes[include]='face'"),
    ],
)
def test_saved_tab_line_prints_on_a_legacy_code_page(monkeypatch, model, classes, printed):
    ui_path = _SCRIPT_PATH.parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(ui_path.read_text(encoding="utf-8"))
    nodes = [
        tree.body[0],  # from __future__ import annotations
        *(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "on_generate_click"
        ),
    ]
    saved = []
    namespace = {
        "ALL_ARGS": SimpleNamespace(
            attrs=(
                "ad_model", "ad_model_classes", "ad_model_classes_exclude",
                "ad_model_classes_excluded",
            )
        ),
        "save_tab_state": lambda mode, tab, state: saved.append(dict(state)) or True,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ui_path), "exec"), namespace)
    buffer = io.BytesIO()
    monkeypatch.setattr(
        sys, "stdout",
        io.TextIOWrapper(buffer, encoding="cp1252", errors="strict", write_through=True),
    )

    state = namespace["on_generate_click"](
        {}, model, classes, False, "", mode="txt2img", tab_index=1
    )

    assert (state["ad_model"], state["ad_model_classes"]) == (model, classes)
    assert [s["ad_model"] for s in saved] == [model]
    assert b"saved tab 2 (txt2img)" in buffer.getvalue()
    assert printed in buffer.getvalue()


@pytest.mark.parametrize("remember", [True, False])
def test_saved_tab_line_only_when_the_tab_was_written(
    tmp_path, monkeypatch, capsys, remember
):
    # With Settings -> ADetailer -> "Remember last-used settings between
    # restarts" off, every Generate still printed "saved tab N" for each tab
    # with a detector, although nothing was written to user_state.json.
    from adetailer import persistence

    state_file = tmp_path / "user_state.json"
    monkeypatch.setattr(persistence, "_STATE_FILE", state_file)
    modules = ModuleType("modules")
    modules.shared = ModuleType("modules.shared")
    modules.shared.opts = SimpleNamespace(data={"ad_remember_last_settings": remember})
    monkeypatch.setitem(sys.modules, "modules", modules)
    monkeypatch.setitem(sys.modules, "modules.shared", modules.shared)
    ui_path = _SCRIPT_PATH.parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(ui_path.read_text(encoding="utf-8"))
    nodes = [
        tree.body[0],  # from __future__ import annotations
        *(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "on_generate_click"
        ),
    ]
    namespace = {
        "ALL_ARGS": SimpleNamespace(
            attrs=(
                "ad_model", "ad_model_classes", "ad_model_classes_exclude",
                "ad_model_classes_excluded",
            )
        ),
        "save_tab_state": persistence.save_tab_state,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ui_path), "exec"), namespace)

    state = namespace["on_generate_click"](
        {}, "face_yolov8n.pt", "face", False, "", mode="txt2img", tab_index=0
    )

    assert (state["ad_model"], state["ad_model_classes"]) == ("face_yolov8n.pt", "face")
    assert state_file.exists() is remember
    assert ("saved tab 1 (txt2img)" in capsys.readouterr().out) is remember


def _start_up_class_filter(monkeypatch, saved):
    # Runs one_ui_group's own start-up code for the detector and its class
    # filter (the fallback for a detector that is gone, the class widgets'
    # starting values and the "tab N restored" / "fell back" line) on a
    # console in a legacy code page. Returns the console output and what the
    # class widgets start with.
    from functools import partial

    ui_path = _SCRIPT_PATH.parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(ui_path.read_text(encoding="utf-8"))
    group = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "one_ui_group"
    )

    def assigned(node, name):
        return isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == name

    def index(body, name):
        return next(i for i, node in enumerate(body) if assigned(node, name))

    start = index(group.body, "_saved_model")
    detection = next(
        node for node in group.body[start:]
        if isinstance(node, ast.With) and any(assigned(s, "_is_world_saved") for s in node.body)
    )
    rows = [s for s in detection.body if isinstance(s, ast.With)]
    row = next(r for r in rows if any(assigned(s, "_saved_exclude") for s in r.body))
    first = index(row.body, "_saved_exclude")
    last = next(
        i for i, s in enumerate(row.body)
        if isinstance(s, ast.If) and "_saved_model_raw" in ast.unparse(s.test)
    )
    body = [
        tree.body[0],  # from __future__ import annotations
        *(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_sv"),
        group.body[index(group.body, "saved")],
        group.body[index(group.body, "sv")],
        *group.body[start : start + 2],  # a detector that is gone falls back
        detection.body[index(detection.body, "_is_world_saved")],
        *row.body[first : last + 1],
    ]
    widgets = {
        ast.unparse(s.targets[0]): {k.arg: k.value for k in s.value.keywords}
        for r in rows
        for s in r.body
        if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]).startswith("w.ad_model")
    }
    namespace = {
        "partial": partial,
        "saved_tab_state": saved,
        "n": 0,
        "model_choices": ["animals.pt", "custom-world.pt", "None"],
        "webui_info": SimpleNamespace(model_mapping={"animals.pt": "animals.pt"}),
        "MEDIAPIPE_FACE_FEATURES_MODEL": "mediapipe_face_features",
        "get_model_class_names": lambda path: {"animals.pt": ["cat", "dog"]}.get(path, []),
    }
    buffer = io.BytesIO()
    monkeypatch.setattr(
        sys, "stdout",
        io.TextIOWrapper(buffer, encoding="cp1252", errors="strict", write_through=True),
    )
    exec(compile(ast.Module(body=body, type_ignores=[]), str(ui_path), "exec"), namespace)

    def value(name):
        node = widgets[f"w.{name}"]["value"]
        return eval(compile(ast.Expression(node), str(ui_path), "eval"), namespace)

    return buffer.getvalue(), {
        "classes": value("ad_model_classes"),
        "selection": value("ad_model_classes_dropdown"),
        "not": value("ad_model_classes_exclude"),
        "excluded": value("ad_model_classes_excluded"),
    }


@pytest.mark.parametrize(
    ("saved", "printed"),
    [
        # YOLO-World: the classes typed in its free-text box.
        (
            {"ad_model": "custom-world.pt", "ad_model_classes": "person,cat"},
            b"tab 1 restored - detector='custom-world.pt', classes[include]='person,cat'",
        ),
        # Saved in NOT mode by an older release: YOLO-World starts with NOT
        # mode off and detects its typed classes.
        (
            {
                "ad_model": "custom-world.pt", "ad_model_classes": "person",
                "ad_model_classes_exclude": True, "ad_model_classes_excluded": "hand",
            },
            b"detector='custom-world.pt', classes[include]='person'",
        ),
        ({"ad_model": "custom-world.pt", "ad_model_classes": "顔"},
         b"classes[include]='\\u9854'"),
        # NOT mode says that the classes are excluded.
        (
            {
                "ad_model": "animals.pt", "ad_model_classes_exclude": True,
                "ad_model_classes_excluded": "dog",
            },
            b"detector='animals.pt', classes[NOT/exclude]=['dog']",
        ),
        ({"ad_model": "animals.pt", "ad_model_classes": "cat"},
         b"detector='animals.pt', classes[include]=['cat']"),
        ({"ad_model": "animals.pt"}, b"detector='animals.pt', classes[include]=[]"),
        # A saved detector that is gone: its saved filter, marked the same way.
        (
            {
                "ad_model": "gone.pt", "ad_model_classes_exclude": True,
                "ad_model_classes_excluded": "dog",
            },
            b"fell back to 'animals.pt'. (saved classes[NOT/exclude]: ['dog'])",
        ),
        ({"ad_model": "gone.pt", "ad_model_classes": "cat"},
         b"(saved classes[include]: ['cat'])"),
    ],
)
def test_start_up_line_names_the_class_filter_each_tab_restored(monkeypatch, saved, printed):
    # A tab with a YOLO-World detector printed "classes=[]" although its typed
    # classes were restored, and a tab in NOT mode listed its excluded classes
    # as "classes=[...]", without saying that they are excluded.
    output, shown = _start_up_class_filter(monkeypatch, saved)

    assert printed in output
    line = output.decode("cp1252").strip()
    assert "\n" not in line  # one line for the tab
    if " restored - " in line:
        # The line names what the class widgets really start with.
        assert ("[NOT/exclude]" in line) is shown["not"]
        world = "-world" in saved["ad_model"]
        assert line.endswith(f"={(shown['classes'] if world else shown['selection'])!a}")
        if shown["not"]:
            assert ",".join(shown["selection"]) == shown["excluded"]


def test_class_skip_block_does_not_skip_the_inverted_background():
    script = _class_prompt_script()
    pred = _detections(["hand"])
    p2 = SimpleNamespace(
        prompt="detailed, [CLASS=hand] [SKIP] [/CLASS]", negative_prompt="blurry"
    )

    script._apply_inline_class_prompts(p2, pred, 0, 1, True)

    assert p2.prompt == "detailed"


@pytest.mark.parametrize(("seq_pass", "guarded"), [(False, True), (True, False)])
def test_class_prompt_line_suppresses_the_guard_only_where_it_applies(
    seq_pass, guarded
):
    # Class prompts are only applied by a sequential pass. Elsewhere a line for
    # the class must not switch the guard off with nothing in its place.
    from adetailer.args import ADetailerArgs

    script = _class_prompt_script()
    args = ADetailerArgs(
        ad_model="faces.pt", ad_class_guard=True,
        ad_class_prompts="face: detailed skin",
    )
    p2 = SimpleNamespace(prompt="detailed", negative_prompt="blurry")

    script._apply_auto_class_guard(
        p2, args, SimpleNamespace(class_names=["face"]), 0, 1, seq_pass
    )

    expected = ("face, detailed", "blurry, hand") if guarded else ("detailed", "blurry")
    assert (p2.prompt, p2.negative_prompt) == expected


@pytest.mark.parametrize(
    ("classes", "sequential", "flags"),
    [("", False, [False]), ("face,hand", True, [True, True])],
)
def test_the_guard_is_told_whether_the_pass_is_sequential(classes, sequential, flags):
    state = SimpleNamespace(
        interrupted=False, skipped=False, job_count=0,
        assign_current_image=lambda _image: None,
    )
    image = Image.new("RGB", (8, 8), "white")
    pred = SimpleNamespace(preview=image)
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
        ensure_pil_image=lambda im, _mode: im,
        process_images=lambda _p: SimpleNamespace(images=[image]),
        NansException=type("NansException", (Exception,), {}),
    )
    script = runtime.AfterDetailerScript()
    script.ultralytics_device = "cpu"
    script.get_i2i_p = lambda *_args: SimpleNamespace(
        init_images=[image], prompt="face", close=lambda: None
    )
    script.get_prompt = lambda *_args: (["face"], [""])
    script.get_ad_model = lambda _name: "model.pt"
    script.pred_preprocessing = lambda *_args: [image]
    script.save_image = lambda *_args, **_kwargs: None
    script.i2i_prompts_replace = lambda *_args: None
    script._apply_inline_class_prompts = lambda *_args: None
    script.fix_p2 = lambda *_args: None
    script.compare_prompt = lambda *_args, **_kwargs: None
    seen = []
    script._apply_auto_class_guard = lambda *call: seen.append(call[-1])

    class Args(SimpleNamespace):
        def copy(self, update):
            return Args(**{**vars(self), **update})

    args = Args(
        ad_classes_sequential=sequential, ad_model_classes=classes,
        ad_model_classes_exclude=False, ad_class_prompts="",
        ad_model="model.pt", ad_confidence=0.3, ad_use_bbox_mask=False,
        ad_detection_resolution=0, is_mediapipe=lambda: False,
    )

    script._postprocess_image_inner(
        SimpleNamespace(extra_generation_params={}), SimpleNamespace(image=image), args
    )

    assert seen == flags


def _ui_suffix():
    ui_path = _SCRIPT_PATH.parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(ui_path.read_text(encoding="utf-8"))
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"ordinal", "suffix"}
    ]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ui_path), "exec"), namespace)
    return namespace["suffix"]


def test_empty_class_prompts_stay_out_of_the_infotext():
    # AUTOMATIC1111's infotext parser reads the first character of every
    # value, so an empty one printed "Error parsing" on every paste.
    from adetailer.args import ADetailerArgs

    runtime = _load_script(
        methods={"extra_params"}, suffix=_ui_suffix(), __version__="test"
    )
    params = runtime.AfterDetailerScript().extra_params(
        [
            ADetailerArgs(ad_model="face_yolov8n.pt"),
            ADetailerArgs(
                ad_model="hand_yolov8n.pt", ad_class_prompts="face: smile | frown"
            ),
            ADetailerArgs(ad_model="face_yolov8n.pt", ad_class_prompts="  "),
        ]
    )

    assert "ADetailer class prompts" not in params
    assert params["ADetailer class prompts 2nd"] == "face: smile | frown"
    assert "ADetailer class prompts 3rd" not in params
    assert "" not in params.values()


@pytest.mark.parametrize(
    ("params", "skipped", "added"),
    [
        ({"ADetailer model": "face_yolov8n.pt"}, [], {"ADetailer class prompts": ""}),
        (
            {"ADetailer model 2nd": "hand_yolov8n.pt"},
            [],
            {"ADetailer class prompts 2nd": ""},
        ),
        (
            {"ADetailer model": "face_yolov8n.pt", "ADetailer class prompts": "face: smile"},
            [],
            {},
        ),
        ({"ADetailer model": "face_yolov8n.pt"}, ["ADetailer class prompts"], {}),
        ({"ADetailer model classes": "face"}, [], {}),
        ({"Steps": "20"}, [], {}),
    ],
)
def test_pasting_parameters_clears_class_prompts_they_leave_out(
    params, skipped, added
):
    runtime = _paste_callback(skipped)
    pasted = dict(params)

    runtime._clear_missing_class_prompts("", pasted)

    assert {
        k: v for k, v in pasted.items() if "class prompts" in k or k in params
    } == {**params, **added}


def _paste_callback(
    skipped=(), schedulers=("Automatic", "Uniform", "Karras", "Exponential"),
    model_mapping=None,
):
    return _load_script(
        functions={"_clear_missing_class_prompts"},
        assigns={
            "_INFOTEXT_MODEL_KEY",
            "_INFOTEXT_PASTE_DEFAULTS",
            "_INFOTEXT_PASTE_DEPENDENT_DEFAULTS",
        },
        # Empty: no installed-detector check, as for a WebUI without models.
        model_mapping=model_mapping or {},
        shared=SimpleNamespace(
            opts=SimpleNamespace(infotext_skip_pasting=list(skipped))
        ),
        all_samplers=[
            SimpleNamespace(name=n) for n in ("DPM++ 2M", "DPM++ 2M SDE", "Euler", "Euler a")
        ],
        schedulers=[SimpleNamespace(label=label) for label in schedulers],
    )


def _infotext(*tabs):
    """Parsed parameters of an image made with these tabs, as the WebUI
    pastes them: every value is text, empty class prompts are left out."""
    suffix = _ui_suffix()
    params = {}
    for n, args in enumerate(tabs):
        params.update(args.extra_params(suffix=suffix(n)))
    params = {
        k: str(v) for k, v in params.items()
        if not (k.startswith("ADetailer class prompts") and not str(v).strip())
    }
    params["ADetailer version"] = "test"
    return params


# Pasted by handlers of their own in aaaaaa/ui.py, not by their key.
_UI_PASTED = {
    "ad_model_classes", "ad_model_classes_exclude", "ad_model_classes_excluded",
    "ad_tab_enable", "ad_inpaint_indices",
}


def _host_paste(params, before, suffix=""):
    """AUTOMATIC1111's and Forge Neo's paste of the string-keyed fields: a
    missing key leaves the field as it is, text is converted to its type."""
    from adetailer.args import ALL_ARGS

    state = dict(before)
    for attr, name in ALL_ARGS:
        value = params.get(name + suffix)
        if attr in _UI_PASTED or value is None:
            continue
        kind = type(before[attr])
        try:
            if kind is bool and value == "False":
                state[attr] = False
            elif kind is int:
                state[attr] = float(value)
            else:
                state[attr] = kind(value)
        except (TypeError, ValueError):
            continue  # the WebUI leaves the field as it is
    return state


_CN_INPAINT = "control_v11p_sd15_inpaint [ebff9138]"


@pytest.mark.parametrize(
    "image",
    [
        {},
        # A separate sampler leaves out "Use same scheduler".
        {"ad_use_sampler": True, "ad_sampler": "Euler a"},
        # A ControlNet model leaves out the "None" module and default weights.
        {"ad_controlnet_model": _CN_INPAINT, "ad_controlnet_module": "None"},
    ],
)
def test_pasted_parameters_reproduce_the_image_tab(image):
    # The infotext leaves out every key at its default, and the WebUI keeps a
    # field whose key is missing: a stale prompt, separate steps, offset or
    # ControlNet setting of the tab used to stay and the next image differed.
    from adetailer.args import ADetailerArgs

    made = ADetailerArgs(ad_model="face_yolov8n.pt", **image)
    before = ADetailerArgs(
        ad_model="hand_yolov8n.pt", ad_prompt="detailed eyes",
        ad_negative_prompt="blurry", ad_prompt_append="smile",
        ad_negative_prompt_append="frown", ad_classes_sequential=True,
        ad_class_guard=True, ad_use_main_loras=True, ad_strip_loras=True,
        ad_detection_resolution=640, ad_mask_k=2, ad_mask_min_ratio=0.1,
        ad_mask_max_ratio=0.9, ad_x_offset=10, ad_y_offset=-5,
        ad_mask_merge_invert="Merge", ad_dynamic_denoise_power=1.5,
        ad_use_inpaint_width_height=True, ad_use_steps=True, ad_steps=50,
        ad_use_cfg_scale=True, ad_use_checkpoint=True, ad_checkpoint="other",
        ad_use_vae=True, ad_vae="vae", ad_use_text_encoder=True,
        ad_text_encoder="te", ad_use_sampler=True, ad_sampler="Euler",
        ad_scheduler="Karras", ad_use_noise_multiplier=True,
        ad_use_clip_skip=True, ad_restore_face=True,
        ad_controlnet_model="control_v11p_sd15_openpose [cab727d4]",
        ad_controlnet_module="inpaint_only+lama", ad_controlnet_weight=0.5,
        ad_controlnet_guidance_start=0.1, ad_controlnet_guidance_end=0.9,
    ).dict()
    params = _infotext(made)

    _paste_callback()._clear_missing_class_prompts("", params)
    after = ADetailerArgs(**_host_paste(params, before))

    assert after.extra_params() == made.extra_params()


def test_pasting_keeps_values_that_only_apply_with_their_toggle():
    from adetailer.args import ADetailerArgs

    params = _infotext(
        ADetailerArgs(ad_model="face_yolov8n.pt"),
        ADetailerArgs(
            ad_model="hand_yolov8n.pt", ad_prompt="hand detail", ad_use_steps=True,
            ad_steps=50,
        ),
    )
    image = dict(params)

    _paste_callback(["ADetailer x offset"])._clear_missing_class_prompts("", params)

    assert params["ADetailer prompt"] == ""
    assert params["ADetailer use separate steps"] == "False"
    assert params["ADetailer ControlNet model"] == "None"
    assert params["ADetailer x offset 2nd"] == "0"
    # Unused while their toggle is off: the tab keeps its own values.
    for name in ("ADetailer steps", "ADetailer sampler", "ADetailer checkpoint",
                 "ADetailer scheduler", "ADetailer ControlNet module"):
        assert name not in params
    assert "ADetailer x offset" not in params  # "Disregard fields ..."
    assert {k: params[k] for k in image} == image  # pasted values stay
    assert not any(k.endswith(" 3rd") for k in params)


def test_pasting_parameters_without_a_detector_adds_nothing():
    params = {"Steps": "20", "ADetailer version": "test"}

    _paste_callback()._clear_missing_class_prompts("", params)

    assert params == {"Steps": "20", "ADetailer version": "test"}


def test_paste_defaults_cover_every_key_the_infotext_leaves_out():
    from adetailer.args import ALL_ARGS, ADetailerArgs

    runtime = _paste_callback()
    defaults = {
        **runtime._INFOTEXT_PASTE_DEFAULTS,
        **{
            name: value
            for extra in runtime._INFOTEXT_PASTE_DEPENDENT_DEFAULTS.values()
            for name, value in extra.items()
        },
    }
    attrs = {name: attr for attr, name in ALL_ARGS}
    args = ADetailerArgs()
    for name, value in defaults.items():
        assert str(getattr(args, attrs[name])) == value, name

    # Used only while their toggle (itself filled) is on.
    dependent = {
        "ADetailer class guard weight", "ADetailer method to decide top k masks",
        "ADetailer inpaint padding", "ADetailer inpaint width",
        "ADetailer inpaint height", "ADetailer steps", "ADetailer CFG scale",
        "ADetailer checkpoint", "ADetailer VAE", "ADetailer text encoder",
        "ADetailer sampler", "ADetailer noise multiplier", "ADetailer CLIP skip",
    }
    ui_pasted = {dict(ALL_ARGS)[attr] for attr in _UI_PASTED}
    for made in (
        ADetailerArgs(ad_model="face_yolov8n.pt"),
        ADetailerArgs(ad_model="face_yolov8n.pt", ad_inpaint_only_masked=False),
        ADetailerArgs(ad_model="face_yolov8n.pt", ad_use_sampler=True),
        ADetailerArgs(ad_model="face_yolov8n.pt", ad_controlnet_model=_CN_INPAINT),
    ):
        left_out = set(ALL_ARGS.names) - set(made.extra_params())
        assert left_out <= set(defaults) | dependent | ui_pasted


@pytest.mark.parametrize("suffix", ["", " 2nd"])
@pytest.mark.parametrize(
    ("choices", "expected"),
    [
        # Forge / Forge Neo list "None" first: the image's preprocessor.
        (["None", "inpaint_global_harmonious", "inpaint_only", "inpaint_only+lama"],
         "None"),
        # AUTOMATIC1111 has no "None": the model's default, as when generating.
        (["inpaint_global_harmonious", "inpaint_only", "inpaint_only+lama"],
         "inpaint_global_harmonious"),
    ],
)
def test_pasted_controlnet_module_none_replaces_a_stale_module(
    suffix, choices, expected
):
    # The infotext leaves out the "None" preprocessor, and the model's change
    # handler kept the tab's old one because it fits the model.
    from adetailer.args import ADetailerArgs

    made = ADetailerArgs(
        ad_model="face_yolov8n.pt", ad_controlnet_model=_CN_INPAINT,
        ad_controlnet_module="None",
    )
    params = {k: str(v) for k, v in made.extra_params(suffix=suffix).items()}
    assert "ADetailer ControlNet module" + suffix not in params

    _paste_callback()._clear_missing_class_prompts("", params)
    # The module field keeps its value when its key is missing.
    module = params.get("ADetailer ControlNet module" + suffix, "inpaint_only+lama")

    ui_path = _SCRIPT_PATH.parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(ui_path.read_text(encoding="utf-8"))
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "on_cn_model_update"
    ]
    namespace = {
        "gr": SimpleNamespace(update=lambda **kwargs: kwargs),
        "cn_module_choices": {"inpaint": choices},
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ui_path), "exec"), namespace)

    assert namespace["on_cn_model_update"](_CN_INPAINT, module)["value"] == expected


@pytest.mark.parametrize(
    ("params", "skipped", "enabled"),
    [
        ({"ADetailer model": "face_yolov8n.pt"}, [], "True"),
        ({"ADetailer model 2nd": "hand_yolov8n.pt"}, [], "True"),
        ({"ADetailer model": "face_yolov8n.pt"}, ["ADetailer enable"], None),
        ({"ADetailer model": "None"}, [], None),
        ({"ADetailer model classes": "face"}, [], None),
        ({"Steps": "20"}, [], None),
        ({"ADetailer model": "face_yolov8n.pt", "ADetailer enable": "False"}, [],
         "False"),
    ],
)
def test_pasting_parameters_with_a_detector_switches_adetailer_on(
    params, skipped, enabled
):
    # "ADetailer enable" is never written, so PNG Info and Send to switched
    # the image's tabs on while ADetailer itself stayed off.
    pasted = dict(params)

    _paste_callback(skipped)._clear_missing_class_prompts("", pasted)

    assert pasted.get("ADetailer enable") == enabled


def test_pasting_splits_an_old_combined_sampler_name():
    # An image from WebUI < 1.9, or from the API with the sampler left at its
    # default, names a sampler such as "DPM++ 2M Karras" that is no choice of
    # the dropdown: Gradio 4 showed it blank, and after a Generate and a
    # restart the tab ran the first sampler with the main pass's scheduler.
    from adetailer.args import ADetailerArgs

    params = _infotext(
        ADetailerArgs(
            ad_model="face_yolov8n.pt", ad_use_sampler=True,
            ad_sampler="DPM++ 2M SDE Karras",
        ),
        ADetailerArgs(ad_model="hand_yolov8n.pt", ad_use_sampler=True),
    )
    assert params["ADetailer sampler 2nd"] == "DPM++ 2M Karras"

    _paste_callback()._clear_missing_class_prompts("", params)

    assert (params["ADetailer sampler"], params["ADetailer scheduler"]) == (
        "DPM++ 2M SDE", "Karras"
    )
    assert (params["ADetailer sampler 2nd"], params["ADetailer scheduler 2nd"]) == (
        "DPM++ 2M", "Karras"
    )
    assert params["ADetailer enable"] == "True"


@pytest.mark.parametrize(
    ("pasted", "skipped", "schedulers", "expected"),
    [
        # The name's scheduler wins, as it does when the WebUI runs it.
        (("True", "DPM++ 2M Karras", "Exponential"), [], None, ("DPM++ 2M", "Karras")),
        # Left as they are:
        (("True", "Euler a", "Karras"), [], None, ("Euler a", "Karras")),
        (("True", "Use same sampler", None), [], None,
         ("Use same sampler", "Use same scheduler")),
        (("False", "DPM++ 2M Karras", None), [], None,
         ("DPM++ 2M Karras", "Use same scheduler")),
        (("True", "Foo Karras", None), [], None, ("Foo Karras", "Use same scheduler")),
        # Unlike Load, a sampler this WebUI does not have and a scheduler's
        # other names are kept too (known issues of beta 2).
        (("True", "Res Multistep", "Karras"), [], None, ("Res Multistep", "Karras")),
        (("True", "DPM++ 2M karras", None), [], None,
         ("DPM++ 2M karras", "Use same scheduler")),
        # Splitting without the scheduler would lose Karras.
        (("True", "DPM++ 2M Karras", None), ["ADetailer scheduler"], None,
         ("DPM++ 2M Karras", None)),
        # WebUI < 1.9 has no schedulers: the combined name is a sampler there.
        (("True", "DPM++ 2M Karras", None), [], (), ("DPM++ 2M Karras", "Use same scheduler")),
    ],
)
def test_pasting_leaves_other_sampler_names_alone(pasted, skipped, schedulers, expected):
    use, sampler, scheduler = pasted
    params = {
        "ADetailer model": "face_yolov8n.pt",
        "ADetailer use separate sampler": use,
        "ADetailer sampler": sampler,
    }
    if scheduler is not None:
        params["ADetailer scheduler"] = scheduler
    callback = (
        _paste_callback(skipped)
        if schedulers is None
        else _paste_callback(skipped, schedulers)
    )

    callback._clear_missing_class_prompts("", params)

    assert (params["ADetailer sampler"], params.get("ADetailer scheduler")) == expected
    assert params["ADetailer enable"] == "True"


def test_class_prompts_paste_callback_is_registered():
    tree = ast.parse(_SCRIPT_PATH.read_text(encoding="utf-8"))
    registered = [
        node.args[0].id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") == "on_infotext_pasted"
    ]
    assert registered == ["_clear_missing_class_prompts"]


def test_negative_denoise_power_on_a_full_frame_region_keeps_the_strength():
    from adetailer.opts import dynamic_denoise_strength

    runtime = _load_script(
        methods={"get_dynamic_denoise_strength"},
        opts=SimpleNamespace(data={"ad_dynamic_denoise_power": -2.0}),
        dynamic_denoise_strength=dynamic_denoise_strength,
    )
    strength = runtime.AfterDetailerScript.get_dynamic_denoise_strength
    per_tab_unset = SimpleNamespace(ad_dynamic_denoise_power=0.0)

    assert strength(0.4, [0.0, 0.0, 800.0, 600.0], (800, 600), per_tab_unset) == 0.4
    # A smaller region is unchanged: a negative power still raises the strength.
    assert strength(0.4, [0, 0, 400, 300], (800, 600), per_tab_unset) == pytest.approx(
        0.4 / 0.75**2
    )


@pytest.mark.parametrize(
    ("mode", "scale", "match", "size"),
    [
        ("Merge and Invert", 1.5, "Off", (1248, 1824)),
        ("Merge and Invert", None, "Free", (832, 1216)),
        # Unchanged: the other modes size the canvas from the detection box.
        ("None", 1.5, "Off", (224, 224)),
        ("Merge", None, "Free", (1216, 360)),
    ],
)
def test_merge_and_invert_sizes_the_inpaint_from_the_inverted_mask(
    mode, scale, match, size
):
    from adetailer.args import ADetailerArgs, InpaintBBoxMatchMode
    from adetailer.opts import dynamic_denoise_strength, optimal_crop_size

    runtime = _load_script(
        methods={
            "fix_p2", "get_dynamic_denoise_strength", "get_optimal_crop_image_size",
            "get_seed", "get_each_tab_seed",
        },
        opts=SimpleNamespace(
            data={"ad_match_inpaint_bbox_size": match, "ad_dynamic_denoise_power": 2}
        ),
        shared=SimpleNamespace(opts=SimpleNamespace(data={}), sd_model=None),
        get_i=lambda _p: 0,
        InpaintBBoxMatchMode=InpaintBBoxMatchMode,
        dynamic_denoise_strength=dynamic_denoise_strength,
        optimal_crop_size=optimal_crop_size,
    )
    image = Image.new("RGB", (832, 1216))
    faces = [(100, 100, 250, 250), (500, 120, 640, 260)]
    if mode == "Merge and Invert":
        mask = Image.new("L", image.size, 255)
        for box in faces:
            mask.paste(0, box)
    else:
        mask = Image.new("L", image.size, 0)
        mask.paste(255, faces[0])
    bbox = [100, 100, 250, 250] if mode == "None" else [100, 100, 640, 260]
    p2 = SimpleNamespace(width=832, height=1216, denoising_strength=0.4, image_mask=mask)
    args = ADetailerArgs(
        ad_model="face_yolov8n.pt",
        ad_mask_merge_invert=mode,
        ad_use_resolution_scale=scale is not None,
        ad_resolution_scale=scale or 1.5,
    )
    outer = SimpleNamespace(seed=1, subseed=1, all_seeds=[1], all_subseeds=[1])

    runtime.AfterDetailerScript().fix_p2(
        outer, p2, SimpleNamespace(image=image), args, SimpleNamespace(bboxes=[bbox]), 0
    )

    assert (p2.width, p2.height) == size
    # Dynamic denoise still follows the detections: a full-frame box would
    # give 0 and leave the background untouched.
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (832 * 1216)
    assert p2.denoising_strength == pytest.approx(0.4 * (1 - area) ** 2)


def _reset_button_js():
    tree = ast.parse(_SCRIPT_PATH.read_text(encoding="utf-8"))
    factory = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_make_reset_settings_button"
    )
    clicks = [
        node for node in ast.walk(factory)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "click"
    ]
    assert len(clicks) == 1
    keywords = {kw.arg: kw.value for kw in clicks[0].keywords}
    return ast.literal_eval(keywords["_js"])


def test_cancelling_the_settings_reset_does_not_return_a_value():
    # Gradio 3.41 and 4.40 send the request with whatever the JS callback
    # returns, so returning on Cancel still reset every option. Only a
    # rejected promise stops the request.
    js = _reset_button_js()
    cancel = js[js.index("if (!confirm(") : js.index("setTimeout")]

    assert "return" not in cancel
    assert "throw" in cancel
    # OK still reaches Python and reloads the page.
    assert js.rstrip().endswith("return [];}")
    assert "location.reload()" in js


def test_readme_sequential_note_matches_mediapipe_face_features():
    from adetailer.args import ADetailerArgs
    from adetailer.classes import MEDIAPIPE_FACE_FEATURES_MODEL, parse_csv

    runtime = _load_script(methods={"_will_run_sequential"}, parse_csv=parse_csv)
    will_run = runtime.AfterDetailerScript._will_run_sequential
    two_parts = {"ad_classes_sequential": True, "ad_model_classes": "eyes,mouth"}

    assert will_run(ADetailerArgs(ad_model=MEDIAPIPE_FACE_FEATURES_MODEL, **two_parts))
    assert not will_run(ADetailerArgs(ad_model="mediapipe_face_full", **two_parts))
    readme = (_SCRIPT_PATH.parents[1] / "README.md").read_text(encoding="utf-8")
    notes = [
        line for line in readme.splitlines()
        if "Sequential mode is **ignored** for MediaPipe" in line
    ]
    assert notes
    assert all(MEDIAPIPE_FACE_FEATURES_MODEL in line for line in notes)


# Final debugging pass, round 2: generation pipeline, prompts and docs.


@pytest.mark.parametrize(
    ("prompt", "negative", "expected"),
    [
        # No eyes in the region: the eyes blocks do not apply to it.
        ("detailed, [CLASS=eyes] [SKIP] [/CLASS]", "blurry", ("detailed", "blurry")),
        (
            "detailed, [CLASS=eyes]blue eyes[/CLASS]",
            "blurry, [CLASS=eyes]red eyes[/CLASS]",
            ("detailed", "blurry"),
        ),
        # The face in the region is not skipped for the hand's sake...
        ("detailed, [CLASS=hand] [SKIP] [/CLASS]", "blurry", ("detailed", "blurry")),
        # ...but the region is skipped when every class it holds is.
        ("detailed, [CLASS=face,hand] [SKIP] [/CLASS]", "blurry", ("[SKIP]", "blurry")),
        (
            "detailed, [CLASS=face] [SKIP] [/CLASS], [CLASS=hand] [SKIP] [/CLASS]",
            "blurry",
            ("[SKIP]", "blurry"),
        ),
    ],
)
def test_merged_mixed_region_follows_only_the_classes_it_holds(
    prompt, negative, expected
):
    from adetailer.args import ADetailerArgs

    script = _class_prompt_script()
    pred = _detections(["face", "hand"])
    args = ADetailerArgs(ad_model="faces.pt", ad_mask_merge_invert="Merge")

    masks = script.pred_preprocessing(SimpleNamespace(), pred, args)
    p2 = SimpleNamespace(prompt=prompt, negative_prompt=negative)
    script._apply_inline_class_prompts(p2, pred, 0, len(masks), False)

    assert len(masks) == 1
    assert (p2.prompt, p2.negative_prompt) == expected


def test_class_blocks_are_dropped_on_a_classless_inverted_background():
    # A class-less detector (the MediaPipe face models) reports no classes,
    # but its Merge and Invert region is still the background.
    script = _class_prompt_script()
    pred = _detections(["face"])
    pred.class_names = []
    p2 = SimpleNamespace(
        prompt="scenery, [CLASS=face] [SKIP] [/CLASS]",
        negative_prompt="blurry, [CLASS=face]ugly[/CLASS]",
    )

    script._apply_inline_class_prompts(p2, pred, 0, 1, True)

    assert (p2.prompt, p2.negative_prompt) == ("scenery", "blurry")
    # Unchanged elsewhere: a class-less region keeps every block.
    p2 = SimpleNamespace(prompt="scenery, [CLASS=face] [SKIP] [/CLASS]", negative_prompt="")
    script._apply_inline_class_prompts(p2, pred, 0, 1, False)
    assert p2.prompt == "[SKIP]"


class _A1111I2I:
    """The host's img2img processing class; like AUTOMATIC1111's dataclass it
    has no Distilled CFG Scale field and refuses an unknown keyword."""

    def __init__(self, **kwargs):
        if "distilled_cfg_scale" in kwargs:
            msg = "unexpected keyword argument 'distilled_cfg_scale'"
            raise TypeError(msg)
        self.__dict__.update(kwargs)


class _ForgeI2I(_A1111I2I):
    distilled_cfg_scale = 3.5  # the Forge / Forge Neo dataclass default


@pytest.mark.parametrize(
    ("host", "user_value", "expected"),
    [(_ForgeI2I, 2.0, 2.0), (_ForgeI2I, 9.0, 9.0), (_A1111I2I, None, None)],
)
def test_detailer_pass_keeps_the_distilled_cfg_scale(host, user_value, expected):
    from adetailer.args import ADetailerArgs

    runtime = _load_script(
        methods={"get_i2i_p"},
        StableDiffusionProcessingImg2Img=host,
        schedulers=None,
        controlnet_type="forge",
        copy_extra_params=dict,
    )
    script = runtime.AfterDetailerScript()
    script.get_seed = lambda _p: (1, 1)
    script.get_width_height = lambda *_args: (512, 512)
    script.get_steps = lambda *_args: 20
    script.get_cfg_scale = lambda *_args: 1.0
    script.get_initial_noise_multiplier = lambda *_args: None
    script.get_sampler = lambda *_args: "Euler"
    script.get_override_settings = lambda *_args: {}
    script.script_filter = lambda *_args: (None, [])
    p = SimpleNamespace(
        sd_model=None, outpath_samples="", outpath_grids="", styles=[],
        subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0,
        tiling=False, extra_generation_params={},
    )
    if user_value is not None:
        p.distilled_cfg_scale = user_value

    i2i = script.get_i2i_p(p, ADetailerArgs(ad_model="face_yolov8n.pt"), None)

    assert getattr(i2i, "distilled_cfg_scale", None) == expected


@pytest.mark.parametrize(
    ("source", "canvas"),
    [
        ((96, 64), (64, 40)),  # capped standalone canvas: not a uniform scale
        ((48, 48), (32, 32)),  # Hires fix x1.5: a scale that is not a 0.25 step
        ((64, 64), (64, 64)),  # the canvas is the image: nothing changes
    ],
)
def test_whole_picture_inpaint_keeps_the_next_mask_on_the_image_grid(source, canvas):
    # With "Inpaint only masked" off the host returns each region at the
    # canvas size. Forge Neo refuses a mask whose scale to the image is not
    # uniform and a multiple of 0.25, which failed the second region.
    state = SimpleNamespace(
        interrupted=False, skipped=False, job_count=0,
        assign_current_image=lambda _image: None,
    )
    image = Image.new("RGB", source, "white")
    masks = [Image.new("L", source, 255), Image.new("L", source, 255)]
    seen = []

    def process_images(p2):
        init, mask = p2.init_images[0], p2.image_mask
        seen.append((init.size, mask.size, any(mask is m for m in masks)))
        sx, sy = (a / b for a, b in zip(init.size, mask.size))
        assert sx == sy, "Only uniform Scaling is supported"
        assert sx % 0.25 == 0, "Inpaint only supports Scale divisible by 0.25"
        return SimpleNamespace(images=[init.resize((p2.width, p2.height))])

    run, pp, _before = _detailer(
        state, process_images, n_masks=2, Image=Image, ordinal=str
    )
    pp.image = image
    i2i = SimpleNamespace(
        init_images=[image], prompt="face", negative_prompt="",
        width=canvas[0], height=canvas[1], close=lambda: None,
    )
    _script_of(run).get_i2i_p = lambda *_args: i2i
    _script_of(run).pred_preprocessing = lambda *_args: masks

    assert run() is True
    assert [init for init, _mask, _same in seen] == [source, canvas]
    assert all(init == mask for init, mask, _same in seen)
    assert pp.image.size == canvas
    if source == canvas:
        assert all(same for _init, _mask, same in seen)


def _script_of(run):
    """The script instance a `_detailer` runner calls."""
    cells = dict(zip(run.__code__.co_freevars, run.__closure__))
    return cells["script"].cell_contents


_INLINE_PROMPT_METHODS = {
    "_get_prompt", "prompt_blank_replacement", "get_prompt",
    "i2i_prompts_replace", "_apply_inline_class_prompts",
}
_INLINE_PROMPT_FUNCTIONS = {
    "_resolve_inline_class_prompt", "_extract_lora_tags", "_extract_lora_triggers",
    "_merge_lora_tags", "_append_lora_triggers", "_strip_lora_tags",
    "_without_style_loras",
}
_INLINE_PROMPT_ASSIGNS = {"_INLINE_CLASS_RE", "_LORA_TAG_RE", "_LORA_TRIGGER_RE"}


@pytest.mark.parametrize(
    ("prompt", "negative", "append", "face", "hand"),
    [
        # The documented way to skip only the hands.
        (
            "[CLASS=hand] [SKIP] [/CLASS]", "", "",
            ("portrait, smiling", "blurry"), ("[SKIP]", "blurry"),
        ),
        # Only the region no block matches gets the main prompt (with the
        # append text); a matching block is kept alone, as before.
        (
            "[CLASS=hand]five fingers[/CLASS]", "[CLASS=hand]extra fingers[/CLASS]",
            "detailed",
            ("portrait, smiling, detailed", "blurry"),
            ("five fingers, detailed", "extra fingers"),
        ),
        (
            "[CLASS=face] detailed face [/CLASS] [CLASS=hand] detailed hand [/CLASS]",
            "[CLASS=face] ugly face [/CLASS] [CLASS=hand] extra fingers [/CLASS]", "",
            ("detailed face", "ugly face"), ("detailed hand", "extra fingers"),
        ),
        # [PROMPT] inside another class's block still leaves the hands without
        # text of their own.
        (
            "[CLASS=face] [PROMPT], smile [/CLASS]",
            "[CLASS=face] [PROMPT], bad teeth [/CLASS]", "",
            ("portrait, smiling, smile", "blurry, bad teeth"),
            ("portrait, smiling", "blurry"),
        ),
        (
            "[CLASS=hand] [SKIP] [/CLASS]", "", "masterpiece",
            ("portrait, smiling, masterpiece", "blurry"), ("[SKIP]", "blurry"),
        ),
        # Unchanged: text outside the blocks, and a prompt without blocks.
        (
            "detailed, [CLASS=hand] [SKIP] [/CLASS]", "", "",
            ("detailed", "blurry"), ("[SKIP]", "blurry"),
        ),
        (
            "[PROMPT], [CLASS=hand] five fingers [/CLASS]", "", "",
            ("portrait, smiling", "blurry"), ("portrait, smiling, five fingers", "blurry"),
        ),
        ("detailed face", "", "", ("detailed face", "blurry"), ("detailed face", "blurry")),
    ],
)
def test_a_prompt_of_only_class_blocks_keeps_the_main_prompt(
    prompt, negative, append, face, hand
):
    # A blank ADetailer prompt means the main prompt. A region that the
    # [CLASS=...] blocks leave with no text of its own used to be inpainted
    # with an empty prompt; a region with a matching block keeps only it.
    from adetailer.args import ADetailerArgs

    runtime = _load_script(
        methods=_INLINE_PROMPT_METHODS,
        functions=_INLINE_PROMPT_FUNCTIONS,
        assigns=_INLINE_PROMPT_ASSIGNS,
        get_i=lambda _p: 0,
    )
    script = runtime.AfterDetailerScript()
    p = SimpleNamespace(
        prompt="portrait, smiling", all_prompts=["portrait, smiling"],
        negative_prompt="blurry", all_negative_prompts=["blurry"],
    )
    args = ADetailerArgs(
        ad_model="faces.pt", ad_prompt=prompt, ad_negative_prompt=negative,
        ad_prompt_append=append,
    )
    prompts, negatives = script.get_prompt(p, args)
    pred = SimpleNamespace(class_names=["face", "hand"])

    regions = []
    for j in range(2):
        p2 = SimpleNamespace()
        script.i2i_prompts_replace(p2, prompts, negatives, j)
        script._apply_inline_class_prompts(p2, pred, j, 2, False, p, args)
        regions.append((p2.prompt, p2.negative_prompt))

    assert regions == [face, hand]


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        # A LoRA the tab prompt names at another weight keeps that weight.
        ("detailed face <lora:detail:0.3>", "detailed face <lora:detail:0.3> <lora:style:0.8>"),
        ("detailed face <lora:detail:1.0>", "detailed face <lora:detail:1.0> <lora:style:0.8>"),
        ("detailed face <lyco:detail:0.3>", "detailed face <lyco:detail:0.3> <lora:style:0.8>"),
        # Unchanged: another name (the host's lookup is case-sensitive), a
        # prompt without LoRAs and a blank prompt.
        (
            "detailed face <lora:Detail:0.3>",
            "detailed face <lora:Detail:0.3> <lora:detail:1> <lora:style:0.8>",
        ),
        ("detailed face,", "detailed face <lora:detail:1> <lora:style:0.8>"),
        ("", "portrait <lora:detail:1> <lora:style:0.8>"),
    ],
)
def test_main_prompt_loras_keep_the_weight_the_tab_prompt_gives(prompt, expected):
    # The main prompt's <lora:detail:1> was added next to the tab's
    # <lora:detail:0.3>, so the host applied that LoRA twice.
    from adetailer.args import ADetailerArgs

    runtime = _load_script(
        methods=_INLINE_PROMPT_METHODS,
        functions=_INLINE_PROMPT_FUNCTIONS,
        assigns=_INLINE_PROMPT_ASSIGNS,
        get_i=lambda _p: 0,
    )
    script = runtime.AfterDetailerScript()
    main = "portrait <lora:detail:1> <lora:style:0.8>"
    p = SimpleNamespace(
        prompt=main, all_prompts=[main], negative_prompt="",
        all_negative_prompts=[""],
    )
    args = ADetailerArgs(ad_model="faces.pt", ad_prompt=prompt, ad_use_main_loras=True)

    prompts, _negatives = script.get_prompt(p, args)

    assert prompts == [expected]


def test_a_merged_region_whose_only_block_is_skipped_keeps_the_main_prompt():
    # A merged face and hand is not skipped by the hands' [SKIP] block, and
    # the blocks leave it no text of its own.
    from adetailer.args import ADetailerArgs

    runtime = _load_script(
        methods=_INLINE_PROMPT_METHODS,
        functions=_INLINE_PROMPT_FUNCTIONS,
        assigns=_INLINE_PROMPT_ASSIGNS,
        get_i=lambda _p: 0,
    )
    script = runtime.AfterDetailerScript()
    p = SimpleNamespace(
        prompt="portrait, smiling", all_prompts=["portrait, smiling"],
        negative_prompt="blurry", all_negative_prompts=["blurry"],
    )
    args = ADetailerArgs(ad_model="faces.pt", ad_prompt="[CLASS=hand] [SKIP] [/CLASS]")
    prompts, negatives = script.get_prompt(p, args)
    pred = SimpleNamespace(class_names=[""], group_classes=[["face", "hand"]])
    p2 = SimpleNamespace()
    script.i2i_prompts_replace(p2, prompts, negatives, 0)

    script._apply_inline_class_prompts(p2, pred, 0, 1, False, p, args)

    assert (p2.prompt, p2.negative_prompt) == ("portrait, smiling", "blurry")


@pytest.mark.parametrize(
    ("scale", "match", "only_masked", "size"),
    [
        (1.5, "Off", False, (1024, 1536)),
        (None, "Free", False, (1024, 1536)),
        (None, "Strict (SDXL only)", False, (1024, 1536)),
        # Unchanged: with Inpaint only masked the canvas follows the box.
        (1.5, "Off", True, (240, 296)),
    ],
)
def test_whole_picture_inpaint_keeps_the_image_size(scale, match, only_masked, size):
    # The host resizes the whole image to the canvas in whole-picture mode:
    # a canvas sized from the box made the final image a small thumbnail or
    # changed its aspect ratio.
    from adetailer.args import ADetailerArgs, InpaintBBoxMatchMode
    from adetailer.opts import dynamic_denoise_strength, optimal_crop_size

    runtime = _load_script(
        methods={
            "fix_p2", "get_dynamic_denoise_strength", "get_optimal_crop_image_size",
            "get_seed", "get_each_tab_seed",
        },
        opts=SimpleNamespace(data={"ad_match_inpaint_bbox_size": match}),
        shared=SimpleNamespace(
            opts=SimpleNamespace(data={}), sd_model=SimpleNamespace(is_sdxl=True)
        ),
        get_i=lambda _p: 0,
        InpaintBBoxMatchMode=InpaintBBoxMatchMode,
        dynamic_denoise_strength=dynamic_denoise_strength,
        optimal_crop_size=optimal_crop_size,
    )
    p2 = SimpleNamespace(width=1024, height=1536, denoising_strength=0.4, image_mask=None)
    args = ADetailerArgs(
        ad_model="face_yolov8n.pt",
        ad_inpaint_only_masked=only_masked,
        ad_use_resolution_scale=scale is not None,
        ad_resolution_scale=scale or 1.5,
    )
    outer = SimpleNamespace(seed=1, subseed=1, all_seeds=[1], all_subseeds=[1])

    runtime.AfterDetailerScript().fix_p2(
        outer, p2, SimpleNamespace(image=Image.new("RGB", (1024, 1536))), args,
        SimpleNamespace(bboxes=[[400, 300, 560, 500]]), 0,
    )

    assert (p2.width, p2.height) == size


def test_readme_says_how_scale_inpaint_to_bbox_rounds_its_sizes():
    # The README said that the sizes are rounded down to a multiple of 8, but
    # fix_p2 first rounds the box's side times the scale to the nearest pixel:
    # a 114-pixel side at scale 1.4 (159.6 pixels) gives 160, not 152.
    from adetailer.args import ADetailerArgs, InpaintBBoxMatchMode
    from adetailer.opts import dynamic_denoise_strength, optimal_crop_size

    runtime = _load_script(
        methods={
            "fix_p2", "get_dynamic_denoise_strength", "get_optimal_crop_image_size",
            "get_seed", "get_each_tab_seed",
        },
        opts=SimpleNamespace(data={}),
        shared=SimpleNamespace(opts=SimpleNamespace(data={})),
        get_i=lambda _p: 0,
        InpaintBBoxMatchMode=InpaintBBoxMatchMode,
        dynamic_denoise_strength=dynamic_denoise_strength,
        optimal_crop_size=optimal_crop_size,
    )
    outer = SimpleNamespace(seed=1, subseed=1, all_seeds=[1], all_subseeds=[1])

    def scaled(side, scale):
        p2 = SimpleNamespace(width=512, height=512, denoising_strength=0.4, image_mask=None)
        args = ADetailerArgs(
            ad_model="face_yolov8n.pt", ad_use_resolution_scale=True,
            ad_resolution_scale=scale,
        )
        runtime.AfterDetailerScript().fix_p2(
            outer, p2, SimpleNamespace(image=Image.new("RGB", (1024, 1024))), args,
            SimpleNamespace(bboxes=[[10, 10, 10 + side, 10 + side]]), 0,
        )
        return p2.width

    section = _readme_section("### Scale inpaint to bbox", "### Dynamic denoise by area")
    rule = next(
        flat for flat in map(_flat_doc_text, section.split("\n- "))
        if "multiple of 8" in flat
    )
    assert "nearest pixel, then down to a multiple of 8" in rule, rule
    assert "at least 64" in rule
    assert scaled(40, 1.0) == 64
    # The README's example is what the code gives, and shows the rounding
    # to the nearest pixel: rounded down at once, its side would be 152.
    side, scale, exact, result = re.search(
        r"(\d+)-pixel side at scale ([\d.]+) \(([\d.]+) pixels\) gives (\d+)", rule
    ).groups()
    assert float(exact) == pytest.approx(int(side) * float(scale))
    assert scaled(int(side), float(scale)) == int(result)
    assert int(float(exact)) // 8 * 8 != int(result)
    assert scaled(101.3, 1.5) == 152  # a fractional box: 151.95 pixels


@pytest.mark.parametrize(
    ("detected", "expected"),
    [
        # The face holds the other parts: none of them is negated there.
        ("face", ("face, detailed", "blurry")),
        # A part never gets "face", which surrounds it, in its negative.
        ("eyes", ("eyes, detailed", "blurry, mouth, nose, eyebrows")),
        ("mouth", ("mouth, detailed", "blurry, eyes, nose, eyebrows")),
    ],
)
def test_class_guard_of_face_features_never_negates_the_face_or_its_parts(
    detected, expected
):
    from adetailer.args import ADetailerArgs
    from adetailer.classes import (
        MEDIAPIPE_FACE_FEATURES_MODEL,
        _has_token,
        build_class_guard,
        get_model_class_names,
    )

    runtime = _load_script(
        methods={"_apply_auto_class_guard"},
        functions={"_parse_class_prompts", "_class_prompt_for"},
        get_model_class_names=get_model_class_names,
        build_class_guard=build_class_guard,
        _has_token=_has_token,
    )
    script = runtime.AfterDetailerScript()
    script.get_ad_model = lambda name: name
    args = ADetailerArgs(ad_model=MEDIAPIPE_FACE_FEATURES_MODEL, ad_class_guard=True)
    p2 = SimpleNamespace(prompt="detailed", negative_prompt="blurry")

    script._apply_auto_class_guard(
        p2, args, SimpleNamespace(class_names=[detected]), 0, 1
    )

    assert (p2.prompt, p2.negative_prompt) == expected
    assert len(get_model_class_names(MEDIAPIPE_FACE_FEATURES_MODEL)) == 5


def test_xyz_prompt_search_replace_writes_the_parameters_before_the_batch_starts():
    # The host sets p.batch_index only after the scripts' process() hook, and
    # the XYZ grid hands every cell a fresh copy of p. Reading the prompt for
    # the Prompt S/R axis there raised AttributeError, logged an error panel
    # for every cell and left the parameters out.
    from aaaaaa.p_method import get_i
    from adetailer.args import ADetailerArgs

    assert get_i(SimpleNamespace(iteration=0, batch_size=2)) == 0
    assert get_i(SimpleNamespace(iteration=2, batch_size=3, batch_index=1)) == 7

    runtime = _load_script(
        methods=_INLINE_PROMPT_METHODS | {"process", "extra_params", "_record_xyz_prompt_sr"},
        functions=_INLINE_PROMPT_FUNCTIONS,
        assigns=_INLINE_PROMPT_ASSIGNS,
        opts=SimpleNamespace(data={}),
        is_img2img_inpaint=lambda _p: False,
        get_i=get_i,
        suffix=_ui_suffix(),
        __version__="test",
    )
    script = runtime.AfterDetailerScript()
    script.is_ad_enabled = lambda *_args: True
    script.set_skip_img2img = lambda *_args: None
    script.get_args = lambda *_args: [
        ADetailerArgs(ad_model="face_yolov8n.pt", ad_prompt="blue eyes")
    ]
    p = SimpleNamespace(
        iteration=0, batch_size=1, prompt="a photo", negative_prompt="",
        all_prompts=["a photo"], all_negative_prompts=[""],
        extra_generation_params={},
        _ad_xyz_prompt_sr=[SimpleNamespace(s="blue", r="red")],
    )

    script.process(p, True, False, {})

    assert p.extra_generation_params["ADetailer prompt"] == "red eyes"
    assert "ADetailer version" in p.extra_generation_params


def test_merged_face_boxes_past_the_frame_get_a_real_denoise_strength():
    # MediaPipe face boxes are not clipped to the image, so the union of two
    # merged faces can be larger than the frame.
    from adetailer.args import ADetailerArgs
    from adetailer.common import PredictOutput
    from adetailer.opts import dynamic_denoise_strength

    boxes = [[-100, -100, 300, 300], [250, 250, 650, 650]]
    masks = []
    for box in boxes:
        mask = Image.new("L", (512, 512), 0)
        mask.paste(255, [max(0, v) for v in box])
        masks.append(mask)
    pred = PredictOutput(
        bboxes=boxes, masks=masks, confidences=[0.9, 0.9],
        preview=Image.new("RGB", (512, 512)),
    )
    args = ADetailerArgs(
        ad_model="mediapipe_face_short", ad_mask_merge_invert="Merge",
        ad_dynamic_denoise_power=2.5,
    )

    _class_prompt_script().pred_preprocessing(SimpleNamespace(), pred, args)
    strength = _load_script(
        methods={"get_dynamic_denoise_strength"},
        opts=SimpleNamespace(data={}),
        dynamic_denoise_strength=dynamic_denoise_strength,
    ).AfterDetailerScript.get_dynamic_denoise_strength(
        0.4, pred.bboxes[0], (512, 512), args
    )

    assert pred.bboxes == [[-100, -100, 650, 650]]
    assert isinstance(strength, float)
    assert 0.0 <= strength <= 0.4


def _nan_standalone(tmp_path):
    """A standalone runner: run(["nan", "ok", ...]) detects one region per
    entry, which fails with a NaN error or succeeds; run(None) detects
    nothing."""
    state = SimpleNamespace(
        interrupted=False, skipped=False, stopping_generation=False,
        job_count=0, assign_current_image=lambda _image: None,
    )
    nans = type("NansException", (Exception,), {})
    image = Image.new("RGB", (64, 64), "white")
    result = Image.new("RGB", (64, 64), "red")
    todo = []

    def process_images(_p2):
        if todo.pop(0) == "nan":
            raise nans("A tensor with NaNs was produced in Unet.")
        return SimpleNamespace(images=[result])

    def predict(*_args, **_kwargs):
        return SimpleNamespace(preview=image if todo else None)

    runtime = _load_script(
        methods={
            "run_detailer_on_image", "read_params_txt", "write_params_txt",
            "_postprocess_image_inner",
        },
        paths=SimpleNamespace(data_path=str(tmp_path)),
        PARAMS_TXT="params.txt",
        shared=SimpleNamespace(
            sd_model=None, state=state, opts=SimpleNamespace(data={})
        ),
        state=state,
        opts=SimpleNamespace(samples_format="png"),
        all_samplers=[],
        images=None,
        AD_APPLY_SUBDIR="ADetailer-Inpaint",
        pause_total_tqdm=nullcontext,
        time=time, copy=copy, get_i=lambda _p: 0,
        parse_csv=lambda text: text.split(","),
        is_skip_img2img=lambda _p: False,
        _ad_verbose=lambda: False,
        _verbose_pass_header=lambda *_args: None,
        _verbose_detection=lambda *_args: None,
        disable_safe_unpickle=nullcontext,
        ultralytics_predict=predict,
        ensure_pil_image=lambda im, _mode: im,
        process_images=process_images,
        NansException=nans,
        ordinal=str,
        Image=Image,
    )
    script = runtime.AfterDetailerScript()
    script.ultralytics_device = "cpu"
    script.get_i2i_p = lambda _p, _args, im: SimpleNamespace(
        init_images=[im], prompt="face", negative_prompt="", close=lambda: None,
        denoising_strength=0.4, width=64, height=64,
    )
    script.get_prompt = lambda *_args: (["face"], [""])
    script.get_ad_model = lambda _name: "model.pt"
    script.pred_preprocessing = lambda *_args: [Image.new("L", (64, 64), 255)] * len(
        todo
    )
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

    def run(regions):
        todo[:] = list(regions or [])
        return script.run_detailer_on_image(image, args, save=False)

    return run, image


def test_standalone_run_reports_a_nan_error_instead_of_nothing_detected(tmp_path):
    # Every detected region failed with a NaN error: the image was detected
    # but not detailed, and a folder run must count it as failed.
    run, image = _nan_standalone(tmp_path)

    result, status = run(["nan", "nan"])
    assert result is None
    assert status.startswith("⚠️ ADetailer run failed")
    assert "NaN" in status

    # Unchanged, and nothing carried over from the failed run.
    assert run(None) == (image, "ℹ️ Nothing detected — image unchanged.")
    result, status = run(["nan", "ok"])
    assert result is not None
    assert status.startswith("✅ ADetailer pass complete.")


def test_readme_says_each_sequential_class_pass_saves_its_own_preview():
    state = SimpleNamespace(
        interrupted=False, skipped=False, job_count=0,
        assign_current_image=lambda _image: None,
    )
    result = Image.new("RGB", (8, 8), "red")
    run, _pp, _before = _detailer(
        state, lambda _p: SimpleNamespace(images=[result]), n_masks=1
    )
    saved = []
    _script_of(run).save_image = lambda _p, _image, *, condition, suffix: saved.append(
        (condition, suffix)
    )

    assert run(classes="face,hand", sequential=True) is True
    assert [s for c, s in saved if c == "ad_save_previews"] == [
        "-ad-preview-1-1-face", "-ad-preview-1-2-hand",
    ]
    readme = (_SCRIPT_PATH.parents[1] / "README.md").read_text(encoding="utf-8")
    row = readme[readme.index("\n### Process classes sequentially"):]
    row = row[: row.index("\n### ", 1)]
    assert "first class only" not in readme
    assert "saves its own mask preview" in row


def test_readme_copy_and_preset_sections_match_the_ui():
    script_text = _SCRIPT_PATH.read_text(encoding="utf-8")
    option = script_text[script_text.index('"ad_max_models",'):]
    default = re.search(r"default=(\d+)", option).group(1)
    maximum = re.search(r'"maximum":\s*(\d+)', option).group(1)
    ui_text = (_SCRIPT_PATH.parents[1] / "aaaaaa" / "ui.py").read_text(encoding="utf-8")
    readme = (_SCRIPT_PATH.parents[1] / "README.md").read_text(encoding="utf-8")
    copy_section = readme[readme.index("## Copy Settings Between Tabs"):]
    copy_section = copy_section[: copy_section.index("## Preset Library")]
    presets = readme[readme.index("## Preset Library"):]

    max_tabs = next(line for line in copy_section.splitlines() if "**Max tabs**" in line)
    assert f"default {default}, up to {maximum}" in max_tabs
    assert 'f"\\U0001F4E5 Paste from {ordinal(idx + 1)} tab"' in ui_text
    assert "📥 Paste from Nth tab" in copy_section
    for escaped, label in [
        ("\\U0001F4C2 Load", "📂 Load"),
        ("✏️ Rename", "✏️ Rename"),
        ("\\U0001F5D1 Delete", "🗑 Delete"),
    ]:
        assert f'value="{escaped}"' in ui_text
        assert f"**{label}**" in presets
    for stale in ("Paste settings from Nth tab here", "**Load preset**",
                  "**Rename preset**", "**Delete preset**"):
        assert stale not in readme


def test_readme_describes_the_current_tab_layout():
    # The README placed the export / import accordion at the bottom and
    # Copy / Paste at the top of the tab, the other way round.
    ui_text = (_SCRIPT_PATH.parents[1] / "aaaaaa" / "ui.py").read_text(encoding="utf-8")
    readme = (_SCRIPT_PATH.parents[1] / "README.md").read_text(encoding="utf-8")
    order = [
        ui_text.index(anchor)
        for anchor in (
            'eid("ad_tab_enable")', '"Preset library export / import"',
            'eid("ad_preset_dropdown")', 'eid("ad_copy_settings")',
        )
    ]
    assert order == sorted(order)
    assert 'label="Overwrite on conflict"' in ui_text
    row = readme[readme.index("\n### Export and import presets"):]
    row = row[: row.index("\n### ", 1)]
    assert "at the top of every tab" in row
    assert '"Overwrite on conflict"' in row
    for stale in ("at the bottom of the preset area", "Overwrite existing on conflict",
                  "`Enable this tab` + `Copy settings` + `Paste settings` row"):
        assert stale not in readme


# Final debugging pass, round 3: generation pipeline, prompts and docs.


def _i2i_script(host=_A1111I2I, methods=(), **host_globals):
    """get_i2i_p with the host's img2img class and the other inputs stubbed."""
    from aaaaaa.p_method import is_skip_img2img

    host_globals.setdefault("is_skip_img2img", is_skip_img2img)
    host_globals.setdefault("opts", SimpleNamespace())  # a host's own options
    runtime = _load_script(
        methods={"get_i2i_p", "get_width_height", *methods},
        StableDiffusionProcessingImg2Img=host,
        schedulers=None,
        controlnet_type="forge",
        copy_extra_params=dict,
        **host_globals,
    )
    script = runtime.AfterDetailerScript()
    script.get_seed = lambda _p: (1, 1)
    script.get_steps = lambda *_args: 20
    script.get_cfg_scale = lambda *_args: 1.0
    script.get_initial_noise_multiplier = lambda *_args: None
    script.get_sampler = lambda *_args: "Euler"
    if "get_override_settings" not in methods:
        script.get_override_settings = lambda *_args: {}
    script.script_filter = lambda *_args: (None, [])
    return script


def _txt2img_p(**extra):
    return SimpleNamespace(
        sd_model=None, outpath_samples="", outpath_grids="", styles=[],
        subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0,
        tiling=False, extra_generation_params={}, width=512, height=768, **extra,
    )


@pytest.mark.parametrize(
    ("only_masked", "enable_hr", "separate", "canvas"),
    [
        # Hires fix x2: the host keeps p.width/height at the first pass.
        (False, True, False, (1024, 1536)),
        # Unchanged: the crop resolution, a generation without Hires fix (and
        # the standalone run's capped canvas), and a separate size.
        (True, True, False, (512, 768)),
        (False, None, False, (512, 768)),
        (False, True, True, (640, 640)),
    ],
)
def test_whole_picture_inpaint_keeps_the_hires_fix_size(
    only_masked, enable_hr, separate, canvas
):
    # The host resizes the whole image to the canvas and returns it at that
    # size, so a 1024x1536 hires image came back at the 512x768 first pass.
    from adetailer.args import ADetailerArgs

    script = _i2i_script()
    p = _txt2img_p() if enable_hr is None else _txt2img_p(enable_hr=enable_hr)
    args = ADetailerArgs(
        ad_model="face_yolov8n.pt", ad_inpaint_only_masked=only_masked,
        ad_use_inpaint_width_height=separate, ad_inpaint_width=640,
        ad_inpaint_height=640,
    )

    i2i = script.get_i2i_p(p, args, Image.new("RGB", (1024, 1536)))

    assert (i2i.width, i2i.height) == canvas


@pytest.mark.parametrize(
    ("only_masked", "separate", "canvas"),
    [
        # Whole picture: the init image's own size, not the sliders' 512x512.
        (False, False, (832, 1216)),
        # Unchanged: the crop resolution and a separate size.
        (True, False, (512, 512)),
        (False, True, (640, 640)),
    ],
)
def test_whole_picture_inpaint_keeps_the_skip_img2img_image_size(
    only_masked, separate, canvas
):
    # With Skip img2img the host resized the whole 832x1216 init image to the
    # img2img sliders, so the result came back squashed to 512x512.
    from adetailer.args import ADetailerArgs

    script = _i2i_script()
    p = _txt2img_p(
        _ad_skip_img2img=True,
        _ad_orig=SimpleNamespace(
            steps=20, sampler_name="Euler a", width=512, height=512
        ),
    )
    p.width = p.height = 128
    args = ADetailerArgs(
        ad_model="face_yolov8n.pt", ad_inpaint_only_masked=only_masked,
        ad_use_inpaint_width_height=separate, ad_inpaint_width=640,
        ad_inpaint_height=640,
    )

    i2i = script.get_i2i_p(p, args, Image.new("RGB", (832, 1216)))

    assert (i2i.width, i2i.height) == canvas


@pytest.mark.parametrize(
    ("use_checkpoint", "checkpoint", "expected"),
    [
        (True, "other.safetensors", 3.5),
        (True, "Use same checkpoint", 9.0),
        (False, "other.safetensors", 9.0),
    ],
)
def test_a_separate_detailer_checkpoint_keeps_the_host_distilled_cfg_scale(
    use_checkpoint, checkpoint, expected
):
    # Forge Neo's SDXL preset submits a hidden Shift of 9.0: a Flux detailer
    # checkpoint read it as guidance 9.0 instead of the host's 3.5.
    from adetailer.args import ADetailerArgs

    script = _i2i_script(
        _ForgeI2I,
        methods={"get_override_settings"},
        _is_forge_modules=lambda: True,
        _forge_wanted_modules=lambda _args: None,
    )
    args = ADetailerArgs(
        ad_model="face_yolov8n.pt", ad_use_checkpoint=use_checkpoint,
        ad_checkpoint=checkpoint,
    )

    i2i = script.get_i2i_p(_txt2img_p(distilled_cfg_scale=9.0), args, None)

    assert i2i.distilled_cfg_scale == expected


def test_saved_step_images_carry_their_own_seed_and_prompt():
    # The -ad-before / -ad-preview / -ad-step files of images 2 and later
    # carried image 1's seed and prompts, so PNG Info recreated image 1.
    from aaaaaa.p_method import get_i

    def create_infotext(
        p, all_prompts, all_seeds, _all_subseeds, _comments=None, iteration=0,
        position_in_batch=0,
    ):
        i = position_in_batch + iteration * p.batch_size
        return f"{all_prompts[i]}|{p.all_negative_prompts[i]}|Seed: {all_seeds[i]}"

    infotext = _load_script(
        methods={"infotext"}, create_infotext=create_infotext, get_i=get_i
    ).AfterDetailerScript.infotext
    p = SimpleNamespace(
        iteration=1, batch_size=2, batch_index=1,
        all_prompts=["p0", "p1", "p2", "p3"],
        all_negative_prompts=["n0", "n1", "n2", "n3"],
        all_seeds=[10, 11, 12, 13], all_subseeds=[20, 21, 22, 23],
    )

    assert infotext(p) == "p3|n3|Seed: 13"
    p.iteration, p.batch_index = 0, 0
    assert infotext(p) == "p0|n0|Seed: 10"


@pytest.mark.parametrize("fails", [True, False])
def test_an_error_in_a_tab_still_restarts_the_scripts_and_keeps_params_txt(
    tmp_path, fails
):
    # A tab that raised (a model that is not installed, out of memory) left
    # the inner pass in params.txt and never re-ran the other scripts'
    # before_process / process, so Forge Neo's ControlNet lost its units for
    # every later batch.
    from aaaaaa.p_method import need_call_postprocess, need_call_process

    params_txt = tmp_path / "params.txt"
    params_txt.write_text("user generation", encoding="utf-8")
    script = _load_script(
        methods={"postprocess_image", "read_params_txt", "write_params_txt"},
        paths=SimpleNamespace(data_path=str(tmp_path)),
        PARAMS_TXT="params.txt",
        opts=SimpleNamespace(data={}),
        ensure_pil_image=lambda image, _mode: image,
        copy=copy,
        _verbose_gen_header=lambda *_args: None,
        need_call_postprocess=need_call_postprocess,
        need_call_process=need_call_process,
        Processed=lambda *_args: "dummy",
        preserve_prompts=lambda _p: nullcontext(),
        CNHijackRestore=nullcontext,
        pause_total_tqdm=nullcontext,
        cn_allow_script_control=nullcontext,
        _should_skip_for_hires_only=lambda _p, _args: False,
        is_skip_img2img=lambda _p: False,
    ).AfterDetailerScript()
    tab = SimpleNamespace(need_skip=lambda: False)
    script.is_ad_enabled = lambda *_args: True
    script.get_i2i_init_image = lambda _p, pp: pp.image
    script.get_args = lambda *_args: [tab, tab]
    script._will_run_sequential = lambda _args: False
    script.save_image = lambda *_args, **_kwargs: None

    def inner(_p, _pp, _args, n=0):
        params_txt.write_text(f"inner pass {n}", encoding="utf-8")
        if fails and n == 1:
            msg = "Model 'missing.pt' not found"
            raise ValueError(msg)
        return True

    script._postprocess_image_inner = inner
    hooks = []
    runner = SimpleNamespace(
        postprocess=lambda *_args: hooks.append("postprocess"),
        before_process=lambda *_args: hooks.append("before_process"),
        process=lambda *_args: hooks.append("process"),
    )
    p = SimpleNamespace(
        batch_index=0, batch_size=1, seed=1, scripts=runner, extra_generation_params={}
    )
    pp = SimpleNamespace(image=Image.new("RGB", (8, 8)))

    if fails:
        with pytest.raises(ValueError, match="not found"):
            script.postprocess_image(p, pp, True)
    else:
        script.postprocess_image(p, pp, True)

    assert params_txt.read_text(encoding="utf-8") == "user generation"
    assert hooks == ["postprocess", "before_process", "process"]


@pytest.mark.parametrize("global_dir", [True, False])
def test_standalone_run_follows_the_global_output_directory(tmp_path, global_dir):
    # Settings > Paths "Output directory for images" holds every generation;
    # the standalone tool still wrote next to the img2img folder.
    img2img_dir = tmp_path / "webui" / "outputs" / "img2img-images"
    override = str(tmp_path / "other-drive" / "images") if global_dir else ""
    saved, shells = [], []
    script = _load_script(
        methods={"run_detailer_on_image", "read_params_txt", "write_params_txt"},
        paths=SimpleNamespace(data_path=str(tmp_path)),
        PARAMS_TXT="params.txt",
        shared=SimpleNamespace(sd_model=None),
        state=SimpleNamespace(interrupted=False, skipped=False),
        opts=SimpleNamespace(
            samples_format="png", outdir_samples=override,
            outdir_img2img_samples=str(img2img_dir),
        ),
        all_samplers=[],
        ensure_pil_image=lambda image, _mode: image,
        images=SimpleNamespace(
            save_image=lambda _image, path, _basename, **_kwargs: saved.append(path)
        ),
        AD_APPLY_SUBDIR="ADetailer-Inpaint",
        pause_total_tqdm=nullcontext,
    ).AfterDetailerScript()

    def inner(p, _pp, _args, n=0):
        shells.append(p)
        return True

    script._postprocess_image_inner = inner

    _image, status = script.run_detailer_on_image(
        Image.new("RGB", (64, 64)), SimpleNamespace(), save=True
    )

    assert "Saved" in status
    if global_dir:
        assert shells[0].outpath_samples == override
        assert saved == [str(Path(override) / "ADetailer-Inpaint")]
    else:
        assert shells[0].outpath_samples == str(img2img_dir)
        assert saved == [str(img2img_dir.resolve().parent / "ADetailer-Inpaint")]


@pytest.mark.parametrize(
    ("skip", "init_images", "attrs", "expected"),
    [
        # An API batch with one init image per batch position.
        (True, ["A", "B"], {"batch_index": 1}, "B"),
        (True, ["A", "B"], {"batch_index": 0}, "A"),
        # Unchanged: one init image, repeated by the host; no batch index.
        (True, ["A"], {"batch_index": 3}, "A"),
        (True, ["A", "B"], {}, "A"),
        (False, ["A", "B"], {"batch_index": 1}, "sample"),
    ],
)
def test_skip_img2img_details_each_batch_images_own_init_image(
    skip, init_images, attrs, expected
):
    get_init = _load_script(
        methods={"get_i2i_init_image"},
        is_skip_img2img=lambda p: getattr(p, "_ad_skip_img2img", False),
    ).AfterDetailerScript.get_i2i_init_image
    p = SimpleNamespace(_ad_skip_img2img=skip, init_images=init_images, **attrs)

    assert get_init(p, SimpleNamespace(image="sample")) == expected


@pytest.mark.parametrize("left_out_by", ["filters", "skip"])
def test_standalone_run_says_when_detections_were_left_out(tmp_path, left_out_by):
    # The detection numbers, the mask filters or [SKIP] can leave nothing to
    # inpaint; the tool said "Nothing detected", against the preview.
    run, image = _nan_standalone(tmp_path)
    script = _script_of(run)
    if left_out_by == "filters":
        script.pred_preprocessing = lambda *_args: []
    else:
        script._apply_inline_class_prompts = lambda p2, *_args: setattr(
            p2, "prompt", "[SKIP]"
        )

    result, status = run(["ok", "ok"])

    assert result is image
    assert status.startswith("ℹ️ Detections found")
    assert "Nothing detected" not in status
    # Nothing carried over to a run that detects nothing.
    assert run(None) == (image, "ℹ️ Nothing detected — image unchanged.")


@pytest.mark.parametrize(
    ("classes", "class_prompts"),
    [
        ("Face,Hand", "face: detailed skin\nhand: five fingers"),
        ("face,hand", "Face: detailed skin\nHand: five fingers"),
        # An exact match wins over one in another case.
        ("face,hand", "Face: other\nface: detailed skin\nhand: five fingers"),
    ],
)
def test_class_prompt_lines_match_the_class_regardless_of_case(classes, class_prompts):
    # The class filter matches regardless of case; its per-class prompt lines
    # did not, so each pass got the tab prompt without the line or the guard.
    from adetailer.args import ADetailerArgs

    runtime = _load_runtime(
        state=SimpleNamespace(interrupted=False, skipped=False),
        copy=copy, re=re,
        parse_csv=lambda text: text.split(","),
        is_skip_img2img=lambda _p: False,
    )
    script = runtime.AfterDetailerScript()
    script.save_image = lambda *_args, **_kwargs: None
    passes = []

    def class_pass(_p, _pp, sub_args, **_kwargs):
        passes.append(sub_args.ad_prompt)
        return True

    script._postprocess_image_inner = class_pass  # each class's own pass
    args = ADetailerArgs(
        ad_model="face_yolov8n.pt", ad_prompt="main", ad_model_classes=classes,
        ad_classes_sequential=True, ad_class_prompts=class_prompts,
    )

    assert runtime.AfterDetailerScript._postprocess_image_inner(
        script, SimpleNamespace(), SimpleNamespace(image=Image.new("RGB", (8, 8))), args
    )
    assert passes == ["detailed skin", "five fingers"]


@pytest.mark.parametrize(
    ("classes", "expected"),
    [
        # "hands" is not a class of the detector: its pass detected every
        # class and repainted the faces with the hands' prompt.
        ("face,hands", [("face", "main")]),
        ("Face,hand,hands", [("Face", "main"), ("hand", "five fingers")]),
        # No known name at all: one pass for every class with the tab prompt,
        # as in single-pass mode, instead of one full pass per name.
        ("faces,hands", [("faces,hands", "main")]),
        # Unchanged: known names.
        ("face,hand", [("face", "main"), ("hand", "five fingers")]),
    ],
)
def test_a_sequential_pass_runs_only_for_a_class_the_detector_has(
    classes, expected, tmp_path
):
    from adetailer.args import ADetailerArgs
    from adetailer.classes import get_model_class_names, resolve_class_ids

    model = tmp_path / "multi.pt"
    model.write_bytes(b"x")
    (tmp_path / "multi.names.json").write_bytes(b'["face","hand"]')
    runtime = _load_runtime(
        state=SimpleNamespace(interrupted=False, skipped=False),
        copy=copy, re=re,
        parse_csv=lambda text: text.split(","),
        is_skip_img2img=lambda _p: False,
        disable_safe_unpickle=nullcontext,
        get_model_class_names=get_model_class_names,
        resolve_class_ids=resolve_class_ids,
    )
    script = runtime.AfterDetailerScript()
    script.save_image = lambda *_args, **_kwargs: None
    script.get_ad_model = lambda _name: model
    passes = []

    def class_pass(_p, _pp, sub_args, **_kwargs):
        passes.append((sub_args.ad_model_classes, sub_args.ad_prompt))
        return True

    script._postprocess_image_inner = class_pass  # each class's own pass
    args = ADetailerArgs(
        ad_model="multi.pt", ad_prompt="main", ad_model_classes=classes,
        ad_classes_sequential=True,
        ad_class_prompts="hand: five fingers\nhands: five fingers",
    )

    assert runtime.AfterDetailerScript._postprocess_image_inner(
        script, SimpleNamespace(), SimpleNamespace(image=Image.new("RGB", (8, 8))), args
    )
    assert passes == expected


def test_an_unknown_non_latin_class_gets_no_sequential_pass_on_a_legacy_console(
    tmp_path, monkeypatch
):
    # A console pipe in a legacy code page (cp1252 here) could not print the
    # "class not found" line: the check that drops the name failed quietly,
    # and the name then got a pass of its own.
    monkeypatch.setattr(
        sys,
        "stdout",
        io.TextIOWrapper(
            io.BytesIO(), encoding="cp1252", errors="strict", write_through=True
        ),
    )
    test_a_sequential_pass_runs_only_for_a_class_the_detector_has(
        "face,\u9854", [("face", "main")], tmp_path
    )


def test_a_digit_class_that_int_rejects_gets_no_sequential_pass(tmp_path):
    # "\u00b2" is a digit to str.isdigit() but not to int(): the check that drops
    # unknown names raised, so "hands" kept its pass too, and the "\u00b2" pass of
    # its own stopped ADetailer for the image.
    test_a_sequential_pass_runs_only_for_a_class_the_detector_has(
        "face,\u00b2,hands", [("face", "main")], tmp_path
    )


def test_an_unconvertible_class_does_not_stop_a_sequential_run_with_no_names_known():
    # With no class names known (a YOLO-World detector, MediaPipe face
    # features) every class gets its pass, a number that cannot be
    # converted included.
    from adetailer.args import ADetailerArgs

    runtime = _load_runtime(
        state=SimpleNamespace(interrupted=False, skipped=False),
        copy=copy, re=re,
        parse_csv=lambda text: text.split(","),
        is_skip_img2img=lambda _p: False,
        disable_safe_unpickle=nullcontext,
        get_model_class_names=lambda _model: [],
        resolve_class_ids=lambda _model, _requested: [],
    )
    script = runtime.AfterDetailerScript()
    script.save_image = lambda *_args, **_kwargs: None
    script.get_ad_model = lambda _name: "multi.pt"
    passes = []

    def class_pass(_p, _pp, sub_args, **_kwargs):
        passes.append((sub_args.ad_model_classes, sub_args.ad_prompt))
        return True

    script._postprocess_image_inner = class_pass  # each class's own pass
    token = "1" * 5000
    args = ADetailerArgs(
        ad_model="multi.pt", ad_prompt="main", ad_model_classes="face," + token,
        ad_classes_sequential=True, ad_class_prompts="face: detailed skin",
    )

    assert runtime.AfterDetailerScript._postprocess_image_inner(
        script, SimpleNamespace(), SimpleNamespace(image=Image.new("RGB", (8, 8))), args
    )
    assert passes == [("face", "detailed skin"), (token, "main")]


def test_a_class_prompt_line_in_another_case_still_takes_priority_over_the_guard():
    from adetailer.args import ADetailerArgs

    script = _class_prompt_script()
    args = ADetailerArgs(
        ad_model="faces.pt", ad_class_guard=True,
        ad_class_prompts="Face: detailed skin",
    )
    p2 = SimpleNamespace(prompt="detailed", negative_prompt="blurry")

    script._apply_auto_class_guard(
        p2, args, SimpleNamespace(class_names=["face"]), 0, 1, True
    )

    assert (p2.prompt, p2.negative_prompt) == ("detailed", "blurry")


@pytest.mark.parametrize("suffix", ["", " 2nd"])
def test_pasting_an_upstream_image_resets_the_forks_own_toggles(suffix):
    # Upstream ADetailer never writes these keys, and this extension always
    # does: a pasted upstream image kept the tab's hires-only toggle (so
    # ADetailer was skipped without hires fix), bbox mask and scale.
    from adetailer.args import ADetailerArgs

    params = {
        name + suffix: value
        for name, value in {
            "ADetailer model": "face_yolov8n.pt",
            "ADetailer confidence": "0.3",
            "ADetailer dilate erode": "4",
            "ADetailer mask blur": "4",
            "ADetailer denoising strength": "0.4",
            "ADetailer inpaint only masked": "True",
            "ADetailer inpaint padding": "32",
        }.items()
    }
    params["ADetailer version"] = "24.11.1"
    before = ADetailerArgs(
        ad_model="hand_yolov8n.pt", ad_apply_on_hires_only=True,
        ad_use_bbox_mask=True, ad_use_resolution_scale=True,
        ad_resolution_scale=2.0, ad_use_main_loras=True, ad_use_lora_triggers=True,
    ).dict()

    _paste_callback()._clear_missing_class_prompts("", params)
    after = ADetailerArgs(**_host_paste(params, before, suffix))

    assert not after.ad_apply_on_hires_only
    assert not after.ad_use_bbox_mask
    assert not after.ad_use_resolution_scale
    assert not after.ad_use_main_loras
    # Only used while its toggle is on: the tab keeps its own value.
    assert after.ad_resolution_scale == 2.0
    assert after.ad_use_lora_triggers


def test_an_image_made_by_this_extension_keeps_its_own_toggles_on_paste():
    from adetailer.args import ADetailerArgs

    made = ADetailerArgs(
        ad_model="face_yolov8n.pt", ad_apply_on_hires_only=True, ad_use_bbox_mask=True
    )
    params = _infotext(made)
    skipped = dict(params)
    del skipped["ADetailer use bbox mask"]

    _paste_callback()._clear_missing_class_prompts("", params)
    _paste_callback(["ADetailer use bbox mask"])._clear_missing_class_prompts(
        "", skipped
    )

    assert params["ADetailer apply on hires only"] == "True"
    assert params["ADetailer use bbox mask"] == "True"
    assert "ADetailer use bbox mask" not in skipped  # "Disregard fields ..."


def _public_docs():
    root = _SCRIPT_PATH.parents[1]
    return {
        name: (root / name).read_text(encoding="utf-8")
        for name in ("README.md", "CHANGELOG.md")
    }


def test_public_docs_are_in_english():
    # Public text is English only; two older changelog passages were reworded.
    changelog = _public_docs()["CHANGELOG.md"]
    assert "Wrap-up of the 2026-05-18..2026-05-19 test session." in changelog
    assert "add a button in the settings that resets the extension's" in changelog


def test_public_docs_match_the_behaviour_of_this_beta():
    docs = _public_docs()
    readme, changelog = docs["README.md"], docs["CHANGELOG.md"]
    paste = next(
        line for line in readme.splitlines()
        if line.startswith("- Loading a preset or pasting a tab")
    )
    # Parameters without ADetailer leave its tabs as they are.
    assert "for an image made with ADetailer, switches off the tabs" in paste
    # Earlier releases never loaded MediaPipe models from such a path.
    assert "MediaPipe detectors work again" not in readme
    # AUTOMATIC1111 now softens the later regions' mask edge at canvas size.
    assert "its result does not change" not in changelog
    # That edge is narrower, not wider, on a canvas larger than the image.
    assert "slightly wider than before" not in changelog
    # The dictionaries lack the reworded tooltips and help text.
    l10n = next(
        line for line in readme.splitlines() if "translated into 10 locales" in line
    )
    assert "Copy settings and Reset tooltips" in l10n
    assert "stay English" in l10n
    # ... and the options added since plus.7 and the Gradio 3 Export tooltip,
    # while the README said every fork label was translated.
    assert not l10n.startswith("- 🟡 Every UI label")
    for name in ("Auto class-guard", "Inpaint only these detections",
                 "Reset every tab", "Verbose diagnostic log", "Export tooltip",
                 "Paste from Nth tab"):
        assert name in l10n
    # The label the other tabs' Paste button shows after a Copy is built at
    # run time and is in no dictionary; only "📥 Paste settings" is.
    ui_text = (_SCRIPT_PATH.parents[1] / "aaaaaa" / "ui.py").read_text(encoding="utf-8")
    assert 'Paste from {ordinal(idx + 1)} tab"' in ui_text
    assert "the Paste settings label follow the UI language" not in readme
    relabel = next(
        line for line in changelog.splitlines()
        if line.startswith("- **On a translated UI, the Paste settings button went back")
    )
    assert "Paste from Nth tab" in relabel
    tooltips = next(
        line for line in changelog.splitlines()
        if line.startswith("- **The button tooltips were always in English,**")
    )
    assert "Export tooltip" in tooltips


# Final debugging pass, round 5: generation pipeline, prompts and docs.


def test_readme_says_when_an_added_sidecar_needs_a_restart():
    # Once names were found for a detector (a generation reads them from the
    # .pt on AUTOMATIC1111), they are cached for the session: a sidecar added
    # afterwards is read only after a restart, which the README left out.
    readme = _public_docs()["README.md"]
    section = readme[readme.index("#### Custom class names via sidecar JSON"):]
    section = section[: section.index("#### Backwards compatibility")]

    assert "adding or changing this file" in section
    assert "still empty" in section
    assert "fills an empty CLASSES dropdown without a restart" in readme


@pytest.mark.parametrize(
    ("main", "prompt", "options", "face", "hand"),
    [
        # A LoRA the face block names at another weight: the hands still get
        # the main prompt's one, the faces keep the block's weight.
        (
            "portrait <lora:style:1>",
            "[CLASS=face]<lora:style:0.5> smiling[/CLASS] detailed", {},
            "<lora:style:0.5> smiling detailed", "detailed <lora:style:1>",
        ),
        (
            "portrait <lora:style:1>",
            "[CLASS=face]<lora:style:1> smiling[/CLASS] detailed", {},
            "<lora:style:1> smiling detailed", "detailed <lora:style:1>",
        ),
        # A trigger phrase written only in the face block.
        (
            "portrait <lora:style (cool look):1>",
            "[CLASS=face]cool look, smiling[/CLASS] detailed",
            {"ad_use_lora_triggers": True},
            "cool look, smiling detailed <lora:style (cool look):1>",
            "detailed <lora:style (cool look):1>, cool look",
        ),
        # Unchanged: no LoRA in the blocks, one outside them, Strip LoRAs.
        (
            "portrait <lora:style:1>", "[CLASS=face]smiling[/CLASS] detailed", {},
            "smiling detailed <lora:style:1>", "detailed <lora:style:1>",
        ),
        (
            "portrait <lora:style:1>",
            "[CLASS=face]smiling[/CLASS] detailed <lora:style:0.5>", {},
            "smiling detailed <lora:style:0.5>", "detailed <lora:style:0.5>",
        ),
        (
            "portrait <lora:style:1>",
            "[CLASS=face]<lora:style:0.5> smiling[/CLASS] detailed",
            {"ad_strip_loras": True},
            "smiling detailed", "detailed",
        ),
    ],
)
def test_a_lora_named_in_another_class_block_still_reaches_the_region(
    main, prompt, options, face, hand
):
    # "Use LoRAs from main prompt" skipped a LoRA the tab prompt named
    # anywhere, also inside the face block the hand region drops, so the
    # hands were inpainted without it.
    from adetailer.args import ADetailerArgs

    runtime = _load_script(
        methods=_INLINE_PROMPT_METHODS,
        functions=_INLINE_PROMPT_FUNCTIONS,
        assigns=_INLINE_PROMPT_ASSIGNS,
        get_i=lambda _p: 0,
    )
    script = runtime.AfterDetailerScript()
    p = SimpleNamespace(
        prompt=main, all_prompts=[main], negative_prompt="",
        all_negative_prompts=[""],
    )
    args = ADetailerArgs(
        ad_model="faces.pt", ad_prompt=prompt, ad_use_main_loras=True, **options
    )
    prompts, negatives = script.get_prompt(p, args)
    pred = SimpleNamespace(class_names=["face", "hand"])

    regions = []
    for j, inverted in ((0, False), (1, False), (0, True)):
        p2 = SimpleNamespace()
        script.i2i_prompts_replace(p2, prompts, negatives, j)
        script._apply_inline_class_prompts(p2, pred, j, 2, inverted, p, args)
        regions.append(p2.prompt)

    # A Merge and Invert background drops every block, like the hands.
    assert regions == [face, hand, hand]


def test_whole_picture_skip_img2img_canvas_is_rounded_down_to_a_multiple_of_8():
    # The docs said the result keeps the init image's size; the WebUI needs a
    # multiple of 8, so a 1080x1350 image comes back at 1080x1344.
    from adetailer.args import ADetailerArgs

    script = _i2i_script()
    p = _txt2img_p(
        _ad_skip_img2img=True,
        _ad_orig=SimpleNamespace(
            steps=20, sampler_name="Euler a", width=512, height=512
        ),
    )
    p.width = p.height = 128
    args = ADetailerArgs(ad_model="face_yolov8n.pt", ad_inpaint_only_masked=False)

    i2i = script.get_i2i_p(p, args, Image.new("RGB", (1080, 1350)))

    assert (i2i.width, i2i.height) == (1080, 1344)
    docs = _public_docs()
    row = next(
        line for line in docs["README.md"].splitlines()
        if line.startswith("| Skip img2img")
    )
    entry = next(
        line for line in docs["CHANGELOG.md"].splitlines()
        if line.startswith('- **With Skip img2img and "Inpaint only masked" off')
    )
    assert "multiple of 8" in row
    assert "multiple of 8" in entry


@pytest.mark.parametrize(
    ("field", "values", "key"),
    [
        ("ad_model", ["face_yolov8n.pt", "None"], "ADetailer model"),
        ("ad_prompt", ["smile", ""], "ADetailer prompt"),
        (
            "ad_controlnet_model", ["control_v11p_sd15_inpaint", "None"],
            "ADetailer ControlNet model",
        ),
    ],
)
def test_an_xyz_cell_does_not_keep_the_previous_cells_parameters(field, values, key):
    # Every X/Y/Z grid cell is a shallow copy of p sharing its parameters:
    # the second cell, which skips the tab or leaves the field at its
    # default, kept the first cell's ADetailer parameters in its images.
    from adetailer.args import ADetailerArgs

    runtime = _load_script(
        methods={
            "process", "is_ad_enabled", "set_skip_img2img", "get_args",
            "extra_params",
        },
        functions={"set_value"},
        opts=SimpleNamespace(data={}),
        is_img2img_inpaint=lambda _p: False,
        ADetailerArgs=ADetailerArgs,
        suffix=_ui_suffix(),
        __version__="test",
    )
    script = runtime.AfterDetailerScript()
    p = SimpleNamespace(
        prompt="a photo", negative_prompt="",
        extra_generation_params={"Hires upscale": 2},
    )
    tab = {
        "ad_model": "face_yolov8n.pt", "ad_prompt": "smiling face",
        "ad_controlnet_model": "control_v11p_sd15_inpaint",
    }

    cells = []
    for value in values:
        pc = copy(p)
        runtime.set_value(pc, value, values, field=field)
        script.process(pc, True, False, tab)
        cells.append(dict(pc.extra_generation_params))

    assert key in cells[0]
    assert key not in cells[1]
    assert cells[1]["Hires upscale"] == 2
    assert "ADetailer version" in cells[1]
    # Unchanged: running process again with the same settings (as the host's
    # re-run after the detailer pass does) keeps the keys and their order.
    single = SimpleNamespace(
        prompt="a photo", negative_prompt="",
        extra_generation_params={"Hires upscale": 2},
    )
    script.process(single, True, False, tab)
    single.extra_generation_params["Later"] = 1
    before = list(single.extra_generation_params.items())
    script.process(copy(single), True, False, tab)
    assert list(single.extra_generation_params.items()) == before


def test_an_empty_mask_in_an_inpaint_batch_does_not_stop_the_later_files():
    # The img2img Batch tab reuses p for every file: one empty mask switched
    # ADetailer off for every later file of the batch.
    from adetailer.mask import is_all_black

    runtime = _load_script(
        methods={"process", "postprocess_image"},
        opts=SimpleNamespace(data={}),
        is_img2img_inpaint=lambda p: p.image_mask is not None,
        is_all_black=is_all_black,
    )
    script = runtime.AfterDetailerScript()
    script.get_image_mask = lambda p: p.image_mask
    script.is_ad_enabled = lambda *_args: True
    script.set_skip_img2img = lambda *_args: None
    script.get_args = lambda *_args: []
    script.extra_params = lambda _args: {"ADetailer model": "face_yolov8n.pt"}

    class Reached(Exception):
        pass

    def reached(*_args):
        raise Reached

    script.get_i2i_init_image = reached
    p = SimpleNamespace(image_mask=None, extra_generation_params={})

    ran = []
    for value in (0, 255, 0, 255):
        p.image_mask = Image.new("L", (64, 64), value)
        p.extra_generation_params = {}
        script.process(p, True, False, {})
        try:
            script.postprocess_image(p, SimpleNamespace(image=None), True, False, {})
        except Reached:
            ran.append(bool(p.extra_generation_params))
        else:
            ran.append(False)

    assert ran == [False, True, False, True]
    assert not getattr(p, "_ad_disabled", False)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        # Prompt alternation: the "|" inside the brackets is not the separator.
        (
            "face: {smiling|laughing}, detailed face",
            ("{smiling|laughing}, detailed face", ""),
        ),
        (
            "face: [smiling|laughing], detailed face | blurry",
            ("[smiling|laughing], detailed face", "blurry"),
        ),
        ("face: smile | {blurry|ugly}", ("smile", "{blurry|ugly}")),
        # Unchanged: plain lines, a second "|", unbalanced or escaped brackets.
        ("face: detailed face | blurry, ugly", ("detailed face", "blurry, ugly")),
        ("face: a | b | c", ("a", "b | c")),
        ("face: (smile | frown", ("(smile", "frown")),
        ("face: \\(smile\\) | blurry", ("\\(smile\\)", "blurry")),
        ("face: detailed face", ("detailed face", "")),
    ],
)
def test_a_pipe_inside_brackets_stays_in_the_class_prompt(line, expected):
    # The line was split at the first "|", so "{smiling|laughing}" moved half
    # of the positive prompt into the negative one.
    runtime = _load_script(functions={"_parse_class_prompts"})

    assert runtime._parse_class_prompts(line) == {"face": expected}


# Final debugging pass, round 6: generation pipeline, prompts and docs.


@pytest.mark.parametrize(
    ("classes", "guarded"),
    [
        # No selected name is known: the one pass for every class uses the
        # tab prompt, so the per-class lines must not switch the guard off.
        (
            "faces,hands",
            [("face, main", "blurry, hand"), ("hand, main", "blurry, face")],
        ),
        # Unchanged: a known name's own line still wins over the guard.
        ("face,hands", [("detailed skin", "blurry")]),
        ("face,hand", [("detailed skin", "blurry"), ("five fingers", "blurry")]),
    ],
)
def test_a_filter_with_no_known_name_keeps_the_auto_class_guard(
    classes, guarded, tmp_path
):
    from adetailer.args import ADetailerArgs
    from adetailer.classes import get_model_class_names, resolve_class_ids

    model = tmp_path / "multi.pt"
    model.write_bytes(b"x")
    (tmp_path / "multi.names.json").write_bytes(b'["face","hand"]')
    runtime = _load_runtime(
        state=SimpleNamespace(interrupted=False, skipped=False),
        copy=copy, re=re,
        parse_csv=lambda text: text.split(","),
        is_skip_img2img=lambda _p: False,
        disable_safe_unpickle=nullcontext,
        get_model_class_names=get_model_class_names,
        resolve_class_ids=resolve_class_ids,
    )
    script = runtime.AfterDetailerScript()
    script.save_image = lambda *_args, **_kwargs: None
    script.get_ad_model = lambda _name: model
    passes = []

    def class_pass(_p, _pp, sub_args, **kwargs):
        passes.append((sub_args, kwargs["_seq_label"] is not None))
        return True

    script._postprocess_image_inner = class_pass  # each class's own pass
    args = ADetailerArgs(
        ad_model="multi.pt", ad_prompt="main", ad_negative_prompt="blurry",
        ad_model_classes=classes, ad_classes_sequential=True, ad_class_guard=True,
        ad_class_prompts="face: detailed skin\nhand: five fingers",
    )

    assert runtime.AfterDetailerScript._postprocess_image_inner(
        script, SimpleNamespace(), SimpleNamespace(image=Image.new("RGB", (8, 8))), args
    )

    # The pass with the whole filter finds a face and a hand, a class's own
    # pass only that class.
    guard = _class_prompt_script()
    regions = []
    for sub_args, seq_pass in passes:
        found = sub_args.ad_model_classes.split(",")
        found = ["face", "hand"] if len(found) > 1 else found
        for j in range(len(found)):
            p2 = SimpleNamespace(
                prompt=sub_args.ad_prompt, negative_prompt=sub_args.ad_negative_prompt
            )
            guard._apply_auto_class_guard(
                p2, sub_args, SimpleNamespace(class_names=found), j, len(found),
                seq_pass,
            )
            regions.append((p2.prompt, p2.negative_prompt))
    assert regions == guarded


def _manual_mode_script(data):
    from aaaaaa.p_method import is_skip_img2img, need_call_postprocess, need_call_process
    from adetailer.args import SkipImg2ImgOrig

    script = _load_script(
        methods={"process", "set_skip_img2img", "postprocess_image", "get_i2i_init_image"},
        opts=SimpleNamespace(data=data),
        is_img2img_inpaint=lambda _p: False,
        SkipImg2ImgOrig=SkipImg2ImgOrig,
        is_skip_img2img=is_skip_img2img,
        ensure_pil_image=lambda image, _mode: image,
        copy=copy,
        _verbose_gen_header=lambda *_args: None,
        need_call_postprocess=need_call_postprocess,
        need_call_process=need_call_process,
        Processed=lambda *_args: "dummy",
        preserve_prompts=lambda _p: nullcontext(),
        CNHijackRestore=nullcontext,
        pause_total_tqdm=nullcontext,
        cn_allow_script_control=nullcontext,
        _should_skip_for_hires_only=lambda _p, _args: False,
    ).AfterDetailerScript()
    tab = SimpleNamespace(need_skip=lambda: False)
    script.is_ad_enabled = lambda *_args: True
    script.get_args = lambda *_args: [tab]
    script.extra_params = lambda _args: {}
    script.read_params_txt = lambda: ""
    script.write_params_txt = lambda _content: None
    script._will_run_sequential = lambda _args: False
    script.save_image = lambda *_args, **_kwargs: None
    script._postprocess_image_inner = lambda _p, _pp, _args, n=0: True
    return script


def test_manual_mode_ticked_during_a_skip_img2img_job_keeps_the_init_image():
    # Manual mode ticked (Settings > Apply is not queue-locked) after the job
    # started: the later images kept the throwaway 128x128 one-step pass.
    data = {"ad_manual_mode": False}
    script = _manual_mode_script(data)
    init = Image.new("RGB", (1024, 768))
    p = SimpleNamespace(
        init_images=[init], width=1024, height=768, steps=30,
        sampler_name="DPM++ 2M", extra_generation_params={},
        batch_index=0, batch_size=1, seed=1, scripts=None,
    )
    script.process(p, True, True, {})
    assert (p.width, p.height) == (128, 128)

    data["ad_manual_mode"] = True
    pp = SimpleNamespace(image=Image.new("RGB", (128, 128)))
    script.postprocess_image(p, pp, True, True, {})

    assert pp.image.size == (1024, 768)


def test_manual_mode_ticked_during_a_batch_restarts_the_other_scripts():
    # Ticked while image 0 of a batch of 4 was detailed: image 0 had shut the
    # other scripts down, and the last image, which starts them again,
    # returned before doing so, so ControlNet stayed off for later batches.
    data = {"ad_manual_mode": False}
    script = _manual_mode_script(data)
    hooks = []
    runner = SimpleNamespace(
        postprocess=lambda *_args: hooks.append("postprocess"),
        before_process=lambda *_args: hooks.append("before_process"),
        process=lambda *_args: hooks.append("process"),
    )
    p = SimpleNamespace(extra_generation_params={}, batch_size=4, seed=1, scripts=runner)
    script.process(p, True, False, {})

    for index in range(4):
        p.batch_index = index
        script.postprocess_image(p, SimpleNamespace(image=Image.new("RGB", (8, 8))), True)
        data["ad_manual_mode"] = True

    assert hooks == ["postprocess", "before_process", "process"]


class _StyleDatabase:
    """The host's StyleDatabase with its merge rule: a style holding
    "{prompt}" wraps the prompt, any other one is appended to it."""

    styles = {"cinematic": ("cinematic still of {prompt}, 35mm film", "lowres, blurry")}

    @staticmethod
    def _merge(prompt, style):
        if "{prompt}" in style:
            return style.replace("{prompt}", prompt)
        return ", ".join(filter(None, (prompt.strip(), style.strip())))

    def apply_styles_to_prompt(self, prompt, styles):
        for name in styles:
            prompt = self._merge(prompt, self.styles[name][0])
        return prompt

    def apply_negative_styles_to_prompt(self, prompt, styles):
        for name in styles:
            prompt = self._merge(prompt, self.styles[name][1])
        return prompt


@pytest.mark.parametrize(
    ("prompt", "negative", "expected"),
    [
        # A blank prompt gives the region the main image's own prompts.
        (
            "", "",
            ("cinematic still of a woman in a park, 35mm film", "bad hands, lowres, blurry"),
        ),
        (
            "[PROMPT], detailed face", "",
            (
                "cinematic still of a woman in a park, detailed face, 35mm film",
                "bad hands, lowres, blurry",
            ),
        ),
        # Unchanged: a prompt of the tab's own is styled once, as before.
        (
            "detailed face", "ugly",
            ("cinematic still of detailed face, 35mm film", "ugly, lowres, blurry"),
        ),
        (
            "detailed face", "",
            ("cinematic still of detailed face, 35mm film", "bad hands, lowres, blurry"),
        ),
    ],
)
def test_the_selected_styles_reach_the_detailer_pass_once(prompt, negative, expected):
    # The host applies the styles to the main prompts and again to the
    # detailer pass, which took a blank or [PROMPT] segment from the styled
    # main prompt: "cinematic still of cinematic still of ...".
    from adetailer.args import ADetailerArgs

    db = _StyleDatabase()
    runtime = _load_script(
        methods=_INLINE_PROMPT_METHODS,
        functions=_INLINE_PROMPT_FUNCTIONS,
        assigns=_INLINE_PROMPT_ASSIGNS,
        get_i=lambda _p: 0,
        shared=SimpleNamespace(prompt_styles=db),
    )
    script = runtime.AfterDetailerScript()
    styles = ["cinematic"]
    p = SimpleNamespace(
        prompt="a woman in a park", negative_prompt="bad hands", styles=styles,
        all_prompts=[db.apply_styles_to_prompt("a woman in a park", styles)],
        all_negative_prompts=[db.apply_negative_styles_to_prompt("bad hands", styles)],
    )
    args = ADetailerArgs(
        ad_model="face_yolov8n.pt", ad_prompt=prompt, ad_negative_prompt=negative
    )

    prompts, negatives = script.get_prompt(p, args)
    # What the detailer pass gets once the host has applied the styles.
    assert (
        db.apply_styles_to_prompt(prompts[0], styles),
        db.apply_negative_styles_to_prompt(negatives[0], styles),
    ) == expected

    # Unchanged: without styles, or when another extension rewrote the main
    # prompts so that the styles no longer give them back, the main prompts
    # are used as they are.
    for selected, main in (([], p.all_prompts[0]), (styles, "a garden")):
        p.styles, p.all_prompts = selected, [main]
        assert script.get_prompt(p, args)[0] == [prompt.replace("[PROMPT]", main) or main]


@pytest.mark.parametrize(
    ("ad_prompt", "main", "sr", "fields", "record"),
    [
        (
            "[SKIP] [SEP] smiling face", "a photo", ("smiling", "laughing"),
            {
                "ad_prompt_append": "masterpiece", "ad_negative_prompt": "ugly",
                "ad_negative_prompt_append": "blurry",
            },
            "[SKIP] [SEP] laughing face",
        ),
        (
            "detailed smiling face", "a photo", ("smiling", "laughing"),
            {"ad_prompt_append": "masterpiece"}, "detailed laughing face",
        ),
        (
            "", "portrait smiling", ("smiling", "laughing"),
            {"ad_prompt_append": "masterpiece"}, "portrait laughing",
        ),
        # The S/R also changes the main prompt a blank prompt stands for.
        (
            "", "a photo, blue eyes <lora:detail:1>", ("blue", "red"),
            {"ad_strip_loras": True, "ad_use_main_loras": True},
            "a photo, red eyes <lora:detail:1>",
        ),
    ],
)
def test_an_xyz_prompt_search_replace_cell_pastes_back_the_same_prompts(
    ad_prompt, main, sr, fields, record
):
    # The saved "ADetailer prompt" was the first [SEP] segment with the append
    # text already in it: pasted back, the append text doubled and the other
    # segments, a [SKIP] included, were lost.
    from aaaaaa.p_method import get_i
    from adetailer.args import ADetailerArgs

    runtime = _load_script(
        methods=_INLINE_PROMPT_METHODS | {"process", "extra_params", "_record_xyz_prompt_sr"},
        functions=_INLINE_PROMPT_FUNCTIONS,
        assigns=_INLINE_PROMPT_ASSIGNS,
        opts=SimpleNamespace(data={}),
        is_img2img_inpaint=lambda _p: False,
        get_i=get_i,
        suffix=_ui_suffix(),
        __version__="test",
    )
    script = runtime.AfterDetailerScript()
    script.is_ad_enabled = lambda *_args: True
    script.set_skip_img2img = lambda *_args: None
    tab = {"ad_model": "face_yolov8n.pt", "ad_prompt": ad_prompt, **fields}
    script.get_args = lambda *_args: [ADetailerArgs(**tab)]

    def photo(**extra):
        return SimpleNamespace(
            iteration=0, batch_size=1, prompt=main, negative_prompt="",
            all_prompts=[main], all_negative_prompts=[""],
            extra_generation_params={}, **extra,
        )

    p = photo(_ad_xyz_prompt_sr=[SimpleNamespace(s=sr[0], r=sr[1])])
    script.process(p, True, False, {})

    params = p.extra_generation_params
    assert params["ADetailer prompt"] == record
    pasted = ADetailerArgs(
        **{
            **tab,
            "ad_prompt": params.get("ADetailer prompt", ""),
            "ad_negative_prompt": params.get("ADetailer negative prompt", ""),
        }
    )
    assert script.get_prompt(photo(), pasted) == script.get_prompt(p, ADetailerArgs(**tab))


def test_a_file_with_an_empty_mask_is_not_saved_with_the_previous_files_parameters():
    # The img2img Batch tab reuses p, and its parameters, for every file: a
    # file whose mask is empty, which ADetailer skips, kept the previous
    # file's ADetailer parameters and was saved as if detailed.
    from adetailer.args import ADetailerArgs
    from adetailer.mask import is_all_black

    runtime = _load_script(
        methods={
            "process", "is_ad_enabled", "set_skip_img2img", "get_args",
            "extra_params",
        },
        opts=SimpleNamespace(data={}),
        is_img2img_inpaint=lambda p: p.image_mask is not None,
        is_all_black=is_all_black,
        ADetailerArgs=ADetailerArgs,
        suffix=_ui_suffix(),
        __version__="test",
    )
    script = runtime.AfterDetailerScript()
    script.get_image_mask = lambda p: p.image_mask
    p = SimpleNamespace(
        prompt="a photo", negative_prompt="", image_mask=None,
        extra_generation_params={"Mask blur": 4},
    )
    tab = {"ad_model": "face_yolov8n.pt", "ad_prompt": "detailed face"}

    files = []
    for value in (255, 0, 255):
        p.image_mask = Image.new("L", (64, 64), value)
        script.process(p, True, False, tab)
        files.append(dict(p.extra_generation_params))

    assert files[0]["ADetailer model"] == "face_yolov8n.pt"
    assert not [k for k in files[1] if k.startswith("ADetailer ")]
    assert files[1]["Mask blur"] == 4
    assert files[2]["ADetailer model"] == "face_yolov8n.pt"


@pytest.mark.parametrize(
    ("prompt", "negative", "append", "expected"),
    [
        # A box that looks empty (a stray newline or space) means the main
        # prompt, as the "If blank" placeholder says.
        ("\n", "\n", "", (["portrait, smiling"], ["blurry"])),
        (" ", " ", "masterpiece", (["portrait, smiling, masterpiece"], ["blurry"])),
        ("  \n ", "", "", (["portrait, smiling"], ["blurry"])),
        # Unchanged.
        ("detailed face", "", "", (["detailed face"], ["blurry"])),
        ("face [SEP] ", "", "", (["face", "portrait, smiling"], ["blurry"])),
        ("[SKIP]", "", "masterpiece", (["[SKIP]"], ["blurry"])),
        (",", "", "", ([","], ["blurry"])),
    ],
)
def test_a_prompt_of_only_spaces_means_the_main_prompt(prompt, negative, append, expected):
    from adetailer.args import ADetailerArgs

    runtime = _load_script(
        methods=_INLINE_PROMPT_METHODS,
        functions=_INLINE_PROMPT_FUNCTIONS,
        assigns=_INLINE_PROMPT_ASSIGNS,
        get_i=lambda _p: 0,
    )
    p = SimpleNamespace(
        prompt="portrait, smiling", all_prompts=["portrait, smiling"],
        negative_prompt="blurry", all_negative_prompts=["blurry"],
    )
    args = ADetailerArgs(
        ad_model="face_yolov8n.pt", ad_prompt=prompt, ad_negative_prompt=negative,
        ad_prompt_append=append,
    )

    assert runtime.AfterDetailerScript().get_prompt(p, args) == expected


def test_readme_says_when_a_folder_run_reports_a_missing_detector():
    # The detector is checked on the first image that opens, not before any
    # file is opened.
    readme = _public_docs()["README.md"]
    assert "reported before any file is opened" not in readme
    assert "at the first image that opens" in readme


def _real_pause_total_tqdm(data, defaults=None):
    from contextlib import contextmanager

    helper = _SCRIPT_PATH.parents[1] / "aaaaaa" / "helper.py"
    tree = ast.parse(helper.read_text(encoding="utf-8"))
    nodes = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "pause_total_tqdm")
        or (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", "") == "_AD_OVERRIDE_KEYS" for t in node.targets)
        )
    ]
    opts = SimpleNamespace(data=data)
    if defaults is not None:
        opts.get_default = defaults.get
    namespace = {"contextmanager": contextmanager, "opts": opts}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(helper), "exec"), namespace)
    return namespace["pause_total_tqdm"]


@pytest.mark.parametrize("saved", [False, True])
def test_standalone_run_leaves_no_override_option_behind(tmp_path, saved):
    # The host puts an override option back after the pass only if it was
    # already in opts.data: without the cleanup of normal generation, a tab's
    # Clip skip and VAE stayed set for every later generation.
    data = {"multiple_tqdm": True, "sd_model_checkpoint": "base.safetensors"}
    if saved:
        data["CLIP_stop_at_last_layers"] = 1
    before = dict(data)
    script = _standalone_runtime(
        tmp_path, pause_total_tqdm=_real_pause_total_tqdm(data)
    ).AfterDetailerScript()
    during = []

    def inner(_p, _pp, _args, n=0):
        override = {"CLIP_stop_at_last_layers": 2, "sd_vae": "detail-vae.safetensors"}
        stored = {k: data[k] for k in override if k in data}
        data.update(override)
        during.append(data["multiple_tqdm"])
        data.update(stored)
        return True

    script._postprocess_image_inner = inner

    image, _status = script.run_detailer_on_image(
        Image.new("RGB", (64, 64)), SimpleNamespace(), save=False
    )

    assert image is not None
    assert during == [False]
    assert data == before


def test_the_applied_prompt_line_prints_on_a_legacy_console(monkeypatch):
    # The "applied ad_prompt" line went through rich with repr(): a prompt
    # with characters a legacy code page lacks made it raise, which dropped
    # the finished region and every later tab for the image. rich also read
    # "[cat:dog:0.5]" as markup and left it out.
    stdout = io.TextIOWrapper(
        io.BytesIO(), encoding="cp1252", errors="strict", write_through=True
    )
    monkeypatch.setattr(sys, "stdout", stdout)
    runtime = _load_script(methods={"compare_prompt"}, ordinal=str, suffix=_ui_suffix())

    runtime.AfterDetailerScript.compare_prompt(
        {"ADetailer prompt": "smile, 笑顔", "ADetailer negative prompt": "blurry"},
        SimpleNamespace(
            all_prompts=["smile, 笑顔, <lora:detail:1>"],
            all_negative_prompts=["blurry, [cat:dog:0.5], \U0001f600"],
        ),
    )

    out = stdout.buffer.getvalue().decode("ascii")
    assert "ad_prompt: 'smile, \\u7b11\\u9854, <lora:detail:1>'" in out
    assert "ad_negative_prompt: 'blurry, [cat:dog:0.5], \\U0001f600'" in out


@pytest.mark.parametrize("tab", [0, 1])
def test_a_standalone_run_prints_no_applied_prompt_line(tmp_path, capsys, tab):
    # From beta 3 the standalone pass carries the tab's "ADetailer ..."
    # parameters, and its inner pass, which ran as the 1st tab (n=0) until the
    # stable plus.8, compared their "ADetailer prompt" with the prompt used: a run from the
    # 1st tab printed "applied 1st ad_prompt: ...", while a run from the 2nd
    # tab, whose key is "ADetailer prompt 2nd", printed nothing. As in beta 2,
    # these runs print no such line; a generation still prints it.
    from adetailer.args import ADetailerArgs

    sfx = _ui_suffix()
    state = SimpleNamespace(
        interrupted=False, skipped=False, stopping_generation=False,
        job_count=0, assign_current_image=lambda _image: None,
    )
    image = Image.new("RGB", (64, 64), "white")
    result = Image.new("RGB", (64, 64), "red")
    runtime = _load_script(
        methods={
            "run_detailer_on_image", "read_params_txt", "write_params_txt",
            "_postprocess_image_inner", "compare_prompt", "extra_params",
        },
        paths=SimpleNamespace(data_path=str(tmp_path)),
        PARAMS_TXT="params.txt",
        shared=SimpleNamespace(
            sd_model=None, state=state, opts=SimpleNamespace(data={})
        ),
        state=state,
        opts=SimpleNamespace(samples_format="png"),
        all_samplers=[],
        images=None,
        AD_APPLY_SUBDIR="ADetailer-Inpaint",
        pause_total_tqdm=nullcontext,
        time=time, copy=copy, get_i=lambda _p: 0,
        parse_csv=lambda text: text.split(","),
        is_skip_img2img=lambda _p: False,
        _ad_verbose=lambda: False,
        _verbose_pass_header=lambda *_args: None,
        _verbose_detection=lambda *_args: None,
        disable_safe_unpickle=nullcontext,
        ultralytics_predict=lambda *_args, **_kwargs: SimpleNamespace(preview=image),
        ensure_pil_image=lambda im, _mode: im,
        process_images=lambda _p2: SimpleNamespace(
            images=[result], all_prompts=["detailed face, sharp eyes"],
            all_negative_prompts=["blurry"],
        ),
        NansException=type("NansException", (Exception,), {}),
        ordinal=sfx.__globals__["ordinal"],
        suffix=sfx,
        __version__="test",
        Image=Image,
    )
    script = runtime.AfterDetailerScript()
    script.ultralytics_device = "cpu"
    shells = []

    def get_i2i_p(p, _args, im):
        shells.append(p)
        return SimpleNamespace(
            init_images=[im], prompt="detailed face", negative_prompt="",
            close=lambda: None, denoising_strength=0.4, width=64, height=64,
        )

    script.get_i2i_p = get_i2i_p
    script.get_prompt = lambda *_args: (["detailed face"], [""])
    script.get_ad_model = lambda _name: "model.pt"
    script.pred_preprocessing = lambda *_args: [Image.new("L", (64, 64), 255)]
    saved = []
    script.save_image = lambda *_args, **kwargs: saved.append(kwargs.get("suffix"))
    script.i2i_prompts_replace = lambda *_args: None
    script._apply_inline_class_prompts = lambda *_args: None
    script._apply_auto_class_guard = lambda *_args: None
    script.fix_p2 = lambda *_args: None
    args = ADetailerArgs(
        ad_model="face_yolov8n.pt", ad_prompt="detailed face",
        ad_negative_prompt="blurry, low quality",
    )

    _image, status = script.run_detailer_on_image(image, args, save=False, tab=tab)

    assert status == "✅ ADetailer pass complete."
    # The preview file names the tab the run was started from.
    assert saved == [f"-ad-preview-{tab + 1}"]
    # Premise: the pass carried the tab's prompts under the tab's own keys.
    params = shells[0].extra_generation_params
    assert params["ADetailer prompt" + sfx(tab)] == "detailed face"
    assert params["ADetailer negative prompt" + sfx(tab)] == "blurry, low quality"
    assert "applied" not in capsys.readouterr().out

    # A generation, here in its 2nd tab, prints it as before.
    generation = SimpleNamespace(extra_generation_params={
        "ADetailer prompt 2nd": "detailed face",
        "ADetailer negative prompt 2nd": "blurry, low quality",
    })
    assert script._postprocess_image_inner(
        generation, SimpleNamespace(image=image), args, n=1
    )
    out = capsys.readouterr().out
    assert "[-] ADetailer: applied 2nd ad_prompt: 'detailed face, sharp eyes'" in out
    assert "[-] ADetailer: applied 2nd ad_negative_prompt: 'blurry'" in out


@pytest.mark.parametrize("outcome", ["detailed", "nothing", "error", "no record"])
def test_standalone_run_leaves_no_params_txt_when_there_was_none(tmp_path, outcome):
    # Before the first generation there is no params.txt: the inner pass
    # created one, and the paste button then filled in its blank prompt,
    # 28 steps and random seed.
    params_txt = tmp_path / "params.txt"
    script = _standalone_runtime(tmp_path).AfterDetailerScript()

    def inner(_p, _pp, _args, n=0):
        if outcome != "no record":  # --no-prompt-history writes none
            params_txt.write_text("inner pass", encoding="utf-8")
        if outcome == "error":
            msg = "CUDA out of memory"
            raise RuntimeError(msg)
        return outcome != "nothing"

    script._postprocess_image_inner = inner

    image, status = script.run_detailer_on_image(
        Image.new("RGB", (64, 64)), SimpleNamespace(), save=False
    )

    assert not params_txt.exists()
    assert ("failed" in status) == (outcome == "error")
    assert (image is None) == (outcome == "error")


@pytest.mark.parametrize("resets_size", [False, True])
@pytest.mark.parametrize("skip", [True, False])
def test_manual_mode_ticked_during_an_img2img_batch_keeps_detailing_its_files(
    skip, resets_size
):
    # The img2img Batch tab and Loopback reuse p and call process() for every
    # file: manual mode ticked after the first file switched ADetailer off for
    # the rest, and with Skip img2img they were saved as the one-step 128x128
    # throwaway pass (at the right size where the host puts it back per file).
    data = {"ad_manual_mode": False}
    script = _manual_mode_script(data)
    p = SimpleNamespace(
        init_images=None, width=1024, height=768, steps=30,
        sampler_name="DPM++ 2M", extra_generation_params={},
        batch_index=0, batch_size=1, seed=1, scripts=None,
    )
    for _file in range(3):
        if resets_size:
            p.width, p.height = 1024, 768
        p.init_images = [Image.new("RGB", (1024, 768))]
        script.process(p, True, skip, {})
        pp = SimpleNamespace(image=Image.new("RGB", (p.width, p.height)))
        script.postprocess_image(p, pp, True, skip, {})
        assert pp.image.size == (1024, 768)
        assert not getattr(p, "_ad_disabled", False)
        data["ad_manual_mode"] = True
    if skip:
        assert (p._ad_orig.steps, p._ad_orig.sampler_name) == (30, "DPM++ 2M")

    # Unchanged: the next job reads the setting again, before Skip img2img.
    job = SimpleNamespace(
        init_images=[Image.new("RGB", (1024, 768))], width=1024, height=768,
        steps=30, sampler_name="DPM++ 2M", extra_generation_params={},
    )
    script.process(job, True, skip, {})
    assert job._ad_disabled
    assert (job.width, job.height, job.steps) == (1024, 768, 30)


@pytest.mark.parametrize(
    ("hand_prompt", "hand_negative"),
    [("", ""), ("blue gloves", "blue nails"), ("[SKIP] [SEP] blue gloves", "")],
)
def test_an_xyz_prompt_search_replace_cell_pastes_back_every_tabs_prompts(
    hand_prompt, hand_negative
):
    # The Prompt S/R axis replaces the text in every tab's detailer prompt,
    # but only the 1st tab's replaced prompt was saved: pasted back, the
    # other tabs ran with the original text.
    from aaaaaa.p_method import get_i
    from adetailer.args import ADetailerArgs

    suffix = _ui_suffix()
    runtime = _load_script(
        methods=_INLINE_PROMPT_METHODS | {"process", "extra_params", "_record_xyz_prompt_sr"},
        functions=_INLINE_PROMPT_FUNCTIONS,
        assigns=_INLINE_PROMPT_ASSIGNS,
        opts=SimpleNamespace(data={}),
        is_img2img_inpaint=lambda _p: False,
        get_i=get_i,
        suffix=suffix,
        __version__="test",
    )
    script = runtime.AfterDetailerScript()
    script.is_ad_enabled = lambda *_args: True
    script.set_skip_img2img = lambda *_args: None
    main = "a woman, blue eyes, blue dress"
    tabs = [
        {"ad_model": "face_yolov8n.pt", "ad_prompt_append": "masterpiece"},
        {
            "ad_model": "hand_yolov8n.pt", "ad_prompt": hand_prompt,
            "ad_negative_prompt": hand_negative, "ad_prompt_append": "detailed",
        },
        # Switched off: saves nothing, as before.
        {"ad_model": "None", "ad_prompt": "blue hat"},
    ]
    script.get_args = lambda *_args: [ADetailerArgs(**tab) for tab in tabs]

    def photo(**extra):
        return SimpleNamespace(
            iteration=0, batch_size=1, prompt=main, negative_prompt="",
            all_prompts=[main], all_negative_prompts=[""],
            extra_generation_params={}, **extra,
        )

    p = photo(_ad_xyz_prompt_sr=[SimpleNamespace(s="blue", r="green")])
    script.process(p, True, False, {})

    params = p.extra_generation_params
    for n, tab in enumerate(tabs[:2]):
        pasted = ADetailerArgs(
            **{
                **tab,
                "ad_prompt": params.get("ADetailer prompt" + suffix(n), ""),
                "ad_negative_prompt": params.get(
                    "ADetailer negative prompt" + suffix(n), ""
                ),
            }
        )
        generated = script.get_prompt(p, ADetailerArgs(**tab))
        assert "blue" not in " ".join(generated[0] + generated[1])
        assert script.get_prompt(photo(), pasted) == generated
    assert not [k for k in params if k.endswith(suffix(2))]


class _LoraStyleDatabase(_StyleDatabase):
    styles = {
        **_StyleDatabase.styles,
        "detail": ("<lora:detail_tweaker:1>, masterpiece", "<lora:blur:1>, lowres"),
    }


@pytest.mark.parametrize("nan_first", [False, True])
@pytest.mark.parametrize("strip", [True, False])
def test_strip_loras_also_strips_the_loras_of_the_selected_styles(strip, nan_first):
    # The host applies the selected styles to the detailer pass after
    # ADetailer's Strip LoRAs, so a style's <lora:...> reached every region.
    from adetailer.args import ADetailerArgs

    db = _LoraStyleDatabase()
    state = SimpleNamespace(
        interrupted=False, skipped=False, job_count=0,
        assign_current_image=lambda _image: None,
    )
    image = Image.new("RGB", (8, 8), "white")
    mask = Image.new("L", (8, 8), 255)
    pred = SimpleNamespace(preview=image, bboxes=[], class_names=[])
    nans = type("NansException", (Exception,), {})
    sent = []

    def process_images(p2):
        # What the host generates with: the styles left on p2 applied.
        sent.append(
            (
                db.apply_styles_to_prompt(p2.prompt, p2.styles),
                db.apply_negative_styles_to_prompt(p2.negative_prompt, p2.styles),
            )
        )
        if nan_first and len(sent) == 1:
            raise nans("NaN in the first region")
        return SimpleNamespace(images=[image])

    runtime = _load_script(
        methods={"_postprocess_image_inner", "i2i_prompts_replace"},
        functions={"_strip_lora_tags"},
        assigns={"_LORA_TAG_RE"},
        state=state,
        shared=SimpleNamespace(state=state, prompt_styles=db),
        time=time, copy=copy, get_i=lambda _p: 0,
        parse_csv=lambda text: text.split(","),
        is_skip_img2img=lambda _p: False,
        _ad_verbose=lambda: False,
        _verbose_pass_header=lambda *_args: None,
        _verbose_detection=lambda *_args: None,
        disable_safe_unpickle=nullcontext,
        ultralytics_predict=lambda *_args, **_kwargs: pred,
        ensure_pil_image=lambda im, _mode: im,
        process_images=process_images,
        NansException=nans,
        ordinal=str,
    )
    script = runtime.AfterDetailerScript()
    script.ultralytics_device = "cpu"
    script.get_i2i_p = lambda *_args: SimpleNamespace(
        init_images=[image], prompt="", negative_prompt="", styles=["detail"],
        close=lambda: None, denoising_strength=0.4, width=512, height=512,
    )
    # A blank prompt resolves to the typed main prompt, its LoRA stripped.
    script.get_prompt = lambda *_args: (["a woman"], ["bad hands"])
    script.get_ad_model = lambda _name: "model.pt"
    script.pred_preprocessing = lambda *_args: [mask, mask]
    script.save_image = lambda *_args, **_kwargs: None
    script._apply_inline_class_prompts = lambda *_args: None
    script._apply_auto_class_guard = lambda *_args: None
    script.fix_p2 = lambda *_args: None
    script.compare_prompt = lambda *_args, **_kwargs: None

    processed = script._postprocess_image_inner(
        SimpleNamespace(extra_generation_params={}),
        SimpleNamespace(image=image),
        ADetailerArgs(ad_model="face_yolov8n.pt", ad_strip_loras=strip),
    )

    assert processed is True
    if strip:
        expected = ("a woman, masterpiece", "bad hands, lowres")
    else:  # Unchanged: the host applies the styles, LoRAs included.
        expected = (
            "a woman, <lora:detail_tweaker:1>, masterpiece",
            "bad hands, <lora:blur:1>, lowres",
        )
    assert sent == [expected, expected]


@pytest.mark.parametrize("vae", ["sdxl_vae.safetensors", "Automatic"])
@pytest.mark.parametrize("forge", [True, False])
def test_a_separate_vae_on_forge_is_not_sent_as_sd_vae_before_the_list_is_saved(
    forge, vae
):
    # Forge writes its VAE / text-encoder list to opts.data only once a
    # module is picked at the top of the page. Until then ADetailer took the
    # host for AUTOMATIC1111 and sent the separate VAE as sd_vae, a bare name
    # Forge Neo fails to load.
    from adetailer.args import ADetailerArgs

    data = {"multiple_tqdm": True}
    defaults = {"CLIP_stop_at_last_layers": 1, "sd_vae": "Automatic"}
    if forge:
        defaults["forge_additional_modules"] = []
    runtime = _load_script(
        methods={"get_override_settings"},
        functions={"_is_forge_modules"},
        shared=SimpleNamespace(opts=SimpleNamespace(data=data)),
        opts=SimpleNamespace(data=data),
        _forge_wanted_modules=lambda _args: None,
    )
    args = ADetailerArgs(ad_model="face_yolov8n.pt", ad_use_vae=True, ad_vae=vae)

    with _real_pause_total_tqdm(data, defaults)():
        override = runtime.AfterDetailerScript().get_override_settings(None, args)

    # Unchanged on AUTOMATIC1111, which has no such list.
    assert ("sd_vae" in override) is not forge
    assert data == {"multiple_tqdm": True}


def test_the_changelog_shows_the_backslash_escape_it_describes():
    # The beta 2 entry gave the raw character as its example of the escape
    # the console lines now write instead of it.
    changelog = _public_docs()["CHANGELOG.md"]
    beta2 = changelog.split("## v26.2.0+plus.8.beta.1", 1)[0]
    marker = "backslash escapes (for example `"
    line = next(text for text in beta2.splitlines() if marker in text)
    example = line.split(marker, 1)[1].split("`", 1)[0]
    assert example.isascii()
    assert example.startswith(chr(92) + "u")


def test_the_docs_say_what_a_folder_run_without_detector_reports():
    # The unreadable files before the first image that opens are not listed:
    # only "Pick a detector model first." is.
    docs = _public_docs()
    readme, changelog = docs["README.md"], docs["CHANGELOG.md"]
    assert "before it that cannot be opened are counted as unreadable" not in readme
    assert "cannot be opened before it are counted as unreadable" not in changelog
    assert "if no image in the folder can be opened" in readme


def test_the_readme_says_mediapipe_face_full_uses_the_short_range_model():
    # MediaPipe's tasks API has only the short-range face model, so
    # mediapipe_face_full gives the same result as mediapipe_face_short.
    readme = _public_docs()["README.md"]
    assert any(
        "`mediapipe_face_full`" in line and "short-range" in line
        for line in readme.splitlines()
    )


# Final debugging pass, round 8: generation pipeline, prompts and docs.


@pytest.mark.parametrize(
    ("only_masked", "registered", "saved", "forced"),
    [
        # "Overlay original for inpaint" unticked: the host returned only the
        # region's canvas, which replaced the picture and fed the next region.
        (True, True, {"overlay_inpaint": False}, True),
        # Unchanged: the option on or at its default, a whole-picture pass,
        # which comes back full size anyway, and a host without the option
        # (AUTOMATIC1111 before 1.8), which cannot set it, also when the
        # settings file keeps a value saved by a newer one.
        (True, True, {"overlay_inpaint": True}, False),
        (True, True, {}, False),
        (False, True, {"overlay_inpaint": False}, False),
        (True, False, {"overlay_inpaint": False}, False),
    ],
)
def test_an_only_masked_pass_keeps_the_host_overlay_on(
    only_masked, registered, saved, forced
):
    from adetailer.args import ADetailerArgs

    data = dict(saved)
    labels = {"overlay_inpaint": object()} if registered else {}
    script = _i2i_script(
        methods={"get_override_settings"},
        opts=SimpleNamespace(data=data, data_labels=labels),
        _is_forge_modules=lambda: False,
        _forge_wanted_modules=lambda _args: None,
    )
    args = ADetailerArgs(ad_model="face_yolov8n.pt", ad_inpaint_only_masked=only_masked)

    i2i = script.get_i2i_p(_txt2img_p(), args, Image.new("RGB", (512, 768)))

    assert i2i.override_settings.get("overlay_inpaint") is (True if forced else None)
    # The host puts back after the pass only the options in opts.data; the
    # user's own setting is not touched here.
    assert set(i2i.override_settings) <= set(data)
    assert data == saved


def test_region_after_a_nan_error_gets_its_own_color_correction():
    # With "Apply color correction to img2img results" the host builds the
    # correction from the region's crop only while p has none, and clears it
    # only after a pass that succeeds: after a NaN error the next region (a
    # hand, say) was matched to the colours of the failed region's crop.
    from adetailer.args import ADetailerArgs, InpaintBBoxMatchMode
    from adetailer.opts import dynamic_denoise_strength, optimal_crop_size

    state = SimpleNamespace(
        interrupted=False, skipped=False, job_count=0,
        assign_current_image=lambda _image: None,
    )
    nans = type("NansException", (Exception,), {})
    image = Image.new("RGB", (1000, 1000), "white")
    mask = Image.new("L", (1000, 1000), 0)
    pred = SimpleNamespace(
        preview=image,
        bboxes=[[0, 0, 316, 316], [500, 500, 816, 816]],
        class_names=[],
    )
    seen = []

    def process_images(p2):
        if p2.color_corrections is None:  # the host's init()
            p2.color_corrections = [f"region {len(seen) + 1}"]
        seen.append(p2.color_corrections[0])
        if len(seen) == 1:
            raise nans("NaN in the first region")
        p2.color_corrections = None  # after the host's batch loop
        return SimpleNamespace(images=[image])

    runtime = _load_script(
        methods={
            "_postprocess_image_inner", "fix_p2", "get_dynamic_denoise_strength",
            "get_optimal_crop_image_size", "get_seed", "get_each_tab_seed",
        },
        state=state,
        shared=SimpleNamespace(
            state=state, opts=SimpleNamespace(data={}), sd_model=None
        ),
        opts=SimpleNamespace(
            data={"ad_match_inpaint_bbox_size": InpaintBBoxMatchMode.OFF.value}
        ),
        time=time, copy=copy, get_i=lambda _p: 0,
        parse_csv=lambda text: text.split(","),
        is_skip_img2img=lambda _p: False,
        _ad_verbose=lambda: False,
        _verbose_pass_header=lambda *_args: None,
        _verbose_detection=lambda *_args: None,
        disable_safe_unpickle=nullcontext,
        ultralytics_predict=lambda *_args, **_kwargs: pred,
        ensure_pil_image=lambda im, _mode: im,
        process_images=process_images,
        NansException=nans,
        ordinal=str,
        InpaintBBoxMatchMode=InpaintBBoxMatchMode,
        dynamic_denoise_strength=dynamic_denoise_strength,
        optimal_crop_size=optimal_crop_size,
    )
    script = runtime.AfterDetailerScript()
    script.ultralytics_device = "cpu"
    script.get_i2i_p = lambda *_args: SimpleNamespace(
        init_images=[image], prompt="face", negative_prompt="",
        close=lambda: None, denoising_strength=0.4, width=512, height=512,
        color_corrections=None,
    )
    script.get_prompt = lambda *_args: (["face"], [""])
    script.get_ad_model = lambda _name: "model.pt"
    script.pred_preprocessing = lambda *_args: [mask, mask]
    script.save_image = lambda *_args, **_kwargs: None
    script.i2i_prompts_replace = lambda *_args: None
    script._apply_inline_class_prompts = lambda *_args: None
    script._apply_auto_class_guard = lambda *_args: None
    script.compare_prompt = lambda *_args, **_kwargs: None
    p = SimpleNamespace(
        extra_generation_params={}, seed=1, subseed=1, all_seeds=[1], all_subseeds=[1]
    )

    processed = script._postprocess_image_inner(
        p, SimpleNamespace(image=image), ADetailerArgs(ad_model="face_yolov8n.pt")
    )

    assert processed is True
    assert seen == ["region 1", "region 2"]


@pytest.mark.parametrize("skip_img2img", [True, False])
def test_an_xyz_cell_switched_to_manual_mode_keeps_no_earlier_cells_parameters(
    skip_img2img,
):
    # Every X/Y/Z cell is a copy of p sharing its extra params. A cell that
    # finds manual mode ticked mid-grid returned before the cleanup: its image,
    # not detailed, was saved with the previous cell's ADetailer model and,
    # with Skip img2img, that cell's Steps, Sampler and Size.
    from adetailer.args import SkipImg2ImgOrig

    data = {"ad_manual_mode": False}
    runtime = _load_runtime(
        opts=SimpleNamespace(data=data),
        is_img2img_inpaint=lambda _p: False,
        SkipImg2ImgOrig=SkipImg2ImgOrig,
    )
    script = runtime.AfterDetailerScript()
    script.is_ad_enabled = lambda *_args: True
    script.get_args = lambda p, *_args: [p._ad_cell_model]
    script.extra_params = lambda arg_list: {"ADetailer model": arg_list[0]}
    p = SimpleNamespace(
        init_images=[object()], steps=20, sampler_name="DPM++ 2M", width=832,
        height=1216, extra_generation_params={"Hires upscale": 2},
    )
    first = copy(p)
    first._ad_cell_model = "face_yolov8n.pt"
    script.process(first, True, skip_img2img, {})
    assert p.extra_generation_params["ADetailer model"] == "face_yolov8n.pt"
    assert ("Steps" in p.extra_generation_params) is skip_img2img

    data["ad_manual_mode"] = True  # ticked while the grid runs
    cell = copy(p)
    cell.steps = 40
    cell._ad_cell_model = "hand_yolov8n.pt"
    script.process(cell, True, skip_img2img, {})

    assert cell._ad_disabled
    assert (cell.steps, cell.width, cell.height) == (40, 832, 1216)
    # Other keys of the shared dict stay.
    assert cell.extra_generation_params == {"Hires upscale": 2}


def _host_flatten(img, bgcolor):
    """The WebUI's images.flatten (AUTOMATIC1111 and Forge Neo)."""
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, bgcolor)
        background.paste(img, mask=img)
        img = background
    return img.convert("RGB")


@pytest.mark.parametrize(
    ("host", "corner"),
    [
        ({"images": SimpleNamespace(flatten=_host_flatten), "bg": "#ffffff"}, (255, 255, 255)),
        ({"images": SimpleNamespace(flatten=_host_flatten), "bg": "#808080"}, (128, 128, 128)),
        # A host without them: the alpha channel is dropped, as before.
        ({"images": SimpleNamespace(), "bg": "#ffffff"}, (0, 255, 0)),
        ({"images": SimpleNamespace(flatten=_host_flatten), "bg": None}, (0, 255, 0)),
    ],
)
def test_skip_img2img_fills_a_transparent_init_image_like_the_host(host, corner):
    # The host fills the transparent parts with its img2img background colour
    # (white on AUTOMATIC1111, grey on Forge Neo) only in a copy of its own;
    # with Skip img2img they came out in the colour stored under the alpha.
    from adetailer.common import ensure_pil_image

    opts = SimpleNamespace() if host["bg"] is None else SimpleNamespace(
        img2img_background_color=host["bg"]
    )
    get_init = _load_script(
        methods={"get_i2i_init_image"},
        is_skip_img2img=lambda p: getattr(p, "_ad_skip_img2img", False),
        images=host["images"],
        opts=opts,
    ).AfterDetailerScript.get_i2i_init_image
    cutout = Image.new("RGBA", (8, 8), (0, 255, 0, 0))
    cutout.paste((200, 30, 30, 255), (2, 2, 6, 6))
    p = SimpleNamespace(_ad_skip_img2img=True, init_images=[cutout])

    image = ensure_pil_image(get_init(p, SimpleNamespace(image="sample")), "RGB")

    assert image.getpixel((0, 0)) == corner
    assert image.getpixel((4, 4)) == (200, 30, 30)
    # The host reuses p.init_images for the next batches.
    assert p.init_images == [cutout]
    assert cutout.mode == "RGBA"
    # Unchanged: an image without transparency is the same object.
    rgb = Image.new("RGB", (8, 8), (10, 20, 30))
    assert get_init(SimpleNamespace(_ad_skip_img2img=True, init_images=[rgb]), None) is rgb


@pytest.mark.parametrize(
    ("host", "corner"),
    [
        ({"images": SimpleNamespace(flatten=_host_flatten), "bg": "#ffffff"}, (255, 255, 255)),
        ({"images": SimpleNamespace(flatten=_host_flatten), "bg": "#808080"}, (128, 128, 128)),
        # A host without them: the alpha channel is dropped, as before.
        ({"images": SimpleNamespace(), "bg": "#ffffff"}, (0, 255, 0)),
    ],
)
def test_standalone_run_fills_a_transparent_image_like_the_host(tmp_path, host, corner):
    # "Run ADetailer on an image" and folder runs dropped the alpha channel of
    # a cut-out: its transparent parts were detected and inpainted black, or in
    # the colour stored under them, instead of the img2img background colour.
    from adetailer.common import ensure_pil_image

    script = _load_script(
        methods={
            "run_detailer_on_image", "read_params_txt", "write_params_txt",
            "get_seed", "get_each_tab_seed",
        },
        paths=SimpleNamespace(data_path=str(tmp_path)),
        PARAMS_TXT="params.txt",
        shared=SimpleNamespace(
            sd_model=None,
            opts=SimpleNamespace(data={"ad_same_seed_for_each_tab": False}),
        ),
        state=SimpleNamespace(interrupted=False, skipped=False, stopping_generation=False),
        opts=SimpleNamespace(samples_format="png", img2img_background_color=host["bg"]),
        all_samplers=[],
        ensure_pil_image=ensure_pil_image,
        images=host["images"],
        AD_APPLY_SUBDIR="ADetailer-Inpaint",
        get_i=lambda _p: 0,
        pause_total_tqdm=nullcontext,
    ).AfterDetailerScript()
    seen = []

    def inner(_p, pp, _args, n=0):
        seen.append(pp.image)
        return False

    script._postprocess_image_inner = inner
    cutout = Image.new("RGBA", (64, 64), (0, 255, 0, 0))
    cutout.paste((200, 30, 30, 255), (16, 16, 48, 48))

    image, status = script.run_detailer_on_image(cutout, SimpleNamespace(), save=False)

    assert seen[0].mode == "RGB"
    assert seen[0].getpixel((0, 0)) == corner
    assert seen[0].getpixel((32, 32)) == (200, 30, 30)
    # Nothing detected: the image handed back is the filled one.
    assert status == "ℹ️ Nothing detected — image unchanged."
    assert image.getpixel((0, 0)) == corner
    # The input box gives an opaque image as RGBA: its pixels are unchanged,
    # and so are those of an RGB image.
    opaque = Image.new("RGB", (64, 64), (10, 20, 30))
    opaque.paste((250, 240, 5), (8, 8, 20, 20))
    for given in (opaque.convert("RGBA"), opaque):
        script.run_detailer_on_image(given, SimpleNamespace(), save=False)
        assert seen[-1].mode == "RGB"
        assert seen[-1].tobytes() == opaque.tobytes()


@pytest.mark.parametrize(
    ("classes", "class_prompts", "expected"),
    [
        # Class ids (from the API, pasted parameters or a preset): each pass
        # got the tab prompt, and the lines still switched the guard off.
        (
            "0,1",
            "face: detailed skin\nhand: five fingers",
            [("0", "detailed skin"), ("1", "five fingers")],
        ),
        ("1,face", "Face: detailed skin\nhand: five fingers",
         [("1", "five fingers"), ("face", "detailed skin")]),
        # Unchanged: a line written for the id itself wins, and names.
        (
            "0,1",
            "0: id line\nface: detailed skin\nhand: five fingers",
            [("0", "id line"), ("1", "five fingers")],
        ),
        (
            "face,hand",
            "face: detailed skin\nhand: five fingers",
            [("face", "detailed skin"), ("hand", "five fingers")],
        ),
    ],
)
def test_a_sequential_class_id_uses_the_line_of_its_class_name(
    classes, class_prompts, expected, tmp_path
):
    from adetailer.args import ADetailerArgs
    from adetailer.classes import get_model_class_names, resolve_class_ids

    model = tmp_path / "multi.pt"
    model.write_bytes(b"x")
    (tmp_path / "multi.names.json").write_bytes(b'["face","hand"]')
    runtime = _load_runtime(
        state=SimpleNamespace(interrupted=False, skipped=False),
        copy=copy, re=re,
        parse_csv=lambda text: text.split(","),
        is_skip_img2img=lambda _p: False,
        disable_safe_unpickle=nullcontext,
        get_model_class_names=get_model_class_names,
        resolve_class_ids=resolve_class_ids,
    )
    script = runtime.AfterDetailerScript()
    script.save_image = lambda *_args, **_kwargs: None
    script.get_ad_model = lambda _name: model
    passes = []

    def class_pass(_p, _pp, sub_args, **_kwargs):
        passes.append((sub_args.ad_model_classes, sub_args.ad_prompt))
        return True

    script._postprocess_image_inner = class_pass  # each class's own pass
    args = ADetailerArgs(
        ad_model="multi.pt", ad_prompt="main", ad_model_classes=classes,
        ad_classes_sequential=True, ad_class_prompts=class_prompts,
    )

    assert runtime.AfterDetailerScript._postprocess_image_inner(
        script, SimpleNamespace(), SimpleNamespace(image=Image.new("RGB", (8, 8))), args
    )
    assert passes == expected


def test_blank_prompts_stay_out_of_the_infotext():
    # A value of only spaces is written unquoted, and AUTOMATIC1111's parser
    # then reads an empty value and prints "Error parsing" on every paste.
    from adetailer.args import ADetailerArgs

    runtime = _load_script(
        methods={"extra_params"}, suffix=_ui_suffix(), __version__="test"
    )
    params = runtime.AfterDetailerScript().extra_params(
        [
            ADetailerArgs(
                ad_model="face_yolov8n.pt", ad_prompt="  ", ad_negative_prompt="\t",
                ad_prompt_append=" ", ad_negative_prompt_append="  ",
                ad_inpaint_indices="  ",
            ),
            ADetailerArgs(
                ad_model="hand_yolov8n.pt", ad_prompt="detailed hand",
                ad_negative_prompt="blurry", ad_prompt_append="five fingers",
                ad_inpaint_indices="1,3",
            ),
        ]
    )

    for name in (
        "ADetailer prompt", "ADetailer negative prompt", "ADetailer prompt append",
        "ADetailer negative prompt append", "ADetailer inpaint indices",
    ):
        assert name not in params
    assert params["ADetailer model"] == "face_yolov8n.pt"
    assert params["ADetailer prompt 2nd"] == "detailed hand"
    assert params["ADetailer negative prompt 2nd"] == "blurry"
    assert params["ADetailer prompt append 2nd"] == "five fingers"
    assert params["ADetailer inpaint indices 2nd"] == "1,3"
    assert all(str(v).strip() for v in params.values())


def test_blank_prompts_left_out_paste_back_blank():
    # A missing key pastes as empty, which a blank value means anyway.
    params = {"ADetailer model": "face_yolov8n.pt"}

    _paste_callback()._clear_missing_class_prompts("", params)

    for name in (
        "ADetailer prompt", "ADetailer negative prompt", "ADetailer prompt append",
        "ADetailer negative prompt append",
    ):
        assert params[name] == ""


def test_the_readme_says_what_the_face_features_do_with_an_unknown_name():
    # Unlike a YOLO model's, an unknown name of the MediaPipe face features is
    # not named in the console, and with no known name nothing is detected.
    readme = _public_docs()["README.md"].splitlines()
    single = next(line for line in readme if "A name the detector does not have" in line)
    sequential = next(
        line for line in readme
        if line.startswith("A class name the detector does not have")
        and "gets no pass of its own" in line
    )

    assert "MediaPipe face features are the exception" in single
    assert "nothing is detected" in single
    assert "With `mediapipe_face_features` such a name is not dropped" in sequential
    assert "nothing is inpainted" in sequential


def test_the_readme_says_the_detection_preview_ignores_mask_preprocessing():
    readme = _public_docs()["README.md"]
    section = readme.split("### Detection preview", 1)[1].split("\n### ", 1)[0]

    assert "mask preprocessing without burning" not in section
    assert "are not applied to it" in section
    for setting in ("erosion/dilation", "merge mode", "Use bbox as mask", "top k"):
        assert setting in section
    # The preview still runs the detector alone: nothing reshapes its masks.
    ui = (_SCRIPT_PATH.parents[1] / "aaaaaa" / "ui.py").read_text(encoding="utf-8")
    tree = ast.parse(ui)
    preview = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_wire_detection_previews"
    )
    called = {
        getattr(node.func, "id", getattr(node.func, "attr", ""))
        for node in ast.walk(preview) if isinstance(node, ast.Call)
    }
    assert not called & {"mask_preprocess", "filter_by_ratio", "filter_k_by"}


def test_the_readme_install_steps_name_the_forks_folder():
    # The WebUI names the folder after the last part of the URL, without
    # ".git": the fork installs into extensions/adetailer-ultimate.
    readme = _public_docs()["README.md"].splitlines()
    url_step = next(line for line in readme if "URL for extension's git repository" in line)
    url = re.search(r"`(https://[^`]+\.git)`", url_step).group(1)
    folder = url.rstrip("/").rsplit("/", 1)[1].replace(".git", "")
    message = next(line for line in readme if "Installed into" in line)

    assert folder == "adetailer-ultimate"
    assert f"extensions\\{folder}. Use Installed tab" in message


# Final debugging pass, round 9: generation pipeline, prompts and docs.


class _ScriptSetupI2I(_A1111I2I):
    """The host's img2img class with its `scripts` / `script_args` setters:
    once both are set, the always-on scripts' setup() runs on the pass."""

    scripts_value = None
    script_args_value = None
    scripts_setup_complete = False
    is_api = False  # ADetailer's own pass is never an API one

    def _setup(self):
        if self.scripts_value and self.script_args_value and not self.scripts_setup_complete:
            self.scripts_setup_complete = True
            self.scripts_value.setup_scrips(self, is_ui=not self.is_api)

    @property
    def scripts(self):
        return self.scripts_value

    @scripts.setter
    def scripts(self, value):
        self.scripts_value = value
        self._setup()

    @property
    def script_args(self):
        return self.script_args_value

    @script_args.setter
    def script_args(self, value):
        self.script_args_value = value
        self._setup()


class _SamplerScriptRunner:
    """The host's built-in Sampler script: its setup copies its slice of the
    script args (the main UI's, or the UI defaults for an API request) onto
    the processing object."""

    def setup_scrips(self, p, *, is_ui):
        if is_ui:
            for name, value in zip(("steps", "sampler_name", "scheduler"), p.script_args):
                setattr(p, name, value)


@pytest.mark.parametrize(
    ("schedulers", "fields", "script_args", "expected"),
    [
        # "Use separate steps" and "Use separate sampler".
        (
            ["karras"],
            {
                "ad_use_steps": True, "ad_steps": 75, "ad_use_sampler": True,
                "ad_sampler": "Euler a", "ad_scheduler": "Karras",
            },
            [20, "DPM++ 2M", "Automatic"],
            (75, "Euler a", "Karras"),
        ),
        # An API request: the Sampler script holds the UI defaults, not the
        # request's steps, sampler and scheduler.
        (["karras"], {}, [20, "DPM++ 2M", "Automatic"], (40, "Euler", "Simple")),
        # A host without schedulers gets no scheduler from ADetailer.
        (None, {"ad_use_steps": True, "ad_steps": 75}, [20, "DPM++ 2M"], (75, "Euler", None)),
    ],
)
def test_the_host_sampler_script_does_not_overwrite_the_detailer_steps_and_sampler(
    schedulers, fields, script_args, expected
):
    # With "Apply only selected scripts to ADetailer" off, assigning the
    # script args ran the built-in Sampler script's setup on the pass, which
    # replaced ADetailer's steps, sampler and scheduler with the main ones.
    from aaaaaa.p_method import is_skip_img2img
    from adetailer.args import ADetailerArgs

    runtime = _load_script(
        methods={
            "get_i2i_p", "get_width_height", "get_steps", "get_sampler",
            "get_scheduler",
        },
        StableDiffusionProcessingImg2Img=_ScriptSetupI2I,
        schedulers=schedulers,
        controlnet_type="forge",
        copy_extra_params=dict,
        is_skip_img2img=is_skip_img2img,
        opts=SimpleNamespace(),
    )
    script = runtime.AfterDetailerScript()
    script.get_seed = lambda _p: (1, 1)
    script.get_cfg_scale = lambda *_args: 7.0
    script.get_initial_noise_multiplier = lambda *_args: None
    script.get_override_settings = lambda *_args: {}
    script.script_filter = lambda *_args: (_SamplerScriptRunner(), list(script_args))
    p = _txt2img_p(steps=40, sampler_name="Euler", scheduler="Simple")
    args = ADetailerArgs(ad_model="face_yolov8n.pt", **fields)

    i2i = script.get_i2i_p(p, args, Image.new("RGB", (512, 768)))

    assert i2i.scripts_setup_complete
    steps, sampler, scheduler = expected
    assert (i2i.steps, i2i.sampler_name) == (steps, sampler)
    if scheduler is None:
        assert "scheduler" not in vars(i2i)
    else:
        assert i2i.scheduler == scheduler


def test_restarting_the_other_scripts_keeps_the_images_template():
    # The copy of p that ADetailer runs the other scripts' process() on shared
    # p's parameters: Dynamic Prompts wrote its "Template" there again, from
    # the prompts it had already resolved, and the images kept that instead
    # of the wildcard template.
    from contextlib import contextmanager

    from aaaaaa.p_method import need_call_postprocess, need_call_process

    @contextmanager
    def preserve_prompts(p):
        saved = (list(p.all_prompts), list(p.all_negative_prompts))
        try:
            yield
        finally:
            p.all_prompts, p.all_negative_prompts = saved

    script = _load_script(
        methods={"postprocess_image"},
        opts=SimpleNamespace(data={}),
        ensure_pil_image=lambda image, _mode: image,
        copy=copy,
        _verbose_gen_header=lambda *_args: None,
        need_call_postprocess=need_call_postprocess,
        need_call_process=need_call_process,
        Processed=lambda *_args: "dummy",
        preserve_prompts=preserve_prompts,
        CNHijackRestore=nullcontext,
        pause_total_tqdm=nullcontext,
        cn_allow_script_control=nullcontext,
        _should_skip_for_hires_only=lambda _p, _args: False,
        is_skip_img2img=lambda _p: False,
    ).AfterDetailerScript()
    tab = SimpleNamespace(need_skip=lambda: False)
    script.is_ad_enabled = lambda *_args: True
    script.get_i2i_init_image = lambda _p, pp: pp.image
    script.get_args = lambda *_args: [tab]
    script.read_params_txt = lambda: ""
    script.write_params_txt = lambda _content: None
    script._will_run_sequential = lambda _args: False
    script.save_image = lambda *_args, **_kwargs: None
    script._postprocess_image_inner = lambda *_args, **_kwargs: True

    def hires_prompt(_params):  # a host callable among the parameters
        return "a face"

    seen = []

    def process(q):
        # What Dynamic Prompts' process() does with "Save template to metadata".
        seen.append((
            q.extra_generation_params is p.extra_generation_params,
            q.extra_generation_params.get("Hires prompt"),
        ))
        q.extra_generation_params["Template"] = q.all_prompts[0]
        q.extra_generation_params["Negative Template"] = q.all_negative_prompts[0]
        q.all_prompts = ["resolved"]

    runner = SimpleNamespace(
        postprocess=lambda *_args: None, before_process=lambda *_args: None,
        process=process,
    )
    params = {
        "Template": "a photo of a face, {blue|green} eyes",
        "Negative Template": "{lowres|blurry}",
        "ADetailer model": "face_yolov8n.pt",
        "Hires prompt": hires_prompt,
    }
    p = SimpleNamespace(
        batch_index=0, batch_size=1, seed=1, scripts=runner,
        all_prompts=["a photo of a face, blue eyes"], all_negative_prompts=["lowres"],
        extra_generation_params=dict(params),
    )

    script.postprocess_image(p, SimpleNamespace(image=Image.new("RGB", (8, 8))), True)

    assert p.extra_generation_params == params
    # The scripts still ran, on parameters of their own that hold p's.
    assert seen == [(False, hires_prompt)]
    assert p.all_prompts == ["a photo of a face, blue eyes"]


def test_standalone_run_says_how_many_regions_failed_with_a_nan_error(tmp_path):
    # One region failed with a NaN error and another was detailed: the status
    # read only "pass complete", although a face was left undetailed.
    run, _image = _nan_standalone(tmp_path)

    for regions, failed in (
        (["nan", "ok"], 1), (["ok", "nan"], 1), (["nan", "nan", "ok"], 2),
    ):
        result, status = run(regions)
        assert result is not None
        assert status.startswith("✅ ADetailer pass complete.")
        assert f"{failed} region(s) failed with a NaN error" in status
        # A folder run still counts the file as detailed and saved.
        assert "couldn't save" not in status

    # A clean run has no note, and none is left over from the runs above.
    assert run(["ok", "ok"])[1] == "✅ ADetailer pass complete."


@pytest.mark.parametrize(
    ("lines", "sr", "record"),
    [
        (
            "face: smiling face\nhand: smiling hand", ("smiling", "laughing"),
            "face: laughing face\nhand: laughing hand",
        ),
        (
            "face: smiling face | smiling blur\nhand:  | smiling hand",
            ("smiling", "laughing"),
            "face: laughing face | laughing blur\nhand:  | laughing hand",
        ),
        # The class name stays as typed, and a line without ":" is left alone.
        (
            "face: detailed face, {smiling|sad}\nhand: open hand\nface detail",
            ("face", "eyes"),
            "face: detailed eyes, {smiling|sad}\nhand: open hand\nface detail",
        ),
    ],
)
def test_an_xyz_prompt_search_replace_cell_pastes_back_the_per_class_lines(
    lines, sr, record
):
    # In sequential mode each class pass uses its per-class line, with the
    # Prompt S/R applied, but the saved lines were the original ones: pasted
    # back, the passes ran with the text the cell had replaced.
    from aaaaaa.p_method import get_i
    from adetailer.args import ADetailerArgs

    runtime = _load_script(
        methods=_INLINE_PROMPT_METHODS | {"process", "extra_params", "_record_xyz_prompt_sr"},
        functions=_INLINE_PROMPT_FUNCTIONS | {"_parse_class_prompts", "_class_prompt_for"},
        assigns=_INLINE_PROMPT_ASSIGNS,
        opts=SimpleNamespace(data={}),
        is_img2img_inpaint=lambda _p: False,
        get_i=get_i,
        suffix=_ui_suffix(),
        __version__="test",
    )
    script = runtime.AfterDetailerScript()
    script.is_ad_enabled = lambda *_args: True
    script.set_skip_img2img = lambda *_args: None
    tab = {
        "ad_model": "face_yolov8n.pt", "ad_model_classes": "face,hand",
        "ad_classes_sequential": True, "ad_prompt": "detailed smiling face",
        "ad_class_prompts": lines,
    }
    script.get_args = lambda *_args: [ADetailerArgs(**tab)]
    main = "a smiling woman"

    def photo(**extra):
        return SimpleNamespace(
            iteration=0, batch_size=1, prompt=main, negative_prompt="",
            all_prompts=[main], all_negative_prompts=[""],
            extra_generation_params={}, **extra,
        )

    def class_pass(args, cls):
        # The per-class update of the sequential branch.
        entry = runtime._class_prompt_for(
            runtime._parse_class_prompts(args.ad_class_prompts), cls
        )
        update = {"ad_model_classes": cls, "ad_classes_sequential": False}
        if entry is not None:
            if entry[0]:
                update["ad_prompt"] = entry[0]
            if entry[1]:
                update["ad_negative_prompt"] = entry[1]
        return args.copy(update=update)

    p = photo(_ad_xyz_prompt_sr=[SimpleNamespace(s=sr[0], r=sr[1])])
    script.process(p, True, False, {})

    params = p.extra_generation_params
    assert params["ADetailer class prompts"] == record
    pasted = ADetailerArgs(
        **{
            **tab,
            "ad_prompt": params.get("ADetailer prompt", ""),
            "ad_negative_prompt": params.get("ADetailer negative prompt", ""),
            "ad_class_prompts": params.get("ADetailer class prompts", ""),
        }
    )
    for cls in ("face", "hand"):
        cell = script.get_prompt(p, class_pass(ADetailerArgs(**tab), cls))
        assert script.get_prompt(photo(), class_pass(pasted, cls)) == cell


def _verbose_helpers():
    import textwrap

    from adetailer.args import ALL_ARGS

    tree = ast.parse(_SCRIPT_PATH.read_text(encoding="utf-8"))
    sections = next(
        ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and getattr(node.target, "id", "") == "_VERBOSE_SECTIONS"
    )
    return _load_script(
        functions={
            "_vprint", "_vfmt", "_vwrap", "_verbose_pass_header",
            "_verbose_pass_result",
        },
        assigns={"_V_LINE"},
        textwrap=textwrap,
        ALL_ARGS=ALL_ARGS,
        _VERBOSE_SECTIONS=sections,
        _ad_verbose=lambda: True,
        _vram_str=lambda: "n/a",
    )


@pytest.mark.parametrize(
    ("encoding", "errors"),
    [
        ("cp1252", "strict"), ("cp1252", "surrogateescape"),
        ("cp932", "strict"), ("cp932", "surrogateescape"), ("utf-8", "strict"),
    ],
)
def test_the_verbose_log_prints_on_a_console_in_a_legacy_code_page(
    monkeypatch, encoding, errors
):
    # Console output to a file or a pipe in a legacy code page refused every
    # block with a character that code page lacks: the tab's settings dump
    # and the closing rule were missing, and on cp932 the region and timing
    # lines too, without any hint.
    from adetailer.args import ADetailerArgs

    helpers = _verbose_helpers()
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding=encoding, errors=errors, write_through=True)
    monkeypatch.setattr(sys, "stdout", stream)
    prompt = "detailed face \U0001f642"

    helpers._verbose_pass_header(
        ADetailerArgs(ad_model="face_yolov8n.pt", ad_prompt=prompt), 0, 0
    )
    helpers._verbose_pass_result(
        [{"j": 0, "inp_ms": 10, "denoise": 0.4, "w": 512, "h": 512,
          "pos": prompt, "neg": "blurry"}],
        5, 10, 20,
    )
    stream.flush()
    out = raw.getvalue().decode(encoding)
    monkeypatch.undo()

    for text in ("tab 1", "[Detection]", "face_yolov8n.pt", "[Prompts]",
                 "prompt: 'detailed face ", "negative: 'blurry'"):
        assert text in out
    assert out.count("[Timing]") == 1
    lines = out.splitlines()
    assert lines[0].startswith("[-] ADetailer ")
    assert lines[-1].startswith("[-] ADetailer ")
    if encoding == "utf-8":
        # A console that has every character prints the block as before.
        assert "─" in out and "═" in lines[-1] and prompt in out
        assert "\\U" not in out
    else:
        assert lines[-1] == "[-] ADetailer ==" + "-" * 66
        assert "detailed face \\U0001f642" in out


# Final debugging pass, round 10: generation pipeline, prompts and docs.


def test_the_docs_quote_each_frozen_settings_error():
    # Only --freeze-settings says "changing settings is disabled": the other
    # two options name the setting. The error stops ADetailer for the image,
    # so its later tabs do not run either, and the earlier tabs are kept.
    docs = _public_docs()
    beta2 = docs["CHANGELOG.md"].split("## v26.2.0+plus.8.beta.1", 1)[0]
    for text in (docs["README.md"], beta2):
        line = next(
            line for line in text.splitlines()
            if "--freeze-specific-settings overlay_inpaint" in line
        )
        assert "not possible to set 'overlay_inpaint'" in line
        assert "later tabs" in line
        assert "leave the image undetailed" not in line


@pytest.mark.parametrize(
    ("encoding", "region", "setting"),
    [
        # cp1252 has the middle dot but no arrow.
        (
            "cp1252",
            "face·detail, (eyes\\u2192left) \\U0001f642", "face·detail",
        ),
        # cp932 has the arrow but no middle dot.
        (
            "cp932",
            "face\\xb7detail, (eyes→left) \\U0001f642", "face\\xb7detail",
        ),
    ],
)
def test_the_verbose_log_fallback_logs_the_prompt_as_typed(
    monkeypatch, encoding, region, setting
):
    # On a legacy code page the fallback turned every middle dot into "|" and
    # every arrow into "->", also inside the logged prompts, where the code
    # page had them and a "|" reads like prompt syntax.
    from adetailer.args import ADetailerArgs

    helpers = _verbose_helpers()
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding=encoding, errors="strict", write_through=True)
    monkeypatch.setattr(sys, "stdout", stream)

    helpers._verbose_pass_header(
        ADetailerArgs(ad_model="face_yolov8n.pt", ad_prompt="face·detail"), 0, 0
    )
    helpers._verbose_pass_result(
        [{"j": 0, "inp_ms": 10, "denoise": 0.4, "w": 512, "h": 512,
          "pos": "face·detail, (eyes→left) \U0001f642", "neg": "blurry"}],
        5, 10, 20,
    )
    stream.flush()
    out = raw.getvalue().decode(encoding)
    monkeypatch.undo()

    assert f"prompt: '{region}'" in out
    assert f"prompt='{setting}'" in out
    assert "face|detail" not in out
    assert "eyes->left" not in out
    # The rules are still drawn in ASCII.
    assert out.splitlines()[-1] == "[-] ADetailer ==" + "-" * 66


@pytest.mark.parametrize("forge", [True, False], ids=["forge", "automatic1111"])
def test_restarting_the_other_scripts_adds_no_second_controlnet_preview(forge):
    # Forge's ControlNet adds its preview (the DetectMap) to
    # p.extra_result_images in process(). The copy of p that ADetailer re-arms
    # the scripts on shared that list, so each batch added one more preview
    # to the gallery and the API images. AUTOMATIC1111's p has no such list.
    from contextlib import contextmanager

    from aaaaaa.p_method import need_call_postprocess, need_call_process

    @contextmanager
    def preserve_prompts(p):
        saved = (list(p.all_prompts), list(p.all_negative_prompts))
        try:
            yield
        finally:
            p.all_prompts, p.all_negative_prompts = saved

    script = _load_script(
        methods={"postprocess_image"},
        opts=SimpleNamespace(data={}),
        ensure_pil_image=lambda image, _mode: image,
        copy=copy,
        _verbose_gen_header=lambda *_args: None,
        need_call_postprocess=need_call_postprocess,
        need_call_process=need_call_process,
        Processed=lambda *_args: "dummy",
        preserve_prompts=preserve_prompts,
        CNHijackRestore=nullcontext,
        pause_total_tqdm=nullcontext,
        cn_allow_script_control=nullcontext,
        _should_skip_for_hires_only=lambda _p, _args: False,
        is_skip_img2img=lambda _p: False,
    ).AfterDetailerScript()
    tab = SimpleNamespace(need_skip=lambda: False)
    script.is_ad_enabled = lambda *_args: True
    script.get_i2i_init_image = lambda _p, pp: pp.image
    script.get_args = lambda *_args: [tab]
    script.read_params_txt = lambda: ""
    script.write_params_txt = lambda _content: None
    script._will_run_sequential = lambda _args: False
    script.save_image = lambda *_args, **_kwargs: None
    script._postprocess_image_inner = lambda *_args, **_kwargs: True
    armed = []

    def process(q):
        armed.append(q is not p)
        if hasattr(q, "extra_result_images"):
            q.extra_result_images.append("detected map")

    runner = SimpleNamespace(
        postprocess=lambda *_args: None, before_process=lambda *_args: None,
        process=process,
    )
    # The job's own process() has already attached its preview.
    own = {"extra_result_images": ["detected map"]} if forge else {}
    p = SimpleNamespace(
        batch_index=0, batch_size=1, seed=1, scripts=runner,
        all_prompts=["a face"], all_negative_prompts=[""],
        extra_generation_params={}, **own,
    )

    for _batch in range(3):
        script.postprocess_image(p, SimpleNamespace(image=Image.new("RGB", (8, 8))), True)

    # ControlNet is still set up again before every batch.
    assert armed == [True, True, True]
    if forge:
        assert p.extra_result_images == ["detected map"]
    else:
        assert not hasattr(p, "extra_result_images")


@pytest.mark.parametrize("skip", [True, False], ids=["skip-img2img", "img2img"])
def test_skip_img2img_saves_the_intermediate_step_images(skip):
    # With Skip img2img, "Save intermediate step images" saved no -ad-step
    # file. Only -ad-before is left out there: it would be the init image.
    script = _manual_mode_script({"ad_manual_mode": False})
    saved = []
    script.save_image = lambda _p, _image, *, condition, suffix: saved.append(
        (condition, suffix)
    )
    p = SimpleNamespace(
        init_images=[Image.new("RGB", (1024, 768))], width=1024, height=768,
        steps=30, sampler_name="DPM++ 2M", extra_generation_params={},
        batch_index=0, batch_size=1, seed=1, scripts=None,
    )
    script.process(p, True, skip, {})
    pp = SimpleNamespace(image=Image.new("RGB", (p.width, p.height)))
    script.postprocess_image(p, pp, True, skip, {})

    expected = [("ad_save_intermediate_steps", "-ad-step-1")]
    if not skip:
        expected.append(("ad_save_images_before", "-ad-before"))
    assert saved == expected


def test_skip_img2img_saves_the_step_image_of_each_sequential_class_pass():
    state = SimpleNamespace(
        interrupted=False, skipped=False, job_count=0,
        assign_current_image=lambda _image: None,
    )
    result = Image.new("RGB", (8, 8), "red")
    run, _pp, _before = _detailer(
        state, lambda _p: SimpleNamespace(images=[result]), n_masks=1
    )
    script = _script_of(run)
    script._postprocess_image_inner.__func__.__globals__["is_skip_img2img"] = (
        lambda _p: True
    )
    saved = []
    script.save_image = lambda _p, _image, *, condition, suffix: saved.append(
        (condition, suffix)
    )

    assert run(classes="face,hand", sequential=True) is True
    assert [s for c, s in saved if c == "ad_save_intermediate_steps"] == [
        "-ad-step-1-1-face", "-ad-step-1-2-hand",
    ]
    assert [s for c, s in saved if c == "ad_save_previews"] == [
        "-ad-preview-1-1-face", "-ad-preview-1-2-hand",
    ]


@pytest.mark.parametrize(
    ("merge", "passes"),
    [
        # One pass inverts every selected class together, with the tab prompt.
        ("Merge and Invert", [("face,hand", "portrait", False)]),
        # The other modes keep one pass per class, each with its own line.
        ("Merge", [("face", "detailed skin", True), ("hand", "five fingers", True)]),
        ("None", [("face", "detailed skin", True), ("hand", "five fingers", True)]),
    ],
)
def test_a_sequential_merge_and_invert_tab_runs_one_pass(merge, passes):
    # Each class pass inverted only its own class: the face pass repainted
    # the hands with the face line, and the hand pass then repainted the
    # faces with the hand line.
    from adetailer.args import ADetailerArgs
    from adetailer.classes import parse_csv

    state = SimpleNamespace(
        interrupted=False, skipped=False, job_count=0,
        assign_current_image=lambda _image: None,
    )
    image = Image.new("RGB", (8, 8), "white")
    pred = SimpleNamespace(preview=image)
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
        ensure_pil_image=lambda im, _mode: im,
        process_images=lambda _p: SimpleNamespace(images=[image]),
        NansException=type("NansException", (Exception,), {}),
    )
    script = runtime.AfterDetailerScript()
    script.ultralytics_device = "cpu"
    script.get_i2i_p = lambda *_args: SimpleNamespace(
        init_images=[image], prompt="face", close=lambda: None
    )
    script.get_prompt = lambda *_args: (["face"], [""])
    script.get_ad_model = lambda _name: "model.pt"
    script.pred_preprocessing = lambda *_args: [image]
    script.save_image = lambda *_args, **_kwargs: None
    script.i2i_prompts_replace = lambda *_args: None
    script._apply_inline_class_prompts = lambda *_args: None
    script.fix_p2 = lambda *_args: None
    script.compare_prompt = lambda *_args, **_kwargs: None
    seen = []
    script._apply_auto_class_guard = lambda _p2, args, *call: seen.append(
        (args.ad_model_classes, args.ad_prompt, call[-1])
    )

    class Args(SimpleNamespace):
        def copy(self, update):
            return Args(**{**vars(self), **update})

    args = Args(
        ad_classes_sequential=True, ad_model_classes="face,hand",
        ad_model_classes_exclude=False, ad_mask_merge_invert=merge,
        ad_prompt="portrait",
        ad_class_prompts="face: detailed skin\nhand: five fingers",
        ad_model="model.pt", ad_confidence=0.3, ad_use_bbox_mask=False,
        ad_detection_resolution=0, is_mediapipe=lambda: False,
    )

    script._postprocess_image_inner(
        SimpleNamespace(extra_generation_params={}), SimpleNamespace(image=image), args
    )

    assert seen == passes
    # The outer loop saves the tab's step image when no class pass does.
    will_run = _load_script(
        methods={"_will_run_sequential"}, parse_csv=parse_csv
    ).AfterDetailerScript._will_run_sequential
    real = ADetailerArgs(
        ad_model="faces.pt", ad_classes_sequential=True,
        ad_model_classes="face,hand", ad_mask_merge_invert=merge,
    )
    assert will_run(real) is (merge != "Merge and Invert")


def _prompt_sr_class():
    """The script's PromptSR (``_load_script`` takes only its main class)."""
    from typing import NamedTuple

    tree = ast.parse(_SCRIPT_PATH.read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PromptSR"
    )
    namespace = {"NamedTuple": NamedTuple}
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_SCRIPT_PATH), "exec"), namespace)
    return namespace["PromptSR"]


def _prompt_sr_script():
    return _load_script(
        methods=_INLINE_PROMPT_METHODS,
        functions=_INLINE_PROMPT_FUNCTIONS | {"search_and_replace_prompt"},
        assigns=_INLINE_PROMPT_ASSIGNS,
        PromptSR=_prompt_sr_class(),
        get_i=lambda _p: 0,
    )


@pytest.mark.parametrize(
    ("ad_prompt", "main", "expected"),
    [
        # A blank or [PROMPT] segment stands for the main prompt, which the
        # axis has already changed: replaced once, not "big big smile".
        ("", True, "a woman, big smile"),
        ("[PROMPT], detailed face", True, "a woman, big smile, detailed face"),
        # The tab's own text is replaced once, as before.
        ("smile, detailed face", True, "big smile, detailed face"),
        ("[PROMPT], smile", True, "a woman, big smile, big smile"),
        # "(AD 1st)" leaves the main prompt as it is: the replacement still
        # reaches the text a blank or [PROMPT] segment stands for.
        ("", False, "a woman, big smile"),
        ("[PROMPT], detailed face", False, "a woman, big smile, detailed face"),
        ("smile, detailed face", False, "big smile, detailed face"),
    ],
)
def test_an_sr_that_changed_the_main_prompt_replaces_it_once(ad_prompt, main, expected):
    from adetailer.args import ADetailerArgs

    runtime = _prompt_sr_script()
    script = runtime.AfterDetailerScript()
    # An X/Y/Z cell: the axis changes a copy of p, then the host builds the
    # prompts from it.
    p = SimpleNamespace(prompt="a woman, smile", negative_prompt="smile lines", styles=[])
    runtime.search_and_replace_prompt(
        p, "big smile", ["smile", "big smile"], replace_in_main_prompt=main
    )
    p.all_prompts, p.all_negative_prompts = [p.prompt], [p.negative_prompt]

    positive, negative = script.get_prompt(
        p, ADetailerArgs(ad_model="face_yolov8n.pt", ad_prompt=ad_prompt)
    )

    assert positive == [expected]
    assert negative == ["big smile lines"]


def test_an_sr_to_nothing_leaves_the_tab_text_blank():
    # The blank check reads the text as typed: a tab whose own text the axis
    # removes does not become the main prompt.
    from adetailer.args import ADetailerArgs

    runtime = _prompt_sr_script()
    script = runtime.AfterDetailerScript()
    p = SimpleNamespace(prompt="a woman, smile", negative_prompt="", styles=[])
    runtime.search_and_replace_prompt(p, "", ["smile", ""], replace_in_main_prompt=True)
    p.all_prompts, p.all_negative_prompts = [p.prompt], [p.negative_prompt]

    positive, _negative = script.get_prompt(
        p, ADetailerArgs(ad_model="face_yolov8n.pt", ad_prompt="smile")
    )

    assert positive == [""]


@pytest.mark.parametrize(
    ("classes", "lines", "seq_pass", "guarded"),
    [
        # A line written for the pass's class id: the pass used it.
        ("0", "0: detailed skin", True, False),
        # A line written with the class name, as before.
        ("face", "face: detailed skin", True, False),
        ("0", "face: detailed skin", True, False),
        # A line for another class id does not apply to this pass.
        ("0", "1: five fingers", True, True),
        # Outside a sequential pass no line applies.
        ("0", "0: detailed skin", False, True),
    ],
)
def test_a_class_line_written_for_the_class_id_switches_the_guard_off(
    classes, lines, seq_pass, guarded
):
    # The sequential pass for class id "0" used a line written for "0", but
    # the guard looked only for a line named like the detected class
    # ("face"), and the region got both.
    from adetailer.args import ADetailerArgs

    script = _class_prompt_script()
    args = ADetailerArgs(
        ad_model="faces.pt", ad_class_guard=True,
        ad_model_classes=classes, ad_class_prompts=lines,
    )
    p2 = SimpleNamespace(prompt="detailed", negative_prompt="blurry")

    script._apply_auto_class_guard(
        p2, args, SimpleNamespace(class_names=["face"]), 0, 1, seq_pass
    )

    expected = ("face, detailed", "blurry, hand") if guarded else ("detailed", "blurry")
    assert (p2.prompt, p2.negative_prompt) == expected


def test_readme_places_the_detection_tools_above_mask_preprocessing():
    # The README put the Detection preview and "Run ADetailer on an image" at
    # the bottom of each tab; they come right after the Detection section.
    root = _SCRIPT_PATH.parents[1]
    ui_text = (root / "aaaaaa" / "ui.py").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    order = [
        ui_text.index(f'eid("{name}")')
        for name in (
            "ad_detection_accordion", "ad_preview_accordion", "ad_apply_accordion",
            "ad_mask_preprocessing_accordion", "ad_inpainting_accordion",
        )
    ]
    assert order == sorted(order)
    section = readme[readme.index("### Detection preview"):]
    section = section[: section.index("### Batch a whole folder")]
    assert "right after the Detection section" in section
    assert "above Mask Preprocessing and Inpainting" in section
    assert "bottom of each tab" not in section


@pytest.mark.parametrize(
    ("enabled", "tab", "xyz", "printed"),
    [
        # ADetailer switched off, or every tab on None: nothing was skipped.
        (False, {"ad_model": "face_yolov8n.pt"}, None, False),
        (True, {"ad_model": "None"}, None, False),
        # Unchanged: ADetailer would have run.
        (True, {"ad_model": "face_yolov8n.pt"}, None, True),
        # An X/Y/Z cell whose axis gives the 1st tab its detector.
        (True, {"ad_model": "None"}, "face_yolov8n.pt", True),
    ],
)
def test_manual_mode_says_it_skips_only_when_adetailer_would_run(
    enabled, tab, xyz, printed, capsys
):
    # With ADetailer switched off, every generation, X/Y/Z cell and API call
    # printed that manual mode skipped it.
    from adetailer.args import ADetailerArgs

    runtime = _load_script(
        methods={"process", "is_ad_enabled"},
        functions={"set_value"},
        opts=SimpleNamespace(data={"ad_manual_mode": True}),
        ADetailerArgs=ADetailerArgs,
    )
    script = runtime.AfterDetailerScript()
    p = SimpleNamespace(init_images=None, extra_generation_params={})
    if xyz:
        runtime.set_value(p, xyz, [xyz], field="ad_model")

    script.process(p, enabled, False, tab)

    assert p._ad_disabled
    assert ("manual mode is ON" in capsys.readouterr().out) == printed


@pytest.mark.parametrize("skip", [False, True])
def test_an_xyz_detector_axis_runs_a_1st_tab_left_on_none(skip):
    # The "[ADetailer] ADetailer model 1st" axis reached only get_args: with
    # the 1st tab on None in the UI, every cell came out without ADetailer
    # although its label named a detector.
    from adetailer.args import ADetailerArgs, SkipImg2ImgOrig

    class Reached(Exception):
        pass

    def reach(*_args):
        raise Reached

    runtime = _load_script(
        methods={
            "process", "postprocess_image", "is_ad_enabled", "set_skip_img2img",
            "get_args", "extra_params",
        },
        functions={"set_value"},
        opts=SimpleNamespace(data={}),
        is_img2img_inpaint=lambda _p: False,
        ADetailerArgs=ADetailerArgs,
        SkipImg2ImgOrig=SkipImg2ImgOrig,
        suffix=_ui_suffix(),
        __version__="test",
    )
    script = runtime.AfterDetailerScript()
    script.get_i2i_init_image = reach
    ui = (True, skip, {"ad_model": "None"}, {"ad_model": "None"})
    p = SimpleNamespace(
        prompt="a photo", negative_prompt="",
        init_images=[Image.new("RGB", (8, 8))] if skip else None,
        steps=30, sampler_name="DPM++ 2M", width=512, height=768,
        extra_generation_params={"Hires upscale": 2},
    )
    values = ["face_yolov8n.pt", "None", "hand_yolov8n.pt"]

    cells = []
    for value in values:
        pc = copy(p)  # a grid cell: its own attributes, the same parameters
        runtime.set_value(pc, value, values, field="ad_model")
        script.process(pc, *ui)
        cells.append((pc, dict(pc.extra_generation_params)))

    for value, (pc, params) in zip(values, cells):
        assert params["Hires upscale"] == 2
        if value == "None":
            # It records nothing of the cell before, Skip img2img's included.
            assert pc._ad_disabled
            assert not [key for key in params if key.startswith("ADetailer ")]
            assert "Steps" not in params
            assert script.postprocess_image(pc, SimpleNamespace(image=None), *ui) is None
        else:
            assert not getattr(pc, "_ad_disabled", False)
            assert params["ADetailer model"] == value
            assert ("Steps" in params) == skip
            with pytest.raises(Reached):
                script.postprocess_image(pc, SimpleNamespace(image=None), *ui)

    # Unchanged: without the axis, and with ADetailer switched off.
    plain = copy(p)
    script.process(plain, *ui)
    assert plain._ad_disabled
    off = copy(p)
    runtime.set_value(off, "face_yolov8n.pt", values, field="ad_model")
    script.process(off, False, skip, {"ad_model": "None"})
    assert off._ad_disabled


@pytest.mark.parametrize(
    ("parameters", "pnginfo", "expected"),
    [
        (
            "face, detailed\nSteps: 20, Sampler: Euler a", True,
            "face, detailed\nSteps: 20, Sampler: Euler a",
        ),
        (None, True, ""),
        # "Write infotext to metadata" off: the host puts no parameters on the
        # result, and "" made it write an empty .txt sidecar with "Create a
        # text file with infotext" on. None writes none, as before beta 2.
        (None, False, None),
    ],
)
def test_a_saved_standalone_result_records_the_passs_parameters(
    tmp_path, parameters, pnginfo, expected
):
    # The result saved to ADetailer-Inpaint (also by a folder batch) was
    # written without parameters: the WebUI stored the text "None" in the
    # PNG, and Send to put "None" in the prompt box.
    saved = []
    script = _load_script(
        methods={"run_detailer_on_image", "read_params_txt", "write_params_txt"},
        paths=SimpleNamespace(data_path=str(tmp_path)),
        PARAMS_TXT="params.txt",
        shared=SimpleNamespace(sd_model=None),
        state=SimpleNamespace(interrupted=False, skipped=False),
        opts=SimpleNamespace(
            samples_format="png", outdir_samples=str(tmp_path / "out"),
            enable_pnginfo=pnginfo,
        ),
        all_samplers=[],
        ensure_pil_image=lambda image, _mode: image,
        images=SimpleNamespace(
            save_image=lambda _image, _path, _basename, **kwargs: saved.append(kwargs)
        ),
        AD_APPLY_SUBDIR="ADetailer-Inpaint",
        pause_total_tqdm=nullcontext,
    ).AfterDetailerScript()

    def inner(_p, pp, _args, n=0):
        result = Image.new("RGB", (64, 64))
        if parameters is not None:
            result.info["parameters"] = parameters  # set by the WebUI's pass
        pp.image = result
        return True

    script._postprocess_image_inner = inner

    _image, status = script.run_detailer_on_image(
        Image.new("RGB", (64, 64)), SimpleNamespace(), save=True
    )

    assert "Saved" in status
    assert saved[0]["info"] == expected


@pytest.mark.parametrize(
    ("size", "working"),
    [
        ((853, 480), (848, 480)),  # rounded down to a multiple of 8
        ((2048, 1536), (1024, 768)),  # at most 1024 on the longest side
        ((40, 40), (64, 64)),  # at least 64
        ((1024, 768), (1024, 768)),
        # Exactly 1024, not 1016: as a float, 1288 * (1024 / 1288) is
        # 1023.99..., and 644 * (1024 / 1288) is 511.99...
        ((1288, 966), (1024, 768)),
        ((966, 1288), (768, 1024)),
        ((1288, 644), (1024, 512)),
        ((1568, 1176), (1024, 768)),
    ],
)
def test_standalone_run_redraws_at_the_documented_working_size(tmp_path, size, working):
    # The Size a standalone result records is this working size (the WebUI
    # writes p.width x p.height for an img2img pass), not the picture's size:
    # the README says how it is made, so the two must not drift apart.
    script = _standalone_runtime(tmp_path).AfterDetailerScript()
    sizes = []

    def inner(p, _pp, _args, n=0):
        sizes.append((p.width, p.height))
        return False

    script._postprocess_image_inner = inner

    script.run_detailer_on_image(Image.new("RGB", size), SimpleNamespace())

    assert sizes == [working]


def test_standalone_working_size_is_exact_for_every_side_length(tmp_path):
    # Each side is the picture's side scaled by 1024 / longest side (not
    # scaled up to 1024), rounded down to a multiple of 8 and at least 64.
    # A float scale lost a pixel on about one long side in seven up to 8192
    # (one in ten over the sizes swept here), so the longest side of such a
    # picture came out at 1016 instead of 1024.
    from fractions import Fraction

    def documented(side, longest):
        exact = Fraction(side * 1024, max(longest, 1024))
        return max(64, exact.numerator // exact.denominator // 8 * 8)

    script = _standalone_runtime(tmp_path).AfterDetailerScript()
    sizes = []

    def inner(p, _pp, _args, n=0):
        sizes.append((p.width, p.height))
        return False

    script._postprocess_image_inner = inner

    longs = range(960, 1700)
    # Premise: the float scale gets some of these long sides wrong.
    assert sum(int(m * (1024 / m)) != 1024 for m in longs if m > 1024) > 50
    cases = []
    for m in longs:
        short = m * 3 // 4
        cases += [(m, short), (short, m)]
    for size in cases:
        # A 1-bit picture: only its size matters here, and it is quick to make.
        script.run_detailer_on_image(Image.new("1", size), SimpleNamespace())

    longest = [max(size) for size in cases]
    expected = [
        (documented(w, top), documented(h, top))
        for (w, h), top in zip(cases, longest)
    ]
    assert sizes == expected
    # The longest side of a large picture is always exactly 1024.
    assert all(max(s) == 1024 for s, top in zip(sizes, longest) if top > 1024)


def test_a_saved_standalone_result_keeps_the_passs_working_size(tmp_path):
    # An 853x480 picture is redrawn at 848x480 and keeps its own pixels: the
    # parameters say "Size: 848x480", as the WebUI writes them for the pass,
    # both in the file saved to ADetailer-Inpaint and on the picture shown in
    # Result. They are passed on as they are, not rewritten.
    parameters = "detailed face\nSteps: 28, Sampler: DPM++ 2M, Size: 848x480"
    saved = []
    script = _load_script(
        methods={"run_detailer_on_image", "read_params_txt", "write_params_txt"},
        paths=SimpleNamespace(data_path=str(tmp_path)),
        PARAMS_TXT="params.txt",
        shared=SimpleNamespace(sd_model=None),
        state=SimpleNamespace(interrupted=False, skipped=False),
        opts=SimpleNamespace(
            samples_format="png", outdir_samples=str(tmp_path / "out"),
            enable_pnginfo=True,
        ),
        all_samplers=[],
        ensure_pil_image=lambda image, _mode: image,
        images=SimpleNamespace(
            save_image=lambda _image, _path, _basename, **kwargs: saved.append(kwargs)
        ),
        AD_APPLY_SUBDIR="ADetailer-Inpaint",
        pause_total_tqdm=nullcontext,
    ).AfterDetailerScript()

    def inner(p, pp, _args, n=0):
        assert (p.width, p.height) == (848, 480)
        result = Image.new("RGB", pp.image.size)
        result.info["parameters"] = parameters  # set by the WebUI's pass
        pp.image = result
        return True

    script._postprocess_image_inner = inner

    image, status = script.run_detailer_on_image(
        Image.new("RGB", (853, 480)), SimpleNamespace(), save=True
    )

    assert "Saved" in status
    assert saved[0]["info"] == parameters
    assert image.info["parameters"] == parameters
    assert image.size == (853, 480)


def _flat_doc_text(text):
    """``text`` on one line without code marks, with "853 x 480" read as
    "853x480" and "1,024" as "1024", so a rewording keeps its facts."""
    text = " ".join(text.replace("`", "").split())
    text = re.sub(r"(\d) ?[x\u00d7] ?(\d)", r"\1x\2", text)
    return re.sub(r"(\d),(\d{3})\b", r"\1\2", text)


def _size_text(doc):
    """The bullet of ``doc`` that says what the Size of a standalone result is."""
    text = _public_docs()[doc]
    if doc == "README.md":
        text = _readme_section("### Run ADetailer on an image", "### Batch a whole folder")
    return next(
        flat for flat in map(_flat_doc_text, text.split("\n- "))
        if "Size" in flat and "848x480" in flat
    )


# Wordings that make "Inpaint only masked" optional where the code needs it on.
_LOOSE_CONDITIONS = (" or off", " or not", "whether or not", "regardless")


def test_readme_says_size_is_the_standalone_working_size():
    # Users read "Size: 848x480" in the parameters of an 853x480 result as a
    # wrong size: no text said that it is the size the regions were redrawn at.
    # Each fact is checked on its own (and
    # test_standalone_run_redraws_at_the_documented_working_size holds the
    # code to the same rule), so a rewording of the bullet keeps it green.
    bullet = _size_text("README.md")

    for fact in ("853x480", "multiple of 8", "1024", "64", "last region"):
        assert fact in bullet


@pytest.mark.parametrize("doc", ["README.md", "CHANGELOG.md"])
def test_the_size_text_says_when_the_picture_keeps_its_pixels(doc):
    # The picture keeps its pixels, and "Scale inpaint to bbox" or the bbox
    # size setting give the Size, only with "Inpaint only masked" on: with it
    # off the whole picture comes back at the working size (see
    # test_whole_picture_inpaint_keeps_the_image_size). Both texts said so
    # without the condition.
    clauses = re.split(r"(?<=[.;]) ", _size_text(doc))
    about = [c for c in clauses if "853x480 pixels" in c or "Scale inpaint to bbox" in c]

    assert len(about) >= 2  # premise: both statements are there
    for clause in about:
        assert '"Inpaint only masked" on' in clause
        # ... and not "on or off", "on or not" or "regardless" of it, which
        # still contain '"Inpaint only masked" on'.
        for loose in _LOOSE_CONDITIONS:
            assert loose not in clause, clause
    # The part that keeps the pixels ties them to the option being on, not
    # to "on or off" or to the option the other way round.
    keeps = next(
        part for clause in about for part in clause.split(", ")
        if "853x480 pixels" in part
    )
    assert '"Inpaint only masked" on' in keeps
    for loose in _LOOSE_CONDITIONS:
        assert loose not in keeps


@pytest.mark.parametrize("doc", ["README.md", "CHANGELOG.md"])
def test_the_size_text_says_the_published_releases_gave_1016(doc):
    # Both texts give the Size as the picture scaled down to 1024 pixels on the
    # longest side, and the CHANGELOG's says it has been so since plus.6. The
    # float scale of plus.6 to beta 2 gave 1016 for some sizes (see
    # test_standalone_working_size_is_exact_for_every_side_length), so without
    # that exception the text told a beta 2 user whose 1288x966 picture
    # recorded 1016x768 that it had recorded 1024x768.
    bullet = _size_text(doc)

    assert "exactly 1024" in bullet
    assert "1016" in bullet
    if doc == "CHANGELOG.md":
        assert "beta 3" in bullet
    else:
        # The README of the stable release names the release, plus.8, from
        # which the size is exact, and the published beta 2 among the
        # versions that gave 1016.
        assert re.search(
            r"\bplus\.8\b[^.;]*\bexactly 1024|\bexactly 1024\b[^.;]*\bplus\.8\b", bullet
        ), bullet
        assert "beta 2" in bullet


def _bbox_cap_note_problems(section):
    """What the note in ``section`` on the 1024-pixel working size gets wrong
    about the options that lift that cap (an empty list when it is right)."""
    sentences = re.split(r"(?<=[.;]) ", _flat_doc_text(section))
    notes = [s for s in sentences if "1024 pixels on the longest side" in s and "unless" in s]
    if len(notes) != 1:
        return [f"expected one note on the working size, found {len(notes)}"]
    lifted = notes[0][notes[0].index("unless"):]
    problems = [
        f"{option} is left out" for option in (
            "Scale inpaint to bbox", "Try to match inpainting size to bounding box size"
        ) if option not in lifted
    ]
    if '"Inpaint only masked"' not in lifted:
        problems.append('the bbox options are not tied to "Inpaint only masked"')
    if re.search(r'"Inpaint only masked" (?:is )?off\b|\bwithout "Inpaint only masked"', lifted):
        problems.append('the bbox options are tied to "Inpaint only masked" off')
    problems += [
        f"{loose.strip()!r} makes \"Inpaint only masked\" optional"
        for loose in _LOOSE_CONDITIONS if loose in lifted
    ]
    return problems


def test_the_working_size_note_says_when_the_bbox_options_lift_the_cap():
    # fix_p2 keeps the capped working size in whole-picture mode before it
    # looks at "Scale inpaint to bbox" or the bbox size setting (see
    # test_whole_picture_inpaint_keeps_the_image_size), but the note on that
    # size said "Scale inpaint to bbox" lifts the cap with no condition, and
    # left the setting out, while the Size note after it gave the condition.
    script = _SCRIPT_PATH.read_text(encoding="utf-8")
    assert (  # premise: the code's order
        script.index("elif not args.ad_inpaint_only_masked:")
        < script.index("elif args.ad_use_resolution_scale:")
    )
    section = _readme_section("### Run ADetailer on an image", "### Batch a whole folder")

    assert _bbox_cap_note_problems(section) == []
    # Only the note is read: a right sentence on another topic that names
    # "Scale inpaint to bbox" (fix_p2 takes "Use separate width/height"
    # first) keeps the test green ...
    control = (
        '\n- A tab\'s "Use separate width/height" takes the place of'
        ' "Scale inpaint to bbox", as in a generation.\n'
    )
    assert _bbox_cap_note_problems(section + control) == []
    # ... and a note that states the wrong rule fails.
    note = (
        "\n- For a large picture, the regions are redrawn at a working size of"
        " at most 1024 pixels on the longest side, unless the tab uses"
        ' "Use separate width/height", {} or Settings → ADetailer →'
        ' "Try to match inpainting size to bounding box size".\n'
    )
    assert _bbox_cap_note_problems(
        note.format('or "Inpaint only masked" with "Scale inpaint to bbox"')
    ) == []
    for wrong in (
        'or "Scale inpaint to bbox"',  # no condition
        'or "Inpaint only masked" off with "Scale inpaint to bbox"',
        'or "Scale inpaint to bbox" (with "Inpaint only masked" on or off)',
        'or "Scale inpaint to bbox" whether or not "Inpaint only masked" is on',
        'or "Scale inpaint to bbox" regardless of "Inpaint only masked"',
        'or "Scale inpaint to bbox" without "Inpaint only masked"',
    ):
        assert _bbox_cap_note_problems(note.format(wrong)), wrong
    # The note as it was, which also left the bbox size setting out.
    assert _bbox_cap_note_problems(
        "\n- For a large picture, the regions are redrawn at a working size of"
        " at most 1024 pixels on the longest side, to avoid running out of"
        ' memory, unless the tab uses "Use separate width/height" or'
        ' "Scale inpaint to bbox".\n'
    )


def test_the_changelog_dates_the_bbox_note_to_beta_2():
    # The README's note that "Scale inpaint to bbox" lifts the 1024-pixel
    # working size first came with beta 2: the READMEs of the stable release
    # and beta 1 only said that the per-region regeneration is capped. The
    # beta 3 entry that corrects the note said "Earlier releases had the
    # same text", which by the CHANGELOG's own rule (see the beta 1 section)
    # takes in the stable release.
    older = re.compile(
        r"\b(?:earlier releases|the stable release|beta 1|plus\.7[\d.]*)\b[^.,;]*"
        r" had the same text",
        re.IGNORECASE,
    )
    bullets = [
        bullet for bullet in map(_flat_doc_text, _public_docs()["CHANGELOG.md"].split("\n- "))
        if "README" in bullet and "Scale inpaint to bbox" in bullet and "working size" in bullet
    ]

    assert bullets, "premise: the CHANGELOG has the entry on the note"
    for bullet in bullets:
        assert not older.search(bullet), bullet
    # The check itself: the old ending fails, the right one and an ending on
    # another topic pass.
    assert older.search("Earlier releases had the same text.")
    assert not older.search("Beta 2 had the same text.")
    assert not older.search("Earlier releases had the same problem.")


def _changelog_beta_heading(changelog, beta, version=None):
    """The ``## `` heading of ``changelog`` for ``version``, or the top one,
    when it is a section of beta number ``beta``; None otherwise."""
    headings = [line for line in changelog.splitlines() if line.startswith("## ")]
    if version is not None:
        headings = [h for h in headings if h.startswith(f"## {version} ")]
    if headings and re.match(rf"## v\S+\.beta\.{beta} ", headings[0]):
        return headings[0]
    return None


def _changelog_stable_heading(changelog, stable, version=None):
    """The ``## `` heading of ``changelog`` for ``version``, or the top one,
    when it is the section of the stable release ``stable`` (such as
    ``v1.0+plus.9``); None otherwise."""
    headings = [line for line in changelog.splitlines() if line.startswith("## ")]
    if version is not None:
        headings = [h for h in headings if h.startswith(f"## {version} ")]
    if headings and headings[0].startswith(f"## {stable} "):
        return headings[0]
    return None


def _current_version():
    """The version in adetailer/__version__.py, such as ``26.2.0+plus.8``."""
    return re.search(
        r'__version__ = "([^"]+)"',
        (_SCRIPT_PATH.parents[1] / "adetailer" / "__version__.py").read_text(encoding="utf-8"),
    ).group(1)


def test_a_summary_points_at_the_changelog_section_of_its_release():
    # Starting the beta 3 section put it on top of the CHANGELOG, and the
    # "Coming in beta 2" summary still sent readers to "the top section" for
    # beta 2's details. Under a beta's heading, or in a beta's own paragraph,
    # "the top" of the CHANGELOG must be that beta's section, and the
    # "The details are in" pointer must name it; headings and wording may
    # change. When the README is that of a stable release, a pointer outside
    # any beta's heading or paragraph describes that release: it must name,
    # or "the top" must be, the section of the version in __version__.py.
    changelog = _public_docs()["CHANGELOG.md"]
    current = _current_version()
    stable = None if ".beta." in current else f"v{current}"
    heading_beta = None
    checked = []
    for line in _public_docs()["README.md"].splitlines():
        if re.match(r"#{1,3} ", line):
            found = re.search(r"\bbeta (\d+)\b", line, re.IGNORECASE)
            heading_beta = found.group(1) if found else None
            continue
        own = re.match(r"\*\*Beta (\d+)\b", line)
        beta = own.group(1) if own else heading_beta
        named = re.search(r"\bthe (v\S+) section of \[CHANGELOG\.md\]", line)
        if not line.startswith("The details are in "):
            named = None  # another line may name another release's section
        top = re.search(r"\bthe top (?:section )?of \[CHANGELOG\.md\]", line)
        if not (named or top):
            continue
        version = named.group(1) if named else None
        if beta is None:
            if stable is None:
                continue
            assert _changelog_stable_heading(changelog, stable, version), (
                f"{stable}: {line}"
            )
            checked.append(stable)
            continue
        assert _changelog_beta_heading(changelog, beta, version), (
            f"beta {beta}: {line}"
        )
        checked.append(beta)

    assert checked, "premise: the README points at the CHANGELOG for its release"
    # The check itself, on a CHANGELOG whose top section is beta 3.
    sample = "# Changelog\n\n## v1.0+plus.9.beta.3 — x\n\n## v1.0+plus.9.beta.2 — y\n"
    assert _changelog_beta_heading(sample, "3")
    assert not _changelog_beta_heading(sample, "2")
    assert _changelog_beta_heading(sample, "2", "v1.0+plus.9.beta.2")
    assert not _changelog_beta_heading(sample, "3", "v1.0+plus.9.beta.2")
    assert not _changelog_beta_heading(sample, "2", "v1.0+plus.9.beta.1")
    # ... and on one whose top section is the stable release built on it.
    released = "# Changelog\n\n## v1.0+plus.9 — x\n\n## v1.0+plus.9.beta.3 — y\n"
    assert _changelog_stable_heading(released, "v1.0+plus.9")
    assert _changelog_stable_heading(released, "v1.0+plus.9", "v1.0+plus.9")
    assert not _changelog_stable_heading(released, "v1.0+plus.9", "v1.0+plus.9.beta.3")
    assert not _changelog_stable_heading(released, "v1.0+plus.8")
    assert not _changelog_stable_heading(sample, "v1.0+plus.9")


def test_the_readme_puts_a_beta_on_the_beta_branch_only_while_the_branch_holds_it():
    # The `beta` branch moved on to beta 3 (its badge shows that version), but
    # the beta 2 paragraph still said beta 2 was on the `beta` branch. A
    # beta's own paragraph may put it on that branch only while the version
    # there is that beta's. Once the stable release is out, the branch holds
    # that stable version until the next beta starts, so no beta's paragraph
    # may put the beta there, and the README says what the branch holds.
    version = _current_version()
    readme = _public_docs()["README.md"].splitlines()
    on_branch = re.compile(r"\bon\b[^.;]*\bthe `beta` branch\b")
    paragraphs = [line for line in readme if re.match(r"\*\*Beta \d+\b", line)]

    def holds(version):
        return re.compile(
            rf"\bthe `beta` branch holds\b[^;]*?\bv{re.escape(version)}(?![.\w])"
        )

    if ".beta." in version:
        assert paragraphs, "premise: the README has a paragraph for a beta"
    else:
        assert any(holds(version).search(line) for line in readme), (
            "premise: the README says that the `beta` branch holds the stable version"
        )
    for paragraph in paragraphs:
        beta = re.match(r"\*\*Beta (\d+)", paragraph).group(1)
        if on_branch.search(paragraph):
            assert version.endswith(f".beta.{beta}"), paragraph
    # The check itself: the wording of the old beta 2 paragraph is caught.
    assert on_branch.search("is on the Releases page and the `beta` branch.")
    assert not on_branch.search("is on the Releases page; the `beta` branch moved on.")
    # ... and the stable wording names the stable version, not a beta of it.
    stable = "the `beta` branch holds the same version as the stable release, v1.0+plus.9, until"
    assert holds("1.0+plus.9").search(stable)
    assert not holds("1.0+plus.9").search(stable.replace("plus.9,", "plus.9.beta.3,"))
    assert not holds("1.0+plus.8").search(stable)


def test_readme_says_which_tabs_generate_names_as_saved():
    # Generate saves every tab but names only the tabs with a detector, and
    # the start-up line too: the README said it named each tab it saved.
    ui_text = (_SCRIPT_PATH.parents[1] / "aaaaaa" / "ui.py").read_text(encoding="utf-8")
    assert 'state.get("ad_model") != "None"' in ui_text  # premise: the code's rule
    line = next(
        line for line in _public_docs()["README.md"].splitlines()
        if "saved tab 1 (" in line
    )
    first = re.sub(r"`[^`]*`", "", line).split(". ")[0]

    assert "names each tab it saved" not in first
    assert "with a detector" in first


_LORA_TAG = re.compile(r"<lora:[^>]+>")


@pytest.mark.parametrize(
    ("typed", "style", "options"),
    [
        ("portrait <lora:char:0.8>", ("<lora:x:1>, cinematic", ""), {}),
        ("portrait <lora:char:0.8>", ("cinematic still of {prompt}, <lora:x:1>", ""), {}),
        # A typed tag at another weight is kept, as in the main image.
        ("portrait <lora:x:0.5>", ("<lora:x:1>, cinematic", ""), {}),
        # The trigger phrase of the style's LoRA is still added.
        (
            "portrait <lora:char:0.8>", ("<lora:x (glow):1>, cinematic", ""),
            {"ad_use_lora_triggers": True},
        ),
        # Unchanged: a style without LoRAs.
        ("portrait <lora:char:0.8>", ("cinematic", ""), {}),
    ],
)
@pytest.mark.parametrize(
    "ad_prompt",
    ["", "[PROMPT], sharp", "detailed face", "[PROMPT] [CLASS=hand] five fingers [/CLASS]"],
)
def test_a_style_lora_reaches_the_detailer_pass_once_with_the_main_loras(
    ad_prompt, typed, style, options
):
    # "Use LoRAs from main prompt" took the LoRAs from the main prompt with
    # the selected styles in it, and the WebUI then applied the styles to the
    # detailer pass again: the style's LoRA was applied twice there.
    from adetailer.args import ADetailerArgs

    db = _StyleDatabase()
    db.styles = {"detail": style}
    runtime = _load_script(
        methods=_INLINE_PROMPT_METHODS,
        functions=_INLINE_PROMPT_FUNCTIONS,
        assigns=_INLINE_PROMPT_ASSIGNS,
        get_i=lambda _p: 0,
        shared=SimpleNamespace(prompt_styles=db),
    )
    script = runtime.AfterDetailerScript()

    def regions(styles):
        p = SimpleNamespace(
            prompt=typed, negative_prompt="", styles=styles,
            all_prompts=[db.apply_styles_to_prompt(typed, styles)],
            all_negative_prompts=[""],
        )
        args = ADetailerArgs(
            ad_model="face_yolov8n.pt", ad_prompt=ad_prompt, ad_use_main_loras=True,
            **options,
        )
        prompts, negatives = script.get_prompt(p, args)
        out = []
        for j in range(2):
            p2 = SimpleNamespace()
            script.i2i_prompts_replace(p2, prompts, negatives, j)
            script._apply_inline_class_prompts(
                p2, SimpleNamespace(class_names=["face", "hand"]), j, 2, False, p, args
            )
            # What the detailer pass gets once the WebUI has applied the styles.
            out.append((p2.prompt, db.apply_styles_to_prompt(p2.prompt, styles)))
        return p.all_prompts[0], out

    for styles in (["detail"], []):  # without styles, as before
        main, out = regions(styles)
        for prompt, final in out:
            # The same LoRAs as the main image, each as many times.
            assert sorted(_LORA_TAG.findall(final)) == sorted(_LORA_TAG.findall(main))
            if options and styles:
                assert "glow" in _LORA_TAG.sub("", prompt)


@pytest.mark.parametrize("main", [False, True])
@pytest.mark.parametrize(
    ("at_process", "resolved", "expected"),
    [
        # A batch whose images have different main prompts.
        (["a red cat", "a red dog"], ["a red cat", "a red dog"], ["a green cat", "a green dog"]),
        # A wildcard script that resolves the prompts after ADetailer's process().
        (["a red {cat|dog}"], ["a red dog"], ["a green dog"]),
    ],
)
def test_each_image_of_an_xyz_prompt_search_replace_cell_records_its_own_prompt(
    at_process, resolved, expected, main
):
    # The record was written once per cell, in process(): every image got the
    # first image's replaced main prompt, or the unresolved wildcard template,
    # and pasted back it detailed the region with another prompt.
    from aaaaaa.p_method import get_i, need_call_postprocess, need_call_process
    from adetailer.args import ADetailerArgs

    runtime = _load_script(
        methods=_INLINE_PROMPT_METHODS | {
            "process", "extra_params", "_record_xyz_prompt_sr", "postprocess_image",
        },
        functions=_INLINE_PROMPT_FUNCTIONS | {"search_and_replace_prompt"},
        assigns=_INLINE_PROMPT_ASSIGNS,
        PromptSR=_prompt_sr_class(),
        opts=SimpleNamespace(data={}),
        is_img2img_inpaint=lambda _p: False,
        get_i=get_i,
        suffix=_ui_suffix(),
        __version__="test",
        ensure_pil_image=lambda image, _mode: image,
        copy=copy,
        _verbose_gen_header=lambda *_args: None,
        need_call_postprocess=need_call_postprocess,
        need_call_process=need_call_process,
        CNHijackRestore=nullcontext,
        pause_total_tqdm=nullcontext,
        cn_allow_script_control=nullcontext,
        _should_skip_for_hires_only=lambda _p, _args: False,
        is_skip_img2img=lambda _p: False,
    )
    script = runtime.AfterDetailerScript()
    tab = {"ad_model": "face_yolov8n.pt", "ad_prompt": "", "ad_prompt_append": "detailed"}
    script.is_ad_enabled = lambda *_args: True
    script.set_skip_img2img = lambda *_args: None
    script.get_args = lambda *_args: [ADetailerArgs(**tab)]
    script.get_i2i_init_image = lambda _p, pp: pp.image
    script.read_params_txt = lambda: ""
    script.write_params_txt = lambda _content: None
    script._will_run_sequential = lambda _args: False
    script.save_image = lambda *_args, **_kwargs: None
    passes = []
    script._postprocess_image_inner = (
        lambda _p, _pp, args, n=0: passes.append(args.ad_prompt) or False
    )

    def as_saved(prompts):  # the "(AD 1st and main prompt)" axis replaced them
        return [text.replace("red", "green") for text in prompts] if main else prompts

    p = SimpleNamespace(
        iteration=0, batch_size=len(at_process), prompt=at_process[0],
        negative_prompt="", styles=[], scripts=None, extra_generation_params={},
    )
    runtime.search_and_replace_prompt(p, "green", ["red", "green"], main)
    p.all_prompts, p.all_negative_prompts = as_saved(at_process), [""] * len(at_process)
    script.process(p, True, False, {})
    p.all_prompts = as_saved(resolved)

    for index, record in enumerate(expected):
        p.batch_index = index
        image = SimpleNamespace(image=Image.new("RGB", (8, 8)))
        script.postprocess_image(p, image, True, False, {})
        params = p.extra_generation_params
        assert params["ADetailer prompt"] == record
        # Pasted back with the image's own main prompt, the region gets the
        # prompt it was detailed with.
        own = SimpleNamespace(
            iteration=0, batch_size=1, prompt=p.all_prompts[index], negative_prompt="",
            all_prompts=[p.all_prompts[index]], all_negative_prompts=[""], styles=[],
        )
        pasted = ADetailerArgs(**{**tab, "ad_prompt": params["ADetailer prompt"]})
        assert script.get_prompt(own, pasted) == script.get_prompt(p, ADetailerArgs(**tab))
    # The pass still gets the tab prompt as typed, and replaces it itself.
    assert passes == [""] * len(expected)


def test_the_readme_says_how_to_stop_a_standalone_or_folder_run():
    # The WebUI shows its Interrupt and Skip buttons only for its own
    # Generate; the README pointed to buttons a folder run never shows.
    readme = _public_docs()["README.md"]
    section = readme[readme.index("### Run ADetailer on an image"):]
    section = section[: section.index("### Manual mode")]
    assert "only while its own Generate runs" in section
    assert "Alt+Enter" in section
    assert "/sdapi/v1/interrupt" in section
    summary = next(line for line in readme.splitlines() if "Interrupt stops a folder run" in line)
    assert "Alt+Enter" in summary


def test_the_readme_reset_row_names_the_first_sampler():
    # Beta 2 puts the ADetailer sampler on the first sampler on Reset; the
    # NEW IN THIS FORK row still listed only the other exceptions to the
    # schema defaults.
    readme = _public_docs()["README.md"]
    row = readme[readme.index("\n### Reset\n"):]
    row = row[: row.index("\n### ", 1)]
    assert "puts every setting back to its default" in row
    assert "first sampler" in row
    ui_text = (_SCRIPT_PATH.parents[1] / "aaaaaa" / "ui.py").read_text(encoding="utf-8")
    assert '"ad_sampler": first_sampler' in ui_text


def test_the_readme_says_automatic1111_shows_the_reset_help_text():
    # AUTOMATIC1111 ships OptionHTML: only the OptionDiv divider is skipped
    # there, and the help text above the Reset button is shown.
    readme = _public_docs()["README.md"]
    row = next(
        line for line in readme.splitlines() if "AUTOMATIC1111" in line and "divider" in line
    )
    assert "divider/help" not in row
    assert "the help text, the button and every feature still work" in row
    source = _SCRIPT_PATH.read_text(encoding="utf-8")
    assert "divider + help block" not in source


# Debugging pass, round 12: generation pipeline, prompts and docs.


def _host_create_binary_mask(image, round=True):  # noqa: A002
    # AUTOMATIC1111 1.10 and Forge Neo modules/processing.py.
    if image.mode == "RGBA" and image.getextrema()[-1] != (255, 255):
        if round:
            image = image.split()[-1].convert("L").point(lambda x: 255 if x > 128 else 0)
        else:
            image = image.split()[-1].convert("L")
    else:
        image = image.convert("L")
    return image


def _mask_script(create_binary_mask=_host_create_binary_mask):
    from PIL import ImageChops

    from adetailer.common import ensure_pil_image

    return _load_script(
        methods={"get_image_mask"},
        Image=Image,
        ImageChops=ImageChops,
        ensure_pil_image=ensure_pil_image,
        create_binary_mask=create_binary_mask,
        images=SimpleNamespace(resize_image=lambda _mode, im, w, h: im.resize((w, h))),
        is_skip_img2img=lambda _p: False,
    ).AfterDetailerScript


def _mask_p(mask, invert):
    return SimpleNamespace(
        image_mask=mask, inpainting_mask_invert=invert, width=64, height=64,
        resize_mode=0,
    )


@pytest.mark.parametrize("invert", [False, True])
@pytest.mark.parametrize("under", [0, 255])
def test_an_inpaint_mask_in_its_alpha_channel_is_read_as_the_host_reads_it(
    under, invert
):
    # A mask painted on a transparent layer (Inpaint upload, an inpaint
    # Batch mask folder, the API) holds its area in the alpha channel, which
    # the host inpaints. ADetailer read its brightness: black under the paint
    # gave no mask ("adetailer disabled"), white the whole image, so faces
    # outside the painted area were detailed too ("Inpaint not masked" swaps
    # the two).
    from PIL import ImageOps

    from adetailer.mask import is_all_black

    mask = Image.new("RGBA", (64, 64), (under, under, under, 0))
    mask.paste((under, under, under, 255), (24, 24, 40, 40))
    host = _host_create_binary_mask(mask)  # the host's init, then its invert
    if invert:
        host = ImageOps.invert(host)

    result = _mask_script().get_image_mask(_mask_p(mask, invert))

    assert result.mode == "L"
    assert result.tobytes() == host.tobytes()
    assert not is_all_black(result)
    assert (result.getbbox() == (24, 24, 40, 40)) is not invert


def test_an_inpaint_mask_without_transparency_is_read_as_before():
    from PIL import ImageChops

    grey = Image.linear_gradient("L").resize((64, 64))
    masks = [
        grey,
        grey.point(lambda x: 255 if x > 128 else 0),
        Image.merge("RGB", (grey, grey.rotate(90), grey)),
        Image.merge("RGBA", (grey, grey.rotate(90), grey, Image.new("L", (64, 64), 255))),
    ]
    script = _mask_script()
    for mask in masks:
        for invert in (False, True):
            before = mask.convert("L")  # its brightness, as before
            if invert:
                before = ImageChops.invert(before)
            result = script.get_image_mask(_mask_p(mask, invert))
            assert result.tobytes() == before.tobytes(), (mask.mode, invert)


# Debugging pass, round 13: generation pipeline, prompts and docs.


def _faint_mask():
    # White paint at about 40% opacity on a transparent layer, with a
    # feathered edge (an alpha-80 ring) and an opaque centre.
    mask = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    mask.paste((255, 255, 255, 80), (8, 8, 56, 56))
    mask.paste((255, 255, 255, 100), (24, 24, 40, 40))
    mask.paste((255, 255, 255, 255), (28, 28, 36, 36))
    return mask


@pytest.mark.parametrize("invert", [False, True])
def test_a_faint_alpha_mask_is_read_unrounded_with_soft_inpainting(invert):
    # Soft inpainting sets p.mask_round = False, so the host keeps the faint
    # alpha and soft-inpaints it. ADetailer rounded it at 128: a mask painted
    # below half opacity became all black ("adetailer disabled"), and with
    # "Whole picture" a face touching only the faint ring was dropped.
    from PIL import ImageOps

    from adetailer.mask import is_all_black

    mask = _faint_mask()
    host = _host_create_binary_mask(mask, round=False)
    assert host.getextrema() == (0, 255)
    assert host.getbbox() == (8, 8, 56, 56)
    if invert:
        host = ImageOps.invert(host)
    p = _mask_p(mask, invert)
    p.mask_round = False

    result = _mask_script().get_image_mask(p)

    assert result.mode == "L"
    assert result.tobytes() == host.tobytes()
    assert not is_all_black(result)
    assert (result.getbbox() == (8, 8, 56, 56)) is not invert


def test_a_faint_alpha_mask_is_still_rounded_without_soft_inpainting():
    # The host rounds when mask_round is True or missing (WebUI 1.6/1.7 have
    # neither mask_round nor the round argument): so does ADetailer, and a
    # host whose create_binary_mask takes no round argument keeps working.
    from PIL import ImageOps

    mask = _faint_mask()
    for invert in (False, True):
        host = _host_create_binary_mask(mask)
        assert host.getbbox() == (28, 28, 36, 36)
        if invert:
            host = ImageOps.invert(host)
        with_flag = _mask_p(mask, invert)
        with_flag.mask_round = True
        result = _mask_script().get_image_mask(with_flag)
        assert result.tobytes() == host.tobytes(), invert
        old_host = _mask_script(lambda image: _host_create_binary_mask(image))
        result = old_host.get_image_mask(_mask_p(mask, invert))
        assert result.tobytes() == host.tobytes(), invert


def test_an_opaque_or_binary_mask_is_read_the_same_with_soft_inpainting():
    from PIL import ImageChops

    grey = Image.linear_gradient("L").resize((64, 64))
    binary = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    binary.paste((0, 0, 0, 255), (24, 24, 40, 40))
    masks = [
        grey,
        Image.merge("RGB", (grey, grey.rotate(90), grey)),
        Image.merge("RGBA", (grey, grey.rotate(90), grey, Image.new("L", (64, 64), 255))),
        binary,
    ]
    script = _mask_script()
    for mask in masks:
        for invert in (False, True):
            rounded = script.get_image_mask(_mask_p(mask, invert))
            p = _mask_p(mask, invert)
            p.mask_round = False
            result = script.get_image_mask(p)
            assert result.tobytes() == rounded.tobytes(), (mask.mode, invert)
    before = ImageChops.invert(grey)  # opaque masks keep their brightness
    p = _mask_p(grey, True)
    p.mask_round = False
    assert script.get_image_mask(p).tobytes() == before.tobytes()


def _a1111_strip_comments(text):
    # AUTOMATIC1111 1.10 modules/processing_scripts/comments.py.
    text = re.sub("(^|\n)#[^\n]*(\n|$)", "\n", text)
    text = re.sub("#[^\n]*(\n|$)", "\n", text)
    return text


def _neo_strip_comments(text):
    # Forge Neo modules/processing_scripts/comments.py, with its option on.
    text = re.sub(r"\/\*.*?\*\/", "", text, flags=re.DOTALL)
    return re.sub(r"[^\S\n]*(\#|\/\/).*", "", text)


def _comments_runtime(db, strip, data=None):
    # The Comments script as the host loads it: a module named "comments.py".
    comments = SimpleNamespace(
        script_class=type("ScriptStripComments", (), {"__module__": "comments.py"}),
        module=SimpleNamespace(strip_comments=strip),
    )
    return _load_script(
        methods=_INLINE_PROMPT_METHODS,
        functions=_INLINE_PROMPT_FUNCTIONS | {"_strip_host_comments"},
        assigns=_INLINE_PROMPT_ASSIGNS,
        get_i=lambda _p: 0,
        shared=SimpleNamespace(prompt_styles=db),
        opts=SimpleNamespace(data=data or {}),
        scripts=SimpleNamespace(scripts_data=[comments] if strip else []),
    )


@pytest.mark.parametrize("ad_prompt", ["", "[PROMPT], detailed face"])
@pytest.mark.parametrize(
    ("strip", "prompt", "negative"),
    [
        (
            _a1111_strip_comments,
            "a woman\n# alt: a man\nin a park", "bad hands\n# old: extra fingers\nugly",
        ),
        (_neo_strip_comments, "a woman\n// alt: a man\nin a park", "bad hands /* old */\nugly"),
    ],
)
def test_the_selected_styles_reach_the_detailer_pass_once_with_prompt_comments(
    strip, prompt, negative, ad_prompt
):
    # The WebUI's Comments script strips the main prompts after styling them,
    # so the typed prompt with its comment never gave them back: the pass
    # took the styled main prompt and the WebUI styled it again.
    from adetailer.args import ADetailerArgs

    db = _StyleDatabase()
    styles = ["cinematic"]
    script = _comments_runtime(db, strip).AfterDetailerScript()
    p = SimpleNamespace(
        prompt=prompt, negative_prompt=negative, styles=styles,
        all_prompts=[strip(db.apply_styles_to_prompt(prompt, styles))],
        all_negative_prompts=[strip(db.apply_negative_styles_to_prompt(negative, styles))],
    )
    args = ADetailerArgs(ad_model="face_yolov8n.pt", ad_prompt=ad_prompt)

    prompts, negatives = script.get_prompt(p, args)

    # What the detailer pass gets once the host has applied the styles.
    positive = db.apply_styles_to_prompt(prompts[0], styles)
    assert positive.count("cinematic still of") == 1
    assert positive.count("35mm film") == 1
    assert "alt: a man" not in positive  # the pass does not strip comments
    assert db.apply_negative_styles_to_prompt(negatives[0], styles) == (
        p.all_negative_prompts[0]
    )
    if not ad_prompt:
        assert positive == p.all_prompts[0]


def test_prompt_comments_leave_the_styles_check_alone_without_the_comments_script():
    from adetailer.args import ADetailerArgs

    db = _StyleDatabase()
    styles = ["cinematic"]
    prompt = "a woman\n# alt: a man\nin a park"
    args = ADetailerArgs(ad_model="face_yolov8n.pt")
    p = SimpleNamespace(
        prompt=prompt, negative_prompt="", styles=styles,
        all_prompts=[db.apply_styles_to_prompt(prompt, styles)],
        all_negative_prompts=[db.apply_negative_styles_to_prompt("", styles)],
    )
    # "Enable comments" off: the main prompt keeps its comment, so does the pass.
    off = _comments_runtime(db, _a1111_strip_comments, {"enable_prompt_comments": False})
    assert off.AfterDetailerScript().get_prompt(p, args)[0] == [prompt]
    # No Comments script (WebUI before 1.8) and main prompts that the styles
    # do not give back: the main prompts, as before.
    p.all_prompts = [_a1111_strip_comments(p.all_prompts[0])]
    missing = _comments_runtime(db, None)
    assert missing.AfterDetailerScript().get_prompt(p, args)[0] == p.all_prompts


_INSTALLED = {"face_yolov8n.pt": "face_yolov8n.pt", "hand_yolov8n.pt": "hand_yolov8n.pt"}


@pytest.mark.parametrize(
    ("tabs", "enabled", "kept", "dropped"),
    [
        # Only a detector that is not installed here.
        ((("face_yolov9c.pt", "face"),), None, set(), {"ADetailer model"}),
        ((("顔_yolov8n.pt", "face"),), None, set(), {"ADetailer model"}),
        # An installed tab still switches ADetailer on and keeps its detector.
        (
            (("face_yolov9c.pt", "face"), ("hand_yolov8n.pt", "")),
            "True", {"ADetailer model 2nd"}, {"ADetailer model"},
        ),
        (
            (("face_yolov8n.pt", "face"), ("hand_yolov9c.pt", "")),
            "True", {"ADetailer model"}, {"ADetailer model 2nd"},
        ),
        # Unchanged: every detector installed.
        ((("face_yolov8n.pt", "face"),), "True", {"ADetailer model"}, set()),
    ],
)
def test_pasting_a_detector_that_is_not_installed_leaves_its_tab_and_adetailer_off(
    capsys, tabs, enabled, kept, dropped
):
    # PNG Info or Send to on an image made with a detector missing here
    # switched ADetailer and that tab on with a detector the dropdown does not
    # have: Generate then failed that tab and every later one ("not found").
    from adetailer.args import ADetailerArgs

    params = _infotext(
        *(ADetailerArgs(ad_model=m, ad_model_classes=c, ad_prompt="a face") for m, c in tabs)
    )

    _paste_callback(model_mapping=_INSTALLED)._clear_missing_class_prompts("", params)

    assert params.get("ADetailer enable") == enabled
    assert kept <= set(params)
    assert not dropped & set(params)
    # The rest of the tab is pasted, as Load applies the rest of a preset.
    # Without its detector key, and with "ADetailer version" there, the paste
    # handlers in aaaaaa/ui.py leave the tab's detector and class filter as
    # they are and switch the tab off.
    assert params["ADetailer prompt"] == "a face"
    assert "ADetailer version" in params
    out = capsys.readouterr().out
    assert out.isascii()
    assert out.count("is not installed") == len(dropped)


def _readme_section(start, end):
    readme = _public_docs()["README.md"]
    section = readme[readme.index(start):]
    return section[: section.index(end)]


def test_the_readme_says_which_base_settings_a_standalone_run_uses():
    # There is no main generation to take them from: without the tab's "Use
    # separate ..." options every region gets the values below, which burn a
    # guidance-distilled or few-step checkpoint, and no text said so.
    tree = ast.parse(_SCRIPT_PATH.read_text(encoding="utf-8"))
    run = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_detailer_on_image"
    )
    shell = {
        k.arg: k.value.value
        for node in ast.walk(run)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "SimpleNamespace"
        for k in node.keywords
        if k.arg in {"steps", "cfg_scale"} and isinstance(k.value, ast.Constant)
    }
    assert set(shell) == {"steps", "cfg_scale"}
    section = _readme_section("### Run ADetailer on an image", "### Manual mode")
    assert f"{shell['steps']} steps" in section
    assert f"CFG scale {shell['cfg_scale']:g}" in section
    for option in ("Use separate steps", "Use separate CFG scale", "Use separate sampler"):
        assert f'"{option}"' in section


def test_the_readme_says_that_a_forge_controlnet_preprocessor_list_starts_with_none():
    # On Forge-based WebUIs the list starts with "None" (the image as it is),
    # which choosing a model keeps or falls back to: the README said the
    # preprocessor was set automatically everywhere.
    ui_text = (_SCRIPT_PATH.parents[1] / "aaaaaa" / "ui.py").read_text(encoding="utf-8")
    assert "current_module if current_module in choices else choices[0]" in ui_text
    section = _readme_section("**ControlNet** (at the bottom of each tab)", "\n## Models")
    assert 'the preprocessor list starts with "None"' in section
    assert "pick the preprocessor yourself" in section


def test_the_readme_says_when_the_styles_still_reach_the_pass_twice():
    # When another extension rewrites the main prompts (wildcards), the
    # styles do not give them back and the pass keeps the old behaviour,
    # as the CHANGELOG says; three README passages promised "once" anyway.
    docs = _public_docs()
    readme, changelog = docs["README.md"], docs["CHANGELOG.md"]
    lines = readme.splitlines()
    row = next(line for line in lines if line.startswith("| ADetailer prompt, ADetailer negative prompt |"))
    loras = next(line for line in lines if "A LoRA that a selected Style adds is not merged" in line)
    summary = next(line for line in lines if "reach the detailer pass once" in line)
    summary = summary[summary.index("reach the detailer pass once"):][:300]
    lora_fix = next(
        line for line in changelog.splitlines()
        if line.startswith('- **With "Use LoRAs from main prompt", a LoRA from a selected Style')
    )
    for text in (row, loras, summary, lora_fix):
        assert "rewrites the main prompts" in text


def test_the_readme_states_the_preset_name_rule_the_code_applies():
    # The README allowed any printable name without path separators or
    # quotes, but ":", "%", "=" and most other symbols are refused.
    import string

    from adetailer import presets

    readme = _public_docs()["README.md"]
    line = next(line for line in readme.splitlines() if line.startswith("- **Save:**"))
    assert "printable, no path separators or quotes" not in line
    listed = set(re.search(r"spaces and `([^`]+)`", line).group(1).split())
    assert len(listed) > 10
    for ch in string.punctuation:
        assert presets.is_valid_name(f"face{ch}v2") is (ch in listed), ch
    assert presets.is_valid_name("face v2")
    assert presets.is_valid_name("x" * 80)
    assert not presets.is_valid_name("x" * 81)


def test_the_beta_1_notes_name_only_reforges_gradio_4_branches_for_the_reset_regression():
    # The cause is Gradio 4, and reForge's main branch runs Gradio 3.
    changelog = _public_docs()["CHANGELOG.md"]
    bullet = next(
        line for line in changelog.splitlines()
        if line.startswith('- **"Reset every tab" reset only the first tab')
    )
    assert "Forge Neo and reForge**" not in bullet
    assert "reForge's Gradio 4 branches" in bullet
    assert "reForge's main branch (Gradio 3)" in bullet


def test_the_readme_says_that_comments_typed_in_an_adetailer_prompt_reach_the_model():
    # The detailer pass keeps only these scripts by default, and not the
    # WebUI's Comments script, which removes the comments of the main prompt.
    from adetailer.args import BUILTIN_SCRIPT, SCRIPT_DEFAULT

    kept = {name.strip() for name in f"{SCRIPT_DEFAULT},{BUILTIN_SCRIPT}".split(",")}
    assert "comments" not in kept
    readme = _public_docs()["README.md"]
    note = next(line for line in readme.splitlines() if line.startswith("Prompt comments ("))
    assert '"Apply only selected scripts to ADetailer"' in note
    assert '"Script names to apply to ADetailer"' in note
    assert "append text" in note


class _HostI2I:
    """The WebUI's img2img pass as get_i2i_p builds it: keeps its keywords."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.mark.parametrize(("tab", "enabled"), [(0, True), (1, True), (1, False)])
def test_a_standalone_result_records_its_tabs_adetailer_parameters(
    tmp_path, tab, enabled
):
    # "Run ADetailer on an image" and folder runs saved only the img2img
    # pass's parameters: PNG Info and Send to brought back that pass but not
    # the tab. The pass now carries the "ADetailer ..." keys a generation
    # where only this tab runs writes, with this tab's suffix (" 2nd" for the
    # 2nd tab), also when its "Enable this tab" is off, as the run ignores it.
    # With the 2nd tab enabled too: a disabled tab writes no keys, so only an
    # enabled one shows that the tabs before it are written as off.
    from adetailer.args import ADetailerArgs

    sfx = _ui_suffix()
    runtime = _load_script(
        methods={
            "run_detailer_on_image", "read_params_txt", "write_params_txt",
            "extra_params", "get_i2i_p",
        },
        paths=SimpleNamespace(data_path=str(tmp_path)),
        PARAMS_TXT="params.txt",
        shared=SimpleNamespace(sd_model=None),
        state=SimpleNamespace(interrupted=False, skipped=False),
        opts=SimpleNamespace(samples_format="png"),
        all_samplers=[],
        ensure_pil_image=lambda image, _mode: image,
        images=None,
        AD_APPLY_SUBDIR="ADetailer-Inpaint",
        pause_total_tqdm=nullcontext,
        suffix=sfx,
        __version__="test",
        StableDiffusionProcessingImg2Img=_HostI2I,
        schedulers=None,
        controlnet_type="forge",
        copy_extra_params=dict,
    )
    script = runtime.AfterDetailerScript()
    script.get_seed = lambda _p: (1, 1)
    script.get_width_height = lambda *_args: (64, 64)
    script.get_steps = lambda *_args: 28
    script.get_cfg_scale = lambda *_args: 7.0
    script.get_initial_noise_multiplier = lambda *_args: None
    script.get_sampler = lambda *_args: "Euler"
    script.get_override_settings = lambda *_args: {}
    script.script_filter = lambda *_args: (None, [])
    made = ADetailerArgs(
        ad_model="face_yolov8n.pt", ad_prompt="detailed face", ad_confidence=0.5,
        ad_tab_enable=enabled,
    )
    passes = []

    def inner(p, pp, args, n=0):
        passes.append(script.get_i2i_p(p, args, pp.image).extra_generation_params)
        return False

    script._postprocess_image_inner = inner

    script.run_detailer_on_image(Image.new("RGB", (64, 64)), made, tab=tab)

    # What a generation writes (process()) when this tab runs and the tabs
    # before it are off.
    ran = made.copy(update={"ad_tab_enable": True})
    expected = runtime.AfterDetailerScript().extra_params(
        [*([ADetailerArgs()] * tab), ran]
    )
    assert passes == [expected]
    assert all(k.endswith(sfx(tab)) for k in expected if k != "ADetailer version")
    assert expected["ADetailer model" + sfx(tab)] == "face_yolov8n.pt"

    # Pasted back (PNG Info, Send to), they switch ADetailer on and give the
    # tab its setup again.
    params = {k: str(v) for k, v in expected.items()}
    _paste_callback()._clear_missing_class_prompts("", params)
    before = ADetailerArgs(ad_model="hand_yolov8n.pt", ad_prompt="detailed eyes").dict()
    after = ADetailerArgs(**_host_paste(params, before, suffix=sfx(tab)))
    assert params["ADetailer enable"] == "True"
    assert after.extra_params() == ran.extra_params()


def test_a_standalone_result_keeps_its_passs_parameters_when_the_tab_cannot_be_read(
    tmp_path,
):
    # Guarded: a tab whose keys cannot be built leaves the pass with its own
    # parameters, as before, and the run goes on.
    script = _standalone_runtime(tmp_path).AfterDetailerScript()
    shells = []

    def inner(p, _pp, _args, n=0):
        shells.append(dict(p.extra_generation_params))
        return False

    script._postprocess_image_inner = inner

    _image, status = script.run_detailer_on_image(
        Image.new("RGB", (64, 64)), SimpleNamespace(), tab=1
    )

    assert shells == [{}]
    assert "Nothing detected" in status
