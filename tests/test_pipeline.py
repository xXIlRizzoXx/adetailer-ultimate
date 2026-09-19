"""Isolated runtime regressions without starting a WebUI or loading a model.

The host script imports WebUI-only modules and downloads detectors at import
time. Extract its actual function bodies through AST and provide the small host
surface needed by each test. These tests cover control flow and override values;
they do not replace an end-to-end generation test in A1111 / Forge.
"""

from __future__ import annotations

import ast
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


def _standalone_runtime(tmp_path, **host_globals):
    state = SimpleNamespace(
        interrupted=False, skipped=False, stopping_generation=False
    )
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

    def inner(_p, _pp, _args):
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


def test_standalone_run_gives_every_region_a_random_seed(tmp_path, monkeypatch):
    values = iter([1000, 2000, 3000, 4000])
    monkeypatch.setattr(random, "randrange", lambda _stop: next(values))
    script = _standalone_runtime(tmp_path).AfterDetailerScript()
    shells = []

    def inner(p, _pp, _args):
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
        functions={"_parse_class_prompts", "_resolve_inline_class_prompt"},
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
        # No readable number keeps every detection, and the console says so.
        ("x", [list(_FACE_BOXES[0]), list(_FACE_BOXES[1]), list(_HAND_BOX)], True),
        # Also for non-Latin text, echoed as ASCII so no console code page
        # can fail to print it.
        ("\u4e8c", [list(_FACE_BOXES[0]), list(_FACE_BOXES[1]), list(_HAND_BOX)], True),
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
    assert out.isascii()


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


def _paste_callback(skipped=()):
    return _load_script(
        functions={"_clear_missing_class_prompts"},
        assigns={
            "_INFOTEXT_MODEL_KEY",
            "_INFOTEXT_PASTE_DEFAULTS",
            "_INFOTEXT_PASTE_DEPENDENT_DEFAULTS",
        },
        shared=SimpleNamespace(
            opts=SimpleNamespace(infotext_skip_pasting=list(skipped))
        ),
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
        (
            "[CLASS=hand]five fingers[/CLASS]", "[CLASS=hand]extra fingers[/CLASS]",
            "detailed",
            ("portrait, smiling, detailed", "blurry"),
            ("portrait, smiling, five fingers, detailed", "blurry, extra fingers"),
        ),
        # Unchanged: text outside the blocks, and a prompt without blocks.
        (
            "detailed, [CLASS=hand] [SKIP] [/CLASS]", "", "",
            ("detailed", "blurry"), ("[SKIP]", "blurry"),
        ),
        ("detailed face", "", "", ("detailed face", "blurry"), ("detailed face", "blurry")),
    ],
)
def test_a_prompt_of_only_class_blocks_keeps_the_main_prompt(
    prompt, negative, append, face, hand
):
    # A blank ADetailer prompt means the main prompt. A prompt made only of
    # [CLASS=...] blocks has blank text outside them, so every other region
    # used to be inpainted with an empty prompt.
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
        script._apply_inline_class_prompts(p2, pred, j, 2, False)
        regions.append((p2.prompt, p2.negative_prompt))

    assert regions == [face, hand]


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
        functions={"_parse_class_prompts"},
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
        methods=_INLINE_PROMPT_METHODS | {"process", "extra_params"},
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
    assert status == "✅ ADetailer pass complete."


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
    row = next(
        line for line in readme.splitlines()
        if line.startswith("| 🟢 | Sequential class detection |")
    )
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
