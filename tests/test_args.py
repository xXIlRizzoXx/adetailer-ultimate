from __future__ import annotations

import pytest

from adetailer.args import ALL_ARGS, ADetailerArgs


def test_all_args() -> None:
    args = ADetailerArgs()
    for attr, _ in ALL_ARGS:
        assert hasattr(args, attr), attr

    for attr, _ in args:
        if attr == "is_api":
            continue
        assert attr in ALL_ARGS.attrs, attr


# The UI registers one .change per entry and tab, and the infotext keys are
# these names: one more entry, a rename or a new order would shift Forge's
# event indices or stop older images from pasting back.
_ALL_ARGS_OF_MAIN = [
    ("ad_model", "ADetailer model"),
    ("ad_model_classes", "ADetailer model classes"),
    ("ad_model_classes_exclude", "ADetailer classes exclude"),
    ("ad_model_classes_excluded", "ADetailer model classes excluded"),
    ("ad_classes_sequential", "ADetailer classes sequential"),
    ("ad_tab_enable", "ADetailer tab enable"),
    ("ad_prompt", "ADetailer prompt"),
    ("ad_negative_prompt", "ADetailer negative prompt"),
    ("ad_prompt_append", "ADetailer prompt append"),
    ("ad_negative_prompt_append", "ADetailer negative prompt append"),
    ("ad_class_prompts", "ADetailer class prompts"),
    ("ad_class_guard", "ADetailer class guard"),
    ("ad_class_guard_weight", "ADetailer class guard weight"),
    ("ad_use_main_loras", "ADetailer use main loras"),
    ("ad_use_lora_triggers", "ADetailer use lora triggers"),
    ("ad_strip_loras", "ADetailer strip loras"),
    ("ad_apply_on_hires_only", "ADetailer apply on hires only"),
    ("ad_use_bbox_mask", "ADetailer use bbox mask"),
    ("ad_confidence", "ADetailer confidence"),
    ("ad_detection_resolution", "ADetailer detection resolution"),
    ("ad_mask_filter_method", "ADetailer method to decide top k masks"),
    ("ad_mask_k", "ADetailer mask only top k"),
    ("ad_mask_min_ratio", "ADetailer mask min ratio"),
    ("ad_mask_max_ratio", "ADetailer mask max ratio"),
    ("ad_x_offset", "ADetailer x offset"),
    ("ad_y_offset", "ADetailer y offset"),
    ("ad_dilate_erode", "ADetailer dilate erode"),
    ("ad_mask_merge_invert", "ADetailer mask merge invert"),
    ("ad_mask_blur", "ADetailer mask blur"),
    ("ad_denoising_strength", "ADetailer denoising strength"),
    ("ad_dynamic_denoise_power", "ADetailer dynamic denoise power"),
    ("ad_inpaint_only_masked", "ADetailer inpaint only masked"),
    ("ad_inpaint_only_masked_padding", "ADetailer inpaint padding"),
    ("ad_use_inpaint_width_height", "ADetailer use inpaint width height"),
    ("ad_inpaint_width", "ADetailer inpaint width"),
    ("ad_inpaint_height", "ADetailer inpaint height"),
    ("ad_use_resolution_scale", "ADetailer use resolution scale"),
    ("ad_resolution_scale", "ADetailer resolution scale"),
    ("ad_use_steps", "ADetailer use separate steps"),
    ("ad_steps", "ADetailer steps"),
    ("ad_use_cfg_scale", "ADetailer use separate CFG scale"),
    ("ad_cfg_scale", "ADetailer CFG scale"),
    ("ad_use_checkpoint", "ADetailer use separate checkpoint"),
    ("ad_checkpoint", "ADetailer checkpoint"),
    ("ad_use_vae", "ADetailer use separate VAE"),
    ("ad_vae", "ADetailer VAE"),
    ("ad_use_text_encoder", "ADetailer use separate text encoder"),
    ("ad_text_encoder", "ADetailer text encoder"),
    ("ad_use_sampler", "ADetailer use separate sampler"),
    ("ad_sampler", "ADetailer sampler"),
    ("ad_scheduler", "ADetailer scheduler"),
    ("ad_use_noise_multiplier", "ADetailer use separate noise multiplier"),
    ("ad_noise_multiplier", "ADetailer noise multiplier"),
    ("ad_use_clip_skip", "ADetailer use separate CLIP skip"),
    ("ad_clip_skip", "ADetailer CLIP skip"),
    ("ad_restore_face", "ADetailer restore face"),
    ("ad_controlnet_model", "ADetailer ControlNet model"),
    ("ad_controlnet_module", "ADetailer ControlNet module"),
    ("ad_controlnet_weight", "ADetailer ControlNet weight"),
    ("ad_controlnet_guidance_start", "ADetailer ControlNet guidance start"),
    ("ad_controlnet_guidance_end", "ADetailer ControlNet guidance end"),
    ("ad_inpaint_indices", "ADetailer inpaint indices"),
    # plus.10: the one entry added since, at the end, so the UI's positions
    # above stay where they were.
    ("ad_inpaint_masked_content", "ADetailer inpaint masked content"),
]


def test_all_args_keep_their_names_and_order() -> None:
    assert len(ALL_ARGS) == 63
    assert [tuple(arg) for arg in ALL_ARGS] == _ALL_ARGS_OF_MAIN


@pytest.mark.parametrize(
    ("choice", "fill"),
    [("fill", 0), ("original", 1), ("latent noise", 2), ("latent nothing", 3)],
)
def test_masked_content_gives_the_webuis_inpainting_fill(choice: str, fill: int) -> None:
    # The WebUI's img2img numbers its "Masked content" choices 0-3.
    args = ADetailerArgs(ad_model="face_yolov8n.pt", ad_inpaint_masked_content=choice)
    assert args.inpainting_fill() == fill


def test_masked_content_defaults_to_original() -> None:
    # Older presets, remembered settings and API requests do not have the
    # key: they keep the fixed "original" (1) the detailer pass always used.
    old = {"ad_model": "face_yolov8n.pt", "ad_prompt": "detailed face"}
    args = ADetailerArgs(**old)
    assert args.ad_inpaint_masked_content == "original"
    assert args.inpainting_fill() == 1


@pytest.mark.parametrize(
    "value", ["noise", "latent  noise", "", "4", "-1", "1.0", 4, -1, 1.5, True, None]
)
def test_masked_content_takes_only_the_webuis_choices(value: object) -> None:
    assert ADetailerArgs(ad_inpaint_masked_content="latent noise")
    with pytest.raises(ValueError, match="ad_inpaint_masked_content"):
        ADetailerArgs(ad_inpaint_masked_content=value)


@pytest.mark.parametrize(
    ("value", "choice"),
    [
        # Any letter case, spaces around it.
        ("Original", "original"),
        ("Latent Noise", "latent noise"),
        (" FILL ", "fill"),
        ("latent nothing\n", "latent nothing"),
        # The WebUI's own inpainting_fill number, as a number or as text.
        (0, "fill"),
        (1, "original"),
        (2, "latent noise"),
        (3, "latent nothing"),
        ("1", "original"),
        (" 2 ", "latent noise"),
    ],
)
def test_masked_content_accepts_the_webuis_number_and_any_letter_case(
    value: object, choice: str
) -> None:
    # An API request with "Original" or 1 was an error and the tab was skipped.
    args = ADetailerArgs(ad_model="face_yolov8n.pt", ad_inpaint_masked_content=value)
    assert args.ad_inpaint_masked_content == choice
    assert args.inpainting_fill() == ["fill", "original", "latent noise",
                                      "latent nothing"].index(choice)


def test_masked_content_is_in_the_parameters_only_when_changed() -> None:
    plain = ADetailerArgs(ad_model="face_yolov8n.pt").extra_params(suffix=" 2nd")
    assert not any(k.startswith("ADetailer inpaint masked content") for k in plain)

    changed = ADetailerArgs(
        ad_model="face_yolov8n.pt", ad_inpaint_masked_content="latent noise"
    ).extra_params(suffix=" 2nd")
    assert changed["ADetailer inpaint masked content 2nd"] == "latent noise"
    assert "ADetailer inpaint masked content" not in changed


@pytest.mark.parametrize(
    ("ad_model", "expect"),
    [("mediapipe_face_full", True), ("face_yolov8n.pt", False)],
)
def test_is_mediapipe(ad_model: str, expect: bool) -> None:
    args = ADetailerArgs(ad_model=ad_model)
    assert args.is_mediapipe() is expect


@pytest.mark.parametrize(
    ("ad_model", "expect"),
    [
        ("mediapipe_face_full", False),
        ("face_yolov8n.pt", False),
        ("None", True),
        # An API request with an empty model: the tab is off, like "None",
        # instead of failing with "Model '' not found" and stopping the tabs
        # after it.
        ("", True),
        ("   ", True),
    ],
)
def test_need_skip(ad_model: str, expect: bool) -> None:
    args = ADetailerArgs(ad_model=ad_model)
    assert args.need_skip() is expect


@pytest.mark.parametrize(
    ("ad_model", "ad_tab_enable", "expect"),
    [
        ("face_yolov8n.pt", False, True),
        ("mediapipe_face_full", False, True),
        ("None", True, True),
        ("ace_yolov8s.pt", True, False),
    ],
)
def test_need_skip_tab_enable(ad_model: str, ad_tab_enable: bool, expect: bool) -> None:
    args = ADetailerArgs(ad_model=ad_model, ad_tab_enable=ad_tab_enable)
    assert args.need_skip() is expect
