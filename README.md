# ADetailer Plus

> **About this fork** — `adetailer-plus` is a soft-fork of [Bing-su/adetailer](https://github.com/Bing-su/adetailer) that adds workflow features on top of upstream ADetailer. The features so far:
>
> - **Per-class filtering** for multiclass YOLO detection models — auto-populated dropdown + "Exclude selected (NOT)" mode (see [Class Filtering](#class-filtering)).
> - **Sequential class detection** — process selected classes one at a time, each pass refining the previous (see [Sequential Mode](#sequential-class-detection)).
> - **Copy settings from 1st** — replicate the 1st tab's processing settings to the 2nd/3rd/4th tabs with one click (see [Copy Between Tabs](#copy-settings-between-tabs)).
> - **Forge Neo compatibility fixes** — `disable_safe_unpickle` patch, JSON sidecar tolerance for civitai_helper-generated metadata.
>
> The implementation was authored by **Claude** (Anthropic's coding assistant) at the request of the repository owner, who is not a Python developer. The class-filtering pattern is borrowed from [wkpark/uddetailer](https://github.com/wkpark/uddetailer). All credit for the original ADetailer goes to **Bing-su**; this fork extends that work — it does not replace it.
>
> If you only need single-class face/hand detection, use upstream ADetailer instead — the additional widgets in this fork are harmless but unnecessary for that case.

---

ADetailer is an extension for the stable diffusion webui that does automatic masking and inpainting. It is similar to the Detection Detailer.

## Install

You can install it directly from the Extensions tab.

![image](https://i.imgur.com/qaXtoI6.png)

Or

(from Mikubill/sd-webui-controlnet)

1. Open "Extensions" tab.
2. Open "Install from URL" tab in the tab.
3. Enter `https://github.com/Bing-su/adetailer.git` to "URL for extension's git repository".
4. Press "Install" button.
5. Wait 5 seconds, and you will see the message "Installed into stable-diffusion-webui\extensions\adetailer. Use Installed tab to restart".
6. Go to "Installed" tab, click "Check for updates", and then click "Apply and restart UI". (The next time you can also use this method to update extensions.)
7. Completely restart A1111 webui including your terminal. (If you do not know what is a "terminal", you can reboot your computer: turn your computer off and turn it on again.)

## Options

| Model, Prompts                    |                                                                                    |                                                                                                                                                        |
| --------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ADetailer model                   | Determine what to detect.                                                          | `None` = disable                                                                                                                                       |
| ADetailer model classes           | **(Fork)** For YOLO-World models: comma separated class names to detect (open vocabulary). For other multiclass YOLO models: classes are auto-populated in a dropdown — see [Class Filtering](#class-filtering-fork-feature) below. | If blank, all classes the model produces are inpainted (upstream behavior). |
| ADetailer prompt, negative prompt | Prompts and negative prompts to apply                                              | If left blank, it will use the same as the input.                                                                                                      |
| Skip img2img                      | Skip img2img. In practice, this works by changing the step count of img2img to 1.  | img2img only                                                                                                                                           |

| Detection                            |                                                                                              |              |
| ------------------------------------ | -------------------------------------------------------------------------------------------- | ------------ |
| Detection model confidence threshold | Only objects with a detection model confidence above this threshold are used for inpainting. |              |
| Mask min/max ratio                   | Only use masks whose area is between those ratios for the area of the entire image.          |              |
| Mask only the top k largest          | Only use the k objects with the largest area of the bbox.                                    | 0 to disable |

If you want to exclude objects in the background, try setting the min ratio to around `0.01`.

| Mask Preprocessing              |                                                                                                                                     |                                                                                         |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Mask x, y offset                | Moves the mask horizontally and vertically by                                                                                       |                                                                                         |
| Mask erosion (-) / dilation (+) | Enlarge or reduce the detected mask.                                                                                                | [opencv example](https://docs.opencv.org/4.7.0/db/df6/tutorial_erosion_dilatation.html) |
| Mask merge mode                 | `None`: Inpaint each mask<br/>`Merge`: Merge all masks and inpaint<br/>`Merge and Invert`: Merge all masks and Invert, then inpaint |                                                                                         |

Applied in this order: x, y offset → erosion/dilation → merge/invert.

#### Inpainting

Each option corresponds to a corresponding option on the inpaint tab. Therefore, please refer to the inpaint tab for usage details on how to use each option.

## Class Filtering (fork feature)

When a multiclass YOLO detection model is selected (one that exposes more than one class, such as the [fdetailer](https://civitai.com/models/1228695) model with classes `face / penis / pussy / anus / sheath / pawpads`, or any custom YOLOv8 segmentation model), the UI auto-populates a multi-select dropdown labelled **ADetailer detector classes** with the class names the model was trained on.

- **Include mode** (default): pick one or more classes. Only detections of those classes will be inpainted.
- **Exclude / NOT mode**: tick the **Exclude selected (NOT)** checkbox. Detections of the selected classes will be skipped; everything else gets inpainted.
- **Empty selection**: behaves exactly like upstream — every class is inpainted.

For YOLO-World models the original text-based interface is preserved (open-vocabulary class names are not known up-front). For MediaPipe models the dropdown stays hidden (those models are not class-based).

#### Custom class names via sidecar JSON

If your `.pt` model does not embed class names in `model.names`, or you want to override them, drop a `.json` file with the same basename next to the `.pt` in `models/adetailer/`. The JSON may be a list, a `{"names": [...]}` object, or a `{"0": "face", "1": "hand", …}` map. Example:

```json
["face", "hand", "eye"]
```

Class names are cached per-session, so changing this file requires a webui restart to take effect.

#### Backwards compatibility

- The Pydantic schema gains optional fields (`ad_model_classes_exclude`, `ad_model_classes_excluded`, `ad_classes_sequential`) with safe defaults. Existing API clients and PNG infotext from earlier ADetailer runs continue to work.
- When the new fields are at their defaults, they are stripped from infotext output — workflows that don't use the feature produce byte-identical infotext to upstream.
- The existing `ad_model_classes` CSV semantic is preserved for YOLO-World models (`model.set_classes`); for multiclass YOLO it now drives the include filter via Ultralytics' native `model(classes=[ids])` argument.

## Sequential Class Detection

Tick **"Process classes sequentially"** (under the class dropdown) to make ADetailer run **one detection + inpaint pass per selected class**, in the order they appear in the dropdown.

Without this flag, picking `face, hand, eye` means ADetailer runs a single inference with `classes=[face, hand, eye]`, gets all detections in one batch, then inpaints them all using the same prompt and settings.

With sequential mode on, it instead runs three full passes:

1. **Pass 1**: detect only `face` → inpaint matched regions → updated image.
2. **Pass 2**: detect only `hand` on the result of pass 1 → inpaint → updated image.
3. **Pass 3**: detect only `eye` on the result of pass 2 → inpaint → final image.

The benefit: each pass operates on a cleaner input. The `hand` detector is not confused by half-inpainted face pixels; per-region padding is applied independently; large masks of one class don't shadow detections of another. Cost: longer total runtime (N passes instead of one).

Sequential mode is **ignored** for MediaPipe models, in NOT/exclude mode, and when fewer than 2 classes are selected — in those cases it has no effect and the regular single-pass flow runs.

## Copy Settings Between Tabs

Every tab — 1st through whatever you set as **Max tabs** in `Settings → ADetailer` (default 2, can be raised to 15+) — has a **"Copy settings"** + **"Paste settings"** button pair at the top.

The flow:

1. Click **Copy settings** in any tab. The clipboard captures that tab's processing settings (prompt, negative prompt, confidence, denoise, padding, sampler/checkpoint/VAE overrides, sampler, ControlNet, restore-faces, all the masking and inpainting knobs).
2. Every OTHER tab's **Paste settings** button enables itself and re-labels to **"Paste settings from Nth tab here"** so you can see at a glance what would be pasted.
3. Click **Paste settings** in any target tab to apply the stashed values. The source tab's own Paste button stays disabled (you can't paste a tab's settings back into itself).
4. The clipboard stays sticky — paste into multiple tabs in sequence, or do another Copy from a different tab to overwrite it.

The detector model and the class filter selection are **deliberately not part of the copy** so that each tab can target a different region or model while sharing all downstream processing. The "Enable this tab" checkbox of the destination tab is also left alone.

## ControlNet Inpainting

You can use the ControlNet extension if you have ControlNet installed and ControlNet models.

Support `inpaint, scribble, lineart, openpose, tile, depth` controlnet models. Once you choose a model, the preprocessor is set automatically. It works separately from the model set by the Controlnet extension.

If you select `Passthrough`, the controlnet settings you set outside of ADetailer will be used.

## Advanced Options

API request example: [wiki/REST-API](https://github.com/Bing-su/adetailer/wiki/REST-API)

`[SEP], [SKIP], [PROMPT]` tokens: [wiki/Advanced](https://github.com/Bing-su/adetailer/wiki/Advanced)

## Media

- 🎥 [どこよりも詳しい After Detailer (adetailer)の使い方 ① 【Stable Diffusion】](https://youtu.be/sF3POwPUWCE)
- 🎥 [どこよりも詳しい After Detailer (adetailer)の使い方 ② 【Stable Diffusion】](https://youtu.be/urNISRdbIEg)

- 📜 [ADetailer Installation and 5 Usage Methods](https://kindanai.com/en/manual-adetailer/)

## Model

| Model                 | Target                | mAP 50                        | mAP 50-95                     |
| --------------------- | --------------------- | ----------------------------- | ----------------------------- |
| face_yolov8n.pt       | 2D / realistic face   | 0.660                         | 0.366                         |
| face_yolov8s.pt       | 2D / realistic face   | 0.713                         | 0.404                         |
| hand_yolov8n.pt       | 2D / realistic hand   | 0.767                         | 0.505                         |
| person_yolov8n-seg.pt | 2D / realistic person | 0.782 (bbox)<br/>0.761 (mask) | 0.555 (bbox)<br/>0.460 (mask) |
| person_yolov8s-seg.pt | 2D / realistic person | 0.824 (bbox)<br/>0.809 (mask) | 0.605 (bbox)<br/>0.508 (mask) |
| mediapipe_face_full   | realistic face        | -                             | -                             |
| mediapipe_face_short  | realistic face        | -                             | -                             |
| mediapipe_face_mesh   | realistic face        | -                             | -                             |

The YOLO models can be found on huggingface [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer).

For a detailed description of the YOLO8 model, see: https://docs.ultralytics.com/models/yolov8/#overview

YOLO World model: https://docs.ultralytics.com/models/yolo-world/

### Additional Model

Put your [ultralytics](https://github.com/ultralytics/ultralytics) yolo model in `models/adetailer`. The model name should end with `.pt`.

It must be a bbox detection or segment model and use all label.

## How it works

ADetailer works in three simple steps.

1. Create an image.
2. Detect object with a detection model and create a mask image.
3. Inpaint using the image from 1 and the mask from 2.

## Development

ADetailer is developed and tested using the stable-diffusion 1.5 model, for the latest version of [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) repository only.

## License

ADetailer is a derivative work that uses two AGPL-licensed works (stable-diffusion-webui, ultralytics) and is therefore distributed under the AGPL license.

## Credits

This fork stands on the shoulders of two prior projects:

- **[Bing-su/adetailer](https://github.com/Bing-su/adetailer)** — the upstream extension. All of the original detection/inpainting pipeline, UI scaffolding, ControlNet integration, infotext handling, and the YOLO-World support are Bing-su's work. This fork is a thin addition on top of it.
- **[wkpark/uddetailer](https://github.com/wkpark/uddetailer)** — a sibling extension whose `scripts/detectors/ultralytics.py` provided the reference implementation for class filtering: passing `classes=[ids]` directly to Ultralytics' inference call for include filtering, and post-filtering predictions by class name for exclude. The dropdown-based UI is also conceptually borrowed from uddetailer's interface.

The class-filtering code in this fork was implemented by **[Claude](https://www.anthropic.com/claude)** (Anthropic's coding assistant) on behalf of the repository owner, who directed the project but does not write Python. The work is licensed under the same AGPL terms as upstream ADetailer.

## See Also

- https://github.com/ototadana/sd-face-editor
- https://github.com/continue-revolution/sd-webui-segment-anything
- https://github.com/portu-sim/sd-webui-bmab
