# ADetailer Ultimate

> **⚠️ This is an unofficial fork.** This project is **not affiliated with or endorsed by Bing-su**, the original ADetailer author. For the official ADetailer go to [Bing-su/adetailer](https://github.com/Bing-su/adetailer).
>
> **About this fork** — a soft-fork of [Bing-su/adetailer](https://github.com/Bing-su/adetailer) that adds workflow features on top of upstream ADetailer. Everything upstream still works the same way; the additions are opt-in widgets layered on top of the existing UI. See the [NEW IN THIS FORK](#new-in-this-fork) section below for the complete list of additions, each compared side-by-side with upstream behavior.
>
> The implementation was authored by **Claude** (Anthropic's coding assistant) at the request of the repository owner, who is not a Python developer. The class-filtering pattern is borrowed from [wkpark/uddetailer](https://github.com/wkpark/uddetailer); the preset library is conceptually inspired by uddetailer too. All credit for the original ADetailer goes to **Bing-su**; this fork extends that work — it does not replace it.
>
> This fork is distributed under the same AGPL-3.0 license as the upstream. See `LICENSE.md` for the full text; Bing-su's copyright notices are intact.
>
> If you only need single-class face/hand detection, use upstream ADetailer instead — the additional widgets in this fork are harmless but unnecessary for that case.

---

## NEW IN THIS FORK

Every row below is an addition this fork makes on top of upstream [Bing-su/adetailer](https://github.com/Bing-su/adetailer). The **Upstream** column describes how the official extension behaves today; the **This fork** column describes what the same workflow looks like here. Anything not listed below works identically to upstream.

> **Status legend** (first column of every table):
> - 🟢 **Tested** — implemented in the codebase and verified hands-on by the repo owner.
> - 🟡 **Under testing** — implemented and shipped, but the repo owner has not finished hands-on verification yet. Should work; issue reports are welcome.
> - 🔴 **Not yet implemented** — on the roadmap, no code in `main` yet. See the [Roadmap](#roadmap-not-yet-implemented) sub-section.

### Per-tab features — visible in **both** the txt2img and img2img ADetailer accordions

Features registered through `adui()` with mode-agnostic visibility — every widget here appears in **both** pipelines' ADetailer tab.

#### Detection

| Status | Feature | Upstream Bing-su | This fork |
| :---: | --- | --- | --- |
| 🟢 | Multiclass detector classes | Either everything the model produces is inpainted (no per-class control), or for YOLO-World a text field accepts open-vocabulary class names. There is no UI to choose among the trained classes of a multiclass YOLO detector. | Auto-populated multi-select dropdown reading class names from `model.names` (or a sidecar `.json`). Choose any subset; the include path uses Ultralytics' native `model(classes=[ids])` so non-matching detections are dropped at inference time. |
| 🟢 | Exclude / NOT mode | Not available. | "Exclude selected (NOT)" checkbox inverts the filter — every class the model produces *except* the selected ones gets inpainted. Implemented as a post-filter on `pred.boxes.cls`. |
| 🟢 | Sequential class detection | All selected classes run in a single inference batch; the inpaint passes all use the same prompt and settings. | Optional "Process classes sequentially" checkbox: runs one detect+inpaint pass per selected class in dropdown order, each pass operating on the output of the previous. Cleaner per-region inpainting at the cost of longer runtime. Top-of-function recursion in `_postprocess_image_inner` with single-class `args.copy(update=...)`. |
| 🟢 | Class pass order (activation order) | N/A (no class-selection UI). | The order in which you click classes in the dropdown is the order they're processed when **Sequential class detection** is on. Gradio's multi-select natively appends each selection to the end of the value list, and the sequential pipeline reads that order verbatim. To re-order, click the × on a token to deselect it, then click its name in the dropdown again — it goes to the end. |
| 🟢 | Detection preview | Not available — you run a full generation to see what the detector matches. | "Run detection preview" button in a per-tab accordion. Runs the configured detector against the most recent generation (or the img2img input) and returns the image annotated with bounding boxes / mask, without inpainting.<br>**Note:** select a model in the **ADetailer detector** dropdown of the same tab *first* — the preview uses the tab's currently-selected detector and the button is a no-op when no detector is chosen. |

#### Prompting

| Status | Feature | Upstream Bing-su | This fork |
| :---: | --- | --- | --- |
| 🟢 | `ad_prompt_append` / `ad_negative_prompt_append` | To add a few inpaint-only tokens you must duplicate the entire main prompt into the tab's `Prompt` field, or leave it blank and use the main one verbatim. | Two single-line "append" fields under the prompts. Their content is appended to the resolved inpaint prompt without forcing you to duplicate the main one. Empty by default, stripped from infotext so workflows that don't use them are byte-identical. |
| 🟢 | Include LoRAs from main prompt (+ optional trigger extraction) | LoRAs in the main txt2img/img2img prompt are not auto-applied to the inpaint pass. There is also no way to surface the *trigger words* embedded in a LoRA filename. | Two paired checkboxes: `Use LoRAs from main prompt` (parent, `ad_use_main_loras`) scrapes `<lora:name:weight>` tags out of the main prompt and merges them into the inpaint prompt — works whether the tab's `Prompt` field is empty or filled. `Append LoRA triggers from name` (child, `ad_use_lora_triggers`) parses the parenthesised substring inside a LoRA name — convention `<lora:name (trigger phrase):weight>` from [Anzhc/aadetailer-reforge](https://github.com/Anzhc/aadetailer-reforge) — and appends the trigger phrase to the prompt. The child only runs when the parent is on. Helpers: `_LORA_TAG_RE`, `_LORA_TRIGGER_RE`, `_extract_lora_tags`, `_extract_lora_triggers`, `_merge_lora_tags`, `_append_lora_triggers`.<br><br>**Requirement for trigger extraction**: the LoRA tag in your main prompt must literally contain `(trigger phrase)` — e.g. `<lora:Some_LoRA (cool style):1>`. Two ways to get this format: (a) type it manually, or (b) set Forge Neo's `Settings → Extra Networks → When adding to prompt, refer to Lora by` from the default `"Alias from file"` to **`"Filename"`**, then save a copy of your LoRA whose filename contains the parenthesised triggers — clicking the LoRA from the picker will then auto-insert the tag with parens. If you keep `"Alias from file"` and the alias has no parens, the picker-inserted tag won't trigger the extraction. |
| 🟢 | Class-specific prompts (sequential mode) | When sequential class detection is on, every class pass uses the tab's single `ad_prompt` and `ad_negative_prompt`. There's no way to give each class its own tailored prompt. | New textbox `ad_class_prompts` in the **Inpaint prompts** accordion. Syntax: one line per class, `classname: positive_prompt [\| negative_prompt]`. Classes without a matching line fall back to the tab's `ad_prompt` for that pass. Lines that don't match the syntax are silently ignored. Applies only when `ad_classes_sequential` is on and ≥2 classes are selected. Parser `_parse_class_prompts`. Inspired by [newtextdoc1111/adetailer](https://github.com/newtextdoc1111/adetailer). |

#### Inpainting

| Status | Feature | Upstream Bing-su | This fork |
| :---: | --- | --- | --- |
| 🟢 | Bounding-box mask (segmentation models) | When a segmentation YOLO model produces a per-pixel mask, ADetailer uses that mask verbatim — sometimes too tight against the subject, leaving little context for the inpaint to blend naturally. | Per-tab checkbox (`ad_use_bbox_mask`) in the **Mask preprocessing** accordion. Forces the rectangular bounding box as the inpaint mask even when the detector provided a precise per-pixel segmentation mask. No effect on bbox-only detectors (they were already using bboxes). Implemented as a single conditional in `ultralytics_predict`; the saved mask preview (`*-ad-preview*.png`) also reflects the bbox substitution. Inspired by [newtextdoc1111/adetailer](https://github.com/newtextdoc1111/adetailer). |
| 🟢 | Scale-based inpaint resolution | Inpaint canvas is either the source `p.width × p.height` or a fixed user-supplied `ad_inpaint_width/height`. There's no way to express "always make the canvas a multiple of the bbox size" — handy when bboxes vary in scale across the same image. | New per-tab `ad_use_resolution_scale` checkbox + `ad_resolution_scale` slider (range 0.5–8.0, default 1.5) in the **Inpainting** section. When the checkbox is on, the inpaint width/height is `bbox_size × scale`, rounded down to a multiple of 8 (SD UNet requirement) with a 64-pixel floor. Mutually exclusive with the existing fixed-width/height toggle — when both are set, fixed wins. Math in `fix_p2`. Inspired by [Anzhc/aadetailer-reforge](https://github.com/Anzhc/aadetailer-reforge). |

#### Tab utilities (clipboard, presets, persistence)

| Status | Feature | Upstream Bing-su | This fork |
| :---: | --- | --- | --- |
| 🟢 | Copy / Paste between tabs | Each tab is configured by hand; there is no built-in way to clone a tab's settings into another. | Clipboard model. Per-tab `Copy settings` button stashes the tab's processing settings into a shared state; every other tab's `Paste settings` button enables and re-labels to `Paste settings from Nth tab here`. The clipboard is sticky — paste into multiple tabs in sequence, or overwrite by Copying from a different tab. Detector / class filter / per-tab enable are deliberately excluded from the snapshot. |
| 🟢 | Named preset library | `Settings → ADetailer` lets you set drop-in defaults (one shared default). There is no per-tab named-preset facility. | Per-tab dropdown + Load / Save / Delete / Rename / Reset row. Each preset is a full widget-state snapshot. Stored in `<extension_root>/user_presets.json` with atomic writes; gitignored. A reserved `(none)` entry at the top of every dropdown lets you clear the active-preset label without touching widget values. Implemented in `adetailer/presets.py`. |
| 🟢 | Preset live preview (`[SEP]` / `[PROMPT]` aware) | Selecting a preset from the dropdown gives no hint of its contents until you commit by clicking Load. | A markdown preview block under the preset dropdown updates on every `preset_dropdown.change` event with the preset's detector, classes, prompts, sequential flag, and class-specific prompt summary. `[SEP]` and `[PROMPT]` tokens are wrapped in backticks and a footnote reminds the user they'll be expanded at generation time. Formatter `_format_preset_preview`. |
| 🟢 | Export / Import preset library JSON | The named preset library lives in `<extension_root>/user_presets.json` but is local to the install — no way to share configurations across machines or back them up. | New "Preset library export / import" accordion at the bottom of the preset area. **Export** generates a `gr.File` download of the entire library. **Import** accepts a JSON file, merges entries by name, with an "Overwrite existing on conflict" toggle for collision behaviour. Status line summarises added/replaced/skipped counts. New library functions `export_presets_json`, `import_presets_json` in `adetailer/presets.py`. Cross-tab dropdown refresh on import is local-to-current-tab; other tabs pick up new presets on next UI reload. |
| 🟢 | Persistent last-used settings (per-tab, per-mode) | Tab values revert to their static defaults at WebUI restart unless you manually re-apply a "set defaults" through Settings. | Every Generate stashes per-tab widget state to `<extension_root>/user_state.json` (atomic write). State is **scoped by pipeline** — `"txt2img:<tab>"` and `"img2img:<tab>"` keys are independent, so changing settings in one mode never affects the other. Restored as initial values at the next WebUI start. Gated by the `Remember last-used settings` toggle in Settings → ADetailer (default on). Implemented in `adetailer/persistence.py`. |

### Per-tab features — visible **only in the txt2img tab**

Widgets whose Gradio `visible=` flag depends on `not is_img2img`, so they don't appear in the img2img ADetailer accordion at all.

| Status | Feature | Upstream Bing-su | This fork |
| :---: | --- | --- | --- |
| 🟢 | Apply only on hires.fix | In txt2img without this toggle the user has no easy way to express "skip ADetailer unless I'm running a hires.fix pass" — useful when iterating fast on a base seed and only wanting the detailer to engage on final hires outputs. | **Txt2img-only** per-tab checkbox (`ad_apply_on_hires_only`, hidden in img2img). When on, ADetailer runs only if `enable_hr=True` (hires.fix is enabled); when hires.fix is off it's skipped entirely. Note: in Forge Neo, `postprocess_image` is called once per generation **after** the hires sampling pass has already completed (and `is_hr_pass` is reset to False before the callback fires), so the original A1111 "skip the lowres pre-hires call" semantics don't apply — there's no double-call to opt out of. The toggle is therefore a "gate on hires.fix" rather than a "pick which postprocess pass to run". A no-op in img2img regardless of how it got set (e.g. via a preset shared from txt2img). Decision in helper `_should_skip_for_hires_only(p, args)`. Inspired by [Anzhc/aadetailer-reforge](https://github.com/Anzhc/aadetailer-reforge). |

### Per-tab features — visible **only in the img2img tab**

| Status | Feature | Upstream Bing-su | This fork |
| :---: | --- | --- | --- |
| — | _No fork-added img2img-only widgets yet._ | The existing `Skip img2img` toggle is an **upstream** feature kept unchanged. | — |

### Global settings — visible in **Settings → ADetailer**

Options registered through `on_ui_settings()` in `scripts/!adetailer.py`. Each one is a single toggle / textbox / button on the **Settings → ADetailer** page of Forge Neo's Settings tab (gear icon).

| Status | Feature | Upstream Bing-su | This fork |
| :---: | --- | --- | --- |
| 🟢 | Manual mode | No equivalent — disabling ADetailer means unticking the accordion's enable, which clears the tab's enable checkboxes as a side effect. | `Settings → ADetailer → Manual mode` is a global toggle that short-circuits the `postprocess_image` hook while leaving every widget value intact. Useful when iterating on prompt / seed / sampler between runs without recomputing the ADetailer pass each time. Primary use case is **batch triage**: generate N raw candidates without burning ADetailer time on each, pick the keepers, then disable Manual mode and Generate again to run ADetailer only on the chosen images. |
| 🟢 | Save intermediate steps | Only the final image (post all ADetailer passes) is saved. | `Settings → ADetailer → Save intermediate steps` writes the after-each-tab images alongside the final result (`_adetailer_step1.png`, `_adetailer_step2.png`, …). Pairs naturally with sequential class detection. |
| 🟢 | Remember last-used settings (toggle) | N/A — no such option exists upstream. | `Settings → ADetailer → Remember last-used settings between restarts` (default on). Gates the per-tab persistence feature listed above: when off, tabs always start with the extension's static defaults; when on, the last `user_state.json` snapshot is restored at boot. |
| 🟢 | Reset ADetailer settings | There is no built-in way to roll back the entire `Settings → ADetailer` page to its factory defaults short of editing `config.json` manually or wiping it entirely. | New red `🔄 Reset ADetailer settings to defaults` button at the bottom of `Settings → ADetailer`. Walks the WebUI options registry and restores every entry registered under the `ADetailer` section to its declared default in the extension's source, then saves `config.json` and reloads the page. Gated by a JS `confirm()` prompt so a stray click can't wipe everything. Per-tab widget state (`user_state.json`) is left untouched — only Settings-page options are reset. Implementation: `_reset_adetailer_settings()` + `_make_reset_settings_button()` in `scripts/!adetailer.py`. |

### Forge Neo compatibility (no UI surface)

Fixes that don't add new widgets — they keep the extension's existing UI working under Forge Neo's slimmer module surface.

| Status | Feature | Upstream Bing-su | This fork |
| :---: | --- | --- | --- |
| 🟢 | `disable_safe_unpickle` patch | Calls `unittest.mock.patch.object(modules.shared.cmd_opts, "disable_safe_unpickle")`. Forge Neo's slimmer `cmd_opts` doesn't expose that attribute, so the patch raises `AttributeError` and ADetailer fails to load the model. | Uses `patch.object(..., create=True)` so the attribute is materialised when missing, and the call is a no-op on Forge Neo. Single-line change in `aaaaaa/helper.py`. |
| 🟢 | `cmd_opts.use_cpu` access at Script init | `get_ultralytics_device()` does a direct `"adetailer" in shared.cmd_opts.use_cpu` membership check. Forge Neo's slimmer `cmd_opts` doesn't expose `use_cpu` → `AttributeError` at Script class instantiation → the runtime hooks (UI accordion, postprocess_image) never register, even though the boot banner still prints "ADetailer initialized". | Wraps the access in `getattr(shared.cmd_opts, "use_cpu", None) or []` so the membership check no-ops on Forge Neo while continuing to work on stock A1111 / Forge classic. Single-line change in `scripts/!adetailer.py`. |
| 🟢 | Civitai_helper JSON sidecars | A `.json` next to a `.pt` is assumed to contain class names; civitai_helper-generated metadata files break the loader. | `_names_from_json` returns `[]` for shapes that don't look like class containers (lists / `{names: [...]}` / `{0: "...", 1: "..."}`); the loader falls back to `model.names` so unrelated `.json` sidecars are silently ignored. |

### UI polish

- 🟢 Section labels (`.ad-section-label`) rendered in bright white, small uppercase — easier to scan the widget groups.
- 🟢 Action buttons (Copy / Paste / preset Load / Save / Rename / Delete / Reset / detection-preview) get rounded 8px corners and `white-space: nowrap` so all heights stay uniform when labels wrap.
- 🟢 A version-badge overlay pinned to the top-right of the accordion header. Auto-hides when the accordion collapses.
- 🟢 Top of every tab: `Enable this tab` + `Copy settings` + `Paste settings` row as direct top-level widgets (no nested clipboard accordion).

### Localisation (10 languages)

- 🟡 Ships ten `localizations/*.json` files (`it_IT`, `es_ES`, `fr_FR`, `de_DE`, `zh_CN`, `ja_JP`, `pt_BR`, `ru_RU`, `ko_KR`, `pl_PL`) covering every UI label, accordion title, button, info hint, placeholder and tooltip exposed by the fork-added widgets. Forge merges them into `window.localization` automatically; pick a language via the WebUI's localization setting (or via the [Language Diffusion](https://github.com/xXIlRizzoXx/sd-webui-language-diffusion) extension's top-bar selector) and the ADetailer panel translates in place — no restart needed. Stable Diffusion technical vocabulary (ADetailer, LoRA, CFG, VAE, ControlNet, hires.fix, img2img, inpaint, bbox, YOLO, MediaPipe, CLIP, SDXL …) is **intentionally left in English** so workflows stay consistent with civitai pages, tutorials, and forum discussion. Native-speaker review of each locale is welcome — open an Issue or PR with corrections.

### Roadmap (not yet implemented)

Candidate items that are not in `main` yet. None is committed to a release date — listed here to make the project's direction transparent. Items are ordered roughly by **value-to-effort ratio** (cheap & impactful first). The **Inspiration** column credits the fork or workflow where the idea originated.

| Status | Feature | What it would do | Inspiration |
| :---: | --- | --- | --- |
| 🔴 | ControlNet crop-aware toggle | Pass the cropped inpaint region through ControlNet's preprocessor instead of the full image, so ControlNet sees what the inpaint actually targets (instead of the whole scene). **Deferred**: requires a working ControlNet install for non-trivial testing, and the ControlNet API surface changes frequently between versions — high risk for a fragile integration. | IOSakaki workflow community |
| 🔴 | WDv3 autotagging | Run a WD14/WDv3-large image tagger over the cropped region *before* the inpaint and prepend the resulting tags to the inpaint prompt. **Deferred (heaviest item)**: ~200 MB model download on first use, cache management, dedicated Settings panel, VRAM impact, error handling for the tagger library. Estimated 1–2 day implementation. | [Anzhc/aadetailer-reforge](https://github.com/Anzhc/aadetailer-reforge) |
| 🔴 | Upstream PRs to `Bing-su/adetailer` | Open the two prepared branches (`fix/forge-neo-cmdopts-compat`, `feat/per-class-filtering`) as PRs against upstream. Both are ready and standing by — pending the repo owner's go-ahead. | (process) |

---

ADetailer is an extension for the stable diffusion webui that does automatic masking and inpainting. It is similar to the Detection Detailer.

## Install

You can install it directly from the Extensions tab.

![image](https://i.imgur.com/qaXtoI6.png)

Or

(from Mikubill/sd-webui-controlnet)

1. Open "Extensions" tab.
2. Open "Install from URL" tab in the tab.
3. Enter `https://github.com/xXIlRizzoXx/adetailer-ultimate.git` to "URL for extension's git repository". (Replace with `https://github.com/Bing-su/adetailer.git` if you want the upstream version instead.)
4. Press "Install" button.
5. Wait 5 seconds, and you will see the message "Installed into stable-diffusion-webui\extensions\adetailer. Use Installed tab to restart".
6. Go to "Installed" tab, click "Check for updates", and then click "Apply and restart UI". (The next time you can also use this method to update extensions.)
7. Completely restart A1111/Forge/Forge Neo webui including your terminal. (If you do not know what is a "terminal", you can reboot your computer: turn your computer off and turn it on again.)

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

When a multiclass YOLO detection or segmentation model is selected — that is, any `.pt` whose `model.names` exposes more than one class — the UI auto-populates a multi-select dropdown labelled **ADetailer detector classes** with the class names the model was trained on. Typical examples are community-published YOLO checkpoints trained to spot multiple facial features at once, or any custom multi-target YOLOv8 model you bring yourself.

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

Tick **"Process classes sequentially"** (under the class dropdown) to make ADetailer run **one detection + inpaint pass per selected class**, in the order they appear in the dropdown — which is the order in which you clicked them (see [Class Pass Order](#class-pass-order)).

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

## Preset Library

Each tab gets a dropdown + Load / Save / Delete / Rename row. A preset is a named snapshot of **every widget value in that tab** — detector, classes, sequential flag, prompts (including the append fields), denoise, padding, sampler, ControlNet, restore-faces, mask preprocessing, the lot.

- **Save** — type a name into the textbox (printable, no path separators or quotes, up to 80 chars) and hit **Save preset**. The dropdown in *every* tab refreshes immediately so you don't have to reload the UI.
- **Load** — pick a name from the dropdown and hit **Load preset**. All widgets in the current tab snap to the preset's values, including the class dropdown for the detector model in use.
- **Rename** — pick a preset, type the new name into the textbox, hit **Rename preset**. The on-disk file is updated atomically.
- **Delete** — pick a preset, hit **Delete preset**. The selection clears.
- **Reset** — hit **Reset preset** to clear the dropdown label to `(none)`. This is a label-only operation: widget values are not touched. Useful when you want to break the association with the currently displayed preset without changing what's on screen.

Presets live in `<extension_root>/user_presets.json` and are also git-ignored. Corrupted JSON or permission errors are swallowed — the worst case is presets stop appearing in the dropdown, never a broken generation.

A reserved `(none)` entry sits at the top of every dropdown so you always have an explicit way to clear the selection.

## Persistent Last-Used Settings

When `Settings → ADetailer → Remember last used settings` is on (default), every Generate click stashes each tab's widget state to `<extension_root>/user_state.json`. The file is rewritten atomically (`.tmp` + `os.replace`), so a crash mid-write can't corrupt it.

On the next WebUI start the values are restored as the tab's initial state. Like presets, errors are silent — if the file goes missing or unreadable, tabs come up at their static defaults.

This is separate from named presets — persistence is the "where did I leave off" recovery; presets are deliberate saved configurations.

## Prompt Append Fields

Below the main `Prompt` and `Negative prompt` textboxes there are two single-line fields:

- **Prompt append** — content appended to whatever ADetailer resolves the inpaint prompt to (either the per-tab prompt or, if blank, the main txt2img/img2img prompt).
- **Negative prompt append** — same idea for the negative.

The point is to add inpaint-only tokens (`ultra sharp eyes`, `detailed iris`, `blurry, lowres` in the negative) without having to duplicate the full main prompt into the tab's prompt field just to tack a few words onto the end.

## Include LoRAs From Main Prompt

A checkbox under the prompt fields. When enabled and the tab's `Prompt` field is **empty**, ADetailer scans the main txt2img/img2img prompt for `<lora:name:weight>` tags and merges them into the inpaint prompt. This means you can leave the inpaint prompt blank and still inherit the style LoRAs from the outer generation.

If the tab's `Prompt` field is non-empty, the main prompt is not consulted — your explicit prompt wins. To still inherit LoRAs in that case, put the LoRA tags in `Prompt append`.

## Detection Preview

Tucked inside an accordion at the bottom of each tab: a **Run detection preview** button. When clicked it loads the most recent generation's image (or the img2img input when in img2img), runs the configured detector against it, and renders the resulting bounding boxes / mask without doing any inpainting.

**Before clicking the button, pick a model in the tab's `ADetailer detector` dropdown.** The preview reuses the tab's currently-selected detector — there is no separate detector picker inside the preview accordion. If you haven't chosen one yet (the dropdown is still on the `None` placeholder), the button has nothing to run against.

Useful for tuning confidence threshold + mask preprocessing without burning a full generation each time.

## Manual Mode

`Settings → ADetailer → Manual mode` is a global toggle. When on, ADetailer's `postprocess_image` short-circuits — no detection, no inpainting — even if the accordion is enabled and tabs are configured. All widget values stay intact. Flip it back off and the next generation runs ADetailer as usual.

The use case is iteration on prompt / sampler / seed without recomputing the ADetailer pass between every txt2img run.

## Save Intermediate Steps

`Settings → ADetailer → Save intermediate steps`, off by default. When on, each tab's inpaint output is saved as a separate file (`_adetailer_step1.png`, `_adetailer_step2.png`, …) alongside the final result. Each step image is the cumulative output of all tabs up to and including that one.

This pairs naturally with sequential class detection — you can inspect what each class pass produced individually.

## Reset Settings

At the bottom of `Settings → ADetailer` there is a red `🔄 Reset ADetailer settings to defaults` button. Clicking it walks the WebUI options registry and restores every entry registered under the `ADetailer` section (every toggle visible on this page — max tabs, save paths, manual mode, remember-last, etc.) to the default value the extension declared in its source. The change is written to `config.json` and the page reloads automatically so every widget re-reads its now-default value.

A `confirm()` prompt gates the action — clicking Cancel does nothing. Per-tab widget state stored in `user_state.json` is **not** touched; only the global Settings options are reset. If you also want to clear per-tab cached values, toggle `Remember last-used settings` off, save once, and toggle it back on.

## Class Pass Order

The class dropdown's `value` list is the source of truth for **Sequential class detection**'s pass order. Gradio's multi-select natively keeps the values in **selection order** — the first class you click is at the front, each subsequent click appends to the end. So:

- **First-time selection** — just click the classes in the order you want them processed. Done.
- **Re-ordering an existing selection** — click the × on a token to remove it from the list, then click its name in the dropdown again. It re-appends at the end. Repeat for any other tokens you want to move.
- **Reset the order to "selection order"** — click × on everything, then click each name fresh in the order you want.

There is no drag-and-drop. An earlier version of the fork shipped an HTML5 drag-and-drop handler, but it relied on a brittle deselect-then-reselect sync against Gradio's reactive store and caused tokens to flicker out during the operation. The native activation-order behaviour is simpler, fully reliable, and produces the same end result.

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
