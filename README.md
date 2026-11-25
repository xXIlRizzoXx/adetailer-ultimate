# ADetailer Ultimate

<div align="center">
  <img src="docs/img/cover-beta.jpg" alt="What's new in the beta v26.2.0+plus.10.beta.3, the third beta of plus.10, compared with beta 2: Run ADetailer on an image is right for Flux (issue #9), shown with two real crops of the same face from Forge Neo 2.33, the Flux picture and the run of this beta with Neo's FLUX preset (20 steps, CFG scale 1, Euler), redrawn cleanly; and the new status that says the values the pass used, for Flux (denoise 0.6, 20 steps, CFG scale 1, Euler (Beta)) and for an SDXL model picked right after it (denoise 0.6, 24 steps, CFG scale 4.5, Euler a); also in this beta: on Forge Neo the img2img preset of the model family (SDXL, Flux, Z-Image Turbo, Qwen-Image, Lumina, Wan, Anima and others), on classic Forge 20 steps, CFG 1, Euler and Simple for Flux, the right values right after a model change, and the tab's own Use separate options still come first" width="100%" />

  <br /><br />

  <img src="docs/img/cover-plus9.jpg" alt="What's new in plus.9, the stable release v26.2.0+plus.9 compared with v26.2.0+plus.8: click the numbered boxes of the Detection preview to choose which detections to inpaint, the [PROMPT=words] token that brings only some parts of the main prompt, a refresh button next to the detector menu that reads the detector folders again without a restart, and X/Y/Z axes for every tab; also tooltips on every option, more sort orders, a max inpaint size, an iris class, color matching, region opacity, an upscaler for the inpaint area, dropping duplicate detections, saving inpaint masks, detection previews in the results gallery and more; plus many fixes, among them ten known issues of plus.8. Presets and settings are kept" width="100%" />

  <br /><br />

  <img src="docs/img/cover.jpg" alt="ADetailer Ultimate: finds faces, hands and other parts in your pictures and redraws them with more detail, for AUTOMATIC1111, Forge, Forge Neo and reForge" width="100%" />

  <br /><br />

  [![Stable release](https://img.shields.io/github/v/release/xXIlRizzoXx/adetailer-ultimate?label=stable)](https://github.com/xXIlRizzoXx/adetailer-ultimate/releases/latest) [![Beta pre-release](https://img.shields.io/github/v/release/xXIlRizzoXx/adetailer-ultimate?include_prereleases&sort=date&label=beta)](https://github.com/xXIlRizzoXx/adetailer-ultimate/releases) [![WebUI: AUTOMATIC1111, Forge, Forge Neo, reForge](https://img.shields.io/badge/WebUI-AUTOMATIC1111%20%7C%20Forge%20%7C%20Forge%20Neo%20%7C%20reForge-blue)](#compatibility) [![License](https://img.shields.io/github/license/xXIlRizzoXx/adetailer-ultimate)](LICENSE.md)

</div>

> **Unofficial fork of [Bing-su/adetailer](https://github.com/Bing-su/adetailer), the extension that finds faces, hands and other parts in your pictures and redraws them with more detail. Everything ADetailer does, plus class filters, presets, remembered settings and more.**
>
> For AUTOMATIC1111, Forge, Forge Neo and reForge (tested live on AUTOMATIC1111 and Forge Neo). Stable [v26.2.0+plus.9](https://github.com/xXIlRizzoXx/adetailer-ultimate/releases/latest) · Beta [v26.2.0+plus.10.beta.3](https://github.com/xXIlRizzoXx/adetailer-ultimate/releases/tag/v26.2.0%2Bplus.10.beta.3) in testing (pre-release: [what is new](#what-is-new-in-plus10), [switch to the beta and back](#switch-to-the-beta-and-back)).
>
> This project is **not affiliated with or endorsed by Bing-su**, the author of the original ADetailer. For the official ADetailer, go to [Bing-su/adetailer](https://github.com/Bing-su/adetailer). All credit for the original ADetailer goes to Bing-su: this fork extends that work, it does not replace it. Licensed under AGPL-3.0, the same license as the original ([LICENSE.md](LICENSE.md)), with the original author's credits kept. Who made what: [Credits](#credits).

**New here?** [Install](#install), then the three-step [Quick start](#quick-start). **Already using the original ADetailer?** Read [Coming from the original ADetailer?](#coming-from-the-original-adetailer) first.

🟡 **The third beta of plus.10 is out** as a pre-release (v26.2.0+plus.10.beta.3). Beta 3 fixes "Run ADetailer on an image" with Flux, which ran with CFG scale 7 and 28 steps and burnt the faces: on Forge Neo it now takes the img2img values of Forge Neo's preset for the model, for every model family, and its status says the denoise, steps, CFG scale and sampler it used. Everything of beta 2 is still there. Beta 2 added an option in Settings → ADetailer that makes "Run ADetailer on an image" and its folder batch use the prompt saved in each picture, so you can detail many pictures, each with its own prompt, in one click; and the YOLO detectors now stay loaded for the whole generation, instead of being read from the disk again for every image, tab and X/Y/Z cell. A new guide in the panel, "Guide: several subjects, one LoRA each", gives each person in a picture their own LoRA: pick a LoRA and a few words for each person, press one button, and it fills the tab with the detector, the `[SEP]` prompt and the settings that keep the parts in order. It also lets YOLOE detectors take the words you type, like YOLO-World; adds an option to detail with the hires fix checkpoint after hires fix, and one to keep the transparency of a cut-out picture in "Run ADetailer on an image" and with Skip img2img; on Forge and Forge Neo, skips a tab whose checkpoint cannot be loaded with the VAE file of its pass, when its files show it for sure, instead of losing the whole generation, and warns when a tab's checkpoint of another model family keeps the main model's VAE or text encoder; saves in the image parameters the prompts a wildcard drew in a tab and the hashes of a tab's detector and checkpoint; says in the console when a detailer prompt names the same LoRA twice, when the model picked is not a detector and when a Python package is missing; applies the selected Styles once also when another extension changed the main prompt; and gives each section of a tab a coloured bar and an emoji. Everything of beta 1 is still there: each tab gets a "Masked content" option, as in the WebUI's own inpaint, to redraw a region from scratch with "latent noise" or "latent nothing"; a wildcard in a tab's prompt that the main prompt also holds takes the value drawn there, so the eye colour stays the same; the Detection preview writes each box's share of the image, the value to set on the "Mask min/max area ratio" sliders; the console says what ADetailer is installing or downloading, so a long first start no longer looks stuck; and detectors with the same file name in different folders are all listed. Dynamic Prompts no longer runs on the regions whose prompt holds no wildcard or variant, so its Magic Prompt no longer rewrites them. It also fixes Soft Inpainting, where ADetailer could turn the main image black or change it outside the mask, makes ADetailer's ControlNet on Forge, Forge Neo and reForge work on each region as on AUTOMATIC1111, and shows the warning when MediaPipe cannot be installed. The stable release is still v26.2.0+plus.9. → [What is new in plus.10](#what-is-new-in-plus10)

🟢 **plus.9 is out** as the stable release v26.2.0+plus.9. You can click the boxes of the Detection preview to choose which detections to inpaint, use X/Y/Z axes for every tab and an Enable axis, bring only the matching parts of the main prompt into the ADetailer prompt with a `[PROMPT=eyes]` token, so a wildcard's eye colour carries over to the detailer pass, pick an `iris` class in the face parts detector, and click a 🔄 button next to the detector menu, so a detector you add or remove while the WebUI runs shows up without a restart. Every option has a tooltip, two tabs can be swapped or one pasted into all the others, and a "Show labels" switch hides the names and numbers on the Detection preview. New settings match the colors of each inpainted region to the picture, blend it with the original, pick the upscaler of the inpaint area, cap the inpaint size, drop a face found twice, sort the detections in more ways, save the inpaint masks, show the detection previews in the results gallery or keep the boxes out of the live preview, and start ADetailer already switched on. Built-in detectors already on your disk load without contacting Hugging Face, the detector runs on the CPU when the whole WebUI does, one failed region no longer stops the others, the LoRAs and embeddings of the ADetailer prompts are listed for sites such as Civitai, ADetailer's detections no longer send Ultralytics' usage analytics, and the Guide tab answers more frequent questions. It also fixes ten known issues of plus.8, among them Export on AUTOMATIC1111, ControlNet preprocessors, prompt comments, Skip img2img, pasted samplers and "Overlay original for inpaint". → [What is new in plus.9](#what-is-new-in-plus9)

## ✨ What it does

<img src="docs/img/01-panel.jpg" alt="One panel: every detailer pass in its own tab, with a counter of the enabled tabs and settings remembered between restarts" width="100%" />

<img src="docs/img/02-classes.jpg" alt="Class filter: detail only the classes you pick, or all but some, and one pass per class in your order" width="100%" />

<img src="docs/img/03-presets.jpg" alt="Presets and tabs: named presets, copy and paste between tabs, reset one tab or every tab" width="100%" />

<img src="docs/img/04-prompts.jpg" alt="Prompts: a prompt for each class with [CLASS] blocks, Auto class-guard and LoRA controls" width="100%" />

<img src="docs/img/05-detection.jpg" alt="Detection: detection resolution per tab and inpaint only the detections you choose by number" width="100%" />

<img src="docs/img/06-combine.jpg" alt="Combine all tabs: the Detection preview runs the detectors of every tab at once, one colour per tab" width="100%" />

<img src="docs/img/10-click-preview.jpg" alt="Plus.9, Detection preview: click the numbered boxes to choose which detections to inpaint" width="100%" />

<img src="docs/img/07-run-on-image.jpg" alt="Run ADetailer on an image: detail a finished picture or a whole folder without generating it again" width="100%" />

<img src="docs/img/08-inpainting.jpg" alt="Inpainting: dynamic denoise per tab, inpaint size scaled to the box, separate checkpoint, VAE and (on Forge) text encoder" width="100%" />

<img src="docs/img/09-settings.jpg" alt="Settings: manual mode, verbose diagnostic log and a reset of the ADetailer settings" width="100%" />

<img src="docs/img/11-xyz.jpg" alt="Plus.9, X/Y/Z plot: every tab gets its own axes, and an Enable axis compares pictures with and without ADetailer" width="100%" />

<img src="docs/img/12-prompt-token.jpg" alt="Plus.9, prompts: [PROMPT=eyes] brings only the parts of the main prompt that name eyes into the detailer prompt" width="100%" />

<img src="docs/img/13-iris.jpg" alt="Plus.9, face parts detector: an iris class with one mask per eye, cut to the eye opening" width="100%" />

<img src="docs/img/14-show-labels.jpg" alt="Plus.9, Detection preview: untick Show labels to see only the boxes and masks, without names and scores" width="100%" />

<img src="docs/img/15-gallery-previews.jpg" alt="Plus.9, results gallery: the picture with the detection boxes of every pass after your results, not saved to disk" width="100%" />

<img src="docs/img/16-settings-beta2.jpg" alt="Plus.9, Settings: four new options, all off by default: gallery previews, hiding the boxes from the live preview, more sort orders and a max inpaint size" width="100%" />

<sub>Screenshots taken on AUTOMATIC1111 1.10. "Exclusive" marks what neither the original ADetailer nor any other fork we know of offers.</sub>

## Contents

1. [Main features](#-main-features)
2. [More features](#more-features)
3. [Requested by users](#-requested-by-users)
4. [What is new in plus.10](#what-is-new-in-plus10) (beta in testing): [new in beta 2](#new-in-plus10-beta-2), [new in beta 1](#new-in-plus10-beta-1), [notes on beta 2](#notes-on-plus10-beta-2) and [beta 1](#notes-on-plus10-beta-1)
5. [What is new in plus.9](#what-is-new-in-plus9): [new in beta 3](#new-in-plus9-beta-3), [new in beta 2](#new-in-plus9-beta-2), [new in beta 1](#new-in-plus9-beta-1), [notes on beta 3](#notes-on-plus9-beta-3), [beta 2](#notes-on-plus9-beta-2) and [beta 1](#notes-on-plus9-beta-1), [known issues](#known-issues)
6. [What is new in plus.8](#what-is-new-in-plus8): [new and fixed](#new-and-fixed), [fixed in beta 1](#fixed-in-beta-1), [known issues of plus.8](#known-issues-of-plus8)
7. [Install](#install): [coming from the original ADetailer?](#coming-from-the-original-adetailer), [update](#update), [switch to the beta and back](#switch-to-the-beta-and-back)
8. [Quick start](#quick-start)
9. [Using each feature](#using-each-feature)
10. [Settings](#settings)
11. [Upstream options](#upstream-options)
12. [Models](#models)
13. [Problems and fixes](#problems-and-fixes)
14. [FAQ](#faq)
15. [Compatibility](#compatibility)
16. [Roadmap](#roadmap)
17. [For developers](#for-developers)
18. [License](#license), [Credits](#credits) and [See also](#see-also)

## ⭐ Main features

🟢 available in the stable release · 🟡 in testing in a beta (plus.10 beta 3). What the beta in testing adds is in [What is new in plus.10](#what-is-new-in-plus10), what the latest release changed in [What is new in plus.9](#what-is-new-in-plus9), and what plus.8 changed in [What is new in plus.8](#what-is-new-in-plus8).

1. 🟢 **Class filter** — detail only the classes you pick from a multi-class detector, for example only face and hand, or all but some with "Exclude selected (NOT)". [How to use it](#class-filter)
2. 🟢 **Run ADetailer on a finished image or a whole folder** — detail pictures you already have with the tab's own settings, without generating them again; originals are never overwritten. [How to use it](#run-adetailer-on-an-image)
3. 🟢 **Detection preview, then pick which detections to inpaint** — see numbered boxes of what the detector finds, without generating anything, then type the numbers you want, such as `1,3`. From plus.8 it also names the class next to each number. From plus.9, you can also click the boxes instead of typing. [How to use it](#detection-preview)
4. 🟢 **Named presets** — save a tab's whole setup under a name, load it in any tab, and export or import the whole library. From plus.8 it also says what a preset from another WebUI needs that this one does not have. [How to use it](#preset-library)
5. 🟢 **Your tabs come back as you last generated with them** — each tab's settings return at the next WebUI start. From plus.8 it also reopens the sections you left open. [How to use it](#remember-last-used-settings)
6. 🟢 **Copy, paste and reset tabs** — copy a tab's setup into any other tab of the same panel, or put one tab or every tab back to its defaults. [How to use it](#copy-settings-between-tabs)
7. 🟢 **A prompt per class** — `[CLASS=hand] five fingers [/CLASS]` in the tab prompt gives the hands their own text, and `[CLASS=hand] [SKIP] [/CLASS]` skips only the hands. [How to use it](#inline-class-blocks-and-skip)
8. 🟢 **Face parts detector** — `mediapipe_face_features` finds eyes, mouth, nose, eyebrows and face separately, so you can detail only the eyes or only the mouth. [How to use it](#face-parts-detector-mediapipe_face_features)
9. 🟢 **Separate checkpoint, VAE and text encoder for the detailer pass** — on Forge and Forge Neo the detailer can also use its own text encoder and VAE, so it can run a checkpoint from another model family than your base model. From plus.8 they are always switched back after the pass. [How to use it](#separate-checkpoint-vae-text-encoder-and-clip-skip)
10. 🟢 **Runs on Forge Neo**, where the original ADetailer stops with errors, and installs on AUTOMATIC1111 without breaking it. Tested live on AUTOMATIC1111 1.10 and Forge Neo; classic Forge and reForge are checked in the code. [How to install](#install)

### Original vs Ultimate

| What you want to do | Original ADetailer | ADetailer Ultimate |
| --- | --- | --- |
| Detail only some classes, for example only the hands of a face-and-hand detector | Every class the detector finds is detailed; only YOLO-World detectors let you type what to look for | Pick the classes in a list, or all but some with "Exclude selected (NOT)" |
| Detail a picture you already have | Send it to img2img and tick "Skip img2img" | "Run ADetailer on an image" in any tab, in txt2img too, or a whole folder in one click |
| Save a setup and use it again | No presets; PNG Info brings back the settings of an image made with ADetailer | Named presets for every tab, with import and export as a file |
| Find your settings after a restart | Every tab starts from its defaults (the WebUI's Settings → Defaults can change them) | Every tab comes back as you last generated with it |
| Use it on Forge Neo | Stops with errors; Forge Neo recommends the separate ADetailer-Neo port instead | Runs there, and is tested live on it |

Everything else the original ADetailer does is still here, some of it with the fixes listed below; its options are listed in [Upstream options](#upstream-options). Moving over from the original? See [Coming from the original ADetailer?](#coming-from-the-original-adetailer).

## More features

Everything else this fork adds, grouped by area, in a short entry each. Smaller fixes are in [CHANGELOG.md](CHANGELOG.md).

🟢 available in the stable release · 🟡 in testing in a beta (plus.10 beta 3)

The 🟡 entries are only in the plus.10 betas, v26.2.0+plus.10.beta.3 and, for the ones beta 2 and beta 1 added, v26.2.0+plus.10.beta.2 and v26.2.0+plus.10.beta.1: [What is new in plus.10](#what-is-new-in-plus10) lists them. Some 🟢 entries came with plus.9 and are not in v26.2.0+plus.8, or came with plus.8 and are not in v26.3.0+plus.7.4: [What is new in plus.9](#what-is-new-in-plus9) and [What is new in plus.8](#what-is-new-in-plus8) list them.

### Detection and classes

- 🟢 **Detection resolution** — the "Detection resolution (0 = default)" slider in each tab runs a YOLO detector at a higher resolution than the one it was trained at (for example 1024 for the built-in detectors, which use 640), so it finds small or distant faces and hands. It uses more VRAM and time. 0 keeps each detector's own size. MediaPipe detectors ignore it, and the "Combine all tabs" preview stays at each detector's own size.
- 🟢 **Process classes sequentially** — runs a separate detailing pass for each selected class, in the order you clicked them, each on the result of the previous one. It needs two or more selected classes and does not run in "Exclude selected (NOT)" mode. MediaPipe detectors other than `mediapipe_face_features` ignore it. With the "Merge and Invert" mask mode, plus.8 runs one pass for all the selected classes instead, so a class is no longer repainted with another class's prompt.
- 🟢 **Detection numbers typed your way** — "Inpaint only these detections" reads `1,3,5` and `1-3`, and also `#2`, numbers separated by spaces, and the commas and dashes a Chinese or Japanese keyboard types (`1，3`, `1、3`, `1～3`). Leave it blank for all. It is not used while "Process classes sequentially" runs its passes.
- 🟢 **Class picks apply at once** — the classes you pick are used even if you press Generate straight away or another job is still running.
- 🟢 **Class names in any case** — a class name such as "Face" from a preset, pasted parameters or an API request matches the detector's "face", in the class filter and in per-class prompt lines.
- 🟢 **Your own class names** — a `<model>.names.json` file next to a detector gives it class names or replaces them. Files saved by Windows PowerShell or with Notepad's "Unicode" option are read too, and unrelated metadata files next to your models are ignored.
- 🟢 **YOLO-World works with class blocks** — regions found by a YOLO-World detector carry the class names typed in its CLASSES field, so `[CLASS=person]` blocks apply to them. "Exclude selected (NOT)", which has no effect with YOLO-World, is hidden for it.
- 🟢 **Accurate masks at the edges** — a mask moved with the "Mask x(→) offset" or "Mask y(↑) offset" slider stops at the image border instead of wrapping round to the other side, and segmentation masks line up on non-square images.
- 🟢 **Class numbers kept** — when PNG Info, Send to, Paste settings or loading a preset brings a class filter written as class numbers (such as `1`, as an API request may send it) and also changes the detector, the class filter keeps the number instead of going empty and detailing every class.
- 🟢 **More ways to type detection numbers** — "Inpaint only these detections" also reads the Japanese long-vowel mark and middle dot (`1ー3`, `1・3`) and a doubled dash (`1——3`, `1--3`).
- 🟢 **Better boxes for faces cut by the edge** — with `mediapipe_face_short` or `mediapipe_face_full`, a face cut off by the left or top edge gets a box that fits the visible face.
- 🟢 **Odd digits in a class filter** — a class written with a character such as `²` or `①` (from the API, a preset or pasted parameters) is ignored and named in the console, like any name the detector does not have, instead of stopping ADetailer for the image.
- 🟢 **Click the boxes to choose detections** — after a single-tab Detection preview, a click on a numbered box adds its number to "Inpaint only these detections" and a second click takes it out; the chosen boxes are marked in green, also when you type the numbers. See [Detection preview](#detection-preview).
- 🟢 **Iris class** — `mediapipe_face_features` has an `iris` class, used only when you pick it: one mask per eye over the whole iris, cut to the opening of the eye, to change the eye colour without redrawing the eyelids. See [Face parts detector](#face-parts-detector-mediapipe_face_features).
- 🟢 **Refresh the detector menu** — the 🔄 next to the "ADetailer detector" menu reads the detector folders again, so a detector you add or remove while the WebUI runs shows up in, or leaves, the menu without a restart: [Add your own detectors](#add-your-own-detectors).
- 🟡 **Detectors with the same file name all listed** — two detectors with the same file name in different folders, as Civitai often saves them, are both in the "ADetailer detector" menu: the first keeps its plain name and the other is listed as "name.pt (folder)", and the console names both files by model folder and path inside it, not by the full path (it adds parent folders only when two model folders end with the same name). The folders are read in a fixed order, so the same file keeps the plain name on every start and every WebUI. See [Notes on plus.10 beta 1](#notes-on-plus10-beta-1) for a change on Forge Neo.
- 🟡 **YOLOE detectors like YOLO-World** — a YOLOE detector with text prompts (a file name with `yoloe`, such as `yoloe-11s-seg.pt`) shows the text field of YOLO-World: type what to find, separated by commas. The prompt-free ones (`-pf`) keep their own class list. The first detection with typed classes downloads a text model of about 600 MB. See [YOLO-World detectors](#yolo-world-detectors).
- 🟡 **Models that are not detectors explained** — an OBB, classification or semantic segmentation model picked as the detector finds nothing to inpaint, and the console, the Detection preview and "Run ADetailer on an image" say why in one line, instead of an error with a traceback. Pose models still work. See [Add your own detectors](#add-your-own-detectors).
- 🟡 **Missing `clip` package named** — when a YOLO-World or YOLOE detector cannot take the typed classes because the Ultralytics library could not install its `clip` package, the tab inpaints nothing and the console gives the pip command that installs it, instead of an error that stops ADetailer for the image. See [YOLO-World detectors](#yolo-world-detectors).

### Prompts

- 🟢 **LoRA control for the detailer pass** — "Use LoRAs from main prompt" brings the main prompt's LoRAs into the detailer prompt. "Strip LoRAs from the detailer prompt" removes them all, so a style LoRA does not bleed onto faces and hands.
- 🟢 **Prompt append fields** — two one-line fields add a few words to the end of the detailer prompt and negative prompt, so you don't have to copy the whole main prompt.
- 🟢 **Auto class-guard** — adds each region's own class to its prompt and the detector's other classes to its negative prompt, so a detected face is not redrawn as a hand, for example. Off by default; the "Class-guard emphasis" slider sets how strongly the class name is weighted. It works with detectors that have class names (YOLO models other than YOLO-World, and `mediapipe_face_features`). In sequential mode a per-class prompt line takes its place.
- 🟢 **Append LoRA triggers from name** — for a LoRA whose name holds a trigger phrase in round brackets (a file named `my_style (cool style)`, used as `<lora:my_style (cool style):1>` in the main prompt), the phrase is added to the detailer prompt. It needs "Use LoRAs from main prompt" turned on.
- 🟢 **No doubled LoRAs** — "Use LoRAs from main prompt" leaves out a LoRA that the tab prompt already names, so the tab's own weight is used.
- 🟢 **Styles applied once** — with Styles selected and a blank or `[PROMPT]` ADetailer prompt, the detailer pass gets the styles once instead of twice, also when the main prompt holds comments (in the stable release, not yet when another extension, such as Dynamic Prompts with wildcards, rewrites the main prompts; 🟡 from plus.10 beta 2 also then, except in three rare cases: see [Notes on plus.10 beta 2](#notes-on-plus10-beta-2)). From the stable release of plus.9, this also holds for a Style that adds its text on the line of a comment or that holds a comment itself. With "Use LoRAs from main prompt" on, a LoRA that a selected Style adds is used once too, and with "Strip LoRAs from the detailer prompt" on, the styles' LoRAs are removed.
- 🟢 **Only some parts of the main prompt** — `[PROMPT=eyes,hair]` in a tab's prompt puts only the parts of this image's main prompt that name one of these words, so the eye colour a wildcard picked reaches the detailer pass without the rest of the main prompt; `[PROMPT=-beach]` puts everything except the parts that name it. See [Upstream options](#upstream-options).
- 🟢 **Spaces count as blank** — an ADetailer prompt that holds only spaces or line breaks now uses the main prompt instead of an empty one.
- 🟢 **Comments left out of the ADetailer prompts** — comments typed in a tab's prompt, negative prompt and append boxes (`#`, and on Forge Neo also `//` and `/* */`) are removed before the detailer pass, as the WebUI removes them from the main prompt, also with "Apply only selected scripts to ADetailer" on. A box that holds only a comment counts as blank. On Forge this works from the stable release: in the three betas a comment there also removed what ADetailer adds after it on the same line, and a box with only a comment gave an empty prompt. From the stable release, on AUTOMATIC1111 and Forge Neo, a comment written in a selected Style stays out of the detailer pass too. Unlike the main prompt's, the comments of the boxes stay in the image parameters ("ADetailer prompt" and the others), so that pasting brings the boxes back as you typed them: take private notes out before you share a picture. In plus.8 they reach the model, except on Forge: see [Known issues of plus.8](#known-issues-of-plus8).
- 🟢 **Per-class prompts with class numbers** — in sequential mode, a class filter given as numbers (for example from an API request) uses the per-class prompt lines written with the class names, and the Auto class-guard stays on when none of the selected class names exists in the detector. A line written for a class number, such as `0: …`, takes the place of the Auto class-guard in its pass, as a line written with the class name does.
- 🟡 **Wildcards keep the main prompt's value** — a `__name__` wildcard (Dynamic Prompts or the "wildcards" extension) in a tab's prompt, negative prompt or "Prompt append" boxes that the main prompt also holds takes the value drawn there for each image, instead of a new draw, so `close-up, __eyecolor__ eyes` gives the same eye colour as the main prompt. A wildcard that is not in the main prompt is drawn as before. Always on, no option. See [Can the ADetailer prompt keep what a wildcard picked in the main prompt?](#can-the-adetailer-prompt-keep-what-a-wildcard-picked-in-the-main-prompt).
- 🟡 **Dynamic Prompts only where a region uses it** — the Dynamic Prompts script no longer runs on a region whose final prompt holds no wildcard or variant, so its "Magic prompt", "I'm feeling lucky" and "Attention grabber" no longer rewrite that region's prompt. A region whose prompt holds a wildcard or a variant runs it, as before; a wildcard that took its value from the main prompt does not count. This changes the default behaviour and has no option. See [Changed in plus.10 beta 1](#changed-in-plus10-beta-1).
- 🟡 **Repeated LoRA note** — when the final prompt of a detailer pass names the same LoRA twice, for example with `[PROMPT]` plus a LoRA that the main prompt already has, or with one more in the append text or in a selected Style, the console says so once per generation, tab and LoRA. The prompt is not changed. See [LoRAs in the detailer prompt](#loras-in-the-detailer-prompt).
- 🟡 **A guide for several subjects, one LoRA each** — the closed box "Guide: several subjects, one LoRA each" above the tabs gives each person in a picture their own LoRA without typing `[SEP]`: pick what to redraw (the whole person inside their outline, the face, or both on two tabs), a LoRA, a weight and a few words for up to four people, and press "✨ Fill the tabs". It fills the tab with the detector, the prompt, the settings that keep the parts in order and a denoising strength of 0.6, and its status says what to check. The subjects are the biggest people in the picture, counted from left to right. Tested live in part on AUTOMATIC1111 and Forge Neo. See [Several subjects, one LoRA each (guide)](#several-subjects-one-lora-each-guide).

### Inpainting

- 🟢 **Apply only on hires.fix** — runs the tab only when hires.fix is on, on the upscaled image; with hires.fix off the tab is skipped (txt2img only).
- 🟢 **Scale inpaint to bbox** — sets each region's inpaint size from the size of its detection box (box size × a factor, 1.5 by default), so the size follows the region. At that factor a small region is redrawn smaller than it is in the picture: see [Scale inpaint to bbox](#scale-inpaint-to-bbox). Only with "Inpaint only masked" on; "Use separate width/height" overrides it.
- 🟢 **Dynamic denoise by area, per tab** — smaller regions get a higher denoising strength. 0 uses the value in Settings → ADetailer, which is off unless you set it there.
- 🟢 **Use bbox as mask** — segmentation models inpaint the whole detection box instead of the tight outline, which gives the region more room to blend in.
- 🟢 **Separate VAE on Forge and Forge Neo** — "Use separate VAE" changes the VAE there too, where it had no effect in the original, and "Automatic" uses the detailer checkpoint's own VAE. In plus.7.4 and beta 1, on Forge Neo, pick a module once in the VAE / Text Encoder selector at the top of the page first, or the detailer pass fails.
- 🟢 **Distilled CFG Scale / Shift kept** — the detailer pass uses your value instead of always 3.5, so Flux guidance, or the shift of models like Z-Image, matches the rest of the image (Forge and Forge Neo only). When a separate checkpoint is picked it keeps 3.5, because a checkpoint of another model family reads this value differently.
- 🟢 **Image size kept when the whole picture is inpainted** — with "Inpaint only masked" off, a hires.fix image is no longer shrunk back to its first-pass size, and with Skip img2img the image keeps its own size (rounded down to a multiple of 8) instead of taking the size of the width and height sliders.
- 🟢 **Skip img2img uses your real settings** — "Use same sampler" uses your sampler, and the saved image records your real steps, sampler and size instead of placeholder values (img2img only).
- 🟢 **Merge and Invert safeguard** — with the "Merge and Invert" mask mode, when the detections cover the whole picture there is nothing left to inpaint, so the pass is skipped instead of repainting the whole picture.
- 🟢 **"Inpaint only masked" always pastes back** — with the WebUI's "Overlay original for inpaint" setting (on Forge Neo: "For inpainting, overlay the resulting image back onto the original image") off, each region is still pasted back into the picture, instead of the region's crop replacing the whole picture. From plus.9, this no longer turns that setting on during the pass, so a settings save during the pass no longer stores it as on and frozen settings no longer stop ADetailer; in plus.8 these two limits remain: see [Known issues of plus.8](#known-issues-of-plus8).
- 🟢 **Transparent input images** — with Skip img2img, the transparent parts of an input image get the WebUI's img2img background colour instead of turning black.
- 🟢 **Separate steps and sampler always used** — with "Apply only selected scripts to ADetailer" turned off in Settings → ADetailer (it is on by default), or with `sampler` added to "Script names to apply to ADetailer", "Use separate steps" and "Use separate sampler" now apply, also through the API, instead of being replaced by the main ones.
- 🟢 **Masks drawn on a transparent layer** — in img2img inpainting, a mask that holds the painted area in its transparency (from the Inpaint upload tab, an inpaint Batch mask folder or the API) is read as the WebUI reads it, also with Soft inpainting on, so ADetailer no longer switches itself off or details faces outside the painted area.
- 🟢 **ControlNet models in any letter case** — a ControlNet model whose file name writes its type with capitals, such as `OpenPoseXL2`, gets its preprocessor like the others.
- 🟢 **Larger mask sliders** — for large pictures, "Mask x(→) offset" and "Mask y(↑) offset" go from -512 to 512, "Mask erosion (-) / dilation (+)" from -256 to 256, "Inpaint mask blur" up to 256 and "Inpaint only masked padding, pixels" up to 512. The defaults are unchanged.
- 🟢 **ControlNet models start with their preprocessor on Forge** — on Forge, Forge Neo and reForge, choosing a ControlNet model selects its default preprocessor, such as `openpose_full` or `depth_midas`, as AUTOMATIC1111 does, instead of "None". From the stable release, so does the X/Y/Z "[ADetailer] ControlNet model 1st" axis on a tab whose ControlNet model is "None" or "Passthrough". In plus.8 it sets "None": see [Known issues of plus.8](#known-issues-of-plus8).
- 🟡 **Masked content** — each tab chooses what fills the region before it is redrawn, as in the WebUI's own inpaint: original (the default, as before), fill, latent noise or latent nothing. The last two redraw the region from scratch and need a denoising strength of about 0.8 or more. See [Masked content](#masked-content).
- 🟡 **Soft Inpainting keeps the main image** — in img2img inpainting with Soft inpainting on, ADetailer no longer turns the main image into a mostly black picture the size of the crop ("Only masked", also with Ultimate SD Upscale) or changes it outside the mask ("Whole picture"), and a batch of 2 or more no longer prints an error from `soft_inpainting.py` ([Bing-su/adetailer#694](https://github.com/Bing-su/adetailer/issues/694)).
- 🟡 **Detail with the hires fix checkpoint** — Settings → ADetailer → "With hires fix, detail with the hires checkpoint" (off by default): after hires fix with its own Hires checkpoint, the tabs without their own checkpoint detail the upscaled picture with that checkpoint, and on Forge and Forge Neo with the hires VAE / text encoder choices and the Hires Distilled CFG Scale. See [Separate checkpoint, VAE, text encoder and CLIP skip](#separate-checkpoint-vae-text-encoder-and-clip-skip).
- 🟡 **Warning for a checkpoint of another model family** — on Forge and Forge Neo, when a tab's own checkpoint is of another model family than the main model (one of them SD 1.5, SD 2 or SDXL) and its pass kept the main model's VAE or text encoder files, the console says so once per generation and tab (in a folder batch, once for the whole folder), and so does the status of a single "Run ADetailer on an image" (the summary of a folder batch does not). The warning blocks nothing.
- 🟡 **A tab whose checkpoint cannot load is skipped** — on Forge and Forge Neo, before a tab's pass ADetailer reads the header of the `.safetensors` files and skips the tab when its checkpoint cannot be loaded with the VAE file of the pass, for example an SDXL checkpoint in the tab with the VAE file of a Flux main model, instead of losing the whole generation. The other tabs run, the image is saved, and the console and the text under the results say what to set in the tab. In any doubt nothing is skipped. Tested live on Forge Neo in two cases; classic Forge and the other cases are covered by the offline tests only. See [Separate checkpoint, VAE, text encoder and CLIP skip](#separate-checkpoint-vae-text-encoder-and-clip-skip).

### Working on existing images

- 🟢 **Save result to outputs** — saves the result of "Run ADetailer on an image" to an ADetailer-Inpaint folder next to your txt2img and img2img outputs, or inside the WebUI's "Output directory for images" (on Forge Neo: "Output Directory") when you have set one.
- 🟢 **Save results in the source folder** — a folder run saves each result next to its original as a new file ending in -ad (`photo.png` → `photo-ad.png`, or `photo-ad-1.png` when that name is taken). Originals are never overwritten, and earlier results are skipped when you run the folder again.
- 🟢 **A new result on every run** — each run of "Run ADetailer on an image" picks new random seeds for all regions, so pressing it again gives a different result.
- 🟢 **Combine all tabs** — the Detection preview runs every configured tab's detector on the image at once, one colour per tab.
- 🟢 **Rotated photos and .tif files** — rotated phone photos are detected upright, and folder runs include .tif files as well as .tiff.
- 🟢 **Tools wait their turn** — the Detection preview and "Run ADetailer on an image" wait for a running generation to finish, Interrupt stops a folder run before the next file, and a new run works after you cancel one. The WebUI hides its Interrupt and Skip buttons during these runs: press Alt+Enter to stop one (see [Run ADetailer on an image](#run-adetailer-on-an-image)).
- 🟢 **Preview on the CPU when ADetailer runs there** — with `--use-cpu adetailer`, or (except on macOS) `--lowvram`, `--medvram` or `--medvram-sdxl`, the Detection preview runs YOLO detectors on the CPU, like the generation does. Of these flags, Forge Neo has only `--lowvram`. From plus.9, both also use the CPU with `--use-cpu all`, `--cpu`, `--always-cpu` and, except on macOS, Forge Neo's `--novram`: see [Compatibility and reliability](#compatibility-and-reliability).
- 🟢 **Clearer folder runs** — the status counts files skipped as earlier results, and Settings → Defaults no longer saves a folder path as a startup default (which made a single-image run detail the whole folder).
- 🟢 **Combine all tabs explains failures** — the combined preview says why a tab's detector failed.
- 🟢 **Show labels** — a checkbox next to "🔍 Run detection preview" hides the class names, scores and numbers drawn on the preview, leaving only the boxes and masks. See [Detection preview](#detection-preview).
- 🟢 **GIFs in the Detection preview** — on Forge and Forge Neo, the MediaPipe detectors now find faces in a GIF dropped into the Detection preview.
- 🟢 **Failed regions counted** — when some regions fail with a NaN error (the WebUI's "A tensor with all NaNs was produced" error, which leaves the region unchanged) and others are detailed, "Run ADetailer on an image" says how many were left unchanged, instead of only reporting the pass as complete (AUTOMATIC1111, classic Forge and reForge).
- 🟢 **Class names on the preview numbers** — the numbers of the single-tab Detection preview also name the class, such as `#1 eyes`, so they no longer hide the part names of `mediapipe_face_features`. When the preview's font cannot draw a class name, as can happen with Chinese or Japanese names on some systems, the number stays plain, and the combined preview leaves the name out of its label instead of failing. On other systems, such as Windows, the font has no Chinese or Japanese letters, so such a class name shows as empty boxes after the number; the number itself is always readable.
- 🟡 **Share of the image on the preview labels** — each label of the Detection preview ends with the share of the image its box takes, such as `#1 face 12.3%`: the value that "Mask min area ratio" and "Mask max area ratio" compare (12.3% is 0.123). See [Detection preview](#detection-preview).
- 🟢 **Transparency and colours kept** — "Run ADetailer on an image" and folder runs fill the transparent parts of a picture, such as a cut-out PNG, with the WebUI's img2img background colour instead of black. A copy saved next to its source keeps the source's colour profile, and a lossless WebP gives a lossless copy.
- 🟢 **Saved results keep their parameters** — results saved to the ADetailer-Inpaint folder record the detailer pass's settings, which PNG Info and Send to read, instead of the word "None". From plus.8 they also record the tab's ADetailer settings, and the PNG, JPEG and WebP copies a folder run saves next to their sources record the same parameters.
- 🟢 **Extra files keep their parameters** — the mask previews, step images and masks that "Run ADetailer on an image" and folder runs save record their parameters, and the console no longer prints "create_infotext failed" for each one. A `[user]` or `[denoising]` token in the WebUI's file or folder name pattern no longer prints a traceback for these files and for the results saved to `ADetailer-Inpaint`.
- 🟡 **The prompt saved in each picture** — Settings → ADetailer → "Run ADetailer on an image uses the prompt saved in the image" (off by default) gives each picture of "Run ADetailer on an image" and its folder batch the main prompt and negative prompt the WebUI saved in it, so a blank tab prompt or `[PROMPT]` uses each picture's own prompt. See [Many pictures, each with its own prompt](#many-pictures-each-with-its-own-prompt).
- 🟡 **Transparency kept on the result** — Settings → ADetailer → "Run ADetailer on an image and Skip img2img keep the transparency of the picture" (off by default): a cut-out PNG is still detailed on the img2img background colour, then the result gets the picture's transparency back, also in a folder batch, and in img2img with "Skip img2img" ticked (that part is tested live on AUTOMATIC1111 only). The saved file keeps it only in a file format that has it (PNG; WebP on Forge Neo); with JPEG, or WebP on AUTOMATIC1111, the saved file has none, and the status (with Skip img2img, the text under the results) and the console say so. See [Run ADetailer on an image](#run-adetailer-on-an-image).

### Presets and tabs

- 🟢 **Complete paste from PNG Info** — PNG Info and Send to txt2img/img2img restore the ADetailer setup more completely than the original: they switch ADetailer and the tabs the image used on, switch the other tabs off (for an image made with ADetailer), and put the settings the image left at their defaults back to those defaults instead of keeping the tab's old values. Fields listed in "Disregard fields from pasted infotext" (on Forge Neo: "Ignore fields when reading infotext") are left alone. From plus.8, a tab whose detector is not installed here keeps its own detector and is switched off, instead of making Generate fail, and an old sampler name such as "DPM++ 2M SDE Karras" is split into the sampler and its scheduler. From plus.9, "Skip img2img" is restored too, for images made with plus.9 or later, and a separate sampler this WebUI does not have becomes the first sampler with a console message; in plus.8, "Skip img2img" is not restored.
- 🟢 **ControlNet preprocessor kept** — loading a preset, pasting a tab or pasting parameters keeps a ControlNet preprocessor that fits the model instead of resetting it, and a restored ControlNet model shows its preprocessor after a restart.
- 🟢 **Only the first tab starts on** — on a fresh setup, "Enable this tab" starts unticked in tabs 2 and later, so the tab counter reads 1x Tab. When you pick a detector in another tab, tick its "Enable this tab" too.
- 🟢 **Safe preset and settings files** — saves from different tabs no longer overwrite each other, and a damaged file is set aside instead of being overwritten.
- 🟢 **Older presets load cleanly** — settings an older preset doesn't include go back to their defaults instead of keeping the tab's current values ("Enable this tab" is left as it is).
- 🟢 **Full preset list after a page reload** — reloading the browser page lists every preset saved, renamed or imported since the WebUI started.
- 🟢 **Presets saved as "Unicode" load** — preset and settings files saved in UTF-16 (for example with Notepad's "Unicode" option) are read instead of being set aside as damaged.
- 🟢 **Sampler after Reset** — Reset, "Reset every tab" and loading a preset without a sampler put "ADetailer sampler" back on the first sampler, as on a fresh setup. In plus.7.4 and beta 1, if you use "Use separate sampler", pick your sampler again after a Reset.
- 🟢 **Presets saved right after a Reset load cleanly** — a preset saved just after a Reset in an earlier release shows "Use same …" in the separate checkpoint, VAE and text encoder lists, instead of empty lists, and the first sampler or, with "Use separate sampler" ticked, the sampler and scheduler that the WebUI ran it with.
- 🟢 **Presets from another WebUI** — when a preset names a detector that is not installed here, Load keeps the tab's detector and class filter and loads the rest. A separate checkpoint, VAE, text encoder, scheduler or ControlNet model that this WebUI does not have is set to "Use same …" or "None", and, with "Use separate sampler" ticked, a missing sampler becomes the first sampler. The status says what was missing, so Generate no longer fails or quietly uses something else.
- 🟢 **Damaged preset files refused cleanly** — Import reports a damaged file as not imported, instead of failing without a message or emptying the preset lists.
- 🟢 **Export downloads on AUTOMATIC1111 too** — on AUTOMATIC1111 and other WebUIs with Gradio 3, such as reForge's main branch, Export now downloads `adetailer-ultimate-presets.json`, as on Forge and Forge Neo. In plus.8 it cannot download a file there: see [Known issues of plus.8](#known-issues-of-plus8).
- 🟢 **Missing samplers named after a restart or a paste** — with "Use separate sampler" ticked, a sampler this WebUI does not have becomes the first sampler, after a restart and after pasting parameters, and the console says so; old names that end with a scheduler are split as Load splits them.

### Settings and output files

- 🟢 **Manual mode** — Settings → ADetailer → "Manual mode — don't auto-run ADetailer after generation" stops the automatic pass but keeps every tab setting, so you can generate drafts quickly and later detail the ones you keep with "Run ADetailer on an image".
- 🟢 **Save intermediate step images** — an option in Settings → ADetailer that saves the image after each tab's pass, and after each class pass in sequential mode (from plus.8 also with Skip img2img).
- 🟢 **Tidy output folder** — mask previews, before-images and step images go into an adetailer-steps sub-folder, so the main folder holds only final images.
- 🟢 **Extra images keep their own seed and prompt** — in a batch, the before, mask-preview and step images each record their own seed and prompt, and in sequential mode each class pass saves its own mask preview, named after the class.
- 🟢 **Reset ADetailer settings to defaults** — a button at the bottom of Settings → ADetailer resets every option on that page after you confirm. Your tabs and presets are not touched.
- 🟢 **Verbose diagnostic log** — an option in Settings → ADetailer that prints each active tab's settings, what was detected, and the time and VRAM each pass used to the console. Useful for bug reports.
- 🟢 **Manual mode changes wait for the next generation** — ticking Manual mode while an image is being made now applies from the next generation, instead of saving Skip img2img images at 128x128 or leaving other scripts such as ControlNet stopped.
- 🟢 **Verbose log complete in a log file** — when the WebUI's console output goes to a log file or another program on Windows, the Verbose diagnostic log keeps all its lines instead of losing the ones with characters the file cannot show.
- 🟢 **Quieter manual mode again** — as in plus.7.4, the console says that ADetailer was skipped only for a generation it would otherwise have detailed; beta 1 also said it with ADetailer switched off.
- 🟢 **Save inpaint masks** — an option in Settings → ADetailer that saves the black-and-white mask each region is inpainted with, as a PNG in the `adetailer-steps` sub-folder. See [Save inpaint masks](#save-inpaint-masks).
- 🟢 **Enable ADetailer at startup** — an option in Settings → ADetailer (Off, txt2img, img2img or Both) that starts the ADetailer panel already switched on in the chosen tabs, after a UI reload. See [Settings](#settings).
- 🟢 **Match the colors of each inpainted region** — an option in Settings → ADetailer (off by default) that matches the colors of each inpainted region to the picture, so a face no longer comes out with another tint. See [Settings](#settings).
- 🟢 **More sort orders** — an option in Settings → ADetailer that details the regions from right to left, from top to bottom, from small to large or by confidence, instead of the order of "Sort bounding boxes by"; it also decides which face gets which part of a `[SEP]` prompt. See [Settings](#settings).
- 🟢 **Max inpaint size** — an option in Settings → ADetailer (0, no limit, by default) that inpaints a region smaller, with the same shape, when its inpaint size would be larger, so "Scale inpaint to bbox" on a large detection does not run out of video memory. See [Scale inpaint to bbox](#scale-inpaint-to-bbox).
- 🟢 **Detection previews in the results gallery** — an option in Settings → ADetailer (off by default) that adds the picture with the detection boxes of every pass after the generated images, with the parameters of its image, without saving it to disk. See [Settings](#settings).

### Interface

- 🟢 **ADetailer Guide tab** — an optional top tab with a plain-language guide to the main options and to common problems (in English); while it is on, a "📖 Guide" link in the ADetailer header opens it. Turn it on in Settings → ADetailer → "Show the ADetailer Guide tab (top tab bar)", then reload the UI.
- 🟢 **Translated interface** — with the companion [Language Diffusion](https://github.com/xXIlRizzoXx/sd-webui-language-diffusion) extension, the fork's labels, buttons and hints can be shown in 10 languages. Options added in recent versions still show in English until the translations are updated.
- 🟢 **Active tabs counter** — a green pill next to the ADetailer title (for example "2x Tabs") shows how many tabs are enabled, even with the panel closed.
- 🟢 **Version badge** — the version label sits on the ADetailer header and also shows the installed build, so you can tell at a glance which ADetailer Ultimate you have.
- 🟢 **Tooltips on every option** — hover over an option of a tab, such as "Mask min area ratio" or "Use separate checkpoint", for about a second to see what it does. The options with a help line under them keep it.
- 🟢 **Interface polish** — tooltips on the fork's buttons, status messages that fade after a few seconds, same-size buttons and readable text on the light theme.
- 🟢 **Open sections are remembered** — the sections you leave open in each tab (Detection, Inpainting, Detection preview and the others) open again after a reload or restart. This is stored in your browser, so another browser starts with them closed ([issue #6](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues/6)).
- 🟢 **Version badge fits a narrow panel** — in a narrow window the version badge no longer covers the ADetailer title and the tab counter; it shortens its text instead.
- 🟢 **Messages and counter keep up on Forge** — on Forge and Forge Neo, a repeated preset or import message shows again also after a very fast answer, and the tab counter follows Paste settings, Load and Reset.
- 🟢 **Translated values stay translated** — on Forge and Forge Neo, with Language Diffusion, the "Use same …" values of a tab's dropdowns stay translated after you change them.
- 🟢 **Guide corrections** — the Guide tab and the "Detection resolution" help text say what 0 means (each detector's own size), what Merge does, where the preview tools are and that tabs 2 and later need "Enable this tab".
- 🟢 **Detection boxes out of the live preview** — an option in Settings → ADetailer (off by default) keeps the picture with the detection boxes out of the WebUI's live preview and the progress API. See [Settings](#settings).
- 🟢 **More answers in the Guide tab** — the order of ADetailer and a face-swap extension, why the detail pass runs fewer steps, model file safety, a switch for "Save images before ADetailer" at the top of the page, and how to leave small or background faces alone. They are in [Problems and fixes](#problems-and-fixes) and the [FAQ](#faq) too.
- 🟡 **Coloured sections** — each section of a tab has a coloured bar and an emoji before its title: blue for the settings (Inpaint prompts, Detection, Mask Preprocessing, Inpainting), amber for the tools (Detection preview, Run ADetailer on an image), grey for the preset library file. The detector and ControlNet controls sit under their own small labels, "Detector" and "ControlNet". Nothing moved and the titles are unchanged, so translations and saved settings keep working. See [Interface details](#interface-details).

### Compatibility and reliability

- 🟢 **MediaPipe detectors on newer builds** — the MediaPipe face detectors find faces on the MediaPipe 0.10.x builds that no longer have its old interface, such as the Python 3.13 builds Forge Neo uses, where they used to find nothing. Their two small model files download by themselves the first time (this needs an internet connection).
- 🟢 **MediaPipe with non-English folder names** — on Windows, the MediaPipe detectors work when the WebUI's folder path has an accented, Cyrillic, Chinese, Japanese or Korean character, for example in your user name.
- 🟢 **MediaPipe safety cap** — installs stay on the tested MediaPipe 0.10.x line, so a fresh install cannot pull MediaPipe 1.0.1, which crashes the whole WebUI on macOS.
- 🟢 **Interrupt and Skip behave predictably** — Interrupt or Skip during a tab's pass throws that whole pass away (the regions it already finished too) instead of keeping a half-finished region. Skip stops ADetailer for the rest of the current batch.
- 🟢 **A failed tab no longer breaks the batch** — when a tab stops with an error (for example a missing detector or out of memory), other scripts such as ControlNet on Forge Neo keep working for the rest of the batch.
- 🟢 **Your changes during a pass are kept** — if you change a setting while ADetailer is working, such as the checkpoint, it is no longer undone when the pass ends, unless the tab uses its own value for that same setting (for example a separate checkpoint). From plus.8 this holds for Clip skip and the VAE also when they have never been saved with Settings → Apply settings (in beta 1 only once they had been saved); plus.7.4 undoes every setting changed during a pass.
- 🟢 **Correct image data in X/Y/Z grids and img2img Batch** — the "[ADetailer] Prompt S/R" axes no longer print an error for every cell, each cell's images save only that cell's settings, and one empty mask no longer switches ADetailer off for the rest of a batch.
- 🟢 **API support for the new options** — every new per-tab option can be sent through the WebUI API. Most of them are written into the image parameters only when you change them; "Apply only on hires.fix", "Use bbox as mask", "Scale inpaint to bbox" (with its factor) and "Append LoRA triggers from name" are always written.
- 🟢 **No startup error when another ADetailer is also installed** — having the original ADetailer installed too no longer prints a startup error. Keep only one ADetailer anyway: with two, the panel appears twice (see [Coming from the original ADetailer?](#coming-from-the-original-adetailer)).
- 🟢 **No "outdated" message on Forge Neo** — Forge Neo's startup message that suggests ADetailer-Neo instead is no longer printed while this fork is installed (Forge Neo only).
- 🟢 **API requests no longer use your remembered UI settings for the tabs they leave out** — with "Remember last-used settings" on, a tab that an API request leaves out now gets a fresh setup's settings, as in the original ADetailer (only the 1st tab on, with the first detector), instead of the settings remembered in the UI.
- 🟢 **Separate CLIP skip, VAE and text encoder always switched back** — after every pass, your main Clip skip, VAE and (on Forge and Forge Neo) text encoder come back, also on a fresh install and after "Run ADetailer on an image", so a tab's own choice no longer reaches the next tabs or later images. On Forge Neo, "Use separate VAE" also works before the VAE / Text Encoder selector at the top of the page has ever been used.
- 🟢 **MediaPipe problems reported** — if MediaPipe is missing or broken, the console says so once, instead of every MediaPipe detector quietly finding nothing.
- 🟢 **Quieter start on AUTOMATIC1111** — a detector without a `<model>.names.json` file no longer fills the console with a long error report at startup. From plus.9, its CLASSES list shows the classes straight away, read from the detector file without loading it; in plus.8 it shows them once a generation or a Detection preview has found something with it and you switch to another detector and back, or straight away with a `<model>.names.json` file. Classic Forge (checked in its code) and Forge Neo list the classes straight away.
- 🟢 **Names and prompts in other alphabets no longer stop ADetailer** — on Windows, when the WebUI's console output is saved to a log file, a detector, class, folder or file name, or a prompt, with characters of another alphabet (for example Chinese or Cyrillic) or an emoji no longer makes ADetailer fail or keeps its panel from appearing.
- 🟢 **Prompt S/R images paste back correctly** — images from an X/Y/Z "[ADetailer] Prompt S/R" cell paste back the same ADetailer prompts in every tab, the per-class prompt lines of a sequential tab included, and each image records its own prompt, also with wildcards. The "(AD 1st and main prompt)" axis no longer replaces twice in the main prompt that a blank or `[PROMPT]` ADetailer prompt stands for.
- 🟢 **Dynamic Prompts templates kept** — with Dynamic Prompts' "Save template to metadata" on, the saved image parameters keep the wildcard template instead of the first image's resolved prompt.
- 🟢 **Detailer ControlNet on classic Forge** — the ControlNet model of an ADetailer tab now works on classic Forge, instead of doing nothing since that WebUI's Gradio 4 update (checked against its source code, not tested live).
- 🟢 **Detailer ControlNet in every pass** — on Forge and Forge Neo, the ControlNet model of an ADetailer tab works for every image, tab and X/Y/Z cell, instead of only in the first pass, with the next ones inpainted without ControlNet and an error in the console (tested live on Forge Neo).
- 🟢 **X/Y/Z detector axis on an empty tab** — an "[ADetailer] ADetailer model 1st" axis also runs a cell whose 1st tab is left on "None" in the panel.
- 🟢 **Starts with `--no-gradio-queue`** — AUTOMATIC1111 and other WebUIs with Gradio 3 start when launched with `--no-gradio-queue`, instead of stopping with a queue error.
- 🟢 **ControlNet preview shown once** — on Forge and Forge Neo, with ControlNet on, its preview appears once in the results instead of once more for every batch.
- 🟢 **X/Y/Z axes for every tab** — the 2nd and later tabs get X/Y/Z axes for their model, prompts, confidence, erosion / dilation, mask blur, denoising strength, padding, CFG scale and steps; the 1st tab gets confidence, mask blur and steps axes too; and "[ADetailer] Enable" switches ADetailer on or off for each cell. A CFG scale or Steps axis also ticks the tab's "Use separate CFG scale" or "Use separate steps" in each cell, so the cells differ, and from the stable release the sampler, scheduler and noise multiplier axes of the 1st tab tick "Use separate sampler" or "Use separate noise multiplier". See [New in plus.9 beta 1](#new-in-plus9-beta-1).
- 🟢 **LoRAs and embeddings of the ADetailer prompts in the image parameters** — a LoRA or an embedding used only in an ADetailer prompt is listed in "Lora hashes" and "TI hashes" with those of the main prompt, so sites such as Civitai recognise it (a WebUI without "TI hashes", such as Forge Neo, lists only the LoRAs). Idea from the [diffus-me/adetailer](https://github.com/diffus-me/adetailer) fork.
- 🟢 **Detectors made with a newer Ultralytics explained** — a detector that the installed Ultralytics cannot load because it is too old, such as a recent YOLO12 or YOLO26 model, is named in the console and in the Detection preview status, with the installed version, instead of only "Can't get attribute '…'".
- 🟢 **`.zip` files in the detector folders pointed out** — at startup the console names up to 5 `.zip` files found in the detector folders and asks you to extract the `.pt` file, since ADetailer reads only `.pt` files.
- 🟢 **MediaPipe leaves your audio alone** — with the `sounddevice` package installed, MediaPipe no longer starts the audio system, which on Windows could stop the sound of other programs and list Python as using the microphone ([Bing-su/adetailer#508](https://github.com/Bing-su/adetailer/issues/508)).
- 🟢 **reForge's Advanced Model Sampling kept** — on reForge, the Advanced Model Sampling scripts always run in the detailer pass, so v-prediction and flow models are detailed correctly also with "Apply only selected scripts to ADetailer" on (checked in reForge's code). With a ControlNet model in the tab, on Forge, Forge Neo and reForge, the detailer pass runs only ControlNet, so these scripts do not run in that tab's pass (see [Known issues](#known-issues)).
- 🟢 **No Ultralytics usage analytics** — ADetailer switches off Ultralytics' anonymous usage analytics before its detections, in memory only: the Ultralytics settings file, which other programs share, is never written.
- 🟢 **MediaPipe pre-releases read correctly** — the macOS safety check no longer takes a MediaPipe pre-release such as 1.0.0rc1 for 1.0.1.
- 🟢 **Built-in detectors from your disk first** — a built-in detector in `models/adetailer` or an extra model folder is used from there, and one downloaded before loads from the Hugging Face cache, without contacting Hugging Face, so at startup the WebUI no longer tries to reach Hugging Face for them, and it starts fine offline. See [Built-in detectors](#built-in-detectors).
- 🟡 **New built-in detectors saved in `models/adetailer`** — a built-in detector downloaded for the first time is saved as a normal file in `models/adetailer`, next to your own detectors, instead of in the Hugging Face cache; the ones already in that cache stay there and are used from there. See [Built-in detectors](#built-in-detectors).
- 🟢 **Detector on the CPU with the whole WebUI** — with `--use-cpu all` (AUTOMATIC1111, Forge, reForge), `--cpu` (Forge Neo) or `--always-cpu` (Forge, reForge), and, except on macOS, with Forge Neo's `--novram`, the YOLO detectors run on the CPU, in generation and in the Detection preview, like the rest of the WebUI, instead of on the graphics card.
- 🟢 **Empty model in an API request** — a tab sent with `"ad_model": ""` counts as off, like `"None"`, instead of stopping with "Model '' not found" and skipping the tabs after it.
- 🟢 **Loads without `NansException`** — a WebUI whose `modules.devices` has no `NansException` still loads ADetailer, instead of the whole extension failing at startup.
- 🟡 **Detailer ControlNet on each region on Forge** — on Forge, Forge Neo and reForge, the ControlNet model of an ADetailer tab reads each region's own picture and mask and fits its preprocessor resolution to each region (Pixel Perfect), as on AUTOMATIC1111, so it acts only inside the region's mask and sees the regions detailed before it. On Forge Neo, a folder in ControlNet's batch input directory or in the img2img Batch tab's input directory no longer replaces its image. This changes the result of a tab with a ControlNet model on these WebUIs, on purpose.
- 🟡 **Console lines while installing and downloading** — when ADetailer installs packages with pip or downloads a built-in detector or a MediaPipe model file, the console prints a line before and after, with the time taken and, for a download, the size of the file, so a long first start no longer looks stuck. On AUTOMATIC1111 and Forge Neo the pip lines appear only after the install step ends.
- 🟡 **MediaPipe install warning shown** — when MediaPipe cannot be installed, the console now shows the warning that the `mediapipe_*` detectors will not work; before, the WebUI dropped it.
- 🟡 **MediaPipe download error without the full folder path** — when a MediaPipe model file cannot be downloaded, the console line that says where to put it by hand names the folder from the extension's folder down, without your system user name, and is printed once per session and model; each later generation prints one short reminder while the model is missing.
- 🟡 **No Hugging Face token to hf-mirror.com** — with the `HF_ENDPOINT` variable set to hf-mirror.com or one of its subdomains, ADetailer's download request no longer sends your Hugging Face token there (the stable plus.9 does). Another mirror host set with `HF_ENDPOINT` still gets it from the Hugging Face library.
- 🟡 **YOLO detectors stay loaded for the whole generation** — a YOLO detector is read from its `.pt` file once per job (one click on Generate, with all its images, tabs and X/Y/Z cells, or one "Run ADetailer on an image", with every picture of a folder) instead of once for every image and tab. At most 4 stay loaded at once, and they are freed when the job ends, also after an error or an Interrupt. No option. See [Does ADetailer Ultimate slow generation down?](#does-adetailer-ultimate-slow-generation-down).
- 🟡 **Drawn wildcards and model hashes in the image parameters** — a tab whose prompt had a wildcard or a `{a|b}` variant drawn in the detailer pass also saves "ADetailer resolved prompt", the prompt each region was really inpainted with, and each tab saves "ADetailer model hash", the short hash of its detector's file, and "ADetailer checkpoint hash" for its own checkpoint when the WebUI knows it. Pasting ignores these keys, and Civitai does not read the two hashes to recognise the models. See [Added in plus.10 beta 2](#added-in-plus10-beta-2).
- 🟡 **Missing packages named** — on a WebUI started with `--skip-install`, or after a failed install, a missing `rich` or `ultralytics` package gets one console line that names it and the pip command that installs it, instead of only an error in a traceback. See [Install and startup](#install-and-startup).

How to use each feature is explained in [Using each feature](#using-each-feature). Every change, version by version, is in [CHANGELOG.md](CHANGELOG.md).

## 🙋 Requested by users

Thank you to everyone who opened an issue, tried a beta and reported back: these requests shaped ADetailer Ultimate.

🟢 available in the stable release · 🟡 in testing in a beta (plus.10 beta 3)

| Request | Asked by | Status | Where to find it |
| --- | --- | --- | --- |
| [#1](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues/1) Install on reForge: a startup error when another ADetailer is installed too | @kvsh88 | 🟢 [v26.3.0+plus.4](https://github.com/xXIlRizzoXx/adetailer-ultimate/releases/tag/v26.3.0%2Bplus.4) | [Install](#install) · [Can I keep the original ADetailer installed too?](#can-i-keep-the-original-adetailer-installed-too) |
| [#2](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues/2) The extension did not load on AUTOMATIC1111 | @arch-official | 🟢 [v26.3.0+plus.4.1](https://github.com/xXIlRizzoXx/adetailer-ultimate/releases/tag/v26.3.0%2Bplus.4.1) and [v26.3.0+plus.5](https://github.com/xXIlRizzoXx/adetailer-ultimate/releases/tag/v26.3.0%2Bplus.5) | [Compatibility](#compatibility) · [Export and import presets](#export-and-import-presets) |
| [#3](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues/3) Choose the text encoder for the detailer pass (Forge, Forge Neo) | @koblue | 🟢 [v26.3.0+plus.5](https://github.com/xXIlRizzoXx/adetailer-ultimate/releases/tag/v26.3.0%2Bplus.5) | [Separate checkpoint, VAE, text encoder and CLIP skip](#separate-checkpoint-vae-text-encoder-and-clip-skip) |
| [#4](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues/4) Detail an image without generating it again | @koblue | 🟢 [v26.3.0+plus.6](https://github.com/xXIlRizzoXx/adetailer-ultimate/releases/tag/v26.3.0%2Bplus.6) | [Run ADetailer on an image](#run-adetailer-on-an-image) |
| [#5](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues/5) Choose which detections to inpaint | @koblue | 🟢 [v26.3.0+plus.7.3](https://github.com/xXIlRizzoXx/adetailer-ultimate/releases/tag/v26.3.0%2Bplus.7.3) | [Inpaint only these detections](#inpaint-only-these-detections) |
| [#6](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues/6) Restore the open sections and the panel's checkboxes at startup | @flux-error-man | 🟢 open sections: [v26.2.0+plus.8](https://github.com/xXIlRizzoXx/adetailer-ultimate/releases/tag/v26.2.0%2Bplus.8)<br>🟢 panel checkboxes: with the WebUI's Settings → Defaults, no update needed | [Remember last-used settings](#remember-last-used-settings) · [Startup defaults for the panel's checkboxes](#startup-defaults-for-the-panels-checkboxes) |
| [#7](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues/7) Comments in the ADetailer prompts reached the model | @flux-error-man | 🟢 comments and console line: [v26.2.0+plus.9](https://github.com/xXIlRizzoXx/adetailer-ultimate/releases/tag/v26.2.0%2Bplus.9) | [Prompts](#prompts) |
| [#8](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues/8) A refresh button for the detector menu | @flux-error-man | 🟢 [v26.2.0+plus.9](https://github.com/xXIlRizzoXx/adetailer-ultimate/releases/tag/v26.2.0%2Bplus.9) | [Add your own detectors](#add-your-own-detectors) |

@arch-official confirmed the fixes for #2 on AUTOMATIC1111, and @koblue confirmed #3, #4 and #5 on Forge Neo. reForge (#1) and classic Forge are supported and checked in the code, but the live tests run on AUTOMATIC1111 1.10 and Forge Neo only. For #6, a preset still holds the settings of one tab on purpose, so loading one never opens or closes sections or switches the whole panel on.

### Also solved: requests from users of the original ADetailer

Requests and reports from users of the original ADetailer, on its tracker and on the trackers of a WebUI and an extension used with it, that ADetailer Ultimate answers:

- 🟢 **Save the settings as presets** — asked in [Bing-su/adetailer#108](https://github.com/Bing-su/adetailer/issues/108) and [discussion #530](https://github.com/Bing-su/adetailer/discussions/530). Save a tab's setup under a name and load it in any tab: [Preset Library](#preset-library).
- 🟢 **Save and load a tab's setup, to reuse it or change the order of the tabs** — asked in [discussion #743](https://github.com/Bing-su/adetailer/discussions/743). Copy one tab and paste it into others, or save each tab as a preset and load them back in the order you want: [Copy Settings Between Tabs](#copy-settings-between-tabs).
- 🟢 **A button to discard all changes** — asked in [Bing-su/adetailer#478](https://github.com/Bing-su/adetailer/issues/478). **🆕 Reset** puts a tab back to its defaults, and "Reset every tab" does it for every tab: [Reset](#reset).
- 🟢 **Add prompt text when a certain class is detected** — asked in [Bing-su/adetailer#412](https://github.com/Bing-su/adetailer/issues/412). Write `[PROMPT], [CLASS=hand] five fingers [/CLASS]` in the tab prompt: the hands get the main prompt plus "five fingers", the other regions the main prompt. Or use one line per class in sequential mode: [Inline class blocks](#inline-class-blocks-and-skip) and [Per-class prompts](#per-class-prompts).
- 🟢 **Detail only the mouth** — asked in [Bing-su/adetailer#807](https://github.com/Bing-su/adetailer/issues/807). The `mediapipe_face_features` detector has a mouth class, next to eyes, nose, eyebrows and face: [Face parts detector](#face-parts-detector-mediapipe_face_features).
- 🟢 **"Use separate steps" ignored** — reported for API requests in [Bing-su/adetailer#795](https://github.com/Bing-su/adetailer/issues/795). With "Apply only selected scripts to ADetailer" off, "Use separate steps" and "Use separate sampler" now work, in the UI and through the API, instead of being replaced by the WebUI's own sampler settings: [Upstream options](#upstream-options).
- 🟢 **Dynamic Prompts' template lost from the saved image** — reported in [adieyal/sd-dynamic-prompts#730](https://github.com/adieyal/sd-dynamic-prompts/issues/730) and [#703](https://github.com/adieyal/sd-dynamic-prompts/issues/703). With Dynamic Prompts' "Save template to metadata" on, the image parameters keep the template with its wildcards instead of the prompt already filled in: [What is new in plus.8](#what-is-new-in-plus8).
- 🟢 **ADetailer's ControlNet on the current classic Forge** — proposed in the upstream pull request [Bing-su/adetailer#745](https://github.com/Bing-su/adetailer/pull/745), not merged there because older Forge versions need the old format. The detailer's ControlNet model works again on classic Forge, and older Forge versions keep working; classic Forge and reForge were checked in their code only, not tested live: [Upstream options](#upstream-options).
- 🟢 **Count the faces and choose which to detail** — asked in [Bing-su/adetailer#635](https://github.com/Bing-su/adetailer/issues/635). Run the Detection preview, then click the numbered boxes to choose: [Detection preview](#detection-preview).
- 🟢 **Larger values for the mask offset and erosion / dilation sliders** — asked in [Bing-su/adetailer#111](https://github.com/Bing-su/adetailer/issues/111). They go from -512 to 512 and from -256 to 256: [Upstream options](#upstream-options).
- 🟢 **Save the masks** — asked in [Bing-su/adetailer#406](https://github.com/Bing-su/adetailer/issues/406). "Save inpaint masks" saves the mask each region is inpainted with: [Save inpaint masks](#save-inpaint-masks).
- 🟢 **Switch ADetailer on and off in an X/Y/Z grid** — asked in [Bing-su/adetailer#251](https://github.com/Bing-su/adetailer/issues/251). The "[ADetailer] Enable" axis does it, and the 2nd and later tabs get axes too: [New in plus.9 beta 1](#new-in-plus9-beta-1).
- 🟢 **Sound cuts out in other programs when MediaPipe starts** — reported in [Bing-su/adetailer#508](https://github.com/Bing-su/adetailer/issues/508). MediaPipe no longer starts the audio system: [Install and startup](#install-and-startup).
- 🟢 **reForge's Advanced Model Sampling in the detailer pass** — proposed in the upstream pull request [Bing-su/adetailer#818](https://github.com/Bing-su/adetailer/pull/818). It now always runs there, except in a tab with a ControlNet model: [Settings](#settings).
- 🟢 **Start with ADetailer switched on** — asked in [lllyasviel/stable-diffusion-webui-forge#1833](https://github.com/lllyasviel/stable-diffusion-webui-forge/issues/1833). Settings → ADetailer → "Enable ADetailer at startup": [Settings](#settings).
- 🟢 **Blend the detailed region with the original** — asked in [Bing-su/adetailer discussion #293](https://github.com/Bing-su/adetailer/discussions/293). Settings → ADetailer → "Detailed region opacity": [Settings](#settings).
- 🟢 **A real upscaler for the detailed region** — asked in [Bing-su/adetailer#711](https://github.com/Bing-su/adetailer/issues/711). Settings → ADetailer → "Upscaler for ADetailer's inpaint area": [Settings](#settings).
- 🟢 **The same face inpainted twice** — reported in [Bing-su/adetailer#669](https://github.com/Bing-su/adetailer/issues/669). Settings → ADetailer → "Drop overlapping detections (max overlap, 0 = off)": [Settings](#settings).
- 🟢 **Swap two tabs, or apply one tab to all** — asked in [Bing-su/adetailer#297](https://github.com/Bing-su/adetailer/issues/297) and [#320](https://github.com/Bing-su/adetailer/issues/320): [Copy Settings Between Tabs](#copy-settings-between-tabs).
- 🟢 **ADetailer runs once more than the tabs of an API request** — reported in [Bing-su/adetailer#485](https://github.com/Bing-su/adetailer/issues/485). A request that sends only the tabs now runs one pass per tab: [Does it work with the WebUI's API?](#does-it-work-with-the-webuis-api).
- 🟢 **A color correction option** — asked in [Bing-su/adetailer#731](https://github.com/Bing-su/adetailer/issues/731). Settings → ADetailer → "Match the colors of each inpainted region to the picture" matches each region to the picture: [Settings](#settings).
- 🟢 **Explain what each option does** — asked in [Bing-su/adetailer#23](https://github.com/Bing-su/adetailer/issues/23). Hover over an option of a tab to see a short explanation: [Interface](#interface).
- 🟢 **No Ultralytics usage analytics** — asked in [Bing-su/adetailer#613](https://github.com/Bing-su/adetailer/issues/613). ADetailer switches them off before its detections, without writing Ultralytics' settings file: [Compatibility](#compatibility).
- 🟢 **Hide the red detection boxes** — asked in [Bing-su/adetailer discussion #482](https://github.com/Bing-su/adetailer/discussions/482). From plus.9, Settings → ADetailer → "Hide detection boxes from the live preview" keeps them out of the live preview and the progress API: [Settings](#settings).
- 🟢 **Run ADetailer after a face swap (roop, ReActor)** — asked in [Bing-su/adetailer#193](https://github.com/Bing-su/adetailer/issues/193), [#327](https://github.com/Bing-su/adetailer/issues/327) and [discussion #159](https://github.com/Bing-su/adetailer/discussions/159). The WebUI's Settings → Callbacks sets the order, when the other extension works in the same step: [The result looks wrong](#the-result-looks-wrong).
- 🟢 **"Use separate steps" runs fewer steps** — reported in [Bing-su/adetailer#649](https://github.com/Bing-su/adetailer/issues/649). The detail pass is an img2img pass, and a WebUI option runs the full number: [The result looks wrong](#the-result-looks-wrong).
- 🟢 **"Save images before ADetailer" in easy reach** — asked in [Bing-su/adetailer#726](https://github.com/Bing-su/adetailer/issues/726). Add it to the WebUI's Quicksettings list: [FAQ](#can-i-switch-save-images-before-adetailer-from-the-top-of-the-page).
- 🟢 **Only part of the main prompt in the ADetailer prompt, also with wildcards** — asked in [Bing-su/adetailer#805](https://github.com/Bing-su/adetailer/issues/805), [#764](https://github.com/Bing-su/adetailer/issues/764), [#673](https://github.com/Bing-su/adetailer/issues/673), discussions [#287](https://github.com/Bing-su/adetailer/discussions/287) and [#553](https://github.com/Bing-su/adetailer/discussions/553), and [adieyal/sd-dynamic-prompts#561](https://github.com/adieyal/sd-dynamic-prompts/issues/561). `[PROMPT=eyes]` brings only the parts of each image's main prompt that name the eyes: [Upstream options](#upstream-options).
- 🟢 **The detection previews in the results gallery, without saving them** — asked in [Bing-su/adetailer#113](https://github.com/Bing-su/adetailer/issues/113) and [#259](https://github.com/Bing-su/adetailer/issues/259). Settings → ADetailer → "Show detection previews in the results gallery": [Settings](#settings).
- 🟡 **Masked content for the detailer pass** — asked in [Bing-su/adetailer#516](https://github.com/Bing-su/adetailer/issues/516) and [#802](https://github.com/Bing-su/adetailer/issues/802). Each tab has a "Masked content" option (fill, original, latent noise, latent nothing), as in the WebUI's own inpaint: [Masked content](#masked-content).
- 🟡 **The same wildcard value in the main prompt and the ADetailer prompt** — asked in [Bing-su/adetailer#764](https://github.com/Bing-su/adetailer/issues/764), and in [adieyal/sd-dynamic-prompts#561](https://github.com/adieyal/sd-dynamic-prompts/issues/561). A wildcard in a tab's prompt that the main prompt also holds takes the value drawn there: [FAQ](#can-the-adetailer-prompt-keep-what-a-wildcard-picked-in-the-main-prompt).
- 🟡 **A black or spoiled image with Soft Inpainting** — reported in [Bing-su/adetailer#694](https://github.com/Bing-su/adetailer/issues/694) and [AUTOMATIC1111/stable-diffusion-webui#16055](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/16055). ADetailer no longer changes the main image's Soft Inpainting result: [What is new in plus.10](#what-is-new-in-plus10).
- 🟡 **The downloaded detectors in `models/adetailer`** — related to the reports about the downloads from Hugging Face in [Bing-su/adetailer#153](https://github.com/Bing-su/adetailer/issues/153), [#163](https://github.com/Bing-su/adetailer/issues/163) and [#619](https://github.com/Bing-su/adetailer/issues/619). A built-in detector downloaded for the first time is saved as a normal file in `models/adetailer`; the ones already in the Hugging Face cache stay there: [Built-in detectors](#built-in-detectors).
- 🟡 **Dynamic Prompts' Magic Prompt rewrites the ADetailer prompts** — reported in [adieyal/sd-dynamic-prompts#669](https://github.com/adieyal/sd-dynamic-prompts/issues/669) and [#711](https://github.com/adieyal/sd-dynamic-prompts/issues/711). Dynamic Prompts no longer runs on the regions whose prompt holds no wildcard or variant; on a region whose prompt holds one it still does: [Changed in plus.10 beta 1](#changed-in-plus10-beta-1).

**Missing something?** [Open an issue](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues/new) and describe what you would like to do. Ideas already being considered are in the [Roadmap](#roadmap).

<a id="beta-in-testing"></a>

## What is new in plus.10

**v26.2.0+plus.10.beta.3**, released on 2026-11-25 as a pre-release on the [Releases page](https://github.com/xXIlRizzoXx/adetailer-ultimate/releases/tag/v26.2.0%2Bplus.10.beta.3), is the third beta of plus.10. It fixes "Run ADetailer on an image" with a Flux model, which ran with CFG scale 7 and 28 steps and burnt the faces ([issue #9](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues/9)): on Forge Neo the run now takes the img2img values of Forge Neo's preset for the family of its model, for every family that has one, and its status says the denoise, steps, CFG scale and sampler the pass used. It keeps everything of beta 2, described below.

**Beta 3** (v26.2.0+plus.10.beta.3) is on the Releases page and the `beta` branch. Its changes since beta 2 are listed at the top of [CHANGELOG.md](CHANGELOG.md); with them, all 3828 offline regression tests pass on AUTOMATIC1111 and Forge Neo. Forge and reForge are supported and checked in the code. Please report any problem in the [Issues](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues).

**v26.2.0+plus.10.beta.2**, released on 2026-11-19 as a pre-release on the [Releases page](https://github.com/xXIlRizzoXx/adetailer-ultimate/releases/tag/v26.2.0%2Bplus.10.beta.2), is the second beta of plus.10. It adds an option in Settings → ADetailer that makes "Run ADetailer on an image" and its folder batch use the prompt saved in each picture, and keeps the YOLO detectors loaded for the whole generation. It adds a guide to the panel, "Guide: several subjects, one LoRA each", that fills a tab so that each person in a picture is redrawn with their own LoRA. It also lets YOLOE detectors take typed classes like YOLO-World, adds an option to detail with the hires fix checkpoint and one to keep the transparency of a picture in "Run ADetailer on an image" and with Skip img2img, skips on Forge and Forge Neo a tab whose checkpoint cannot be loaded with the VAE file of its pass when its files show it for sure, warns there when a tab's checkpoint of another model family keeps the main model's VAE or text encoder files, saves in the image parameters the prompts a wildcard drew in a tab and the hashes of a tab's detector and checkpoint, says in the console when a detailer prompt names the same LoRA twice, when the model picked is not a detector and when a Python package is missing, applies the selected Styles once also when another extension changed the main prompt, and gives each section of a tab a coloured bar and an emoji. It keeps everything of the first beta, v26.2.0+plus.10.beta.1, released on 2026-11-11. Beta 1 adds a "Masked content" option to every tab, a "1 megapixel" inpaint size choice in Settings, Anytest ControlNet models to the menu, lets a wildcard in a tab's prompt take the value drawn in the main prompt, writes each box's share of the image on the Detection preview, prints console lines while ADetailer installs packages or downloads models, lists every detector when two have the same file name in different folders, and saves a built-in detector downloaded for the first time in `models/adetailer`, and three new guides explain the same face from a reference picture, many pictures each with its own prompt, and how to reinstall MediaPipe. Beta 1 changes one behaviour: Dynamic Prompts no longer runs on the regions whose prompt holds no wildcard or variant, so its Magic Prompt no longer rewrites them. Beta 1 fixes Soft Inpainting, where ADetailer spoiled the main image, ADetailer's ControlNet on Forge, Forge Neo and reForge, which now works on each region as on AUTOMATIC1111, the warning for a MediaPipe that cannot be installed, which was never shown, the Hugging Face token, which is no longer sent to hf-mirror.com when you set `HF_ENDPOINT` to it, and the console line for a MediaPipe model file that cannot be downloaded, which no longer shows the full folder path. The stable version is still v26.2.0+plus.9: what it changed is in [What is new in plus.9](#what-is-new-in-plus9). In this section, "beta 1" and "beta 2" mean the betas of plus.10.

**Beta 2** (v26.2.0+plus.10.beta.2), released on 2026-11-19, stays on the Releases page; the `beta` branch moved on to beta 3, which includes everything of beta 2. Its changes since beta 1 are listed in the v26.2.0+plus.10.beta.2 section of [CHANGELOG.md](CHANGELOG.md), right below beta 3's; with them, all 3790 offline regression tests passed on AUTOMATIC1111 and Forge Neo.

**Beta 1** (v26.2.0+plus.10.beta.1), released on 2026-11-11, stays on the Releases page; the `beta` branch moved on to beta 2, which includes everything of beta 1. Its changes are listed in the v26.2.0+plus.10.beta.1 section of [CHANGELOG.md](CHANGELOG.md), right below beta 2's; with them, all 2381 offline regression tests passed on AUTOMATIC1111 and Forge Neo.

**To try the beta**, follow [Switch to the beta and back](#switch-to-the-beta-and-back): two commands in a terminal, then a restart. The version badge in the ADetailer panel then shows `v26.2.0+plus.10.beta.3`. plus.10 adds one tab setting, "Masked content" (from beta 1), at the end of the list of per-tab options; its default, "original", does what ADetailer always did, so your presets and remembered settings work in both versions, and switching does not touch them (see [Your data](#your-data)). Beta 3 adds no tab setting and no option. Beta 2 adds no tab setting, only three options in Settings → ADetailer: "Run ADetailer on an image uses the prompt saved in the image", "Run ADetailer on an image and Skip img2img keep the transparency of the picture" and "With hires fix, detail with the hires checkpoint". The guide box that beta 2 adds to the panel, "Guide: several subjects, one LoRA each", is not a setting: its choices are not saved, and the tabs it fills use only tab settings that plus.9 and beta 1 already have. What changes if you go back to plus.9 or to beta 1 is in [Notes on plus.10 beta 2](#notes-on-plus10-beta-2) and [Notes on plus.10 beta 1](#notes-on-plus10-beta-1).

### New in plus.10 beta 3

🟢 available in the stable release · 🟡 in testing in a beta (plus.10 beta 3)

> [!NOTE]
> **plus.10 beta 3 in short:**
>
> - 🟡 **Fixed: Run ADetailer on an image with Flux:** it no longer runs a Flux model with CFG scale 7 and 28 steps, which burnt the faces and took about twice the time.
> - 🟡 **Changed: the img2img values of Forge Neo's presets:** on Forge Neo, "Run ADetailer on an image" takes the img2img values of the preset of its model's family, as Forge Neo's img2img tab does.
> - 🟡 **Changed: the status says the values used:** the status under the result adds the denoise, steps, CFG scale and sampler of the pass.

The details are in the v26.2.0+plus.10.beta.3 section of [CHANGELOG.md](CHANGELOG.md).

#### Fixed in plus.10 beta 3

- 🟡 **Run ADetailer on an image with Flux** — without the tab's "Use separate steps", "Use separate CFG scale" and "Use separate sampler", the run used 28 steps, CFG scale 7 and the WebUI's first sampler, which burnt the faces of a Flux model and took about twice the time ([issue #9](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues/9)). On Forge and Forge Neo a Flux model now gets Flux's values, and right after another main model was picked, the run no longer takes the values of the model still loaded. Tested live on Forge Neo 2.33 with a Flux and an SDXL main model. Details: [Run ADetailer on an image](#run-adetailer-on-an-image).

#### Changed in plus.10 beta 3

- 🟡 **The img2img values of Forge Neo's presets** — on Forge Neo, without the tab's own "Use separate ..." options, "Run ADetailer on an image" takes the img2img Sampler, Scheduler, Steps, CFG and Distilled CFG (or Shift) of the preset in Settings for the family of its model, for every family that has one (SD 1.5, SDXL, Flux, Flux.2 Klein, Z-Image Turbo, Qwen-Image, Lumina, Wan, Anima and others). SD 2, Chroma, AUTOMATIC1111 and reForge keep the old values. Tested live on Forge Neo 2.33 with Flux and SDXL; the other families are covered by the offline tests only.
- 🟡 **The status says the values used** — the status of "Run ADetailer on an image" adds the denoise, steps, CFG scale and sampler of the pass, read from the parameters the WebUI wrote on the result, for example "✅ ADetailer pass complete. Used: denoise 0.6, 20 steps, CFG scale 1, Euler (Beta)." Tested live on Forge Neo 2.33.

### New in plus.10 beta 2

🟢 available in the stable release · 🟡 in testing in a beta (plus.10 beta 3)

> [!NOTE]
> **plus.10 beta 2 in short:**
>
> - 🟡 **The prompt saved in each picture:** a new option in Settings → ADetailer makes "Run ADetailer on an image" and its folder batch use the prompt and negative prompt saved in each picture, so a blank tab prompt or `[PROMPT]` details each picture with its own prompt.
> - 🟡 **A guide for several subjects, one LoRA each:** a closed box in the panel, "Guide: several subjects, one LoRA each". Pick a LoRA and a few words for up to four people and press "✨ Fill the tabs": it sets the detector, writes the `[SEP]` prompt and the settings that keep the parts in order, and its status says what to check.
> - 🟡 **YOLO detectors stay loaded:** most detectors are read from the disk once per generation, not again for every image, tab and X/Y/Z cell.
> - 🟡 **Transparency kept:** a new option in Settings → ADetailer gives the result of "Run ADetailer on an image", and of img2img with Skip img2img, the transparency of a cut-out picture back.
> - 🟡 **Hires fix checkpoint:** a new option in Settings → ADetailer details the picture with the hires fix checkpoint after hires fix, in the tabs without their own checkpoint.
> - 🟡 **A tab whose checkpoint cannot load is skipped:** on Forge and Forge Neo, a tab whose checkpoint cannot be loaded with the VAE file of its pass is skipped before the pass when its `.safetensors` files show it for sure, instead of losing the whole generation.
> - 🟡 **Model family warning:** on Forge and Forge Neo, the console warns when a tab's checkpoint of another model family keeps the main model's VAE or text encoder files.
> - 🟡 **More in the image parameters:** the prompts a wildcard drew in a tab, and the hashes of a tab's detector and checkpoint.
> - 🟡 **Clearer console:** a line when a detailer prompt names the same LoRA twice, when the model picked is not a detector, and when the `clip`, `rich` or `ultralytics` package is missing.
> - 🟡 **Fixed: Styles applied once:** also when another extension, such as Dynamic Prompts or the "wildcards" extension, changed the main prompt.
> - 🟡 **Changed: YOLOE detectors take typed classes:** like YOLO-World, they show a text field for the words to find.
> - 🟡 **Coloured sections:** each section of a tab has a coloured bar and an emoji before its title, and the detector and ControlNet controls have their own labels.

The details are in the v26.2.0+plus.10.beta.2 section of [CHANGELOG.md](CHANGELOG.md).

#### Added in plus.10 beta 2

- 🟡 **The prompt saved in each picture** — Settings → ADetailer → "Run ADetailer on an image uses the prompt saved in the image" (off by default). With it on, each picture of "Run ADetailer on an image", alone or in a folder batch, gets as its main prompt and negative prompt the ones the WebUI saved in it: the parameters of a PNG, or the EXIF comment of a JPEG or WebP, the text PNG Info shows. A blank tab prompt or `[PROMPT]` then uses that prompt, as in the generation that made the picture, and so do `[PROMPT=words]` and "Use LoRAs from main prompt". Only the two prompts are taken: not the seed, sampler, steps or size, and no ADetailer setting. A picture without saved parameters keeps an empty main prompt, and the status line and the console say so; the summary of a folder batch counts those pictures, and the console gives their number in one line. A camera photo with a blank EXIF comment counts as having none, and so does a picture pasted from the clipboard. Each result records the prompts actually used: the picture's own, or an empty prompt when the picture had none (the tab's prompt stays in "ADetailer prompt"), so running the tool again on a result reuses the right prompt. Off, nothing changes. Tested live on AUTOMATIC1111 and Forge Neo. Details: [Many pictures, each with its own prompt](#many-pictures-each-with-its-own-prompt).
- 🟡 **Transparency kept on the result** — Settings → ADetailer → "Run ADetailer on an image and Skip img2img keep the transparency of the picture" (off by default). With it on, a picture with transparent parts, such as a cut-out PNG, is still detailed with those parts filled with the img2img background colour, then the result gets the picture's transparency back, alone or in a folder batch, also when nothing was detected. A picture whose pixels are all opaque stays as before. The file saved to `ADetailer-Inpaint` keeps the WebUI's file format and has the transparency only in a format that has it (PNG; WebP on Forge Neo): with JPEG, or WebP on AUTOMATIC1111, the status and the console say so instead of changing the format. With "📁 Save results in the source folder instead", PNG, WebP and TIFF copies keep it, and a copy in a format without it, such as a transparent picture named `.jpg`, is saved filled instead of failing; the summary of a folder batch counts those files and the console gives their number in one line. The same option holds in img2img with "Skip img2img" ticked, also in the Batch tab of img2img: the picture the WebUI shows and saves gets the input's transparency back, and with a file format without transparency the console and the text under the results say so once. On Forge Neo, a Batch tab file whose sides are not multiples of the WebUI's "Resolution Step" setting (64 by default) comes out without it. Off, nothing changes. "Run ADetailer on an image" is tested live on AUTOMATIC1111 and Forge Neo (PNG on both, WebP on Forge Neo); the Skip img2img part is tested live on AUTOMATIC1111 only, with an API request (off, the result is filled; on, its transparency is the same as the input's, and the region is detailed), and its Batch tab and Forge Neo are covered by the offline tests only. Details: [Run ADetailer on an image](#run-adetailer-on-an-image) and the notes on Skip img2img in [Upstream options](#upstream-options).
- 🟡 **A guide for several subjects, one LoRA each** — a closed box in the ADetailer panel, above the tabs, in txt2img and img2img: "Guide: several subjects, one LoRA each". It is the guided way of [A different prompt or LoRA for each face](#a-different-prompt-or-lora-for-each-face), made for people. Pick "What to redraw" ("Whole person (outline)", "Face only" or "Whole person, then face (2 tabs)") and "People in the picture" (Auto, 2, 3 or 4), give each of the four rows a LoRA from its menu, a weight and a few words, and press "✨ Fill the tabs". The guide switches ADetailer on and fills the 1st tab (with the last choice, the 2nd too): the first installed detector of its list (`person_yolov8s-seg.pt` or `person_yolov8n-seg.pt` for a whole person; `face_yolov8s.pt`, `face_yolov8n.pt` or `mediapipe_face_full` for a face), a prompt with one part per subject, such as `woman, red hair, green dress, <lora:my_lora_a:1> [SEP] man, grey beard, <lora:my_lora_b:1> [SEP] [SKIP]`, and the settings that keep the parts in order: "Mask only the top k (0 to disable)" set to the number of subjects, filtered by Area, "Mask merge mode" on None, "Inpaint only masked" on, "Strip LoRAs from the detailer prompt" and "Process classes sequentially" off, the class filter and "Inpaint only these detections" emptied and, for a whole person, "Use bbox as mask (segmentation models)" off, so that the outline is used. It also sets "Inpaint denoising strength" to 0.6, and the last line of its status says that the people are easier to recognise when the main prompt already describes them, without their LoRAs. A row left empty among the people counted becomes `[SKIP]`, so that person is left as they are, and a row with a LoRA and no words gets `[PROMPT]`, the main prompt, before its tag. The subjects are the biggest people in the picture, as many as the last filled row or "People in the picture" says, counted from left to right: the status under the button reads, for example, "Subject 1 is the leftmost of the 2 biggest people." The status also says what can go wrong: Settings → ADetailer → "Sort bounding boxes by" not on "Position (left to right)" (the guide never changes the Settings: it says what to set there), a detector that is not installed, a LoRA that is also in the main prompt, a LoRA name the WebUI does not list, a prompt that was replaced, another tab that is still switched on. The prompt the filled tabs had is replaced; their other settings, the other tabs and the main prompt are left alone, and everything can be edited afterwards. The guide's own choices are not saved anywhere. It adds no tab setting and no option in Settings. Tested live on AUTOMATIC1111 1.10.1 and Forge Neo 2.32. On AUTOMATIC1111, each of the three choices of "What to redraw" gave every part to the right person with "Position (left to right)", also with "People in the picture" on 2 and one row filled; three status warnings (the sort order, a 2nd tab still switched on, "Max tabs" at 1), a click on the button during a generation, presets, Reset, the parameters of a two-tab picture pasted back and the gallery buttons after a reload of the page were tried too. On Forge Neo, the button filled the tabs, the status warned about a prompt comment mark and a generation was made from the page. Not tried live: LoRA aliases, `lyco` tags, a typed LoRA name that the WebUI does not list and a translated interface, which are covered by the offline tests only. Details: [Several subjects, one LoRA each (guide)](#several-subjects-one-lora-each-guide); its limits are in [Notes on plus.10 beta 2](#notes-on-plus10-beta-2).
- 🟡 **Detail with the hires fix checkpoint** — Settings → ADetailer → "With hires fix, detail with the hires checkpoint" (off by default). After a txt2img hires fix with its own Hires checkpoint, the tabs without their own checkpoint detail the upscaled picture with that checkpoint instead of the main one, and on Forge and Forge Neo with the "Hires VAE / Text Encoder" choices and the "Hires Distilled CFG Scale". A tab's own VAE or text encoder still replaces the hires ones, and a tab with its own checkpoint keeps it. The WebUI puts the main checkpoint back after the pass. No effect without hires fix, in img2img and in "Run ADetailer on an image"; with frozen settings the main checkpoint is used, and the console says so. Tested live on AUTOMATIC1111 and Forge Neo. Details: [Separate checkpoint, VAE, text encoder and CLIP skip](#separate-checkpoint-vae-text-encoder-and-clip-skip).
- 🟡 **A tab whose checkpoint cannot load is skipped** — on Forge and Forge Neo, a checkpoint that fails to load leaves the WebUI without a model, and on Forge Neo the whole generation was lost with it. Before a tab's pass, ADetailer now reads the header of the `.safetensors` files, without loading them, and skips the tab when the checkpoint of its pass cannot be loaded with the VAE file the pass would give it: the other tabs run and the image is saved. It does so when a tab with its own checkpoint would keep a VAE file of the main model with another number of latent channels (an SD 1.5 or SDXL checkpoint in the tab with the VAE file of a Flux, Chroma, Z-Image or Lumina main model, or a Flux checkpoint in the tab with the VAE file selected for an SD 1.5, SD 2 or SDXL main model), and when the VAE file chosen in the tab's own "Use separate VAE" does not fit, also in a tab without its own checkpoint. The console says it in one line per generation and tab, with what to set in the tab, and the same line shows under the results of txt2img and img2img; "Run ADetailer on an image" gives no result, and its status says that ADetailer did not run because the tab was skipped. In any doubt nothing is skipped: a `.ckpt` or GGUF file, a checkpoint in the tab that is not recognised as SD 1.5, SDXL or Flux, a VAE of another layout among the files, the hires fix checkpoint. Only the VAE file is checked, not the text encoders. Nothing changes on AUTOMATIC1111 and reForge. Tested live on Forge Neo 2.32 in two cases: a Flux main model with its VAE and text encoder files and a tab with an SDXL checkpoint that kept them, and an SDXL main model with Flux's VAE file chosen in the "Use separate VAE" of a tab with an SDXL checkpoint. In both the tab was skipped with its console line, the generation ended, the picture was saved and the WebUI kept working, and the same tab with files that fit ran and detailed the picture. Classic Forge and the other cases are covered by the offline tests only. Details: [Separate checkpoint, VAE, text encoder and CLIP skip](#separate-checkpoint-vae-text-encoder-and-clip-skip).
- 🟡 **Warning for a checkpoint of another model family** — on Forge and Forge Neo, when a tab's own checkpoint is of another model family than the main model (one of them SD 1.5, SD 2 or SDXL) and its pass kept the main model's VAE or text encoder files, ADetailer prints one console warning per generation and tab (in a folder batch of "Run ADetailer on an image", one for the whole folder), after the tab's first region, with what to set in the tab; a single "Run ADetailer on an image" repeats it in its status, the summary of a folder batch does not. Once the tab's "Use separate VAE" has a `.safetensors` VAE file with the latent channels of its checkpoint's family (for an SD 1.5, SD 2, SDXL, Flux, Chroma, Z-Image or Lumina checkpoint), the main model's VAE file no longer counts: the line names only the main model's other files (its text encoders), and nothing is printed when none is kept. The line stays, though, while a VAE file of the main model that the checkpoint loads with too (the same latent channels, or a layout that is not read; one saved with a checkpoint's `first_stage_model.` prefix only for an SD 1.5, SD 2 or SDXL checkpoint, as Forge Neo lays such a file over no other family) is still among the files of the pass: Forge Neo reads the files in the order of their paths, so the main model's can be the one used. When a file the line names is where no choice of the tab takes it out (a VAE file of the main model outside a folder named VAE once the tab has its own, a text encoder file outside a folder named text_encoder), the line says to move it into such a folder and to pick it again in the WebUI's "VAE / Text Encoder" list. When the tab's checkpoint does not even load with those files, in a case the skip above does not recognise, its family cannot be read: the console then says that the checkpoint could not be loaded while the pass kept the main model's files, with the advice for the family that the header of the checkpoint's file tells (the same advice for SD 1.5 or SDXL; for Flux, a Flux VAE file in the tab's "Use separate VAE" instead of Automatic; both when the header does not tell; when the main model's `.safetensors` VAE file is outside a folder named VAE, so that Automatic would keep it, first to move that file into a folder named VAE and to pick it again in the WebUI's "VAE / Text Encoder" list, and only then the advice for the checkpoint's family, or only to move it when the tab already has a VAE file of that family; a text encoder file outside a folder named text_encoder is told to be moved as above, and "None" and Automatic are not told again when they would take out nothing; when the header tells the main model's family, no text encoder file is told to be moved), and so does the status of a single "Run ADetailer on an image" (when the line says to move a file, the status says it too). The warning blocks nothing. No warning on AUTOMATIC1111 and reForge, or when a checkpoint that did load is of an unknown family. Tested live on Forge Neo with a Flux main model and an SDXL checkpoint in the tab, before the skip above was added: that setup is now skipped when its files are `.safetensors` files (the skip is tested live on Forge Neo too), so the warning is left for the cases that the skip does not recognise; in those cases it is covered by the offline tests only. The status line of a checkpoint that does not load is covered by the offline tests only, and on Forge it is all covered by the offline tests only.
- 🟡 **The prompts a wildcard drew in a tab, saved in the image** — a wildcard or a `{a|b}` variant written only in an ADetailer prompt is drawn in the detailer pass, and the saved image kept only the text as typed. Each tab where one was drawn now also writes "ADetailer resolved prompt" (and "ADetailer resolved negative prompt"; "ADetailer resolved prompt 2nd" and so on for the other tabs), with the whole prompt each region was inpainted with, one `[SEP]` part per region. The key is written only when a wildcard or a variant was really drawn, also when the wildcard took the main prompt's value; a tab where nothing was drawn writes nothing. "ADetailer prompt" stays as typed, so pasted parameters draw again, and pasting ignores the new keys. A long value is cut and says so (about 4000 bytes for each key and about 12000 for all those of an image): the regions that do not fit are left out and counted, as in "(+2 more)", and when not even the first region's prompt fits, it is cut and ends with " ... (cut)". Each image of a batch saves its own values; "Run ADetailer on an image" writes none. Not tested live yet.
- 🟡 **The hashes of a tab's detector and checkpoint, saved in the image** — each tab also writes "ADetailer model hash", the first 10 characters of the sha256 of its detector's file, like the WebUI's "Model hash"; a MediaPipe detector has no file and writes none, and nothing is hashed with `--no-hashing`. A tab with its own checkpoint writes "ADetailer checkpoint hash" when the WebUI already knows that hash; ADetailer never hashes a checkpoint itself. Pasting ignores both keys. They are information for whoever reads the parameters: Civitai does not read these two keys to recognise the models a picture used. Tested live: "ADetailer model hash" on AUTOMATIC1111 1.10.1, in the saved parameters and in PNG Info, and both keys on Forge Neo 2.32; the rest is covered by the offline tests only.
- 🟡 **Repeated LoRA note** — the WebUI loads a LoRA once for each tag in a prompt (classic Forge loads two identical tags once), so a LoRA named twice in the final prompt of a detailer pass can get a stronger effect than one tag asks for: for example with `[PROMPT], <lora:detail:0.5>` in the tab while the main prompt already has `<lora:detail:1>`, or with one more in the "Prompt append" text or in a selected Style. The console now says so in one line per generation, tab and LoRA (in a folder batch, one for the whole folder). The same LoRA in the main prompt and in a tab that has its own prompt is not doubled and gets no line, and neither does a LoRA in the negative prompt. The prompt is not changed. Tested live on AUTOMATIC1111 1.10.1 with a folder batch of "Run ADetailer on an image" (one line for the whole folder); the rest is covered by the offline tests only. Details: [LoRAs in the detailer prompt](#loras-in-the-detailer-prompt).

#### Changed in plus.10 beta 2

- 🟡 **YOLO detectors stay loaded for the whole generation** — a YOLO detector was read from its `.pt` file again for every image of a batch, every tab and every X/Y/Z cell. It now stays loaded for the whole job: one click on Generate, with all its images, tabs and X/Y/Z cells, or one click on "Run ADetailer on an image", with every picture of a folder. At most 4 detectors stay loaded at once, the ones used last, and they are freed when the job ends, also after an error or an Interrupt. A detector file replaced by another one with the same name is loaded again. YOLO-World detectors and the Detection preview load their detector each time, as before. It adds no option, and a detector that runs on the CPU (for example with `--lowvram` or `--medvram`) still runs there. Tested live on AUTOMATIC1111 and Forge Neo with a 2 × 2 batch and two tabs.
- 🟡 **YOLOE detectors take typed classes, like YOLO-World** — a YOLOE detector with text prompts (`yoloe-11s-seg.pt`, `yoloe-v8l-seg.pt`, `yoloe-26n-seg.pt`, in any letter case) now shows the text field "ADetailer detector CLASSES (YOLO-World / YOLOE)", next to an empty CLASSES list: type comma-separated words to find. The regions carry those names, so `[CLASS=…]` blocks apply; "Exclude selected (NOT)" is hidden and off. Left empty, the detector finds the classes saved in its file, and nothing is downloaded. The prompt-free `-pf` models stay ordinary detectors with their own classes. The first time a detection runs with typed classes, the Ultralytics library downloads its text model from github.com/ultralytics/assets (`mobileclip_blt.ts`, about 600 MB; `mobileclip2_b.ts`, about 254 MB, for YOLOE-26) and installs the `clip` package from GitHub (ultralytics/CLIP) when it is missing (Forge Neo). If the classes cannot be applied, the console says why and that tab inpaints nothing. A YOLOE detector is never kept loaded between tabs. This changes what a YOLOE detector does for anyone who used one: before, it was an ordinary detector with the CLASSES list of its file, and the classes picked there now show in the text field and are looked for as text prompts. Tested live on AUTOMATIC1111 and Forge Neo with `yoloe-26n-seg.pt`. Details: [YOLO-World detectors](#yolo-world-detectors).
- 🟡 **Coloured sections** — each section of a tab now has a coloured bar and an emoji before its title: blue for settings (Inpaint prompts, Detection, Mask Preprocessing, Inpainting), amber for tools (Detection preview, Run ADetailer on an image), grey for the preset library file. The detector and ControlNet controls now sit under their own small labels, "Detector" and "ControlNet". Nothing moved and the titles are unchanged, so translations and saved settings keep working. Tested live on AUTOMATIC1111 and Forge Neo.

#### Fixed in plus.10 beta 2

- 🟡 **Styles applied once also when another extension changed the main prompt** — with Styles selected and a blank or `[PROMPT]` ADetailer prompt, the styles, their LoRAs included, were still applied twice in the detailer pass when a wildcard was drawn by the "wildcards" extension or Dynamic Prompts was enabled: a known issue of plus.9, also in the original ADetailer and in beta 1. ADetailer now finds the text that the styles wrap and gives the pass only that, when the styles give back exactly the WebUI's main prompt from it. Three rare cases remain: see [Notes on plus.10 beta 2](#notes-on-plus10-beta-2). Not tested live with those extensions yet.
- 🟡 **A model that is not a detector gets one clear line** — an Ultralytics `.pt` file whose results have no boxes (an OBB model, a classification model or a semantic segmentation model), picked as the detector, ended in "'NoneType' object has no attribute 'xyxy'" with a traceback. Now nothing is detected and the console says why once per generation and tab; the Detection preview and "Run ADetailer on an image" give the reason in their status, and a folder batch says it once and counts its files as failed. Detection, segmentation and pose models work as before. Covered by the offline tests only: not tested live yet.
- 🟡 **YOLO-World or YOLOE without the `clip` package** — when the Ultralytics library cannot install its `clip` package by itself (no internet connection, or its automatic install switched off), a YOLO-World detector with typed classes stopped ADetailer for the image with a traceback. A YOLO-World or YOLOE detector now leaves its tab without inpainting, and the console names the missing package and the pip command to run with the WebUI's Python, once per generation and tab; the Detection preview and "Run ADetailer on an image" show the reason in their status. A file path in the text of another error of the text model is cut down to its file name, and a web address in it is left as it is. Nothing is guessed about where a path ends. The folders the program knows for sure (the user's, the WebUI's, this extension's, the detector's, Python's, the temporary one and those named by the WebUI and by the environment) are taken out wherever they are written; then any other word that is plainly a piece of a path is cut down to what follows its last slash or backslash. Plainly: it starts with `/`, `~/`, `~\`, `.\`, `..\`, a drive letter and a slash or backslash, a network share (`\\server\share`, also with the doubled backslashes of an error's repr), the long form `\\?\` (also `\\?\UNC\`), or a backslash, a name that starts with a capital letter and another backslash (`\Users\name`); or it holds a backslash and ends with a file name such as `weights\text.ts`; or it holds a backslash and comes after a piece of a path that did not end with a file name. The other words of the text, also the ones like "and/or" and a backslash that belongs to the message such as `'\t'` or `\d`, are left as they are, unless they come after a piece of a path that did not end with a file name (`D:\Other: bad escape \d` becomes `Other: bad escape d`). One limit is left: a folder that is not one of those and has a space in its name leaves the words of that name in the text (`D:\Other Disk\a.pt` becomes `Other a.pt`), and, written with forward slashes, also what follows the space (`/mnt/Other Disk/b/a.pt` becomes `Other Disk/b/a.pt`). Covered by the offline tests only: not tested live yet.
- 🟡 **A missing `rich` or `ultralytics` package named** — on a WebUI started with `--skip-install`, or after a failed install: without `rich`, a plain console line names the package and the pip command before the error (ADetailer still cannot load without it); without `ultralytics`, the console says so at startup and again before the error of a detection, with the pip command, instead of the extension loading without a word and the first generation failing. Covered by the offline tests only: not tested live yet.

### Notes on plus.10 beta 2

- **While a job runs, up to 4 YOLO detectors stay in memory**, in video memory when they run on the graphics card. They are freed when the job ends, so between generations the memory is as before.
- **A picture pasted from the clipboard, or cropped in the Input box, has no saved prompt:** with the prompt option ("Run ADetailer on an image uses the prompt saved in the image") on, it keeps an empty main prompt, and the status says so. Drop the picture's file instead, or use the folder batch.
- **Only the two prompts come from the picture:** its seed, steps, sampler and size do not. To use those too, use the img2img Batch recipe in [Many pictures, each with its own prompt](#many-pictures-each-with-its-own-prompt).
- **Going back to beta 1 or plus.9:** they ignore the prompt option, so "Run ADetailer on an image" starts every picture with an empty main prompt again. The saved value is kept for when you come back.
- **The prompt option stays in English on a translated interface**, with its help text, until the Language Diffusion dictionaries have it.
- **The two other new options and the new "Detector" label stay in English on a translated interface**, with the help texts of the two options, until the Language Diffusion dictionaries have them.
- **Going back to beta 1 or plus.9 with the other changes:** they ignore the two other new options (the saved values are kept), the sections lose their bar, emoji and labels, and a YOLOE detector is an ordinary detector again.
- **A YOLOE detector with typed classes is slow:** the Ultralytics library loads its text model again for every detection with typed classes, and the first one downloads it (about 600 MB, or about 254 MB for YOLOE-26).
- **The new label, example text and tooltip of the typed classes field ("ADetailer detector CLASSES (YOLO-World / YOLOE)") stay in English on a translated interface** until the Language Diffusion dictionaries have them.
- **Only the file name tells a YOLOE detector:** a renamed one without "yoloe" in its name is an ordinary detector.
- **With the transparency option on, a JPEG (and, on AUTOMATIC1111, a WebP) saved to `ADetailer-Inpaint`, or by an img2img job with Skip img2img, has no transparency:** set the WebUI's "File format for images" (on Forge Neo: "Image Format") to png to keep it there.
- **With Skip img2img, the transparency is not kept everywhere:** on Forge Neo, in the Batch tab of img2img, a file whose sides are not multiples of the WebUI's "Resolution Step" setting (64 by default) comes out without it, because the WebUI resizes the file before ADetailer sees it; on AUTOMATIC1111 the grid of several results has none; and in the Inpaint tabs of img2img, Skip img2img switches ADetailer off, as before. The Skip img2img part of the option is tested live on AUTOMATIC1111 only, with an API request: its Batch tab and Forge Neo are not tested live yet.
- **With several images in one generation, the WebUI's own grid of the batch carries the "ADetailer resolved prompt" of the last image**, and so does, on Forge Neo with a batch size above 1, the `params.txt` record that the paste button reads. Each image has its own values, and pasting ignores these keys. With a batch size above 1, the copy that the WebUI can save before face restoration carries, from the second image of a batch on, the keys of the image before it; the copies saved before hires fix do not carry them.
- **Styles applied once: three rare cases remain.** With a blank or `[PROMPT]` ADetailer prompt, the selected Styles still reach the detailer pass twice with a Style that itself holds a wildcard or a `{a|b}` variant, with a Style that has `{prompt}` in it twice, and when an extension adds text after the styled prompt, as the Magic Prompt of Dynamic Prompts does. Write the tab's own prompt instead: it gets the styles once.
- **The model family warning comes after the tab's first region**, once the tab's checkpoint is loaded.
- **In an X/Y/Z plot, the text under the results shows only the notes of the first cell:** the WebUI keeps only those, so a tab skipped for its checkpoint in a later cell is said in the console only.
- **On Forge Neo, a tab's checkpoint that does not load still makes the whole generation fail in the cases that the new skip does not recognise**, and its picture is not saved: the WebUI is left without a model and stops with an error such as `'FakeInitialModel' object has no attribute 'use_distilled_cfg_scale'`. It is not new in this beta: the earlier versions behave the same, also when a tab with an SDXL checkpoint keeps the VAE and text encoders of a Flux main model. This beta now skips the tab, and saves the picture, when it can tell for sure from the `.safetensors` files that the VAE file of the pass does not fit the checkpoint (see [Added in plus.10 beta 2](#added-in-plus10-beta-2)). It can still happen in the cases left alone: a `.ckpt` or GGUF checkpoint or VAE, a checkpoint in the tab that is not recognised as SD 1.5, SDXL or Flux, a main model whose VAE has 32 latent channels (Flux.2 and similar) while a tab with an SD 1.5 or SDXL checkpoint keeps its VAE file, a VAE of another layout or saved with a checkpoint's `first_stage_model.` prefix among the files, the hires fix checkpoint, a tab without its own checkpoint whose "Use separate VAE" file does not fit when the main model is a GGUF file or uses a GGUF text encoder, a tab whose own "Use separate VAE" file fits its checkpoint while a VAE file of the main model with another number of latent channels stays among the files, outside a folder named VAE (no choice of the tab takes it out, and Forge Neo reads the files in the order of their paths, so it can be read after the tab's: move it into a folder named VAE and pick it again in the WebUI's "VAE / Text Encoder" list), and text encoder files that do not fit, which are not checked. A tab with a Flux checkpoint whose file holds neither a VAE nor text encoders still fails too when its pass has no Flux VAE file or no Flux text encoders (with an SD 1.5 or SDXL main model that has no VAE file selected, there is no file to check), also with a Flux VAE file in "Use separate VAE": a tab can pick only one text encoder file, so use a Flux checkpoint that holds its text encoders. For the other cases, the console line of the warning says what to set in that tab: for an SD 1.5 or SDXL checkpoint, "Use separate text encoder" with "None (use detailer checkpoint's own)" and "Use separate VAE" with Automatic; for a Flux one, a Flux VAE file in "Use separate VAE". "Run ADetailer on an image" is not affected: its status says that the run failed. Once the tab has files that fit, the next generation works again without a restart. The failure was seen in the live tests of this beta on Forge Neo, before the skip was added. The cases that the skip recognises no longer fail the generation: two of them were tested live on Forge Neo 2.32 (a tab with an SDXL checkpoint that kept the VAE and text encoder files of a Flux main model, and Flux's VAE file chosen in the "Use separate VAE" of a tab with an SDXL checkpoint), and each time the tab was skipped and the picture was saved. The cases left alone were not tested live, and nothing is promised for them. Forge was not tested live, and there the WebUI may need a restart.
- **The guide "Guide: several subjects, one LoRA each" is tested live in part:** on AUTOMATIC1111 1.10.1 and, with fewer checks, on Forge Neo 2.32. LoRA aliases, `lyco` tags, a typed LoRA name that the WebUI does not list and a translated interface are covered by the offline tests only. How to use it: [Several subjects, one LoRA each (guide)](#several-subjects-one-lora-each-guide).
- **The built-in face detectors can miss the face of an animal-like character:** in the live tests a built-in YOLO face detector did not find the muzzle of an animal character that stands like a person, so "Face only" and the 2nd tab of "Whole person, then face (2 tabs)" can leave such a face as it is. Pick "Whole person (outline)" for those characters. It is a limit of the detector, not of the guide.
- **The guide does not change the Settings.** The parts go to the people from left to right only with Settings → ADetailer → "Sort bounding boxes by" on "Position (left to right)" and "More sort orders (overrides Sort bounding boxes by)" on Off. On a fresh setup "Sort bounding boxes by" is on None: the status under the button then warns and says what to set: set it once and click **Apply settings**. With one subject there is no order to follow and no warning.
- **The guide's subjects are the biggest people, counted from left to right:** the tab keeps as many of the biggest detections as there are subjects, by the area of their box, and gives them the parts by the left edge of each box. A person the detector does not find, or a figure with a bigger box than a subject, moves the parts to other people.
- **With "Whole person, then face (2 tabs)", no setting ties a face to its person:** they are matched only by the order of their boxes, because the 2nd tab counts on its own. It finds the faces by itself, keeps as many of the biggest faces as there are subjects and gives them the same parts from left to right. They are usually the faces of the people of the 1st tab, but a face may not get the part of its own person when another person in the picture has a bigger face, when people overlap or hug, or when a face is not found; the status says so every time. With a sort order other than "Position (left to right)", the 2nd tab sorts the faces with that same setting, and the order of the faces may not be the order of the people: a face can then get another person's part.
- **In the Inpaint tabs of img2img with "Inpaint area" on "Whole picture", the guide's order can fail:** after the tab has kept the biggest people and put them in order, ADetailer leaves out the ones that do not touch your inpaint mask. A person left out does not use up a part: the people left take the parts from the first one on, so Subject 1 is the leftmost of the people left, and a person can get the part written for another. The guide's status does not mention this.
- **A LoRA of the guide that is also in the main prompt acts on the whole picture**, and with "Use LoRAs from main prompt" ticked in the tab it is added to the parts of the other subjects too. The status warns; the guide changes neither the main prompt nor that checkbox.
- **The guide takes at most 4 subjects and fills only the 1st tab** (with "Whole person, then face (2 tabs)", the 2nd too). "✨ Fill the tabs" replaces the prompt those tabs had and sets the options listed in its section, with no undo: save the tab as a preset first to keep its old setup.
- **A LoRA name the WebUI does not list is written anyway**, as typed, and the status warns with the number of its row: with a wrong name that subject is redrawn without a LoRA.
- **The guide's LoRA menus do not follow the disk:** they get the WebUI's current LoRA list when you press "✨ Fill the tabs", and the guide never scans the folders. A second browser window that was already open keeps its old list until that page is reloaded.
- **The guide's choices are not saved:** presets, remembered settings, the WebUI's Settings → Defaults and the image parameters do not hold them, and an API request cannot use the guide. Only what the guide wrote in the tabs is saved, as for any tab.
- **The texts of the guide stay in English on a translated interface** until the Language Diffusion dictionaries have them; its status is made when you press the button and is in English.
- **Going back to beta 1 or plus.9 after using the guide:** the box is not there, and a tab it filled works as it is, because the guide writes only tab settings that those versions have.
- The notes on beta 1 below still apply.

### New in plus.10 beta 1

🟢 available in the stable release · 🟡 in testing in a beta (plus.10 beta 1)

> [!NOTE]
> **plus.10 beta 1 in short:**
>
> - 🟡 **Masked content in every tab:** choose what fills the region before it is redrawn, as in the WebUI's own inpaint; "latent noise" and "latent nothing" redraw it from scratch.
> - 🟡 **1 megapixel inpaint size:** a new choice in Settings redraws each region at about 1024×1024 with the shape of its box, for SDXL, Flux and newer models.
> - 🟡 **Anytest ControlNet models listed:** they run with no preprocessor; the `anima-lllite` ones need an Anima checkpoint.
> - 🟡 **Wildcards keep the main prompt's value:** a wildcard in a tab's prompt that the main prompt also holds takes the value drawn there, so the eye colour stays the same.
> - 🟡 **Share of the image on the Detection preview:** each label ends with the share of the image its box takes, such as "#1 face 12.3%", the value to set on the "Mask min/max area ratio" sliders (12.3% = 0.123).
> - 🟡 **Console lines while installing and downloading:** a long first start no longer looks stuck.
> - 🟡 **Detectors with the same file name all listed:** the second one no longer disappears from the menu.
> - 🟡 **New built-in detectors saved in `models/adetailer`:** a built-in detector downloaded for the first time is a normal file there; the ones already in the Hugging Face cache stay where they are.
> - 🟡 **Changed: Dynamic Prompts skips regions without wildcards:** it no longer runs on the regions whose prompt holds no wildcard or variant, so its Magic Prompt no longer rewrites them.
> - 🟡 **Soft Inpainting fixed:** ADetailer no longer turns the main image black or changes it outside the mask.
> - 🟡 **ADetailer's ControlNet on Forge, Forge Neo and reForge works on each region**, as on AUTOMATIC1111.
> - 🟡 **MediaPipe install warning shown:** when MediaPipe cannot be installed, the console says so.
> - 🟡 **No Hugging Face token to hf-mirror.com:** also when you set `HF_ENDPOINT` to that mirror yourself.
> - 🟡 **MediaPipe download error without the full folder path:** printed once, then a short reminder per generation.

The details are in the v26.2.0+plus.10.beta.1 section of [CHANGELOG.md](CHANGELOG.md).

#### Added in plus.10 beta 1

- 🟡 **Masked content** — a new option in every tab, under "Dynamic denoise by area": "original" (the default, what ADetailer always did) keeps the region as it is, "fill" paints it with the colours around it, and "latent noise" and "latent nothing" redraw it from scratch and need a denoising strength of about 0.8 or more. It is written into the image parameters ("ADetailer inpaint masked content") only when it is not "original", pastes back, is saved in presets and remembered between restarts, can be sent through the API as `ad_inpaint_masked_content` (the four names in any letter case, or the WebUI's `inpainting_fill` number 0 to 3), and has an X/Y/Z axis for the 1st tab, "[ADetailer] Masked content 1st", added after all the other ADetailer axes. It uses the same names as the [IOSakaki/adetailer](https://github.com/IOSakaki/adetailer) fork, the idea's source, so pictures made with that fork paste back too. Asked upstream in [Bing-su/adetailer#516](https://github.com/Bing-su/adetailer/issues/516) and [#802](https://github.com/Bing-su/adetailer/issues/802). Details: [Masked content](#masked-content).
- 🟡 **1 megapixel inpaint size** — Settings → ADetailer → "Try to match inpainting size to bounding box size" has a fourth choice, "1 megapixel (SDXL, Flux and newer)": each region is inpainted at about 1024×1024 pixels with the shape of its box (kept between 1:4 and 4:1), both sides in multiples of 64, for any model. With SD1.5 models it takes about four times the time and video memory. As with Strict and Free, it works only with "Inpaint only masked"; a tab's "Use separate width/height" or "Scale inpaint to bbox" takes its place, and "Max inpaint size" still caps it. When it applies, the image parameters say "ADetailer inpaint size match: 1 MP". Off stays the default. Idea from the [IOSakaki/adetailer](https://github.com/IOSakaki/adetailer) and [Haoming02/ADetailer-Neo](https://github.com/Haoming02/ADetailer-Neo) forks.
- 🟡 **Anytest ControlNet models listed** — a ControlNet model whose file name contains "anytest", "any-test" or "any_test" (any letter case), such as `anima-lllite-any-test-like-v2`, is now in the ADetailer ControlNet model menu with no preprocessor ("None" on Forge, "none" on AUTOMATIC1111), the only one the tab offers for it, so the crop goes to ControlNet as it is. A name that also holds another type word, such as "anytest_depth", keeps that type. On an X/Y/Z "[ADetailer] ControlNet model" axis, an Anytest cell keeps the tab's preprocessor unless the tab's is "None", as for every model type. The other models are unchanged. The `anima-lllite` models work only with Anima checkpoints, on Forge Neo. Idea from the [IOSakaki/adetailer](https://github.com/IOSakaki/adetailer) fork.
- 🟡 **Wildcards keep the main prompt's value** — a `__name__` wildcard of Dynamic Prompts or of the "wildcards" extension, in a tab's prompt, negative prompt or "Prompt append" boxes, that the main prompt (or, for the negative prompt, the main negative prompt) also holds now takes the value drawn there for each image, instead of a new draw. For example, with `1girl, __eyecolor__ eyes` as the main prompt, drawn as `green eyes`, and `close-up, __eyecolor__ eyes` in the tab, the detailer prompt is `close-up, green eyes`. A wildcard that is not in the main prompt is drawn as before, so use one of those to get a different value on purpose. Only `__name__` wildcards are carried over, not Dynamic Prompts' `{a|b}` choices or `${name}` variables. When the value cannot be read for sure (the same wildcard drawn twice with different values, two wildcards with nothing between them, more than one possible reading, or a main prompt that another extension rewrote), the tab's prompt is left as before, and "Verbose diagnostic log" says why. A value that holds a LoRA is not loaded twice with "Use LoRAs from main prompt", and "Strip LoRAs from the detailer prompt" removes it too. The regions that no `[CLASS=…]` block matches get the values as well. A very long main prompt that cannot be compared quickly is given up on at once, and the tab's wildcards are drawn again: this takes something like many wildcards with only spaces between them in a prompt of more than about 800 words, while a prompt of 500 words that starts with six wildcards is read. The comparison is remembered, so the other tabs and boxes of the same image do not repeat it. Its limits are in [Notes on plus.10 beta 1](#notes-on-plus10-beta-1). Always on, no option. Asked upstream in [Bing-su/adetailer#764](https://github.com/Bing-su/adetailer/issues/764), and in [adieyal/sd-dynamic-prompts#561](https://github.com/adieyal/sd-dynamic-prompts/issues/561).
- 🟡 **Share of the image on the Detection preview** — each label of the Detection preview ends with the share of the image that its box takes: "#1 face 12.3%" on the single-tab preview, "#1 1:face 0.90 12.3%" with "🔁 Combine all tabs". It is the value that "Mask min area ratio" and "Mask max area ratio" compare, written as a percentage: 12.3% is 0.123, and 0.4% is 0.004. It always has one decimal, which is one step of the sliders, and it is cut down, never rounded up, so the value read on a box keeps that box when set as the minimum; as the maximum, set one step more (11.6% is 0.117). It reads "100%" for a box as large as the image and "<0.1%" for a box smaller than one step. Unticking "Show labels" hides it with the rest of the label; the previews in the results gallery and the files of "Save mask previews" do not change. Always on, no option. Tested live on AUTOMATIC1111 and Forge Neo. Asked by users on Reddit. Details: [Detection preview](#detection-preview).
- 🟡 **Console lines while installing and downloading** — when ADetailer has to install packages with pip, or to download a built-in YOLO detector (from Hugging Face or its hf-mirror.com mirror) or a MediaPipe model file, the console prints a line before and one after, with the time taken and, for a download, the size of the file. The line before a download reads "trying to download …", and the line after it comes only when the file has arrived. When a MediaPipe model file cannot be downloaded, its "trying to download" line and the error line after it are printed once per session and model, not for every image. Nothing is printed when everything is already there, with `--ad-no-huggingface`, with the 🔄 button or in Hugging Face's offline mode.
- 🟡 **Detectors with the same file name in different folders all listed** — Civitai often saves different detectors under the same file name, and the second one disappeared from the "ADetailer detector" menu without a word. The first keeps its plain name; each other one is listed as "name.pt (folder)", and the console names both files by model folder and path inside it, not by the full path, and repeats the line only when those files change. When two model folders end with the same name, the line adds their parent folders, one at a time and three at most, or puts the drive or network share after the text ("adetailer/hand.pt (on D:)"), only until the files can be told apart. It never shows the name of your home folder: that folder is written as "~", and a parent folder or network share with the same name is left out, even if two folders then read the same. The folders are read in a fixed order (the files at the top of a folder first, then its subfolders from A to Z, ignoring letter case), so the same file keeps the plain name on every start and every WebUI; identical copies still count as one detector. A link with its own file name to a detector listed elsewhere stays a separate menu entry, as in plus.9.
- 🟡 **New built-in detectors saved in `models/adetailer`** — a built-in detector that is in none of the model folders and not in the Hugging Face cache is now saved as a normal file in `models/adetailer`, next to your own detectors, and the console line of its download ends with "saved in models/adetailer". The YOLO-World model `yolov8x-worldv2.pt` is saved there too. Detectors already in the Hugging Face cache stay there and are used from there: nothing is copied and nothing is downloaded again. From the next start, the saved file is read from the folder like a detector you put there yourself, in its usual place in the menu, and plus.9 reads it from there too. This needs huggingface_hub 0.23 or newer and a `models/adetailer` folder that can be written; without them, or when saving there fails, the detector goes to the Hugging Face cache as before. The Hugging Face library keeps a small `.cache` subfolder in `models/adetailer` with the state of its downloads: it holds no model, and nothing in it is listed in the menu. Tested live with `hand_yolov8n.pt`, which was saved there, listed once in the menu and used on AUTOMATIC1111 and Forge Neo. Related upstream reports about the downloads from Hugging Face: [Bing-su/adetailer#153](https://github.com/Bing-su/adetailer/issues/153), [#163](https://github.com/Bing-su/adetailer/issues/163) and [#619](https://github.com/Bing-su/adetailer/issues/619). Idea from the [abzaloff/aadetailer-neoforge](https://github.com/abzaloff/aadetailer-neoforge) fork. Details: [Built-in detectors](#built-in-detectors).
- 🟡 **Guides** — three recipes with options that already exist: [Same face from a reference picture](#same-face-from-a-reference-picture) (an IP-Adapter unit in the WebUI's ControlNet with "Passthrough" in the tab), [Many pictures, each with its own prompt](#many-pictures-each-with-its-own-prompt) (the img2img Batch tab with PNG info and "Skip img2img", since "Run ADetailer on an image" and its folder batch start with an empty prompt), and the commands that reinstall MediaPipe when the `mediapipe_*` detectors cannot load it (see [Install and startup](#install-and-startup)). The first two are in the Guide tab too.

#### Changed in plus.10 beta 1

- 🟡 **Dynamic Prompts no longer runs on the regions whose prompt holds no wildcard or variant** — this changes what the detailer pass does by default for anyone who uses the Dynamic Prompts extension with ADetailer, on purpose, and adds no option. The detailer pass ran the Dynamic Prompts script on every region, so with its "Magic prompt", "I'm feeling lucky" or "Attention grabber" on, the prompt of each region was rewritten although it had nothing to draw. A region is now inpainted without that script when nothing is there for it: no `{a|b}` variant, no `__name__` wildcard and no Dynamic Prompts comment in the region's final prompt and negative prompt, with the selected Styles. The final prompt is the text the region is inpainted with: its `[SEP]` part of the tab's prompt, the "Prompt append" text, and the `[CLASS=…]` block or per-class prompt of its class. So a wildcard in only one `[SEP]` part or in one class's block runs the script only on those regions; a wildcard of the tab that took its value from the main prompt (see "Wildcards keep the main prompt's value" above) does not run it, and neither does a comment that ADetailer removes. A tab with an empty prompt uses the main prompt as it was already drawn, so its regions run without the script too. A region whose final prompt still holds a wildcard or a variant runs Dynamic Prompts, as before: to get the old behaviour in a tab, put a variant, or a wildcard that the main prompt does not hold, in its "Prompt append" box, which reaches every region. The check errs on the side of running the script: `${name}` variables, Jinja2 blocks and the variant and wildcard markers changed in Settings → Dynamic Prompts count too, and whenever ADetailer is not sure, the script runs as before. The other scripts of "Script names to apply to ADetailer" are not touched. With "Verbose diagnostic log" on, the console says for each region when Dynamic Prompts was not run. Tested live on AUTOMATIC1111 with "Attention grabber" (not with "Magic prompt" and "I'm feeling lucky", which take the same path); at the time of writing, the Dynamic Prompts extension did not load on Forge Neo in our test (an error of its own at startup). Its limits are in [Notes on plus.10 beta 1](#notes-on-plus10-beta-1). Reported in [adieyal/sd-dynamic-prompts#669](https://github.com/adieyal/sd-dynamic-prompts/issues/669) and [#711](https://github.com/adieyal/sd-dynamic-prompts/issues/711).

#### Fixed in plus.10 beta 1

- 🟡 In img2img inpainting with Soft inpainting on, ADetailer no longer spoils the main image. With "Inpaint area" on "Only masked" (and with Ultimate SD Upscale) the result came out the size of the crop and mostly black; with "Whole picture" the area outside the mask lost its original pixels; and a batch of 2 or more printed an error from `soft_inpainting.py`. The detailer pass shares the Soft Inpainting script with the main image and left its own overlay and mask there; the main image's are now put back after the pass. Reported upstream in [Bing-su/adetailer#694](https://github.com/Bing-su/adetailer/issues/694) and [AUTOMATIC1111/stable-diffusion-webui#16055](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/16055).
- 🟡 On Forge, Forge Neo and reForge, the ADetailer ControlNet reads each region's own picture and mask, as on AUTOMATIC1111: it acts only inside the region's mask (an inpaint preprocessor no longer redraws the whole crop), and from the second region on it sees the regions already detailed. Before, it got the picture as it was when the tab started, with an all-white mask. This also fixes a ControlNet error on an almost black picture on classic Forge and Forge Neo.
- 🟡 On Forge, Forge Neo and reForge, the ADetailer ControlNet preprocessor resolution follows each region's crop and canvas (Pixel Perfect), as on AUTOMATIC1111. Before, one resolution was worked out from the whole picture, before each region's size was set.
- 🟡 On Forge Neo, a folder typed in ControlNet's batch input directory or in the img2img Batch tab's input directory no longer replaces the ADetailer ControlNet image: the detailer pass used a file of that folder as its control image, so a region got the control map of another picture. The folder is ignored only while the ADetailer ControlNet reads a region's image and stays set for the main ControlNet; if you empty or change the box during a pass, your new value is kept. Idea from the [IOSakaki/adetailer](https://github.com/IOSakaki/adetailer) fork.
- 🟡 When MediaPipe could not be installed, the console warning that the `mediapipe_*` detectors will not work was never shown, because the WebUI drops the error output of the extension's install step when that step ends normally. It is now printed where the WebUI shows it, after the install step ends. Earlier releases had the same problem.
- 🟡 The console line for a MediaPipe model file that could not be downloaded no longer shows the full path of the model folder, which holds the system user name: it names the folder from the extension's folder down, and is printed once per session and model instead of once per image. While the model is missing, each later generation prints one short reminder per model ("… is still missing (see the message above) …"), so a detector without its model does not look like a picture where nothing was detected. The detection preview prints that reminder at most once a minute. An incomplete model file put there by hand is still replaced by a new download, and the console says so. If that incomplete file cannot be removed (read-only, or in use by another program), the console says so once and asks you to remove or replace it by hand, and no download is tried meanwhile. plus.9 and earlier releases had the same problem.
- 🟡 With the `HF_ENDPOINT` variable set to hf-mirror.com (also written with a final dot, or as a subdomain such as www.hf-mirror.com), ADetailer's request for a built-in detector no longer sends your Hugging Face token to the mirror, which is not run by Hugging Face, and a download that fails there is no longer tried a second time on the same host. plus.9 and earlier releases sent the token. Requests to Hugging Face itself are unchanged, and another mirror host set with `HF_ENDPOINT` still gets the token from the Hugging Face library.

The three ControlNet fixes change the result of a tab with a ControlNet model on Forge, Forge Neo and reForge, on purpose. AUTOMATIC1111 is unchanged.

### Notes on plus.10 beta 1

- **X/Y/Z plot requests through the API:** a request picks an X/Y/Z axis by its position in the list. The new "[ADetailer] Masked content 1st" axis is added after all the other ADetailer axes, so every ADetailer axis that existed before keeps its position. Only the axes of extensions loaded after ADetailer move one place down; as before, their position also changes with "Max tabs". Look up the position from the axis name instead of keeping a fixed number. The X/Y/Z plot in the browser is not affected.
- **Going back to plus.9:** presets, remembered settings and pasted parameters load as before, and "Masked content" is ignored there, so every tab runs as "original". An API request that sends `ad_inpaint_masked_content` to plus.9 skips that tab with a ValidationError, as happens with any per-tab option a release does not know.
- **The pip lines appear late on AUTOMATIC1111 and Forge Neo:** the WebUI shows the output of the extension's install step only once that step ends, so the lines before and after a pip install appear together, afterwards. The lines for the downloads of the detectors and of the MediaPipe model files appear on time.
- **Without a network connection, the "trying to download" lines come back at every start** for each built-in detector that is not downloaded yet. Start the WebUI with `--ad-no-huggingface` or in Hugging Face's offline mode to stop them.
- **Wildcards that do not take the main prompt's value:** text that another extension adds after the main prompt (for example Dynamic Prompts' Magic Prompt) becomes part of the value of a wildcard at the very end of the main prompt. Wildcards inside a Style are drawn again in the detailer pass. Wildcards with parameters, such as `__name(x=1)__`, are not recognised and are drawn again. A main prompt whose spaces Dynamic Prompts changes may not match, and then the wildcards are drawn again ("Verbose diagnostic log" says so). So are the wildcards of a tab when the main prompt is too long to be compared quickly (something like many wildcards with only spaces between them in a prompt of more than about 800 words).
- **A region whose prompt holds a wildcard or a variant is still rewritten by Dynamic Prompts' Magic Prompt:** the Dynamic Prompts script runs on such a region, as in plus.9, with "Magic prompt", "I'm feeling lucky" and "Attention grabber" if they are on: a region whose final prompt holds a variant, a Dynamic Prompts comment, or a wildcard that did not take its value from the main prompt (the main prompt does not hold it, or the value could not be read). Switch them off in Dynamic Prompts, or keep such text out of that region's prompt.
- **With Dynamic Prompts' "Combinatorial generation" on, a tab with variants inpaints each region once for every combination.** This was read in the code and not tried.
- **A mirror other than hf-mirror.com set with `HF_ENDPOINT` still gets your Hugging Face token:** only hf-mirror.com and its subdomains are known to ADetailer; a request to any other host you put in `HF_ENDPOINT` is sent by the Hugging Face library with your token, as it would be to Hugging Face. Log out of Hugging Face (or unset `HF_TOKEN`) if you do not want that host to receive it.
- **On Forge Neo, with the tab's ControlNet model on "Passthrough", your own ControlNet units still read the batch input directory during the ADetailer pass:** region 1 gets the first file, region 2 the second, and so on. Empty the directory box, or pick a ControlNet model in the ADetailer tab, to avoid it. The main image is not affected. plus.9 does the same.
- **On Forge Neo, with the same detector file name in two subfolders of the same folder, the plain name now goes to the other file**, the one whose subfolder comes first from A to Z. A tab, a preset or a picture that names that detector by its plain name now uses that file; pick the other one, listed as "name.pt (folder)", if you want it.
- **"Masked content" stays in English on a translated interface**, with its help text, until the Language Diffusion dictionaries have it.
- The known issues of plus.9 still apply: see [Known issues](#known-issues).

## What is new in plus.9

**v26.2.0+plus.9**, released on 2026-11-09, is the stable release of plus.9. It adds clickable boxes to the Detection preview, X/Y/Z axes for every tab and an Enable axis, a `[PROMPT=words]` token that brings only some parts of the main prompt, an `iris` class for the face parts detector, a 🔄 button next to the detector menu that updates the list when a detector is added or removed while the WebUI runs (and finds such a detector also when Generate or pasted parameters name it), a "Show labels" switch for the Detection preview, tooltips on every option, swapping and pasting tabs, higher limits for the mask sliders and new settings for the colors, the opacity, the upscaler and the maximum size of the inpainted regions, for duplicate detections, more sort orders, inpaint masks, the detection boxes in the live preview or in the results gallery, and starting ADetailer switched on. It loads built-in detectors from your disk before contacting Hugging Face, runs the detector on the CPU when the whole WebUI runs there, no longer sends Ultralytics' usage analytics, answers more frequent questions in the Guide tab and below, and fixes problems with detectors, MediaPipe, reForge, ControlNet, prompts, Export, API requests and saved and pasted parameters, among them ten known issues of plus.8. It was tested in three betas, each listed below with what it added: beta 1 (v26.2.0+plus.9.beta.1, 2026-10-22), beta 2 (v26.2.0+plus.9.beta.2, 2026-10-24) and beta 3 (v26.2.0+plus.9.beta.3, 2026-10-29). The stable release is beta 3 with the fixes listed under "Fixed since beta 3" in the v26.2.0+plus.9 section of [CHANGELOG.md](CHANGELOG.md): on Forge, Forge Neo and reForge, with a ControlNet model in the tab and the WebUI's "Overlay original for inpaint" setting (on Forge Neo: "For inpainting, overlay the resulting image back onto the original image") off, "Inpaint only masked" again pastes each region back, where in the three betas the region's crop replaced the whole picture; loading a preset finds a detector added while the WebUI runs, as Generate does, and Load and Paste settings list it in the tab's detector menu; with Batch count 2 or more on AUTOMATIC1111 and reForge, the later batches keep the main prompt's embeddings in "TI hashes", and on Forge and Forge Neo they no longer get the "TI" and "Emphasis" of the detailer pass; the download button of the Detection preview is no longer hidden by the picture without labels or blocked by a box in its corner; the console note for a prompt that `[PROMPT=words]` left empty is said in two more cases; Settings → Reload UI also reads the class names of the detectors again, as the 🔄 does; the X/Y/Z sampler, scheduler and noise multiplier axes of the 1st tab change each cell also when its "Use separate sampler" or "Use separate noise multiplier" is not ticked; and Paste settings keeps the tab's detector, with a message, when the copied detector was removed since the Copy; the X/Y/Z "[ADetailer] ControlNet model 1st" axis runs each model with its default preprocessor on Forge, Forge Neo and reForge, as on AUTOMATIC1111, when the tab's ControlNet model is "None"; on Forge Neo, the picture of "PiD Integrated" reaches the results also with ADetailer on, with its own parameters; the console note for an emptied prompt shows a token with a `[` in it as typed, instead of stopping ADetailer for that image; on Forge, the comments typed in the ADetailer prompts are removed before the detailer pass, so they no longer take with them what ADetailer adds after them on the same line; "Or batch a whole folder" is refused on a WebUI started with `--hide-ui-dir-config`, as the WebUI's own folder batches are; an img2img API request with the `infotext` of a picture made without Skip img2img no longer skips the img2img pass; an image or an HTML tag in a preset is shown as text in its preview and in the status of Load, so picking a preset from someone else's library no longer makes your browser contact another server; on AUTOMATIC1111, a detector file made to ask for a huge amount of memory when its class names are read shows no classes instead; and ADetailer's own request to the hf-mirror.com mirror for a built-in detector goes without your Hugging Face token, which in the three betas and in plus.8 was sent to the mirror too (a mirror you set yourself with the `HF_ENDPOINT` variable still gets it, as Hugging Face would); on Forge and Forge Neo, the ControlNet model of a tab works in every pass, where from the second image, tab or X/Y/Z cell the region was inpainted without ControlNet; with "Max tabs" at 1 the 🔄 next to the detector menu works, where it left a wrong value in the menu and Generate then skipped the tab; and on AUTOMATIC1111 and Forge Neo a comment written in a Style no longer reaches the model in the detailer pass, and that Style is applied once. The previous stable release is v26.2.0+plus.8, called plus.8 below. In this section, "beta 1", "beta 2" and "beta 3" mean the betas of plus.9.

The details are in the v26.2.0+plus.9 section of [CHANGELOG.md](CHANGELOG.md) and in the sections of the three betas below it. All 2004 offline regression tests pass on AUTOMATIC1111 and Forge Neo, and the 🔄 button of beta 3 and the fixes since beta 3 that these two WebUIs can show were also tested live in the browser on both, except the one for PiD Integrated. Forge and reForge are supported and checked in the code, but not tested live. Please report any problem in the [Issues](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues).

**To update** from plus.8, follow [Update](#update). plus.9 adds no tab setting (only, in Settings → ADetailer, "Enable ADetailer at startup", "Save inpaint masks", "Match the colors of each inpainted region to the picture", "Detailed region opacity", "Drop overlapping detections", "Upscaler for ADetailer's inpaint area", "Hide detection boxes from the live preview", "Show detection previews in the results gallery", "More sort orders (overrides Sort bounding boxes by)" and "Max inpaint size, pixels per side (0 = no limit)"), so your presets and remembered settings work as before, and updating does not touch them (see [Your data](#your-data)). The notes of each beta below say what changes if you go back to plus.8.

### New in plus.9 beta 3

🟢 available in the stable release · 🟡 in testing in a beta (none right now)

#### New in plus.9 beta 3

- 🟢 **🔄 next to the detector menu** — reads the detector folders again (`models/adetailer` and the extra model folders of Settings → ADetailer), so a detector you add or remove while the WebUI runs shows up in, or leaves, the "ADetailer detector" menu of every tab, without a restart. It does not contact the network. A tab keeps its detector while it is still there; a tab whose detector was removed takes its first choice, as at startup. The menus on the same side (txt2img or img2img) change at once, the other side's after a page reload or with its own 🔄, and a page reload keeps the new list. A detector replaced by a new file of the same name uses the new file's classes after the 🔄 (choose another detector and back to update the CLASSES menu of a tab that shows it) and, from the stable release, after Settings → Reload UI. Asked in [issue #8](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues/8). Details: [Add your own detectors](#add-your-own-detectors). From the stable release it also works with "Max tabs" at 1: in beta 3 it left a wrong value in the menu there, until you chose a detector again or reloaded the page.
- 🟢 **A detector added while the WebUI runs is found** — Generate, an API request and pasted parameters (PNG Info, Send to) read the detector folders again when they name a detector that is not in the list, before they report it as missing. From the stable release, loading a preset does the same: in beta 3 Load said such a detector is not installed (click 🔄, then Load again). From the stable release, Load and Paste settings (also Swap and Paste into every tab) list such a detector in the tab's detector menu, also when Generate or another tab found it first; in beta 3 a second Load or a Paste on Forge and Forge Neo showed the menu empty, though Generate used the detector, and so did a detector removed with 🔄 and put back. Pasted parameters still show it empty there: see [Known issues](#known-issues).

#### Fixed in plus.9 beta 3

- 🟢 The console line "[-] ADetailer: applied 1st ad_prompt: …" is no longer printed when the prompt differs from the typed one only by the comments ADetailer removed (from beta 1): it shows only when something else changed the prompt, as in plus.8. On that line, a line break of the prompt is written `\n`; the model gets a real line break. Asked in [issue #7](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues/7).
- 🟢 An X/Y/Z "[ADetailer] Prompt S/R" image whose ADetailer prompt has comments around its `[SEP]` parts (such as `blue face # note`, then `[SEP] blue eyes` on the next line, or `blue face # note [SEP] blue eyes` with the `[SEP]` hidden by the comment) pastes back the same parts as the generation (from beta 1). The saved parameters split the prompt only at the `[SEP]` that the WebUI's own comment rules leave active, keep a hidden `[SEP]` inside its comment, and end the line before `[SEP]` after a part that ends inside a comment, in the prompt and in the negative prompt. The comments stay in the saved parameters, except in a part that, without them, would not give the generation's text (such as a `[PROMPT=words]` that matched nothing right after a comment): that part is saved as the generation used it. Prompts without comments are saved as before.
- 🟢 The console line about `.zip` files in the detector folders now says that a click on 🔄 next to the detector menu is enough after you extract the `.pt` file, instead of only asking you to restart the WebUI. It is printed again at every 🔄, and when a detector name that is not in the list makes ADetailer read the folders again, as a reminder while the `.zip` is still there.

### Notes on plus.9 beta 3

- **The 🔄 button does not download a built-in detector that is missing**: like a WebUI started offline, it lists only those already in the folders or in the Hugging Face cache. Restart the WebUI online to download one.
- **Settings → Reload UI still works** and reads the checkpoint, VAE and text encoder lists of the ADetailer tabs again too (from the stable release, also the class names of the detectors, as the 🔄 does; in beta 3 they changed only with the 🔄 or a restart); refreshing the browser page (F5) does not read any folder.
- **The 🔄 button's hover text stays in English on a translated interface** until the Language Diffusion dictionaries have it.
- **Going back to plus.8, beta 1 or beta 2:** they have no 🔄 button; after adding or removing a detector, use Settings → Reload UI or restart the WebUI.
- **An X/Y/Z "[ADetailer] Prompt S/R" value that adds a comment or a `[SEP]`**: a replacement that writes comment characters (`#`, and on Forge Neo also `//` or `/*`), such as `face` → `face #`, is plain text in the generation but a comment once the saved parameters are pasted back; one that writes `[SEP]`, such as `face` → `face [SEP] eyes`, is one part in the generation and two pasted back. Keep comment characters and `[SEP]` out of the replacement values.

### New in plus.9 beta 2

🟢 available in the stable release · 🟡 in testing in a beta (none right now)

> [!NOTE]
> **plus.9 beta 2 in short:**
>
> - 🟢 **Only some parts of the main prompt:** `[PROMPT=eyes]` in a tab's prompt brings only the parts of each image's main prompt that name the eyes, so the eye colour a wildcard picked reaches the detailer pass; `[PROMPT=-beach]` brings everything else.
> - 🟢 **Iris class:** `mediapipe_face_features` can detect the iris, one mask per eye, to change the eye colour without redrawing the eyelids.
> - 🟢 **Detection previews in the results gallery:** a new setting adds the picture with the detection boxes of every pass after the generated images, without saving it to disk.
> - 🟢 **Show labels:** a checkbox next to the Detection preview button hides the names, scores and numbers on the preview, leaving only the boxes and masks.
> - 🟢 **More sort orders:** a new setting details the detections from right to left, from top to bottom, from small to large or by confidence, and gives the parts of a `[SEP]` prompt in that order.
> - 🟢 **Max inpaint size:** a new setting limits the size a region is inpainted at, so "Scale inpaint to bbox" on a large detection does not run out of video memory.
> - 🟢 **Hide the detection boxes from the live preview:** a new setting keeps the picture with the boxes out of the live preview and the progress API.
> - 🟢 **Built-in detectors from your disk:** a built-in detector in `models/adetailer` or in the Hugging Face cache loads without contacting Hugging Face, so at startup the WebUI no longer tries to reach Hugging Face for them, and it starts fine offline.
> - 🟢 **Detector on the CPU with the whole WebUI:** `--use-cpu all`, `--cpu`, `--always-cpu` and, except on macOS, Forge Neo's `--novram` also run the detector on the CPU.
> - 🟢 **Empty model in an API request:** a tab sent with `"ad_model": ""` counts as off, like `"None"`, instead of failing.
> - 🟢 **Five new answers in the Guide tab:** the order of ADetailer and a face-swap extension, why the detail pass runs fewer steps, model file safety, a switch for "Save images before ADetailer" at the top of the page, and how to leave small or background faces alone.

The details are in the v26.2.0+plus.9.beta.2 section of [CHANGELOG.md](CHANGELOG.md).

#### New in plus.9 beta 2

- 🟢 **`[PROMPT=words]`** — puts only the parts of this image's main prompt (the parts between its commas, new lines, `BREAK` and `AND`) that name one of the words, with their weights: with `close-up, [PROMPT=eyes]` and the main prompt `1girl, green eyes, beach`, the detailer prompt is `close-up, green eyes`. `[PROMPT=-beach]` puts everything except the parts that name it. It works in batches, with wildcards and in the negative prompt too. Details: [Upstream options](#upstream-options).
- 🟢 **Iris class** — pick `iris` in the CLASSES list of `mediapipe_face_features` to get one mask per eye over the whole iris, cut to the opening of the eye (none for a closed or almost closed eye). It is never used when no class is picked. Details: [Face parts detector](#face-parts-detector-mediapipe_face_features).
- 🟢 **Detection previews in the results gallery** — Settings → ADetailer → "Show detection previews in the results gallery" (off by default) adds the picture with the detection boxes of every ADetailer pass after the generated images, each with the parameters of its image, so Send to, PNG Info and Save work on it; up to 32 per generation. They are not saved to disk, API responses do not include them, and they are left out while a script of the Script menu (X/Y/Z plot, SD upscale…) runs. On Forge Neo with "PiD Integrated" on, the PiD picture comes after them with its own parameters, from the stable release: in earlier releases it did not reach the results with ADetailer on, though its file was saved. Details: [Settings](#settings).
- 🟢 **Show labels** — a "Show labels" checkbox next to "🔍 Run detection preview" (on by default, remembered by the browser) hides the class names, scores and numbers drawn on the preview, for the single-tab preview, where the boxes can still be clicked, and for "🔁 Combine all tabs". Details: [Detection preview](#detection-preview).
- 🟢 **More sort orders** — Settings → ADetailer → "More sort orders (overrides Sort bounding boxes by)": Off (the default), Position (right to left), Position (top to bottom), Area (small to large) or Confidence (high to low). Any choice but Off is used instead of "Sort bounding boxes by", for the order the regions are detailed in and for the parts of a `[SEP]` prompt; for example, with right to left the first part goes to the face on the right. Details: [Settings](#settings) and [A different prompt or LoRA for each face](#a-different-prompt-or-lora-for-each-face).
- 🟢 **Max inpaint size** — Settings → ADetailer → "Max inpaint size, pixels per side (0 = no limit)" (0 by default). With "Inpaint only masked" on and "Use separate width/height" off, a region whose inpaint size would have a side longer than the limit is inpainted smaller, with the same shape, and the console says so. When set, the limit is written into the image parameters ("ADetailer max inpaint size") if a tab it applies to is on. Details: [Scale inpaint to bbox](#scale-inpaint-to-bbox).
- 🟢 **Hide detection boxes from the live preview** — Settings → ADetailer → "Hide detection boxes from the live preview" (off by default) keeps the picture with the boxes ADetailer detected out of the WebUI's live preview and the progress API; the preview keeps its last picture until the inpainting steps replace it. The Detection preview button and "Save mask previews" are not affected. Asked for in [Bing-su/adetailer discussion #482](https://github.com/Bing-su/adetailer/discussions/482).
- 🟢 **Built-in detectors from your disk first** — a built-in detector such as `face_yolov8n.pt` in `models/adetailer` or an extra model folder is used from there, and one downloaded before loads from the Hugging Face cache, both without contacting Hugging Face; only a missing one is downloaded. Details: [Built-in detectors](#built-in-detectors).
- 🟢 **The detector on the CPU with the whole WebUI** — `--use-cpu all` (AUTOMATIC1111, Forge, reForge), `--cpu` (Forge Neo) and `--always-cpu` (Forge, reForge) run the detector, and the Detection preview, on the CPU, also on macOS; Forge Neo's `--novram` does too, except on macOS, like `--lowvram`.
- 🟢 **New answers in the Guide tab and the README** — [ADetailer and a face-swap extension run in the wrong order](#the-result-looks-wrong), [the detail pass runs fewer steps than you set](#the-result-looks-wrong), [small or background faces get detailed too](#the-result-looks-wrong), [is a detector file safe?](#is-a-detector-file-safe) and [a switch for "Save images before ADetailer"](#can-i-switch-save-images-before-adetailer-from-the-top-of-the-page).

#### Fixed in plus.9 beta 2

- 🟢 An API request that sends `"ad_model": ""` (or only spaces) for a tab no longer fails with "Model '' not found" and skips the tabs after it: that tab counts as off, like `"None"` (in plus.8 and beta 1, send `"None"`). See [Does it work with the WebUI's API?](#does-it-work-with-the-webuis-api).
- 🟢 The Guide tab says that "🔁 Combine all tabs" runs each detector at its own size, not at a fixed default.
- 🟢 ADetailer still loads on a WebUI whose `modules.devices` has no `NansException`: before, that missing name stopped the whole extension from loading. AUTOMATIC1111, Forge, Forge Neo and reForge all have it today, so nothing changes there.
- 🟢 With "Skip img2img", the "Verbose diagnostic log" header shows the width and height set in img2img instead of size=128x128.

### Notes on plus.9 beta 2

- **MediaPipe sends Google its own usage statistics.** Recent versions of the MediaPipe library, such as 0.10.35, the newest 0.10 version that ADetailer installs when MediaPipe is missing (on Python 3.13 the MediaPipe detectors need 0.10.30 or later), send Google statistics about how the library is used, to play.googleapis.com; Google says it never sends the pictures or other input. Unlike Ultralytics' analytics, this cannot be switched off from ADetailer: Google offers no setting for it ([google-ai-edge/mediapipe#6291](https://github.com/google-ai-edge/mediapipe/issues/6291)). To stop it, block play.googleapis.com in your firewall, or do not use the MediaPipe detectors. When the upload fails, the console can show a line such as "Failed to send to clearcut" from time to time; Google says a later MediaPipe release will stop printing it. Earlier releases of ADetailer Ultimate behave the same way with these MediaPipe versions.
- **A built-in detector already in the Hugging Face cache is no longer checked for a newer version.** If the Bingsu/adetailer page ever publishes an updated file under the same name, delete that detector from the Hugging Face cache (the `models--Bingsu--adetailer` folder, or `models--Bingsu--yolo-world-mirror` for the YOLO-World detector) and restart the WebUI online to download it again.
- **The four new options stay in English on a translated interface**, and so may the "Show labels" switch, until the Language Diffusion dictionaries have them.
- **A detection preview in the results gallery gives the first image's seed** to the WebUI's "Reuse seed" button, not the seed of its own image.
- **In an img2img batch from a folder, the detection previews count toward the WebUI's "Show the first N batch img2img results in UI" limit (on Forge Neo "Show the first N batch of img2img results in UI")**, so with "Show detection previews in the results gallery" on, fewer result images may be shown in the gallery (they are all saved as usual).
- **The download button of the Detection preview always saves the picture with the labels**, also while "Show labels" is off.
- **The face-swap tip works only when the other extension changes the image in the same step as ADetailer.** In which step ReActor and roop work was not checked: if moving the entry changes nothing, that extension works in a later step.
- **Going back to plus.8 or beta 1:** they ignore the four new options, so the order of "Sort bounding boxes by" applies, the inpaint size is not limited, the live preview shows the boxes and the results gallery shows no detection previews. The saved values are kept for when you come back. They do not know `[PROMPT=words]` either, so a prompt that holds it reaches the model as typed, and a `mediapipe_face_features` tab with only `iris` picked finds nothing there.
- The notes on beta 1 below still apply.

### New in plus.9 beta 1

🟢 available in the stable release · 🟡 in testing in a beta (none right now)

> [!NOTE]
> **plus.9 beta 1 in short:**
>
> - 🟢 **Click to choose detections:** click the numbered boxes of the Detection preview to choose which detections to inpaint, instead of typing their numbers.
> - 🟢 **Save inpaint masks:** a new option saves the black-and-white mask each region is inpainted with.
> - 🟢 **X/Y/Z axes for every tab:** the 2nd and later tabs get their own axes, the 1st tab gets three more, and an "[ADetailer] Enable" axis compares pictures with and without ADetailer.
> - 🟢 **Higher slider limits:** the mask offset, erosion / dilation, mask blur and padding sliders go further, for large pictures.
> - 🟢 **Clearer messages:** a detector made with a newer Ultralytics, or left in a `.zip` file, is explained in the console.
> - 🟢 **Works better with other tools:** the LoRAs and embeddings of the ADetailer prompts are listed in the image parameters for sites such as Civitai, reForge's Advanced Model Sampling stays on in the detailer pass, and MediaPipe no longer starts the audio system.
> - 🟢 **Extra files with their parameters:** the mask previews, step images and masks of "Run ADetailer on an image" record their parameters.
> - 🟢 **Match the colors of each region:** a new setting matches the colors of each inpainted region to the picture, so a face no longer comes out with another tint.
> - 🟢 **New guides:** how to give each face its own prompt or LoRA, and where to find more detectors.
> - 🟢 **Region opacity and upscaler:** two new settings blend each inpainted region with the original and pick the upscaler that enlarges it.
> - 🟢 **Duplicate detections dropped:** a new setting inpaints a face found twice only once.
> - 🟢 **Swap tabs or paste into every tab:** two checkboxes next to Paste settings.
> - 🟢 **Sturdier passes:** an error in one region no longer stops the others, and a short API request no longer runs an extra pass.
> - 🟢 **Tooltips on the options:** hover over an option of a tab to see what it does.
> - 🟢 **Enable ADetailer at startup:** a new setting starts the ADetailer panel already switched on in txt2img, img2img or both.
> - 🟢 **No Ultralytics usage analytics:** ADetailer's detections no longer send Ultralytics' anonymous usage analytics.
> - 🟢 **Known issues of plus.8 fixed:** Export downloads on AUTOMATIC1111, a detector without a class-name file lists its classes straight away there, ControlNet models get their default preprocessor on Forge, comments typed in the ADetailer prompts are removed, pasted parameters restore "Skip img2img" and name a missing sampler, and "Inpaint only masked" no longer changes the WebUI's "Overlay original for inpaint" setting (on Forge Neo: "For inpainting, overlay the resulting image back onto the original image").

The details are in the v26.2.0+plus.9.beta.1 section of [CHANGELOG.md](CHANGELOG.md).

#### New

- 🟢 **Click the boxes of the Detection preview** — after a single-tab preview, a click on a numbered box adds its number to "Inpaint only these detections", a second click takes it out, and the chosen boxes are marked in green; numbers typed by hand mark their boxes too. Not with "🔁 Combine all tabs". Asked for upstream in [Bing-su/adetailer#635](https://github.com/Bing-su/adetailer/issues/635). Details: [Detection preview](#detection-preview).
- 🟢 **Save inpaint masks** — Settings → ADetailer → "Save inpaint masks" (off by default) saves, for every region inpainted, the black-and-white mask img2img gets, as a PNG ending in `-ad-mask-N-K` (tab N, region K) in the `adetailer-steps` folder. Asked for upstream in [Bing-su/adetailer#406](https://github.com/Bing-su/adetailer/issues/406). Details: [Save inpaint masks](#save-inpaint-masks).
- 🟢 **X/Y/Z axes for every tab** — the 2nd and later tabs, up to "Max tabs", get axes for ADetailer model, ADetailer prompt, ADetailer negative prompt, Detection confidence threshold, Mask erosion / dilation, Inpaint mask blur, Inpaint denoising strength, Inpaint only masked padding, CFG scale and Steps, such as "[ADetailer] Inpaint denoising strength 2nd". The 1st tab gets "[ADetailer] Detection confidence threshold 1st", "[ADetailer] Inpaint mask blur 1st" and "[ADetailer] Steps 1st". A tab's axes apply when its "Enable this tab" is ticked. The axes that existed before keep their names. Asked for upstream in [Bing-su/adetailer#251](https://github.com/Bing-su/adetailer/issues/251).
- 🟢 **"[ADetailer] Enable" axis** — False or True switches ADetailer off or on for each cell of an X/Y/Z grid, whatever "Enable ADetailer" says, so one grid compares pictures with and without it. The tabs run as set in the panel.
- 🟢 **Higher limits for the mask sliders** — "Mask x(→) offset" and "Mask y(↑) offset" go from -512 to 512, "Mask erosion (-) / dilation (+)" from -256 to 256, "Inpaint mask blur" up to 256 and "Inpaint only masked padding, pixels" up to 512. The defaults are unchanged. Asked for upstream in [Bing-su/adetailer#111](https://github.com/Bing-su/adetailer/issues/111). Details: [Upstream options](#upstream-options).
- 🟢 **Match the colors of each inpainted region** — Settings → ADetailer → "Match the colors of each inpainted region to the picture" (off by default) matches the colors of each region, with a little of its surroundings, to the same area of the picture before the pass, with the WebUI's own color correction, and blends the edge with "Inpaint mask blur". It is the WebUI's "Apply color correction to img2img results" for ADetailer's regions only. On AUTOMATIC1111 the WebUI's correction fixes the tint and keeps the brightness of the inpainted region; on Forge Neo it matches the brightness too. Asked for upstream in [Bing-su/adetailer#731](https://github.com/Bing-su/adetailer/issues/731).
- 🟢 **Detailed region opacity** — Settings → ADetailer → "Detailed region opacity" (100 by default, as before) blends each inpainted region with the picture before it: 50 keeps half of the change. Asked for in [Bing-su/adetailer discussion #293](https://github.com/Bing-su/adetailer/discussions/293).
- 🟢 **Upscaler for the inpaint area** — Settings → ADetailer → "Upscaler for ADetailer's inpaint area" picks an installed upscaler, such as an ESRGAN one, to enlarge the area ADetailer inpaints, and with "Inpaint only masked" the result back when the crop is bigger than the inpaint size; the first choice keeps the WebUI's "Upscaler for img2img", as before. Asked for upstream in [Bing-su/adetailer#711](https://github.com/Bing-su/adetailer/issues/711).
- 🟢 **Drop overlapping detections** — Settings → ADetailer → "Drop overlapping detections (max overlap, 0 = off)" (off by default) inpaints only the more confident of two detections of the same class that overlap, so a face found twice is inpainted once; the much smaller eyes inside a face are kept. Asked for upstream in [Bing-su/adetailer#669](https://github.com/Bing-su/adetailer/issues/669).
- 🟢 **Swap tabs or paste into every tab** — tick "Swap with the copied tab" or "Paste into every tab" next to "📥 Paste settings" before you click it. Asked for upstream in [Bing-su/adetailer#297](https://github.com/Bing-su/adetailer/issues/297) and [#320](https://github.com/Bing-su/adetailer/issues/320).
- 🟢 **Tooltips on the options** — hovering over an option of a tab, such as "Detection model confidence threshold" or "Inpaint mask blur", shows a short explanation, as the fork's buttons already did. Every option without a help line under it gets one, in every tab; the layout of the panel is unchanged. Asked for upstream in [Bing-su/adetailer#23](https://github.com/Bing-su/adetailer/issues/23).
- 🟢 **Enable ADetailer at startup** — a new option in Settings → ADetailer (Off, txt2img, img2img or Both; Off by default; it needs a UI reload) starts the ADetailer panel already switched on in the chosen tabs. With Off, the WebUI's Settings → Defaults decide as before; any other choice wins over Defaults for the on or off state of those tabs. API requests are not affected, and "Remember last-used settings" does not store the on or off state. Asked for upstream in [lllyasviel/stable-diffusion-webui-forge#1833](https://github.com/lllyasviel/stable-diffusion-webui-forge/issues/1833). Details: [Settings](#settings).
- 🟢 **No Ultralytics usage analytics** — ADetailer switches off Ultralytics' anonymous usage analytics before its detections, in memory only. The shared Ultralytics settings file is never written, and in the same WebUI session the analytics stay off for other extensions that use Ultralytics. Asked for upstream in [Bing-su/adetailer#613](https://github.com/Bing-su/adetailer/issues/613).

#### Fixed

**Detectors and MediaPipe**

- 🟢 On AUTOMATIC1111, a detector without a `<model>.names.json` now lists its classes straight away, as on Forge and Forge Neo. The names are read from the detector file's metadata without loading it: nothing in the file is run, and AUTOMATIC1111's safe-unpickle check stays on. In plus.8 the list stays empty until the first detection (see [Known issues of plus.8](#known-issues-of-plus8)). From the stable release, a file made to ask for a huge amount of memory while its names are read shows no classes.
- 🟢 The MediaPipe safety check for macOS no longer takes a pre-release build such as 1.0.0rc1 for 1.0.1.

- 🟢 A detector made with a newer Ultralytics than the one in your WebUI, such as a recent YOLO12 or YOLO26 model, failed with only a message such as "Can't get attribute '…'". The console and the Detection preview status now say that the detector needs a newer Ultralytics, with the file name and the installed version (see [Nothing runs or nothing is detected](#nothing-runs-or-nothing-is-detected)).
- 🟢 A detector left in a `.zip` file, as many detectors on Civitai download, never showed up in the list and nothing said why. At startup, the console now names up to 5 `.zip` files in the detector folders and asks you to extract the `.pt` file (see [Install and startup](#install-and-startup)).
- 🟢 With the `sounddevice` package installed, the first MediaPipe detection started the audio system: on Windows, sound in other programs could stop, and Python was listed as using the microphone. ADetailer now loads MediaPipe without `sounddevice`, which detection does not need. Reported upstream in [Bing-su/adetailer#508](https://github.com/Bing-su/adetailer/issues/508).

**Saved parameters and X/Y/Z**

- 🟢 A LoRA or an embedding used only in an ADetailer prompt is now listed in the image's "Lora hashes" and "TI hashes", next to those of the main prompt and without doubles, so sites such as Civitai recognise it. Each image of a batch and each X/Y/Z cell lists only its own. A WebUI that writes no "TI hashes", such as Forge Neo, lists only the LoRAs. From the stable release, with Batch count 2 or more on AUTOMATIC1111 and reForge, the later batches also keep the main prompt's embeddings in "TI hashes"; in the betas, as in plus.8, they could lose them or list the previous batch's ADetailer embedding instead. On Forge and Forge Neo, the later batches no longer list in "TI" and "Emphasis" an embedding or weights used only in an ADetailer prompt.
- 🟢 An X/Y/Z "CFG scale" or "Steps" axis, the existing "[ADetailer] CFG scale 1st" included, also ticks the tab's "Use separate CFG scale" or "Use separate steps" in each cell. Before, without that box ticked, every cell came out the same (in plus.8, tick it in the tab before you start the grid). From the stable release, the "[ADetailer] ADetailer sampler 1st", "[ADetailer] ADetailer scheduler 1st" and "[ADetailer] noise multiplier 1st" axes also tick "Use separate sampler" or "Use separate noise multiplier", and the images record the axis's value; a sampler axis keeps the main generation's scheduler, and a scheduler axis its sampler, unless the tab already uses its own (in beta 1 to beta 3, tick the box in the tab before you start the grid).
- 🟢 With "Save mask previews", "Save intermediate step images" or "Save inpaint masks" on, the extra files of "Run ADetailer on an image" and folder runs record their parameters (the detailer pass, the loaded checkpoint and VAE, and the tab's ADetailer settings), and the console no longer prints "create_infotext failed" for each one.
- 🟢 A `[user]` or `[denoising]` token in the WebUI's file or folder name pattern no longer prints a traceback for each extra file and each result saved to `ADetailer-Inpaint` by "Run ADetailer on an image" and folder runs.
- 🟢 Pasted parameters (PNG Info, Send to, the paste button) and a restart now split a ticked separate sampler name the way Load and the WebUI do. A sampler this WebUI does not have, such as Res Multistep from Forge Neo on AUTOMATIC1111, becomes the first sampler, and the console says so, instead of the dropdown showing it blank (in plus.8, check "ADetailer sampler" and "ADetailer scheduler" after a restart or a paste).
- 🟢 img2img pictures now record "ADetailer skip img2img" (True or False), so PNG Info and Send to tick or untick the box as the picture was made. txt2img pictures and older images without the key leave it as it is (in plus.8, set it by hand). From the stable release, an API request with that picture's `infotext` reads the key as saved too: in the three betas, "False" could skip the img2img pass when ADetailer's arguments came in the short form.

**Prompts and ControlNet**

- 🟢 Comments typed in the ADetailer prompt, negative prompt and append boxes (`#`, and on Forge Neo also `//` and `/* */`) are now removed before the detailer pass, following the WebUI's own comment setting, also with "Apply only selected scripts to ADetailer" on. A box that holds only a comment counts as blank and takes the main prompt (in plus.8, leave comments out of those boxes).
- 🟢 On Forge, Forge Neo and reForge, choosing a ControlNet model now selects its default preprocessor, for example `openpose_full` or `depth_midas`, as AUTOMATIC1111 does, instead of "None". Picking "None" afterwards keeps it, and a saved "None" survives a restart; loading a preset or pasting parameters that change the model gives the model's default. An API request without `ad_controlnet_module` also uses the default (in plus.8, pick the preprocessor yourself).

**Inpainting**

- 🟢 "Inpaint only masked" no longer turns on the WebUI's "Overlay original for inpaint" setting (on Forge Neo: "For inpainting, overlay the resulting image back onto the original image") to paste each region back. It now gives the pass the WebUI's own overlay, so the setting is no longer saved as on when the WebUI saves its settings during the pass, and frozen settings no longer stop ADetailer. The pasted result is unchanged (in plus.8, keep that setting on, or untick "Inpaint only masked").

**Presets and interface**

- 🟢 On AUTOMATIC1111 and other WebUIs with Gradio 3, such as reForge's main branch, Export now downloads `adetailer-ultimate-presets.json`, as on Forge. The browser saves the file itself, and no new event handler is added (in plus.8, back up `user_presets.json` by hand).
- 🟢 The preset export button reads "📤 Export"; before, its English label had a typo. With Language Diffusion, the button stays in English until its dictionaries are updated for the new label.

**reForge**

- 🟢 With "Apply only selected scripts to ADetailer" on (the default), reForge's Advanced Model Sampling was left out of the detailer pass, so the detailed regions of a v-prediction or flow model came out broken. It now always runs there, like Soft Inpainting and HyperTile, and so does the reForge-RescaleCFG extension, whose script file has the same name as one of Advanced Model Sampling's scripts. Checked in reForge's code. With a ControlNet model in the tab, on Forge, Forge Neo and reForge, the detailer pass runs only ControlNet, so these scripts do not run in that tab's pass (see [Known issues](#known-issues)). Idea from [Anzhc/aadetailer-reforge](https://github.com/Anzhc/aadetailer-reforge), also proposed upstream in [Bing-su/adetailer#818](https://github.com/Bing-su/adetailer/pull/818). In plus.8, add `advanced_model_sampling_script` (and, where reForge has it, `advanced_model_sampling_script_backported`) to "Script names to apply to ADetailer".

### Notes on plus.9 beta 1

- **X/Y/Z plot requests through the API:** a request picks an X/Y/Z axis by its position in the list. The new ADetailer axes move the axes of the extensions loaded after ADetailer further down, and their position also changes with "Max tabs". Look up the position from the axis name before a request instead of keeping a fixed number. ADetailer's own axes that existed before keep their positions, and the X/Y/Z plot in the browser is not affected.
- **The message for a detector made with a newer Ultralytics can appear twice in the console:** once as its own line and once in the error report that follows it.
- **Going back to plus.8:** its mask sliders do not go as far. If a tab or a preset uses a value above plus.8's limits (for example a padding of 300), check those sliders after switching back.
- **MediaPipe on Linux aarch64:** MediaPipe 0.10.x has Linux aarch64 wheels only up to 0.10.18 (Python 3.9 to 3.12), so those systems get 0.10.18; on Python 3.13 there is no supported MediaPipe wheel for aarch64, and the MediaPipe detectors cannot be used there; the other packages are still installed, so the YOLO detectors work.
- **The export button in English on a translated interface:** with Language Diffusion, "📤 Export" stays in English until its dictionaries have the new label.
- **The tooltips of the options in English on a translated interface:** with Language Diffusion, they stay in English until its dictionaries have them, and so do the new checkboxes next to Paste settings.
- **"Detailed region opacity" with a high denoising strength** can show a faint double outline: test it on your own pictures.
- **"Upscaler for ADetailer's inpaint area" is set for the pass only:** a settings save during a pass stores it as "Upscaler for img2img", as for the separate CLIP skip, checkpoint and VAE.
- **A pasted ControlNet preprocessor "None" on Forge, Forge Neo and reForge:** when PNG Info, Send to, Load or Paste settings also change the ControlNet model, the model's default preprocessor is chosen instead of "None". Pick "None" again after pasting, if you want it.
- **"Enable ADetailer at startup" needs a UI reload** to apply, and it does not change API requests.
- The known issues of v26.2.0+plus.8 still apply, except those this beta fixes (Export on AUTOMATIC1111, the classes of a detector without a class-name file on AUTOMATIC1111, "Skip img2img" and samplers in pasted parameters, prompt comments, the ControlNet preprocessor "None" on Forge, and the two about "Overlay original for inpaint"): see [Known issues of plus.8](#known-issues-of-plus8).

### Known issues

Still open in v26.2.0+plus.9, with what to do meanwhile. The notes of each beta above ([beta 3](#notes-on-plus9-beta-3), [beta 2](#notes-on-plus9-beta-2) and [beta 1](#notes-on-plus9-beta-1)) still apply too. The first three come from plus.8 or earlier; the other known issues of plus.8 are fixed (see [Known issues of plus.8](#known-issues-of-plus8)); the last two are limits of plus.9 found after beta 3.

- **Newer options are in English on a translated interface.** The options added since plus.7, among them those of plus.9, and some reworded tooltips and help texts, stay in English until the Language Diffusion dictionaries are updated (see [Translations](#translations)). This includes the reworded help text of **🔄 Reset ADetailer settings to defaults** on the Settings page and the "📥 Paste from Nth tab" label that the other tabs show after a Copy.
- **A tab prompt made only of `[CLASS=…]` blocks gives the main prompt to the regions that no block matches** (from plus.8; plus.7.4 gave them an empty prompt). With `[CLASS=face] smiling [/CLASS]` and a detector that finds faces and hands, the hands are redrawn with the main prompt. To leave a class as it is, skip it with `[CLASS=hand] [SKIP] [/CLASS]` or leave it out of the class filter; to give it other text, add a block for it.
- **Styles can still be applied twice.** With a blank or `[PROMPT]` ADetailer prompt, the styles, their LoRAs included, still reach the detailer pass twice when another extension rewrites the main prompts (for example Dynamic Prompts with wildcards). Write the tab's own prompt instead: it gets the styles once. 🟡 Fixed in plus.10 beta 2, except in three rare cases: see [Fixed in plus.10 beta 2](#fixed-in-plus10-beta-2).
- **Pasted parameters with a detector added while the WebUI runs show an empty detector menu on Forge, Forge Neo and reForge's Gradio 4 branches.** PNG Info, Send to and the paste button find the detector and Generate uses it, but the tab's "ADetailer detector" menu looks empty until you click 🔄 next to it; AUTOMATIC1111 and reForge's main branch (Gradio 3) show the name. Load and Paste settings list it.
- **With a ControlNet model in the tab, on Forge, Forge Neo and reForge, the detailer pass runs only ControlNet**, as in the original ADetailer: reForge's Advanced Model Sampling, Soft Inpainting, HyperTile and the scripts of "Script names to apply to ADetailer" do not run in that tab's pass, whatever "Apply only selected scripts to ADetailer" is set to. For a v-prediction or flow model on reForge, leave that tab's ControlNet model on "None".

More symptoms, with what to do, are in [Problems and fixes](#problems-and-fixes).

## What is new in plus.8

**v26.2.0+plus.8**, released on 2026-10-10, is the stable release of the plus.8 reliability update: no new buttons, but the existing ones now do what you asked, keep your settings and report what really happened, and the sections you leave open are remembered. It was tested in three betas, each listed with its changes in the [CHANGELOG](CHANGELOG.md): beta 1 (v26.2.0+plus.8.beta.1, 2026-09-23), beta 2 (v26.2.0+plus.8.beta.2, 2026-10-01) and beta 3 (v26.2.0+plus.8.beta.3, 2026-10-08). The stable release is beta 3 with one more fix. The previous stable release is v26.3.0+plus.7.4, called plus.7.4 below. In this section and elsewhere in this README, except in [What is new in plus.9](#what-is-new-in-plus9), "beta 1", "beta 2" and "beta 3" without a version mean the betas of plus.8.

All 1014 offline regression tests pass on AUTOMATIC1111 and Forge Neo, and the betas were also tested live in the browser on both (see [Testing](#testing)). Forge and reForge are supported and checked in the code, but not tested live. Please report any problem in the [Issues](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues).

**To update** from plus.7.4, follow [Update](#update). Your presets and remembered settings are not touched (see [Your data](#your-data)).

Why the smaller number: version numbers now start with the version of the original ADetailer this fork is built on, 26.2.0, its latest release. Releases up to v26.3.0+plus.7.4 used 26.3.0, so by number plus.8 sorts before them although it is newer.

<a id="coming-in-beta-2"></a>

### New and fixed

🟢 available in the stable release · 🟡 in testing in a beta (none right now)

> [!NOTE]
> **plus.8 in short:**
>
> - 🟢 **Open sections are remembered:** the sections you leave open in each tab open again after a reload or restart ([issue #6](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues/6)).
> - 🟢 **The "Use separate …" options do what they say:** a tab's own steps and sampler are used also with "Apply only selected scripts to ADetailer" off, and your main Clip skip, VAE and text encoder always come back after its pass.
> - 🟢 **Works better with other tools:** ADetailer's ControlNet model on classic Forge (checked in its code), ControlNet's preview shown once on Forge and Forge Neo, Dynamic Prompts templates in the saved image, Styles and their LoRAs applied once (unless another extension rewrites the main prompts), X/Y/Z Prompt S/R images whose ADetailer prompts paste back as they were made, and AUTOMATIC1111 started with `--no-gradio-queue`.
> - 🟢 **Presets, pasted parameters and API requests behave:** a browser reload lists every preset, Reset puts the sampler back on the first one, a preset that names a detector, model or sampler this WebUI does not have loads safely and says what is missing, pasted parameters with a detector you do not have switch that tab off instead of making Generate fail, and a tab an API request leaves out gets a fresh setup, not your remembered UI settings.
> - 🟢 **Fewer silent failures:** a broken MediaPipe, a failed detector in the combined preview and regions left undetailed by a NaN error are reported, an inpaint mask drawn on a transparent layer no longer switches ADetailer off, and names and prompts in other alphabets no longer stop ADetailer when the console output goes to a log file or another program on Windows.
> - 🟢 **Your pictures keep what they had:** "Run ADetailer on an image" and folder runs fill a transparent background with the img2img background colour instead of black, a copy saved next to its source keeps its colour profile and lossless WebP quality, and saved results record their parameters with the tab's settings, also in the PNG, JPEG and WebP copies next to their sources, so pasting them brings back the tab's setup.
> - 🟢 **A clearer Detection preview:** each number also names its class, and a number or label that would cover another moves next to its box.
> - 🟢 **And the fixes of beta 1:** Interrupt and Skip, image sizes, presets, pasted parameters, X/Y/Z grids, folder runs and more: see [Fixed in beta 1](#fixed-in-beta-1).

The details are in the v26.2.0+plus.8 section of [CHANGELOG.md](CHANGELOG.md) and in the sections of the three betas below it. Where a fix removes a problem of plus.7.4 and beta 1, the line also says what to do there, when there is a way.

#### New

- 🟢 **Open sections are remembered** — the sections you leave open in each tab open again after a reload or restart, in the same browser ([issue #6](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues/6)). Details: [Remember last-used settings](#remember-last-used-settings).

#### Fixed

**Generation**

- 🟢 A tab's separate CLIP skip or VAE no longer carries over to the next tabs or stays loaded afterwards, also after "Run ADetailer on an image" (in plus.7.4 and beta 1, save Clip skip once with Settings → Apply settings, on AUTOMATIC1111 the VAE too, and on Forge and Forge Neo pick a module once in the VAE / Text Encoder selector at the top of the page).
- 🟢 "Use separate steps" and "Use separate sampler" also apply with Settings → ADetailer → "Apply only selected scripts to ADetailer" turned off (it is on by default), or with `sampler` added to "Script names to apply to ADetailer". Before, the detailer pass then ran with the main generation's steps, sampler and scheduler, and through the API with the WebUI's default steps and sampler (for example 20 steps). Reported upstream in [Bing-su/adetailer#795](https://github.com/Bing-su/adetailer/issues/795) (in plus.7.4 and beta 1, keep that option on and leave `sampler` out of that list).
- 🟢 With the WebUI's "Overlay original for inpaint" setting off (Settings → img2img; on Forge Neo, "For inpainting, overlay the resulting image back onto the original image"), "Inpaint only masked" still pastes each region back into the picture, instead of keeping only the region's crop. Two limits remain: see [Known issues of plus.8](#known-issues-of-plus8) (in plus.7.4 and beta 1, keep that setting on, or untick "Inpaint only masked").
- 🟢 Ticking Manual mode while an image is being made applies from the next generation (also in the img2img Batch tab and Loopback), instead of saving Skip img2img images at 128x128 or leaving other scripts stopped.
- 🟢 With Skip img2img, the transparent parts of an input image get the WebUI's img2img background colour instead of black.
- 🟢 After a NaN error on one region, the next region gets its own colour correction instead of the failed region's, when the WebUI's "Apply color correction to img2img results to match original colors" setting is on (AUTOMATIC1111, classic Forge and reForge).
- 🟢 In img2img inpainting, a mask drawn on a transparent layer (from the Inpaint upload tab, an inpaint Batch mask folder or the API) is read from its transparency, as the WebUI reads it, also with Soft inpainting on. Before, ADetailer read its brightness: it printed "img2img inpainting with no mask -- adetailer disabled." and detailed nothing, or detailed faces outside the painted area (in plus.7.4 and beta 1, use a black-and-white mask without transparency).
- 🟢 With Skip img2img, "Save intermediate step images" saves the step images of each tab and of each class pass; before, only the mask previews were saved there.
- 🟢 A ControlNet model whose file name writes its type with capitals, such as `OpenPoseXL2`, gets its preprocessor; before, no preprocessor ran for it (in plus.7.4 and beta 1, rename the file so that its type is in lower case, such as `openposeXL2`, and restart the WebUI).

**Classes and prompts**

- 🟢 With Styles selected and a blank or `[PROMPT]` ADetailer prompt, the selected Styles reach the detailer pass once, not twice, also when the main prompt holds comments. This does not apply when another extension rewrites the main prompts, for example Dynamic Prompts with wildcards, or when a Style adds its text on the line of a comment: there the styles can still be applied twice. "Strip LoRAs from the detailer prompt" removes the styles' LoRAs too.
- 🟢 With "Use LoRAs from main prompt", a LoRA that a selected Style adds is used once in the detailer pass, not twice; its trigger phrase is still added with "Append LoRA triggers from name" (in plus.7.4 and beta 1, move that LoRA from the Style into the main prompt).
- 🟢 An ADetailer prompt of only spaces or line breaks uses the main prompt, and it is no longer written into the image parameters, where AUTOMATIC1111 could not paste it back.
- 🟢 A class picked by its number (such as `1`) in pasted parameters or a preset is kept when the detector changes too.
- 🟢 In sequential mode, a class filter given as numbers uses the per-class prompt lines written with the class names, and the Auto class-guard stays on when none of the selected class names exists in the detector (in plus.7.4 and beta 1, remove the unknown names or turn sequential mode off).
- 🟢 In sequential mode, a per-class line written for a class number (`0: …`, with a class filter given as numbers) takes the place of the Auto class-guard in its pass, as a line written with the class name does (in plus.7.4 and beta 1, turn the Auto class-guard off in that tab).
- 🟢 In sequential mode with the "Merge and Invert" mask mode, the tab runs one pass that inverts every selected class together, with the tab prompt, as without sequential mode. Before, each class pass inverted only its own class and repainted the other classes with its prompt (in plus.7.4 and beta 1, turn sequential mode off when you use Merge and Invert).
- 🟢 A class such as `²` or `①` in the class filter of a YOLO detector (from the API, a preset or pasted parameters), which looks like a number but is not one, is ignored and named in the console like an unknown class name, instead of stopping ADetailer for the image.
- 🟢 "Inpaint only these detections" reads the Japanese long-vowel mark and middle dot (`1ー3`, `1・3`) and a doubled dash (`1——3`, `1--3`) (in beta 1, type a range with a single `-` or `～` and separate numbers with commas or spaces; in plus.7.4, use only a plain `-` and commas, such as `1-3,5`).

**Presets and settings**

- 🟢 Reset, "Reset every tab" and loading a preset without a sampler put "ADetailer sampler" back on the first sampler (in plus.7.4 and beta 1, if you use "Use separate sampler", pick your sampler again after a Reset).
- 🟢 A preset saved right after a Reset in an earlier release loads with "Use same checkpoint", "Use same VAE", "Use same text encoder" and the first sampler, as on a fresh setup, instead of empty lists and a sampler that is not in the list. With "Use separate sampler" ticked, its sampler name, such as "DPM++ 2M Karras", is shown as the sampler and scheduler the WebUI runs it as, DPM++ 2M with the Karras scheduler, so images come out the same (in plus.7.4 and beta 1, pick those entries and the sampler again after loading it, then save the preset).
- 🟢 Reloading the browser page lists every preset saved, renamed or imported since the WebUI started.
- 🟢 `user_presets.json`, `user_state.json` and imported preset files saved as UTF-16 (for example with Notepad's "Unicode" option) are read instead of being set aside.
- 🟢 Settings → Defaults no longer saves a path typed in "Or batch a whole folder" as a startup default.
- 🟢 The "Manual mode" help text on the Settings page points to "Run ADetailer on an image".
- 🟢 Loading a preset whose detector is not installed here (moved from another WebUI, or its file deleted or renamed) keeps the tab's detector and class filter, loads the rest and says so, instead of saying "Loaded" and making Generate fail for that tab and the later ones (in plus.7.4 and beta 1, pick an installed detector in the tab after loading such a preset).
- 🟢 Loading a preset whose separate checkpoint, VAE, text encoder, scheduler or ControlNet model is not in this WebUI sets that list to "Use same …" or "None", and with "Use separate sampler" ticked a sampler this WebUI does not have becomes the first sampler; the status names each one. Before, Load set them anyway and Generate quietly used something else. A checkpoint or ControlNet model named with or without its " [hash]" is still found, so a preset moved between AUTOMATIC1111 and Forge finds its ControlNet model (in plus.7.4 and beta 1, check those lists after loading a preset from another WebUI).
- 🟢 After a restart, a remembered old sampler name such as "DPM++ 2M Karras", with "Use separate sampler" ticked, comes back as that sampler and its scheduler, instead of the first sampler with "Use same scheduler", which then ran with the main pass's scheduler without a message. Some names are still not split: see [Known issues of plus.8](#known-issues-of-plus8).
- 🟢 Import reports a damaged preset file as not imported, instead of failing without a message or emptying every preset list (in plus.7.4 and beta 1, import only files exported by ADetailer Ultimate or copies of your own `user_presets.json`).

**Saved and pasted parameters, X/Y/Z and the API**

- 🟢 A tab that an API request leaves out gets a fresh setup's settings, as in the original ADetailer (only the 1st tab on, with the first detector), instead of the settings remembered in the UI: a request that sends only a face tab no longer also runs, for example, a hand tab enabled in the UI.
- 🟢 With the Dynamic Prompts extension and its "Save template to metadata" option, the images keep the prompt template with its wildcards ("Template" and "Negative Template") in their saved parameters, instead of the first image's finished prompt. Reported to Dynamic Prompts in [#703](https://github.com/adieyal/sd-dynamic-prompts/issues/703) and [#730](https://github.com/adieyal/sd-dynamic-prompts/issues/730).
- 🟢 Images from an X/Y/Z "[ADetailer] Prompt S/R" cell paste back the same ADetailer prompts, in every tab and in the per-class prompt lines of sequential mode. With the "(AD 1st)" axis, a region that no `[CLASS=…]` block matches and a `[PROMPT]` in a per-class line still paste back with the main prompt as typed; the "(AD 1st and main prompt)" axis pastes them back the same as they were made.
- 🟢 An X/Y/Z cell that starts after Manual mode is ticked, and a file of the img2img Batch tab with an empty mask, no longer record the ADetailer parameters of the cell or file before them.
- 🟢 Pasted parameters (PNG Info, Send to, the paste button) whose detector is not installed here, as is common with pictures shared online, keep that tab's detector and class filter and switch the tab off: they no longer switch ADetailer on or make Generate fail, and the console names the missing detector (in plus.7.4 and beta 1, pick an installed detector in that tab, or switch the tab off, before you press Generate).
- 🟢 With "Use separate sampler" ticked in the pasted parameters, pasting splits an old sampler name made of a sampler of this WebUI and a scheduler's label, such as "DPM++ 2M SDE Karras" from a WebUI before 1.9, into the sampler and the scheduler; before, the "ADetailer sampler" list got a name it does not have. For the names it does not split, see [Known issues of plus.8](#known-issues-of-plus8).
- 🟢 Each image of an X/Y/Z "[ADetailer] Prompt S/R" cell records its own ADetailer prompt, also in a batch whose images have different prompts and with a wildcard extension such as Dynamic Prompts. The "(AD 1st and main prompt)" axis no longer replaces twice in the main prompt that a blank or `[PROMPT]` ADetailer prompt stands for (with the values `smile, big smile`, the faces got "big big smile").
- 🟢 An X/Y/Z "[ADetailer] ADetailer model 1st" axis runs its detector in every cell, also when the 1st tab is left on "None" in the panel; a cell whose value is "None" still does not run ADetailer (in plus.7.4 and beta 1, pick a detector in the 1st tab before you start the grid).

**Detection and tools**

- 🟢 With `mediapipe_face_short` or `mediapipe_face_full`, a face cut off by the left or top edge gets a box that fits the visible face.
- 🟢 The Detection preview runs YOLO detectors on the CPU when ADetailer does (`--use-cpu adetailer`, or `--lowvram`, `--medvram` or `--medvram-sdxl` except on macOS). Of these flags, Forge Neo has only `--lowvram`.
- 🟢 With "🔁 Combine all tabs", the preview says why a tab's detector failed.
- 🟢 A folder run in the source folder counts the files it skips as earlier results (`-ad`, `-ad-1`, …) instead of leaving them out without a word.
- 🟢 When "Run ADetailer on an image" details some regions and others fail with a NaN error, the status says how many were left unchanged, instead of only "pass complete" (AUTOMATIC1111, classic Forge and reForge).
- 🟢 "Run ADetailer on an image" no longer creates the WebUI's record of your last generation when there was none yet.
- 🟢 On the single-tab Detection preview, each number also names its class, such as `#1 eyes` or `#1 face`, so the numbers no longer hide the part names of `mediapipe_face_features`. When the preview's font cannot draw a class name (Chinese or Japanese names, for example, on AUTOMATIC1111, classic Forge or reForge when no TrueType font is found, as on macOS), that number stays plain, and the combined preview leaves the name out of its label instead of failing. On other systems, such as Windows, the font has no Chinese or Japanese letters, so such a class name shows as empty boxes after the number; the number itself is always readable.
- 🟢 "Run ADetailer on an image" and folder runs fill the transparent parts of a picture with the WebUI's img2img background colour (white on AUTOMATIC1111 and grey on Forge Neo by default) instead of black.
- 🟢 A copy that a folder run saves next to its source keeps the source's colour profile (such as Display P3 or Adobe RGB), and a WebP copy is lossless when its source is, and at quality 95 otherwise, instead of lossy at quality 80.
- 🟢 Results saved to the `ADetailer-Inpaint` folder record the detailer pass's parameters, which PNG Info and Send to read, instead of the word "None" (with the WebUI's "Write infotext to metadata of the generated image" setting off, on Forge Neo "Write infotext to metadata of generated images", they record none, and no empty text file is written beside them).
- 🟢 AUTOMATIC1111 and other WebUIs with Gradio 3 start when launched with `--no-gradio-queue`: the Detection preview and "Run ADetailer on an image" buttons now follow the WebUI's queue setting (in plus.7.4 and beta 1, start the WebUI without that option).
- 🟢 On the Detection preview, a number or label that would cover one drawn before it moves next to its box, the labels of "🔁 Combine all tabs" are drawn on top of every box outline, and the text of each number and label sits inside its background also on tall pictures (see [Detection preview](#detection-preview)).
- 🟢 The working size of a picture larger than 1024 pixels in "Run ADetailer on an image" and folder runs is exactly 1024 pixels on its longest side; before, it was 1016 for some sizes, such as 1016x768 for a 1288x966 picture.
- 🟢 The parameters of a "Run ADetailer on an image" or folder-run result also record the tab's ADetailer settings, under the tab's own number, so PNG Info and Send to switch that tab on and restore its setup. The PNG, JPEG and WebP copies that a folder run saves next to their sources record the same parameters; before, they recorded none (see [Run ADetailer on an image](#run-adetailer-on-an-image)).
- 🟢 "Run ADetailer on an image" and folder runs started from the 2nd tab or a later one name that tab in their console messages and in the Verbose diagnostic log, and number the `-ad-preview` files (and the `-ad-step` files of a tab with "Process classes sequentially") with it; before, they named the 1st tab there. The result itself already used the right tab and recorded its settings under the right tab's number.

**Interface and messages**

- 🟢 In a narrow window, the version badge no longer covers the ADetailer title and the tab counter.
- 🟢 On AUTOMATIC1111, a detector without a `<model>.names.json` no longer prints a long error report in the console at startup or when you pick it. Its CLASSES list shows the classes once a generation or a Detection preview has found something with it and you switch to another detector and back (see [Known issues of plus.8](#known-issues-of-plus8)). Classic Forge and Forge Neo were not affected.
- 🟢 If MediaPipe is missing or broken, the console says so once, instead of every MediaPipe detector quietly finding nothing; when one of its libraries fails to load, its model files are no longer downloaded again at every detection.
- 🟢 When the WebUI's console output goes to a log file or to another program, such as a launcher, instead of a console window (on Windows), a detector, class, folder or file name, or a prompt, with characters of another alphabet (for example Chinese or Cyrillic) or an emoji no longer stops ADetailer or keeps its panel from appearing. A folder run also no longer counts a result it could not save as an unreadable file and leaves it out of the gallery (in plus.7.4 and beta 1, set the environment variable `PYTHONIOENCODING=utf-8` for the WebUI, or keep names in plain Latin letters and avoid such characters in the prompts; on a Japanese or Korean system plus.7.4 needs the environment variable).
- 🟢 In the same setup, the "Verbose diagnostic log" keeps all its lines. Before, every block with a character the log could not hold was left out, often the whole list of tab settings.
- 🟢 On Forge and Forge Neo, a repeated preset message, after Load, Save, Rename, Delete, Reset, Export or Import, shows again also when the answer comes back very fast.
- 🟢 On Forge and Forge Neo, the tab counter (the green pill next to the ADetailer title) follows Paste settings, Load and Reset, instead of keeping the old count until something else changed on the page.
- 🟢 On Forge and Forge Neo, with Language Diffusion, the translated "Use same …" values of a tab's dropdowns stay translated after the value changes; before, they went back to English until the page was reloaded.
- 🟢 With Manual mode on, the console says that ADetailer was skipped only for a generation it would otherwise have detailed; beta 1 said so also with ADetailer switched off.
- 🟢 The "Detection resolution" help text and the Guide tab say that 0 keeps each detector's own size: 640 for the built-in detectors, 1024 for some community detectors, where a lower value such as 768 can miss parts. The Guide also says that Merge joins every mask, also masks that do not touch, where the Detection preview tools are, that the MediaPipe detectors download their small model files on first use, and that tabs 2 and later need "Enable this tab".
- 🟢 Generate prints its `saved tab` line only for a tab whose settings were really written, so no longer with "Remember last-used settings between restarts" off or when the file cannot be written. At startup, the line for a restored tab says whether its classes are included or excluded (NOT), and shows the classes typed for a YOLO-World detector (see [Remember last-used settings](#remember-last-used-settings)).

**Forge and Forge Neo**

- 🟢 A tab's separate VAE or text encoder works, and is switched back after the pass, also before the VAE / Text Encoder selector at the top of the page has ever been used (in plus.7.4 and beta 1, on Forge Neo, pick a module in that selector once first).
- 🟢 On classic Forge, the ControlNet model chosen in a tab works with Forge's current built-in ControlNet. Before, every region printed a ControlNet error in the console and was detailed without ControlNet. This follows upstream [pull request #745](https://github.com/Bing-su/adetailer/pull/745); classic Forge and reForge were checked in their code only, not tested live.
- 🟢 The MediaPipe detectors find faces in a GIF dropped into the Detection preview.
- 🟢 With ControlNet on, its preview (for example a depth or pose map) appears once in the results, instead of once more for every batch. With ControlNet's "Save DetectMap to disk" option on, the extra copies are still saved to disk.

#### Testing

- All 1014 offline regression tests pass on AUTOMATIC1111 and Forge Neo. Beta 2 also passed the 27 tests with real detectors, and was tested live in the browser on AUTOMATIC1111 1.10 and Forge Neo, with 38 checks on each covering the open sections and many of the fixes above; beta 3 was tested live with 9 checks on each, which cover its fixes. The others are covered by the offline tests. Anything found later is fixed in a later release and added to [CHANGELOG.md](CHANGELOG.md).
- Classic Forge and reForge are not part of the live tests: the fixes that concern them, such as ADetailer's ControlNet on classic Forge, are checked in their code only.

#### Fixed in beta 1

<details>
<summary>Show the fixes of beta 1</summary>

The main fixes, by area. Most of these problems were in earlier releases too. With beta 1, all 491 offline regression tests and the 27 tests with real detectors pass.

**Generation**

- 🟢 Interrupt or Skip during a pass throws that whole pass away instead of keeping a half-finished region; Skip stops ADetailer for the rest of the batch.
- 🟢 With "Inpaint only masked" off, the image keeps its size: a hires.fix result is no longer shrunk back, and with Skip img2img the input keeps its own size (rounded down to a multiple of 8).
- 🟢 A setting you change while ADetailer is working, such as the checkpoint, is no longer undone when the pass ends.
- 🟢 With Skip img2img, "Use same sampler" uses your sampler and the saved image records your real steps, sampler and size; with Manual mode on, Skip img2img no longer changes the normal generation's size, sampler or steps.
- 🟢 A tab that stops with an error no longer leaves other scripts, such as ControlNet on Forge Neo, switched off for the rest of the batch.
- 🟢 Merge and Invert skips the pass, instead of repainting the whole picture, when the detections cover all of it.

**Presets and settings**

- 🟢 A preset or settings file that cannot be read is kept under a new name (`…unreadable-<date>.json`) instead of being overwritten, and saves from different tabs no longer overwrite each other.
- 🟢 Loading a preset or pasting a tab restores the whole setup: detector, class choices, excluded classes, YOLO-World text and a ControlNet preprocessor that fits the model.
- 🟢 A preset saved by an older version puts the settings it does not have back to their defaults, instead of keeping the tab's current values.
- 🟢 Save, Delete and Rename no longer change the preset selected in the other tabs, and the preview shows the new contents of a preset you save over.
- 🟢 Reset leaves tabs 2 and later switched off and puts the separate checkpoint, VAE and text encoder back on "Use same …", as on a fresh setup.
- 🟢 Import works on AUTOMATIC1111 and other WebUIs with Gradio 3; before, it stopped with an error.
- 🟢 Cancel in the confirmation of **🔄 Reset ADetailer settings to defaults** no longer resets and saves every option on the Settings page.

**Pasted parameters and X/Y/Z**

- 🟢 Pasting generation parameters (PNG Info, Send to txt2img/img2img) switches ADetailer and every tab the image used on and, for an image made with ADetailer, switches off the tabs it did not use; settings the image left at their defaults go back to those defaults.
- 🟢 Pasting restores a tab's class filter together with its visible selection, clears an old detection-number filter, and no longer makes AUTOMATIC1111 print "Error parsing" for empty class prompts.
- 🟢 In X/Y/Z grids, the "[ADetailer] Prompt S/R" axes no longer print an error for every cell, and each cell's images record only that cell's settings.
- 🟢 In the img2img Batch tab, one empty mask no longer switches ADetailer off for the rest of the batch.

**Classes and prompts**

- 🟢 Class names match in any case ("Face" selects "face"), in the class filter and in per-class prompt lines.
- 🟢 On AUTOMATIC1111, a class filter now works for a detector without a `<model>.names.json`, and a translated class name no longer switches the filter off.
- 🟢 With `[CLASS=hand] [SKIP] [/CLASS]` alone, the regions that are not hands get the main prompt instead of an empty one.
- 🟢 YOLO-World regions carry their class name, so `[CLASS=…]` blocks match them; "Exclude selected (NOT)", which had no effect there, is hidden.
- 🟢 "Use LoRAs from main prompt" no longer adds a LoRA that the tab prompt already names, so the tab's weight is used.

**Run ADetailer on an image and folder runs**

- 🟢 Each run picks new random seeds for all regions, so running again changes every region, not only the first.
- 🟢 The Detection preview and "Run ADetailer on an image" wait for a running generation to finish, Interrupt stops a folder run before the next file (also with AUTOMATIC1111's default "stop after the current image"), and a new run works after a cancel.
- 🟢 The status says what really happened: "Pick a detector model first.", the reason a run failed, "Batch failed" or "Batch interrupted" instead of "Batch done", and when detections were found but none was left to inpaint.
- 🟢 Results follow the WebUI's "Output directory for images" (on Forge Neo: "Output Directory") when one is set, folder runs include `.tif` files, and the WebUI's record of your last generation, which the paste button reads, is left as it was.

**Detection**

- 🟢 A mask moved with the offset sliders stops at the image border, and segmentation masks line up on non-square images.
- 🟢 "Inpaint only these detections" reads `#2`, numbers separated by spaces, and Chinese or Japanese commas and dashes (`1，3`, `1、3`, `1～3`).
- 🟢 On Windows, the MediaPipe detectors work when the WebUI's folder path has a non-ASCII character, such as an accented user name.

**Interface and translations**

- 🟢 On the light theme, the prompt placeholders, the "Preset library" heading, the status of the Detection preview and "Run ADetailer on an image", and the "📖 Guide" link are readable.
- 🟢 With Language Diffusion, button tooltips follow the interface language (the reworded ones stay in English for now), and the Paste settings button stays translated after Copy or Reset.
- 🟢 On AUTOMATIC1111, the version badge sits on the ADetailer header again, and Export explains that a download needs a WebUI with Gradio 4, such as Forge or Forge Neo, and that you can back up `user_presets.json` instead.

**Forge and Forge Neo**

- 🟢 The detailer pass uses your Distilled CFG Scale / Shift instead of always 3.5 (when a separate checkpoint is picked it keeps 3.5).
- 🟢 On Forge Neo, with "Inpaint only masked" off, the second region no longer fails, and a very early Interrupt counts as a cancel, not as an error.

Every fix of beta 1, with the details, is in the v26.2.0+plus.8.beta.1 section of [CHANGELOG.md](CHANGELOG.md).

</details>

### Known issues of plus.8

Still open in v26.2.0+plus.8, with what to do meanwhile. The four known issues listed in the beta 1 notes (at the end of the beta 1 section in [CHANGELOG.md](CHANGELOG.md)) are fixed: see the lines of [New and fixed](#new-and-fixed) that say what to do in plus.7.4 and beta 1. Unless an issue says otherwise, it also affects plus.7.4 and the betas; the two at the end come with plus.8 (from beta 2). The issues marked "fixed in plus.9 beta 1" are fixed in plus.9; in plus.8 they still apply as described. Those still open in plus.9 are also in [Known issues](#known-issues).

- **Export does not download a file on AUTOMATIC1111** (and on other WebUIs with Gradio 3, such as reForge's main branch). 🟢 Fixed in plus.9 beta 1: Export downloads the file there too. In plus.8, back up `user_presets.json` from the extension's folder by hand; Import works on every WebUI from plus.8 on (in plus.7.4 it stops with an error there, so copy the file by hand to restore it too).
- **On AUTOMATIC1111, a detector without a `<model>.names.json` shows no classes until its first detection** (Forge Neo lists the classes straight away, and so does classic Forge, checked in its code; reForge is not tested). 🟢 Fixed in plus.9 beta 1: the classes are listed straight away, read from the detector file without loading it. In plus.8, run a generation or a Detection preview with it that finds something, then switch to another detector and back, or add that file to see the classes straight away. In plus.7.4 the list stays empty for the whole session, even after a detection, and a class filter for such a detector is ignored (see [Fixed in beta 1](#fixed-in-beta-1)): add the file and restart the WebUI.
- **Newer options are in English on a translated interface.** The options added since plus.7, and some reworded tooltips and help texts, stay in English until the Language Diffusion dictionaries are updated (see [Translations](#translations)). From plus.8 this includes the reworded help text of **🔄 Reset ADetailer settings to defaults** on the Settings page; the "📥 Paste from Nth tab" label that the other tabs show after a Copy stays English too.
- **Pasted parameters do not restore "Skip img2img".** Tick it again by hand after PNG Info or Send to img2img. 🟢 Fixed in plus.9 beta 1 for img2img pictures: they record "ADetailer skip img2img" (True or False), which PNG Info and Send to read.
- **The export button's label will be renamed.** Its English label has a typo. It stays as it is for now, because the Language Diffusion dictionaries translate the button by that label: it will be renamed together with them. The button works normally. 🟢 Fixed in plus.9 beta 1: the button reads "📤 Export" (in English until the dictionaries have the new label).
- **Prompt comments typed in the ADetailer prompts reach the model.** 🟢 Fixed in plus.9 beta 1: they are removed before the detailer pass, as the WebUI removes them from the main prompt. Forge removes these comments itself, also in the detailer pass; there, from the stable plus.9, a comment no longer takes with it what ADetailer adds after it, and a box with only a comment takes the main prompt. In plus.8: the WebUI removes comments (`#`, and on Forge Neo also `//` and `/* */`) from the main prompt, and a blank or `[PROMPT]` ADetailer prompt takes the main prompt without them. Comments typed in a tab's own prompt boxes are not removed on AUTOMATIC1111, Forge Neo and reForge while "Apply only selected scripts to ADetailer" is on (the default), so their text is used as prompt text. Leave comments out of the ADetailer prompts. Adding `comments` to Settings → ADetailer → "Script names to apply to ADetailer" removes them, but a comment on the last line of a typed prompt then also removes what ADetailer adds after it: the append text, the main prompt's LoRAs and what the selected Styles add.
- **On Forge, Forge Neo and reForge, choosing a ControlNet model sets its preprocessor to "None".** 🟢 Fixed in plus.9 beta 1: a new model gets its default preprocessor, as on AUTOMATIC1111, and so does an API request that leaves the preprocessor out. In plus.8: there the preprocessor list starts with "None", which hands the picture to ControlNet as it is, and a new model keeps the tab's preprocessor only when it fits. For pose, depth, lineart, scribble and inpaint models, pick the preprocessor in "ControlNet module" yourself, for example `openpose_full`, `depth_midas`, `lineart_coarse` or `inpaint_global_harmonious`. An API request that leaves the preprocessor out also runs without one there; AUTOMATIC1111, whose lists have no "None", uses the model's default. On Forge, Forge Neo and reForge the X/Y/Z "[ADetailer] ControlNet model 1st" axis also ran each model without a preprocessor when the tab's ControlNet model was "None" or "Passthrough": fixed in the stable plus.9.
- **A restart or pasted parameters do not warn about a separate sampler this WebUI does not have.** 🟢 Fixed in plus.9 beta 1: a restart and pasting split the name as Load does, and a sampler this WebUI does not have becomes the first sampler with a console message. In plus.8: with "Use separate sampler" ticked, such a sampler, for example Res Multistep, which Forge Neo has and AUTOMATIC1111 does not, runs as the WebUI's first sampler. After a restart the "ADetailer sampler" list shows the first sampler; after pasting, it shows the missing name (empty on Forge and Forge Neo), and Generate runs the first sampler with only a console warning. From plus.8, a restart and pasting split an old name made of a sampler of this WebUI and a scheduler's label, such as "DPM++ 2M Karras", into the sampler and the scheduler. A restart and pasting do not split other spellings of a scheduler, such as "DPM++ 2M karras" or "Euler a SGMUniform", nor a name whose sampler part this WebUI does not have, such as "DPM++ 2M SDE Heun Karras" on Forge Neo: pasting keeps such a name, and a restart shows the first sampler with the saved scheduler. After a restart or a paste, check "ADetailer sampler" and "ADetailer scheduler" and pick them again if needed. From plus.8, loading a preset names such a sampler in its status.
- **A tab prompt made only of `[CLASS=…]` blocks gives the main prompt to the regions that no block matches** (from plus.8; plus.7.4 gave them an empty prompt). With `[CLASS=face] smiling [/CLASS]` and a detector that finds faces and hands, the hands are redrawn with the main prompt. To leave a class as it is, skip it with `[CLASS=hand] [SKIP] [/CLASS]` or leave it out of the class filter; to give it other text, add a block for it.
- **Styles can still be applied twice.** With a blank or `[PROMPT]` ADetailer prompt, the styles, their LoRAs included, still reach the detailer pass twice when another extension rewrites the main prompts (for example Dynamic Prompts with wildcards), or when a Style adds its text on the line of a prompt comment. Write the tab's own prompt instead: it gets the styles once. 🟢 The case of the prompt comment is fixed in the stable release of plus.9. 🟡 The case of another extension is fixed in plus.10 beta 2, except in three rare cases: see [Fixed in plus.10 beta 2](#fixed-in-plus10-beta-2).
- **From plus.8: "Overlay original for inpaint" can be saved as on.** 🟢 Fixed in plus.9 beta 1: "Inpaint only masked" no longer changes that setting. In plus.8: if you turned this setting off (on Forge Neo: "For inpainting, overlay the resulting image back onto the original image"), it can be saved as on when the WebUI saves its settings while an "Inpaint only masked" pass runs: for example when you change a quick setting at the top of the page, such as Clip skip, pick a checkpoint or VAE there on Forge Neo, or an API request changes a setting. Untick it again in Settings.
- **From plus.8: with the WebUI's settings frozen, "Inpaint only masked" can fail.** 🟢 Fixed in plus.9 beta 1: that setting is no longer changed, so frozen settings no longer stop ADetailer. In plus.8: when the WebUI is started with a launch option that freezes this setting (`--freeze-settings`, `--freeze-settings-in-sections img2img` or `--freeze-specific-settings overlay_inpaint`) and "Overlay original for inpaint" (on Forge Neo: "For inpainting, overlay the resulting image back onto the original image") is off, the first "Inpaint only masked" pass stops ADetailer for that image with the WebUI's frozen-setting error in the console ("changing settings is disabled" with `--freeze-settings`, "not possible to set 'overlay_inpaint' because … frozen …" with the other two): that tab and the later tabs are not detailed, and the earlier tabs keep their result. Keep that setting on when you freeze the settings, or untick "Inpaint only masked" in the tab.

More symptoms, with what to do, are in [Problems and fixes](#problems-and-fixes).

## Install

**You need** a working AUTOMATIC1111 1.10, Forge, Forge Neo or reForge, and an internet connection the first time ADetailer starts.

> [!WARNING]
> **Keep only one ADetailer.** If the original ADetailer, `ADetailer-Neo` or another ADetailer-based extension is in your WebUI's `extensions` folder, move it out before you install: see [Coming from the original ADetailer?](#coming-from-the-original-adetailer).

### Coming from the original ADetailer?

1. **Move the original out.** Close the WebUI and move the `adetailer` folder out of the WebUI's `extensions` folder, to any place outside it. Keep it: it is your way back. Do the same with `ADetailer-Neo` or any other ADetailer-based extension: with two ADetailers the panel appears twice, and running both is not supported.
2. **Install ADetailer Ultimate:** start the WebUI again and follow the steps in [Install from the Extensions tab](#install-from-the-extensions-tab), which end with a complete restart of the WebUI.
3. **Check that only one ADetailer is left:** the ADetailer panel appears only once. Open it (this also switches ADetailer on) and the top right of its header shows "ADetailer Ultimate" and the version (with v26.3.0+plus.7.4 on AUTOMATIC1111, just above the header).

**What comes with you**

- Your detectors: the files in `models/adetailer` and in the folders listed in Settings → ADetailer → "Extra paths to scan adetailer models…", and the built-in detectors the original already downloaded, which are not downloaded again.
- Your Settings → ADetailer options: this fork reads the same ones. The options it adds start at their defaults.
- Images made with the original ADetailer: their parameters paste back with PNG Info or Send to txt2img / img2img (from plus.8, a tab whose detector you do not have is switched off instead of making Generate fail). API requests written for the original work too. From plus.8, a tab that a request leaves out gets a fresh setup, as in the original. In plus.7.4 and beta 1 it gets the settings remembered in the UI: if you use the API there, turn off Settings → ADetailer → "Remember last-used settings between restarts" and restart the WebUI.
- The Python packages the original installed, except MediaPipe 1.0 or newer: the install replaces it with the tested 0.10 version (see [MediaPipe detectors](#mediapipe-detectors)).

**What starts fresh**

- Presets: the original has none, so the [Preset Library](#preset-library) starts empty.
- Remembered tab settings: the tabs start with their defaults, and from your first Generate on they come back as you last generated with them (see [Remember last-used settings](#remember-last-used-settings)). Startup values you saved for the original's tab settings with Settings → Defaults are not used; those of "Enable ADetailer" and "Skip img2img" still are.

**Going back:** close the WebUI, move the `adetailer-ultimate` folder out of `extensions`, put the folder you moved out before back and restart the WebUI completely. If you no longer have that folder, install `https://github.com/Bing-su/adetailer.git` from **Install from URL** in the same way (on Forge Neo, where the original does not run, reinstall the ADetailer port you used before). Your detectors and Settings → ADetailer options work there as before; the options only this fork adds are not used. Your presets and remembered settings stay inside the `adetailer-ultimate` folder you moved, ready if you come back.

### Install from the Extensions tab

1. Open the **Extensions** tab, then its **Install from URL** tab.
2. Enter `https://github.com/xXIlRizzoXx/adetailer-ultimate.git` in "URL for extension's git repository".
3. Click **Install**.
4. Wait until you see the message "Installed into stable-diffusion-webui\extensions\adetailer-ultimate. Use Installed tab to restart." (the start of the path is your own WebUI folder). ADetailer installs the Python packages it needs at this point, so on a WebUI that does not have them yet this can take a few minutes.
5. Restart the WebUI completely: close it, including its console window or launcher, and start it again. If you are not sure how, restart your computer.

These steps install the stable release (v26.2.0+plus.9). To try a beta when one is in testing, see [Switch to the beta and back](#switch-to-the-beta-and-back).

![The Install from URL tab of the Extensions tab](https://i.imgur.com/qaXtoI6.png)

(Screenshot and steps adapted from the instructions of [Mikubill/sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet).)

The first start takes a little longer: ADetailer downloads its built-in detectors from Hugging Face (and, if you installed with git, the Python packages it needs), so the WebUI needs an internet connection that time. The MediaPipe detectors download their two small model files the first time you use one of them, which needs an internet connection too. Then an **ADetailer** panel appears in txt2img and img2img: see [Quick start](#quick-start).

The steps are the same on AUTOMATIC1111, Forge, Forge Neo and reForge. They are tested on AUTOMATIC1111 1.10 and Forge Neo.

If you prefer git, open a terminal in the WebUI's `extensions` folder, run this command and restart the WebUI:

```bash
git clone https://github.com/xXIlRizzoXx/adetailer-ultimate.git
```

### Update

- **From the WebUI:** open Extensions → Installed, click **Check for updates**, then **Apply and restart UI** (called **Apply and quit** when the WebUI cannot restart itself; then start it again). This updates the version you are on: the stable release, or the beta if you switched to it.
- **With git:** open a terminal in the extension's folder and run `git pull`, then restart the WebUI. On the beta you can also run the two beta commands of the next section again; they also work when `git pull` stops with an error.

Updating keeps your presets and remembered settings (see [Your data](#your-data)). To check which version you have, open the ADetailer panel (this switches ADetailer on; close it again if you do not want it). The badge at its top right shows the version, for example `ADetailer Ultimate · v26.2.0+plus.9`, followed by a short code of the exact build.

### Switch to the beta and back

The beta is the next version while it is being tested. Right now the `beta` branch holds the third beta of plus.10, v26.2.0+plus.10.beta.3 (see [What is new in plus.10](#what-is-new-in-plus10)), and the stable release is v26.2.0+plus.9. Switching uses git, the same tool the WebUI uses to install extensions. It works for a copy installed with **Install from URL** or `git clone`, not for a ZIP file downloaded from the Releases page, and it needs a terminal where the `git` command works.

1. Open a terminal in the extension's folder: the folder you installed it in, usually `extensions/adetailer-ultimate` inside your WebUI folder. On Windows: open the folder in File Explorer, click the address bar, type `cmd` and press Enter.
2. Run these two commands:

   ```bash
   git fetch origin
   git checkout -B beta origin/beta
   ```

3. Restart the WebUI.

To go back to the stable release, run these two commands in the same folder and restart the WebUI:

```bash
git fetch origin
git checkout -B main origin/main
```

The `-B` in the commands makes them work also when your copy already has an older beta from an earlier try. Your presets and remembered settings stay as they are in both directions. 🟡 The three options that plus.10 beta 2 adds in Settings → ADetailer are ignored by plus.9 and beta 1. A beta in testing is also on the Releases page, marked as a pre-release; the `beta` branch can be ahead of it.

### Uninstall

1. Close the WebUI.
2. If you want to keep your presets and remembered settings, copy `user_presets.json` and `user_state.json` out of the extension's folder first (see [Your data](#your-data)).
3. Move the `adetailer-ultimate` folder out of `extensions`, or delete it.
4. Start the WebUI again.

To switch ADetailer Ultimate off only for a while, untick it in Extensions → Installed and click **Apply and restart UI** (or **Apply and quit**, then start the WebUI again).

To go back to the original ADetailer, see "Going back" in [Coming from the original ADetailer?](#coming-from-the-original-adetailer).

### Your data

Your presets and remembered tab settings are two files in the extension's folder (usually `extensions/adetailer-ultimate` inside your WebUI folder). Everything else is kept by the WebUI or your browser, or is downloaded again when needed (the built-in detectors in the Hugging Face download cache or, 🟡 when plus.10 beta 1 downloads one for the first time, in the WebUI's `models/adetailer` folder; the MediaPipe model files in the extension's folder).

| What | Where it is kept | Kept when you update or switch between stable and beta? |
| --- | --- | --- |
| Your presets | `user_presets.json`, in the extension's folder | Yes |
| Remembered tab settings | `user_state.json`, in the extension's folder | Yes |
| Settings → ADetailer options | the WebUI's own settings (`config.json`) | Yes |
| Startup defaults of the panel's checkboxes | the WebUI's `ui-config.json` (see [Startup defaults](#startup-defaults-for-the-panels-checkboxes)) | Yes |
| 🟢 Which sections of the panel are open (from plus.8) | your browser | Yes; another browser, or another address for the same WebUI (such as `localhost` instead of `127.0.0.1`), starts with every section closed |
| Your own detector models | the WebUI's `models/adetailer` folder (see [Models](#models)) | Yes |

- **The two files are yours.** They are not part of the download, so updating and switching between stable and beta leave them as they are. Deleting the extension's folder deletes them too, so copy them first.
- **Back them up** by copying `user_presets.json` and `user_state.json` somewhere safe. To restore them, close the WebUI, put the copies back in the extension's folder and start the WebUI.
- **Move presets to another computer** with Export and Import (see [Export and import presets](#export-and-import-presets)), or copy `user_presets.json`. From plus.9, the export button downloads the file on every WebUI; in plus.8, on AUTOMATIC1111 and other WebUIs with Gradio 3 such as reForge's main branch, it cannot download a file, so copying the file is the way there.
- 🟢 **A damaged file is never overwritten** (from plus.8). If one of the two files cannot be read, for example after a hand edit that broke it, ADetailer keeps it under a new name, such as `user_presets.unreadable-<date>.json` or `user_state.unreadable-<date>.json`, before it saves a new one. Files saved as UTF-8 are read, and from plus.8 also files saved as UTF-16, as Notepad's "Unicode" option saves them.
- 🟢 **Saves are safe.** A crash during a save leaves the old or the new file, not an empty one, and from plus.8 so does a power cut or system crash right after a save, on drives that support it.

## Quick start

ADetailer works on the finished image: it finds the parts you choose (faces, hands and so on), makes a mask for each one and redraws only those parts, with more detail. Your first fix takes three steps:

1. **Turn it on.** In the txt2img (or img2img) tab, scroll to the **ADetailer** panel below the generation settings and click the **ADetailer** title in its header. The panel opens and the checkbox next to the title (Enable ADetailer) is ticked with it. Ticking only the checkbox switches ADetailer on without opening the panel; click the title to open it.
2. **Pick a detector.** In the panel's **1st** tab, "ADetailer detector" already shows `face_yolov8n.pt` on a new install, once the first start could download the built-in detectors: keep it to fix faces. "Enable this tab (1st)" is already ticked.
3. **Generate.** Click **Generate** as usual. When the image is ready, ADetailer finds each face and redraws it, and the gallery shows the fixed image.

**Next steps:**

- **Fix hands too:** open the **2nd** tab, pick `hand_yolov8n.pt` and tick "Enable this tab (2nd)" (tabs 2 and later start unticked). Tabs run in order: first the faces, then the hands. The green pill next to the ADetailer title shows how many tabs are on.
- **Fix only some classes** of a detector that knows several (for example only the eyes of the face parts detector): see [Class filter](#class-filter).
- **See what a detector finds** before you generate: see [Detection preview](#detection-preview).
- **Keep a setup you like:** save it as a preset ([Preset Library](#preset-library)) or copy it into another tab ([Copy Settings Between Tabs](#copy-settings-between-tabs)).
- **Fix a picture you already have**, without generating it again: see [Run ADetailer on an image](#run-adetailer-on-an-image).
- **Every other option:** [Using each feature](#using-each-feature), [Settings](#settings) and [Upstream options](#upstream-options).

If a fixed face looks too different from the rest of the picture, lower "Inpaint denoising strength" in the tab's **Inpainting** section (0.4 by default). More tips are in [Problems and fixes](#problems-and-fixes).

## Using each feature

Every option below sits in each ADetailer tab, in both txt2img and img2img, unless it says otherwise. All of it is in the stable release, except the lines and sections marked 🟡, which are new in the plus.10 betas (beta 1 or beta 2), in testing.

### Class filter

Picks which classes of a multi-class detector get detailed, for example only `face` and `hand` from a detector trained on several classes. Left empty, every class is detailed, as in the original ADetailer.

**Where:** the "ADetailer detector CLASSES" list in the detector area of each tab, just below "ADetailer detector". It fills itself with the detector's class names.

**How to use:**

1. Pick a multi-class detector in "ADetailer detector".
2. Click the classes you want in "ADetailer detector CLASSES", for example `face` and `hand`.
3. Generate. Only the regions of those classes are inpainted.

**Good to know:**

- The classes you pick are used even if you press Generate straight away, or while another job is still running. One exception: if a UI translation shows a class in the list under another name, your pick reaches ADetailer only after the current job ends, so wait for that job to finish before you press Generate.
- From plus.9, on AUTOMATIC1111 too, a detector without a `<model>.names.json` file lists its classes straight away, read from the detector file without loading it. In plus.8, on AUTOMATIC1111, a detector without a `<model>.names.json` file (or the older `<model>.json`) lists its classes only after its first detection in each session, in a generation or a Detection preview: then switch to another detector and back. Add that file (see [Custom class names via sidecar JSON](#custom-class-names-via-sidecar-json)) to see them straight away. A filter that comes from a preset, pasted parameters or the API works on such a detector anyway. Forge Neo lists the classes at once, and so does classic Forge, checked in its code; reForge is not tested. This is so from plus.8 on: in v26.3.0+plus.7.4 the list stays empty for the whole session, even after a detection, and the class filter of such a detector is ignored, so every class is detailed. There, add the file and restart the WebUI.
- From plus.8, names from a preset, pasted parameters or the API match regardless of case: `Face` selects `face` (an exact match wins). In plus.7.4 a name must match exactly, or it is ignored.
- 🟢 A class number from pasted parameters or a preset (for example `1` for the detector's second class) is kept when loading or pasting also changes the detector. In plus.7.4 and beta 1 the filter then went empty and every class was inpainted. (Class numbers sent in an API request always worked.) The face parts detector, `mediapipe_face_features`, matches names only.
- A name the detector does not have is ignored and named in the console; when none of the names of an include filter is known, every class is inpainted, as with an empty selection. The MediaPipe face features are the exception: a name they do not have is ignored without a console line, and when none of the names matches, nothing is detected.
- 🟢 In the class filter of a YOLO detector other than YOLO-World, a class written with a character such as `²` or `①` (from the API, a preset or pasted parameters), which looks like a number but is not one, is ignored and named in the console in the same way. In beta 1 it stopped ADetailer for that image, and so did plus.7.4 with an include filter (in NOT mode it was only ignored).
- YOLO-World detectors (names with `-world`) show an extra text field next to the CLASSES list, "ADetailer detector CLASSES (YOLO-World / YOLOE)" (before plus.10 beta 2: "ADetailer detector CLASSES (YOLO-World)"); the CLASSES list stays empty for them. Type what to find in the text field, separated by commas, for example `person, cat`. Left empty, the detector looks for its 80 default classes (the COCO classes). The regions it finds carry those names, so inline `[CLASS=…]` blocks work with them. See [YOLO-World detectors](#yolo-world-detectors). 🟡 From plus.10 beta 2, YOLOE detectors with text prompts (names with `yoloe`, not the prompt-free `-pf` ones) show the same text field; left empty, they find the classes saved in their file.
- The other MediaPipe detectors have no classes. The [face parts detector](#face-parts-detector-mediapipe_face_features) has five (six from plus.9, with the opt-in `iris`).
- A `<model>.names.json` file next to a detector gives it class names, or other names: see [Custom class names via sidecar JSON](#custom-class-names-via-sidecar-json). How the class options behave with images, presets and API requests made before them: [Backwards compatibility](#backwards-compatibility).

### Exclude selected (NOT)

Turns the class filter around: every class except the ones you pick is detailed.

**Where:** the "Exclude selected (NOT)" checkbox below the CLASSES list.

**How to use:** pick the classes to leave out, for example `hand`, and tick "Exclude selected (NOT)".

**Good to know:**

- With no class picked, nothing is left out.
- "Process classes sequentially" does not run in NOT mode.
- It works with `mediapipe_face_features` too. It has no effect on detectors without classes, and it is hidden for YOLO-World, which looks for the names you type, or its default classes.

### Process classes sequentially

Runs a separate detect-and-inpaint pass for each selected class, in the order you clicked them, each on the result of the previous pass. Each class is looked for on the picture the earlier classes already fixed (a normal pass finds every region on the original picture), and each class can have its own [per-class prompt](#per-class-prompts). It takes a little longer, because the detector runs once per class.

**Where:** the "Process classes sequentially" checkbox beside "Exclude selected (NOT)".

**How to use:** pick two or more classes in the order you want, for example `face` and then `hand`, and tick the box. Pass 1 details the faces; pass 2 looks for hands on that result and details them.

**Good to know:**

- Sequential mode is **ignored** for MediaPipe models other than `mediapipe_face_features` (which runs one pass per selected face part), in "Exclude selected (NOT)" mode, with the "Merge and Invert" mask merge mode (from plus.8), and when fewer than 2 classes are selected: the tab then runs its usual single pass.
- "Inpaint only these detections" is not used while the passes run: each class is detailed in full.
- Each class pass saves its own mask preview, named after the class (`…-ad-preview-1-1-face.png`, `…-ad-preview-1-2-hand.png`), when "Save mask previews" is on. It also saves its own step image (`…-ad-step-1-1-face.png`, `…-ad-step-1-2-hand.png`) when "Save intermediate step images (one per tab pass)" is on (from plus.8 also with img2img's "Skip img2img"; before, not with it). Both are in Settings → ADetailer. The files go into the `adetailer-steps` sub-folder, so you can go back to the face-only result if the hand pass spoils something.
- Skip or Interrupt during the passes rolls the whole tab back to the picture from before its first pass. Step images already saved are kept.
- 🟢 A class filter given as class numbers (for example from the API) uses the per-class prompt lines written with the class names.
- 🟢 With "Mask merge mode" on Merge and Invert, the single pass inverts every selected class together, with the tab prompt, so all of them are kept. In plus.7.4 and beta 1 each class pass inverted only its own class and repainted the other classes with its prompt: turn sequential mode off when you use Merge and Invert.

A class name the detector does not have (for example `hands` from a preset, pasted parameters or the API, for a model whose class is `hand`) gets no pass of its own and is named in the console. With `mediapipe_face_features` such a name is not dropped: its pass finds nothing, and when no selected part is known nothing is inpainted. When none of the selected names is known to a YOLO detector, one pass details every class with the tab prompt.

### Class pass order

In sequential mode the classes are processed in the order you clicked them in the CLASSES list.

- To set the order, click the classes in the order you want.
- To move a class to the end, remove it with the × on its tag and click its name in the list again. Repeat for any other class you want to move.

There is no drag-and-drop.

### Per-class prompts

Gives each class its own prompt and negative prompt in sequential mode.

**Where:** the per-class prompt box in the "Inpaint prompts" section, below the prompt fields (its placeholder starts with "Per-class prompt overrides").

**How to use:** write one line per class, `class: prompt | negative prompt`, and use "Process classes sequentially" with two or more classes selected:

```text
face: detailed face, sharp eyes
hand: five fingers | blurry hands, extra fingers
```

**Good to know:**

- Used only in sequential mode.
- The part after the first `|` is the negative prompt, and it is optional. A `|` inside brackets, as in `{smiling|laughing}` or `[smiling|laughing]`, stays in the prompt.
- A line replaces the tab prompt for that class's pass; `[PROMPT]` in it inserts the main prompt. A class without a line, or with one half of its line empty, uses the tab's prompt or negative prompt for that half. The append fields and LoRA options still apply.
- Class names match regardless of case (an exact match wins). Lines without a `:` are ignored.
- 🟢 With a class filter given as class numbers, the line written with the class name applies (a line written for the number itself wins), and either one takes the place of the Auto class-guard in its pass.

### Inline `[CLASS=…]` blocks and `[SKIP]`

Targets a class right inside the tab prompt, in normal and in sequential mode: each region keeps only the blocks for its own class, and text outside the blocks applies to every region.

**Where:** type the blocks in the tab's prompt or negative prompt, in the "Inpaint prompts" section.

**How to use:**

```text
detailed, [CLASS=face] smiling [/CLASS] [CLASS=hand] five fingers [/CLASS]
```

Faces get "detailed, smiling" and hands "detailed, five fingers".

**Good to know:**

- A tag can name several classes: `[CLASS=face,eyes] … [/CLASS]`.
- `[CLASS=hand] [SKIP] [/CLASS]` skips only the hands. A `[SKIP]` segment of its own (between `[SEP]`s) works as in the original ADetailer. In a prompt with class blocks, a `[SKIP]` outside the blocks skips every region.
- A region that the blocks leave without text of its own (no block matches it and there is no text outside the blocks) gets the main prompt, as a blank prompt does: `[CLASS=hand] [SKIP] [/CLASS]` alone details every other region with the main prompt. A region with a matching block keeps only that text: `[CLASS=hand] five fingers [/CLASS]` gives the hands just "five fingers"; write `[PROMPT], [CLASS=hand] five fingers [/CLASS]` to add the main prompt.
- YOLO-World regions match the names typed in its CLASSES field. With a detector that has no classes (the MediaPipe detectors other than `mediapipe_face_features`), every block applies to every region.
- With "Mask merge mode" on Merge, a merged region keeps the blocks of every class it holds and is skipped only when all of them are skipped. With Merge and Invert the region is the background, so no block applies to it.
- A prompt without `[CLASS=` works exactly as before.

### A different prompt or LoRA for each face

Gives each face of a picture, or each hand or person, its own prompt and its own LoRA, with `[SEP]`.

**Where:** the tab's prompt, in the "Inpaint prompts" section, and Settings → ADetailer → "Sort bounding boxes by" (from plus.9, also "More sort orders (overrides Sort bounding boxes by)").

**How to use:**

1. In Settings → ADetailer, set "Sort bounding boxes by" to "Position (left to right)" and click **Apply settings**, so the parts go to the faces from left to right.
2. In the tab prompt, write one part per face, separated by `[SEP]`:

```text
smiling woman, red hair <lora:style_a:0.8> [SEP] old man, grey beard [SEP] [SKIP]
```

The leftmost face gets the first part and its LoRA, the next face the second part, and any other face is left as it is.

**Good to know:**

- Each face is inpainted in a pass of its own, so a LoRA written in one part is used only for that face. "Use LoRAs from main prompt" still adds the main prompt's LoRAs to every part, and "Strip LoRAs from the detailer prompt" removes them all.
- With more faces than parts, the faces left over get the last part: end with `[SKIP]` to leave them as they are, or with a part that suits them all.
- The negative prompt can be split with `[SEP]` the same way, and `[PROMPT]` in a part puts the main prompt there.
- With "Mask merge mode" on Merge, all the faces are inpainted as one region, which gets only the first part; with Merge and Invert the background is inpainted instead, also with the first part.
- From plus.9, Settings → ADetailer → "More sort orders (overrides Sort bounding boxes by)" gives the parts in other orders: Position (right to left), Position (top to bottom), Area (small to large) or Confidence (high to low). Any choice but Off is used instead of "Sort bounding boxes by"; for example, with Position (right to left) the first part goes to the face on the right. Keep "More sort orders" on Off to use "Sort bounding boxes by" as in step 1.
- The numbers of the Detection preview (#1, #2, …) follow the detector's own order, not "Sort bounding boxes by" or "More sort orders", so they can differ from the order of the parts. "Inpaint only these detections" picks faces by those numbers; the faces it keeps are then sorted and get the parts in order.
- To give different classes of one detector their own text, such as faces and hands, use [`[CLASS=…]` blocks](#inline-class-blocks-and-skip) instead.
- 🟡 From plus.10 beta 2, for people (the whole person or the face) the box "Guide: several subjects, one LoRA each" above the tabs writes this prompt and sets up the tab for you: see [Several subjects, one LoRA each (guide)](#several-subjects-one-lora-each-guide).

### Several subjects, one LoRA each (guide)

🟡 New in plus.10 beta 2. A box in the panel that sets up a tab so that each person in a picture is redrawn with their own LoRA and their own words. It is the guided way of [A different prompt or LoRA for each face](#a-different-prompt-or-lora-for-each-face): it picks the detector, writes the `[SEP]` prompt and sets the options that keep the parts in order, and everything it fills can be edited afterwards. Tested live on AUTOMATIC1111 1.10.1 and Forge Neo 2.32. On AUTOMATIC1111, each of the three choices of "What to redraw" gave every part to the right person with "Position (left to right)", also with "People in the picture" on 2 and one row filled; three status warnings (the sort order, a 2nd tab still switched on, "Max tabs" at 1), a click on the button during a generation, presets, Reset, the parameters of a two-tab picture pasted back and the gallery buttons after a reload of the page were tried too. On Forge Neo, the button filled the tabs, the status warned about a prompt comment mark and a generation was made from the page. Not tried live: LoRA aliases, `lyco` tags, a typed LoRA name that the WebUI does not list and a translated interface, which are covered by the offline tests only.

**Where:** the closed box "Guide: several subjects, one LoRA each" in the ADetailer panel, above the tabs, in txt2img and img2img.

**How to use:**

1. Once: in Settings → ADetailer, set "Sort bounding boxes by" to "Position (left to right)", leave "More sort orders (overrides Sort bounding boxes by)" on Off and click **Apply settings**. The guide never changes the Settings; its status says when this is not set.
2. Write the main prompt for the whole picture, without the LoRAs of the subjects: a LoRA in the main prompt acts on everybody. Describe the people there with plain words (colours, clothes, who is on the left and who on the right): they are then easier to recognise after the guide's pass.
3. Open the box and pick "What to redraw" (see the table below).
4. Fill one row for each person, from left to right: pick the LoRA in "Subject 1 LoRA", set "Subject 1 weight" (it starts at 1) and type a few words in "Subject 1 words", such as `woman, red hair, green dress`. Then the same in the row of Subject 2, such as `man, grey beard`, and so on, up to four. Leave a row empty to leave that person as they are.
5. If the picture has more people than the rows you filled, set "People in the picture" to their number (see below).
6. Press "✨ Fill the tabs" and read the status under the button.
7. Generate.

**What to redraw:**

| Choice | What the guide fills | When to pick it |
| --- | --- | --- |
| "Whole person (outline)" (the default) | The 1st tab, with a person detector that gives the outline of each person: `person_yolov8s-seg.pt`, or `person_yolov8n-seg.pt` when the first is not installed. Each person is redrawn inside their outline, so body and clothes follow the LoRA too. | A LoRA of a whole character or of an outfit. |
| "Face only" | The 1st tab, with a face detector: `face_yolov8s.pt`, then `face_yolov8n.pt`, then `mediapipe_face_full`, the first one installed. Only the faces are redrawn. | A LoRA of a face, or when body and clothes should stay as they were generated. |
| "Whole person, then face (2 tabs)" | The 1st tab as with the first choice and the 2nd tab as with the second, both with the same prompt: each person is redrawn on the 1st tab, then their face on the 2nd. It needs Settings → ADetailer → "Max tabs" at 2 or more. | People who are small in the picture, whose faces need a pass of their own after the body. |

**Animal-like characters:** in the live tests a built-in YOLO face detector did not find the muzzle of an animal character that stands like a person. For them pick "Whole person (outline)": "Face only" and the 2nd tab of the last choice do not redraw a face that the detector does not find.

**People in the picture:** how many people the tab keeps, the biggest ones first: Auto, 2, 3 or 4. "Auto" counts up to the last row you filled, and a number lower than the last filled row is not used. Example: a picture with two people, and a LoRA for one of them only. With row 1 filled and "Auto", the tab keeps one person, the biggest, and that one gets the part of Subject 1, whichever side they stand on. With "People in the picture" on 2, the tab keeps the two biggest people: Subject 1 is the one on the left, and the other one is left as they are. To give the LoRA to the person on the right instead, leave row 1 empty and fill row 2.

**What the button writes** in the 1st tab (with the last choice, in the 2nd too); it also ticks "Enable ADetailer":

| Setting of the tab | Value |
| --- | --- |
| "Enable this tab" | On. |
| "ADetailer detector" | The first installed detector of the choice (see the table above). When none of them is installed, the tab keeps its detector and its classes, and the status says so. |
| "ADetailer detector CLASSES" and "Exclude selected (NOT)" | Emptied and off, together with the detector. |
| "Process classes sequentially" | Off. |
| The tab's prompt | One part per subject, separated by `[SEP]`, with a last `[SKIP]` (see below). The prompt the tab had is replaced. |
| "Strip LoRAs from the detailer prompt" | Off, or it would take the LoRAs out again. |
| "Use bbox as mask (segmentation models)" | Off in a tab that redraws the whole person, so that the outline is used; not touched in a tab that redraws faces. |
| "Mask only the top k (0 to disable)" and "Method to filter top k masks by (confidence or area)" | The number of subjects, and Area: the tab keeps that many of the biggest people (or faces), so a small figure in the background does not take a part. |
| "Mask merge mode" | None, so each person is a region of their own. |
| "Inpaint only these detections" | Emptied. |
| "Inpaint only masked" | On. |
| "Inpaint denoising strength" | 0.6, whatever the tab had: enough for each person to take the look of their LoRA inside their region (see "Good to know" below). |

Everything else is left as it is: the tab's negative prompt and "Prompt append" fields, "Use LoRAs from main prompt", the confidence threshold and the "Mask min area ratio" and "Mask max area ratio" sliders, the inpaint size, the checkpoint, sampler and ControlNet settings, the other tabs, the main prompt and the Settings.

**The prompt it writes:** with the two rows of the example, and the LoRAs `my_lora_a` and `my_lora_b`:

```text
woman, red hair, green dress, <lora:my_lora_a:1> [SEP] man, grey beard, <lora:my_lora_b:1> [SEP] [SKIP]
```

- Each part holds the words of its row, then the tag of its LoRA with its weight. A row with a LoRA and no words becomes `[PROMPT], <lora:my_lora_a:1>`: the main prompt with that LoRA. A row with words and no LoRA is the words alone.
- A row left empty among the people counted becomes `[SKIP]`, which leaves that person as they are. With row 1 filled and "People in the picture" on 2, the prompt is `woman, red hair, green dress, <lora:my_lora_a:1> [SEP] [SKIP] [SEP] [SKIP]`.
- The last `[SKIP]` is always there: without it, any other region the tab finds would be redrawn with the last part.
- A LoRA whose tag you already typed in the words of its row is not written a second time: the tag you typed stays, with its weight, and the status says so. A tag typed in another letter case than the WebUI lists that LoRA, or with the ending of its file (such as `.safetensors`), counts as the row's LoRA too: no second tag is added. The words are never rewritten, so when none of that LoRA's tags in the words has the name as the WebUI lists it, the status warns that the subject would be redrawn without a LoRA and gives the name to write in the tag. A tag under the LoRA's alias while the row's menu has its file name, or the other way round, counts as the row's LoRA too: the guide reads from the WebUI's LoRA list which file each alias stands for, so no second tag is added. A second tag is still added for a tag in the words that does not hold exactly the row's name, when the guide cannot tell that it is the row's LoRA: on a WebUI that gives no LoRA list, for a LoRA the WebUI does not list (the status warns about it), for a tag that matches two names the WebUI lists that differ only in letter case (such as `MY_LORA` with `my_lora` and `My_Lora`), and for an alias on a WebUI that does not tell which file it stands for.
- A `[SEP]` or `[SKIP]` typed in the words is taken out, and the status says so.
- A LoRA name typed in another letter case than the WebUI lists it, or with the ending of its file (such as `.safetensors`), is written as the WebUI lists it. The exception is a row whose words hold tags of that LoRA, none of them with the name as listed (see the point above): the row then takes the name of the first of those tags, which stays as typed, no second tag is added under the listed name, and the status warns that the name is not the one the WebUI lists, with the name to write in the tag.

**What the status says:** the text under the button stays until you press the button again.

- "✅ Filled the 1st tab for 2 subjects with person_yolov8s-seg.pt. Subject 1 is the leftmost of the 2 biggest people." Everything was written, and the parts go from left to right. With one subject it reads "Subject 1 is the biggest person.", and with "Face only" it counts faces.
- "⚠️ The parts follow Settings, ADetailer, Sort bounding boxes by, now None: set it to Position (left to right) and click Apply settings." The tab is filled, but the parts would not go to the people from left to right: do step 1. When "More sort orders" is not on Off, the status names that setting instead and says to set it to Off.
- "⚠️ No person detector is installed (person_yolov8s-seg.pt or person_yolov8n-seg.pt): the 1st tab keeps its detector and classes. Pick a detector that finds a person there." The rest of the tab is filled.
- "⚠️ my_lora_a is also in the main prompt: there a LoRA acts on the whole picture and, with Use LoRAs from main prompt on, on every subject. Take it out of the main prompt."
- "⚠️ Subject 2: the WebUI lists no LoRA named my_lora_b. Its part was written as typed: with a wrong name that subject is redrawn without a LoRA. Pick the LoRA from the menu; a file added just now is listed after a refresh of the WebUI's LoRA list."
- "⚠️ Subject 1: the tag typed in the words names My_Lora_A, but the WebUI lists that LoRA as my_lora_a: with the name as typed that subject is redrawn without a LoRA. Write my_lora_a in the tag, or take the tag out of the words." A tag typed in the row's words names the row's LoRA in another letter case than the WebUI lists it, or with the file ending, and no tag in those words has the listed name: the tag stays as typed and no second one is added.
- "ℹ️ Subject 1: the words already hold my_lora_a, so the tag typed there is kept and no second one is added."
- "⚠️ Subject 1: the words hold a lyco tag. Forge and Forge Neo do not read lyco tags, so there that subject is redrawn without the LoRA of that tag. Write lora instead of lyco in the tag." The guide counts a `<lyco:…>` tag as a LoRA tag, as AUTOMATIC1111 reads it, so the prompt is the same; on a WebUI that reads lyco tags as LoRA tags (AUTOMATIC1111 and reForge) this line is not shown.
- "ℹ️ The prompt the 1st tab had was replaced."
- "ℹ️ The 2nd tab is switched on too: it runs as well when it has a detector." A tab the guide did not fill is still on. When it is the 2nd tab and its prompt holds `[SEP]` or a LoRA, for example after an earlier fill with "Whole person, then face (2 tabs)", the status warns that it runs too, with that prompt: switch it off or clear its prompt. The prompts of the 3rd and later tabs are not read.
- "ℹ️ The 2nd tab counts on its own: it takes the 2 biggest faces, from left to right. They are usually the faces of the same people, but not when another person has a bigger face, people overlap or a face is not found: check the result." Shown every time with "Whole person, then face (2 tabs)": the 2nd tab keeps the biggest faces, and no setting ties them to the biggest people of the 1st tab. The line has other forms. When the status warns about the order, after "it takes the 2 biggest faces" it goes on with "and sorts them with the same setting as the 1st tab sorts the people. Their order can differ from the order of the people, and a face then gets another person's part." instead of "from left to right". When no face detector is installed, so that the 2nd tab keeps its detector, it reads "it takes the 2 biggest regions its detector finds" and "With a face detector they are usually the faces of the same people" (and, with the order warning, "a region then gets another person's part"). With one subject it reads "it takes the biggest face. It is usually the face of the same person, but not when another person has a bigger face or that face is not found: check the result.", or, without a face detector, "it takes the biggest region its detector finds. With a face detector that is usually the face of the same person".
- "ℹ️ The guide set Inpaint denoising strength to 0.6 in the 1st tab. The subjects are easier to recognise when the main prompt already describes them (colours, clothes, who is on the left and who on the right) without their LoRAs." The last line every time the tabs are filled, whatever denoising strength they had; with "Whole person, then face (2 tabs)" it reads "in the 1st tab and the 2nd tab".
- "⚠️ Pick a LoRA or type some words for at least one subject.", "⚠️ This choice needs 2 tabs and the panel has 1: raise Settings, ADetailer, Max tabs to 2 or more and reload the UI, or pick another choice. Nothing was changed." and the status for a name that cannot be a LoRA name (one with `:`, `<` or `>`): nothing was written.
- A `#` in the words of a row (on Forge Neo, also `//` and `/*`) gets a warning when the WebUI's prompt comments are on: the WebUI would drop what follows it, the other subjects included.

**Good to know:**

- The subjects are not "the people you mean": the tab keeps the biggest detections, by the area of their box, and gives them the parts from left to right, by the left edge of each box. A person the detector does not find, or a figure in the background with a bigger box than a subject, moves the parts. The [Detection preview](#detection-preview) shows what the detector finds, with each box's share of the picture.
- Each person is inpainted in a pass of their own, so a LoRA written in one part is used only for that person. A LoRA in the main prompt is used for the whole picture, and "Use LoRAs from main prompt", which the guide does not change, adds the main prompt's LoRAs to every part.
- The guide sets "Inpaint denoising strength" to 0.6 in the tabs it fills: with the tab's default of 0.4 a LoRA can change little more than the eyes, the hair and the colours, and the person is not recognisable. Describe the people in the main prompt too, without their LoRAs: their colours, their clothes, who is on the left and who on the right. If a person still changes too little, raise the tab's denoising strength a little; near 0.75 the bodies can break, because the new character does not fit the outline of the person in the picture. Lower it if a person changes too much.
- The weight sliders start at 1: at that weight the people are easier to recognise. Lower a weight if its LoRA changes the person too much or spoils the style of the picture.
- With "Whole person, then face (2 tabs)" the 2nd tab counts on its own: it finds the faces by itself, keeps as many of the biggest faces as there are subjects and gives them the same parts from left to right. No setting ties a face to its person: they are matched only by the order of their boxes. The biggest faces are usually those of the people of the 1st tab, but not when another person in the picture has a bigger face, when people overlap or when a face is not found: check the result. With a sort order other than "Position (left to right)" (see step 1), the faces are sorted with that same setting and their order may not be the order of the people.
- The menus list the LoRAs the WebUI knows, with "None" first; a name can also be typed there, without the ending of its file. The guide reads the WebUI's list and does not scan the disk: for a LoRA file added while the WebUI runs, see [Problems and fixes](#the-result-looks-wrong).
- At most four subjects, and only the 1st tab (and the 2nd) are filled. For more people, or for hands and other classes, write the prompt by hand: [A different prompt or LoRA for each face](#a-different-prompt-or-lora-for-each-face).
- The choices of the guide are not saved: not in presets, not in the remembered settings and not in the image parameters, and a reload of the page puts them back to their defaults. What is saved is what the guide wrote in the tabs, as for any tab.
- The other limits are in [Notes on plus.10 beta 2](#notes-on-plus10-beta-2).

### Auto class-guard

Keeps each region looking like its own class: it adds the region's class name at the start of its prompt and the detector's other classes to its negative prompt, so a face is not redrawn as a hand, for example.

**Where:** "Auto class-guard" and "Class-guard emphasis" in the "Inpaint prompts" section. Off by default.

**How to use:** tick "Auto class-guard". To weight the class name, raise "Class-guard emphasis" (0.5 to 2.0; at 1.0 the plain name is added, at 1.2 it becomes `(face:1.20)`).

**Good to know:**

- It reads the other classes from the detector itself (or from its `<model>.names.json`), so there is nothing to type.
- It works with detectors that have class names: YOLO models other than YOLO-World, and `mediapipe_face_features`. It does nothing with YOLO-World and the other MediaPipe detectors.
- It works in normal and sequential mode, with include and NOT filters. In sequential mode, a per-class prompt line for a class takes its place for that class, from plus.8 also a line written for the class number, such as `0: …` (before, the guard was added on top of such a line).
- It leaves out a merged region that mixes classes and the Merge and Invert background.
- With `mediapipe_face_features`, whose face region holds the other parts, the face gets no negatives and "face" is never put in a part's negative.
- A class name already in the prompt as its own comma-separated item, such as `face` or `(face:1.2)`, is not added again.
- 🟢 In sequential mode, when none of the selected class names exists in the detector, the single pass that details every class keeps the guard, also when some classes have per-class prompt lines.

### Prompt append fields

Two one-line fields add a few words to the end of the detailer prompt and negative prompt, so you do not have to copy the whole main prompt into the tab.

**Where:** in the "Inpaint prompts" section, under the prompt ("Always appended to the prompt above") and under the negative prompt ("Always appended to the negative prompt above").

**How to use:** leave the tab prompt blank and type `detailed eyes, sharp pupils` in the first field: each region gets the main prompt followed by those words. In the negative field, add for example `blurry, lowres`.

**Good to know:**

- The words are added to whatever the prompt becomes: the tab prompt, or the main prompt when the tab prompt is blank. With `[SEP]`, they are added to every part, and a `[SKIP]` part still skips.
- Left empty, the fields change nothing and are not written into the image parameters.
- 🟢 A tab prompt of only spaces or line breaks counts as blank and uses the main prompt. In plus.7.4 and beta 1 the region got an empty prompt, or only the append text.

### LoRAs in the detailer prompt

Three checkboxes decide which LoRAs the detailer pass uses.

**Where:** in the "Inpaint prompts" section, below the class-guard options.

- **Use LoRAs from main prompt** adds the `<lora:…>` tags of the main prompt to the tab's own prompt, so you can write a short detailer prompt and keep your style LoRAs. A LoRA the tab prompt already names is not added again, so the tab's weight wins: with `<lora:detail:1>` in the main prompt and `detailed face <lora:detail:0.3>` in the tab, the pass uses it once, at 0.3. The names must match exactly, letter case included. A LoRA named only inside a `[CLASS=…]` block counts only for that class's regions; the other regions still get the main prompt's one. A LoRA that a selected Style adds is not merged: the WebUI applies the selected Styles to the detailer pass too, so the pass uses it once (in plus.7.4 and beta 1 it was applied twice), and its trigger phrase is still added with "Append LoRA triggers from name". With a blank or `[PROMPT]` tab prompt, while another extension (for example with wildcards) rewrites the main prompts, the whole Style, its LoRA included, still reaches the pass twice in the stable release; 🟡 from plus.10 beta 2 it reaches the pass once then too, except in three rare cases (see [Notes on plus.10 beta 2](#notes-on-plus10-beta-2)).
- **Append LoRA triggers from name** also adds the trigger phrase that a LoRA's name holds in round brackets: for a LoRA file named `my_style (cool style)`, the tag `<lora:my_style (cool style):1>` in the main prompt adds `cool style` to the detailer prompt, unless it is already there. It needs "Use LoRAs from main prompt" on. The brackets must be part of the LoRA's file name: brackets typed into the tag of a LoRA whose name has none make the WebUI look for a LoRA of that name, which it does not find. With Settings → Extra Networks → "When adding to prompt, refer to Lora by" set to "Filename", clicking the LoRA card inserts the tag with the brackets.
- **Strip LoRAs from the detailer prompt** removes every `<lora:…>` and `<lyco:…>` tag from the detailer prompt and negative prompt, so a style LoRA of the main prompt does not bleed onto faces and hands (the problem described upstream in [Bing-su/adetailer#805](https://github.com/Bing-su/adetailer/issues/805)). It runs last, so it wins over "Use LoRAs from main prompt" when both are on. From plus.8 it also removes the LoRAs that the selected Styles add; the rest of their text is kept.

**Good to know:** with a blank tab prompt, the detailer pass already uses the whole main prompt, LoRAs included. "Use LoRAs from main prompt" matters when the tab has a prompt of its own, and "Strip LoRAs from the detailer prompt" is the way to keep them out.

🟡 **The same LoRA twice (from plus.10 beta 2):** the WebUI loads a LoRA once for each tag it finds in a prompt (classic Forge loads two identical tags once), so a LoRA named twice in the final prompt of a detailer pass can get a stronger effect than one tag asks for. The same LoRA written in the main prompt and in a tab that has its own prompt is not doubled: the tab's prompt takes the place of the main one. It is doubled when the tab's final prompt names it twice: with `[PROMPT], <lora:detail:0.5>` in the tab while the main prompt already has `<lora:detail:1>`, or when the "Prompt append" text or a selected Style adds a LoRA that the prompt already has. The console then prints one line per generation, tab and LoRA (in a folder batch of "Run ADetailer on an image", one for the whole folder), with the LoRA's name and how many times it is named: keep one tag. Nothing is said for the negative prompt, whose LoRAs are never loaded, and the branches of a Dynamic Prompts `{a|b}` block are not added up; with Dynamic Prompts in the pass, a tag whose name it still has to draw (`<lora:{detail|skin}:1>`, `<lora:__style__:1>`) is not counted, nor is a tag with a `__name__` wildcard in its name with the "wildcards" extension in the pass, and a tag without a name never is. The prompt is not changed. Tested live on AUTOMATIC1111 1.10.1 with a folder batch of "Run ADetailer on an image" (one line for the whole folder); the rest is covered by the offline tests only.

### Apply only on hires.fix

Runs the tab only when the WebUI's hires.fix is on, so quick drafts made without it are not detailed and only the final, upscaled pictures are (txt2img only).

**Where:** the "Apply only on hires.fix" checkbox at the bottom of the "Inpaint prompts" section, below the LoRA checkboxes. It is shown only in txt2img.

**How to use:** tick it in the tabs you want only for the final pictures, then generate with hires.fix on. With hires.fix off, those tabs are skipped.

**Good to know:**

- The tab details the upscaled picture, after the hires.fix pass.
- It has no effect in img2img, also when it comes from a preset saved in txt2img, and "Run ADetailer on an image" ignores it.
- Off by default.

### Detection resolution

Runs a YOLO detector at a higher resolution than the one it was trained at, so it finds small or distant faces and hands that it misses at its own size (640 pixels for the built-in detectors).

**Where:** the "Detection resolution (0 = default)" slider in the "Detection" section (0 to 1536).

**How to use:** set it above the detector's own size, for example to 1024 for a built-in detector, check what it finds with the [Detection preview](#detection-preview), then generate.

**Good to know:**

- 0 keeps the detector's own size, the one it was trained at: 640 for the built-in detectors, 1024 for some community detectors. Only a value above that size finds more; a lower one, such as 768 for a 1024 detector, lowers the resolution and can miss parts. Higher values use more VRAM and time.
- It applies to generation, "Run ADetailer on an image" and the single-tab Detection preview; the "🔁 Combine all tabs" preview stays at each detector's own size.
- MediaPipe detectors ignore it.
- Only the detection changes: masks and inpainting still use the full picture.
- If you use "Inpaint only these detections", keep the same value in the preview and in the run.

### Face parts detector (`mediapipe_face_features`)

The `mediapipe_face_features` detector finds the parts of each face as separate classes, `eyes`, `mouth`, `nose`, `eyebrows` and `face`, with masks that follow each part's outline, so you can detail only the eyes or only the mouth. From plus.9, it also has an `iris` class, used only when you pick it (see below).

**Where:** pick `mediapipe_face_features` in "ADetailer detector"; its classes appear in the CLASSES list.

**How to use:** pick `eyes` in the CLASSES list to detail only the eyes, or leave the list empty for all five parts. With an empty list the whole face oval is redrawn too: with Settings → ADetailer → "Sort bounding boxes by" on None (the default) it comes last for each face, after the other parts; another sort order mixes the parts and the ovals by box position or size, and so does a position or area order of "More sort orders" (from plus.9; its Confidence (high to low) keeps them as with None, since every part has the same score). To detail only some parts, pick them, for example `eyes` and `mouth`.

**The iris (from plus.9):** pick `iris` in the CLASSES list to change the eye colour. It gives one mask per eye: a circle over the whole iris, from the iris points of MediaPipe's face landmarker, cut to the opening of the eye so the eyelids are left alone; a closed or almost closed eye gives no iris mask, also on a tilted head. It is used only when you pick it: an empty CLASSES list still means the five other parts, and "Exclude selected (NOT)" does not add it either. With "Position (left to right)" in "Sort bounding boxes by" and `[SEP]` in the prompt, each eye can get its own prompt, for example `blue eyes [SEP] green eyes`; the "Merge" mask mode inpaints both eyes together. A small dilation (about 4 pixels) and a denoising strength of about 0.3 to 0.45 change the colour without moving the gaze; leave "Scale inpaint to bbox" off, because it gives a region as small as the iris too few pixels (see [Scale inpaint to bbox](#scale-inpaint-to-bbox)). With the Auto class-guard, `eyes` and `iris` do not go into each other's negative prompt, and `iris` never goes into another part's. The iris needs MediaPipe's newer tasks interface: with the old interface, which has no iris points, it is simply not found, without an error.

**Good to know:**

- Everything that works with classes works here: "Exclude selected (NOT)", "Process classes sequentially" (one pass per part), per-class prompts, inline blocks such as `[CLASS=eyes]` and the Auto class-guard.
- It matches class names only, not class numbers. When none of the names in the filter is one of its parts, it finds nothing at all, where a YOLO detector would detail every class (see [Class filter](#class-filter)).
- 🟢 On the single-tab Detection preview each number also names its part, such as `#1 eyes`, so you can read which number to type in "Inpaint only these detections". In plus.7.4 and beta 1 the numbers cover the part names that the detector writes on its boxes. When boxes are very close, as with the eyes and the eyebrows, a number that would cover the name of an earlier one moves next to its box instead; beta 2 still drew it over that name (see [Detection preview](#detection-preview)).
- Like the other MediaPipe detectors, it is made for realistic faces, ignores "Detection resolution" and downloads its small model files by itself the first time it is used. The other MediaPipe detectors have no classes: see [MediaPipe detectors](#mediapipe-detectors).

### Inpaint only these detections

Details only the detections you choose, by the number the Detection preview shows on them. Requested in [#5](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues/5).

**Where:** the "Inpaint only these detections, e.g. 1,3,5 (blank = all)" field in the "Detection" section.

**How to use:**

1. Drop the picture into the [Detection preview](#detection-preview) and click **🔍 Run detection preview**: each box gets a number (#1, #2, …), from plus.8 followed by its class when the detector has class names (`#1 face`).
2. Type the numbers to keep: `1,3,5`, `1 3 5` or a range such as `1-3`. From plus.9, you can also click the boxes on the preview instead (see [Detection preview](#detection-preview)).
3. Detail the finished picture with "Run ADetailer on an image", or generate. To use the numbers in a generation, generate the same picture again (same seed and settings); a 2nd or later tab looks at the picture after the earlier tabs have changed it.

**Good to know:**

- Blank details every detection, and so does a value with no number in it (the console says so). When no number matches a detection, the tab inpaints nothing, and the console says that too.
- The numbers follow the detector's own order, so they match only when the picture, the detector, its classes (and "Exclude selected (NOT)"), the confidence threshold and the detection resolution are the same in the preview and in the run. When the tab has a Detection resolution, take the numbers from the single-tab preview ("🔁 Combine all tabs" off), because the combined preview detects at each detector's own size. That makes it most reliable on a finished picture with "Run ADetailer on an image". A pause in the middle of a generation is not possible in the WebUI; this two-step flow stands in for it.
- The mask filters ("Mask min area ratio", "Mask max area ratio", "Mask only the top k") apply after your choice.
- Not used while "Process classes sequentially" runs its passes (two or more classes selected): each class is then detailed in full.
- It also reads `#2` as written on the preview, and the full-width digits, commas and dashes that a Chinese or Japanese keyboard types, such as `1，3`, `1、3` or `1～3`.
- 🟢 It also reads the Japanese long-vowel mark and middle dot (`1ー3`, `1・3`) and a doubled dash (`1——3`, `1--3`). In beta 1, type a range with a single `-` or `～` and separate numbers with commas or spaces; in plus.7.4, use only a plain `-` and commas, such as `1-3,5`.
- 🟢 A click on a box of the single-tab Detection preview writes the field as a plain list in order, such as `1,3`: a range typed as `1-3` then becomes `1,2,3`.

### Use bbox as mask

Inpaints the whole detection box instead of the tight outline that a segmentation model draws, which gives the region more room to blend in.

**Where:** "Use bbox as mask (segmentation models)" in the "Mask Preprocessing" section.

**How to use:** tick it when the mask of a segmentation model hugs the subject too tightly, or its edges show.

**Good to know:**

- Only YOLO segmentation models (such as `person_yolov8n-seg.pt`) change. Box-only detectors already use the box, and the MediaPipe detectors keep their own masks.
- The saved mask preview shows the box; the Detection preview shows the detector's own masks.
- "Mask erosion (-) / dilation (+)" and "Inpaint mask blur" are other ways to widen or soften a mask.

### Scale inpaint to bbox

Sets each region's inpaint size from the size of its detection box, so the size follows the region. It suits larger regions: a small one gets too few pixels (see below).

**Where:** "Scale inpaint to bbox" and "Inpaint resolution scale" in the "Inpainting" section.

**How to use:** tick "Scale inpaint to bbox" and set the scale (0.5 to 8, default 1.5): a 256×320 box is then inpainted at 384×480.

**Good to know:**

- The scaled sizes are rounded to the nearest pixel, then down to a multiple of 8, and are at least 64 pixels: a 114-pixel side at scale 1.4 (159.6 pixels) gives 160.
- The size comes from the box alone, while the WebUI redraws the box plus "Inpaint only masked padding, pixels" (32 by default), the mask's dilation and the spread of "Inpaint mask blur" (4 by default). So, with the default values and scale 1.5, a region under about 180 pixels per side is redrawn smaller than it is in the picture and can come back blurred: a 45×35 box gets 64×64 for a crop of about 133×133. For a small region, such as an iris, leave this option off (by default the region is then redrawn at the size of the picture) or use a high scale: at scale 8 the same box gets 360×280.
- It works only with "Inpaint only masked" on. When the whole picture is inpainted, the picture keeps its own size.
- "Use separate width/height" wins when both are on. This option in turn replaces the Settings → ADetailer option "Try to match inpainting size to bounding box size" for this tab.
- With "Mask merge mode" on Merge and Invert, the size comes from the inverted region.
- From plus.9, Settings → ADetailer → "Max inpaint size, pixels per side (0 = no limit)" caps the size: a region whose scaled size would have a side longer than that value is inpainted smaller, with the same shape, so a large detection at a high scale does not run out of video memory. For example, with a limit of 1024, a 1024×1365 box at scale 1.5 (1536×2048) is inpainted at 768×1024, and the console says so. 0, the default, sets no limit.

### Dynamic denoise by area

Gives small regions a higher denoising strength than large ones, so small faces are redrawn enough while large ones keep their shape.

**Where:** the "Dynamic denoise by area" slider in the "Inpainting" section (0 to 8).

**How to use:** set it between 2 and 4. "Inpaint denoising strength" is then the strength for the smallest regions, and the larger a region is compared with the picture, the lower its strength.

**Good to know:**

- 0 uses the value of Settings → ADetailer → "Power scaling for dynamic denoise strength based on bounding box size", which is off unless you set it there. A value here overrides it for this tab only.
- The strength is "Inpaint denoising strength" × (1 − box area ÷ picture area) raised to this value, where the box is the region's detection box.
- It is written into the image parameters only when it is not 0.

### Masked content

🟡 New in plus.10 beta 1. Chooses what fills the detected region before it is redrawn, as "Masked content" in the WebUI's own inpaint.

**Where:** the "Masked content" choice in the "Inpainting" section, under "Dynamic denoise by area".

**How to use:** leave it on "original" (the default) to redraw the region from what is there, as ADetailer always did. Pick "latent noise" or "latent nothing" to redraw it from scratch, for example to replace an object instead of touching it up, with a denoising strength of about 0.8 or more. "fill" starts from the colours around the region.

**Good to know:**

- It is written into the image parameters ("ADetailer inpaint masked content") only when it is not "original", so PNG Info and Send to bring it back; pasted parameters without it set "original".
- It is saved in presets and remembered between restarts, and the 1st tab has an X/Y/Z axis, "[ADetailer] Masked content 1st", added after all the other ADetailer axes.
- Through the API, `ad_inpaint_masked_content` accepts `"fill"`, `"original"`, `"latent noise"` and `"latent nothing"` in any letter case, or the WebUI's `inpainting_fill` number 0 to 3 (as a number or as text). Any other value is rejected.
- plus.9 ignores it: see [Notes on plus.10 beta 1](#notes-on-plus10-beta-1).

### Separate checkpoint, VAE, text encoder and CLIP skip

The detailer pass can use another checkpoint, VAE, text encoder or CLIP skip than the main generation; your main settings come back after the pass. The separate checkpoint, VAE and CLIP skip come from the original ADetailer (see [Upstream options](#upstream-options)). This fork adds the separate text encoder, makes "Use separate VAE" work on Forge and Forge Neo, and makes the return of your main settings more reliable (see the lines below).

**Where:** the "Inpainting" section: "Use separate checkpoint" with "ADetailer checkpoint", "Use separate VAE" with "ADetailer VAE", "Use separate text encoder (Forge/Forge Neo)" with "ADetailer text encoder", and "Use separate CLIP skip" with "ADetailer CLIP skip". Steps, CFG scale, sampler and noise multiplier work the same way (see [Upstream options](#upstream-options)).

**How to use:** tick the "Use separate …" box and pick the value in the dropdown or slider next to it.

**Good to know:**

- The text encoder option works on Forge and Forge Neo only (tested live on Forge Neo; classic Forge is checked in the code only); on AUTOMATIC1111 it does nothing. "Use same text encoder" keeps the main one, "None (use detailer checkpoint's own)" lets the detailer checkpoint use its own, or you pick one from the list. This lets you use a detailer checkpoint from another model family than your base model (requested in [#3](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues/3)).
- On Forge and Forge Neo, "Use separate VAE" works too (in the original ADetailer it has no effect there), and "Automatic" uses the detailer checkpoint's own VAE. A VAE or text encoder chosen there that is no longer installed is left out: the pass keeps the main one for it, and the console says so.
- On Forge and Forge Neo the detailer pass uses your Distilled CFG Scale / Shift. When "Use separate checkpoint" is on and a checkpoint is picked, it uses the default, 3.5, because a checkpoint from another model family reads that value differently.
- From plus.8, a setting you change in the UI while ADetailer is working, such as the checkpoint, is kept when the pass ends, unless the tab uses its own value for that same setting (for example a separate checkpoint); plus.7.4 undoes every setting changed during a pass. This also holds for a Clip skip or VAE that has never been saved in Settings; in beta 1 such a first change was undone when the pass ended.
- Reset puts the checkpoint, VAE and text encoder dropdowns back on "Use same …".
- 🟢 After every pass, the main Clip skip, VAE and (on Forge and Forge Neo) text encoder come back, also when they have never been saved in Settings, and also after "Run ADetailer on an image". In plus.7.4 and beta 1, save Clip skip once with Settings → Apply settings (on AUTOMATIC1111, the VAE too), and on Forge and Forge Neo pick a module once in the VAE / Text Encoder selector at the top of the page, so that a tab's own choice does not carry over to the next tabs and images.
- 🟢 On Forge Neo, "Use separate VAE" also works before the VAE / Text Encoder selector at the top of the page has ever been used. In plus.7.4 and beta 1, pick a module there once first, or the detailer pass fails.
- 🟡 From plus.10 beta 2, Settings → ADetailer → "With hires fix, detail with the hires checkpoint" (off by default) gives the hires fix checkpoint to the tabs without their own: after a txt2img hires fix with a Hires checkpoint other than "Use same checkpoint", they detail the upscaled picture with it, the model that drew it, instead of the main checkpoint. A tab with "Use separate checkpoint" ticked but left on "Use same checkpoint" counts as one without. On Forge and Forge Neo the hires fix's "Hires VAE / Text Encoder" choices come with it, a tab's own "Use separate VAE" or "Use separate text encoder" still replaces them, and the pass uses the "Hires Distilled CFG Scale". Tabs with their own checkpoint keep it, and the main checkpoint comes back after the pass. It has no effect without hires fix, in img2img and in "Run ADetailer on an image"; with frozen settings (`--freeze-settings` and the like) the main checkpoint is used, and the console says so once. The hires fix checkpoint menu shows with the WebUI's "Hires fix: show hires checkpoint and sampler selection" setting (on Forge Neo: "[Hires. fix]: Show checkpoint, sampler, scheduler, and cfg options"). Tested live on AUTOMATIC1111 and Forge Neo.
- 🟡 From plus.10 beta 2, on Forge and Forge Neo, when a tab's own checkpoint is of another model family than the main model (one of them SD 1.5, SD 2 or SDXL) and its pass kept the main model's VAE or text encoder files, the console prints one warning per generation and tab (in a folder batch of "Run ADetailer on an image", one for the whole folder), after the tab's first region: the result can be broken or fail. Give the tab its own files: "Use separate text encoder" with "None (use detailer checkpoint's own)" and "Use separate VAE" with Automatic; when the main model's `.safetensors` VAE file is outside a folder named VAE (Automatic would then keep it), the line says to pick a VAE file of the checkpoint's family in the tab's "Use separate VAE" instead. Once the tab's "Use separate VAE" has a `.safetensors` VAE file with the latent channels of its checkpoint's family (for an SD 1.5, SD 2, SDXL, Flux, Chroma, Z-Image or Lumina checkpoint), that file takes the place of the main model's VAE file: the line then names only the main model's other files (its text encoders) and says to set "Use separate text encoder" to "None (use detailer checkpoint's own)", and when no such file is kept either, nothing is printed. The line stays, though, while a VAE file of the main model that the checkpoint loads with too (the same latent channels, or a layout that is not read; one saved with a checkpoint's `first_stage_model.` prefix only for an SD 1.5, SD 2 or SDXL checkpoint, as Forge Neo lays such a file over no other family) is still among the files of the pass: Forge Neo reads the files in the order of their paths, so the main model's can be the one used. When a file the line names is where no choice of the tab takes it out (a VAE file of the main model outside a folder named VAE once the tab has its own, a text encoder file outside a folder named text_encoder), the line says to move it into such a folder (such as `models/VAE` or `models/text_encoder`) and to pick it again in the WebUI's "VAE / Text Encoder" list; from there the tab's own VAE file takes the place of a VAE file, and "Use separate text encoder" with "None (use detailer checkpoint's own)" takes out a text encoder. A single "Run ADetailer on an image" repeats the warning in its status line; the summary of a folder batch does not. When the tab's checkpoint does not even load with those files, in a case that the skip of the next point does not recognise, its family cannot be read: the console then says that the checkpoint could not be loaded while the pass kept the main model's files, with the advice for the family that the header of the checkpoint's file tells (the same advice for SD 1.5 or SDXL; for Flux, a Flux VAE file in the tab's "Use separate VAE" instead of Automatic; both when the header does not tell; when the main model's `.safetensors` VAE file is outside a folder named VAE, so that Automatic would keep it, first to move that file into a folder named VAE and to pick it again in the WebUI's "VAE / Text Encoder" list, and only then the advice for the checkpoint's family, or only to move it when the tab already has a VAE file of that family; a text encoder file outside a folder named text_encoder is told to be moved as above, and "None" and Automatic are not told again when they would take out nothing; when the header tells the main model's family, no text encoder file is told to be moved), and so does the status of a single "Run ADetailer on an image" (when the line says to move a file, the status says it too). The warning blocks nothing, and on Forge Neo a checkpoint that does not load makes the whole generation fail in the cases that the skip of the next point does not recognise (see [Notes on plus.10 beta 2](#notes-on-plus10-beta-2)); there is no warning on AUTOMATIC1111 and reForge, or when a checkpoint that did load is of an unknown family. Tested live on Forge Neo with a Flux main model and an SDXL checkpoint in the tab, before the skip of the next point was added: that setup is now skipped when its files are `.safetensors` files (the skip is tested live on Forge Neo too), so the warning is left for the cases that the skip does not recognise; in those cases it is covered by the offline tests only. The status line of a checkpoint that does not load is covered by the offline tests only, and on Forge it is all covered by the offline tests only.
- 🟡 From plus.10 beta 2, on Forge and Forge Neo, a tab whose checkpoint cannot be loaded with the VAE file of its pass is skipped before the pass, so that the failed load no longer costs the whole generation: the other tabs run and the image is saved. ADetailer reads only the header of the `.safetensors` files, without loading them. It skips the tab in two cases. First, a tab with its own checkpoint would keep a VAE file of the main model with another number of latent channels: an SD 1.5 or SDXL checkpoint in the tab (4 channels) with the VAE file of a Flux, Chroma, Z-Image or Lumina main model (16 channels), or a Flux checkpoint in the tab with the VAE file selected for an SD 1.5, SD 2 or SDXL main model. Second, the VAE file chosen in the tab's own "Use separate VAE" does not fit, such as Flux's VAE file for an SDXL checkpoint; this also holds for a tab without its own checkpoint, where the WebUI loads the main checkpoint again with the tab's file. The console says it in one line per generation and tab (in a folder batch of "Run ADetailer on an image", one for the whole folder), with the name of the VAE file and what to set, and the same line shows under the results of txt2img and img2img (an API response does not carry it). What to set: for an SD 1.5 or SDXL checkpoint, "Use separate VAE" with Automatic and "Use separate text encoder" with "None (use detailer checkpoint's own)"; for a Flux checkpoint, a Flux VAE file (16 latent channels, such as `ae.safetensors`) in the tab's "Use separate VAE", not Automatic, and a checkpoint file that holds its text encoders; for a VAE file chosen in the tab, another one there. When the tab's checkpoint is of another family than the main model and a file of the main model that the pass keeps is where no choice of the tab takes it out (a VAE file outside a folder named VAE, as it can be with `--vae-dirs`, or a text encoder outside a folder named text_encoder), the line says first to move it into such a folder (such as `models/VAE` or `models/text_encoder`) and to pick it again in the WebUI's "VAE / Text Encoder" list, and the status of "Run ADetailer on an image" says so too: a choice of the tab alone could end the skip while that file stays, and Forge Neo, which reads the files in the order of their paths, could read it after the tab's, so that the load would fail with the whole generation. With a checkpoint of the main model's family such a file is the checkpoint's own (a text encoder that "None" would take out, and the pass would fail without it), so the line only says to choose another VAE file in the tab. "Run ADetailer on an image" then gives no result: its status reads "ADetailer did not run" and says that the tab was skipped, and a folder batch counts its files as failed with that reason. It works also right after the WebUI started, before the main model was ever loaded. In any doubt nothing is skipped and the pass runs as before, with the warning of the point above: a checkpoint or VAE file that is not a readable `.safetensors` file (a `.ckpt` or a GGUF file, for example), a checkpoint in the tab that is not recognised for sure as SD 1.5, SDXL or Flux, a main model of a family that is not checked, a VAE of another layout or one saved with a checkpoint's `first_stage_model.` prefix among the files, a VAE file that fits the checkpoint among the files (also when one of the main model that does not fit stays next to it, outside a folder named VAE), and the hires fix checkpoint. Only the VAE file is checked, not the text encoders, so a failed load is still possible (see [Notes on plus.10 beta 2](#notes-on-plus10-beta-2)). Nothing changes on AUTOMATIC1111 and reForge. Tested live on Forge Neo 2.32 in two cases: a Flux main model with its VAE and text encoder files and a tab with an SDXL checkpoint that kept them, and an SDXL main model with Flux's VAE file chosen in the "Use separate VAE" of a tab with an SDXL checkpoint. In both the tab was skipped with its console line, the generation ended, the picture was saved and the WebUI kept working, and the same tab with files that fit ran and detailed the picture. Classic Forge and the other cases are covered by the offline tests only.
- 🟢 "Use separate steps" and "Use separate sampler" also apply with Settings → ADetailer → "Apply only selected scripts to ADetailer" turned off, or with `sampler` in "Script names to apply to ADetailer". In plus.7.4 and beta 1 the detailer pass then takes the main generation's steps, sampler and scheduler (through the API, the WebUI's default steps and sampler): keep that option on, as it is by default, and leave `sampler` out of that list.

### Same face from a reference picture

New guide in plus.10 beta 1. Redraw the detected face so it looks like the face in a picture you choose, with an IP-Adapter model in the WebUI's own ControlNet and "Passthrough" in the ADetailer tab. It uses options that already exist. The ADetailer part was tested live on Forge Neo: "Passthrough" brings the IP-Adapter unit into the detailer pass, and "Skip img2img" leaves the rest of the picture as it was. How well the face matches depends on the IP-Adapter model and the checkpoint (see "Good to know").

**What you need:** ControlNet: on AUTOMATIC1111 the sd-webui-controlnet extension; on Forge and Forge Neo it is built in. An IP-Adapter model that fits your checkpoint, for example an `ip-adapter-plus-face` model for SDXL, in the WebUI's ControlNet models folder. The first time an IP-Adapter preprocessor runs, the WebUI may download the CLIP image encoder it needs by itself: that is the WebUI's own behaviour, not ADetailer's.

**How to use:**

1. Open img2img (not Inpaint) and drop the picture to fix into it.
2. In the WebUI's ControlNet section, outside ADetailer, enable one unit, tick "Upload independent control image" and drop the reference face into the unit. Pick an IP-Adapter preprocessor and the IP-Adapter model.
3. Switch every other ControlNet unit off.
4. In the ADetailer tab, pick a face detector, set "ControlNet model" to "Passthrough", and tick "Skip img2img".
5. Click Generate.

**Good to know:**

- "Passthrough" gives the ADetailer pass the ControlNet units set in the WebUI, as they are. The tab's own "ControlNet model" menu lists only inpaint, scribble, lineart, openpose, tile, depth, union and Anytest models, so an IP-Adapter model works only this way.
- Every unit that is on takes part in the ADetailer pass: a pose or depth unit made for the whole picture would steer the face too. Keep only the IP-Adapter unit on.
- With "Skip img2img" the img2img pass is left out, so the reference changes only what ADetailer redraws. In txt2img the unit also acts on the main image, so the whole picture is made with the reference.
- With two or more tabs switched on, set "Passthrough" in each of them: users of the original ADetailer found that in batches, with "Passthrough" in only some tabs, the later pictures can come out as noise.
- The reference and a ControlNet model picked in the tab do not go together: with a model picked, the ADetailer pass uses only that model and leaves the WebUI's units out.
- The SDXL IP-Adapter models were trained on SDXL base, mostly on photos. On anime checkpoints, such as Illustrious ones, a weight of 1 can spoil the picture even without ADetailer: try the unit alone in txt2img first, and lower its weight to about 0.5-0.7 if the result breaks up.
- "Run ADetailer on an image" runs without the WebUI's ControlNet units, so use img2img for this.
- On Forge Neo, leave ControlNet's batch input directory empty and do not use the img2img Batch tab for this: with "Passthrough", the units read the files of that folder during the ADetailer pass instead of the reference (see [Notes on plus.10 beta 1](#notes-on-plus10-beta-1)).

### Copy Settings Between Tabs

Copy one tab's whole setup and paste it into other tabs of the same panel, as many times as you like.

**Where:** the **📋 Copy settings** and **📥 Paste settings** buttons, just below the preset rows. Every tab, from the 1st up to the number set in **Max tabs** (Settings → ADetailer, default 4, up to 15, needs a UI reload), has them.

**How to use:**

1. Click **📋 Copy settings** in the tab you want to copy.
2. In every other tab the Paste button turns into **📥 Paste from Nth tab** (for example "📥 Paste from 1st tab"), so you can see where the settings come from.
3. Click it in the tab you want to fill: its settings are replaced by the copied ones.

**Good to know:**

- Everything in the tab is copied: detector, class filter, prompts, "Enable this tab", "Inpaint only these detections" and every detection, mask, inpainting and ControlNet setting.
- The copy stays until you copy another tab, press **🆕 Reset** or reload the page, so you can paste it into several tabs. A tab cannot paste onto itself.
- From plus.9, tick **Swap with the copied tab** before clicking Paste to swap the two tabs: this tab gets the copied settings and the copied tab gets this tab's, then the copy is emptied. For example, Copy in the 1st tab and Paste with the box ticked in the 3rd swaps the 1st and the 3rd. The copied tab gets this tab's settings as they are; this tab gets the copy made when you clicked Copy.
- 🟢 Tick **Paste into every tab** before clicking Paste to put the copy into every tab of the panel except the copied one. Tick only one of the two boxes: with both ticked nothing changes. Both start unticked and are never saved.
- 🟢 If the copied detector was removed after the Copy (its file deleted or renamed, then 🔄), Paste, Swap and Paste into every tab keep the tab's own detector and class filter and paste the rest, and the tab's message says that the detector is not installed, as Load does. From the stable release; in beta 3 the tab got the missing detector and Generate stopped with "Model '…' not found".
- txt2img and img2img each have their own copy. To move a setup from one to the other, save it as a [preset](#preset-library) and load it there.
- On a translated interface the "📥 Paste from Nth tab" label stays English: it is made when you click Copy and has no dictionary entry.
- Loading a preset or pasting a tab restores the whole setup, class filter included. Pasting generation parameters (PNG Info, Send to txt2img/img2img, or the WebUI's paste button) also restores the whole ADetailer setup: it switches ADetailer and the tabs the image used on and, for an image made with ADetailer, switches off the tabs it did not use; settings that the image does not record go back to their defaults. Skip img2img is not restored (from plus.9, it is, for images made with plus.9 or later), and fields listed in the WebUI's "Disregard fields from pasted infotext" (on Forge Neo: "Ignore fields when reading infotext") are left alone. From plus.8, a tab whose detector is not installed in this WebUI keeps its own detector and class filter and is switched off: it no longer switches ADetailer on or makes Generate fail, and the console names the missing detector. Pasting also splits an old sampler name that holds its scheduler, such as "DPM++ 2M SDE Karras", into the sampler and the scheduler; for the names plus.8 does not split, see [Known issues of plus.8](#known-issues-of-plus8). From plus.9, pasting splits the name as Load does, and a separate sampler this WebUI does not have becomes the first sampler, with a console message.

### Preset Library

Save a tab's whole setup under a name, and load it again later in any tab, in txt2img or img2img.

**Where:** near the top of every tab, under "Preset library" (below the export / import section): the list of saved presets with **📂 Load**, **✏️ Rename** and **🗑 Delete**, and below it a name box with **💾 Save preset**.

**How to use:**

- **Save:** set up the tab, type a name in the box and click **💾 Save preset**. A name can have letters, digits, spaces and `- _ . , ( ) [ ] + ! ? @ # &`, up to 80 characters; other symbols, such as `:`, `%`, `=`, `/` or quotes, are refused, also by Rename and Import. Saving under a name that already exists replaces that preset.
- **Check before loading:** pick a preset in the list and a summary appears below the Copy and Paste buttons: its detector, classes, sequential mode, prompts and the classes of its per-class prompt lines. `[SEP]` and `[PROMPT]` are highlighted, because they only take effect at generation time: `[SEP]` splits the prompt between the regions and `[PROMPT]` is replaced by the main prompt (from plus.9, `[PROMPT=words]` is highlighted too). Saving over a preset updates its summary in every tab of the same panel that shows it.
- **Load:** pick a preset and click **📂 Load**. Every setting of the tab changes to the preset's, including the detector, the class filter and "Enable this tab". From plus.8, when the preset names something this WebUI does not have, the status says so and the tab keeps working (see below).
- **Rename:** pick a preset, type the new name in the name box and click **✏️ Rename**. A name that another preset already has is refused.
- **Delete:** pick a preset and click **🗑 Delete**. It is deleted at once, without asking, so back up your library first if you may want it back: [export](#export-and-import-presets) it, or copy `user_presets.json` from the extension's folder.
- Pick `(none)` in the list to clear the selection without changing any setting.

**Good to know:**

- One library serves every tab, in txt2img and img2img. Save, Rename and Delete update the list of every tab in the same panel at once. The other panel (img2img after a save in txt2img, and the other way round) lists the change after its own next Save, Rename or Delete, or after a restart (from plus.8: also after a browser reload); an Import there updates only the tab you import in. Each other tab keeps its own selection: a tab that shows a renamed preset follows the new name, and one that shows a deleted preset goes back to `(none)`.
- If a change cannot be written to disk, the status line says so, and the library stays as it was.
- A preset saved by an older version loads cleanly: the settings it does not have go back to their defaults, and "Enable this tab" is left as it is.
- The library is the file `user_presets.json` in the extension's folder, and updates keep it (see [Your data](#your-data)). If the file cannot be read, it is kept under a new name, `user_presets.unreadable-<date>.json`, before the next save, and the status line says so.
- The name `(none)` is reserved. Status messages under the buttons fade after about 4 seconds, and repeating an action shows its message again (on Forge and Forge Neo, from plus.8 also when the answer comes back very fast).
- 🟢 A browser reload lists every saved preset, in every tab. In plus.7.4 and beta 1 a reload shows only the presets that existed when the WebUI started, until a restart or the next Save, Rename or Delete in that panel (an Import updates only its own tab).
- 🟢 A preset without a sampler setting, such as a file edited by hand, puts "ADetailer sampler" on the first sampler, and so does a preset whose sampler this WebUI does not have while its "Use separate sampler" is off. A preset saved right after a Reset in an earlier release shows "Use same checkpoint", "Use same VAE" and "Use same text encoder" instead of empty lists; with "Use separate sampler" ticked, its sampler name, such as "DPM++ 2M Karras", is shown as the sampler and scheduler the WebUI runs it as (DPM++ 2M and Karras), so images come out the same.
- 🟢 A preset whose detector is not installed in this WebUI (moved from another WebUI, or its file deleted or renamed) loads everything else and keeps the tab's detector and class filter, and the status says "⚠️ Loaded '…', but its detector '…' is not installed: the tab keeps its detector and classes." From the stable plus.9, Load first reads the detector folders again, as Generate does, so a detector added while the WebUI runs loads with the preset. In plus.7.4 and beta 1, Load set a detector the list does not have (Forge and Forge Neo show it empty), and Generate then failed that tab and the later ones: pick an installed detector after loading such a preset.
- 🟢 A separate checkpoint, VAE, text encoder, scheduler or ControlNet model that is not in this WebUI's list is set to the list's first choice ("Use same …", or "None" for the ControlNet model and its preprocessor), as after a restart, and the status names it. With "Use separate sampler" ticked, a sampler that is not in this WebUI's list, for example Res Multistep on AUTOMATIC1111, or another name for a sampler such as `k_euler_a`, is set to the first sampler and named in the status too. A checkpoint named without the " [hash]" it gains once first loaded, and a ControlNet model named with or without its hash, still find their file. In plus.7.4 and beta 1, Load set those names anyway and Generate quietly used something else: check those lists after loading a preset from another WebUI.

### Export and import presets

Save the whole preset library as a file, to back it up or move it to another computer, and bring it back.

**Where:** at the top of every tab, in the collapsed section "Preset library export / import", just under "Enable this tab" and above the preset rows.

**How to use:**

- **Export:** click the export button. On Forge and Forge Neo your browser downloads the whole library as `adetailer-ultimate-presets.json`; from plus.9, on AUTOMATIC1111 and the other WebUIs with Gradio 3 too.
- **Import:** click **📥 Import** and choose a preset file, either an exported file or a copy of `user_presets.json`. Presets with a new name are added. When you already have a preset with the same name, yours is kept, unless "Overwrite on conflict" is ticked: then the imported one replaces it.

**Good to know:**

- On Forge and Forge Neo, plus.7.4 downloads nothing on the first click of the export button and the previous click's file after that: click it twice without saving presets in between. This is fixed in plus.8; the fix is not yet tested in a browser.
- On AUTOMATIC1111, and on other WebUIs with Gradio 3 such as reForge's main branch, the export button cannot download a file in plus.8 (from plus.9, it downloads it, as on Forge). In plus.8 its status line says so; in plus.7.4 it wrongly reports the presets as exported, and the button's label changes to a file path. Back up the library by copying `user_presets.json` from the extension's folder instead. Import works on every WebUI from plus.8 on; in plus.7.4 it stops with an error on those WebUIs, so copy `user_presets.json` by hand there too.
- The status line counts the presets added, replaced and not imported (because of a name conflict, an invalid name, a library that could not be written or, from plus.8, a damaged preset).
- The tab you import in lists the new presets at once; the other tabs list them after the next Save, Rename or Delete in the same panel (from plus.8: also after a browser reload).
- Files saved as UTF-8 import fine.
- 🟢 Files saved as "Unicode" (UTF-16), for example by Notepad or Windows PowerShell, import too (see also [Your data](#your-data)).
- 🟢 A file that cannot be read as a preset file imports nothing, says "no presets imported (file empty, invalid, or all names skipped)" and leaves the library as it was. In plus.7.4 and beta 1 some damaged files made Import fail without a message or emptied every preset list: import only preset files from sources you trust.
- 🟢 On Forge and Forge Neo, importing the same file again shows its message again.
- A preset moved to another WebUI may name a detector, checkpoint or sampler that WebUI does not have: from plus.8, Load says so (see [Preset Library](#preset-library)).
- 🟢 From the stable release of plus.9, an image or an HTML tag in a preset appears as typed in its preview and in the status of Load, so your browser does not load it. In plus.8 and the betas, picking a preset from someone else's library could make your browser contact another server.
- "Overwrite on conflict" starts unticked; to change that, see [Startup defaults for the panel's checkboxes](#startup-defaults-for-the-panels-checkboxes).

### Remember last-used settings

🟢 Part of this is new in plus.8: remembering which sections are open.

Your tabs come back as you last generated with them: every Generate saves each tab's settings, and the next WebUI start brings them back.

**Where:** on by default. Settings → ADetailer → "Remember last-used settings between restarts" turns it off.

**How it works:**

- Each click on Generate saves the settings of every tab, separately for txt2img and img2img, so changing one never affects the other.
- At the next start, every tab shows the settings saved last. The CLASSES list of a remembered detector is ready, with your selection ticked (for a detector without a class-names file on AUTOMATIC1111, see [Class filter](#class-filter)).

**What is remembered:** every setting inside the tabs, from the detector, classes and prompts to "Enable this tab" and the mask, inpainting and ControlNet settings.

**What is not remembered:**

- The panel's own checkboxes, "Enable ADetailer" and "Skip img2img", and the tools' checkboxes: "💾 Save result to outputs", "📁 Save results in the source folder instead", "🔁 Combine all tabs" and "Overwrite on conflict". The WebUI can give them a startup default instead: see [Startup defaults for the panel's checkboxes](#startup-defaults-for-the-panels-checkboxes).
- The images dropped into the tools, the folder path, the preset selected in the list, and "Reset every tab", which always starts unticked.
- Changes you only try out with the Detection preview or "Run ADetailer on an image": the settings are saved only when you click Generate.

**Good to know:**

- The settings are kept in `user_state.json` in the extension's folder, and updates keep it (see [Your data](#your-data)). If the file cannot be read, the tabs start with their defaults, and the next save keeps the old file as `user_state.unreadable-<date>.json`.
- Turning the option off deletes nothing: the tabs start with their defaults and nothing new is saved, and turning it on again brings back the settings saved last. To start from the defaults for good, press **🆕 Reset** (with "Reset every tab" if you like), pick a detector again, then Generate with the option on.
- On every Generate the console names each tab with a detector that it saved (`[-] ADetailer: saved tab 1 (txt2img) - detector=…`), and at every start each such tab it restored. From plus.8 the saved line appears only for a tab whose settings were really written: not with the option off, and not when the file cannot be written. In plus.7.4, beta 1 and beta 2 it appears for every tab with a detector, also when nothing was saved. From plus.8 the start-up line names the class filter the same way as the saved line, `classes[include]=…` or `classes[NOT/exclude]=…`, with the classes typed for a YOLO-World detector; in plus.7.4, beta 1 and beta 2 it shows `classes=[]` for a YOLO-World tab and lists the classes excluded in NOT mode without saying that they are excluded.
- A remembered detector that is no longer installed is replaced by the tab's default choice.
- 🟢 A remembered old sampler name made of a sampler of this WebUI and a scheduler's label, such as "DPM++ 2M Karras" with "Use separate sampler" ticked (from a preset loaded or parameters pasted in an earlier release, or an earlier release's Reset followed by ticking the box), comes back as the sampler and scheduler the WebUI ran it with, DPM++ 2M and Karras. In plus.7.4 and beta 1 the tab came back with the first sampler and "Use same scheduler", and then ran with the main pass's scheduler without a message. In plus.8, any other name that this WebUI does not have, such as "DPM++ 2M karras" or "DPM++ 2M SDE Heun Karras" on Forge Neo, comes back as the first sampler with the saved scheduler, without a message: see [Known issues of plus.8](#known-issues-of-plus8). From plus.9, a restart splits the name as Load does, so "DPM++ 2M karras" comes back as DPM++ 2M and Karras, and a sampler this WebUI does not have becomes the first sampler, with a console message.
- 🟢 A tab that an API request leaves out gets the settings of a fresh setup (only the 1st tab on, with the first detector), as in the original ADetailer. In plus.7.4 and beta 1 it gets your remembered UI settings, so a request could also run, for example, a hand detector enabled in the 2nd tab.
- 🟢 **Open sections are remembered** ([issue #6](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues/6)): the sections you leave open (Inpaint prompts, Detection, Detection preview, Run ADetailer on an image, Mask Preprocessing, Inpainting and Preset library export / import) open again when the page loads. Each tab of txt2img and img2img remembers its own. This is stored only in your browser: another browser, another address for the same WebUI (for example `localhost` instead of `127.0.0.1`) or a browser that blocks site data starts with every section closed. The main ADetailer panel is left out on purpose, because opening it is the same switch as "Enable ADetailer".

### Reset

Put a tab, or every tab, back to its default settings. The Settings page has its own reset.

**Where:** **🆕 Reset** and the "Reset every tab" checkbox sit next to **💾 Save preset** in every tab. For the Settings page: Settings → ADetailer → **🔄 Reset ADetailer settings to defaults**, at the bottom of the page.

**How to use:**

- Click **🆕 Reset** to reset this tab. Tick "Reset every tab" first to reset all the tabs of the panel (txt2img or img2img) at once.
- On the Settings page, click **🔄 Reset ADetailer settings to defaults** and confirm. Every option on that page goes back to its default, the settings are saved and the page reloads. Options that need a UI reload, such as Max tabs and Show the ADetailer Guide tab, take effect after Reload UI or a restart.

**Good to know:**

- A tab reset puts every setting back to its default. The detector list shows `None` in every reset tab, the 1st tab included, so pick a detector again: a fresh setup would start the 1st tab on the first detector. "Enable this tab" is ticked in the 1st tab and unticked in the others, and the separate checkpoint, VAE and text encoder lists show "Use same …". It also clears the preset selection and the name box, and empties the copy made with **📋 Copy settings**. Saved presets are kept.
- A reset cannot be undone: save a preset first if you may want the settings back. The remembered settings change at the next Generate.
- 🟢 Reset also puts "ADetailer sampler" back on the first sampler. In plus.7.4 and beta 1, Reset sets it to "DPM++ 2M Karras", which is not in the list on AUTOMATIC1111 1.9 and later, Forge and Forge Neo: if you use "Use separate sampler", pick your sampler again after a Reset.
- The Settings reset does not touch your tabs, presets or remembered settings. From plus.8 on, Cancel in its confirmation changes nothing; in plus.7.4, Cancel still resets and saves every ADetailer option (the page just does not reload), so click the button only when you mean it. To put the remembered tab settings back to their defaults, use **🆕 Reset** in the panel, pick a detector again, then Generate.

### Detection preview

See what the tab's detector finds on an image, without generating or changing anything.

**Where:** in every tab, the "Detection preview" section right after the Detection section, above Mask Preprocessing and Inpainting.

**How to use:**

1. Pick a detector in the tab's "ADetailer detector" list first. The preview uses the tab's detector; without one it only asks you to pick one.
2. Drop or paste an image into **Input**.
3. Click **🔍 Run detection preview**. **Detections** shows the image with the box or mask of every region found and its number (#1, #2, …); YOLO detectors also write each box's class and confidence. From plus.8 the number also names the class when the detector has class names, such as `#1 face` or `#1 eyes`. 🟡 From plus.10 beta 1 each label ends with the share of the image its box takes, such as `#1 face 12.3%`. The status line counts the detections.
4. From plus.9, click the boxes on **Detections** to choose which ones to inpaint: a click adds the box's number to the tab's "Inpaint only these detections" field, a second click takes it out, and the chosen boxes are marked in green. A smaller box inside a larger one, such as the eyes inside the face, is drawn on top, so it can be clicked too. After a single-tab preview with detections, the status line reads "✅ N detection(s). Click the boxes to choose which to inpaint."

**Good to know:**

- The preview follows the tab's CLASSES, "Exclude selected (NOT)", confidence threshold and "Detection resolution". It shows the detector's raw boxes and masks: mask preprocessing (x/y offset, erosion/dilation, merge mode, Use bbox as mask) and the mask filters (min/max area ratio, top k, Inpaint only these detections) are not applied to it.
- Type the numbers into "Inpaint only these detections" to detail only those regions. They match the real pass only with the same detector, class filter, confidence and detection resolution.
- 🟢 Clicking the boxes works with the single-tab preview, not with **🔁 Combine all tabs**. Numbers typed by hand mark their boxes too, and an empty field still means every detection. The boxes belong to the last preview: run it again after you change the picture, the detector, its classes, the confidence threshold or the detection resolution. If the boxes do not respond, for example on a WebUI whose page the script does not recognise, type the numbers as before.
- **🔁 Combine all tabs** runs the detector of every tab that has one picked, whether or not its "Enable this tab" is ticked, and draws all the results on one image, one colour per tab (the colours repeat after the 6th tab), each labelled with its number, tab, class and confidence. The status line counts the detections of each tab. This combined view always runs each detector at its own size, whatever the tab's "Detection resolution".
- From plus.9, untick **Show labels**, next to **🔍 Run detection preview**, to see only the boxes and masks: the class names, scores and numbers drawn on the picture are hidden, on the single-tab preview, where the boxes can still be clicked, and on **🔁 Combine all tabs**. Tick it again to see them. It is on by default, and the browser remembers your choice, one for every tab of txt2img and img2img. The download button of the preview always saves the picture with the labels; from the stable plus.9 it stays visible and clickable also with Show labels off or a box in the top right corner (in the betas, the picture without labels hid it). If the picture without labels cannot be made, the switch changes nothing.
- 🟡 From plus.10 beta 1, each label ends with the share of the image that its box takes: `#1 face 12.3%` on the single-tab preview, `#1 1:face 0.90 12.3%` with **🔁 Combine all tabs**. It is the value that "Mask min area ratio" and "Mask max area ratio" compare, written as a percentage: 12.3% is 0.123, and 0.4% is 0.004, so you can read on the preview what to set on those sliders. It always has one decimal, which is one step of the sliders, and it is cut down, never rounded up: a box at 11.68% reads `11.6%`. So the value read on a box, set as "Mask min area ratio", keeps that box; to keep it with "Mask max area ratio", set one step more (11.6% is a maximum of 0.117). It reads `100%` for a box as large as the image and `<0.1%` for a box smaller than one step of the sliders. On a picture too narrow for a whole label (a long class name on a narrow, tall picture), the class name is cut, as in `#1 left_ey... 3.0%`, or left out, and with "🔁 Combine all tabs" the confidence is left out too, so the number and the share can still be read. On a very small picture, less than about 100 pixels wide, even those can run past the edge. Unticking **Show labels** hides it with the rest of the label. The previews in the results gallery and the files of "Save mask previews" do not show it. It is always on and has no option.
- 🟢 To see the boxes of every generation without running the preview, turn on Settings → ADetailer → "Show detection previews in the results gallery" (see [Settings](#settings)).
- "Enable ADetailer" and "Enable this tab" do not need to be ticked. If a generation is running, the preview waits for it to finish.
- 🟢 When the WebUI is started with `--use-cpu adetailer`, or (except on macOS) with `--lowvram`, `--medvram` or `--medvram-sdxl`, the preview runs YOLO detectors on the CPU, as generation does. Of these flags, Forge Neo has only `--lowvram`. In plus.7.4 and plus.8 beta 1 the preview always uses the graphics card. From plus.9, the preview and generation also run them on the CPU when the whole WebUI does: with `--use-cpu all` (AUTOMATIC1111, Forge, reForge), `--cpu` (Forge Neo) or `--always-cpu` (Forge, reForge), also on macOS, and, except on macOS, with Forge Neo's `--novram`.
- 🟢 With **🔁 Combine all tabs**, a tab whose detector fails shows the reason, and the status reads "Preview failed" when every tab fails.
- 🟢 On Forge and Forge Neo, the MediaPipe detectors find faces in a GIF too.
- 🟢 When the preview's font cannot draw a class name (Chinese or Japanese names, for example, on AUTOMATIC1111, classic Forge or reForge when no TrueType font is found, as on macOS), that number stays plain (`#1`), and the combined preview leaves the name out of its label instead of failing. On other systems, such as Windows, the font has no Chinese or Japanese letters, so such a class name shows as empty boxes after the number; the number itself is always readable.
- From plus.8, a number or label that would cover one drawn before it moves next to its box: above it, into another corner of it, below it or beside it. This works for the single-tab numbers and for the labels of **🔁 Combine all tabs**, and a number cut off by the edge of the image is moved inside it. A number that has room stays where it was; one moved above its box can hide the class and confidence that a YOLO detector writes there. With a very small face in a large image the numbers can still partly overlap, and with more than 40 numbers around one box, as with hundreds of boxes at a confidence near 0, a number stays where it was, so the preview stays quick. In plus.7.4, beta 1 and beta 2 a number drawn later can cover an earlier one.
- From plus.8, the labels of **🔁 Combine all tabs** are drawn on top of every box outline, so the edge of another box no longer cuts through them; a label can cover a short stretch of another box's outline instead. In plus.7.4, beta 1 and beta 2 the outline of a box drawn later can run across an earlier label. The numbers of the single-tab preview are always drawn on top; the class and confidence that a YOLO detector writes on that preview are drawn by the detector itself, and the outline of a box it draws later can still cross them.
- From plus.8, the text of every number and label sits inside its dark or coloured background at every picture size. The text grows with the picture's height, and in plus.7.4, beta 1 and beta 2 it hangs below its background on tall pictures: by about 5 pixels on a picture 2048 pixels high and about 15 on one 4096 pixels high, depending on the font.

### Run ADetailer on an image

Detail a picture you already have with this tab's settings, without generating it again (requested in [#4](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues/4)).

**Where:** in every tab of txt2img and img2img, the "Run ADetailer on an image" section below "Detection preview".

**How to use:**

1. Set up the tab: detector, classes, prompt and the other settings.
2. Drop or paste the picture into **Input**.
3. Click **✨ Run ADetailer on this image**. The retouched picture appears in **Result**: click it to see it full size, or use its download button.

**Good to know:**

- The run uses the tab's detector, class filter, "Inpaint only these detections", prompt, LoRAs, mask and inpainting settings, and its separate checkpoint, VAE and text encoder when set. "Enable ADetailer" and "Enable this tab" do not need to be ticked, and "Apply only on hires.fix" does not apply here.
- There is no main prompt here, so write the prompt in the tab: a blank prompt inpaints with an empty prompt. The prompt and negative prompt saved in the picture are not read: every run starts with an empty prompt and negative prompt, and only the tab's are used. 🟡 From plus.10 beta 2, Settings → ADetailer → "Run ADetailer on an image uses the prompt saved in the image" (off by default) changes this: the prompt and negative prompt saved in the picture become the main ones, so a blank prompt or `[PROMPT]` uses them. To detail pictures each with its own prompt, see [Many pictures, each with its own prompt](#many-pictures-each-with-its-own-prompt). Steps, CFG scale and sampler come from the tab's "Use separate steps", "Use separate CFG scale" and "Use separate sampler" options; otherwise the run uses 28 steps, CFG scale 7, the WebUI's first sampler (DPM++ 2M on AUTOMATIC1111 and Forge Neo) and the Automatic scheduler, and on Forge and Forge Neo the default Distilled CFG Scale, 3.5. 🟡 From plus.10 beta 3, on Forge Neo the run takes instead the img2img Sampler, Scheduler, Steps, CFG and Distilled CFG (Shift, for the models that use one) of the preset in Settings for the family of its model, as Neo's img2img tab does: the main model (from the header of its `.safetensors` file while the WebUI has not loaded it yet, for example right after picking it), or the tab's own checkpoint when the header of its `.safetensors` file says it is SD 1.5, SDXL or Flux. SD 2, Chroma and the families without a preset keep the values above. With a Flux model on a WebUI without such a preset, such as classic Forge (not tested live), it uses 20 steps, CFG scale 1, Euler and the Simple scheduler. A sampler or scheduler the WebUI does not list is left out, and the tab's own options still come first ([issue #9](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues/9)). The status under the result then says the values the pass used, read from the parameters the WebUI wrote on it (those of the last region redrawn), for example "✅ ADetailer pass complete. Used: denoise 0.6, 20 steps, CFG scale 1, Euler (Beta)." Tested live on Forge Neo 2.33 with a Flux and an SDXL main model, the status included; the other families, a tab checkpoint and classic Forge are covered by the offline tests only. For another guidance-distilled or few-step checkpoint, such as SDXL Lightning or Turbo, tick those options and set the model's own values. A folder batch uses the same settings.
- Every run picks new random seeds, so clicking again gives a new variation.
- In img2img, "Skip img2img" gives a similar result; this tool does it in one click, in txt2img too.
- **💾 Save result to outputs** (off by default) also saves the result in an `ADetailer-Inpaint` folder (see [Output folders](#output-folders)). From plus.8 these results record the detailer pass's parameters, which PNG Info and Send to read; in plus.7.4 and beta 1 a PNG records the word "None". With the WebUI's "Write infotext to metadata of the generated image" setting off (on Forge Neo: "Write infotext to metadata of generated images"), they record none.
- From plus.8 the parameters of a result (the file saved to `ADetailer-Inpaint`, the picture downloaded from **Result**, and the PNG, JPEG and WebP copies a folder run saves next to their sources) also record the tab's ADetailer settings under the tab's own number, as a generation in which only that tab runs records them, also when "Enable this tab" is off. PNG Info and Send to then switch ADetailer and that tab on, switch the other tabs off and restore the tab's setup, together with the detailer pass's steps, sampler, seed and size. With the WebUI's "Write infotext to metadata" setting off none of them records parameters (see above), and a copy whose parameters cannot be written is saved without them (see Batch a whole folder, below). In beta 2 the file saved to `ADetailer-Inpaint` and the Result picture record only the detailer pass, so pasting them leaves the ADetailer tabs as they were; in plus.7.4 and beta 1 so does the Result picture of a single run that is not saved, while a saved PNG records the word "None" (see above). In plus.7.4, beta 1 and beta 2 the copies next to their sources record no parameters at all (see Batch a whole folder, below).
- It works while [Manual mode](#manual-mode) is on. If a generation is running, the run waits for it.
- **To stop a run:** the WebUI shows its Interrupt and Skip buttons only while its own Generate runs, so they stay hidden here. Press **Alt+Enter**, the WebUI's Skip shortcut, with the txt2img or img2img tab open: the region being redrawn is thrown away and the picture comes back unchanged ("Cancelled — image unchanged."), and a folder run stops before the next file. Esc does nothing here. With the WebUI started with `--api`, a request to `/sdapi/v1/interrupt` or `/sdapi/v1/skip` stops it too. Clicking Generate also shows the Interrupt button, but that generation then runs as soon as this job ends.
- The status line says what happened: done, nothing detected, detections found but none left to inpaint (because of the detection numbers, the mask filters or `[SKIP]`), cancelled, failed with a NaN error (AUTOMATIC1111, classic Forge and reForge), or from plus.9 failed with another error while inpainting, with the reason of the first one. 🟡 From plus.10 beta 2 it also says when the detector could not be used at all (a model that is not a detector, or a YOLO-World or YOLOE detector that could not take the typed classes), with the reason, and, on Forge and Forge Neo, when ADetailer did not run because the tab was skipped for a checkpoint that cannot be loaded with the VAE file of its pass (see [Separate checkpoint, VAE, text encoder and CLIP skip](#separate-checkpoint-vae-text-encoder-and-clip-skip)); in both cases there is no result, and a folder run counts the file as failed.
- 🟢 When some regions are detailed and others fail with a NaN error (from plus.9, also with another error), the status also says how many were left unchanged. A folder run still counts such a file as detailed.
- Rotated phone photos are detected upright. For a large picture, the regions are redrawn at a working size of at most 1024 pixels on the longest side, to avoid running out of memory, unless the tab uses "Use separate width/height", or "Inpaint only masked" with "Scale inpaint to bbox" or Settings → ADetailer → "Try to match inpainting size to bounding box size". With "Inpaint only masked" on (the default) the result keeps the picture's full size (in plus.7.4 and beta 1, only while the WebUI's "Overlay original for inpaint" setting (on Forge Neo: "For inpainting, overlay the resulting image back onto the original image") is on; from plus.8 the region is pasted back anyway, see [Known issues of plus.8](#known-issues-of-plus8)); with it off, the result comes back at the smaller working size.
- The Size in a result's parameters (in the file saved to `ADetailer-Inpaint`, in the picture downloaded from **Result**, or, from plus.8, in a PNG, JPEG or WebP copy saved next to its source) is the working size the regions were redrawn at, not the picture's own size, as the WebUI records the Width and Height of an img2img inpaint: the picture's size, scaled down to 1024 pixels on the longest side when it is larger (keeping its proportions), then rounded down to a multiple of 8, and at least 64. So an 853x480 picture records 848x480; with "Inpaint only masked" on (the default) it keeps its 853x480 pixels, with it off it comes back at 848x480. With "Use separate width/height", or with "Inpaint only masked" on and "Scale inpaint to bbox" or Settings → ADetailer → "Try to match inpainting size to bounding box size", it is the size that option gives instead. With several regions, the Size, prompt, seed and denoising strength are those of the last region redrawn. Pasting the parameters into txt2img puts that size into Width and Height. From plus.8 the working size of a larger picture is exactly 1024 pixels on its longest side, so a 1288x966 picture records 1024x768; in plus.7.4, beta 1 and beta 2 it is 1016 for some sizes, such as 1016x768 for that picture.
- 🟢 The transparent parts of a picture, such as a cut-out PNG, are filled with the WebUI's img2img background colour, the "With img2img, fill transparent parts of the input image with this color" setting (on Forge Neo: "For img2img, fill the transparent parts of the input image with this color"), white on AUTOMATIC1111 and grey on Forge Neo by default, before the pass, as img2img does, also in a folder batch. In plus.7.4 and beta 1 they turn black, or the colour they hold. 🟡 From plus.10 beta 2, Settings → ADetailer → "Run ADetailer on an image and Skip img2img keep the transparency of the picture" (off by default) gives the result the picture's transparency back after the pass, also when nothing was detected or the run was cancelled; with "Inpaint only masked" off, a result at the smaller working size gets it stretched to its size. A picture whose pixels are all opaque stays as before, and the parameters of the result do not change. The file saved to `ADetailer-Inpaint` keeps the WebUI's file format ("File format for images", on Forge Neo "Image Format"), so it has the transparency only in a format that has it: PNG, and WebP on Forge Neo. With JPEG, or WebP on AUTOMATIC1111, the status adds "The saved file has no transparency" and the console says so; set the format to png to keep it. Tested live on AUTOMATIC1111 and Forge Neo (PNG on both, WebP on Forge Neo). The same option holds in img2img with "Skip img2img": see the notes on Skip img2img in [Upstream options](#upstream-options).
- The run leaves the WebUI's record of your last generation as it was, so the WebUI's paste button with an empty prompt still brings back your last generation. It does not change the remembered settings either.
- 🟢 A tab's separate CLIP skip, VAE and (on Forge and Forge Neo) text encoder are always put back after the run, also when those settings were never saved on the Settings page.

### Batch a whole folder

Detail every picture in a folder in one click, with the tab's settings.

**Where:** in "Run ADetailer on an image", the box "Or batch a whole folder (optional)".

**How to use:**

1. Paste the path of a folder into the box.
2. Tick **📁 Save results in the source folder instead** if you want each result next to its original.
3. Click **✨ Run ADetailer on this image**.

**Good to know:**

- Every detailed picture is saved, whether or not **💾 Save result to outputs** is ticked: in the `ADetailer-Inpaint` folder or, with "📁 Save results in the source folder instead", next to its source as `name-ad` (`photo.png` gives `photo-ad.png`). A picture where nothing was detected is counted as left unchanged and is not written. Originals are never overwritten: when the name is taken, the result becomes `name-ad-1`, `name-ad-2` and so on.
- With that option, files whose name already ends in `-ad` or `-ad-<number>` count as earlier results and are skipped, so running the folder again does not detail the results.
- PNG, JPEG, WebP, BMP and TIFF (`.tif` or `.tiff`) files are read, from the folder itself, not from its sub-folders. A folder path wins over a picture dropped into **Input**.
- Each file is processed on its own: an unreadable file is skipped and counted, and a file whose run fails (for example with CUDA out of memory) is counted as failed; the status gives the reason of the first failure, and the console has the rest.
- A missing detector stops the batch at the first image that opens, with "Pick a detector model first." and nothing written; if no image in the folder can be opened, the batch is reported as failed instead, with every file skipped as unreadable.
- The status line sums up the run: "Batch done", "Batch failed" when nothing could be written, or "Batch interrupted" when you stop it, which ends the run before the next file (press Alt+Enter: see "To stop a run" in [Run ADetailer on an image](#run-adetailer-on-an-image)). **Result** shows the first 30 pictures.
- A generation you start during a folder run waits until the run ends.
- 🟢 On a WebUI started with `--hide-ui-dir-config`, the option for a WebUI shared with other people, the folder box is switched off, as the WebUI's own folder batches are: a path in it gets "⚠️ Folder batch is disabled: the WebUI was started with --hide-ui-dir-config." and nothing is read or written, even when a picture is dropped into **Input**. Empty the box to detail a single picture. In plus.8 and in the three betas of plus.9 the box ignored that option. The two folder options of Settings → ADetailer are hidden there too: see the [Settings table](#settings).
- With "📁 Save results in the source folder instead", a result that cannot be written next to its source is counted as detailed but not saved, and the console says why.
- 🟢 A copy saved next to its source keeps the source's colour profile, such as Display P3 or Adobe RGB, so it looks like the original in colour-managed viewers (a greyscale or CMYK profile does not fit the RGB copy and is left out; results in `ADetailer-Inpaint` are saved by the WebUI, which writes no profile). The copy keeps its source's format: a JPEG at quality 95, a WebP lossless when its source is lossless and at quality 95 otherwise, and PNG, TIFF and BMP lossless. In plus.7.4 and beta 1 the copy has no colour profile, and every WebP copy is lossy at quality 80, so a lossless source loses detail over the whole picture.
- From plus.8 a copy saved next to its source records the result's parameters, as the WebUI records them in its own pictures: in the PNG text, or in the EXIF comment of a JPEG or WebP, which PNG Info reads. BMP and TIFF copies record none, like the WebUI's own BMP and TIFF pictures, and with the WebUI's "Write infotext to metadata of the generated image" setting off (on Forge Neo: "Write infotext to metadata of generated images") none does. When they cannot be written, for example in a JPEG copy with parameters longer than about 32,000 characters, the copy is saved without them and the console says so. The source picture is never changed. In plus.7.4, beta 1 and beta 2 these copies record no parameters at all.
- 🟢 The status also counts the files skipped as earlier results. In plus.7.4 and beta 1 they are left out without a word.
- 🟢 The folder path is never saved as a startup default by Settings → Defaults. In plus.7.4 and beta 1 it can be, and every single-picture run then details the whole folder: clear the box before you use Settings → Defaults.
- Every file starts with an empty prompt and negative prompt, as a single run does: the prompt saved in each picture is not read, and only the tab's prompt and negative prompt are used. 🟡 From plus.10 beta 2, with Settings → ADetailer → "Run ADetailer on an image uses the prompt saved in the image" on, each file gets the prompt and negative prompt saved in it, and the status counts the files that have none. See the next section.
- 🟡 From plus.10 beta 2, with Settings → ADetailer → "Run ADetailer on an image and Skip img2img keep the transparency of the picture" on, each result gets its picture's transparency back (see [Run ADetailer on an image](#run-adetailer-on-an-image)). With "📁 Save results in the source folder instead", PNG, WebP and TIFF copies keep it; a copy in a format without it, such as that of a transparent picture whose file name ends in `.jpg` (a PNG inside), is saved filled, as with the option off, instead of failing. The summary counts the results saved without their transparency, here and in `ADetailer-Inpaint`, and the console gives their number in one line, without file names.

### Many pictures, each with its own prompt

New guide in plus.10 beta 1. Detail many pictures you already made, each with the prompt it was made with. 🟡 From plus.10 beta 2 the simplest way is an option of "Run ADetailer on an image". The img2img Batch tab with "Skip img2img" is the other way: it also works in the stable release, where "Run ADetailer on an image" and its folder batch start every picture with an empty prompt and negative prompt and use only the tab's.

**🟡 With "Run ADetailer on an image" (from plus.10 beta 2):**

1. In Settings → ADetailer, tick "Run ADetailer on an image uses the prompt saved in the image" and click **Apply settings**.
2. In the ADetailer tab, set up the detector and the other settings. Leave the prompt and negative prompt empty, so each picture uses its own (or put `[PROMPT]` where it should go).
3. In "Run ADetailer on an image", paste the folder's path into "Or batch a whole folder (optional)", or drop one picture into **Input**.
4. Click **✨ Run ADetailer on this image**.

**Good to know:**

- The prompt and negative prompt are read from the parameters the WebUI saved in each picture: the PNG text, or the EXIF comment of a JPEG or WebP, the text PNG Info shows. A text counts as saved parameters only when its last line is the WebUI's parameters line, which starts with "Steps:"; if a PNG's EXIF comment is blank or written by another program, its own PNG text is read instead. A JPEG with an EXIF rotation (a camera's portrait photo, or a picture turned in a viewer) is read too, also where PNG Info cannot read it. LoRAs written in them are used, as in a normal generation, and so are `[PROMPT=words]` and "Use LoRAs from main prompt".
- Only the two prompts are taken. The seed, steps, sampler and size are not: the run picks new seeds, and the steps and sampler come from the tab (see [Run ADetailer on an image](#run-adetailer-on-an-image)). The text of a Style is already part of a saved prompt, so it is used too; the Styles selected in the WebUI are not applied. The WebUI's "Disregard fields from pasted infotext" setting (on Forge Neo: "Ignore fields when reading infotext") does not apply here.
- A picture without saved parameters, such as one pasted from the clipboard, cropped in the Input box or saved by another program, keeps an empty main prompt. The status line and the console say so, and the summary of a folder batch counts those pictures.
- Tested live on AUTOMATIC1111 and Forge Neo, on a single picture and on a folder of four.

**With the img2img Batch tab:**

This uses options that already exist, also in the stable release, and can take each picture's seed, sampler and steps too. The recipe is checked in the code, not tested live yet.

1. In img2img, open the Batch tab: add the pictures in "Upload", or type their folder in "From directory" → "Input directory", with a folder for the results in "Output directory". Leave "Inpaint batch mask directory" empty: ADetailer does not support "Skip img2img" with inpainting.
2. Turn on the PNG info option, which reads the prompt and other parameters saved in each picture: on AUTOMATIC1111, open the "PNG info" section and tick "Append png info to prompts"; on Forge Neo, switch on "Append PNG Info". Under "Parameters to take from png info", tick "Prompt" and "Negative prompt" (none is ticked by default), and "Seed", "Sampler" and "Steps" if you want those of each picture too.
3. Leave the main prompt and negative prompt empty: the WebUI adds each picture's prompt after the text in those boxes.
4. In the ADetailer tab, leave the prompt empty, so the detailer pass uses each picture's prompt (or put `[PROMPT]` where it should go), and tick "Skip img2img".
5. Click Generate.

**Good to know:**

- With "Skip img2img" the img2img pass is left out, and ADetailer details each picture as it is. When "Steps" and "Sampler" are ticked in the PNG info option, the detailer pass uses each picture's own steps and sampler, unless the tab uses "Use separate steps" or "Use separate sampler".
- The same "Skip img2img" works with a single picture in img2img, with the prompt you type or paste there.

### Manual mode

Stop ADetailer from running after every generation, while every tab keeps its settings.

**Where:** Settings → ADetailer → "Manual mode — don't auto-run ADetailer after generation" (off by default).

**How to use:** turn it on, generate your drafts quickly, then detail only the pictures you keep with [Run ADetailer on an image](#run-adetailer-on-an-image), which works while manual mode is on. You can also send a picture to img2img and turn manual mode off. Turn it off to go back to automatic detailing.

**Good to know:**

- Pictures come out untouched, and at every generation that ADetailer would otherwise have detailed, the console says that manual mode is on (beta 1 also said it with ADetailer switched off; plus.8, like plus.7.4, no longer does).
- With Skip img2img ticked, manual mode keeps your normal img2img generation, with its own size, steps and sampler.
- 🟢 The option is read once when a generation starts (in an X/Y/Z grid, when each cell starts). Switching it during a running generation, including the img2img Batch tab and Loopback, takes effect from the next generation. In manual mode, a grid cell's picture no longer carries the previous cell's ADetailer parameters.

### Save intermediate step images

Keep a copy of the picture after each ADetailer pass, so you can go back when a later pass spoils something.

**Where:** Settings → ADetailer → "Save intermediate step images (one per tab pass)" (off by default).

**How it works:**

- One file per tab that detailed something: `…-ad-step-1` after the 1st tab, `…-ad-step-2` after the 2nd, and so on, each with the result of every tab up to that one.
- In a tab with "Process classes sequentially", one file per class pass: for example `…-ad-step-1-1-face` (face fixed), then `…-ad-step-1-2-hand` (face and hand fixed).

**Good to know:**

- The files go into the `adetailer-steps` sub-folder, not next to the final pictures (see [Output folders](#output-folders)).
- Two more options of Settings → ADetailer save extra files there: "Save mask previews" (the detected regions; in sequential mode each class pass saves its own mask preview, named like its step file, such as `…-ad-preview-1-1-face`) and "Save images before ADetailer" (`…-ad-before`, the picture before any pass).
- Each file records the parameters of its own picture, also for the 2nd and later pictures of a batch, so PNG Info or Send to txt2img on it brings back that picture's seed and prompts.
- If you press Skip or Interrupt during a sequential tab, the step files of the classes already done stay on disk, while the final picture goes back to how it was before that tab.
- With Skip img2img, the before image is not saved, as it would be the input picture itself. From plus.8 the step images are saved there too; in plus.7.4 and beta 1 they are not.

### Save inpaint masks

New in plus.9. Keep the black-and-white mask each region is inpainted with, to check what ADetailer redrew or to use the mask elsewhere.

**Where:** Settings → ADetailer → "Save inpaint masks" (off by default).

**How it works:**

- One PNG file per region inpainted, in the `adetailer-steps` sub-folder (see [Output folders](#output-folders)), whatever image format the WebUI saves in: a lossy format would blur the mask's edges.
- The files end in `-ad-mask-N-K`: N is the tab and K the region's number in that pass, counted in the order the regions are inpainted, for example `…-ad-mask-1-1` and `…-ad-mask-1-2` for two faces found by the 1st tab. K can differ from the numbers of the Detection preview.
- In a tab with "Process classes sequentially", each class pass names its masks like its mask preview, for example `…-ad-mask-1-1-face-1`.

**Good to know:**

- The mask is the one img2img gets: after the mask offset, erosion / dilation, merge mode and the mask filters, at the size of the picture the pass works on. "Inpaint mask blur" is added later by img2img, so it is not in the file.
- A region that is skipped (for example with `[SKIP]`), fails or is cancelled saves no mask, so its number is missing.
- "Run ADetailer on an image" and folder runs save the masks too, numbered with the tab they were started from.
- Each file records the parameters of its picture, like the step images. If a mask cannot be saved, the console says so and the pass goes on.

### Output folders

Where the files ADetailer saves end up.

| Files | Saved when | Folder |
| --- | --- | --- |
| Final pictures | always, by the WebUI | the WebUI's usual output folder, unchanged |
| Step images (`-ad-step`), mask previews (`-ad-preview`), before images (`-ad-before`), inpaint masks (`-ad-mask`, from plus.9) | the matching option in Settings → ADetailer is on | an `adetailer-steps` sub-folder inside the folder of the final picture, for example `txt2img-images/2026-06-03/adetailer-steps/` (for "Run ADetailer on an image", see below) |
| Results of "Run ADetailer on an image" | "💾 Save result to outputs" is ticked | an `ADetailer-Inpaint` folder next to the img2img output folder (by default beside `txt2img-images` and `img2img-images`), in a date sub-folder when the WebUI saves images to sub-folders |
| Results of a folder batch | for every picture that was detailed | `ADetailer-Inpaint`, or next to each source with "📁 Save results in the source folder instead" |

**Good to know:**

- When "Output directory for adetailer images" (Settings → ADetailer) is set, `adetailer-steps` is created inside that folder instead.
- When the WebUI's "Output directory for images" (on Forge Neo: "Output Directory") is set (one folder for every generation), `ADetailer-Inpaint` is created inside that folder, and the step and preview images of "Run ADetailer on an image" go into its `adetailer-steps` sub-folder too. Otherwise those images go into the `adetailer-steps` sub-folder of the img2img output folder (for example `img2img-images/2026-06-03/adetailer-steps/`), also when you run the tool in txt2img.
- If `adetailer-steps` cannot be created (for example because a file with that name is already there), the files are saved in the folder above it and the console says so.
- The final picture never moves, whatever the ADetailer options.
- 🟢 From plus.8, the results in `ADetailer-Inpaint` record the detailer pass's parameters, so PNG Info and Send to read them; their Size is the size the regions were redrawn at, not the picture's (see [Run ADetailer on an image](#run-adetailer-on-an-image)). From plus.8 they also record the tab's ADetailer settings, and PNG, JPEG and WebP copies saved next to their sources record the same parameters.
- From plus.9, the mask previews, step images and masks of "Run ADetailer on an image" and folder runs record their parameters too (the detailer pass, the loaded checkpoint and VAE, and the tab's ADetailer settings); in plus.8 they record none, and the console prints "create_infotext failed" for each one. A `[user]` or `[denoising]` token in the WebUI's pattern for file or folder names no longer prints a traceback for these files and for the results in `ADetailer-Inpaint`.

### ADetailer Guide tab

A built-in guide to the main options and to common problems, in plain language, in its own tab of the WebUI.

**Where:** turn on Settings → ADetailer → "Show the ADetailer Guide tab (top tab bar)" (off by default), then reload the UI. The "ADetailer Guide" tab appears in the WebUI's top tab bar.

**What is in it:** a table of contents and one section per topic: getting started, choosing a detector, detection, mask preprocessing, prompts, inpainting, per-pass overrides, the preview and run-on-image tools, presets and saving, fixing common problems, the special tokens `[SEP]`, `[SKIP]`, `[PROMPT]` and (from plus.9) `[PROMPT=words]`, and what is new in this fork.

**Good to know:**

- With the tab on, the version badge starts with a **📖 Guide** link (visible while the panel is open) that opens the guide.
- A line at the bottom of the ADetailer panel says where to turn the guide on.
- The guide is in English only; it is not translated.
- 🟢 From plus.8 some of its explanations are corrected: what 0 means for Detection resolution, what Merge does in Mask merge mode, where the Detection preview tools are, that the MediaPipe detectors download small model files on first use, and that tabs 2 and later need "Enable this tab".
- 🟡 From plus.10 beta 2 it also explains the colours of the sections, YOLOE detectors, the options "With hires fix, detail with the hires checkpoint" and "Run ADetailer on an image and Skip img2img keep the transparency of the picture", the console warning for a tab's checkpoint of another model family and the tab that is skipped when its checkpoint cannot be loaded with the VAE file of its pass, the models that are not detectors, the console line for a LoRA named twice, and the box "Guide: several subjects, one LoRA each".

### Interface details

Small things you see in the ADetailer panel.

- **Version badge:** while the panel is open, the top right of the ADetailer header shows `ADetailer Ultimate · v<version> · <commit>`, so you can tell at a glance which build you have (a copy installed without git shows only the version).
- 🟢 In a narrow window the version badge shortens its text with "…" instead of covering the title and the tab counter.
- **Tab counter:** a green pill next to the ADetailer title, for example "2x Tabs", counts the tabs with "Enable this tab" ticked. It shows even when the panel is closed and hides at 0; txt2img and img2img count separately. It counts the ticked boxes only: a tab whose detector is `None` is counted but does nothing. On Forge and Forge Neo it also follows Paste settings, Load and Reset from plus.8; before, a fast answer there could leave the old count until something else changed on the page.
- **Only the 1st tab starts on:** on a fresh setup, "Enable this tab" is ticked only in the 1st tab, so one detailing pass runs by default. When you pick a detector in another tab, tick its "Enable this tab" too.
- **Tab layout:** each tab starts with "Enable this tab", then the collapsed "Preset library export / import" section, the preset rows and the Copy and Paste buttons, then the detector with its classes, followed by the sections Inpaint prompts, Detection, Detection preview, Run ADetailer on an image, Mask Preprocessing and Inpainting, and the ControlNet settings.
- 🟡 **Section colours:** from plus.10 beta 2 each section has a coloured bar and an emoji before its title: blue for the settings of the pass (✍️ Inpaint prompts, 🎯 Detection, 🎭 Mask Preprocessing, 🎨 Inpainting), amber for the tools that work on a picture (🔍 Detection preview, ✨ Run ADetailer on an image) and grey for the preset library file (📦 Preset library export / import). The detector and the ControlNet settings sit under their own small labels, "Detector" and "ControlNet". The bar and the emoji are drawn by the extension's style sheet, so the titles stay word for word: nothing moved, and translations, saved settings and the sections you left open keep working. On a translated interface "Detector" stays in English until the Language Diffusion dictionaries have it.
- 🟡 **Guide box:** from plus.10 beta 2 a closed box, "Guide: several subjects, one LoRA each", sits above the tabs, in txt2img and img2img. Its choices are not tab settings: presets, remembered settings, the WebUI's Settings → Defaults and the image parameters do not hold them. The status under its "✨ Fill the tabs" button does not fade: it stays until the next press. See [Several subjects, one LoRA each (guide)](#several-subjects-one-lora-each-guide).
- **Tooltips and status messages:** hover over the fork's buttons, and over the options of a tab, for a short explanation. The messages under the preset and export / import buttons fade after about 4 seconds; those of the Detection preview and "Run ADetailer on an image" stay until the next run. Repeating an action shows its message again (on Forge and Forge Neo, from plus.8 also after a very fast answer, and after an Import).
- **Themes:** labels, hints and messages are readable on both the dark and the light theme.
- **Translations:** with the companion [Language Diffusion](https://github.com/xXIlRizzoXx/sd-webui-language-diffusion) extension the interface can be shown in 10 languages; the options added in recent versions still show in English until the dictionaries are updated, and, from plus.8, on Forge and Forge Neo the translated "Use same …" values of the dropdowns stay translated after a change. See [Translations](#translations).

## Settings

Open the WebUI's **Settings** tab and pick **ADetailer** in the list on the left. Change what you need and click **Apply settings**; for the options that say so, click **Reload UI** too. These options apply to every tab, in txt2img and img2img alike, and the WebUI keeps them with its own settings.

Options marked *(fork)* are added by ADetailer Ultimate; the others come from the original ADetailer, and where this fork changes how one works, the row says so. They are listed in the order the page shows them.

| Option | What it does | Default |
| --- | --- | --- |
| Max tabs | How many detector tabs (1st, 2nd, …) each ADetailer panel has, from 1 to 15. Click Reload UI after changing it. | 4 |
| Show the ADetailer Guide tab (top tab bar) *(fork)* | Adds an "ADetailer Guide" tab next to Settings, with a plain-language guide to the main options and to common problems (in English), and a "📖 Guide" link in the ADetailer header that opens it. Click Reload UI after changing it. See [ADetailer Guide tab](#adetailer-guide-tab). | Off |
| Verbose diagnostic log *(fork)* | Prints a detailed report to the console for every pass: each active tab's settings, what was detected (class, confidence, mask size), the prompt used for each region, and the time and VRAM each step took. It never changes the result. Useful when you report a problem. From plus.8, when the console output goes to a log file or to another program on Windows, the report is no longer lost; before, whole blocks (often the header and every tab's settings) were dropped when they held a character the log could not store. | Off |
| Extra paths to scan adetailer models separated by vertical bars(\|) | More folders where ADetailer looks for detector models (`.pt` files), besides the WebUI's `models/adetailer` folder. Separate the folders with `\|`. Click Reload UI after changing it (from plus.9, or 🔄 next to the detector menu). A model file can contain code, so put there only files from sources you trust. Hidden on a WebUI started with `--hide-ui-dir-config`, where the saved value stays in use (from the stable plus.9). | Empty |
| Output directory for adetailer images | The folder for the extra images of the options below (mask previews, inpaint masks, images before ADetailer, step images). Empty means the WebUI's usual txt2img or img2img output folder. This fork puts these images in an `adetailer-steps` sub-folder there, so the main folder holds only final images (see [Output folders](#output-folders)). Hidden on a WebUI started with `--hide-ui-dir-config`, where the saved value stays in use (from the stable plus.9). | Empty |
| Save mask previews | Also saves a copy of each image with the detected regions drawn on it (file name ending in `-ad-preview-1` for the 1st tab, `-ad-preview-2` for the 2nd, …; the original names them `-ad-preview`, `-ad-preview-2nd`, …). In sequential mode this fork saves one per class pass, named after the class, such as `-ad-preview-1-1-face`. | Off |
| Show detection previews in the results gallery *(fork, from plus.9)* | Adds the picture with the detection boxes of every ADetailer pass (the one "Save mask previews" saves; in sequential mode one per class pass) after the generated images in the results gallery, each with the parameters of its image, so Send to, PNG Info and Save work on it. At most 32 per generation; past that the console says so once, and "Save mask previews" still saves them all. This option does not save them to disk, and API responses do not include them. The "🔍 Run detection preview" button and "Run ADetailer on an image" are not affected. While a script of the Script menu runs, the previews are not kept or shown, because some of those scripts, such as X/Y/Z plot, SD upscale and Prompt matrix, read the results by their position. In an img2img batch from a folder, the WebUI's "Show the first N batch img2img results in UI" limit (on Forge Neo "Show the first N batch of img2img results in UI") counts the previews too. The WebUI's "Reuse seed" button gives the first image's seed for a preview. | Off |
| Hide detection boxes from the live preview *(fork, from plus.9)* | While ADetailer works, the WebUI's live preview, and the progress API, no longer show the picture with the boxes it detected: they keep their last picture until the inpainting steps replace it. The "🔍 Run detection preview" button and "Save mask previews" are not affected. | Off |
| Save inpaint masks *(fork, from plus.9)* | Also saves, as a PNG, the black-and-white mask each region is inpainted with (file name ending in `-ad-mask-N-K`: tab N, K-th region inpainted; in sequential mode `-ad-mask-1-1-face-1` and so on). No file for a region that is skipped, fails or is cancelled. See [Save inpaint masks](#save-inpaint-masks). | Off |
| Match the colors of each inpainted region to the picture *(fork, from plus.9)* | After each region is inpainted, matches the colors of its box, with the "Inpaint only masked padding, pixels" around it, to the same area of the picture before the pass, with the WebUI's own color correction, and puts the result back only inside the mask, its edge softened by "Inpaint mask blur". It is the WebUI's "Apply color correction to img2img results to match original colors" for ADetailer only: with that WebUI setting on, the regions are already corrected and this option adds nothing. On AUTOMATIC1111 the WebUI's correction fixes the tint and keeps the brightness of the inpainted region; on Forge Neo it matches the brightness too. Being a setting, it is not written into the image parameters. | Off |
| Detailed region opacity *(fork, from plus.9)* | How much of each inpainted region is kept, from 0 to 100. Below 100, each region is blended with the picture as it was before that region (50 keeps exactly half of the change); at 100 each region is kept as inpainted, as before, and at 0 the picture stays as it was before that region. Written into the image parameters as "ADetailer region opacity" only when below 100; pasted parameters do not set it. With a high denoising strength a faint double outline can show. | 100 |
| Save images before ADetailer | Also saves each image as it was before ADetailer (file name ending in `-ad-before`). To switch it from the top of the page, add `ad_save_images_before` to the WebUI's Quicksettings list (see [Can I switch "Save images before ADetailer" from the top of the page?](#can-i-switch-save-images-before-adetailer-from-the-top-of-the-page)). | Off |
| Save intermediate step images (one per tab pass) *(fork)* | Also saves the image after each tab's pass (`-ad-step-1`, `-ad-step-2`, …) and, in sequential mode, after each class pass (`-ad-step-1-1-face`, `-ad-step-1-2-hand`, …). If a later pass spoils something, the earlier step is still on disk. From plus.8 also with Skip img2img; before, not with it. See [Save intermediate step images](#save-intermediate-step-images). | Off |
| Manual mode — don't auto-run ADetailer after generation *(fork)* | ADetailer no longer runs after each generation, but every tab keeps its settings. Handy for making many drafts quickly: detail the ones you keep with "Run ADetailer on an image", which works while this option is on, or turn the option off and generate again. See [Manual mode](#manual-mode). | Off |
| 🟡 Run ADetailer on an image uses the prompt saved in the image *(fork, from plus.10 beta 2)* | For every picture of "Run ADetailer on an image", also in a folder batch, the main prompt and negative prompt become the ones the WebUI saved in that picture (the PNG text, or the EXIF comment of a JPEG or WebP), so a blank tab prompt or `[PROMPT]` uses them, as in a normal generation. Only the two prompts are taken, not the seed, sampler, steps or size. A picture without saved parameters, or pasted from the clipboard, keeps an empty main prompt, and the status says so. Off: the main prompt is always empty, as before. See [Many pictures, each with its own prompt](#many-pictures-each-with-its-own-prompt). | Off |
| 🟡 Run ADetailer on an image and Skip img2img keep the transparency of the picture *(fork, from plus.10 beta 2)* | For a picture with transparent parts, such as a cut-out PNG, in "Run ADetailer on an image" and its folder batch, the result gets the picture's transparency back; the pass still works on the picture filled with the img2img background colour. The same in img2img with "Skip img2img" ticked, also in its Batch tab; on Forge Neo, a Batch tab file whose sides are not multiples of the WebUI's "Resolution Step" setting (64 by default) is resized by the WebUI first and comes out without its transparency. The saved file keeps the transparency only in a file format that has it (PNG; WebP on Forge Neo); with JPEG, or WebP on AUTOMATIC1111, the saved file has none, and the status (with Skip img2img, the text under the results) and the console say so. Off: the transparent parts stay filled, as before. See [Run ADetailer on an image](#run-adetailer-on-an-image). | Off |
| Apply only selected scripts to ADetailer | On: the detailer pass runs only the scripts listed in the next option, plus the WebUI's own Soft Inpainting and HyperTile, and ControlNet when the tab uses it. From plus.9, reForge's Advanced Model Sampling scripts always run there too (needed by v-prediction and flow models), and so does the reForge-RescaleCFG extension, whose script file has the same name as one of these scripts; in plus.8, add `advanced_model_sampling_script` (and `advanced_model_sampling_script_backported` where reForge has it) to the next option. Off: every script active in your generation also runs in the detailer pass. Either way, on Forge, Forge Neo and reForge a tab with a ControlNet model runs only ControlNet in its pass, without Soft Inpainting, HyperTile, Advanced Model Sampling or the listed scripts. From plus.8, with it off, a tab's "Use separate steps" and "Use separate sampler" still apply; before, the detailer pass took the main generation's steps, sampler and scheduler. | On |
| Script names to apply to ADetailer (separated by comma) | The scripts (from other extensions) that also run in the detailer pass while the option above is on, as a comma-separated list of script names. In plus.8, adding `comments` also removes the prompt comments typed in the ADetailer prompts, with a limit (see [Known issues of plus.8](#known-issues-of-plus8)); from plus.9 they are removed without it. 🟡 From plus.10 beta 1, `dynamic_prompting` (Dynamic Prompts) is left out for each region whose final prompt holds no wildcard or variant, also with the option above off (see [Changed in plus.10 beta 1](#changed-in-plus10-beta-1)). | `dynamic_prompting,dynamic_thresholding,lora_block_weight,negpip,wildcard_recursive,wildcards` |
| Sort bounding boxes by | The order in which the detected regions are detailed: None (the detector's own order), Position (left to right), Position (center to edge) or Area (large to small). With `[SEP]` in the prompt, it also sets which region gets which part of the prompt. The numbers of the Detection preview and of "Inpaint only these detections" always follow the detector's own order. From plus.9, the next option offers more orders and, when it is not Off, is used instead of this one. 🟡 From plus.10 beta 2, the box "Guide: several subjects, one LoRA each" follows this setting without changing it, and with two or more subjects its status warns when it is not on Position (left to right), or when the next option is not on Off: see [Several subjects, one LoRA each (guide)](#several-subjects-one-lora-each-guide). | None |
| More sort orders (overrides Sort bounding boxes by) *(fork, from plus.9)* | More orders for the detected regions: Position (right to left), by the right edge of each box; Position (top to bottom), by its top edge; Area (small to large); or Confidence (high to low), the most confident detection first (equal scores keep the detector's order; with a detector that gives no score per box the order is left as it is). Any choice but Off is used instead of "Sort bounding boxes by", also for the parts of a `[SEP]` prompt; Off keeps "Sort bounding boxes by". The numbers of the Detection preview still follow the detector's own order. It is a separate option so that plus.8 and beta 1 ignore it and use "Sort bounding boxes by". | Off |
| Drop overlapping detections (max overlap, 0 = off) *(fork, from plus.9)* | When two detections of the same class overlap more than this (intersection over union, 0 to 0.95), only the more confident one is inpainted, so a face found twice is inpainted once. A much smaller box inside a bigger one, such as the eyes in a face, is kept (of two boxes one inside the other, the less confident one drops when the smaller's area is more than this share of the bigger one's), and detections of different classes never drop each other (when the detector has class names). It applies after "Inpaint only these detections", so the numbers of the Detection preview keep working; the preview itself still shows every detection. About 0.4 to 0.6 works well: the built-in YOLO detectors already drop boxes of the same class that overlap more than 0.7. | 0 |
| Use same seed for each tab in adetailer | On: every detected region is redrawn with the image's seed. Off: each region gets its own seed, counting up from the image's seed. | Off |
| Power scaling for dynamic denoise strength based on bounding box size | With a value above 0, smaller regions get more denoising and larger ones less, and the tab's "Inpaint denoising strength" is the most any region gets. 0 = off, 1 = linear, 2 to 4 recommended. A tab's own "Dynamic denoise by area" slider *(fork)* is used instead when it is not 0. | 0 |
| Try to match inpainting size to bounding box size, if 'Use separate width/height' is not set | Sets each region's inpaint size from the shape of its box. Off; Strict (SDXL only), which uses the sizes SDXL was trained on; or Free, for any model and any size; from plus.10, also "1 megapixel (SDXL, Flux and newer)" *(fork)*, about 1024×1024 with the shape of the box, for any model. With this fork (from plus.8) it works only with "Inpaint only masked" on, and a tab's "Scale inpaint to bbox" *(fork)* takes its place when that is ticked. | Off |
| Max inpaint size, pixels per side (0 = no limit) *(fork, from plus.9)* | From 0 to 4096 in steps of 64. With "Inpaint only masked" on and "Use separate width/height" off, a region whose inpaint size would have a side longer than this is inpainted smaller, with the same shape: the longer side becomes this value, both sides are rounded down to a multiple of 8 (at least 64), and the console prints a line such as `[-] ADetailer: inpaint size 1536x2048 -> 768x1024 (max inpaint size 1024).` It limits the sizes of "Scale inpaint to bbox", of the option above and the size of a large picture, so a large detection does not run out of video memory. Written into the image parameters as "ADetailer max inpaint size" when it is not 0 and a tab it applies to is on; pasted parameters do not set it. | 0 (no limit) |
| Upscaler for ADetailer's inpaint area *(fork, from plus.9)* | The upscaler that enlarges the area ADetailer inpaints when it is smaller than the inpaint size: the crop of "Inpaint only masked", or the whole picture with a larger "Use separate width/height". With "Inpaint only masked", it also enlarges the result back to the crop's size when the crop is bigger than the inpaint size; in "Run ADetailer on an image" and folder runs with the WebUI's "Overlay original for inpaint" (on Forge Neo: "For inpainting, overlay the resulting image back onto the original image") off, that last step uses the WebUI's own upscaler (on Forge, Forge Neo and reForge with a ControlNet model in the tab, the chosen one, as in a generation). The first choice uses the WebUI's "Upscaler for img2img", as before; another choice uses that upscaler for ADetailer only, and the WebUI's setting comes back after each pass. Left out when the upscaler is no longer installed or the settings are frozen. If the WebUI saves its settings during a pass, the chosen upscaler is saved as "Upscaler for img2img" (the separate CLIP skip, checkpoint and VAE have the same limit). From plus.9, with frozen settings (`--freeze-settings` and the like) the tab's own CLIP skip, checkpoint and VAE are left out too, with a console message, instead of stopping ADetailer for the image. | Use the WebUI's img2img upscaler |
| 🟡 With hires fix, detail with the hires checkpoint *(fork, from plus.10 beta 2)* | When txt2img uses hires fix with a Hires checkpoint other than "Use same checkpoint", the tabs without their own checkpoint detail the upscaled picture with that checkpoint instead of the main one; on Forge and Forge Neo with the hires VAE / text encoder choices and the Hires Distilled CFG Scale, while a tab's own VAE or text encoder still replaces them. Tabs with their own checkpoint keep it. No effect without hires fix, in img2img and in "Run ADetailer on an image"; with frozen settings, the main checkpoint. Off: the main checkpoint, as before. See [Separate checkpoint, VAE, text encoder and CLIP skip](#separate-checkpoint-vae-text-encoder-and-clip-skip). | Off |
| Remember last-used settings between restarts *(fork)* | Saves each tab's settings every time you click Generate, separately for txt2img and img2img, and brings them back at the next start (file `user_state.json` in the extension's folder, see [Your data](#your-data)). A change you never generate with is not saved. Off: tabs start with the defaults, and the saved file is kept for when you turn it on again. See [Remember last-used settings](#remember-last-used-settings). | On |
| Enable ADetailer at startup *(fork, from plus.9)* | Starts the ADetailer panel already switched on in the chosen tabs: Off, txt2img, img2img or Both. Click Reload UI after changing it. With Off, the WebUI's Settings → Defaults decide as before (see [Startup defaults for the panel's checkboxes](#startup-defaults-for-the-panels-checkboxes)); any other choice wins over Defaults for the on or off state of those tabs. API requests are not affected, and "Remember last-used settings between restarts" does not store the on or off state. | Off |
| 🔄 Reset ADetailer settings to defaults *(fork)* | A button at the bottom of the page. After you confirm, it puts every option on this page back to its default and reloads the page; from plus.8, Cancel changes nothing (in plus.7.4, Cancel still resets and saves every option). Max tabs and the Guide tab change only after you click Reload UI. From plus.9, the extra model folders change after Reload UI, after a click on 🔄 next to the detector menu, and the first time ADetailer meets a detector name that is not in the list (it then reads the folders with the current setting): after you change or reset them, click 🔄 or Reload UI, so that the menus and the list Generate uses stay the same. See [Reset](#reset). | – |

Good to know:

- The Reset button resets only this page: your tabs, remembered settings and presets are not touched. To put the tabs back to their defaults, use **🆕 Reset** in the ADetailer panel (see [Reset](#reset)).
- A few options added in recent versions, such as Show the ADetailer Guide tab and Verbose diagnostic log, stay in English in the translated interface until the translations are updated (see [Translations](#translations)).

### Startup defaults for the panel's checkboxes

A few checkboxes of the ADetailer panel are not tab settings, so neither presets nor "Remember last-used settings between restarts" keep them: "Enable ADetailer", "Skip img2img", "💾 Save result to outputs", "📁 Save results in the source folder instead", "🔁 Combine all tabs" and "Overwrite on conflict". The WebUI itself keeps a startup value for each of them, in its `ui-config.json` file, on AUTOMATIC1111 and Forge Neo alike. To change how they start:

1. Set them the way you want. For the ones inside a tab, use the 1st tab. txt2img and img2img are set separately.
2. Open **Settings → Defaults** and click **View changes**. Each row shows a path, for example `…/ADetailer/value` for "Enable ADetailer" or `…/💾 Save result to outputs/value`. Make sure only the rows you want are listed: **Apply** saves every row shown.
3. Click **Apply** and restart the WebUI.

Good to know:

- Each mode keeps one value for all its tabs, taken from the 1st tab.
- With "Enable ADetailer" on by default, the ADetailer panel starts open and ADetailer runs on every generation.
- From plus.9, Settings → ADetailer → "Enable ADetailer at startup" does the same for "Enable ADetailer" without Settings → Defaults: set it to txt2img, img2img or Both and reload the UI. Any choice but Off wins over the value kept here for those tabs.
- 🟢 From plus.8, a folder path typed in "Or batch a whole folder" is never saved as a startup default. Before, it could be, and every single-image run of "Run ADetailer on an image" then detailed the whole folder.
- 🟡 The choices of the box "Guide: several subjects, one LoRA each" (plus.10 beta 2) are not kept there either: they start from their defaults every time.

## Upstream options

These are ADetailer's own options, found in every tab of the ADetailer panel. They work as in the original extension; where this fork changes how one behaves, the table says so. The options this fork adds to the same sections, such as the class filter, the prompt extras, "Apply only on hires.fix", "Detection resolution", "Inpaint only these detections", "Use bbox as mask", "Dynamic denoise by area", "Scale inpaint to bbox" and "Use separate text encoder", are explained in [Using each feature](#using-each-feature).

**Top of the panel and of each tab**

| Option | What it does | Default |
| --- | --- | --- |
| Skip img2img | img2img only, at the top of the panel. Skips the img2img generation and runs only ADetailer on your input image. With this fork (from plus.8) the saved image keeps your own steps, sampler and size, and with "Inpaint only masked" off the image keeps its own size, rounded down to a multiple of 8. From plus.8, transparent parts get the WebUI's img2img background colour instead of black. 🟡 From plus.10 beta 2, an option in Settings → ADetailer gives the result the input's transparency back (see below). | Off |
| Enable this tab (1st, 2nd, …) | Runs this tab. A tab also needs a detector other than `None`. | On in the 1st tab, off in the others (this fork; the original starts with every tab on) |
| ADetailer detector | What the tab detects: faces, hands, people and so on (see [Models](#models)). `None` turns the tab off. From plus.9, the 🔄 next to it reads the detector folders again, for a detector added or removed while the WebUI runs (see [Add your own detectors](#add-your-own-detectors)). | `face_yolov8n.pt` in the 1st tab, `None` in the others (after **🆕 Reset**, `None` in every tab it resets) |

Good to know about Skip img2img (apart from the one-step pass and the last point, these are changes made by this fork, from plus.8):

- In practice the img2img pass is cut to one step. In the img2img Batch tab each file keeps its own steps, sampler and size, and "Use same sampler" uses your sampler.
- With "Inpaint only masked" off, the size is rounded down as the WebUI needs: a 1080x1350 image comes back at 1080x1344, instead of the size of the width and height sliders.
- An API request with one input image per batch image details each image's own input.
- 🟡 From plus.10 beta 2, with Settings → ADetailer → "Run ADetailer on an image and Skip img2img keep the transparency of the picture" on (off by default), an input picture with transparent parts, such as a cut-out PNG, is still detailed with its transparent parts filled with the img2img background colour, and the picture the WebUI shows and saves gets the input's transparency back, right before the WebUI saves it; in the Batch tab and in an API batch, each result gets the transparency of its own input. An opaque input stays as before, and with "Inpaint only masked" off a result of another size gets the transparency stretched to it. The file keeps the WebUI's file format: with one that has no transparency (JPEG, or WebP on AUTOMATIC1111) the console and the text under the results say so once for the job. On Forge Neo, in the Batch tab, a file whose sides are not multiples of the WebUI's "Resolution Step" setting (64 by default) is resized by the WebUI first and comes out without its transparency. The grid the WebUI makes of several results is its own picture: on AUTOMATIC1111 it has no transparency. Tested live on AUTOMATIC1111 1.10.1 with an API request (off, the result is filled; on, its transparency is the same as the input's, and the region is detailed); the Batch tab and Forge Neo are covered by the offline tests only.
- It does not work with a mask from img2img's Inpaint tabs: ADetailer is then skipped for that generation, and the console says so.

**Inpaint prompts**

| Option | What it does | Default |
| --- | --- | --- |
| ADetailer prompt, ADetailer negative prompt | The prompts for the detailer pass. Blank means the main prompt. From plus.8, a prompt of only spaces or line breaks counts as blank, and with a blank or `[PROMPT]` prompt the selected Styles are applied once instead of twice, also when the main prompt holds comments (in the stable release, unless another extension, such as Dynamic Prompts with wildcards, rewrites the main prompts; 🟡 from plus.10 beta 2 also then, except in three rare cases: see [Notes on plus.10 beta 2](#notes-on-plus10-beta-2)). | Blank |

Three special words work in these prompts: `[SEP]` splits the prompt so the 1st, 2nd, … region each get their own part; `[SKIP]` skips a region; `[PROMPT]` puts the main prompt at that spot. See the original ADetailer's [Advanced wiki page](https://github.com/Bing-su/adetailer/wiki/Advanced). A request example for the WebUI's API is on its [REST API wiki page](https://github.com/Bing-su/adetailer/wiki/REST-API).

From plus.9, this fork adds `[PROMPT=words]`, which puts only some parts of the main prompt:

- `[PROMPT=eyes,hair]` puts the parts of this image's main prompt (the parts between its commas, new lines, `BREAK` and `AND`; a comma inside brackets does not split, unless the brackets are left open, as in `i <3 cats`) that name one of these words, with their weights; two parts it brings keep the `AND` or `BREAK` that was between them (`AND` when there were both), otherwise they are joined with a comma (also for a new line). For example, with `close-up, [PROMPT=eyes]` in the tab and `1girl, green eyes, beach` as the main prompt a wildcard produced, the detailer prompt is `close-up, green eyes`. A word matches whole, in any letter case, and `_` counts as a space; the numbers of the weights are not searched, so `[PROMPT=1]` does not find `(green eyes:1.2)`.
- A word with a minus sign drops parts instead: `[PROMPT=-beach,-heels]` puts everything except the parts that name them. With both, as in `[PROMPT=eyes,-dress]`, a part is kept when it names a plain word and none of the minus words.
- With plain words it never brings the main prompt's LoRAs ("Use LoRAs from main prompt" adds them); with only minus words it keeps them ("Strip LoRAs from the detailer prompt" removes them). The token treats hypernetwork tags (`<hypernet:…>`) the same way; the two LoRA options do not act on them. The names inside these tags are not searched.
- When no part matches, the token adds nothing, and the weight or comma around it goes too; when that leaves the prompt, or a `[SEP]` part of it, empty, the console says so once per tab (from the stable plus.9 also when it leaves the regions of a `[CLASS=…]` block empty, and with a token that holds a `[` as typed: in beta 2 and beta 3 such a token lost text in the note or stopped ADetailer for that image). `[PROMPT=]` brings the whole main prompt, like `[PROMPT]`.
- It works in the negative prompt too (with the main negative prompt), in every part of a `[SEP]` prompt, inside `[CLASS=…]` blocks and inside a weight, such as `([PROMPT=eyes]:1.2)`. In a batch, each image uses its own main prompt, with the wildcards already filled in. An X/Y/Z "[ADetailer] Prompt S/R" never rewrites the words written inside the token, so `[PROMPT=eyes]` still looks for "eyes".

Prompt comments (`#`, and on Forge Neo also `//` and `/* */`), which the WebUI removes from the main prompt, are left out of a blank or `[PROMPT]` ADetailer prompt too, because that takes the main prompt. From plus.9, comments typed in the ADetailer prompt, negative prompt and append boxes are removed too, before the detailer pass, following the WebUI's own comment setting, also with "Apply only selected scripts to ADetailer" on; a box that holds only a comment counts as blank and takes the main prompt. On Forge, which removes the comments while it applies the styles, this works from the stable release of plus.9: in its three betas the WebUI removed them only from the finished prompt, so a comment also removed the append text, the main prompt's LoRAs and what came after it on the same line, and a box with only a comment gave an empty prompt. Those comments stay in the image parameters ("ADetailer prompt", "ADetailer negative prompt" and the append boxes), unlike the main prompt's, which the WebUI leaves out (on Forge Neo, unless its "Save Raw Comments" setting is on): pasting brings the boxes back as typed, so take private notes out of them before you share a picture. From the stable release of plus.9, on AUTOMATIC1111 and Forge Neo, a comment written in a selected Style stays out of the detailer pass too: ADetailer then applies the selected Styles itself, so the comment's text no longer reaches the model and a blank or `[PROMPT]` prompt gets that Style once. In plus.8, comments typed in the ADetailer prompt boxes themselves are not removed on AUTOMATIC1111, Forge Neo and reForge (Forge removes them itself): with "Apply only selected scripts to ADetailer" on (the default) the detailer pass runs without the WebUI's comment removal, so their text reaches the model. Adding `comments` to "Script names to apply to ADetailer" removes them, but a comment on the last line of a typed prompt then also removes what is added after it on that line: the append text, the main prompt's LoRAs and what the selected Styles add after the prompt. See [Known issues of plus.8](#known-issues-of-plus8).

**Detection**

| Option | What it does | Default |
| --- | --- | --- |
| Detection model confidence threshold | Only detections the detector is more sure of than this are used. Lower finds more, and more false, detections. | 0.3 |
| Method to filter top k masks by (confidence or area) | What "Mask only the top k" ranks the detections by: Area or Confidence. | Area |
| Mask only the top k (0 to disable) | Keeps only the k largest (or most confident) detections. | 0 |
| Mask min area ratio, Mask max area ratio | Keeps only detections whose area, as a share of the whole image, is between these two values. To leave out small things in the background, try a minimum of about 0.01. 🟡 From plus.10 beta 1, the [Detection preview](#detection-preview) writes this share on each box as a percentage, cut down to one decimal: 12.3% is 0.123. As the minimum that value keeps the box; as the maximum, set one step more (0.124). | 0 and 1 |

**Mask Preprocessing**

| Option | What it does | Default |
| --- | --- | --- |
| Mask x(→) offset, Mask y(↑) offset | Moves the mask sideways and up or down, in pixels, from -200 to 200 (from plus.9, from -512 to 512). With this fork (from plus.8) a moved mask stops at the image border instead of wrapping round to the other side. | 0 |
| Mask erosion (-) / dilation (+) | Shrinks (negative values) or grows (positive values) the mask, from -128 to 128 (from plus.9, from -256 to 256). | 4 |
| Mask merge mode | None: each mask is inpainted on its own. Merge: all masks are joined and inpainted once. Merge and Invert: all masks are joined and everything else is inpainted; with this fork (from plus.8) the pass is skipped when the detections cover the whole image, since nothing is left to inpaint. | None |

They are applied in this order: offset, then erosion or dilation, then merge mode.

From plus.9, these sliders, and "Inpaint mask blur" and "Inpaint only masked padding, pixels" below, go further, for large pictures ([Bing-su/adetailer#111](https://github.com/Bing-su/adetailer/issues/111)). The defaults are unchanged, and presets, pasted parameters, remembered settings and API requests already accepted such values.

**Inpainting**

These work like the options of the same name in img2img's Inpaint tab.

| Option | What it does | Default |
| --- | --- | --- |
| Inpaint mask blur | Softens the edge of the mask, for smoother blending, up to 64 (from plus.9, up to 256). | 4 |
| Inpaint denoising strength | How much the region changes: lower keeps more of the original, higher redraws more. About 0.3 to 0.5 is typical. | 0.4 |
| Inpaint only masked, Inpaint only masked padding, pixels | On: only the region is redrawn, at full resolution, with this many pixels of surroundings (up to 256; from plus.9, up to 512), and then pasted back into the picture. Off: the whole picture is inpainted, and with this fork (from plus.8) it keeps its size, also after hires.fix. From plus.8, the region is always pasted back, also when the WebUI's "Overlay original for inpaint" setting (on Forge Neo: "For inpainting, overlay the resulting image back onto the original image") is off (two limits in plus.8, gone from plus.9: see [Known issues of plus.8](#known-issues-of-plus8)). | On, 32 |
| Use separate width/height (inpaint width, inpaint height) | The resolution the region is redrawn at. | Off, 512 x 512 |
| Use separate steps, Use separate CFG scale | Steps and CFG scale for the detailer pass only. From plus.8, the separate steps apply also with Settings → ADetailer → "Apply only selected scripts to ADetailer" off. | Off (28 steps, CFG 7) |
| Use separate checkpoint, Use separate VAE | Another checkpoint or VAE for the detailer pass only. On Forge and Forge Neo, "Use separate VAE" works with this fork (it had no effect there in the original). | Off |
| Use separate sampler (ADetailer sampler, ADetailer scheduler) | Another sampler, and scheduler where the WebUI has them, for the detailer pass only. From plus.8, also with "Apply only selected scripts to ADetailer" off; and an old sampler name that holds its scheduler, such as "DPM++ 2M Karras", from an older preset, pasted parameters or remembered settings, is shown as the sampler and its scheduler (for the names a restart and a paste do not split in plus.8, see [Known issues of plus.8](#known-issues-of-plus8); from plus.9 they split them as Load does). | Off |
| Use separate noise multiplier | "Noise multiplier for img2img" for the detailer pass only. | Off (1.0) |
| Use separate CLIP skip | CLIP skip for the detailer pass only. | Off (1) |
| Restore faces after ADetailer | Runs the WebUI's face restoration, where your WebUI has one, on the result of the detailer pass. | Off |

How the separate checkpoint, VAE, text encoder and CLIP skip work in this fork, and what plus.8 changes, is in [Separate checkpoint, VAE, text encoder and CLIP skip](#separate-checkpoint-vae-text-encoder-and-clip-skip).

In img2img inpainting (an Inpaint tab, an inpaint Batch mask folder or an API request with a mask), ADetailer does nothing when the mask is empty, and with the WebUI's "Inpaint area" on "Whole picture" it details only the detections that touch the area the WebUI inpaints. From plus.8 a mask drawn on a transparent layer is read from its transparency, as the WebUI reads it, also with Soft inpainting on, where the WebUI keeps its semi-transparent parts; before, ADetailer read its brightness, so it switched itself off, or also detailed the faces outside the painted area.

**ControlNet** (at the bottom of each tab)

These need ControlNet installed, with ControlNet models; without ControlNet the options are greyed out. They work separately from the ControlNet settings of your main generation.

| Option | What it does | Default |
| --- | --- | --- |
| ControlNet model | A ControlNet model for the detailer pass. Models whose name contains inpaint, scribble, lineart, openpose, tile, depth or union, in any letter case, are listed. Passthrough uses the ControlNet settings you made outside ADetailer. From plus.8, on classic Forge the model works with its current built-in ControlNet; before, every region printed a ControlNet error and was detailed without ControlNet (checked in classic Forge's code, not tested live). | None |
| ControlNet module | The preprocessor. Choosing a model lists the preprocessors that fit it and picks the first, unless the tab's preprocessor fits the new model; on Forge, Forge Neo and reForge that first one is "None" in plus.8 (see below), and from plus.9 the model's default preprocessor is picked there instead. With this fork (from plus.8) a saved preprocessor that fits the model is kept when you load a preset or paste settings, and a restored model shows its preprocessor after a restart. From plus.8, a model whose file name writes its type with capitals, such as `OpenPoseXL2`, gets its preprocessors too; before, it got none. | Set by the model |
| ControlNet weight | How strongly ControlNet guides the pass. | 1.0 |
| ControlNet guidance start, ControlNet guidance end | When ControlNet starts and stops guiding, as a share of the steps. | 0 and 1 |

On Forge, Forge Neo and reForge the preprocessor list starts with "None", which hands the picture to ControlNet as it is. From plus.9, choosing a ControlNet model there selects its default preprocessor (for example `openpose_full` or `depth_midas`), as AUTOMATIC1111 does; picking "None" afterwards keeps it, a saved "None" survives a restart, loading a preset or pasting parameters that change the model gives the model's default, and an API request without `ad_controlnet_module` also uses the default. In plus.8, for pose, depth, lineart, scribble and inpaint models, pick the preprocessor yourself there, for example `openpose_full`, `depth_midas`, `lineart_coarse` or `inpaint_global_harmonious`. An API request that leaves the preprocessor out also runs without one there, while AUTOMATIC1111, whose lists have no "None", uses the model's default preprocessor.

## Models

A detector is the model that finds what to fix: faces, hands, people and so on. Each tab uses the detector chosen in its **ADetailer detector** list. Detector files can contain code: use only files from sources you trust (see [Add your own detectors](#add-your-own-detectors)).

### Built-in detectors

| Detector | Finds | Good to know |
| --- | --- | --- |
| `face_yolov8n.pt` | Faces, drawn and realistic | Fast; a good first choice |
| `face_yolov8s.pt` | Faces, drawn and realistic | A little slower, finds more; better for small faces |
| `hand_yolov8n.pt` | Hands, drawn and realistic | |
| `person_yolov8n-seg.pt` | Whole people, with their outline | Segmentation model |
| `person_yolov8s-seg.pt` | Whole people, with their outline | Segmentation model, finds more |
| `yolov8x-worldv2.pt` | Whatever you name | YOLO-World, see [YOLO-World detectors](#yolo-world-detectors) |
| `mediapipe_face_short` | Realistic faces close to the camera | See [MediaPipe detectors](#mediapipe-detectors) |
| `mediapipe_face_full` | Realistic faces | Same result as `mediapipe_face_short` |
| `mediapipe_face_mesh` | Realistic faces, following the face's shape | |
| `mediapipe_face_mesh_eyes_only` | The eyes of realistic faces | |
| `mediapipe_face_features` | Eyes, mouth, nose, eyebrows and face, as separate classes (and the opt-in iris, from plus.9) | Added by this fork |

- The `n` and `s` in a name are sizes: `n` is the fastest, `s` is a little slower and more accurate.
- The `.pt` detectors in this table download by themselves the first time the WebUI starts with ADetailer, from the [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer) page on Hugging Face (the YOLO-World one from [Bingsu/yolo-world-mirror](https://huggingface.co/Bingsu/yolo-world-mirror)). That page also lists how accurate each one is. When Hugging Face cannot be reached, they are fetched from the hf-mirror.com mirror of the same page instead, from the stable plus.9 without your Hugging Face token. If you set the `HF_ENDPOINT` variable to the mirror yourself, in the stable plus.9 the Hugging Face library sends the token there, as it does to Hugging Face; 🟡 from plus.10 beta 1 the token is not sent to hf-mirror.com or its subdomains, and only another host set with `HF_ENDPOINT` still gets it from the Hugging Face library. The `--ad-no-huggingface` option (next point) stops both. In the stable plus.9 they are kept in the Hugging Face download folder (its cache), not in `models/adetailer`. 🟡 From plus.10 beta 1, a built-in detector downloaded for the first time is saved as a normal file in `models/adetailer`, next to your own detectors, and the console line of its download ends with "saved in models/adetailer"; the ones already in the Hugging Face download folder stay there and are used from there, with no copy and no new download. This needs huggingface_hub 0.23 or newer and a `models/adetailer` folder that can be written; without them, or when saving there fails, the detector goes to the Hugging Face download folder as before. The Hugging Face library keeps a small `.cache` subfolder in `models/adetailer` with the state of its downloads: it holds no model, and nothing in it is listed in the menu.
- Without an internet connection, the list shows the built-in detectors downloaded before and your own. To stop these downloads altogether, start the WebUI with the `--ad-no-huggingface` option.
- From plus.9, a built-in detector file with its usual name, such as `face_yolov8n.pt`, put in `models/adetailer` or in an extra model folder is used from there, in its usual place in the list, and a built-in detector downloaded before loads from the Hugging Face download folder first: neither contacts Hugging Face, so at startup the WebUI no longer tries to reach Hugging Face for them, and it starts fine offline. Only a built-in detector found in neither place is downloaded (🟡 from plus.10 beta 1, into `models/adetailer`, as said above). A detector already downloaded is then no longer checked for a newer version on Hugging Face: see [Notes on plus.9 beta 2](#notes-on-plus9-beta-2). In plus.8, every start checks each built-in detector on Hugging Face, and a copy in `models/adetailer` is used only when that check and the download folder both fail.
- More about these model types in the Ultralytics documentation: [YOLOv8](https://docs.ultralytics.com/models/yolov8/#overview) and [YOLO-World](https://docs.ultralytics.com/models/yolo-world/).

### MediaPipe detectors

The five `mediapipe_…` detectors need no file in `models/adetailer`.

- They are made for realistic faces.
- They use two small model files, `face_landmarker.task` and `blaze_face_short_range.tflite` (about 4 MB together). These download by themselves the first time you use one of these detectors, into the `adetailer/_mediapipe_models/` folder inside the extension's folder, and stay there when you update. On a computer without internet, the console names each missing file and the official address it comes from: download it from that address on another computer and copy it into that folder. Do not use copies from other sites. A download that stopped partway, much smaller than the real file, is not used and is downloaded again.
- `mediapipe_face_full` uses the same short-range face model as `mediapipe_face_short`, made for faces within about 2 m of the camera, so both normally give the same result: MediaPipe's newer face detector, which ADetailer uses, has no full-range model. For small or distant faces, use a YOLO face detector such as `face_yolov8s.pt`, with a higher "Detection resolution" if needed.
- 🟢 With `mediapipe_face_short` and `mediapipe_face_full`, a face cut by the left or top edge of the picture gets a box that fits its visible part, instead of one that covers the neck or the background beside it.
- "Detection resolution" has no effect on MediaPipe detectors.
- Only `mediapipe_face_features` has classes (eyes, mouth, nose, eyebrows, face and, from plus.9, iris, used only when picked). Pick them in the CLASSES list, and use "Process classes sequentially" or per-class prompts with them. The other MediaPipe detectors have no class filter.
- If MediaPipe is missing or broken, these detectors find nothing, while the other detectors keep working. From plus.8 the console says so once per session, and why (see [Problems and fixes](#install-and-startup)).
- ADetailer Ultimate keeps MediaPipe on the tested 0.10 versions, because MediaPipe 1.0.1 closes the whole WebUI on macOS.
- From plus.9, MediaPipe no longer starts the audio system when the `sounddevice` package is installed, which on Windows could stop the sound of other programs (see [Install and startup](#install-and-startup)).

### Add your own detectors

> [!WARNING]
> A `.pt` detector file can contain code, not only data. Use only detector files from sources you trust.

1. Copy the `.pt` file into the `models/adetailer` folder of your WebUI. Subfolders work too. ADetailer creates this folder the first time the WebUI starts with it.
2. Restart the WebUI. The detector appears in the **ADetailer detector** list under its file name. From plus.9, you can instead click **🔄** next to the **ADetailer detector** menu: the folders are read again without a restart (the same works after removing a detector). Settings → Reload UI does it too; refreshing the browser page (F5) does not.

Good to know:

- To keep detectors in other folders, list those folders in Settings → ADetailer → "Extra paths to scan adetailer models separated by vertical bars(|)", then restart the WebUI (from plus.9, click Apply settings and then 🔄).
- A detector must be an [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO model that finds boxes (detection) or outlines (segmentation), with a file name ending in `.pt`. 🟡 From plus.10 beta 2, a model of another kind picked as the detector, such as an OBB model (rotated boxes), a classification model or a semantic segmentation model, inpaints nothing, and the console says why in one line, once per generation and tab; the Detection preview and "Run ADetailer on an image" say it in their status. Before, it ended in an error with a traceback. Pose models work as detectors, with the boxes they find.
- A detector trained on several classes lists them in the CLASSES list, so you can choose which ones to detail (see [Class names](#class-names)).
- A detector whose file name contains `-world` is treated as a YOLO-World detector.
- A detector that downloads as a `.zip` file must be extracted first: ADetailer reads only the `.pt` file. From plus.9, the console names the `.zip` files it finds in the detector folders at startup, at every 🔄, and when a detector name that is not in the list makes ADetailer read the folders again.
- A detector made with a newer Ultralytics than the one in your WebUI, such as a recent YOLO12 or YOLO26 model, does not load until Ultralytics is updated. From plus.9, the console and the Detection preview say so (see [Nothing runs or nothing is detected](#nothing-runs-or-nothing-is-detected)).

### Find more detectors

The built-in detectors cover faces, hands and people. For other parts of a picture, add a detector file as in [Add your own detectors](#add-your-own-detectors):

- **More from the same page:** the [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer) page on Hugging Face (Apache-2.0 license), where most of the built-in detectors come from, also has larger face and hand detectors (`face_yolov8m.pt`, `face_yolov9c.pt`, `hand_yolov8s.pt`, `hand_yolov9c.pt`), `person_yolov8m-seg.pt`, and `deepfashion2_yolov8s-seg.pt`, which finds 13 kinds of clothing as separate classes.
- **Community detectors:** Civitai and Hugging Face host many detectors made for ADetailer, for eyes, hair, feet, clothing and more: search those sites for "ADetailer". Read the page first: prefer a detector with many downloads and reviews, check its license, and look at its classes, which the CLASSES list will show.
- **Eyes, mouth, nose and eyebrows** need no file: use `mediapipe_face_features` (see [Face parts detector](#face-parts-detector-mediapipe_face_features)).
- **Anything you can name**, with mixed results: a YOLO-World detector looks for the words you type (see [YOLO-World detectors](#yolo-world-detectors)).

A `.pt` file can contain code, not only data: download detectors only from sources you trust.

### Class names

When a detector knows several classes, the **ADetailer detector CLASSES** list fills with their names, for example face and hand. Pick the classes to detail, or leave the list empty to detail every class. From plus.8, names that come from a preset, pasted parameters or an API request match regardless of case, so `Face` selects `face` (in plus.7.4 a name must match exactly, or it is ignored). How to use the list: [Class filter](#class-filter).

#### Custom class names via sidecar JSON

A detector that does not list its classes, or lists them under names you don't like, can get them from a small text file. Save it next to the detector, in the same folder, with the detector's name plus `.names.json`: for `hands_and_faces.pt`, name it `hands_and_faces.names.json`. The first name is class 0, the second class 1, and so on:

```json
["face", "hand", "eyes"]
```

- These forms work too: `{"names": ["face", "hand"]}`, `{"names": {"0": "face", "1": "hand"}}` and `{"0": "face", "1": "hand"}`.
- From plus.8, the file can be saved as UTF-8 (with or without a byte-order mark) or UTF-16, which is what Notepad's "Unicode" option and Windows PowerShell write. In plus.7.4, save it as plain UTF-8 without a byte-order mark (in Notepad: "UTF-8", not "UTF-8 with BOM" or "Unicode"); other encodings do not work there.
- Other `.json` files next to your detectors, such as the metadata some model managers write, are ignored.
- Class names are looked up once per session: after they have been found for a detector (from this file or from the model), adding or changing this file needs a restart of the WebUI (from plus.9, click 🔄 next to the "ADetailer detector" menu instead, then choose another detector and back to update the CLASSES menu of a tab that shows it; from the stable release, Settings → Reload UI works too).
- On AUTOMATIC1111, a detector without this file lists its classes only after its first detection in plus.8, in a generation or a Detection preview: then switch to another detector and back (from plus.9, it lists them straight away, read from the detector file without loading it). From plus.8, the class filter works in generation either way (in plus.7.4 it is ignored for such a detector until you add the file and restart the WebUI). With the file, the classes are listed straight away, and from plus.8 a file added while the WebUI is running fills an empty CLASSES dropdown without a restart: switch to another detector and back, as long as that detector's CLASSES dropdown is still empty. From plus.9, after you replace a detector with a new file of the same name, click 🔄 next to the detector menu: the new classes are used by the class filter, and the CLASSES dropdown shows them after you switch to another detector and back (from the stable release, Settings → Reload UI also reads the new classes). Classic Forge (checked in its code) and Forge Neo list the classes straight away; reForge is not tested.

#### Backwards compatibility

- The older file name `<model>.json` still works when there is no `<model>.names.json`. The `.names.json` name is better, because some launchers and model managers write their own `<model>.json`.
- An empty class filter details every class, as the original ADetailer does. Images and API requests made with the original ADetailer, and presets saved without a class filter, work as before, and while the class options are not used they are left out of the image parameters.
- For YOLO-World, the classes field works as in the original ADetailer. For other multi-class detectors the same field (`ad_model_classes` in the API) holds the class filter: class names or class numbers, separated by commas. In the API, NOT mode is `ad_model_classes_exclude: true` with the classes to leave out in `ad_model_classes_excluded`, and sequential passes are `ad_classes_sequential: true`.

### YOLO-World detectors

`yolov8x-worldv2.pt`, and any detector whose file name contains `-world`, finds whatever you name.

- A text box, **ADetailer detector CLASSES (YOLO-World / YOLOE)** (before plus.10 beta 2: "ADetailer detector CLASSES (YOLO-World)"), appears next to the CLASSES list, which stays empty for YOLO-World. Type the names separated by commas, for example `person, hand`. Left empty, it looks for the 80 everyday classes it knows by default (the COCO classes).
- From plus.8, each region carries the name it was found with, so `[CLASS=person] … [/CLASS]` blocks in the prompt apply to it.
- From plus.8, "Exclude selected (NOT)" is hidden, because YOLO-World looks only for the names you type (or for its default classes); in plus.7.4 it is shown but has no effect. The Auto class-guard does nothing with YOLO-World.
- Typed names need a one-time extra download of about 340 MB, the CLIP text model, which the Ultralytics library makes the first time a YOLO-World detector runs with them (see [Does it upload my images?](#does-it-upload-my-images)).
- 🟡 From plus.10 beta 2, when the typed names cannot be given to a YOLO-World or YOLOE detector because the `clip` package is missing and the Ultralytics library could not install it by itself (no internet connection, or its automatic install switched off), that tab inpaints nothing and the console says why once per generation and tab, with the pip command to run with the WebUI's Python; the Detection preview and "Run ADetailer on an image" say it in their status. Any other failure of the text model gets one line with the error. Before, a YOLO-World detector stopped ADetailer for the image with a traceback.
- 🟡 From plus.10 beta 2, **YOLOE detectors** with text prompts work the same way: a file whose name contains `yoloe` in any letter case, such as `yoloe-11s-seg.pt`, `yoloe-v8l-seg.pt` or `yoloe-26n-seg.pt`, gets the same text field, "Exclude selected (NOT)" is hidden, and the regions carry the names you type. Left empty, the detector finds the classes saved in its file, and nothing is downloaded. The prompt-free YOLOE detectors, marked `-pf` in their names (such as `yoloe-11s-seg-pf.pt`), stay ordinary detectors with their own CLASSES list. The first time a detection runs with typed names, the Ultralytics library downloads its text model from github.com/ultralytics/assets (`mobileclip_blt.ts`, about 600 MB; for YOLOE-26 `mobileclip2_b.ts`, about 254 MB), and installs the `clip` package from GitHub (ultralytics/CLIP) when it is missing (on Forge Neo). It loads that text model again for every detection with typed names, so these detections are slow. When the names cannot be given to the detector (an Ultralytics too old for YOLOE text prompts, a prompt-free model renamed without `-pf`, or no text model), the console says why in one line and that tab inpaints nothing. Like YOLO-World, a YOLOE detector does not stay loaded for the whole generation: it is read from its file each time. Before beta 2, a YOLOE detector was an ordinary detector with the CLASSES list of its file; the classes picked there now show in the text field. The field's label, example text and tooltip name both YOLO-World and YOLOE, and say what an empty field does for each. Tested live on AUTOMATIC1111 and Forge Neo with `yoloe-26n-seg.pt`.

## Problems and fixes

A few of the most common tips are also in the optional ADetailer Guide tab, under "🛠️ Fixing common problems" (turn it on in Settings → ADetailer → "Show the ADetailer Guide tab (top tab bar)", then reload the UI). Problems still open in the current release are listed in [Known issues](#known-issues). If your problem is not here, please open an Issue (see [For developers](#for-developers)).

Entries marked 🟢 "fixed in plus.8" still happen in plus.7.4 and in beta 1: their tip is the workaround there, where there is one. Entries marked "fixed in plus.9" still happen in v26.2.0+plus.8 in the same way.

### Install and startup

**The ADetailer panel appears twice**

- Two ADetailer extensions are installed (older versions also printed "Error running preload()" at startup). Move the other one's folder (for example the original `adetailer` or `ADetailer-Neo`) out of the WebUI's `extensions` folder, keeping it as a backup, then restart the WebUI completely. See [Coming from the original ADetailer?](#coming-from-the-original-adetailer).

**On macOS, the console says that the MediaPipe detectors are disabled**

- MediaPipe 1.0.1 or newer is installed, and on macOS it closes the whole WebUI, so ADetailer Ultimate does not use it. The console shows the command that installs a working version. This can happen when the WebUI is started with its package installation turned off; otherwise the extension installs the tested 0.10 version by itself when the WebUI starts.

**Every MediaPipe detector finds nothing, on every picture**

- MediaPipe is missing or its install is broken. From plus.8 the console says "MediaPipe could not be loaded", with the reason. If it is missing, restarting the WebUI with its package installation turned on installs it; if it is broken, reinstall MediaPipe in the WebUI's Python environment, as the console suggests. The YOLO detectors are not affected.
- To reinstall it, close the WebUI and run pip with the WebUI's own Python (for example `venv\Scripts\python.exe -m pip` in the WebUI folder on Windows, `venv/bin/python -m pip` on Linux and macOS). First remove it with `pip uninstall -y mediapipe`. Then start the WebUI with its package installation turned on, so ADetailer installs the version it supports, or install that version yourself with `pip install "mediapipe>=0.10.13,<=0.10.99"`, the same command ADetailer runs. On a WebUI that uses NumPy 1, such as AUTOMATIC1111, add `"numpy<2"` to that command, as ADetailer does, or pip can bring in NumPy 2 and break the WebUI. MediaPipe 1.x is not supported.

**The ADetailer panel does not appear after installing**

- Restart the WebUI completely: close its console window or launcher and start it again.
- In Extensions → Installed, check that `adetailer-ultimate` is listed and ticked.
- Keep only one ADetailer (see [Coming from the original ADetailer?](#coming-from-the-original-adetailer)).
- Read the console at startup: if ADetailer prints an error there, report it with the full log (see [Report a problem](#report-a-problem)).

**🟡 The console says that the Python package 'rich' or 'ultralytics' cannot be imported (from plus.10 beta 2)**

- The WebUI was started with its package installation turned off (`--skip-install`), or the install failed. Start the WebUI once without `--skip-install`, or run the pip command of that console line with the WebUI's own Python. Without `rich`, ADetailer cannot load and its panel is missing; without `ultralytics`, the panel loads but the YOLO detectors cannot run, and the line comes at startup and again before the error of a detection. In the earlier versions the only sign is an error such as "No module named 'rich'" in a traceback, or a first generation that fails.

**The built-in detectors are missing from the "ADetailer detector" list**

- The first start had no internet or could not reach Hugging Face, or the WebUI was started with the `--ad-no-huggingface` option. Restart the WebUI once with an internet connection and without that option. Your own detectors in `models/adetailer` are listed either way. You can also download the built-in `.pt` files from the [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer) page on another computer, put them in `models/adetailer` with their names unchanged and restart the WebUI; from plus.9 they are listed in their usual place, without contacting Hugging Face.

**A detector you added is not in the "ADetailer detector" list**

- ADetailer reads only `.pt` files. Many detectors on Civitai download as a `.zip` file: extract the `.pt` file from it into the same folder, then click 🔄 next to the "ADetailer detector" menu (from plus.9) or restart the WebUI. From plus.9, the console names up to 5 `.zip` files it finds in the detector folders at startup, at every 🔄 and when a detector name that is not in the list makes ADetailer read the folders again.
- The file must be in `models/adetailer`, in one of its sub-folders or in a folder listed in Settings → ADetailer → "Extra paths to scan adetailer models separated by vertical bars(|)", and the WebUI must be restarted after you add it (from plus.9, click 🔄 next to the "ADetailer detector" menu instead; for a folder just added to that setting, click Apply settings first; see [Add your own detectors](#add-your-own-detectors)).

**Sound in other programs stops, or Windows lists Python as using the microphone, when a MediaPipe detector runs (fixed in plus.9)**

- This happens when the `sounddevice` package is installed in the WebUI's Python environment: loading MediaPipe also starts the audio system. In plus.8, use a YOLO face detector instead of the MediaPipe ones if this bothers you. From plus.9, ADetailer loads MediaPipe without `sounddevice`, which detection does not need, unless another extension has already loaded MediaPipe or `sounddevice`. Reported upstream in [Bing-su/adetailer#508](https://github.com/Bing-su/adetailer/issues/508).

**AUTOMATIC1111 stops at startup with "The queue is enabled for event … but the queue has not been enabled for the app" (🟢 fixed in plus.8)**

- This happens when a WebUI with Gradio 3 (AUTOMATIC1111, older Forge versions, reForge's main branch) is started with the `--no-gradio-queue` option. In plus.7.4 and beta 1, start it without that option.

**ADetailer stops with an error, or its panel is missing, when a name or a prompt has other characters (🟢 fixed in plus.8)**

- This happens on Windows when the WebUI's console output goes to a log file or to another program, such as a launcher, and a detector, class, folder or file name, or the prompt ADetailer applies, has characters such as Chinese, Cyrillic or an emoji. In plus.7.4 and beta 1, give detectors, classes, folders and files names in plain Latin letters and avoid such characters in the prompts. If you know how to set an environment variable for your WebUI (for example with a `set PYTHONIOENCODING=utf-8` line in `webui-user.bat` on AUTOMATIC1111), that works too.

### Nothing runs or nothing is detected

**ADetailer does not run at all**

- Tick the checkbox on the ADetailer panel's header, and "Enable this tab" in each tab you want.
- Pick a detector: a tab whose detector is `None` does nothing.
- Turn off Settings → ADetailer → "Manual mode — don't auto-run ADetailer after generation".
- In txt2img, "Apply only on hires.fix" skips the tab when hires.fix is off.

**Generate stops a tab with a "not found" error after you load a preset or paste parameters (🟢 fixed in plus.8)**

- The preset or the picture names a detector that is not installed in this WebUI, for example a custom face model, as is common with pictures shared online; that tab and the later ones are then not detailed. Pick an installed detector in that tab, or install the missing one. From plus.8 the tab keeps its own detector instead and the missing one is named: in the Load status, or in the console after a paste, which also switches that tab off.

**In img2img inpainting the console says "img2img inpainting with no mask -- adetailer disabled." although you painted a mask (🟢 fixed in plus.8)**

- The mask holds the painted area in its transparency, as a mask from the Inpaint upload tab, an inpaint Batch mask folder or the API can, and ADetailer read only its brightness; with other such masks it also detailed faces outside the painted area. Use a black-and-white mask without transparency, white where the picture is inpainted. From plus.8 the mask is read as the WebUI reads it.

**"Nothing detected" or "No detections"**

- Run the Detection preview on the same picture to see what the detector finds.
- Lower "Detection model confidence threshold" (0.3 by default).
- Check the class filter: with classes picked in the CLASSES list, the detector looks only for those (with "Exclude selected (NOT)", only for the others).
- "Nothing detected" and "No detections" mean that the detector itself found nothing. The mask filters and the detection numbers act after detection. When they leave nothing, "Run ADetailer on an image" says "Detections found, but the detection numbers, mask filters or [SKIP] left none to inpaint", and a generation leaves the picture unchanged although the Detection preview shows boxes. Then check "Mask min area ratio" and "Mask max area ratio", the numbers in "Inpaint only these detections" (the console then says so) and `[SKIP]` in the prompt.
- Check that the detector fits the picture: the MediaPipe detectors are made for realistic faces, and a hand detector finds no faces.
- `mediapipe_face_features` finds nothing for class names it does not have. Pick the names from its CLASSES list.

**🟡 The console says "tab 1 inpaints nothing", with the name of the detector (from plus.10 beta 2)**

- "… is an OBB model", "a classification model" or "a semantic segmentation model": the file picked as the detector is not a detector. ADetailer needs a detection or segmentation model (a pose model works too): pick another file. The earlier versions stop with "'NoneType' object has no attribute 'xyxy'" instead.
- "… could not take the typed classes: the Python package 'clip' is missing": a YOLO-World or YOLOE detector needs that package for the names you type, and the Ultralytics library could not install it by itself. Run the pip command of that line with the WebUI's own Python, with an internet connection, then generate again. Or empty the classes field: the detector then looks for its own classes.
- The Detection preview and "Run ADetailer on an image" give the same reason in their status.

**A detector fails with "Can't get attribute 'C3k2'" or a similar name**

- The detector was made with a newer Ultralytics than the one in your WebUI, as can happen with recent models such as YOLO12 or YOLO26. Update Ultralytics in the WebUI's Python environment, or choose another detector. From plus.9, the console and the Detection preview status say so, naming the file and the installed Ultralytics version (in the console the message can appear twice).

**Small or distant faces are missed**

- Raise "Detection resolution (0 = default)" in the tab above the detector's own size, for example to 1024 for a built-in detector (they use 640). It works with YOLO detectors and uses more VRAM and time.
- Lower the confidence threshold, or use `face_yolov8s.pt` instead of `face_yolov8n.pt`. `mediapipe_face_short` and `mediapipe_face_full` are made for faces close to the camera.

**The CLASSES list is empty**

- The MediaPipe detectors other than `mediapipe_face_features` have no classes, and YOLO-World uses the text box next to the list.
- From plus.9, a detector without a `<model>.names.json` file lists its classes straight away on AUTOMATIC1111 too. In plus.8, on AUTOMATIC1111, such a detector lists its classes only after its first detection: run a generation or a Detection preview with it, then switch to another detector and back. Or add that file (see [Class names](#class-names)). In plus.7.4 the list stays empty even after a detection: add the file and restart the WebUI. Classic Forge (checked in its code) and Forge Neo list them straight away.

**"Inpaint only these detections" details the wrong regions, or all of them**

- The numbers are the ones the Detection preview shows (#1, #2…). They match only when the preview and the run use the same picture, detector, class filter, confidence threshold and detection resolution.
- The field is not used while "Process classes sequentially" runs its passes.
- 🟢 In beta 1, the Japanese long-vowel mark and middle dot (`1ー3`, `1・3`) are not read, so every detection is inpainted, and a doubled dash (`1——3`, `1--3`) inpaints only the first number's detection: type a range with a single `-` or `～`, and separate numbers with commas or spaces. v26.3.0+plus.7.4 reads only commas and a plain `-`. There, `1--3` inpaints only the first number's detection, and numbers separated only by spaces (`1 3`), `～` or other Chinese or Japanese punctuation inpaint every detection, so type it as `1-3,5`.

### The result looks wrong

**A detailed face looks pale, washed out, or like someone else**

- Lower "Inpaint denoising strength" (0.4 by default). A lower value keeps more of the original face.
- If the colours don't match the rest of the picture, turn on Settings → ADetailer → "Match the colors of each inpainted region to the picture" (from plus.9), which corrects only ADetailer's regions, or the WebUI's "Apply color correction to img2img results to match original colors" setting (in the img2img part of the WebUI's Settings), which corrects every img2img result; and raise "Inpaint mask blur" so the edge blends in.
- With "Use separate VAE" or "Use separate checkpoint", check that they fit your main model: a VAE made for another model family gives wrong or washed-out colours.

**One thing is redrawn as another, for example a face as a hand**

- Keep "Inpaint denoising strength" low.
- Turn on "Auto class-guard" in the tab's prompt section. It works with detectors that have class names.
- Or give each class its own text with `[CLASS=face] … [/CLASS]` blocks in the tab prompt or, with "Process classes sequentially", with per-class prompt lines.

**A style LoRA shows up on faces and hands**

- Tick "Strip LoRAs from the detailer prompt".
- 🟢 In plus.7.4 and beta 1, a LoRA inside a selected Style is not stripped: leave that Style out while you need the option.
- 🟢 In plus.7.4 and beta 1, "Use LoRAs from main prompt" also applies a LoRA that a selected Style adds a second time, so the detailed regions get it twice as strong: move that LoRA from the Style into the main prompt. From plus.8 it is used once.

**The detailed regions get the selected Styles twice (🟢 fixed in plus.8)**

- With Styles selected and a blank or `[PROMPT]` ADetailer prompt, the style is applied twice in the detailer pass. Write the tab's own prompt instead: it gets the styles once. From plus.8 they are applied once also with a blank or `[PROMPT]` prompt, and also when the main prompt holds comments, except when another extension, such as Dynamic Prompts with wildcards, rewrites the main prompts. In plus.8 and the betas of plus.9 they are still applied twice also when a Style adds its text on the line of a comment or holds a comment itself. 🟡 From plus.10 beta 2 they are applied once also when another extension changed the main prompt, except in three rare cases (see [Notes on plus.10 beta 2](#notes-on-plus10-beta-2)).

**🟡 The console says that a tab names the same LoRA more than once (from plus.10 beta 2)**

- The final prompt of that tab's detailer pass has the tag of the same LoRA twice, and the WebUI loads a LoRA once for each tag (classic Forge loads two identical tags once), so its effect can be stronger than you asked for. Keep one tag: `[PROMPT]` and a blank prompt already bring the main prompt's LoRAs, and the "Prompt append" text and the selected Styles add their own. See [LoRAs in the detailer prompt](#loras-in-the-detailer-prompt).

**🟡 After "✨ Fill the tabs", a LoRA goes to the wrong person, or two people swap their LoRAs (from plus.10 beta 2)**

- Read the status under the button of the guide "Guide: several subjects, one LoRA each". When one of its lines starts with "⚠️ The parts follow Settings, ADetailer", the parts are not given from left to right: see the entry below that starts with "The guide's status says".
- Count the people. The parts go to the biggest people of the picture, from left to right by the left edge of their boxes, not to the people you have in mind: a figure in the background with a big box, or a person the detector does not find, moves every part after it. Set "People in the picture" to the number of people of about the same size, fill the rows by their place from the left, leave empty the rows of the people to leave alone, and press "✨ Fill the tabs" again. The [Detection preview](#detection-preview) of the tab shows the boxes the detector finds.
- Two people who hug, or stand one behind the other, can have boxes that start at almost the same point: the order goes by the left edge of each box, so which one counts as the left one can change from picture to picture.
- With "Whole person, then face (2 tabs)" the 2nd tab keeps the biggest faces on its own, and the faces get their parts by the order of their own boxes, not by the person they belong to: when another person in the picture has a bigger face, when a face is not found, or when people overlap, a face can get the part of another person. Use "Whole person (outline)" or "Face only" for that picture. With a sort order other than "Position (left to right)" the faces can also come in another order than the people: set the order as the entry below that starts with "The guide's status says" explains.
- In the Inpaint tabs of img2img with "Inpaint area" on "Whole picture", the people that do not touch your inpaint mask are left out and the others move up: see [Notes on plus.10 beta 2](#notes-on-plus10-beta-2).

**🟡 The guide's status says "The parts follow Settings, ADetailer, Sort bounding boxes by, now None" (from plus.10 beta 2)**

- The tab is filled, but the parts of its prompt would not go to the people from left to right. The order is one setting for every tab, and on a fresh setup it is on None. Open Settings → ADetailer, set "Sort bounding boxes by" to "Position (left to right)" and click **Apply settings**. When the status names "More sort orders" instead, set "More sort orders (overrides Sort bounding boxes by)" to Off too: any other choice there is used instead of "Sort bounding boxes by".
- The guide only reads these two settings, because changing them changes the order of every tab and of every `[SEP]` prompt. The order is read when you generate, so there is no need to fill the tabs again; press "✨ Fill the tabs" once more if you want the status to confirm it.
- With one subject there is no order to follow, and the status does not warn.

**🟡 With the guide, the LoRAs mix: every person gets something of the other LoRAs (from plus.10 beta 2)**

- Take the LoRAs of the subjects out of the main prompt. A LoRA written there acts on the whole picture, so the main generation draws every person with all of them, before ADetailer starts. The guide's status warns when a LoRA of a row is also in the main prompt. Describe the people in the main prompt with plain words, and let the filled tab give each one their LoRA.
- In the filled tab, untick "Use LoRAs from main prompt" if the main prompt holds LoRAs that the people should not get: the guide does not change it, and it adds the main prompt's LoRAs to every part.
- A row with a LoRA and no words becomes `[PROMPT]` plus its tag, and `[PROMPT]` is the whole main prompt: type a few words in that row to use them instead.

**🟡 A LoRA I just added is not in the guide's menus (from plus.10 beta 2)**

- The menus show the LoRA list the WebUI had at its start, or at the last press of "✨ Fill the tabs", and the guide does not scan the disk. Refresh the WebUI's own LoRA list, then press "✨ Fill the tabs": the menus get the WebUI's current list with the answer, also when the status only asks you to fill a row.
- Or type the LoRA's name in the menu, without the ending of its file. While the WebUI does not list that name, the status warns, and the part is written as typed.
- A second browser window that was already open keeps its old list until you reload that page.

**The edge of the detailed region shows, or the mask is too tight**

- Tick "Use bbox as mask (segmentation models)", or raise "Mask erosion (-) / dilation (+)" (4 by default).
- Raise "Inpaint mask blur" (4 by default) and "Inpaint only masked padding, pixels".
- If the edge shows because the region came out in other colours, turn on Settings → ADetailer → "Match the colors of each inpainted region to the picture" (from plus.9).

**The result is only a crop of the detailed region (🟢 fixed in plus.8)**

- The WebUI's "Overlay original for inpaint" setting (Settings → img2img; on Forge Neo, "For inpainting, overlay the resulting image back onto the original image") is off, and "Inpaint only masked" then returns only the region. Turn that setting on, or untick "Inpaint only masked". From plus.8 the region is pasted back anyway, with two limits in plus.8 (see [Known issues of plus.8](#known-issues-of-plus8)); from plus.9 it is pasted back without changing that setting, so those limits are gone.

**"Use separate steps" or "Use separate sampler" has no effect (🟢 fixed in plus.8)**

- This happens with Settings → ADetailer → "Apply only selected scripts to ADetailer" turned off, or with `sampler` in "Script names to apply to ADetailer": the detailer pass then takes the main generation's steps, sampler and scheduler. Turn the option back on (the default) and leave `sampler` out of the list.

**On classic Forge, ADetailer's ControlNet model does nothing and the console shows a ControlNet error for every region (🟢 fixed in plus.8)**

- v26.3.0+plus.7.4 and beta 1 do not work with classic Forge's current built-in ControlNet, and there is no workaround. It is fixed in plus.8 (checked in classic Forge's code, not tested live).

**With Dynamic Prompts, the saved parameters show a finished prompt instead of the wildcard template (🟢 fixed in plus.8)**

- With ADetailer on and Dynamic Prompts' "Save template to metadata" option, the "Template" and "Negative Template" saved with the images are replaced by the first image's finished prompt. There is no workaround in plus.7.4 and beta 1; plus.8 keeps the template.

**In sequential mode with Merge and Invert, the classes are repainted with each other's prompt (🟢 fixed in plus.8)**

- Each class pass inverted only its own class, so the face pass also repainted the hands with the face prompt, and the hand pass the faces. Turn "Process classes sequentially" off in a tab that uses Merge and Invert. From plus.8 the tab runs one pass for all the selected classes there.

**On reForge, the detailed regions of a v-prediction or flow model come out broken (fixed in plus.9)**

- With Settings → ADetailer → "Apply only selected scripts to ADetailer" on (the default), plus.8 leaves reForge's Advanced Model Sampling out of the detailer pass. Add `advanced_model_sampling_script` (and `advanced_model_sampling_script_backported`, where reForge has it) to "Script names to apply to ADetailer", or turn "Apply only selected scripts to ADetailer" off. From plus.9 these scripts always run in the detailer pass (checked in reForge's code), except in a tab with a ControlNet model: there the pass runs only ControlNet, so for a v-prediction or flow model leave that tab's ControlNet model on "None".

**Every cell of an X/Y/Z grid with an "[ADetailer] CFG scale 1st" axis looks the same (fixed in plus.9)**

- The detailer pass uses a tab's own CFG scale only with "Use separate CFG scale" ticked: in plus.8, tick it in the tab before you start the grid. From plus.9 a CFG scale or Steps axis ticks "Use separate CFG scale" or "Use separate steps" in each cell. The same goes for the "[ADetailer] ADetailer sampler 1st", "[ADetailer] ADetailer scheduler 1st" and "[ADetailer] noise multiplier 1st" axes with "Use separate sampler" and "Use separate noise multiplier", from the stable release; in plus.8 and the plus.9 betas, tick that box in the tab first.

**ADetailer's ControlNet model has little or no effect**

- In plus.8, on Forge, Forge Neo and reForge, choosing a model sets "ControlNet module" to "None", which hands the picture to ControlNet as it is: pick the preprocessor that fits the model (see [Known issues of plus.8](#known-issues-of-plus8)). From plus.9, the model's default preprocessor is selected there; if "ControlNet module" shows "None", it was picked by hand or restored as saved, so pick the preprocessor again.
- 🟢 In plus.7.4 and beta 1, a model whose file name writes its type with capitals, such as `OpenPoseXL2`, gets no preprocessor at all: rename the file so that its type is in lower case, such as `openposeXL2`, and restart the WebUI. From plus.8 the type matches in any case.

**Small, distant or background faces get detailed too**

- Raise "Mask min area ratio" in the tab's Detection section. It is the detection box's share of the whole picture, not of its width: 0.01 is 1%, 0.05 is 5%, and a box half as wide and half as tall as the picture is 0.25, not 0.5. Boxes smaller than that are skipped. 🟡 From plus.10 beta 1, the [Detection preview](#detection-preview) writes each box's share next to its number, such as `#2 face 0.8%` (0.8% is 0.008), so you can read there which value leaves the small ones out.
- Or set "Mask only the top k (0 to disable)" with "Method to filter top k masks by (confidence or area)" on Area: it keeps only the k biggest detections, so 1 keeps the main subject.
- Or run the Detection preview and type, or click, the numbers of the faces you want in "Inpaint only these detections" (see [Detection preview](#detection-preview)).

**The detail pass runs fewer steps than you set**

- The detail pass is an img2img pass, and img2img runs only about steps × denoising strength: 30 steps at a denoising strength of 0.4 run about 12. This also applies with "Use separate steps". To run the full number, turn on the WebUI option "With img2img, do exactly the amount of steps the slider specifies" in Settings → img2img (on Forge Neo, "During img2img, do exactly the number of Steps the slider specifies"). It applies to every img2img generation, not only to ADetailer. Reported upstream in [Bing-su/adetailer#649](https://github.com/Bing-su/adetailer/issues/649).

**ADetailer and a face-swap extension (ReActor, roop…) run in the wrong order**

- In Settings → Callbacks, find "script_postprocess_image callback priority": the entries listed there run first, in that order, in the step where ADetailer works. To detail the swapped face, put the face-swap extension's entry before ADetailer's, the one ending in `!adetailer.py/script_postprocess_image/AfterDetailerScript`; to swap the face after ADetailer, put ADetailer's first. Apply and restart the WebUI.
- This works only when the other extension changes the image in that same step. An extension that works in the later postprocess step always runs after ADetailer, whatever this list says. In which step ReActor and roop work was not checked: if moving the entry changes nothing, check that extension's own settings. Asked upstream in [Bing-su/adetailer#193](https://github.com/Bing-su/adetailer/issues/193), [#327](https://github.com/Bing-su/adetailer/issues/327) and [discussion #159](https://github.com/Bing-su/adetailer/discussions/159).

**CUDA out of memory**

- Detail at a smaller size: set a smaller size with "Use separate width/height", or a lower "Inpaint resolution scale" (with "Scale inpaint to bbox" ticked).
- From plus.9, set Settings → ADetailer → "Max inpaint size, pixels per side (0 = no limit)", for example to 1024: with "Inpaint only masked", a larger region is then inpainted smaller, with the same shape (see [Scale inpaint to bbox](#scale-inpaint-to-bbox)).
- Lower "Detection resolution".
- For many images, raise Batch count instead of Batch size.
- On Forge, try without the `--cuda-malloc`, `--cuda-stream` and `--pin-shared-memory` launch options if you use them, and turn off on-the-fly LoRA patching where your WebUI offers it.

### Existing images and folder runs

**"Run ADetailer on an image" gives a result that does not match the picture's prompt**

- The tool has no main prompt: write the prompt in the tab. A blank one inpaints with an empty prompt. 🟡 From plus.10 beta 2, turn on Settings → ADetailer → "Run ADetailer on an image uses the prompt saved in the image" to use the prompt saved in the picture instead (see [Many pictures, each with its own prompt](#many-pictures-each-with-its-own-prompt)).
- Steps, CFG scale and sampler come from the tab's "Use separate steps", "Use separate CFG scale" and "Use separate sampler" options. Without them it uses 28 steps, CFG scale 7, the WebUI's first sampler and the Automatic scheduler (on Forge and Forge Neo with a Distilled CFG Scale of 3.5). For a guidance-distilled or few-step checkpoint, such as Flux dev or SDXL Lightning, tick those options and set the model's own values, for example CFG scale 1.
- Every run picks new random seeds, so run it again for another variation.

**The Interrupt and Skip buttons are missing while "Run ADetailer on an image" or a folder run works**

- The WebUI shows them only while its own Generate runs. Press Alt+Enter, the WebUI's Skip shortcut, with the txt2img or img2img tab open: a single picture comes back unchanged, and a folder run stops before the next file. Esc does nothing here. See "To stop a run" in [Run ADetailer on an image](#run-adetailer-on-an-image).

**A cut-out PNG comes out on black (🟢 fixed in plus.8)**

- In plus.7.4 and beta 1, "Run ADetailer on an image", folder runs and Skip img2img drop the transparency of the picture: its transparent parts turn black, or the colour they hold. Save the picture on the background you want, without transparency, before the run. From plus.8 they are filled with the WebUI's img2img background colour, the "With img2img, fill transparent parts of the input image with this color" setting (on Forge Neo: "For img2img, fill the transparent parts of the input image with this color"). 🟡 From plus.10 beta 2, to get the transparency back on the result, turn on Settings → ADetailer → "Run ADetailer on an image and Skip img2img keep the transparency of the picture" and save in a file format that has transparency, such as PNG.

**A copy saved next to its source looks duller, or a WebP copy loses quality (🟢 fixed in plus.8)**

- With "📁 Save results in the source folder instead", plus.7.4 and beta 1 save the copy without the source's colour profile, so colour-managed viewers show it duller or shifted, and save every WebP copy lossy at quality 80. If this matters, convert such sources to sRGB, and WebP sources to PNG, before the run. From plus.8 the copy keeps the profile, and a lossless WebP stays lossless.

**PNG Info shows "None" for a result saved in the `ADetailer-Inpaint` folder (🟢 fixed in plus.8)**

- v26.3.0+plus.7.4 and beta 1 save these results with the word "None" as their parameters, and Send to then puts "None" in the prompt box. The picture itself is fine. From plus.8 they record the detailer pass's parameters.

**PNG Info or Send to brings back only the detailer pass of a "Run ADetailer on an image" or folder-run result (fixed in plus.8)**

- In beta 2 the parameters of these results hold only the detailer pass's prompt, steps, sampler, seed and size, with no ADetailer setting, so pasting them leaves the ADetailer tabs as they were. In plus.7.4 and beta 1 so does the picture shown in Result after a single run that is not saved, while a PNG saved to `ADetailer-Inpaint` records the word "None" (see the entry above). In plus.7.4, beta 1 and beta 2 a copy saved next to its source holds no parameters at all. To keep a tab's setup, save it as a preset. With the WebUI's "Write infotext to metadata" setting on, plus.8 records the tab's settings in all of them, except BMP and TIFF copies, which record no parameters, and a copy whose parameters cannot be written (see Batch a whole folder); with the setting off none of them records parameters.

**The console says "create_infotext failed" for each mask preview or step image of "Run ADetailer on an image" (fixed in plus.9)**

- With "Save mask previews" or "Save intermediate step images" on, plus.8 saves these files of "Run ADetailer on an image" and folder runs without parameters and prints this message for each one; the pictures themselves are fine. From plus.9 they record their parameters, and so do the masks of "Save inpaint masks".

**A folder run skips files or says "No images found in that folder."**

- The folder box, "Or batch a whole folder (optional)", reads PNG, JPEG, WebP, BMP and TIFF (`.tif` or `.tiff`) files directly in that folder; files in its sub-folders are not read.
- With "📁 Save results in the source folder instead", files whose name ends in `-ad` or `-ad-<number>` count as earlier results and are skipped. From plus.8 the status says how many; before, they were left out without a word, and a folder of only such files reported "No images found".
- "Pick a detector model first." means that the tab has no detector: the run stops at the first image that opens and writes nothing (if no image in the folder can be opened, the batch is reported as failed instead).

**A single dropped image details a whole folder (🟢 fixed in plus.8)**

- Settings → Defaults saved a folder path as a startup default. Empty the "Or batch a whole folder (optional)" box of the 1st tab, save the defaults again with Settings → Defaults (see [Startup defaults for the panel's checkboxes](#startup-defaults-for-the-panels-checkboxes)), and empty the box in the other tabs too, or restart the WebUI.

### Tabs, presets and settings

**Only the first tab runs**

- On a fresh setup only the 1st tab starts with "Enable this tab" ticked, and the other tabs start with the detector `None`. In each other tab you use, tick "Enable this tab" and pick a detector. The green pill next to the ADetailer title shows how many tabs have "Enable this tab" ticked.

**The panel keeps coming back with old settings, or you want a clean start**

- Every Generate saves each tab's settings, and the next WebUI start brings them back. For a clean start, press Reset (tick "Reset every tab" for all tabs), pick a detector again, then Generate once.
- To start with the defaults every time, turn off Settings → ADetailer → "Remember last-used settings between restarts". Turning it off and on again does not erase the saved settings.
- The "🔄 Reset ADetailer settings to defaults" button on the Settings page resets only the Settings options, not your tabs or presets.

**ADetailer is on at every start, or another panel checkbox always starts the same way**

- Settings → Defaults has saved a startup default for it. See [Startup defaults for the panel's checkboxes](#startup-defaults-for-the-panels-checkboxes).
- From plus.9, Settings → ADetailer → "Enable ADetailer at startup" set to txt2img, img2img or Both also starts ADetailer switched on there, whatever Settings → Defaults says. Set it to Off and reload the UI to let Defaults decide again.

**The export button does not download a file on AUTOMATIC1111**

- Fixed in plus.9: Export downloads `adetailer-ultimate-presets.json` on these WebUIs too. If it still does not, check that your browser allows downloads from the WebUI's page.
- In plus.8, WebUIs with Gradio 3 (AUTOMATIC1111, and reForge's main branch) cannot download a file from that button; from plus.8 the status line says so (plus.7.4 wrongly reports the presets as exported). Back up the preset library by copying `user_presets.json` from the extension's folder (for example `extensions/adetailer-ultimate`). Import works on every WebUI from plus.8 on, and from plus.8 Export downloads the file on Forge and Forge Neo.

**A preset is missing from another tab's list**

- Import refreshes the importing tab's list at once. The other tabs show the new presets after a Save, Delete or Rename in the same txt2img or img2img panel (from plus.8: also after a browser reload). A preset saved in txt2img reaches img2img's lists (and the other way round) after a Save, Delete, Rename or Import there, or after a restart.
- 🟢 In plus.7.4 and beta 1, a browser reload shows only the presets that existed when the WebUI started: use Settings → Reload UI instead.

**Load says "⚠️ Loaded '…', but …" (🟢 from plus.8)**

- The preset names something this WebUI does not have: a detector (the tab keeps its own detector and classes), or a separate checkpoint, VAE, text encoder, scheduler, ControlNet model or sampler (the tab uses "Use same …", "None" or the first sampler there instead). Install what is missing and load the preset again, or pick other entries and save the preset.

**After Reset, "ADetailer sampler" looks empty or shows "DPM++ 2M Karras" (🟢 fixed in plus.8)**

- The detailer uses this list only with "Use separate sampler" ticked. If you use that option, pick the sampler again after a Reset or "Reset every tab": a preset saved or a tab copied right after the Reset keeps the wrong value too.
- A preset saved right after a Reset in an earlier release can also show the separate checkpoint, VAE and text encoder lists empty when you load it: pick "Use same …" in them and the sampler again, then save the preset. From plus.8, loading such a preset shows them as on a fresh setup, and with "Use separate sampler" ticked it shows the sampler name as the sampler and scheduler the WebUI runs it as.

**After a restart or a paste, "ADetailer sampler" is not the sampler you used**

- With "Use separate sampler" ticked, a sampler this WebUI does not have comes back as the first sampler after a restart, and in plus.8 pasted parameters put its name in the list (empty on Forge and Forge Neo), while Generate runs the first sampler: pick the sampler and scheduler again (see [Known issues of plus.8](#known-issues-of-plus8)). From plus.9, a restart and pasting split the name as Load does, and a sampler this WebUI does not have becomes the first sampler, with a console message. From plus.8 an old name such as "DPM++ 2M Karras" is split into the sampler and the scheduler; in plus.7.4 and beta 1, after a restart such a tab ran with the main pass's scheduler, without a message.

**The tab counter shows the old number after Paste settings, Load or Reset (🟢 fixed in plus.8)**

- On Forge and Forge Neo a fast answer can leave the old count until something else changes on the page. Generation uses the real "Enable this tab" checkboxes, so only the counter is wrong.

### Separate VAE, text encoder and CLIP skip

**On Forge Neo, a tab with "Use separate VAE" fails on every image (🟢 fixed in plus.8)**

- Pick a module once in the VAE / Text Encoder selector at the top of the page, then generate again. This also stops a tab's separate VAE or text encoder from staying loaded after its pass.

**A tab's own CLIP skip or VAE carries over to the next tabs or to later images (🟢 fixed in plus.8)**

- The WebUI puts them back after the pass only once they have been saved in its settings. Save Clip skip once with Settings → Apply settings (on AUTOMATIC1111, the VAE too); on Forge and Forge Neo, pick a module once in the VAE / Text Encoder selector at the top of the page.

**🟡 The console says a tab's checkpoint is of another model family and kept the main model's VAE / text encoder files (Forge and Forge Neo, from plus.10 beta 2)**

- The tab's own checkpoint ran with VAE or text encoder files made for the main model's family, so its result can be broken or the pass can fail. In that tab, tick "Use separate text encoder" with "None (use detailer checkpoint's own)" and "Use separate VAE" with Automatic, or pick the files that fit its checkpoint. When the line says that Automatic would keep the main model's VAE file (a VAE file outside a folder named VAE), pick a VAE file of the checkpoint's family in "Use separate VAE" instead of Automatic. When the line says that "Use separate VAE" already has a VAE file of the checkpoint's family, leave that choice as it is and only set "Use separate text encoder" to "None (use detailer checkpoint's own)": the line is then not printed again. When the line says that no choice of the tab takes out a file of the main model, move that file into the folder the line names (a folder named VAE, such as `models/VAE`, for a VAE file; a folder named text_encoder, such as `models/text_encoder`, for a text encoder) and pick it again in the WebUI's "VAE / Text Encoder" list: from there the tab's own VAE file takes the place of the VAE file, and "None (use detailer checkpoint's own)" takes out the text encoder.
- When the console says instead that the tab's checkpoint could not be loaded while the pass kept the main model's files, the checkpoint did not load at all with them, and on Forge Neo the whole generation then stops with an error; this is left only for the cases that the skip of the point below does not recognise (see [Notes on plus.10 beta 2](#notes-on-plus10-beta-2)). The same two settings fix it for an SD 1.5 or SDXL checkpoint in the tab; for a Flux checkpoint the line says to pick a Flux VAE file in the tab's "Use separate VAE" instead. When the main model's VAE file is outside a folder named VAE, where Automatic would keep it, the line says first to move it into a folder named VAE and to pick it again in the WebUI's "VAE / Text Encoder" list, then what to set in the tab: do it in that order, because a VAE file picked in the tab first leaves the main model's file in place, and the WebUI can read it after the tab's. When the line says that no choice of the tab takes out a text encoder, move it as above; when it is the only file of the main model that the pass kept, "None" and Automatic are not told. When the tab already has such a VAE file, the line says to move the main model's VAE file into a folder named VAE and to pick it again in the WebUI's "VAE / Text Encoder" list: outside such a folder no choice of the tab takes it out, and the WebUI can read it after the tab's. On Forge, not tested live, the WebUI may also need a restart.

**🟡 The console, or the text under the results, says that a tab was skipped because its checkpoint cannot be loaded (Forge and Forge Neo, from plus.10 beta 2)**

- The checkpoint of that tab's pass does not fit the VAE file the pass would use, and loading it would have cost the whole generation, so ADetailer skipped the tab: the picture is saved without that tab's work. Do what the line says, in its order. When the line says first to move the main model's VAE file into a folder named VAE (a VAE file outside such a folder, as it can be with `--vae-dirs`), or a text encoder into a folder named text_encoder, do that before anything else: move the file, for example into `models/VAE` or `models/text_encoder`, and pick it again in the WebUI's "VAE / Text Encoder" list. No choice of the tab takes such a file out: a choice in the tab alone can end the skip while the file stays, and on Forge Neo the WebUI can read it after the tab's, so the checkpoint still fails to load and the whole generation is lost (see the known issue in [Notes on plus.10 beta 2](#notes-on-plus10-beta-2)). Then, when the line names the main model's VAE file and the tab has an SD 1.5 or SDXL checkpoint: tick "Use separate VAE" with Automatic and "Use separate text encoder" with "None (use detailer checkpoint's own)". When the tab has a Flux checkpoint: pick a Flux VAE file, such as `ae.safetensors`, in the tab's "Use separate VAE", and use a checkpoint file that holds its text encoders. When it names the VAE file chosen in the tab: choose one made for that checkpoint's family there.
- "Run ADetailer on an image" then says "ADetailer did not run" with the same reason and gives no result. When the line says first to move a file of the main model, its status says it too.

## FAQ

### Can I use my presets and settings from the original ADetailer?

The original ADetailer has no named presets, so there is nothing to import. Your Settings → ADetailer options and your detectors carry over, images made with the original paste back, and API requests written for it work too (see [Does it work with the WebUI's API?](#does-it-work-with-the-webuis-api)). What carries over and what starts fresh: [Coming from the original ADetailer?](#coming-from-the-original-adetailer).

### Can I move my presets to another computer?

Yes. Export the preset library as a file (on Forge and Forge Neo, from plus.8) and import it on the other computer, or copy `user_presets.json` from the extension's folder. From plus.9, Export downloads the file on AUTOMATIC1111 and other Gradio 3 WebUIs too; in plus.8, copying the file is the only way there, because the export button cannot download a file. From plus.8, loading a preset that names a detector, checkpoint, VAE, text encoder, scheduler, ControlNet model or sampler the other WebUI does not have says so and keeps the tab working; in plus.7.4 and beta 1, check those settings after loading. See [Your data](#your-data).

### Does ADetailer Ultimate slow generation down?

With the same settings it does about the same work as the original ADetailer, and its extra detailing options are off by default. Generation takes longer only with what you turn on: every enabled tab is one more detailing pass, "Process classes sequentially" runs one pass per selected class, and a higher "Detection resolution" makes detection slower and uses more VRAM. The Detection preview and "Run ADetailer on an image" run only when you press their buttons. 🟡 From plus.10 beta 2, a YOLO detector is read from its file once per generation, not again for every image, tab and X/Y/Z cell; YOLO-World detectors and detectors that store no detection size of their own are read as before, and so are all of them when more than 4 different detectors take turns on each image, for example 5 tabs with 5 different detectors (an X/Y/Z axis over more than 4 detectors still reads each one once per cell). Up to 4 detectors stay in memory (video memory, when they run on the graphics card) until the generation ends.

### Is it safe to switch to the beta and back?

Yes. Switching changes only the extension's code. Your presets (`user_presets.json`) and remembered settings (`user_state.json`) are not part of that code, so the switch leaves them alone, and neither the plus.8 betas nor the plus.9 betas added new tab settings, so the stable releases and the betas read them. Back them up first anyway. After going back from plus.9 to plus.8, check the mask sliders of a tab or preset where you used a value above plus.8's limits (see [Notes on plus.9 beta 1](#notes-on-plus9-beta-1)); the four options that beta 2 adds in Settings → ADetailer are simply ignored there, and a prompt with `[PROMPT=words]` or a tab with only the `iris` class picked does not work there (see [Notes on plus.9 beta 2](#notes-on-plus9-beta-2)). 🟡 plus.10 beta 1 adds one tab setting, "Masked content": plus.9 ignores it in presets, remembered settings and pasted parameters, so its tabs run as "original" there, but an API request that sends it to plus.9 skips that tab (see [Notes on plus.10 beta 1](#notes-on-plus10-beta-1)). The three options that beta 2 adds in Settings → ADetailer are ignored by plus.9 and beta 1, which keep their saved values for when you come back: with "Run ADetailer on an image uses the prompt saved in the image", that tool starts every picture with an empty main prompt again; with "With hires fix, detail with the hires checkpoint", the tabs detail a hires fix picture with the main checkpoint; with "Run ADetailer on an image and Skip img2img keep the transparency of the picture", the transparent parts of a result stay filled with the img2img background colour (see [Notes on plus.10 beta 2](#notes-on-plus10-beta-2)). See [Switch to the beta and back](#switch-to-the-beta-and-back) and [Your data](#your-data).

### Can I keep the original ADetailer installed too?

Not at the same time. ADetailer Ultimate replaces it: with two ADetailers the panel appears twice. Move the other copy's folder out of the `extensions` folder; you can move it back at any time (see [Coming from the original ADetailer?](#coming-from-the-original-adetailer)).

### Does it work with the WebUI's API?

Yes. API requests work as with the original ADetailer (see its [REST API guide](https://github.com/Bing-su/adetailer/wiki/REST-API)), and every new per-tab option of this fork can be sent too. From plus.9, a request in the short form, with only the tabs (`[{...}]`, without the values for "Enable ADetailer" and "Skip img2img"), runs one pass per tab it sends (to be exact in every case, use the full form `[true, false, {...}]`); before, it also ran the WebUI's default 1st tab ([Bing-su/adetailer#485](https://github.com/Bing-su/adetailer/issues/485)). From plus.9, a tab sent with an empty model, `"ad_model": ""`, counts as off, like `"None"`; in plus.8 and plus.9 beta 1 it stops with "Model '' not found" and the tabs after it are not detailed, so send `"None"` there. From plus.8, a tab that a request leaves out gets a fresh setup's settings, not the settings remembered in the UI (in plus.7.4 and beta 1, send every tab in the request, with `"ad_model": "None"` for the tabs you do not want, or turn off "Remember last-used settings between restarts" and restart the WebUI); and with "Apply only selected scripts to ADetailer" off, the detailer pass uses the request's steps, sampler and scheduler (or the tab's separate ones), where before it used the defaults of the WebUI's Sampler section, such as 20 steps. In plus.8, on Forge, Forge Neo and reForge, a request that leaves the ControlNet preprocessor out runs without one (see [Known issues of plus.8](#known-issues-of-plus8)); from plus.9 it uses the model's default, and from plus.8 an inpaint mask sent with transparency is read as the WebUI reads it. An X/Y/Z plot request picks its axes by position: from plus.9 the axes of the extensions loaded after ADetailer sit further down the list, and their position changes with "Max tabs", so look the position up from the axis name (see [Notes on plus.9 beta 1](#notes-on-plus9-beta-1)); 🟡 plus.10 beta 1 adds one more ADetailer axis after all the others of ADetailer, so only the axes of the extensions loaded after ADetailer move one place down, and a per-tab option, `ad_inpaint_masked_content`, which takes the four names in any letter case or the WebUI's `inpainting_fill` number 0 to 3 (see [Masked content](#masked-content) and [Notes on plus.10 beta 1](#notes-on-plus10-beta-1)).

### Can the ADetailer prompt keep what a wildcard picked in the main prompt?

Yes. A blank ADetailer prompt or `[PROMPT]` already reuses each image's main prompt as it was resolved, also in batches, so what a wildcard picked is the same. In the stable release, a `__wildcard__` typed in the ADetailer prompt itself is drawn again, so it can differ from the main prompt. 🟡 From plus.10 beta 1, a `__name__` wildcard in a tab's prompt, negative prompt or "Prompt append" boxes that the main prompt also holds takes the value drawn there for each image, with nothing to switch on: with `1girl, __eyecolor__ eyes` as the main prompt and `close-up, __eyecolor__ eyes` in the tab, both get the same eye colour. A wildcard that is not in the main prompt is still drawn again, so use one of those to get a different value on purpose; when the value cannot be read for sure, the tab's prompt is left as before and "Verbose diagnostic log" says why. A value that holds a LoRA is not loaded twice with "Use LoRAs from main prompt", and "Strip LoRAs from the detailer prompt" removes it too (see [New in plus.10 beta 1](#new-in-plus10-beta-1), and [Notes on plus.10 beta 1](#notes-on-plus10-beta-1) for the wildcards that are still drawn again). From plus.9, to reuse only the part the wildcard picked, write for example `[PROMPT=eyes]`: it brings only the parts of the main prompt that name the eyes, such as `green eyes` (see [Upstream options](#upstream-options)). To use the whole main prompt at a lower weight, write `([PROMPT]:0.5)`. 🟡 From plus.10 beta 2, when a wildcard or a `{a|b}` variant of a tab's prompt is drawn in the detailer pass, the image parameters also record the prompt each region was really inpainted with, as "ADetailer resolved prompt" (see [Added in plus.10 beta 2](#added-in-plus10-beta-2)).

### Can each person in a picture get their own LoRA?

Yes. Each detected region is inpainted in a pass of its own, so a LoRA written in one `[SEP]` part of a tab's prompt is used only for that region: [A different prompt or LoRA for each face](#a-different-prompt-or-lora-for-each-face) shows how to write it by hand, in the stable release too. 🟡 From plus.10 beta 2, the box "Guide: several subjects, one LoRA each" above the tabs does it for you, for up to four people: pick a LoRA and a few words for each one and press "✨ Fill the tabs" (see [Several subjects, one LoRA each (guide)](#several-subjects-one-lora-each-guide)). Either way, keep those LoRAs out of the main prompt, where a LoRA acts on the whole picture, and mind that the parts go to the people by the order of their boxes, not by who they are.

### Does it upload my images?

No. ADetailer Ultimate works on your computer and does not upload your images, prompts or settings. It does download files: in plus.8, at every start it checks the built-in detectors on Hugging Face (or on the hf-mirror.com mirror when Hugging Face cannot be reached) and downloads the missing ones, which the `--ad-no-huggingface` option turns off (from plus.9, it contacts Hugging Face only for a built-in detector that is neither in the model folders nor downloaded before, see [Built-in detectors](#built-in-detectors); from the stable plus.9, ADetailer's own request to the mirror goes without your Hugging Face token, which plus.8 and the plus.9 betas sent to it too when you were logged in to Hugging Face; if you set the `HF_ENDPOINT` variable to the mirror yourself, in the stable plus.9 the Hugging Face library sends the token there, as it does to Hugging Face; 🟡 from plus.10 beta 1 the token is not sent to hf-mirror.com or its subdomains, and only another host set with `HF_ENDPOINT` still gets it from the Hugging Face library); it installs the Python packages it needs when they are missing; it fetches the two small MediaPipe model files from Google the first time you use a MediaPipe detector; and the first time a YOLO-World detector runs with names you typed, the Ultralytics library downloads the CLIP text model it needs (about 340 MB, from OpenAI) and installs its CLIP package from GitHub if it is missing (🟡 from plus.10 beta 2, the same happens with a YOLOE detector: the Ultralytics library downloads its text model, about 600 MB, or about 254 MB for YOLOE-26, from github.com/ultralytics/assets, and installs the `clip` package with pip when it is missing). With a detector whose class names, as stored in the detector file (a `.names.json` next to it does not change them; for YOLO-World, the names you type), have non-ASCII letters (for example Japanese or Cyrillic), when a detection finds something the Ultralytics library looks for the `Arial.Unicode.ttf` font to write the names on its picture of the detections, and when the font is neither in its own folder of your user profile nor among the system fonts (on Windows it usually is not), it downloads it from GitHub into that folder, once, also with `--ad-no-huggingface`; without a connection it uses its default font and the detection goes on. The image parameters that ADetailer adds to each picture keep the comments typed in its prompt boxes, unlike the WebUI's main prompt (see [Prompts](#prompts)), so take private notes out of them before you share a picture. Your presets and remembered settings stay in the extension's folder, and from plus.8 the sections you leave open are remembered in your browser. The Ultralytics library, which runs the YOLO detectors, can send anonymous usage analytics of its own: from plus.9 ADetailer switches them off before its detections (see [Compatibility](#compatibility)); in plus.8, its documentation explains how to turn them off. Recent versions of the MediaPipe library, such as 0.10.35, send Google usage statistics of their own (never the pictures), which ADetailer cannot switch off: see [Notes on plus.9 beta 2](#notes-on-plus9-beta-2).

### Is a detector file safe?

Only as safe as its source. A `.pt` detector is a pickle file, and loading one can run code stored in it. ADetailer reads only `.pt` detectors (no `.safetensors`) and, like the original ADetailer, loads them with AUTOMATIC1111's safe-unpickle check and PyTorch's weights-only loading switched off, because that check does not accept the Python classes a YOLO model contains; Forge Neo has no such check. So a virus scanner or a model site that warns about a YOLO `.pt` file is common and proves neither that the file is harmful nor that it is safe. Download detectors only from sources you trust: the built-in ones come from the [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer) page on Hugging Face. On AUTOMATIC1111 with the check on, a `<model>.names.json` beside a custom detector lets the panel list its classes without opening the file (see [Class names](#class-names)). From plus.9, the Guide tab says this too.

### Can I switch "Save images before ADetailer" from the top of the page?

Yes, with the WebUI's quick settings. "Save images before ADetailer" (Settings → ADetailer) also saves each image as it was before ADetailer changed it, with a file name ending in `-ad-before`, which helps to compare checkpoints or LoRAs. Add `ad_save_images_before` to Settings → User interface → "Quicksettings list" (on Forge Neo, User Interface → "Quicksettings List"), click **Apply settings** and reload the UI: the option then appears at the top of the page. Asked upstream in [Bing-su/adetailer#726](https://github.com/Bing-su/adetailer/issues/726). From plus.9, the Guide tab says this too.

### Why are some labels still in English when I use Language Diffusion?

The options added in recent versions are not in the translation dictionaries yet, and the Guide tab is in English only. Some labels are made while you use the panel, such as "📥 Paste from Nth tab", and have no translation. See [Translations](#translations).

### How do I know which version I have?

Open the ADetailer panel (this also switches ADetailer on): the badge at the top right of its header shows "ADetailer Ultimate", the version and the build. The console also prints it at startup, in the line that starts with `[-] ADetailer initialized. version:`.

## Compatibility

| WebUI | Status | Notes |
| --- | --- | --- |
| AUTOMATIC1111 1.10 | 🟢 Supported, tested live | Gradio 3: in plus.8 the export button cannot download a file (Import works from plus.8 on); from plus.9 it downloads it. "Use separate text encoder" and the Distilled CFG setting apply only on Forge and Forge Neo. A detector without a `<model>.names.json` file lists its classes after its first detection (from plus.8; in plus.7.4 only with that file); from plus.9, straight away. On the Settings page only the thin divider line above **🔄 Reset ADetailer settings to defaults** is missing: the help text, the button and every feature still work. From plus.8 it also starts with the `--no-gradio-queue` option. |
| Forge Neo | 🟢 Supported, tested live | Runs on Python 3.13 with the newer MediaPipe 0.10 builds. "Use separate text encoder (Forge/Forge Neo)" works here. Forge Neo's startup message that suggests another ADetailer is not printed while this fork is installed. |
| Forge (classic) | 🟢 Supported, not tested live | Its differences are handled in the code. Gradio 4, as on Forge Neo: from plus.8 the export button downloads the file (in plus.7.4 the first click downloads nothing and later clicks the previous click's file). "Use separate text encoder (Forge/Forge Neo)" works here too, and detectors list their classes straight away, also without a `<model>.names.json` file. From plus.8, ADetailer's ControlNet options work with its current built-in ControlNet (checked in its code). |
| reForge | 🟢 Supported, not tested live | Its differences are handled in the code. On its main branch, which uses Gradio 3, the export button cannot download a file in plus.8, as on AUTOMATIC1111 (from plus.9, it downloads it). From plus.9, its Advanced Model Sampling scripts always run in the detailer pass, for v-prediction and flow models (checked in its code), except in a tab with a ControlNet model. |

- **Live tests** (real generations in the browser) run on AUTOMATIC1111 1.10 and Forge Neo, on Windows. Classic Forge and reForge are supported and their differences are handled in the code, but they are not part of the live tests. Other WebUIs built on AUTOMATIC1111 may work, but they are not checked.
- **When a WebUI lacks a feature,** ADetailer Ultimate keeps working without it instead of stopping with an error.
- **Python:** from 3.10 (AUTOMATIC1111 1.10) to 3.13 (Forge Neo). The automatic tests on GitHub run on Windows (Python 3.12, offline tests) and on macOS (Python 3.10 to 3.14, also with the real detectors).
- **Gradio:** Gradio 3 (AUTOMATIC1111, reForge's main branch) and Gradio 4 (Forge, Forge Neo).
- **Packages:** at every start, unless the WebUI's package installation is turned off, the extension installs what it needs when it is missing or its version is outside the tested range: Ultralytics 8.3.75 or newer, MediaPipe 0.10 (a MediaPipe 1.x is replaced by 0.10) and Rich. On WebUIs that use NumPy 1, such as AUTOMATIC1111, it keeps NumPy 1, so the WebUI's other packages keep working. On Linux aarch64, MediaPipe 0.10.x has wheels only up to 0.10.18 (Python 3.9 to 3.12), so those systems get 0.10.18; on Python 3.13 there is no supported MediaPipe wheel for aarch64, so the MediaPipe detectors are not available there. MediaPipe is installed with a command of its own, so when it cannot be installed the other packages still are.
- **Ultralytics usage analytics:** from plus.9, ADetailer switches off Ultralytics' anonymous usage analytics before its detections, in memory only; the shared Ultralytics settings file is never written, and in the same WebUI session the analytics stay off for other extensions that use Ultralytics.
- **Model families:** ADetailer details whatever checkpoint the WebUI runs. The original ADetailer was made for Stable Diffusion 1.5 and SDXL. On Forge and Forge Neo, a detailer checkpoint of another family can be used with a separate text encoder, and from plus.8 the detailer pass keeps your Distilled CFG Scale / Shift, used by newer model families such as Flux or Z-Image (with a separate checkpoint it uses the WebUI's default, 3.5); these mixed setups are still being checked for a later release. 🟡 From plus.10 beta 2, on Forge and Forge Neo, a tab whose checkpoint cannot be loaded with the VAE file of its pass is skipped before the pass when its `.safetensors` files show it for sure, instead of losing the whole generation, and the console warns when a tab's checkpoint of another model family keeps the main model's VAE or text encoder files (see [Separate checkpoint, VAE, text encoder and CLIP skip](#separate-checkpoint-vae-text-encoder-and-clip-skip)).

### Translations

With the [Language Diffusion](https://github.com/xXIlRizzoXx/sd-webui-language-diffusion) extension installed next to ADetailer Ultimate, the labels, buttons, hints, placeholders and tooltips of the fork's own controls up to plus.6 are translated into 10 locales: Chinese (Simplified), French, German, Italian, Japanese, Korean, Polish, Portuguese (Brazil), Russian and Spanish. The options added since plus.7 stay English until the dictionaries get them: Auto class-guard and Class-guard emphasis, Strip LoRAs from the detailer prompt, Detection resolution, Dynamic denoise by area, Inpaint only these detections, the folder batch of "Run ADetailer on an image" with its help text, Reset every tab, the Guide link and the pointer to the Guide tab, and the Settings options Show the ADetailer Guide tab and Verbose diagnostic log; so do the "(none)" preset entry, the reworded Copy settings and Reset tooltips, the Export tooltip shown on Gradio 3 WebUIs (in plus.8), the "📤 Export" label and the "Enable ADetailer at startup" option of plus.9, the reworded help texts of "Reset ADetailer settings", "Apply only on hires.fix" and "Manual mode", and the "📥 Paste from Nth tab" label that the other tabs' Paste settings button shows after a Copy, which is made when you click Copy.

- Language Diffusion also adds the language selector at the top of the page.
- Technical words (ADetailer, LoRA, CFG, VAE, ControlNet, hires.fix, img2img, inpaint, bbox, YOLO, MediaPipe and so on) stay English in every language on purpose, so they match tutorials and model pages.
- The ADetailer Guide tab is in English only.
- The "Save inpaint masks" option of plus.9 and its help text stay English until the dictionaries get them, and so do the tooltips of the options.
- Four more options of plus.9, "More sort orders (overrides Sort bounding boxes by)", "Max inpaint size, pixels per side (0 = no limit)", "Hide detection boxes from the live preview" and "Show detection previews in the results gallery", with their help texts and choices, stay English until the dictionaries get them, and so may the "Show labels" switch of the Detection preview.
- 🟡 The "Masked content" option of plus.10 beta 1 and its help text stay English until the dictionaries get them, and so do the option "Run ADetailer on an image uses the prompt saved in the image" of plus.10 beta 2 and its help text, the options "Run ADetailer on an image and Skip img2img keep the transparency of the picture" and "With hires fix, detail with the hires checkpoint" with their help texts, the "Detector" label above the detector controls, and the label, example text and tooltip of "ADetailer detector CLASSES (YOLO-World / YOLOE)", and the texts of the box "Guide: several subjects, one LoRA each" of plus.10 beta 2: its title, its introduction, "What to redraw" and "People in the picture" with their choices and help texts, the labels "Subject 1 LoRA", "Subject 1 weight" and "Subject 1 words" of its four rows, the example text of the words and the "✨ Fill the tabs" button. The status of that box is made when you press the button and is in English. The status notes that beta 2 adds to "Run ADetailer on an image" (a tab skipped for its checkpoint, a detector that could not be used) are made while the tool runs and are in English.
- 🟢 On Forge and Forge Neo, plus.8 keeps the translated "Use same …" values of the ADetailer dropdowns translated after a change; before, they went back to English until the page was reloaded.
- Corrections by native speakers are welcome in the Language Diffusion repository.

## Roadmap

Ideas being considered. None of them is implemented, and none has a release date.

- 🔴 **A switch for what ControlNet sees, the region or the whole picture**: the [IOSakaki/adetailer](https://github.com/IOSakaki/adetailer) fork has a "Use ADetailer crop as ControlNet input" option that gives ControlNet either the crop of the region being detailed or the whole picture. This fork has no such switch. 🟡 Part of the idea is in plus.10 beta 1: on Forge, Forge Neo and reForge, ADetailer's ControlNet now reads each region's own picture and mask, as it already did on AUTOMATIC1111, so with "Inpaint only masked" on it works on the region being detailed, and with it off on the whole picture with the region's mask (see [What is new in plus.10](#what-is-new-in-plus10)). The switch itself is on hold: ControlNet changes often between versions, so this needs a lot of testing.
- 🔴 **Automatic tags for each region**: an image tagger (WD tagger, WDv3) would describe each detected region before it is detailed and add those tags to its prompt. On hold: the tagger is a large extra download (hundreds of megabytes) and needs extra memory. Idea from [Anzhc/aadetailer-reforge](https://github.com/Anzhc/aadetailer-reforge).

More ideas and requests are welcome in the [Issues](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues).

## For developers

### Run the offline tests

In the repository's folder, in a Python environment with the extension's dependencies and the test tools `pytest`, `hypothesis` and `piexif` (without `piexif` the tests of the parameters in JPEG and WebP copies are skipped):

```bash
python -m pytest -m "not integration"
```

With [uv](https://docs.astral.sh/uv/), `uv run --all-extras pytest -m "not integration"` sets up that environment for you.

- The tests marked `integration` run the real detectors and download detector models and sample images, so the command above leaves them out.
- The tests also run on GitHub for pull requests and for pushes to the `main` and `beta` branches. Some of them compare README.md and CHANGELOG.md with the code, so update both files together with a change.

### Report a problem

- Open an [Issue](https://github.com/xXIlRizzoXx/adetailer-ultimate/issues) with the "Bug report" form. Say what you did, what you expected and what happened, which WebUI you use and its version, the list of your installed extensions, and paste the full console log, from start to end. Any language is welcome.
- Please answer follow-up questions: an issue with no activity is marked as stale after 17 days and closed 3 days later.
- For a more useful log, turn on Settings → ADetailer → "Verbose diagnostic log" before you repeat the problem: the console then shows each tab's settings, what was detected and how long each pass took.
- Before you post a log, a screenshot or an image, remove anything personal: your user name in folder paths (for example `C:\Users\your-name\...`, `/home/your-name/...` or `/Users/your-name/...`), folder and file names you do not want to show, any password, API key or token in the "Launching Web UI with arguments" line and in the "Commandline" row of the "System info" table that ADetailer prints with an error, any public link to your WebUI (a `gradio.live`, ngrok or similar address after "Running on public URL"), and prompts you do not want to share. The Verbose diagnostic log also prints the prompt of every region. An image made by the WebUI keeps its prompt and settings inside the file: post a screenshot instead if you do not want to share them.
- If the same problem happens with the original ADetailer, it may belong in [its issue tracker](https://github.com/Bing-su/adetailer/issues).

### Contribute

- Send pull requests to the `beta` branch: changes are tested there first, then reach the stable `main` branch. For a large change, open an Issue first.
- Give every fix a test, and describe changes users will notice in CHANGELOG.md and README.md. Public texts are in English, with generic examples (face, hand, eyes).
- **One rule for new per-tab options:** add them at the end of the list of per-tab options, and never rename, remove or reorder the existing ones, so saved settings, pasted parameters and API calls keep working.

## License

ADetailer Ultimate is distributed under the GNU Affero General Public License v3.0 (AGPL-3.0), like the original ADetailer; the full text is in [LICENSE.md](LICENSE.md). The original ADetailer uses this license because it builds on two AGPL-licensed works, the Stable Diffusion WebUI and Ultralytics. The original author's credits are kept.

## Credits

This fork is not affiliated with or endorsed by Bing-su. It builds on the work of others:

- **[Bing-su/adetailer](https://github.com/Bing-su/adetailer)**: the original ADetailer. Detection and inpainting, the panel and its options, ControlNet support, image parameters, YOLO-World support, the original MediaPipe detectors and the built-in detector models are all Bing-su's work. This fork adds to it; it does not replace it.
- **[wkpark/uddetailer](https://github.com/wkpark/uddetailer)**: the model for the class filter (choosing classes to detail or to leave out, from a list); the preset library was inspired by it too.
- **[Anzhc/aadetailer-reforge](https://github.com/Anzhc/aadetailer-reforge)**: the ideas behind "Apply only on hires.fix", "Scale inpaint to bbox" and the `<lora:name (trigger phrase):weight>` form read by "Append LoRA triggers from name".
- **[newtextdoc1111/adetailer](https://github.com/newtextdoc1111/adetailer)**: the ideas behind "Use bbox as mask" and per-class prompts.
- **[djunk1159/ADetailer_mediapipe_mouth_only](https://github.com/djunk1159/ADetailer_mediapipe_mouth_only)**: the idea behind the `mediapipe_face_features` detector.
- **[Ultralytics](https://github.com/ultralytics/ultralytics)** and **[MediaPipe](https://github.com/google-ai-edge/mediapipe)**: the detection libraries the detectors run on.
- 🟢 The plus.8 update fixes problems that others reported or fixed first: [Bing-su/adetailer#795](https://github.com/Bing-su/adetailer/issues/795), upstream pull request [#796](https://github.com/Bing-su/adetailer/pull/796) and [TheLastAncient/adetailer_steps_fix](https://github.com/TheLastAncient/adetailer_steps_fix) (separate steps); [adieyal/sd-dynamic-prompts#703](https://github.com/adieyal/sd-dynamic-prompts/issues/703), [#730](https://github.com/adieyal/sd-dynamic-prompts/issues/730) and [grey-aogames/adetailer](https://github.com/grey-aogames/adetailer) (Dynamic Prompts template); upstream pull request [#745](https://github.com/Bing-su/adetailer/pull/745) (ControlNet image and mask on Forge).
- 🟢 plus.9 takes up ideas of other forks and requests: [diffus-me/adetailer](https://github.com/diffus-me/adetailer) (LoRA and embedding hashes of the ADetailer prompts), [Anzhc/aadetailer-reforge](https://github.com/Anzhc/aadetailer-reforge) and upstream pull request [#818](https://github.com/Bing-su/adetailer/pull/818) (reForge's Advanced Model Sampling), [otacoo/ADetailer-Classic](https://github.com/otacoo/ADetailer-Classic) (one failed region no longer stops the tab), [y-lan/adetailer](https://github.com/y-lan/adetailer) (the reason of a failed detector download), [abzaloff/aadetailer-neoforge](https://github.com/abzaloff/aadetailer-neoforge) (built-in detectors loaded from your disk first, loading without `NansException`), [IOSakaki/adetailer](https://github.com/IOSakaki/adetailer) (the detector on the CPU when the whole WebUI runs there), [heatmobcompany/sd-webui-adetailer](https://github.com/heatmobcompany/sd-webui-adetailer) (an empty `ad_model` in an API request counts as off), and upstream requests [#111](https://github.com/Bing-su/adetailer/issues/111), [#251](https://github.com/Bing-su/adetailer/issues/251), [#406](https://github.com/Bing-su/adetailer/issues/406), [#508](https://github.com/Bing-su/adetailer/issues/508) and [#635](https://github.com/Bing-su/adetailer/issues/635).
- 🟡 plus.10 beta 1 takes up ideas of other forks and requests: [IOSakaki/adetailer](https://github.com/IOSakaki/adetailer) (the per-tab "Masked content", with the same option name and the same "ADetailer inpaint masked content" key in the image parameters, and Forge Neo's ControlNet batch folder ignored in the detailer pass, both from its commit 649ef43), upstream requests [#516](https://github.com/Bing-su/adetailer/issues/516) and [#802](https://github.com/Bing-su/adetailer/issues/802) (Masked content), [#764](https://github.com/Bing-su/adetailer/issues/764) and [adieyal/sd-dynamic-prompts#561](https://github.com/adieyal/sd-dynamic-prompts/issues/561) (the same wildcard value in the main prompt and the ADetailer prompt), [abzaloff/aadetailer-neoforge](https://github.com/abzaloff/aadetailer-neoforge) and the related upstream reports [#153](https://github.com/Bing-su/adetailer/issues/153), [#163](https://github.com/Bing-su/adetailer/issues/163) and [#619](https://github.com/Bing-su/adetailer/issues/619) (built-in detectors downloaded into `models/adetailer`), the reports [adieyal/sd-dynamic-prompts#669](https://github.com/adieyal/sd-dynamic-prompts/issues/669) and [adieyal/sd-dynamic-prompts#711](https://github.com/adieyal/sd-dynamic-prompts/issues/711) (Dynamic Prompts' Magic Prompt in the detailer pass), and the reports [Bing-su/adetailer#694](https://github.com/Bing-su/adetailer/issues/694) and [AUTOMATIC1111/stable-diffusion-webui#16055](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/16055) (Soft Inpainting).
- Thanks to everyone who reported problems and tested new features in the Issues, and to the users named in [Requested by users](#-requested-by-users).

The fork is directed by its owner, xXIlRizzoXx, who does not write Python. Its code was written by **[Claude](https://www.anthropic.com/claude)**, Anthropic's coding assistant. The plus.8 reliability update was started with **OpenAI Codex** and reviewed and completed with Claude. 🇮🇹

## See also

- [Language Diffusion](https://github.com/xXIlRizzoXx/sd-webui-language-diffusion): the companion extension that translates the WebUI, ADetailer Ultimate included.
- The original [ADetailer](https://github.com/Bing-su/adetailer) and its wiki: [REST API](https://github.com/Bing-su/adetailer/wiki/REST-API) and [Advanced](https://github.com/Bing-su/adetailer/wiki/Advanced) (the `[SEP]`, `[SKIP]` and `[PROMPT]` prompt tokens).
- Guides to the original ADetailer, which apply to this fork too: [ADetailer Installation and 5 Usage Methods](https://kindanai.com/en/manual-adetailer/), and a detailed video guide in Japanese, [part 1](https://youtu.be/sF3POwPUWCE) and [part 2](https://youtu.be/urNISRdbIEg).
- Other extensions for detailing and masking: [sd-face-editor](https://github.com/ototadana/sd-face-editor), [sd-webui-segment-anything](https://github.com/continue-revolution/sd-webui-segment-anything) and [sd-webui-bmab](https://github.com/portu-sim/sd-webui-bmab).
