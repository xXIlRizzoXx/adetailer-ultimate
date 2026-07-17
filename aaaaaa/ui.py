from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gradio as gr

from aaaaaa.conditional import InputAccordion
from adetailer import ADETAILER, __version__
from adetailer.args import ALL_ARGS, MASK_MERGE_INVERT
from adetailer.classes import MEDIAPIPE_FACE_FEATURES_MODEL, get_model_class_names
from adetailer.persistence import load_state, save_tab_state
from adetailer.presets import (
    delete_preset,
    export_presets_json,
    get_preset,
    get_preset_names,
    import_presets_json,
    rename_preset,
    save_preset,
)
from controlnet_ext import controlnet_exists, controlnet_type, get_cn_models

if controlnet_type == "forge":
    from lib_controlnet import global_state

    cn_module_choices = {
        "inpaint": list(global_state.get_filtered_preprocessors("Inpaint")),
        "lineart": list(global_state.get_filtered_preprocessors("Lineart")),
        "openpose": list(global_state.get_filtered_preprocessors("OpenPose")),
        "tile": list(global_state.get_filtered_preprocessors("Tile")),
        "scribble": list(global_state.get_filtered_preprocessors("Scribble")),
        "depth": list(global_state.get_filtered_preprocessors("Depth")),
    }
else:
    cn_module_choices = {
        "inpaint": [
            "inpaint_global_harmonious",
            "inpaint_only",
            "inpaint_only+lama",
        ],
        "lineart": [
            "lineart_coarse",
            "lineart_realistic",
            "lineart_anime",
            "lineart_anime_denoise",
        ],
        "openpose": ["openpose_full", "dw_openpose_full"],
        "tile": ["tile_resample", "tile_colorfix", "tile_colorfix+sharp"],
        "scribble": ["t2ia_sketch_pidi"],
        "depth": ["depth_midas", "depth_hand_refiner"],
    }

union = list(chain.from_iterable(cn_module_choices.values()))
cn_module_choices["union"] = union


def _apply_exif_orientation(im):
    """Rotate a user-supplied image to upright per its EXIF orientation tag.

    Phone/camera photos store the sensor orientation as a tag (274) instead of
    rotating the pixels; a detector fed the un-rotated pixels sees a sideways
    face and misses it (or finds garbage). This mirrors Gradio's own upload
    guard exactly — it only acts when the tag says "rotated" and
    ``exif_transpose`` clears the tag, so it is a strict no-op on images Gradio
    (or anyone else) already transposed and can never double-rotate. Guarded:
    returns the image unchanged on anything unexpected (e.g. a numpy array)."""
    try:
        from PIL import Image as _PImg
        from PIL import ImageOps

        if not isinstance(im, _PImg.Image):
            return im
        if im.getexif().get(274, 1) != 1:
            return ImageOps.exif_transpose(im)
    except Exception:  # noqa: BLE001
        pass
    return im


class Widgets(SimpleNamespace):
    def tolist(self):
        return [getattr(self, attr) for attr in ALL_ARGS.attrs]


@dataclass
class WebuiInfo:
    ad_model_list: list[str]
    sampler_names: list[str]
    scheduler_names: list[str]
    t2i_button: gr.Button
    i2i_button: gr.Button
    checkpoints_list: list[str]
    vae_list: list[str]
    # Forge / Forge Neo only: available text-encoder modules for the per-pass
    # "separate text encoder" option. Empty on A1111 (no such concept there), so
    # the dropdown just offers the pass-through / "None" choices — index-safe.
    text_encoders_list: list[str] = field(default_factory=list)
    model_mapping: dict[str, str] = field(default_factory=dict)


def gr_interactive(value: bool = True):
    return gr.update(interactive=value)


def _read_git_short_hash() -> str:
    """Best-effort: return the current commit's 7-char short hash by
    reading `<extension_root>/.git/HEAD` and following the ref chain.

    Falls back to an empty string if anything goes wrong (e.g. the user
    installed from a downloaded zip with no .git directory, or the
    HEAD/ref files are corrupt). Used by the accordion header overlay
    so the user can tell at a glance which commit they have installed —
    the overlay auto-updates the next time `adui()` rebuilds the panel
    after a `git pull`.

    Implementation note: avoids `subprocess` so this is cheap (just two
    file reads) and works on any platform without needing `git` on PATH.
    """
    try:
        # Walk up from aaaaaa/ui.py to the extension root (which has .git).
        ext_root = Path(__file__).resolve().parent.parent
        git_dir = ext_root / ".git"
        head_file = git_dir / "HEAD"
        if not head_file.is_file():
            return ""
        head = head_file.read_text(encoding="utf-8").strip()
        if not head:
            return ""
        if head.startswith("ref: "):
            # Symbolic ref → resolve to actual hash.
            ref_name = head[5:].strip()
            ref_file = git_dir / ref_name
            if ref_file.is_file():
                return ref_file.read_text(encoding="utf-8").strip()[:7]
            # Packed refs fallback (after `git gc`).
            packed = git_dir / "packed-refs"
            if packed.is_file():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if not line or line.startswith("#") or line.startswith("^"):
                        continue
                    parts = line.split()
                    if len(parts) == 2 and parts[1] == ref_name:
                        return parts[0][:7]
            return ""
        # Detached HEAD — `head` itself is the full hash.
        return head[:7]
    except OSError:
        return ""


def _build_overlay_text() -> str:
    """Assemble the top-right header overlay text:
    `ADetailer Ultimate · v<version> · <git-short>`

    The git short-hash is appended only when readable, so packaged
    installs without a .git directory degrade to the brand+version form.
    """
    git_short = _read_git_short_hash()
    base = f"ADetailer Ultimate · v{__version__}"
    if git_short:
        return f"{base} · {git_short}"
    return base


# In-extension user guide — rendered as a dedicated top-level "ADetailer Guide"
# tab (see `build_guide_blocks` here + the `on_ui_tabs` callback in
# scripts/!adetailer.py): an intro, a table of contents, and one collapsible
# section per topic. Deliberately English-only reference text — like the
# technical vocabulary it is NOT a translation key, so the markdown formatting
# here does NOT fall under the single-plain-block i18n rule. Keep it in sync
# with README.md when features change. Items marked (new) are fork additions.

_GUIDE_INTRO_MD = """\
# 📖 ADetailer Guide

ADetailer automatically finds parts of an image — faces, hands, eyes, whole subjects… — and re-generates just those regions at higher quality, with no manual inpainting. It runs **after** your normal generation, on the finished image.

Everything is optional: turn ADetailer on, pick a detector, and the defaults already work. Expand a topic below to read about it. Controls marked **(new)** are additions of this fork.
"""

_GUIDE_SECTIONS: list[tuple[str, str]] = [
    (
        "🚀 Getting started",
        """
- **Enable ADetailer** — the checkbox on the accordion header turns it on for this generation.
- **ADetailer detector** — the first dropdown at the top of each tab. This is the model that finds what to fix (a face model, hand model, a segmentation or MediaPipe model…). `None` disables that tab.
- **Tabs (1st, 2nd, …)** — each numbered tab is an independent detector + its own settings, and they run **in order** (e.g. tab 1 = faces, tab 2 = hands). Change how many tabs you have in `Settings → ADetailer`.
- **First run:** enable ADetailer, pick `face_yolov8n.pt` in the detector dropdown, and generate — every face is detected and re-detailed automatically.
""",
    ),
    (
        "🔎 Choosing a detector",
        """
- **YOLO models** (`face_yolov8n`, `hand_yolov8n`, `person_yolov8n`, …) — fast detectors. The `n`/`s`/`m` suffix is size; `n` is fastest, larger is more accurate.
- **Segmentation vs box** — segmentation models give a precise mask shape; box-only models give a rectangle. You can force the rectangle with *Use bbox as mask* (see Mask preprocessing).
- **Multi-class models** — a model that knows several classes (e.g. a person/face/hand model, or a custom one) shows the **CLASSES** dropdown so you can pick which parts to detail.
- **MediaPipe** — `mediapipe_face_mesh` and `..._eyes_only` detect faces/eyes by landmarks, without a model file.
- **MediaPipe face features (new)** — `mediapipe_face_features` treats the face as classes **eyes, mouth, nose, eyebrows, face**; pick all, some, or just one in the CLASSES dropdown.
- **YOLO-World (`…-world`)** — open vocabulary: type comma-separated words in the classes field to detect almost anything you can name.
- **Custom models** — drop a `.pt` into your ADetailer models folder. If its classes don't appear, add a sidecar file `<model>.names.json` shaped like `{"names": {"0": "face", "1": "hand"}}`.
""",
    ),
    (
        "🎯 Detection settings",
        """
- **Detection model confidence threshold** — how sure the detector must be. Lower finds more (and more false) detections.
- **Detection resolution (0 = default) (new)** — run the detector at a higher internal resolution (e.g. 1024 instead of the default 640) to catch **small or distant** parts. Higher finds more but uses more VRAM and time. The single-tab *Detection preview* honours it too.
- **ADetailer detector CLASSES** — for multi-class models, choose which parts to detail. Empty = all of them.
- **Exclude selected (NOT)** — invert the choice: detail everything **except** the selected classes.
- **Process classes sequentially** — one detect+inpaint pass per selected class, in the order you clicked them. Required for per-class prompts.
- **Method to filter / Mask only the top k** — keep only the `k` biggest or most-confident detections; `0` keeps all.
""",
    ),
    (
        "🖌️ Mask preprocessing",
        """
- **Use bbox as mask (segmentation models)** — inpaint the rectangle instead of the tight segmentation shape, giving the region more room to blend. No effect on box-only detectors.
- **Mask erosion (−) / dilation (+)** — shrink or grow the detected mask.
- **Mask x / y offset** — nudge the mask left/right or up/down.
- **Mask merge mode** — None, Merge, or Merge and Invert: combine overlapping masks, optionally inverting what gets inpainted.
""",
    ),
    (
        "✍️ Prompts",
        """
- **Prompt / Negative prompt** — text used for the detail pass. Leave blank to reuse the main prompt.
- **Prompt append / Negative prompt append** — add a few words to the end without retyping the whole prompt.
- **Per-class prompts** — a different prompt (and optional negative) per class. One line each: `classname: positive | negative`. Works with **Process classes sequentially** on and 2+ classes selected.
- **Inline `[CLASS=name] … [/CLASS]` (new)** — write class-specific text right inside the prompt, e.g. `portrait [CLASS=hand] five fingers [/CLASS]`. Each detected region keeps only the blocks matching its class; text outside blocks applies to every region. Works in **normal and sequential** mode, the tag can list several classes (`[CLASS=face,eyes]`), and `[CLASS=hand] [SKIP] [/CLASS]` skips only that class. Needs a class-aware detector; inert unless you type the tags.
- **Auto class-guard (new)** — for each detected region, adds its own class name to the positive and every **other** class of the model to the negative, so a detected part isn't re-drawn as a different one. Your per-class prompts always override it. **Class-guard emphasis** weights the added class name (1.0 = plain).
- **Use LoRAs from main prompt** (+ **Append LoRA triggers from name**) — copy the LoRA tags (and their trigger words) from the main prompt into the detail pass.
- **Strip LoRAs from the detailer prompt (new)** — remove all `<lora:…>` / `<lyco:…>` tags from the detail prompt so the main prompt's LoRAs don't bleed onto the region. Wins over *Use LoRAs from main prompt* when both are on.
""",
    ),
    (
        "🎨 Inpainting",
        """
- **Inpaint denoising strength** — how much the region changes. Lower keeps the original shape; higher redraws more. ~0.3–0.5 is typical; low values help a detected class keep looking like itself.
- **Dynamic denoise by area (new)** — automatically gives smaller detected regions more denoise. `0` = use the global Settings value; `2`–`4` is a good range.
- **Inpaint mask blur** — soften the mask edge for smoother blending.
- **Inpaint only masked** (+ padding) — regenerate just the region at full detail (recommended), keeping some surrounding context.
- **Separate inpaint width/height** or **scale** — set the working resolution of the region.
""",
    ),
    (
        "⚙️ Per-pass overrides (optional)",
        """
Use different settings **only** for the detail pass; each is off (inherits the main generation) by default.
- **Detailer checkpoint / VAE** — run the detail pass with a different model / VAE.
- **Separate text encoder** — Forge / Forge Neo only; use a different text encoder for the detail pass (e.g. an SDXL detailer under a different base).
- **Sampler / scheduler / steps / CFG scale / CLIP skip** — per-pass sampling settings.
- **Apply only on hires. fix** — run this tab only when hires.fix is on.
""",
    ),
    (
        "🧰 Preview & run-on-image tools",
        """
Two sibling tools live inside each tab's **Detection** section (roughly the middle of the tab, above Mask preprocessing):
- **Detection preview** — drop an image to see what the detector would find (boxes), without generating. The single-tab preview honours this tab's **Detection resolution**. **Combine all tabs** overlays every tab's detector at once (that combined view stays at the default resolution).
- **Run ADetailer on an image** — drop a finished image and run the full detect + inpaint pass on it, without regenerating. Optional **Save result to outputs** writes to a dedicated `ADetailer-Inpaint` folder; click the result to enlarge it.
- **Batch a whole folder (new)** — paste a **folder path** in that same tool to detail every image inside it in one go; each result is always saved to the `ADetailer-Inpaint` folder. A folder path takes priority over a single dropped image.
""",
    ),
    (
        "💾 Presets, copy/paste & saving",
        """
- **Preset library** — save / load / rename / delete a whole tab's settings by name; **export / import** the library as a file to move it between machines.
- **Copy settings / Paste settings** — clone one tab's settings into another.
- **Remember last-used settings between restarts** (`Settings → ADetailer`) — restore your last setup after a restart.
- **Reset ADetailer settings** (`Settings → ADetailer`) — restore factory defaults.
- **Where files go** — intermediate step/preview files land in an `adetailer-steps/` sub-folder; the standalone *Run/Batch* results go in a top-level `ADetailer-Inpaint/` folder next to your normal outputs.
""",
    ),
    (
        "🛠️ Fixing common problems",
        """
- **The wrong thing gets regenerated** (e.g. one part redrawn as another) — keep denoise low and turn on **Auto class-guard**, or write **per-class prompts** / **inline `[CLASS=]`** so each class stays itself.
- **Small or distant faces are missed** — raise **Detection resolution** (e.g. 1024), and/or lower the confidence threshold.
- **The face no longer looks like the character** — lower **Inpaint denoising strength**.
- **A style LoRA bleeds onto the detailed region** — turn on **Strip LoRAs from the detailer prompt**.
- **The mask is too tight / edges show** — use **bbox as mask** or **dilation**, and increase **mask blur**.
- **CUDA out of memory** — lower the inpaint resolution; for many images use **batch count**, not batch size; drop the `--cuda-malloc` / `--cuda-stream` / `--pin-shared-memory` launch flags; and turn off on-the-fly LoRA patching if your UI has it.
""",
    ),
    (
        "🔡 Special tokens (advanced)",
        """
These go inside the tab's **Prompt** field:
- **`[SEP]`** — split the prompt so each detected region (1st, 2nd, …) gets its own part.
- **`[SKIP]`** — skip a region entirely. On its own it skips all; inside a class block (`[CLASS=hand] [SKIP] [/CLASS]`) it skips only that class.
- **`[PROMPT]`** — insert the main generation prompt at that spot.
""",
    ),
    (
        "✨ What's new in this fork",
        """
Highlights added on top of upstream ADetailer:
- **Class filtering** (pick which parts), **Exclude (NOT)** mode, **sequential** per-class passes, and **per-class prompts**.
- **Auto class-guard**, **inline `[CLASS=]`** prompts, **Strip LoRAs**, **LoRAs + triggers from the main prompt**.
- **HD Detection resolution**, **Dynamic denoise by area**, **per-pass text encoder** (Forge/Neo).
- **MediaPipe face-features** detector (eyes/mouth/nose/eyebrows/face).
- **Run ADetailer on an image** and **batch a whole folder** without regenerating.
- **Preset library** with export/import, **copy/paste between tabs**, **remembered settings**, and this guide.
- Built to work across **A1111, Forge, Forge Neo and reForge**.
""",
    ),
]


def build_guide_blocks():
    """Build the standalone "📖 ADetailer Guide" top-level tab: an intro, a
    table of contents, and one collapsible section per topic. Reference
    documentation (English); static — no event listeners — so it is fully
    isolated from the main UI wiring. Returns a `gr.Blocks` for `on_ui_tabs`.
    """
    with gr.Blocks(analytics_enabled=False) as blocks:
        with gr.Column(elem_classes=["ad-guide-tab"]):
            gr.Markdown(_GUIDE_INTRO_MD, elem_classes=["ad-guide"])
            toc = "\n".join(f"1. {title}" for title, _ in _GUIDE_SECTIONS)
            gr.Markdown(
                "### Contents\n" + toc, elem_classes=["ad-guide", "ad-guide-toc"]
            )
            for i, (title, body) in enumerate(_GUIDE_SECTIONS):
                with gr.Accordion(
                    title, open=(i == 0), elem_classes=["ad-guide-section"]
                ):
                    gr.Markdown(body.strip(), elem_classes=["ad-guide"])
    return blocks


def _format_preset_preview(name: str | None):
    """Build a compact markdown summary of a preset's contents for the
    live preview area below the preset dropdown.

    Triggers on every preset_dropdown.change event so the user can flip
    through saved presets and read a one-glance summary before deciding
    whether to Load.

    Returns a `gr.update(...)` rather than a raw string so the function
    also controls the markdown container's visibility — when there's
    nothing to show, the container hides entirely (avoiding an empty
    styled box between the preset row and the detector section).

    `[SEP]` and `[PROMPT]` tokens are flagged in the rendered preview
    because they will be expanded at generation time against the main
    txt2img/img2img prompt — the user is warned to expect different
    final prompts than what they see here verbatim.
    """
    from adetailer.presets import get_preset

    name = (name or "").strip()
    if not name or name == PRESET_NONE:
        return gr.update(value="", visible=False)
    data = get_preset(name)
    if not data:
        return gr.update(
            value=f"_(preset '{name}' not found on disk)_", visible=True
        )

    def _trunc(s: str, n: int = 140) -> str:
        s = (s or "").strip()
        return s if len(s) <= n else s[: n - 1].rstrip() + "…"

    def _flag_tokens(s: str) -> str:
        # Wrap [SEP]/[PROMPT] in backticks so they're visually distinct
        # — note to the user that these are placeholders, not literals.
        return (
            (s or "")
            .replace("[SEP]", "`[SEP]`")
            .replace("[PROMPT]", "`[PROMPT]`")
        )

    pos = _flag_tokens(_trunc(data.get("ad_prompt", "")))
    neg = _flag_tokens(_trunc(data.get("ad_negative_prompt", "")))
    model = (data.get("ad_model") or "").strip() or "_(not set)_"
    classes = (data.get("ad_model_classes") or "").strip()
    excluded = (data.get("ad_model_classes_excluded") or "").strip()
    seq = bool(data.get("ad_classes_sequential", False))

    lines: list[str] = [f"**Preview — {name}**"]
    lines.append(f"- **Detector:** {model}")
    if classes:
        lines.append(f"- **Classes (include):** {classes}")
    if excluded:
        lines.append(f"- **Classes (exclude):** {excluded}")
    if seq:
        lines.append("- **Sequential class detection:** on")
    if pos:
        lines.append(f"- **Prompt:** {pos}")
    if neg:
        lines.append(f"- **Negative:** {neg}")
    # Per-class prompts (fork feature) — flag presence without dumping all.
    class_prompts = (data.get("ad_class_prompts") or "").strip()
    if class_prompts:
        first_classes = [
            line.split(":", 1)[0].strip()
            for line in class_prompts.splitlines()
            if ":" in line
        ][:5]
        lines.append(
            "- **Class-specific prompts:** "
            + (", ".join(first_classes) if first_classes else "_(set)_")
        )
    if "[SEP]" in (data.get("ad_prompt", "") + data.get("ad_negative_prompt", "")) or "[PROMPT]" in (
        data.get("ad_prompt", "") + data.get("ad_negative_prompt", "")
    ):
        lines.append(
            "- _`[SEP]` / `[PROMPT]` tokens will be expanded against the main prompt at generation time._"
        )
    return gr.update(value="\n".join(lines), visible=True)


def ordinal(n: int) -> str:
    d = {1: "st", 2: "nd", 3: "rd"}
    return str(n) + ("th" if 11 <= n % 100 <= 13 else d.get(n % 10, "th"))


def suffix(n: int, c: str = " ") -> str:
    return "" if n == 0 else c + ordinal(n + 1)


def on_widget_change(state: dict, value: Any, *, attr: str):
    if "is_api" in state:
        state = state.copy()
        state.pop("is_api")
    state[attr] = value
    return state


def on_generate_click(
    state: dict, *values: Any, mode: str = "txt2img", tab_index: int = 0
):
    for attr, value in zip(ALL_ARGS.attrs, values):
        state[attr] = value  # noqa: PERF403
    state["is_api"] = ()
    # Best-effort persistence: stash the just-clicked values so they come
    # back as the defaults at next WebUI start. The (mode, tab_index)
    # pair scopes the persisted state by pipeline so txt2img and img2img
    # don't overwrite each other. Never raise — see adetailer.persistence
    # for the swallowed-error policy.
    save_tab_state(mode, tab_index, state)
    # Diagnostic log (console) — show exactly what each active tab persisted on
    # this Generate, so saved settings are visible/auditable in the log (paired
    # with the restore log in one_ui_group). Only logs tabs with a real
    # detector to avoid noise. Plain print → index-safe.
    if state.get("ad_model") and state.get("ad_model") != "None":
        _cls = (
            state.get("ad_model_classes_excluded")
            if state.get("ad_model_classes_exclude")
            else state.get("ad_model_classes")
        )
        _mode_word = "NOT/exclude" if state.get("ad_model_classes_exclude") else "include"
        print(
            f"[-] ADetailer: saved tab {tab_index + 1} ({mode}) — "
            f"detector={state.get('ad_model')!r}, classes[{_mode_word}]={_cls!r}"
        )
    return state


def _sv(saved: dict[str, Any], attr: str, default):
    """Saved value for `attr` if present, else `default`."""
    return saved.get(attr, default) if attr in saved else default


def on_ad_model_update(
    model: str,
    current_selection: list | None = None,
    model_mapping: dict[str, str] | None = None,
):
    """Return updates for (textbox, dropdown, exclude-checkbox, excluded-textbox).

    The dropdown and exclude checkbox stay always visible so the layout is
    predictable across model changes. They're populated with the model's
    class names when applicable, empty otherwise.

    - YOLO-World: ALSO shows the free-text textbox (open-vocabulary).
    - Other multiclass YOLO: dropdown populated from model.names. Any
      current selections that are still valid in the new model are
      preserved — this is what lets Copy/Paste between tabs keep the
      class-filter state when the detector matches.
    - MediaPipe / None: dropdown shown but empty.
    """
    if (
        not model
        or model == "None"
        or (
            model.lower().startswith("mediapipe")
            and model != MEDIAPIPE_FACE_FEATURES_MODEL
        )
    ):
        return (
            gr.update(visible=False, value=""),
            gr.update(visible=True, choices=[], value=[]),
            gr.update(visible=True, value=False),
            gr.update(value=""),
        )

    if "-world" in model:
        return (
            gr.update(
                visible=True,
                value="",
                placeholder="Comma separated class names to detect, ex: 'person,cat'. default: COCO 80 classes",
            ),
            gr.update(visible=True, choices=[], value=[]),
            gr.update(visible=True, value=False),
            gr.update(value=""),
        )

    mapping = model_mapping or {}
    path = mapping.get(model, "")
    names = get_model_class_names(path) if path else []
    # Preserve selections that still exist in the new model's class list.
    preserved = [s for s in (current_selection or []) if s in names]
    # Also feed the preserved classes into the hidden backing textbox.
    # Programmatic dropdown updates don't fire `.change`, so without this the
    # textbox would stay empty even though the dropdown shows the preserved
    # tokens (e.g. after Copy/Paste or re-selecting the same model), desyncing
    # the filter. Checkbox is reset to include-mode here, so the value goes to
    # ad_model_classes (not ad_model_classes_excluded).
    return (
        gr.update(visible=False, value=",".join(preserved)),
        gr.update(visible=True, choices=names, value=preserved),
        gr.update(visible=True, value=False),
        gr.update(value=""),
    )


def on_cn_model_update(cn_model_name: str):
    cn_model_name = cn_model_name.replace("inpaint_depth", "depth")
    for t in cn_module_choices:
        if t in cn_model_name:
            choices = cn_module_choices[t]
            return gr.update(visible=True, choices=choices, value=choices[0])
    return gr.update(visible=False, choices=["None"], value="None")


def elem_id(item_id: str, n: int, is_img2img: bool) -> str:
    tab = "img2img" if is_img2img else "txt2img"
    suf = suffix(n, "_")
    return f"script_{tab}_adetailer_{item_id}{suf}"


def state_init(w: Widgets) -> dict[str, Any]:
    return {attr: getattr(w, attr).value for attr in ALL_ARGS.attrs}


def adui(
    num_models: int,
    is_img2img: bool,
    webui_info: WebuiInfo,
    script=None,
):
    states = []
    infotext_fields = []
    eid = partial(elem_id, n=0, is_img2img=is_img2img)

    # Load per-tab saved state from disk once per UI build. Scoped by
    # mode (txt2img vs img2img) so each pipeline has its own last-used
    # values. The dict is passed down into one_ui_group so each widget
    # can read its previous value as a default.
    saved_state = load_state("img2img" if is_img2img else "txt2img")

    with InputAccordion(
        value=False,
        elem_id=eid("ad_main_accordion"),
        label=ADETAILER,
        visible=True,
    ) as ad_enable:
        # Version "about" badge — CSS pulls it out of normal flow and overlays
        # it onto the accordion header. Lives inside the accordion content so
        # it's automatically hidden when the accordion is collapsed.
        # Format (since 2026-05-16): brand prefix + locked __version__ +
        # current git short-hash. The hash gives a live, auto-updating
        # indicator of "which commit is installed" — refreshes every time
        # adui() rebuilds the panel after a `git pull`. Locked __version__
        # is left alone per the no-auto-bump rule.
        gr.Markdown(
            _build_overlay_text(),
            elem_id=eid("ad_version"),
            elem_classes=["ad-version-overlay"],
        )

        with gr.Row():
            ad_skip_img2img = gr.Checkbox(
                label="Skip img2img",
                value=False,
                visible=is_img2img,
                elem_id=eid("ad_skip_img2img"),
            )

        infotext_fields.append((ad_enable, "ADetailer enable"))
        infotext_fields.append((ad_skip_img2img, "ADetailer skip img2img"))

        # Shared clipboard for per-tab copy/paste. Tuple of (source_tab_idx,
        # list_of_values_in_copyable_attrs_order). source_tab_idx == -1 means
        # "clipboard empty" — paste buttons stay disabled until the first copy.
        clipboard_state = gr.State((-1, []))

        all_widgets: list[Widgets] = []
        all_copy_btns: list[gr.Button] = []
        all_paste_btns: list[gr.Button] = []
        all_presets: list[tuple] = []

        with gr.Group(), gr.Tabs():
            for n in range(num_models):
                with gr.Tab(ordinal(n + 1)):
                    (
                        w,
                        copy_btn,
                        paste_btn,
                        preset_widgets,
                        state,
                        infofields,
                    ) = one_ui_group(
                        n=n,
                        is_img2img=is_img2img,
                        webui_info=webui_info,
                        saved_tab_state=saved_state.get(str(n), {}),
                    )

                all_widgets.append(w)
                all_copy_btns.append(copy_btn)
                all_paste_btns.append(paste_btn)
                all_presets.append(preset_widgets)
                states.append(state)
                infotext_fields.extend(infofields)

        # Second pass: now that every tab's widgets + buttons exist, wire the
        # cross-tab copy/paste handlers. We need refs to ALL paste buttons to
        # update their labels when ANY tab does a copy, so this can't be done
        # inside one_ui_group.
        _wire_copy_paste(
            all_widgets, all_copy_btns, all_paste_btns, clipboard_state, num_models
        )
        _wire_presets(
            all_widgets,
            all_presets,
            all_paste_btns,
            clipboard_state,
            num_models,
        )
        # Cross-tab Detection-preview wiring (also a post-loop "second pass"):
        # lets each tab's "Combine all tabs" checkbox run every tab's detector.
        _wire_detection_previews(all_widgets, webui_info, num_models, script)

        # The full user guide lives in its own top-level "ADetailer Guide" tab
        # (opt-in, off by default — see the on_ui_tabs gate + the
        # ad_show_guide_tab setting). Leave a small, non-invasive pointer that is
        # accurate whether the tab is on or off. Static markdown only (no event
        # listeners) → index-safe. Single plain-text line (no inline markdown) so
        # it stays translatable.
        gr.Markdown(
            "📖 A full guide to every option is available as its own top tab — "
            "turn on \"Show the ADetailer Guide tab\" in Settings → ADetailer to "
            "display it.",
            elem_classes=["ad-guide-pointer"],
        )

    # components: [bool, bool, dict, dict, ...]
    components = [ad_enable, ad_skip_img2img, *states]
    return components, infotext_fields


def _wire_detection_previews(all_widgets, webui_info, num_models, script=None):
    """Wire every tab's Detection-preview button AFTER all tabs exist, so the
    per-tab "Combine all tabs" checkbox can run EVERY configured tab's detector
    on one image and overlay all the boxes.

    Index-safe by construction: registers exactly ONE ``.click`` per tab — the
    same count the old per-tab wiring inside ``one_ui_group`` had, just
    relocated here (mirrors ``_wire_copy_paste`` / ``_wire_presets``). The net
    number of Gradio event listeners is unchanged, so Forge's gallery buttons
    (wired after the scripts' UI) keep their fn_index and don't break. The
    checkbox has NO listener of its own; it is only read as an ``inputs`` value.
    """
    model_mapping = webui_info.model_mapping
    tab_colors = [
        (255, 64, 64),
        (64, 200, 64),
        (64, 160, 255),
        (255, 200, 0),
        (210, 64, 255),
        (0, 210, 210),
    ]

    def _predict(
        image, model_name, classes_csv, exclude_csv, exclude_mode, confidence, imgsz=0
    ):
        """Run one detector. Returns (PredictOutput | None, error_str | None).
        `imgsz` (0 = default 640) is the optional HD detection resolution; it
        only affects ultralytics models — mediapipe ignores it."""
        if not model_name or model_name == "None":
            return None, "no model"
        try:
            if model_name.lower().startswith("mediapipe"):
                from adetailer.mediapipe import mediapipe_predict

                return (
                    mediapipe_predict(
                        model_name,
                        image,
                        float(confidence),
                        classes=classes_csv or "",
                        exclude_classes=(exclude_csv if exclude_mode and exclude_csv else ""),
                    ),
                    None,
                )
            from adetailer.ultralytics import ultralytics_predict

            path = model_mapping.get(model_name, "")
            if not path:
                return None, f"path not found for '{model_name}'"
            try:
                from aaaaaa.helper import disable_safe_unpickle

                cm = disable_safe_unpickle()
            except Exception:  # noqa: BLE001
                from contextlib import nullcontext

                cm = nullcontext()
            with cm:
                pred = ultralytics_predict(
                    path,
                    image=image,
                    confidence=float(confidence),
                    device="",
                    classes=classes_csv or "",
                    exclude_classes=(exclude_csv if exclude_mode and exclude_csv else ""),
                    imgsz=int(imgsz or 0),
                )
            return pred, None
        except Exception as e:  # noqa: BLE001 — surface to the UI
            return None, str(e)

    def _make_handler(tab_idx):
        def _run(image, combine, run_inpaint, save, *flat):
            if image is None:
                return None, "⚠️ Drop an image into the Input box first."
            image = _apply_exif_orientation(image)
            per_tab = [flat[i * 5 : (i + 1) * 5] for i in range(num_models)]

            # "Also run ADetailer": run the FULL detect+inpaint pass on the
            # input image using THIS tab's settings, without regenerating
            # (issue #4). This tab's full ADetailerArgs are appended to `flat`
            # after the per-tab detector fields; rebuild them and hand off to
            # the script. run_detailer_on_image is guarded end-to-end.
            if run_inpaint:
                if script is None:
                    return None, "⚠️ ADetailer run isn't available here."
                from adetailer.args import ADetailerArgs

                arg_vals = flat[num_models * 5 :]
                try:
                    # is_api's pre-validator resolves the () sentinel to False
                    # (a plain False would be inverted to True); () is exactly
                    # what a real UI generation pass sends, so this is parity.
                    args_obj = ADetailerArgs(
                        **dict(zip(list(ALL_ARGS.attrs), arg_vals)),
                        is_api=(),
                    )
                except Exception as e:  # noqa: BLE001
                    return None, f"⚠️ Couldn't read this tab's settings: {e}"
                if not args_obj.ad_model or args_obj.ad_model == "None":
                    return None, "⚠️ Pick a detector model first."
                return script.run_detailer_on_image(image, args_obj, save=save)

            # Single-tab: reuse the rich ultralytics/mediapipe plot (class
            # labels + confidence baked in by the detector's own plotter).
            if not combine:
                model_name, classes_csv, exclude_csv, exclude_mode, confidence = per_tab[
                    tab_idx
                ]
                if not model_name or model_name == "None":
                    return None, "⚠️ Pick a detector model first."
                # HD preview: honour THIS tab's "Detection resolution" so the
                # single-tab preview reflects what the real pass will detect.
                # Combine-all-tabs stays at the default 640 — it only has the 5
                # detector fields per tab, not each tab's full args. 0 = default.
                imgsz = 0
                try:
                    _full = flat[num_models * 5 :]
                    _i = list(ALL_ARGS.attrs).index("ad_detection_resolution")
                    imgsz = int(_full[_i] or 0)
                except Exception:  # noqa: BLE001
                    imgsz = 0
                pred, err = _predict(
                    image, model_name, classes_csv, exclude_csv, exclude_mode,
                    confidence, imgsz,
                )
                if err:
                    return None, f"⚠️ Preview failed: {err}"
                n = len(pred.bboxes) if pred and pred.bboxes else 0
                if (pred is None or pred.preview is None) and n == 0:
                    return None, "ℹ️ No detections."
                return pred.preview, f"✅ {n} detection(s)."

            # Combined: overlay every configured tab's detections on one
            # canvas. For SEGMENTATION models we tint the real mask SHAPE (not
            # just the bbox square) so the analysed part is shown in full;
            # box-only models fall back to a filled rectangle. Readable labels
            # (tab# + class name + confidence on a filled background) go on top.
            from PIL import Image as _PILImage
            from PIL import ImageDraw, ImageFont

            targets = [
                i for i in range(num_models) if per_tab[i][0] and per_tab[i][0] != "None"
            ]
            if not targets:
                return None, "⚠️ No tab has a detector model selected."

            # Run each tab's detector once; collect its results.
            results = []  # (tab_idx, color, bboxes, confs, names, masks)
            summary = []
            total = 0
            for i in targets:
                model_name, classes_csv, exclude_csv, exclude_mode, confidence = per_tab[i]
                pred, err = _predict(
                    image, model_name, classes_csv, exclude_csv, exclude_mode, confidence
                )
                if err:
                    summary.append(f"Tab{i + 1}: ERR")
                    continue
                bboxes = (pred.bboxes if pred else None) or []
                results.append(
                    (
                        i,
                        tab_colors[i % len(tab_colors)],
                        bboxes,
                        (pred.confidences if pred else None) or [],
                        getattr(pred, "class_names", None) or [],
                        (pred.masks if pred else None) or [],
                    )
                )
                total += len(bboxes)
                summary.append(f"Tab{i + 1}: {len(bboxes)}")

            if total == 0:
                return (
                    image.convert("RGB"),
                    "ℹ️ No detections across tabs — " + " | ".join(summary),
                )

            # Pass 1: tint each region with its tab colour using the detector's
            # mask (real seg shape, or bbox rectangle for box-only models) at
            # ~43% opacity via alpha compositing.
            canvas = image.convert("RGBA")
            for _i, color, _bb, _cf, _nm, masks in results:
                for m in masks:
                    if m is None:
                        continue
                    try:
                        alpha = m.convert("L").point(lambda v: v * 110 // 255)
                        tint = _PILImage.new("RGBA", canvas.size, color + (0,))
                        tint.putalpha(alpha)
                        canvas = _PILImage.alpha_composite(canvas, tint)
                    except Exception:  # noqa: BLE001 — cosmetic; skip a bad mask
                        continue
            canvas = canvas.convert("RGB")

            # Pass 2: box outline + readable label on top of the tints.
            draw = ImageDraw.Draw(canvas)
            fsize = max(15, canvas.height // 45)
            line_w = max(3, canvas.height // 320)
            font = None
            for _fname in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
                try:
                    font = ImageFont.truetype(_fname, fsize)
                    break
                except Exception:  # noqa: BLE001
                    continue
            if font is None:
                try:
                    font = ImageFont.load_default(size=fsize)  # Pillow >= 10.1
                except Exception:  # noqa: BLE001
                    font = ImageFont.load_default()

            for i, color, bboxes, confs, names, _masks in results:
                for j, box in enumerate(bboxes):
                    x1, y1, x2, y2 = [int(v) for v in box[:4]]
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
                    cls = names[j] if j < len(names) else f"T{i + 1}"
                    c = confs[j] if j < len(confs) else None
                    label = f"{i + 1}:{cls}" + (f" {c:.2f}" if c is not None else "")
                    tb = draw.textbbox((0, 0), label, font=font)
                    tw, th = tb[2] - tb[0], tb[3] - tb[1]
                    ly = y1 - th - 7 if (y1 - th - 7) >= 0 else y1 + 2
                    draw.rectangle([x1, ly, x1 + tw + 8, ly + th + 7], fill=color)
                    draw.text((x1 + 4, ly + 3), label, fill=(255, 255, 255), font=font)

            return (
                canvas,
                f"✅ {total} detection(s) across {len(targets)} tab(s) — "
                + " | ".join(summary),
            )

        # Two thin wrappers over the one _run body, one per dedicated button.
        # They bake the branch constants so _run's *flat layout is untouched:
        # _detect -> detection only, _apply -> full detect+inpaint.
        def _detect(image, combine, *flat):
            return _run(image, combine, False, False, *flat)

        # Cap how many result thumbnails we hand back to the gallery: a huge
        # folder would otherwise base64 hundreds of full-res images into the
        # browser and freeze the tab. Every result is still written to disk.
        _batch_gallery_cap = 30

        def _run_folder(folder, same_folder, *flat):
            """Batch: run the full detect+inpaint pass on EVERY image in
            `folder`. Each result is saved to the ADetailer-Inpaint outputs folder,
            or — when ``same_folder`` is on — beside its source file as
            ``<name>-ad`` (a new file; originals are kept, existing ``-ad`` inputs
            skipped). Reuses the single-image _run per file and is fully guarded,
            so one unreadable file never aborts the batch. Returns (gallery,
            status)."""
            import re as _re
            from pathlib import Path as _Path

            # Recognise this tool's own outputs: "<name>-ad" or "<name>-ad-<n>".
            _AD_OUT_RE = _re.compile(r"-ad(?:-\d+)?$", _re.IGNORECASE)

            try:
                base = _Path(folder)
                is_dir = base.is_dir()
            except Exception:  # noqa: BLE001
                return None, f"⚠️ Invalid folder path: {folder}"
            if not is_dir:
                return None, f"⚠️ Folder not found: {folder}"

            exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
            try:
                files = sorted(
                    f
                    for f in base.iterdir()
                    if f.is_file()
                    and f.suffix.lower() in exts
                    # In same-folder mode our own outputs land here as name-ad /
                    # name-ad-N; skip them so a re-run doesn't reprocess them.
                    and not (same_folder and _AD_OUT_RE.search(f.stem))
                )
            except Exception as e:  # noqa: BLE001
                return None, f"⚠️ Couldn't read the folder: {e}"
            if not files:
                return None, "ℹ️ No images found in that folder."

            from PIL import Image as _PILImage

            def _gc():
                # Reclaim VRAM between images so a long batch doesn't fragment
                # the card (helps on tight GPUs). Guarded: a no-op on any WebUI
                # that doesn't expose modules.devices.torch_gc.
                try:
                    from modules import devices

                    devices.torch_gc()
                except Exception:  # noqa: BLE001
                    pass

            gallery = []
            saved = 0
            unchanged = 0
            not_saved = 0
            failed = 0
            interrupted = False
            for f in files:
                # Honour the WebUI Interrupt/Skip button between files. Each image's
                # run_detailer_on_image clears these flags at its start, so a click
                # during one image survives to this top-of-loop check for the next.
                try:
                    from modules import shared as _sh

                    if _sh.state.interrupted or _sh.state.skipped:
                        interrupted = True
                        break
                except Exception:  # noqa: BLE001
                    pass
                try:
                    with _PILImage.open(f) as _im:
                        # Opened straight from disk (not via Gradio), so honour
                        # the EXIF orientation ourselves or a rotated photo won't
                        # be detected.
                        im = _apply_exif_orientation(_im).convert("RGB")
                    if same_folder:
                        # Detail WITHOUT the built-in ADetailer-Inpaint save
                        # (save=False), then write the result beside the source as
                        # <name>-ad.<ext> — originals are kept.
                        img, st = _run(im, False, True, False, *flat)
                        if img is None:
                            failed += 1
                            continue
                        st = st if isinstance(st, str) else ""
                        if st.startswith("ℹ️"):  # nothing detected -> not written
                            unchanged += 1
                        else:
                            # NEVER overwrite an existing file — an original in
                            # this folder may already be named "<x>-ad", and prior
                            # runs leave their own outputs. Find the first free
                            # "<name>-ad[-N]" slot so no existing file is clobbered.
                            dst = f.with_name(f"{f.stem}-ad{f.suffix}")
                            _n = 1
                            while dst.exists() and _n <= 9999:
                                dst = f.with_name(f"{f.stem}-ad-{_n}{f.suffix}")
                                _n += 1
                            if dst.exists():
                                not_saved += 1  # gave up finding a free name
                            else:
                                try:
                                    if f.suffix.lower() in {".jpg", ".jpeg"}:
                                        img.save(dst, quality=95, subsampling=0)
                                    else:
                                        img.save(dst)
                                    saved += 1
                                except Exception:  # noqa: BLE001 — keep the result
                                    not_saved += 1
                        if len(gallery) < _batch_gallery_cap:
                            gallery.append(img)
                        continue
                    # Default: save=True -> run_detailer_on_image writes to the
                    # ADetailer-Inpaint folder; its inpaint branch is itself
                    # guarded end-to-end and never raises.
                    img, st = _run(im, False, True, True, *flat)
                    if img is None:
                        failed += 1
                        continue
                    st = st if isinstance(st, str) else ""
                    # run_detailer_on_image reports "ℹ️ …unchanged" when nothing
                    # was detected (not written); "✅ … (couldn't save …)" when it
                    # detailed the image but the disk write failed; a plain "✅ …"
                    # when it was detailed AND written. Categorise by that so the
                    # summary is honest about what actually hit disk.
                    if st.startswith("ℹ️"):
                        unchanged += 1
                    elif "couldn't save" in st:
                        not_saved += 1
                    else:
                        saved += 1
                    if len(gallery) < _batch_gallery_cap:
                        gallery.append(img)
                except Exception:  # noqa: BLE001 — skip a bad file, keep going
                    failed += 1
                finally:
                    _gc()

            # "ADetailer-Inpaint" mirrors AD_APPLY_SUBDIR in scripts/!adetailer.py
            # (that file isn't importable here — its name starts with "!").
            have = saved + unchanged + not_saved
            dest_txt = (
                "beside each source file as name-ad"
                if same_folder
                else "saved to the 'ADetailer-Inpaint' folder"
            )
            head = "⏹️ Batch interrupted" if interrupted else "✅ Batch done"
            scope = f"{have}/{len(files)}" if interrupted else f"{len(files)}"
            status = (
                f"{head} — {scope} image(s): {saved} detailed and "
                f"{dest_txt}"
            )
            if unchanged:
                status += f", {unchanged} left unchanged (nothing detected)"
            if not_saved:
                status += f", {not_saved} detailed but not saved (see console)"
            if failed:
                status += f", {failed} skipped (unreadable)"
            if have > len(gallery):
                status += f" — showing the first {len(gallery)} in the gallery"
            status += "."
            return (gallery or None), status

        def _apply(image, folder, same_folder, save, *flat):
            # ad_apply_output is a gr.Gallery, so results are ALWAYS a list
            # (None -> clears the gallery on error / nothing detected). A folder
            # path takes priority over the single dropped image.
            folder = (folder or "").strip().strip('"').strip("'")
            if folder:
                return _run_folder(folder, same_folder, *flat)
            img, status = _run(image, False, True, save, *flat)
            return ([img] if img is not None else None), status

        return _detect, _apply

    for i, w in enumerate(all_widgets):
        # Shared tail read by BOTH buttons: every tab's 5 detector fields (so
        # "Combine all tabs" can reach them) + THIS tab's full ADetailerArgs in
        # ALL_ARGS order (so the apply branch can rebuild the args). Read as
        # inputs only — adds NO .change listener (index-safe).
        tail = []
        for t in all_widgets:
            tail += [
                t.ad_model,
                t.ad_model_classes,
                t.ad_model_classes_excluded,
                t.ad_model_classes_exclude,
                t.ad_confidence,
            ]
        tail += w.tolist()

        detect_fn, apply_fn = _make_handler(i)
        # One dedicated button per sub-accordion. Two per-tab .click handlers are
        # index-safe: the host gallery's send-to buttons don't depend on
        # ADetailer's listener count (which already scales 1-15x with the "max
        # models" slider). The checkboxes stay listener-free.
        w.ad_preview_btn.click(
            fn=detect_fn,
            inputs=[w.ad_preview_input, w.ad_preview_all_tabs, *tail],
            outputs=[w.ad_preview_output, w.ad_preview_status],
            queue=True,
        )
        w.ad_apply_btn.click(
            fn=apply_fn,
            inputs=[
                w.ad_apply_input,
                w.ad_apply_folder,
                w.ad_apply_same_folder,
                w.ad_apply_save,
                *tail,
            ],
            outputs=[w.ad_apply_output, w.ad_apply_status],
            queue=True,
        )


# Copy/paste includes EVERY ADetailer arg by default — user explicitly
# requested this. Keeping the exclusion mechanism in place (empty for now)
# so it's a one-line revert if we ever want to carve out exceptions.
_COPY_EXCLUDE_ATTRS: frozenset[str] = frozenset()

# Sentinel "no preset selected" entry. Lives as the first choice in every
# preset dropdown so the user can switch back to a clean state without
# losing the saved presets.
PRESET_NONE = "(none)"


def _copyable_attrs() -> list[str]:
    return [a for a in ALL_ARGS.attrs if a not in _COPY_EXCLUDE_ATTRS]


def _wire_copy_paste(
    all_widgets: list[Widgets],
    all_copy_btns: list[gr.Button],
    all_paste_btns: list[gr.Button],
    clipboard_state: gr.State,
    num_models: int,
) -> None:
    attrs = _copyable_attrs()

    # Wire each Copy button: capture this tab's values into clipboard_state and
    # update all paste-button labels in one event.
    for src_idx in range(num_models):
        src_widget_refs = [getattr(all_widgets[src_idx], a) for a in attrs]

        def _make_copy_fn(idx: int):
            def _copy_fn(*values):
                new_clip = (idx, list(values))
                # Short post-Copy label (was "📥 Paste settings from Nth tab
                # here" — 36 chars, overflowed the 160px min_width and the
                # 📥 emoji rendered weirdly in Chromium). Shorter form
                # fits cleanly + keeps the emoji legible.
                label = f"\U0001F4E5 Paste from {ordinal(idx + 1)} tab"
                paste_updates = []
                for j in range(num_models):
                    if j == idx:
                        paste_updates.append(
                            gr.update(
                                value="\U0001F4E5 Paste settings",
                                interactive=False,
                            )
                        )
                    else:
                        paste_updates.append(
                            gr.update(value=label, interactive=True)
                        )
                return (new_clip, *paste_updates)

            return _copy_fn

        all_copy_btns[src_idx].click(
            fn=_make_copy_fn(src_idx),
            inputs=src_widget_refs,
            outputs=[clipboard_state, *all_paste_btns],
            queue=False,
        )

    # Wire each Paste button: read clipboard_state and apply to this tab's
    # widgets. No-op if clipboard is empty or paste is on the source tab.
    # The UI-only ad_model_classes_dropdown isn't part of ALL_ARGS, so we
    # explicitly compute its new value from the pasted ad_model_classes CSV
    # and append it to the outputs.
    try:
        _classes_attr_idx = attrs.index("ad_model_classes")
    except ValueError:
        _classes_attr_idx = -1

    for dst_idx in range(num_models):
        dst_widget_refs = [getattr(all_widgets[dst_idx], a) for a in attrs]
        dst_dropdown = all_widgets[dst_idx].ad_model_classes_dropdown

        def _make_paste_fn(idx: int, n_attrs: int):
            def _paste_fn(clipboard):
                source_idx, values = clipboard
                if (
                    source_idx < 0
                    or source_idx == idx
                    or len(values) != n_attrs
                ):
                    return [gr.update() for _ in range(n_attrs)] + [gr.update()]
                # Parse the pasted CSV into a multi-select value list for
                # the class dropdown. on_ad_model_update will filter any
                # entries that aren't valid for the destination's detector.
                csv = values[_classes_attr_idx] if _classes_attr_idx >= 0 else ""
                selected = [c.strip() for c in (csv or "").split(",") if c.strip()]
                return list(values) + [gr.update(value=selected)]

            return _paste_fn

        all_paste_btns[dst_idx].click(
            fn=_make_paste_fn(dst_idx, len(attrs)),
            inputs=clipboard_state,
            outputs=[*dst_widget_refs, dst_dropdown],
            queue=False,
        )


def _wire_presets(
    all_widgets: list[Widgets],
    all_presets: list[tuple],
    all_paste_btns: list[gr.Button],
    clipboard_state: gr.State,
    num_models: int,
) -> None:
    """Wire each tab's preset Load/Save/Delete/Rename/Reset buttons.

    Layout reminder — each entry of `all_presets` is the 9-tuple returned
    from one_ui_group: (dropdown, load_btn, rename_btn, delete_btn,
    name_box, save_btn, reset_btn, status_md, reset_all_cb).

    Saving / deleting / renaming from any tab refreshes ALL tabs'
    dropdown choices. Load applies a saved preset to THIS tab's widgets
    (including the UI-only classes dropdown). Reset rolls widgets back to
    their pydantic defaults AND clears the preset+clipboard state for a
    fresh start — for this tab alone, or for every tab when that tab's
    "Reset every tab" checkbox is ticked.
    """
    from adetailer.args import ADetailerArgs

    attrs = list(ALL_ARGS.attrs)
    all_dropdowns = [p[0] for p in all_presets]
    # Cross-tab refs for the "Reset every tab" scope. Reset is the only handler
    # that can reach outside its own tab, so these are built once here and used
    # as its static output list; per-tab wiring below still uses the narrow refs.
    all_name_boxes = [p[4] for p in all_presets]
    all_status_mds = [p[7] for p in all_presets]
    all_classes_dds = [_w.ad_model_classes_dropdown for _w in all_widgets]
    all_widget_refs = [
        getattr(all_widgets[i], a) for i in range(num_models) for a in attrs
    ]

    # Pydantic field defaults — used by the Reset handler to roll widgets
    # back to a pristine state. Falls back to None for any attr that
    # somehow isn't on the schema (defensive — shouldn't happen).
    _defaults = {
        a: ADetailerArgs.__fields__[a].default
        for a in attrs
        if a in ADetailerArgs.__fields__
    }

    def _refresh_dropdowns_update(selected: str | None = None) -> list:
        # PRESET_NONE is always the first entry so the user has a no-op
        # option to switch the dropdown back to "nothing selected".
        names = [PRESET_NONE] + get_preset_names()
        return [
            gr.update(
                choices=names,
                value=(selected if selected in names else PRESET_NONE),
            )
            for _ in range(num_models)
        ]

    def _is_none(selected: str | None) -> bool:
        return not selected or selected == PRESET_NONE

    for idx in range(num_models):
        (
            dropdown,
            load_btn,
            rename_btn,
            delete_btn,
            name_box,
            save_btn,
            reset_btn,
            status_md,
            reset_all_cb,
        ) = all_presets[idx]
        widget_refs = [getattr(all_widgets[idx], a) for a in attrs]

        # LOAD: pull the selected preset from disk, apply its values to this
        # tab's widgets — including the UI-only classes dropdown (parsed
        # from the preset's ad_model_classes CSV). No-op if (none) or the
        # preset name is missing.
        dst_classes_dd = all_widgets[idx].ad_model_classes_dropdown

        def _make_load(idx: int, n_attrs: int):
            def _load(selected: str | None):
                if _is_none(selected):
                    return ["", *(gr.update() for _ in range(n_attrs)), gr.update()]
                preset = get_preset(selected)
                if not preset:
                    return [
                        f"⚠️ Preset '{selected}' not found.",
                        *(gr.update() for _ in range(n_attrs)),
                        gr.update(),
                    ]
                widget_updates = [
                    gr.update(value=preset[a]) if a in preset else gr.update()
                    for a in attrs
                ]
                # Parse the saved CSV into the multi-select dropdown.
                csv = preset.get("ad_model_classes", "") or ""
                selected_classes = [c.strip() for c in csv.split(",") if c.strip()]
                return [
                    f"✅ Loaded '{selected}'.",
                    *widget_updates,
                    gr.update(value=selected_classes),
                ]

            return _load

        load_btn.click(
            fn=_make_load(idx, len(attrs)),
            inputs=dropdown,
            outputs=[status_md, *widget_refs, dst_classes_dd],
            queue=False,
        )

        # SAVE: capture current widget values and write a new preset. Refresh
        # every tab's dropdown so the preset becomes selectable everywhere.
        def _make_save(idx: int):
            def _save(name: str, *values):
                name = (name or "").strip()
                if not name:
                    return ["⚠️ Enter a preset name first.", *_refresh_dropdowns_update()]
                state_dict = {a: v for a, v in zip(attrs, values)}
                ok = save_preset(name, state_dict)
                if not ok:
                    return [
                        f"⚠️ Invalid preset name '{name}'.",
                        *_refresh_dropdowns_update(),
                    ]
                return [
                    f"✅ Saved preset '{name}'.",
                    *_refresh_dropdowns_update(selected=name),
                ]

            return _save

        save_btn.click(
            fn=_make_save(idx),
            inputs=[name_box, *widget_refs],
            outputs=[status_md, *all_dropdowns],
            queue=False,
        )

        # DELETE: remove the selected preset; refresh dropdowns.
        def _make_delete():
            def _delete(selected: str | None):
                if _is_none(selected):
                    return ["⚠️ Pick a preset first.", *_refresh_dropdowns_update()]
                selected = selected.strip()
                ok = delete_preset(selected)
                if not ok:
                    return [
                        f"⚠️ Preset '{selected}' not found.",
                        *_refresh_dropdowns_update(),
                    ]
                return [
                    f"\U0001F5D1 Deleted '{selected}'.",
                    *_refresh_dropdowns_update(),
                ]

            return _delete

        delete_btn.click(
            fn=_make_delete(),
            inputs=dropdown,
            outputs=[status_md, *all_dropdowns],
            queue=False,
        )

        # RENAME: take the currently-selected preset and the value of
        # the 'Preset name to save' textbox; rename the preset on disk
        # to the new name and refresh every tab's dropdown so it points
        # at the renamed entry.
        def _make_rename():
            def _rename(selected: str | None, new_name: str):
                if _is_none(selected):
                    return [
                        "⚠️ Pick a preset first.",
                        *_refresh_dropdowns_update(),
                    ]
                new_name = (new_name or "").strip()
                if not new_name:
                    return [
                        "⚠️ Enter the new name in the 'Preset name to save' box first.",
                        *_refresh_dropdowns_update(selected=selected),
                    ]
                ok, msg = rename_preset(selected, new_name)
                if not ok:
                    return [
                        f"⚠️ Rename failed: {msg}.",
                        *_refresh_dropdowns_update(selected=selected),
                    ]
                return [
                    f"✏️ Renamed '{selected}' → '{new_name}'.",
                    *_refresh_dropdowns_update(selected=new_name),
                ]

            return _rename

        rename_btn.click(
            fn=_make_rename(),
            inputs=[dropdown, name_box],
            outputs=[status_md, *all_dropdowns],
            queue=False,
        )

        # RESET: roll a tab back to a pristine state. Per target tab:
        # - All ALL_ARGS widgets (detector, classes, prompts, denoise,
        #   padding, sampler, ControlNet, ...) -> pydantic defaults
        # - UI-only classes multi-select dropdown -> empty
        # - Preset library: dropdown back to (none), name box emptied, status set
        # The global clipboard is wiped either way: state -> (-1, []), and every
        # paste button on every tab returns to "📥 Paste settings" disabled.
        #
        # Scope comes from this tab's "Reset every tab" checkbox: unticked resets
        # only this tab (unchanged behaviour), ticked resets all of them. Gradio
        # output lists are static, so the outputs always span every tab and
        # non-target tabs simply receive gr.update() no-ops. Still ONE .click per
        # tab, exactly as before → index-safe.
        def _make_reset(idx: int):
            def _reset(reset_all: bool):
                targets = set(range(num_models)) if reset_all else {idx}
                note = (
                    "\U0001F195 All tabs reset to defaults."
                    if reset_all
                    else "\U0001F195 Tab reset to defaults."
                )
                status_updates = [
                    note if i in targets else gr.update() for i in range(num_models)
                ]
                dd_updates = [
                    gr.update(value=PRESET_NONE) if i in targets else gr.update()
                    for i in range(num_models)
                ]
                name_updates = [
                    "" if i in targets else gr.update() for i in range(num_models)
                ]
                paste_updates = [
                    gr.update(value="\U0001F4E5 Paste settings", interactive=False)
                    for _ in range(num_models)
                ]
                # Flattened tab-major to match all_widget_refs exactly.
                widget_updates = [
                    gr.update(value=_defaults.get(a))
                    if (i in targets and a in _defaults)
                    else gr.update()
                    for i in range(num_models)
                    for a in attrs
                ]
                classes_updates = [
                    gr.update(value=[]) if i in targets else gr.update()
                    for i in range(num_models)
                ]
                return [
                    *status_updates,
                    *dd_updates,
                    *name_updates,
                    (-1, []),  # clipboard_state (global)
                    *paste_updates,
                    *widget_updates,
                    *classes_updates,
                ]

            return _reset

        reset_btn.click(
            fn=_make_reset(idx),
            inputs=[reset_all_cb],
            outputs=[
                *all_status_mds,
                *all_dropdowns,
                *all_name_boxes,
                clipboard_state,
                *all_paste_btns,
                *all_widget_refs,
                *all_classes_dds,
            ],
            queue=False,
        )


def one_ui_group(
    n: int,
    is_img2img: bool,
    webui_info: WebuiInfo,
    saved_tab_state: "dict[str, Any] | None" = None,
):
    w = Widgets()
    eid = partial(elem_id, n=n, is_img2img=is_img2img)

    saved = saved_tab_state or {}
    sv = partial(_sv, saved)

    model_choices = (
        [*webui_info.ad_model_list, "None"]
        if n == 0
        else ["None", *webui_info.ad_model_list]
    )

    # Top of the tab content — all directly visible, no accordion wrapping.
    # Layout reorganised 2026-05-16:
    #   - Enable this tab           (sits alone at the top)
    #   - Preset library export / import (accordion, collapsed default) —
    #     was previously at the BOTTOM of the preset block; moved here so
    #     it sits above the daily-use preset controls.
    #   - Preset library            (dropdown + load/rename/delete row,
    #                                then name-textbox + save + reset row)
    #   - Copy / Paste settings     (moved DOWN from the top — now sits
    #                                directly under the preset name-to-save
    #                                row so all "tab-state operations" are
    #                                grouped together)
    #   - Preset status + preview   (just below Copy/Paste)
    with gr.Row(variant="compact"):
        w.ad_tab_enable = gr.Checkbox(
            label=f"Enable this tab ({ordinal(n + 1)})",
            # Default: only the 1st tab is enabled; extra tabs start OFF (they
            # have no detector yet). Saved/last-used state still wins via sv().
            value=sv("ad_tab_enable", n == 0),
            visible=True,
            elem_id=eid("ad_tab_enable"),
        )

    # Export / Import preset library — power-user controls, collapsed by
    # default. Uses the compact gr.DownloadButton + gr.UploadButton pair
    # instead of two big gr.File drop-zones so the accordion stays tiny
    # vertically when expanded.
    with gr.Accordion(
        "Preset library export / import",
        open=False,
        elem_id=eid("ad_preset_io_accordion"),
    ):
        with gr.Row(variant="compact"):
            # gr.DownloadButton needs Gradio 4 (Forge / Forge Neo). AUTOMATIC1111
            # ships Gradio 3, which lacks it — a hard reference crashed the whole
            # ADetailer tab at build there (#2). Fall back to a plain Button so the
            # tab still loads; preset export just isn't one-click-downloadable on
            # Gradio 3 (everything else, incl. Import, works). The .click wiring
            # below is harmless on a Button.
            _DownloadButton = getattr(gr, "DownloadButton", None)
            if _DownloadButton is not None:
                preset_export_btn = _DownloadButton(
                    # Short label per user request 2026-05-18 — the previous
                    # "Export to JSON" wrapped on two lines until the CSS
                    # nowrap rule was extended; the user then asked for a
                    # tighter label outright. Emoji kept to mirror the
                    # Import button's 📥 marker visually.
                    label="\U0001F4E4 Esport",
                    elem_id=eid("ad_preset_export_btn"),
                    scale=0,
                    min_width=160,
                )
            else:
                preset_export_btn = gr.Button(
                    value="\U0001F4E4 Esport",
                    elem_id=eid("ad_preset_export_btn"),
                    scale=0,
                    min_width=160,
                )
            preset_import_btn = gr.UploadButton(
                label="\U0001F4E5 Import",
                file_types=[".json"],
                elem_id=eid("ad_preset_import_btn"),
                scale=0,
                min_width=110,
            )
            preset_import_overwrite = gr.Checkbox(
                label="Overwrite on conflict",
                value=False,
                scale=1,
                elem_id=eid("ad_preset_import_overwrite"),
            )
        preset_io_status = gr.Markdown(
            value="",
            elem_id=eid("ad_preset_io_status"),
            elem_classes=["ad-preset-status"],
        )

    # Preset library — load/save/delete/rename named tab configurations.
    # Shared storage across tabs; cross-tab dropdown refresh on save/delete
    # is wired in _wire_presets().
    initial_presets = [PRESET_NONE] + get_preset_names()
    gr.Markdown("Preset library", elem_classes=["ad-section-label"])
    with gr.Row(variant="compact"):
        preset_dropdown = gr.Dropdown(
            choices=initial_presets,
            value=PRESET_NONE,
            label="Saved presets" + suffix(n),
            show_label=False,
            interactive=True,
            scale=4,
            elem_id=eid("ad_preset_dropdown"),
        )
        preset_load_btn = gr.Button(
            value="\U0001F4C2 Load",
            elem_id=eid("ad_preset_load"),
            scale=0,
            min_width=90,
        )
        preset_rename_btn = gr.Button(
            value="✏️ Rename",
            elem_id=eid("ad_preset_rename"),
            scale=0,
            min_width=110,
        )
        preset_delete_btn = gr.Button(
            value="\U0001F5D1 Delete",
            elem_id=eid("ad_preset_delete"),
            scale=0,
            min_width=100,
        )
    with gr.Row(variant="compact", elem_classes=["ad-preset-save-row"]):
        preset_name_box = gr.Textbox(
            value="",
            placeholder="Preset name to save (letters, digits, basic punctuation)",
            show_label=False,
            scale=4,
            elem_id=eid("ad_preset_name"),
        )
        preset_save_btn = gr.Button(
            value="\U0001F4BE Save preset",
            elem_id=eid("ad_preset_save"),
            scale=0,
            min_width=130,
        )
        preset_reset_btn = gr.Button(
            value="\U0001F195 Reset",
            elem_id=eid("ad_preset_reset"),
            scale=0,
            min_width=90,
        )
        # Scope switch for the Reset button next to it: off = this tab only
        # (the long-standing behaviour), on = every ADetailer tab at once
        # (requested 2026-07-17 — a single control to reset the whole panel).
        # A checkbox rather than a second button on purpose: a button would be
        # a new Gradio event listener, and those shift dependency indices and
        # break Forge's gallery JS (learned the hard way 2026-06-04). A plain
        # component with no listener of its own is index-safe; _wire_presets
        # just reads it as an extra input to the existing .click.
        preset_reset_all = gr.Checkbox(
            label="Reset every tab",
            value=False,
            scale=0,
            min_width=140,
            elem_id=eid("ad_preset_reset_all"),
        )
        # Transient scope switch, not a setting: keep the host WebUI's
        # ui-config.json from freezing it on across restarts (it tracks
        # labelled components by label). Plain attribute set → index-safe.
        try:
            preset_reset_all.do_not_save_to_config = True
        except Exception:  # noqa: BLE001 — never break UI build over a flag
            pass

    # Copy / Paste settings — inter-tab clipboard. Moved here from the top
    # of the tab so it sits directly under the preset name-to-save row,
    # grouping all tab-state copying operations (Save preset, Reset, Copy,
    # Paste) in one visual block. The `.ad-tab-clipboard-row` class adds
    # a margin-top so the buttons don't touch the textbox above them
    # (user reported the rows were rendering with zero spacing in Forge
    # Neo).
    with gr.Row(variant="compact", elem_classes=["ad-tab-clipboard-row"]):
        copy_btn = gr.Button(
            value="\U0001F4CB Copy settings",
            elem_id=eid("ad_copy_settings"),
            scale=0,
            min_width=160,
        )
        paste_btn = gr.Button(
            value="\U0001F4E5 Paste settings",
            elem_id=eid("ad_paste_settings"),
            interactive=False,
            scale=0,
            min_width=160,
        )

    preset_status = gr.Markdown(
        value="",
        elem_id=eid("ad_preset_status"),
        elem_classes=["ad-preset-status"],
    )

    # Live preview of what the currently-highlighted preset contains — updates
    # on dropdown change BEFORE the user clicks Load. Shows the prompts (with
    # [SEP]/[PROMPT] tokens flagged so the user knows whether they will be
    # expanded against the main prompt), the detector, and a class summary.
    # Hidden by default so it doesn't render an empty styled box between
    # the preset row and the detector section when no preset is selected
    # — visibility is flipped on by `_format_preset_preview` when there's
    # content to show.
    preset_preview = gr.Markdown(
        value="",
        visible=False,
        elem_id=eid("ad_preset_preview"),
        elem_classes=["ad-preset-preview"],
    )
    preset_dropdown.change(
        fn=_format_preset_preview,
        inputs=preset_dropdown,
        outputs=preset_preview,
        queue=False,
    )

    # Wire Export (DownloadButton) and Import (UploadButton) locally.
    # DownloadButton: click handler returns a file path; Gradio triggers the
    # download automatically and updates the button's `value` so subsequent
    # clicks re-serve the file. UploadButton: upload event yields the path
    # of the uploaded file via the button's `inputs` value.
    def _do_export() -> str:
        """Click handler for the export DownloadButton. Writes the current
        preset library to a temp file and returns its path so Gradio can
        serve the download. Also updates the status line side-effect-free
        via a separate output target."""
        import tempfile
        from pathlib import Path

        payload = export_presets_json()
        tmp_dir = Path(tempfile.gettempdir())
        out = tmp_dir / "adetailer-ultimate-presets.json"
        out.write_text(payload, encoding="utf-8")
        return str(out)

    def _do_export_status() -> str:
        return f"✅ Exported **{len(get_preset_names())}** preset(s)."

    def _do_import(uploaded_path: str | None, overwrite: bool) -> tuple[Any, str]:
        """Upload handler. `uploaded_path` is the local filesystem path of
        the file the user dropped on the UploadButton."""
        if not uploaded_path:
            return gr.update(), "_no file received._"
        try:
            with open(uploaded_path, "r", encoding="utf-8") as f:
                payload = f.read()
        except OSError as e:
            return gr.update(), f"_could not read file: {e}_"
        added, replaced, skipped = import_presets_json(
            payload, overwrite=overwrite
        )
        parts: list[str] = []
        if added:
            parts.append(f"➕ **{added}** added")
        if replaced:
            parts.append(f"\U0001F501 **{replaced}** replaced")
        if skipped:
            parts.append(
                f"⏭ **{len(skipped)}** skipped (already present; tick 'Overwrite' to replace)"
            )
        if not parts:
            msg = "_no presets imported (file empty, invalid, or all names skipped)._"
        else:
            msg = " · ".join(parts)
        names = [PRESET_NONE] + get_preset_names()
        return gr.update(choices=names), msg

    # DownloadButton wiring: the click both refreshes the button's `value`
    # (triggering the download) AND updates the status line.
    preset_export_btn.click(
        fn=_do_export,
        inputs=None,
        outputs=preset_export_btn,
        queue=False,
    ).then(
        fn=_do_export_status,
        inputs=None,
        outputs=preset_io_status,
        queue=False,
    )
    # UploadButton fires `.upload` when the user picks/drops a file; the
    # button's value (the upload path) is included in `inputs`.
    preset_import_btn.upload(
        fn=_do_import,
        inputs=[preset_import_btn, preset_import_overwrite],
        outputs=[preset_dropdown, preset_io_status],
        queue=False,
    )

    # Saved model name may refer to a model the user deleted between sessions.
    # Fall back to the default first choice if it's not in current choices.
    _saved_model = sv("ad_model", model_choices[0])
    if _saved_model not in model_choices:
        _saved_model = model_choices[0]

    with gr.Group():
        with gr.Row():
            w.ad_model = gr.Dropdown(
                label="ADetailer detector" + suffix(n),
                choices=model_choices,
                value=_saved_model,
                visible=True,
                type="value",
                elem_id=eid("ad_model"),
                info="Select a model to use for detection.",
            )

        with gr.Row():
            w.ad_model_classes = gr.Textbox(
                label="ADetailer detector CLASSES (YOLO-World)" + suffix(n),
                value=sv("ad_model_classes", ""),
                visible=False,
                elem_id=eid("ad_model_classes"),
            )
            # Restore the saved class filter into the VISIBLE dropdown on load.
            # The dropdown is UI-only (not in ALL_ARGS): its backing values live
            # in the hidden ad_model_classes / ad_model_classes_excluded
            # textboxes, which ARE restored from user_state.json. But Gradio
            # `.change` events don't fire on initial render, so on_ad_model_update
            # never runs at startup — leaving this dropdown empty even though the
            # filter was saved. Worse, the first interaction could then sync the
            # empty dropdown back over the restored textbox, silently wiping the
            # user's classes. Fix: pre-seed `choices`/`value` here from the saved
            # model (+ any saved selection) so the class list is visible AND the
            # filter is preserved across restarts.
            # Seed the dropdown's FULL class list at build via
            # get_model_class_names so EVERY class is selectable immediately at
            # startup — even when NO class filter was saved. Previously this ran
            # only for tabs that had restored a filter, so a tab whose detector
            # was already a multiclass model (e.g. it was the last-used detector)
            # but with no classes picked showed an EMPTY dropdown: you had to
            # switch detector and switch back to make the classes appear
            # (reported 2026-07-17). Now it seeds whenever the saved detector is a
            # class-based model (any multiclass YOLO or mediapipe_face_features),
            # so the classes show straight away with the saved selection (if any)
            # pre-applied. get_model_class_names reads the `.names.json`/`.json`
            # sidecar first (fast, no .pt load) and is lru_cached; the (rarer) .pt
            # fallback only happens for a sidecar-less model, and this runs once
            # per tab for its own saved detector — so startup cost stays bounded.
            # Index-safe: only an existing widget's initial values change, no
            # event listener is added.
            _saved_exclude = bool(sv("ad_model_classes_exclude", False))
            _saved_classes_csv = (
                sv("ad_model_classes_excluded", "")
                if _saved_exclude
                else sv("ad_model_classes", "")
            )
            _wanted_classes = [
                c.strip() for c in (_saved_classes_csv or "").split(",") if c.strip()
            ]
            if (
                _saved_model
                and _saved_model != "None"
                and (
                    not _saved_model.lower().startswith("mediapipe")
                    or _saved_model == MEDIAPIPE_FACE_FEATURES_MODEL
                )
                and "-world" not in _saved_model
            ):
                try:
                    if _saved_model == MEDIAPIPE_FACE_FEATURES_MODEL:
                        _full_classes = get_model_class_names(_saved_model)
                    else:
                        _mpath = webui_info.model_mapping.get(_saved_model, "")
                        _full_classes = (
                            get_model_class_names(_mpath) if _mpath else []
                        )
                except Exception:  # noqa: BLE001
                    _full_classes = []
                # Full list first; append any saved token not in it so the value
                # stays a subset of the choices (Gradio requirement). Falls back
                # to the saved tokens only if the full list couldn't be read.
                _dd_choices: list[str] = list(
                    dict.fromkeys([*(_full_classes or []), *_wanted_classes])
                )
                _dd_value: list[str] = list(dict.fromkeys(_wanted_classes))
            else:
                _dd_choices = []
                _dd_value = []

            # Diagnostic log (console) — surfaces, at every UI build / WebUI
            # restart, exactly which detector + classes each tab restored from
            # user_state.json. Confirms persistence is working and, crucially,
            # flags when a SAVED detector is no longer present in the current
            # model list (so it fell back to the default) — the usual reason a
            # tab "resets" on restart. Plain print → index-safe.
            _saved_model_raw = saved.get("ad_model")
            if _saved_model_raw and _saved_model_raw != _saved_model:
                print(
                    f"[-] ADetailer: tab {n + 1} — saved detector "
                    f"{_saved_model_raw!r} is NOT in the current model list; "
                    f"fell back to {_saved_model!r}. (saved classes: "
                    f"{_wanted_classes!r})"
                )
            elif _saved_model_raw and _saved_model_raw != "None":
                print(
                    f"[-] ADetailer: tab {n + 1} restored — detector="
                    f"{_saved_model!r}, classes={_dd_value!r}"
                )

            # UI-only dropdown: not in ALL_ARGS. It syncs into ad_model_classes
            # (CSV) for the include path or ad_model_classes_excluded for exclude.
            # Info text added 2026-05-18 per user UX feedback: the dropdown is
            # NOT auto-populated when a multiclass model is chosen; the user
            # must opt in by selecting classes. Make the "empty = all" default
            # explicit so people don't think the empty dropdown means "no
            # detection will happen".
            w.ad_model_classes_dropdown = gr.Dropdown(
                label="ADetailer detector CLASSES" + suffix(n),
                info="If empty, ALL classes the model produces are inpainted. Select to narrow down to specific classes.",
                choices=_dd_choices,
                value=_dd_value,
                multiselect=True,
                visible=True,
                elem_id=eid("ad_model_classes_dropdown"),
            )

        with gr.Row(variant="compact", elem_classes=["ad-2up-row"]):
            w.ad_model_classes_exclude = gr.Checkbox(
                label="Exclude selected (NOT)" + suffix(n),
                value=sv("ad_model_classes_exclude", False),
                visible=True,
                elem_id=eid("ad_model_classes_exclude"),
            )
            w.ad_classes_sequential = gr.Checkbox(
                label="Process classes sequentially" + suffix(n),
                value=sv("ad_classes_sequential", False),
                visible=True,
                elem_id=eid("ad_classes_sequential"),
            )
            # Mirror of the dropdown when exclude=True; hidden, used as the
            # backing arg in ALL_ARGS.
            w.ad_model_classes_excluded = gr.Textbox(
                value=sv("ad_model_classes_excluded", ""),
                visible=False,
                elem_id=eid("ad_model_classes_excluded"),
            )

        _on_ad_model_update = partial(
            on_ad_model_update, model_mapping=webui_info.model_mapping
        )
        w.ad_model.change(
            _on_ad_model_update,
            inputs=[w.ad_model, w.ad_model_classes_dropdown],
            outputs=[
                w.ad_model_classes,
                w.ad_model_classes_dropdown,
                w.ad_model_classes_exclude,
                w.ad_model_classes_excluded,
            ],
            queue=False,
        )

        def _sync_dropdown(selected: list[str] | None, exclude: bool):
            csv = ",".join(selected or [])
            return ("" if exclude else csv), (csv if exclude else "")

        # NOTE: deliberately NOT queue=False here. With queue=False each
        # selection fires an independent, unordered request; selecting two
        # classes quickly (face then hand) could let the earlier "face"
        # response land last and overwrite "face,hand", silently dropping the
        # 2nd class. Going through the queue serialises the syncs so the final
        # value wins. We do NOT add a `.input` handler — adding a Gradio event
        # handler shifts dependency indices and breaks Forge's gallery JS
        # (learned the hard way 2026-06-04). Same 2 handlers as before, just
        # without queue=False → index-safe.
        #
        # Queueing costs LATENCY, though, and that latency was a real bug
        # (2026-07-17): while the queue is busy — a batch run, say — this sync
        # can sit unprocessed for minutes, so a generation started in the
        # meantime submits the stale hidden value. Empty means "inpaint every
        # class", so the user silently got classes they never selected.
        # javascript/class-sync.js now mirrors the dropdown into these hidden
        # fields client-side, instantly, which removes the race outright; these
        # two handlers stay as the backstop, computing the identical CSV.
        w.ad_model_classes_dropdown.change(
            _sync_dropdown,
            inputs=[w.ad_model_classes_dropdown, w.ad_model_classes_exclude],
            outputs=[w.ad_model_classes, w.ad_model_classes_excluded],
        )
        w.ad_model_classes_exclude.change(
            _sync_dropdown,
            inputs=[w.ad_model_classes_dropdown, w.ad_model_classes_exclude],
            outputs=[w.ad_model_classes, w.ad_model_classes_excluded],
        )

    with gr.Accordion(
        "Inpaint prompts",
        open=False,
        elem_id=eid("ad_prompts_accordion"),
    ):
        with gr.Row(elem_id=eid("ad_toprow_prompt"), elem_classes=["ad-prompt-row"]):
            w.ad_prompt = gr.Textbox(
                value=sv("ad_prompt", ""),
                label="ad_prompt" + suffix(n),
                show_label=False,
                lines=3,
                placeholder="ADetailer prompt"
                + suffix(n)
                + "\nIf blank, the main prompt is used.",
                elem_id=eid("ad_prompt"),
            )

        with gr.Row(elem_id=eid("ad_toprow_prompt_append"), elem_classes=["ad-prompt-row"]):
            w.ad_prompt_append = gr.Textbox(
                value=sv("ad_prompt_append", ""),
                label="ad_prompt_append" + suffix(n),
                show_label=False,
                lines=1,
                placeholder="Always appended to the prompt above (e.g. 'detailed eyes, sharp pupils')",
                elem_id=eid("ad_prompt_append"),
            )

        with gr.Row(elem_id=eid("ad_toprow_negative_prompt"), elem_classes=["ad-prompt-row"]):
            w.ad_negative_prompt = gr.Textbox(
                value=sv("ad_negative_prompt", ""),
                label="ad_negative_prompt" + suffix(n),
                show_label=False,
                lines=2,
                placeholder="ADetailer negative prompt"
                + suffix(n)
                + "\nIf blank, the main negative prompt is used.",
                elem_id=eid("ad_negative_prompt"),
            )

        with gr.Row(elem_id=eid("ad_toprow_negative_prompt_append"), elem_classes=["ad-prompt-row"]):
            w.ad_negative_prompt_append = gr.Textbox(
                value=sv("ad_negative_prompt_append", ""),
                label="ad_negative_prompt_append" + suffix(n),
                show_label=False,
                lines=1,
                placeholder="Always appended to the negative prompt above",
                elem_id=eid("ad_negative_prompt_append"),
            )

        # Class-specific prompts: pairs with Sequential class detection so
        # each class in the sequential queue can override the tab's prompt /
        # negative prompt with its own. Lines that don't match the syntax
        # are silently ignored. Empty entries fall back to the tab defaults.
        with gr.Row(elem_id=eid("ad_toprow_class_prompts"), elem_classes=["ad-prompt-row"]):
            w.ad_class_prompts = gr.Textbox(
                value=sv("ad_class_prompts", ""),
                label="ad_class_prompts" + suffix(n),
                show_label=False,
                # 5 visible lines so the multi-line placeholder example
                # (4 lines: format intro + 'Example:' label + 2 sample
                # entries) fits without scrolling. Bumped from 4 on
                # 2026-05-18 per user UX feedback.
                lines=5,
                placeholder=(
                    "Per-class prompt overrides for Sequential class detection.\n"
                    "Format (one per line): classname: positive_prompt | negative_prompt\n"
                    "Example:\n"
                    "  face: detailed face, sharp eyes\n"
                    "  hand: five fingers | blurry hands, extra fingers"
                ),
                elem_id=eid("ad_class_prompts"),
            )

        with gr.Row(variant="compact", elem_classes=["ad-2up-row"]):
            w.ad_class_guard = gr.Checkbox(
                label="Auto class-guard" + suffix(n),
                info=(
                    "For each detected region, adds its class name to the "
                    "positive prompt and every other class of the current "
                    "detector model to the negative prompt, so a correct "
                    "detection is not regenerated as a different class. Your "
                    "per-class prompts override it. Works with any class-based "
                    "detector including mediapipe face-features; no effect on "
                    "class-less detectors (the other mediapipe face-box models)."
                ),
                value=sv("ad_class_guard", False),
                visible=True,
                elem_id=eid("ad_class_guard"),
            )
            w.ad_class_guard_weight = gr.Slider(
                label="Class-guard emphasis" + suffix(n),
                info=(
                    "Weight applied to the class name added to the positive "
                    "prompt, e.g. 1.2 gives (face:1.20). 1.0 = no emphasis. "
                    "Only used when Auto class-guard is on."
                ),
                minimum=0.5,
                maximum=2.0,
                step=0.05,
                value=sv("ad_class_guard_weight", 1.0),
                visible=True,
                elem_id=eid("ad_class_guard_weight"),
            )

        with gr.Row(variant="compact", elem_classes=["ad-2up-row"]):
            w.ad_use_main_loras = gr.Checkbox(
                label="Use LoRAs from main prompt" + suffix(n),
                value=sv("ad_use_main_loras", False),
                visible=True,
                elem_id=eid("ad_use_main_loras"),
            )
            # Sub-toggle: only relevant when ad_use_main_loras is on, but kept
            # always visible so the user can preset it and it appears next to
            # its parent. The script-side wiring already guards on both flags.
            w.ad_use_lora_triggers = gr.Checkbox(
                label="Append LoRA triggers from name" + suffix(n),
                info="Expects the convention <lora:name (trigger):weight>",
                value=sv("ad_use_lora_triggers", False),
                visible=True,
                elem_id=eid("ad_use_lora_triggers"),
            )

        with gr.Row():
            w.ad_strip_loras = gr.Checkbox(
                label="Strip LoRAs from the detailer prompt" + suffix(n),
                info=(
                    "Removes <lora:...> tags from the prompt sent to the "
                    "detailer, so LoRAs in the main prompt don't bleed onto the "
                    "detailed region. Runs last, so it wins over Use LoRAs from "
                    "main prompt when both are on."
                ),
                value=sv("ad_strip_loras", False),
                visible=True,
                elem_id=eid("ad_strip_loras"),
            )

        # Hires-only toggle on its own row below the LoRA checkboxes — keeps
        # related top-level prompt/pipeline toggles in the same visual area
        # of the tab without needing an accordion expansion. Hidden in
        # img2img (no hires.fix concept there) — symmetric with the
        # existing `ad_skip_img2img` widget which is shown only in img2img.
        # Even when hidden the widget still exists in the components list,
        # so its value (read from persistence/preset) is honoured by the
        # runtime check — see `_should_skip_for_hires_only`.
        with gr.Row(variant="compact"):
            w.ad_apply_on_hires_only = gr.Checkbox(
                label="Apply only on hires.fix" + suffix(n),
                info="Skip the lowres pre-hires call; run ADetailer only on the upscale output. Has no effect in img2img or when hires.fix is off.",
                value=sv("ad_apply_on_hires_only", False),
                visible=not is_img2img,
                elem_id=eid("ad_apply_on_hires_only"),
            )

    with gr.Group():
        with gr.Accordion(
            "Detection", open=False, elem_id=eid("ad_detection_accordion")
        ):
            detection(w, n, is_img2img, saved)

        # Two sibling sub-accordions inside one group: a detection-only preview
        # and the full "run ADetailer on an image" tool (issue #4). Each has its
        # OWN Input / Run button / Output; settings are shared automatically
        # because both handlers read the same per-tab detector fields + tolist().
        with gr.Group():
            with gr.Accordion(
                "Detection preview",
                open=False,
                elem_id=eid("ad_preview_accordion"),
            ):
                gr.Markdown(
                    "Drop or paste an image and press the button to run the "
                    "detector with the current settings (classes, NOT, "
                    "confidence) and outline the detected regions with bounding "
                    "boxes — detection only, no inpainting. To actually run "
                    "ADetailer on an image, use \"Run ADetailer on an image\" "
                    "below.",
                    elem_classes=["ad-preview-hint"],
                )
                with gr.Row():
                    w.ad_preview_input = gr.Image(
                        label="Input",
                        type="pil",
                        interactive=True,
                        elem_id=eid("ad_preview_input"),
                    )
                    w.ad_preview_output = gr.Image(
                        label="Detections",
                        type="pil",
                        interactive=False,
                        elem_id=eid("ad_preview_output"),
                    )
                with gr.Row():
                    w.ad_preview_btn = gr.Button(
                        "🔍 Run detection preview",
                        elem_id=eid("ad_preview_btn"),
                        scale=0,
                        min_width=200,
                    )
                    # Component only — NO listener of its own (read purely as an
                    # input by the preview button's .click, wired later in
                    # _wire_detection_previews). When on, the button runs EVERY
                    # configured tab's detector and overlays all boxes at once.
                    w.ad_preview_all_tabs = gr.Checkbox(
                        label="🔁 Combine all tabs",
                        value=False,
                        scale=0,
                        min_width=220,
                        elem_classes=["ad-preview-combine"],
                        elem_id=eid("ad_preview_all_tabs"),
                    )
                    w.ad_preview_status = gr.Markdown(
                        value="",
                        elem_id=eid("ad_preview_status"),
                        elem_classes=["ad-preview-status"],
                    )

            with gr.Accordion(
                "Run ADetailer on an image",
                open=False,
                elem_id=eid("ad_apply_accordion"),
            ):
                gr.Markdown(
                    "Drop or paste an image and press the button to run the full "
                    "ADetailer detect + inpaint pass on it — using this tab's "
                    "detector, detailer checkpoint, prompt, LoRAs, text encoder "
                    "and VAE — without regenerating the base image. Handy for "
                    "trying different detailer setups on a finished picture "
                    "(requested in #4). Or paste a folder path below to batch-"
                    "process every image inside it, saving each result.",
                    elem_classes=["ad-preview-hint"],
                )
                with gr.Row():
                    w.ad_apply_input = gr.Image(
                        label="Input",
                        type="pil",
                        interactive=True,
                        elem_id=eid("ad_apply_input"),
                    )
                    # A gr.Gallery (not gr.Image) for the result: on Forge Neo's
                    # Gradio 4.40 gr.Image has NO fullscreen button, but a Gallery
                    # enlarges the image to fill the window on click (allow_preview,
                    # default True) — exactly like a normal generated image, which
                    # is what koblue asked for in #4. Every param below exists on
                    # BOTH Gradio 3.41.2 (A1111) and 4.40 (Forge/Neo); `interactive`
                    # is omitted (4.x-only, and redundant for a display gallery).
                    w.ad_apply_output = gr.Gallery(
                        label="Result",
                        columns=1,
                        rows=1,
                        object_fit="contain",
                        preview=True,
                        show_download_button=True,
                        elem_id=eid("ad_apply_output"),
                    )
                with gr.Row(elem_classes=["ad-apply-folder-row"]):
                    # Batch mode: point this at a folder of finished images and
                    # the Run button below details EVERY image in it, saving each
                    # result. A folder path here wins over the single image above.
                    # Listener-free input (no .change) and it reuses the existing
                    # Run button, so it adds no event — index-safe.
                    w.ad_apply_folder = gr.Textbox(
                        label="Or batch a whole folder (optional)",
                        placeholder="Paste a folder path to detail every image inside it",
                        info="Runs on every image in this folder and saves each result to the ADetailer-Inpaint outputs folder (or beside each source, see below). Leave empty to use the single image above; if both are set, the folder wins.",
                        lines=1,
                        max_lines=1,
                        interactive=True,
                        elem_id=eid("ad_apply_folder"),
                    )
                    # Batch-only save destination toggle: write each result NEXT TO
                    # its source file as name-ad instead of the ADetailer-Inpaint
                    # folder. Listener-free input (no .change), read by the same Run
                    # button — index-safe, adds no ALL_ARGS field.
                    w.ad_apply_same_folder = gr.Checkbox(
                        label="📁 Save results in the source folder instead",
                        value=False,
                        info="Batch only: write each result beside its source file as name-ad (a new file), instead of the ADetailer-Inpaint folder. Existing files are NEVER overwritten — if the name is taken it uses name-ad-1, name-ad-2, and so on — so your originals are always kept. Files already ending in -ad are skipped, so re-runs don't reprocess earlier results.",
                        elem_id=eid("ad_apply_same_folder"),
                    )
                with gr.Row():
                    w.ad_apply_btn = gr.Button(
                        "✨ Run ADetailer on this image",
                        elem_id=eid("ad_apply_btn"),
                        scale=0,
                        min_width=240,
                    )
                    # "Save result to outputs": listener-free input — index-safe.
                    # Writes the result to your outputs folder instead of only
                    # Gradio's temp dir (koblue's request, #4).
                    w.ad_apply_save = gr.Checkbox(
                        label="💾 Save result to outputs",
                        value=False,
                        scale=0,
                        min_width=210,
                        elem_classes=["ad-preview-combine"],
                        elem_id=eid("ad_apply_save"),
                    )
                    w.ad_apply_status = gr.Markdown(
                        value="",
                        elem_id=eid("ad_apply_status"),
                        elem_classes=["ad-preview-status"],
                    )

        with gr.Accordion(
            "Mask Preprocessing",
            open=False,
            elem_id=eid("ad_mask_preprocessing_accordion"),
        ):
            mask_preprocessing(w, n, is_img2img, saved)

        with gr.Accordion(
            "Inpainting", open=False, elem_id=eid("ad_inpainting_accordion")
        ):
            inpainting(w, n, is_img2img, webui_info, saved)

    with gr.Group():
        controlnet(w, n, is_img2img, saved)

    # Opt every persistence-managed widget OUT of the host WebUI's
    # ui-config.json. Forge/A1111's modules/ui_loadsave.py tracks each
    # labelled component BY LABEL and, on every restart, OVERRIDES its value
    # with whatever it froze in ui-config.json — which silently defeated
    # "Remember last-used settings": it forced the detector back to the first
    # model (e.g. face_yolov8n.pt) and emptied the CLASSES dropdown, no matter
    # what user_state.json had saved. ui_loadsave honours a per-component
    # `do_not_save_to_config` flag (ui_loadsave.py: `if getattr(obj,
    # "do_not_save_to_config", False): return`), so setting it makes our
    # persistence the single source of truth for these widgets. Covers all
    # ALL_ARGS widgets (detector, classes textboxes, confidence, prompts,
    # denoise, …) plus the UI-only visible classes dropdown. Plain attribute
    # set → NO Gradio listener added → index-safe.
    for _persisted in (*w.tolist(), w.ad_model_classes_dropdown):
        try:
            _persisted.do_not_save_to_config = True
        except Exception:  # noqa: BLE001 — never break UI build over a flag
            pass

    # The "Detection preview" and "Run ADetailer on an image" buttons' .click
    # handlers are wired LATER, in _wire_detection_previews(), AFTER every tab's
    # widgets exist — so the per-tab "Combine all tabs" checkbox can feed every
    # tab's detector settings into one handler and overlay all the boxes on a
    # single image. This mirrors how _wire_copy_paste / _wire_presets do
    # cross-tab wiring. Index-safe: these two per-tab .click handlers add NO
    # .change listener and don't disturb Forge's gallery send-to buttons, whose
    # fn_index doesn't depend on ADetailer's .click count (that count already
    # scales 1-15x with the max-models slider).

    state = gr.State(lambda: state_init(w))

    for attr in ALL_ARGS.attrs:
        widget = getattr(w, attr)
        on_change = partial(on_widget_change, attr=attr)
        widget.change(fn=on_change, inputs=[state, widget], outputs=state, queue=False)

    all_inputs = [state, *w.tolist()]
    target_button = webui_info.i2i_button if is_img2img else webui_info.t2i_button
    target_button.click(
        fn=partial(
            on_generate_click,
            mode="img2img" if is_img2img else "txt2img",
            tab_index=n,
        ),
        inputs=all_inputs,
        outputs=state,
        queue=False,
    )

    infotext_fields = [(getattr(w, attr), name + suffix(n)) for attr, name in ALL_ARGS]

    preset_widgets = (
        preset_dropdown,
        preset_load_btn,
        preset_rename_btn,
        preset_delete_btn,
        preset_name_box,
        preset_save_btn,
        preset_reset_btn,
        preset_status,
        preset_reset_all,
    )
    return w, copy_btn, paste_btn, preset_widgets, state, infotext_fields


def detection(
    w: Widgets, n: int, is_img2img: bool, saved: dict[str, Any] | None = None
):
    eid = partial(elem_id, n=n, is_img2img=is_img2img)
    sv = partial(_sv, saved or {})

    with gr.Row(elem_classes=["ad-2col-row"]):
        with gr.Column(variant="compact"):
            w.ad_confidence = gr.Slider(
                label="Detection model confidence threshold" + suffix(n),
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=sv("ad_confidence", 0.3),
                visible=True,
                elem_id=eid("ad_confidence"),
            )
            w.ad_detection_resolution = gr.Slider(
                label="Detection resolution (0 = default)" + suffix(n),
                info=(
                    "Detector inference resolution. 0 keeps the default (640); "
                    "higher (e.g. 1024) finds smaller or more distant parts but "
                    "uses more VRAM and time."
                ),
                minimum=0,
                maximum=1536,
                step=64,
                value=sv("ad_detection_resolution", 0),
                visible=True,
                elem_id=eid("ad_detection_resolution"),
            )
            w.ad_mask_filter_method = gr.Radio(
                choices=["Area", "Confidence"],
                value=sv("ad_mask_filter_method", "Area"),
                label="Method to filter top k masks by (confidence or area)"
                + suffix(n),
                visible=True,
                elem_id=eid("ad_mask_filter_method"),
            )
            w.ad_mask_k = gr.Slider(
                label="Mask only the top k (0 to disable)" + suffix(n),
                minimum=0,
                maximum=10,
                step=1,
                value=sv("ad_mask_k", 0),
                visible=True,
                elem_id=eid("ad_mask_k"),
            )

        with gr.Column(variant="compact"):
            w.ad_mask_min_ratio = gr.Slider(
                label="Mask min area ratio" + suffix(n),
                minimum=0.0,
                maximum=1.0,
                step=0.001,
                value=sv("ad_mask_min_ratio", 0.0),
                visible=True,
                elem_id=eid("ad_mask_min_ratio"),
            )
            w.ad_mask_max_ratio = gr.Slider(
                label="Mask max area ratio" + suffix(n),
                minimum=0.0,
                maximum=1.0,
                step=0.001,
                value=sv("ad_mask_max_ratio", 1.0),
                visible=True,
                elem_id=eid("ad_mask_max_ratio"),
            )


def mask_preprocessing(
    w: Widgets, n: int, is_img2img: bool, saved: dict[str, Any] | None = None
):
    eid = partial(elem_id, n=n, is_img2img=is_img2img)
    sv = partial(_sv, saved or {})

    with gr.Group():
        with gr.Row(elem_classes=["ad-2col-row"]):
            with gr.Column(variant="compact"):
                w.ad_x_offset = gr.Slider(
                    label="Mask x(→) offset" + suffix(n),
                    minimum=-200,
                    maximum=200,
                    step=1,
                    value=sv("ad_x_offset", 0),
                    visible=True,
                    elem_id=eid("ad_x_offset"),
                )
                w.ad_y_offset = gr.Slider(
                    label="Mask y(↑) offset" + suffix(n),
                    minimum=-200,
                    maximum=200,
                    step=1,
                    value=sv("ad_y_offset", 0),
                    visible=True,
                    elem_id=eid("ad_y_offset"),
                )

            with gr.Column(variant="compact"):
                w.ad_dilate_erode = gr.Slider(
                    label="Mask erosion (-) / dilation (+)" + suffix(n),
                    minimum=-128,
                    maximum=128,
                    step=4,
                    value=sv("ad_dilate_erode", 4),
                    visible=True,
                    elem_id=eid("ad_dilate_erode"),
                )

        with gr.Row():
            w.ad_mask_merge_invert = gr.Radio(
                label="Mask merge mode" + suffix(n),
                choices=MASK_MERGE_INVERT,
                value=sv("ad_mask_merge_invert", "None"),
                elem_id=eid("ad_mask_merge_invert"),
                info="None: do nothing, Merge: merge masks, Merge and Invert: merge all masks and invert",
            )

        with gr.Row(variant="compact"):
            # Forces the bbox to be used as the mask even when the detection
            # model provides a per-pixel segmentation mask. Useful for seg
            # models that produce overly-tight masks where the inpaint needs
            # more padding around the subject for natural-looking blending.
            w.ad_use_bbox_mask = gr.Checkbox(
                label="Use bbox as mask (segmentation models)" + suffix(n),
                info="Force the rectangular bounding box as the inpaint mask, even when the detector produced a precise per-pixel segmentation mask. No effect on bbox-only detectors.",
                value=sv("ad_use_bbox_mask", False),
                visible=True,
                elem_id=eid("ad_use_bbox_mask"),
            )


def inpainting(  # noqa: PLR0915
    w: Widgets,
    n: int,
    is_img2img: bool,
    webui_info: WebuiInfo,
    saved: dict[str, Any] | None = None,
):
    eid = partial(elem_id, n=n, is_img2img=is_img2img)
    sv = partial(_sv, saved or {})

    with gr.Group():
        with gr.Row(elem_classes=["ad-2up-row"]):
            w.ad_mask_blur = gr.Slider(
                label="Inpaint mask blur" + suffix(n),
                minimum=0,
                maximum=64,
                step=1,
                value=sv("ad_mask_blur", 4),
                visible=True,
                elem_id=eid("ad_mask_blur"),
            )

            w.ad_denoising_strength = gr.Slider(
                label="Inpaint denoising strength" + suffix(n),
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=sv("ad_denoising_strength", 0.4),
                visible=True,
                elem_id=eid("ad_denoising_strength"),
            )

        with gr.Row():
            w.ad_dynamic_denoise_power = gr.Slider(
                label="Dynamic denoise by area" + suffix(n),
                info=(
                    "Scales this tab's denoise by the detected region's size — "
                    "smaller regions get more denoise. 0 = use the global "
                    "Settings value (off unless set there); 2-4 is a good range."
                ),
                minimum=0.0,
                maximum=8.0,
                step=0.1,
                value=sv("ad_dynamic_denoise_power", 0.0),
                visible=True,
                elem_id=eid("ad_dynamic_denoise_power"),
            )

        with gr.Row(elem_classes=["ad-2col-row"]):
            with gr.Column(variant="compact"):
                w.ad_inpaint_only_masked = gr.Checkbox(
                    label="Inpaint only masked" + suffix(n),
                    value=sv("ad_inpaint_only_masked", True),
                    visible=True,
                    elem_id=eid("ad_inpaint_only_masked"),
                )
                w.ad_inpaint_only_masked_padding = gr.Slider(
                    label="Inpaint only masked padding, pixels" + suffix(n),
                    minimum=0,
                    maximum=256,
                    step=4,
                    value=sv("ad_inpaint_only_masked_padding", 32),
                    visible=True,
                    elem_id=eid("ad_inpaint_only_masked_padding"),
                )

                w.ad_inpaint_only_masked.change(
                    gr_interactive,
                    inputs=w.ad_inpaint_only_masked,
                    outputs=w.ad_inpaint_only_masked_padding,
                    queue=False,
                )

            with gr.Column(variant="compact"):
                w.ad_use_inpaint_width_height = gr.Checkbox(
                    label="Use separate width/height" + suffix(n),
                    value=sv("ad_use_inpaint_width_height", False),
                    visible=True,
                    elem_id=eid("ad_use_inpaint_width_height"),
                )

                w.ad_inpaint_width = gr.Slider(
                    label="inpaint width" + suffix(n),
                    minimum=64,
                    maximum=2048,
                    step=4,
                    value=sv("ad_inpaint_width", 512),
                    visible=True,
                    elem_id=eid("ad_inpaint_width"),
                )

                w.ad_inpaint_height = gr.Slider(
                    label="inpaint height" + suffix(n),
                    minimum=64,
                    maximum=2048,
                    step=4,
                    value=sv("ad_inpaint_height", 512),
                    visible=True,
                    elem_id=eid("ad_inpaint_height"),
                )

                w.ad_use_inpaint_width_height.change(
                    lambda value: (gr_interactive(value), gr_interactive(value)),
                    inputs=w.ad_use_inpaint_width_height,
                    outputs=[w.ad_inpaint_width, w.ad_inpaint_height],
                    queue=False,
                )

                # Scale-based resolution: alternative to absolute width/height,
                # computes the inpaint canvas as `bbox_size * multiplier` so
                # the canvas always exceeds the source bbox. Mutually exclusive
                # with `Use separate width/height` (the fixed-dim toggle wins).
                w.ad_use_resolution_scale = gr.Checkbox(
                    label="Scale inpaint to bbox" + suffix(n),
                    info="Use bbox_size × scale as the inpaint canvas. Overridden by 'Use separate width/height' when both are on.",
                    value=sv("ad_use_resolution_scale", False),
                    visible=True,
                    elem_id=eid("ad_use_resolution_scale"),
                )
                w.ad_resolution_scale = gr.Slider(
                    label="Inpaint resolution scale" + suffix(n),
                    minimum=0.5,
                    maximum=8.0,
                    step=0.05,
                    value=sv("ad_resolution_scale", 1.5),
                    visible=True,
                    elem_id=eid("ad_resolution_scale"),
                )
                # Disable the slider unless the toggle is on, so the UI
                # makes the binding obvious.
                w.ad_use_resolution_scale.change(
                    lambda value: gr_interactive(value),
                    inputs=w.ad_use_resolution_scale,
                    outputs=w.ad_resolution_scale,
                    queue=False,
                )

        with gr.Row(elem_classes=["ad-2col-row"]):
            with gr.Column(variant="compact"):
                w.ad_use_steps = gr.Checkbox(
                    label="Use separate steps" + suffix(n),
                    value=sv("ad_use_steps", False),
                    visible=True,
                    elem_id=eid("ad_use_steps"),
                )

                w.ad_steps = gr.Slider(
                    label="ADetailer steps" + suffix(n),
                    minimum=1,
                    maximum=150,
                    step=1,
                    value=sv("ad_steps", 28),
                    visible=True,
                    elem_id=eid("ad_steps"),
                )

                w.ad_use_steps.change(
                    gr_interactive,
                    inputs=w.ad_use_steps,
                    outputs=w.ad_steps,
                    queue=False,
                )

            with gr.Column(variant="compact"):
                w.ad_use_cfg_scale = gr.Checkbox(
                    label="Use separate CFG scale" + suffix(n),
                    value=sv("ad_use_cfg_scale", False),
                    visible=True,
                    elem_id=eid("ad_use_cfg_scale"),
                )

                w.ad_cfg_scale = gr.Slider(
                    label="ADetailer CFG scale" + suffix(n),
                    minimum=0.0,
                    maximum=30.0,
                    step=0.5,
                    value=sv("ad_cfg_scale", 7.0),
                    visible=True,
                    elem_id=eid("ad_cfg_scale"),
                )

                w.ad_use_cfg_scale.change(
                    gr_interactive,
                    inputs=w.ad_use_cfg_scale,
                    outputs=w.ad_cfg_scale,
                    queue=False,
                )

        with gr.Row(elem_classes=["ad-2col-row"]):
            with gr.Column(variant="compact"):
                w.ad_use_checkpoint = gr.Checkbox(
                    label="Use separate checkpoint" + suffix(n),
                    value=sv("ad_use_checkpoint", False),
                    visible=True,
                    elem_id=eid("ad_use_checkpoint"),
                )

                ckpts = ["Use same checkpoint", *webui_info.checkpoints_list]
                _saved_ckpt = sv("ad_checkpoint", ckpts[0])
                if _saved_ckpt not in ckpts:
                    _saved_ckpt = ckpts[0]

                w.ad_checkpoint = gr.Dropdown(
                    label="ADetailer checkpoint" + suffix(n),
                    choices=ckpts,
                    value=_saved_ckpt,
                    visible=True,
                    elem_id=eid("ad_checkpoint"),
                )

            with gr.Column(variant="compact"):
                w.ad_use_vae = gr.Checkbox(
                    label="Use separate VAE" + suffix(n),
                    value=sv("ad_use_vae", False),
                    visible=True,
                    elem_id=eid("ad_use_vae"),
                )

                vaes = ["Use same VAE", *webui_info.vae_list]
                _saved_vae = sv("ad_vae", vaes[0])
                if _saved_vae not in vaes:
                    _saved_vae = vaes[0]

                w.ad_vae = gr.Dropdown(
                    label="ADetailer VAE" + suffix(n),
                    choices=vaes,
                    value=_saved_vae,
                    visible=True,
                    elem_id=eid("ad_vae"),
                )

        # Per-pass text encoder (Forge / Forge Neo). Lets the detailer step use
        # a different text encoder than the base generation — e.g. drop a ZiT
        # (Z-Image) encoder so an SDXL detailer checkpoint can run (issue #3).
        # Index-safe: mirrors the checkpoint/VAE pair — the checkbox has NO
        # grey-out .change() of its own, so only the uniform per-ALL_ARGS
        # listener is added. On A1111 the encoders list is empty and the whole
        # thing is a harmless no-op (guarded again at override time).
        with gr.Row(), gr.Column(variant="compact"):
            w.ad_use_text_encoder = gr.Checkbox(
                label="Use separate text encoder (Forge/Forge Neo)" + suffix(n),
                value=sv("ad_use_text_encoder", False),
                visible=True,
                elem_id=eid("ad_use_text_encoder"),
            )

            tes = [
                "Use same text encoder",
                "None (use detailer checkpoint's own)",
                *webui_info.text_encoders_list,
            ]
            _saved_te = sv("ad_text_encoder", tes[0])
            if _saved_te not in tes:
                _saved_te = tes[0]

            w.ad_text_encoder = gr.Dropdown(
                label="ADetailer text encoder" + suffix(n),
                choices=tes,
                value=_saved_te,
                visible=True,
                elem_id=eid("ad_text_encoder"),
            )

        with gr.Row(), gr.Column(variant="compact"):
            w.ad_use_sampler = gr.Checkbox(
                label="Use separate sampler" + suffix(n),
                value=sv("ad_use_sampler", False),
                visible=True,
                elem_id=eid("ad_use_sampler"),
            )

            sampler_names = [
                "Use same sampler",
                *webui_info.sampler_names,
            ]
            _saved_sampler = sv("ad_sampler", sampler_names[1])
            if _saved_sampler not in sampler_names:
                _saved_sampler = sampler_names[1]

            with gr.Row(elem_classes=["ad-2up-row"]):
                w.ad_sampler = gr.Dropdown(
                    label="ADetailer sampler" + suffix(n),
                    choices=sampler_names,
                    value=_saved_sampler,
                    visible=True,
                    elem_id=eid("ad_sampler"),
                )

                scheduler_names = [
                    "Use same scheduler",
                    *webui_info.scheduler_names,
                ]
                _saved_scheduler = sv("ad_scheduler", scheduler_names[0])
                if _saved_scheduler not in scheduler_names:
                    _saved_scheduler = scheduler_names[0]

                w.ad_scheduler = gr.Dropdown(
                    label="ADetailer scheduler" + suffix(n),
                    choices=scheduler_names,
                    value=_saved_scheduler,
                    visible=len(scheduler_names) > 1,
                    elem_id=eid("ad_scheduler"),
                )

                w.ad_use_sampler.change(
                    lambda value: (gr_interactive(value), gr_interactive(value)),
                    inputs=w.ad_use_sampler,
                    outputs=[w.ad_sampler, w.ad_scheduler],
                    queue=False,
                )

        with gr.Row(elem_classes=["ad-2col-row"]):
            with gr.Column(variant="compact"):
                w.ad_use_noise_multiplier = gr.Checkbox(
                    label="Use separate noise multiplier" + suffix(n),
                    value=sv("ad_use_noise_multiplier", False),
                    visible=True,
                    elem_id=eid("ad_use_noise_multiplier"),
                )

                w.ad_noise_multiplier = gr.Slider(
                    label="Noise multiplier for img2img" + suffix(n),
                    minimum=0.5,
                    maximum=1.5,
                    step=0.01,
                    value=sv("ad_noise_multiplier", 1.0),
                    visible=True,
                    elem_id=eid("ad_noise_multiplier"),
                )

                w.ad_use_noise_multiplier.change(
                    gr_interactive,
                    inputs=w.ad_use_noise_multiplier,
                    outputs=w.ad_noise_multiplier,
                    queue=False,
                )

            with gr.Column(variant="compact"):
                w.ad_use_clip_skip = gr.Checkbox(
                    label="Use separate CLIP skip" + suffix(n),
                    value=sv("ad_use_clip_skip", False),
                    visible=True,
                    elem_id=eid("ad_use_clip_skip"),
                )

                w.ad_clip_skip = gr.Slider(
                    label="ADetailer CLIP skip" + suffix(n),
                    minimum=1,
                    maximum=12,
                    step=1,
                    value=sv("ad_clip_skip", 1),
                    visible=True,
                    elem_id=eid("ad_clip_skip"),
                )

                w.ad_use_clip_skip.change(
                    gr_interactive,
                    inputs=w.ad_use_clip_skip,
                    outputs=w.ad_clip_skip,
                    queue=False,
                )

        with gr.Row(), gr.Column(variant="compact"):
            w.ad_restore_face = gr.Checkbox(
                label="Restore faces after ADetailer" + suffix(n),
                value=sv("ad_restore_face", False),
                elem_id=eid("ad_restore_face"),
            )


def controlnet(
    w: Widgets, n: int, is_img2img: bool, saved: dict[str, Any] | None = None
):
    eid = partial(elem_id, n=n, is_img2img=is_img2img)
    sv = partial(_sv, saved or {})
    cn_models = ["None", "Passthrough", *get_cn_models()]
    _saved_cn = sv("ad_controlnet_model", "None")
    if _saved_cn not in cn_models:
        _saved_cn = "None"

    # `ad-cn-row` class added 2026-05-18 — user reported the stacked
    # dropdowns/sliders inside the two columns were visually stuck
    # together (no breathing room). The CSS rule in style.css adds a
    # 10px margin-top between siblings inside each column.
    with gr.Row(variant="panel", elem_classes=["ad-cn-row"]):
        with gr.Column(variant="compact"):
            w.ad_controlnet_model = gr.Dropdown(
                label="ControlNet model" + suffix(n),
                choices=cn_models,
                value=_saved_cn,
                visible=True,
                type="value",
                interactive=controlnet_exists,
                elem_id=eid("ad_controlnet_model"),
            )

            w.ad_controlnet_module = gr.Dropdown(
                label="ControlNet module" + suffix(n),
                choices=["None"],
                value=sv("ad_controlnet_module", "None"),
                visible=False,
                type="value",
                interactive=controlnet_exists,
                elem_id=eid("ad_controlnet_module"),
            )

            w.ad_controlnet_weight = gr.Slider(
                label="ControlNet weight" + suffix(n),
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=sv("ad_controlnet_weight", 1.0),
                visible=True,
                interactive=controlnet_exists,
                elem_id=eid("ad_controlnet_weight"),
            )

            w.ad_controlnet_model.change(
                on_cn_model_update,
                inputs=w.ad_controlnet_model,
                outputs=w.ad_controlnet_module,
                queue=False,
            )

        with gr.Column(variant="compact"):
            w.ad_controlnet_guidance_start = gr.Slider(
                label="ControlNet guidance start" + suffix(n),
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=sv("ad_controlnet_guidance_start", 0.0),
                visible=True,
                interactive=controlnet_exists,
                elem_id=eid("ad_controlnet_guidance_start"),
            )

            w.ad_controlnet_guidance_end = gr.Slider(
                label="ControlNet guidance end" + suffix(n),
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=sv("ad_controlnet_guidance_end", 1.0),
                visible=True,
                interactive=controlnet_exists,
                elem_id=eid("ad_controlnet_guidance_end"),
            )
