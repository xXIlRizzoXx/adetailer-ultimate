# Changelog

## 2026-05-18 (ui + ux: native hover tooltips on buttons + taller class-prompts box)

Two user-feedback follow-ups:

**V66 — `ad_class_prompts` textbox felt cramped.** The multi-line
placeholder example (4 lines: syntax intro + "Example:" label + 2
sample entries) was bumping against the bottom of the box. Bumped
`lines=4` → `lines=5` so the placeholder fits without scrolling.

**V68 — buttons had no hover tooltips.** `gr.Button` doesn't expose an
`info=` parameter the way `gr.Checkbox` / `gr.Dropdown` / `gr.Slider`
do, so there was no built-in mechanism. Added a small standalone
JavaScript file that walks the fork's action buttons and sets the
native HTML `title` attribute — the browser then renders the standard
tooltip after ~1 s of hover. Buttons covered:

- Copy / Paste settings (top-of-tab clipboard)
- Load / Rename / Delete / Save preset / Reset preset (preset library)
- Export to JSON / Import (preset library export-import accordion)
- Run detection preview (Detection preview accordion)

Files:
- `aaaaaa/ui.py`: `w.ad_class_prompts = gr.Textbox(..., lines=5, ...)`
  (was 4). Comment block notes the user-feedback rationale.
- `javascript/button-tooltips.js`: new file. ~70 LOC pure JS, no
  dependencies. Maps each elem_id fragment to a tooltip string and
  applies it via `el.title = "..."`. Re-runs via `MutationObserver`
  so tooltips survive Gradio's reactive rerenders (preset load,
  tab switches). Idempotent (guarded by `!target.title`).
- `preview.py`: extended to read and inject every `javascript/*.js`
  file via `gr.Blocks(head="<script>...</script>")`, the same way
  Forge Neo auto-loads them at the extension root. Without this,
  the Claude Preview tab wouldn't show the tooltips.

Verified live via Claude Preview: tooltip text appears on hover for
all 7 fork buttons (4 immediately, 3 after opening their parent
accordions — the MutationObserver picks them up automatically).

## 2026-05-18 (fix: amber-pill not nested anymore — single chip)

User-reported follow-up on the previous amber-pill commit: the warning
rendered as TWO nested amber chips (one inside the other) because
Gradio propagates `elem_classes=["ad-preview-status"]` to BOTH the
outer `.block` wrapper AND the inner `.prose` markdown wrapper. The
`.ad-preview-status { background; border-left; padding; rounded }` rule
was matching both, drawing two boxes.

Fix: scope the pill styling to only the outer wrapper via
`div.block.ad-preview-status` (more specific selector that doesn't
match the inner `.prose` markdown div). Text properties (color, font
size, opacity) stay on `.ad-preview-status p` so they apply regardless
of which wrapper the `<p>` is nested under.

style.css
- `.ad-preview-status, .ad-preview-status p { ... }` → split into:
    `div.block.ad-preview-status { background; border-left; padding; rounded }`
    `.ad-preview-status p          { color; font-size; opacity; reset margins }`
- Comment block expanded to warn future edits about the dual-class
  Gradio propagation.

Verified live via Claude Preview: outer wrapper has the amber pill
styling (bg + border + padding), inner wrapper has transparent
background + 0 padding. Single chip on screen, no nesting.

## 2026-05-18 (ux: legible amber-pill warning for Detection preview status)

User-reported: the "⚠️ Pick a detector model first." warning emitted by
the Detection preview button was rendering as faded 11px grey text
because the status markdown widget shared the `.ad-preset-status`
class (which is intentionally dim, designed for non-shouting status
lines like "preset saved"). Warnings need to be readable at a glance.

- aaaaaa/ui.py: `w.ad_preview_status` now uses `elem_classes=["ad-preview-status"]`
  (dedicated class) instead of sharing `.ad-preset-status`. Comment
  block notes the rationale.
- style.css: new `.ad-preview-status` ruleset — full opacity, 12px font,
  amber color (#fbbf24), subtle amber pill background, amber-tinted
  border-left. Auto-hide-when-empty rule extended to include the new
  class so the pill only appears when there's a message to show.

Verified live via Claude Preview by injecting the warning text into
the markdown widget: the pill renders as a clearly-visible amber
"chip" inline with the Run-detection-preview button.

## 2026-05-18 (ux: info text on CLASSES dropdown explains "empty = all")

User-reported confusion: when a multiclass detector is selected, the
CLASSES dropdown is intentionally NOT auto-populated (it's a filter —
empty means "no narrowing, all classes detected"). But the empty state
looks identical to "broken, nothing will be detected".

Added a `gr.Dropdown(..., info="If empty, ALL classes the model
produces are inpainted. Select to narrow down to specific classes.")`
to the `ad_model_classes_dropdown` widget so the empty default is
self-explanatory.

- aaaaaa/ui.py: `gr.Dropdown(info=...)` parameter added to
  `w.ad_model_classes_dropdown`. Comment block notes the UX rationale.

## 2026-05-18 (ui: margin between preset-name-to-save row and Copy/Paste row)

User reported the Copy/Paste settings buttons were touching the
"Preset name to save" textbox directly above them in Forge Neo — no
breathing room. Added a `.ad-tab-clipboard-row` class on the Copy/Paste
row and a `margin-top: 10px` rule in `style.css` mirroring the
existing `.ad-preset-save-row` spacing. The two rows now sit with a
visible 10px gap.

- aaaaaa/ui.py: gr.Row for Copy/Paste now has elem_classes=["ad-tab-clipboard-row"].
- style.css: new rule for `.ad-tab-clipboard-row` { margin-top: 10px !important }.

## 2026-05-18 (ui: shorter label on the export button — "📤 Esport")

User-requested label tweak after the nowrap fix landed: the export
DownloadButton now reads `📤 Esport` (was `📤 Export to JSON`).
Shorter label = tighter button, more consistent visual weight with the
adjacent `📥 Import` UploadButton. Emoji kept on both buttons for
parallel visual markers.

## 2026-05-18 (ui: nowrap on Export/Import buttons + preview.py brand rename)

Visual-inspection sweep via Claude Preview turned up a regression on the new
Export/Import buttons added 2026-05-16: the "Export to JSON" label wrapped
onto two lines (taller than its single-line siblings) because the CSS
nowrap rule had not been extended to cover the new button elem_ids.

- `style.css`: `ad_preset_export_btn` and `ad_preset_import_btn` added to
  the `button[id*="..."]` nowrap selector list. `gr.DownloadButton` /
  `gr.UploadButton` render as `<button>` at the root so the same rule
  applies — confirmed via preview restart.
- `preview.py` + `.claude/launch.json`: title strings updated from
  "ADetailer Plus" → "ADetailer Ultimate" so the Claude Preview header
  is consistent with the current brand. Launch-config name renamed
  `adetailer-plus-preview` → `adetailer-ultimate-preview`.

Same-pass also confirmed via Preview the following are working as
expected: brand-prefixed live overlay with current git short-hash
(7fe99b9), Copy/Paste row below preset-name-to-save row, compact
Export/Import accordion (~70px expanded), empty preset_status /
preset_io_status hidden via `:has(.md:empty)` (no dead bands).

## 2026-05-16 (ui: hide empty preview/status markdown containers)

User reported a visual artifact: an empty styled band between the preset row and the detector section, even with no preset selected. The cause: `gr.Markdown` widgets with empty content still render their outer `div.block` container, and our CSS adds padding / border-left / background to `.ad-preset-preview`, making the empty shell visible as a useless box. Two-pronged fix:

- **Python side** (`aaaaaa/ui.py`): `_format_preset_preview` now returns `gr.update(value=..., visible=bool)` instead of a raw string, so the preview component is hidden entirely when no preset is selected (or `(none)` sentinel chosen). Initial `preset_preview = gr.Markdown(..., visible=False)`.
- **CSS fallback** (`style.css`): new rule using the `:has(.md:empty)` selector to collapse any `.ad-preset-preview` or `.ad-preset-status` container whose inner markdown span is empty. Covers `preset_status` and `preset_io_status` (whose handlers live in `_wire_presets` and weren't refactored), and acts as a safety net for the preview if Gradio ever fails to apply the visibility update.

End result: the preset block now has zero vertical footprint between rows when no message / preview is showing. The styled border + background only appear when there's actual content.

## 2026-05-16 (ui: tab-state controls reordered + Export/Import compacted)

Two layout-level changes to the tab-state block at the top of each ADetailer tab. The widgets and behaviour are unchanged; only the order and the rendering of Export/Import are different.

**Reorder (top-to-bottom of the tab now)**:
1. `Enable this tab` — alone on its row (was sharing the row with Copy/Paste).
2. `▾ Preset library export / import` accordion — **moved from the bottom of the preset block to immediately under Enable**. Collapsed by default, so the daily-use preset row stays the first thing the eye lands on after Enable.
3. **Preset library** — `Saved presets ▾ + Load + Rename + Delete` row, then `Preset name + Save preset + Reset` row (unchanged).
4. `[📋 Copy settings]  [📥 Paste settings]` — **moved DOWN from the very top of the tab to directly under the preset-name-to-save row**. All "tab-state copying operations" (Save preset, Reset, Copy, Paste) are now grouped in one visual block.
5. Preset status + live preview markdown (unchanged).

**Compaction of the Export/Import accordion**:
- The two big `gr.File` drop-zones (each rendering ~100px tall whether or not they have content) are replaced by `gr.DownloadButton` for export and `gr.UploadButton` for import — both render as ordinary buttons, much shorter vertically.
- Three controls now fit on a single row: `[📤 Export to JSON]  [📥 Import]  ☐ Overwrite on conflict`.
- The accordion's expanded height drops from ~280px to ~70px.
- Wiring change: `preset_export_btn.click(...).then(...)` — the click handler returns a file path that Gradio uses to trigger the download AND updates `preset_export_btn` itself (modern DownloadButton pattern), followed by a `.then(...)` that writes the status line. `preset_import_btn.upload(...)` fires on file pick/drop and the button's value carries the uploaded path.

## 2026-05-16 (ui: overlay auto-updates with current git short-hash)

The top-right overlay used to be a static `"ADetailer Ultimate · v26.2.0+plus.2"` string — informative but unable to signal "is my install current?" because the locked `__version__` never changes between commits. Now the overlay also appends the current commit's 7-char short hash, read directly from `<extension_root>/.git/HEAD` at UI-build time:

```
ADetailer Ultimate · v26.2.0+plus.2 · 63a9dd2
```

- `aaaaaa/ui.py`: two new helpers `_read_git_short_hash()` (reads HEAD + the referenced ref file, with packed-refs fallback) and `_build_overlay_text()` (composes the final string). The `gr.Markdown` for the overlay now calls `_build_overlay_text()` instead of a hardcoded f-string.
- `style.css`: `.ad-version-overlay` `max-width` raised from 280px → 360px to fit the longer string (~47 chars at 11px).
- No `subprocess` call — pure file reads, works without `git` on PATH, degrades to brand-only when `.git` is missing (e.g. zip installs).
- The hash auto-updates every time `adui()` rebuilds the panel (Forge Neo restart, full UI reload). User can verify their install is at the latest commit by comparing the overlay hash to the one reported at the end of each push report.

## 2026-05-16 (ui: version overlay now brand-prefixed)

- The version badge in the top-right of the accordion header used to read just `v26.2.0+plus.2`. After the rename to **ADetailer Ultimate** + the addition of ~37 fork features, that string alone was ambiguous (the `+plus.2` build-metadata refers to a previous fork name kept locked per the no-auto-bump rule). The overlay now reads `ADetailer Ultimate · v26.2.0+plus.2` so the brand is visible at a glance without altering the locked version string.
- `aaaaaa/ui.py`: `gr.Markdown(f"ADetailer Ultimate · v{__version__}", ...)`.
- `style.css`: `.ad-version-overlay` `max-width` bumped from 220px to 280px to fit the longer string without horizontal clipping.

## 2026-05-16 (feat: 5-feature batch — peer-fork roadmap items 3, 4, 5, 8, 9)

Implements five of the remaining roadmap items in one batch. All five are 🟡 (in the codebase, awaiting hands-on verification by the repo owner — Tests 22 through 26 added to the pending sticky list).

### Detection / mask

- **Bounding-box mask for segmentation models** (`ad_use_bbox_mask`, default off) — new per-tab checkbox in the **Mask preprocessing** accordion. Forces the rectangular bounding box as the inpaint mask even when the YOLO model produced a precise per-pixel segmentation mask. Useful when the seg mask is too tight against the subject and the inpaint needs more breathing room. Implemented as a single conditional in `adetailer.ultralytics.ultralytics_predict` + a new `use_bbox_mask` kwarg in its signature. Inspired by [newtextdoc1111/adetailer](https://github.com/newtextdoc1111/adetailer).

### Inpainting / resolution

- **Scale-based inpaint resolution** (`ad_use_resolution_scale` + `ad_resolution_scale`, default off / 1.5×) — new checkbox + slider in the **Inpainting** section. When the toggle is on, the inpaint canvas is `bbox_size × scale` (rounded down to a multiple of 8 for SD UNet compatibility, 64-pixel floor). Mutually exclusive with the existing `Use separate width/height` toggle — when both are on, the fixed-dimensions toggle wins. Math centralised in `fix_p2`. Inspired by [Anzhc/aadetailer-reforge](https://github.com/Anzhc/aadetailer-reforge).

### Sequential class detection / prompts

- **Class-specific prompts** (`ad_class_prompts`, default empty) — new multiline textbox in the **Inpaint prompts** accordion. Syntax (one per line): `classname: positive_prompt [| negative_prompt]`. When the sequential class detection feature is on, each class's pass reads its dedicated prompt from this textbox; entries with empty values fall back to the tab's default `ad_prompt`/`ad_negative_prompt`. Lines that don't match the syntax are silently ignored. Parser `_parse_class_prompts` in `scripts/!adetailer.py`. Inspired by [newtextdoc1111/adetailer](https://github.com/newtextdoc1111/adetailer).

### Preset library polish

- **Live preset preview with `[SEP]`/`[PROMPT]` awareness** — new markdown block under the preset dropdown that updates on every `preset_dropdown.change` event. Shows the highlighted preset's detector, classes (include/exclude), sequential flag, prompts (truncated), and class-specific prompts summary. `[SEP]` and `[PROMPT]` tokens are wrapped in backticks with a footnote reminding the user they'll be expanded at generation time. Formatter `_format_preset_preview` in `aaaaaa/ui.py`. CSS scoped via `.ad-preset-preview` in `style.css`.
- **Export / Import preset library to JSON** — new "Preset library export / import" accordion (collapsed by default) under the preset row. **Export**: button generates a `gr.File` download of the entire `user_presets.json` (sorted, indented). **Import**: drop a JSON file in the upload box, optionally tick "Overwrite existing on conflict", click Import. Status line summarises added / replaced / skipped counts. Cross-tab dropdown refresh after import is local-to-current-tab (other tabs pick up new presets on next UI reload). Library helpers `export_presets_json` and `import_presets_json` in `adetailer/presets.py`.

### Pydantic schema additions

- `ad_use_bbox_mask: bool = False`
- `ad_use_resolution_scale: bool = False`
- `ad_resolution_scale: confloat(ge=0.5, le=8.0) = 1.5`
- `ad_class_prompts: str = ""`

All four ship with infotext mapping entries so they round-trip through PNG-info save/load.

## 2026-05-16 (audit + fixes: txt2img/img2img parity)

Code-review audit of every fork feature against both `StableDiffusionProcessingTxt2Img` and `StableDiffusionProcessingImg2Img` pipelines. Two issues found and fixed; everything else was already mode-agnostic.

- **Fix A — "Apply only on hires.fix" wrongly skipped img2img.** The toggle, when on, was treating img2img runs as "hires.fix is off → skip the tab entirely". Img2img has no hires.fix concept, so this manifested as the tab silently doing nothing when a user enabled the toggle in txt2img and later opened img2img. Fix on two layers:
  - `_should_skip_for_hires_only` now early-returns `False` when `isinstance(p, StableDiffusionProcessingImg2Img)`. The toggle becomes a no-op in img2img.
  - UI checkbox is now `visible=not is_img2img` (symmetric with the existing `ad_skip_img2img` widget which is `visible=is_img2img`). The widget still exists in the component list — its value from persistence/preset is honoured by the runtime check above as defense-in-depth.
- **Fix B — persistence shared state between txt2img and img2img.** `user_state.json` was keyed by tab index only (`"0"`, `"1"`, …), so a Generate click in img2img Tab 1 overwrote whatever txt2img Tab 1 had stashed. Now keys are scoped as `"<mode>:<tab_index>"` (e.g. `"txt2img:0"`, `"img2img:2"`). Legacy unscoped keys still load for both modes for backwards compatibility on upgrade — the next Generate writes the scoped form and the legacy entry stays dormant until the file is overwritten.
- Files touched: `scripts/!adetailer.py` (helper), `aaaaaa/ui.py` (checkbox visibility + `mode` param wiring), `adetailer/persistence.py` (new scoping logic + back-compat legacy reads).

Other audited features confirmed mode-agnostic: class filtering (include + NOT + sequential + activation order), detection preview, JSON sidecar tolerance, prompt append fields, LoRA inclusion + trigger extraction, Copy/Paste between tabs, named preset library, manual mode, save intermediate steps, all UI polish.

## 2026-05-16 (feat: "Apply only on hires.fix" toggle)

- **"Apply only on hires.fix"** — new per-tab checkbox `ad_apply_on_hires_only` (default off) that skips the tab's ADetailer pass during the lowres pre-hires.fix postprocess call and runs it only when the post-upscale image is ready. Saves compute when hires.fix is going to overwrite the lowres detail anyway. Inspired by [Anzhc/aadetailer-reforge](https://github.com/Anzhc/aadetailer-reforge).
- Decision matrix (helper `_should_skip_for_hires_only(p, args)`):
  - Toggle off → never skip.
  - Toggle on, hires.fix enabled, **in hires pass** (`p.is_hr_pass == True`) → run normally.
  - Toggle on, hires.fix enabled, lowres pre-hires call (`is_hr_pass == False`) → skip.
  - Toggle on, hires.fix off, or img2img run → skip entirely (the user explicitly asked for hires-only and no hires step is coming).
- Files:
  - `adetailer/args.py`: new pydantic field + infotext mapping `"ADetailer apply on hires only"`.
  - `scripts/!adetailer.py`: new helper `_should_skip_for_hires_only`, called inside the per-tab loop in `postprocess_image` right after `args.need_skip()`.
  - `aaaaaa/ui.py`: new `gr.Checkbox` on its own row below the LoRA checkboxes.
- Status: implemented, **awaiting hands-on verification** by the repo owner (Test 21 added to the pending list).

## 2026-05-16 (feat: LoRA trigger extraction)

- **LoRA trigger extraction** — new sub-toggle `Append LoRA triggers from name` (`ad_use_lora_triggers`, default off) under the existing `Use LoRAs from main prompt` checkbox. When both checkboxes are on, ADetailer parses the convention `<lora:name (trigger phrase):weight>` (from [Anzhc/aadetailer-reforge](https://github.com/Anzhc/aadetailer-reforge)) and appends the parenthesised trigger phrase to the inpaint prompt. Triggers are deduplicated case-insensitively against the existing prompt body. Backwards-compatible: LoRA tags without parentheses are unaffected, and the negative-prompt pipeline is left untouched (triggers only make sense in the positive).
- Implementation:
  - New regex `_LORA_TRIGGER_RE = re.compile(r"\(([^)]+)\)")` matching the first non-greedy parenthesised substring inside a LoRA tag's name.
  - New helpers `_extract_lora_triggers(tags)` and `_append_lora_triggers(prompt, triggers)` in `scripts/!adetailer.py`.
  - `_get_prompt` gains an `include_triggers: bool = False` keyword. `get_prompt` passes `bool(args.ad_use_main_loras and args.ad_use_lora_triggers)`.
  - New pydantic field `ad_use_lora_triggers: bool = False` and infotext mapping entry `"ADetailer use lora triggers"`.
  - UI: a second checkbox added to the existing LoRA row, with `info=` hint showing the expected convention.
- Status: implemented, **awaiting hands-on verification** by the repo owner (Test 20 added to the pending list).

## 2026-05-16 (rename → ADetailer Ultimate)

- Project renamed to **ADetailer Ultimate**: GitHub repo `xXIlRizzoXx/adetailer-plus` → `xXIlRizzoXx/adetailer-ultimate`. README title and install URL updated. `style.css` header comment updated. The slug was briefly `adetailer_ultimate` (underscore) for a few minutes before being normalised to `adetailer-ultimate` (hyphen) to match the SD WebUI extension-ecosystem convention. All previous URLs continue to work via GitHub's automatic redirect chain (`xXIlRizzoXx/adetailer` → `adetailer-plus` → `adetailer_ultimate` → `adetailer-ultimate`).

## 2026-05-16 (rename → ADetailer Plus + roadmap expansion)

- Project renamed back to **ADetailer Plus**: GitHub repo `xXIlRizzoXx/adetailer` → `xXIlRizzoXx/adetailer-plus`. README title and install URL updated. `style.css` header comment updated. Old `xXIlRizzoXx/adetailer` URLs continue to work via GitHub's automatic redirect.
- README **Roadmap (not yet implemented)** section expanded from 4 to 10 items after analysing the two most-starred forks of upstream `Bing-su/adetailer`:
  - From [Anzhc/aadetailer-reforge](https://github.com/Anzhc/aadetailer-reforge) (22★): LoRA trigger extraction (`<lora:name (trigger):1>` parsing), "Apply only on hires.fix" toggle, scale-based resolution, WDv3 autotagging.
  - From [newtextdoc1111/adetailer](https://github.com/newtextdoc1111/adetailer) (14★): class-specific prompts (per-class prompt in sequential mode), bounding-box mask option for segmentation models.
- The new Roadmap table includes an **Inspiration** column crediting the upstream fork where each idea originated.

## 2026-05-15 → 2026-05-16 (plus: workflow ergonomics — extended)

- v26.2.0+plus.2 (version locked here per repo-owner request; further fork features ship under the same string until an explicit bump)
- Project renaming history (pre-2026-05-16): `adetailer-classfilter` (initial) → `adetailer-plus` (scope expansion) → `adetailer` (briefly simplified) → `adetailer-plus` (current). GitHub redirects keep all old URLs working.

### Detection

- **Sequential class detection** — new "Process classes sequentially" checkbox. When multiple classes are selected in the dropdown, runs one detect+inpaint pass per class in dropdown order, each operating on the output of the previous. Better separation of regions and cleaner per-class inpainting at the cost of longer runtime. Ignored for MediaPipe, NOT mode, and single-class selections. Implemented via top-of-function recursion in `_postprocess_image_inner` with single-class `args.copy(update=...)`.
- **Class pass order = activation order** — the order in which the user clicks classes in the multi-select dropdown is the order they're processed under Sequential class detection. Re-ordering = click × on a token then re-click its name (it goes to the end). Native Gradio behaviour; no JS. An earlier iteration shipped a `javascript/class-reorder.js` HTML5 drag-and-drop handler with a deselect-then-reselect sync; it caused tokens to flicker out of the DOM during the operation, and the simpler native-order approach makes it unnecessary. The JS file is removed.
- **Detection preview** — accordion at the bottom of each tab with a "Run detection preview" button. Runs the configured detector against the most recent generation (or img2img input) and renders bounding boxes / mask without inpainting. Useful for tuning confidence + mask preprocessing without burning a full generation.

### Workflow & prompting

- **`ad_prompt_append` / `ad_negative_prompt_append`** — two new single-line fields under the main prompt textboxes that append to the resolved inpaint prompt without forcing the user to duplicate the main prompt. New pydantic fields with empty-string defaults; stripped from infotext when at defaults.
- **Include LoRAs from main prompt** — when the tab's prompt is blank and the checkbox is on, `<lora:name:weight>` tags are scraped out of the main txt2img/img2img prompt and merged into the inpaint prompt. New pydantic field `ad_use_main_loras: bool`.
- **Copy / Paste between tabs** — clipboard-style flow: one "Copy settings" button per tab snapshots the current tab's processing settings; every other tab's "Paste settings" button enables and re-labels to "Paste settings from Nth tab here", clicking it applies the snapshot. Detector, class filter and per-tab enable are deliberately excluded from the snapshot. The clipboard is sticky — paste into multiple tabs in a row, or overwrite by Copying from a different tab.
- **Named preset library** — Load / Save / Delete / Rename per tab, dropdown shared across tabs. Each preset stores every widget value in the tab. Persisted to `<extension_root>/user_presets.json` with atomic writes; corruption-tolerant. A `(none)` sentinel entry sits at the top of every dropdown for explicit clearing without touching widget state. `Reset preset` clears the dropdown label without modifying widgets. Implemented in `adetailer/presets.py`.
- **Persistent last-used settings** — every Generate click stashes per-tab widget state to `<extension_root>/user_state.json` (atomic write). Restored as initial values at the next WebUI start. Toggle in `Settings → ADetailer → Remember last used settings` (default on). Implemented in `adetailer/persistence.py`.
- **Manual mode** — `Settings → ADetailer → Manual mode` short-circuits `postprocess_image` while preserving widget state, for iterating on prompt/seed/sampler without ADetailer between every run.
- **Save intermediate steps** — `Settings → ADetailer → Save intermediate steps` writes out the after-each-tab images alongside the final result (`_adetailer_step1.png`, `_adetailer_step2.png`, …).

### Forge Neo compatibility

- `aaaaaa/helper.py`: `disable_safe_unpickle` switched to `patch.object(..., create=True)` so Forge Neo's slimmer `modules.shared.cmd_opts` (which doesn't expose the legacy `disable_safe_unpickle` attribute) no longer crashes ADetailer's model loading.
- `adetailer/classes.py`: `_names_from_json` is tolerant of civitai_helper-style metadata JSON sidecars — when the file shape doesn't look like a class-name container, it returns `[]` so the loader falls back to `model.names` instead of raising.

### UI polish

- Section labels (`.ad-section-label`) in bright white, small uppercase, scoped via CSS.
- Action buttons (`Copy`, `Paste`, preset Load/Save/Rename/Delete/Reset, detection preview) get rounded corners (8px) and `white-space: nowrap` so widths don't double the height on label wrap.
- Version badge overlay (`.ad-version-overlay`) pinned to the top-right of the accordion header — auto-hides when the accordion collapses.
- Top of every tab: `Enable this tab` checkbox + `Copy settings` + `Paste settings` row as direct top-level widgets (no nested accordion).

## 2026-05-15 (fork: class-filtering)

- v26.2.0+classfilter.1
- **Fork only** — per-class filtering for multiclass YOLO detection models.
  - New auto-populated multi-select dropdown `ADetailer detector classes` for non-YOLO-World models. Reads class names from `model.names` or a sidecar `<model>.json`.
  - New `Exclude selected (NOT)` checkbox to invert the filter (inpaint everything except the selected classes).
  - Include path uses Ultralytics' native `model(classes=[ids])` keyword — zero post-processing cost.
  - Exclude path filters `pred[0].boxes.cls` after inference.
  - New Pydantic fields `ad_model_classes_exclude: bool` and `ad_model_classes_excluded: str`; defaults preserve byte-identical infotext for workflows that don't use the feature.
  - YOLO-World text-based class entry is preserved unchanged.
  - MediaPipe models keep all class widgets hidden.
- Design inspired by [wkpark/uddetailer](https://github.com/wkpark/uddetailer); implementation by Claude (Anthropic).

## 2026-02-05

- v26.2.0
- segmentation 모델의 마스크 dtype이 uint8로 변경된 것에 대응

## 2025-03-10

- v25.3.0
- unsafe pickling 방법 변경

## 2024-11-13

- v24.11.1
- `mediapipe_face_mesh`, `mediapipe_face_mesh_eyes_only` 모델에 confidences가 없어 발생하는 에러 수정

## 2024-11-10

- v24.11.0
- `disable_controlnet_units` 함수가 `script_args`의 상태를 변경된 상태로 저장하는 문제 수정
- XYZ Grid에 CFG Scale, scheduler, noise multiplier 추가
- Area 또는 Confidence를 기준으로 마스크 최대 갯수를 지정할 수 있도록 함 (PR #720)

- `ADetailer detector classes`의 element id를 `ad_classes`에서 `ad_model_classes`로 변경
- `mediapipe` 최대 버전을 0.10.15로 제한

## 2024-09-02

- v24.9.0
- Dynamic Denoising, Inpaint bbox sizing 기능 (PR #678)
- `ad_save_images_dir` 옵션 추가 - ad 이미지를 저장하는 장소 지정 (PR #689)

- forge와 관련된 버그 몇 개 수정
- pydantic validation에 실패해도 에러를 일으키지 않고 넘어가도록 수정

## 2024-08-03

- v24.8.0
- 샘플러 선택칸에 Use same sampler 옵션 추가
- 컨트롤넷 유니온 모델을 선택할 수 있게 함

- webui 1.9.0이상에서 기본 스케줄러가 설정되지 않던 문제 수정
- issus #656의 문제 해결을 위해 v24.4.0에 적용되었던 프롬프트 표시 기능을 되돌림
- mediapipe에서 에러가 발생하면 추론이 실패한 것으로 처리하고 조용히 넘어감

## 2024-06-16

- v24.6.0
- webui 1.6.0 미만 버전을 위한 기능들을 제거하고, 최소 버전을 1.6.0으로 올림
- 허깅페이스 연결을 체크하는데 1초만 소요되도록 함
  - 허깅페이스 미러 (hf-mirror.com)도 체크함 (합쳐서 2초)
- InputAccordion을 적용함

## 2024-05-20

- v24.5.1
- uv를 사용하지 않게 함
- 모든 허깅페이스 모델을 동시에 다운로드 시도함
- 기본 탭 수를 2에서 4로 변경

## 2024-05-19

- v24.5.0
- 개별 탭 활성화/비활성화 체크박스 추가
- ad_extra_model_dir 옵션에 |로 구분된 여러 디렉토리를 추가할 수 있게 함 (PR #596)
- `hypertile` 빌트인 확장이 지원되도록 함
- 항상 cond 캐시를 비움
- 설치 스크립트에 uv를 사용함
- mediapipe 최소 버전을 올려 protobuf 버전 4를 사용하게 함

## 2024-04-17

- v24.4.2
- `params.txt` 파일이 없을 때 에러가 발생하지 않도록 수정
- 파이썬 3.9 이하에서 유니온 타입 에러 방지

## 2024-04-14

- v24.4.1
- webui 1.9.0에서 발생한 에러 수정
  - extra generation params에 callable이 들어와서 생긴 문제
  - assign_current_image에 None이 들어갈 수 있던 문제
- webui 1.9.0에서 변경된 scheduler 지원
- 컨트롤넷 모델을 찾을 때, 대소문자 구분을 하지 않음 (PR #577)
- 몇몇 기능을 스크립트에서 분리하여 별도 파일로 빼냄

## 2024-04-10

- v24.4.0
- txt2img에서 hires를 설정했을 때, 이미지의 exif에서 Denoising Strength가 adetailer의 denoisiog stregnth로 덮어 쓰이는 문제 수정
- ad prompt, ad negative prompt에 프롬프트를 변경하는 기능을 적용했을 때(와일드카드 등), 적용된 프롬프트가 이미지의 exif에 제대로 표시됨

## 2024-03-29

- v24.3.5
- 알 수 없는 이유로 인페인팅을 확인하는 과정에서 Txt2Img 인스턴스가 들어오는 문제에 대한 임시 해결

## 2024-03-28

- v24.3.4
- 인페인트에서, 이미지 해상도가 16의 배수가 아닐 때 사이즈 불일치로 인한 opencv 에러 방지

## 2024-03-25

- v24.3.3
- webui 1.6.0 미만 버전에서 create_binary_mask 함수에 대해 ImportError가 발생하는 것 수정

## 2024-03-21

- v24.3.2
- UI를 거치지 않은 입력에 대해, image_mask를 입력했을 때 opencv 에러가 발생하는 것 수정
- img2img inpaint에서 skip img2img 옵션을 활성화할 경우, adetailer를 비활성화함
  - 마스크 크기에 대해 해결하기 힘든 문제가 있음

## 2024-03-16

- v24.3.1
- YOLO World v2, YOLO9 지원가능한 버전으로 ultralytics 업데이트
- inpaint full res인 경우 인페인트 모드에서 동작하게 변경
- inpaint full res가 아닌 경우, 사용자가 입력한 마스크와 교차점이 있는 마스크만 선택하여 사용함

## 2024-03-01

- v24.3.0
- YOLO World 모델 추가: 가장 큰 yolov8x-world.pt 모델만 기본적으로 선택할 수 있게 함.
- lllyasviel/stable-diffusion-webui-forge에서 컨트롤넷을 사용가능하게 함 (PR #517)
- 기본 스크립트 목록에 soft_inpainting 추가 (https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14208)
  - 기존에 설치한 사람에게 소급적용되지는 않음

- 감지모델에 대한 간단한 pytest 추가함
- xyz grid 컨트롤넷 모델 옵션에 `Passthrough` 추가함

## 2024-01-23

- v24.1.2
- controlnet 모델에 `Passthrough` 옵션 추가. 입력으로 들어온 컨트롤넷 옵션을 그대로 사용
- fastapi 엔드포인트 추가

## 2024-01-10

- v24.1.1
- SDNext 호환 업데이트 (issue #466)
  - 설정 값 state에 초기값 추가
  - 위젯 값을 변경할 때마다 state도 변경되게 함 (기존에는 생성 버튼을 누를 때 적용되었음)
- `inpaint_depth_hand` 컨트롤넷 모델이 depth 모델로 인식되게 함 (issue #463)

## 2024-01-04

- v24.1.0
- `depth_hand_refiner` ControlNet 추가 (PR #460)

## 2023-12-30

- v23.12.0
- 파일을 인자로 추가하는 몇몇 스크립트에 대해 deepcopy의 에러를 피하기 위해 script_args 복사 방법을 변경함
- skip img2img 기능을 사용할 때 너비, 높이를 128로 고정하여 스킵 과정이 조금 더 나아짐
- img2img inpainting 모드에서 adetailer 자동 비활성화
- 처음 생성된 params.txt 파일을 항상 유지하도록 변경함

## 2023-11-19

- v23.11.1
- 기본 스크립트 목록에 negpip 추가
  - 기존에 설치한 사람에게 소급적용되지는 않음
- skip img2img 옵션이 2스텝 이상일 때, 제대로 적용되지 않는 문제 수정
- SD.Next에서 이미지가 np.ndarray로 입력되는 경우 수정
- 컨트롤넷 경로를 sys.path에 추가하여 --data-dir등을 지정한 경우에도 임포트 에러가 일어나지 않게 함.

## 2023-10-30

- v23.11.0
- 이미지의 인덱스 계산방법 변경
  - webui 1.1.0 미만에서 adetailer 실행 불가능하게 함
- 컨트롤넷 preprocessor 선택지 늘림
- 추가 yolo 모델 디렉터리를 설정할 수 있는 옵션 추가
- infotext에 `/`가 있는 항목이 exif에서 복원되지 않는 문제 수정
  - 이전 버전에 생성된 이미지는 여전히 복원안됨
- 같은 탭에서 항상 같은 시드를 적용하게 하는 옵션 추가
- 컨트롤넷 1.1.411 (f2aafcf2beb99a03cbdf7db73852228ccd6bd1d6) 버전을 사용중일 경우,
  webui 버전 1.6.0 미만에서 사용할 수 없다는 메세지 출력

## 2023-10-15

- v23.10.1
- xyz grid에 prompt S/R 추가
- img2img에서 steps가 1일때 에러가 발생하는 샘플러의 처리를 위해 샘플러 이름도 변경하게 수정

## 2023-10-07

- v23.10.0
- 허깅페이스 모델을 다운로드 실패했을 때, 계속 다운로드를 시도하지 않음
- img2img에서 img2img단계를 건너뛰는 기능 추가
- live preview에서 감지 단계를 보여줌 (PR #352)

## 2023-09-20

- v23.9.3
- ultralytics 버전 8.0.181로 업데이트 (https://github.com/ultralytics/ultralytics/pull/4891)
- mediapipe와 ultralytics의 lazy import

## 2023-09-10

- v23.9.2
- (실험적) VAE 선택 기능

## 2023-09-01

- v23.9.1
- webui 1.6.0에 추가된 인자를 사용해서 생긴 하위 호환 문제 수정

## 2023-08-31

- v23.9.0
- (실험적) 체크포인트 선택기능
  - 버그가 있어 리프레시 버튼은 구현에서 빠짐
- 1.6.0 업데이트에 따라 img2img에서 사용불가능한 샘플러를 선택했을 때 더이상 Euler로 변경하지 않음
- 유효하지 않은 인자가 전달되었을 때, 에러를 일으키지 않고 대신 adetailer를 비활성화함

## 2023-08-25

- v23.8.1
- xyz grid에서 model을 `None`으로 설정한 이후에 adetailer가 비활성화 되는 문제 수정
- skip을 눌렀을 때 진행을 멈춤
- `--medvram-sdxl`을 설정했을 때에도 cpu를 사용하게 함

## 2023-08-14

- v23.8.0
- `[PROMPT]` 키워드 추가. `ad_prompt` 또는 `ad_negative_prompt`에 사용하면 입력 프롬프트로 대체됨 (PR #243)
- Only top k largest 옵션 추가 (PR #264)
- ultralytics 버전 업데이트

## 2023-07-31

- v23.7.11
- separate clip skip 옵션 추가
- install requirements 정리 (ultralytics 새 버전, mediapipe~=3.20)

## 2023-07-28

- v23.7.10
- ultralytics, mediapipe import문 정리
- traceback에서 컬러를 없앰 (api 때문), 라이브러리 버전도 보여주게 설정.
- huggingface_hub, pydantic을 install.py에서 없앰
- 안쓰는 컨트롤넷 관련 코드 삭제

## 2023-07-23

- v23.7.9
- `ultralytics.utils` ModuleNotFoundError 해결 (https://github.com/ultralytics/ultralytics/issues/3856)
- `pydantic` 2.0 이상 버전 설치안되도록 함
- `controlnet_dir` cmd args 문제 수정 (PR #107)

## 2023-07-20

- v23.7.8
- `paste_field_names` 추가했던 것을 되돌림

## 2023-07-19

- v23.7.7
- 인페인팅 단계에서 별도의 샘플러를 선택할 수 있게 옵션을 추가함 (xyz그리드에도 추가)
- webui 1.0.0-pre 이하 버전에서 batch index 문제 수정
- 스크립트에 `paste_field_names`을 추가함. 사용되는지는 모르겠음

## 2023-07-16

- v23.7.6
- `ultralytics 8.0.135`에 추가된 cpuinfo 기능을 위해 `py-cpuinfo`를 미리 설치하게 함. (미리 설치 안하면 cpu나 mps사용할 때 재시작해야함)
- init_image가 RGB 모드가 아닐 때 RGB로 변경.

## 2023-07-07

- v23.7.4
- batch count > 1일때 프롬프트의 인덱스 문제 수정

- v23.7.5
- i2i의 `cached_uc`와 `cached_c`가 p의 `cached_uc`와 `cached_c`가 다른 인스턴스가 되도록 수정

## 2023-07-05

- v23.7.3
- 버그 수정
  - `object()`가 json 직렬화 안되는 문제
  - `process`를 호출함에 따라 배치 카운트가 2이상일 때, all_prompts가 고정되는 문제
  - `ad-before`와 `ad-preview` 이미지 파일명이 실제 파일명과 다른 문제
  - pydantic 2.0 호환성 문제

## 2023-07-04

- v23.7.2
- `mediapipe_face_mesh_eyes_only` 모델 추가: `mediapipe_face_mesh`로 감지한 뒤 눈만 사용함.
- 매 배치 시작 전에 `scripts.postprocess`를, 후에 `scripts.process`를 호출함.
  - 컨트롤넷을 사용하면 소요 시간이 조금 늘어나지만 몇몇 문제 해결에 도움이 됨.
- `lora_block_weight`를 스크립트 화이트리스트에 추가함.
  - 한번이라도 ADetailer를 사용한 사람은 수동으로 추가해야함.

## 2023-07-03

- v23.7.1
- `process_images`를 진행한 뒤 `StableDiffusionProcessing` 오브젝트의 close를 호출함
- api 호출로 사용했는지 확인하는 속성 추가
- `NansException`이 발생했을 때 중지하지 않고 남은 과정 계속 진행함

## 2023-07-02

- v23.7.0
- `NansException`이 발생하면 로그에 표시하고 원본 이미지를 반환하게 설정
- `rich`를 사용한 에러 트레이싱
  - install.py에 `rich` 추가
- 생성 중에 컴포넌트의 값을 변경하면 args의 값도 함께 변경되는 문제 수정 (issue #180)
- 터미널 로그로 ad_prompt와 ad_negative_prompt에 적용된 실제 프롬프트 확인할 수 있음 (입력과 다를 경우에만)

## 2023-06-28

- v23.6.4
- 최대 모델 수 5 -> 10개
- ad_prompt와 ad_negative_prompt에 빈칸으로 놔두면 입력 프롬프트가 사용된다는 문구 추가
- huggingface 모델 다운로드 실패시 로깅
- 1st 모델이 `None`일 경우 나머지 입력을 무시하던 문제 수정
- `--use-cpu` 에 `adetailer` 입력 시 cpu로 yolo모델을 사용함

## 2023-06-20

- v23.6.3
- 컨트롤넷 inpaint 모델에 대해, 3가지 모듈을 사용할 수 있도록 함
- Noise Multiplier 옵션 추가 (PR #149)
- pydantic 최소 버전 1.10.8로 설정 (Issue #146)

## 2023-06-05

- v23.6.2
- xyz_grid에서 ADetailer를 사용할 수 있게함.
  - 8가지 옵션만 1st 탭에 적용되도록 함.

## 2023-06-01

- v23.6.1
- `inpaint, scribble, lineart, openpose, tile` 5가지 컨트롤넷 모델 지원 (PR #107)
- controlnet guidance start, end 인자 추가 (PR #107)
- `modules.extensions`를 사용하여 컨트롤넷 확장을 불러오고 경로를 알아내로록 변경
- ui에서 컨트롤넷을 별도 함수로 분리

## 2023-05-30

- v23.6.0
- 스크립트의 이름을 `After Detailer`에서 `ADetailer`로 변경
  - API 사용자는 변경 필요함
- 몇몇 설정 변경
  - `ad_conf` → `ad_confidence`. 0~100 사이의 int → 0.0~1.0 사이의 float
  - `ad_inpaint_full_res` → `ad_inpaint_only_masked`
  - `ad_inpaint_full_res_padding` → `ad_inpaint_only_masked_padding`
- mediapipe face mesh 모델 추가
  - mediapipe 최소 버전 `0.10.0`

- rich traceback 제거함
- huggingface 다운로드 실패할 때 에러가 나지 않게 하고 해당 모델을 제거함

## 2023-05-26

- v23.5.19
- 1번째 탭에도 `None` 옵션을 추가함
- api로 ad controlnet model에 inpaint가 아닌 다른 컨트롤넷 모델을 사용하지 못하도록 막음
- adetailer 진행중에 total tqdm 진행바 업데이트를 멈춤
- state.inturrupted 상태에서 adetailer 과정을 중지함
- 컨트롤넷 process를 각 batch가 끝난 순간에만 호출하도록 변경

### 2023-05-25

- v23.5.18
- 컨트롤넷 관련 수정
  - unit의 `input_mode`를 `SIMPLE`로 모두 변경
  - 컨트롤넷 유넷 훅과 하이잭 함수들을 adetailer를 실행할 때에만 되돌리는 기능 추가
  - adetailer 처리가 끝난 뒤 컨트롤넷 스크립트의 process를 다시 진행함. (batch count 2 이상일때의 문제 해결)
- 기본 활성 스크립트 목록에서 컨트롤넷을 뺌

### 2023-05-22

- v23.5.17
- 컨트롤넷 확장이 있으면 컨트롤넷 스크립트를 활성화함. (컨트롤넷 관련 문제 해결)
- 모든 컴포넌트에 elem_id 설정
- ui에 버전을 표시함

### 2023-05-19

- v23.5.16
- 추가한 옵션
  - Mask min/max ratio
  - Mask merge mode
  - Restore faces after ADetailer
- 옵션들을 Accordion으로 묶음

### 2023-05-18

- v23.5.15
- 필요한 것만 임포트하도록 변경 (vae 로딩 오류 없어짐. 로딩 속도 빨라짐)

### 2023-05-17

- v23.5.14
- `[SKIP]`으로 ad prompt 일부를 건너뛰는 기능 추가
- bbox 정렬 옵션 추가
- sd_webui 타입힌트를 만들어냄
- enable checker와 관련된 api 오류 수정?

### 2023-05-15

- v23.5.13
- `[SEP]`으로 ad prompt를 분리하여 적용하는 기능 추가
- enable checker를 다시 pydantic으로 변경함
- ui 관련 함수를 adetailer.ui 폴더로 분리함
- controlnet을 사용할 때 모든 controlnet unit 비활성화
- adetailer 폴더가 없으면 만들게 함

### 2023-05-13

- v23.5.12
- `ad_enable`을 제외한 입력이 dict타입으로 들어오도록 변경
  - web api로 사용할 때에 특히 사용하기 쉬움
  - web api breaking change
- `mask_preprocess` 인자를 넣지 않았던 오류 수정 (PR #47)
- huggingface에서 모델을 다운로드하지 않는 옵션 추가 `--ad-no-huggingface`

### 2023-05-12

- v23.5.11
- `ultralytics` 알람 제거
- 필요없는 exif 인자 더 제거함
- `use separate steps` 옵션 추가
- ui 배치를 조정함

### 2023-05-09

- v23.5.10
- 선택한 스크립트만 ADetailer에 적용하는 옵션 추가, 기본값 `True`. 설정 탭에서 지정가능.
  - 기본값: `dynamic_prompting,dynamic_thresholding,wildcards,wildcard_recursive`
- `person_yolov8s-seg.pt` 모델 추가
- `ultralytics`의 최소 버전을 `8.0.97`로 설정 (C:\\ 문제 해결된 버전)

### 2023-05-08

- v23.5.9
- 2가지 이상의 모델을 사용할 수 있음. 기본값: 2, 최대: 5
- segment 모델을 사용할 수 있게 함. `person_yolov8n-seg.pt` 추가

### 2023-05-07

- v23.5.8
- 프롬프트와 네거티브 프롬프트에 방향키 지원 (PR #24)
- `mask_preprocess`를 추가함. 이전 버전과 시드값이 달라질 가능성 있음!
- 이미지 처리가 일어났을 때에만 before이미지를 저장함
- 설정창의 레이블을 ADetailer 대신 더 적절하게 수정함

### 2023-05-06

- v23.5.7
- `ad_use_cfg_scale` 옵션 추가. cfg 스케일을 따로 사용할지 말지 결정함.
- `ad_enable` 기본값을 `True`에서 `False`로 변경
- `ad_model`의 기본값을 `None`에서 첫번째 모델로 변경
- 최소 2개의 입력(ad_enable, ad_model)만 들어오면 작동하게 변경.

- v23.5.7.post0
- `init_controlnet_ext`을 controlnet_exists == True일때에만 실행
- webui를 C드라이브 바로 밑에 설치한 사람들에게 `ultralytics` 경고 표시

### 2023-05-05 (어린이날)

- v23.5.5
- `Save images before ADetailer` 옵션 추가
- 입력으로 들어온 인자와 ALL_ARGS의 길이가 다르면 에러메세지
- README.md에 설치방법 추가

- v23.5.6
- get_args에서 IndexError가 발생하면 자세한 에러메세지를 볼 수 있음
- AdetailerArgs에 extra_params 내장
- scripts_args를 딥카피함
- postprocess_image를 약간 분리함

- v23.5.6.post0
- `init_controlnet_ext`에서 에러메세지를 자세히 볼 수 있음

### 2023-05-04

- v23.5.4
- use pydantic for arguments validation
- revert: ad_model to `None` as default
- revert: `__future__` imports
- lazily import yolo and mediapipe

### 2023-05-03

- v23.5.3.post0
- remove `__future__` imports
- change to copy scripts and scripts args

- v23.5.3.post1
- change default ad_model from `None`

### 2023-05-02

- v23.5.3
- Remove `None` from model list and add `Enable ADetailer` checkbox.
- install.py `skip_install` fix.
