from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from types import SimpleNamespace
from typing import Any

import gradio as gr

from aaaaaa.conditional import InputAccordion
from adetailer import ADETAILER, __version__
from adetailer.args import ALL_ARGS, MASK_MERGE_INVERT
from adetailer.classes import get_model_class_names
from adetailer.persistence import load_state, save_tab_state
from adetailer.presets import (
    delete_preset,
    get_preset,
    get_preset_names,
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
    model_mapping: dict[str, str] = field(default_factory=dict)


def gr_interactive(value: bool = True):
    return gr.update(interactive=value)


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


def on_generate_click(state: dict, *values: Any, tab_index: int = 0):
    for attr, value in zip(ALL_ARGS.attrs, values):
        state[attr] = value  # noqa: PERF403
    state["is_api"] = ()
    # Best-effort persistence: stash the just-clicked values so they come
    # back as the defaults at next WebUI start. Never raise — see
    # adetailer.persistence for the swallowed-error policy.
    save_tab_state(tab_index, state)
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
    if not model or model == "None" or model.lower().startswith("mediapipe"):
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
    return (
        gr.update(visible=False, value=""),
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
):
    states = []
    infotext_fields = []
    eid = partial(elem_id, n=0, is_img2img=is_img2img)

    # Load per-tab saved state from disk once per UI build. The dict is
    # passed down into one_ui_group so each widget can read its previous
    # value as a default.
    saved_state = load_state()

    with InputAccordion(
        value=False,
        elem_id=eid("ad_main_accordion"),
        label=ADETAILER,
        visible=True,
    ) as ad_enable:
        # Version "about" badge — CSS pulls it out of normal flow and overlays
        # it onto the accordion header. Lives inside the accordion content so
        # it's automatically hidden when the accordion is collapsed.
        gr.Markdown(
            f"v{__version__}",
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

    # components: [bool, bool, dict, dict, ...]
    components = [ad_enable, ad_skip_img2img, *states]
    return components, infotext_fields


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
                label = f"\U0001F4E5 Paste settings from {ordinal(idx + 1)} tab here"
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

    Layout reminder — each entry of `all_presets` is the 8-tuple returned
    from one_ui_group: (dropdown, load_btn, rename_btn, delete_btn,
    name_box, save_btn, reset_btn, status_md).

    Saving / deleting / renaming from any tab refreshes ALL tabs'
    dropdown choices. Load applies a saved preset to THIS tab's widgets
    (including the UI-only classes dropdown). Reset resets THIS tab's
    widgets to their pydantic defaults AND clears the preset+clipboard
    state for a fresh start.
    """
    from adetailer.args import ADetailerArgs

    attrs = list(ALL_ARGS.attrs)
    all_dropdowns = [p[0] for p in all_presets]

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

        # RESET: roll THIS tab back to a pristine state.
        # - All ALL_ARGS widgets (detector, classes, prompts, denoise,
        #   padding, sampler, ControlNet, ...) -> pydantic defaults
        # - UI-only classes multi-select dropdown -> empty
        # - Preset library on this tab: dropdown back to (none), name box
        #   emptied, status cleared
        # - Global clipboard wiped: state -> (-1, []), every paste button
        #   on every tab returns to "📥 Paste settings" disabled
        # Other tabs' widgets are NOT touched.
        def _make_reset():
            def _reset():
                widget_updates = [
                    gr.update(value=_defaults.get(a))
                    if a in _defaults
                    else gr.update()
                    for a in attrs
                ]
                paste_updates = [
                    gr.update(value="\U0001F4E5 Paste settings", interactive=False)
                    for _ in range(num_models)
                ]
                return [
                    "\U0001F195 Tab reset to defaults.",  # status_md
                    gr.update(value=PRESET_NONE),         # this tab's preset dropdown
                    "",                                    # name_box
                    (-1, []),                              # clipboard_state (global)
                    *paste_updates,                        # every paste button (global)
                    *widget_updates,                       # every ALL_ARGS widget
                    gr.update(value=[]),                   # classes multiselect dropdown
                ]

            return _reset

        reset_btn.click(
            fn=_make_reset(),
            inputs=None,
            outputs=[
                status_md,
                dropdown,
                name_box,
                clipboard_state,
                *all_paste_btns,
                *widget_refs,
                dst_classes_dd,
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

    # Preset row: load/save/delete named tab configurations. Shared storage
    # across tabs — saving from one tab makes the preset visible in all
    # tabs' dropdowns. The wiring (which needs refs to every tab's dropdown
    # for cross-tab refresh) is done in _wire_presets() called from adui().
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
    preset_status = gr.Markdown(
        value="",
        elem_id=eid("ad_preset_status"),
        elem_classes=["ad-preset-status"],
    )

    # 'Tab clipboard' accordion grouping the tab-level toggles + Copy/Paste
    # inter-tab clipboard. Closed by default; user opens it when they need
    # to enable/disable the tab or move settings between tabs.
    with gr.Accordion(
        "Tab clipboard",
        open=False,
        elem_id=eid("ad_tab_clipboard_accordion"),
    ):
        with gr.Row(variant="compact"):
            w.ad_tab_enable = gr.Checkbox(
                label=f"Enable this tab ({ordinal(n + 1)})",
                value=sv("ad_tab_enable", True),
                visible=True,
                elem_id=eid("ad_tab_enable"),
            )
        with gr.Row(variant="compact"):
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
                label="ADetailer detector classes (YOLO-World)" + suffix(n),
                value=sv("ad_model_classes", ""),
                visible=False,
                elem_id=eid("ad_model_classes"),
            )
            # UI-only dropdown: not in ALL_ARGS. It syncs into ad_model_classes
            # (CSV) for the include path or ad_model_classes_excluded for exclude.
            w.ad_model_classes_dropdown = gr.Dropdown(
                label="ADetailer detector classes" + suffix(n),
                choices=[],
                value=[],
                multiselect=True,
                visible=True,
                elem_id=eid("ad_model_classes_dropdown"),
            )

        with gr.Row(variant="compact"):
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

        w.ad_model_classes_dropdown.change(
            _sync_dropdown,
            inputs=[w.ad_model_classes_dropdown, w.ad_model_classes_exclude],
            outputs=[w.ad_model_classes, w.ad_model_classes_excluded],
            queue=False,
        )
        w.ad_model_classes_exclude.change(
            _sync_dropdown,
            inputs=[w.ad_model_classes_dropdown, w.ad_model_classes_exclude],
            outputs=[w.ad_model_classes, w.ad_model_classes_excluded],
            queue=False,
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

        with gr.Row(variant="compact"):
            w.ad_use_main_loras = gr.Checkbox(
                label="Use LoRAs from main prompt" + suffix(n),
                value=sv("ad_use_main_loras", False),
                visible=True,
                elem_id=eid("ad_use_main_loras"),
            )

    with gr.Group():
        with gr.Accordion(
            "Detection", open=False, elem_id=eid("ad_detection_accordion")
        ):
            detection(w, n, is_img2img, saved)

        with gr.Accordion(
            "Detection preview (no inpaint)",
            open=False,
            elem_id=eid("ad_preview_accordion"),
        ):
            gr.Markdown(
                "Drop or paste an image, then click the button to run the detector "
                "with the current settings (classes, NOT, confidence) WITHOUT inpainting. "
                "The result shows the detected regions with bounding boxes.",
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
                w.ad_preview_status = gr.Markdown(
                    value="",
                    elem_id=eid("ad_preview_status"),
                    elem_classes=["ad-preset-status"],
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

    # Wire the detection-preview button now that w.ad_confidence and the
    # class-filter widgets all exist. The handler is closed over
    # webui_info.model_mapping so it can resolve the dropdown choice to
    # the on-disk model path.
    _model_mapping = webui_info.model_mapping

    def _run_detection_preview(
        image, model_name, classes_csv, exclude_csv, exclude_mode, confidence
    ):
        if image is None:
            return None, "⚠️ Drop an image into the Input box first."
        if not model_name or model_name == "None":
            return None, "⚠️ Pick a detector model first."
        try:
            if model_name.lower().startswith("mediapipe"):
                from adetailer.mediapipe import mediapipe_predict

                pred = mediapipe_predict(model_name, image, confidence)
            else:
                from adetailer.ultralytics import ultralytics_predict

                path = _model_mapping.get(model_name, "")
                if not path:
                    return None, f"⚠️ Model path not found for '{model_name}'."

                # disable_safe_unpickle is a contextmanager defined in
                # aaaaaa.helper that toggles torch.load weights-only +
                # cmd_opts.disable_safe_unpickle. Imported lazily because
                # the preview module is shared with the standalone Gradio
                # preview which doesn't have modules.shared.
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
                        exclude_classes=(
                            exclude_csv if exclude_mode and exclude_csv else ""
                        ),
                    )
        except Exception as e:  # noqa: BLE001 — surface error to the UI
            return None, f"⚠️ Preview failed: {e}"

        n_detections = len(pred.bboxes) if pred.bboxes else 0
        if pred.preview is None and n_detections == 0:
            return None, "ℹ️ No detections."
        return pred.preview, f"✅ {n_detections} detection(s)."

    w.ad_preview_btn.click(
        fn=_run_detection_preview,
        inputs=[
            w.ad_preview_input,
            w.ad_model,
            w.ad_model_classes,
            w.ad_model_classes_excluded,
            w.ad_model_classes_exclude,
            w.ad_confidence,
        ],
        outputs=[w.ad_preview_output, w.ad_preview_status],
        queue=False,
    )

    state = gr.State(lambda: state_init(w))

    for attr in ALL_ARGS.attrs:
        widget = getattr(w, attr)
        on_change = partial(on_widget_change, attr=attr)
        widget.change(fn=on_change, inputs=[state, widget], outputs=state, queue=False)

    all_inputs = [state, *w.tolist()]
    target_button = webui_info.i2i_button if is_img2img else webui_info.t2i_button
    target_button.click(
        fn=partial(on_generate_click, tab_index=n),
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
    )
    return w, copy_btn, paste_btn, preset_widgets, state, infotext_fields


def detection(
    w: Widgets, n: int, is_img2img: bool, saved: dict[str, Any] | None = None
):
    eid = partial(elem_id, n=n, is_img2img=is_img2img)
    sv = partial(_sv, saved or {})

    with gr.Row():
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
        with gr.Row():
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
        with gr.Row():
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

        with gr.Row():
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

        with gr.Row():
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

            with gr.Row():
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

        with gr.Row():
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

    with gr.Row(variant="panel"):
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
