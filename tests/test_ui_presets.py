"""Callback-level regressions; these do not start a live Gradio/WebUI server."""

import ast
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from adetailer.args import ALL_ARGS
from adetailer.classes import MEDIAPIPE_FACE_FEATURES_MODEL


class Component:
    def click(self, fn, **kwargs):
        self.callback = fn
        self.inputs = kwargs.get("inputs")
        self.outputs = kwargs.get("outputs")
        return self


@pytest.fixture
def callbacks():
    path = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "ordinal", "_copyable_attrs", "_wire_copy_paste", "_wire_presets",
        "_restored_tab_updates", "_sync_class_dropdown", "on_ad_model_update",
        "_do_import", "suffix", "_class_filter_infotext_fields",
        "_skipped_infotext_keys",
    }
    functions = [
        node for node in ast.walk(source)
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", "") == "_CLASS_FILTER_DEFAULTS" for t in node.targets)
        )
    ]
    module = ast.Module(
        body=[source.body[0], *functions], type_ignores=[]
    )
    namespace = {
        "gr": SimpleNamespace(update=lambda **kwargs: kwargs),
        "Path": Path,
        "ALL_ARGS": ALL_ARGS,
        "MEDIAPIPE_FACE_FEATURES_MODEL": MEDIAPIPE_FACE_FEATURES_MODEL,
        "_COPY_EXCLUDE_ATTRS": frozenset(),
        "PRESET_NONE": "(none)",
        "get_model_class_names": lambda path: {
            "faces.pt": ["face", "hand"],
            "animals.pt": ["cat", "dog"],
            MEDIAPIPE_FACE_FEATURES_MODEL: ["eyes", "mouth", "nose"],
        }.get(path, []),
        "get_preset_names": lambda: ["saved"],
        "take_recovery_note": lambda: "",
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def widgets():
    return SimpleNamespace(
        **{attr: Component() for attr in ALL_ARGS.attrs},
        ad_model_classes_dropdown=Component(),
    )


@pytest.mark.parametrize("action", ["load", "paste"])
@pytest.mark.parametrize(
    "scenario",
    [
        ("animals.pt", "cat", False, "", ["cat"], False),
        ("animals.pt", "", True, "dog", ["dog"], False),
        ("custom-world.pt", "red hat,glasses", False, "", [], True),
    ],
)
def test_restore_changes_detector_and_keeps_class_filter(
    callbacks, action, scenario
):
    model, include, exclude, excluded, selected, visible = scenario
    state = {
        "ad_model": model,
        "ad_model_classes": include,
        "ad_model_classes_exclude": exclude,
        "ad_model_classes_excluded": excluded,
    }
    attrs = list(ALL_ARGS.attrs)
    all_widgets = [widgets(), widgets()]
    mapping = {"faces.pt": "faces.pt", "animals.pt": "animals.pt"}
    if action == "paste":
        copy_buttons = [Component(), Component()]
        paste_buttons = [Component(), Component()]
        callbacks["_wire_copy_paste"](
            all_widgets, copy_buttons, paste_buttons, Component(), 2, mapping
        )
        result = paste_buttons[1].callback((0, [state.get(a) for a in attrs]))
    else:
        callbacks["get_preset"] = lambda name: state
        preset_widgets = [tuple(Component() for _ in range(9)) for _ in range(2)]
        callbacks["_wire_presets"](
            all_widgets, preset_widgets, [Component(), Component()], Component(), 2,
            mapping,
        )
        result = preset_widgets[1][1].callback("saved")[1:]

    restored = dict(zip(attrs, result[:-1]))
    assert restored["ad_model"]["value"] == model
    assert restored["ad_model_classes"]["value"] == include
    assert restored["ad_model_classes"]["visible"] is visible
    assert restored["ad_model_classes_exclude"]["value"] is exclude
    assert restored["ad_model_classes_excluded"]["value"] == excluded
    assert result[-1]["value"] == selected
    assert result[-1]["choices"] == ([] if visible else ["cat", "dog"])
    # The existing follow-up class-sync callback must agree with the
    # restored values, including World where it must leave free text alone.
    sync = callbacks["_sync_class_dropdown"](selected, exclude, model)
    assert sync == (({}, {}) if visible else (include, excluded))
    # Gradio .change runs after a programmatic detector update. A selection
    # left from the prior detector must not overwrite the new backing CSV.
    after_change = callbacks["on_ad_model_update"](
        model, ["face"], mapping, current_include=include,
        current_exclude=exclude, current_excluded=excluded,
    )
    assert after_change[0]["value"] == include
    assert after_change[0]["visible"] is visible
    assert after_change[1]["value"] == selected
    assert after_change[2]["value"] is exclude
    assert after_change[3]["value"] == excluded


def test_reset_hides_previous_world_vocabulary_field(callbacks):
    preset_widgets = [tuple(Component() for _ in range(9))]
    callbacks["_wire_presets"](
        [widgets()], preset_widgets, [Component()], Component(), 1
    )
    result = preset_widgets[0][6].callback(False)
    # status, preset choice, name, clipboard and paste button precede args.
    restored = dict(zip(ALL_ARGS.attrs, result[5:-1]))
    assert restored["ad_model_classes"]["value"] == ""
    assert restored["ad_model_classes"]["visible"] is False
    assert result[-1]["value"] == []


def test_reset_every_tab_gives_each_tab_its_own_updates(callbacks):
    # Gradio 4 pops "value" out of an update dict while post-processing it, in
    # place. An update object shared by two tabs therefore reaches the second
    # tab empty, and that tab is silently left unchanged. Every target tab must
    # receive its own update objects, as it did before plus.7.5.
    tabs = 3
    preset_widgets = [tuple(Component() for _ in range(9)) for _ in range(tabs)]
    callbacks["_wire_presets"](
        [widgets() for _ in range(tabs)], preset_widgets,
        [Component() for _ in range(tabs)], Component(), tabs,
    )
    result = preset_widgets[0][6].callback(True)
    count = len(ALL_ARGS.attrs)
    base = 4 * tabs + 1  # status, preset, name and paste per tab + clipboard
    per_tab = [result[base + t * count: base + (t + 1) * count] for t in range(tabs)]
    classes = result[base + tabs * count:]
    assert len(classes) == tabs

    restored = [*(u for tab in per_tab for u in tab), *classes]
    assert len({id(u) for u in restored}) == len(restored)

    # Consume the outputs in order the way Gradio 4 does.
    values = [[u.pop("value", "<missing>") for u in tab] for tab in per_tab]
    class_values = [c.pop("value", "<missing>") for c in classes]
    assert values[1] == values[0]
    assert values[2] == values[0]
    assert values[0][ALL_ARGS.attrs.index("ad_model")] != "<missing>"
    assert class_values == [[], [], []]


def test_mediapipe_face_feature_selection_has_choices(callbacks):
    result = callbacks["on_ad_model_update"](
        MEDIAPIPE_FACE_FEATURES_MODEL, ["eyes"], {}
    )
    assert result[1]["choices"] == ["eyes", "mouth", "nose"]
    assert result[1]["value"] == ["eyes"]


@pytest.mark.parametrize(
    "scenario",
    [
        ("animals.pt", "cat", False, "", ["cat"]),
        ("animals.pt", "", True, "dog", ["dog"]),
        ("custom-world.pt", "red hat,glasses", False, "", []),
    ],
)
def test_png_style_programmatic_values_restore_without_preset_helper(
    callbacks, scenario
):
    model, include, exclude, excluded, selected = scenario
    result = callbacks["on_ad_model_update"](
        model, ["face"], {"animals.pt": "animals.pt"},
        current_include=include, current_exclude=exclude,
        current_excluded=excluded,
    )
    assert result[0]["value"] == include
    assert result[0]["visible"] is ("-world" in model)
    assert result[1]["value"] == selected
    assert result[2]["value"] is exclude
    assert result[3]["value"] == excluded


@pytest.mark.parametrize("upload_kind", ["path", "named_string", "file_wrapper"])
def test_import_accepts_gradio_3_and_4_upload_values(callbacks, tmp_path, upload_kind):
    payload = '{"saved":{"ad_model":"animals.pt"}}'
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=tmp_path, delete=False
    ) as uploaded_file:
        uploaded_file.write(payload)

    class NamedString(str):
        @property
        def name(self):
            return str(self)

    uploaded = {
        "path": uploaded_file.name,
        "named_string": NamedString(uploaded_file.name),
        "file_wrapper": uploaded_file,
    }[upload_kind]
    received = []

    def import_payload(value, *, overwrite):
        received.append((value, overwrite))
        return 1, 0, []

    callbacks["import_presets_json"] = import_payload
    choices, status = callbacks["_do_import"](uploaded, True)
    assert received == [(payload, True)]
    assert choices["choices"] == ["(none)", "saved"]
    assert "added" in status


def _paste(fields, params):
    """AUTOMATIC1111's paste loop for callable infotext keys."""
    out = {}
    for component, key in fields:
        value = key(params)
        out[component] = value
    return out


@pytest.mark.parametrize(
    ("tab", "params", "expected"),
    [
        # NOT mode image pasted onto the same detector: nothing else would
        # update the visible selection, so the paste must.
        (
            0,
            {"ADetailer model": "faces.pt", "ADetailer classes exclude": "True",
             "ADetailer model classes excluded": "hand"},
            ("", True, "hand", ["hand"], ["face", "hand"]),
        ),
        # Include-mode image: its missing exclude key means include mode.
        (
            0,
            {"ADetailer model": "faces.pt", "ADetailer model classes": "face"},
            ("face", False, "", ["face"], ["face", "hand"]),
        ),
        # No class filter at all: every field goes back to its default.
        (
            1,
            {"ADetailer model 2nd": "animals.pt"},
            ("", False, "", [], ["cat", "dog"]),
        ),
        # World vocabulary lives in the free-text field, never the selection.
        (
            0,
            {"ADetailer model": "custom-world.pt",
             "ADetailer model classes": "red hat,glasses"},
            ("red hat,glasses", False, "", [], []),
        ),
    ],
)
def test_png_info_paste_restores_class_filter_and_selection(
    callbacks, tab, params, expected
):
    w = widgets()
    fields = callbacks["_class_filter_infotext_fields"](
        w, tab, {"faces.pt": "faces.pt", "animals.pt": "animals.pt"}
    )
    assert [c for c, _ in fields] == [
        w.ad_model_classes, w.ad_model_classes_exclude,
        w.ad_model_classes_excluded, w.ad_model_classes_dropdown,
    ]
    pasted = _paste(fields, params)
    include, exclude, excluded, selected, choices = expected
    assert pasted[w.ad_model_classes] == include
    assert pasted[w.ad_model_classes_exclude] is exclude
    assert pasted[w.ad_model_classes_excluded] == excluded
    assert pasted[w.ad_model_classes_dropdown]["value"] == selected
    assert pasted[w.ad_model_classes_dropdown]["choices"] == choices


def test_png_info_without_this_tab_leaves_its_class_filter_alone(callbacks):
    w = widgets()
    fields = callbacks["_class_filter_infotext_fields"](w, 1, {})
    pasted = _paste(fields, {"ADetailer model": "faces.pt"})
    assert set(pasted.values()) == {None}


def test_png_info_paste_leaves_fields_the_user_disregards(callbacks, monkeypatch):
    # "Disregard fields from pasted infotext" removes keys before the paste
    # handlers run; those fields must stay as they are, not reset to defaults.
    modules = ModuleType("modules")
    modules.shared = SimpleNamespace(
        opts=SimpleNamespace(infotext_skip_pasting=["ADetailer classes exclude"])
    )
    monkeypatch.setitem(sys.modules, "modules", modules)
    w = widgets()
    fields = callbacks["_class_filter_infotext_fields"](w, 0, {"faces.pt": "faces.pt"})
    pasted = _paste(
        fields, {"ADetailer model": "faces.pt", "ADetailer model classes": "face"}
    )
    assert pasted[w.ad_model_classes] == "face"
    assert pasted[w.ad_model_classes_exclude] is None
    assert pasted[w.ad_model_classes_excluded] == ""
    assert pasted[w.ad_model_classes_dropdown] is None


def test_each_tab_registers_the_class_filter_paste_handlers():
    path = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    group = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "one_ui_group"
    )
    calls = {
        node.func.id
        for node in ast.walk(group)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_class_filter_infotext_fields" in calls
    source = ast.get_source_segment(path.read_text(encoding="utf-8"), group)
    assert "if attr not in _CLASS_FILTER_DEFAULTS" in source
