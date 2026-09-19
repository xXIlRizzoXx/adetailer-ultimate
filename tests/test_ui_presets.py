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
        "_skipped_infotext_keys", "_tab_infotext_fields", "on_cn_model_update",
        "_do_export", "_do_export_status",
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
        "export_presets_json": lambda: '{"saved": {}}',
        "cn_module_choices": {
            "inpaint": ["inpaint_global_harmonious", "inpaint_only", "inpaint_only+lama"],
            "openpose": ["openpose", "openpose_face", "openpose_full"],
        },
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
    # Every tab gets the same values, except "Enable this tab" (only the
    # first tab starts enabled, as on a fresh setup).
    enable = ALL_ARGS.attrs.index("ad_tab_enable")
    assert [tab.pop(enable) for tab in values] == [True, False, False]
    assert values[1] == values[0]
    assert values[2] == values[0]
    assert values[0][ALL_ARGS.attrs.index("ad_model")] != "<missing>"
    assert class_values == [[], [], []]


def test_reset_leaves_extra_tabs_disabled_like_a_fresh_setup(callbacks):
    # A fresh build starts only the first tab enabled. Reset used the schema
    # default (enabled) for every tab, so all extra tabs came back ticked.
    tabs = 3
    preset_widgets = [tuple(Component() for _ in range(9)) for _ in range(tabs)]
    callbacks["_wire_presets"](
        [widgets() for _ in range(tabs)], preset_widgets,
        [Component() for _ in range(tabs)], Component(), tabs,
    )
    count = len(ALL_ARGS.attrs)
    base = 4 * tabs + 1  # status, preset, name and paste per tab + clipboard
    enable = ALL_ARGS.attrs.index("ad_tab_enable")

    def enabled(result):
        return [result[base + t * count + enable].get("value") for t in range(tabs)]

    assert enabled(preset_widgets[0][6].callback(True)) == [True, False, False]
    # Resetting the second tab alone turns it off and leaves the others.
    assert enabled(preset_widgets[1][6].callback(False)) == [None, False, None]


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


def test_import_accepts_a_file_with_a_byte_order_mark(callbacks, tmp_path):
    # Windows editors and PowerShell 5.1 can save UTF-8 with a byte-order
    # mark; the library reader accepts it, so Import must too.
    payload = '{"saved":{"ad_model":"animals.pt"}}'
    uploaded = tmp_path / "presets.json"
    uploaded.write_bytes(b"\xef\xbb\xbf" + payload.encode("utf-8"))
    received = []

    def import_payload(value, *, overwrite):
        received.append(value)
        return 1, 0, []

    callbacks["import_presets_json"] = import_payload
    _choices, status = callbacks["_do_import"](str(uploaded), False)
    assert received == [payload]
    assert "added" in status


def test_export_on_gradio_3_keeps_the_button_label(callbacks, tmp_path, monkeypatch):
    # Gradio 3 has no DownloadButton: a returned path became the plain
    # Button's label and "Exported" was reported although nothing downloaded.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    callbacks["_DownloadButton"] = None
    label, status = callbacks["_do_export"]()
    assert label == {}
    assert list(tmp_path.iterdir()) == []
    assert "Exported" not in status
    assert "Gradio 4" in status


def test_export_on_gradio_4_still_serves_the_file(callbacks, tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    callbacks["_DownloadButton"] = object
    out = tmp_path / "adetailer-ultimate-presets.json"
    path, status = callbacks["_do_export"]()
    assert path == str(out)
    assert out.read_text(encoding="utf-8") == '{"saved": {}}'
    assert "Exported" in status


@pytest.mark.parametrize("gradio_4", [True, False])
def test_export_downloads_the_file_its_click_wrote(gradio_4):
    # Gradio 4's DownloadButton downloads the value it holds when clicked,
    # before the server has written this click's file: the first Export got
    # nothing (while the status said it worked) and later ones the previous
    # file. A front-end step now downloads the new value and clears it.
    path = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    click = next(
        node for node in calls
        if node.func.attr == "click"
        and getattr(node.func.value, "id", "") == "preset_export_btn"
    )
    then = next(
        node for node in calls if node.func.attr == "then" and node.func.value is click
    )
    outputs = next(k.value for k in click.keywords if k.arg == "outputs")
    assert [e.id for e in outputs.elts] == ["preset_export_btn", "preset_io_status"]
    assert [ast.unparse(k) for k in then.keywords] == [
        "queue=False", "**_export_then"
    ]
    assign = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", "") == "_export_then"
    )
    namespace = {
        "preset_export_btn": "button",
        "_EXPORT_JS": "js",
        "_DownloadButton": object if gradio_4 else None,
    }
    kwargs = eval(compile(ast.Expression(assign.value), str(path), "eval"), namespace)
    if gradio_4:
        assert kwargs == {
            "fn": None, "inputs": "button", "outputs": "button", "js": "js"
        }
    else:
        # Gradio 3 names it `_js` and rejects `js`; nothing to download there.
        assert kwargs == {"fn": None}


CN_INPAINT = "control_v11p_sd15_inpaint [ebff9138]"


@pytest.mark.parametrize(
    ("model", "module", "expected"),
    [
        # Load / Paste / infotext paste set model and module together; the
        # model's change handler runs afterwards and must keep the module.
        (CN_INPAINT, "inpaint_only+lama", "inpaint_only+lama"),
        # A module that does not fit the new model falls back to the first.
        ("control_v11p_sd15_openpose [cab727d4]", "inpaint_only+lama", "openpose"),
        (CN_INPAINT, None, "inpaint_global_harmonious"),
    ],
)
def test_controlnet_model_change_keeps_a_fitting_module(
    callbacks, model, module, expected
):
    update = callbacks["on_cn_model_update"](model, module)
    assert update["visible"] is True
    assert update["value"] == expected


def test_controlnet_model_none_hides_the_module(callbacks):
    update = callbacks["on_cn_model_update"]("None", "inpaint_only")
    assert (update["visible"], update["value"]) == (False, "None")


def test_controlnet_model_change_reads_the_current_module():
    path = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "change"
        and node.args
        and getattr(node.args[0], "id", "") == "on_cn_model_update"
    ]
    assert len(calls) == 1
    inputs = next(k.value for k in calls[0].keywords if k.arg == "inputs")
    assert isinstance(inputs, ast.List)
    assert [e.attr for e in inputs.elts] == [
        "ad_controlnet_model", "ad_controlnet_module"
    ]


class Block:
    """A Gradio block that records its settings; also a no-op layout."""

    def __init__(self, *_args, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def change(self, *_args, **_kwargs):
        return self


@pytest.mark.parametrize(
    ("saved", "expected"),
    [
        # A restored model shows its module with the saved choice.
        (
            {"ad_controlnet_model": CN_INPAINT, "ad_controlnet_module": "inpaint_only+lama"},
            (True, "inpaint_only+lama"),
        ),
        # A saved module that does not fit falls back to the model's first.
        (
            {"ad_controlnet_model": CN_INPAINT, "ad_controlnet_module": "openpose"},
            (True, "inpaint_global_harmonious"),
        ),
        # No model, or one that is gone: hidden as before, value untouched.
        ({"ad_controlnet_model": "None", "ad_controlnet_module": "inpaint_only"},
         (False, "inpaint_only")),
        ({"ad_controlnet_model": "deleted_model", "ad_controlnet_module": "x"},
         (False, "x")),
    ],
)
def test_restored_controlnet_model_shows_its_module_at_startup(saved, expected):
    # .change does not fire at the first render, so after a restart the
    # restored model's module dropdown stayed hidden with only "None" in it.
    from functools import partial

    path = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    names = {"controlnet", "on_cn_model_update", "elem_id", "_sv", "suffix", "ordinal"}
    nodes = [
        node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    choices = ["inpaint_global_harmonious", "inpaint_only", "inpaint_only+lama"]
    namespace = {
        "gr": SimpleNamespace(
            update=lambda **kwargs: kwargs, Row=Block, Column=Block,
            Dropdown=Block, Slider=Block,
        ),
        "partial": partial,
        "get_cn_models": lambda: [CN_INPAINT],
        "controlnet_exists": True,
        "cn_module_choices": {"inpaint": choices},
    }
    module = ast.Module(body=[source.body[0], *nodes], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    w = SimpleNamespace()

    namespace["controlnet"](w, 0, False, saved)

    visible, value = expected
    dropdown = w.ad_controlnet_module.kwargs
    assert (dropdown["visible"], dropdown["value"]) == (visible, value)
    assert dropdown["choices"] == (choices if visible else ["None"])


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
    assert "_tab_infotext_fields" in calls
    source = ast.get_source_segment(path.read_text(encoding="utf-8"), group)
    assert "if attr not in _CLASS_FILTER_DEFAULTS" in source
    # Each widget is pasted once: by its callable handler, not its label too.
    assert "and attr not in _TAB_PASTE_ATTRS" in source
    attrs = next(
        node.value for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", "") == "_TAB_PASTE_ATTRS" for t in node.targets)
    )
    assert ast.literal_eval(attrs) == ("ad_tab_enable", "ad_inpaint_indices")


@pytest.mark.parametrize(
    ("tab", "params", "expected"),
    [
        # Tab 2 is in the image: it was enabled, and no indices means none.
        (1, {"ADetailer model": "face.pt", "ADetailer model 2nd": "hand.pt"}, (True, "")),
        (
            1,
            {"ADetailer model 2nd": "hand.pt", "ADetailer inpaint indices 2nd": "1,3"},
            (True, "1,3"),
        ),
        (0, {"ADetailer model": "face.pt"}, (True, "")),
        # Tab 3 is not in the image: leave it as it is.
        (2, {"ADetailer model": "face.pt", "ADetailer model 2nd": "hand.pt"}, (None, None)),
        (1, {"ADetailer model": "face.pt"}, (None, None)),
        # An image made with ADetailer lists every tab that ran: a tab that
        # is not in it did not run, so it is switched off.
        (1, {"ADetailer model": "face.pt", "ADetailer version": "26.3.0"}, (False, None)),
        (
            0,
            {"ADetailer model 2nd": "hand.pt", "ADetailer version": "26.3.0"},
            (False, None),
        ),
        # Parameters without ADetailer leave every tab as it is.
        (1, {"Steps": "20"}, (None, None)),
    ],
)
def test_png_info_paste_enables_the_tab_and_clears_old_indices(
    callbacks, tab, params, expected
):
    w = widgets()
    fields = callbacks["_tab_infotext_fields"](w, tab)
    assert [c for c, _ in fields] == [w.ad_tab_enable, w.ad_inpaint_indices]
    pasted = _paste(fields, params)
    assert (pasted[w.ad_tab_enable], pasted[w.ad_inpaint_indices]) == expected


def test_png_info_paste_of_a_two_tab_image_runs_the_second_pass(callbacks):
    # Real infotext for an image made with two enabled tabs. It never lists
    # "tab enable" and leaves out empty indices, and the WebUI keeps a field
    # whose key is missing, so tab 2 (off by default) stayed off and an old
    # "2" in its indices field stayed in place.
    from adetailer.args import ADetailerArgs

    params = {
        **ADetailerArgs(ad_model="face.pt").extra_params(),
        **ADetailerArgs(ad_model="hand.pt", ad_prompt="detailed hand").extra_params(
            suffix=" 2nd"
        ),
    }
    params = {k: str(v) for k, v in params.items()}  # parsed from text
    assert "ADetailer tab enable 2nd" not in params
    assert "ADetailer inpaint indices 2nd" not in params

    w = widgets()
    fields = [
        (getattr(w, attr), name + " 2nd")
        for attr, name in ALL_ARGS
        if attr in ("ad_model", "ad_prompt")
    ]
    fields.extend(callbacks["_tab_infotext_fields"](w, 1))
    pasted = _paste(
        [(c, k if callable(k) else (lambda p, k=k: p.get(k))) for c, k in fields],
        params,
    )
    tab = {"ad_tab_enable": False, "ad_inpaint_indices": "2"}  # before pasting
    tab.update(
        (attr, pasted[getattr(w, attr)])
        for attr in ("ad_model", "ad_prompt", "ad_tab_enable", "ad_inpaint_indices")
        if pasted[getattr(w, attr)] is not None
    )
    args = ADetailerArgs(**tab)
    assert args.ad_model == "hand.pt"
    assert not args.need_skip()
    assert args.ad_inpaint_indices == ""


def test_png_info_paste_leaves_tab_fields_the_user_disregards(callbacks, monkeypatch):
    modules = ModuleType("modules")
    modules.shared = SimpleNamespace(
        opts=SimpleNamespace(infotext_skip_pasting=["ADetailer inpaint indices 2nd"])
    )
    monkeypatch.setitem(sys.modules, "modules", modules)
    w = widgets()
    pasted = _paste(
        callbacks["_tab_infotext_fields"](w, 1), {"ADetailer model 2nd": "hand.pt"}
    )
    assert pasted[w.ad_tab_enable] is True
    assert pasted[w.ad_inpaint_indices] is None


@pytest.mark.parametrize(
    "skipped", [["ADetailer model 2nd"], ["ADetailer tab enable 2nd"]]
)
def test_png_info_paste_keeps_a_tab_whose_detector_the_user_disregards(
    callbacks, monkeypatch, skipped
):
    # A disregarded detector key is removed before pasting: its absence does
    # not mean that the tab did not run.
    modules = ModuleType("modules")
    modules.shared = SimpleNamespace(
        opts=SimpleNamespace(infotext_skip_pasting=skipped)
    )
    monkeypatch.setitem(sys.modules, "modules", modules)
    w = widgets()
    pasted = _paste(
        callbacks["_tab_infotext_fields"](w, 1),
        {"ADetailer model": "face.pt", "ADetailer version": "26.3.0"},
    )
    assert pasted[w.ad_tab_enable] is None


def test_the_master_switch_is_pasted_from_its_own_key():
    # The infotext_pasted callback in scripts/!adetailer.py adds "ADetailer
    # enable" to parameters with a detector; the switch must keep reading it
    # (a string key also keeps it in "Disregard fields from pasted infotext").
    path = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    source = path.read_text(encoding="utf-8")
    assert 'infotext_fields.append((ad_enable, "ADetailer enable"))' in source


# YOLO-World detects the typed classes and has no NOT mode: its "Exclude
# selected (NOT)" checkbox must be hidden and off, never a visible promise of
# an inversion that does not happen. Other detectors keep NOT mode.
_WORLD_NOT = {
    "ad_model": "custom-world.pt",
    "ad_model_classes": "person",
    "ad_model_classes_exclude": True,
    "ad_model_classes_excluded": "hand",
}
_FIXED_NOT = {
    "ad_model": "animals.pt",
    "ad_model_classes": "",
    "ad_model_classes_exclude": True,
    "ad_model_classes_excluded": "dog",
}


@pytest.mark.parametrize(
    ("state", "shown"), [(_WORLD_NOT, False), (_FIXED_NOT, True)]
)
def test_detector_change_hides_not_mode_for_yolo_world(callbacks, state, shown):
    result = callbacks["on_ad_model_update"](
        state["ad_model"], [], {"animals.pt": "animals.pt"},
        current_include=state["ad_model_classes"],
        current_exclude=True,
        current_excluded=state["ad_model_classes_excluded"],
    )
    assert result[2] == {"visible": shown, "value": shown}
    # With NOT mode off, a kept excluded list went into the image's
    # parameters ("classes excluded: hand") and the preset preview.
    assert result[3]["value"] == (state["ad_model_classes_excluded"] if shown else "")
    if not shown:
        assert result[0]["value"] == "person"  # the typed vocabulary is kept


@pytest.mark.parametrize(
    ("state", "shown"), [(_WORLD_NOT, False), (_FIXED_NOT, True)]
)
def test_load_and_paste_hide_not_mode_for_yolo_world(callbacks, state, shown):
    attrs = list(ALL_ARGS.attrs)
    restored = dict(
        zip(
            attrs,
            callbacks["_restored_tab_updates"](
                state, attrs, {"animals.pt": "animals.pt"}
            )[:-1],
        )
    )
    assert restored["ad_model_classes_exclude"] == {"visible": shown, "value": shown}
    assert restored["ad_model_classes_excluded"] == {
        "value": state["ad_model_classes_excluded"] if shown else ""
    }


@pytest.mark.parametrize(
    ("model", "expected"), [("custom-world.pt", False), ("animals.pt", True)]
)
def test_png_info_paste_leaves_not_mode_off_for_yolo_world(callbacks, model, expected):
    w = widgets()
    fields = callbacks["_class_filter_infotext_fields"](w, 0, {"animals.pt": "animals.pt"})
    pasted = _paste(
        fields,
        {"ADetailer model": model, "ADetailer classes exclude": "True",
         "ADetailer model classes": "person",
         "ADetailer model classes excluded": "hand"},
    )
    assert pasted[w.ad_model_classes_exclude] is expected
    assert pasted[w.ad_model_classes_excluded] == ("hand" if expected else "")


def test_yolo_world_switch_leaves_no_excluded_classes_in_the_parameters(callbacks):
    from adetailer.args import ADetailerArgs

    result = callbacks["on_ad_model_update"](
        "custom-world.pt", ["hand"], {},
        current_include="", current_exclude=True, current_excluded="hand",
    )
    args = ADetailerArgs(
        ad_model="custom-world.pt", ad_model_classes="person,cat",
        ad_model_classes_exclude=result[2]["value"],
        ad_model_classes_excluded=result[3]["value"],
    )
    assert args.extra_params()["ADetailer model classes"] == "person,cat"
    assert "ADetailer model classes excluded" not in args.extra_params()


@pytest.mark.parametrize(("world", "shown"), [(True, False), (False, True)])
def test_saved_yolo_world_tab_starts_with_not_mode_hidden(world, shown):
    path = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    checkbox = next(
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "attr", "") == "ad_model_classes_exclude"
    )
    keywords = {k.arg: k.value for k in checkbox.keywords}
    namespace = {"sv": lambda attr, default: True, "_is_world_saved": world}

    def evaluate(node):
        return eval(compile(ast.Expression(node), str(path), "eval"), namespace)

    assert evaluate(keywords["visible"]) is shown
    assert evaluate(keywords["value"]) is shown


@pytest.mark.parametrize(("world", "expected"), [(True, ""), (False, "hand")])
def test_saved_yolo_world_tab_starts_without_excluded_classes(world, expected):
    # Like its hidden NOT checkbox: an excluded list saved with a YOLO-World
    # detector would otherwise go into the next image's parameters.
    path = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    textbox = next(
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "attr", "") == "ad_model_classes_excluded"
    )
    keywords = {k.arg: k.value for k in textbox.keywords}
    namespace = {"sv": lambda attr, default: "hand", "_is_world_saved": world}
    value = eval(compile(ast.Expression(keywords["value"]), str(path), "eval"), namespace)
    assert value == expected
