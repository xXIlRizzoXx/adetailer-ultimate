"""Callback-level regressions; these do not start a live Gradio/WebUI server."""

import ast
import gc
import sys
import tempfile
import weakref
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
        "_do_export", "_do_export_status", "_format_preset_preview",
        "_sync_preset_choices",
    }
    functions = [
        node for node in ast.walk(source)
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                getattr(t, "id", "") in {"_CLASS_FILTER_DEFAULTS", "_PRESET_DROPDOWNS"}
                for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
            )
        )
    ]
    module = ast.Module(
        body=[source.body[0], *functions], type_ignores=[]
    )
    namespace = {
        "gr": SimpleNamespace(update=lambda **kwargs: kwargs),
        "Path": Path,
        "weakref": weakref,
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
    mapping = {
        "faces.pt": "faces.pt", "animals.pt": "animals.pt",
        "custom-world.pt": "custom-world.pt",
    }
    if action == "paste":
        copy_buttons = [Component(), Component()]
        paste_buttons = [Component(), Component()]
        callbacks["_wire_copy_paste"](
            all_widgets, copy_buttons, paste_buttons, Component(), 2, mapping
        )
        result = paste_buttons[1].callback((0, [state.get(a) for a in attrs]))
    else:
        callbacks["get_preset"] = lambda name: state
        preset_widgets = [tuple(Component() for _ in range(10)) for _ in range(2)]
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
    preset_widgets = [tuple(Component() for _ in range(10))]
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
    # receive its own update objects, as it did before plus.8.
    tabs = 3
    preset_widgets = [tuple(Component() for _ in range(10)) for _ in range(tabs)]
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
    preset_widgets = [tuple(Component() for _ in range(10)) for _ in range(tabs)]
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


@pytest.mark.parametrize("reset_all", [False, True])
def test_reset_puts_override_dropdowns_back_on_their_first_choice(callbacks, reset_all):
    # Their schema default is None, which is not one of their choices, so
    # Reset left the separate checkpoint, VAE and text encoder dropdowns blank.
    # The sampler's schema default "DPM++ 2M Karras" is no sampler name on
    # WebUI 1.9+ (the scheduler is a setting of its own): Gradio 4 showed the
    # dropdown blank. A fresh build shows the first sampler.
    tabs = 2
    preset_widgets = [tuple(Component() for _ in range(10)) for _ in range(tabs)]
    callbacks["_wire_presets"](
        [widgets() for _ in range(tabs)], preset_widgets,
        [Component() for _ in range(tabs)], Component(), tabs,
        first_sampler="DPM++ 2M",
    )
    result = preset_widgets[0][6].callback(reset_all)
    count = len(ALL_ARGS.attrs)
    base = 4 * tabs + 1  # status, preset, name and paste per tab + clipboard
    first = {
        "ad_checkpoint": "Use same checkpoint",
        "ad_vae": "Use same VAE",
        "ad_text_encoder": "Use same text encoder",
        "ad_sampler": "DPM++ 2M",
    }
    for t in range(tabs if reset_all else 1):
        restored = dict(zip(ALL_ARGS.attrs, result[base + t * count : base + (t + 1) * count]))
        for attr, value in first.items():
            assert restored[attr]["value"] == value
        assert restored["ad_use_checkpoint"]["value"] is False
        assert restored["ad_use_sampler"]["value"] is False
        assert restored["ad_scheduler"]["value"] == "Use same scheduler"
    if not reset_all:
        # The tab this reset leaves out is untouched.
        assert result[base + count + ALL_ARGS.attrs.index("ad_sampler")] == {}
    assert not any(isinstance(u, dict) and u.get("value", "") is None for u in result)
    # These are the first choices a fresh build gives the three dropdowns.
    path = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lists = {
        node.targets[0].id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.List)
        and getattr(node.targets[0], "id", "") in ("ckpts", "vaes", "tes")
    }
    assert [ast.literal_eval(lists[n].elts[0]) for n in ("ckpts", "vaes", "tes")] == [
        first[a] for a in ("ad_checkpoint", "ad_vae", "ad_text_encoder")
    ]


def test_reset_is_given_the_sampler_a_fresh_build_shows():
    # The fixture never runs adui: check that it hands _wire_presets the first
    # real sampler, the one the tab's sampler dropdown shows on a fresh build.
    path = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    adui = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "adui"
    )
    call = next(
        node for node in ast.walk(adui)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_wire_presets"
    )
    first = next(k.value for k in call.keywords if k.arg == "first_sampler")
    for sampler_names, expected in ((["DPM++ 2M", "Euler a"], "DPM++ 2M"), ([], None)):
        webui_info = SimpleNamespace(sampler_names=sampler_names)
        assert eval(compile(ast.Expression(first), str(path), "eval"), {"webui_info": webui_info}) == expected
    built = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "inpainting"
    )
    built = ast.unparse(built).replace("'", '"')
    assert '["Use same sampler", *webui_info.sampler_names]' in built
    assert 'sv("ad_sampler", sampler_names[1])' in built


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
    choices, status, _preview = callbacks["_do_import"](uploaded, True)
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
    _choices, status, _preview = callbacks["_do_import"](str(uploaded), False)
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


def test_only_the_gradio_3_export_button_is_marked_for_its_tooltip():
    # The tooltip promised a JSON download that the Gradio 3 fallback button
    # cannot give; button-tooltips.js tells it apart by this class.
    path = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    branch = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "_DownloadButton is not None"
        and any("ad_preset_export_btn" in ast.unparse(n) for n in node.body)
    )

    def classes(nodes):
        calls = [n for s in nodes for n in ast.walk(s) if isinstance(n, ast.Call)]
        call = next(c for c in calls if "ad_preset_export_btn" in ast.unparse(c))
        kw = {k.arg: k.value for k in call.keywords}
        return ast.literal_eval(kw["elem_classes"]) if "elem_classes" in kw else []

    assert "ad-export-unavailable" in classes(branch.orelse)
    assert "ad-export-unavailable" not in classes(branch.body)


def test_hires_only_help_says_the_tab_is_skipped_without_hires_fix():
    # The help text said the option has no effect when hires.fix is off, but
    # the tab is then skipped; it also named a pre-hires call no WebUI makes.
    path = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and any(
            k.arg == "elem_id" and "ad_apply_on_hires_only" in ast.unparse(k.value)
            for k in node.keywords
        )
    )
    info = ast.literal_eval(next(k.value for k in call.keywords if k.arg == "info"))
    assert "skipped" in info
    assert "when hires.fix is off" not in info
    assert "pre-hires" not in info
    assert not any(mark in info for mark in ("**", "`", "["))


def test_manual_mode_help_points_to_run_adetailer_on_an_image():
    # The Settings help text only offered Detection preview plus a round trip
    # through img2img with the option turned off, although "Run ADetailer on
    # an image" details a finished image while manual mode is on.
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "scripts" / "!adetailer.py").read_text(encoding="utf-8"))
    option = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_option"
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "ad_manual_mode"
    )
    assert option.args[1].func.attr == "info"
    info = ast.literal_eval(option.args[1].args[0])
    assert "'Run ADetailer on an image'" in info
    assert "Detection preview" not in info
    assert not any(mark in info for mark in ("**", "`", "["))
    # The section the text names still exists.
    ui = (root / "aaaaaa" / "ui.py").read_text(encoding="utf-8")
    assert '"Run ADetailer on an image",' in ui


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


@pytest.mark.parametrize(
    ("model", "module", "expected"),
    [
        ("OpenPoseXL2 [9a0c1e]", None, "openpose"),
        ("OpenPoseXL2 [9a0c1e]", "openpose_full", "openpose_full"),
        ("controlnetxlCNXL_sdxlOpenpose", "None", "openpose"),
        ("Controlnet_Tile_Realistic_v2_fp16", None, "tile_resample"),
        ("control_sd15_Inpaint_Depth_Hand_fp16", None, "depth_midas"),
        ("control_sd15_inpaint_depth_hand_fp16", None, "depth_midas"),
        ("Anime_Inpaint", "inpaint_only", "inpaint_only"),
        ("Passthrough", "openpose", "None"),
    ],
)
def test_controlnet_model_matches_its_type_in_any_case(
    callbacks, model, module, expected
):
    # The model list accepts a capitalised type (OpenPoseXL2, ..._Tile_...)
    # in any case, but the preprocessor was matched case-sensitively: the
    # module dropdown was hidden on "None", so the model got the raw crop.
    choices = callbacks["cn_module_choices"]
    choices["tile"] = ["tile_resample", "tile_colorfix"]
    choices["depth"] = ["depth_midas", "depth_hand_refiner"]
    update = callbacks["on_cn_model_update"](model, module)
    assert update["value"] == expected
    assert update["visible"] is (expected != "None")


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


@pytest.mark.parametrize(
    ("saved", "schedulers", "expected"),
    [
        # A ticked old name is split by the scheduler's label: the WebUI ran it
        # as the sampler plus that scheduler, and a restart must keep it so.
        ((True, "DPM++ 2M Karras", "Use same scheduler"), None, ("DPM++ 2M", "Karras")),
        ((True, "DPM++ 2M SDE Exponential", "Uniform"), None,
         ("DPM++ 2M SDE", "Exponential")),
        # Everything else is restored as before.
        ((False, "DPM++ 2M Karras", "Use same scheduler"), None,
         ("DPM++ 2M", "Use same scheduler")),
        ((True, "Foo Karras", "Use same scheduler"), None,
         ("DPM++ 2M", "Use same scheduler")),
        ((True, "Euler a", "Karras"), None, ("Euler a", "Karras")),
        ((True, "Use same sampler", "Karras"), None, ("Use same sampler", "Karras")),
        # Another WebUI's sampler: the first sampler, as Load shows it.
        ((True, "Res Multistep", "Karras"), None, ("DPM++ 2M", "Karras")),
        # Only Load also splits a scheduler's other names (a known issue).
        ((True, "DPM++ 2M karras", "Uniform"), None, ("DPM++ 2M", "Uniform")),
        # WebUI < 1.9 has no schedulers: nothing to split.
        ((True, "DPM++ 2M Karras", "Use same scheduler"), [],
         ("DPM++ 2M", "Use same scheduler")),
    ],
)
def test_restart_splits_a_ticked_old_sampler_name(saved, schedulers, expected):
    # Remembered settings could hold a ticked "DPM++ 2M Karras" (an older
    # preset or Reset, a pasted image). At the next start the name is not a
    # choice and fell back to the first sampler with the saved scheduler:
    # the tab silently ran DPM++ 2M with the main pass's scheduler.
    from functools import partial

    path = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    names = {"inpainting", "elem_id", "_sv", "suffix", "ordinal", "gr_interactive"}
    nodes = [
        node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "gr": SimpleNamespace(
            update=lambda **kwargs: kwargs, Row=Block, Column=Block, Group=Block,
            Dropdown=Block, Slider=Block, Checkbox=Block,
        ),
        "partial": partial,
    }
    module = ast.Module(body=[source.body[0], *nodes], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    webui_info = SimpleNamespace(
        sampler_names=["DPM++ 2M", "DPM++ 2M SDE", "Euler a"],
        scheduler_names=(
            ["Automatic", "Uniform", "Karras", "Exponential"]
            if schedulers is None
            else schedulers
        ),
        checkpoints_list=[], vae_list=[], text_encoders_list=[],
    )
    use, sampler, scheduler = saved
    w = SimpleNamespace()

    namespace["inpainting"](
        w, 0, False, webui_info,
        {"ad_use_sampler": use, "ad_sampler": sampler, "ad_scheduler": scheduler},
    )

    assert (w.ad_sampler.kwargs["value"], w.ad_scheduler.kwargs["value"]) == expected


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


def test_load_resets_settings_an_older_preset_does_not_have(callbacks):
    # A preset saved by an older version lacks the settings added since. They
    # kept the tab's current values, so a leftover detection-number filter
    # ("2") inpainted only that detection after loading the preset.
    callbacks["get_preset"] = lambda name: {
        "ad_model": "faces.pt", "ad_prompt": "old preset", "ad_model_classes": "face",
    }
    preset_widgets = [tuple(Component() for _ in range(10))]
    callbacks["_wire_presets"](
        [widgets()], preset_widgets, [Component()], Component(), 1,
        {"faces.pt": "faces.pt"},
    )
    result = preset_widgets[0][1].callback("saved")[1:]
    restored = dict(zip(ALL_ARGS.attrs, result[:-1]))
    assert restored["ad_inpaint_indices"]["value"] == ""
    assert restored["ad_detection_resolution"]["value"] == 0
    assert restored["ad_class_guard"]["value"] is False
    assert restored["ad_confidence"]["value"] == 0.3
    assert restored["ad_prompt"]["value"] == "old preset"
    assert restored["ad_model_classes"]["value"] == "face"
    assert result[-1]["value"] == ["face"]
    # "Enable this tab" is not in the preset, so it is left as it is.
    assert "value" not in restored["ad_tab_enable"]


def test_load_keeps_an_override_choice_an_older_preset_does_not_have(callbacks):
    # Presets saved before the per-pass text encoder existed have no
    # ad_text_encoder. Its schema default is None, which is not one of the
    # dropdown's choices, so Load left the dropdown blank.
    callbacks["get_preset"] = lambda name: {
        "ad_model": "faces.pt",
        "ad_checkpoint": "Use same checkpoint",
        "ad_vae": "Use same VAE",
    }
    preset_widgets = [tuple(Component() for _ in range(10))]
    callbacks["_wire_presets"](
        [widgets()], preset_widgets, [Component()], Component(), 1,
        {"faces.pt": "faces.pt"},
    )
    result = preset_widgets[0][1].callback("saved")[1:]
    restored = dict(zip(ALL_ARGS.attrs, result[:-1]))
    assert "value" not in restored["ad_text_encoder"]  # the tab keeps its choice
    assert restored["ad_use_text_encoder"]["value"] is False
    assert restored["ad_checkpoint"]["value"] == "Use same checkpoint"
    assert not any(
        "value" in u and u["value"] is None for u in result if isinstance(u, dict)
    )


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ({"ad_model": "faces.pt", "ad_prompt": "old preset"}, "DPM++ 2M"),
        ({"ad_model": "faces.pt", "ad_sampler": "Euler a"}, "Euler a"),
    ],
)
def test_load_gives_a_preset_without_a_sampler_the_first_sampler(
    callbacks, preset, expected
):
    # The sampler's schema default ("DPM++ 2M Karras") is not a sampler name
    # since the WebUI split out the scheduler, so a preset without a sampler
    # left the dropdown on a choice that does not exist. A preset that names
    # its sampler keeps it.
    callbacks["get_preset"] = lambda name: dict(preset)
    preset_widgets = [tuple(Component() for _ in range(10))]
    callbacks["_wire_presets"](
        [widgets()], preset_widgets, [Component()], Component(), 1,
        {"faces.pt": "faces.pt"}, first_sampler="DPM++ 2M",
    )
    result = preset_widgets[0][1].callback("saved")[1:]
    restored = dict(zip(ALL_ARGS.attrs, result[:-1]))
    assert restored["ad_sampler"]["value"] == expected
    assert restored["ad_use_sampler"]["value"] is False


@pytest.mark.parametrize(
    ("sampler", "use_sampler", "names", "expected"),
    [
        # What Reset set in plus.7.x and beta 1: no sampler on WebUI 1.9+.
        ("DPM++ 2M Karras", False, ["DPM++ 2M", "Euler a"], "DPM++ 2M"),
        ("Euler a", False, ["DPM++ 2M", "Euler a"], "Euler a"),
        ("Use same sampler", False, ["DPM++ 2M", "Euler a"], "Use same sampler"),
        # A real sampler on a WebUI before 1.9.
        ("DPM++ 2M Karras", False, ["Euler a", "DPM++ 2M Karras"], "DPM++ 2M Karras"),
        # Used: the WebUI runs it as DPM++ 2M with the Karras scheduler.
        ("DPM++ 2M Karras", True, ["DPM++ 2M", "Euler a"], "DPM++ 2M Karras"),
    ],
)
def test_load_shows_a_preset_saved_after_an_older_reset_as_a_fresh_setup(
    callbacks, sampler, use_sampler, names, expected
):
    # Reset in plus.7.x and beta 1 set the schema defaults, and a preset saved
    # right afterwards kept them: Load then blanked the separate checkpoint,
    # VAE and text encoder dropdowns and put the sampler on a non-choice.
    callbacks["get_preset"] = lambda name: {
        "ad_model": "faces.pt",
        "ad_use_checkpoint": False,
        "ad_checkpoint": None,
        "ad_use_vae": False,
        "ad_vae": None,
        "ad_use_text_encoder": False,
        "ad_text_encoder": None,
        "ad_use_sampler": use_sampler,
        "ad_sampler": sampler,
    }
    preset_widgets = [tuple(Component() for _ in range(10))]
    callbacks["_wire_presets"](
        [widgets()], preset_widgets, [Component()], Component(), 1,
        {"faces.pt": "faces.pt"}, first_sampler=names[0], sampler_names=names,
    )
    result = preset_widgets[0][1].callback("saved")[1:]
    restored = dict(zip(ALL_ARGS.attrs, result[:-1]))
    assert restored["ad_sampler"]["value"] == expected
    assert restored["ad_use_sampler"]["value"] is use_sampler
    assert restored["ad_checkpoint"]["value"] == "Use same checkpoint"
    assert restored["ad_vae"]["value"] == "Use same VAE"
    assert restored["ad_text_encoder"]["value"] == "Use same text encoder"
    assert not any(
        "value" in u and u["value"] is None for u in result if isinstance(u, dict)
    )


_SCHEDULERS = ["Automatic", "Uniform", "Karras", "Exponential", "SGM Uniform"]


@pytest.mark.parametrize(
    ("sampler", "use_sampler", "schedulers", "expected"),
    [
        # Ticked: shown as the sampler and scheduler the WebUI runs it as.
        ("DPM++ 2M Karras", True, _SCHEDULERS, ("DPM++ 2M", "Karras")),
        ("DPM++ 2M SDE Exponential", True, _SCHEDULERS, ("DPM++ 2M SDE", "Exponential")),
        # No split into two choices: the WebUI runs its first sampler with the
        # scheduler the name ends with. Without a scheduler list: kept.
        ("Foo Karras", True, _SCHEDULERS, ("DPM++ 2M", "Karras")),
        ("DPM++ 2M Karras", True, None, ("DPM++ 2M Karras", "Use same scheduler")),
        # The WebUI takes "Uniform" off the name first, which leaves no
        # sampler ("Euler SGM"): not split as "SGM Uniform".
        ("Euler SGM Uniform", True, _SCHEDULERS, ("DPM++ 2M", "Uniform")),
        # Unchanged: a real choice, "Use same sampler", or the box unticked.
        ("Euler a", True, _SCHEDULERS, ("Euler a", "Use same scheduler")),
        ("Use same sampler", True, _SCHEDULERS, ("Use same sampler", "Use same scheduler")),
        ("DPM++ 2M Karras", False, _SCHEDULERS, ("DPM++ 2M", "Use same scheduler")),
    ],
)
def test_load_shows_a_ticked_old_sampler_name_as_sampler_and_scheduler(
    callbacks, sampler, use_sampler, schedulers, expected
):
    # With "Use separate sampler" ticked, Load kept "DPM++ 2M Karras": Gradio 4
    # WebUIs showed the dropdown empty, and after Generate and a restart the
    # tab ran DPM++ 2M with the main pass's scheduler instead of Karras.
    callbacks["get_preset"] = lambda name: {
        "ad_model": "faces.pt",
        "ad_use_sampler": use_sampler,
        "ad_sampler": sampler,
        "ad_scheduler": "Use same scheduler",
    }
    names = ["DPM++ 2M", "DPM++ 2M SDE", "Euler", "Euler a"]
    preset_widgets = [tuple(Component() for _ in range(10))]
    callbacks["_wire_presets"](
        [widgets()], preset_widgets, [Component()], Component(), 1,
        {"faces.pt": "faces.pt"}, first_sampler=names[0], sampler_names=names,
        scheduler_names=schedulers,
    )
    result = preset_widgets[0][1].callback("saved")[1:]
    restored = dict(zip(ALL_ARGS.attrs, result[:-1]))
    assert (restored["ad_sampler"]["value"], restored["ad_scheduler"]["value"]) == expected
    assert restored["ad_use_sampler"]["value"] is use_sampler


def _scheduler(label, name, aliases=None):
    return SimpleNamespace(label=label, name=name, aliases=aliases)


# Some schedulers of AUTOMATIC1111 1.10 and of Forge Neo, in each one's order,
# with the label, name and aliases the WebUI takes off a sampler name.
_WEBUI_SCHEDULERS = {
    "automatic1111": [
        _scheduler("Automatic", "automatic"), _scheduler("Uniform", "uniform"),
        _scheduler("Karras", "karras"), _scheduler("Exponential", "exponential"),
        _scheduler("SGM Uniform", "sgm_uniform", ["SGMUniform"]),
        _scheduler("DDIM", "ddim"), _scheduler("Beta", "beta"),
    ],
    "forge-neo": [
        _scheduler("Automatic", "automatic"), _scheduler("Karras", "karras"),
        _scheduler("Exponential", "exponential"), _scheduler("Uniform", "uniform"),
        _scheduler("SGM Uniform", "sgm_uniform", ["SGMUniform"]),
        _scheduler("DDIM", "ddim"), _scheduler("Beta", "beta"),
    ],
}
_WEBUI_SAMPLERS = ["DPM++ 2M", "DPM++ 2M SDE", "Euler", "Euler a", "DDIM", "LCM"]


def _wire_sampler_load(callbacks, schedulers):
    preset_widgets = [tuple(Component() for _ in range(10))]
    callbacks["_wire_presets"](
        [widgets()], preset_widgets, [Component()], Component(), 1,
        {"faces.pt": "faces.pt"}, first_sampler=_WEBUI_SAMPLERS[0],
        sampler_names=_WEBUI_SAMPLERS, scheduler_names=[x.label for x in schedulers],
        scheduler_endings=[[x.label, x.name, *(x.aliases or ())] for x in schedulers],
    )

    def load(preset):
        callbacks["get_preset"] = lambda name: {"ad_model": "faces.pt", **preset}
        status, *result = preset_widgets[0][1].callback("saved")
        restored = dict(zip(ALL_ARGS.attrs, result[:-1]))
        return status, restored["ad_sampler"]["value"], restored["ad_scheduler"]["value"]

    return load


@pytest.mark.parametrize(
    ("sampler", "use_sampler", "expected", "missing"),
    [
        # Another WebUI's sampler, an alias or a lower-case name: the WebUI
        # looks samplers up by their exact name and runs its first one.
        ("Res Multistep", True, ("DPM++ 2M", "Beta"), True),
        ("k_euler_a", True, ("DPM++ 2M", "Beta"), True),
        ("euler a", True, ("DPM++ 2M", "Beta"), True),
        (None, True, ("DPM++ 2M", "Beta"), True),
        ("Foo Karras", True, ("DPM++ 2M", "Karras"), True),
        # Names the WebUI runs: a scheduler's label, name or alias taken off.
        ("DPM++ 2M Karras", True, ("DPM++ 2M", "Karras"), False),
        ("DPM++ 2M karras", True, ("DPM++ 2M", "Karras"), False),
        ("Euler a SGMUniform", True, ("Euler a", "SGM Uniform"), False),
        ("Euler sgm_uniform", True, ("Euler", "SGM Uniform"), False),
        # One after the other, in the WebUI's order: the last one counts.
        ("DPM++ 2M SDE Exponential Karras", True, ("DPM++ 2M SDE", "Exponential"), False),
        # Unchanged: a real choice, "Use same sampler", or the box unticked.
        ("Euler a", True, ("Euler a", "Beta"), False),
        ("DDIM", True, ("DDIM", "Beta"), False),
        ("Use same sampler", True, ("Use same sampler", "Beta"), False),
        ("Res Multistep", False, ("DPM++ 2M", "Beta"), False),
    ],
)
def test_load_puts_a_ticked_sampler_this_webui_does_not_have_on_the_first_sampler(
    callbacks, sampler, use_sampler, expected, missing
):
    # A preset from another WebUI (for example "Res Multistep" from Forge Neo
    # on AUTOMATIC1111) said "Loaded" with "Use separate sampler" ticked and
    # set a sampler the dropdown does not have (Forge and Forge Neo showed it
    # empty): Generate ran the first sampler with only a console warning,
    # while the image's parameters named the missing one.
    load = _wire_sampler_load(callbacks, _WEBUI_SCHEDULERS["automatic1111"])
    status, *shown = load(
        {"ad_use_sampler": use_sampler, "ad_sampler": sampler, "ad_scheduler": "Beta"}
    )
    assert tuple(shown) == expected
    if missing:
        assert status == (
            f"⚠️ Loaded 'saved', but this WebUI does not have its sampler "
            f"'{sampler}': the tab uses \"DPM++ 2M\" there instead."
        )
    else:
        assert status == "✅ Loaded 'saved'."


def test_load_names_a_missing_sampler_with_the_other_missing_settings(callbacks):
    status, restored = _load_with_lists(callbacks, {
        "ad_model": "faces.pt", "ad_use_sampler": True, "ad_sampler": "Res Multistep",
        "ad_use_checkpoint": True, "ad_checkpoint": "not-here.safetensors",
    })
    assert status == (
        "⚠️ Loaded 'saved', but this WebUI does not have its sampler 'Res Multistep', "
        "checkpoint 'not-here.safetensors': the tab uses \"DPM++ 2M\", "
        "\"Use same …\" or \"None\" there instead."
    )
    assert restored["ad_sampler"]["value"] == "DPM++ 2M"
    assert restored["ad_checkpoint"]["value"] == "Use same checkpoint"
    # Without the sampler the text is the one of the other settings alone.
    status, _ = _load_with_lists(callbacks, {
        "ad_model": "faces.pt", "ad_checkpoint": "not-here.safetensors",
    })
    assert status.endswith("the tab uses \"Use same …\" or \"None\" there instead.")
    # A detector that is not installed comes first.
    status, restored = _load_with_lists(callbacks, {
        "ad_model": "eyes.pt", "ad_use_sampler": True, "ad_sampler": "Res Multistep",
    })
    assert status == (
        "⚠️ Loaded 'saved', but its detector 'eyes.pt' is not installed: the tab "
        "keeps its detector and classes. This WebUI also does not have its sampler "
        "'Res Multistep': the tab uses \"DPM++ 2M\" there instead."
    )
    assert restored["ad_sampler"]["value"] == "DPM++ 2M"


def _webui_runs(sampler, scheduler, schedulers):
    # modules/sd_samplers.py get_sampler_and_scheduler(convert_automatic=False)
    # of AUTOMATIC1111 1.10 and Forge Neo, which Generate runs on the pass.
    by_label = {**{x.name: x for x in schedulers}, **{x.label: x for x in schedulers}}
    found = by_label.get(scheduler, schedulers[0])
    name = sampler or _WEBUI_SAMPLERS[0]
    for x in schedulers:
        for option in [x.label, x.name, *(x.aliases or [])]:
            if name.endswith(" " + option):
                found = x
                name = name[0:-(len(option) + 1)]
                break
    known = name in _WEBUI_SAMPLERS
    return (name if known else _WEBUI_SAMPLERS[0]), found.label, known


@pytest.mark.parametrize("webui", sorted(_WEBUI_SCHEDULERS))
def test_load_shows_the_sampler_and_scheduler_the_webui_runs(callbacks, webui):
    # Every name Load accepts is one the WebUI runs as it is shown, and every
    # name it replaces is one the WebUI replaces with its first sampler.
    schedulers = _WEBUI_SCHEDULERS[webui]
    load = _wire_sampler_load(callbacks, schedulers)
    endings = [e for x in schedulers for e in (x.label, x.name, *(x.aliases or ()))]
    bases = [*_WEBUI_SAMPLERS, "Res Multistep", "k_euler_a", "Foo", "Euler SGM", "DPM++"]
    names = [
        " ".join((base, *tail))
        for base in bases
        for tail in [(), *((e,) for e in endings), *((e, f) for e in endings for f in endings)]
    ]
    kept = replaced = 0
    for name in names:
        status, *shown = load(
            {"ad_use_sampler": True, "ad_sampler": name, "ad_scheduler": "Beta"}
        )
        runs = _webui_runs(name, "Beta", schedulers)
        assert tuple(shown) == runs[:2], name
        assert status.startswith("✅" if runs[2] else "⚠️"), name
        kept += runs[2]
        replaced += not runs[2]
    # Both kinds of names are there.
    assert kept > 100
    assert replaced > 100


def test_load_is_given_the_endings_the_webui_takes_off_a_sampler_name():
    # The fixture never runs adui: check that it hands _wire_presets each
    # scheduler's label, name and aliases, from the list the WebUI uses.
    path = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.ImportFrom) and node.module == "aaaaaa.conditional"
        and "schedulers" in {a.name for a in node.names}
        for node in tree.body
    )
    adui = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "adui"
    )
    call = next(
        node for node in ast.walk(adui)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_wire_presets"
    )
    endings = next(k.value for k in call.keywords if k.arg == "scheduler_endings")
    schedulers = [
        _scheduler("Karras", "karras"),
        _scheduler("SGM Uniform", "sgm_uniform", ["SGMUniform"]),
        SimpleNamespace(label="Beta"),  # a WebUI whose schedulers have a label only
    ]
    value = eval(compile(ast.Expression(endings), str(path), "eval"), {"schedulers": schedulers})
    assert value == [["Karras", "karras"], ["SGM Uniform", "sgm_uniform", "SGMUniform"], ["Beta", None]]


def test_docs_do_not_say_a_restart_and_pasting_split_a_sampler_as_load_does():
    # Load splits a ticked sampler name as the WebUI does at Generate and names
    # a sampler this WebUI does not have; a restart and pasting split a
    # scheduler's label only. Their comments and the docs said "as Load does".
    root = Path(__file__).resolve().parents[1]
    texts = {
        name: (root / name).read_text(encoding="utf-8")
        for name in ("README.md", "CHANGELOG.md", "aaaaaa/ui.py", "scripts/!adetailer.py")
    }
    for name, text in texts.items():
        flat = " ".join(text.replace("#", " ").split())
        for claim in (
            "as Load splits it", "split it as Load does",
            "runs it as, as Load does", "with Karras, as Load does",
        ):
            assert claim not in flat, (name, claim)
    beta2 = next(
        section for section in texts["CHANGELOG.md"].split("\n## ")
        if section.startswith("v26.2.0+plus.8.beta.2")
    )
    known = beta2.split("**Known issues in this beta**")[1]
    assert "Res Multistep" in known
    assert "DPM++ 2M karras" in known


def test_load_keeps_the_override_choices_a_preset_names(callbacks):
    callbacks["get_preset"] = lambda name: {
        "ad_model": "faces.pt",
        "ad_checkpoint": "detail.safetensors",
        "ad_vae": "detail.vae.pt",
        "ad_text_encoder": "None (use detailer checkpoint's own)",
        "ad_sampler": "Euler a",
    }
    preset_widgets = [tuple(Component() for _ in range(10))]
    callbacks["_wire_presets"](
        [widgets()], preset_widgets, [Component()], Component(), 1,
        {"faces.pt": "faces.pt"}, first_sampler="DPM++ 2M",
        sampler_names=["DPM++ 2M", "Euler a"],
    )
    result = preset_widgets[0][1].callback("saved")[1:]
    restored = dict(zip(ALL_ARGS.attrs, result[:-1]))
    assert restored["ad_checkpoint"]["value"] == "detail.safetensors"
    assert restored["ad_vae"]["value"] == "detail.vae.pt"
    assert restored["ad_text_encoder"]["value"] == "None (use detailer checkpoint's own)"
    assert restored["ad_sampler"]["value"] == "Euler a"


@pytest.mark.parametrize("model", ["eyes.pt", None])
def test_load_keeps_the_detector_when_the_preset_names_one_not_installed(
    callbacks, model
):
    # A preset moved with Export/Import, or whose model was deleted or
    # renamed, said "Loaded" and set a detector the dropdown does not have:
    # Generate then failed that tab and every tab after it.
    callbacks["get_preset"] = lambda name: {
        "ad_model": model, "ad_model_classes": "eyes", "ad_prompt": "detailed eyes",
    }
    preset_widgets = [tuple(Component() for _ in range(10))]
    callbacks["_wire_presets"](
        [widgets()], preset_widgets, [Component()], Component(), 1,
        {"faces.pt": "faces.pt"},
    )
    status, *result = preset_widgets[0][1].callback("saved")
    restored = dict(zip(ALL_ARGS.attrs, result[:-1]))

    assert status.startswith("⚠️ Loaded 'saved'")
    assert f"'{model}' is not installed" in status
    for attr in ("ad_model", "ad_model_classes", "ad_model_classes_exclude",
                 "ad_model_classes_excluded"):
        assert restored[attr] == {}
    assert result[-1] == {}  # the CLASSES selection is left alone too
    assert restored["ad_prompt"]["value"] == "detailed eyes"


_OVERRIDE_LISTS = {
    "checkpoint_names": ["detail [abc1234567]"],
    "vae_names": ["Automatic", "None", "detail.vae.pt"],
    "text_encoder_names": ["clip_l.safetensors"],
    "cn_model_names": ["control_v11p_sd15_inpaint [ebff9138]"],
}


def _load_with_lists(callbacks, preset, **lists):
    callbacks["get_preset"] = lambda name: dict(preset)
    preset_widgets = [tuple(Component() for _ in range(10))]
    callbacks["_wire_presets"](
        [widgets()], preset_widgets, [Component()], Component(), 1,
        {"faces.pt": "faces.pt"}, first_sampler="DPM++ 2M",
        sampler_names=["DPM++ 2M", "Euler a"], scheduler_names=["Automatic", "Karras"],
        **{**_OVERRIDE_LISTS, **lists},
    )
    status, *result = preset_widgets[0][1].callback("saved")
    return status, dict(zip(ALL_ARGS.attrs, result[:-1]))


@pytest.mark.parametrize("detector", ["faces.pt", "eyes.pt"])
def test_load_puts_an_override_this_webui_does_not_have_on_its_first_choice(
    callbacks, detector
):
    # A preset moved with Export / Import named a checkpoint, VAE, text
    # encoder, scheduler and ControlNet model this WebUI does not have: Load
    # said "Loaded" and set them (Forge and Forge Neo showed the dropdowns
    # empty), Generate swapped them for the WebUI's own without a word, and
    # the image's parameters still named them.
    status, restored = _load_with_lists(callbacks, {
        "ad_model": detector, "ad_prompt": "detailed face",
        "ad_use_checkpoint": True, "ad_checkpoint": "not-here.safetensors",
        "ad_use_vae": True, "ad_vae": "not-here.vae.pt",
        "ad_use_text_encoder": True, "ad_text_encoder": "not-here-te.safetensors",
        "ad_use_sampler": True, "ad_sampler": "Euler a", "ad_scheduler": "Not A Scheduler",
        "ad_controlnet_model": "control_missing [0000]",
        "ad_controlnet_module": "openpose_full",
    })

    names = [
        "checkpoint 'not-here.safetensors'", "VAE 'not-here.vae.pt'",
        "text encoder 'not-here-te.safetensors'", "scheduler 'Not A Scheduler'",
        "ControlNet model 'control_missing [0000]'",
    ]
    assert status.startswith("⚠️ Loaded 'saved', but ")
    assert all(name in status for name in names)
    if detector == "eyes.pt":  # both warnings, the detector's first
        assert "its detector 'eyes.pt' is not installed" in status
        assert status.index("eyes.pt") < status.index("not-here.safetensors")
        assert restored["ad_model"] == {}
    else:
        assert restored["ad_model"]["value"] == "faces.pt"
    first = {
        "ad_checkpoint": "Use same checkpoint", "ad_vae": "Use same VAE",
        "ad_text_encoder": "Use same text encoder", "ad_scheduler": "Use same scheduler",
        "ad_controlnet_model": "None", "ad_controlnet_module": "None",
    }
    for attr, value in first.items():
        assert restored[attr]["value"] == value
    # The rest of the preset is applied, its "Use separate …" boxes included.
    assert restored["ad_prompt"]["value"] == "detailed face"
    assert restored["ad_use_checkpoint"]["value"] is True
    assert restored["ad_use_vae"]["value"] is True
    assert restored["ad_sampler"]["value"] == "Euler a"


@pytest.mark.parametrize(
    ("attr", "saved", "lists", "expected"),
    [
        # A checkpoint's short name gains " [hash]" once the WebUI first loads
        # it: the name saved before that is the same checkpoint, and back.
        ("ad_checkpoint", "detail", {}, "detail [abc1234567]"),
        ("ad_checkpoint", "detail [abc1234567]", {"checkpoint_names": ["detail"]}, "detail"),
        ("ad_checkpoint", "detail [abc1234567]", {}, "detail [abc1234567]"),
        # AUTOMATIC1111's ControlNet names carry a hash, Forge's do not.
        ("ad_controlnet_model", "control_v11p_sd15_inpaint", {}, "control_v11p_sd15_inpaint [ebff9138]"),
        (
            "ad_controlnet_model", "control_v11p_sd15_inpaint [ebff9138]",
            {"cn_model_names": ["control_v11p_sd15_inpaint"]}, "control_v11p_sd15_inpaint",
        ),
        # Two checkpoints of that name: left to the WebUI, as before.
        (
            "ad_checkpoint", "detail",
            {"checkpoint_names": ["detail [abc1234567]", "detail [0123456789]"]}, "detail",
        ),
        # The first choices are always there, also with nothing installed.
        ("ad_checkpoint", "Use same checkpoint", {"checkpoint_names": []}, "Use same checkpoint"),
        ("ad_vae", "None", {}, "None"),
        ("ad_vae", "detail.vae.pt", {}, "detail.vae.pt"),
        (
            "ad_text_encoder", "None (use detailer checkpoint's own)",
            {"text_encoder_names": []}, "None (use detailer checkpoint's own)",
        ),
        ("ad_scheduler", "Karras", {}, "Karras"),
        ("ad_controlnet_model", "Passthrough", {"cn_model_names": []}, "Passthrough"),
    ],
)
def test_load_keeps_an_override_this_webui_has(callbacks, attr, saved, lists, expected):
    status, restored = _load_with_lists(
        callbacks,
        {"ad_model": "faces.pt", attr: saved, "ad_controlnet_module": "inpaint_only"},
        **lists,
    )
    assert status == "✅ Loaded 'saved'."
    assert restored[attr]["value"] == expected
    assert restored["ad_controlnet_module"]["value"] == "inpaint_only"


def test_load_does_not_take_another_file_of_the_same_name(callbacks):
    # Two different hashes are two different files: the WebUI would not run
    # the one the preset names either.
    status, restored = _load_with_lists(
        callbacks, {"ad_model": "faces.pt", "ad_checkpoint": "detail [0123456789]"}
    )
    assert status.startswith("⚠️ Loaded 'saved', but this WebUI does not have its ")
    assert "checkpoint 'detail [0123456789]'" in status
    assert restored["ad_checkpoint"]["value"] == "Use same checkpoint"


@pytest.mark.parametrize("model", ["faces.pt", "None"])
def test_load_still_sets_an_installed_detector_or_none(callbacks, model):
    callbacks["get_preset"] = lambda name: {"ad_model": model}
    preset_widgets = [tuple(Component() for _ in range(10))]
    callbacks["_wire_presets"](
        [widgets()], preset_widgets, [Component()], Component(), 1,
        {"faces.pt": "faces.pt"},
    )
    status, *result = preset_widgets[0][1].callback("saved")

    assert status == "✅ Loaded 'saved'."
    assert dict(zip(ALL_ARGS.attrs, result))["ad_model"] == {"value": model}


def test_load_is_given_the_sampler_choices_a_fresh_build_shows():
    # The fixture never runs adui: check that it hands _wire_presets the real
    # samplers, the choices of the tab's sampler dropdown besides "Use same".
    path = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    adui = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "adui"
    )
    call = next(
        node for node in ast.walk(adui)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_wire_presets"
    )
    names = next(k.value for k in call.keywords if k.arg == "sampler_names")
    webui_info = SimpleNamespace(
        sampler_names=["DPM++ 2M", "Euler a"], scheduler_names=["Automatic", "Karras"]
    )
    value = eval(compile(ast.Expression(names), str(path), "eval"), {"webui_info": webui_info})
    assert value == ["DPM++ 2M", "Euler a"]
    # And the schedulers, the choices of its scheduler dropdown besides "Use same".
    names = next(k.value for k in call.keywords if k.arg == "scheduler_names")
    value = eval(compile(ast.Expression(names), str(path), "eval"), {"webui_info": webui_info})
    assert value == ["Automatic", "Karras"]


def test_load_is_given_the_override_choices_a_fresh_build_shows():
    # The fixture never runs adui: check that it hands _wire_presets the lists
    # the checkpoint, VAE, text encoder and ControlNet dropdowns are built from.
    path = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    call = next(
        node for node in ast.walk(functions["adui"])
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_wire_presets"
    )
    keywords = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert keywords["checkpoint_names"] == "webui_info.checkpoints_list"
    assert keywords["vae_names"] == "webui_info.vae_list"
    assert keywords["text_encoder_names"] == "webui_info.text_encoders_list"
    assert keywords["cn_model_names"] == "_cn_models"
    adui = ast.unparse(functions["adui"])
    assert "_cn_models = get_cn_models()" in adui
    built = ast.unparse(functions["inpainting"]).replace("'", '"')
    assert '["Use same checkpoint", *webui_info.checkpoints_list]' in built
    assert '["Use same VAE", *webui_info.vae_list]' in built
    assert "*webui_info.text_encoders_list" in built
    built = ast.unparse(functions["controlnet"]).replace("'", '"')
    assert '["None", "Passthrough", *get_cn_models()]' in built


@pytest.mark.parametrize("exclude", [False, True])
def test_detector_change_keeps_a_class_name_in_another_case(callbacks, exclude):
    # Pasted parameters or a preset may hold "Face" for the model's "face".
    # The engine matches it, but the detector change that follows a paste or
    # Load dropped it, so every class was inpainted (in NOT mode, none excluded).
    result = callbacks["on_ad_model_update"](
        "faces.pt", ["Face"], {"faces.pt": "faces.pt"},
        current_include="" if exclude else "Face",
        current_exclude=exclude,
        current_excluded="Face" if exclude else "",
    )
    assert result[1]["value"] == ["face"]
    assert result[1]["choices"] == ["face", "hand"]
    assert result[0]["value"] == ("" if exclude else "face")
    assert result[3]["value"] == ("face" if exclude else "")


def test_detector_change_drops_an_ambiguous_or_unknown_class_name(callbacks):
    # As in the engine: an exact match wins, a name matching several classes
    # in another case and a name the model does not have are dropped.
    result = callbacks["on_ad_model_update"](
        "faces.pt", [], {"faces.pt": "faces.pt"}, current_include="Face,face,HAND",
    )
    assert result[0]["value"] == "face,hand"  # listed once
    callbacks["get_model_class_names"] = lambda path: ["Face", "face", "hand"]
    result = callbacks["on_ad_model_update"](
        "faces.pt", [], {"faces.pt": "faces.pt"},
        current_include="FACE,Face,cat,HAND",
    )
    assert result[0]["value"] == "Face,hand"
    # Without class names every requested name is kept, as before.
    callbacks["get_model_class_names"] = lambda path: []
    result = callbacks["on_ad_model_update"](
        "faces.pt", [], {"faces.pt": "faces.pt"}, current_include="Face",
    )
    assert result[0]["value"] == "Face"


@pytest.mark.parametrize("exclude", [False, True])
def test_detector_change_keeps_a_numeric_class_id(callbacks, exclude):
    # The API may send a class id ("1" is the 2nd class, hand) and detection
    # accepts it, but the detector change that follows a paste or Load dropped
    # it, so every class was inpainted (in NOT mode, none excluded). An id the
    # detector does not have (7) is dropped, as detection drops it, and a
    # superscript digit is only an unknown name.
    csv = "1,Face,7,²"
    result = callbacks["on_ad_model_update"](
        "faces.pt", [], {"faces.pt": "faces.pt"},
        current_include="" if exclude else csv,
        current_exclude=exclude,
        current_excluded=csv if exclude else "",
    )
    assert result[1]["value"] == ["1", "face"]
    assert result[1]["choices"] == ["face", "hand", "1"]
    assert result[0]["value"] == ("" if exclude else "1,face")
    assert result[3]["value"] == ("1,face" if exclude else "")
    # The MediaPipe face features match names only: an id stays dropped.
    result = callbacks["on_ad_model_update"](
        MEDIAPIPE_FACE_FEATURES_MODEL, [], {},
        current_include="" if exclude else "1,eyes",
        current_exclude=exclude,
        current_excluded="1,eyes" if exclude else "",
    )
    assert result[1]["value"] == ["eyes"]
    assert result[0]["value"] == ("" if exclude else "eyes")
    assert result[3]["value"] == ("eyes" if exclude else "")


@pytest.mark.parametrize("exclude", [False, True])
def test_detector_change_drops_a_class_id_that_cannot_be_converted(callbacks, exclude):
    # A number that cannot be converted is dropped like an unknown name. A
    # zero-padded id is still kept as written.
    csv = "face,001," + "9" * 5000
    result = callbacks["on_ad_model_update"](
        "faces.pt", [], {"faces.pt": "faces.pt"},
        current_include="" if exclude else csv,
        current_exclude=exclude,
        current_excluded=csv if exclude else "",
    )
    assert result[1]["value"] == ["face", "001"]
    assert result[0]["value"] == ("" if exclude else "face,001")
    assert result[3]["value"] == ("face,001" if exclude else "")


def test_folder_box_is_never_saved_as_a_startup_default():
    # Settings -> Defaults -> Apply saved a path typed in "Or batch a whole
    # folder" to ui-config.json, and after a restart every single-image run
    # became a folder batch. The host skips a component with this flag.
    path = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body, i = next(
        (node.body, i)
        for node in ast.walk(tree)
        if isinstance(getattr(node, "body", None), list)
        for i, stmt in enumerate(node.body)
        if isinstance(stmt, ast.Assign)
        and ast.unparse(stmt.targets[0]) == "w.ad_apply_folder"
    )
    namespace = {
        "gr": SimpleNamespace(
            Textbox=lambda **kw: SimpleNamespace(**kw),
            Checkbox=lambda **kw: SimpleNamespace(**kw),
        ),
        "w": SimpleNamespace(),
        "eid": str,
    }
    module = ast.Module(body=body[i : i + 2], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    assert getattr(namespace["w"].ad_apply_folder, "do_not_save_to_config", False)
    # The checkboxes beside it keep the startup defaults the README documents.
    flagged = {
        ast.unparse(node.targets[0])
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]).endswith(".do_not_save_to_config")
    }
    assert "w.ad_apply_same_folder.do_not_save_to_config" not in flagged
    assert "w.ad_apply_save.do_not_save_to_config" not in flagged


def test_preset_save_delete_rename_keep_other_tabs_selection(callbacks, monkeypatch):
    # Save, Delete and Rename refresh every tab's list of presets, but they
    # also moved every tab's selection to the acting tab's preset (or to
    # "(none)" on an error), so a Delete in another tab removed the wrong one.
    disk = {"hands": {}, "eyes": {}}
    # Save also refreshes the preview of tabs showing the saved preset.
    monkeypatch.setattr("adetailer.presets.get_preset", lambda name: {})

    def rename(old, new):
        disk[new] = disk.pop(old)
        return True, ""

    callbacks.update(
        get_preset_names=lambda: sorted(disk),
        is_valid_name=lambda name: "/" not in name,
        save_preset=lambda name, state: disk.__setitem__(name, state) is None,
        delete_preset=lambda name: disk.pop(name, None) is not None,
        rename_preset=rename,
    )
    tabs = 3
    preset_widgets = [tuple(Component() for _ in range(10)) for _ in range(tabs)]
    callbacks["_wire_presets"](
        [widgets() for _ in range(tabs)], preset_widgets,
        [Component() for _ in range(tabs)], Component(), tabs,
    )
    dropdowns = [p[0] for p in preset_widgets]
    for button in (2, 3, 5):  # Rename, Delete, Save read every dropdown
        assert preset_widgets[0][button].inputs[-tabs:] == dropdowns
    values = [None] * len(ALL_ARGS.attrs)

    def chosen(result):
        return [u["value"] for u in result[1 : 1 + tabs]]

    save = preset_widgets[0][5].callback
    assert chosen(save("faces", *values, "(none)", "hands", "eyes")) == [
        "faces", "hands", "eyes",
    ]
    assert "faces" in disk
    for name in ("", "bad/name"):
        assert chosen(save(name, *values, "eyes", "hands", "(none)")) == [
            "eyes", "hands", "(none)",
        ]
    # Nothing picked in tab 2: no tab changes.
    delete = preset_widgets[1][3].callback
    assert chosen(delete("eyes", "(none)", "hands")) == ["eyes", "(none)", "hands"]
    # Deleting "eyes" in tab 2 clears it there and in tab 1, which showed it.
    assert chosen(delete("eyes", "eyes", "hands")) == ["(none)", "(none)", "hands"]
    assert "eyes" not in disk
    # Every tab that showed the renamed preset follows its new name.
    rename_cb = preset_widgets[1][2].callback
    assert chosen(rename_cb("hand", "hands", "hands", "faces")) == [
        "hand", "hand", "faces",
    ]
    assert chosen(rename_cb("", "faces", "hand", "hand")) == ["faces", "hand", "hand"]
    result = save("faces", *values, "(none)", "hand", "faces")
    assert len({id(u) for u in result[1 : 1 + tabs]}) == tabs
    assert all(u["choices"] == ["(none)", "faces", "hand"] for u in result[1 : 1 + tabs])


def test_save_over_the_selected_preset_refreshes_its_preview(callbacks, monkeypatch):
    # Saving over the preset a tab shows keeps the dropdown's value, so its
    # .change never rebuilt the preview, which kept the old contents.
    disk = {"faces": {"ad_prompt": "old face prompt", "ad_negative_prompt": ""}, "hands": {}}
    monkeypatch.setattr("adetailer.presets.get_preset", lambda name: disk.get(name, {}))
    callbacks.update(
        get_preset_names=lambda: sorted(disk),
        is_valid_name=lambda name: True,
        save_preset=lambda name, state: disk.__setitem__(name, state) is None,
    )
    tabs = 3
    preset_widgets = [tuple(Component() for _ in range(10)) for _ in range(tabs)]
    callbacks["_wire_presets"](
        [widgets() for _ in range(tabs)], preset_widgets,
        [Component() for _ in range(tabs)], Component(), tabs,
    )
    save = preset_widgets[0][5]
    assert save.outputs[1 + tabs :] == [p[9] for p in preset_widgets]
    values = dict.fromkeys(ALL_ARGS.attrs)
    values.update(ad_prompt="new face prompt", ad_negative_prompt="")

    result = save.callback("faces", *values.values(), "faces", "faces", "hands")

    assert len(result) == 1 + 2 * tabs
    previews = result[1 + tabs :]
    for preview in previews[:2]:  # the acting tab and another showing "faces"
        assert preview["visible"] is True
        assert "new face prompt" in preview["value"]
        assert "old face prompt" not in preview["value"]
    assert previews[0] is not previews[1]
    assert previews[2] == {}
    # A save that fails changes no preview.
    result = save.callback("", *values.values(), "faces", "faces", "hands")
    assert result[1 + tabs :] == [{}, {}, {}]
    # The tuple built by one_ui_group ends with the preview.
    path = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    built = next(
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", "") == "preset_widgets"
    )
    assert [ast.unparse(e) for e in built.elts][9:] == ["preset_preview"]


def test_import_replacing_the_selected_preset_refreshes_its_preview(
    callbacks, monkeypatch, tmp_path
):
    # An import with "Overwrite on conflict" replaced the selected preset but
    # kept the dropdown's value, so the preview kept the old contents.
    disk = {"faces": {"ad_prompt": "old face prompt", "ad_negative_prompt": ""}}
    monkeypatch.setattr("adetailer.presets.get_preset", lambda name: disk.get(name, {}))

    def import_payload(_value, *, overwrite):
        disk["faces"] = {"ad_prompt": "imported face prompt", "ad_negative_prompt": ""}
        return 0, 1, []

    callbacks["import_presets_json"] = import_payload
    uploaded = tmp_path / "presets.json"
    uploaded.write_text('{"faces": {}}', encoding="utf-8")

    _choices, status, preview = callbacks["_do_import"](str(uploaded), True, "faces")

    assert "replaced" in status
    assert preview["visible"] is True
    assert "imported face prompt" in preview["value"]
    # With no preset selected the preview stays hidden; an unreadable file
    # changes nothing.
    assert callbacks["_do_import"](str(uploaded), True, "(none)")[2] == {
        "value": "", "visible": False,
    }
    assert callbacks["_do_import"](str(tmp_path / "missing.json"), True, "faces")[2] == {}
    # A selected preset the preview cannot show (edited by hand) does not
    # cost the import its status.
    disk["hands"] = {"ad_prompt": None, "ad_negative_prompt": None}
    _choices, status, preview = callbacks["_do_import"](str(uploaded), True, "hands")
    assert "replaced" in status
    assert preview == {}
    # The upload reads the tab's dropdown and updates its preview.
    path = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    upload = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "preset_import_btn.upload"
    )
    kw = {k.arg: ast.unparse(k.value) for k in upload.keywords}
    assert kw["inputs"].endswith("preset_dropdown]")
    assert kw["outputs"].endswith("preset_preview]")


def test_guide_describes_source_folder_batches_and_the_export_fallback():
    # The Guide said folder-batch results always go to ADetailer-Inpaint and
    # that the library can be exported as a file, which is wrong with "Save
    # results in the source folder instead" and on AUTOMATIC1111 (Gradio 3).
    path = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        node for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and getattr(node.target, "id", "") == "_GUIDE_SECTIONS"
    )
    sections = dict(ast.literal_eval(node.value))
    tools = next(body for title, body in sections.items() if "run-on-image" in title)
    presets = next(body for title, body in sections.items() if "Presets" in title)
    assert "always saved to the `ADetailer-Inpaint` folder" not in tools
    assert "Save results in the source folder instead" in tools
    assert "`name-ad`" in tools
    where = next(line for line in presets.splitlines() if "Where files go" in line)
    assert "Save results in the source folder instead" in where
    library = next(line for line in presets.splitlines() if "Preset library" in line)
    assert "AUTOMATIC1111" in library and "user_presets.json" in library
    # reForge's main branch ships Gradio 3 too, so the limit is the Gradio
    # version, not one WebUI.
    assert "Gradio 3" in library and "reForge" in library


def test_guide_says_merge_joins_every_mask():
    # The Guide said Merge combines overlapping masks, but it joins every
    # mask into one region, also far-apart faces (as README and the hint say).
    from PIL import Image

    from adetailer.mask import mask_merge_invert

    path = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        node for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and getattr(node.target, "id", "") == "_GUIDE_SECTIONS"
    )
    sections = dict(ast.literal_eval(node.value))
    masks = next(body for title, body in sections.items() if "Mask preprocessing" in title)
    line = next(line for line in masks.splitlines() if "Mask merge mode" in line)
    assert "overlapping" not in line
    assert "Merge: merge all masks into one" in line
    assert "None: inpaint each mask separately" in line
    # What the Guide now says: two masks that do not touch become one.
    a = Image.new("L", (64, 64))
    a.paste(255, (2, 2, 12, 12))
    b = Image.new("L", (64, 64))
    b.paste(255, (40, 40, 60, 60))
    merged = mask_merge_invert([a, b], "Merge")
    assert len(merged) == 1
    assert merged[0].getbbox() == (2, 2, 60, 60)
    assert len(mask_merge_invert([a, b], "None")) == 2


def _guide_sections():
    path = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        node for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and getattr(node.target, "id", "") == "_GUIDE_SECTIONS"
    )
    return tree, dict(ast.literal_eval(node.value))


def test_detection_resolution_help_names_the_detectors_own_size():
    # 0 runs a detector at the size stored in its checkpoint: 640 for the
    # bundled models, 1024 for some community ones. The help said 0 was a fixed
    # 640, so on a 1024 model a "higher" 768 lowered the resolution.
    tree, sections = _guide_sections()
    slider = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") == "Slider"
        and any(
            k.arg == "elem_id" and "ad_detection_resolution" in ast.unparse(k.value)
            for k in node.keywords
        )
    )
    info = ast.literal_eval(next(k.value for k in slider.keywords if k.arg == "info"))
    assert "(640)" not in info
    assert "own size" in info
    assert "1024" in info
    lines = [
        line for body in sections.values() for line in body.splitlines()
        if line.startswith(("- **Detection resolution", "- **Small or distant"))
    ]
    assert len(lines) == 2
    assert all("own size" in line and "default 640" not in line for line in lines)
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    for old in ("default inference size of 640px", "640 default", "stays at the default 640"):
        assert old not in readme


def test_guide_says_the_mediapipe_detectors_download_their_model_files():
    # The Guide said they work "without a model file": since plus.7 they need
    # MediaPipe's own files, downloaded on first use, so an offline machine
    # found nothing.
    from adetailer.mediapipe import _FACE_LANDMARKER_ASSET, _MODEL_DIR

    _tree, sections = _guide_sections()
    body = next(body for title, body in sections.items() if "Choosing a detector" in title)
    line = next(line for line in body.splitlines() if "`mediapipe_face_mesh`" in line)
    assert "without a model file" not in line
    assert f"adetailer/{_MODEL_DIR.name}/" in line
    assert _FACE_LANDMARKER_ASSET[0] in line
    assert "offline" in line


def test_guide_places_the_detection_tools_after_the_detection_section():
    # The Guide put them inside the Detection section, as the README did
    # before; they are separate accordions right after it.
    tree, sections = _guide_sections()
    body = next(body for title, body in sections.items() if "run-on-image" in title)
    first = body.strip().splitlines()[0]
    assert "inside each tab's" not in first
    assert "right after each tab's **Detection** section" in first
    assert "above Mask preprocessing" in first

    def title(node):
        call = node.items[0].context_expr if isinstance(node, ast.With) else None
        if (
            isinstance(call, ast.Call)
            and getattr(call.func, "attr", "") == "Accordion"
            and call.args
            and isinstance(call.args[0], ast.Constant)
        ):
            return call.args[0].value
        return None

    detection = next(node for node in ast.walk(tree) if title(node) == "Detection")
    nested = [title(node) for node in ast.walk(detection) if node is not detection]
    assert not {"Detection preview", "Run ADetailer on an image"} & set(nested)
    parent = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.With) and detection in node.body
    )
    order = [
        title(node) or [title(inner) for inner in node.body]
        for node in parent.body
        if isinstance(node, ast.With)
    ]
    assert order[:3] == [
        "Detection", ["Detection preview", "Run ADetailer on an image"], "Mask Preprocessing"
    ]


def test_guide_getting_started_places_the_detector_and_names_enable_this_tab():
    # It called the detector "the first dropdown at the top of each tab" (that
    # is the Saved presets dropdown) and never named "Enable this tab", which
    # tabs 2 and later start without: its own "tab 2 = hands" did nothing.
    tree, sections = _guide_sections()
    body = next(body for title, body in sections.items() if "Getting started" in title)
    assert "first dropdown" not in body
    detector = next(line for line in body.splitlines() if "**ADetailer detector**" in line)
    assert "below the Preset library" in detector
    enable = next(line for line in body.splitlines() if "**Enable this tab**" in line)
    assert "only the 1st tab is switched on" in enable
    assert "skipped" in enable

    group = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "one_ui_group"
    )

    def widget(fragment):
        return next(
            node for node in ast.walk(group)
            if isinstance(node, ast.Call)
            and any(
                k.arg == "elem_id" and fragment == ast.unparse(k.value)
                for k in node.keywords
            )
        )

    tab_enable = widget("eid('ad_tab_enable')")
    value = next(k.value for k in tab_enable.keywords if k.arg == "value")
    assert ast.unparse(value) == "sv('ad_tab_enable', n == 0)"
    label = next(k.value for k in tab_enable.keywords if k.arg == "label")
    assert "Enable this tab" in ast.unparse(label)
    presets = widget("eid('ad_preset_dropdown')")
    paste = widget("eid('ad_paste_settings')")
    detector_dd = widget("eid('ad_model')")
    assert tab_enable.lineno < presets.lineno < paste.lineno < detector_dd.lineno


@pytest.mark.parametrize("gradio4", [False, True])
def test_saved_presets_reach_the_page_a_browser_reload_is_served(
    callbacks, monkeypatch, gradio4
):
    # A browser reload (F5) is served the page config Gradio built at startup,
    # which holds each dropdown's own choices list: a preset saved, deleted or
    # renamed afterwards was missing (or back) after F5 until a restart.
    # Gradio 3 keeps plain strings, Gradio 4 (label, value) pairs.
    def listed(*names):
        return [(n, n) for n in names] if gradio4 else list(names)

    def preset_row():
        dropdown = Component()
        dropdown.choices = listed("(none)", "saved")
        return (dropdown, *(Component() for _ in range(9)))

    disk = ["saved"]
    monkeypatch.setattr("adetailer.presets.get_preset", lambda name: {})
    callbacks["get_preset_names"] = lambda: sorted(disk)
    callbacks["is_valid_name"] = lambda name: True
    callbacks["save_preset"] = lambda name, state: disk.append(name) or True
    callbacks["delete_preset"] = lambda name: disk.remove(name) or True

    def rename(old, new):
        disk[disk.index(old)] = new
        return True, ""

    callbacks["rename_preset"] = rename
    # txt2img and img2img each wire their own tabs; a reload shows both.
    modes = []
    for _mode in ("txt2img", "img2img"):
        rows = [preset_row(), preset_row()]
        callbacks["_wire_presets"](
            [widgets(), widgets()], rows, [Component(), Component()], Component(), 2
        )
        modes.append(rows)
    dropdowns = [row[0] for rows in modes for row in rows]
    served = [dd.choices for dd in dropdowns]

    # A UI built and dropped before (the host's API builds one at startup,
    # "Reload UI" another) is not kept alive and does not break a save.
    dropped = [preset_row()]
    callbacks["_wire_presets"]([widgets()], dropped, [Component()], Component(), 1)
    probe = weakref.ref(dropped[0][0])
    del dropped
    gc.collect()
    assert probe() is None

    def check(*names):
        assert [dd.choices for dd in dropdowns] == [listed(*names)] * 4
        assert all(dd.choices is before for dd, before in zip(dropdowns, served))

    values = [None] * len(ALL_ARGS.attrs)
    modes[0][0][5].callback("faces-2", *values, "(none)", "saved")
    check("(none)", "faces-2", "saved")
    modes[1][1][2].callback("faces-3", "(none)", "faces-2")
    check("(none)", "faces-3", "saved")
    modes[1][0][3].callback("saved", "(none)")
    check("(none)", "faces-3")
    assert len(callbacks["_PRESET_DROPDOWNS"]) == 4


def test_import_reaches_the_page_a_browser_reload_is_served(callbacks, tmp_path):
    dropdown = Component()
    dropdown.choices = before = ["(none)", "saved"]
    callbacks["_wire_presets"](
        [widgets()], [(dropdown, *(Component() for _ in range(9)))],
        [Component()], Component(), 1,
    )
    callbacks["get_preset_names"] = lambda: ["imported", "saved"]
    callbacks["import_presets_json"] = lambda _value, *, overwrite: (1, 0, [])
    uploaded = tmp_path / "presets.json"
    uploaded.write_text('{"imported": {}}', encoding="utf-8")

    choices, status, _preview = callbacks["_do_import"](str(uploaded), False)

    assert "added" in status
    assert choices["choices"] == ["(none)", "imported", "saved"]
    assert dropdown.choices is before
    assert before == ["(none)", "imported", "saved"]


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-be", "utf-32"])
def test_import_accepts_a_utf16_file(callbacks, tmp_path, encoding):
    # Windows PowerShell 5.1 (">", Out-File) and Notepad's "Unicode" save
    # UTF-16; class-name sidecars were read in it, preset files were not.
    payload = '{"saved":{"ad_model":"animals.pt"}}'
    uploaded = tmp_path / "presets.json"
    uploaded.write_bytes(payload.encode(encoding))
    received = []

    def import_payload(value, *, overwrite):
        received.append(value)
        return 1, 0, []

    callbacks["import_presets_json"] = import_payload
    _choices, status, _preview = callbacks["_do_import"](str(uploaded), False)
    assert received == [payload]
    assert "added" in status


def test_import_refuses_a_file_in_another_encoding(callbacks, tmp_path):
    uploaded = tmp_path / "presets.json"
    uploaded.write_bytes('{"caf\xe9": {}}'.encode("cp1252"))
    received = []
    callbacks["import_presets_json"] = lambda value, *, overwrite: received.append(value)

    choices, status, preview = callbacks["_do_import"](str(uploaded), False)

    assert "could not read file" in status
    assert received == []
    assert (choices, preview) == ({}, {})


def test_api_default_of_a_tab_is_not_its_remembered_settings():
    # The host's API reads each script input's .value once at startup as the
    # default for a tab a request leaves out. With remembered settings, a
    # request that sent only the 1st tab (the upstream REST-API format) also
    # ran a remembered, enabled 2nd tab: a hand pass it never asked for.
    from adetailer.args import ADetailerArgs

    path = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    group = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "one_ui_group"
    )
    at = next(
        i for i, node in enumerate(group.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "state"
    )
    code = compile(ast.Module(body=group.body[at : at + 2], type_ignores=[]), str(path), "exec")

    class State:
        # Gradio 3.41 and 4.40: a callable value is called once for .value (what
        # the API reads) and attached as a page-load event (what the UI gets).
        def __init__(self, value):
            self.load_fn = value
            self.value = value()

    remembered = [
        {"ad_model": "face_yolov8n.pt", "ad_tab_enable": True, "ad_prompt": "smile"},
        {"ad_model": "hand_yolov8n.pt", "ad_tab_enable": True, "ad_prompt": "hand"},
    ]
    models = ["face_yolov8n.pt", "hand_yolov8n.pt"]

    def build(n, saved):
        namespace = {
            "gr": SimpleNamespace(State=State),
            "state_init": lambda _w: dict(remembered[n]),
            "w": None,
            "n": n,
            "saved": saved,
            "model_choices": [*models, "None"] if n == 0 else ["None", *models],
        }
        exec(code, namespace)
        return namespace["state"]

    states = [build(n, remembered[n]) for n in range(2)]
    # A page load still gives the UI the restored settings.
    assert [s.load_fn() for s in states] == remembered
    # The host fills what the request leaves out with each input's .value.
    request = [True, False, {"ad_model": "face_yolov8n.pt"}]
    args = [*request, *[False, False, *(s.value for s in states)][len(request) :]]
    tabs = [ADetailerArgs(**a) for a in args if isinstance(a, dict)]
    assert [t.ad_model for t in tabs if not t.need_skip()] == ["face_yolov8n.pt"]
    # A request with no tab at all runs a fresh setup's 1st tab.
    first = ADetailerArgs(**states[0].value)
    assert (first.ad_model, first.ad_tab_enable, first.ad_prompt) == ("face_yolov8n.pt", True, "")
    # With nothing remembered, the fresh build's own values, as before.
    assert build(1, {}).value == remembered[1]
