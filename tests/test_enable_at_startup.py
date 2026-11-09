"""'Enable ADetailer at startup' (Settings -> ADetailer). These tests do not
start a WebUI: the functions are taken from the source and run with stand-ins."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_UI_PATH = _ROOT / "aaaaaa" / "ui.py"
_SCRIPT_PATH = _ROOT / "scripts" / "!adetailer.py"


def _load(path, functions=(), assigns=(), **namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = ast.parse("from __future__ import annotations").body
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(getattr(t, "id", "") in assigns for t in targets):
                body.append(node)
    namespace = {"Any": Any, "sys": sys, **namespace}
    module = ast.Module(body=body, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture
def ui():
    return _load(
        _UI_PATH,
        functions=("apply_enable_at_startup",),
        assigns=(
            "ENABLE_AT_STARTUP_CHOICES",
            "_ENABLE_AT_STARTUP_MODES",
            "_MAIN_ACCORDIONS",
        ),
    )


def _accordion():
    # Like the WebUI's InputAccordion: a hidden checkbox plus its gr.Accordion.
    return SimpleNamespace(
        value=False,
        accordion=SimpleNamespace(open=False, elem_classes=["input-accordion"]),
    )


def _built(ui):
    built = {"txt2img": _accordion(), "img2img": _accordion()}
    ui._MAIN_ACCORDIONS.update(built)
    return built


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("txt2img", ["txt2img"]),
        ("img2img", ["img2img"]),
        ("Both", ["txt2img", "img2img"]),
    ],
)
def test_chosen_modes_start_switched_on(ui, choice, expected):
    built = _built(ui)

    assert ui.apply_enable_at_startup(choice) == expected

    for mode, acc in built.items():
        on = mode in expected
        assert acc.value is on
        assert acc.accordion.open is on
        assert ("input-accordion-open" in acc.accordion.elem_classes) is on
        # A switched-on panel opts out of ui-config.json; the other one is
        # still left to Settings > Defaults.
        assert getattr(acc, "do_not_save_to_config", False) is on
        assert getattr(acc.accordion, "do_not_save_to_config", False) is on
    assert ui._MAIN_ACCORDIONS == {}


def test_off_changes_nothing(ui, capsys):
    built = _built(ui)

    assert ui.apply_enable_at_startup("Off") == []

    for acc in built.values():
        assert acc.value is False
        assert acc.accordion.open is False
        assert acc.accordion.elem_classes == ["input-accordion"]
        assert not hasattr(acc, "do_not_save_to_config")
        assert not hasattr(acc.accordion, "do_not_save_to_config")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("choice", ["both", "TXT2IMG", "", None, 1, True, ["Both"], "../Both"])
def test_unknown_value_counts_as_off_with_one_line(ui, capsys, choice):
    built = _built(ui)

    assert ui.apply_enable_at_startup(choice) == []

    assert all(acc.value is False for acc in built.values())
    out = capsys.readouterr().out
    assert out.count("\n") == 1
    assert "using Off" in out
    assert out.isascii()


def test_a_mode_without_its_panel_is_skipped(ui):
    ui._MAIN_ACCORDIONS["txt2img"] = _accordion()

    assert ui.apply_enable_at_startup("Both") == ["txt2img"]


def test_a_panel_that_refuses_the_change_does_not_stop_the_other(ui):
    class Refusing:
        def __setattr__(self, key, value):
            raise RuntimeError("read-only")

    ok = _accordion()
    ui._MAIN_ACCORDIONS.update({"txt2img": Refusing(), "img2img": ok})

    assert ui.apply_enable_at_startup("Both") == ["img2img"]
    assert ok.value is True


def _host_loadsave(acc, ui_settings):
    """The part of the WebUI's ui_loadsave.add_component (A1111 1.10 and
    Forge Neo) that handles an InputAccordion: a saved value overrides the
    built one unless the component opts out."""
    key = "customscript/!adetailer.py/txt2img/ADetailer/value"
    for obj, field in ((acc, "value"), (acc.accordion, "open")):
        if getattr(obj, "do_not_save_to_config", False):
            continue
        saved = ui_settings.get(key)
        if saved is None:
            ui_settings[key] = getattr(obj, field)
        else:
            setattr(obj, field, saved)


def test_the_off_that_ui_config_already_holds_does_not_switch_it_back_off(ui):
    # Every existing ui-config.json holds "off" for the panel: the WebUI
    # writes the built value there on its first start.
    built = _built(ui)
    ui.apply_enable_at_startup("txt2img")
    ui_settings = {"customscript/!adetailer.py/txt2img/ADetailer/value": False}

    _host_loadsave(built["txt2img"], ui_settings)

    assert built["txt2img"].value is True
    assert built["txt2img"].accordion.open is True


def test_with_off_settings_defaults_still_decide(ui):
    built = _built(ui)
    ui.apply_enable_at_startup("Off")
    ui_settings = {"customscript/!adetailer.py/txt2img/ADetailer/value": True}

    _host_loadsave(built["txt2img"], ui_settings)

    assert built["txt2img"].value is True
    assert built["txt2img"].accordion.open is True


def _adui_tree():
    tree = ast.parse(_UI_PATH.read_text(encoding="utf-8"))
    return next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "adui"
    )


def test_panel_is_still_built_switched_off_so_the_api_default_stays_off():
    # The API takes the script defaults from the UI as built (Forge Neo right
    # after ui(), A1111 by calling ui() again): building the panel switched
    # on would run ADetailer for every API request that leaves it out.
    adui = _adui_tree()
    calls = [
        node for node in ast.walk(adui)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "InputAccordion"
    ]
    assert len(calls) == 1
    value = {kw.arg: kw.value for kw in calls[0].keywords}["value"]
    assert isinstance(value, ast.Constant) and value.value is False
    registered = [
        node for node in ast.walk(adui)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Subscript)
        and getattr(node.targets[0].value, "id", "") == "_MAIN_ACCORDIONS"
        and getattr(node.value, "id", "") == "ad_enable"
    ]
    assert len(registered) == 1


def test_script_inputs_keep_enable_skip_and_tabs_in_this_order():
    # is_ad_enabled, set_skip_img2img and _drop_api_default_tab read these
    # inputs by position, and API requests send them by position: with the
    # first two swapped, Skip img2img would be read as Enable.
    adui = _adui_tree()
    assigned = [
        node for node in ast.walk(adui)
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", "") == "components"
    ]
    assert len(assigned) == 1
    assert ast.unparse(assigned[0].value) == "[ad_enable, ad_skip_img2img, *states]"
    # The names in that list are bound once (the Enable accordion and the Skip
    # checkbox), and the tabs are only appended, in order.
    for name in ("ad_enable", "ad_skip_img2img", "states"):
        bound = [
            node for node in ast.walk(adui)
            if isinstance(node, ast.Name) and node.id == name
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ]
        assert len(bound) == 1, name
    on_states = {
        node.func.attr for node in ast.walk(adui)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and getattr(node.func.value, "id", "") == "states"
    }
    assert on_states == {"append"}
    assert not [
        node for node in ast.walk(adui)
        if isinstance(node, ast.Subscript) and getattr(node.value, "id", "") == "states"
        and isinstance(node.ctx, (ast.Store, ast.Del))
    ]
    # Nothing between the list and the return can reorder it (an insert, a
    # reverse, a slice assignment): the list is built right before returning.
    assert adui.body[-2] is assigned[0]
    returned = adui.body[-1]
    assert isinstance(returned, ast.Return)
    assert isinstance(returned.value, ast.Tuple)
    assert [node.id for node in returned.value.elts] == ["components", "infotext_fields"]
    # The script's ui() hands that list to the WebUI as it is.
    script = ast.parse(_SCRIPT_PATH.read_text(encoding="utf-8"))
    cls = next(
        node for node in script.body
        if isinstance(node, ast.ClassDef) and node.name == "AfterDetailerScript"
    )
    ui = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "ui"
    )
    names = sorted(
        (
            node for node in ast.walk(ui)
            if isinstance(node, ast.Name) and node.id == "components"
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    assert [type(node.ctx).__name__ for node in names] == ["Store", "Load"]
    assert ast.unparse(ui.body[-1]) == "return components"
    unpacked = next(
        node for node in ui.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple)
    )
    assert [node.id for node in unpacked.targets[0].elts] == ["components", "infotext_fields"]
    assert getattr(unpacked.value.func, "id", "") == "adui"


class _OptionInfo:
    def __init__(self, default=None, label="", component=None, component_args=None, section=None, **_kw):
        self.default = default
        self.label = label
        self.component = component
        self.component_args = component_args or {}
        self.section = section
        self.info_text = None
        self.reload = False

    def info(self, text):
        self.info_text = text
        return self

    def needs_reload_ui(self):
        self.reload = True
        return self


def _registered_options(ui, **host):
    options = {}
    shared = SimpleNamespace(
        opts=SimpleNamespace(add_option=lambda key, info: options.__setitem__(key, info)),
        OptionInfo=_OptionInfo,
        **host,
    )
    gr = SimpleNamespace(
        Slider="Slider", Checkbox="Checkbox", Textbox="Textbox", Radio="Radio",
        Dropdown="Dropdown",
    )
    script = _load(
        _SCRIPT_PATH,
        functions=("on_ui_settings",),
        shared=shared,
        gr=gr,
        ADETAILER="ADetailer",
        OptionDiv=None,
        SCRIPT_DEFAULT="",
        BBOX_SORTBY=["None"],
        BBOX_SORTBY_MORE=["Off"],
        INPAINT_BBOX_MATCH_MODES=["Off"],
        InpaintBBoxMatchMode=SimpleNamespace(OFF=SimpleNamespace(value="Off")),
        _make_reset_settings_button=object(),
        ENABLE_AT_STARTUP_CHOICES=ui.ENABLE_AT_STARTUP_CHOICES,
        AD_CROP_UPSCALER_WEBUI="Use the WebUI's img2img upscaler",
        _crop_upscaler_choices=lambda: {"choices": []},
    )
    script.on_ui_settings()
    return options


def test_option_is_registered_off_by_default(ui):
    info = _registered_options(ui)["ad_enable_at_startup"]

    assert info.default == "Off"
    assert info.component == "Radio"
    assert info.component_args["choices"] == ["Off", "txt2img", "img2img", "Both"]
    assert info.section == ("ADetailer", "ADetailer")
    assert info.reload is True
    # Translatable text: one plain block, no inline markdown.
    for text in (info.label, info.info_text):
        assert not any(mark in text for mark in ("**", "`", "[", "](", "\n"))


@pytest.mark.parametrize("visible", [False, True])
@pytest.mark.parametrize("key", ["ad_extra_models_dir", "ad_save_images_dir"])
def test_folder_options_follow_the_webui_hide_dirs(ui, key, visible):
    # With --hide-ui-dir-config the WebUI sets hide_dirs to visible False: its
    # setter then refuses the option, so a guest cannot point it at any folder
    # of the server. Only component_args carries that to the WebUI.
    info = _registered_options(ui, hide_dirs={"visible": visible})[key]

    assert info.component_args == {"visible": visible}
    assert info.component == "Textbox"


@pytest.mark.parametrize("key", ["ad_extra_models_dir", "ad_save_images_dir"])
def test_folder_options_register_on_a_host_without_hide_dirs(ui, key):
    assert _registered_options(ui)[key].component_args == {}


def _on_ui_tabs(data, apply):
    return _load(
        _SCRIPT_PATH,
        functions=("on_ui_tabs",),
        shared=SimpleNamespace(opts=SimpleNamespace(data=data)),
        apply_enable_at_startup=apply,
    ).on_ui_tabs


def test_ui_tabs_callback_applies_the_saved_choice():
    seen = []

    assert _on_ui_tabs({"ad_enable_at_startup": "Both"}, seen.append)() == []
    assert _on_ui_tabs({}, seen.append)() == []

    assert seen == ["Both", "Off"]


def test_ui_tabs_callback_survives_a_failure(capsys):
    def boom(_choice):
        raise RuntimeError("broken")

    assert _on_ui_tabs({"ad_enable_at_startup": "Both"}, boom)() == []
    err = capsys.readouterr().err
    assert "Enable ADetailer at startup" in err
