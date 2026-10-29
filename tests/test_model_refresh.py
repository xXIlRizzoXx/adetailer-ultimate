from __future__ import annotations

import ast
import weakref
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]
_UI_PATH = _ROOT / "aaaaaa" / "ui.py"


def _ui_functions():
    source = ast.parse(_UI_PATH.read_text(encoding="utf-8"))
    names = {"_model_choices", "_sync_model_choices", "_wire_model_refresh"}
    body = [
        node for node in source.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                getattr(t, "id", "") == "_MODEL_DROPDOWNS"
                for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
            )
        )
    ]
    namespace = {"gr": SimpleNamespace(update=lambda **kw: kw), "weakref": weakref}
    module = ast.Module(body=[source.body[0], *body], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_UI_PATH), "exec"), namespace)
    return namespace


class _Dropdown:
    def __init__(self, choices):
        self.choices = choices


class _Button:
    def click(self, **kwargs):
        self.kwargs = kwargs


def _tabs(*dropdowns):
    return [SimpleNamespace(ad_model=dd, ad_model_refresh=_Button()) for dd in dropdowns]


def test_the_refresh_button_updates_the_detector_menu_of_every_tab():
    # Issue #8: a detector added or removed while the WebUI runs needed a
    # restart (or Settings > Reload UI) to show up in the menu.
    ns = _ui_functions()
    # Gradio 3 keeps the choices as names, Gradio 4 as (label, value) pairs.
    first = _Dropdown(["a.pt", "gone.pt", "None"])
    second = _Dropdown([("None", "None"), ("a.pt", "a.pt"), ("gone.pt", "gone.pt")])
    ns["_MODEL_DROPDOWNS"][:] = [weakref.ref(first), weakref.ref(second)]
    tabs = _tabs(first, second)
    reads = []
    info = SimpleNamespace(
        refresh_models=lambda: reads.append(1) or ["a.pt", "new.pt"], model_mapping={}
    )

    ns["_wire_model_refresh"](tabs, info, 2)

    # One click per tab, each updating the menus of every tab.
    assert [t.ad_model_refresh.kwargs["outputs"] for t in tabs] == [[first, second]] * 2
    # It reads every tab's detector, and never waits in the queue behind a
    # running generation.
    assert [t.ad_model_refresh.kwargs["inputs"] for t in tabs] == [[first, second]] * 2
    assert all(t.ad_model_refresh.kwargs["queue"] is False for t in tabs)
    cfg1, cfg2 = first.choices, second.choices
    fn = tabs[1].ad_model_refresh.kwargs["fn"]
    updates = fn("new.pt", "gone.pt")
    assert reads == [1]
    # A detector still installed stays, even when it is not the first choice;
    # a removed one gives way to the first choice, as at startup ("None" is
    # last on the 1st tab, first after).
    assert updates == [
        {"choices": ["a.pt", "new.pt", "None"], "value": "new.pt"},
        {"choices": ["None", "a.pt", "new.pt"], "value": "None"},
    ]
    assert [u["value"] for u in fn("None", "a.pt")] == ["None", "a.pt"]
    # A browser reload is served these lists: they are refreshed too, in
    # place, since the startup page config holds these very list objects.
    assert first.choices is cfg1 and second.choices is cfg2
    assert cfg1 == ["a.pt", "new.pt", "None"]
    assert cfg2 == [("None", "None"), ("a.pt", "a.pt"), ("new.pt", "new.pt")]


def test_a_failed_refresh_keeps_the_detector_menus():
    ns = _ui_functions()
    dd = _Dropdown(["a.pt", "None"])
    ns["_MODEL_DROPDOWNS"][:] = [weakref.ref(dd)]
    tabs = _tabs(dd)

    def broken():
        raise OSError("disk")

    ns["_wire_model_refresh"](tabs, SimpleNamespace(refresh_models=broken, model_mapping={}), 1)

    assert tabs[0].ad_model_refresh.kwargs["fn"]("a.pt") == [{}]
    assert dd.choices == ["a.pt", "None"]


def test_every_tab_gets_a_refresh_button_with_a_tooltip():
    source = _UI_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    group = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "one_ui_group"
    )
    assert "eid('ad_model_refresh')" in ast.unparse(group)
    # Every detector menu is kept for the refresh, so a page reload after the
    # 🔄 lists the new detectors too.
    assert "_MODEL_DROPDOWNS.append(weakref.ref(w.ad_model))" in ast.unparse(group)
    adui = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "adui"
    )
    assert "_wire_model_refresh(all_widgets, webui_info, num_models)" in ast.unparse(adui)
    tooltips = (_ROOT / "javascript" / "button-tooltips.js").read_text(encoding="utf-8")
    assert "adetailer_ad_model_refresh:" in tooltips


def test_the_refresh_tooltip_names_the_side_it_updates():
    # The 🔄 updates the menus of its own side (txt2img or img2img) only: the
    # tooltip said "every tab", so the other side's menus looked broken.
    import re

    tooltips = (_ROOT / "javascript" / "button-tooltips.js").read_text(encoding="utf-8")
    text = re.search(r'adetailer_ad_model_refresh:\s*"([^"]*)"', tooltips).group(1)
    assert "this side" in text
    assert "page reload" in text
    assert "every tab." not in text
