"""The guide "several subjects, one LoRA each": components and wiring.

The functions are taken out of aaaaaa/ui.py and scripts/!adetailer.py by
name and run with stand-ins for Gradio and the WebUI: no server is started.
"""

from __future__ import annotations

import ast
import re
import weakref
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from adetailer import subject_guide
from adetailer.args import ALL_ARGS, BBOX_SORTBY, BBOX_SORTBY_MORE
from adetailer.classes import MEDIAPIPE_FACE_FEATURES_MODEL, is_yoloe_model
from adetailer.subject_guide import (
    GUIDE_ATTRS,
    GUIDE_MODES,
    LEFT_TO_RIGHT,
    MODE_FACE,
    MODE_PERSON,
    MODE_PERSON_FACE,
)

_ROOT = Path(__file__).resolve().parents[1]
# The last line of the status when the 1st tab is filled.
DENOISE_LINE = (
    "ℹ️ The guide set Inpaint denoising strength to 0.6 in the 1st tab. The "
    "subjects are easier to recognise when the main prompt already describes "
    "them (colours, clothes, who is on the left and who on the right) without "
    "their LoRAs."
)
_UI_PATH = _ROOT / "aaaaaa" / "ui.py"
_SCRIPT_PATH = _ROOT / "scripts" / "!adetailer.py"

_UI_NAMES = {
    "subject_guide_ui", "_wire_subject_guide", "_guide_lora_names",
    "_restored_tab_updates", "_added_model_update", "_model_choices", "_md_safe",
    "elem_id", "suffix", "ordinal", "_subject_row_id", "_sync_lora_choices",
    "_hidden_if_unfinished", "_no_latex",
}
_UI_VALUES = {
    "_SUBJECT_GUIDE_INTRO", "_SUBJECT_GUIDE_MODE_INFO", "_SUBJECT_GUIDE_PEOPLE_INFO",
    "_GUIDE_LORA_DROPDOWNS",
}


class _Component:
    def __init__(self, kind, *args, **kwargs):
        self.kind = kind
        self.args = args
        self.kwargs = kwargs
        self.clicks = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def click(self, **kwargs):
        self.clicks.append(kwargs)

    def __repr__(self):
        return f"<{self.kind} {self.kwargs.get('label') or self.kwargs.get('elem_id')}>"


class _Gradio:
    """Records every component built; gr.update gives the plain dict."""

    def __init__(self):
        self.created = []

    @staticmethod
    def update(**kwargs):
        return kwargs

    def __getattr__(self, kind):
        def make(*args, **kwargs):
            component = _Component(kind, *args, **kwargs)
            self.created.append(component)
            return component

        return make


def _top_level(path, names, values=()):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return tree, [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                getattr(t, "id", "") in values
                for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
            )
        )
    ]


def _ui(class_names=None):
    tree, body = _top_level(_UI_PATH, _UI_NAMES, _UI_VALUES)
    assert {getattr(n, "name", None) for n in body} >= _UI_NAMES
    gr = _Gradio()
    printed = []
    namespace = {
        "gr": gr,
        "partial": partial,
        "weakref": weakref,
        "contextmanager": contextmanager,
        "SimpleNamespace": SimpleNamespace,
        "ALL_ARGS": ALL_ARGS,
        "MEDIAPIPE_FACE_FEATURES_MODEL": MEDIAPIPE_FACE_FEATURES_MODEL,
        "is_yoloe_model": is_yoloe_model,
        "get_model_class_names": class_names or (lambda path: ["person"] if path else []),
        "print": lambda *args: printed.append(" ".join(map(str, args))),
        **{
            name: getattr(subject_guide, name)
            for name in (
                "DEFAULT_WEIGHT", "GUIDE_ATTRS", "GUIDE_MODES", "GUIDE_SUBJECTS",
                "NO_LORA", "PEOPLE_CHOICES", "build_subject_guide",
                "host_comment_marks", "lora_choices",
            )
        },
    }
    module = ast.Module(body=[tree.body[0], *body], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_UI_PATH), "exec"), namespace)
    namespace["printed"] = printed
    return namespace


MAPPING = {
    "face_yolov8n.pt": "face.pt",
    "person_yolov8n-seg.pt": "person.pt",
    "mediapipe_face_full": "mediapipe_face_full",
}


def _hold_choices(menus, pairs=False):
    """Give each stand-in menu the `choices` list a Gradio Dropdown keeps,
    which the page config of a browser reload is made from: plain names on
    Gradio 3, (label, value) pairs on Gradio 4."""
    for menu in menus:
        names = menu.kwargs["choices"]
        menu.choices = [(n, n) for n in names] if pairs else list(names)


def _page_choices(menu):
    return [c[1] if isinstance(c, tuple) else c for c in menu.choices]


def _panel(tabs, *, main_prompt=None, lora_names=("bob", "alice"), sort=(LEFT_TO_RIGHT, "Off"),
           mapping=MAPPING, class_names=None, pairs=False, **more):
    ns = _ui(class_names)
    info = SimpleNamespace(
        model_mapping=mapping,
        lora_names=(lambda: list(lora_names)) if not callable(lora_names) else lora_names,
        sort_order=(lambda: sort) if not callable(sort) else sort,
        main_prompt=main_prompt,
        **more,
    )
    guide = ns["subject_guide_ui"](False, ns["_guide_lora_names"](info))
    _hold_choices(guide.loras, pairs)
    widgets = [
        SimpleNamespace(
            **{attr: _Component(f"tab{n}.{attr}") for attr in ALL_ARGS.attrs},
            ad_model_classes_dropdown=_Component(f"tab{n}.classes"),
        )
        for n in range(tabs)
    ]
    ad_enable = _Component("enable")
    ns["_wire_subject_guide"](guide, ad_enable, widgets, tabs, info)
    return SimpleNamespace(
        ns=ns, gr=ns["gr"], guide=guide, widgets=widgets, ad_enable=ad_enable, info=info,
        click=guide.button.clicks[0] if guide.button.clicks else None,
    )


def _values(panel, mode=MODE_PERSON, subjects=None, enabled=None, prompts=None, main=None,
            people="Auto"):
    subjects = subjects or [("alice", 0.8, "woman, red hair"), ("bob", 0.8, "man")]
    subjects = [*subjects, *[("None", 0.8, "")] * 4][:4]
    tabs = len(panel.widgets)
    values = [mode, people, *(v for row in subjects for v in row)]
    values += list(enabled or [n == 0 for n in range(tabs)])
    values += list(prompts or [""] * min(tabs, 2))
    if panel.info.main_prompt is not None:
        values.append(main if main is not None else "")
    assert len(values) == len(panel.click["inputs"])
    return values


def _answer(panel, **kwargs):
    answer = panel.click["fn"](*_values(panel, **kwargs))
    assert isinstance(answer, list)
    # One answer per output, whatever happened.
    assert len(answer) == len(panel.click["outputs"])
    return answer


def _split(panel, answer):
    """(status, enable, {tab: {attr: update}}, [classes update per tab], menus)."""
    targets = min(len(panel.widgets), 2)
    n = len(GUIDE_ATTRS)
    per_tab = {
        t: dict(zip(GUIDE_ATTRS, answer[2 + t * n : 2 + (t + 1) * n])) for t in range(targets)
    }
    classes = answer[2 + targets * n : 2 + targets * n + targets]
    menus = answer[2 + targets * n + targets :]
    assert len(menus) == 4
    return answer[0], answer[1], per_tab, classes, menus


# ---------------------------------------------------------------- components


def test_the_guide_is_a_closed_box_with_four_subject_rows():
    panel = _panel(2)
    kinds = [c.kind for c in panel.gr.created]
    assert kinds.count("Accordion") == 1
    assert kinds.count("Radio") == 2
    assert kinds.count("Dropdown") == 4
    assert kinds.count("Slider") == 4
    assert kinds.count("Textbox") == 4
    assert kinds.count("Button") == 1
    assert kinds.count("Markdown") == 2
    assert kinds.count("Row") == 4
    assert len(kinds) == 22

    accordion = panel.gr.created[0]
    assert accordion.kind == "Accordion"
    assert accordion.args == ("Guide: several subjects, one LoRA each",)
    assert accordion.kwargs["open"] is False

    radio = panel.guide.mode
    assert radio.kwargs["label"] == "What to redraw"
    assert radio.kwargs["choices"] == list(GUIDE_MODES) == [
        "Whole person (outline)", "Face only", "Whole person, then face (2 tabs)",
    ]
    assert radio.kwargs["value"] == "Whole person (outline)"
    assert radio.kwargs["info"] == (
        "Whole person redraws each person inside their outline, so body and "
        "clothes follow the LoRA too. Face only redraws the faces. The last choice "
        "redraws each person on the 1st tab and then their face on the 2nd."
    )

    people = panel.guide.people
    assert people.kind == "Radio"
    assert people.kwargs["label"] == "People in the picture"
    assert people.kwargs["choices"] == ["Auto", "2", "3", "4"]
    assert people.kwargs["value"] == "Auto"
    assert people.kwargs["info"] == (
        "How many people the guide counts: the biggest ones, from left to right. "
        "Auto counts up to the last row filled. Set it when the picture has more "
        "people than rows filled: the rows then go to the leftmost of the people "
        "counted, not to the biggest ones."
    )
    # Not "so that the others are left as they are": with Auto the others are
    # left as they are too. What the number changes is who gets the rows.
    assert "left as they are" not in people.kwargs["info"]
    row = [("alice", 0.8, "woman")]
    for count, who in (("Auto", "the biggest person"), ("2", "the leftmost of the 2 biggest people")):
        result = _ui()["build_subject_guide"](
            MODE_PERSON, row, installed_models=list(MAPPING), num_tabs=2,
            sort_order=LEFT_TO_RIGHT, people=count,
        )
        assert result.tabs[0]["ad_prompt"].count("<lora:") == 1
        assert f"Subject 1 is {who}." in result.status
    # Between "What to redraw" and the rows.
    created = panel.gr.created
    assert created.index(radio) < created.index(people) < created.index(panel.guide.loras[0])

    for i, (lora, weight, words) in enumerate(
        zip(panel.guide.loras, panel.guide.weights, panel.guide.words), start=1
    ):
        assert lora.kwargs["label"] == f"Subject {i} LoRA"
        assert lora.kwargs["choices"] == ["None", "alice", "bob"]
        assert lora.kwargs["value"] == "None"
        # A LoRA added since the start, or a WebUI without a LoRA list.
        assert lora.kwargs["allow_custom_value"] is True
        assert weight.kwargs["label"] == f"Subject {i} weight"
        assert (weight.kwargs["minimum"], weight.kwargs["maximum"]) == (0.0, 2.0)
        assert weight.kwargs["step"] == 0.05
        # 1 from the start (a live test, 19/11/2026: at 0.8 the characters
        # were less recognisable), written "1" in the tag.
        assert weight.kwargs["value"] == 1
        assert subject_guide.lora_tag("alice", weight.kwargs["value"]) == "<lora:alice:1>"
        assert words.kwargs["label"] == f"Subject {i} words"
        assert words.kwargs["placeholder"] == "woman, red hair, green dress"
        assert words.kwargs["value"] == ""
    # A slider left as it is gives the tag "<lora:alice:1>" on a click.
    start = panel.guide.weights[0].kwargs["value"]
    tabs = _split(panel, _answer(panel, subjects=[("alice", start, "woman")]))[2]
    assert tabs[0]["ad_prompt"] == {"value": "woman, <lora:alice:1> [SEP] [SKIP]"}
    # Each menu has a choices list of its own.
    lists = [lora.kwargs["choices"] for lora in panel.guide.loras]
    assert len({id(x) for x in lists}) == 4

    assert panel.guide.button.kwargs["value"] == "✨ Fill the tabs"
    assert panel.guide.status.kwargs["value"] == ""


def test_the_status_of_the_guide_reads_no_formula():
    # The status quotes LoRA names and words as typed. Gradio's Markdown
    # reads "$...$" (3.41, in the page it has drawn) or "$$...$$" (4.40) as a
    # formula: a name such as a$b$c was not shown as typed. The component
    # asks for no formula at all where Gradio has the parameter for it.
    def build(markdown):
        ns = _ui()
        gr = ns["gr"]

        def make(*args, **kwargs):
            component = _Component("Markdown", *args, **kwargs)
            gr.created.append(component)
            return component

        gr.Markdown = markdown(make)
        return ns["subject_guide_ui"](False, []).status.kwargs

    def gradio_3_41_and_4(make):
        def markdown(value="", *, latex_delimiters=None, elem_id=None, elem_classes=None):
            return make(
                value=value, latex_delimiters=latex_delimiters, elem_id=elem_id,
                elem_classes=elem_classes,
            )

        return markdown

    def older_gradio(make):
        def markdown(value="", *, elem_id=None, elem_classes=None):
            return make(value=value, elem_id=elem_id, elem_classes=elem_classes)

        return markdown

    assert build(gradio_3_41_and_4)["latex_delimiters"] == []
    # A Gradio without the parameter still builds the panel.
    older = build(older_gradio)
    assert "latex_delimiters" not in older
    assert older["elem_classes"] == ["ad-subject-guide-status"]
    no_latex = _ui()["_no_latex"]
    for odd in (None, 3, "Markdown", lambda *args, **kwargs: None, len):
        assert no_latex(odd) == {}
    # The status alone: the intro is the guide's own text.
    source = _UI_PATH.read_text(encoding="utf-8")
    assert source.count("**_no_latex(gr.Markdown)") == 1


def test_the_status_of_the_guide_does_not_fade_away():
    # preset-status-fade.js hides every "ad-preset-status" line a few seconds
    # after it appears: the guide's warnings have to stay on screen.
    fade = (_ROOT / "javascript" / "preset-status-fade.js").read_text(encoding="utf-8")
    assert 'const WATCH_CLASS = "ad-preset-status";' in fade
    panel = _panel(2)
    for component in panel.gr.created:
        classes = component.kwargs.get("elem_classes") or []
        assert "ad-preset-status" not in classes
        assert not any(c in fade for c in classes), classes


def test_every_component_of_the_guide_has_an_id_of_its_side():
    for is_img2img, side in ((False, "txt2img"), (True, "img2img")):
        ns = _ui()
        ns["subject_guide_ui"](is_img2img, [])
        ids = [c.kwargs.get("elem_id") for c in ns["gr"].created if c.kind not in ("Row", "Markdown")]
        assert all(i and i.startswith(f"script_{side}_adetailer_ad_subject_guide") for i in ids)
        assert len(set(ids)) == len(ids) == 16


def test_every_component_of_the_guide_stays_out_of_ui_config():
    # ui-config.json tracks labelled components by label: without the flag
    # the WebUI would bring the guide's choices back at every start.
    panel = _panel(2)
    built = [c for c in panel.gr.created if c.kind != "Row"]
    assert len(built) == 18
    assert all(getattr(c, "do_not_save_to_config", False) is True for c in built)
    assert {id(c) for c in panel.guide.components} == {id(c) for c in built}


def test_the_guide_lists_no_lora_when_the_webui_has_no_list():
    def broken():
        raise RuntimeError("no LoRA module")

    for names in ([], None, broken, lambda: None, lambda: [3, None, "ok"]):
        ns = _ui()
        info = SimpleNamespace(lora_names=names)
        found = ns["_guide_lora_names"](info)
        assert found == (["ok"] if found else [])
        guide = ns["subject_guide_ui"](False, found)
        assert guide.loras[0].kwargs["choices"] == ["None", *found]
    assert _ui()["_guide_lora_names"](SimpleNamespace()) == []


def _texts():
    panel = _panel(2)
    texts = []
    for c in panel.gr.created:
        texts += [a for a in c.args if isinstance(a, str)]
        # The value of a menu or of the choice is not a text of its own.
        keys = ["label", "placeholder", "info"]
        if c.kind not in ("Dropdown", "Radio"):
            keys.append("value")
        texts += [c.kwargs[k] for k in keys if isinstance(c.kwargs.get(k), str)]
        if c.kind == "Radio":
            texts += c.kwargs["choices"]
    return [t for t in texts if t]


def test_every_text_of_the_guide_is_one_plain_block():
    texts = _texts()
    assert "Guide: several subjects, one LoRA each" in texts
    # Box, intro, the choice (label, info, 3 choices), the people (label,
    # info, 4 choices), 4 rows of 3 labels and an example, the button.
    assert len(texts) == 1 + 1 + 5 + 6 + 4 * 4 + 1
    for text in texts:
        # The WebUI's localization replaces whole text nodes: inline
        # markdown would split a text into pieces no translation matches.
        assert not re.search(r"\*|`|_{2}|\[[^\]]*\]\(|\n|<|#", text), text
        assert text == text.strip()


def test_the_intro_says_what_the_button_does():
    intro = _ui()["_SUBJECT_GUIDE_INTRO"]
    assert _texts().count(intro) == 1
    # Two sets of choices are below the intro: "the last choice below" could
    # be the "4" of People in the picture. The set is named.
    assert "last choice below" not in intro
    radios = [c for c in _panel(2).gr.created if c.kind == "Radio"]
    assert [r.kwargs["label"] for r in radios] == ["What to redraw", "People in the picture"]
    assert radios[0].kwargs["choices"][-1] == MODE_PERSON_FACE
    for said in (
        "fills the 1st tab (and the 2nd with the last choice of What to redraw)",
        # The denoise the button writes (subject_guide.GUIDE_DENOISE), and
        # the main prompt that makes it enough: a live test on 19/11/2026.
        "The prompt those tabs had is replaced, and their Inpaint denoising "
        f"strength is set to {subject_guide.GUIDE_DENOISE:g}.",
        "The subjects are easier to recognise when the main prompt already "
        "describes them: colours, clothes, who is on the left and who on the right.",
        # Not "the leftmost" alone: the biggest people, then left to right.
        "The subjects are the biggest people in the picture, as many as People "
        "in the picture below says, counted from left to right: Subject 1 is "
        "the leftmost of them.",
        "Sort bounding boxes by set to Position (left to right)",
        "Leave a row empty to leave that subject as it is.",
        "Keep these LoRAs out of the main prompt.",
    ):
        assert said in intro
    assert LEFT_TO_RIGHT in BBOX_SORTBY
    # The setting's own label, as Settings shows it.
    assert 'label="Sort bounding boxes by"' in _SCRIPT_PATH.read_text(encoding="utf-8")


def test_the_options_of_the_guide_explain_themselves():
    # The panel's rule (tests/test_frontend.py): an option has an info line
    # or a tooltip. The choice has an info line; the rows have their label,
    # an example and the text above them, and take no tooltip key.
    from tests.test_frontend import _option_tooltips, _panel_widgets

    widgets = {key: (kind, has_info) for key, kind, has_info in _panel_widgets()}
    assert widgets["ad_subject_guide_mode"] == ("Radio", True)
    assert widgets["ad_subject_guide_people"] == ("Radio", True)
    guide_keys = [key for key in widgets if key.startswith("ad_subject_guide")]
    assert sorted(guide_keys) == [
        "ad_subject_guide", "ad_subject_guide_fill", "ad_subject_guide_mode",
        "ad_subject_guide_people", "ad_subject_guide_status",
    ]
    assert not [key for key in _option_tooltips() if "subject_guide" in key]
    panel = _panel(2)
    for words in panel.guide.words:
        assert words.kwargs["placeholder"]
    for field in (*panel.guide.loras, *panel.guide.weights, *panel.guide.words):
        assert field.kwargs["label"].startswith("Subject ")
        assert "info" not in field.kwargs


# -------------------------------------------------------------------- wiring


def test_building_the_guide_registers_no_listener():
    ns = _ui()
    ns["subject_guide_ui"](False, ["alice"])
    assert all(c.clicks == [] for c in ns["gr"].created)


@pytest.mark.parametrize("tabs", [1, 2, 3, 15])
def test_the_button_has_one_click_that_never_waits_in_the_queue(tabs):
    panel = _panel(tabs)
    assert len(panel.guide.button.clicks) == 1
    assert sum(len(c.clicks) for c in panel.gr.created) == 1
    assert not any(c.clicks for w in panel.widgets for c in vars(w).values())
    assert panel.ad_enable.clicks == []
    assert set(panel.click) == {"fn", "inputs", "outputs", "queue"}
    assert panel.click["queue"] is False


@pytest.mark.parametrize("tabs", [1, 2, 3, 15])
@pytest.mark.parametrize("with_main", [False, True])
def test_what_the_click_reads_and_writes(tabs, with_main):
    main = _Component("main prompt") if with_main else None
    panel = _panel(tabs, main_prompt=main)
    guide, widgets = panel.guide, panel.widgets
    targets = widgets[:2]
    rows = [c for row in zip(guide.loras, guide.weights, guide.words) for c in row]
    assert panel.click["inputs"] == [
        guide.mode,
        guide.people,
        *rows,
        *(w.ad_tab_enable for w in widgets),
        *(w.ad_prompt for w in targets),
        *([main] if with_main else []),
    ]
    assert panel.click["outputs"] == [
        guide.status,
        panel.ad_enable,
        *(getattr(w, attr) for w in targets for attr in GUIDE_ATTRS),
        *(w.ad_model_classes_dropdown for w in targets),
        *guide.loras,
    ]
    # No output twice, the main prompt never written, no tab after the 2nd.
    outputs = panel.click["outputs"]
    assert len({id(c) for c in outputs}) == len(outputs)
    assert all(c is not main for c in outputs)
    written = {id(c) for c in outputs}
    for w in widgets[2:]:
        assert not written & {id(c) for c in vars(w).values()}
    # Of a written tab, only the guide's settings.
    for w in targets:
        others = [c for attr, c in vars(w).items() if attr not in (*GUIDE_ATTRS, "ad_model_classes_dropdown")]
        assert len(others) == len(ALL_ARGS.attrs) - len(GUIDE_ATTRS)
        assert not written & {id(c) for c in others}


@pytest.mark.parametrize("tabs", [1, 2, 4])
@pytest.mark.parametrize("mode", GUIDE_MODES)
@pytest.mark.parametrize("with_main", [False, True])
def test_every_output_gets_an_update_of_its_own(tabs, mode, with_main):
    panel = _panel(tabs, main_prompt=_Component("main") if with_main else None)
    for kwargs in (
        {"mode": mode},
        {"mode": mode, "subjects": [("None", 0.8, "")]},  # nothing chosen
        {"mode": mode, "subjects": [("a:b", 0.8, "")]},  # a name refused
        {"mode": None},
    ):
        answer = _answer(panel, **kwargs)
        assert isinstance(answer[0], str)
        updates = answer[1:]
        assert all(isinstance(u, dict) for u in updates)
        # Gradio 4 empties an update while it sends it: one shared by two
        # outputs would reach only the first.
        assert len({id(u) for u in updates}) == len(updates)


def test_whole_person_fills_the_first_tab_and_switches_adetailer_on():
    panel = _panel(3)
    status, enable, tabs, classes, menus = _split(
        panel, _answer(panel, prompts=["old", "kept"], enabled=[False, True, False])
    )
    prompt = "woman, red hair, <lora:alice:0.8> [SEP] man, <lora:bob:0.8> [SEP] [SKIP]"
    assert status.split("\n\n") == [
        "✅ Filled the 1st tab for 2 subjects with person_yolov8n-seg.pt. "
        "Subject 1 is the leftmost of the 2 biggest people.",
        "ℹ️ The prompt the 1st tab had was replaced.",
        "ℹ️ The 2nd tab is switched on too: it runs as well when it has a "
        "detector.",
        DENOISE_LINE,
    ]
    assert enable == {"value": True}
    assert tabs[0] == {
        "ad_tab_enable": {"value": True},
        # Its menu lists the detectors of now, "None" last on the 1st tab.
        "ad_model": {
            "choices": ["face_yolov8n.pt", "person_yolov8n-seg.pt", "mediapipe_face_full", "None"],
            "value": "person_yolov8n-seg.pt",
        },
        "ad_model_classes": {"value": "", "visible": False},
        "ad_model_classes_exclude": {"value": False, "visible": True},
        "ad_model_classes_excluded": {"value": ""},
        "ad_classes_sequential": {"value": False},
        "ad_prompt": {"value": prompt},
        "ad_strip_loras": {"value": False},
        "ad_use_bbox_mask": {"value": False},
        "ad_mask_k": {"value": 2},
        "ad_mask_filter_method": {"value": "Area"},
        "ad_mask_merge_invert": {"value": "None"},
        "ad_inpaint_indices": {"value": ""},
        "ad_inpaint_only_masked": {"value": True},
        "ad_denoising_strength": {"value": 0.6},
    }
    # The classes menu follows the detector, as after Load preset.
    assert classes[0] == {"choices": ["person"], "value": [], "visible": True}
    # The 2nd tab is not this choice's: nothing of it changes.
    assert tabs[1] == {attr: {} for attr in GUIDE_ATTRS}
    assert classes[1] == {}
    # The LoRA list is the one the menus were built with: nothing to send.
    assert menus == [{}] * 4


def test_people_in_the_picture_is_one_more_input_of_the_same_click():
    # With two people and only the 1st row filled, "2" keeps the two biggest
    # and skips the second: the LoRA goes to the left one. No new listener.
    panel = _panel(2)
    assert panel.click["inputs"][1] is panel.guide.people
    assert panel.guide.people not in panel.click["outputs"]
    assert sum(len(c.clicks) for c in panel.gr.created) == 1
    one_row = [("alice", 0.8, "woman")]
    for people, prompt, k, said in (
        ("Auto", "woman, <lora:alice:0.8> [SEP] [SKIP]", 1,
         "for 1 subject with person_yolov8n-seg.pt. Subject 1 is the biggest person."),
        ("2", "woman, <lora:alice:0.8> [SEP] [SKIP] [SEP] [SKIP]", 2,
         "for 2 subjects with person_yolov8n-seg.pt (subject 2 stays as it is). "
         "Subject 1 is the leftmost of the 2 biggest people."),
        ("4", "woman, <lora:alice:0.8> [SEP] [SKIP] [SEP] [SKIP] [SEP] [SKIP] [SEP] [SKIP]", 4,
         "for 4 subjects with person_yolov8n-seg.pt (subjects 2, 3 and 4 stay as "
         "they are). Subject 1 is the leftmost of the 4 biggest people."),
        # Nothing picked, a translated text, a number: Auto.
        (None, "woman, <lora:alice:0.8> [SEP] [SKIP]", 1, "Subject 1 is the biggest person."),
        ("Automatico", "woman, <lora:alice:0.8> [SEP] [SKIP]", 1, "the biggest person."),
        (3, "woman, <lora:alice:0.8> [SEP] [SKIP]", 1, "the biggest person."),
    ):
        status, enable, tabs, _, _ = _split(
            panel, _answer(panel, subjects=one_row, people=people)
        )
        assert tabs[0]["ad_prompt"] == {"value": prompt}
        assert tabs[0]["ad_mask_k"] == {"value": k}
        if said.startswith("for"):
            assert status.split("\n\n") == ["✅ Filled the 1st tab " + said, DENOISE_LINE]
        else:
            assert said in status
        assert enable == {"value": True}


def test_after_two_tabs_a_one_tab_choice_tells_the_second_tab_left_on():
    # The whole path: "person, then face" fills both tabs; then "Whole
    # person" with the 2nd tab as that click left it (on, same prompt).
    panel = _panel(3)
    _, _, tabs, _, _ = _split(panel, _answer(panel, mode=MODE_PERSON_FACE))
    first, second = (tabs[t]["ad_prompt"]["value"] for t in (0, 1))
    assert tabs[1]["ad_tab_enable"] == {"value": True}
    status, _, tabs, classes, _ = _split(
        panel,
        _answer(panel, mode=MODE_PERSON, enabled=[True, True, False], prompts=[first, second]),
    )
    assert (
        "⚠️ The 2nd tab is still switched on and its prompt holds [SEP] or a "
        "LoRA, maybe from an earlier fill: when it has a detector it runs too, "
        "with that prompt. The guide did not change it. Switch it off or clear "
        "its prompt if you do not want that."
    ) in status.split("\n\n")
    assert "switched on too" not in status
    # Told, not touched: every output of the 2nd tab changes nothing.
    assert tabs[1] == {attr: {} for attr in GUIDE_ATTRS}
    assert classes[1] == {}
    # With the 2nd tab switched off there is nothing to tell.
    status = _answer(
        panel, mode=MODE_PERSON, enabled=[True, False, False], prompts=[first, second]
    )[0]
    assert "still switched on" not in status


def test_face_only_leaves_the_bbox_mask_switch_alone():
    panel = _panel(2)
    status, enable, tabs, classes, _ = _split(panel, _answer(panel, mode=MODE_FACE))
    assert "with face_yolov8n.pt" in status
    assert enable == {"value": True}
    assert tabs[0]["ad_model"]["value"] == "face_yolov8n.pt"
    assert tabs[0]["ad_use_bbox_mask"] == {}
    assert tabs[0]["ad_mask_k"] == {"value": 2}
    assert tabs[0]["ad_denoising_strength"] == {"value": 0.6}
    assert tabs[1] == {attr: {} for attr in GUIDE_ATTRS}


def test_person_then_face_fills_both_tabs_each_with_its_own_updates():
    panel = _panel(4)
    status, enable, tabs, classes, _ = _split(panel, _answer(panel, mode=MODE_PERSON_FACE))
    assert status.startswith(
        "✅ Filled the 1st tab with person_yolov8n-seg.pt and the 2nd tab with "
        "face_yolov8n.pt, for 2 subjects."
    )
    assert enable == {"value": True}
    assert tabs[0]["ad_model"]["value"] == "person_yolov8n-seg.pt"
    assert tabs[1]["ad_model"] == {
        # "None" first after the 1st tab, as the menus are built.
        "choices": ["None", "face_yolov8n.pt", "person_yolov8n-seg.pt", "mediapipe_face_full"],
        "value": "face_yolov8n.pt",
    }
    assert tabs[0]["ad_prompt"] == tabs[1]["ad_prompt"]
    assert tabs[0]["ad_prompt"] is not tabs[1]["ad_prompt"]
    assert tabs[0]["ad_use_bbox_mask"] == {"value": False}
    assert tabs[1]["ad_use_bbox_mask"] == {}
    assert classes == [{"choices": ["person"], "value": [], "visible": True}] * 2
    # The denoise slider of each tab is an output, written with its own update.
    for w in panel.widgets[:2]:
        assert w.ad_denoising_strength in panel.click["outputs"]
    assert tabs[0]["ad_denoising_strength"] == tabs[1]["ad_denoising_strength"] == {"value": 0.6}
    assert tabs[0]["ad_denoising_strength"] is not tabs[1]["ad_denoising_strength"]
    assert status.split("\n\n")[-1] == DENOISE_LINE.replace(
        "the 1st tab", "the 1st tab and the 2nd tab"
    )


def test_person_then_face_with_a_single_tab_only_says_why():
    panel = _panel(1)
    status, enable, tabs, classes, menus = _split(panel, _answer(panel, mode=MODE_PERSON_FACE))
    assert status.startswith("⚠️ This choice needs 2 tabs and the panel has 1")
    assert enable == {}
    assert tabs == {0: {attr: {} for attr in GUIDE_ATTRS}}
    assert classes == [{}]
    assert menus == [{}] * 4
    # The menus still take the LoRA list of now, when it changed.
    names = [["alice"]]
    panel = _panel(1, lora_names=lambda: names[0])
    names[0] = ["carol"]
    menus = _split(panel, _answer(panel, mode=MODE_PERSON_FACE))[-1]
    assert menus == [{"choices": ["None", "carol"]}] * 4


def test_with_a_single_tab_the_one_tab_modes_work():
    panel = _panel(1)
    status, enable, tabs, classes, _ = _split(panel, _answer(panel))
    assert status.startswith("✅ Filled the 1st tab for 2 subjects")
    assert enable == {"value": True}
    assert tabs[0]["ad_mask_k"] == {"value": 2}
    assert len(classes) == 1


def test_nothing_chosen_changes_nothing_and_does_not_switch_adetailer_on():
    panel = _panel(2)
    answer = _answer(panel, subjects=[("None", 0.8, " ")])
    assert answer[0] == "⚠️ Pick a LoRA or type some words for at least one subject."
    assert answer[1:] == [{}] * (len(answer) - 1)


def test_a_missing_detector_keeps_the_tabs_detector_and_classes():
    panel = _panel(2, mapping={"face_yolov8n.pt": "face.pt"})
    status, enable, tabs, classes, _ = _split(panel, _answer(panel))
    assert "No person detector is installed" in status
    assert enable == {"value": True}
    for attr in ("ad_model", "ad_model_classes", "ad_model_classes_exclude", "ad_model_classes_excluded"):
        assert tabs[0][attr] == {}
    assert classes[0] == {}
    assert tabs[0]["ad_prompt"]["value"].endswith("[SEP] [SKIP]")


def test_the_status_cannot_carry_markup_typed_by_the_user():
    panel = _panel(2)
    answer = _answer(panel, subjects=[("<img src=x onerror=1>", 0.8, "")])
    assert "<" not in answer[0]
    assert ">" not in answer[0]
    assert "&lt;img src=x onerror=1&gt;" in answer[0]
    # The explanation's own "< >" too: the whole status is escaped.
    assert "without &lt; &gt; or :" in answer[0]
    assert answer[1] == {}


def test_md_safe_is_left_as_the_presets_use_it():
    # The guide protects the values it quotes by itself (md_plain); the
    # panel's _md_safe, which the preset preview and the detection status
    # use as well, does what it did.
    md_safe = _ui()["_md_safe"]
    assert md_safe("a_b *c* `d` [e] | <f> ![g](h) &#95;") == (
        "a_b *c* `d` [e] | &lt;f&gt; !\\[g](h) &#95;"
    )
    panel = _panel(2, main_prompt=_Component("main prompt"))
    status = _answer(
        panel, subjects=[("_new_", 0.8, "w <lora:_new_:1>")], main="<lora:_new_:1>"
    )[0]
    assert "the words already hold &#95;new&#95;, so" in status
    assert "⚠️ &#95;new&#95; is also in the main prompt" in status
    assert "_new_" not in status


def test_the_status_of_a_filled_tab_is_escaped_too():
    # A tag typed in the words can hold "<" in its name, and the status names
    # the LoRAs the main prompt holds as well.
    panel = _panel(2, main_prompt=_Component("main prompt"))
    tag = "<lora:a<img src=x:1>"
    answer = _answer(panel, subjects=[("None", 0.8, f"woman {tag}")], main=f"photo {tag}")
    assert answer[1] == {"value": True}
    assert "a&lt;img src=x is also in the main prompt" in answer[0]
    assert "<" not in answer[0]
    # The prompt itself is written as typed.
    assert _split(panel, answer)[2][0]["ad_prompt"] == {
        "value": f"woman {tag} [SEP] [SKIP]"
    }


def test_the_main_prompt_is_read_for_the_warning_only():
    main = _Component("main prompt")
    panel = _panel(2, main_prompt=main)
    assert panel.click["inputs"][-1] is main
    status = _answer(panel, main="a photo <lora:alice:1>")[0]
    assert "⚠️ alice is also in the main prompt" in status
    assert "main prompt" not in _answer(panel, main="a photo")[0].replace(DENOISE_LINE, "")
    # A value that is not a text (another kind of component) is ignored.
    answer = panel.click["fn"](*_values(panel)[:-1], None)[0]
    assert "main prompt" not in answer.replace(DENOISE_LINE, "")
    # Without the box the guide works the same, without that warning.
    without = _panel(2)
    assert main not in without.click["inputs"]
    assert _answer(without)[0].startswith("✅ ")


def test_the_comment_marks_told_are_those_the_webui_strips_at_the_click():
    # "//" is a prompt comment on Forge Neo, plain text on AUTOMATIC1111.
    rows = [("alice", 0.8, "blush, ///"), ("bob", 0.8, "man #2")]
    told = lambda panel: [  # noqa: E731
        line.split(":")[0] for line in _answer(panel, subjects=rows)[0].split("\n\n")
        if " holds a " in line
    ]
    option = [True]

    def neo(text):
        if not option[0]:
            return text
        text = re.sub(r"\/\*.*?\*\/", "", text, flags=re.DOTALL)
        return re.sub(r"[^\S\n]*(\#|\/\/).*", "", text)

    panel = _panel(2, strip_comments=neo)
    assert told(panel) == ["⚠️ Subject 1 holds a //", "⚠️ Subject 2 holds a #"]
    # The option is read at each click: off, nothing is dropped.
    option[0] = False
    assert told(panel) == []
    a1111 = _panel(2, strip_comments=lambda text: re.sub("#[^\n]*(\n|$)", "\n", text))
    assert told(a1111) == ["⚠️ Subject 2 holds a #"]
    # A panel that was given no function, or one that fails: every mark.
    for strip in (None, lambda text: 1 / 0):
        assert told(_panel(2, strip_comments=strip)) == [
            "⚠️ Subject 1 holds a //", "⚠️ Subject 2 holds a #",
        ]
    assert told(_panel(2)) == ["⚠️ Subject 1 holds a //", "⚠️ Subject 2 holds a #"]


def test_a_name_the_webui_does_not_know_is_told_and_one_it_knows_otherwise_is_fixed():
    names = [["bob", "alice"]]
    aliases = [["Bobby"]]
    panel = _panel(2, lora_names=lambda: names[0], lora_aliases=lambda: aliases[0])
    rows = [("Nessuno", 0.8, "woman"), ("ALICE.safetensors", 0.8, "girl"), ("Bobby", 1, "man")]
    status, enable, tabs, _, _ = _split(panel, _answer(panel, subjects=rows))
    lines = status.split("\n\n")
    assert lines[0].startswith("⚠️ Filled the 1st tab for 3 subjects")
    assert lines[1].startswith("⚠️ Subject 1: the WebUI lists no LoRA named Nessuno. Its part")
    assert lines[2] == (
        "ℹ️ Subject 2: ALICE.safetensors was written as alice, the name in the "
        "WebUI's LoRA list."
    )
    assert lines[3:] == [DENOISE_LINE]
    assert enable == {"value": True}
    assert tabs[0]["ad_prompt"] == {
        "value": "woman, <lora:Nessuno:0.8> [SEP] girl, <lora:alice:0.8> [SEP] "
        "man, <lora:Bobby:1> [SEP] [SKIP]"
    }
    # The list is read at each click: the file added since is known now.
    names[0] = ["bob", "alice", "Nessuno"]
    assert "lists no LoRA" not in _answer(panel, subjects=rows)[0]
    # Without the aliases (a WebUI that has none) an alias is told.
    aliases[0] = None
    assert "Subject 3: the WebUI lists no LoRA named Bobby." in _answer(panel, subjects=rows)[0]
    # A WebUI without a LoRA list: nothing is compared.
    names[0] = []
    status = _answer(panel, subjects=rows)[0]
    assert status.startswith("✅ ")
    assert "lists no LoRA" not in status
    assert "was written as" not in status
    assert _answer(_panel(2, lora_names=None), subjects=rows)[0].startswith("✅ ")


def test_the_click_gives_the_guide_the_file_of_each_alias():
    # The WebUI's aliases with their LoRA file (_lora_aliases): the menu's
    # file name and the alias in the words are one LoRA, one tag.
    panel = _panel(
        2, lora_names=lambda: ["bob", "alice"],
        lora_aliases=lambda: {"bob": "bob", "Bobby": "bob", "alice": "alice", 3: "x"},
    )
    rows = [("bob", 0.8, "man <lora:Bobby:0.5>")]
    status, _, tabs, _, _ = _split(panel, _answer(panel, subjects=rows))
    assert tabs[0]["ad_prompt"] == {"value": "man <lora:Bobby:0.5> [SEP] [SKIP]"}
    assert status.startswith("✅ ")
    assert _ui()["_guide_lora_names"](
        SimpleNamespace(lora_aliases=lambda: {"Bobby": "bob", 3: "x", "odd": None}),
        "lora_aliases",
    ) == {"Bobby": "bob"}


def test_the_sort_order_is_read_at_each_click_and_only_read():
    order = [("None", "Off")]
    panel = _panel(2, sort=lambda: order[0])
    assert "Sort bounding boxes by, now None: set it to" in _answer(panel)[0]
    order[0] = (LEFT_TO_RIGHT, BBOX_SORTBY_MORE[1])
    assert f"More sort orders, now {BBOX_SORTBY_MORE[1]}: set it to Off" in _answer(panel)[0]
    order[0] = (LEFT_TO_RIGHT, "Off")
    assert _answer(panel)[0].split("\n\n")[0].endswith(
        "Subject 1 is the leftmost of the 2 biggest people."
    )


def test_a_sort_order_that_cannot_be_read_still_fills_the_tab():
    def broken():
        raise KeyError("opts")

    for reader in (broken, None, lambda: None, lambda: ("only one",)):
        panel = _panel(2, sort=reader if reader is not None else (lambda: 1 / 0))
        status, enable, tabs, _, _ = _split(panel, _answer(panel))
        assert "now not readable" in status
        assert "leftmost" not in status
        assert enable == {"value": True}
        assert tabs[0]["ad_mask_k"] == {"value": 2}


@pytest.mark.parametrize("pairs", [False, True])
def test_the_lora_menus_follow_the_list_of_the_click(pairs):
    names = [["bob", "alice"]]
    panel = _panel(2, lora_names=lambda: names[0], pairs=pairs)
    # The list the menus were built with: an update that changes nothing.
    assert _answer(panel)[-4:] == [{}] * 4
    names[0] = ["carol", "bob", "alice"]
    menus = _answer(panel)[-4:]
    assert menus == [{"choices": ["None", "alice", "bob", "carol"]}] * 4
    assert len({id(m["choices"]) for m in menus}) == 4
    # The same list again is not sent again: with thousands of LoRAs it
    # is megabytes at every click.
    assert _answer(panel)[-4:] == [{}] * 4
    assert _answer(panel, subjects=[("None", 0.8, "")])[-4:] == [{}] * 4
    # A list that is empty or cannot be read keeps the menus as they are:
    # the value picked or typed is never touched.
    names[0] = []
    assert _answer(panel)[-4:] == [{}] * 4
    names[0] = None
    assert _answer(panel)[-4:] == [{}] * 4
    assert all("value" not in m for m in menus)
    # Back to a list after that: still the one the menus show.
    names[0] = ["alice", "bob", "carol"]
    assert _answer(panel)[-4:] == [{}] * 4
    names[0] = ["alice"]
    assert _answer(panel)[-4:] == [{"choices": ["None", "alice"]}] * 4


@pytest.mark.parametrize("pairs", [False, True])
def test_a_page_reload_lists_the_loras_of_the_last_click(pairs):
    # After F5 the browser is served the choices each component holds: they
    # were the list of the start-up until the next click, since the click
    # only updated the browser. The detector menus update the page too.
    names = [["bob", "alice"]]
    panel = _panel(2, lora_names=lambda: names[0], pairs=pairs)
    held = [menu.choices for menu in panel.guide.loras]
    names[0] = ["carol", "bob", "alice"]
    _answer(panel)
    for menu, before in zip(panel.guide.loras, held):
        assert _page_choices(menu) == ["None", "alice", "bob", "carol"]
        # In place, in the shape the component keeps (pairs on Gradio 4).
        assert menu.choices is before
        assert all(isinstance(c, tuple) is pairs for c in menu.choices)
    assert len({id(menu.choices) for menu in panel.guide.loras}) == 4
    # A click that only has something to say updates the page as well.
    names[0] = ["dave"]
    assert _answer(panel, mode=None)[-4:] == [{"choices": ["None", "dave"]}] * 4
    assert _page_choices(panel.guide.loras[3]) == ["None", "dave"]


def test_the_menus_of_the_other_side_are_updated_for_a_reload_too():
    # txt2img and img2img have a guide each, built in the same process.
    names = [["bob"]]
    ns = _ui()
    info = SimpleNamespace(model_mapping=MAPPING, lora_names=lambda: names[0])
    sides = []
    for is_img2img in (False, True):
        guide = ns["subject_guide_ui"](is_img2img, ["bob"])
        _hold_choices(guide.loras)
        widgets = [SimpleNamespace(
            **{attr: _Component(attr) for attr in ALL_ARGS.attrs},
            ad_model_classes_dropdown=_Component("classes"),
        )]
        ns["_wire_subject_guide"](guide, _Component("enable"), widgets, 1, info)
        sides.append(guide)
    assert len(ns["_GUIDE_LORA_DROPDOWNS"]) == 8
    txt, img = sides
    names[0] = ["bob", "carol"]
    click = txt.button.clicks[0]
    count = len(click["inputs"])
    assert click["fn"](*[None] * count)[-4:] == [{"choices": ["None", "bob", "carol"]}] * 4
    for menu in (*txt.loras, *img.loras):
        assert _page_choices(menu) == ["None", "bob", "carol"]
    # The other side's browser still shows the old list: its click sends it.
    click = img.button.clicks[0]
    assert click["fn"](*[None] * count)[-4:] == [{"choices": ["None", "bob", "carol"]}] * 4
    assert click["fn"](*[None] * count)[-4:] == [{}] * 4
    # A menu of a panel that is gone (the UI was rebuilt) is passed over.
    kept = ns["_GUIDE_LORA_DROPDOWNS"]
    gone = _Component("Dropdown")
    kept.append(weakref.ref(gone))
    del gone
    names[0] = ["zoe"]
    assert txt.button.clicks[0]["fn"](*[None] * count)[-1] == {"choices": ["None", "zoe"]}
    assert _page_choices(img.loras[0]) == ["None", "zoe"]


def test_a_page_that_cannot_be_updated_keeps_getting_the_list():
    # Without the choices list of its own (another Gradio, a stand-in) a
    # reload would show the old list: the click then sends the list each
    # time, as it did before.
    names = [["bob", "alice"]]
    panel = _panel(2, lora_names=lambda: names[0])
    for menu in panel.guide.loras:
        del menu.choices
    names[0] = ["carol"]
    assert _answer(panel)[-4:] == [{"choices": ["None", "carol"]}] * 4
    assert _answer(panel)[-4:] == [{"choices": ["None", "carol"]}] * 4
    sync = panel.ns["_sync_lora_choices"]
    assert sync([], ["None"]) is True
    assert sync([SimpleNamespace(choices=[])], ["None"]) is False
    assert sync([SimpleNamespace(choices=None)], ["None"]) is False
    assert sync([object()], ["None"]) is False


def test_a_box_that_fails_while_it_is_built_is_hidden():
    # Something inside the box fails (a Gradio without one of its options):
    # the panel goes on without the guide, but the box stayed on screen with
    # the rows built so far and no button.
    ns = _ui()
    made = ns["gr"].__getattr__

    class Failing:
        created = ns["gr"].created
        update = staticmethod(ns["gr"].update)

        def __getattr__(self, kind):
            if kind == "Button":
                def broken(*args, **kwargs):
                    raise TypeError("unexpected keyword argument")
                return broken
            return made(kind)

    ns["gr"] = Failing()
    with pytest.raises(TypeError):
        ns["subject_guide_ui"](False, ["alice"])
    created = Failing.created
    accordion = created[0]
    assert accordion.kind == "Accordion"
    assert accordion.visible is False
    # The rows were built before the button failed: without this they show.
    assert [c.kind for c in created].count("Dropdown") == 4
    # Nothing of a box that failed is kept for the page updates.
    assert ns["_GUIDE_LORA_DROPDOWNS"] == []
    # A box that is built whole stays as it is.
    whole = _ui()
    whole["subject_guide_ui"](False, ["alice"])
    assert not hasattr(whole["gr"].created[0], "visible")
    # And adui goes on without the guide (it catches what the box raises).
    adui = _function(ast.parse(_UI_PATH.read_text(encoding="utf-8")), "adui")
    box = _function(ast.parse(_UI_PATH.read_text(encoding="utf-8")), "subject_guide_ui")
    opened = next(n for n in ast.walk(box) if isinstance(n, ast.With))
    assert [ast.unparse(item.context_expr.func) for item in opened.items] == [
        "gr.Accordion", "_hidden_if_unfinished",
    ]
    assert ast.unparse(opened.items[1].context_expr) == "_hidden_if_unfinished(accordion)"
    assert "subject_guide = None" in ast.unparse(adui)


def test_an_error_inside_the_guide_changes_nothing_and_says_so():
    def no_names(path):
        raise OSError("cannot read the model at some place")

    panel = _panel(2, class_names=no_names)
    answer = _answer(panel)
    assert answer[0] == (
        "⚠️ The guide could not fill the tabs (OSError): nothing was changed. "
        "The console says where it stopped."
    )
    assert answer[1:] == [{}] * (len(answer) - 1)
    assert len({id(u) for u in answer[1:]}) == len(answer) - 1
    # The console does say it: the kind of error and the place, in ASCII,
    # without the error's own text (it can hold a full path of the PC).
    assert len(panel.ns["printed"]) == 1
    line = panel.ns["printed"][0]
    assert re.fullmatch(
        r"\[-\] ADetailer: the subject guide could not fill the tabs: "
        r"OSError in no_names, line \d+",
        line,
    )
    assert line.isascii()
    assert "some place" not in line


def test_values_of_the_wrong_kind_never_raise():
    panel = _panel(2)
    count = len(panel.click["inputs"])
    for values in ([None] * count, [3] * count, [["x"]] * count, [MODE_PERSON]):
        answer = panel.click["fn"](*values)
        assert len(answer) == len(panel.click["outputs"])
        assert answer[0].startswith("⚠️ ")
        assert answer[1] == {}


# ------------------------------------------------- where it sits in the panel


def _function(tree, name):
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _calls(node, name):
    return [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == name
    ]


def test_the_guide_sits_between_skip_img2img_and_the_tabs():
    adui = _function(ast.parse(_UI_PATH.read_text(encoding="utf-8")), "adui")
    built = _calls(adui, "subject_guide_ui")
    assert len(built) == 1
    skip = next(
        n.lineno for n in ast.walk(adui)
        if isinstance(n, ast.Constant) and n.value == "Skip img2img"
    )
    tabs = next(
        n.lineno for n in ast.walk(adui)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "Tabs"
    )
    assert skip < built[0].lineno < tabs
    assert ast.unparse(built[0]) == "subject_guide_ui(is_img2img, _guide_lora_names(webui_info))"


def test_the_guide_button_is_the_last_listener_of_the_panel():
    from tests.test_jobs import _is_listener_call

    source = _UI_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    adui = _function(tree, "adui")
    wired = _calls(adui, "_wire_subject_guide")
    assert len(wired) == 1
    assert ast.unparse(wired[0]) == (
        "_wire_subject_guide(subject_guide, ad_enable, all_widgets, num_models, webui_info)"
    )
    # After every other call of adui that wires listeners or builds the
    # tabs, so the listeners the panel already had keep their order.
    for name in (
        "one_ui_group", "_wire_copy_paste", "_wire_presets",
        "_wire_detection_previews", "_wire_model_refresh",
    ):
        assert max(c.lineno for c in _calls(adui, name)) < wired[0].lineno
    refresh = _calls(adui, "_wire_model_refresh")[0]
    # Nothing that registers a listener comes after it in adui.
    later = [
        n for n in ast.walk(adui)
        if isinstance(n, ast.Call) and n.lineno > wired[0].end_lineno
        and (_is_listener_call(n) or getattr(n.func, "id", "").startswith(("_wire_", "one_ui_")))
    ]
    assert later == []
    assert refresh.lineno < wired[0].lineno
    # One .click in the wiring function, none in the one that builds the box.
    wire = _function(tree, "_wire_subject_guide")
    assert [n.func.attr for n in ast.walk(wire) if isinstance(n, ast.Call) and _is_listener_call(n)] == ["click"]
    box = _function(tree, "subject_guide_ui")
    assert [n for n in ast.walk(box) if isinstance(n, ast.Call) and _is_listener_call(n)] == []


def test_the_guide_adds_no_script_argument_and_no_infotext_field():
    assert len(ALL_ARGS) == 63
    tree = ast.parse(_UI_PATH.read_text(encoding="utf-8"))
    adui = _function(tree, "adui")
    # What adui returns is what the WebUI passes to the script at Generate,
    # offers to the API and saves in ui-config.json: the guide is not there.
    assigned = [
        ast.unparse(n.value) for n in ast.walk(adui)
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "components"
    ]
    assert assigned == ["[ad_enable, ad_skip_img2img, *states]"]
    returned = [ast.unparse(n.value) for n in ast.walk(adui) if isinstance(n, ast.Return)]
    assert returned == ["(components, infotext_fields)"]
    for node in ast.walk(adui):
        if isinstance(node, ast.Call) and "infotext_fields" in ast.unparse(node.func):
            assert "subject_guide" not in ast.unparse(node)
    # The box hangs nothing on a tab's Widgets (ALL_ARGS, presets, state).
    box = _function(tree, "subject_guide_ui")
    names = {n.id for n in ast.walk(box) if isinstance(n, ast.Name)}
    assert not names & {"w", "Widgets", "ALL_ARGS", "infotext_fields", "states"}
    assert "State" not in {n.attr for n in ast.walk(box) if isinstance(n, ast.Attribute)}


def test_a_guide_that_cannot_be_built_or_wired_does_not_break_the_panel():
    adui = _function(ast.parse(_UI_PATH.read_text(encoding="utf-8")), "adui")
    guarded = {
        ast.unparse(call.func): True
        for node in ast.walk(adui) if isinstance(node, ast.Try)
        and any(
            isinstance(h.type, ast.Name) and h.type.id == "Exception" for h in node.handlers
        )
        for stmt in node.body for call in ast.walk(stmt) if isinstance(call, ast.Call)
    }
    assert guarded.get("subject_guide_ui")
    assert guarded.get("_wire_subject_guide")


# ------------------------------------------------------------ the script side


def _script(sys_modules=None, opts_data=None):
    tree, body = _top_level(
        _SCRIPT_PATH,
        {"_lora_names", "_lora_aliases", "_reads_lyco_tags", "_sort_order_settings"},
    )
    assert len(body) == 4
    namespace = {
        "sys": SimpleNamespace(modules=sys_modules if sys_modules is not None else {}),
        "opts": SimpleNamespace(data=opts_data if opts_data is not None else {}),
        "BBOX_SORTBY": BBOX_SORTBY,
        "BBOX_SORTBY_MORE": BBOX_SORTBY_MORE,
        "Any": object,
    }
    module = ast.Module(body=[tree.body[0], *body], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_SCRIPT_PATH), "exec"), namespace)
    return namespace


def _networks(**attrs):
    module = ModuleType("networks")
    for name, value in attrs.items():
        setattr(module, name, value)
    return module


def test_the_lora_names_come_from_the_webuis_own_list():
    available = {"bob": object(), "alice": object(), "": object(), 3: object()}
    names = _script({"networks": _networks(available_networks=available)})["_lora_names"]()
    assert names == ["bob", "alice"]


@pytest.mark.parametrize(
    "modules",
    [
        {},  # a WebUI without the LoRA extension
        {"networks": None},
        {"networks": _networks()},  # another module of that name
        {"networks": _networks(available_networks=None)},
        {"networks": _networks(available_networks=["alice"])},
        {"networks": _networks(available_networks="alice")},
        {"networks": SimpleNamespace()},
    ],
)
def test_without_a_lora_list_there_are_no_names_and_no_error(modules):
    assert _script(modules)["_lora_names"]() == []


def test_a_lora_list_that_raises_gives_no_names():
    class Raising(dict):
        def __iter__(self):
            raise RuntimeError("changed during iteration")

    class Module:
        def __getattr__(self, name):
            raise RuntimeError("lazy import failed")

    for networks in (_networks(available_networks=Raising(a=1)), Module()):
        assert _script({"networks": networks})["_lora_names"]() == []


def test_the_lora_module_is_looked_up_never_imported():
    tree = ast.parse(_SCRIPT_PATH.read_text(encoding="utf-8"))
    for name in ("_lora_names", "_lora_aliases"):
        lora = _function(tree, name)
        assert "sys.modules.get('networks')" in ast.unparse(lora)
        assert not [n for n in ast.walk(lora) if isinstance(n, (ast.Import, ast.ImportFrom))]
        called = {
            getattr(n.func, "attr", getattr(n.func, "id", ""))
            for n in ast.walk(lora) if isinstance(n, ast.Call)
        }
        # Nor a scan of the LoRA folders: the names the WebUI already has.
        assert called <= {"get", "getattr", "isinstance", "list", "lower"}


def test_the_aliases_are_the_other_names_a_tag_can_take():
    # As AUTOMATIC1111 and Forge Neo keep them (networks.py): every name and
    # every alias is a key; an alias two files share is refused by the WebUI,
    # which keeps those in lower case, as a dict on one and a set on the other.
    aliases = {"bob": 1, "Bobby": 1, "alice": 1, "Shared": 1, "": 1, 3: 1}
    for forbidden in ({"none": 1, "shared": 1}, {"none", "shared"}):
        module = _networks(
            available_networks={"bob": 1, "alice": 1},
            available_network_aliases=aliases,
            forbidden_network_aliases=forbidden,
        )
        assert _script({"networks": module})["_lora_aliases"]() == {
            "bob": "bob", "Bobby": "Bobby", "alice": "alice",
        }
    # Without the refused ones, every key.
    module = _networks(available_network_aliases={"Bobby": 1, "Shared": 1})
    assert _script({"networks": module})["_lora_aliases"]() == {"Bobby": "Bobby", "Shared": "Shared"}
    odd = _networks(available_network_aliases={"Bobby": 1}, forbidden_network_aliases=7)
    assert _script({"networks": odd})["_lora_aliases"]() == {"Bobby": "Bobby"}


def test_each_alias_comes_with_the_name_of_its_lora_file():
    # The value of each key is the LoRA it gives (NetworkOnDisk), whose
    # .name is its file name, the key of available_networks: the guide then
    # knows that "Bobby" and "bob" are one LoRA. Without a name, the key.
    bob = SimpleNamespace(name="bob", alias="Bobby")
    alice = SimpleNamespace(name="alice", alias="alice")
    module = _networks(
        available_networks={"bob": bob, "alice": alice},
        available_network_aliases={"bob": bob, "Bobby": bob, "alice": alice, "odd": 1, "x": None},
        forbidden_network_aliases={"none": 1},
    )
    assert _script({"networks": module})["_lora_aliases"]() == {
        "bob": "bob", "Bobby": "bob", "alice": "alice", "odd": "odd", "x": "x",
    }


@pytest.mark.parametrize(
    "modules",
    [
        {}, {"networks": None}, {"networks": _networks()},
        {"networks": _networks(available_network_aliases=None)},
        {"networks": _networks(available_network_aliases=["Bobby"])},
        {"networks": SimpleNamespace()},
    ],
)
def test_without_aliases_there_are_none_and_no_error(modules):
    assert _script(modules)["_lora_aliases"]() == {}


def test_aliases_that_raise_give_none():
    class Raising(dict):
        def __iter__(self):
            raise RuntimeError("changed during iteration")

    class Module:
        def __getattr__(self, name):
            raise RuntimeError("lazy import failed")

    for networks in (_networks(available_network_aliases=Raising(a=1)), Module()):
        assert _script({"networks": networks})["_lora_aliases"]() == {}


def test_a_webui_reads_lyco_tags_only_where_lyco_is_the_lora_network():
    # AUTOMATIC1111 registers "lyco" as an alias of the LoRA network
    # (extensions-builtin/Lora/scripts/lora_script.py); Forge Neo does not,
    # and modules/extra_networks.py skips such a tag.
    lora = object()

    def host(**attrs):
        return {"modules.extra_networks": SimpleNamespace(**attrs)}

    read = _script(host(extra_network_registry={"lora": lora},
                        extra_network_aliases={"lyco": lora}))["_reads_lyco_tags"]
    assert read() is True
    for modules in (
        host(extra_network_registry={"lora": lora}, extra_network_aliases={}),
        host(extra_network_registry={"lora": lora, "lyco": object()}, extra_network_aliases={}),
        host(extra_network_registry={}, extra_network_aliases={}),
        host(extra_network_registry={"lora": lora}),
        {}, {"modules.extra_networks": None},
    ):
        assert _script(modules)["_reads_lyco_tags"]() is False


def test_the_click_tells_a_lyco_tag_unless_the_webui_reads_it():
    rows = [("alice", 0.8, "woman <lyco:alice:0.5>")]
    told = "Subject 1: the words hold a lyco tag."
    for reads, said in ((lambda: True, False), (lambda: False, True), (lambda: 1, True),
                        (lambda: 1 / 0, True), (None, True)):
        panel = _panel(2, reads_lyco=reads)
        status, _, tabs, _, _ = _split(panel, _answer(panel, subjects=rows))
        assert (told in status) is said
        assert tabs[0]["ad_prompt"] == {"value": "woman <lyco:alice:0.5> [SEP] [SKIP]"}


def test_the_sort_settings_are_read_with_their_defaults():
    read = _script(opts_data={})["_sort_order_settings"]
    assert read() == ("None", "Off")
    data = {"ad_bbox_sortby": LEFT_TO_RIGHT, "ad_bbox_sortby_more": "Area (small to large)"}
    assert _script(opts_data=data)["_sort_order_settings"]() == (
        LEFT_TO_RIGHT, "Area (small to large)",
    )
    # Read only: the guide never changes a setting.
    node = _function(
        ast.parse(_SCRIPT_PATH.read_text(encoding="utf-8")), "_sort_order_settings"
    )
    assert "opts.data.get('ad_bbox_sortby', BBOX_SORTBY[0])" in ast.unparse(node)
    assert "opts.data.get('ad_bbox_sortby_more', BBOX_SORTBY_MORE[0])" in ast.unparse(node)
    assert not [
        n for n in ast.walk(node)
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete))
    ]
    assert {n.func.attr for n in ast.walk(node) if isinstance(n, ast.Call)} == {"get"}


def test_the_script_hands_the_panel_what_the_guide_reads():
    ui = _function(ast.parse(_SCRIPT_PATH.read_text(encoding="utf-8")), "ui")
    info = _calls(ui, "WebuiInfo")
    assert len(info) == 1
    given = {k.arg: ast.unparse(k.value) for k in info[0].keywords}
    assert given["lora_names"] == "_lora_names"
    assert given["lora_aliases"] == "_lora_aliases"
    assert given["reads_lyco"] == "_reads_lyco_tags"
    assert given["sort_order"] == "_sort_order_settings"
    # The WebUI's own comment rules, with its option (see _strip_host_comments).
    assert given["strip_comments"] == "_strip_host_comments"
    # Each side reads its own main prompt box.
    assert given["main_prompt"] == (
        "img2img_prompt_box if is_img2img else txt2img_prompt_box"
    )
    # And the panel takes them as optional: an older caller still works.
    cls = next(
        n for n in ast.walk(ast.parse(_UI_PATH.read_text(encoding="utf-8")))
        if isinstance(n, ast.ClassDef) and n.name == "WebuiInfo"
    )
    defaults = {
        n.target.id: ast.unparse(n.value) for n in cls.body
        if isinstance(n, ast.AnnAssign) and n.value is not None
    }
    optional = (
        "lora_names", "lora_aliases", "sort_order", "main_prompt", "strip_comments",
        "reads_lyco",
    )
    assert {k: defaults.get(k) for k in optional} == dict.fromkeys(optional, "None")


# ------------------------------------------------------- the main prompt box


def _capture():
    tree, body = _top_level(_SCRIPT_PATH, {"on_after_component"})
    assert len(body) == 1
    namespace = {
        "txt2img_submit_button": None, "img2img_submit_button": None,
        "txt2img_prompt_box": None, "img2img_prompt_box": None,
    }
    module = ast.Module(body=[tree.body[0], *body], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_SCRIPT_PATH), "exec"), namespace)
    return namespace


def _kept(namespace):
    return {
        name: getattr(value, "elem_id", None)
        for name, value in namespace.items()
        if name.endswith(("_button", "_box"))
    }


def test_the_main_prompt_boxes_are_kept_as_the_webui_builds_them():
    ns = _capture()
    seen = ns["on_after_component"]
    # In the order a WebUI builds a side: prompt, negative prompt, Generate.
    for elem_id in (
        "txt2img_prompt", "txt2img_neg_prompt", "txt2img_generate", "txt2img_steps",
        "img2img_prompt", "img2img_neg_prompt", "img2img_generate", None,
    ):
        seen(SimpleNamespace(elem_id=elem_id), elem_id=elem_id)
    assert _kept(ns) == {
        "txt2img_submit_button": "txt2img_generate",
        "img2img_submit_button": "img2img_generate",
        "txt2img_prompt_box": "txt2img_prompt",
        "img2img_prompt_box": "img2img_prompt",
    }


def test_a_webui_without_those_prompt_boxes_leaves_them_unset():
    ns = _capture()
    seen = ns["on_after_component"]
    for component in (
        SimpleNamespace(elem_id="txt2img_generate"),
        SimpleNamespace(elem_id="img2img_generate"),
        SimpleNamespace(elem_id="some_other_prompt"),
        SimpleNamespace(elem_id="txt2img_prompt_container"),
        SimpleNamespace(elem_id=["txt2img_prompt"]),
        SimpleNamespace(),  # a component without an id
        object(),
    ):
        seen(component)
    assert _kept(ns) == {
        "txt2img_submit_button": "txt2img_generate",
        "img2img_submit_button": "img2img_generate",
        "txt2img_prompt_box": None,
        "img2img_prompt_box": None,
    }
    # And the panel then wires the guide without it (see
    # test_the_main_prompt_is_read_for_the_warning_only).


def test_the_prompt_boxes_start_unset_and_are_only_handed_to_the_guide():
    source = _SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assigned = [
        ast.unparse(node) for node in tree.body
        if isinstance(node, ast.Assign) and "prompt_box" in ast.unparse(node)
    ]
    assert assigned == ["txt2img_prompt_box = img2img_prompt_box = None"]
    # Read in one place, the WebuiInfo of ui(): never written to, never used
    # as an output of a listener of the script.
    uses = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id.endswith("_prompt_box")
        and isinstance(node.ctx, ast.Load)
    ]
    assert sorted(n.id for n in uses) == ["img2img_prompt_box", "txt2img_prompt_box"]
    ui = _function(tree, "ui")
    assert all(ui.lineno <= n.lineno <= ui.end_lineno for n in uses)
    # In the panel, the box is an input of the guide's click and nothing else.
    panel_tree = ast.parse(_UI_PATH.read_text(encoding="utf-8"))
    readers = {
        fn.name for fn in ast.walk(panel_tree) if isinstance(fn, ast.FunctionDef)
        for n in ast.walk(fn)
        if (isinstance(n, ast.Attribute) and n.attr == "main_prompt")
        or (isinstance(n, ast.Constant) and n.value == "main_prompt")
    }
    # (test_what_the_click_reads_and_writes: never one of its outputs.)
    assert readers == {"_wire_subject_guide"}

def test_the_guide_tab_line_names_the_denoise_the_button_writes():
    # The line of the ADetailer Guide tab is written by hand: it has to
    # follow subject_guide.GUIDE_DENOISE when that changes.
    text = (Path(__file__).parents[1] / "aaaaaa" / "ui.py").read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.startswith("- Guide: several subjects, one LoRA each")]
    assert len(lines) == 1
    assert f"with Inpaint denoising strength at {subject_guide.GUIDE_DENOISE:g}." in lines[0]
