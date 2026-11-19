"""The parameters ADetailer adds to a saved image beside the tabs' settings.

Whole generations on a stand-in host: the real process(), postprocess_image()
and detailer pass of scripts/!adetailer.py, a detector that finds what the
test says, and a host whose img2img pass applies the styles and runs the
pass's scripts, as AUTOMATIC1111 and Forge do. No WebUI, no model.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import time
from contextlib import nullcontext
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from aaaaaa import p_method
from aaaaaa.p_method import get_i, need_call_postprocess, need_call_process
from adetailer.args import ADetailerArgs
from tests.test_pipeline import (
    _INLINE_PROMPT_ASSIGNS,
    _INLINE_PROMPT_FUNCTIONS,
    _INLINE_PROMPT_METHODS,
    _a1111_strip_comments,
    _host_paste,
    _load_script,
    _paste_callback,
    _StyleDatabase,
    _ui_suffix,
)

_ROOT = Path(__file__).resolve().parents[1]
_RESOLVED = "ADetailer resolved prompt"
_RESOLVED_NEGATIVE = "ADetailer resolved negative prompt"


# AUTOMATIC1111's and Forge Neo's modules/infotext_utils.py: how a value is
# written in the parameters line and how the line is read back.
_HOST_PARAM = re.compile(r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)')


def _host_quote(text):
    if "," not in str(text) and "\n" not in str(text) and ":" not in str(text):
        return text
    return json.dumps(text, ensure_ascii=False)


def _host_line(params: dict) -> str:
    return ", ".join(
        k if k == v else f"{k}: {_host_quote(v)}" for k, v in params.items() if v is not None
    )


def _host_read(line: str) -> dict:
    found = {}
    for key, value in _HOST_PARAM.findall(line):
        if value[0] == '"' and value[-1] == '"':
            value = json.loads(value)
        found[key] = value
    return found


class _Draws:
    """A script of the detailer pass that draws wildcards and variants, as
    Dynamic Prompts and the "wildcards" extension do in their process() hook:
    each __name__ and each {a|b} of the prompts the host built becomes the
    value listed for the region being inpainted."""

    filename = "extensions/sd-dynamic-prompts/scripts/dynamic_prompting.py"

    def __init__(self, values, host):
        self.values = values
        self.host = host

    def _draw(self, text):
        region = self.host.region

        def value(match):
            options = self.values[match.group(1) or match.group(2)]
            return options[region % len(options)]

        return re.sub(r"__(\w+)__|\{([^{}]*)\}", value, text)

    def process(self, p):
        p.all_prompts = [self._draw(text) for text in p.all_prompts]
        p.all_negative_prompts = [self._draw(text) for text in p.all_negative_prompts]


class _Runner:
    """The host's ScriptRunner, as far as the pass uses it."""

    def __init__(self, *scripts):
        self.alwayson_scripts = list(scripts)
        self.callback_map = {}

    def process(self, p):
        for script in self.alwayson_scripts:
            script.process(p)


class _Host:
    """The generation the tests run: its main `p`, the script and the host's
    img2img pass (process_images)."""

    def __init__(  # noqa: PLR0913
        self, tabs, *, main=("1girl, green eyes",), typed=None, values=None,
        classes=("face", "face"), styles=(), db=None, comments=None, fails=(),
        cancel_at=None, standalone=False, data=None, detectors=None,
        on_pass=None,
    ):
        self.tabs = [dict(tab) for tab in tabs]
        self.classes = list(classes)
        self.db = db or _StyleDatabase()
        self.styles = list(styles)
        self.fails = set(fails)
        self.cancel_at = cancel_at
        self.on_pass = on_pass  # called when the host starts a region's pass
        self.region = 0  # the regions inpainted so far, over the whole job
        self.passes = []  # the prompts each region was handed to the host with
        self.image = Image.new("RGB", (8, 8), "white")
        self.state = SimpleNamespace(
            interrupted=False, skipped=False, job_count=0,
            assign_current_image=lambda _image: None,
        )
        self.runner = _Runner(_Draws(values or {}, self))
        host_opts = SimpleNamespace(data={"ad_verbose_log": False, **(data or {})})
        mask = Image.new("L", (8, 8), 255)

        def detect(*_args, **kwargs):
            # A sequential class pass detects one class.
            wanted = [c for c in (kwargs.get("classes") or "").split(",") if c]
            names = wanted * 2 if wanted else list(self.classes)
            return SimpleNamespace(
                preview=self.image if names else None, bboxes=[], class_names=names
            )

        runtime = _load_script(
            methods=_INLINE_PROMPT_METHODS | {
                "process", "extra_params", "postprocess_image",
                "_postprocess_image_inner", "is_ad_enabled", "get_args",
                "_will_run_sequential", "postprocess", "before_process_batch",
            },
            functions=_INLINE_PROMPT_FUNCTIONS | {"_parse_class_prompts", "_class_prompt_for"},
            assigns=_INLINE_PROMPT_ASSIGNS,
            state=self.state,
            shared=SimpleNamespace(state=self.state, prompt_styles=self.db, opts=host_opts),
            opts=host_opts,
            scripts=SimpleNamespace(scripts_data=[
                SimpleNamespace(
                    script_class=type("ScriptStripComments", (), {"__module__": "comments.py"}),
                    module=SimpleNamespace(strip_comments=comments),
                )
            ] if comments else []),
            time=time, copy=copy, get_i=get_i, suffix=_ui_suffix(), ordinal=str,
            # The detectors found at startup: name -> file.
            model_mapping=dict(detectors or {}),
            __version__="test", ADetailerArgs=ADetailerArgs,
            parse_csv=lambda text: [c.strip() for c in text.split(",") if c.strip()],
            is_img2img_inpaint=lambda _p: False,
            is_skip_img2img=lambda _p: False,
            _verbose_gen_header=lambda *_args: None,
            _verbose_pass_header=lambda *_args: None,
            _verbose_detection=lambda *_args: None,
            _verbose_pass_result=lambda *_args: None,
            disable_safe_unpickle=nullcontext,
            ultralytics_predict=detect,
            mediapipe_predict=detect,
            get_model_class_names=lambda _model: [],
            ensure_pil_image=lambda im, _mode: im,
            process_images=self.process_images,
            NansException=type("NansException", (Exception,), {}),
            need_call_postprocess=need_call_postprocess,
            need_call_process=need_call_process,
            CNHijackRestore=nullcontext,
            pause_total_tqdm=nullcontext,
            cn_allow_script_control=nullcontext,
            _should_skip_for_hires_only=lambda _p, _args: False,
        )
        script = runtime.AfterDetailerScript()
        script.ultralytics_device = "cpu"
        script.set_skip_img2img = lambda *_args: None
        script.get_i2i_init_image = lambda _p, pp: pp.image
        script.read_params_txt = lambda: ""
        script.write_params_txt = lambda _content: None
        script.get_ad_model = lambda _name: "model.pt"
        script.pred_preprocessing = lambda _p, pred, _args: [mask] * len(pred.class_names)
        script.save_image = lambda *_args, **_kwargs: None
        script._apply_auto_class_guard = lambda *_args: None
        script.fix_p2 = lambda *_args: None
        script.compare_prompt = lambda *_args, **_kwargs: None
        script.get_i2i_p = lambda p, _args, _image: SimpleNamespace(
            init_images=[self.image], prompt="", negative_prompt="",
            styles=list(p.styles), scripts=self.runner, close=lambda: None,
            denoising_strength=0.4, width=512, height=512,
            extra_generation_params=dict(p.extra_generation_params),
        )
        self.script = script
        drawn = [self.db.apply_styles_to_prompt(text, self.styles) for text in main]
        self.p = SimpleNamespace(
            prompt=main[0] if typed is None else typed, negative_prompt="",
            all_prompts=drawn, all_negative_prompts=[""] * len(drawn),
            styles=list(self.styles), scripts=None, iteration=0,
            batch_size=len(drawn), batch_index=0, extra_generation_params={},
        )
        if standalone:
            self.p._ad_standalone = True

    def process_images(self, p2):
        # The host's setup_prompts(), then the scripts' process() hook.
        self.passes.append((p2.prompt, p2.negative_prompt))
        if self.on_pass is not None:
            self.on_pass()
        styles = list(getattr(p2, "styles", None) or [])
        p2.all_prompts = [self.db.apply_styles_to_prompt(p2.prompt, styles)]
        p2.all_negative_prompts = [
            self.db.apply_negative_styles_to_prompt(p2.negative_prompt, styles)
        ]
        if p2.scripts is not None:
            p2.scripts.process(p2)
        region = self.region
        self.region += 1
        if region in self.fails:
            msg = "out of memory"
            raise RuntimeError(msg)
        if region == self.cancel_at:
            self.state.interrupted = True
        return SimpleNamespace(
            images=[self.image],
            all_prompts=list(p2.all_prompts),
            all_negative_prompts=list(p2.all_negative_prompts),
        )

    def start(self):
        """The script's process() hook: the tabs' own parameters."""
        self.script.process(self.p, True, False, *self.tabs)
        return dict(self.p.extra_generation_params)

    def image_parameters(self, index=0):
        """The parameters the host saves image `index` with: what the dict
        holds when postprocess_image() gives the image back."""
        self.p.batch_index = index
        self.script.postprocess_image(
            self.p, SimpleNamespace(image=self.image), True, False, *self.tabs
        )
        return dict(self.p.extra_generation_params)


def _tab(**settings):
    return {"ad_model": "face_yolov8n.pt", **settings}


def _saved(tabs, **host):
    """The parameters of the only image of a generation, and what the tabs'
    settings alone wrote for it."""
    job = _Host(tabs, **host)
    own = job.start()
    return job.image_parameters(), own


def test_a_wildcard_only_in_the_adetailer_prompt_is_saved_with_its_values():
    # The value a wildcard written only in a tab's prompt drew was printed in
    # the console and nowhere else: the image kept the text with the wildcard.
    params, own = _saved(
        [_tab(ad_prompt="__eyecolor__ eyes")], values={"eyecolor": ["green", "blue"]}
    )

    # Each face drew its own value.
    assert params[_RESOLVED] == "green eyes [SEP] blue eyes"
    # The tab's prompt stays as typed: pasted back, it draws again.
    assert params["ADetailer prompt"] == "__eyecolor__ eyes"
    # The negative prompt had nothing to draw: no key for it.
    assert {k: v for k, v in params.items() if k != _RESOLVED} == own


def test_a_variant_in_the_negative_prompt_is_saved_with_its_values():
    params, own = _saved(
        [_tab(ad_prompt="detailed face", ad_negative_prompt="{blurry|lowres}, ugly")],
        values={"blurry|lowres": ["lowres", "blurry"]},
    )

    assert params[_RESOLVED_NEGATIVE] == "lowres, ugly [SEP] blurry, ugly"
    assert params["ADetailer negative prompt"] == "{blurry|lowres}, ugly"
    assert {k: v for k, v in params.items() if k != _RESOLVED_NEGATIVE} == own


@pytest.mark.parametrize(
    ("tab", "host"),
    [
        ({}, {}),
        ({"ad_prompt": "detailed face", "ad_negative_prompt": "blurry"}, {}),
        ({"ad_prompt": "detailed face [SEP] detailed eyes [SEP] [SKIP]"}, {}),
        ({"ad_prompt": "[PROMPT], detailed face", "ad_prompt_append": "sharp"}, {}),
        ({"ad_prompt": "[PROMPT=eyes], [CLASS=face]detailed face[/CLASS]"}, {}),
        ({"ad_prompt": "face <lora:detail:0.8>", "ad_use_main_loras": True}, {}),
        # The host adds the selected styles to the pass's prompts.
        ({"ad_prompt": "detailed face", "ad_negative_prompt": "ugly"}, {"styles": ["cinematic"]}),
        ({}, {"styles": ["cinematic"]}),
        # The host's Comments script takes a comment out: not a wildcard.
        (
            {"ad_prompt": "detailed face\n# my note"},
            {"comments": _a1111_strip_comments, "data": {"enable_prompt_comments": False}},
        ),
        # Sequential classes, each with its own prompt.
        (
            {
                "ad_model_classes": "face,hand", "ad_classes_sequential": True,
                "ad_prompt": "detailed", "ad_class_prompts": "hand: five fingers",
            },
            {},
        ),
    ],
)
def test_a_tab_without_wildcards_saves_the_same_parameters_as_before(tab, host):
    # Nothing was drawn in the pass: the image's parameters are, byte for
    # byte, the ones the tabs' settings wrote before this key existed.
    params, own = _saved([_tab(**tab)], values={"eyecolor": ["green"]}, **host)

    assert _host_line(params) == _host_line(own)
    assert list(params) == list(own)
    # Premise: on the same host, a wildcard in the tab is saved.
    with_wildcard, _own = _saved(
        [_tab(**{**tab, "ad_prompt_append": "__eyecolor__ eyes"})],
        values={"eyecolor": ["green"]}, **host,
    )
    assert "green eyes" in with_wildcard[_RESOLVED]


class _HostAddsNetwork(_Host):
    """A WebUI with Settings > Extra Networks "Add network to prompt" set:
    AUTOMATIC1111's and Forge Neo's ExtraNetworkLora.activate() puts the
    LoRA's tag at the end of the prompts of every pass."""

    def process_images(self, p2):
        result = super().process_images(p2)
        result.all_prompts = [text + "<lora:detail:1.0>" for text in result.all_prompts]
        return result


@pytest.mark.parametrize(
    ("tab", "host", "draws"),
    [
        ({"ad_prompt": "detailed face", "ad_negative_prompt": "blurry"}, {}, True),
        ({}, {}, True),
        ({"ad_prompt": "detailed face [SEP] detailed eyes"}, {}, True),
        ({"ad_prompt": "detailed face"}, {"styles": ["cinematic"]}, True),
        # Two underscores in a LoRA's name are not a wildcard, nor is a brace
        # nobody drew (a WebUI without Dynamic Prompts).
        ({"ad_prompt": "face <lora:soft__light__v2:0.8>"}, {}, False),
        ({"ad_prompt": "face, {best quality}"}, {}, False),
    ],
)
def test_a_lora_tag_the_host_adds_to_every_pass_is_not_a_wildcard_drawn(tab, host, draws):
    # Every image had "ADetailer resolved prompt", without any wildcard: any
    # other text than the one handed over counted as one drawn.
    job = _HostAddsNetwork([_tab(**tab)], values={"eyecolor": ["green"]}, **host)
    if not draws:
        job.runner.alwayson_scripts.clear()
    own = job.start()

    saved = job.image_parameters()

    assert _host_line(saved) == _host_line(own)
    assert list(saved) == list(own)
    assert len(job.passes) == 2
    if draws:
        # On the same host a wildcard is saved, with the tag the pass used.
        drawn = _HostAddsNetwork(
            [_tab(**{**tab, "ad_prompt_append": "__eyecolor__ eyes"})],
            values={"eyecolor": ["green"]}, **host,
        )
        drawn.start()
        resolved = drawn.image_parameters()[_RESOLVED]
        assert "green eyes" in resolved
        assert resolved.endswith("<lora:detail:1.0>")


def test_a_wildcard_with_the_main_prompt_value_is_saved_also_with_a_tag_added():
    # ADetailer itself gave the tab's wildcard the main prompt's value: the
    # text handed to the pass has no wildcard left, and the key is written.
    job = _HostAddsNetwork(
        [_tab(ad_prompt="__eyecolor__ iris [SEP] detailed eyes")],
        typed="1girl, __eyecolor__ eyes", main=("1girl, green eyes",),
    )
    job.start()

    params = job.image_parameters()

    assert params[_RESOLVED] == (
        "green iris<lora:detail:1.0> [SEP] detailed eyes<lora:detail:1.0>"
    )
    assert _RESOLVED_NEGATIVE not in params


class _DrawsOwnMarks(_Draws):
    """Dynamic Prompts with its own wildcard marker (Settings > Dynamic
    Prompts "Wildcard wrap"): @@name@@."""

    def _draw(self, text):
        region = self.host.region
        return re.sub(
            r"@@(\w+)@@",
            lambda m: self.values[m.group(1)][region % len(self.values[m.group(1)])],
            text,
        )


@pytest.mark.parametrize("own_marks", [True, False])
def test_dynamic_prompts_own_wildcard_marker_counts_as_a_wildcard(own_marks):
    data = {"dp_parser_wildcard_wrap": "@@"} if own_marks else {}
    job = _Host([_tab(ad_prompt="@@eyecolor@@ eyes")], data=data)
    job.runner.alwayson_scripts[:] = [_DrawsOwnMarks({"eyecolor": ["green", "blue"]}, job)]
    own = job.start()

    params = job.image_parameters()

    if own_marks:
        assert params[_RESOLVED] == "green eyes [SEP] blue eyes"
    else:
        # Nothing Dynamic Prompts reads: the pass runs without it.
        assert job.passes == [("@@eyecolor@@ eyes", "")] * 2
        assert params == own


def test_each_sep_part_keeps_its_place_among_the_resolved_prompts():
    # Upstream once saved the pass's prompt in place of the tab's and lost
    # every [SEP] part but one: here each region has its own part, the ones
    # with nothing to draw and the skipped ones included.
    params, _own = _saved(
        [_tab(ad_prompt="detailed face [SEP] [SKIP] [SEP] __eyecolor__ eyes")],
        values={"eyecolor": ["green", "blue", "grey"]},
        classes=("face", "face", "face", "face"),
    )

    # The 2nd region is skipped, so the host inpaints three: the last part
    # serves the 3rd and the 4th face, each with the value it drew.
    assert params[_RESOLVED] == "detailed face [SEP] [SKIP] [SEP] blue eyes [SEP] grey eyes"


def test_regions_that_used_the_same_prompt_are_listed_once():
    # As in a prompt typed with [SEP], the last part is also the one of the
    # regions after it.
    params, _own = _saved(
        [_tab(ad_prompt="__eyecolor__ eyes")], values={"eyecolor": ["green"]},
        classes=("face", "face", "face"),
    )

    assert params[_RESOLVED] == "green eyes"


def test_each_image_of_a_batch_saves_its_own_values():
    # The images of a batch, the batches of a job and the cells of an X/Y/Z
    # grid share the parameters dict.
    job = _Host(
        [_tab(ad_prompt="__eyecolor__ eyes")],
        main=("1girl", "1boy", "landscape"),
        values={"eyecolor": ["green", "blue", "grey", "brown"]},
    )
    own = job.start()

    first = job.image_parameters(0)
    second = job.image_parameters(1)
    # Nothing detected on the third image: it keeps none of the second's.
    job.classes = []
    third = job.image_parameters(2)

    assert first[_RESOLVED] == "green eyes [SEP] blue eyes"
    assert second[_RESOLVED] == "grey eyes [SEP] brown eyes"
    assert third == own


def test_after_a_job_of_several_images_the_last_images_values_do_not_stay():
    # The X/Y/Z plot script builds its grid's parameters from this dict when
    # the job of its first cell is over, with the prompt and seed of the
    # cell's first image: the keys left there were the last image's.
    job = _Host(
        [_tab(ad_prompt="__eyecolor__ eyes")],
        main=("1girl", "1boy"), values={"eyecolor": ["green", "blue", "red", "grey"]},
    )
    own = job.start()
    first = job.image_parameters(0)
    last = job.image_parameters(1)
    assert first[_RESOLVED] == "green eyes [SEP] blue eyes"
    assert last[_RESOLVED] == "red eyes [SEP] grey eyes"

    # The stand-in result postprocess_image() gives the scripts has no
    # images: the image being detailed keeps its keys.
    job.script.postprocess(copy(job.p), SimpleNamespace(images=[]))
    assert job.p.extra_generation_params == last
    job.script.postprocess(job.p, SimpleNamespace(images=[job.image, job.image]))

    assert job.p.extra_generation_params == own
    # The images were saved before, each with its own values.
    assert (first[_RESOLVED], last[_RESOLVED]) == (
        "green eyes [SEP] blue eyes", "red eyes [SEP] grey eyes",
    )


def _host_before_process_batch(job, batch):
    """The WebUI's ScriptRunner.before_process_batch for batch `batch` (Batch
    count), with the arguments AUTOMATIC1111 and Forge Neo give it. Their
    Script class has an empty one for a script without its own."""
    job.p.iteration = batch
    hook = getattr(job.script, "before_process_batch", lambda *_args, **_kwargs: None)
    hook(
        job.p, True, False, *job.tabs,
        batch_number=batch, prompts=["1girl"], seeds=[1], subseeds=[1],
    )


def test_the_next_batch_of_a_job_starts_without_the_values_of_the_batch_before():
    # Batch count 2 with hires fix: the WebUI saves the "before highres fix"
    # copies of the second batch while it samples, before the first
    # postprocess_image() of that batch, with what the dict holds then: it
    # still held the keys of the first batch's last image.
    job = _Host(
        [_tab(ad_prompt="__eyecolor__ eyes")],
        main=("1girl", "1boy"), values={"eyecolor": ["green", "blue", "red", "grey"]},
    )
    job.p.batch_size = 1
    own = job.start()

    _host_before_process_batch(job, 0)
    assert job.p.extra_generation_params == own
    first = job.image_parameters(0)
    _host_before_process_batch(job, 1)
    # What the second batch's copies are saved with.
    held = dict(job.p.extra_generation_params)
    second = job.image_parameters(0)

    assert held == own
    # Each image is still saved with its own values.
    assert first[_RESOLVED] == "green eyes [SEP] blue eyes"
    assert second[_RESOLVED] == "red eyes [SEP] grey eyes"


def test_the_batch_hook_takes_any_arguments_and_any_p():
    # Called by the WebUI for every script, also for a pass of ADetailer's
    # own and for a p without parameters: never an error.
    job = _Host([_tab()])
    hook = job.script.before_process_batch
    for p in (SimpleNamespace(), SimpleNamespace(extra_generation_params=None), None):
        hook(p)
        hook(p, True, False, batch_number=0, prompts=[], seeds=[], subseeds=[])
    other = {"Steps": 20, "ADetailer model": "face_yolov8n.pt", _RESOLVED: "green eyes"}
    p = SimpleNamespace(extra_generation_params=dict(other))
    hook(p, batch_number=1)
    # Only the keys of the resolved prompts go.
    assert p.extra_generation_params == {"Steps": 20, "ADetailer model": "face_yolov8n.pt"}


def test_after_a_job_of_one_image_its_values_stay():
    # They are the ones of the image the prompt and seed belong to.
    job = _Host([_tab(ad_prompt="__eyecolor__ eyes")], values={"eyecolor": ["green", "blue"]})
    job.start()
    saved = job.image_parameters(0)

    job.script.postprocess(job.p, SimpleNamespace(images=[job.image]))

    assert job.p.extra_generation_params == saved
    assert saved[_RESOLVED] == "green eyes [SEP] blue eyes"


def test_each_tab_saves_its_own_resolved_prompts():
    params, own = _saved(
        [
            _tab(ad_prompt="detailed face"),
            _tab(ad_prompt="__eyecolor__ eyes", ad_negative_prompt="__flaws__"),
            _tab(ad_prompt="{long|short} hair"),
        ],
        values={
            "eyecolor": ["green"], "flaws": ["blurry"], "long|short": ["long", "short"],
        },
        classes=("face",),
    )

    # The 1st tab drew nothing; the regions are counted over the whole job,
    # so the 3rd tab's only region is the third one inpainted.
    assert {k: v for k, v in params.items() if k not in own} == {
        _RESOLVED + " 2nd": "green eyes",
        _RESOLVED_NEGATIVE + " 2nd": "blurry",
        _RESOLVED + " 3rd": "long hair",
    }


def test_the_resolved_prompt_is_the_whole_prompt_the_region_was_inpainted_with():
    # [PROMPT], the text always added, the LoRAs of the main prompt and the
    # selected styles are part of what the pass used.
    db = _StyleDatabase()
    params, _own = _saved(
        [
            _tab(
                ad_prompt="[PROMPT], __eyecolor__ eyes", ad_prompt_append="sharp",
                ad_negative_prompt="ugly",
            )
        ],
        main=("1girl, park",), values={"eyecolor": ["green", "blue"]},
        styles=["cinematic"], db=db,
    )

    assert params[_RESOLVED] == (
        "cinematic still of 1girl, park, green eyes, sharp, 35mm film"
        " [SEP] cinematic still of 1girl, park, blue eyes, sharp, 35mm film"
    )
    # The negative prompt only got the style, which is not a wildcard drawn.
    assert _RESOLVED_NEGATIVE not in params


def test_a_wildcard_that_took_the_main_prompt_value_is_saved_too():
    # The tab's __eyecolor__ takes the value the main prompt drew, before the
    # pass and without a script: the tab's prompt as typed does not say it.
    job = _Host(
        [_tab(ad_prompt="__eyecolor__ iris [SEP] detailed eyes")],
        typed="1girl, __eyecolor__ eyes", main=("1girl, green eyes",),
    )
    job.start()

    params = job.image_parameters()

    assert job.passes == [("green iris", ""), ("detailed eyes", "")]
    assert params[_RESOLVED] == "green iris [SEP] detailed eyes"
    assert params["ADetailer prompt"] == "__eyecolor__ iris [SEP] detailed eyes"
    assert _RESOLVED_NEGATIVE not in params


def test_a_region_that_fails_keeps_its_place_with_the_prompt_it_was_given():
    params, _own = _saved(
        [_tab(ad_prompt="__eyecolor__ eyes")],
        values={"eyecolor": ["green", "blue", "grey"]},
        classes=("face", "face", "face"), fails={1},
    )

    assert params[_RESOLVED] == "green eyes [SEP] __eyecolor__ eyes [SEP] grey eyes"


def test_a_cancelled_pass_saves_no_resolved_prompt():
    # The pass is discarded and the image keeps none of its regions.
    job = _Host(
        [_tab(ad_prompt="__eyecolor__ eyes")],
        values={"eyecolor": ["green", "blue"]}, cancel_at=1,
    )
    own = job.start()

    assert job.image_parameters() == own
    # Premise: without the cancel the same job saves the key.
    assert _RESOLVED in _saved(
        [_tab(ad_prompt="__eyecolor__ eyes")], values={"eyecolor": ["green", "blue"]}
    )[0]


@pytest.mark.parametrize(("cancel_at", "saved"), [(None, True), (3, False)])
def test_sequential_classes_save_the_prompts_of_every_class_or_of_none(cancel_at, saved):
    # Each class is a pass of its own with the tab's number. A sequential tab
    # cancelled halfway is rolled back, the classes done so far included.
    job = _Host(
        [
            _tab(
                ad_model_classes="face,hand", ad_classes_sequential=True,
                ad_prompt="__eyecolor__ eyes",
                ad_class_prompts="hand: {five|slender} fingers",
            )
        ],
        values={"eyecolor": ["green", "blue"], "five|slender": ["five", "slender"]},
        cancel_at=cancel_at,
    )
    own = job.start()

    params = job.image_parameters()

    if saved:
        assert params[_RESOLVED] == (
            "green eyes [SEP] blue eyes [SEP] five fingers [SEP] slender fingers"
        )
    else:
        assert job.region == 4  # the first class was inpainted
        assert params == own


def test_run_adetailer_on_an_image_records_no_resolved_prompt():
    # Its pass runs no script that draws wildcards, and its result keeps the
    # parameters the host put on it: the shell's own are left as they were.
    job = _Host(
        [_tab(ad_prompt="__eyecolor__ eyes")], values={"eyecolor": ["green", "blue"]},
        standalone=True,
    )
    before = dict(job.p.extra_generation_params)

    args = ADetailerArgs(**job.tabs[0])
    assert job.script._postprocess_image_inner(
        job.p, SimpleNamespace(image=job.image), args
    ) is True

    assert job.p.extra_generation_params == before
    # Premise: the same pass of a generation records it.
    del job.p._ad_standalone
    job.region = 0
    job.script._postprocess_image_inner(job.p, SimpleNamespace(image=job.image), args)
    assert job.p.extra_generation_params[_RESOLVED] == "green eyes [SEP] blue eyes"


def test_the_next_xyz_cell_does_not_keep_the_resolved_prompts_of_the_one_before():
    # A cell is a copy of p that shares the parameters dict: its process()
    # hook writes the tabs' own keys again, before any image of the cell.
    job = _Host([_tab(ad_prompt="__eyecolor__ eyes")], values={"eyecolor": ["green"]})
    own = job.start()
    assert _RESOLVED in job.image_parameters()

    cell = copy(job.p)
    job.script.process(cell, True, False, *job.tabs)

    assert cell.extra_generation_params is job.p.extra_generation_params
    assert cell.extra_generation_params == own


@pytest.mark.parametrize(
    "value",
    [
        "green eyes, long lashes",  # commas
        "eyes: green",  # a colon
        'a "quoted" word, and a comma',
        "first line\nsecond line",
        "detailed face [SEP] 1girl, {green} eyes",
        "Steps: 20, Negative prompt: none",  # looks like other parameters
    ],
)
def test_pasting_an_image_with_a_resolved_prompt_works_as_before(value):
    # The WebUI quotes the value like any other one, and pasting ignores the
    # key: the tab gets exactly what an image without it gives.
    job = _Host(
        [_tab(ad_prompt="__eyecolor__ eyes", ad_negative_prompt="__eyecolor__ ugly")],
        values={"eyecolor": [value]}, classes=("face",),
    )
    own = job.start()
    params = job.image_parameters()
    assert params[_RESOLVED] == f"{value} eyes"

    line = _host_line({"Steps": 20, "Sampler": "Euler a", **params})
    read = _host_read(line)
    old = _host_read(_host_line({"Steps": 20, "Sampler": "Euler a", **own}))

    # Read back whole, and nothing else on the line is disturbed.
    assert read[_RESOLVED] == f"{value} eyes"
    assert read[_RESOLVED_NEGATIVE] == f"{value} ugly"
    assert {k: v for k, v in read.items() if k in old} == old
    # The paste callback and the paste itself: the same tab as an old image.
    callback = _paste_callback()
    callback._clear_missing_class_prompts("", read)
    callback._clear_missing_class_prompts("", old)
    before = ADetailerArgs(ad_model="hand_yolov8n.pt", ad_prompt="old prompt").dict()
    assert _host_paste(read, before) == _host_paste(old, before)
    assert _host_paste(read, before)["ad_prompt"] == "__eyecolor__ eyes"


@pytest.mark.parametrize(
    ("handed", "used", "resolved"),
    [
        ("__eyecolor__ eyes", "green eyes", True),
        ("detailed face", "detailed face", False),
        # Other spaces and line breaks are not a wildcard drawn.
        ("detailed face,\n sharp", "detailed face, sharp", False),
        # Nor is a comment the host's Comments script took out of the pass.
        ("detailed face\n# my note", "detailed face", False),
        ("__eyecolor__ eyes\n# my note", "green eyes", True),
        # Nor a text the host or another script added to a prompt without
        # wildcards, or put in its place.
        ("detailed face", "detailed face<lora:detail:1.0>", False),
        ("detailed face", "portrait, intricate details, 8k", False),
        ("detailed face\n# my note", "detailed face<lora:detail:1.0>", False),
        # A wildcard or variant still there whole was not drawn.
        ("{long|short} hair", "{long|short} hair<lora:detail:1.0>", False),
        ("__eyecolor__ eyes\n# my note", "__eyecolor__ eyes <lora:detail:1.0>", False),
        ("{long|short} hair", "long hair<lora:detail:1.0>", True),
    ],
)
def test_only_a_text_the_pass_changed_counts_as_resolved(handed, used, resolved):
    region = [handed, handed, False, False]
    processed = SimpleNamespace(all_prompts=[used], all_negative_prompts=[used])

    p_method.note_used_prompts(region, processed, strip=_a1111_strip_comments)

    assert region == [used, used, resolved, resolved]
    # A host answer without the prompts leaves the region as it was.
    region = [handed, handed, False, False]
    p_method.note_used_prompts(region, SimpleNamespace(images=[]))
    assert region == [handed, handed, False, False]
    # And a wildcard ADetailer itself resolved stays resolved.
    region = [used, used, True, False]
    p_method.note_used_prompts(region, processed)
    assert region == [used, used, True, False]
    # Other spaces alone never count, also on a WebUI without comments.
    region = ["detailed face,\n sharp", "", False, False]
    spaced = SimpleNamespace(all_prompts=["detailed face, sharp"], all_negative_prompts=[""])
    p_method.note_used_prompts(region, spaced)
    assert region == ["detailed face, sharp", "", False, False]


def test_a_very_long_resolved_prompt_is_cut_at_whole_regions():
    # A JPEG holds its parameters in 64 KB and the WebUI fails to save one
    # that needs more: many regions with a long prompt each stay below that.
    long_prompt = ", ".join(["a very detailed face"] * 60)  # about 1300 characters
    p = SimpleNamespace(extra_generation_params={})
    regions = [[f"{long_prompt} {k}", "", True, False] for k in range(8)]

    p_method.keep_resolved_prompts(p, 1, regions)

    value = p.extra_generation_params[_RESOLVED + " 2nd"]
    assert len(value) <= p_method.RESOLVED_PROMPT_MAX + len(" [SEP] (+8 more)")
    kept = value.split(" [SEP] ")
    assert kept[:-1] == [region[0] for region in regions[: len(kept) - 1]]
    assert kept[-1] == f"(+{8 - (len(kept) - 1)} more)"
    assert 1 < len(kept) < 8


def _utf8(text):
    return len(text.encode("utf-8"))


@pytest.mark.parametrize("letter", ["x", "あ"])
def test_a_region_too_long_for_the_key_is_cut_and_says_so(letter):
    # One region with a prompt longer than the key's room wrote no key at
    # all, without a word: its beginning is saved, with a note at the end.
    long_prompt = "detailed face, " + letter * 4001
    p = SimpleNamespace(extra_generation_params={})

    p_method.keep_resolved_prompts(p, 0, [[long_prompt, "blurry", True, False]])

    value = p.extra_generation_params[_RESOLVED]
    assert value.endswith(p_method.RESOLVED_PROMPT_CUT)
    head = value[: -len(p_method.RESOLVED_PROMPT_CUT)]
    # The beginning of the prompt, cut at a whole character.
    assert long_prompt.startswith(head)
    assert head.startswith("detailed face, " + letter * 1000)
    assert p_method.RESOLVED_PROMPT_MAX - 4 < _utf8(value) <= p_method.RESOLVED_PROMPT_MAX
    assert _RESOLVED_NEGATIVE not in p.extra_generation_params  # nothing drawn there
    # More regions after it: the note says how many are left out.
    p_method.start_resolved_prompts(p)
    p_method.keep_resolved_prompts(
        p, 0, [[long_prompt, "", True, False], ["eyes", "", False, False]]
    )
    value = p.extra_generation_params[_RESOLVED]
    assert value.endswith(p_method.RESOLVED_PROMPT_CUT + " [SEP] (+1 more)")
    # A prompt that just fits is saved whole.
    p_method.start_resolved_prompts(p)
    fits = "y" * p_method.RESOLVED_PROMPT_MAX
    p_method.keep_resolved_prompts(p, 0, [[fits, "", True, False]])
    assert p.extra_generation_params == {_RESOLVED: fits}


@pytest.mark.parametrize("letter", ["x", "あ"])
def test_the_resolved_prompts_of_all_the_tabs_share_one_room(letter):
    # The limit was for each key: with many tabs the keys together came near
    # the 64 KB a JPEG holds, where the WebUI fails to save the parameters.
    p = SimpleNamespace(extra_generation_params={"Steps": 20})
    tabs = 8
    for n in range(tabs):
        regions = [
            [f"{n}-{j} " + letter * 660, f"{n}-{j} " + letter * 660, True, True]
            for j in range(6)
        ]
        p_method.keep_resolved_prompts(p, n, regions)

    params = p.extra_generation_params
    values = {k: v for k, v in params.items() if k.startswith(p_method.RESOLVED_PROMPT_KEYS)}
    # Every tab and key says something, the cut ones with their note.
    assert len(values) == tabs * 2
    note = len(" [SEP] (+6 more)") + len(p_method.RESOLVED_PROMPT_CUT)
    total = sum(_utf8(value) for value in values.values())
    assert total <= p_method.RESOLVED_PROMPTS_TOTAL_MAX + note * len(values)
    for value in values.values():
        assert _utf8(value) <= p_method.RESOLVED_PROMPT_MAX + note
    # In the JPEG's comment (UTF-16) they leave most of the 64 KB free.
    assert len(_host_line(params).encode("utf_16_be")) < 30000
    # The first tabs are whole regions, the last ones only the note.
    assert values[_RESOLVED].startswith(f"0-0 {letter}")
    assert values[_RESOLVED].endswith(" more)")
    last = p_method._tab_suffix(tabs - 1)
    assert values[_RESOLVED + last] == values[_RESOLVED_NEGATIVE + last] == "(+6 more)"
    # The tab that meets the end of the room saves the beginning of a region.
    assert [
        value for value in values.values()
        if value.endswith(p_method.RESOLVED_PROMPT_CUT + " [SEP] (+5 more)")
    ]
    # The line is read back whole and pasting works as before.
    assert _host_read(_host_line(params))[_RESOLVED + last] == "(+6 more)"


def test_the_documents_say_how_a_long_resolved_prompt_is_cut():
    # The README and the CHANGELOG quote the marks of a value that is cut, and
    # how many bytes a key and an image hold: each one as the code writes it.
    root = Path(__file__).parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert (
        "about 4000 bytes for each key and about 12000 for all those of an image): the "
        'regions that do not fit are left out and counted, as in "(+2 more)", and when not '
        "even the first region's prompt fits, it is cut and ends with \" ... (cut)\"."
    ) in readme
    assert (
        "each key holds about 4000 bytes and all those of one image about 12000. From the "
        "first region's prompt that does not fit, the regions are left out and counted, as "
        'in "(+2 more)"; when not even the first one fits, it is cut and ends with " ... '
        '(cut)".'
    ) in changelog
    assert (p_method.RESOLVED_PROMPT_MAX, p_method.RESOLVED_PROMPTS_TOTAL_MAX) == (4000, 12000)
    # Three regions with room for one: the next two are counted.
    parts = ["a" * 1000, "b" * 1000, "c" * 10]
    assert p_method._resolved_value(parts, 1500) == "a" * 1000 + " [SEP] (+2 more)"
    # Not even the first fits: its beginning, with the mark, then the others.
    value = p_method._resolved_value(parts, 500)
    assert value.endswith(" ... (cut) [SEP] (+2 more)")
    assert value.startswith("a" * 400)


def test_a_sequential_tab_does_not_count_its_own_keys_twice():
    # Each class pass writes the tab's keys again with the regions so far.
    # Here the two tabs before it leave room for little more than its value.
    regions = [[f"{j} " + "x" * 1500, "", True, False] for j in range(4)]
    whole = " [SEP] ".join([regions[0][0], regions[1][0], "(+2 more)"])
    p = SimpleNamespace(extra_generation_params={})
    for n in range(2):
        p_method.keep_resolved_prompts(p, n, [["y" * 3990, "", True, False]])
    for j in range(4):
        p_method.keep_resolved_prompts(p, 2, [regions[j]])

    assert p.extra_generation_params[_RESOLVED + " 3rd"] == whole
    # A tab rolled back gives its room back to the next one.
    p_method.drop_resolved_prompts(p, 2)
    p_method.keep_resolved_prompts(p, 3, regions)
    assert _RESOLVED + " 3rd" not in p.extra_generation_params
    assert p.extra_generation_params[_RESOLVED + " 4th"] == whole
    # Premise: with the 3rd tab's value still there, the 4th has to be cut.
    p_method.keep_resolved_prompts(p, 4, regions)
    assert p.extra_generation_params[_RESOLVED + " 5th"].endswith(
        p_method.RESOLVED_PROMPT_CUT + " [SEP] (+3 more)"
    )


def test_a_blank_resolved_prompt_is_never_written():
    # Also with no room left: an empty prompt has nothing to cut.
    p = SimpleNamespace(extra_generation_params={})
    for n in range(4):
        p_method.keep_resolved_prompts(p, n, [["x" * 3990, "", True, True]])
    p_method.keep_resolved_prompts(p, 4, [["", "", True, True], ["", "", True, True]])

    assert _RESOLVED + " 5th" not in p.extra_generation_params
    assert _RESOLVED_NEGATIVE not in p.extra_generation_params
    assert p.extra_generation_params[_RESOLVED + " 4th"].endswith(p_method.RESOLVED_PROMPT_CUT)


def test_the_resolved_prompt_keys_end_like_the_tabs_other_keys():
    # aaaaaa/p_method.py cannot import the UI module: its own tab ending is
    # the one of aaaaaa.ui.suffix.
    suffix = _ui_suffix()
    assert [p_method._tab_suffix(n) for n in range(130)] == [suffix(n) for n in range(130)]


def test_no_script_or_ui_code_reads_the_resolved_prompt_as_a_tab_setting():
    # Output only: not an argument of a tab, not a paste field, not read by
    # the browser scripts.
    from adetailer.args import ALL_ARGS

    assert len(ALL_ARGS) == 63
    names = [name for _attr, name in ALL_ARGS]
    for key in p_method.RESOLVED_PROMPT_KEYS:
        assert key not in names
        # No tab setting's key starts with it, and it starts with none that
        # is matched by its beginning.
        assert not [name for name in names if name.startswith(key)]
        assert key.startswith("ADetailer ")  # dropped with the other keys
    ui = (_ROOT / "aaaaaa" / "ui.py").read_text(encoding="utf-8")
    assert "resolved prompt" not in ui
    for js in (_ROOT / "javascript").glob("*.js"):
        assert "resolved" not in js.read_text(encoding="utf-8")
    # The pattern that finds a tab's detector in pasted parameters matches the
    # whole key only.
    tree = ast.parse((_ROOT / "scripts" / "!adetailer.py").read_text(encoding="utf-8"))
    source = ast.unparse(tree)
    assert "_INFOTEXT_MODEL_KEY.fullmatch(" in source
    assert "_INFOTEXT_MODEL_KEY.match(" not in source


# The hashes of a tab's detector file and of its own checkpoint.

_MODEL_HASH = "ADetailer model hash"
_CHECKPOINT_HASH = "ADetailer checkpoint hash"


def _detector_file(folder, name, content):
    """A detector file, and the first 10 characters of its sha256, as the
    WebUI shortens the one of a checkpoint for "Model hash"."""
    file = folder / name
    file.write_bytes(content)
    return str(file), hashlib.sha256(content).hexdigest()[:10]


@pytest.fixture
def fresh_hashes(monkeypatch):
    # The hashes remembered so far belong to the session, not to a test.
    monkeypatch.setattr(p_method, "_DETECTOR_HASHES", {})


def _webui_checkpoints(monkeypatch, checkpoints, aliases=True, no_hashing=False):
    """The WebUI's modules.sd_models with these checkpoints (title -> the
    short hash it holds, None for one never hashed) and modules.shared with
    its --no-hashing flag. Returns the checkpoints' records, and the ones the
    code asked the WebUI to hash."""
    asked = []

    class Info:
        def __init__(self, title, short):
            self.title = title
            self.shorthash = short
            self.sha256 = short and short + "0" * 54

        def calculate_shorthash(self):
            # Reads the whole file, several GB: only the WebUI may call it.
            asked.append(self.title)
            return self.shorthash

    records = {title: Info(title, short) for title, short in checkpoints.items()}

    def closest_match(name):
        # The WebUI's get_closet_checkpoint_match: a title that holds the text.
        found = [info for title, info in records.items() if name in title]
        return records.get(name) or (found[0] if found else None)

    sd_models = SimpleNamespace(get_closet_checkpoint_match=closest_match)
    if aliases:
        sd_models.checkpoint_aliases = records
    shared = SimpleNamespace(cmd_opts=SimpleNamespace(no_hashing=no_hashing))
    monkeypatch.setitem(
        sys.modules, "modules", SimpleNamespace(sd_models=sd_models, shared=shared)
    )
    return records, asked


def _old_parameters(tabs):
    """What the tabs' settings wrote before the hash keys existed."""
    suffix = _ui_suffix()
    params = {}
    for n, tab in enumerate(tabs):
        params.update(ADetailerArgs(**tab).extra_params(suffix=suffix(n)))
    # An empty prompt, class prompt or detection number is left out.
    params = {k: v for k, v in params.items() if str(v).strip()}
    params["ADetailer version"] = "test"
    return params


def test_the_hash_of_each_tabs_detector_file_is_saved(tmp_path, monkeypatch, fresh_hashes):
    # Only the detector's name was saved: two files with the same name could
    # not be told apart, and no site could look the detector up.
    face, face_hash = _detector_file(tmp_path, "face_yolov8n.pt", b"face weights")
    hand, hand_hash = _detector_file(tmp_path, "hand_yolov8n.pt", b"hand weights")
    # A MediaPipe detector is a name, not a file: a file that happens to have
    # that name in the folder the WebUI runs from is not its file.
    _detector_file(tmp_path, "mediapipe_face_full", b"something else")
    monkeypatch.chdir(tmp_path)
    detectors = {
        "face_yolov8n.pt": face,
        "hand_yolov8n.pt": hand,
        "mediapipe_face_full": "mediapipe_face_full",
        # A detector whose file is gone.
        "eyes.pt": str(tmp_path / "eyes.pt"),
    }
    tabs = [
        _tab(),
        _tab(ad_model="hand_yolov8n.pt"),
        _tab(ad_model="mediapipe_face_full"),
        _tab(ad_model="eyes.pt"),
        _tab(ad_model="face_yolov8n.pt", ad_tab_enable=False),  # does not run
    ]

    params = _Host(tabs, detectors=detectors).start()

    assert {k: v for k, v in params.items() if "hash" in k} == {
        _MODEL_HASH: face_hash,
        _MODEL_HASH + " 2nd": hand_hash,
    }
    assert len(face_hash) == 10
    # The other parameters are the ones written before, in the same order.
    old = _old_parameters(tabs)
    assert [(k, v) for k, v in params.items() if "hash" not in k] == list(old.items())


def test_a_detector_file_is_read_once_until_it_is_replaced(
    tmp_path, monkeypatch, fresh_hashes
):
    # Every generation writes the parameters: the file is hashed the first
    # time and again only when another file took its place.
    face, face_hash = _detector_file(tmp_path, "face_yolov8n.pt", b"face weights")
    opened = []
    real_open = open

    def counting_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(p_method, "open", counting_open, raising=False)
    job = _Host([_tab(), _tab()], detectors={"face_yolov8n.pt": face})

    for _generation in range(3):
        params = job.start()
        assert params[_MODEL_HASH] == params[_MODEL_HASH + " 2nd"] == face_hash
    assert opened == [face]

    _face, new_hash = _detector_file(tmp_path, "face_yolov8n.pt", b"other, longer weights")
    assert job.start()[_MODEL_HASH] == new_hash != face_hash
    assert opened == [face, face]


@pytest.mark.parametrize("case", ["no_hashing", "folder", "empty", "too_large"])
def test_a_detector_that_cannot_be_hashed_writes_no_hash(
    tmp_path, monkeypatch, fresh_hashes, case
):
    file, _hash = _detector_file(tmp_path, "face_yolov8n.pt", b"face weights")
    job = _Host([_tab()], detectors={"face_yolov8n.pt": file})
    # Premise: this detector's hash is written.
    assert _MODEL_HASH in job.start()
    monkeypatch.setattr(p_method, "_DETECTOR_HASHES", {})

    if case == "no_hashing":
        # The WebUI's --no-hashing: the user asked for no file to be hashed.
        _webui_checkpoints(monkeypatch, {}, no_hashing=True)
    elif case == "folder":
        Path(file).unlink()
        Path(file).mkdir()
    elif case == "empty":
        Path(file).write_bytes(b"")
    else:
        monkeypatch.setattr(p_method, "DETECTOR_HASH_MAX_BYTES", 4)

    assert job.start() == _old_parameters(job.tabs)


def test_the_checkpoint_hash_saved_is_the_one_the_webui_already_knows(monkeypatch):
    # The tab's own checkpoint was saved by name only. Its hash comes from
    # the WebUI's record of the checkpoint and is never worked out here:
    # that reads a file of several GB.
    records, asked = _webui_checkpoints(
        monkeypatch,
        {"known.safetensors [1234567890]": "1234567890", "fresh.safetensors": None},
    )
    own = {"ad_use_checkpoint": True}
    tabs = [
        _tab(ad_checkpoint="known.safetensors [1234567890]", **own),
        _tab(ad_checkpoint="fresh.safetensors", **own),  # never hashed
        _tab(ad_checkpoint="Use same checkpoint", **own),
        _tab(ad_checkpoint="gone.safetensors", **own),  # not a checkpoint here
        _tab(ad_checkpoint="known.safetensors [1234567890]"),  # box not ticked
    ]

    params = _Host(tabs).start()

    assert {k: v for k, v in params.items() if "hash" in k} == {
        _CHECKPOINT_HASH: "1234567890"
    }
    assert asked == []
    assert [(k, v) for k, v in params.items() if "hash" not in k] == list(
        _old_parameters(tabs).items()
    )
    # A record with the long hash only gives its first 10 characters.
    records["fresh.safetensors"].sha256 = "abcdef0123456789" * 4
    assert _Host(tabs).start()[_CHECKPOINT_HASH + " 2nd"] == "abcdef0123"
    assert asked == []


@pytest.mark.parametrize("host", ["no_aliases", "no_checkpoints"])
def test_a_webui_without_the_checkpoint_records_writes_what_it_can(monkeypatch, host):
    tabs = [_tab(ad_use_checkpoint=True, ad_checkpoint="known.safetensors")]
    if host == "no_aliases":
        # Looked up with the WebUI's other function, which also finds a
        # checkpoint whose name holds the text: "None" and "Use same
        # checkpoint" are choices of the tab's menu, not checkpoints.
        _webui_checkpoints(
            monkeypatch,
            {"known.safetensors": "1234567890", "None of Use same checkpoint.ckpt": "aaaaaaaaaa"},
            aliases=False,
        )
        tabs += [
            _tab(ad_use_checkpoint=True, ad_checkpoint="None"),
            _tab(ad_use_checkpoint=True, ad_checkpoint="Use same checkpoint"),
        ]
        params = _Host(tabs).start()
        assert {k: v for k, v in params.items() if "hash" in k} == {
            _CHECKPOINT_HASH: "1234567890"
        }
    else:
        monkeypatch.setitem(sys.modules, "modules", SimpleNamespace())
        assert _Host(tabs).start() == _old_parameters(tabs)
        # Premise: with the records, the same tab writes the hash.
        _webui_checkpoints(monkeypatch, {"known.safetensors": "1234567890"})
        assert _CHECKPOINT_HASH in _Host(tabs).start()


def test_a_checkpoint_hash_the_webui_works_out_while_detailing_is_saved(monkeypatch):
    # The first time a checkpoint is used, the WebUI hashes it as the pass
    # loads it: after the generation started. The image still gets the hash.
    records, asked = _webui_checkpoints(monkeypatch, {"fresh.safetensors": None})

    def host_loads_the_checkpoint():
        records["fresh.safetensors"].shorthash = "feedc0ffee"

    tabs = [
        _tab(ad_prompt="detailed face"),
        _tab(ad_use_checkpoint=True, ad_checkpoint="fresh.safetensors"),
    ]
    job = _Host(tabs, on_pass=host_loads_the_checkpoint, classes=("face",))
    started = job.start()
    assert started == _old_parameters(tabs)

    params = job.image_parameters()

    assert params == {**started, _CHECKPOINT_HASH + " 2nd": "feedc0ffee"}
    assert asked == []
    # The next generation writes it with the tab's other keys.
    assert job.start()[_CHECKPOINT_HASH + " 2nd"] == "feedc0ffee"


def test_without_wildcards_and_known_hashes_the_parameters_are_the_same_as_before(
    tmp_path, monkeypatch, fresh_hashes
):
    # Byte for byte: no detector file to hash, a checkpoint the WebUI never
    # hashed, and nothing for the pass to draw.
    _webui_checkpoints(monkeypatch, {"fresh.safetensors": None})
    detectors = {"face_yolov8n.pt": str(tmp_path / "face_yolov8n.pt")}
    own = {"ad_use_checkpoint": True, "ad_checkpoint": "fresh.safetensors"}
    tabs = [
        _tab(ad_prompt="[PROMPT], detailed face", ad_negative_prompt="blurry"),
        _tab(ad_prompt="eyes [SEP] iris", **own),
        _tab(ad_model="mediapipe_face_mesh", **own),
    ]
    job = _Host(tabs, detectors=detectors, values={"eyecolor": ["green"]})
    old = _old_parameters(tabs)

    started = job.start()
    saved = job.image_parameters()

    for params in (started, saved):
        assert _host_line({"Steps": 20, **params}) == _host_line({"Steps": 20, **old})
    # Premise: the same job with the detector's file in place writes its hash
    # for the two tabs that use it, and nothing else changes.
    _file, face_hash = _detector_file(tmp_path, "face_yolov8n.pt", b"face weights")
    assert job.start() == {
        **old, _MODEL_HASH: face_hash, _MODEL_HASH + " 2nd": face_hash,
    }


def test_pasting_an_image_with_the_hashes_works_as_before(
    tmp_path, monkeypatch, capsys, fresh_hashes
):
    # Pasting ignores the keys. "ADetailer model hash" is not the key of a
    # tab's detector: read as one, its value would be a detector that is not
    # installed, and the paste would leave that tab's detector alone.
    face, face_hash = _detector_file(tmp_path, "face_yolov8n.pt", b"face weights")
    hand, hand_hash = _detector_file(tmp_path, "hand_yolov8n.pt", b"hand weights")
    _webui_checkpoints(monkeypatch, {"known.safetensors [1234567890]": "1234567890"})
    detectors = {"face_yolov8n.pt": face, "hand_yolov8n.pt": hand}
    tabs = [
        _tab(ad_use_checkpoint=True, ad_checkpoint="known.safetensors [1234567890]"),
        _tab(ad_model="hand_yolov8n.pt", ad_prompt="detailed hand"),
    ]
    params = _Host(tabs, detectors=detectors).start()
    assert params[_MODEL_HASH + " 2nd"] == hand_hash

    read = _host_read(_host_line({"Steps": 20, "Sampler": "Euler a", **params}))
    old = _host_read(
        _host_line({"Steps": 20, "Sampler": "Euler a", **_old_parameters(tabs)})
    )

    assert {k: v for k, v in read.items() if k not in old} == {
        _MODEL_HASH: face_hash,
        _CHECKPOINT_HASH: "1234567890",
        _MODEL_HASH + " 2nd": hand_hash,
    }
    callback = _paste_callback(model_mapping=detectors)
    capsys.readouterr()
    callback._clear_missing_class_prompts("", read)
    callback._clear_missing_class_prompts("", old)
    assert capsys.readouterr().out == ""  # no "detector ... is not installed"
    assert {k: v for k, v in read.items() if "hash" not in k} == old
    assert read["ADetailer enable"] == "True"
    for n, sfx in enumerate(("", " 2nd")):
        before = ADetailerArgs(ad_model="None", ad_prompt="old prompt").dict()
        pasted = _host_paste(read, before, sfx)
        assert pasted == _host_paste(old, before, sfx)
        assert pasted["ad_model"] == tabs[n]["ad_model"]


def test_the_hash_keys_are_adetailers_own():
    # Not a setting of a tab, and never a key of the WebUI or of another
    # extension: "Model hash", "Lora hashes" and "TI hashes" are the WebUI's,
    # "Hashes" is written by the Civitai extension.
    from adetailer.args import ALL_ARGS

    assert len(ALL_ARGS) == 63
    names = [name for _attr, name in ALL_ARGS]
    for key in p_method.MODEL_HASH_KEYS:
        assert key not in names
        assert key.startswith("ADetailer ")  # dropped with the tab's other keys
        # Sites read a key that ends in "hashes" as a list of name: hash.
        assert key.endswith(" hash")
    ui = (_ROOT / "aaaaaa" / "ui.py").read_text(encoding="utf-8")
    assert "model hash" not in ui
    assert "checkpoint hash" not in ui
    for js in (_ROOT / "javascript").glob("*.js"):
        assert "ADetailer model" not in js.read_text(encoding="utf-8")
