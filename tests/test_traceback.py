"""The error report built by rich_traceback; no WebUI is needed."""

import io

import pytest
from rich.console import Console

from aaaaaa import traceback as ad_traceback


def test_error_report_keeps_the_error_when_a_prompt_has_class_blocks(monkeypatch):
    # rich parsed every table cell as markup: the documented "[/CLASS]" is a
    # closing tag with no opening one, so building the report raised a
    # MarkupError that replaced the real error.
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr(ad_traceback, "sys_info", lambda: {"Platform": "test"})
    monkeypatch.setattr(ad_traceback, "processing", lambda *args: {})

    @ad_traceback.rich_traceback
    def process(p, *args):
        msg = "CUDA out of memory"
        raise RuntimeError(msg)

    tab = {
        "ad_model": "face_yolov8n.pt",
        "ad_prompt": "[CLASS=hand] [SKIP] [/CLASS]",
        "ad_negative_prompt": "[CLASS=hand]extra fingers[/CLASS]",
    }
    with pytest.raises(RuntimeError, match="CUDA out of memory") as exc:
        process(object(), True, False, tab)
    assert "[CLASS=hand] [SKIP] [/CLASS]" in str(exc.value)
    assert "[CLASS=hand]extra fingers[/CLASS]" in str(exc.value)


def test_error_report_shows_prompts_as_typed():
    # Also no emoji codes: ":dog:" in "[cat:dog:0.5]" became an emoji.
    data = {"ad_prompt": "[class=hand]five fingers[/class]", "prompt": "[cat:dog:0.5]"}
    out = io.StringIO()
    Console(file=out, width=200, color_system=None).print(
        ad_traceback.get_table("ADetailer", data)
    )
    assert "[class=hand]five fingers[/class]" in out.getvalue()
    assert "[cat:dog:0.5]" in out.getvalue()
