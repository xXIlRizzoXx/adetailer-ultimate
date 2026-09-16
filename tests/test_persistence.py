from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from adetailer import persistence


@pytest.fixture
def state_file(tmp_path, monkeypatch):
    path = tmp_path / "user_state.json"
    monkeypatch.setattr(persistence, "_STATE_FILE", path)
    monkeypatch.setattr(persistence, "_enabled", lambda: True)
    return path


def test_simultaneous_tab_saves_preserve_both_tabs(state_file, monkeypatch):
    first_read = Event()
    second_read = Event()
    original_load = persistence._read_raw

    def overlapping_load():
        current = original_load()
        if not first_read.is_set():
            first_read.set()
            # Without the transaction lock both callbacks read the same file.
            # With it, the second callback waits outside _load_raw instead.
            second_read.wait(timeout=0.2)
        else:
            second_read.set()
        return current

    monkeypatch.setattr(persistence, "_read_raw", overlapping_load)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            persistence.save_tab_state, "txt2img", 0, {"ad_prompt": "face"}
        )
        assert first_read.wait(timeout=2)
        second = pool.submit(
            persistence.save_tab_state, "txt2img", 1, {"ad_prompt": "hands"}
        )
        first.result(timeout=2)
        second.result(timeout=2)

    assert persistence.load_state() == {
        "0": {"ad_prompt": "face"},
        "1": {"ad_prompt": "hands"},
    }


def test_scoped_state_and_legacy_fallback_survive_saving(state_file):
    state_file.write_text('{"0": {"ad_prompt": "legacy"}}', encoding="utf-8")
    persistence.save_tab_state(
        "img2img", 0, {"ad_prompt": "new", "is_api": ()}
    )
    assert persistence.load_state("txt2img") == {"0": {"ad_prompt": "legacy"}}
    assert persistence.load_state("img2img") == {"0": {"ad_prompt": "new"}}


def test_unreadable_encoding_does_not_break_ui_startup(state_file):
    state_file.write_bytes(b"\xff\xfe{\x00}\x00")
    assert persistence.load_state() == {}


def test_failed_replace_preserves_previous_state(state_file, monkeypatch):
    persistence.save_tab_state("txt2img", 0, {"ad_prompt": "original"})

    def fail_replace(*args):
        raise OSError

    monkeypatch.setattr(persistence.os, "replace", fail_replace)
    persistence.save_tab_state("txt2img", 0, {"ad_prompt": "new"})
    assert persistence.load_state() == {"0": {"ad_prompt": "original"}}


@pytest.mark.parametrize(
    "original",
    [
        '{"txt2img:1": {"ad_prompt": "caf\xe9"}}'.encode("cp1252"),
        '{"txt2img:1": {"ad_prompt": "hands"}}'.encode("utf-16"),
        b'{"txt2img:1": {"ad_prompt": "hands"},}',
    ],
    ids=["cp1252", "utf16", "trailing-comma"],
)
def test_unreadable_state_is_kept_aside_before_a_save(state_file, original):
    state_file.write_bytes(original)
    persistence.save_tab_state("txt2img", 0, {"ad_prompt": "face"})
    assert persistence.load_state() == {"0": {"ad_prompt": "face"}}
    backups = list(state_file.parent.glob("user_state.unreadable-*.json"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original


def test_state_with_byte_order_mark_is_read(state_file):
    state_file.write_bytes(b'\xef\xbb\xbf{"txt2img:1": {"ad_prompt": "hands"}}')
    persistence.save_tab_state("txt2img", 0, {"ad_prompt": "face"})
    assert persistence.load_state() == {
        "0": {"ad_prompt": "face"},
        "1": {"ad_prompt": "hands"},
    }


def test_state_that_cannot_be_read_right_now_is_not_replaced(
    state_file, monkeypatch
):
    persistence.save_tab_state("txt2img", 1, {"ad_prompt": "hands"})
    original = state_file.read_bytes()

    def locked(*_args, **_kwargs):
        raise PermissionError("file in use")

    with monkeypatch.context() as patched:
        patched.setattr(type(state_file), "read_text", locked)
        persistence.save_tab_state("txt2img", 0, {"ad_prompt": "face"})
    assert state_file.read_bytes() == original
    assert not list(state_file.parent.glob("user_state.unreadable-*"))
