from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from adetailer import presets


@pytest.fixture
def preset_file(tmp_path, monkeypatch):
    path = tmp_path / "user_presets.json"
    monkeypatch.setattr(presets, "_PRESETS_FILE", path)
    return path


def test_simultaneous_saves_preserve_both_presets(preset_file, monkeypatch):
    first_read = Event()
    second_read = Event()
    original_load = presets._load_raw

    def overlapping_load():
        current = original_load()
        if not first_read.is_set():
            first_read.set()
            second_read.wait(timeout=0.2)
        else:
            second_read.set()
        return current

    monkeypatch.setattr(presets, "_load_raw", overlapping_load)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(presets.save_preset, "faces", {"ad_prompt": "face"})
        assert first_read.wait(timeout=2)
        second = pool.submit(presets.save_preset, "hands", {"ad_prompt": "hand"})
        assert first.result(timeout=2)
        assert second.result(timeout=2)

    assert presets.load_presets() == {
        "faces": {"ad_prompt": "face"},
        "hands": {"ad_prompt": "hand"},
    }


@pytest.mark.parametrize("operation", ["save", "delete", "rename", "import"])
def test_failed_writes_are_not_reported_as_success(
    preset_file, monkeypatch, operation
):
    assert presets.save_preset("original", {"ad_prompt": "keep me"})
    original_bytes = preset_file.read_bytes()

    def fail_replace(*args):
        raise OSError

    monkeypatch.setattr(presets.os, "replace", fail_replace)
    if operation == "save":
        assert not presets.save_preset("new", {"ad_prompt": "new"})
    elif operation == "delete":
        assert not presets.delete_preset("original")
    elif operation == "rename":
        ok, message = presets.rename_preset("original", "renamed")
        assert not ok
        assert "write" in message
    else:
        added, replaced, skipped = presets.import_presets_json(
            '{"new":{"ad_prompt":"new"},"original":{"ad_prompt":"replacement"}}',
            overwrite=True,
        )
        assert (added, replaced) == (0, 0)
        assert set(skipped) == {"new", "original"}

    assert preset_file.read_bytes() == original_bytes


def test_import_keeps_conflicts_unless_overwrite_requested(preset_file):
    assert presets.save_preset("original", {"ad_prompt": "keep me"})
    payload = '{"original":{"ad_prompt":"replace me","is_api":false}}'
    assert presets.import_presets_json(payload) == (0, 0, ["original"])
    assert presets.get_preset("original") == {"ad_prompt": "keep me"}
    assert presets.import_presets_json(payload, overwrite=True) == (0, 1, [])
    assert presets.get_preset("original") == {"ad_prompt": "replace me"}


def test_unreadable_encoding_does_not_break_ui_startup(preset_file):
    preset_file.write_bytes(b"\xff\xfe{\x00}\x00")
    assert presets.load_presets() == {}
