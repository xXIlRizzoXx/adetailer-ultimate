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
    original_load = presets._read_raw

    def overlapping_load():
        current = original_load()
        if not first_read.is_set():
            first_read.set()
            second_read.wait(timeout=0.2)
        else:
            second_read.set()
        return current

    monkeypatch.setattr(presets, "_read_raw", overlapping_load)
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


def test_library_with_byte_order_mark_is_read(preset_file):
    preset_file.write_bytes(b'\xef\xbb\xbf{"faces": {"ad_prompt": "face"}}')
    assert presets.load_presets() == {"faces": {"ad_prompt": "face"}}
    assert presets.save_preset("hands", {"ad_prompt": "hand"})
    assert set(presets.load_presets()) == {"faces", "hands"}


DAMAGED_LIBRARIES = {
    "cp1252": '{"caf\xe9": {"ad_prompt": "face"}}'.encode("cp1252"),
    "utf16": '{"faces": {"ad_prompt": "face"}}'.encode("utf-16"),
    "trailing-comma": b'{"faces": {"ad_prompt": "face"},}',
    "not-an-object": b'[{"ad_prompt": "face"}]',
}


@pytest.mark.parametrize("operation", ["save", "import"])
@pytest.mark.parametrize("damage", DAMAGED_LIBRARIES)
def test_unreadable_library_is_kept_aside_before_a_save(
    preset_file, operation, damage
):
    original = DAMAGED_LIBRARIES[damage]
    preset_file.write_bytes(original)
    presets.take_recovery_note()

    if operation == "save":
        assert presets.save_preset("new", {"ad_prompt": "new"})
    else:
        assert presets.import_presets_json('{"new":{"ad_prompt":"new"}}') == (1, 0, [])

    assert presets.load_presets() == {"new": {"ad_prompt": "new"}}
    backups = list(preset_file.parent.glob("user_presets.unreadable-*.json"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    note = presets.take_recovery_note()
    assert backups[0].name in note
    assert presets.take_recovery_note() == ""


@pytest.mark.parametrize("operation", ["delete", "rename"])
def test_unreadable_library_is_left_alone_when_nothing_is_written(
    preset_file, operation
):
    original = DAMAGED_LIBRARIES["utf16"]
    preset_file.write_bytes(original)
    if operation == "delete":
        assert not presets.delete_preset("faces")
    else:
        assert not presets.rename_preset("faces", "renamed")[0]
    assert preset_file.read_bytes() == original
    assert not list(preset_file.parent.glob("user_presets.unreadable-*"))


@pytest.mark.parametrize("operation", ["save", "import"])
def test_library_that_cannot_be_read_right_now_is_not_replaced(
    preset_file, monkeypatch, operation
):
    assert presets.save_preset("original", {"ad_prompt": "keep me"})
    original = preset_file.read_bytes()

    def locked(*_args, **_kwargs):
        raise PermissionError("file in use")

    with monkeypatch.context() as patched:
        patched.setattr(type(preset_file), "read_text", locked)
        if operation == "save":
            assert not presets.save_preset("new", {"ad_prompt": "new"})
        else:
            added, replaced, skipped = presets.import_presets_json(
                '{"new":{"ad_prompt":"new"}}'
            )
            assert (added, replaced, skipped) == (0, 0, ["new"])
    assert preset_file.read_bytes() == original
    assert not list(preset_file.parent.glob("user_presets.unreadable-*"))
