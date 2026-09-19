from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from adetailer import json_file, presets


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


def test_dropdown_placeholder_name_is_reserved(preset_file):
    # "(none)" is the dropdowns' "no preset selected" entry: a preset with that
    # name could be saved but never loaded, renamed or deleted in the UI.
    assert not presets.is_valid_name("(none)")
    assert not presets.is_valid_name(" (none) ")
    assert not presets.save_preset("(none)", {"ad_prompt": "face"})
    assert presets.save_preset("faces", {"ad_prompt": "face"})
    assert presets.rename_preset("faces", "(none)") == (False, "invalid name '(none)'")
    payload = '{"(none)": {"ad_prompt": "face"}}'
    assert presets.import_presets_json(payload, overwrite=True) == (0, 0, ["(none)"])
    assert presets.get_preset_names() == ["faces"]
    # Parentheses stay allowed in ordinary names.
    assert presets.is_valid_name("face (close-up)")


def test_reserved_name_matches_the_ui_placeholder():
    import ast
    from pathlib import Path

    ui = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"
    tree = ast.parse(ui.read_text(encoding="utf-8"))
    placeholder = next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", "") == "PRESET_NONE" for t in node.targets)
    )
    assert placeholder == presets._RESERVED_NAME


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


def test_deeply_nested_library_is_damaged_not_fatal(preset_file):
    preset_file.write_text("[" * 100_000 + "]" * 100_000, encoding="utf-8")
    assert presets.load_presets() == {}
    assert presets.save_preset("new", {"ad_prompt": "new"})
    assert len(list(preset_file.parent.glob("user_presets.unreadable-*.json"))) == 1


def test_backup_made_in_the_same_second_is_never_overwritten(preset_file, monkeypatch):
    monkeypatch.setattr(json_file.time, "strftime", lambda _fmt: "20260916-120000")
    earlier = preset_file.parent / "user_presets.unreadable-20260916-120000.json"
    earlier.write_bytes(b"earlier backup")
    preset_file.write_bytes(DAMAGED_LIBRARIES["trailing-comma"])

    assert presets.save_preset("new", {"ad_prompt": "new"})
    assert earlier.read_bytes() == b"earlier backup"
    second = preset_file.parent / "user_presets.unreadable-20260916-120000-2.json"
    assert second.read_bytes() == DAMAGED_LIBRARIES["trailing-comma"]


def test_library_whose_status_cannot_be_checked_does_not_break_the_ui(
    preset_file, monkeypatch
):
    def denied(*_args, **_kwargs):
        raise PermissionError("access denied")

    monkeypatch.setattr(type(preset_file), "is_file", denied)
    assert presets.load_presets() == {}
    assert not presets.save_preset("new", {"ad_prompt": "new"})


def test_library_is_on_disk_before_it_replaces_the_old_one(preset_file, monkeypatch):
    # Closing a file does not put it on disk. After a power cut the renamed
    # library could hold only zero bytes, and the next save dropped every
    # preset it had.
    events = []
    real_fsync, real_replace = presets.os.fsync, presets.os.replace

    def fsync(fd):
        events.append(("fsync", presets.os.fstat(fd).st_size))
        real_fsync(fd)

    def replace(src, dst):
        events.append(("replace", str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(presets.os, "fsync", fsync)
    monkeypatch.setattr(presets.os, "replace", replace)
    assert presets.save_preset("face", {"ad_prompt": "x"})

    tmp = preset_file.with_suffix(".json.tmp")
    assert events == [
        ("fsync", preset_file.stat().st_size),
        ("replace", str(tmp), str(preset_file)),
    ]
    assert presets.get_preset("face") == {"ad_prompt": "x"}


def test_a_file_system_that_cannot_sync_still_saves(preset_file, monkeypatch):
    def fail_fsync(_fd):
        raise OSError

    monkeypatch.setattr(presets.os, "fsync", fail_fsync)
    assert presets.save_preset("face", {"ad_prompt": "x"})
    assert presets.get_preset("face") == {"ad_prompt": "x"}
    assert not preset_file.with_suffix(".json.tmp").exists()
