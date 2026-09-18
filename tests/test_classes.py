"""Class-name lookup: caching and sidecar reading (offline, no real models)."""

import codecs
import sys
from types import ModuleType, SimpleNamespace

import pytest

from adetailer.classes import get_model_class_names, resolve_class_ids


@pytest.fixture(autouse=True)
def _fresh_cache():
    get_model_class_names.cache_clear()
    yield
    get_model_class_names.cache_clear()


def _fake_ultralytics(monkeypatch, load):
    """Install a fake `ultralytics` whose YOLO(path) calls `load(path)`."""
    module = ModuleType("ultralytics")
    module.calls = []

    class YOLO:
        def __init__(self, path):
            module.calls.append(path)
            self.names = load(path)

    module.YOLO = YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return module


def _unreadable(path):
    raise RuntimeError("this .pt cannot be read")


def _fake_host(monkeypatch):
    """A WebUI whose safe-unpickle check is on (AUTOMATIC1111's default)."""
    cmd_opts = SimpleNamespace(disable_safe_unpickle=False)
    modules = ModuleType("modules")
    modules.shared = SimpleNamespace(cmd_opts=cmd_opts)
    monkeypatch.setitem(sys.modules, "modules", modules)
    monkeypatch.setitem(sys.modules, "modules.shared", modules.shared)
    return cmd_opts


def test_failed_lookup_is_retried_and_success_is_cached(tmp_path, monkeypatch):
    pt = tmp_path / "multi.pt"
    pt.write_bytes(b"x")
    attempts = []

    def load(path):
        attempts.append(path)
        if len(attempts) == 1:
            raise RuntimeError("transient load failure")
        return {0: "face", 1: "hand"}

    fake = _fake_ultralytics(monkeypatch, load)

    assert get_model_class_names(str(pt)) == []
    assert get_model_class_names(str(pt)) == ["face", "hand"]
    assert resolve_class_ids(str(pt), ["hand"]) == [1]
    # Found names are still cached: no further .pt loads.
    assert len(fake.calls) == 2


def test_detection_reads_names_the_ui_lookup_could_not(tmp_path, monkeypatch):
    # AUTOMATIC1111: the UI looks names up with the safe-unpickle check on,
    # which refuses an Ultralytics .pt without a sidecar; detection bypasses
    # the check. The UI's empty result must not be reused by detection.
    pt = tmp_path / "multi.pt"
    pt.write_bytes(b"x")
    cmd_opts = _fake_host(monkeypatch)

    def load(path):
        if not cmd_opts.disable_safe_unpickle:
            raise AttributeError("the host's safe loader returned None")
        return {0: "face", 1: "hand"}

    fake = _fake_ultralytics(monkeypatch, load)

    # UI (check on): nothing readable, and the refused load is not repeated
    # on the next UI lookup (the host prints an error report each time).
    assert get_model_class_names(str(pt)) == []
    assert get_model_class_names(str(pt)) == []
    assert len(fake.calls) == 1

    # Detection (check bypassed): the real names, so a class filter works.
    cmd_opts.disable_safe_unpickle = True
    assert resolve_class_ids(str(pt), ["hand"]) == [1]
    assert get_model_class_names(str(pt)) == ["face", "hand"]

    # Back in the UI: the names found by detection are reused.
    cmd_opts.disable_safe_unpickle = False
    assert get_model_class_names(str(pt)) == ["face", "hand"]
    assert len(fake.calls) == 2


@pytest.mark.parametrize(
    "payload",
    [
        b'["face","hand"]',
        codecs.BOM_UTF8 + b'["face","hand"]',
        '["face","hand"]'.encode("utf-16"),
    ],
    ids=["utf-8", "utf-8-bom", "utf-16"],
)
def test_sidecar_is_read_in_the_encodings_windows_writes(
    tmp_path, monkeypatch, payload
):
    # Windows PowerShell 5.1 writes UTF-16 with `>` and a UTF-8 byte-order
    # mark with -Encoding UTF8. The sidecar exists for models whose .pt
    # cannot be read, so the .pt fallback finds nothing here.
    pt = tmp_path / "x.pt"
    pt.write_bytes(b"x")
    (tmp_path / "x.names.json").write_bytes(payload)
    _fake_ultralytics(monkeypatch, _unreadable)

    assert get_model_class_names(str(pt)) == ["face", "hand"]
    assert resolve_class_ids(str(pt), ["hand"]) == [1]


def test_undecodable_sidecar_falls_through_to_the_next_one(tmp_path, monkeypatch):
    pt = tmp_path / "x.pt"
    pt.write_bytes(b"x")
    # ANSI (cp1252) text with a non-ASCII character is not valid UTF-8.
    (tmp_path / "x.names.json").write_bytes('["caf\xe9"]'.encode("cp1252"))
    (tmp_path / "x.json").write_bytes(b'["face","hand"]')
    _fake_ultralytics(monkeypatch, _unreadable)

    assert get_model_class_names(str(pt)) == ["face", "hand"]


def test_undecodable_legacy_json_does_not_raise(tmp_path, monkeypatch):
    # An unrelated <model>.json (e.g. model-manager metadata) in cp1252.
    pt = tmp_path / "x.pt"
    pt.write_bytes(b"x")
    (tmp_path / "x.json").write_bytes('{"description": "caf\xe9"}'.encode("cp1252"))
    _fake_ultralytics(monkeypatch, lambda path: {0: "face"})

    assert get_model_class_names(str(pt)) == ["face"]
