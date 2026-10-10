"""Class-name lookup: caching and sidecar reading (offline, no real models)."""

import codecs
import io
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
    """A WebUI that refuses to read a detector file in the panel (as on
    AUTOMATIC1111)."""
    import torch

    cmd_opts = SimpleNamespace(disable_safe_unpickle=False)
    modules = ModuleType("modules")
    modules.shared = SimpleNamespace(cmd_opts=cmd_opts)
    modules.safe = SimpleNamespace(
        load=lambda *args, **kwargs: None, unsafe_torch_load=torch.load
    )
    monkeypatch.setitem(sys.modules, "modules", modules)
    monkeypatch.setitem(sys.modules, "modules.shared", modules.shared)
    monkeypatch.setitem(sys.modules, "modules.safe", modules.safe)
    monkeypatch.setattr(torch, "load", modules.safe.load)
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
    # AUTOMATIC1111: the UI's lookup of a .pt without a sidecar comes back
    # empty, and detection's lookup reads the real names. The UI's empty
    # result must not be reused by detection.
    pt = tmp_path / "multi.pt"
    pt.write_bytes(b"x")
    cmd_opts = _fake_host(monkeypatch)

    def load(path):
        if not cmd_opts.disable_safe_unpickle:
            raise AttributeError("the host would not read this file")
        return {0: "face", 1: "hand"}

    fake = _fake_ultralytics(monkeypatch, load)

    # UI: nothing readable, and the .pt is never read there, so the host
    # prints no error report.
    assert get_model_class_names(str(pt)) == []
    assert get_model_class_names(str(pt)) == []
    assert len(fake.calls) == 0

    # Detection: the real names, so a class filter works.
    cmd_opts.disable_safe_unpickle = True
    assert resolve_class_ids(str(pt), ["hand"]) == [1]
    assert get_model_class_names(str(pt)) == ["face", "hand"]

    # Back in the UI: the names found by detection are reused.
    cmd_opts.disable_safe_unpickle = False
    assert get_model_class_names(str(pt)) == ["face", "hand"]
    assert len(fake.calls) == 1


def test_sidecar_added_after_a_refused_ui_lookup_is_read(tmp_path, monkeypatch):
    # AUTOMATIC1111: the UI lookup found no sidecar and the CLASSES dropdown
    # came up empty; a sidecar written afterwards is read on the next lookup,
    # and the UI never loads the .pt there.
    pt = tmp_path / "multi.pt"
    pt.write_bytes(b"x")
    _fake_host(monkeypatch)
    fake = _fake_ultralytics(monkeypatch, _unreadable)

    assert get_model_class_names(str(pt)) == []
    assert len(fake.calls) == 0

    # An unrelated <model>.json (e.g. model-manager metadata) is not a class
    # list: still nothing, and the .pt is still not loaded.
    (tmp_path / "multi.json").write_bytes(b'{"modelId": 1}')
    assert get_model_class_names(str(pt)) == []
    assert len(fake.calls) == 0

    (tmp_path / "multi.names.json").write_bytes(b'["face","hand"]')
    assert get_model_class_names(str(pt)) == ["face", "hand"]
    assert resolve_class_ids(str(pt), ["hand"]) == [1]
    assert len(fake.calls) == 0


def test_ui_lookup_never_hands_the_pt_to_a_refusing_host(tmp_path, monkeypatch):
    # AUTOMATIC1111: reading a .pt without a sidecar while the panel was built
    # made the host print an error report with a traceback at every start, and
    # no names came of it.
    pt = tmp_path / "multi.pt"
    pt.write_bytes(b"x")
    cmd_opts = _fake_host(monkeypatch)

    def load(path):
        if not cmd_opts.disable_safe_unpickle:
            raise AssertionError("the host would print its error report")
        return {0: "face", 1: "hand"}

    fake = _fake_ultralytics(monkeypatch, load)

    assert get_model_class_names(str(pt)) == []
    assert fake.calls == []

    # Detection reads the real names once; the UI then reuses them.
    cmd_opts.disable_safe_unpickle = True
    assert resolve_class_ids(str(pt), ["hand"]) == [1]
    cmd_opts.disable_safe_unpickle = False
    assert get_model_class_names(str(pt)) == ["face", "hand"]
    assert len(fake.calls) == 1


def test_ui_reads_the_pt_where_the_webui_allows_it(tmp_path, monkeypatch):
    # A WebUI that lets the UI read the detector file: the UI keeps reading
    # class names from a .pt without a sidecar, as before.
    pt = tmp_path / "multi.pt"
    pt.write_bytes(b"x")
    modules = ModuleType("modules")
    modules.shared = SimpleNamespace(cmd_opts=SimpleNamespace())
    monkeypatch.setitem(sys.modules, "modules", modules)
    monkeypatch.setitem(sys.modules, "modules.shared", modules.shared)
    fake = _fake_ultralytics(monkeypatch, lambda path: {0: "face", 1: "hand"})

    assert get_model_class_names(str(pt)) == ["face", "hand"]
    assert len(fake.calls) == 1


@pytest.mark.parametrize("host", ["host-a", "host-b"])
def test_ui_reads_the_pt_on_other_webuis(tmp_path, monkeypatch, host):
    # Another WebUI that lets the UI read the detector file: the UI must keep
    # reading class names from a .pt without a sidecar there, or the CLASSES
    # dropdown stays empty until a generation has run.
    import torch

    pt = tmp_path / "multi.pt"
    pt.write_bytes(b"x")
    cmd_opts = _fake_host(monkeypatch)
    safe = sys.modules["modules.safe"]
    monkeypatch.setattr(torch, "load", safe.unsafe_torch_load)
    if host == "host-b":
        monkeypatch.delitem(sys.modules, "modules.safe")
        monkeypatch.delattr(sys.modules["modules"], "safe")
    fake = _fake_ultralytics(monkeypatch, lambda path: {0: "face", 1: "hand"})

    assert cmd_opts.disable_safe_unpickle is False
    assert get_model_class_names(str(pt)) == ["face", "hand"]
    assert len(fake.calls) == 1

    if host == "host-a":
        # The same lookup on a WebUI that refuses the read (AUTOMATIC1111):
        # only a sidecar is read again, never the .pt.
        get_model_class_names.cache_clear()
        monkeypatch.setattr(torch, "load", safe.load)
        assert get_model_class_names(str(pt)) == []
        assert len(fake.calls) == 1


def test_the_readme_says_only_automatic1111_waits_for_the_first_detection():
    # The README sentence saying that the CLASSES dropdown lists a .pt's
    # class names only after the first detection is about AUTOMATIC1111: it
    # must not follow the classic Forge / Forge Neo sentence, where it read
    # as if those hosts, which list them straight away, waited too.
    import re
    from pathlib import Path

    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    paragraph = next(
        line for line in readme.splitlines()
        if line.startswith("- On AUTOMATIC1111, a detector without this file")
    )
    assert "Classic Forge" in paragraph
    assert paragraph.count("after its first detection") == 1
    before = paragraph[: paragraph.index("after its first detection")]
    hosts = re.findall(r"AUTOMATIC1111|classic Forge|Forge Neo|reForge", before, re.I)
    assert hosts
    assert hosts[-1] == "AUTOMATIC1111", hosts


def test_sidecar_added_after_names_were_found_waits_for_a_restart(
    tmp_path, monkeypatch
):
    # The README promises a sidecar added while the WebUI runs only while no
    # names were found: once a generation has read the .pt, the names found
    # are cached for the session, so adding a sidecar needs a restart.
    pt = tmp_path / "multi.pt"
    pt.write_bytes(b"x")
    cmd_opts = _fake_host(monkeypatch)

    def load(path):
        if not cmd_opts.disable_safe_unpickle:
            raise AttributeError("the host would not read this file")
        return {0: "class0", 1: "class1"}

    _fake_ultralytics(monkeypatch, load)

    assert get_model_class_names(str(pt)) == []  # UI lookup
    cmd_opts.disable_safe_unpickle = True  # detection's lookup
    assert get_model_class_names(str(pt)) == ["class0", "class1"]
    cmd_opts.disable_safe_unpickle = False

    (tmp_path / "multi.names.json").write_bytes(b'["face","hand"]')
    assert get_model_class_names(str(pt)) == ["class0", "class1"]
    get_model_class_names.cache_clear()  # a restart
    assert get_model_class_names(str(pt)) == ["face", "hand"]


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


def test_class_names_match_regardless_of_case(tmp_path, monkeypatch, capsys):
    # An API call or an old preset may send "Face" for the model's "face"; it
    # used to match nothing, so the class filter was dropped silently.
    pt = tmp_path / "x.pt"
    pt.write_bytes(b"x")
    (tmp_path / "x.names.json").write_bytes(b'["face","hand","Eye","eye"]')
    _fake_ultralytics(monkeypatch, _unreadable)

    assert resolve_class_ids(str(pt), ["Face", "HAND"]) == [0, 1]
    # An exact match wins over a case-insensitive one.
    assert resolve_class_ids(str(pt), ["eye", "Eye"]) == [3, 2]
    assert capsys.readouterr().out == ""

    # Unknown, out-of-range and ambiguous entries are dropped and named.
    assert resolve_class_ids(str(pt), ["hands", "9", "EYE", "face"]) == [0]
    assert "hands, 9, EYE" in capsys.readouterr().out


@pytest.mark.parametrize("names_known", [True, False])
def test_a_digit_that_int_rejects_is_an_unknown_class(
    tmp_path, monkeypatch, capsys, names_known
):
    # "²" and "①" are digits to str.isdigit() but not to int(), and a number
    # that cannot be converted is refused too: such a class from the API, a
    # preset or pasted parameters is dropped like any other unknown class.
    pt = tmp_path / "x.pt"
    if names_known:
        pt.write_bytes(b"x")
        (tmp_path / "x.names.json").write_bytes(b'["face","hand"]')
    _fake_ultralytics(monkeypatch, _unreadable)
    bad = ["²", "①", "1" * 5000]

    assert resolve_class_ids(str(pt), ["face", *bad]) == ([0] if names_known else [])
    out = capsys.readouterr().out
    if names_known:
        assert "ignored: \\xb2, \\u2460, 111" in out
    else:
        assert out == ""
    # Decimal digits of other scripts are still ids, as before.
    assert resolve_class_ids(str(pt), ["１"]) == [1]


def test_class_not_found_line_prints_on_a_legacy_code_page(tmp_path, monkeypatch):
    # Console output redirected to a file or pipe in a legacy code page
    # (cp1252 here): a non-Latin class name from a preset, pasted parameters
    # or the API, or a non-ASCII model file name, could not be printed, and
    # the error stopped ADetailer for the image.
    out = io.TextIOWrapper(
        io.BytesIO(), encoding="cp1252", errors="strict", write_through=True
    )
    monkeypatch.setattr(sys, "stdout", out)
    _fake_ultralytics(monkeypatch, _unreadable)
    pt = tmp_path / "x.pt"
    pt.write_bytes(b"x")
    (tmp_path / "x.names.json").write_bytes(b'["face","hand"]')
    other = tmp_path / "x\u9854.pt"
    other.write_bytes(b"x")
    (tmp_path / "x\u9854.names.json").write_bytes(b'["face","hand"]')

    assert resolve_class_ids(str(pt), ["face", "\u9854"]) == [0]
    assert resolve_class_ids(str(other), ["hands"]) == []

    # Other characters are escaped; ASCII names print as they are.
    printed = out.buffer.getvalue()
    assert b"class not found in x.pt, ignored: \\u9854" in printed
    assert b"class not found in x\\u9854.pt, ignored: hands" in printed
