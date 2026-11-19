"""YOLOE detectors with text prompts are handled like YOLO-World.

A fake Ultralytics: no real model is loaded and nothing is downloaded."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from adetailer.classes import (
    get_model_class_names,
    is_open_vocab_model,
    is_yoloe_model,
)
from adetailer.ultralytics import apply_classes, ultralytics_predict


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("yoloe-11s-seg.pt", True),
        ("yoloe-v8l-seg.pt", True),
        ("yoloe-26n-seg.pt", True),
        ("YOLOE-11S-SEG.PT", True),
        ("Yoloe-v8s-seg.pt", True),
        ("my_yoloe_face.pt", True),
        ("models/detectors/yoloe-11s-seg.pt", True),
        # A copy in another folder, as the menu names it.
        ("yoloe-11s-seg.pt (pack/v2)", True),
        ("yoloe-11m-seg.pt (feet 2)", True),
        # Prompt-free: the classes saved in the model, like any detector.
        ("yoloe-11s-seg-pf.pt", False),
        ("yoloe-v8l-seg-pf.pt", False),
        ("YOLOE-11S-SEG-PF.pt", False),
        ("yoloe-11s-seg-pf.pt (pack)", False),
        # Only the file name counts, never a folder.
        ("yoloe/face_yolov8n.pt", False),
        ("face_yolov8n.pt (yoloe)", False),
        ("hand.pt (yoloe-models)", False),
        ("face_yolov8n.pt", False),
        ("yoloeyes_v8.pt", False),
        ("yolov8s-worldv2.pt", False),
        ("mediapipe_face_mesh", False),
        ("None", False),
    ],
)
def test_yoloe_is_recognised_from_the_file_name(name, expected):
    assert is_yoloe_model(name) is expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("yoloe-11s-seg.pt", True),
        ("yolov8s-worldv2.pt", True),
        ("yoloe-11s-seg-pf.pt", False),
        ("face_yolov8n.pt", False),
    ],
)
def test_open_vocabulary_detectors(name, expected):
    assert is_open_vocab_model(name) is expected


def _with_names(path: Path) -> Path:
    path.write_bytes(b"weights")
    path.with_name(path.stem + ".names.json").write_text(
        '["person", "car"]', encoding="utf-8"
    )
    return path


def test_yoloe_has_no_class_list_but_a_prompt_free_one_does(tmp_path):
    # The classes dropdown, the class guard and the sequential passes read
    # this list: a YOLOE with text prompts takes typed words instead.
    yoloe = _with_names(tmp_path / "yoloe-11s-seg.pt")
    pf = _with_names(tmp_path / "yoloe-11s-seg-pf.pt")
    get_model_class_names.cache_clear()
    try:
        assert get_model_class_names(str(yoloe)) == []
        assert get_model_class_names(str(pf)) == ["person", "car"]
    finally:
        get_model_class_names.cache_clear()


class _Boxes:
    def __init__(self, cls):
        n = len(cls)
        self.xyxy = torch.tensor([[1.0, 1.0, 3.0, 3.0]] * n)
        self.conf = torch.tensor([0.9] * n)
        self.cls = torch.tensor(cls)

    def __len__(self):
        return len(self.cls)


class _Result:
    def __init__(self, names, cls):
        self.names = names
        self.boxes = _Boxes(cls)
        self.masks = None

    def plot(self, **_kwargs):
        return np.zeros((4, 4, 3), np.uint8)


def _fake_yoloe(monkeypatch, *, text_pe=True, pe_error=None, prompt_free=False):
    """A fake Ultralytics whose YOLO is a YOLOE: set_classes(names, embeddings)
    and get_text_pe(names), as in Ultralytics 8.4. `text_pe=False` is an
    Ultralytics without text prompts for YOLOE."""
    calls: list = []
    module = ModuleType("ultralytics")

    class YOLO:
        def __init__(self, path):
            self.names = {0: "person", 1: "bus"}
            head = SimpleNamespace(lrpc=[]) if prompt_free else SimpleNamespace()
            self.model = SimpleNamespace(model=[head])

        def set_classes(self, classes, embeddings=None):
            calls.append(("set_classes", list(classes), embeddings))
            self.names = dict(enumerate(classes))

        def __call__(self, image, **kwargs):
            calls.append(("predict", kwargs))
            return [_Result(self.names, [0.0])]

    if text_pe:

        def get_text_pe(self, names):
            calls.append(("get_text_pe", list(names)))
            if pe_error is not None:
                raise pe_error
            return f"pe:{','.join(names)}"

        YOLO.get_text_pe = get_text_pe

    module.YOLO = YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return calls


def _predict(path, **kwargs):
    return ultralytics_predict(str(path), Image.new("RGB", (4, 4)), **kwargs)


def test_typed_classes_reach_yoloe_as_text_prompts(tmp_path, monkeypatch):
    calls = _fake_yoloe(monkeypatch)

    result = _predict(tmp_path / "yoloe-11s-seg.pt", classes="face, hand")

    assert calls[0] == ("get_text_pe", ["face", "hand"])
    assert calls[1] == ("set_classes", ["face", "hand"], "pe:face,hand")
    # No class filter by number and no NOT mode: the typed words are the
    # whole vocabulary, as with YOLO-World.
    assert "classes" not in calls[2][1]
    assert result.class_names == ["face"]


def test_yoloe_ignores_an_excluded_list_like_yolo_world(tmp_path, monkeypatch):
    calls = _fake_yoloe(monkeypatch)

    result = _predict(
        tmp_path / "YOLOE-11S-SEG.pt", classes="face", exclude_classes="face"
    )

    assert result.bboxes
    assert calls[1][0] == "set_classes"


def test_yoloe_without_typed_classes_downloads_nothing(tmp_path, monkeypatch):
    # The text model is needed only for typed classes: without them the
    # model detects its own vocabulary and Ultralytics fetches nothing.
    calls = _fake_yoloe(monkeypatch)

    result = _predict(tmp_path / "yoloe-11s-seg.pt")

    assert [c[0] for c in calls] == ["predict"]
    assert result.class_names == ["person"]


@pytest.mark.parametrize(
    ("fake", "said"),
    [
        ({"text_pe": False}, "no text prompts for YOLOE"),
        ({"pe_error": ImportError("No module named 'clip'")}, "ImportError"),
        ({"pe_error": ConnectionError("offline")}, "(ConnectionError: offline)."),
        ({"prompt_free": True}, "prompt-free"),
    ],
    ids=["old-ultralytics", "no-clip", "offline", "renamed-prompt-free"],
)
def test_yoloe_that_cannot_take_the_classes_skips_the_tab(
    tmp_path, monkeypatch, capfd, fake, said
):
    calls = _fake_yoloe(monkeypatch, **fake)

    result = _predict(tmp_path / "yoloe-11s-seg.pt", classes="face")

    # Nothing detected (no inpainting with the model's own classes), no
    # exception for the generation, and the reason in the result's note: the
    # caller says it once per tab and generation, so nothing is printed here.
    assert result.bboxes == []
    assert result.preview is None
    assert not any(c[0] == "predict" for c in calls)
    assert said in result.note
    assert result.note.startswith("the YOLOE detector 'yoloe-11s-seg.pt' c")
    assert "the typed classes" in result.note
    said_here = capfd.readouterr()
    assert said_here.out == ""
    assert said_here.err == ""
    if fake.get("prompt_free"):
        # Not even the text model is asked for.
        assert not any(c[0] == "get_text_pe" for c in calls)


def _no_module(name):
    """What `import name` raises when the package is not installed."""
    return ModuleNotFoundError(f"No module named {name!r}", name=name)


def _wrapped(error):
    try:
        try:
            raise error
        except ImportError as inner:
            raise RuntimeError("text model not built") from inner
    except RuntimeError as outer:
        return outer


_CLIP_COMMAND = "python -m pip install git+https://github.com/ultralytics/CLIP.git"


@pytest.mark.parametrize(
    ("error", "package", "command", "reason"),
    [
        (
            _no_module("clip"), "clip", _CLIP_COMMAND,
            "(ModuleNotFoundError: No module named 'clip')",
        ),
        (
            ImportError("No module named 'clip'"), "clip", _CLIP_COMMAND,
            "(ImportError: No module named 'clip')",
        ),
        (
            _wrapped(_no_module("clip")), "clip", _CLIP_COMMAND,
            "(RuntimeError: text model not built)",
        ),
        (
            _no_module("mobileclip"), "mobileclip",
            "python -m pip install git+https://github.com/ultralytics/mobileclip.git",
            "(ModuleNotFoundError: No module named 'mobileclip')",
        ),
    ],
    ids=["clip", "clip-without-a-name", "clip-wrapped", "mobileclip"],
)
@pytest.mark.parametrize(
    ("model", "kind"),
    [("yolov8s-worldv2.pt", "YOLO-World"), ("yoloe-11s-seg.pt", "YOLOE")],
)
def test_a_missing_clip_package_is_named_with_the_command_that_installs_it(
    tmp_path, monkeypatch, capfd, model, kind, error, package, command, reason
):
    # Typed classes need Ultralytics' text model, whose "clip" package
    # Ultralytics installs by itself; without a network (or with its
    # auto-install off) the import fails. With YOLO-World that error stopped
    # ADetailer for the image with a traceback; with YOLOE the console only
    # had the error's name.
    calls = _fake_yoloe(monkeypatch, pe_error=error)
    yolo = sys.modules["ultralytics"].YOLO
    yoloe_set_classes = yolo.set_classes

    def set_classes(self, classes, embeddings=None):
        if "world" in model:
            raise error
        yoloe_set_classes(self, classes, embeddings)

    yolo.set_classes = set_classes

    result = _predict(tmp_path / model, classes="face, hand")

    assert result.bboxes == []
    assert result.preview is None
    # Not detected with the model's own classes instead of the typed ones.
    assert not any(c[0] == "predict" for c in calls)
    assert result.note == (
        f"the {kind} detector {model!r} could not take the typed classes: the "
        f"Python package {package!r} is missing and Ultralytics could not "
        f"install it by itself {reason}. Run {command} with the WebUI's Python "
        "(its virtual environment)."
    )
    # The command is copied with what touches it: a space on both sides, and
    # the note ends its sentence, so a status can go on after it.
    assert f" {command} " in result.note
    assert result.note.endswith(".")
    said_here = capfd.readouterr()
    assert said_here.out == ""
    assert said_here.err == ""


def test_a_long_reason_of_the_typed_classes_note_is_cut_to_160_characters(
    tmp_path, monkeypatch
):
    # An error text can be very long (a path, a list of files): the note
    # keeps the first 160 characters of its first line, with and without the
    # install command.
    _fake_yoloe(monkeypatch)
    long_text = "".join(chr(ord("a") + k % 26) for k in range(400))
    missing = ModuleNotFoundError(long_text + "\nsecond line", name="clip")
    other = ConnectionError(long_text + "\nsecond line")
    for error, ending in (
        (missing, f"). Run {_CLIP_COMMAND} with the WebUI's Python (its virtual environment)."),
        (other, ")."),
    ):

        def set_classes(self, classes, embeddings=None, error=error):
            raise error

        sys.modules["ultralytics"].YOLO.set_classes = set_classes
        note = _predict(tmp_path / "yolov8s-worldv2.pt", classes="face").note
        assert f"({type(error).__name__}: {long_text[:160]}){ending[1:]}" in note
        assert note.endswith(ending)
        assert long_text[:161] not in note
        assert "second line" not in note


def _w(text: str) -> str:
    """`text` with a backslash for each "|": a Windows path that reads well."""
    return text.replace("|", "\\")


_WINDOWS_FOLDER = "C:\\Users\\some one\\.cache\\clip\\"
# What _known_folders gives in the tests below, whatever the computer they
# run on: users, WebUIs and detectors' folders with spaces in their names.
_KNOWN = [
    _w("C:|Users|some one"),
    "/home/some one",
    "/home/some one/My notes.txt old",
    "/home/someone/My own models",
    _w("C:|Users|someone|Cloud Drive - Some Company Name"),
    _w("C:|Users|someone|My own models - new"),
    _w("C:|Users|someone|SD v1.5 models, new"),
    _w("C:|Users|Some Body Else"),
    _w("C:|Users|Some Body Else|SD v1.5 new"),
    "/Users/Some Body Else",
    _w("C:|Users|S. B. Else"),
    _w("C:|Users|Some R. Body"),
    _w("C:|Users|Some Body, Jr"),
    "/Users/S. Body",
    _w("D:|AI|Some WebUI Name - New"),
    _w("D:|My Own Big Disk"),
    _w("D:|My Disk v1.5 new|Private Folder"),
]


@pytest.fixture
def known_folders(monkeypatch):
    """_typed_classes_note with the folders of _KNOWN instead of the ones of
    this computer; the detector paths it asked the folders for."""
    import adetailer.ultralytics as module

    asked = []

    def fake(model_path):
        asked.append(model_path)
        return list(_KNOWN)

    monkeypatch.setattr(module, "_known_folders", fake)
    return asked


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        # A file name in quotes, as Python writes it in an OSError.
        (
            "[Errno 2] No such file or directory: '/home/someone/.cache/clip/weights.pt'",
            "[Errno 2] No such file or directory: 'weights.pt'",
        ),
        (
            "[Errno 13] Permission denied: "
            "'C:\\\\Users\\\\some one\\\\.cache\\\\clip\\\\weights.pt'",
            "[Errno 13] Permission denied: 'weights.pt'",
        ),
        # Without quotes, with a space in a folder's name.
        (
            _WINDOWS_FOLDER + "weights.pt exists and is not a regular file",
            "weights.pt exists and is not a regular file",
        ),
        (
            "open file failed, file path: /home/some one/models/text.ts",
            "open file failed, file path: text.ts",
        ),
        (
            "cannot copy ~/.cache/clip/a.pt to \\\\server\\someone\\b.pt (C:/Users/someone/c.pt)",
            "cannot copy a.pt to b.pt (c.pt)",
        ),
        # Longer than the cut: half a path would still be a path.
        (
            "no text model in " + _WINDOWS_FOLDER * 20 + "weights.pt",
            "no text model in weights.pt",
        ),
        # Not paths: a web address, a slash between two words.
        (
            "Download failure for https://example.com/assets/v1/text.ts. Retry limit reached.",
            "Download failure for https://example.com/assets/v1/text.ts. Retry limit reached.",
        ),
        ("read/write error and/or 3/4 of 1 / 2", "read/write error and/or 3/4 of 1 / 2"),
        # A word with a backslash is a piece of a path, wherever it starts.
        ("weights\\text.ts is not a model", "text.ts is not a model"),
    ],
    ids=[
        "quoted", "quoted-windows", "bare-windows", "bare-with-a-space", "three-paths",
        "longer-than-the-cut", "web-address", "slash-between-words", "relative",
    ],
)
@pytest.mark.parametrize("missing_package", [False, True])
def test_the_typed_classes_note_keeps_only_the_file_name_of_a_path(
    text, kept, missing_package, known_folders
):
    # The first line of the error went into the note as it was: a path in it
    # (a weights file under the user's folder) was shown in the console and
    # in the status of the interface.
    from adetailer.ultralytics import _typed_classes_note

    error = ImportError(text, name="clip") if missing_package else OSError(text)
    note = _typed_classes_note("YOLOE", "/models/adetailer/yoloe-11s-seg.pt", error)

    assert f"({type(error).__name__}: {kept})" in note
    for folder in ("someone", "some one", ".cache", "Users", "home", "server"):
        assert folder not in note
    assert note.startswith("the YOLOE detector 'yoloe-11s-seg.pt' could not take")
    # The folders the program is sure of are asked for this detector.
    assert known_folders == ["/models/adetailer/yoloe-11s-seg.pt"]


_CLIP_ADDRESS = "git+https://github.com/ultralytics/CLIP.git"


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        # A path without quotes, then a text with a slash of its own.
        (
            "cannot load /home/someone/models/text.ts: bad header, 3/4 of the bytes read",
            "cannot load text.ts: bad header, 3/4 of the bytes read",
        ),
        (
            "C:\\Users\\someone\\w\\text.ts is corrupt, see the log and/or retry",
            "text.ts is corrupt, see the log and/or retry",
        ),
        (
            "cannot read /home/someone/models with a bad header, n/a on 11/14/2026",
            "cannot read models with a bad header, n/a on 11/14/2026",
        ),
        (
            "cannot read /home/someone/models and 3/4 of the bytes and/or the rest",
            "cannot read models and 3/4 of the bytes and/or the rest",
        ),
        # ...also right after a path that has no file name at its end.
        (
            "cannot write to /home/someone/models and/or the disk is full",
            "cannot write to models and/or the disk is full",
        ),
        ("C:\\Users\\someone\\models 3/4 full", "models 3/4 full"),
        # ...or a web address: never cut, also right after the path.
        (
            f"no text model in /home/someone/w/text.ts. Run pip install {_CLIP_ADDRESS}",
            f"no text model in text.ts. Run pip install {_CLIP_ADDRESS}",
        ),
        (
            "ERROR /home/someone/w/yolo.pt appears to be a model trained with "
            "https://github.com/ultralytics/yolov5.",
            "ERROR yolo.pt appears to be a model trained with "
            "https://github.com/ultralytics/yolov5.",
        ),
        (
            "see https://example.com/get?file=/v1/text.ts for C:\\Users\\someone\\text.ts",
            "see https://example.com/get?file=/v1/text.ts for text.ts",
        ),
        # A command in quotes that starts with a path.
        (
            f"Command 'C:\\venv\\someone\\python.exe -m pip install {_CLIP_ADDRESS}' "
            "returned non-zero exit status 1.",
            f"Command 'python.exe -m pip install {_CLIP_ADDRESS}' "
            "returned non-zero exit status 1.",
        ),
        (
            "Command '/home/someone/venv/bin/python -m pytest tests/test_one.py' failed",
            "Command 'python -m pytest tests/test_one.py' failed",
        ),
        (
            "Command '/home/someone/bin/python /home/someone/run.py' failed, 1/2 done",
            "Command 'python run.py' failed, 1/2 done",
        ),
        # Still a whole path: spaces in quotes, a space in a folder's name.
        (
            "[Errno 2] No such file or directory: "
            "'C:\\Users\\someone\\My own models - new\\clip\\weights.pt' (1/2)",
            "[Errno 2] No such file or directory: 'weights.pt' (1/2)",
        ),
        (
            "[Errno 13] Permission denied: '/home/some one/.cache/clip' and/or locked",
            "[Errno 13] Permission denied: 'clip' and/or locked",
        ),
        (
            "C:\\Users\\some one\\.cache\\clip is not a folder, 3/4 read",
            "clip is not a folder, 3/4 read",
        ),
        (
            "copy /home/some one/a.pt /home/someone/b.pt (C:/Users/someone/c.pt) and/or retry",
            "copy a.pt b.pt (c.pt) and/or retry",
        ),
    ],
    ids=[
        "fraction", "and-or", "date", "no-file-name", "and-or-next", "fraction-next",
        "pip-command", "web-address",
        "address-with-a-path", "quoted-command", "quoted-option", "quoted-two-paths",
        "quoted-spaces", "quoted-folder", "bare-folder", "three-paths",
    ],
)
def test_a_path_in_an_error_text_does_not_take_the_text_after_it(text, kept, known_folders):
    # A path without quotes was cut up to the last slash of the line: "3/4",
    # "and/or", a web address or the pip command after it lost their start
    # ("cannot load 4 of the bytes read", "no text model in CLIP.git").
    from adetailer.ultralytics import _file_names_only, _typed_classes_note

    assert _file_names_only(text, _KNOWN) == kept
    note = _typed_classes_note("YOLOE", "/models/adetailer/yoloe-11s-seg.pt", OSError(text))
    assert f"(OSError: {kept[:160]})" in note
    for folder in ("someone", "some one", "Users", "home", "venv/", "venv\\"):
        assert folder not in kept


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        # Without quotes, a folder named with three words or more.
        (
            "ERROR C:\\Users\\Some Body Else\\sd\\models\\yolo.pt appears to be a YOLOv5 model",
            "ERROR yolo.pt appears to be a YOLOv5 model",
        ),
        (
            "open file failed, file path: D:\\AI\\Some WebUI Name - New\\models\\text.ts",
            "open file failed, file path: text.ts",
        ),
        (
            "cannot load C:\\Users\\someone\\Cloud Drive - Some Company Name\\sd\\text.ts: bad header",
            "cannot load text.ts: bad header",
        ),
        ("cannot read C:\\Users\\Some Body Else\\models", "cannot read models"),
        (
            "cannot read C:\\Users\\Some Body Else\\models and 3/4 of the bytes, n/a",
            "cannot read models and 3/4 of the bytes, n/a",
        ),
        (
            "cannot load /home/someone/My own models/clip/text.ts: bad header, 3/4 read",
            "cannot load text.ts: bad header, 3/4 read",
        ),
        # A folder name that ends like a file name, or with a punctuation mark.
        (
            "ERROR C:\\Users\\S. B. Else\\models\\yolo.pt is not a model and/or is corrupt",
            "ERROR yolo.pt is not a model and/or is corrupt",
        ),
        (
            "no text model in C:\\Users\\someone\\SD v1.5 models, new\\clip\\text.ts (1/2)",
            "no text model in text.ts (1/2)",
        ),
        # Two paths, each with such a folder.
        (
            "copy C:\\Users\\Some Body Else\\a.pt to D:\\My Own Big Disk\\b.pt and/or retry",
            "copy a.pt to b.pt and/or retry",
        ),
        # A web address after it is still never cut.
        (
            "ERROR C:\\Users\\Some Body Else\\w\\yolo.pt was trained with "
            "https://github.com/ultralytics/yolov5.",
            "ERROR yolo.pt was trained with https://github.com/ultralytics/yolov5.",
        ),
        # The text between two pieces of a path stays, whatever it holds.
        (
            "cannot read C:\\Users\\someone\\models, then weights\\text.ts is bad",
            "cannot read models, then text.ts is bad",
        ),
    ],
    ids=[
        "three-words", "five-words", "company-folder", "no-file-name", "then-a-fraction",
        "forward-slashes", "initials", "dot-and-comma", "two-paths", "web-address",
        "two-pieces",
    ],
)
def test_a_folder_named_with_three_words_is_not_left_in_the_text(text, kept, known_folders):
    # A path without quotes stopped at the first word without a backslash:
    # with a folder of three words or more ("Some Body Else", a user's full
    # name) the rest of the path, that folder included, stayed in the note.
    # Now such a folder goes when it is one the program is sure of.
    from adetailer.ultralytics import _file_names_only, _typed_classes_note

    assert _file_names_only(text, _KNOWN) == kept
    note = _typed_classes_note("YOLOE", "/models/adetailer/yoloe-11s-seg.pt", OSError(text))
    assert f"(OSError: {kept[:160]})" in note
    for folder in ("Some", "Body", "Else", "someone", "Users", "home", "Company",
                   "My own", "My Own", "S. B.", "v1.5", "Disk"):
        assert folder not in kept


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        # In quotes (the usual form of an OSError): a folder with a word that
        # ends with a punctuation mark, or like a file name.
        (
            "[Errno 2] No such file or directory: 'C:\\Users\\Some R. Body\\models\\yolo.pt'",
            "[Errno 2] No such file or directory: 'yolo.pt'",
        ),
        (
            "[Errno 2] No such file or directory: '/Users/S. Body/models/yolo.pt'",
            "[Errno 2] No such file or directory: 'yolo.pt'",
        ),
        (
            'cannot open "C:\\Users\\Some Body, Jr\\models\\x.pt" now, 3/4 read',
            'cannot open "x.pt" now, 3/4 read',
        ),
        (
            "cannot open 'D:\\My Disk v1.5 new\\Private Folder\\x.pt' for reading",
            "cannot open 'x.pt' for reading",
        ),
        (
            "[Errno 13] Permission denied: '/home/some one/My notes.txt old/clip' and/or locked",
            "[Errno 13] Permission denied: 'clip' and/or locked",
        ),
        # Without quotes, forward slashes and no file name at the end.
        (
            "cannot read /Users/Some Body Else/models/adetailer",
            "cannot read adetailer",
        ),
        (
            "cannot read C:/Users/Some Body Else/models/adetailer today, 3/4 on 11/14/2026",
            "cannot read adetailer today, 3/4 on 11/14/2026",
        ),
    ],
    ids=[
        "quoted-initial", "quoted-initial-slashes", "quoted-comma", "quoted-version",
        "quoted-file-name-in-a-folder", "bare-slashes", "bare-slashes-then-a-date",
    ],
)
def test_a_quoted_path_or_one_with_slashes_leaves_no_folder(text, kept, known_folders):
    # In quotes the path stopped at the first word that ends with a
    # punctuation mark or like a file name ("R.", "v1.5"), and the folders
    # after it stayed ("Body\models\yolo.pt"); without quotes and without a
    # file name at its end, one with forward slashes stopped at its first
    # word ("Some Body Else/models/adetailer" stayed).
    from adetailer.ultralytics import _file_names_only, _typed_classes_note

    assert _file_names_only(text, _KNOWN) == kept
    note = _typed_classes_note("YOLOE", "/models/adetailer/yoloe-11s-seg.pt", OSError(text))
    assert f"(OSError: {kept[:160]})" in note
    for folder in ("Some", "Body", "Else", "Users", "home", "some one", "Jr", "Disk",
                   "Private", "notes", "models"):
        assert folder not in kept


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        # After the file name, a word with a backslash or a relative path: a
        # word with a backslash is cut by itself when it is a path, the
        # other words stay.
        (
            "cannot parse C:\\models\\data.yaml: found character '\\t' that cannot start any token",
            "cannot parse data.yaml: found character '\\t' that cannot start any token",
        ),
        (
            "C:\\m\\yolo.pt is not a valid model, did you mean yolo11n.pt or models\\yolo12n.pt?",
            "yolo.pt is not a valid model, did you mean yolo11n.pt or yolo12n.pt?",
        ),
        (
            "cannot load C:\\m\\yolo.pt (expected ultralytics/nn/tasks.py 8.3)",
            "cannot load yolo.pt (expected ultralytics/nn/tasks.py 8.3)",
        ),
        (
            "ERROR /home/someone/w/YOLO.PT is not a model, see weights/last.pt and runs/a/b.yaml",
            "ERROR YOLO.PT is not a model, see weights/last.pt and runs/a/b.yaml",
        ),
        # After a folder of three words and one that ends like a file name.
        (
            "ERROR C:\\Users\\Some Body Else\\SD v1.5 new\\yolo.safetensors is bad, try a\\b.ckpt",
            "ERROR yolo.safetensors is bad, try b.ckpt",
        ),
        # A file of any kind.
        (
            "cannot read C:\\Users\\Some Body Else\\pictures\\face.webp or the folder",
            "cannot read face.webp or the folder",
        ),
    ],
    ids=["escape", "did-you-mean", "relative-path", "upper-case", "three-words", "other-kind"],
)
def test_the_text_after_a_file_name_is_not_lost_to_a_later_word(text, kept, known_folders):
    # When a path without quotes went on to the LAST word with a backslash
    # (or a slash and a file name), such a word in the message took the file
    # name and the words before it: "... did you mean yolo11n.pt or
    # models\yolo12n.pt?" came out as "yolo12n.pt?".
    from adetailer.ultralytics import _file_names_only, _typed_classes_note

    assert _file_names_only(text, _KNOWN) == kept
    note = _typed_classes_note("YOLOE", "/models/adetailer/yoloe-11s-seg.pt", OSError(text))
    assert f"(OSError: {kept[:160]})" in note
    for folder in ("someone", "Some", "Body", "Else", "Users", "home", "v1.5", "C:"):
        assert folder not in kept


def test_a_long_text_without_spaces_is_read_once():
    # The web address pattern was tried again from each character of a run
    # of letters: about a second for 32,000 of them, four times that for
    # twice as many. A wide limit: the text is read in a few milliseconds.
    import time

    from adetailer.ultralytics import _file_names_only

    run = "a1+.-" * 12_800
    address = f"1{run}://example.com/get?file=/v1/text.ts"
    started = time.perf_counter()
    assert _file_names_only(run) == run
    assert _file_names_only("a" * 64_000) == "a" * 64_000
    assert time.perf_counter() - started < 1.5
    # An address is found as before, whatever stands before its first letter.
    for text in (address, f"see -e{address} and C:\\Users\\someone\\text.ts"):
        assert _file_names_only(text) == text.replace("C:\\Users\\someone\\", "")


@pytest.mark.parametrize(
    "text",
    [
        # What Ultralytics and Python write today.
        "Download failure for https://ultralytics.com/assets/mobileclip_blt.ts. "
        "Retry limit reached. <urlopen error [Errno 11001] getaddrinfo failed>",
        "No module named 'clip'",
        "Model ViT-B/32 not found; available models = ['RN50', 'ViT-B/32', 'ViT-B/16']",
        "PytorchStreamReader failed reading file data/0: invalid header or archive is corrupted",
        # A slash that is not a path's, an address with a path in it.
        "3/4 of the bytes read on 11/14/2026, n/a and/or 5 it/s",
        f"Run pip install {_CLIP_ADDRESS} (see https://example.com/a?b=/c/d.pt)",
    ],
)
def test_an_error_text_without_a_file_path_is_left_as_it_is(text):
    from adetailer.ultralytics import _file_names_only

    assert _file_names_only(text) == text
    assert _file_names_only(text, _KNOWN) == text


@pytest.mark.parametrize(
    ("text", "folders", "kept"),
    [
        # The folder ends like a file ("anna.ts dev", "my.pt files"): where a
        # path with spaces ends is not guessed any more.
        (
            "cannot open C:|Users|anna.ts dev|sd|models|face.webp now",
            ["C:|Users|anna.ts dev"],
            "cannot open face.webp now",
        ),
        (
            "cannot open C:|Users|someone|my.pt files|model.onnx now",
            ["C:|Users|someone", "C:|Users|someone|my.pt files"],
            "cannot open model.onnx now",
        ),
        (
            "cannot read /Volumes/data.bin backup/Private Folder/clip now",
            ["/Volumes/data.bin backup/Private Folder"],
            "cannot read clip now",
        ),
        # A later word with slashes of its own is not a piece of the path.
        (
            "cannot find /home/someone/models/clip - see docs/usage/cli for help",
            ["/home/someone"],
            "cannot find clip - see docs/usage/cli for help",
        ),
        (
            "cannot open C:|Users|someone|face.webp (see docs/usage/cli)",
            ["C:|Users|someone"],
            "cannot open face.webp (see docs/usage/cli)",
        ),
        (
            "tried /home/someone/a.pt, /home/someone/b and c/d/e",
            ["/home/someone"],
            "tried a.pt, b and c/d/e",
        ),
        # Forward slashes, a name with a space, no file name at the end.
        ("cannot read C:/Users/Some Body/adetailer", ["C:|Users|Some Body"], "cannot read adetailer"),
        (
            "cannot read D:/Users/Some Body Jr/clip, 3/4 read",
            ["D:|Users|Some Body Jr"],
            "cannot read clip, 3/4 read",
        ),
        ("env HOME=/Users/Some Body", ["/Users/Some Body"], "env HOME="),
        # An apostrophe in a path in single quotes, as Ultralytics writes it.
        (
            "'C:|Users|O'Brien|models|yolo.pt' does not exist",
            ["C:|Users|O'Brien"],
            "'yolo.pt' does not exist",
        ),
        (
            "cannot open '/Users/O'Brien Smith/models/clip' now",
            ["/Users/O'Brien Smith"],
            "cannot open 'clip' now",
        ),
        # After another sign, in another case, with other separators.
        (
            "cannot open \u2018/home/some one/models/clip\u2019 now",
            ["/home/some one"],
            "cannot open \u2018clip\u2019 now",
        ),
        ("cannot open `C:|Users|Some Body|models|clip` now", ["C:|Users|Some Body"], "cannot open `clip` now"),
        ("<C:|Users|Some Body|x.pt>", ["C:|Users|Some Body"], "x.pt>"),
        (
            "cannot open file:///C:/Users/Some Body/models/clip.pt now",
            ["C:|Users|Some Body"],
            "cannot open file:clip.pt now",
        ),
        (
            "cannot open //server/share/some one/clip now",
            ["||server|share|some one"],
            "cannot open clip now",
        ),
        ("||?|C:|Users|Some Body|x.pt is locked", ["C:|Users|Some Body"], "x.pt is locked"),
        ("gcc -IC:|Users|Some Body|include a.c", ["C:|Users|Some Body"], "gcc include a.c"),
        ("open c:/users/SOME BODY|AppData/x.pt", ["C:|Users|Some Body"], "open x.pt"),
        (
            "No such file: 'c:||users||some body||x.pt'",
            ["C:|Users|Some Body"],
            "No such file: 'x.pt'",
        ),
        ("cannot read C:|Users|Some Body.", ["C:|Users|Some Body|"], "cannot read ."),
        # The longest first: the temporary folder inside the user's.
        (
            "cannot write to C:|Users|Some Body|AppData|Local|Temp now",
            ["C:|Users|Some Body", "C:|Users|Some Body|AppData|Local|Temp"],
            "cannot write to  now",
        ),
        # Inside a longer path, glued to an option.
        ("cannot read /mnt/backup/home/some one/x.pt", ["/home/some one"], "cannot read x.pt"),
        ("gcc -I/home/some one/include a.c", ["/home/some one"], "gcc include a.c"),
        # A name alone under the root is not looked for (it is a piece of
        # other texts too): it is cut as any path, where a path starts.
        (
            "see docs/app/a.md and https://app.example.com/app/a, not /app/x.pt in /app",
            ["/app"],
            "see docs/app/a.md and https://app.example.com/app/a, not x.pt in app",
        ),
        # A web address stays whole, also with a folder written in it.
        (
            "see https://example.com/get?file=/home/someone/a.pt for /home/someone/a.pt",
            ["/home/someone"],
            "see https://example.com/get?file=/home/someone/a.pt for a.pt",
        ),
    ],
    ids=[
        "user-folder-like-a-script", "folder-like-a-model", "slashes-like-a-file",
        "later-word-with-slashes", "later-word-in-brackets", "second-path",
        "name-with-a-space", "then-a-fraction", "the-home-itself",
        "apostrophe-windows", "apostrophe-posix", "curly-quotes", "backticks",
        "angle-brackets", "file-address", "network-share", "long-path-prefix",
        "glued-option", "another-case", "repr-in-another-case", "full-stop",
        "longest-first", "inside-a-path", "glued-option-slashes", "one-name-folder",
        "in-a-web-address",
    ],
)
def test_a_known_folder_is_taken_out_wherever_it_is_written(text, folders, kept):
    # The end of a path with spaces was guessed, and each guess left a folder
    # in the note or took words of the message with it ("cannot find cli for
    # help"). A folder the program is sure of is now looked for as it is.
    from adetailer.ultralytics import _file_names_only

    assert _file_names_only(_w(text), [_w(folder) for folder in folders]) == _w(kept)


@pytest.mark.parametrize(
    ("word", "kept"),
    [
        # Plainly a piece of a path: cut down to what follows its last
        # separator, with the signs around it left where they are.
        ("'C:|a|b.pt'", "'b.pt'"),
        ('"C:/a/b.pt":', '"b.pt":'),
        ("(~/a/b.pt),", "(b.pt),"),
        ("name=/a/b.pt?", "name=b.pt?"),
        ("--weights=C:|a|b.pt", "--weights=b.pt"),
        ("['/a/b.pt',", "['b.pt',"),
        ("a|b.pt", "b.pt"),
        ("||server|share|b.pt", "b.pt"),
        ("C://a//b.pt", "b.pt"),
        ("file:///C:/a/b.pt", "file:b.pt"),
        ("C:|a|", ""),
        # Not one: slashes only, and no start of a path.
        ("and/or", "and/or"),
        ("n/a", "n/a"),
        ("3/4", "3/4"),
        ("(1/2)", "(1/2)"),
        ("11/14/2026", "11/14/2026"),
        ("docs/usage/cli", "docs/usage/cli"),
        ("'ViT-B/32',", "'ViT-B/32',"),
        ("ratio=16/9", "ratio=16/9"),
        ("/", "/"),
        ("//", "//"),
        ("</title></head>", "</title></head>"),
        # A web address, whatever it holds.
        ("https://example.com/a/b.pt", "https://example.com/a/b.pt"),
        ("git+https://github.com/a/b.git", "git+https://github.com/a/b.git"),
        ("(https://example.com/a?b=/c/d.pt)", "(https://example.com/a?b=/c/d.pt)"),
        ("url='HTTPS://example.com/C:|a|b.pt'", "url='HTTPS://example.com/C:|a|b.pt'"),
        ("s3://bucket/a/b.pt", "s3://bucket/a/b.pt"),
    ],
)
def test_a_word_is_cut_only_when_it_is_plainly_a_piece_of_a_path(word, kept):
    from adetailer.ultralytics import _file_names_only

    assert _file_names_only(_w(f"cannot open {word} now")) == _w(f"cannot open {kept} now")
    assert _file_names_only(_w(word)) == _w(kept)


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        # A backslash that is a character of the message, not a folder: the
        # word stays whole (it came out as "'t'", "d", "'nhand'", "d+").
        (
            "cannot parse data.yaml: found character '|t' that cannot start any token",
            "cannot parse data.yaml: found character '|t' that cannot start any token",
        ),
        ("bad escape |d at position 3", "bad escape |d at position 3"),
        ("invalid class 'face|nhand' in names", "invalid class 'face|nhand' in names"),
        ('{"re": "||d+", "n": 1}', '{"re": "||d+", "n": 1}'),
        ("pattern |d+|.|d+ failed", "pattern |d+|.|d+ failed"),
        # A folder without a file name at the end, from a start of a path
        # that is not a drive: a network share in the repr of an error (its
        # backslashes doubled), the same in a JSON, the long form of a share,
        # a path from the root without its drive. Left whole, they kept the
        # name of the user's folder.
        (
            "The network path was not found: '||||nas||home||someone||clip'",
            "The network path was not found: 'clip'",
        ),
        ('{"path": "||||nas||someone||cache"}', '{"path": "cache"}'),
        ("cannot open ||?|UNC|nas|someone|cache now", "cannot open cache now"),
        ("cannot open ||||?||UNC||nas||someone||cache now", "cannot open cache now"),
        ("cannot open |Users|someone|cache now", "cannot open cache now"),
        # A piece of a path: a drive (also glued to an option), the start of
        # a path, a file name after the last separator, or a word after a
        # folder that did not end with a file name.
        ("see weights|text.ts now", "see text.ts now"),
        ("did you mean models|yolo12n.pt?", "did you mean yolo12n.pt?"),
        ("cannot open Some Body|models|yolo.pt now", "cannot open Some yolo.pt now"),
        ("cannot open D:|Other Disk|My Models|x now", "cannot open Other My x now"),
        ("cannot open ..|Other Disk|x now", "cannot open Other x now"),
        ("gcc -ID:|Other Disk|include a.c", "gcc -ID:Other include a.c"),
        ("cannot open ||server|share|Other Disk|x now", "cannot open Other x now"),
        # After a file name the next word with a backslash is read by itself.
        ("cannot open D:|a|b.pt: bad escape |d", "cannot open b.pt: bad escape |d"),
        # Escapes followed by letters are no path from the root: after ONE
        # backslash a folder's name starts with a capital letter.
        ("found '|tface|thand' here", "found '|tface|thand' here"),
        ("found '|bface|b' here", "found '|bface|b' here"),
        ("found '|x00|x01|x02' here", "found '|x00|x01|x02' here"),
    ],
    ids=[
        "yaml-tab", "regex", "repr-of-a-class", "json", "regex-of-two-pieces",
        "share-in-a-repr", "share-in-a-json", "long-form", "long-form-in-a-repr", "from-the-root",
        "file-name", "file-name-and-sign",
        "relative-file", "after-a-folder", "relative-start", "glued-option", "network-share",
        "after-a-file-name", "escapes-and-words", "backspace-escapes", "hex-escapes",
    ],
)
def test_a_word_with_a_backslash_is_cut_only_when_it_is_a_piece_of_a_path(text, kept):
    # Every word with a backslash was cut as a path: the error a user read
    # was another one ("found character 't'", "bad escape d", "'nhand'").
    from adetailer.ultralytics import _file_names_only

    assert _file_names_only(_w(text), [_w("C:|Users|Some Body")]) == _w(kept)
    assert _file_names_only(_w(text)) == _w(kept)


def test_the_documents_name_every_start_of_a_path_as_the_code_reads_it():
    # README and CHANGELOG list the starts of a path a word is cut from; each
    # one cuts a folder without a file name, the one-letter "name" does not,
    # and their example of a word after such a folder comes out as they say.
    from adetailer.ultralytics import _file_names_only

    said = _w(
        "Plainly: it starts with `/`, `~/`, `~|`, `.|`, `..|`, a drive letter and a slash "
        "or backslash, a network share (`||server|share`, also with the doubled backslashes "
        "of an error's repr), the long form `||?|` (also `||?|UNC|`), or a backslash, a name "
        "that starts with a capital letter and another backslash (`|Users|name`); or it holds a "
        "backslash and ends with a file name such as `weights|text.ts`; or it holds a "
        "backslash and comes after a piece of a path that did not end with a file name. The "
        'other words of the text, also the ones like "and/or" and a backslash that belongs '
        "to the message such as `'|t'` or `|d`, are left as they are, unless they come after "
        "a piece of a path that did not end with a file name (`D:|Other: bad escape |d` "
        "becomes `Other: bad escape d`)."
    )
    root = Path(__file__).parents[1]
    for name in ("README.md", "CHANGELOG.md"):
        assert said in (root / name).read_text(encoding="utf-8"), name
    for path in (
        "/someone/cache", "~/someone/cache", "~|someone|cache", ".|someone|cache",
        "..|someone|cache", "D:|someone|cache", "D:/someone/cache", "||server|someone|cache",
        "||||server||someone||cache", "||?|D:|someone|cache", "||?|UNC|server|someone|cache",
        "|Users|someone|cache",
    ):  # fmt: skip
        assert _file_names_only(_w(f"cannot open {path} now")) == "cannot open cache now", path
    for word in ("'|t'", "|d", "|d|someone", "|tface|thand", "and/or"):
        assert _file_names_only(_w(f"bad escape {word} now")) == _w(f"bad escape {word} now")
    assert _file_names_only(_w("see weights|text.ts")) == "see text.ts"
    assert _file_names_only(_w("D:|Other: bad escape |d")) == "Other: bad escape d"


def test_a_network_share_in_the_folders_is_not_asked_to_the_disk(monkeypatch):
    # Resolving "\\server\share" or "//server/share" asks a computer of the
    # network, that can wait long when it does not answer; the folder is
    # still in the list as it is written.
    import adetailer.ultralytics as module

    asked = []
    resolve = module.Path.resolve

    def recorded(self, *args, **kwargs):
        asked.append(str(self))
        return resolve(self, *args, **kwargs)

    monkeypatch.setattr(module.Path, "resolve", recorded)
    monkeypatch.setenv("HF_HOME", "\\\\server\\share\\HF Cache")
    monkeypatch.setenv("TORCH_HOME", "//server/share/Torch Cache")
    folders = module._known_folders("/models/yoloe-11s-seg.pt")

    assert "\\\\server\\share\\HF Cache" in folders
    assert "//server/share/Torch Cache" in folders
    assert asked
    assert not [path for path in asked if "server" in path]


def test_a_drive_letter_of_a_network_share_is_not_asked_to_the_disk(monkeypatch):
    # The same for a drive letter given to a network share ("Z:\"): Windows'
    # table of the drive names tells it, without asking the network.
    import sys
    from types import SimpleNamespace

    import adetailer.ultralytics as module

    targets = {
        "Z:": "\\Device\\LanmanRedirector\\;Z:0000000000012345\\server\\share",
        "Y:": "\\Device\\Mup\\;WebDavRedirector\\server\\share",
        "X:": "\\??\\UNC\\server\\share\\x",
        "C:": "\\Device\\HarddiskVolume3",
    }

    def query(name, buffer, size):
        buffer.value = targets.get(name, "")
        return len(buffer.value)

    class Buffer:
        def __init__(self, size):
            self.value, self.size = "", size

        def __len__(self):
            return self.size

    fake = SimpleNamespace(
        create_unicode_buffer=Buffer,
        windll=SimpleNamespace(kernel32=SimpleNamespace(QueryDosDeviceW=query)),
    )
    monkeypatch.setitem(sys.modules, "ctypes", fake)
    # Only this module's "os" is replaced: os.name itself changed makes
    # pathlib build a path of the other system, which Python before 3.12
    # refuses (the next Path() of the whole run fails).
    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt"))
    assert module._network_drive("Z:\\HF Cache")
    assert module._network_drive("y:/x")
    assert module._network_drive("X:")
    assert not module._network_drive("C:\\Users\\someone")
    assert not module._network_drive("W:\\not a drive")
    assert not module._network_drive("/home/someone")
    monkeypatch.undo()
    # Not on Windows, and without the table: never.
    monkeypatch.setattr(module, "os", SimpleNamespace(name="posix"))
    assert not module._network_drive("Z:\\HF Cache")
    monkeypatch.undo()

    asked = []
    resolve = module.Path.resolve

    def recorded(self, *args, **kwargs):
        asked.append(str(self))
        return resolve(self, *args, **kwargs)

    monkeypatch.setattr(module.Path, "resolve", recorded)
    monkeypatch.setattr(module, "_network_drive", lambda folder: folder.startswith("Z:"))
    monkeypatch.setenv("HF_HOME", "Z:\\HF Cache")
    folders = module._known_folders("/models/yoloe-11s-seg.pt")

    assert "Z:\\HF Cache" in folders
    assert asked
    assert not [path for path in asked if path.startswith("Z:")]


@pytest.mark.parametrize(
    ("text", "left", "folder", "clean"),
    [
        (
            "cannot open D:|Other Disk|x.pt now",
            "cannot open Other x.pt now",
            "D:|Other Disk",
            "cannot open x.pt now",
        ),
        (
            "cannot open 'D:|Other Disk|My Models|yolo.pt' now",
            "cannot open 'Other My yolo.pt' now",
            "D:|Other Disk|My Models",
            "cannot open 'yolo.pt' now",
        ),
        (
            "cannot open /mnt/Other Disk/b/a.pt now",
            "cannot open Other Disk/b/a.pt now",
            "/mnt/Other Disk",
            "cannot open a.pt now",
        ),
        (
            "cannot load C:|Users|someone|Cloud Drive - Some Company|sd|text.ts: bad header",
            "cannot load Cloud Drive - Some text.ts: bad header",
            "C:|Users|someone|Cloud Drive - Some Company",
            "cannot load text.ts: bad header",
        ),
    ],
    ids=["bare", "quoted", "slashes", "inside-the-user-folder"],
)
def test_a_folder_with_spaces_that_is_not_known_leaves_words_of_its_name(
    text, left, folder, clean
):
    # The limit of the rule, as it is: nothing says where a path with spaces
    # ends, so only its words with a separator are cut. The words of a
    # folder's name stay, and with forward slashes what follows them too.
    from adetailer.ultralytics import _file_names_only

    sure_of = [_w("C:|Users|someone"), "/home/someone"]
    assert _file_names_only(_w(text), sure_of) == left
    # The root of the path is gone, and no word of the message is lost.
    for root in ("D:", "mnt", "Users", "someone"):
        assert root not in left
    words = iter(left.split())
    assert all(word in words for word in text.split() if "|" not in word and "/" not in word)
    # The same folder, once the program is sure of it, leaves nothing.
    assert _file_names_only(_w(text), [*sure_of, _w(folder)]) == clean


def test_no_word_of_any_text_is_lost_and_no_backslash_is_left_in_a_cut_word():
    # Texts made by chance from one seed: a word with neither a slash nor a
    # backslash comes out as it went in, at its place; a word that starts
    # with a web address too; a word that keeps a backslash out of a web
    # address is kept whole (it is no path: "'\t'", "\d+"), a word that
    # starts as a path ("C:\", "~\", ".\", "\\server\share", "\\?\",
    # "\Users\name", after the signs a path can follow) keeps no backslash,
    # and reading the result again changes nothing.
    import random
    import re

    from adetailer.ultralytics import _file_names_only

    signs = r"""[\s'"`(\[{=:,;\u2018\u201c]*"""
    # The long form, two backslashes or more before a name and one more
    # piece, or one backslash before a name with a capital letter first.
    new_starts = (
        r"\\{2,}\?\\+[^\s\\/]|\\{2,}[\w.$\-]{2,}\\+[^\s\\/]|\\[A-Z][\w.$\-]+\\+[^\s\\/]"
    )
    starts = re.compile(rf"{signs}(?:[A-Za-z]:[\\/]|~?/+[^\s/]|~\\|\.\.?\\|{new_starts})")
    chance = random.Random(20)
    pieces = [
        "a", "B", "7", "x.pt", "some", " ", " ", "\t", "\\", "\\", "/", "/", "'", '"', ":",
        ".", ",", "(", ")", "=", "-", "~", "`", "C:", "https://", "git+https://", "file://",
        "and/or", "?", "[", "]", "\\\\?\\", "\\nas\\", "\\\\nas", "\\Users\\", "\\tface",
    ]  # fmt: skip
    seen = new = 0
    for _ in range(4000):
        text = "".join(chance.choice(pieces) for _ in range(chance.randrange(40)))
        out = _file_names_only(text)
        words, kept = re.split(r"\s", text), re.split(r"\s", out)
        assert len(kept) == len(words), text
        for word, now in zip(words, kept):
            if ("\\" not in word and "/" not in word) or re.match(
                r"""['"(=]*(?:git\+)?https://""", word
            ):
                assert now == word, text
            assert len(now) <= len(word), text
            assert "\\" not in now or "://" in now or now == word, text
            if starts.match(word) and "://" not in word:
                seen += 1
                new += bool(re.match(rf"{signs}(?:{new_starts})", word))
                assert "\\" not in now, text
        assert _file_names_only(out) == out, text
    # The texts made hold many such words, also of the forms added last.
    assert seen > 500
    assert new > 300


def test_known_folders_leave_nothing_in_any_text_and_the_rest_stays():
    # Texts made by chance from one seed, with the folders the program is
    # sure of written in any case, with any separator, bare or between
    # signs, alone or with a path after them. None of them is left, nor its
    # last name (the user's); the words of the message, the words with a
    # slash of their own and the web addresses stay as they are, in order.
    import random
    import re

    from adetailer.ultralytics import _file_names_only

    chance = random.Random(2026)
    folders = [
        _w("C:|Users|Some Body"),
        _w("C:|Users|Some Body|AppData|Local|Temp"),
        "/home/some one",
        "/Users/anna.ts dev",
        _w("D:|Stable WebUI - New|venv"),
        _w("||server|share|Team Folder"),
        "/srv/Shared App",
    ]
    names = [re.split(r"[\\/]", folder)[-1].lower() for folder in folders]
    stays = [
        "cannot", "open", "the", "file", "now", "error:", "retry", "(1/2)", "and/or", "3/4",
        "11/14/2026", "docs/usage/cli", "https://example.com/a/b.pt", _CLIP_ADDRESS,
        "(https://example.com/get?file=/v1/text.ts)", "http://localhost:7860/app/x",
    ]  # fmt: skip

    def path() -> str:
        folder = chance.choice(folders)
        tail = chance.sample(["models", "clip", "sd", "weights.pt", "text.ts"], chance.randrange(4))
        written = chance.choice([str, str.lower, str.upper])("\\".join([folder, *tail]))
        separator = chance.choice(["\\", "/", "\\\\"])
        written = re.sub(r"[\\/]", lambda _: separator, written)
        before = chance.choice(["", "", "'", '"', "(", "name=", "file://"])
        return before + written + chance.choice(["", "", "'", '"', ")", ":", ",", "."])

    for _ in range(3000):
        words = [
            chance.choice(stays) if chance.random() < 0.6 else path()
            for _ in range(chance.randrange(1, 12))
        ]
        text = " ".join(words)
        out = _file_names_only(text, folders)
        for name in names:
            assert name not in out.lower(), text
        assert "\\" not in out, text
        assert [word for word in out.split() if word in stays] == [
            word for word in words if word in stays
        ], text


@pytest.mark.parametrize(
    "text",
    [
        "C:\\Users\\someone\\a.pt /home/x/b.pt " * 3_000,
        "http://a /b " * 8_400,
        "C:\\Users\\Some Body\\x.pt /home/some one/y.pt " * 2_400,
        "C:\\Users\\Some " * 7_000,
        "/home" + "/" * 100_000,
        "C:" + "\\" * 100_000,
        "/home/" * 17_000,
        "/" * 100_000,
        "\\" * 100_000,
        "'" * 100_000,
        "'C:\\a b " * 12_500,
        "'\\ab" * 25_000,
        "'" + "\\" * 50_000 + "ab" * 25_000,
        "'\\\\?" * 25_000,
        "a=" * 50_000,
        ":/" * 50_000,
        "a1+.-" * 20_000,
        "a" * 100_000,
        " " * 100_000,
    ],
    ids=[
        "paths", "addresses-and-paths", "known-folders", "starts-of-a-known-folder",
        "slashes-after-a-folder", "backslashes-after-a-drive", "folder-names", "slashes",
        "backslashes", "quotes",
        "open-quotes", "starts-from-the-root", "backslashes-before-a-name", "long-forms",
        "equals", "colons", "no-space", "letters", "spaces",
    ],
)
def test_a_hundred_thousand_characters_are_read_in_no_time(text):
    # Thousands of paths in one line took seconds (each path was compared
    # with every web address and every other start of a path). About 100,000
    # characters of any shape take some hundredths of a second: the limit is
    # wide, for a slow computer.
    import time

    from adetailer.ultralytics import _file_names_only

    assert len(text) >= 96_000
    folders = [*_KNOWN, _w("C:|Users|Some Body"), "/home", "/home/some one", "/usr", "/tmp"]
    started = time.perf_counter()
    out = _file_names_only(text, folders)
    assert time.perf_counter() - started < 1.0
    assert "Some Body" not in out
    assert "some one" not in out


def test_the_folders_of_this_computer_are_listed_without_an_error(tmp_path, monkeypatch):
    # The true list, whatever the system: only that it is made, of texts,
    # with the user's folder, the working one and the detector's in it.
    import adetailer.ultralytics as module

    detector = tmp_path / "My detectors" / "yoloe-11s-seg.pt"
    folders = module._known_folders(detector)

    assert folders
    assert all(isinstance(folder, str) for folder in folders)
    for folder in (Path.home(), Path.cwd(), detector.parent):
        assert str(folder) in folders
    assert module._known_folders("yoloe-11s-seg.pt")
    # What the WebUI tells and the environment names is in it; a folder that
    # cannot be read (no user's folder, no working folder) is left out.
    webui = ModuleType("modules.paths")
    webui.models_path = str(tmp_path / "Shared Models")
    webui.data_path = None
    monkeypatch.setitem(sys.modules, "modules.paths", webui)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "HF Cache"))
    monkeypatch.delenv("TORCH_HOME", raising=False)

    def unreadable(*args, **kwargs):
        raise RuntimeError

    monkeypatch.setattr(module.Path, "home", unreadable)
    monkeypatch.setattr(module.Path, "cwd", unreadable)
    monkeypatch.setattr(module.tempfile, "gettempdir", unreadable)
    folders = module._known_folders(detector)
    for folder in ("Shared Models", "HF Cache", "My detectors"):
        assert str(tmp_path / folder) in folders
    # Whatever is in the list, a text is read: what is not a whole folder
    # ("", ".", a root, a drive alone, a name) is never looked for.
    text = "a.b c:d e/f g\\h . / C:\\ models None"
    not_folders = ["", ".", "/", "\\", "C:", "C:\\", "C:models", "models", "None", "/tmp/"]
    assert module._file_names_only(text, not_folders) == "a.b c:d e/f g\\h . /  models None"


def test_the_typed_classes_note_leaves_out_the_folders_of_this_detector(tmp_path):
    # With the true list of folders: the detector's own folder, with spaces
    # in its name, goes from the error text in any way it is written, before
    # the cut to 160 characters; the detector is still named by its file.
    from adetailer.ultralytics import _typed_classes_note

    detector = tmp_path / "My Own Detectors" / "yoloe-11s-seg.pt"
    folder = str(detector.parent)
    for written in (
        folder,
        folder.upper(),
        folder.replace("\\", "/"),
        repr(folder)[1:-1],
    ):
        text = (
            f"cannot find {written}/clip - see docs/usage/cli for help, "
            + f"not {written} " * 8
            + f"(1/2) '{written}/text.ts'"
        )
        note = _typed_classes_note("YOLOE", detector, OSError(text))

        assert note == (
            "the YOLOE detector 'yoloe-11s-seg.pt' could not take the typed classes "
            "(OSError: cannot find clip - see docs/usage/cli for help, "
            + "not  " * 8
            + "(1/2) 'text.ts')."
        )


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("offline\nsecond line " + "x" * 400),
        AttributeError("module 'clip' has no attribute 'load'"),
        ModuleNotFoundError("No module named 'ftfy'", name="ftfy"),
    ],
    ids=["download", "another-clip", "another-package"],
)
def test_yolo_world_that_cannot_take_the_classes_skips_the_tab(
    tmp_path, monkeypatch, error
):
    # Any other failure of the text model (its weights not downloaded, a
    # package of another kind under the same name): the tab is skipped with
    # the error in one line, as for YOLOE, and no install command is made up.
    calls = _fake_yoloe(monkeypatch)

    def set_classes(self, classes, embeddings=None):
        raise error

    sys.modules["ultralytics"].YOLO.set_classes = set_classes

    result = _predict(tmp_path / "yolov8s-worldv2.pt", classes="face")

    assert result.bboxes == []
    assert not any(c[0] == "predict" for c in calls)
    first = str(error).splitlines()[0]
    assert result.note == (
        "the YOLO-World detector 'yolov8s-worldv2.pt' could not take the typed "
        f"classes ({type(error).__name__}: {first})."
    )
    assert "pip install" not in result.note
    assert "\n" not in result.note


def test_yolo_world_without_typed_classes_needs_no_text_model(tmp_path, monkeypatch):
    # Its own classes: set_classes is not called, so a missing "clip" package
    # changes nothing.
    calls = _fake_yoloe(monkeypatch)

    def set_classes(self, classes, embeddings=None):
        raise _no_module("clip")

    sys.modules["ultralytics"].YOLO.set_classes = set_classes

    result = _predict(tmp_path / "yolov8s-worldv2.pt")

    assert [c[0] for c in calls] == ["predict"]
    assert result.class_names == ["person"]
    assert result.note == ""


def test_apply_classes_shim_says_why_yoloe_could_not_take_the_classes(
    monkeypatch, capfd
):
    _fake_yoloe(monkeypatch, pe_error=_no_module("clip"))
    model = sys.modules["ultralytics"].YOLO("yoloe-11s-ü.pt")

    apply_classes(model, "yoloe-11s-ü.pt", "face")

    err = capfd.readouterr().err
    assert err.startswith(
        "[-] ADetailer: the YOLOE detector 'yoloe-11s-\\xfc.pt' could not take "
        "the typed classes: the Python package 'clip' is missing"
    )
    assert err.rstrip().endswith(
        f". Run {_CLIP_COMMAND} with the WebUI's Python (its virtual environment)."
    )
    assert err.isascii()


def test_prompt_free_yoloe_is_an_ordinary_detector(tmp_path, monkeypatch):
    calls = _fake_yoloe(monkeypatch, prompt_free=True)
    pf = _with_names(tmp_path / "yoloe-11s-seg-pf.pt")
    get_model_class_names.cache_clear()
    try:
        result = _predict(pf, classes="car")
    finally:
        get_model_class_names.cache_clear()

    # Its own classes, filtered by number; never set_classes.
    assert calls == [("predict", {"conf": 0.3, "device": "", "classes": [1]})]
    assert result.class_names == ["person"]


def test_yolo_world_still_gets_one_argument_set_classes(tmp_path, monkeypatch):
    calls = _fake_yoloe(monkeypatch)

    _predict(tmp_path / "yolov8s-worldv2.pt", classes="face")

    assert calls[0] == ("set_classes", ["face"], None)


def test_apply_classes_shim_uses_text_prompts_for_yoloe(monkeypatch):
    calls = _fake_yoloe(monkeypatch)
    model = sys.modules["ultralytics"].YOLO("yoloe-11s-seg.pt")

    apply_classes(model, "yoloe-11s-seg.pt", "face")
    apply_classes(model, "yoloe-11s-seg-pf.pt", "face")

    assert calls == [
        ("get_text_pe", ["face"]),
        ("set_classes", ["face"], "pe:face"),
    ]
