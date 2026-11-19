import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from adetailer.classes import (
    _names_from_json,
    get_model_class_names,
    parse_csv,
    resolve_class_ids,
)
from adetailer.ultralytics import mask_to_pil, ultralytics_predict


@pytest.mark.integration
@pytest.mark.parametrize(
    "model_name",
    [
        "face_yolov8n.pt",
        "face_yolov8n_v2.pt",
        "face_yolov8s.pt",
        "face_yolov9c.pt",
        "hand_yolov8n.pt",
        "hand_yolov8s.pt",
        "hand_yolov9c.pt",
        "person_yolov8n-seg.pt",
        "person_yolov8s-seg.pt",
        "person_yolov8m-seg.pt",
        "deepfashion2_yolov8s-seg.pt",
    ],
)
def test_ultralytics_hf_models(sample_image: Image.Image, model_name: str):
    model_path = hf_hub_download("Bingsu/adetailer", model_name)
    result = ultralytics_predict(model_path, sample_image)
    assert result.preview is not None
    assert len(result.bboxes) > 0
    assert len(result.masks) > 0
    assert len(result.confidences) > 0
    assert len(result.bboxes) == len(result.masks) == len(result.confidences)


@pytest.mark.integration
def test_yolo_world_default(sample_image: Image.Image):
    model_path = hf_hub_download("Bingsu/yolo-world-mirror", "yolov8x-worldv2.pt")
    result = ultralytics_predict(model_path, sample_image)
    assert result.preview is not None
    assert len(result.bboxes) > 0
    assert len(result.masks) > 0
    assert len(result.confidences) > 0
    assert len(result.bboxes) == len(result.masks) == len(result.confidences)


@pytest.mark.integration
@pytest.mark.parametrize(
    "klass",
    [
        "person",
        "bird",
        "yellow bird",
        "person,glasses,headphone",
        "person,bird",
        "glasses,yellow bird",
    ],
)
def test_yolo_world(sample_image2: Image.Image, klass: str):
    model_path = hf_hub_download("Bingsu/yolo-world-mirror", "yolov8x-worldv2.pt")
    result = ultralytics_predict(model_path, sample_image2, classes=klass)
    assert result.preview is not None
    assert len(result.bboxes) > 0
    assert len(result.masks) > 0
    assert len(result.confidences) > 0
    assert len(result.bboxes) == len(result.masks) == len(result.confidences)


@pytest.mark.integration
def test_class_filter_include_person(sample_image: Image.Image):
    """Single-class model: filtering on its only class is a no-op vs unfiltered."""
    model_path = hf_hub_download("Bingsu/adetailer", "person_yolov8n-seg.pt")
    full = ultralytics_predict(model_path, sample_image)
    filtered = ultralytics_predict(model_path, sample_image, classes="person")
    assert len(filtered.bboxes) == len(full.bboxes)
    assert len(filtered.masks) == len(full.masks)


@pytest.mark.integration
def test_class_filter_include_unknown_falls_back(sample_image: Image.Image):
    """Unknown class names are dropped; if no valid id remains, falls back to no filter."""
    model_path = hf_hub_download("Bingsu/adetailer", "person_yolov8n-seg.pt")
    full = ultralytics_predict(model_path, sample_image)
    filtered = ultralytics_predict(model_path, sample_image, classes="nonexistent_class")
    assert len(filtered.bboxes) == len(full.bboxes)


@pytest.mark.integration
def test_class_filter_exclude_person(sample_image: Image.Image):
    """Excluding the only class the model produces should yield zero detections."""
    model_path = hf_hub_download("Bingsu/adetailer", "person_yolov8n-seg.pt")
    result = ultralytics_predict(model_path, sample_image, exclude_classes="person")
    assert len(result.bboxes) == 0
    assert len(result.masks) == 0
    assert len(result.confidences) == 0


def test_parse_csv():
    assert parse_csv("") == []
    assert parse_csv("face") == ["face"]
    assert parse_csv("face, hand ,eye") == ["face", "hand", "eye"]
    assert parse_csv(",,,") == []


class TestNamesFromJson:
    def test_list(self):
        assert _names_from_json(["face", "hand"]) == ["face", "hand"]

    def test_names_list(self):
        assert _names_from_json({"names": ["face", "hand"]}) == ["face", "hand"]

    def test_names_dict(self):
        assert _names_from_json({"names": {"0": "face", "1": "hand"}}) == ["face", "hand"]

    def test_bare_int_keys(self):
        assert _names_from_json({"0": "face", "1": "hand", "2": "eye"}) == [
            "face",
            "hand",
            "eye",
        ]

    def test_civitai_helper_sidecar_ignored(self):
        """Regression: civitai_helper-style metadata JSON must not be mistaken
        for a class-names file. It has string keys and nested dicts; we should
        return [] so the caller falls through to model.names."""
        civitai_blob = {
            "description": "<h2>...</h2>",
            "sd version": "Unknown",
            "extensions": {"sd_civitai_helper": {"version": "1.8.13"}},
        }
        assert _names_from_json(civitai_blob) == []

    def test_partial_int_keys_rejected(self):
        """A dict with some int keys and some non-int keys is NOT a class map."""
        assert _names_from_json({"0": "face", "description": "x"}) == []

    def test_nested_values_rejected(self):
        """A dict with int keys whose values are dicts is NOT a class map."""
        assert _names_from_json({"0": {"name": "face"}, "1": {"name": "hand"}}) == []

    def test_empty_returns_empty(self):
        assert _names_from_json({}) == []
        assert _names_from_json([]) == []
        assert _names_from_json(None) == []
        assert _names_from_json("face") == []


@pytest.mark.integration
def test_resolve_class_ids_with_known_model():
    """resolve_class_ids should accept names AND numeric strings."""
    model_path = hf_hub_download("Bingsu/adetailer", "person_yolov8n-seg.pt")
    ids = resolve_class_ids(model_path, ["person", "0", "unknown"])
    assert 0 in ids
    assert all(isinstance(i, int) for i in ids)


class _FakeBoxes:
    def __init__(self, cls):
        n = len(cls)
        self.xyxy = torch.tensor([[1.0, 1.0, 3.0, 3.0]] * n)
        self.conf = torch.tensor([0.9] * n)
        self.cls = torch.tensor(cls)

    def __len__(self):
        return len(self.cls)


class _FakeResult:
    def __init__(self, names, cls):
        self.names = names
        self.boxes = _FakeBoxes(cls)
        self.masks = None

    def plot(self):
        return np.zeros((4, 4, 3), np.uint8)


def _fake_yolo(monkeypatch, cls, names):
    """Fake Ultralytics model returning one result with boxes of class ids
    `cls`. Like YOLO-World, set_classes replaces its vocabulary, which the
    result reports as `names`."""
    module = ModuleType("ultralytics")

    class YOLO:
        def __init__(self, path):
            self.names = names

        def set_classes(self, classes):
            self.names = list(classes)

        def __call__(self, image, **kwargs):
            return [_FakeResult(self.names, cls)]

    module.YOLO = YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)


@pytest.mark.parametrize(
    ("classes", "cls", "expected"),
    [
        # No typed classes: the model's own vocabulary names the regions.
        ("", [1.0, 0.0], ["bicycle", "person"]),
        # Typed classes: the region carries the typed name, so inline
        # [CLASS=hand] blocks and [SKIP] match it.
        ("face, hand", [1.0], ["hand"]),
    ],
)
def test_world_regions_are_named_after_their_classes(
    tmp_path, monkeypatch, classes, cls, expected
):
    _fake_yolo(monkeypatch, cls, {0: "person", 1: "bicycle"})
    model = tmp_path / "yolov8x-worldv2.pt"

    result = ultralytics_predict(str(model), Image.new("RGB", (4, 4)), classes=classes)

    assert result.class_names == expected


def test_looked_up_class_names_still_win_over_the_result(tmp_path, monkeypatch):
    model = tmp_path / "multi.pt"
    model.write_bytes(b"x")
    (tmp_path / "multi.names.json").write_text('["face", "hand"]', encoding="utf-8")
    _fake_yolo(monkeypatch, [1.0, 5.0], {0: "other", 1: "thing"})
    get_model_class_names.cache_clear()
    try:
        result = ultralytics_predict(str(model), Image.new("RGB", (4, 4)))
    finally:
        get_model_class_names.cache_clear()

    # An id outside the looked-up names stays numeric, as before.
    assert result.class_names == ["hand", "5"]


def test_class_filter_names_match_regardless_of_case(tmp_path, monkeypatch, capsys):
    # "Face" from an API call or an old preset used to match no class, so the
    # include filter was dropped and every class was inpainted.
    model = tmp_path / "multi.pt"
    model.write_bytes(b"x")
    (tmp_path / "multi.names.json").write_text('["face", "hand"]', encoding="utf-8")
    calls = []
    module = ModuleType("ultralytics")

    class YOLO:
        def __init__(self, path):
            self.names = {0: "face", 1: "hand"}

        def __call__(self, image, **kwargs):
            calls.append(kwargs)
            wanted = kwargs.get("classes")
            cls = [float(c) for c in (0, 1) if wanted is None or c in wanted]
            return [_FakeResult(self.names, cls)]

    module.YOLO = YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    image = Image.new("RGB", (4, 4))
    get_model_class_names.cache_clear()
    try:
        include = ultralytics_predict(str(model), image, classes="Face")
        exclude = ultralytics_predict(str(model), image, exclude_classes="HAND")
        unknown = ultralytics_predict(str(model), image, classes="hands")
    finally:
        get_model_class_names.cache_clear()

    assert calls[0]["classes"] == [0]
    assert include.class_names == ["face"]
    assert exclude.class_names == ["face"]
    # A name the model does not have is still dropped (every class is then
    # detected, as before), but the console now says so.
    assert "classes" not in calls[2]
    assert unknown.class_names == ["face", "hand"]
    assert "hands" in capsys.readouterr().out


@pytest.mark.parametrize(
    "token", ["²", "①", "1" * 5000], ids=["superscript", "circled", "bad-number"]
)
def test_a_digit_that_int_rejects_does_not_stop_the_class_filter(
    tmp_path, monkeypatch, token
):
    # A class with a digit that int() rejects, or a number that cannot be
    # converted, in the include or the NOT filter is dropped as an unknown
    # class.
    model = tmp_path / "multi.pt"
    model.write_bytes(b"x")
    (tmp_path / "multi.names.json").write_text('["face", "hand"]', encoding="utf-8")
    _fake_yolo(monkeypatch, [0.0, 1.0], {0: "face", 1: "hand"})
    image = Image.new("RGB", (4, 4))
    get_model_class_names.cache_clear()
    try:
        include = ultralytics_predict(str(model), image, classes="face," + token)
        exclude = ultralytics_predict(str(model), image, exclude_classes=token)
    finally:
        get_model_class_names.cache_clear()

    assert include.class_names == ["face", "hand"]
    assert exclude.class_names == ["face", "hand"]


_NEWER_LAYER = (
    "Can't get attribute 'C3k2' on <module 'ultralytics.nn.modules.block'"
    " from 'site-packages/ultralytics/nn/modules/block.py'>"
)


def _yolo_that_fails(monkeypatch, error):
    """Fake Ultralytics whose model load raises `error`, as torch.load does
    when the checkpoint needs a layer class this Ultralytics does not have."""
    module = ModuleType("ultralytics")

    class YOLO:
        def __init__(self, path):
            raise error

    module.YOLO = YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    import importlib.metadata

    real_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "8.3.75" if name == "ultralytics" else real_version(name),
    )


def _load_error_from_a_wrapper():
    try:
        raise AttributeError(
            "Can't get attribute 'Segment26' on <module 'ultralytics.nn.modules.head'>"
        )
    except AttributeError as inner:
        try:
            raise RuntimeError("model load failed") from inner
        except RuntimeError as outer:
            return outer


@pytest.mark.parametrize(
    "error",
    [AttributeError(_NEWER_LAYER), _load_error_from_a_wrapper()],
    ids=["unpickle-error", "wrapped"],
)
def test_detector_from_a_newer_ultralytics_gets_a_clear_message(
    tmp_path, monkeypatch, capsys, error
):
    # A YOLO11 / YOLO26 detector loaded by an older Ultralytics used to fail
    # with only "Can't get attribute 'C3k2' on <module ...>".
    from adetailer.ultralytics import UltralyticsTooOldError

    _yolo_that_fails(monkeypatch, error)
    model = tmp_path / "face_yolo26n-ü.pt"

    with pytest.raises(UltralyticsTooOldError) as info:
        ultralytics_predict(str(model), Image.new("RGB", (4, 4)))

    msg = str(info.value)
    assert "face_yolo26n-" in msg
    assert "newer Ultralytics than Ultralytics 8.3.75" in msg
    assert "Can't get attribute" in msg
    assert "site-packages" not in msg
    # Still an AttributeError, chained to the original error.
    assert isinstance(info.value, AttributeError)
    assert info.value.__cause__ is error
    err = capsys.readouterr().err
    assert "newer Ultralytics than Ultralytics 8.3.75" in err
    assert "face_yolo26n-\\xfc.pt" in err
    assert err.isascii()


@pytest.mark.parametrize(
    "error",
    [
        AttributeError("'NoneType' object has no attribute 'names'"),
        AttributeError("Can't get attribute 'Thing' on <module 'otherlib.layers'>"),
        FileNotFoundError("face.pt does not exist"),
    ],
    ids=["other-attribute-error", "other-library", "other-error"],
)
def test_other_detector_load_errors_are_raised_unchanged(
    tmp_path, monkeypatch, capsys, error
):
    _yolo_that_fails(monkeypatch, error)

    with pytest.raises(type(error)) as info:
        ultralytics_predict(str(tmp_path / "face.pt"), Image.new("RGB", (4, 4)))

    assert info.value is error
    assert "newer Ultralytics" not in capsys.readouterr().err


class _NoBoxesResult:
    """What Ultralytics returns for a model that is not a detector: no boxes,
    and the field of its own task instead (`semantic_mask` only exists in a
    recent Ultralytics)."""

    def __init__(self, **fields):
        self.names = {0: "plane"}
        self.boxes = None
        self.masks = None
        self.probs = None
        self.obb = None
        for name, value in fields.items():
            setattr(self, name, value)

    def plot(self, **_kwargs):
        msg = "a result without boxes has no preview to draw"
        raise AssertionError(msg)


def _fake_task_model(monkeypatch, result, **attrs):
    """Fake Ultralytics model with the given attributes (`task`, as the real
    one sets from the loaded file) that returns `result`."""
    calls = []
    module = ModuleType("ultralytics")

    class YOLO:
        def __init__(self, path):
            self.names = {0: "plane"}
            for name, value in attrs.items():
                setattr(self, name, value)

        def __call__(self, image, **kwargs):
            calls.append(kwargs)
            return [result]

    module.YOLO = YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return calls


@pytest.mark.parametrize(
    ("task", "kind"),
    [
        ("obb", "an OBB model (rotated boxes)"),
        ("classify", "a classification model"),
        ("semantic", "a semantic segmentation model"),
    ],
)
def test_a_model_that_is_not_a_detector_detects_nothing_and_says_why(
    tmp_path, monkeypatch, capsys, task, kind
):
    # An OBB or classification model as the detector ended in "'NoneType'
    # object has no attribute 'xyxy'": its results have no boxes.
    calls = _fake_task_model(monkeypatch, _NoBoxesResult(), task=task)
    model = tmp_path / "planes-ü.pt"

    result = ultralytics_predict(str(model), Image.new("RGB", (4, 4)))

    assert result.bboxes == []
    assert result.masks == []
    assert result.preview is None
    assert "'planes-ü.pt'" in result.note
    assert f"is {kind}" in result.note
    assert "needs a detection or segmentation model" in result.note
    # Only the name of the file, never its folder.
    assert tmp_path.name not in result.note
    # Told from the loaded model: nothing is detected with it.
    assert calls == []
    # The caller says it, once: nothing is printed here.
    said = capsys.readouterr()
    assert said.out == ""
    assert said.err == ""


@pytest.mark.parametrize(
    ("fields", "kind"),
    [
        ({"obb": object()}, "an OBB model (rotated boxes)"),
        ({"probs": object()}, "a classification model"),
        ({"semantic_mask": object()}, "a semantic segmentation model"),
        ({}, "a model whose results have no boxes"),
    ],
    ids=["obb", "classify", "semantic", "unknown"],
)
@pytest.mark.parametrize("attrs", [{}, {"task": None}, {"task": "detect"}])
def test_a_result_without_boxes_is_not_read_as_detections(
    tmp_path, monkeypatch, fields, kind, attrs
):
    # The same when the model's task did not tell (none, or a wrong one).
    calls = _fake_task_model(monkeypatch, _NoBoxesResult(**fields), **attrs)

    result = ultralytics_predict(str(tmp_path / "planes.pt"), Image.new("RGB", (4, 4)))

    assert len(calls) == 1
    assert result.bboxes == []
    assert result.preview is None
    assert f"'planes.pt' is {kind}" in result.note
    assert "needs a detection or segmentation model" in result.note


@pytest.mark.parametrize("task", ["detect", "segment", "pose", None, "new-task"])
def test_detection_segmentation_and_pose_models_detect_as_before(
    tmp_path, monkeypatch, task
):
    # Their results have boxes (a pose model adds keypoints to them): no note.
    names = {0: "face"}
    calls = _fake_task_model(monkeypatch, _FakeResult(names, [0.0]), task=task)

    result = ultralytics_predict(str(tmp_path / "face.pt"), Image.new("RGB", (4, 4)))

    assert calls == [{"conf": 0.3, "device": ""}]
    assert result.bboxes == [[1.0, 1.0, 3.0, 3.0]]
    assert result.preview is not None
    assert result.note == ""


_MISSING_ULTRALYTICS = (
    "[-] ADetailer: the Python package 'ultralytics' cannot be imported ({}), so "
    "the YOLO detectors cannot run. Start the WebUI once without --skip-install, "
    "or install it with the WebUI's Python: python -m pip install {}"
)


@pytest.mark.parametrize(
    ("numpy_version", "specs"),
    [
        # AUTOMATIC1111: a plain install of Ultralytics can pull NumPy 2 in.
        ("1.26.4", '"ultralytics>=8.3.75" "numpy<2"'),
        ("2.1.3", '"ultralytics>=8.3.75"'),
        (None, '"ultralytics>=8.3.75"'),
    ],
    ids=["numpy-1", "numpy-2", "no-numpy"],
)
def test_missing_ultralytics_line_gives_the_pip_command_of_install_py(
    monkeypatch, numpy_version, specs
):
    from adetailer.ultralytics import missing_ultralytics_line
    from tests.test_pipeline import _load_install

    if numpy_version is None:
        monkeypatch.setitem(sys.modules, "numpy", None)
    else:
        monkeypatch.setattr(np, "__version__", numpy_version)

    line = missing_ultralytics_line()

    assert line == _MISSING_ULTRALYTICS.format("it is not installed", specs)
    assert line.isascii()
    # The same packages and versions install.py asks pip for.
    install = _load_install(monkeypatch)
    calls = []
    monkeypatch.setattr(install, "run_pip", lambda *args: calls.append(args))
    monkeypatch.setattr(install, "is_installed", lambda *a: False)
    install.install()
    wanted = [arg for arg in calls[0] if not arg.startswith("rich")]
    assert specs == " ".join(f'"{arg}"' for arg in wanted)


@pytest.mark.parametrize(
    ("found", "said"),
    [(None, True), (object(), False), (ValueError("no spec"), False)],
    ids=["not-installed", "installed", "cannot-tell"],
)
def test_startup_says_when_ultralytics_is_not_installed(
    monkeypatch, capsys, found, said
):
    # Ultralytics is imported at the first detection only: a WebUI started
    # with --skip-install loaded ADetailer without a word and failed at the
    # first generation. Looked up without importing it.
    import importlib.util

    from adetailer.ultralytics import warn_missing_ultralytics

    asked = []

    def find_spec(name):
        asked.append(name)
        if isinstance(found, Exception):
            raise found
        return found

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(np, "__version__", "2.1.3")

    assert warn_missing_ultralytics() is said

    assert asked == ["ultralytics"]
    captured = capsys.readouterr()
    assert captured.out == ""
    expected = _MISSING_ULTRALYTICS.format(
        "it is not installed", '"ultralytics>=8.3.75"'
    )
    assert captured.err == (expected + "\n" if said else "")


def _webui_job(monkeypatch, stamp):
    state = SimpleNamespace(
        job="txt2img", job_count=1, time_start=100.0, job_timestamp=stamp
    )
    monkeypatch.setitem(sys.modules, "modules.shared", SimpleNamespace(state=state))
    return state


def test_detection_without_ultralytics_names_the_package_and_the_pip_command(
    tmp_path, monkeypatch, capsys
):
    # "No module named 'ultralytics'" came alone, in a long error report, for
    # every image. One clear line first, once per WebUI job (each time in the
    # Detection preview, which begins none); the error itself is unchanged.
    import adetailer.ultralytics as ultra

    monkeypatch.setitem(sys.modules, "ultralytics", None)
    monkeypatch.setattr(ultra, "_MISSING_SAID_JOB", None)
    monkeypatch.setattr(np, "__version__", "1.26.4")
    image = Image.new("RGB", (4, 4))

    def detect():
        with pytest.raises(ModuleNotFoundError) as info:
            ultralytics_predict(str(tmp_path / "face.pt"), image)
        assert info.value.name == "ultralytics"
        return capsys.readouterr().err, str(info.value)

    state = _webui_job(monkeypatch, "20261114120000")
    err, error = detect()
    assert err == (
        _MISSING_ULTRALYTICS.format(error, '"ultralytics>=8.3.75" "numpy<2"') + "\n"
    )
    assert err.isascii()
    # The next images and tabs of the same job.
    assert detect()[0] == ""
    assert detect()[0] == ""
    # The next job says it again.
    state.job_timestamp = "20261114120500"
    assert detect()[0].count("'ultralytics' cannot be imported") == 1
    assert detect()[0] == ""
    # No job (the Detection preview): each time.
    state.job, state.job_count = "", 0
    assert detect()[0].count("'ultralytics' cannot be imported") == 1
    assert detect()[0].count("'ultralytics' cannot be imported") == 1
    # And without a WebUI.
    monkeypatch.delitem(sys.modules, "modules.shared")
    assert detect()[0].count("'ultralytics' cannot be imported") == 1


def test_a_missing_package_of_ultralytics_is_named_in_the_line(
    tmp_path, monkeypatch, capsys
):
    # Ultralytics is there but a package it needs is not: the same command
    # installs it.
    import adetailer.ultralytics as ultra

    module = ModuleType("ultralytics")

    def lazy(name):
        raise ModuleNotFoundError("No module named 'polars'", name="polars")

    module.__getattr__ = lazy
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    monkeypatch.setattr(ultra, "_MISSING_SAID_JOB", None)
    monkeypatch.setattr(np, "__version__", "2.1.3")

    with pytest.raises(ModuleNotFoundError, match="polars"):
        ultralytics_predict(str(tmp_path / "face.pt"), Image.new("RGB", (4, 4)))

    assert capsys.readouterr().err == (
        _MISSING_ULTRALYTICS.format(
            "No module named 'polars'", '"ultralytics>=8.3.75"'
        )
        + "\n"
    )


def test_another_import_error_of_ultralytics_is_raised_without_the_line(
    tmp_path, monkeypatch, capsys
):
    # An Ultralytics that is installed but broken in another way: installing
    # it again is not known to help, so no command is made up.
    monkeypatch.setitem(sys.modules, "ultralytics", ModuleType("ultralytics"))

    with pytest.raises(ImportError, match="cannot import name 'YOLO'"):
        ultralytics_predict(str(tmp_path / "face.pt"), Image.new("RGB", (4, 4)))

    assert capsys.readouterr().err == ""


class TestMaskToPil:
    @pytest.mark.parametrize(
        ("content_size", "padding", "output_size"),
        [
            ((8, 6), (0, 0, 1, 1), (16, 12)),  # landscape
            ((6, 8), (1, 1, 0, 0), (12, 16)),  # portrait
            ((8, 5), (0, 0, 1, 2), (16, 10)),  # odd vertical padding
            ((5, 8), (1, 2, 0, 0), (10, 16)),  # odd horizontal padding
            ((8, 8), (0, 0, 0, 0), (16, 16)),  # square, no padding
            ((8, 6), (0, 0, 0, 0), (16, 12)),  # rectangular, no padding
            ((8, 6), (0, 0, 0, 0), (8, 6)),  # native-resolution mask
        ],
    )
    def test_mask_to_pil_preserves_regions_after_letterbox(
        self, content_size, padding, output_size
    ):
        width, height = content_size
        original = torch.zeros((2, height, width), dtype=torch.float32)
        original[0, :2, :3] = 1
        original[1, -2:, -3:] = 1
        letterboxed = torch.nn.functional.pad(original, padding)

        results = mask_to_pil(letterboxed, output_size)

        assert len(results) == 2
        for content, result in zip(original, results):
            expected = Image.fromarray((content.numpy() * 255).astype(np.uint8))
            expected = expected.resize(output_size)
            np.testing.assert_array_equal(np.array(result), np.array(expected))

    def test_mask_to_pil_float32(self):
        mask = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]], dtype=torch.float32)
        imgs = mask_to_pil(mask, shape=(2, 2))

        assert len(imgs) == 1
        img = imgs[0]
        assert isinstance(img, Image.Image)

        arr = np.array(img)
        assert arr.shape == (2, 2)
        assert arr.dtype == np.uint8

        expected = np.array([[0, 255], [0, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(arr, expected)

    def test_mask_to_pil_uint8(self):
        mask = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.uint8)
        imgs = mask_to_pil(mask, shape=(2, 2))

        assert len(imgs) == 1
        img = imgs[0]
        assert isinstance(img, Image.Image)

        arr = np.array(img)
        assert arr.shape == (2, 2)
        assert arr.dtype == np.uint8

        expected = np.array([[0, 255], [0, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(arr, expected)


class _FakeEvents:
    """Stand-in for Ultralytics' usage-analytics object (`events`)."""

    def __init__(self):
        self.enabled = True


def _fake_events_module(monkeypatch, name, events):
    module = ModuleType(name)
    if events is not None:
        module.events = events
    monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize(
    "layout",
    ["ultralytics.utils.events", "ultralytics.hub.utils"],
    ids=["new-layout", "old-layout"],
)
def test_detection_switches_off_ultralytics_usage_analytics(
    tmp_path, monkeypatch, layout
):
    # Ultralytics sends anonymous usage analytics on every predict while its
    # `sync` setting is on (the default). ADetailer switches the in-memory
    # object off, in both module layouts, and never writes its settings.
    _fake_yolo(monkeypatch, [0.0], {0: "face"})
    settings = {"sync": True}
    sys.modules["ultralytics"].SETTINGS = settings
    events = _FakeEvents()
    if layout == "ultralytics.hub.utils":
        # Older Ultralytics: no ultralytics.utils.events module at all.
        monkeypatch.setitem(sys.modules, "ultralytics.utils.events", None)
    else:
        _fake_events_module(monkeypatch, "ultralytics.hub.utils", None)
    _fake_events_module(monkeypatch, layout, events)

    result = ultralytics_predict(str(tmp_path / "face.pt"), Image.new("RGB", (4, 4)))

    assert events.enabled is False
    assert settings == {"sync": True}
    assert len(result.bboxes) == 1


@pytest.mark.parametrize("case", ["no-module", "no-attribute", "broken-object"])
def test_detection_works_when_ultralytics_analytics_cannot_be_found(
    tmp_path, monkeypatch, case
):
    _fake_yolo(monkeypatch, [0.0], {0: "face"})
    if case == "no-module":
        monkeypatch.setitem(sys.modules, "ultralytics.utils.events", None)
        monkeypatch.setitem(sys.modules, "ultralytics.hub.utils", None)
    elif case == "no-attribute":
        _fake_events_module(monkeypatch, "ultralytics.utils.events", object())
        _fake_events_module(monkeypatch, "ultralytics.hub.utils", None)
    else:

        class _ReadOnly:
            enabled = True

            def __setattr__(self, key, value):
                raise RuntimeError("read-only")

        _fake_events_module(monkeypatch, "ultralytics.utils.events", _ReadOnly())
        _fake_events_module(monkeypatch, "ultralytics.hub.utils", None)

    result = ultralytics_predict(str(tmp_path / "face.pt"), Image.new("RGB", (4, 4)))

    assert len(result.bboxes) == 1


class _PlotResult(_FakeResult):
    """A result whose plot() takes Ultralytics' labels/conf options."""

    def __init__(self, names, cls, calls):
        super().__init__(names, cls)
        self.calls = calls

    def plot(self, **kwargs):
        self.calls.append(kwargs)
        img = np.zeros((4, 4, 3), np.uint8)
        if kwargs.get("labels", True):
            img[0, 0] = 255  # stands for the text of a label
        return img


@pytest.mark.parametrize("plain", [True, False])
def test_plain_preview_is_plotted_without_labels(tmp_path, monkeypatch, plain):
    # The Detection preview's "Show labels" switch needs the same picture
    # without class names and scores; only the preview asks for it.
    calls = []
    module = ModuleType("ultralytics")

    class YOLO:
        def __init__(self, path):
            self.names = {0: "face"}

        def __call__(self, image, **kwargs):
            return [_PlotResult(self.names, [0.0], calls)]

    module.YOLO = YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    kwargs = {"plain_preview": True} if plain else {}

    result = ultralytics_predict(str(tmp_path / "face.pt"), Image.new("RGB", (4, 4)), **kwargs)

    assert result.preview.getpixel((0, 0)) == (255, 255, 255)
    if plain:
        assert calls == [{}, {"labels": False, "conf": False}]
        assert result.preview_plain.getpixel((0, 0)) == (0, 0, 0)
        assert result.preview_plain.size == result.preview.size
    else:
        assert calls == [{}]
        assert result.preview_plain is None


def test_plain_preview_is_skipped_by_an_old_plot(tmp_path, monkeypatch):
    # An Ultralytics whose plot() has no labels/conf options: no plain
    # picture, and the detection still works.
    _fake_yolo(monkeypatch, [0.0], {0: "face"})
    result = ultralytics_predict(
        str(tmp_path / "face.pt"), Image.new("RGB", (4, 4)), plain_preview=True
    )
    assert result.bboxes and result.preview is not None
    assert result.preview_plain is None
