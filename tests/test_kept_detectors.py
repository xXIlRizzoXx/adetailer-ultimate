"""The detector stays loaded while one WebUI job runs (keep_loaded=True).

A fake Ultralytics and a fake WebUI job state: no real model is loaded."""

from __future__ import annotations

import ast
import os
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import adetailer.ultralytics as ultra
from adetailer.classes import get_model_class_names
from adetailer.ultralytics import release_kept_detectors, ultralytics_predict

_ROOT = Path(__file__).resolve().parent.parent


class _Boxes:
    def __init__(self):
        self.xyxy = torch.tensor([[1.0, 1.0, 3.0, 3.0]])
        self.conf = torch.tensor([0.9])
        self.cls = torch.tensor([0.0])

    def __len__(self):
        return 1


class _Result:
    def __init__(self, names):
        self.names = names
        self.boxes = _Boxes()
        self.masks = None

    def plot(self, **_kwargs):
        return np.zeros((4, 4, 3), np.uint8)


def _fake_ultralytics(monkeypatch, *, own_imgsz=640, fail_calls=0):
    """Count the loads (YOLO(path)) and record each call's arguments."""
    loads: list[str] = []
    calls: list[dict] = []
    failing = [fail_calls]
    module = ModuleType("ultralytics")

    class YOLO:
        def __init__(self, path):
            loads.append(Path(str(path)).name)
            self.names = {0: "face", 1: "hand"}
            self.predictor = object()
            self.overrides = {"imgsz": own_imgsz} if own_imgsz else {}

        def set_classes(self, classes):
            self.names = list(classes)
            calls.append({"set_classes": list(classes)})

        def __call__(self, image, **kwargs):
            calls.append(kwargs)
            if failing[0]:
                failing[0] -= 1
                raise RuntimeError("out of memory")
            return [_Result(self.names)]

    module.YOLO = YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return loads, calls


def _model(tmp_path, name="face.pt", data=b"weights"):
    path = tmp_path / name
    path.write_bytes(data)
    # Class names from a sidecar, so no extra YOLO() load looks them up.
    path.with_name(path.stem + ".names.json").write_text(
        '["face", "hand"]', encoding="utf-8"
    )
    return path


@pytest.fixture
def job(monkeypatch):
    """A running WebUI job; the watcher thread is recorded, not started."""
    state = SimpleNamespace(
        job="txt2img", job_count=1, time_start=100.0, job_timestamp="20261111120000"
    )
    monkeypatch.setitem(sys.modules, "modules.shared", SimpleNamespace(state=state))
    devices = SimpleNamespace(gc=[])
    devices.torch_gc = lambda: devices.gc.append(1)
    monkeypatch.setitem(sys.modules, "modules.devices", devices)
    watched: list[tuple] = []
    monkeypatch.setattr(ultra, "_start_watch", watched.append)
    get_model_class_names.cache_clear()
    release_kept_detectors()
    yield SimpleNamespace(state=state, watched=watched, devices=devices)
    release_kept_detectors()
    get_model_class_names.cache_clear()


def _predict(path, **kwargs):
    kwargs.setdefault("keep_loaded", True)
    return ultralytics_predict(str(path), Image.new("RGB", (4, 4)), **kwargs)


def test_two_detections_of_one_job_load_the_detector_once(tmp_path, monkeypatch, job):
    loads, _calls = _fake_ultralytics(monkeypatch)
    face = _model(tmp_path)

    first = _predict(face)
    second = _predict(face)

    assert loads == ["face.pt"]
    assert first.bboxes and second.bboxes
    # One watcher for the job, which drops the detector when the job ends.
    assert len(job.watched) == 1


def test_the_next_job_loads_the_detector_again(tmp_path, monkeypatch, job):
    loads, _calls = _fake_ultralytics(monkeypatch)
    face = _model(tmp_path)

    _predict(face)
    job.state.time_start = 200.0  # a new click on Generate
    job.state.job_timestamp = "20261111120500"
    _predict(face)

    assert loads == ["face.pt", "face.pt"]


@pytest.mark.parametrize("change", ["size", "time"])
def test_a_file_replaced_under_the_same_name_is_loaded_again(
    tmp_path, monkeypatch, job, change
):
    loads, _calls = _fake_ultralytics(monkeypatch)
    face = _model(tmp_path)
    _predict(face)

    if change == "size":
        face.write_bytes(b"other, longer weights")
    else:
        info = face.stat()
        os.utime(face, ns=(info.st_atime_ns, info.st_mtime_ns + 5_000_000_000))
    _predict(face)

    assert loads == ["face.pt", "face.pt"]


def test_a_tab_without_class_filter_after_one_with_it_passes_no_classes(
    tmp_path, monkeypatch, job
):
    # Ultralytics' reused predictor keeps every argument not passed again:
    # without classes=None the second tab would find only hands.
    _loads, calls = _fake_ultralytics(monkeypatch)
    model = _model(tmp_path, "multi.pt")

    _predict(model, classes="hand")
    _predict(model)

    assert calls[0]["classes"] == [1]
    assert "classes" in calls[1]
    assert calls[1]["classes"] is None


def test_the_detection_size_is_always_passed_to_a_kept_detector(
    tmp_path, monkeypatch, job
):
    _loads, calls = _fake_ultralytics(monkeypatch, own_imgsz=640)
    face = _model(tmp_path)

    _predict(face, imgsz=1024)
    _predict(face)  # 0 = the detector's own size

    assert calls[0]["imgsz"] == 1024
    assert calls[1]["imgsz"] == 640
    assert calls[1]["conf"] == 0.3


def test_a_detector_without_its_own_size_is_not_kept(tmp_path, monkeypatch, job):
    # Its size could not be passed again, so it is loaded every time as before.
    loads, calls = _fake_ultralytics(monkeypatch, own_imgsz=None)
    face = _model(tmp_path)

    _predict(face)
    _predict(face)

    assert loads == ["face.pt", "face.pt"]
    assert "imgsz" not in calls[1]
    assert "classes" not in calls[1]


def test_an_error_while_detecting_still_releases_at_the_end_of_the_job(
    tmp_path, monkeypatch, job
):
    loads, _calls = _fake_ultralytics(monkeypatch, fail_calls=1)
    face = _model(tmp_path)

    with pytest.raises(RuntimeError):
        _predict(face)
    owner = job.watched[0]
    # Still running: the watcher keeps it.
    assert ultra._job_watch_step(owner) is False
    assert ultra._KEPT

    # The WebUI ends the job (also after an error, Interrupt or Skip).
    job.state.job = ""
    job.state.job_count = 0
    assert ultra._job_watch_step(owner) is True

    assert not ultra._KEPT
    assert ultra._KEPT_JOB is None
    assert job.devices.gc == [1]
    _predict(face)
    assert loads == ["face.pt", "face.pt"]


def test_the_watcher_releases_when_a_new_job_has_already_begun(
    tmp_path, monkeypatch, job
):
    _fake_ultralytics(monkeypatch)
    _predict(_model(tmp_path))
    owner = job.watched[0]

    job.state.time_start = 300.0
    assert ultra._job_watch_step(owner) is True
    assert not ultra._KEPT


def test_the_watcher_thread_drops_the_detectors_when_the_job_ends(
    tmp_path, monkeypatch
):
    # The real thread, without the job fixture that replaces it.
    state = SimpleNamespace(
        job="txt2img", job_count=1, time_start=400.0, job_timestamp="20261111121000"
    )
    monkeypatch.setitem(sys.modules, "modules.shared", SimpleNamespace(state=state))
    monkeypatch.delitem(sys.modules, "modules.devices", raising=False)
    monkeypatch.setattr(ultra, "_WATCH_EVERY", 0.01)
    started = []
    real_start = ultra._start_watch

    def start(owner):
        real_start(owner)
        started.extend(
            t for t in threading.enumerate() if t.name == "ADetailer-detectors"
        )

    monkeypatch.setattr(ultra, "_start_watch", start)
    release_kept_detectors()
    get_model_class_names.cache_clear()
    _fake_ultralytics(monkeypatch)
    try:
        _predict(_model(tmp_path))
        assert ultra._KEPT
        assert started

        state.job, state.job_count = "", 0
        for thread in started:
            thread.join(timeout=10)

        assert not ultra._KEPT
        assert not any(t.is_alive() for t in started)
    finally:
        state.job, state.job_count = "", 0
        release_kept_detectors()
        get_model_class_names.cache_clear()


def test_world_detectors_are_never_kept(tmp_path, monkeypatch, job):
    # set_classes changes the model itself: each detection loads it anew.
    loads, calls = _fake_ultralytics(monkeypatch)
    world = tmp_path / "yolov8s-worldv2.pt"
    world.write_bytes(b"weights")

    _predict(world, classes="face")
    _predict(world)

    assert loads == ["yolov8s-worldv2.pt", "yolov8s-worldv2.pt"]
    assert calls[0] == {"set_classes": ["face"]}
    assert "classes" not in calls[-1]
    assert not ultra._KEPT


def test_yoloe_detectors_are_never_kept_but_prompt_free_ones_are(
    tmp_path, monkeypatch, job
):
    # Text prompts change a YOLOE model like set_classes changes a YOLO-World
    # one; a prompt-free YOLOE keeps its own classes and is kept as usual.
    loads, calls = _fake_ultralytics(monkeypatch)
    yolo = sys.modules["ultralytics"].YOLO
    yolo.get_text_pe = lambda self, names: f"pe:{','.join(names)}"
    yolo.set_classes = lambda self, classes, embeddings=None: calls.append(
        {"set_classes": list(classes), "embeddings": embeddings}
    )
    yoloe = _model(tmp_path, "yoloe-11s-seg.pt")
    pf = _model(tmp_path, "yoloe-11s-seg-pf.pt")

    _predict(yoloe, classes="face")
    _predict(yoloe, classes="hand")
    _predict(pf)
    _predict(pf)

    assert loads == ["yoloe-11s-seg.pt", "yoloe-11s-seg.pt", "yoloe-11s-seg-pf.pt"]
    assert calls[0] == {"set_classes": ["face"], "embeddings": "pe:face"}
    assert list(ultra._KEPT) and all("yoloe-11s-seg-pf.pt" in k[0] for k in ultra._KEPT)


def test_without_keep_loaded_the_detector_is_loaded_each_time(
    tmp_path, monkeypatch, job
):
    # The Detection preview, which does not pass it: as before, also inside
    # a job.
    loads, calls = _fake_ultralytics(monkeypatch)
    face = _model(tmp_path)

    ultralytics_predict(str(face), Image.new("RGB", (4, 4)))
    ultralytics_predict(str(face), Image.new("RGB", (4, 4)))

    assert loads == ["face.pt", "face.pt"]
    assert "classes" not in calls[1]
    assert not ultra._KEPT


@pytest.mark.parametrize("state", ["no webui", "no job"])
def test_outside_a_job_nothing_is_kept(tmp_path, monkeypatch, job, state):
    loads, _calls = _fake_ultralytics(monkeypatch)
    if state == "no webui":
        monkeypatch.delitem(sys.modules, "modules.shared")
    else:
        job.state.job, job.state.job_count = "", 0
    face = _model(tmp_path)

    _predict(face)
    _predict(face)

    assert loads == ["face.pt", "face.pt"]
    assert not job.watched


def test_another_thread_in_the_same_job_does_not_share_them(
    tmp_path, monkeypatch, job
):
    loads, _calls = _fake_ultralytics(monkeypatch)
    face = _model(tmp_path)
    _predict(face)

    other = threading.Thread(target=_predict, args=(face,))
    other.start()
    other.join()
    _predict(face)

    # The other thread loaded its own; the job's thread still has its own.
    assert loads == ["face.pt", "face.pt"]


def test_only_the_most_recent_detectors_are_kept(tmp_path, monkeypatch, job):
    loads, _calls = _fake_ultralytics(monkeypatch)
    paths = [_model(tmp_path, f"d{n}.pt") for n in range(ultra._KEPT_MAX + 1)]
    for path in paths:
        _predict(path)

    _predict(paths[-1])  # still kept
    _predict(paths[0])  # dropped as the least recently used

    assert len(loads) == len(paths) + 1
    assert len(ultra._KEPT) == ultra._KEPT_MAX


def test_a_failed_load_is_not_kept(tmp_path, monkeypatch, job):
    loads, _calls = _fake_ultralytics(monkeypatch)
    module = sys.modules["ultralytics"]
    real = module.YOLO
    attempts = []

    def flaky(path):
        attempts.append(path)
        if len(attempts) == 1:
            raise OSError("file busy")
        return real(path)

    module.YOLO = flaky
    face = _model(tmp_path)

    with pytest.raises(OSError):
        _predict(face)
    _predict(face)
    _predict(face)

    assert loads == ["face.pt"]


def test_a_model_that_is_not_a_detector_is_read_once_per_job_too(
    tmp_path, monkeypatch, job
):
    # An OBB model picked as the detector is refused from its task, for every
    # image of the job, without reading its file again and without detecting;
    # an ordinary detector next to it is kept and used as before.
    loads, calls = _fake_ultralytics(monkeypatch)
    yolo = sys.modules["ultralytics"].YOLO
    real_init = yolo.__init__

    def init(self, path):
        real_init(self, path)
        if "obb" in Path(str(path)).name:
            self.task = "obb"

    yolo.__init__ = init
    obb = _model(tmp_path, "planes-obb.pt")
    face = _model(tmp_path)

    notes = [_predict(obb).note, _predict(face).note, _predict(obb).note]

    assert loads == ["planes-obb.pt", "face.pt"]
    assert len(calls) == 1
    assert notes[0] == notes[2]
    assert "is an OBB model" in notes[0]
    assert notes[1] == ""
    assert len(ultra._KEPT) == 2
    assert len(job.watched) == 1


def test_the_standalone_run_releases_them_before_ending_its_job(monkeypatch):
    from aaaaaa.jobs import wrap_adetailer_job

    order = []
    state = SimpleNamespace(
        begin=lambda job=None: order.append("begin"),
        end=lambda: order.append("end"),
        skipped=False, interrupted=False, stopping_generation=False,
    )
    shared = ModuleType("modules.shared")
    shared.state = state
    call_queue = ModuleType("modules.call_queue")
    call_queue.queue_lock = threading.Lock()
    monkeypatch.setitem(sys.modules, "modules.shared", shared)
    monkeypatch.setitem(sys.modules, "modules.call_queue", call_queue)
    modules = ModuleType("modules")
    modules.shared = shared
    modules.call_queue = call_queue
    monkeypatch.setitem(sys.modules, "modules", modules)
    monkeypatch.setattr(
        ultra, "release_kept_detectors", lambda *a: order.append("release")
    )

    def fails():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        wrap_adetailer_job(fails)()

    assert order == ["begin", "release", "end"]


def _predict_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", "")) == "ultralytics_predict"
    ]


def test_only_the_generation_pass_keeps_its_detector():
    def keep(call):
        for kw in call.keywords:
            if kw.arg == "keep_loaded":
                return getattr(kw.value, "value", None)
        return False

    script = _predict_calls(_ROOT / "scripts" / "!adetailer.py")
    preview = _predict_calls(_ROOT / "aaaaaa" / "ui.py")

    assert [keep(call) for call in script] == [True]
    assert preview and all(keep(call) is False for call in preview)
