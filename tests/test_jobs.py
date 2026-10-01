"""Standalone action lifecycle with synthetic host state; no WebUI/GPU needed."""

from __future__ import annotations

import ast
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Event, Lock
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

from aaaaaa.jobs import wrap_adetailer_detection, wrap_adetailer_job

_UI_PATH = Path(__file__).resolve().parents[1] / "aaaaaa" / "ui.py"


class _ObservedLock:
    def __init__(self):
        self.lock = Lock()
        self.attempted = Event()

    def __enter__(self):
        self.attempted.set()
        self.lock.acquire()

    def __exit__(self, *_exc):
        self.lock.release()


@pytest.fixture
def host(monkeypatch):
    lock = _ObservedLock()
    events = []

    class State:
        skipped = True
        interrupted = True
        stopping_generation = True
        job = "previous job"
        job_count = 9

        def begin(self, job):
            assert lock.lock.locked()
            events.append("begin")
            self.skipped = self.interrupted = self.stopping_generation = False
            self.job = job
            self.job_count = -1

        def end(self):
            assert lock.lock.locked()
            events.append("end")
            self.job = ""
            self.job_count = 0

    state = State()
    modules = ModuleType("modules")
    shared = ModuleType("modules.shared")
    shared.state = state
    queue = ModuleType("modules.call_queue")
    queue.queue_lock = lock
    modules.shared = shared
    modules.devices = SimpleNamespace(torch_gc=lambda: None)
    monkeypatch.setitem(sys.modules, "modules", modules)
    monkeypatch.setitem(sys.modules, "modules.shared", shared)
    monkeypatch.setitem(sys.modules, "modules.call_queue", queue)
    return SimpleNamespace(state=state, lock=lock, events=events)


@pytest.mark.parametrize("fails", [False, True])
def test_action_waits_for_host_lock_before_touching_state_and_cleans_up(host, fails):
    def action():
        assert not host.state.interrupted
        assert not host.state.skipped
        host.events.append("action")
        host.state.interrupted = host.state.skipped = True
        if fails:
            msg = "simulated callback failure"
            raise RuntimeError(msg)
        return ["result"], "original status"

    host.lock.lock.acquire()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(wrap_adetailer_job(action))
        try:
            assert host.lock.attempted.wait(timeout=2)
            assert host.events == []
            assert host.state.interrupted
            assert host.state.job == "previous job"
        finally:
            host.lock.lock.release()

        if fails:
            with pytest.raises(RuntimeError, match="simulated callback failure"):
                future.result(timeout=2)
        else:
            assert future.result(timeout=2) == (["result"], "original status")

    assert host.events == ["begin", "action", "end"]
    assert not host.lock.lock.locked()
    assert not host.state.interrupted
    assert not host.state.skipped
    assert not host.state.stopping_generation
    assert (host.state.job, host.state.job_count) == ("", 0)


def test_action_runs_on_a_host_whose_begin_takes_no_job_label(host, monkeypatch):
    def begin_without_label(self):
        host.events.append("begin")
        self.skipped = self.interrupted = self.stopping_generation = False

    monkeypatch.setattr(type(host.state), "begin", begin_without_label)
    assert wrap_adetailer_job(lambda: "done")() == "done"
    assert host.events == ["begin", "end"]
    assert not host.lock.lock.locked()


def test_detector_serializes_without_resetting_generation_state(host):
    def detect():
        assert host.lock.lock.locked()
        return "preview", "status"

    assert wrap_adetailer_detection(detect)() == ("preview", "status")
    assert host.events == []
    assert host.state.interrupted
    assert host.state.skipped
    assert host.state.job == "previous job"


def test_no_event_forces_the_gradio_queue():
    # Gradio 3 (AUTOMATIC1111) refuses to launch when the WebUI runs with
    # --no-gradio-queue and any event passes queue=True: the Detection preview
    # and Run buttons did, so the whole WebUI failed to start.
    root = _UI_PATH.parents[1]
    forced = []
    for path in [*(root / "aaaaaa").glob("*.py"), *(root / "scripts").glob("*.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        forced.extend(
            f"{path.name}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for k in node.keywords
            if k.arg == "queue"
            and isinstance(k.value, ast.Constant)
            and k.value.value is True
        )
    assert forced == []


_EVENTS = {"click", "change", "input", "select", "upload", "submit", "release", "then"}


@pytest.mark.parametrize(
    ("relpath", "expected"), [("aaaaaa/ui.py", 27), ("scripts/!adetailer.py", 1)]
)
def test_no_new_gradio_event_listener(relpath, expected):
    # Forge numbers the event handlers (fn_index) in registration order, so a
    # new listener shifts the WebUI's own handlers: the project adds none.
    # This counts the registration calls in the source (the project's grep
    # also counts a comment in ui.py, hence its 28); a call inside a loop,
    # such as the per-setting .change of ALL_ARGS, counts once here, which is
    # why tests/test_args.py pins ALL_ARGS too.
    tree = ast.parse((_UI_PATH.parents[1] / relpath).read_text(encoding="utf-8"))
    calls = [
        f"{node.lineno}:{node.func.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _EVENTS
    ]
    assert len(calls) == expected, calls


def _wired_apply(monkeypatch, script):
    """Execute the actual callback wiring, omitting only WebUI import side effects."""
    args_module = ModuleType("adetailer.args")
    args_module.ADetailerArgs = SimpleNamespace
    monkeypatch.setitem(sys.modules, "adetailer.args", args_module)
    tree = ast.parse(_UI_PATH.read_text(encoding="utf-8"))
    body = ast.parse("from __future__ import annotations").body
    body.extend(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_wire_detection_previews"
    )
    namespace = {
        "ALL_ARGS": SimpleNamespace(attrs=("ad_model",)),
        "_apply_exif_orientation": lambda image: image,
        "wrap_adetailer_job": wrap_adetailer_job,
        "wrap_adetailer_detection": wrap_adetailer_detection,
    }
    module = ast.Module(body=body, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_UI_PATH), "exec"), namespace)

    class Widget:
        def __init__(self):
            self.ad_preview_btn = SimpleNamespace(click=lambda **_kwargs: None)
            self.ad_apply_btn = SimpleNamespace(click=self.capture)
            self.callback = None

        def __getattr__(self, name):
            return name

        def tolist(self):
            return ["model component"]

        def capture(self, **kwargs):
            self.callback = kwargs["fn"]

    widget = Widget()
    namespace["_wire_detection_previews"](
        [widget], SimpleNamespace(model_mapping={}), 1, script
    )
    return widget.callback


@pytest.mark.parametrize("cancel_after_first", [False, True])
def test_wired_folder_restarts_after_cancel_and_preserves_new_interrupt(
    host, monkeypatch, tmp_path, cancel_after_first
):
    for name in ("a.png", "b.png"):
        Image.new("RGB", (8, 8), "white").save(tmp_path / name)
    calls = []

    def detail(image, _args, save):
        assert not host.state.interrupted
        assert not host.state.skipped
        assert save
        calls.append(image)
        if cancel_after_first:
            host.state.interrupted = True
        return image, "✅ ADetailer pass complete."

    script = SimpleNamespace(run_detailer_on_image=detail)
    callback = _wired_apply(monkeypatch, script)
    gallery, status = callback(
        None, str(tmp_path), False, True,
        "face.pt", "", "", False, 0.3, "face.pt",
    )

    assert len(calls) == (1 if cancel_after_first else 2)
    assert len(gallery) == len(calls)
    assert ("Batch interrupted" if cancel_after_first else "Batch done") in status
    assert host.events == ["begin", "end"]
    assert not host.state.interrupted


def test_wired_folder_stops_before_next_file_when_host_requests_stop(
    host, monkeypatch, tmp_path
):
    # AUTOMATIC1111's Interrupt button calls stop_generating() instead of
    # interrupt() once job_count > 1 (its default "interrupt after current"
    # option). That sets only stopping_generation. The folder must finish the
    # current file and stop, not run every remaining file for nothing.
    for name in ("a.png", "b.png", "c.png"):
        Image.new("RGB", (8, 8), "white").save(tmp_path / name)
    calls = []

    def detail(image, _args, save):
        # Record, don't assert: an exception here would be swallowed by the
        # folder loop as "unreadable" and hide a missing stop check.
        calls.append(host.state.stopping_generation)
        host.state.stopping_generation = True
        return image, "✅ ADetailer pass complete."

    script = SimpleNamespace(run_detailer_on_image=detail)
    callback = _wired_apply(monkeypatch, script)
    _gallery, status = callback(
        None, str(tmp_path), False, True,
        "face.pt", "", "", False, 0.3, "face.pt",
    )

    assert calls == [False]
    assert "Batch interrupted" in status
    assert "unreadable" not in status
    assert host.events == ["begin", "end"]
    assert not host.state.stopping_generation


@pytest.mark.parametrize("same_folder", [False, True])
def test_wired_folder_lists_cancelled_files_separately(
    host, monkeypatch, tmp_path, same_folder
):
    # A file whose pass was cancelled did have detections: the summary must not
    # call it "nothing detected", and it must not be written anywhere.
    for name in ("a.png", "b.png"):
        Image.new("RGB", (8, 8), "white").save(tmp_path / name)
    calls = []

    def detail(image, _args, save):
        calls.append(save)
        host.state.interrupted = True
        return image, "ℹ️ Cancelled — image unchanged."

    script = SimpleNamespace(run_detailer_on_image=detail)
    callback = _wired_apply(monkeypatch, script)
    _gallery, status = callback(
        None, str(tmp_path), same_folder, True,
        "face.pt", "", "", False, 0.3, "face.pt",
    )

    assert calls == [not same_folder]
    assert "Batch interrupted" in status
    assert "1 cancelled" in status
    assert "nothing detected" not in status
    assert "nothing to inpaint" not in status
    assert sorted(f.name for f in tmp_path.iterdir()) == ["a.png", "b.png"]


@pytest.mark.parametrize("same_folder", [False, True])
def test_wired_folder_without_detector_says_so(
    host, monkeypatch, tmp_path, same_folder
):
    # A tab without a detector used to report "Batch done ... 2 skipped
    # (unreadable)" for perfectly readable images.
    for name in ("a.png", "b.png"):
        Image.new("RGB", (8, 8), "white").save(tmp_path / name)
    calls = []

    def detail(image, _args, save):
        calls.append(save)
        return image, "✅ ADetailer pass complete."

    script = SimpleNamespace(run_detailer_on_image=detail)
    callback = _wired_apply(monkeypatch, script)
    gallery, status = callback(
        None, str(tmp_path), same_folder, True,
        "None", "", "", False, 0.3, "None",
    )

    assert calls == []
    assert gallery is None
    assert status == "⚠️ Pick a detector model first."
    assert sorted(f.name for f in tmp_path.iterdir()) == ["a.png", "b.png"]


@pytest.mark.parametrize("same_folder", [False, True])
@pytest.mark.parametrize("readable", [True, False])
def test_wired_folder_without_detector_and_an_unreadable_file(
    host, monkeypatch, tmp_path, same_folder, readable
):
    # What the README says: a missing detector stops the batch at the first
    # image that opens, with only its message; when no image opens, the batch
    # is reported as failed, with every file skipped as unreadable.
    (tmp_path / "a.png").write_bytes(b"not an image")
    names = ["a.png"]
    if readable:
        Image.new("RGB", (8, 8), "white").save(tmp_path / "b.png")
        names.append("b.png")
    calls = []

    def detail(image, _args, save):
        calls.append(save)
        return image, "✅ ADetailer pass complete."

    script = SimpleNamespace(run_detailer_on_image=detail)
    callback = _wired_apply(monkeypatch, script)
    gallery, status = callback(
        None, str(tmp_path), same_folder, True,
        "None", "", "", False, 0.3, "None",
    )

    assert calls == []
    assert gallery is None
    if readable:
        assert status == "⚠️ Pick a detector model first."
    else:
        assert status.startswith("⚠️ Batch failed")
        assert "1 skipped (unreadable)" in status
        assert "Pick a detector" not in status
    assert sorted(f.name for f in tmp_path.iterdir()) == names


@pytest.mark.parametrize("same_folder", [False, True])
def test_wired_folder_reports_failed_runs_with_their_reason(
    host, monkeypatch, tmp_path, same_folder
):
    for name in ("a.png", "b.png"):
        Image.new("RGB", (8, 8), "white").save(tmp_path / name)
    calls = []

    def detail(image, _args, save):
        calls.append(save)
        return None, "⚠️ ADetailer run failed: CUDA out of memory."

    script = SimpleNamespace(run_detailer_on_image=detail)
    callback = _wired_apply(monkeypatch, script)
    _gallery, status = callback(
        None, str(tmp_path), same_folder, True,
        "face.pt", "", "", False, 0.3, "face.pt",
    )

    assert len(calls) == 2
    assert status.startswith("⚠️ Batch failed")
    assert "2 failed (ADetailer run failed: CUDA out of memory — see console)" in status
    assert "unreadable" not in status


class _Unsavable:
    def save(self, *_args, **_kwargs):
        msg = "disk full"
        raise OSError(msg)


@pytest.mark.parametrize("reason", ["save fails", "no free name"])
def test_wired_same_folder_explains_unsaved_results_in_the_console(
    host, monkeypatch, tmp_path, capsys, reason
):
    # The status says "detailed but not saved (see console)"; the console
    # must then say why.
    Image.new("RGB", (8, 8), "white").save(tmp_path / "a.png")
    result = Image.new("RGB", (8, 8), "black")
    if reason == "save fails":
        result = _Unsavable()
    else:
        exists = Path.exists
        monkeypatch.setattr(
            Path, "exists", lambda self: "-ad" in self.stem or exists(self)
        )

    def detail(_image, _args, save):
        return result, "✅ ADetailer pass complete."

    script = SimpleNamespace(run_detailer_on_image=detail)
    callback = _wired_apply(monkeypatch, script)
    _gallery, status = callback(
        None, str(tmp_path), True, False,
        "face.pt", "", "", False, 0.3, "face.pt",
    )

    assert "1 detailed but not saved (see console)" in status
    console = "".join(capsys.readouterr())
    if reason == "save fails":
        assert "a-ad.png" in console
        assert "disk full" in console
    else:
        assert "a.png" in console
        assert "no free" in console
    assert sorted(f.name for f in tmp_path.iterdir()) == ["a.png"]


@pytest.mark.parametrize("reason", ["save fails", "no free name"])
def test_wired_same_folder_console_line_prints_on_a_legacy_code_page(
    host, monkeypatch, tmp_path, reason
):
    # With the output redirected to a file in a legacy code page, a path with
    # characters that page lacks made the console line itself raise: the file
    # was counted a second time as unreadable and left out of the gallery, and
    # the console said nothing although the status said "see console".
    folder = tmp_path / "写真"
    folder.mkdir()
    name = "a.png" if reason == "save fails" else "顔.png"
    Image.new("RGB", (8, 8), "white").save(folder / name)
    result = Image.new("RGB", (8, 8), "black")
    if reason == "save fails":
        result = _Unsavable()
    else:
        exists = Path.exists
        monkeypatch.setattr(
            Path, "exists", lambda self: "-ad" in self.stem or exists(self)
        )

    def detail(_image, _args, save):
        return result, "✅ ADetailer pass complete."

    callback = _wired_apply(monkeypatch, SimpleNamespace(run_detailer_on_image=detail))
    buffer = io.BytesIO()
    monkeypatch.setattr(
        sys, "stdout",
        io.TextIOWrapper(buffer, encoding="cp1252", errors="strict", write_through=True),
    )
    gallery, status = callback(
        None, str(folder), True, False,
        "face.pt", "", "", False, 0.3, "face.pt",
    )

    assert "1 detailed but not saved (see console)" in status
    assert "unreadable" not in status
    assert gallery is not None
    assert len(gallery) == 1
    console = buffer.getvalue()
    if reason == "save fails":
        assert b"\\u5199\\u771f" in console
        assert b"a-ad.png (disk full)." in console
    else:
        assert b"\\u9854.png: no free '-ad' file name left." in console


def _wired_preview(monkeypatch, predict, script=None, **extra):
    """The real two-tab Detection preview wiring, with a stub detector."""
    detector = ModuleType("adetailer.ultralytics")
    detector.ultralytics_predict = predict
    monkeypatch.setitem(sys.modules, "adetailer.ultralytics", detector)
    helper = ModuleType("aaaaaa.helper")
    helper.disable_safe_unpickle = nullcontext
    monkeypatch.setitem(sys.modules, "aaaaaa.helper", helper)
    tree = ast.parse(_UI_PATH.read_text(encoding="utf-8"))
    body = ast.parse("from __future__ import annotations").body
    body.extend(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_wire_detection_previews"
    )
    namespace = {
        "ALL_ARGS": SimpleNamespace(attrs=("ad_model",)),
        "_apply_exif_orientation": lambda image: image,
        "wrap_adetailer_job": wrap_adetailer_job,
        "wrap_adetailer_detection": wrap_adetailer_detection,
        **extra,
    }
    module = ast.Module(body=body, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_UI_PATH), "exec"), namespace)

    class Widget:
        def __init__(self):
            self.ad_preview_btn = SimpleNamespace(click=self.capture)
            self.ad_apply_btn = SimpleNamespace(click=lambda **_kwargs: None)
            self.callback = None

        def __getattr__(self, name):
            return name

        def tolist(self):
            return ["model component"]

        def capture(self, **kwargs):
            self.callback = kwargs["fn"]

    widgets = [Widget(), Widget()]
    mapping = {"face.pt": "models/face.pt", "hand.pt": "models/hand.pt"}
    namespace["_wire_detection_previews"](
        widgets, SimpleNamespace(model_mapping=mapping), 2, script
    )
    return widgets[0].callback


@pytest.mark.parametrize("combine", [False, True])
@pytest.mark.parametrize(
    ("script", "device"),
    [(SimpleNamespace(ultralytics_device="cpu"), "cpu"), (None, "")],
)
def test_preview_detects_on_the_device_generation_uses(
    host, monkeypatch, combine, script, device
):
    # With --use-cpu adetailer, --lowvram or --medvram generation and "Run
    # ADetailer on an image" detect on the CPU, but the Detection preview
    # always asked for the default device (the GPU when there is one).
    devices = []

    def predict(_path, **kwargs):
        devices.append(kwargs["device"])
        return SimpleNamespace(bboxes=[], confidences=[], masks=[], preview=None)

    callback = _wired_preview(monkeypatch, predict, script)
    callback(
        Image.new("RGB", (8, 8), "white"), combine,
        "face.pt", "", "", False, 0.3, "hand.pt", "", "", False, 0.3, "face.pt",
    )

    assert devices == [device] * (2 if combine else 1)


@pytest.mark.parametrize("second_tab_runs", [False, True])
def test_combined_preview_reports_a_failed_detector(host, monkeypatch, second_tab_runs):
    # With "Combine all tabs", a tab whose detector failed (a model file removed
    # or corrupt, out of memory) showed only "ERR", and when every tab failed
    # the status said "No detections" with the plain image. The single-tab
    # preview reports "Preview failed" and the reason.
    def predict(path, **_kwargs):
        if path == "models/face.pt" or not second_tab_runs:
            msg = "simulated detector failure"
            raise RuntimeError(msg)
        return SimpleNamespace(bboxes=[], confidences=[], masks=[], preview=None)

    callback = _wired_preview(monkeypatch, predict)
    image, status = callback(
        Image.new("RGB", (8, 8), "white"), True,
        "face.pt", "", "", False, 0.3, "hand.pt", "", "", False, 0.3, "face.pt",
    )

    assert "Tab1: ERR (simulated detector failure)" in status
    if second_tab_runs:
        # One detector ran and found nothing: that is still "No detections".
        assert image is not None
        assert status.startswith("ℹ️ No detections across tabs")
        assert "Tab2: 0" in status
    else:
        assert image is None
        assert status.startswith("⚠️ Preview failed")
        assert "Tab2: ERR (simulated detector failure)" in status
        assert "No detections" not in status


def _load_draw_detection_numbers():
    tree = ast.parse(_UI_PATH.read_text(encoding="utf-8"))
    body = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_draw_detection_numbers"
    ]
    namespace = {}
    module = ast.Module(body=body, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_UI_PATH), "exec"), namespace)
    return namespace["_draw_detection_numbers"]


@pytest.mark.parametrize(
    ("names", "labels"),
    [
        (["eyes", "nose"], ["#1 eyes", "#2 nose"]),
        (["eyes"], ["#1 eyes", "#2"]),
        (["", "nose"], ["#1", "#2 nose"]),
        ([], ["#1", "#2"]),
        (None, ["#1", "#2"]),
    ],
)
def test_detection_number_chips_name_the_class(monkeypatch, names, labels):
    # mediapipe_face_features writes each part's name at the top-left corner of
    # its box, where the number chip covered it: the five overlapping boxes of
    # a face showed only #1 to #5, so the number to type was a guess.
    from PIL import ImageDraw

    draw_numbers = _load_draw_detection_numbers()
    image = Image.new("RGB", (256, 256), "gray")
    bboxes = [[10, 10, 100, 60], [20, 70, 120, 130]]
    plain = draw_numbers(image, bboxes)
    drawn = []
    text = ImageDraw.ImageDraw.text

    def record(self, xy, label, *args, **kwargs):
        drawn.append(label)
        return text(self, xy, label, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record)
    out = draw_numbers(image, bboxes, names)

    assert drawn == labels
    # Without a name the chips are drawn as before.
    assert (out.tobytes() == plain.tobytes()) is (labels == ["#1", "#2"])


@pytest.mark.parametrize(
    ("names", "labels"),
    [
        (["顔", "hand"], ["#1", "#2 hand"]),
        (["hand", "łapa"], ["#1 hand", "#2"]),
    ],
)
def test_detection_number_chips_keep_the_number_with_a_bitmap_font(
    monkeypatch, names, labels
):
    # With no TrueType font found (macOS / minimal Linux) Pillow < 10.1 falls
    # back to its Latin-1 bitmap font: one class name outside Latin-1 made the
    # label measurement raise, and the preview lost every number chip.
    from PIL import ImageDraw, ImageFont

    draw_numbers = _load_draw_detection_numbers()
    bitmap = getattr(ImageFont, "load_default_imagefont", ImageFont.load_default)()

    def no_truetype(*_args, **_kwargs):
        raise OSError("cannot open resource")

    # No `size` argument, as on Pillow < 10.1.
    monkeypatch.setattr(ImageFont, "truetype", no_truetype)
    monkeypatch.setattr(ImageFont, "load_default", lambda: bitmap)
    drawn = []
    text = ImageDraw.ImageDraw.text

    def record(self, xy, label, *args, **kwargs):
        drawn.append(label)
        return text(self, xy, label, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record)
    image = Image.new("RGB", (256, 256), "gray")
    out = draw_numbers(image, [[10, 10, 100, 60], [20, 70, 120, 130]], names)

    assert drawn == labels
    assert out.tobytes() != image.tobytes()


@pytest.mark.parametrize(
    ("names", "confidences", "labels"),
    [
        (["hand"], [0.9], ["#1 1:hand 0.90", "#1 2:hand 0.90"]),
        (["visage"], [0.9], ["#1 1:visage 0.90", "#1 2:visage 0.90"]),
        (["顔"], [0.9], ["#1 1 0.90", "#1 2 0.90"]),
        (["łapa"], [], ["#1 1", "#1 2"]),
        ([], [0.9], ["#1 1:T1 0.90", "#1 2:T2 0.90"]),
    ],
)
def test_combined_preview_keeps_its_labels_with_a_bitmap_font(
    host, monkeypatch, names, confidences, labels
):
    # The same Latin-1 bitmap font fallback as above: with "Combine all tabs"
    # one class name outside Latin-1 (a YOLO-World word, a custom model's
    # class) raised from the label measurement and the preview failed.
    from PIL import ImageDraw, ImageFont

    bitmap = getattr(ImageFont, "load_default_imagefont", ImageFont.load_default)()

    def no_truetype(*_args, **_kwargs):
        raise OSError("cannot open resource")

    # No `size` argument, as on Pillow < 10.1.
    monkeypatch.setattr(ImageFont, "truetype", no_truetype)
    monkeypatch.setattr(ImageFont, "load_default", lambda: bitmap)
    drawn = []
    text = ImageDraw.ImageDraw.text

    def record(self, xy, label, *args, **kwargs):
        drawn.append(label)
        return text(self, xy, label, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record)

    def predict(_path, **_kwargs):
        return SimpleNamespace(
            bboxes=[[10, 20, 60, 70]], confidences=confidences, masks=[],
            preview=None, class_names=names,
        )

    callback = _wired_preview(monkeypatch, predict)
    image, status = callback(
        Image.new("RGB", (128, 128), "white"), True,
        "face.pt", "", "", False, 0.3, "hand.pt", "", "", False, 0.3, "face.pt",
    )

    assert image is not None
    assert status.startswith("✅ 2 detection(s) across 2 tab(s)")
    assert drawn == labels


@pytest.mark.parametrize("model", ["mediapipe_face_features", "face.pt"])
def test_single_tab_preview_chips_get_the_class_names(host, monkeypatch, model):
    pred = SimpleNamespace(
        bboxes=[[0, 0, 10, 10], [2, 2, 12, 12]], confidences=[0.9, 0.8], masks=[],
        preview=Image.new("RGB", (16, 16)), class_names=["eyes", "nose"],
    )
    mediapipe = ModuleType("adetailer.mediapipe")
    mediapipe.mediapipe_predict = lambda *_args, **_kwargs: pred
    monkeypatch.setitem(sys.modules, "adetailer.mediapipe", mediapipe)
    calls = []

    def draw(img, bboxes, *names):
        calls.append((bboxes, *names))
        return img

    callback = _wired_preview(
        monkeypatch, lambda _path, **_kwargs: pred, _draw_detection_numbers=draw
    )
    _image, status = callback(
        Image.new("RGB", (16, 16), "white"), False,
        model, "", "", False, 0.3, "hand.pt", "", "", False, 0.3, model,
    )

    assert status == "✅ 2 detection(s)."
    assert calls == [(pred.bboxes, ["eyes", "nose"])]


def test_wired_folder_still_skips_an_unreadable_file(host, monkeypatch, tmp_path):
    Image.new("RGB", (8, 8), "white").save(tmp_path / "a.png")
    (tmp_path / "b.png").write_bytes(b"not an image")

    def detail(image, _args, save):
        return image, "✅ ADetailer pass complete."

    script = SimpleNamespace(run_detailer_on_image=detail)
    callback = _wired_apply(monkeypatch, script)
    _gallery, status = callback(
        None, str(tmp_path), False, True,
        "face.pt", "", "", False, 0.3, "face.pt",
    )

    assert status.startswith("✅ Batch done")
    assert "1 detailed" in status
    assert "1 skipped (unreadable)" in status
    assert "failed" not in status


@pytest.mark.parametrize("same_folder", [False, True])
def test_wired_folder_counts_files_with_nothing_to_inpaint(
    host, monkeypatch, tmp_path, same_folder
):
    # A file whose detections were all left out (detection numbers, mask
    # filters or [SKIP]) was counted as "nothing detected".
    Image.new("RGB", (8, 8), "white").save(tmp_path / "a.png")

    def detail(image, _args, save):
        return image, (
            "ℹ️ Detections found, but the detection numbers, mask filters or "
            "[SKIP] left none to inpaint — image unchanged."
        )

    script = SimpleNamespace(run_detailer_on_image=detail)
    callback = _wired_apply(monkeypatch, script)
    _gallery, status = callback(
        None, str(tmp_path), same_folder, True,
        "face.pt", "", "", False, 0.3, "face.pt",
    )

    assert "1 left unchanged (nothing to inpaint)" in status
    assert "nothing detected" not in status
    assert sorted(f.name for f in tmp_path.iterdir()) == ["a.png"]


@pytest.mark.parametrize("same_folder", [False, True])
def test_wired_folder_that_wrote_nothing_is_headed_failed(
    host, monkeypatch, tmp_path, same_folder
):
    # Every file was detailed but none could be written (a full disk or a
    # folder you cannot write to): the batch said "Batch done".
    for name in ("a.png", "b.png"):
        Image.new("RGB", (8, 8), "white").save(tmp_path / name)

    def detail(image, _args, save):
        if same_folder:
            return _Unsavable(), "✅ ADetailer pass complete."
        return image, "✅ ADetailer pass complete. (couldn't save — see console)"

    script = SimpleNamespace(run_detailer_on_image=detail)
    callback = _wired_apply(monkeypatch, script)
    _gallery, status = callback(
        None, str(tmp_path), same_folder, True,
        "face.pt", "", "", False, 0.3, "face.pt",
    )

    assert status.startswith("⚠️ Batch failed")
    assert "2 detailed but not saved" in status
    assert sorted(f.name for f in tmp_path.iterdir()) == ["a.png", "b.png"]


@pytest.mark.parametrize("same_folder", [False, True])
@pytest.mark.parametrize("kind", ["not saved", "failed"])
def test_wired_folder_that_saved_none_beside_unchanged_files_is_headed_failed(
    host, monkeypatch, tmp_path, same_folder, kind
):
    # One file with nothing to inpaint made a run whose every detailed
    # result failed or could not be saved "Batch done", although nothing
    # was written.
    for name in ("a.png", "b.png", "c.png"):
        Image.new("RGB", (8, 8), "white").save(tmp_path / name)
    calls = []

    def detail(image, _args, save):
        calls.append(save)
        if len(calls) == 1:
            return image, "ℹ️ Nothing detected — image unchanged."
        if kind == "failed":
            return None, "⚠️ ADetailer run failed: CUDA out of memory."
        if same_folder:
            return _Unsavable(), "✅ ADetailer pass complete."
        return image, "✅ ADetailer pass complete. (couldn't save — see console)"

    script = SimpleNamespace(run_detailer_on_image=detail)
    callback = _wired_apply(monkeypatch, script)
    _gallery, status = callback(
        None, str(tmp_path), same_folder, True,
        "face.pt", "", "", False, 0.3, "face.pt",
    )

    assert len(calls) == 3
    assert status.startswith("⚠️ Batch failed")
    assert "1 left unchanged (nothing to inpaint)" in status
    assert ("2 failed" if kind == "failed" else "2 detailed but not saved") in status
    assert sorted(f.name for f in tmp_path.iterdir()) == ["a.png", "b.png", "c.png"]


@pytest.mark.parametrize("same_folder", [False, True])
def test_wired_folder_details_tif_images(host, monkeypatch, tmp_path, same_folder):
    # Only the ".tiff" spelling was accepted: a folder of scans saved as
    # ".tif" had "No images", and in a mixed folder they were left out.
    Image.new("RGB", (8, 8), "white").save(tmp_path / "a.png")
    Image.new("RGB", (8, 8), "white").save(tmp_path / "scan.tif")
    calls = []

    def detail(image, _args, save):
        calls.append(save)
        return image, "✅ ADetailer pass complete."

    script = SimpleNamespace(run_detailer_on_image=detail)
    callback = _wired_apply(monkeypatch, script)
    _gallery, status = callback(
        None, str(tmp_path), same_folder, True,
        "face.pt", "", "", False, 0.3, "face.pt",
    )

    assert calls == [not same_folder] * 2
    assert status.startswith("✅ Batch done — 2 image(s): 2 detailed")
    if same_folder:
        assert sorted(f.name for f in tmp_path.iterdir()) == [
            "a-ad.png", "a.png", "scan-ad.tif", "scan.tif",
        ]
        with Image.open(tmp_path / "scan-ad.tif") as result:
            assert result.format == "TIFF"


def test_wired_same_folder_counts_files_skipped_as_earlier_results(
    host, monkeypatch, tmp_path
):
    # In same-folder mode a file named like an earlier result (name-ad or
    # name-ad-<n>, in any case) is skipped, but the status did not count it,
    # and a folder of only such files reported "No images found".
    for name in ("banner-ad-1.png", "summer-AD.jpg"):
        Image.new("RGB", (8, 8), "white").save(tmp_path / name)
    calls = []

    def detail(image, _args, save):
        calls.append(save)
        return image, "✅ ADetailer pass complete."

    callback = _wired_apply(monkeypatch, SimpleNamespace(run_detailer_on_image=detail))
    tab = ("face.pt", "", "", False, 0.3, "face.pt")
    _gallery, status = callback(None, str(tmp_path), True, True, *tab)
    assert calls == []
    assert status.startswith("ℹ️ No images to detail: the 2 image(s)")
    assert "<" not in status  # shown as Markdown

    Image.new("RGB", (8, 8), "white").save(tmp_path / "photo.png")
    _gallery, status = callback(None, str(tmp_path), True, True, *tab)
    assert calls == [False]
    assert status.startswith("✅ Batch done — 1 image(s): 1 detailed")
    assert status.endswith(", 2 skipped as earlier results (name ending in -ad, -ad-1, -ad-2…).")
    assert sorted(f.name for f in tmp_path.iterdir()) == [
        "banner-ad-1.png", "photo-ad.png", "photo.png", "summer-AD.jpg",
    ]

    # Without the option every image is detailed and nothing is skipped.
    calls.clear()
    _gallery, status = callback(None, str(tmp_path), False, True, *tab)
    assert calls == [True] * 4
    assert status.startswith("✅ Batch done — 4 image(s): 4 detailed")
    assert "skipped" not in status


@pytest.mark.parametrize("same_folder", [False, True])
@pytest.mark.parametrize("files", [1, 3])
def test_wired_folder_cancelled_on_its_last_file_is_headed_interrupted(
    host, monkeypatch, tmp_path, same_folder, files
):
    # The loop checks for a stop only before each file, so a cancel during the
    # last (or only) file ended the batch as "Batch done".
    for i in range(files):
        Image.new("RGB", (8, 8), "white").save(tmp_path / f"{i}.png")
    calls = []

    def detail(image, _args, save):
        calls.append(save)
        if len(calls) < files:
            return image, "ℹ️ Nothing detected — image unchanged."
        host.state.interrupted = True
        return image, "ℹ️ Cancelled — image unchanged."

    script = SimpleNamespace(run_detailer_on_image=detail)
    callback = _wired_apply(monkeypatch, script)
    _gallery, status = callback(
        None, str(tmp_path), same_folder, True,
        "face.pt", "", "", False, 0.3, "face.pt",
    )

    assert len(calls) == files
    assert status.startswith(f"⏹️ Batch interrupted — {files}/{files} image(s)")
    assert "1 cancelled" in status


def test_wired_folder_interrupted_count_includes_failed_files(
    host, monkeypatch, tmp_path
):
    # The "done/total" figure of an interrupted batch left out the files that
    # failed: "0/3" after one failed run.
    for name in ("a.png", "b.png", "c.png"):
        Image.new("RGB", (8, 8), "white").save(tmp_path / name)

    def detail(image, _args, save):
        host.state.interrupted = True
        return None, "⚠️ ADetailer run failed: CUDA out of memory."

    script = SimpleNamespace(run_detailer_on_image=detail)
    callback = _wired_apply(monkeypatch, script)
    _gallery, status = callback(
        None, str(tmp_path), False, True,
        "face.pt", "", "", False, 0.3, "face.pt",
    )

    assert status.startswith("⏹️ Batch interrupted — 1/3 image(s)")
    assert "1 failed" in status


def test_wired_folder_with_nothing_to_inpaint_is_still_done(
    host, monkeypatch, tmp_path
):
    # Files left unchanged are a success: with an unreadable file beside
    # them the batch is still "done", not "failed".
    Image.new("RGB", (8, 8), "white").save(tmp_path / "a.png")
    (tmp_path / "b.png").write_bytes(b"not an image")

    def detail(image, _args, save):
        return image, "ℹ️ Nothing detected — image unchanged."

    script = SimpleNamespace(run_detailer_on_image=detail)
    callback = _wired_apply(monkeypatch, script)
    _gallery, status = callback(
        None, str(tmp_path), False, True,
        "face.pt", "", "", False, 0.3, "face.pt",
    )

    assert status.startswith("✅ Batch done")
    assert "1 left unchanged" in status
    assert "1 skipped (unreadable)" in status


def test_wired_folder_keeps_the_transparency_of_a_cut_out(host, monkeypatch, tmp_path):
    # A folder run opened every file as RGB, which drops the alpha channel: the
    # transparent parts reached the pass black, or in the colour stored under
    # them, and the pass could not fill them like the WebUI's img2img.
    cutout = Image.new("RGBA", (8, 8), (10, 200, 10, 0))
    cutout.paste((200, 30, 30, 255), (2, 2, 6, 6))
    cutout.save(tmp_path / "a.png")
    palette = Image.new("P", (8, 8), 0)
    palette.putpalette([10, 200, 10, 200, 30, 30] + [0] * 762)
    palette.paste(1, (2, 2, 6, 6))
    palette.save(tmp_path / "b.png", transparency=0)
    Image.new("LA", (8, 8), (40, 0)).save(tmp_path / "c.png")
    Image.new("RGB", (8, 8), (10, 20, 30)).save(tmp_path / "d.png")
    Image.new("RGB", (8, 8), (10, 20, 30)).save(tmp_path / "e.jpg")
    seen = []

    def detail(image, _args, save):
        seen.append((image.mode, image.getpixel((0, 0))))
        return image, "✅ ADetailer pass complete."

    callback = _wired_apply(monkeypatch, SimpleNamespace(run_detailer_on_image=detail))
    _gallery, status = callback(
        None, str(tmp_path), False, True,
        "face.pt", "", "", False, 0.3, "face.pt",
    )

    assert status.startswith("✅ Batch done — 5 image(s): 5 detailed")
    assert seen[:4] == [
        ("RGBA", (10, 200, 10, 0)),
        ("RGBA", (10, 200, 10, 0)),
        ("RGBA", (40, 40, 40, 0)),
        # Without transparency: RGB, as before.
        ("RGB", (10, 20, 30)),
    ]
    assert seen[4][0] == "RGB"


def test_run_on_an_image_input_keeps_transparency():
    # Gradio's default image_mode, "RGB", dropped a dropped cut-out's alpha
    # channel before the pass could fill it. The Detection preview input is
    # left as it was.
    tree = ast.parse(_UI_PATH.read_text(encoding="utf-8"))
    modes = {}
    for node in ast.walk(tree):
        target = node.targets[0] if isinstance(node, ast.Assign) else None
        if (
            isinstance(target, ast.Attribute)
            and target.attr in ("ad_apply_input", "ad_preview_input")
            and isinstance(node.value, ast.Call)
        ):
            keywords = {k.arg: k.value for k in node.value.keywords}
            modes[target.attr] = tuple(
                ast.literal_eval(keywords[k]) if k in keywords else None
                for k in ("type", "image_mode")
            )
    assert modes == {"ad_apply_input": ("pil", "RGBA"), "ad_preview_input": ("pil", None)}


# An ICC header names its colour space at bytes 16 to 19; Pillow stores the
# profile as it is, so no colour management is needed to write one.
def _icc(space):
    return bytes(16) + space + bytes(108)


@pytest.mark.parametrize("ext", [".jpg", ".png", ".webp", ".tif"])
def test_wired_same_folder_copy_keeps_the_colour_profile(
    host, monkeypatch, tmp_path, ext
):
    # The copy saved beside the source had no colour profile: a wide-gamut
    # photo (Display P3, Adobe RGB) then looked duller than its original in a
    # colour-managed viewer.
    Image.new("RGB", (8, 8), "red").save(tmp_path / f"a{ext}", icc_profile=_icc(b"RGB "))
    Image.new("RGB", (8, 8), "red").save(tmp_path / f"b{ext}")

    def detail(image, _args, save):
        # Like the host's pass: a new image, without the source's info.
        return (
            Image.frombytes(image.mode, image.size, image.tobytes()),
            "✅ ADetailer pass complete.",
        )

    callback = _wired_apply(monkeypatch, SimpleNamespace(run_detailer_on_image=detail))
    _gallery, status = callback(
        None, str(tmp_path), True, False,
        "face.pt", "", "", False, 0.3, "face.pt",
    )

    assert status.startswith("✅ Batch done — 2 image(s): 2 detailed")
    with Image.open(tmp_path / f"a-ad{ext}") as result:
        assert result.info.get("icc_profile") == _icc(b"RGB ")
    # A source without a profile gives a copy without one, as before.
    with Image.open(tmp_path / f"b-ad{ext}") as result:
        assert not result.info.get("icc_profile")


def _webp_bitstream(data):
    """The fourcc of a WebP file's image chunk: b"VP8L" (lossless) or b"VP8 "."""
    pos = 12
    while pos + 8 <= len(data):
        fourcc, size = data[pos : pos + 4], int.from_bytes(data[pos + 4 : pos + 8], "little")
        if fourcc in (b"VP8 ", b"VP8L"):
            return fourcc
        pos += 8 + size + (size & 1)
    return None


@pytest.mark.parametrize(
    ("mode", "icc", "lossless"),
    [
        ("RGB", None, True),
        ("RGB", _icc(b"RGB "), True),  # extended file: VP8X and ICCP first
        ("RGBA", None, True),
        ("RGB", None, False),
    ],
    ids=["lossless", "lossless-profile", "lossless-alpha", "lossy"],
)
def test_wired_same_folder_webp_copy_keeps_a_lossless_source_lossless(
    host, monkeypatch, tmp_path, mode, icc, lossless
):
    # Pillow writes WebP lossy at quality 80 unless told otherwise: the copy of
    # a lossless WebP saved beside it lost detail all over the image, not only
    # in the detailed regions.
    import random

    rng = random.Random(0)
    noise = bytes(rng.randrange(1, 256) for _ in range(16 * 16 * len(mode)))
    extra = {"icc_profile": icc} if icc else {}
    Image.frombytes(mode, (16, 16), noise).save(
        tmp_path / "a.webp", lossless=lossless, **extra
    )
    assert _webp_bitstream((tmp_path / "a.webp").read_bytes()) == (
        b"VP8L" if lossless else b"VP8 "
    )
    with Image.open(tmp_path / "a.webp") as source:
        pixels = source.convert(mode).tobytes()

    def detail(image, _args, save):
        return (
            Image.frombytes(image.mode, image.size, image.tobytes()),
            "✅ ADetailer pass complete.",
        )

    callback = _wired_apply(monkeypatch, SimpleNamespace(run_detailer_on_image=detail))
    _gallery, status = callback(
        None, str(tmp_path), True, False,
        "face.pt", "", "", False, 0.3, "face.pt",
    )

    assert status.startswith("✅ Batch done — 1 image(s): 1 detailed")
    data = (tmp_path / "a-ad.webp").read_bytes()
    # A lossy source stays lossy, so its copy does not grow several times.
    assert _webp_bitstream(data) == (b"VP8L" if lossless else b"VP8 ")
    with Image.open(tmp_path / "a-ad.webp") as result:
        assert (result.info.get("icc_profile") or None) == icc
        if lossless:
            assert result.convert(mode).tobytes() == pixels
    if not lossless:
        # At the JPEG copy's quality (95), not Pillow's default 80.
        default = io.BytesIO()
        Image.frombytes(mode, (16, 16), pixels).save(default, "WEBP")
        assert len(data) > len(default.getvalue())


@pytest.mark.parametrize(("mode", "space"), [("L", b"GRAY"), ("CMYK", b"CMYK")])
def test_wired_same_folder_copy_leaves_out_a_profile_for_other_colours(
    host, monkeypatch, tmp_path, mode, space
):
    # The copy is always RGB: a grey or CMYK profile would not fit it.
    Image.new(mode, (8, 8)).save(tmp_path / "a.jpg", icc_profile=_icc(space))

    def detail(image, _args, save):
        return (
            Image.frombytes(image.mode, image.size, image.tobytes()),
            "✅ ADetailer pass complete.",
        )

    callback = _wired_apply(monkeypatch, SimpleNamespace(run_detailer_on_image=detail))
    callback(
        None, str(tmp_path), True, False,
        "face.pt", "", "", False, 0.3, "face.pt",
    )

    with Image.open(tmp_path / "a-ad.jpg") as result:
        assert result.mode == "RGB"
        assert not result.info.get("icc_profile")
