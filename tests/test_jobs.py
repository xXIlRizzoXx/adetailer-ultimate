"""Standalone action lifecycle with synthetic host state; no WebUI/GPU needed."""

from __future__ import annotations

import ast
import sys
from concurrent.futures import ThreadPoolExecutor
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
