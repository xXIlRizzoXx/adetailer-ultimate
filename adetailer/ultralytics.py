from __future__ import annotations

import os
import re
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from adetailer import PredictOutput
from adetailer.classes import (
    get_model_class_names,
    is_open_vocab_model,
    is_world_model,
    parse_csv,
    resolve_class_ids,
)
from adetailer.common import create_mask_from_bbox

if TYPE_CHECKING:
    import torch
    from ultralytics import YOLO, YOLOWorld


class UltralyticsTooOldError(AttributeError):
    """A detector made with a newer Ultralytics than the installed one.

    Still an AttributeError, the error the load raised before, so any code that
    catches that keeps working; only the message is clearer."""


def _too_old_ultralytics_message(model_path: str | Path, exc: BaseException) -> str | None:
    """A clear message when loading ``model_path`` failed because the model was
    saved with a newer Ultralytics (YOLO11, YOLO12, YOLO26...): unpickling it
    needs a layer class this Ultralytics does not have, e.g.
    "Can't get attribute 'C3k2' on <module 'ultralytics.nn.modules.block'>".
    None for any other error, which is then raised unchanged."""
    seen: set[int] = set()
    err: BaseException | None = exc
    while err is not None and id(err) not in seen:
        seen.add(id(err))
        text = str(err)
        if (
            isinstance(err, AttributeError)
            and "Can't get attribute" in text
            and "ultralytics" in text
        ):
            break
        err = err.__cause__ or err.__context__
    else:
        return None
    # The module's file path adds nothing here and makes the line very long.
    text = re.sub(r" from '[^']*'>", ">", text)
    try:
        from importlib.metadata import version

        installed = f"Ultralytics {version('ultralytics')}"
    except Exception:  # noqa: BLE001
        installed = "the installed Ultralytics"
    return (
        f"[-] ADetailer: the detector {Path(str(model_path)).name!r} was made with"
        f" a newer Ultralytics than {installed}, so it cannot be loaded"
        f" ({text}). Update Ultralytics in the WebUI's environment, or choose"
        " another detector."
    )


def _print_ascii(msg: str) -> None:
    # ASCII only, so a console in a legacy code page can print it.
    try:
        print(msg.encode("ascii", "backslashreplace").decode("ascii"), file=sys.stderr)
    except Exception:  # noqa: BLE001
        pass


# Ultralytics models whose results hold nothing to inpaint (no box and no mask
# of a single object), by the task Ultralytics gives the loaded model, and the
# field of a result each of them fills instead of the boxes.
_NOT_DETECTORS = {
    "obb": ("an OBB model (rotated boxes)", "obb"),
    "classify": (
        "a classification model (one label for the whole image)",
        "probs",
    ),
    "semantic": (
        "a semantic segmentation model (one class map for the whole image)",
        "semantic_mask",
    ),
}


def _not_a_detector_note(model_path: str | Path, model=None, result=None) -> str:
    """Why this model cannot be a detector, for PredictOutput.note; "" for one
    that can. Told from the loaded model's task, before anything is detected,
    or from a result that has no boxes. A detection, segmentation or pose
    model (its results have boxes) is never one of these."""
    kind = ""
    if model is not None:
        try:
            task = model.task
        except Exception:  # noqa: BLE001 — no task: detect as before
            task = None
        if isinstance(task, str) and task in _NOT_DETECTORS:
            kind = _NOT_DETECTORS[task][0]
    elif getattr(result, "boxes", None) is None:
        kind = "a model whose results have no boxes"
        for name, field in _NOT_DETECTORS.values():
            if getattr(result, field, None) is not None:
                kind = name
                break
    if not kind:
        return ""
    return (
        f"the detector {Path(str(model_path)).name!r} is {kind}; ADetailer"
        " needs a detection or segmentation model."
    )


# Where Ultralytics keeps its anonymous usage-analytics object (`events`),
# newest layout first: ultralytics.utils.events (recent releases such as 8.4) and
# ultralytics.hub.utils (older 8.x).
_EVENTS_MODULES = ("ultralytics.utils.events", "ultralytics.hub.utils")


def _disable_ultralytics_events() -> None:
    """Stop Ultralytics from sending anonymous usage analytics, from ADetailer's
    first detection on. The object is shared by the whole WebUI process, so it
    stays off for other extensions that use Ultralytics too.

    Ultralytics queues an analytics event on every predict (its on_predict_start
    callback) and posts it to google-analytics when its `sync` setting is on,
    which is the default. Only the in-memory `events` object is switched off:
    the shared settings file of Ultralytics (also used by other programs) is
    never written. Idempotent and best-effort: a missing module or attribute
    in another Ultralytics version changes nothing and never stops a detection.
    """
    import importlib

    for name in _EVENTS_MODULES:
        try:
            events = getattr(importlib.import_module(name), "events", None)
            if events is not None and hasattr(events, "enabled"):
                events.enabled = False
                return
        except Exception:  # noqa: BLE001
            continue


# Detectors kept loaded while one WebUI job runs (one click on Generate with
# all its images, tabs and X/Y/Z cells, or one "Run ADetailer on an image"
# with its whole folder), so the .pt file is not read and the model rebuilt
# for every tab of every image. Only the generation pass asks for it
# (keep_loaded=True); the Detection preview loads its detector each time.
# They are dropped when the job ends, so between jobs the memory (also the
# video memory) is as before. Key: the resolved file with its modification
# time and size (a file replaced under the same name is loaded again) and the
# device. YOLO-World and YOLOE (text prompts) detectors are never kept:
# set_classes changes the model.
_KEPT_LOCK = threading.Lock()
_KEPT: OrderedDict[tuple, tuple[Any, Any]] = OrderedDict()
# The job the kept detectors belong to: (thread, job start, job stamp).
_KEPT_JOB: tuple | None = None
# The job a watcher thread already follows.
_KEPT_WATCHED: tuple | None = None
# At most this many detectors at once: an X/Y/Z axis over many detectors
# keeps only the most recently used ones.
_KEPT_MAX = 4
# How often the watcher looks whether the job has ended (seconds).
_WATCH_EVERY = 0.5


def _job_id() -> tuple | None:
    """The running WebUI job as (start time, job stamp), the same for all its
    images, tabs and X/Y/Z cells and new for every job; None when no job
    runs (the Detection preview begins none) or there is no WebUI. Read only
    if the WebUI has loaded modules.shared itself."""
    try:
        state = sys.modules["modules.shared"].state
        if state.job == "" and state.job_count == 0:
            return None
        if state.time_start is None:
            return None
        return (state.time_start, str(state.job_timestamp))
    except Exception:  # noqa: BLE001
        return None


def _detector_key(model_path: str | Path, device: str) -> tuple | None:
    """The file as it is now and the device; None if the file cannot be read
    (then the detector is loaded as before and its own error is raised)."""
    try:
        path = Path(model_path).resolve()
        info = path.stat()
        return (str(path), info.st_mtime_ns, info.st_size, str(device))
    except Exception:  # noqa: BLE001
        return None


def _own_imgsz(model) -> Any:
    """The detection size the model uses when none is passed (stored in the
    checkpoint); None when it has none, and then the model is not kept: a
    reused model would keep the size of the call before it."""
    try:
        size = model.overrides.get("imgsz")
    except Exception:  # noqa: BLE001
        return None
    return size or None


def _free_video_memory() -> None:
    # The WebUI's own cleanup, which its end of the job already ran before
    # the watcher dropped the detectors. Best effort.
    try:
        devices = sys.modules.get("modules.devices")
        if devices is not None:
            devices.torch_gc()
    except Exception:  # noqa: BLE001
        pass


def release_kept_detectors(job: tuple | None = None) -> bool:
    """Drop the kept detectors (only those of `job` when given). True when
    any was dropped. Safe from any thread."""
    global _KEPT_JOB, _KEPT_WATCHED
    with _KEPT_LOCK:
        if job is not None and _KEPT_JOB != job:
            return False
        dropped = bool(_KEPT)
        _KEPT.clear()
        _KEPT_JOB = None
        _KEPT_WATCHED = None
    return dropped


def _job_watch_step(owner: tuple) -> bool:
    """One look of the watcher: True when it can stop (the detectors of
    `owner` were dropped because its job has ended, or they are gone)."""
    with _KEPT_LOCK:
        if _KEPT_JOB != owner:
            return True
    if _job_id() == owner[1:]:
        return False
    if release_kept_detectors(owner):
        _free_video_memory()
    return True


def _watch_job(owner: tuple) -> None:
    try:
        while not _job_watch_step(owner):
            time.sleep(_WATCH_EVERY)
    except Exception:  # noqa: BLE001 — never leave a traceback in the console
        release_kept_detectors(owner)


def _start_watch(owner: tuple) -> None:
    """Drop the detectors of `owner` as soon as its job ends: also when the
    job stops with an error, an Interrupt or a Skip, and after the last X/Y/Z
    cell, where no ADetailer code runs any more."""
    threading.Thread(
        target=_watch_job, args=(owner,), name="ADetailer-detectors", daemon=True
    ).start()


def _load_detector(YOLO, model_path: str | Path, device: str, keep: bool) -> tuple[Any, Any]:  # noqa: N803
    """(model, its own detection size if it is a kept one, else None).

    With `keep`, inside a WebUI job and on the job's thread, the model loaded
    for the same file and device earlier in the same job is reused."""
    if not keep or is_open_vocab_model(model_path):
        return YOLO(model_path), None
    job = _job_id()
    key = _detector_key(model_path, device)
    if job is None or key is None:
        return YOLO(model_path), None
    owner = (threading.get_ident(), *job)
    global _KEPT_JOB, _KEPT_WATCHED
    with _KEPT_LOCK:
        if _KEPT_JOB != owner:
            if _KEPT_JOB is not None and _KEPT_JOB[1:] == job:
                # The same job on another thread: leave its detectors alone.
                return YOLO(model_path), None
            # Detectors of a job that has ended: drop them.
            _KEPT.clear()
            _KEPT_JOB = owner
        kept = _KEPT.get(key)
        if kept is not None:
            _KEPT.move_to_end(key)
            return kept
    # Loaded outside the lock: it reads the file. Only a loaded model is kept.
    model = YOLO(model_path)
    size = _own_imgsz(model)
    if size is None:
        return model, None
    watch = False
    with _KEPT_LOCK:
        if _KEPT_JOB != owner:
            return model, None
        _KEPT[key] = (model, size)
        while len(_KEPT) > _KEPT_MAX:
            _KEPT.popitem(last=False)
        if _KEPT_WATCHED != owner:
            _KEPT_WATCHED = owner
            watch = True
    if watch:
        try:
            _start_watch(owner)
        except Exception:  # noqa: BLE001 — no thread: drop them now
            release_kept_detectors(owner)
    return model, size


# The Ultralytics that install.py installs when it is missing.
_ULTRALYTICS_SPEC = "ultralytics>=8.3.75"
# The WebUI job whose console already has the line of a missing Ultralytics.
_MISSING_SAID_JOB: tuple | None = None


def missing_ultralytics_line(detail: str = "it is not installed") -> str:
    """The console line for an Ultralytics that cannot be imported (a WebUI
    started with --skip-install, a failed install): the package, what it
    costs, and the pip command to run with the WebUI's Python.

    With the NumPy pin of install.py: on a WebUI with NumPy 1.x
    (AUTOMATIC1111) a plain install of Ultralytics can pull NumPy 2.x in,
    which breaks the whole WebUI."""
    specs = f'"{_ULTRALYTICS_SPEC}"'
    try:
        import numpy

        if int(numpy.__version__.split(".")[0]) < 2:
            specs += ' "numpy<2"'
    except Exception:  # noqa: BLE001 — no NumPy to keep
        pass
    return (
        "[-] ADetailer: the Python package 'ultralytics' cannot be imported"
        f" ({detail}), so the YOLO detectors cannot run. Start the WebUI once"
        " without --skip-install, or install it with the WebUI's Python:"
        f" python -m pip install {specs}"
    )


def warn_missing_ultralytics() -> bool:
    """For the start of the WebUI: Ultralytics is imported at the first
    detection only, so say now when it is not installed. Looked up without
    importing it; True when the line was said."""
    try:
        import importlib.util

        if importlib.util.find_spec("ultralytics") is not None:
            return False
    except Exception:  # noqa: BLE001 — cannot tell: the first detection will
        return False
    _print_ascii(missing_ultralytics_line())
    return True


def _say_missing_ultralytics(exc: BaseException) -> None:
    """The same line when a detection cannot import Ultralytics, before the
    error itself: once per WebUI job (the error is raised for every image),
    each time outside a job (the Detection preview)."""
    global _MISSING_SAID_JOB
    job = _job_id()
    if job is not None and job == _MISSING_SAID_JOB:
        return
    _MISSING_SAID_JOB = job
    _print_ascii(missing_ultralytics_line(str(exc)))


def ultralytics_predict(
    model_path: str | Path,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
    classes: str = "",
    exclude_classes: str = "",
    use_bbox_mask: bool = False,
    imgsz: int = 0,
    plain_preview: bool = False,
    keep_loaded: bool = False,
) -> PredictOutput[float]:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as e:
        # Ultralytics, or a package it needs, is not installed: one clear
        # line first, then the error as before.
        _say_missing_ultralytics(e)
        raise

    _disable_ultralytics_events()
    try:
        # keep_loaded: reuse the model loaded earlier in the same WebUI job.
        model, kept_imgsz = _load_detector(YOLO, model_path, device, keep_loaded)
    except Exception as e:
        msg = _too_old_ultralytics_message(model_path, e)
        if msg is None:
            raise
        _print_ascii(msg)
        raise UltralyticsTooOldError(msg) from e

    # A model whose results have nothing to inpaint (rotated boxes, a label
    # for the whole image): nothing is detected, and the caller says why.
    note = _not_a_detector_note(model_path, model)
    if note:
        return PredictOutput(note=note)

    requested = parse_csv(classes)
    excluded = parse_csv(exclude_classes)

    # Higher detector inference resolution (e.g. 1024) finds smaller / distant
    # parts than the model's own size (640 for the bundled models). 0 keeps the
    # size stored in the checkpoint, which Ultralytics uses when none is passed.
    # Detected boxes are rescaled back to the original image space internally,
    # so nothing downstream changes. imgsz is a universally-supported predict
    # kwarg; the multiclass branch additionally drops it on TypeError alongside
    # `classes=` for very old Ultralytics.
    if is_open_vocab_model(model_path):
        # YOLO-World / YOLOE open-vocab path.
        if requested:
            if is_world_model(model_path):
                try:
                    model.set_classes(requested)
                except Exception as e:  # noqa: BLE001 — never stop the generation
                    # Its text model is not there (the "clip" package, or the
                    # weights it downloads the first time): as for YOLOE.
                    world_note = _typed_classes_note("YOLO-World", model_path, e)
                    return PredictOutput(note=world_note)
            else:
                yoloe_note = _set_yoloe_classes(model, model_path, requested)
                if yoloe_note:
                    # The model would detect its own classes, not the typed
                    # ones: nothing is detected, and the caller says why.
                    return PredictOutput(note=yoloe_note)
        world_kw: dict = {"conf": confidence, "device": device}
        if imgsz:
            world_kw["imgsz"] = imgsz
        pred = model(image, **world_kw)
    else:
        # Multiclass YOLO: include-by-id at inference time, exclude post-hoc.
        kw: dict = {"conf": confidence, "device": device}
        if imgsz:
            kw["imgsz"] = imgsz
        elif kept_imgsz is not None:
            # A kept model reuses its predictor, which keeps every argument
            # of the call before that is not passed again: pass its own size.
            kw["imgsz"] = kept_imgsz
        if requested:
            ids = resolve_class_ids(str(model_path), requested)
            if ids:
                kw["classes"] = ids
        if kept_imgsz is not None:
            # The same for the class filter: a tab without one would keep
            # the classes of the tab before it.
            kw.setdefault("classes", None)
        try:
            pred = model(image, **kw)
        except TypeError:
            # Older ultralytics may not accept the `classes=` / `imgsz=` kwargs
            # — drop them and fall back so detection still runs.
            kw.pop("classes", None)
            kw.pop("imgsz", None)
            if kept_imgsz is not None:
                # Without them a reused predictor would keep the old ones:
                # a new one starts from the defaults, as a new model does.
                try:
                    model.predictor = None
                except Exception:  # noqa: BLE001
                    pass
            pred = model(image, **kw)

        if (
            excluded
            and pred[0].boxes is not None
            and pred[0].boxes.cls is not None
            and len(pred[0].boxes) > 0
        ):
            names = get_model_class_names(str(model_path))
            # Same name matching as the include filter ("Hand" excludes "hand").
            excluded_ids = resolve_class_ids(str(model_path), excluded)
            cls_ids = pred[0].boxes.cls.cpu().numpy().astype(int).tolist()
            keep = [
                i
                for i, cid in enumerate(cls_ids)
                if (names[cid] if 0 <= cid < len(names) else str(cid)) not in excluded
                and str(cid) not in excluded
                and cid not in excluded_ids
            ]
            if not keep:
                return PredictOutput()
            try:
                pred[0] = pred[0][keep]
            except (TypeError, IndexError):
                # Older ultralytics Results may not support list indexing.
                # Subset manually after extraction below by reusing `keep`.
                pred = _SubsetWrapper(pred, keep)

    if getattr(pred[0], "boxes", None) is None:
        # The same for a model whose task did not tell: no boxes to read.
        return PredictOutput(note=_not_a_detector_note(model_path, result=pred[0]))

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if bboxes.size == 0:
        return PredictOutput()
    bboxes = bboxes.tolist()

    # Mask source: by default we trust segmentation masks when the model
    # provides them and fall back to bbox-derived rectangles otherwise.
    # When `use_bbox_mask` is on, we always use bbox-derived rectangles
    # — useful when the seg mask is too tight against the subject and
    # the inpaint needs more breathing room around the edges.
    if use_bbox_mask or pred[0].masks is None:
        masks = create_mask_from_bbox(bboxes, image.size)
    else:
        masks = mask_to_pil(pred[0].masks.data, image.size)

    confidences = pred[0].boxes.conf.cpu().numpy().tolist()

    # Per-box class names (parallel to bboxes/confidences) so the combined
    # multi-tab Detection Preview can label each box like single mode does.
    # Cosmetic only — wrapped so a names lookup never breaks prediction.
    try:
        _names = get_model_class_names(str(model_path))
        _cids = pred[0].boxes.cls.cpu().numpy().astype(int).tolist()
        # No names for YOLO-World (or a model whose names could not be looked
        # up): use the result's own vocabulary, which follows set_classes.
        _result_names = None if _names else getattr(pred[0], "names", None)
        if isinstance(_result_names, (list, tuple)):
            _result_names = dict(enumerate(_result_names))
        if not isinstance(_result_names, dict):
            _result_names = {}
        class_names = [
            _names[c] if 0 <= c < len(_names) else str(_result_names.get(c, c))
            for c in _cids
        ]
    except Exception:  # noqa: BLE001
        class_names = []

    # When `use_bbox_mask` is on we discard the seg silhouette in favour of
    # the bbox rectangle; clear `pred[0].masks` so `plot()` skips the seg
    # overlay and the saved preview accurately reflects what the inpaint
    # will receive (otherwise the saved `*-ad-preview*.png` shows the seg
    # shape while the inpaint actually used the bbox — confusing).
    if use_bbox_mask:
        try:
            pred[0].masks = None
        except (AttributeError, TypeError):
            pass

    preview = pred[0].plot()
    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    preview = Image.fromarray(preview)

    # The Detection preview also asks for the same picture with the boxes and
    # masks but no class names and scores (its "Show labels" switch). Best
    # effort: an Ultralytics whose plot() lacks these options gives none.
    preview_plain = None
    if plain_preview:
        try:
            preview_plain = Image.fromarray(
                cv2.cvtColor(pred[0].plot(labels=False, conf=False), cv2.COLOR_BGR2RGB)
            )
        except Exception:  # noqa: BLE001
            preview_plain = None

    return PredictOutput(
        bboxes=bboxes,
        masks=masks,
        confidences=confidences,
        class_names=class_names,
        preview=preview,
        preview_plain=preview_plain,
    )


class _SubsetWrapper:
    """Fallback for old Ultralytics: subset Results-like objects by index list."""

    def __init__(self, orig, keep: list[int]):
        self._orig = orig
        self._keep = keep
        self._wrapped = _SubsetResults(orig[0], keep)

    def __getitem__(self, idx):
        if idx == 0:
            return self._wrapped
        return self._orig[idx]


class _SubsetResults:
    def __init__(self, r, keep: list[int]):
        self._r = r
        self._keep = keep

    @property
    def boxes(self):
        return _SubsetBoxes(self._r.boxes, self._keep) if self._r.boxes is not None else None

    @property
    def masks(self):
        return _SubsetMasks(self._r.masks, self._keep) if self._r.masks is not None else None

    def plot(self, *a, **kw):
        return self._r.plot(*a, **kw)


class _SubsetBoxes:
    def __init__(self, b, keep: list[int]):
        self._b = b
        self._keep = keep

    @property
    def xyxy(self):
        return self._b.xyxy[self._keep]

    @property
    def conf(self):
        return self._b.conf[self._keep]

    @property
    def cls(self):
        return self._b.cls[self._keep]


class _SubsetMasks:
    def __init__(self, m, keep: list[int]):
        self._m = m
        self._keep = keep

    @property
    def data(self):
        return self._m.data[self._keep]


def _prompt_free(model) -> bool:
    """A YOLOE model whose head has a fixed vocabulary (a "-pf" model under
    another name): Ultralytics refuses to set its classes."""
    try:
        return hasattr(model.model.model[-1], "lrpc")
    except Exception:  # noqa: BLE001
        return False


# The packages of the text models Ultralytics gives typed classes to (CLIP for
# YOLO-World, MobileCLIP for YOLOE), and where Ultralytics itself installs
# each from, with pip, when it is missing.
_TEXT_MODEL_PACKAGES = {
    "clip": "git+https://github.com/ultralytics/CLIP.git",
    "mobileclip": "git+https://github.com/ultralytics/mobileclip.git",
}


def _missing_text_model_package(exc: BaseException) -> str:
    """The text model package whose import failed ("clip"), looked for in
    `exc` and the errors it came from; "" for any other error."""
    seen: set[int] = set()
    err: BaseException | None = exc
    while err is not None and id(err) not in seen:
        seen.add(id(err))
        if isinstance(err, ImportError):
            name = getattr(err, "name", None)
            if not isinstance(name, str):
                found = re.search(r"No module named '([^']+)'", str(err))
                name = found.group(1) if found else ""
            name = name.split(".")[0]
            if name in _TEXT_MODEL_PACKAGES:
                return name
        err = err.__cause__ or err.__context__
    return ""


# A web address in an error text ("https://...", "git+https://..."), up to
# the next space: never a file path, never cut. Not a "file://" one, which is
# a path, nor a drive with two slashes ("C://"). It is tried only where a run
# of such characters starts: a long text without spaces is read once.
_WEB_ADDRESS = r"(?<![\w+.\-])(?!file:)[\w+.\-]{2,}://\S*"
# What a path can come right after, inside a word: 'a.pt', (a.pt), name=a.pt.
_BEFORE_A_PATH = r"""\s'"`(\[{=:,;\u2018\u201c"""
# Left where a known folder was taken out of a text (_file_names_only): the
# word it is in is a piece of a path, whatever else it holds.
_FOLDER_MARK = "\0"
# A web address (the group 1), or a word that can be a piece of a path, from
# where the path starts in it to its end: it starts as a path does (a drive,
# "~/", a "/" that opens a name) or it holds a backslash or the mark of a
# known folder. _path_word tells whether one with a backslash is a path.
_PATH_WORD_RE = re.compile(
    rf"({_WEB_ADDRESS})|(?:^|(?<=[{_BEFORE_A_PATH}]))"
    rf"(?=(?:[A-Za-z]:)?(?:~?/+[^\s/]|[^{_BEFORE_A_PATH}\\\0]*[\\\0]))\S+",
    re.IGNORECASE,
)
# Where a path plainly starts: where a word can start one, with "/", "~", a
# ".\", the long form of a path ("\\?\", "\\?\UNC\"), or backslashes before
# a name and one more piece: a network share ("\\server\share", with the
# backslashes doubled of a repr too), a path from the root without its drive
# ("\Users\name": after ONE backslash the name starts with a capital letter);
# anywhere, with a drive (one letter alone or glued to an option as in
# "-ID:\", a colon, a separator). A name of one letter after a backslash is
# none ("\d", "\t": a character of the message), and neither is a lower-case
# name after one backslash ("\tface\thand", "\x00\x01": escapes of a text).
_PATH_START_RE = re.compile(
    rf"(?:^|(?<=[{_BEFORE_A_PATH}]))"
    r"(?:(?:[A-Za-z]:)?~?/+[^\s/]|~\\|\.\.?\\|\\{2,}\?\\+[^\s\\/]"
    r"|\\{2,}[\w.$\-]{2,}\\+[^\s\\/]|\\[A-Z][\w.$\-]+\\+[^\s\\/])"
    r"|(?:(?<![A-Za-z0-9])|(?<=-[A-Za-z]))[A-Za-z]:[\\/]"
)
# A file name with its extension at the end of a word ("yolo.pt", "a.yaml:",
# "b.ts'?"), not a number ("1.5").
_FILE_NAME_RE = re.compile(r"\w\.(?=[0-9]*[A-Za-z])[A-Za-z0-9]{1,12}\W*$")
# A folder that can be looked for in a text: a drive and a name, or two names
# under the root. Not a name alone ("/tmp", "/app": it is also a piece of
# other texts), which is cut as any path.
_KNOWN_FOLDER_RE = re.compile(r"(?:[A-Za-z]:[\\/]+|[\\/]+[^\\/]+[\\/]+)[^\\/]")
# Where a running WebUI (modules.paths) and the environment name a folder.
_WEBUI_FOLDERS = ("models_path", "data_path", "script_path", "extensions_dir")
_FOLDER_VARIABLES = (
    "USERPROFILE", "HOME", "APPDATA", "LOCALAPPDATA", "OneDrive", "OneDriveCommercial",
    "OneDriveConsumer", "VIRTUAL_ENV", "XDG_CACHE_HOME", "HF_HOME", "TORCH_HOME",
)  # fmt: skip


def _known_folders(model_path: str | Path) -> list[str]:
    """The folders this program is sure of when it is called, for
    _file_names_only: the user's, the working one (the WebUI's), the
    temporary one, Python's, the detector's, this extension's, the ones the
    WebUI tells when it runs and the ones named in the environment. Each as
    it is written and as the disk resolves it (a link, a junction); one that
    cannot be read is left out."""
    webui = sys.modules.get("modules.paths")
    sources = [
        Path.home,
        Path.cwd,
        tempfile.gettempdir,
        lambda: sys.prefix,
        lambda: sys.base_prefix,
        lambda: Path(str(model_path)).parent,
        lambda: Path(__file__).parent.parent,
        *(lambda name=name: getattr(webui, name) for name in _WEBUI_FOLDERS),
        *(lambda name=name: os.environ[name] for name in _FOLDER_VARIABLES),
    ]
    folders = []
    for source in sources:
        try:
            folders.append(str(source()))
            # Not a network share ("\\server\...", "//server/...", or a drive
            # letter given to one): asking the disk about one can wait for a
            # computer that does not answer.
            if not re.match(r"[\\/]{2}", folders[-1]) and not _network_drive(folders[-1]):
                folders.append(str(Path(folders[-1]).resolve()))
        except Exception:  # noqa: BLE001, PERF203 — a note, never an error
            continue
    return folders


def _network_drive(folder: str) -> bool:
    """Windows: True when `folder` is on a drive letter given to a network
    share ("Z:\\..."). Asked to Windows' table of the drive names only
    (QueryDosDeviceW), never to the disk or the network; False on the other
    systems, for any other folder and when it cannot tell."""
    drive = re.match(r"[A-Za-z]:", folder)
    if os.name != "nt" or not drive:
        return False
    try:
        import ctypes

        target = ctypes.create_unicode_buffer(1024)
        if not ctypes.windll.kernel32.QueryDosDeviceW(drive.group().upper(), target, len(target)):
            return False
        # A redirector of the network ("\Device\LanmanRedirector\;Z:...",
        # "\Device\Mup\..."), or a drive made with subst on a share.
        name = target.value.lower()
        return "\\;" in name or name.startswith(("\\device\\mup", "\\??\\unc\\"))
    except Exception:  # noqa: BLE001 — unknown: the folder is resolved, as before
        return False


def _path_word(found: re.Match[str], in_a_folder: list[bool]) -> str:
    """What _file_names_only leaves of a word found by _PATH_WORD_RE: a web
    address whole; a piece of a path cut down to what follows its last
    separator; a word with a backslash that is no path ("'\\t'", "\\d+",
    "face\\nhand") as it is. `in_a_folder[0]`: the last piece of a path
    before this word did not end with a file name, so its folder can go on
    after a space ("D:\\Other Disk\\a.pt"); set here for the next word."""
    if found.group(1):
        return found.group(1)
    word = found.group()
    name = re.split(r"[\\/\0]", word)[-1]
    named = bool(_FILE_NAME_RE.search(name))
    before = ""
    if not (_FOLDER_MARK in word or named or in_a_folder[0]):
        # Two characters back: a drive before a sign that is not part of a
        # path ("<C:\a>" is found from its backslash).
        start = found.start()
        path = _PATH_START_RE.search(found.string, max(start - 2, 0), found.end())
        if path is None:
            return word
        before = word[: max(path.start() - start, 0)]
        before = "" if re.search(r"[\\/]", before) else before
    in_a_folder[0] = not named
    return before + name


def _file_names_only(text: str, folders: list[str] | tuple[str, ...] = ()) -> str:
    """`text` (one line of an error) without the folders of the disk, often
    the user's own: they are not for the console or the status. Nothing is
    guessed about where a path with spaces in it ends.

    1. Each of `folders` (the ones the program is sure of: _known_folders)
       is taken out wherever it is written, the longest first, with or
       without spaces, in any case and with any separator (a backslash, a
       slash, the doubled backslash of a repr).
    2. Then each word that is plainly a piece of a path (_PATH_WORD_RE,
       _path_word) is cut down to what follows its last separator; the signs
       around it stay ('a.pt', "a.pt:"). Plainly: it starts as a path does
       (a drive, "~", a "/" that opens a name, ".\\", a network share, the
       long form "\\\\?\\", "\\" and two names as in "\\Users\\name"), or
       it stood right after one of `folders`, or it holds a backslash and
       ends with a file name ("weights\\text.ts", "models\\yolo12n.pt?"),
       or it holds a backslash and comes after a piece of a path that did
       not end with a file name ("Disk\\My" in "D:\\Other Disk\\My
       Models\\a.pt").

    A word with neither a slash nor a backslash is never touched, unless it
    is inside one of `folders`; a web address stays whole, with whatever is
    written in it; a word with slashes only that does not start as a path
    stays as it is ("and/or", "3/4", "docs/usage/cli"), and so does a word
    with a backslash that is none of the above ("'\\t'", "\\d+",
    "face\\nhand"): it is a character of the message, not a folder. A word
    that still holds a backslash is left whole. The limit: a folder that is
    not one of `folders` and has a space in its name leaves the words of
    that name that hold no separator ("D:\\Other Disk\\a.pt" comes out as
    "Other a.pt"), and, written with slashes, what follows them too
    ("/mnt/Other Disk/b/a.pt" as "Other Disk/b/a.pt")."""
    known = {
        ("[\\\\/]" if folder[0] in "\\/" else "")
        + "[\\\\/]+".join(re.escape(name) for name in re.split(r"[\\/]+", folder) if name)
        for folder in folders
        if _KNOWN_FOLDER_RE.match(folder)
    }
    if known:
        # A mark is left in its place: what follows it in the same word is
        # still a piece of a path, for the second step.
        text = re.sub(
            rf"({_WEB_ADDRESS})|{'|'.join(sorted(known, key=len, reverse=True))}",
            lambda found: found.group(1) or _FOLDER_MARK,
            text,
            flags=re.IGNORECASE,
        )
    in_a_folder = [False]
    return _PATH_WORD_RE.sub(lambda found: _path_word(found, in_a_folder), text)


def _typed_classes_note(kind: str, model_path: str | Path, exc: BaseException) -> str:
    """Why a YOLO-World or YOLOE detector could not take the typed classes, for
    PredictOutput.note. Ultralytics installs the "clip" package by itself the
    first time: when that did not work (no network, its auto-install switched
    off) the note names the package and the command that installs it. The
    command is not at the end of the note: what follows the note (a full
    stop, the next sentence of a status) would be copied with its address."""
    name = Path(str(model_path)).name
    # One line: an error text from torch or a download often has several.
    # Its paths as file names, before the cut: half a path is still a path.
    line = (str(exc).strip().splitlines() or [""])[0]
    detail = _file_names_only(line, _known_folders(model_path))[:160]
    error = f"{type(exc).__name__}: {detail}"
    package = _missing_text_model_package(exc)
    if package:
        return (
            f"the {kind} detector {name!r} could not take the typed classes: the"
            f" Python package {package!r} is missing and Ultralytics could not"
            f" install it by itself ({error}). Run python -m pip install"
            f" {_TEXT_MODEL_PACKAGES[package]} with the WebUI's Python (its"
            " virtual environment)."
        )
    return f"the {kind} detector {name!r} could not take the typed classes ({error})."


def _set_yoloe_classes(model, model_path: str | Path, names: list[str]) -> str:
    """Give a YOLOE detector the typed classes as text prompts, the way
    Ultralytics documents it: set_classes(names, get_text_pe(names)). "" when
    done; when this Ultralytics or this model cannot, the reason, for
    PredictOutput.note. The first time, Ultralytics itself fetches its text
    model for this."""
    name = Path(str(model_path)).name
    set_classes = getattr(model, "set_classes", None)
    get_text_pe = getattr(model, "get_text_pe", None)
    if not callable(set_classes) or not callable(get_text_pe) or _prompt_free(model):
        reason = (
            "it is a prompt-free model"
            if callable(set_classes) and callable(get_text_pe)
            else "this Ultralytics has no text prompts for YOLOE (update it)"
        )
        return f"the YOLOE detector {name!r} cannot take the typed classes: {reason}."
    try:
        set_classes(names, get_text_pe(names))
    except Exception as e:  # noqa: BLE001 — never stop the generation
        return _typed_classes_note("YOLOE", model_path, e)
    return ""


def apply_classes(model: YOLO | YOLOWorld, model_path: str | Path, classes: str):
    """Backward-compatible shim: only sets classes on YOLO-World and YOLOE
    (text prompts) models."""
    if not classes or not is_open_vocab_model(model_path):
        return
    parsed = parse_csv(classes)
    if parsed:
        if is_world_model(model_path):
            model.set_classes(parsed)
        else:
            note = _set_yoloe_classes(model, model_path, parsed)
            if note:
                _print_ascii(f"[-] ADetailer: {note}")


def mask_to_pil(masks: torch.Tensor, shape: tuple[int, int]) -> list[Image.Image]:
    """
    Parameters
    ----------
    masks: torch.Tensor, dtype=torch.float32 or torch.uint8, shape=(N, H, W).
        uint8 tensor is expected to have values 0 or 1 (not 0-255).

    shape: tuple[int, int]
        (W, H) of the original image
    """
    masks = masks.float()
    n, mask_height, mask_width = masks.shape
    width, height = shape
    # Standard Ultralytics segmentation returns masks in letterboxed inference
    # coordinates, while boxes are already in original-image coordinates.
    # Remove the centered padding before resizing (same rounding as scale_masks)
    # so a rectangular image's mask stays aligned with the detected subject.
    gain = min(mask_height / height, mask_width / width)
    pad_x = (mask_width - round(width * gain)) / 2
    pad_y = (mask_height - round(height * gain)) / 2
    left, top = round(pad_x - 0.1), round(pad_y - 0.1)
    right = mask_width - round(pad_x + 0.1)
    bottom = mask_height - round(pad_y + 0.1)
    masks = masks[:, top:bottom, left:right]
    return [to_pil_image(masks[i], mode="L").resize(shape) for i in range(n)]
