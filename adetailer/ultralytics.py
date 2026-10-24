from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from adetailer import PredictOutput
from adetailer.classes import (
    get_model_class_names,
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
) -> PredictOutput[float]:
    from ultralytics import YOLO

    _disable_ultralytics_events()
    try:
        model = YOLO(model_path)
    except Exception as e:
        msg = _too_old_ultralytics_message(model_path, e)
        if msg is None:
            raise
        _print_ascii(msg)
        raise UltralyticsTooOldError(msg) from e

    requested = parse_csv(classes)
    excluded = parse_csv(exclude_classes)

    # Higher detector inference resolution (e.g. 1024) finds smaller / distant
    # parts than the model's own size (640 for the bundled models). 0 keeps the
    # size stored in the checkpoint, which Ultralytics uses when none is passed.
    # Detected boxes are rescaled back to the original image space internally,
    # so nothing downstream changes. imgsz is a universally-supported predict
    # kwarg; the multiclass branch additionally drops it on TypeError alongside
    # `classes=` for very old Ultralytics.
    if is_world_model(model_path):
        # YOLO-World open-vocab path (unchanged behavior).
        if requested:
            model.set_classes(requested)
        world_kw: dict = {"conf": confidence, "device": device}
        if imgsz:
            world_kw["imgsz"] = imgsz
        pred = model(image, **world_kw)
    else:
        # Multiclass YOLO: include-by-id at inference time, exclude post-hoc.
        kw: dict = {"conf": confidence, "device": device}
        if imgsz:
            kw["imgsz"] = imgsz
        if requested:
            ids = resolve_class_ids(str(model_path), requested)
            if ids:
                kw["classes"] = ids
        try:
            pred = model(image, **kw)
        except TypeError:
            # Older ultralytics may not accept the `classes=` / `imgsz=` kwargs
            # — drop them and fall back so detection still runs.
            kw.pop("classes", None)
            kw.pop("imgsz", None)
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


def apply_classes(model: YOLO | YOLOWorld, model_path: str | Path, classes: str):
    """Backward-compatible shim: only sets classes on YOLO-World models."""
    if not classes or not is_world_model(model_path):
        return
    parsed = parse_csv(classes)
    if parsed:
        model.set_classes(parsed)


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
