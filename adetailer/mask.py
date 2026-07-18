from __future__ import annotations

from enum import IntEnum
from functools import partial, reduce
from math import dist
from typing import Any, TypeVar

import cv2
import numpy as np
from PIL import Image, ImageChops

from adetailer.args import MASK_MERGE_INVERT
from adetailer.common import PredictOutput, ensure_pil_image


class SortBy(IntEnum):
    NONE = 0
    LEFT_TO_RIGHT = 1
    CENTER_TO_EDGE = 2
    AREA = 3


class MergeInvert(IntEnum):
    NONE = 0
    MERGE = 1
    MERGE_INVERT = 2


T = TypeVar("T", int, float)


def _dilate(arr: np.ndarray, value: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    return cv2.dilate(arr, kernel, iterations=1)


def _erode(arr: np.ndarray, value: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    return cv2.erode(arr, kernel, iterations=1)


def dilate_erode(img: Image.Image, value: int) -> Image.Image:
    """
    The dilate_erode function takes an image and a value.
    If the value is positive, it dilates the image by that amount.
    If the value is negative, it erodes the image by that amount.

    Parameters
    ----------
        img: PIL.Image.Image
            the image to be processed
        value: int
            kernel size of dilation or erosion

    Returns
    -------
        PIL.Image.Image
            The image that has been dilated or eroded
    """
    if value == 0:
        return img

    arr = np.array(img)
    arr = _dilate(arr, value) if value > 0 else _erode(arr, -value)

    return Image.fromarray(arr)


def offset(img: Image.Image, x: int = 0, y: int = 0) -> Image.Image:
    """
    The offset function takes an image and offsets it by a given x(→) and y(↑) value.

    Parameters
    ----------
        mask: Image.Image
            Pass the mask image to the function
        x: int
            →
        y: int
            ↑

    Returns
    -------
        PIL.Image.Image
            A new image that is offset by x and y
    """
    return ImageChops.offset(img, x, -y)


def is_all_black(img: Image.Image | np.ndarray) -> bool:
    if isinstance(img, Image.Image):
        img = np.array(ensure_pil_image(img, "L"))
    return cv2.countNonZero(img) == 0


def has_intersection(im1: Any, im2: Any) -> bool:
    arr1 = np.array(ensure_pil_image(im1, "L"))
    arr2 = np.array(ensure_pil_image(im2, "L"))
    return not is_all_black(cv2.bitwise_and(arr1, arr2))


def bbox_area(bbox: list[T]) -> T:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def mask_preprocess(
    masks: list[Image.Image],
    kernel: int = 0,
    x_offset: int = 0,
    y_offset: int = 0,
    merge_invert: int | MergeInvert | str = MergeInvert.NONE,
) -> list[Image.Image]:
    """
    The mask_preprocess function takes a list of masks and preprocesses them.
    It dilates and erodes the masks, and offsets them by x_offset and y_offset.

    Parameters
    ----------
        masks: list[Image.Image]
            A list of masks
        kernel: int
            kernel size of dilation or erosion
        x_offset: int
            →
        y_offset: int
            ↑

    Returns
    -------
        list[Image.Image]
            A list of processed masks
    """
    if not masks:
        return []

    if x_offset != 0 or y_offset != 0:
        masks = [offset(m, x_offset, y_offset) for m in masks]

    if kernel != 0:
        masks = [dilate_erode(m, kernel) for m in masks]
        masks = [m for m in masks if not is_all_black(m)]

    return mask_merge_invert(masks, mode=merge_invert)


# Bbox sorting
def _key_left_to_right(bbox: list[T]) -> T:
    """
    Left to right

    Parameters
    ----------
    bbox: list[int] | list[float]
        list of [x1, y1, x2, y2]
    """
    return bbox[0]


def _key_center_to_edge(bbox: list[T], *, center: tuple[float, float]) -> float:
    """
    Center to edge

    Parameters
    ----------
    bbox: list[int] | list[float]
        list of [x1, y1, x2, y2]
    image: Image.Image
        the image
    """
    bbox_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    return dist(center, bbox_center)


def _key_area(bbox: list[T]) -> T:
    """
    Large to small

    Parameters
    ----------
    bbox: list[int] | list[float]
        list of [x1, y1, x2, y2]
    """
    return -bbox_area(bbox)


def sort_bboxes(
    pred: PredictOutput[T], order: int | SortBy = SortBy.NONE
) -> PredictOutput[T]:
    if order == SortBy.NONE or len(pred.bboxes) <= 1:
        return pred

    if order == SortBy.LEFT_TO_RIGHT:
        key = _key_left_to_right
    elif order == SortBy.CENTER_TO_EDGE:
        width, height = pred.preview.size
        center = (width / 2, height / 2)
        key = partial(_key_center_to_edge, center=center)
    elif order == SortBy.AREA:
        key = _key_area
    else:
        raise RuntimeError

    items = len(pred.bboxes)
    idx = sorted(range(items), key=lambda i: key(pred.bboxes[i]))
    pred.bboxes = [pred.bboxes[i] for i in idx]
    pred.masks = [pred.masks[i] for i in idx]
    if pred.class_names:
        pred.class_names = [pred.class_names[i] for i in idx]
    return pred


# Filter by ratio
def is_in_ratio(bbox: list[T], low: float, high: float, orig_area: int) -> bool:
    area = bbox_area(bbox)
    return low <= area / orig_area <= high


def filter_by_ratio(
    pred: PredictOutput[T], low: float, high: float
) -> PredictOutput[T]:
    if not pred.bboxes:
        return pred

    w, h = pred.preview.size
    orig_area = w * h
    items = len(pred.bboxes)
    idx = [i for i in range(items) if is_in_ratio(pred.bboxes[i], low, high, orig_area)]
    pred.bboxes = [pred.bboxes[i] for i in idx]
    pred.masks = [pred.masks[i] for i in idx]
    pred.confidences = [pred.confidences[i] for i in idx]
    if pred.class_names:
        pred.class_names = [pred.class_names[i] for i in idx]
    return pred


def parse_indices(spec: str, n: int) -> list[int] | None:
    """Parse a 1-based selection string into sorted, de-duped 0-based indices
    within ``[0, n)``.

    Accepts comma- (or semicolon-) separated single numbers and inclusive
    ranges, e.g. ``"1,3,5"`` or ``"1-3,5"``. Whitespace is ignored; tokens that
    aren't a positive int or an ``a-b`` range are skipped; indices outside
    ``[1, n]`` are dropped. Returns ``None`` when the string contains no usable
    number at all — the caller treats that (and a blank string) as "keep all".
    An in-range parse that resolves to nothing (e.g. only out-of-range numbers)
    returns an empty list, i.e. "keep none".
    """
    seen: set[int] = set()
    order: list[int] = []
    found_any = False
    for tok in spec.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok[1:]:  # inclusive range like "2-4" (a leading "-" is not a range)
            a, _, b = tok.partition("-")
            try:
                lo, hi = int(a), int(b)
            except ValueError:
                continue
            found_any = True
            if lo > hi:
                lo, hi = hi, lo
            values: range | tuple[int, ...] = range(lo, hi + 1)
        else:
            try:
                v = int(tok)
            except ValueError:
                continue
            found_any = True
            values = (v,)
        for one in values:
            i = one - 1  # 1-based -> 0-based
            if 0 <= i < n and i not in seen:
                seen.add(i)
                order.append(i)
    if not found_any:
        return None
    return sorted(order)


def filter_by_indices(pred: PredictOutput[T], spec: str) -> PredictOutput[T]:
    """Keep only the detections whose 1-based number appears in ``spec``.

    The numbers refer to the RAW detector order (the order shown by the numbered
    Detection preview), so this must run BEFORE any ratio/top-k/sort step. A
    blank or unparseable spec is a no-op (all detections kept).
    """
    if not spec or not spec.strip() or not pred.bboxes:
        return pred
    keep = parse_indices(spec, len(pred.bboxes))
    if keep is None:  # nothing parseable -> keep all
        return pred
    pred.bboxes = [pred.bboxes[i] for i in keep]
    pred.masks = [pred.masks[i] for i in keep]
    if pred.confidences:
        pred.confidences = [pred.confidences[i] for i in keep]
    if pred.class_names:
        pred.class_names = [pred.class_names[i] for i in keep]
    return pred


def filter_k_largest(pred: PredictOutput[T], k: int = 0) -> PredictOutput[T]:
    if not pred.bboxes or k == 0:
        return pred
    areas = [bbox_area(bbox) for bbox in pred.bboxes]
    idx = np.argsort(areas)[-k:]
    idx = idx[::-1]
    pred.bboxes = [pred.bboxes[i] for i in idx]
    pred.masks = [pred.masks[i] for i in idx]
    pred.confidences = [pred.confidences[i] for i in idx]
    if pred.class_names:
        pred.class_names = [pred.class_names[i] for i in idx]
    return pred


def filter_k_most_confident(pred: PredictOutput[T], k: int = 0) -> PredictOutput[T]:
    if not pred.bboxes or not pred.confidences or k == 0:
        return pred
    idx = np.argsort(pred.confidences)[-k:]
    idx = idx[::-1]
    pred.bboxes = [pred.bboxes[i] for i in idx]
    pred.masks = [pred.masks[i] for i in idx]
    pred.confidences = [pred.confidences[i] for i in idx]
    if pred.class_names:
        pred.class_names = [pred.class_names[i] for i in idx]
    return pred


def filter_k_by(
    pred: PredictOutput[T], k: int = 0, by: str = "Area"
) -> PredictOutput[T]:
    if by == "Area":
        return filter_k_largest(pred, k)
    if by == "Confidence":
        return filter_k_most_confident(pred, k)
    raise RuntimeError


# Merge / Invert
def mask_merge(masks: list[Image.Image]) -> list[Image.Image]:
    arrs = [np.array(m) for m in masks]
    arr = reduce(cv2.bitwise_or, arrs)
    return [Image.fromarray(arr)]


def mask_invert(masks: list[Image.Image]) -> list[Image.Image]:
    return [ImageChops.invert(m) for m in masks]


def mask_merge_invert(
    masks: list[Image.Image], mode: int | MergeInvert | str
) -> list[Image.Image]:
    if isinstance(mode, str):
        mode = MASK_MERGE_INVERT.index(mode)

    if mode == MergeInvert.NONE or not masks:
        return masks

    if mode == MergeInvert.MERGE:
        return mask_merge(masks)

    if mode == MergeInvert.MERGE_INVERT:
        merged = mask_merge(masks)
        return mask_invert(merged)

    raise RuntimeError
