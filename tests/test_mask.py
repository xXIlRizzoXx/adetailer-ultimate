import math

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from adetailer.common import PredictOutput
from adetailer.mask import (
    bbox_area,
    bbox_iou,
    bbox_ratio,
    bbox_ratio_label,
    dilate_erode,
    filter_by_ratio,
    filter_overlapping,
    has_intersection,
    is_all_black,
    is_in_ratio,
    mask_invert,
    mask_merge,
    mask_preprocess,
    offset,
    parse_indices,
    SortBy,
    sort_bboxes,
)


def test_dilate_positive_value():
    img = Image.new("L", (10, 10), color="black")
    draw = ImageDraw.Draw(img)
    draw.rectangle((3, 3, 5, 5), fill="white")
    value = 3

    result = dilate_erode(img, value)

    assert isinstance(result, Image.Image)
    assert result.size == (10, 10)

    expect = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 255, 255, 255, 255, 255, 0, 0, 0],
            [0, 0, 255, 255, 255, 255, 255, 0, 0, 0],
            [0, 0, 255, 255, 255, 255, 255, 0, 0, 0],
            [0, 0, 255, 255, 255, 255, 255, 0, 0, 0],
            [0, 0, 255, 255, 255, 255, 255, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(np.array(result), expect)


def test_offset():
    img = Image.new("L", (10, 10), color="black")
    draw = ImageDraw.Draw(img)
    draw.rectangle((4, 4, 5, 5), fill="white")

    result = offset(img, x=1, y=2)

    assert isinstance(result, Image.Image)
    assert result.size == (10, 10)

    expect = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 255, 255, 0, 0, 0],
            [0, 0, 0, 0, 0, 255, 255, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(np.array(result), expect)


@pytest.mark.parametrize(
    ("bbox", "x", "y", "expected"),
    [
        ((7, 4, 9, 5), 2, 0, (9, 4, 10, 6)),
        ((0, 4, 2, 5), -2, 0, (0, 4, 1, 6)),
        ((4, 0, 5, 2), 0, 2, (4, 0, 6, 1)),
        ((4, 7, 5, 9), 0, -2, (4, 9, 6, 10)),
        ((4, 4, 5, 5), 10, 0, None),
        ((4, 4, 5, 5), 0, -10, None),
    ],
)
def test_offset_clips_at_image_edges(bbox, x, y, expected):
    mask = Image.new("L", (10, 10), 0)
    ImageDraw.Draw(mask).rectangle(bbox, fill=255)

    shifted = offset(mask, x=x, y=y)

    assert shifted.getbbox() == expected
    assert mask.getbbox() == (bbox[0], bbox[1], bbox[2] + 1, bbox[3] + 1)


@pytest.mark.parametrize("merge_invert", ["None", "Merge", "Merge and Invert"])
def test_offset_drops_off_canvas_masks_and_keeps_source_indices(merge_invert):
    masks = [Image.new("L", (10, 10), 0) for _ in range(2)]
    ImageDraw.Draw(masks[0]).rectangle((8, 4, 9, 5), fill=255)
    ImageDraw.Draw(masks[1]).rectangle((3, 4, 4, 5), fill=255)

    processed, groups = mask_preprocess(
        masks, x_offset=2, merge_invert=merge_invert
    )

    assert len(processed) == 1
    assert groups == [[1]]


def test_offset_drops_all_off_canvas_masks():
    mask = Image.new("L", (10, 10), 255)

    assert mask_preprocess([mask], x_offset=10) == ([], [])


def _box_mask(box):
    mask = Image.new("L", (10, 10), 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    return mask


@pytest.mark.parametrize(
    ("boxes", "kernel"),
    [
        ([(0, 0, 9, 9)], 4),  # a box clamped to the whole frame
        ([(0, 0, 9, 9)], 0),
        ([(1, 1, 8, 8)], 4),  # dilation fills the last pixels
        ([(0, 0, 4, 9), (5, 0, 9, 9)], 0),  # two halves cover the frame
    ],
)
def test_merge_invert_of_detections_filling_the_frame_inpaints_nothing(
    boxes, kernel
):
    # The inverted mask would be blank, and the host repaints the whole image
    # for a blank mask instead of nothing.
    masks = [_box_mask(box) for box in boxes]

    assert mask_preprocess(
        masks, kernel=kernel, merge_invert="Merge and Invert"
    ) == ([], [])


def test_merge_invert_keeps_the_background_around_detections():
    masks = [_box_mask((0, 0, 4, 9)), _box_mask((5, 0, 8, 9))]

    processed, groups = mask_preprocess(masks, merge_invert="Merge and Invert")

    assert len(processed) == 1
    assert processed[0].getbbox() == (9, 0, 10, 10)
    assert groups == [[0, 1]]


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("1,3;2,2", [0, 1, 2]),
        ("3-1", [0, 1, 2]),
        ("0-2", [0, 1]),
        ("4-7", []),
        ("-2,0", []),
        ("bad,1-x", None),
        ("", None),
        # The Detection preview labels boxes "#1", "#2", ...: typing the label,
        # or separating numbers with spaces, must select those detections
        # instead of silently keeping every one of them.
        ("#2", [1]),
        ("#1,#3", [0, 2]),
        ("2 3", [1, 2]),
        ("1, #3", [0, 2]),
        ("#1-#2", [0, 1]),
        (" #2 ", [1]),
        # A label typed whole, with its class name and its share of the
        # picture, still picks only its number.
        ("#1 12%", [0]),
        ("#2 eyes 0.4%, #3 <0.1%", [1, 2]),
        ("#3 100%", [2]),
        # A spaced range stays a range (not the two numbers 1 and 3).
        ("1 - 3", [0, 1, 2]),
        ("1 -3", [0, 1, 2]),
        ("#", None),
        # A Chinese or Japanese IME types full-width or CJK commas, dashes and
        # digits: they select those detections instead of keeping every one.
        ("1\uff0c3", [0, 2]),
        ("1\u30013", [0, 2]),
        ("\uff11\uff0c\uff13", [0, 2]),
        ("2\uff1b3", [1, 2]),
        ("\uff032", [1]),
        ("1\uff5e3", [0, 1, 2]),
        ("1\u301c3", [0, 1, 2]),
        ("1\u20133", [0, 1, 2]),
        ("1\uff0d3", [0, 1, 2]),
        ("1 \u2014 3", [0, 1, 2]),
        # The Japanese long-vowel mark (the minus key in Japanese mode) and the
        # middle dot, full-width or half-width, work like "-" and ",".
        ("1\u30fc3", [0, 1, 2]),
        ("\uff11\u30fc\uff13", [0, 1, 2]),
        ("1\uff703", [0, 1, 2]),
        ("1\u30fb3", [0, 2]),
        ("1\uff653", [0, 2]),
        ("\u30fc2", []),
        # A doubled dash (a Chinese IME types two em dashes for Shift and
        # minus) is still a range, instead of silently keeping only the first
        # number, or a detection before it.
        ("1\u2014\u20143", [0, 1, 2]),
        ("1\uff0d\uff0d3", [0, 1, 2]),
        ("1--3", [0, 1, 2]),
        ("2\u2014\u20143", [1, 2]),
        ("1 - - 3", [0, 1, 2]),
        ("1\u30fc\u30fc3", [0, 1, 2]),
        ("3--1", [0, 1, 2]),
        ("1--", None),
    ],
)
def test_parse_indices_preserves_selection_rules(spec, expected):
    assert parse_indices(spec, 3) == expected


@pytest.mark.parametrize(
    "spec", ["1-999999999999", "999999999999-1", "1--999999999999"]
)
def test_parse_indices_bounds_range_before_iteration(monkeypatch, spec):
    import adetailer.mask as mask_module

    def bounded_range(start, stop):
        # Fail immediately on the regression; never really iterate a huge range.
        assert stop - start <= 3
        return range(start, stop)

    monkeypatch.setattr(mask_module, "range", bounded_range, raising=False)
    assert parse_indices(spec, 3) == [0, 1, 2]


class TestIsAllBlack:
    def test_is_all_black_1(self):
        img = Image.new("L", (10, 10), color="black")
        assert is_all_black(img)

        draw = ImageDraw.Draw(img)
        draw.rectangle((4, 4, 5, 5), fill="white")
        assert not is_all_black(img)

    def test_is_all_black_2(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        assert is_all_black(img)

        img[4:6, 4:6] = 255
        assert not is_all_black(img)

    def test_is_all_black_rgb_image_pil(self):
        img = Image.new("RGB", (10, 10), color="red")
        assert not is_all_black(img)

        img = Image.new("RGBA", (10, 10), color="red")
        assert not is_all_black(img)

    def test_is_all_black_rgb_image_numpy(self):
        img = np.full((10, 10, 4), 127, dtype=np.uint8)
        with pytest.raises(cv2.error):
            is_all_black(img)

        img = np.full((4, 10, 10), 0.5, dtype=np.float32)
        with pytest.raises(cv2.error):
            is_all_black(img)


class TestHasIntersection:
    def test_has_intersection_1(self):
        arr1 = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        arr2 = arr1.copy()
        assert not has_intersection(arr1, arr2)

    def test_has_intersection_2(self):
        arr1 = np.array(
            [
                [0, 0, 0, 0],
                [0, 255, 255, 0],
                [0, 255, 255, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        arr2 = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 255, 255],
                [0, 0, 255, 255],
            ],
            dtype=np.uint8,
        )
        assert has_intersection(arr1, arr2)

        arr3 = np.array(
            [
                [255, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 255],
                [0, 0, 255, 255],
            ],
            dtype=np.uint8,
        )
        assert not has_intersection(arr1, arr3)

    def test_has_intersection_3(self):
        img1 = Image.new("L", (10, 10), color="black")
        draw1 = ImageDraw.Draw(img1)
        draw1.rectangle((3, 3, 5, 5), fill="white")
        img2 = Image.new("L", (10, 10), color="black")
        draw2 = ImageDraw.Draw(img2)
        draw2.rectangle((6, 6, 8, 8), fill="white")
        assert not has_intersection(img1, img2)

        img3 = Image.new("L", (10, 10), color="black")
        draw3 = ImageDraw.Draw(img3)
        draw3.rectangle((2, 2, 8, 8), fill="white")
        assert has_intersection(img1, img3)

    def test_has_intersection_4(self):
        img1 = Image.new("RGB", (10, 10), color="black")
        draw1 = ImageDraw.Draw(img1)
        draw1.rectangle((3, 3, 5, 5), fill="white")
        img2 = Image.new("RGBA", (10, 10), color="black")
        draw2 = ImageDraw.Draw(img2)
        draw2.rectangle((2, 2, 8, 8), fill="white")
        assert has_intersection(img1, img2)

    def test_has_intersection_5(self):
        img1 = Image.new("RGB", (10, 10), color="black")
        draw1 = ImageDraw.Draw(img1)
        draw1.rectangle((4, 4, 5, 5), fill="white")
        img2 = np.full((10, 10, 4), 255, dtype=np.uint8)
        assert has_intersection(img1, img2)


def test_bbox_area():
    bbox = [0.0, 0.0, 10.0, 10.0]
    assert bbox_area(bbox) == 100


@pytest.mark.parametrize(
    ("bbox", "size", "label"),
    [
        # As large as the picture, square or not.
        ([0, 0, 512, 512], (512, 512), "100%"),
        ([0, 0, 640, 480], (640, 480), "100%"),
        ([0, 0, 480, 640], (480, 640), "100%"),
        # Half as wide and half as tall is a quarter, on any shape.
        ([0, 0, 50, 50], (100, 100), "25.0%"),
        ([256, 96, 768, 480], (1024, 768), "25.0%"),
        ([0.5, 0.5, 100.5, 50.5], (200, 100), "25.0%"),
        # Always one decimal.
        ([0, 0, 400, 300], (1000, 1000), "12.0%"),
        ([0, 0, 200, 170], (1000, 1000), "3.4%"),
        ([100, 50, 300, 250], (2000, 500), "4.0%"),
        ([0, 0, 80, 50], (1000, 1000), "0.4%"),
        # Cut down to a step of the sliders, never rounded up: 11.68% set as
        # "12%" (0.12) on "Mask min area ratio" dropped its own box.
        ([0, 0, 400, 292], (1000, 1000), "11.6%"),
        ([0, 0, 1234, 100], (1000, 1000), "12.3%"),
        ([0, 0, 47, 100], (1000, 1000), "0.4%"),
        ([0, 0, 996, 100], (1000, 1000), "9.9%"),
        ([0, 0, 999, 1000], (1000, 1000), "99.9%"),
        ([0, 0, 9996, 100], (1000, 1000), "99.9%"),
        # Exactly on a step: 0.116 is "11.6%", not "11.5%".
        ([0, 0, 116, 1000], (1000, 1000), "11.6%"),
        ([0, 0, 290, 100], (100, 1000), "29.0%"),
        ([0, 0, 1, 100], (100, 1000), "0.1%"),
        # Smaller than one step of the sliders (0.001 = 0.1%).
        ([0, 0, 2, 2], (1024, 1024), "<0.1%"),
        ([10, 10, 10, 40], (64, 64), "0%"),
        # Past the picture's edge: still the value the filter works out.
        ([0, 0, 110, 100], (100, 100), "110.0%"),
        # Nothing to write.
        ([1, 2], (100, 100), ""),
        (None, (100, 100), ""),
        ([0, 0, 10, 10], (0, 100), ""),
        ([0, 0, 10, 10], None, ""),
        ([30, 0, 10, 10], (100, 100), ""),
        ([0, 0, float("nan"), 10], (100, 100), ""),
        ([0, 0, float("inf"), 10], (100, 100), ""),
    ],
)
def test_bbox_ratio_label(bbox, size, label):
    # The Detection preview writes a box's share of the picture next to its
    # number, to be set on "Mask min area ratio" / "Mask max area ratio".
    assert bbox_ratio_label(bbox, size) == label


def _slider(label, steps=0):
    """The value read on a label, typed on a "Mask min/max area ratio" slider
    (11.6% is 0.116), ``steps`` steps of the slider (0.001) further up."""
    whole, _, tenth = label[:-1].partition(".")
    count = int(whole) * 10 + int(tenth or 0) + steps
    return float(f"{count // 1000}.{count % 1000:03d}")


def _ratio_label_boxes():
    """(box, picture size) pairs: whole-pixel and fractional boxes on pictures
    of several shapes, and boxes exactly on a step of the sliders."""
    for width, height in ((640, 480), (768, 1344), (1024, 1024), (1920, 816)):
        for box_w in range(4, width + 1, 37):
            for box_h in range(4, height + 1, 53):
                yield [0.0, 0.0, float(box_w), float(box_h)], (width, height)
    # The scan of the review that found the rounded label.
    for box_w in range(8, 1024, 7):
        for box_h in range(8, 1024, 11):
            yield [0, 0, box_w, box_h], (1024, 1024)
    for box in ([0, 0, 400, 304], [0, 0, 400, 322], [0, 0, 190, 190], [0, 0, 60, 60]):
        yield box, (1024, 1024)
    # Exactly on a step: 0.001, 0.002, ... 1.0 of the picture.
    for steps in range(1, 1001):
        yield [0, 0, steps, 1000], (1000, 1000)
        yield [0.0, 0.0, steps / 8, 80.0], (100, 100)
    # Fractional corners, as a detector gives them.
    rng = np.random.default_rng(620)
    for x1, y1, x2, y2 in rng.uniform(0, 1, (4000, 4)).tolist():
        yield (
            [x1 * 300, y1 * 400, 300 + x2 * 468, 400 + y2 * 624], (768, 1024)
        )


def test_bbox_ratio_label_set_on_the_sliders_keeps_its_own_box():
    # The label was rounded: a box at 11.68% read "12%", and "Mask min area
    # ratio" at 0.12 dropped that very box (half of the boxes of a scan). It is
    # now cut down to a step of the sliders, so the value read keeps its box as
    # the minimum, and as the maximum one step further up.
    kinds = set()
    checked = 0
    for box, (width, height) in _ratio_label_boxes():
        area = width * height
        label = bbox_ratio_label(box, (width, height))
        ratio = bbox_ratio(box, area)
        if label == "<0.1%":
            # Below every setting of the sliders but 0.
            assert 0 < ratio < 0.001, (box, label)
            assert not is_in_ratio(box, 0.001, 1.0, area)
            assert is_in_ratio(box, 0.0, 0.001, area)
            kinds.add("tiny")
            continue
        if label == "100%":
            assert is_in_ratio(box, 1.0, 1.0, area), (box, label)
            kinds.add("full")
            continue
        # Always one decimal.
        whole, dot, tenth = label[:-1].partition(".")
        assert label.endswith("%") and dot, (box, label)
        assert whole.isdigit() and tenth.isdigit() and len(tenth) == 1, (box, label)
        low, high = _slider(label), _slider(label, 1)
        assert round(high - low, 6) == 0.001
        # As the minimum it keeps its own box; one step more would not.
        assert is_in_ratio(box, low, 1.0, area), (box, label)
        assert not is_in_ratio(box, high, 1.0, area), (box, label)
        # As the maximum plus one step it keeps its own box.
        assert is_in_ratio(box, 0.0, high, area), (box, label)
        assert is_in_ratio(box, low, high, area), (box, label)
        # Through the filter itself.
        for lowest, highest in ((low, 1.0), (0.0, high)):
            pred = PredictOutput(
                bboxes=[box], masks=["m"], confidences=[0.9],
                preview=Image.new("1", (width, height)),
            )
            assert filter_by_ratio(pred, lowest, highest).masks == ["m"], (box, label)
        kinds.add("decimal")
        checked += 1
    # Premise: the scan has every kind of label, and many boxes.
    assert kinds == {"tiny", "decimal", "full"}
    assert checked > 10000


def test_bbox_ratio_label_of_a_box_on_a_step_is_that_step():
    # 0.57 of the picture is "57.0%": cut from the percentage it would read
    # one step less on 81 of the steps (100 * 0.57 is 56.99999999999999).
    for steps in range(1, 1000):
        label = bbox_ratio_label([0, 0, steps, 1000], (1000, 1000))
        assert label == f"{steps // 10}.{steps % 10}%", steps
    assert bbox_ratio_label([0, 0, 1000, 1000], (1000, 1000)) == "100%"


def test_bbox_ratio_label_of_a_box_a_hair_below_a_step_is_the_step_before():
    # A fractional box can be the smallest possible amount below a step:
    # the filter set to that step drops it, so the label must not name it
    # (the ratio times 1000 alone gives the step itself for 0.117, 0.203...).
    for steps in range(2, 1001):
        ratio = math.nextafter(steps / 1000, 0.0)
        box = [0.0, 0.0, ratio, 1.0]
        assert bbox_ratio(box, 1) == ratio
        assert not is_in_ratio(box, steps / 1000, 1.0, 1)
        label = bbox_ratio_label(box, (1, 1))
        assert label == f"{(steps - 1) // 10}.{(steps - 1) % 10}%", steps
        assert is_in_ratio(box, _slider(label), _slider(label, 1), 1)


@pytest.mark.parametrize("size", [(400, 200), (200, 400)])
def test_the_ratio_read_on_the_preview_works_on_the_sliders(size):
    # 30.0% read on the label is 0.3 on the sliders: as the minimum it keeps
    # that box and the larger one, as the maximum that box and the smaller.
    def boxes():
        found = [[0, 0, 400, 200], [0, 0, 200, 120], [100, 100, 120, 116]]
        if size[0] < size[1]:
            found = [[y1, x1, y2, x2] for x1, y1, x2, y2 in found]
        return found

    assert [bbox_ratio_label(b, size) for b in boxes()] == ["100%", "30.0%", "0.4%"]

    def kept(low, high):
        pred = PredictOutput(
            bboxes=boxes(), masks=["a", "b", "c"], confidences=[0.9, 0.8, 0.7],
            preview=Image.new("RGB", size),
        )
        return filter_by_ratio(pred, low, high).masks

    assert kept(0.3, 1.0) == ["a", "b"]
    assert kept(0.0, 0.3) == ["b", "c"]
    assert kept(0.004, 0.3) == ["b", "c"]
    assert kept(0.005, 0.3) == ["b"]


class TestMaskMerge:
    def test_mask_merge(self):
        img1 = Image.new("L", (10, 10), color="black")
        draw1 = ImageDraw.Draw(img1)
        draw1.rectangle((3, 3, 5, 5), fill="white")

        img2 = Image.new("L", (10, 10), color="black")
        draw2 = ImageDraw.Draw(img2)
        draw2.rectangle((6, 6, 8, 8), fill="white")

        merged = mask_merge([img1, img2])
        assert len(merged) == 1

        expect = Image.new("L", (10, 10), color="black")
        draw3 = ImageDraw.Draw(expect)
        draw3.rectangle((3, 3, 5, 5), fill="white")
        draw3.rectangle((6, 6, 8, 8), fill="white")

        assert np.array_equal(np.array(merged[0]), np.array(expect))

    def test_merge_mask_different_size(self):
        img1 = Image.new("L", (10, 10), color="black")
        draw1 = ImageDraw.Draw(img1)
        draw1.rectangle((3, 3, 5, 5), fill="white")

        img2 = Image.new("L", (20, 20), color="black")
        draw2 = ImageDraw.Draw(img2)
        draw2.rectangle((6, 6, 8, 8), fill="white")

        with pytest.raises(
            cv2.error, match="-209:Sizes of input arguments do not match"
        ):
            mask_merge([img1, img2])


def test_mask_invert():
    img = Image.new("L", (10, 10), color="black")
    draw = ImageDraw.Draw(img)
    draw.rectangle((3, 3, 5, 5), fill="white")

    inverted = mask_invert([img])
    assert len(inverted) == 1

    expect = Image.new("L", (10, 10), color="white")
    draw = ImageDraw.Draw(expect)
    draw.rectangle((3, 3, 5, 5), fill="black")

    assert np.array_equal(np.array(inverted[0]), np.array(expect))


def _overlap_pred(bboxes, confidences=None, class_names=None):
    from adetailer.common import PredictOutput

    masks = []
    for box in bboxes:
        mask = Image.new("L", (512, 512), 0)
        mask.paste(255, tuple(int(v) for v in box))
        masks.append(mask)
    return PredictOutput(
        bboxes=[list(b) for b in bboxes],
        masks=masks,
        confidences=list(confidences or []),
        preview=Image.new("RGB", (512, 512)),
        class_names=list(class_names or []),
    )


def test_bbox_iou():
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert bbox_iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0  # touching edges
    assert bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)
    assert bbox_iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0  # empty boxes


def test_filter_overlapping_keeps_the_more_confident_duplicate():
    # One face found twice (boxes almost the same) plus a second face.
    face = [100, 100, 200, 200]
    twin = [104, 102, 204, 202]
    other = [300, 100, 400, 200]
    pred = _overlap_pred([face, twin, other], confidences=[0.6, 0.9, 0.8])
    twin_mask = pred.masks[1]

    out = filter_overlapping(pred, 0.6)

    # Detector order is kept; every parallel list stays aligned.
    assert out.bboxes == [twin, other]
    assert out.confidences == [0.9, 0.8]
    assert out.masks[0] is twin_mask
    assert len(out.masks) == 2


@pytest.mark.parametrize("max_overlap", [0, 0.0, -1])
def test_filter_overlapping_off_changes_nothing(max_overlap):
    pred = _overlap_pred([[0, 0, 100, 100], [0, 0, 100, 100]], [0.5, 0.9])
    before = (list(pred.bboxes), list(pred.masks), list(pred.confidences))
    out = filter_overlapping(pred, max_overlap)
    assert (out.bboxes, out.masks, out.confidences) == before


def test_filter_overlapping_keeps_small_boxes_inside_a_big_one():
    # Eyes inside a face cover only a small part of it: the IoU stays low.
    face = [100, 100, 300, 300]
    eye_l = [140, 160, 180, 190]
    eye_r = [220, 160, 260, 190]
    pred = _overlap_pred([face, eye_l, eye_r], [0.9, 0.8, 0.7])
    out = filter_overlapping(pred, 0.3)
    assert out.bboxes == [face, eye_l, eye_r]


def test_filter_overlapping_compares_only_the_same_class():
    box = [100, 100, 200, 200]
    near = [102, 100, 202, 200]
    pred = _overlap_pred(
        [box, near, box], [0.9, 0.95, 0.5], class_names=["face", "hand", "face"]
    )
    out = filter_overlapping(pred, 0.5)
    # The hand stays (another class); the weaker second face goes.
    assert out.bboxes == [box, near]
    assert out.class_names == ["face", "hand"]
    assert out.confidences == [0.9, 0.95]


@pytest.mark.parametrize("confidences", [None, [1.0, 1.0, 1.0]])
def test_filter_overlapping_without_or_with_equal_confidences(confidences):
    # MediaPipe may report no score (1.0 for all): the earlier box wins.
    a = [10, 10, 110, 110]
    b = [12, 10, 112, 110]
    c = [14, 10, 114, 110]
    pred = _overlap_pred([a, b, c], confidences)
    out = filter_overlapping(pred, 0.5)
    assert out.bboxes == [a]
    assert out.confidences == ([] if confidences is None else [1.0])
    assert len(out.masks) == 1


def test_filter_overlapping_one_or_no_detection():
    from adetailer.common import PredictOutput

    empty = PredictOutput()
    assert filter_overlapping(empty, 0.5) is empty
    one = _overlap_pred([[0, 0, 10, 10]], [0.9])
    assert filter_overlapping(one, 0.5).bboxes == [[0, 0, 10, 10]]


def test_filter_overlapping_threshold_is_strict():
    # IoU exactly 1/3: dropped only when it is MORE than the setting.
    a = [0, 0, 10, 10]
    b = [5, 0, 15, 10]
    assert len(filter_overlapping(_overlap_pred([a, b], [0.9, 0.8]), 0.4).bboxes) == 2
    assert len(filter_overlapping(_overlap_pred([a, b], [0.9, 0.8]), 0.3).bboxes) == 1


# Settings -> ADetailer -> "Sort bounding boxes by".


def _sort_pred(confidences=(0.5, 0.9, 0.7)):
    from adetailer.common import PredictOutput

    # Three boxes: a small one on the right, a large one at the bottom left
    # and a middle one at the top.
    bboxes = [[300, 100, 360, 160], [10, 200, 210, 400], [120, 20, 220, 120]]
    masks = [Image.new("L", (400, 400), n) for n in (1, 2, 3)]
    return PredictOutput(
        bboxes=[list(b) for b in bboxes],
        masks=masks,
        confidences=list(confidences),
        preview=Image.new("RGB", (400, 400)),
        class_names=["face", "hand", "eyes"][: len(bboxes)],
    )


def test_sort_choices_are_exactly_the_original_four():
    from adetailer.args import BBOX_SORTBY

    # An older version reads ad_bbox_sortby with BBOX_SORTBY.index() and fails
    # every generation on a choice it does not know: new orders go in the
    # separate "More sort orders" setting (BBOX_SORTBY_MORE), never here.
    assert BBOX_SORTBY == [
        "None",
        "Position (left to right)",
        "Position (center to edge)",
        "Area (large to small)",
    ]


def test_more_sort_orders_map_to_the_sort_values_after_the_original_ones():
    from adetailer.args import BBOX_SORTBY, BBOX_SORTBY_MORE

    assert BBOX_SORTBY_MORE == [
        "Off",
        "Position (right to left)",
        "Position (top to bottom)",
        "Area (small to large)",
        "Confidence (high to low)",
    ]
    assert len(BBOX_SORTBY) + len(BBOX_SORTBY_MORE) - 1 == len(SortBy)
    assert [SortBy(len(BBOX_SORTBY) - 1 + i) for i in range(1, 5)] == [
        SortBy.RIGHT_TO_LEFT,
        SortBy.TOP_TO_BOTTOM,
        SortBy.AREA_SMALL_TO_LARGE,
        SortBy.CONFIDENCE,
    ]


@pytest.mark.parametrize(
    ("order_by", "order"),
    [
        (SortBy.LEFT_TO_RIGHT, [1, 2, 0]),
        (SortBy.AREA, [1, 2, 0]),
        (SortBy.RIGHT_TO_LEFT, [0, 2, 1]),
        (SortBy.TOP_TO_BOTTOM, [2, 0, 1]),
        (SortBy.AREA_SMALL_TO_LARGE, [0, 2, 1]),
        (SortBy.CONFIDENCE, [1, 2, 0]),
    ],
)
def test_sort_bboxes_keeps_masks_scores_and_classes_with_their_box(order_by, order):
    original = _sort_pred()
    pred = sort_bboxes(_sort_pred(), order_by)

    assert pred.bboxes == [original.bboxes[i] for i in order]
    assert [m.getpixel((0, 0)) for m in pred.masks] == [i + 1 for i in order]
    assert pred.confidences == [original.confidences[i] for i in order]
    assert pred.class_names == [original.class_names[i] for i in order]


@pytest.mark.parametrize(
    ("order_by", "bboxes"),
    [
        # The wide box ends further right but starts further left.
        (SortBy.RIGHT_TO_LEFT, [[100, 0, 150, 50], [0, 0, 200, 100]]),
        # The tall box starts higher but ends lower.
        (SortBy.TOP_TO_BOTTOM, [[0, 10, 50, 20], [60, 0, 100, 200]]),
    ],
)
def test_right_to_left_and_top_to_bottom_use_the_right_and_top_edges(order_by, bboxes):
    from adetailer.common import PredictOutput

    # Right to left goes by the right edge of each box and top to bottom by
    # its top edge: here the other edge gives the opposite order.
    pred = sort_bboxes(
        PredictOutput(
            bboxes=[list(b) for b in bboxes],
            masks=[Image.new("L", (200, 200), n) for n in (1, 2)],
            confidences=[0.9, 0.8],
            preview=Image.new("RGB", (200, 200)),
            class_names=["face", "hand"],
        ),
        order_by,
    )

    assert pred.bboxes == [bboxes[1], bboxes[0]]
    assert [m.getpixel((0, 0)) for m in pred.masks] == [2, 1]
    assert pred.confidences == [0.8, 0.9]
    assert pred.class_names == ["hand", "face"]


@pytest.mark.parametrize("confidences", [(), (0.9, 0.8)])
def test_sort_by_confidence_without_one_score_per_box_keeps_the_order(confidences):
    original = _sort_pred()
    pred = sort_bboxes(_sort_pred(confidences), SortBy.CONFIDENCE)
    assert pred.bboxes == original.bboxes


def test_sort_by_confidence_keeps_the_detector_order_on_equal_scores():
    original = _sort_pred((0.8, 0.8, 0.9))
    pred = sort_bboxes(_sort_pred((0.8, 0.8, 0.9)), SortBy.CONFIDENCE)
    assert pred.bboxes == [original.bboxes[i] for i in (2, 0, 1)]
