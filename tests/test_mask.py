import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from adetailer.mask import (
    bbox_area,
    bbox_iou,
    dilate_erode,
    filter_overlapping,
    has_intersection,
    is_all_black,
    mask_invert,
    mask_merge,
    mask_preprocess,
    offset,
    parse_indices,
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
