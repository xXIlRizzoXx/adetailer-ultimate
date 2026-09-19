import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from adetailer.mask import (
    bbox_area,
    dilate_erode,
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
    ],
)
def test_parse_indices_preserves_selection_rules(spec, expected):
    assert parse_indices(spec, 3) == expected


@pytest.mark.parametrize("spec", ["1-999999999999", "999999999999-1"])
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
