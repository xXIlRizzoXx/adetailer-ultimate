from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from adetailer import mediapipe as detector
from adetailer.mediapipe import mediapipe_predict


@pytest.mark.integration
@pytest.mark.parametrize(
    "model_name",
    [
        "mediapipe_face_short",
        "mediapipe_face_full",
        "mediapipe_face_mesh",
        "mediapipe_face_mesh_eyes_only",
        "mediapipe_face_features",
    ],
)
def test_mediapipe(sample_image2: Image.Image, model_name: str):
    result = mediapipe_predict(model_name, sample_image2)
    if result.preview is None:
        # mediapipe_predict swallows every failure and returns an empty result, so
        # asserting only "if preview is not None" made this test pass silently on a
        # completely broken MediaPipe stack — a green run proved nothing. Skipping
        # keeps the suite green where MediaPipe genuinely cannot run, but says so
        # out loud in the pytest summary instead of hiding it as a pass.
        pytest.skip(f"{model_name}: MediaPipe returned no result in this environment")

    assert len(result.bboxes) > 0
    assert len(result.masks) > 0
    assert len(result.confidences) > 0
    assert len(result.bboxes) == len(result.masks) == len(result.confidences)


@pytest.mark.parametrize(
    ("platform", "version", "expected"),
    [
        ("darwin", "0.10.35", False),
        ("darwin", "1.0.0", False),
        ("darwin", "1.0.1", True),
        ("darwin", "1.1.0", True),
        ("win32", "1.0.1", False),
        ("linux", "1.0.1", False),
    ],
)
def test_tasks_api_platform_guard(monkeypatch, platform, version, expected):
    # Bypass the memoized wrapper so platform fixtures cannot pollute later tests.
    monkeypatch.setattr(detector, "sys", SimpleNamespace(platform=platform, stderr=None))
    monkeypatch.setattr("importlib.metadata.version", lambda _: version)
    assert detector._tasks_api_aborts.__wrapped__() is expected


def test_unsafe_tasks_api_never_downloads_or_constructs_detector(monkeypatch):
    monkeypatch.setattr(detector, "_tasks_api_aborts", lambda: True)
    monkeypatch.setattr(detector, "_LANDMARKER_CACHE", {})
    monkeypatch.setattr(detector, "_DETECTOR_CACHE", {})
    download = Mock(side_effect=AssertionError("Unsafe detector must not initialize"))
    monkeypatch.setattr(detector, "_ensure_model", download)
    assert detector._get_face_detector(0.3) is None
    assert detector._get_face_landmarker(0.3, 1) is None
    download.assert_not_called()


@pytest.mark.parametrize(
    ("include", "exclude", "expected"),
    [
        ("", "", ["eyes", "mouth", "nose", "eyebrows", "face"]),
        ("mouth,eyes", "", ["eyes", "mouth"]),
        ("", "eyes,mouth", ["nose", "eyebrows", "face"]),
        ("MOUTH", "mouth", []),
    ],
)
def test_face_features_filters_keep_masks_and_labels_aligned(
    monkeypatch, include, exclude, expected
):
    monkeypatch.setattr(detector, "_run_face_mesh", lambda *_: [object()])
    monkeypatch.setattr(
        detector, "_hull_polys", lambda *_: [[(2, 2), (8, 2), (8, 8), (2, 8)]]
    )
    result = detector.mediapipe_face_features(
        Image.new("RGB", (12, 12)), classes=include, exclude_classes=exclude
    )
    assert result.class_names == expected
    assert len(result.bboxes) == len(result.masks) == len(result.confidences) == len(expected)
    assert all(mask.getbbox() == tuple(box) for mask, box in zip(result.masks, result.bboxes))
