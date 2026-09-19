import sys
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


def _fake_tasks_api(seen):
    """A fake `mediapipe.tasks.python` whose loader, like MediaPipe's native one
    on Windows, cannot open a model asset by a path with non-ASCII characters."""

    class BaseOptions:
        def __init__(self, model_asset_path=None, model_asset_buffer=None):
            seen["path"], seen["buffer"] = model_asset_path, model_asset_buffer

    def create_from_options(options):
        path = seen["path"]
        if path is not None and not str(path).isascii():
            raise FileNotFoundError(f"Unable to open file at {path}")
        return object()

    factory = SimpleNamespace(create_from_options=create_from_options)
    vision = SimpleNamespace(
        FaceDetector=factory,
        FaceLandmarker=factory,
        FaceDetectorOptions=lambda **kw: kw,
        FaceLandmarkerOptions=lambda **kw: kw,
        RunningMode=SimpleNamespace(IMAGE=0),
    )
    return SimpleNamespace(BaseOptions=BaseOptions, vision=vision)


@pytest.mark.parametrize("which", ["detector", "landmarker"])
def test_model_asset_in_non_ascii_folder_loads_and_is_kept(
    monkeypatch, tmp_path, which
):
    # A WebUI installed under a user or folder name with an accented or CJK
    # character: the asset failed to load, was deleted as "corrupt" and was
    # downloaded again on every call, while nothing was ever detected.
    asset = tmp_path / "café_日本" / "model.bin"
    asset.parent.mkdir()
    asset.write_bytes(b"x" * 4096)
    seen = {}
    monkeypatch.setitem(sys.modules, "mediapipe.tasks.python", _fake_tasks_api(seen))
    monkeypatch.setattr(detector, "_tasks_api_aborts", lambda: False)
    monkeypatch.setattr(detector, "_ensure_model", lambda *_: str(asset))
    monkeypatch.setattr(detector, "_DETECTOR_CACHE", {})
    monkeypatch.setattr(detector, "_LANDMARKER_CACHE", {})

    if which == "detector":
        assert detector._get_face_detector(0.3) is not None
    else:
        assert detector._get_face_landmarker(0.3, 20) is not None
    assert asset.exists()
    assert seen["buffer"] == asset.read_bytes()
