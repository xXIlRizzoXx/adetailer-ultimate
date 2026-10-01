import io
import sys
from types import ModuleType, SimpleNamespace
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


@pytest.mark.parametrize(
    ("include", "expected"), [("hands", []), ("eye,lips", []), ("eyes,Hand", ["eyes"])]
)
def test_face_features_find_nothing_for_names_they_do_not_have(
    monkeypatch, capsys, include, expected
):
    # As the README says: unlike a YOLO model, which ignores such a name and
    # detects every class, the face features detect nothing for it, silently.
    mesh = Mock(return_value=[object()])
    monkeypatch.setattr(detector, "_run_face_mesh", mesh)
    monkeypatch.setattr(
        detector, "_hull_polys", lambda *_: [[(2, 2), (8, 2), (8, 8), (2, 8)]]
    )
    result = detector.mediapipe_face_features(Image.new("RGB", (12, 12)), classes=include)
    assert result.class_names == expected
    assert mesh.called is bool(expected)
    assert capsys.readouterr().out == ""


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


def _block_mediapipe(monkeypatch):
    """Make every `import mediapipe...` fail, as with MediaPipe missing or broken."""
    for name in [k for k in sys.modules if k == "mediapipe" or k.startswith("mediapipe.")]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "mediapipe", None)


def _fresh_detectors(monkeypatch):
    monkeypatch.setattr(detector, "_tasks_api_aborts", lambda: False)
    monkeypatch.setattr(detector, "_LANDMARKER_CACHE", {})
    monkeypatch.setattr(detector, "_DETECTOR_CACHE", {})
    monkeypatch.setattr(detector, "_UNAVAILABLE_REPORTED", False)


# The face landmarker models and the face detector models report the error
# each from their own loader: a session with only one family, and a mixed one
# that still reports it once.
_REPORT_SESSIONS = [
    ("mediapipe_face_mesh", "mediapipe_face_mesh_eyes_only", "mediapipe_face_features"),
    ("mediapipe_face_short", "mediapipe_face_full"),
    (
        "mediapipe_face_mesh",
        "mediapipe_face_mesh",
        "mediapipe_face_short",
        "mediapipe_face_features",
    ),
]


@pytest.mark.parametrize("models", _REPORT_SESSIONS)
def test_mediapipe_that_cannot_be_loaded_is_reported_once(monkeypatch, capsys, models):
    # With MediaPipe missing or broken, every MediaPipe detector found nothing
    # and the console only said "nothing detected", exactly as for an image
    # without a face. It still finds nothing and never crashes, but says why.
    _block_mediapipe(monkeypatch)
    _fresh_detectors(monkeypatch)
    download = Mock(side_effect=AssertionError("no model download without MediaPipe"))
    monkeypatch.setattr(detector, "_ensure_model", download)

    image = Image.new("RGB", (64, 64))
    for model in models:
        result = mediapipe_predict(model, image)
        assert result.preview is None
        assert result.bboxes == []

    err = capsys.readouterr().err
    assert err.count("MediaPipe could not be loaded") == 1
    assert "ModuleNotFoundError" in err
    download.assert_not_called()


def test_missing_model_asset_is_not_reported_as_a_broken_mediapipe(
    monkeypatch, capsys
):
    # MediaPipe loads but its model asset cannot be obtained (the download code
    # says so itself), and the legacy `solutions` API is gone, as in current
    # MediaPipe: this is not a MediaPipe that cannot be loaded.
    monkeypatch.setitem(sys.modules, "mediapipe", ModuleType("mediapipe"))
    monkeypatch.setitem(sys.modules, "mediapipe.tasks.python", _fake_tasks_api({}))
    _fresh_detectors(monkeypatch)
    monkeypatch.setattr(detector, "_ensure_model", lambda *_: None)

    image = Image.new("RGB", (64, 64))
    for model in ("mediapipe_face_mesh", "mediapipe_face_short"):
        assert mediapipe_predict(model, image).bboxes == []
    assert "could not be loaded" not in capsys.readouterr().err


def test_mediapipe_load_error_prints_on_a_legacy_code_page(monkeypatch):
    # The reason can name a file or folder with non-Latin characters; on a
    # console in a legacy code page the line must still print.
    out = io.TextIOWrapper(
        io.BytesIO(), encoding="cp1252", errors="strict", write_through=True
    )
    monkeypatch.setattr(sys, "stderr", out)
    monkeypatch.setattr(detector, "_UNAVAILABLE_REPORTED", False)

    detector._report_mediapipe_unavailable(ImportError("DLL load failed: \u9854"))

    assert b"MediaPipe could not be loaded (ImportError: DLL load failed: \\u9854)" in (
        out.buffer.getvalue()
    )


def _native_library_api(monkeypatch, load_raw_library, create_error=None):
    """A MediaPipe laid out like 0.10.3x: the import is pure Python and the
    native library loads only when create_from_options calls
    ``mediapipe_c_bindings.load_raw_library()``."""
    bindings = ModuleType("mediapipe.tasks.python.core.mediapipe_c_bindings")
    bindings.load_raw_library = load_raw_library

    def create_from_options(options):
        load_raw_library()
        if create_error is not None:
            raise create_error
        return object()

    api = _fake_tasks_api({})
    api.vision.FaceDetector = SimpleNamespace(create_from_options=create_from_options)
    api.vision.FaceLandmarker = api.vision.FaceDetector
    monkeypatch.setitem(sys.modules, "mediapipe", ModuleType("mediapipe"))
    monkeypatch.setitem(sys.modules, "mediapipe.tasks.python", api)
    monkeypatch.setitem(sys.modules, bindings.__name__, bindings)
    _fresh_detectors(monkeypatch)


def _model_assets(monkeypatch, tmp_path):
    """Both model assets already downloaded; returns {name: path}."""
    assets = {}
    for name, _url, _min in (
        detector._FACE_LANDMARKER_ASSET,
        detector._FACE_DETECTOR_ASSET,
    ):
        assets[name] = tmp_path / name
        assets[name].write_bytes(b"x" * 4096)
    monkeypatch.setattr(detector, "_ensure_model", lambda name, *_: str(assets[name]))
    return assets


@pytest.mark.parametrize("models", _REPORT_SESSIONS)
def test_mediapipe_native_library_that_cannot_load_is_reported_and_keeps_assets(
    monkeypatch, tmp_path, capsys, models
):
    # Current MediaPipe imports fine and loads its native library only when a
    # detector is created. When that library could not load, every detector
    # found nothing without a word, and the valid model assets were deleted as
    # corrupt and downloaded again at every detection.
    def load_raw_library(signatures=()):
        msg = "Could not find module 'libmediapipe.dll' (or one of its dependencies)"
        raise FileNotFoundError(msg)

    _native_library_api(monkeypatch, load_raw_library)
    assets = _model_assets(monkeypatch, tmp_path)

    image = Image.new("RGB", (64, 64))
    for model in models:
        result = mediapipe_predict(model, image)
        assert result.preview is None
        assert result.bboxes == []

    err = capsys.readouterr().err
    assert err.count("MediaPipe could not be loaded") == 1
    assert "FileNotFoundError" in err
    assert "libmediapipe" in err
    assert all(path.exists() for path in assets.values())


@pytest.mark.parametrize("which", ["detector", "landmarker"])
@pytest.mark.parametrize("bindings", ["current", "older wheel"])
def test_corrupt_model_asset_is_still_discarded_and_not_reported(
    monkeypatch, tmp_path, capsys, which, bindings
):
    # The native library loads but the asset does not: it is still deleted so
    # that the next detection downloads it again, and MediaPipe itself is not
    # reported as broken. Older wheels have no native loader to ask.
    _native_library_api(
        monkeypatch, lambda signatures=(): object(), ValueError("corrupt model")
    )
    if bindings == "older wheel":
        monkeypatch.setitem(
            sys.modules, "mediapipe.tasks.python.core.mediapipe_c_bindings", None
        )
    assets = _model_assets(monkeypatch, tmp_path)

    if which == "detector":
        assert detector._get_face_detector(0.3) is None
        name = detector._FACE_DETECTOR_ASSET[0]
    else:
        assert detector._get_face_landmarker(0.3, 20) is None
        name = detector._FACE_LANDMARKER_ASSET[0]
    assert not assets[name].exists()
    assert "could not be loaded" not in capsys.readouterr().err


class _ClampingFaceDetector:
    """A stub tasks FaceDetector for a 200x200 image and one face per entry of
    ``faces`` (true x, y, width, height, score). Like MediaPipe's, it clamps the
    origin of a face cut by the left or top edge to 0 and keeps the full width
    and height. On a padded copy it finds the faces shifted by the padding
    (and by ``jitter``, as a real detector's boxes move a little), finds nothing
    (``padded="none"``) or fails (``padded="raise"``)."""

    def __init__(self, faces, padded="find", jitter=(0, 0)):
        self.faces, self.padded, self.calls = faces, padded, 0
        self.jitter = jitter

    def detect(self, data):
        self.calls += 1
        pad_y, pad_x = data.shape[0] - 200, data.shape[1] - 200
        if pad_x or pad_y:
            if self.padded == "raise":
                msg = "detect failed"
                raise RuntimeError(msg)
            if self.padded == "none":
                return SimpleNamespace(detections=[])
            pad_x, pad_y = pad_x + self.jitter[0], pad_y + self.jitter[1]
        return SimpleNamespace(
            detections=[
                SimpleNamespace(
                    bounding_box=SimpleNamespace(
                        origin_x=max(0, x + pad_x),
                        origin_y=max(0, y + pad_y),
                        width=w,
                        height=h,
                    ),
                    categories=[SimpleNamespace(score=score)],
                )
                for x, y, w, h, score in self.faces
            ]
        )


def _run_face_short(monkeypatch, stub):
    fake_mp = ModuleType("mediapipe")
    fake_mp.Image = lambda image_format, data: data
    fake_mp.ImageFormat = SimpleNamespace(SRGB="srgb")
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)
    monkeypatch.setattr(detector, "_get_face_detector", lambda *_: stub)
    return mediapipe_predict("mediapipe_face_short", Image.new("RGB", (200, 200)))


@pytest.mark.parametrize(
    ("faces", "expected"),
    [
        # cut by the left edge
        ([(-40, 50, 100, 100, 0.9)], [[0, 50, 60, 150]]),
        # cut by the top edge
        ([(50, -30, 100, 100, 0.9)], [[50, 0, 150, 70]]),
        # cut by both
        ([(-20, -30, 100, 100, 0.9)], [[0, 0, 80, 70]]),
        # more than half outside: the centre of the full face is outside too
        ([(-60, 50, 100, 100, 0.9)], [[0, 50, 40, 150]]),
        ([(50, -70, 100, 100, 0.9)], [[50, 0, 150, 30]]),
        ([(-60, -60, 100, 100, 0.9)], [[0, 0, 40, 40]]),
        # two faces cut by the left edge, each keeps its own box
        (
            [(-40, 10, 80, 80, 0.9), (-10, 110, 60, 60, 0.8)],
            [[0, 10, 40, 90], [0, 110, 50, 170]],
        ),
    ],
)
def test_face_box_cut_by_the_left_or_top_edge_fits_the_visible_face(
    monkeypatch, faces, expected
):
    # The box of such a face slid past it (onto the neck or the background) by
    # the part outside the image, and that area was inpainted with the face.
    stub = _ClampingFaceDetector(faces)
    result = _run_face_short(monkeypatch, stub)
    assert result.bboxes == expected
    assert result.confidences == [face[4] for face in faces]
    assert [m.getbbox() for m in result.masks] == [
        (x1, y1, x2 + 1, y2 + 1) for x1, y1, x2, y2 in expected
    ]
    assert result.preview is not None
    assert stub.calls == 2


def test_face_box_cut_by_the_left_edge_and_touching_the_top_fits_the_face(
    monkeypatch,
):
    # The box touches both edges, so it is matched on both, but the face is cut
    # only by the left one: a re-detected box a few pixels lower still matches,
    # and the box still reaches the top edge the detector saw the face touch.
    stub = _ClampingFaceDetector([(-40, 0, 100, 100, 0.9)], jitter=(0, 3))
    result = _run_face_short(monkeypatch, stub)
    assert result.bboxes == [[0, 0, 60, 103]]
    assert stub.calls == 2


def test_face_boxes_away_from_the_left_and_top_edges_are_unchanged(monkeypatch):
    # Inside the image, or cut by the right or bottom edge (where the box runs
    # past the image and the mask is clipped): one detection, boxes as before.
    faces = [(40, 50, 60, 60, 0.8), (150, 20, 80, 80, 0.7), (20, 160, 60, 60, 0.6)]
    stub = _ClampingFaceDetector(faces)
    result = _run_face_short(monkeypatch, stub)
    assert result.bboxes == [
        [40, 50, 100, 110],
        [150, 20, 230, 100],
        [20, 160, 80, 220],
    ]
    assert result.confidences == [0.8, 0.7, 0.6]
    assert stub.calls == 1


class _GrowingFaceDetector(_ClampingFaceDetector):
    """Like _ClampingFaceDetector, but on the padded copy, where the whole face
    fits, it sees the face larger (``grown``: true x, y, width, height), as the
    real detector does with a face that fills the frame."""

    def __init__(self, face, grown):
        super().__init__([face])
        self.grown = grown

    def detect(self, data):
        if data.shape[:2] != (200, 200):
            self.faces = [(*self.grown, 0.9)]
        return super().detect(data)


@pytest.mark.parametrize(
    ("face", "grown"),
    [
        # cut by the top edge, re-detected past the bottom one
        ((20, 0, 180, 180, 0.9), (0, -20, 200, 230)),
        # cut by the left edge, re-detected past the right one
        ((0, 20, 180, 180, 0.9), (-20, 0, 230, 200)),
    ],
)
def test_frame_filling_edge_face_box_stays_inside_the_image(
    monkeypatch, face, grown
):
    # The fitted box ran past the far edge of the image: its area ratio went
    # over 1.0, the highest "Mask max area ratio", and the face was dropped.
    from adetailer.mask import filter_by_ratio

    stub = _GrowingFaceDetector(face, grown)
    result = _run_face_short(monkeypatch, stub)
    assert result.bboxes == [[0, 0, 200, 200]]
    assert stub.calls == 2
    assert filter_by_ratio(result, 0.0, 1.0).bboxes == [[0, 0, 200, 200]]


@pytest.mark.parametrize("padded", ["find", "none", "raise"])
def test_edge_face_box_falls_back_to_the_detector_box(monkeypatch, padded):
    # A face inside the image stays as it is, in its place and with its score,
    # and is not taken for the edge face although it lies in the edge face's
    # box; when the padded copy finds no match or fails, the edge box is kept.
    faces = [(40, 50, 60, 60, 0.8), (-40, 50, 100, 100, 0.9)]
    result = _run_face_short(monkeypatch, _ClampingFaceDetector(faces, padded))
    edge = [0, 50, 60, 150] if padded == "find" else [0, 50, 100, 150]
    assert result.bboxes == [[40, 50, 100, 110], edge]
    assert result.confidences == [0.8, 0.9]
    assert len(result.masks) == 2


@pytest.mark.parametrize(
    ("model", "target"),
    [
        ("mediapipe_face_short", "mediapipe_face_detection"),
        ("mediapipe_face_full", "mediapipe_face_detection"),
        ("mediapipe_face_mesh", "mediapipe_face_mesh"),
        ("mediapipe_face_mesh_eyes_only", "mediapipe_face_mesh_eyes_only"),
        ("mediapipe_face_features", "mediapipe_face_features"),
    ],
)
def test_every_mediapipe_detector_gets_an_rgb_image(monkeypatch, model, target):
    # On Gradio 4 WebUIs a GIF dropped into the Detection preview arrives as a
    # palette ("P") image. An RGB image is handed over as it is.
    seen = []

    def fake(*args, **kwargs):
        seen.extend(a for a in args if isinstance(a, Image.Image))
        return detector.PredictOutput()

    monkeypatch.setattr(detector, target, fake)
    mediapipe_predict(model, Image.new("P", (16, 8)))
    assert [(img.mode, img.size) for img in seen] == [("RGB", (16, 8))]

    rgb = Image.new("RGB", (16, 8))
    mediapipe_predict(model, rgb)
    assert seen[1] is rgb


@pytest.mark.parametrize("mode", ["P", "L", "RGBA"])
def test_face_in_a_gif_or_grayscale_image_is_detected(monkeypatch, mode):
    # MediaPipe takes only 3-channel RGB data and refused the 2-D array of a
    # palette or grayscale image, so every MediaPipe detector found nothing.
    stub = _ClampingFaceDetector([(40, 50, 60, 60, 0.8)])
    fake_mp = ModuleType("mediapipe")

    def mp_image(image_format, data):
        if data.ndim != 3 or data.shape[2] != 3:
            msg = "Pixel data size is too small"
            raise ValueError(msg)
        return data

    fake_mp.Image = mp_image
    fake_mp.ImageFormat = SimpleNamespace(SRGB="srgb")
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)
    monkeypatch.setattr(detector, "_get_face_detector", lambda *_: stub)

    image = Image.new("RGB", (200, 200), "white").convert(mode)
    result = mediapipe_predict("mediapipe_face_short", image)
    assert result.bboxes == [[40, 50, 100, 110]]
    assert result.confidences == [0.8]
    assert result.preview is not None
