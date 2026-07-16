from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from adetailer import PredictOutput
from adetailer.classes import FACE_FEATURE_CLASSES, parse_csv
from adetailer.common import create_bbox_from_mask, create_mask_from_bbox

# --- Model assets (new MediaPipe "tasks" API) -------------------------------
# Recent MediaPipe wheels — notably the Python 3.13 builds Forge Neo ships —
# DROPPED the legacy `solutions` API (FaceMesh / FaceDetection) entirely, leaving
# only the new `tasks` API. So these detectors now run on `tasks` (FaceLandmarker
# / FaceDetector), downloading the small model assets on first use, and fall back
# to the old `solutions` API when it IS still present (older mediapipe on
# Python < 3.13) and the download isn't available — universal-WebUI compat.
_MODEL_DIR = Path(__file__).resolve().parent / "_mediapipe_models"
# (filename, url, min_valid_bytes) — the min size is a floor (~80% of the real
# asset) used to reject truncated / interleaved / corrupt downloads instead of
# caching a >1 KB but broken file forever. Real sizes: .task ~3.76 MB, .tflite
# ~0.23 MB.
_FACE_LANDMARKER_ASSET = (
    "face_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    3_000_000,
)
_FACE_DETECTOR_ASSET = (
    "blaze_face_short_range.tflite",
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    200_000,
)

# Feature -> the FaceLandmarksConnections constant name(s) whose vertices build
# each convex-hull sub-polygon (left+right kept separate so a mask never bridges
# across the face). Same 468/478 FaceMesh topology as the old FACEMESH_* sets.
_FEATURE_TASK_GROUPS: dict[str, list[tuple[str, ...]]] = {
    "eyes": [("FACE_LANDMARKS_LEFT_EYE",), ("FACE_LANDMARKS_RIGHT_EYE",)],
    "mouth": [("FACE_LANDMARKS_LIPS",)],
    "nose": [("FACE_LANDMARKS_NOSE",)],
    "eyebrows": [("FACE_LANDMARKS_LEFT_EYEBROW",), ("FACE_LANDMARKS_RIGHT_EYEBROW",)],
    "face": [("FACE_LANDMARKS_FACE_OVAL",)],
}
# Old `solutions` FACEMESH_* constant names, for the fallback path.
_FEATURE_SOLUTION_GROUPS: dict[str, list[tuple[str, ...]]] = {
    "eyes": [("FACEMESH_LEFT_EYE",), ("FACEMESH_RIGHT_EYE",)],
    "mouth": [("FACEMESH_LIPS",)],
    "nose": [("FACEMESH_NOSE",)],
    "eyebrows": [("FACEMESH_LEFT_EYEBROW",), ("FACEMESH_RIGHT_EYEBROW",)],
    "face": [("FACEMESH_FACE_OVAL",)],
}
# Hard-coded nose landmark indices (468-point FaceMesh topology, all < 468). Last
# resort when neither the tasks connections nor the solutions FACEMESH_NOSE
# constant is available.
_FACEMESH_NOSE_FALLBACK = frozenset(
    {
        1, 2, 4, 5, 6, 19, 94, 168, 195, 197,  # midline: nasion -> bridge -> tip
        45, 48, 64, 97, 98, 115, 220,  # left ala / wing
        275, 278, 294, 326, 327, 344, 440,  # right ala / wing
    }
)


def _ensure_model(name: str, url: str, min_bytes: int = 1024) -> str | None:
    """Return the local path to a tasks model asset, downloading it on first use.

    Cached under ``adetailer/_mediapipe_models/`` (git-ignored; survives updates).
    Fully guarded — returns None if the file is absent and can't be fetched, so
    the caller degrades to the solutions fallback / an empty result instead of
    crashing. Users can also drop the file into that folder manually.

    ``min_bytes`` is the smallest size a valid asset can be: a cached OR freshly
    downloaded file below it is treated as truncated/corrupt (rejected, not
    installed) so a broken download never gets cached forever.
    """
    dst = _MODEL_DIR / name
    try:
        if dst.exists():
            if dst.stat().st_size >= min_bytes:
                return str(dst)
            # A previously-cached-but-corrupt file: drop it so we re-fetch.
            dst.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        return None
    try:
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        return None

    # Unique temp name (per process) so two concurrent first-use downloads can't
    # interleave into one shared .part and produce a >min_bytes but corrupt file.
    import os

    tmp = dst.with_name(f"{dst.name}.{os.getpid()}.part")
    ok = False
    # Prefer requests (robust SSL/proxy handling inside the WebUI); fall back to
    # urllib. The WebUI process has working HTTPS (it downloads the YOLO models
    # the same way).
    try:
        import requests

        with requests.get(url, timeout=(10, 180), stream=True) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)
        ok = True
    except Exception:  # noqa: BLE001
        try:
            import urllib.request

            urllib.request.urlretrieve(url, tmp)  # noqa: S310
            ok = True
        except Exception:  # noqa: BLE001
            ok = False

    if ok:
        try:
            if tmp.stat().st_size >= min_bytes:
                tmp.replace(dst)
                return str(dst)
        except Exception:  # noqa: BLE001
            pass
    try:
        tmp.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass
    print(
        f"[-] ADetailer: couldn't obtain MediaPipe model {name!r}. Place it in "
        f"{_MODEL_DIR} manually (from {url}) to use this detector.",
        file=sys.stderr,
    )
    return None


# One FaceLandmarker per (confidence, max_faces) — created lazily and REUSED
# across detect() calls (IMAGE running mode allows it). This avoids reloading the
# ~3.7 MB model (and re-emitting MediaPipe's init logs) for every image in a
# batch. Never closed (a WebUI keeps a couple of these for its lifetime).
_LANDMARKER_CACHE: dict[tuple[float, int], object] = {}


def _get_face_landmarker(confidence: float, max_faces: int):
    key = (round(float(confidence), 3), int(max_faces))
    if key in _LANDMARKER_CACHE:
        return _LANDMARKER_CACHE[key]
    try:
        from mediapipe.tasks.python import BaseOptions, vision
    except Exception:  # noqa: BLE001
        return None
    path = _ensure_model(*_FACE_LANDMARKER_ASSET)
    if not path:
        return None
    try:
        landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=path),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=int(max_faces),
                min_face_detection_confidence=float(confidence),
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
        )
    except Exception:  # noqa: BLE001
        return None
    _LANDMARKER_CACHE[key] = landmarker
    return landmarker


# One FaceDetector per confidence — cached and reused for the same reason as the
# landmarker above (IMAGE mode is stateless per detect(), so a batch of images
# reuses one detector instead of reloading blaze_face + re-logging per image).
_DETECTOR_CACHE: dict[float, object] = {}


def _get_face_detector(confidence: float):
    key = round(float(confidence), 3)
    if key in _DETECTOR_CACHE:
        return _DETECTOR_CACHE[key]
    try:
        from mediapipe.tasks.python import BaseOptions, vision
    except Exception:  # noqa: BLE001
        return None
    path = _ensure_model(*_FACE_DETECTOR_ASSET)
    if not path:
        return None
    try:
        detector = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=path),
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=float(confidence),
            )
        )
    except Exception:  # noqa: BLE001
        return None
    _DETECTOR_CACHE[key] = detector
    return detector


def _run_face_mesh(
    image: Image.Image, confidence: float, max_faces: int = 20
) -> list[np.ndarray]:
    """Detect faces and return one ``(N, 2)`` int array of pixel-space landmark
    points per face. Prefers the tasks FaceLandmarker; falls back to the legacy
    solutions FaceMesh when tasks isn't usable. Empty list if neither works."""
    w, h = image.size
    arr = np.array(image)

    landmarker = _get_face_landmarker(confidence, max_faces)
    if landmarker is not None:
        try:
            import mediapipe as mp

            res = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
            )
        except Exception:  # noqa: BLE001
            res = None
        if res is not None:
            faces = getattr(res, "face_landmarks", None) or []
            return [
                np.array([[p.x * w, p.y * h] for p in face], dtype=int)
                for face in faces
            ]

    # Legacy solutions fallback (older mediapipe that still ships `solutions`).
    try:
        import mediapipe as mp

        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            min_detection_confidence=confidence,
        ) as face_mesh:
            pred = face_mesh.process(arr)
        if pred.multi_face_landmarks is None:
            return []
        return [
            np.array([[lm.x * w, lm.y * h] for lm in f.landmark], dtype=int)
            for f in pred.multi_face_landmarks
        ]
    except Exception:  # noqa: BLE001
        return []


@lru_cache(maxsize=8)
def _feature_indices(feature: str) -> tuple[tuple[int, ...], ...]:
    """Vertex-index tuples, one per convex-hull sub-polygon of ``feature``
    (e.g. "eyes" -> (left_eye_idx, right_eye_idx)). Topology-constant, so this is
    cached. Source order: tasks FaceLandmarksConnections -> solutions FACEMESH_*
    -> hard-coded nose. Sub-polygons with < 3 unique points are dropped."""
    polys: list[tuple[int, ...]] = []

    # 1) tasks connections (present on modern mediapipe)
    conns = None
    try:
        from mediapipe.tasks.python import vision

        conns = vision.FaceLandmarksConnections
    except Exception:  # noqa: BLE001
        conns = None
    if conns is not None:
        for subgroup in _FEATURE_TASK_GROUPS.get(feature, []):
            verts: set[int] = set()
            for name in subgroup:
                group = getattr(conns, name, None)
                if group:
                    for c in group:
                        verts.add(int(c.start))
                        verts.add(int(c.end))
            if len(verts) >= 3:
                polys.append(tuple(sorted(verts)))
        if polys:
            return tuple(polys)

    # 2) legacy solutions constants
    try:
        import mediapipe as mp

        mp_face_mesh = mp.solutions.face_mesh
        for subgroup in _FEATURE_SOLUTION_GROUPS.get(feature, []):
            arrs: list[np.ndarray] = []
            for name in subgroup:
                group = getattr(mp_face_mesh, name, None)
                if group is not None:
                    arrs.append(np.array(list(group)).flatten())
            if arrs:
                idx = np.unique(np.concatenate(arrs))
                if idx.size >= 3:
                    polys.append(tuple(int(v) for v in idx))
        if polys:
            return tuple(polys)
    except Exception:  # noqa: BLE001
        pass

    # 3) hard-coded nose (last resort)
    if feature == "nose":
        return (tuple(sorted(_FACEMESH_NOSE_FALLBACK)),)
    return ()


def _hull_polys(points: np.ndarray, feature: str) -> list[list[int]]:
    """Convex-hull outlines (flat [x0,y0,x1,y1,...]) for ``feature`` on one face's
    landmark ``points``. Skips indices past the landmark count (468-vs-478 drift)
    and sub-polygons with < 3 points."""
    n_pts = points.shape[0]
    out: list[list[int]] = []
    for idx_tuple in _feature_indices(feature):
        idx = np.array(idx_tuple, dtype=int)
        idx = idx[idx < n_pts]
        if idx.size < 3:
            continue
        flat = cv2.convexHull(points[idx]).reshape(-1).tolist()
        # A face so small that every landmark of this feature rounds to the same
        # pixel yields a 1-point hull; ImageDraw.polygon() would raise on < 2
        # points. Drop it so only this sub-polygon is skipped, never the whole
        # image's result (each flat point is 2 coords -> need >= 6 for a triangle).
        if len(flat) >= 6:
            out.append(flat)
    return out


def mediapipe_predict(
    model_type: str,
    image: Image.Image,
    confidence: float = 0.3,
    classes: str = "",
    exclude_classes: str = "",
) -> PredictOutput:
    mapping = {
        "mediapipe_face_short": lambda img, conf: mediapipe_face_detection(0, img, conf),
        "mediapipe_face_full": lambda img, conf: mediapipe_face_detection(1, img, conf),
        "mediapipe_face_mesh": mediapipe_face_mesh,
        "mediapipe_face_mesh_eyes_only": mediapipe_face_mesh_eyes_only,
        "mediapipe_face_features": lambda img, conf: mediapipe_face_features(
            img, conf, classes=classes, exclude_classes=exclude_classes
        ),
    }
    if model_type in mapping:
        try:
            return mapping[model_type](image, confidence)
        except Exception:  # noqa: BLE001
            return PredictOutput()
    msg = f"[-] ADetailer: Invalid mediapipe model type: {model_type}, Available: {list(mapping.keys())!r}"
    raise RuntimeError(msg)


def mediapipe_face_detection(
    model_type: int, image: Image.Image, confidence: float = 0.3
) -> PredictOutput[float]:
    """Face bounding-box detector. Uses the tasks FaceDetector (blaze short-range;
    the tasks API ships only that model, so both short/full map to it), falling
    back to the legacy solutions FaceDetection with ``model_selection`` when
    available."""
    img_width, img_height = image.size
    arr = np.array(image)

    # --- tasks FaceDetector (preferred) ---
    detections = None
    detector = _get_face_detector(confidence)
    if detector is not None:
        try:
            import mediapipe as mp

            res = detector.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
            )
            detections = res.detections
        except Exception:  # noqa: BLE001
            detections = None

    if detections is not None:
        bboxes = []
        confidences = []
        for d in detections:
            bb = d.bounding_box
            x1 = float(bb.origin_x)
            y1 = float(bb.origin_y)
            bboxes.append([x1, y1, x1 + float(bb.width), y1 + float(bb.height)])
            confidences.append(
                float(d.categories[0].score) if getattr(d, "categories", None) else 1.0
            )
        if not bboxes:
            return PredictOutput()
        masks = create_mask_from_bbox(bboxes, image.size)
        preview = draw_preview(image.convert("RGB").copy(), bboxes, masks)
        return PredictOutput(
            bboxes=bboxes, masks=masks, confidences=confidences, preview=preview
        )

    # --- legacy solutions fallback ---
    try:
        import mediapipe as mp

        mp_face_detection = mp.solutions.face_detection
        draw_util = mp.solutions.drawing_utils
        with mp_face_detection.FaceDetection(
            model_selection=model_type, min_detection_confidence=confidence
        ) as face_detector:
            pred = face_detector.process(arr)
        if pred.detections is None:
            return PredictOutput()
        preview_array = arr.copy()
        bboxes = []
        confidences = []
        for detection in pred.detections:
            draw_util.draw_detection(preview_array, detection)
            bbox = detection.location_data.relative_bounding_box
            x1 = bbox.xmin * img_width
            y1 = bbox.ymin * img_height
            x2 = x1 + bbox.width * img_width
            y2 = y1 + bbox.height * img_height
            # `solutions` Detection.score is a repeated-float container: take the
            # scalar so confidences stays a list[float] (matches the tasks path).
            score = detection.score
            confidences.append(float(score[0]) if len(score) else 1.0)
            bboxes.append([x1, y1, x2, y2])
        masks = create_mask_from_bbox(bboxes, image.size)
        return PredictOutput(
            bboxes=bboxes,
            masks=masks,
            confidences=confidences,
            preview=Image.fromarray(preview_array),
        )
    except Exception:  # noqa: BLE001
        return PredictOutput()


def mediapipe_face_mesh(
    image: Image.Image, confidence: float = 0.3
) -> PredictOutput[int]:
    """Whole-face detector: one convex-hull mask over all landmarks per face."""
    faces = _run_face_mesh(image, confidence)
    if not faces:
        return PredictOutput()

    masks: list[Image.Image] = []
    confidences: list[float] = []
    for points in faces:
        hull = cv2.convexHull(points).reshape(-1).tolist()
        mask = Image.new("L", image.size, "black")
        if len(hull) >= 6:  # need >= 3 points, else polygon() raises
            ImageDraw.Draw(mask).polygon(hull, fill="white")
        if mask.getbbox() is None:
            continue
        masks.append(mask)
        confidences.append(1.0)  # FaceMesh gives no per-face score

    if not masks:
        return PredictOutput()
    bboxes = create_bbox_from_mask(masks, image.size)
    preview = draw_preview(image.convert("RGB").copy(), bboxes, masks)
    return PredictOutput(
        bboxes=bboxes, masks=masks, confidences=confidences, preview=preview
    )


def mediapipe_face_mesh_eyes_only(
    image: Image.Image, confidence: float = 0.3
) -> PredictOutput[int]:
    """Eyes-only detector: one mask (left+right eye hulls) per face."""
    faces = _run_face_mesh(image, confidence)
    if not faces:
        return PredictOutput()

    masks: list[Image.Image] = []
    confidences: list[float] = []
    for points in faces:
        mask = Image.new("L", image.size, "black")
        draw = ImageDraw.Draw(mask)
        for hull in _hull_polys(points, "eyes"):
            draw.polygon(hull, fill="white")
        if mask.getbbox() is None:
            continue
        masks.append(mask)
        confidences.append(1.0)

    if not masks:
        return PredictOutput()
    bboxes = create_bbox_from_mask(masks, image.size)
    preview = draw_preview(image.convert("RGB").copy(), bboxes, masks)
    return PredictOutput(
        bboxes=bboxes, masks=masks, confidences=confidences, preview=preview
    )


def mediapipe_face_features(
    image: Image.Image,
    confidence: float = 0.3,
    classes: str = "",
    exclude_classes: str = "",
) -> PredictOutput[int]:
    """Multi-class facial-feature detector: one precise landmark-group mask per
    detected face per REQUESTED feature class (eyes, mouth, nose, eyebrows,
    face). ``class_names`` is parallel to bboxes/masks, so the CLASSES filter,
    Auto class-guard, sequential passes and the combined Detection-Preview labels
    all light up. Empty include selection means ALL features."""
    requested = {c.casefold() for c in parse_csv(classes)}
    excluded = {c.casefold() for c in parse_csv(exclude_classes)}
    wanted = [
        f
        for f in FACE_FEATURE_CLASSES
        if (not requested or f in requested) and f not in excluded
    ]
    if not wanted:
        return PredictOutput()

    faces = _run_face_mesh(image, confidence)
    if not faces:
        return PredictOutput()

    bboxes: list[list[int]] = []
    masks: list[Image.Image] = []
    confidences: list[float] = []
    class_names: list[str] = []

    for points in faces:
        for feature in wanted:
            mask = Image.new("L", image.size, "black")
            draw = ImageDraw.Draw(mask)
            for hull in _hull_polys(points, feature):
                draw.polygon(hull, fill="white")
            # Per-mask bbox keeps all four lists strictly parallel: a feature
            # that produced no polygon yields an all-black mask -> getbbox()
            # None -> dropped together with its conf/class, never desyncing.
            bbox = mask.getbbox()
            if bbox is None:
                continue
            bboxes.append(list(bbox))
            masks.append(mask)
            confidences.append(1.0)  # FaceMesh gives no per-part score
            class_names.append(feature)

    if not masks:
        return PredictOutput()

    preview = draw_preview(
        image.convert("RGB").copy(), bboxes, masks, class_names=class_names
    )
    return PredictOutput(
        bboxes=bboxes,
        masks=masks,
        confidences=confidences,
        class_names=class_names,
        preview=preview,
    )


def draw_preview(
    preview: Image.Image,
    bboxes: list[list[int]],
    masks: list[Image.Image],
    class_names: list[str] | None = None,
) -> Image.Image:
    red = Image.new("RGB", preview.size, "red")
    for mask in masks:
        masked = Image.composite(red, preview, mask)
        preview = Image.blend(preview, masked, 0.25)

    draw = ImageDraw.Draw(preview)
    for i, bbox in enumerate(bboxes):
        draw.rectangle(bbox, outline="red", width=2)
        if class_names and i < len(class_names):
            draw.text((bbox[0] + 2, bbox[1] + 2), class_names[i], fill="red")

    return preview
