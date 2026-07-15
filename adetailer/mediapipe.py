from __future__ import annotations

from functools import partial

import cv2
import numpy as np
from PIL import Image, ImageDraw

from adetailer import PredictOutput
from adetailer.classes import FACE_FEATURE_CLASSES, parse_csv
from adetailer.common import create_bbox_from_mask, create_mask_from_bbox

# Which MediaPipe FaceMesh landmark-group constant(s) build each facial-feature
# class. Each inner tuple is ONE convex-hull polygon; a feature may be several
# polygons (left+right eyes/eyebrows get separate hulls so a mask never bridges
# across the face). Resolved by name via getattr so a group missing on an older
# MediaPipe build degrades gracefully instead of raising.
_FEATURE_SUBGROUPS: dict[str, list[tuple[str, ...]]] = {
    "eyes": [("FACEMESH_LEFT_EYE",), ("FACEMESH_RIGHT_EYE",)],
    "mouth": [("FACEMESH_LIPS",)],
    "nose": [("FACEMESH_NOSE",)],
    "eyebrows": [("FACEMESH_LEFT_EYEBROW",), ("FACEMESH_RIGHT_EYEBROW",)],
    "face": [("FACEMESH_FACE_OVAL",)],
}

# Hard-coded nose landmark indices (468-point FaceMesh topology, all < 468 so no
# dependency on refine_landmarks). Used ONLY when this MediaPipe build predates
# FACEMESH_NOSE (absent in older versions shipped with some A1111/reForge venvs)
# — universal-WebUI compat. Flat int set so the same flatten path handles it.
_FACEMESH_NOSE_FALLBACK = frozenset(
    {
        1, 2, 4, 5, 6, 19, 94, 168, 195, 197,  # midline: nasion -> bridge -> tip
        45, 48, 64, 97, 98, 115, 220,  # left ala / wing
        275, 278, 294, 326, 327, 344, 440,  # right ala / wing
    }
)


def mediapipe_predict(
    model_type: str,
    image: Image.Image,
    confidence: float = 0.3,
    classes: str = "",
    exclude_classes: str = "",
) -> PredictOutput:
    mapping = {
        "mediapipe_face_short": partial(mediapipe_face_detection, 0),
        "mediapipe_face_full": partial(mediapipe_face_detection, 1),
        "mediapipe_face_mesh": mediapipe_face_mesh,
        "mediapipe_face_mesh_eyes_only": mediapipe_face_mesh_eyes_only,
        "mediapipe_face_features": partial(
            mediapipe_face_features,
            classes=classes,
            exclude_classes=exclude_classes,
        ),
    }
    if model_type in mapping:
        func = mapping[model_type]
        try:
            return func(image, confidence)
        except Exception:
            return PredictOutput()
    msg = f"[-] ADetailer: Invalid mediapipe model type: {model_type}, Available: {list(mapping.keys())!r}"
    raise RuntimeError(msg)


def mediapipe_face_detection(
    model_type: int, image: Image.Image, confidence: float = 0.3
) -> PredictOutput[float]:
    import mediapipe as mp

    img_width, img_height = image.size

    mp_face_detection = mp.solutions.face_detection
    draw_util = mp.solutions.drawing_utils

    img_array = np.array(image)

    with mp_face_detection.FaceDetection(
        model_selection=model_type, min_detection_confidence=confidence
    ) as face_detector:
        pred = face_detector.process(img_array)

    if pred.detections is None:
        return PredictOutput()

    preview_array = img_array.copy()

    bboxes = []
    confidences = []
    for detection in pred.detections:
        draw_util.draw_detection(preview_array, detection)

        bbox = detection.location_data.relative_bounding_box
        x1 = bbox.xmin * img_width
        y1 = bbox.ymin * img_height
        w = bbox.width * img_width
        h = bbox.height * img_height
        x2 = x1 + w
        y2 = y1 + h

        confidences.append(detection.score)
        bboxes.append([x1, y1, x2, y2])

    masks = create_mask_from_bbox(bboxes, image.size)
    preview = Image.fromarray(preview_array)

    return PredictOutput(
        bboxes=bboxes, masks=masks, confidences=confidences, preview=preview
    )


def mediapipe_face_mesh(
    image: Image.Image, confidence: float = 0.3
) -> PredictOutput[int]:
    import mediapipe as mp

    mp_face_mesh = mp.solutions.face_mesh
    draw_util = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles

    w, h = image.size

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=20, min_detection_confidence=confidence
    ) as face_mesh:
        arr = np.array(image)
        pred = face_mesh.process(arr)

        if pred.multi_face_landmarks is None:
            return PredictOutput()

        preview = arr.copy()
        masks = []
        confidences = []

        for landmarks in pred.multi_face_landmarks:
            draw_util.draw_landmarks(
                image=preview,
                landmark_list=landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
            )

            points = np.array(
                [[land.x * w, land.y * h] for land in landmarks.landmark], dtype=int
            )
            outline = cv2.convexHull(points).reshape(-1).tolist()

            mask = Image.new("L", image.size, "black")
            draw = ImageDraw.Draw(mask)
            draw.polygon(outline, fill="white")
            masks.append(mask)
            confidences.append(1.0)  # Confidence is unknown

        bboxes = create_bbox_from_mask(masks, image.size)
        preview = Image.fromarray(preview)
        return PredictOutput(
            bboxes=bboxes, masks=masks, confidences=confidences, preview=preview
        )


def mediapipe_face_mesh_eyes_only(
    image: Image.Image, confidence: float = 0.3
) -> PredictOutput[int]:
    import mediapipe as mp

    mp_face_mesh = mp.solutions.face_mesh

    left_idx = np.array(list(mp_face_mesh.FACEMESH_LEFT_EYE)).flatten()
    right_idx = np.array(list(mp_face_mesh.FACEMESH_RIGHT_EYE)).flatten()

    w, h = image.size

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=20, min_detection_confidence=confidence
    ) as face_mesh:
        arr = np.array(image)
        pred = face_mesh.process(arr)

        if pred.multi_face_landmarks is None:
            return PredictOutput()

        preview = image.copy()
        masks = []
        confidences = []

        for landmarks in pred.multi_face_landmarks:
            points = np.array(
                [[land.x * w, land.y * h] for land in landmarks.landmark], dtype=int
            )
            left_eyes = points[left_idx]
            right_eyes = points[right_idx]
            left_outline = cv2.convexHull(left_eyes).reshape(-1).tolist()
            right_outline = cv2.convexHull(right_eyes).reshape(-1).tolist()

            mask = Image.new("L", image.size, "black")
            draw = ImageDraw.Draw(mask)
            for outline in (left_outline, right_outline):
                draw.polygon(outline, fill="white")
            masks.append(mask)
            confidences.append(1.0)  # Confidence is unknown

        bboxes = create_bbox_from_mask(masks, image.size)
        preview = draw_preview(preview, bboxes, masks)
        return PredictOutput(
            bboxes=bboxes, masks=masks, confidences=confidences, preview=preview
        )


def _feature_polygons(mp_face_mesh, feature: str) -> list[np.ndarray]:
    """Landmark-index arrays, one per convex-hull polygon that makes up
    ``feature`` (e.g. "eyes" -> [left_eye_idx, right_eye_idx]).

    Version-safe: a missing FACEMESH_* constant is skipped, and "nose" falls
    back to ``_FACEMESH_NOSE_FALLBACK``. Never raises. Sub-polygons with fewer
    than 3 unique points are dropped (cv2.convexHull needs a triangle).
    """
    polys: list[np.ndarray] = []
    for subgroup in _FEATURE_SUBGROUPS.get(feature, []):
        arrs: list[np.ndarray] = []
        for name in subgroup:
            group = getattr(mp_face_mesh, name, None)
            if group is not None:
                # Edge-tuple frozensets flatten to all vertices; a flat int set
                # flattens to itself — the same code path handles both.
                arrs.append(np.array(list(group)).flatten())
        if not arrs and feature == "nose":
            arrs.append(np.array(list(_FACEMESH_NOSE_FALLBACK)).flatten())
        if not arrs:
            continue
        idx = np.unique(np.concatenate(arrs))
        if idx.size >= 3:
            polys.append(idx)
    return polys


def mediapipe_face_features(
    image: Image.Image,
    confidence: float = 0.3,
    classes: str = "",
    exclude_classes: str = "",
) -> PredictOutput[int]:
    """Multi-class FaceMesh detector: one precise landmark-group mask per
    detected face per REQUESTED facial-feature class (eyes, mouth, nose,
    eyebrows, face). ``class_names`` is parallel to bboxes/masks, so the CLASSES
    filter, Auto class-guard, sequential passes and the combined Detection
    Preview labels all light up. Empty include selection means ALL features.
    """
    import mediapipe as mp

    mp_face_mesh = mp.solutions.face_mesh

    requested = {c.casefold() for c in parse_csv(classes)}
    excluded = {c.casefold() for c in parse_csv(exclude_classes)}
    wanted = [
        f
        for f in FACE_FEATURE_CLASSES
        if (not requested or f in requested) and f not in excluded
    ]
    if not wanted:
        return PredictOutput()

    w, h = image.size
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=20,
        min_detection_confidence=confidence,
    ) as face_mesh:
        arr = np.array(image)
        pred = face_mesh.process(arr)

    if pred.multi_face_landmarks is None:
        return PredictOutput()

    bboxes: list[list[int]] = []
    masks: list[Image.Image] = []
    confidences: list[float] = []
    class_names: list[str] = []

    for landmarks in pred.multi_face_landmarks:
        points = np.array(
            [[lm.x * w, lm.y * h] for lm in landmarks.landmark], dtype=int
        )
        n_pts = points.shape[0]

        for feature in wanted:
            mask = Image.new("L", image.size, "black")
            draw = ImageDraw.Draw(mask)
            for idx in _feature_polygons(mp_face_mesh, feature):
                idx = idx[idx < n_pts]  # survive 468-vs-478 topology drift
                if idx.size < 3:
                    continue
                hull = cv2.convexHull(points[idx]).reshape(-1).tolist()
                draw.polygon(hull, fill="white")

            # Per-mask bbox keeps all four lists strictly parallel: a feature
            # unavailable on this build yields an all-black mask -> getbbox()
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

    preview = draw_preview(image.copy(), bboxes, masks, class_names=class_names)
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
