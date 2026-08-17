import pytest
from PIL import Image

from adetailer.mediapipe import mediapipe_predict


@pytest.mark.parametrize(
    "model_name",
    [
        "mediapipe_face_short",
        "mediapipe_face_full",
        "mediapipe_face_mesh",
        "mediapipe_face_mesh_eyes_only",
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
