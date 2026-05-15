import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from adetailer.ultralytics import mask_to_pil, ultralytics_predict


@pytest.mark.parametrize(
    "model_name",
    [
        "face_yolov8n.pt",
        "face_yolov8n_v2.pt",
        "face_yolov8s.pt",
        "face_yolov9c.pt",
        "hand_yolov8n.pt",
        "hand_yolov8s.pt",
        "hand_yolov9c.pt",
        "person_yolov8n-seg.pt",
        "person_yolov8s-seg.pt",
        "person_yolov8m-seg.pt",
        "deepfashion2_yolov8s-seg.pt",
    ],
)
def test_ultralytics_hf_models(sample_image: Image.Image, model_name: str):
    model_path = hf_hub_download("Bingsu/adetailer", model_name)
    result = ultralytics_predict(model_path, sample_image)
    assert result.preview is not None
    assert len(result.bboxes) > 0
    assert len(result.masks) > 0
    assert len(result.confidences) > 0
    assert len(result.bboxes) == len(result.masks) == len(result.confidences)


def test_yolo_world_default(sample_image: Image.Image):
    model_path = hf_hub_download("Bingsu/yolo-world-mirror", "yolov8x-worldv2.pt")
    result = ultralytics_predict(model_path, sample_image)
    assert result.preview is not None
    assert len(result.bboxes) > 0
    assert len(result.masks) > 0
    assert len(result.confidences) > 0
    assert len(result.bboxes) == len(result.masks) == len(result.confidences)


@pytest.mark.parametrize(
    "klass",
    [
        "person",
        "bird",
        "yellow bird",
        "person,glasses,headphone",
        "person,bird",
        "glasses,yellow bird",
    ],
)
def test_yolo_world(sample_image2: Image.Image, klass: str):
    model_path = hf_hub_download("Bingsu/yolo-world-mirror", "yolov8x-worldv2.pt")
    result = ultralytics_predict(model_path, sample_image2, classes=klass)
    assert result.preview is not None
    assert len(result.bboxes) > 0
    assert len(result.masks) > 0
    assert len(result.confidences) > 0
    assert len(result.bboxes) == len(result.masks) == len(result.confidences)


class TestMaskToPil:
    def test_mask_to_pil_float32(self):
        mask = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]], dtype=torch.float32)
        imgs = mask_to_pil(mask, shape=(2, 2))

        assert len(imgs) == 1
        img = imgs[0]
        assert isinstance(img, Image.Image)

        arr = np.array(img)
        assert arr.shape == (2, 2)
        assert arr.dtype == np.uint8

        expected = np.array([[0, 255], [0, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(arr, expected)

    def test_mask_to_pil_uint8(self):
        mask = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.uint8)
        imgs = mask_to_pil(mask, shape=(2, 2))

        assert len(imgs) == 1
        img = imgs[0]
        assert isinstance(img, Image.Image)

        arr = np.array(img)
        assert arr.shape == (2, 2)
        assert arr.dtype == np.uint8

        expected = np.array([[0, 255], [0, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(arr, expected)
