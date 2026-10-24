import numpy as np
from PIL import Image, ImageDraw

from adetailer.common import create_bbox_from_mask, create_mask_from_bbox


def test_create_mask_from_bbox():
    img = Image.new("L", (10, 10), color="black")
    bbox = [[1.0, 1.0, 2.0, 2.0], [7.0, 7.0, 8.0, 8.0]]
    masks = create_mask_from_bbox(bbox, img.size)
    expect1 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 255, 255, 0, 0, 0, 0, 0, 0, 0],
            [0, 255, 255, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    expect2 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 255, 255, 0],
            [0, 0, 0, 0, 0, 0, 0, 255, 255, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert len(masks) == len(bbox)
    arr1 = np.array(masks[0])
    arr2 = np.array(masks[1])
    assert arr1.shape == expect1.shape
    assert arr2.shape == expect2.shape
    assert arr1.shape == (10, 10)
    assert arr1.dtype == expect1.dtype
    assert arr2.dtype == expect2.dtype
    assert np.array_equal(arr1, expect1)
    assert np.array_equal(arr2, expect2)

    # The function correctly receives a list of masks and the shape of the image.


def test_create_bbox_from_mask():
    mask = Image.new("L", (10, 10), color="black")
    draw = ImageDraw.Draw(mask)
    draw.rectangle((2, 2, 5, 5), fill="white")

    result = create_bbox_from_mask([mask], (10, 10))

    assert isinstance(result, list)
    assert len(result) == 1
    assert all(isinstance(bbox, list) for bbox in result)
    assert all(len(bbox) == 4 for bbox in result)
    assert result[0] == [2, 2, 6, 6]

    result = create_bbox_from_mask([mask], (256, 256))
    assert result[0] == [38, 38, 166, 166]


def _model_list(monkeypatch, *dirs):
    import adetailer.common as common

    monkeypatch.setattr(common, "download_models", lambda *a, **k: {})
    return list(common.get_models(*dirs, huggingface=False))


def test_zip_in_the_detector_folders_is_named_once_and_the_list_is_unchanged(
    tmp_path, monkeypatch, capsys
):
    # Many detectors on Civitai download as a .zip, which ADetailer does not
    # read: the model never showed up and the console said nothing.
    (tmp_path / "face_detector.pt").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "hand_detector.zip").write_bytes(b"x")
    (tmp_path / "eyes [v2].ZIP").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    empty = tmp_path.parent / (tmp_path.name + "-empty")
    empty.mkdir()
    expected = _model_list(monkeypatch, empty)
    capsys.readouterr()

    # The same folder listed twice, plus a folder inside it.
    models = _model_list(monkeypatch, tmp_path, "", tmp_path, tmp_path / "sub")

    assert models == [*expected, "face_detector.pt"]
    err = capsys.readouterr().err
    assert err.count("[-] ADetailer:") == 1
    assert err.count("hand_detector.zip") == 1
    assert "eyes [v2].ZIP" in err
    assert "found 2 .zip file(s)" in err
    assert "extract the .pt file" in err


def test_no_zip_message_without_zip_files(tmp_path, monkeypatch, capsys):
    (tmp_path / "face_detector.pt").write_bytes(b"x")
    _model_list(monkeypatch, tmp_path, tmp_path / "missing")
    assert ".zip" not in capsys.readouterr().err


def test_zip_message_names_a_few_files_in_ascii(tmp_path, monkeypatch, capsys):
    for i in range(7):
        (tmp_path / f"detector_{i}_ü.zip").write_bytes(b"x")
    _model_list(monkeypatch, tmp_path)
    err = capsys.readouterr().err
    assert "found 7 .zip file(s)" in err
    assert "and 2 more" in err
    assert err.count("detector_") == 5
    assert "detector_0_\\xfc.zip" in err
    assert err.isascii()


def test_zip_check_that_fails_does_not_stop_the_model_list(
    tmp_path, monkeypatch, capsys
):
    from pathlib import Path

    (tmp_path / "face_detector.pt").write_bytes(b"x")
    real_rglob = Path.rglob

    def rglob(self, pattern):
        if pattern == "*" and self == tmp_path and not hasattr(rglob, "used"):
            rglob.used = True  # the .pt scan runs first and works
            return real_rglob(self, pattern)
        raise OSError("folder cannot be read")

    monkeypatch.setattr(Path, "rglob", rglob)
    assert "face_detector.pt" in _model_list(monkeypatch, tmp_path)


def _fake_hub(monkeypatch, *errors, result=None):
    """Replace hf_hub_download: the n-th call raises errors[n] (or returns
    `result` when that entry is None); collect what the console would print."""
    from adetailer import common

    calls, printed = [], []
    queue = list(errors)

    def fake(repo_id, file, **kwargs):
        calls.append(kwargs)
        error = queue.pop(0) if queue else None
        if error is None:
            return result
        raise error

    monkeypatch.setattr(common, "hf_hub_download", fake)
    monkeypatch.setattr(common, "print", lambda *a, **k: printed.append(" ".join(map(str, a))))
    return calls, printed


def test_hf_download_says_why_the_model_failed(monkeypatch):
    from adetailer import common

    calls, printed = _fake_hub(
        monkeypatch,
        FileNotFoundError("not in the cache"),
        OSError("[Errno 11001] getaddrinfo failed\nmore details"),
        ConnectionError("mirror unreachable"),
    )

    assert common.hf_download("face_yolov8n.pt") == "INVALID"

    # The local cache first, then Hugging Face and its mirror; the reason is
    # the first remote error, not the cache miss.
    assert calls == [
        {"local_files_only": True},
        {"etag_timeout": 1},
        {"etag_timeout": 1, "endpoint": "https://hf-mirror.com"},
    ]
    assert printed == [
        "[-] ADetailer: Failed to load model 'face_yolov8n.pt' from huggingface"
        " (OSError: [Errno 11001] getaddrinfo failed)"
    ]


def test_hf_download_reason_is_short_ascii_and_not_markup(monkeypatch):
    from adetailer import common

    long_text = "[bold]" + "é" * 400
    _calls, printed = _fake_hub(monkeypatch, *[RuntimeError(long_text)] * 3)

    assert common.hf_download("face_yolov8n.pt") == "INVALID"

    reason = printed[0].split(" (", 1)[1]
    assert printed[0].isascii()
    assert reason.startswith("RuntimeError: \\[bold]")  # escaped for rich
    assert reason.endswith("...)")
    assert len(reason) <= 205


def test_hf_download_without_huggingface_hub(monkeypatch):
    from adetailer import common

    printed = []
    monkeypatch.setattr(common, "hf_hub_download", None)
    monkeypatch.setattr(common, "print", lambda *a, **k: printed.append(a[0]))

    assert common.hf_download("face_yolov8n.pt") == "INVALID"
    assert printed == [
        "[-] ADetailer: Failed to load model 'face_yolov8n.pt' from huggingface"
        " (huggingface_hub is not installed)"
    ]


def test_hf_download_offline_mode_stays_silent(monkeypatch):
    from adetailer import common

    calls, printed = _fake_hub(monkeypatch, OSError("not in the cache"))

    assert common.hf_download("face_yolov8n.pt", check_remote=False) == "INVALID"
    assert calls == [{"local_files_only": True}]
    assert printed == []


def test_hf_download_success_prints_nothing(monkeypatch):
    from adetailer import common

    calls, printed = _fake_hub(
        monkeypatch, OSError("offline"), None, result="cached/face_yolov8n.pt"
    )

    assert common.hf_download("face_yolov8n.pt") == "cached/face_yolov8n.pt"
    assert len(calls) == 2
    assert printed == []


def test_hf_download_uses_the_cached_model_without_the_network(monkeypatch):
    # A model already in the Hugging Face cache loads with no request to
    # Hugging Face or its mirror (offline start, no traffic at each start).
    from adetailer import common

    calls, printed = _fake_hub(monkeypatch, None, result="cache/face_yolov8n.pt")

    assert common.hf_download("face_yolov8n.pt") == "cache/face_yolov8n.pt"
    assert calls == [{"local_files_only": True}]
    assert printed == []


def test_builtin_model_in_the_model_folder_is_used_without_hugging_face(
    tmp_path, monkeypatch
):
    # face_yolov8n.pt already in models/adetailer (or an extra folder): that
    # file is used and only the missing built-in models are looked up.
    import adetailer.common as common

    main = tmp_path / "adetailer"
    extra = tmp_path / "extra"
    main.mkdir()
    extra.mkdir()
    (main / "face_yolov8n.pt").write_bytes(b"x")
    (extra / "hand_yolov8n.pt").write_bytes(b"x")
    (main / "my_eyes.pt").write_bytes(b"x")
    asked = []

    def fake_download(*names, check_remote=True):
        asked.extend(names)
        return {name: f"cache/{name}" for name in names}

    monkeypatch.setattr(common, "download_models", fake_download)
    models = common.get_models(main, extra)

    assert "face_yolov8n.pt" not in asked
    assert "hand_yolov8n.pt" not in asked
    assert "face_yolov8s.pt" in asked
    assert models["face_yolov8n.pt"] == str(main / "face_yolov8n.pt")
    assert models["hand_yolov8n.pt"] == str(extra / "hand_yolov8n.pt")
    assert models["face_yolov8s.pt"] == "cache/face_yolov8s.pt"
    # Menu order unchanged: the built-in models first, in their usual order.
    assert list(models)[:6] == [
        "face_yolov8n.pt",
        "face_yolov8s.pt",
        "hand_yolov8n.pt",
        "person_yolov8n-seg.pt",
        "person_yolov8s-seg.pt",
        "yolov8x-worldv2.pt",
    ]
    assert list(models)[-1] == "my_eyes.pt"


def test_every_builtin_model_on_disk_means_no_lookup_at_all(tmp_path, monkeypatch):
    import adetailer.common as common

    for name in (
        "face_yolov8n.pt", "face_yolov8s.pt", "hand_yolov8n.pt",
        "person_yolov8n-seg.pt", "person_yolov8s-seg.pt", "yolov8x-worldv2.pt",
    ):
        (tmp_path / name).write_bytes(b"x")

    def no_download(*names, **_kwargs):
        assert not names, "no lookup expected"
        return {}

    monkeypatch.setattr(common, "download_models", no_download)
    models = common.get_models(tmp_path)

    assert models["yolov8x-worldv2.pt"] == str(tmp_path / "yolov8x-worldv2.pt")
