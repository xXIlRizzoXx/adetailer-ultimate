import numpy as np
import pytest
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
    # From beta 3 the refresh button next to the detector menu is enough.
    assert "click the refresh button next to the detector menu" in err


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
        {"etag_timeout": 1, "endpoint": "https://hf-mirror.com", "token": False},
    ]
    assert printed == [
        "[-] ADetailer: Failed to load model 'face_yolov8n.pt' from huggingface"
        " (OSError: [Errno 11001] getaddrinfo failed)"
    ]


def test_hf_download_does_not_send_the_hugging_face_token_to_the_mirror(monkeypatch):
    # hf-mirror.com is not Hugging Face: the user's saved token (or HF_TOKEN)
    # must not reach it. The detector repos are public, so it is not needed.
    from adetailer import common

    calls, _printed = _fake_hub(
        monkeypatch,
        FileNotFoundError("not in the cache"),
        OSError("huggingface.co unreachable"),
        OSError("mirror unreachable"),
    )

    assert common.hf_download("face_yolov8n.pt") == "INVALID"

    mirror = [c for c in calls if c.get("endpoint") == "https://hf-mirror.com"]
    assert len(mirror) == 1
    assert mirror[0].get("token", "missing") is False
    # The cache and huggingface.co calls are unchanged.
    assert calls[0] == {"local_files_only": True}
    assert calls[1] == {"etag_timeout": 1}


def test_token_false_sends_no_authorization_header(monkeypatch):
    # What token=False means for the installed huggingface_hub: no
    # "authorization" header, even when a token is set.
    pytest.importorskip("huggingface_hub")
    from huggingface_hub.utils import build_hf_headers

    # huggingface_hub 1.x looks up the agent running it for the User-Agent
    # header: it fetches a list from huggingface.co and saves it in HF_HOME.
    # An offline test does neither.
    lookups = []
    monkeypatch.setattr(
        "huggingface_hub.utils._headers.detect_agent",
        lambda: lookups.append(1),
        raising=False,  # not in the 0.x versions of the hosts
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_DISABLE_TELEMETRY", True)

    monkeypatch.setenv("HF_TOKEN", "hf_FAKEtokenFORtest0000000000")
    headers = {k.lower(): v for k, v in build_hf_headers(token=False).items()}
    assert "authorization" not in headers
    assert lookups == []


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
    # No failure line. The progress lines of a real download go to stdout,
    # see test_hf_download_announces_a_real_download below.
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



def _hub_online(monkeypatch, endpoint="https://huggingface.co"):
    """The Hugging Face library online and on its usual address, whatever the
    machine running the tests has set (HF_HUB_OFFLINE, HF_ENDPOINT)."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    try:
        import huggingface_hub.constants  # noqa: F401
    except Exception:  # noqa: BLE001
        return
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", False)
    monkeypatch.setattr("huggingface_hub.constants.ENDPOINT", endpoint)


def _downloaded_file(tmp_path, size=2_500_000):
    path = tmp_path / "face_yolov8n.pt"
    path.write_bytes(b"x" * size)
    return str(path)


def test_hf_download_announces_a_real_download(tmp_path, monkeypatch, capsys):
    # First start: the model is in no cache, so Hugging Face sends it. The
    # console says so before (the WebUI looks stuck meanwhile) and after.
    from adetailer import common

    _hub_online(monkeypatch)
    path = _downloaded_file(tmp_path)
    calls, printed = _fake_hub(monkeypatch, OSError("not in the cache"), None, result=path)

    assert common.hf_download("face_yolov8n.pt") == path

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == (
        "[-] ADetailer: trying to download 'face_yolov8n.pt' from Hugging Face..."
    )
    assert lines[1].startswith(
        "[-] ADetailer: downloaded 'face_yolov8n.pt' from Hugging Face (2.5 MB, "
    )
    assert lines[1].endswith(" s).")
    assert len(lines) == 2
    assert all(line.isascii() for line in lines)
    assert printed == []
    assert len(calls) == 2


def test_hf_download_says_nothing_for_a_cached_model(monkeypatch, capsys):
    # Every later start: the model comes from the local cache, no line.
    from adetailer import common

    _hub_online(monkeypatch)
    _fake_hub(monkeypatch, None, result="cache/face_yolov8n.pt")

    assert common.hf_download("face_yolov8n.pt") == "cache/face_yolov8n.pt"
    assert capsys.readouterr().out == ""


def test_hf_download_names_the_mirror_that_sent_the_model(
    tmp_path, monkeypatch, capsys
):
    # Hugging Face unreachable, hf-mirror.com answers: the lines say where
    # the model really came from.
    from adetailer import common

    _hub_online(monkeypatch)
    path = _downloaded_file(tmp_path)
    _fake_hub(
        monkeypatch, OSError("not in the cache"), OSError("unreachable"), None,
        result=path,
    )

    assert common.hf_download("face_yolov8n.pt") == path

    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == [
        "[-] ADetailer: trying to download 'face_yolov8n.pt' from Hugging Face...",
        "[-] ADetailer: could not download 'face_yolov8n.pt' from Hugging Face,"
        " trying the hf-mirror.com mirror...",
    ]
    assert lines[2].startswith(
        "[-] ADetailer: downloaded 'face_yolov8n.pt' from hf-mirror.com (2.5 MB, "
    )
    assert len(lines) == 3


def test_hf_download_failure_has_no_downloaded_line(monkeypatch, capsys):
    from adetailer import common

    _hub_online(monkeypatch)
    _calls, printed = _fake_hub(monkeypatch, *[OSError("unreachable")] * 3)

    assert common.hf_download("face_yolov8n.pt") == "INVALID"

    out = capsys.readouterr().out
    assert "downloaded" not in out
    # Nothing was downloaded: no line may say it is, or promise "first run
    # only" (without a connection these lines come back at every start).
    assert out.splitlines() == [
        "[-] ADetailer: trying to download 'face_yolov8n.pt' from Hugging Face...",
        "[-] ADetailer: could not download 'face_yolov8n.pt' from Hugging Face,"
        " trying the hf-mirror.com mirror...",
    ]
    assert printed[0].startswith("[-] ADetailer: Failed to load model")


def test_hf_download_does_not_announce_the_mirror_as_its_own_fallback(
    monkeypatch, capsys
):
    # HF_ENDPOINT is hf-mirror.com already: "could not download from
    # hf-mirror.com, trying the hf-mirror.com mirror" made no sense.
    pytest.importorskip("huggingface_hub")
    from adetailer import common

    _hub_online(monkeypatch, endpoint="https://hf-mirror.com")
    _calls, printed = _fake_hub(monkeypatch, *[OSError("unreachable")] * 3)

    assert common.hf_download("face_yolov8n.pt") == "INVALID"

    assert capsys.readouterr().out.splitlines() == [
        "[-] ADetailer: trying to download 'face_yolov8n.pt' from hf-mirror.com..."
    ]
    assert printed[0].startswith("[-] ADetailer: Failed to load model")


def test_hf_download_does_not_ask_the_mirror_twice(monkeypatch):
    # HF_ENDPOINT is hf-mirror.com already: a failed download is not tried a
    # second time on the same host.
    pytest.importorskip("huggingface_hub")
    from adetailer import common

    _hub_online(monkeypatch, endpoint="https://hf-mirror.com")
    calls, printed = _fake_hub(
        monkeypatch,
        FileNotFoundError("not in the cache"),
        OSError("mirror unreachable"),
        OSError("mirror still unreachable"),
    )

    assert common.hf_download("face_yolov8n.pt") == "INVALID"

    assert len(calls) == 2
    assert calls[0] == {"local_files_only": True}
    assert "endpoint" not in calls[1]
    assert printed == [
        "[-] ADetailer: Failed to load model 'face_yolov8n.pt' from huggingface"
        " (OSError: mirror unreachable)"
    ]


@pytest.mark.parametrize("answers", [True, False])
def test_hf_download_sends_no_token_when_hf_endpoint_is_the_mirror(
    tmp_path, monkeypatch, answers
):
    # HF_ENDPOINT is hf-mirror.com: the first (and only) request goes to the
    # mirror, so the user's saved Hugging Face token must not go with it.
    pytest.importorskip("huggingface_hub")
    from adetailer import common

    _hub_online(monkeypatch, endpoint="https://hf-mirror.com")
    path = _downloaded_file(tmp_path)
    calls, _printed = _fake_hub(
        monkeypatch,
        FileNotFoundError("not in the cache"),
        None if answers else OSError("mirror unreachable"),
        result=path,
    )

    assert common.hf_download("face_yolov8n.pt") == (path if answers else "INVALID")

    assert calls == [
        {"local_files_only": True},
        {"etag_timeout": 1, "token": False},
    ]


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://hf-mirror.com.",
        "https://www.hf-mirror.com",
        "https://WWW.HF-Mirror.com.",
        "https://cdn.eu.hf-mirror.com:443",
    ],
)
def test_hf_download_knows_the_mirror_by_other_spellings_of_its_host(
    monkeypatch, endpoint
):
    # A final dot or a subdomain is still the mirror: no token, and no second
    # request to the same place.
    pytest.importorskip("huggingface_hub")
    from adetailer import common

    _hub_online(monkeypatch, endpoint=endpoint)
    calls, _printed = _fake_hub(monkeypatch, *[OSError("unreachable")] * 3)

    assert common.hf_download("face_yolov8n.pt") == "INVALID"

    assert calls == [
        {"local_files_only": True},
        {"etag_timeout": 1, "token": False},
    ]


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://hf-mirror.com.evil.example",
        "https://nothf-mirror.com",
        "https://hf-mirror.community",
    ],
)
def test_hf_download_does_not_take_a_lookalike_host_for_the_mirror(
    monkeypatch, endpoint
):
    # Another host with the mirror's name in it is handled as any HF_ENDPOINT:
    # the request there is unchanged, then the real mirror, without a token.
    pytest.importorskip("huggingface_hub")
    from adetailer import common

    _hub_online(monkeypatch, endpoint=endpoint)
    calls, _printed = _fake_hub(monkeypatch, *[OSError("unreachable")] * 3)

    assert common.hf_download("face_yolov8n.pt") == "INVALID"

    assert calls == [
        {"local_files_only": True},
        {"etag_timeout": 1},
        {"etag_timeout": 1, "endpoint": "https://hf-mirror.com", "token": False},
    ]


def test_hub_name_drops_the_final_dot_of_the_host(monkeypatch):
    pytest.importorskip("huggingface_hub")
    from adetailer import common

    _hub_online(monkeypatch, endpoint="https://hf-mirror.com.")
    assert common._hub_name() == "hf-mirror.com"
    _hub_online(monkeypatch, endpoint="https://huggingface.co.")
    assert common._hub_name() == "Hugging Face"


def test_hf_download_keeps_the_token_for_hugging_face_itself(tmp_path, monkeypatch):
    # The usual address: the request to huggingface.co is unchanged (a token
    # is sent as before), only the mirror gets none.
    pytest.importorskip("huggingface_hub")
    from adetailer import common

    _hub_online(monkeypatch)
    path = _downloaded_file(tmp_path)
    calls, _printed = _fake_hub(
        monkeypatch, FileNotFoundError("not in the cache"), None, result=path
    )

    assert common.hf_download("face_yolov8n.pt") == path
    assert calls == [{"local_files_only": True}, {"etag_timeout": 1}]


def test_hf_download_names_the_hf_endpoint_host(tmp_path, monkeypatch, capsys):
    # HF_ENDPOINT set to a mirror: the first request goes there, not to
    # Hugging Face, and the line says so.
    pytest.importorskip("huggingface_hub")
    from adetailer import common

    _hub_online(monkeypatch, endpoint="https://hf-mirror.com")
    path = _downloaded_file(tmp_path)
    _fake_hub(monkeypatch, OSError("not in the cache"), None, result=path)

    common.hf_download("face_yolov8n.pt")

    out = capsys.readouterr().out
    assert "trying to download 'face_yolov8n.pt' from hf-mirror.com..." in out
    assert "Hugging Face" not in out


@pytest.mark.parametrize("variable", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_hf_download_says_nothing_in_hugging_face_offline_mode(
    monkeypatch, capsys, variable
):
    # Offline mode refuses every request: no line may say "downloading".
    pytest.importorskip("huggingface_hub")
    from adetailer import common

    _hub_online(monkeypatch)
    monkeypatch.setenv(variable, "1")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    _calls, printed = _fake_hub(monkeypatch, *[OSError("offline mode")] * 3)

    assert common.hf_download("face_yolov8n.pt") == "INVALID"
    assert capsys.readouterr().out == ""
    assert len(printed) == 1  # the failure line, as before


def test_hf_download_says_nothing_without_the_network_step(monkeypatch, capsys):
    # --ad-no-huggingface and the refresh button: no download, no line.
    from adetailer import common

    _hub_online(monkeypatch)
    _fake_hub(monkeypatch, OSError("not in the cache"))

    assert common.hf_download("face_yolov8n.pt", check_remote=False) == "INVALID"
    assert capsys.readouterr().out == ""


def test_hf_download_says_nothing_without_huggingface_hub(monkeypatch, capsys):
    from adetailer import common

    _hub_online(monkeypatch)
    monkeypatch.setattr(common, "hf_hub_download", None)
    monkeypatch.setattr(common, "print", lambda *a, **k: None)

    assert common.hf_download("face_yolov8n.pt") == "INVALID"
    assert capsys.readouterr().out == ""


def test_progress_line_that_cannot_be_printed_is_dropped(monkeypatch):
    from adetailer import common

    class Broken:
        def write(self, _text):
            raise OSError("console closed")

        def flush(self):
            raise OSError("console closed")

    monkeypatch.setattr(common.sys, "stdout", Broken())
    common.say_progress("[-] ADetailer: downloading 'face.pt'...")  # no error


def test_start_with_every_builtin_model_present_prints_no_progress(
    tmp_path, monkeypatch, capsys
):
    # Through get_models, as at WebUI start: one built-in model in the model
    # folder, the others in the Hugging Face cache. Not a single line.
    from adetailer import common

    _hub_online(monkeypatch)
    (tmp_path / "face_yolov8n.pt").write_bytes(b"x")
    calls = []

    def cached(repo_id, file, **kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            return f"cache/{file}"
        raise AssertionError("no network request expected")

    monkeypatch.setattr(common, "hf_hub_download", cached)
    common.get_models(tmp_path)

    assert all(kwargs == {"local_files_only": True} for kwargs in calls)
    assert "downloading" not in capsys.readouterr().out


def test_start_announces_only_the_missing_builtin_model(
    tmp_path, monkeypatch, capsys
):
    from adetailer import common

    _hub_online(monkeypatch)
    path = _downloaded_file(tmp_path)

    def one_missing(repo_id, file, **kwargs):
        if kwargs.get("local_files_only") and file != "hand_yolov8n.pt":
            return f"cache/{file}"
        if kwargs.get("local_files_only"):
            raise OSError("not in the cache")
        return path

    monkeypatch.setattr(common, "hf_hub_download", one_missing)
    common.get_models(tmp_path / "empty")

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert all("'hand_yolov8n.pt'" in line for line in lines)


_BUILTIN_MODELS = (
    "face_yolov8n.pt",
    "face_yolov8s.pt",
    "hand_yolov8n.pt",
    "person_yolov8n-seg.pt",
    "person_yolov8s-seg.pt",
    "yolov8x-worldv2.pt",
)


class _HubError(OSError):
    """An error of the request itself (no connection, a file that is not
    there), as the Hugging Face library raises them."""


_HubError.__module__ = "huggingface_hub.errors"


def _failing_hub(monkeypatch):
    """Replace hf_hub_download with one that finds nothing, as with an empty
    cache and no connection; collect (repo_id, file, kwargs) of each call."""
    from adetailer import common

    calls = []

    def fake(repo_id, file, **kwargs):
        calls.append((repo_id, file, kwargs))
        raise _HubError("not available")

    monkeypatch.setattr(common, "hf_hub_download", fake)
    monkeypatch.setattr(common, "print", lambda *a, **k: None)
    return calls


def _repo_of(name):
    from adetailer import common

    return "Bingsu/yolo-world-mirror" if name == "yolov8x-worldv2.pt" else common.REPO_ID


def test_get_models_without_hugging_face_only_reads_the_local_cache(
    tmp_path, monkeypatch
):
    # The refresh button (and Load, Paste or Generate with an unknown detector
    # through it) and --ad-no-huggingface: a built-in model missing from the
    # folders is looked up in the local cache, with no request to Hugging Face
    # or its mirror.
    from adetailer import common

    calls = _failing_hub(monkeypatch)

    models = common.get_models(tmp_path, huggingface=False)

    assert sorted(file for _repo, file, _kwargs in calls) == sorted(_BUILTIN_MODELS)
    assert all(kwargs == {"local_files_only": True} for _r, _f, kwargs in calls)
    assert all(repo == _repo_of(file) for repo, file, _kwargs in calls)
    assert not set(_BUILTIN_MODELS) & set(models)


def test_get_models_by_default_tries_hugging_face_and_its_mirror(tmp_path, monkeypatch):
    # The start of the WebUI: the same lookup goes on to the network, and a
    # download would be saved in the model folder (the first folder given).
    from adetailer import common

    _hub_online(monkeypatch)
    calls = _failing_hub(monkeypatch)

    models = common.get_models(tmp_path)

    for name in _BUILTIN_MODELS:
        steps = [kwargs for _repo, file, kwargs in calls if file == name]
        assert steps == [
            {"local_files_only": True},
            {"etag_timeout": 1, "local_dir": str(tmp_path)},
            {
                "etag_timeout": 1,
                "endpoint": "https://hf-mirror.com",
                "token": False,
                "local_dir": str(tmp_path),
            },
        ]
    assert all(repo == _repo_of(file) for repo, file, _kwargs in calls)
    assert not set(_BUILTIN_MODELS) & set(models)


def test_builtin_model_in_the_model_folder_is_used_without_hugging_face(
    tmp_path, monkeypatch
):
    # face_yolov8n.pt already in models/adetailer (or an extra folder): that
    # file is used and only the missing built-in models are looked up. With a
    # copy in both folders, the one in the first folder given wins.
    import adetailer.common as common

    main = tmp_path / "adetailer"
    extra = tmp_path / "extra"
    main.mkdir()
    extra.mkdir()
    (main / "face_yolov8n.pt").write_bytes(b"x")
    (extra / "face_yolov8n.pt").write_bytes(b"x")
    (extra / "hand_yolov8n.pt").write_bytes(b"x")
    (main / "my_eyes.pt").write_bytes(b"x")
    asked = []

    def fake_download(*names, check_remote=True, local_dir=None):
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


# --- New downloads are saved in the model folder -----------------------------
# A built-in model that is in no model folder and not in the Hugging Face cache
# used to be downloaded into that cache, where the user does not see it. It is
# now saved as a file in models/adetailer (the first folder given). Models
# already in the cache stay there.


def _saving_hub(monkeypatch, cached=()):
    """Replace hf_hub_download with one that has the `cached` models in its
    cache and sends every other one: as a file in the folder it is asked to
    save in (local_dir), else into its cache. Collect (file, kwargs) of each
    call."""
    from pathlib import Path

    from adetailer import common

    calls = []

    def fake(repo_id, file, **kwargs):
        calls.append((file, kwargs))
        if kwargs.get("local_files_only"):
            if file in cached:
                return f"cache/{file}"
            raise _HubError("not in the cache")
        if "local_dir" not in kwargs:
            return f"cache/{file}"
        path = Path(kwargs["local_dir"]) / file
        path.write_bytes(file.encode())
        return str(path)

    monkeypatch.setattr(common, "hf_hub_download", fake)
    return calls


def test_missing_builtin_model_is_downloaded_into_the_model_folder(
    tmp_path, monkeypatch, capsys
):
    from adetailer import common

    _hub_online(monkeypatch)
    folder = tmp_path / "models" / "adetailer"
    extra = tmp_path / "extra"
    folder.mkdir(parents=True)
    extra.mkdir()
    (folder / "my_eyes.pt").write_bytes(b"x")
    calls = _saving_hub(monkeypatch, cached={"face_yolov8n.pt"})

    models = common.get_models(folder, extra)

    # The model that is in the cache is used from there: no request, no copy.
    assert models["face_yolov8n.pt"] == "cache/face_yolov8n.pt"
    assert [kwargs for file, kwargs in calls if file == "face_yolov8n.pt"] == [
        {"local_files_only": True}
    ]
    assert not (folder / "face_yolov8n.pt").exists()
    # Every other one is now a file in the model folder, not in an extra one.
    missing = _BUILTIN_MODELS[1:]
    for name in missing:
        assert models[name] == str(folder / name)
        assert (folder / name).is_file()
        assert [kwargs for file, kwargs in calls if file == name] == [
            {"local_files_only": True},
            {"etag_timeout": 1, "local_dir": str(folder)},
        ]
    assert not list(extra.iterdir())
    # Same names in the same order as ever.
    assert list(models)[:6] == list(_BUILTIN_MODELS)
    assert list(models)[-1] == "my_eyes.pt"

    # The console says where the file went, without the path of the folder.
    out = capsys.readouterr().out
    lines = out.splitlines()
    done = [line for line in lines if ": downloaded " in line]
    assert len(lines) == 2 * len(missing)
    assert len(done) == len(missing)
    assert all(line.endswith(" s), saved in models/adetailer.") for line in done)
    assert out.isascii()
    assert str(tmp_path) not in out
    assert tmp_path.name not in out

    # The next start, or the refresh button: each model is found in the
    # folder, listed once under its usual name, and not asked for again.
    calls.clear()
    again = common.get_models(folder, extra)

    assert list(again) == list(models)
    assert calls == [("face_yolov8n.pt", {"local_files_only": True})]
    for name in missing:
        assert again[name] == str(folder / name)
    assert capsys.readouterr().out == ""


def test_model_sent_by_the_mirror_is_saved_in_the_model_folder_too(
    tmp_path, monkeypatch, capsys
):
    from adetailer import common

    _hub_online(monkeypatch)
    path = _downloaded_file(tmp_path)
    calls, printed = _fake_hub(
        monkeypatch, _HubError("not in the cache"), _HubError("unreachable"), None,
        result=path,
    )

    assert common.hf_download("face_yolov8n.pt", local_dir=tmp_path) == path

    # A failed request is not a problem of the folder: the mirror is asked
    # next, with the same folder and still without the token.
    assert calls == [
        {"local_files_only": True},
        {"etag_timeout": 1, "local_dir": str(tmp_path)},
        {
            "etag_timeout": 1,
            "endpoint": "https://hf-mirror.com",
            "token": False,
            "local_dir": str(tmp_path),
        },
    ]
    out = capsys.readouterr().out
    last = out.splitlines()[-1]
    assert last.startswith(
        "[-] ADetailer: downloaded 'face_yolov8n.pt' from hf-mirror.com (2.5 MB, "
    )
    assert last.endswith(" s), saved in models/adetailer.")
    assert str(tmp_path) not in out
    assert printed == []


def test_hugging_face_library_without_local_dir_keeps_its_cache(
    tmp_path, monkeypatch, capsys
):
    # An old huggingface_hub whose hf_hub_download has no local_dir: the
    # models go to the Hugging Face cache as before, without an error.
    from adetailer import common

    _hub_online(monkeypatch)
    calls = []

    def old_hub(
        repo_id, filename, local_files_only=False, etag_timeout=10, endpoint=None,
        token=None,
    ):
        calls.append(filename)
        if local_files_only:
            raise _HubError("not in the cache")
        return f"cache/{filename}"

    monkeypatch.setattr(common, "hf_hub_download", old_hub)
    models = common.get_models(tmp_path)

    for name in _BUILTIN_MODELS:
        assert models[name] == f"cache/{name}"
        assert calls.count(name) == 2
    assert not list(tmp_path.iterdir())
    out = capsys.readouterr().out
    assert "saved in" not in out
    assert "Failed to load" not in out


@pytest.mark.parametrize(
    ("version", "saved"),
    [
        ("0.14.1", False),
        ("0.22.2", False),
        ("0.23.0", True),
        ("0.36.2", True),
        ("1.0.0rc1", True),
        ("1.31.0", True),
        ("unknown", False),
    ],
)
def test_only_a_hugging_face_library_from_0_23_saves_in_the_model_folder(
    tmp_path, monkeypatch, version, saved
):
    # Before 0.23 local_dir gave a link into the cache, or moved the file in
    # from another disk in a way an interrupted download could leave half
    # done: those versions keep their cache.
    pytest.importorskip("huggingface_hub")
    from adetailer import common

    _hub_online(monkeypatch)
    monkeypatch.setattr("huggingface_hub.__version__", version)
    calls = _saving_hub(monkeypatch)

    path = common.hf_download("face_yolov8n.pt", local_dir=tmp_path)

    in_folder = str(tmp_path / "face_yolov8n.pt")
    assert path == (in_folder if saved else "cache/face_yolov8n.pt")
    assert ("local_dir" in calls[-1][1]) is saved
    assert len(calls) == 2


@pytest.mark.parametrize("folder", ["missing", "", None])
def test_model_folder_that_is_not_there_keeps_the_cache(
    tmp_path, monkeypatch, folder
):
    from adetailer import common

    _hub_online(monkeypatch)
    calls = _saving_hub(monkeypatch)
    where = tmp_path / folder if folder else folder

    assert common.hf_download("face_yolov8n.pt", local_dir=where) == (
        "cache/face_yolov8n.pt"
    )
    assert [kwargs for _file, kwargs in calls] == [
        {"local_files_only": True},
        {"etag_timeout": 1},
    ]
    # The folder is not created on the way.
    assert not list(tmp_path.iterdir())


def test_model_folder_that_cannot_be_written_keeps_the_cache(
    tmp_path, monkeypatch, capsys
):
    # A read-only folder (or a shared one without the right to write): seen
    # before the request, with a file that is created and removed at once.
    from adetailer import common

    _hub_online(monkeypatch)
    calls = _saving_hub(monkeypatch)

    def no_file(*_args, **kwargs):
        raise PermissionError(13, "Permission denied", str(kwargs["dir"]))

    monkeypatch.setattr(common.tempfile, "TemporaryFile", no_file)

    assert common.hf_download("face_yolov8n.pt", local_dir=tmp_path) == (
        "cache/face_yolov8n.pt"
    )
    assert [kwargs for _file, kwargs in calls] == [
        {"local_files_only": True},
        {"etag_timeout": 1},
    ]
    assert not list(tmp_path.iterdir())
    out = capsys.readouterr().out
    assert "saved in" not in out
    assert str(tmp_path) not in out


def test_download_that_fails_on_the_model_folder_goes_to_the_cache(
    tmp_path, monkeypatch, capsys
):
    # The folder can be written, but the download into it fails for a reason
    # that is not the request (no file locks on that disk, disk full...): the
    # same request is made again into the cache, as before this change.
    from adetailer import common

    _hub_online(monkeypatch)
    calls, printed = _fake_hub(
        monkeypatch,
        _HubError("not in the cache"),
        PermissionError(13, "Permission denied", str(tmp_path / ".cache")),
        None,
        result="cache/face_yolov8n.pt",
    )

    assert common.hf_download("face_yolov8n.pt", local_dir=tmp_path) == (
        "cache/face_yolov8n.pt"
    )
    assert calls == [
        {"local_files_only": True},
        {"etag_timeout": 1, "local_dir": str(tmp_path)},
        {"etag_timeout": 1},
    ]
    out = capsys.readouterr().out
    # The file is not in the folder, so the line does not say it is.
    assert "saved in" not in out
    assert str(tmp_path) not in out
    assert printed == []


def test_error_of_the_model_folder_is_not_the_reason_printed(
    tmp_path, monkeypatch, capsys
):
    # The folder fails and then the network too: the mirror is asked without
    # the folder, and the failure line gives the reason of the network, not
    # the error of the folder with its path.
    from adetailer import common

    _hub_online(monkeypatch)
    calls, printed = _fake_hub(
        monkeypatch,
        _HubError("not in the cache"),
        PermissionError(13, "Permission denied", str(tmp_path / ".cache")),
        _HubError("no connection"),
        _HubError("no connection"),
    )

    assert common.hf_download("face_yolov8n.pt", local_dir=tmp_path) == "INVALID"

    assert calls == [
        {"local_files_only": True},
        {"etag_timeout": 1, "local_dir": str(tmp_path)},
        {"etag_timeout": 1},
        {"etag_timeout": 1, "endpoint": "https://hf-mirror.com", "token": False},
    ]
    assert printed == [
        "[-] ADetailer: Failed to load model 'face_yolov8n.pt' from huggingface"
        " (_HubError: no connection)"
    ]
    assert str(tmp_path) not in capsys.readouterr().out


def _count_write_tests(monkeypatch):
    # The write test of the model folder, counted.
    from adetailer import common

    real = common.tempfile.TemporaryFile
    made = []

    def counted(*args, **kwargs):
        made.append(kwargs.get("dir"))
        return real(*args, **kwargs)

    monkeypatch.setattr(common.tempfile, "TemporaryFile", counted)
    return made


def test_without_hugging_face_the_model_folder_is_left_alone(tmp_path, monkeypatch):
    # --ad-no-huggingface and the refresh button, unchanged: only the local
    # cache is read, and nothing is written or tried in the model folder.
    from adetailer import common

    _hub_online(monkeypatch)
    made = _count_write_tests(monkeypatch)
    calls = _failing_hub(monkeypatch)

    models = common.get_models(tmp_path, huggingface=False)

    assert len(calls) == len(_BUILTIN_MODELS)
    assert all(kwargs == {"local_files_only": True} for _r, _f, kwargs in calls)
    assert not set(_BUILTIN_MODELS) & set(models)
    assert made == []
    assert not list(tmp_path.iterdir())


def test_offline_mode_does_not_touch_the_model_folder(tmp_path, monkeypatch):
    # Hugging Face's offline mode refuses every request anyway.
    pytest.importorskip("huggingface_hub")
    from adetailer import common

    _hub_online(monkeypatch)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    made = _count_write_tests(monkeypatch)
    calls, _printed = _fake_hub(monkeypatch, *[_HubError("offline mode")] * 3)

    assert common.hf_download("face_yolov8n.pt", local_dir=tmp_path) == "INVALID"
    assert all("local_dir" not in kwargs for kwargs in calls)
    assert made == []


def test_no_model_folder_argument_without_huggingface_hub(tmp_path, monkeypatch):
    from adetailer import common

    monkeypatch.setattr(common, "hf_hub_download", None)
    assert common._model_folder(tmp_path) == {}


def test_the_installed_hugging_face_library_can_save_in_the_model_folder(tmp_path):
    # The real hf_hub_download of this environment (0.x on AUTOMATIC1111,
    # 0.x or 1.x on the Forge family), not a stand-in: it knows local_dir.
    # Nothing is requested; the write test leaves no file behind.
    pytest.importorskip("huggingface_hub", minversion="0.23")
    from adetailer import common

    assert common._model_folder(tmp_path) == {"local_dir": str(tmp_path)}
    assert not list(tmp_path.iterdir())


def test_hugging_face_service_folder_in_the_model_folder_is_not_listed(
    tmp_path, monkeypatch, capsys
):
    # The Hugging Face library keeps a ".cache" subfolder in a folder it saves
    # into: locks, download state and the unfinished file of an interrupted
    # download. None of it is a model, and the model whose download did not
    # finish is still missing.
    import adetailer.common as common

    (tmp_path / "my_eyes.pt").write_bytes(b"x")
    expected = _model_list(monkeypatch, tmp_path)
    state = tmp_path / ".cache" / "huggingface"
    (state / "download").mkdir(parents=True)
    (state / ".gitignore").write_bytes(b"*")
    (state / "CACHEDIR.TAG").write_bytes(b"Signature")
    for name in (
        "face_yolov8n.pt.metadata",
        "face_yolov8n.pt.lock",
        "aBcD1234=.0123abcd.incomplete",
    ):
        (state / "download" / name).write_bytes(b"x")
    asked = []

    def fake_download(*names, **_kwargs):
        asked.extend(names)
        return {}

    monkeypatch.setattr(common, "download_models", fake_download)
    capsys.readouterr()
    models = list(common.get_models(tmp_path, huggingface=False))

    assert models == expected
    assert "face_yolov8n.pt" in asked
    assert capsys.readouterr().err == ""


# --- Detectors with the same file name in different folders -----------------
# Civitai gives many detectors the same file name (for example four different
# detectors all saved as "detector_v10.pt"). The second one used to vanish
# from the menu without a word.


def _model_dict(monkeypatch, *dirs):
    # The detectors read from the folders (the MediaPipe ones are always there).
    import adetailer.common as common

    monkeypatch.setattr(common, "download_models", lambda *a, **k: {})
    models = common.get_models(*dirs, huggingface=False)
    return {k: v for k, v in models.items() if not k.startswith("mediapipe")}


def _put(path, data=None):
    # Each file a different detector, unless the same bytes are given.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(str(path).encode() if data is None else data)
    return path


def test_same_file_name_in_two_subfolders_lists_both(tmp_path, monkeypatch, capsys):
    feet = _put(tmp_path / "feet" / "detector_v10.pt")
    hair = _put(tmp_path / "hair" / "detector_v10.pt")
    _put(tmp_path / "eyes.pt")

    models = _model_dict(monkeypatch, tmp_path)

    # The first one keeps the plain name, as before; the other one is listed
    # with its folder after the name.
    assert models["detector_v10.pt"] == str(feet)
    assert models["detector_v10.pt (hair)"] == str(hair)
    assert models["eyes.pt"] == str(tmp_path / "eyes.pt")
    assert len(models) == 3
    err = capsys.readouterr().err
    assert err.count("detector files are named") == 1
    # The files are named from their model folder on: never the full path.
    root = tmp_path.name
    assert f"{root}/feet/detector_v10.pt keeps the name" in err
    assert f"{root}/hair/detector_v10.pt is listed as" in err
    assert str(tmp_path) not in err
    assert ":\\" not in err and "\\" not in err
    assert "'detector_v10.pt (hair)'" in err


def test_the_same_file_keeps_the_plain_name_whatever_the_disk_order(
    tmp_path, monkeypatch
):
    # Python 3.13 (Forge Neo) walks the subfolders in reverse order and Linux
    # in the order of the disk: the plain name must not depend on it.
    import adetailer.common as common

    alpha = _put(tmp_path / "alpha" / "detector_v10.pt")
    zed = _put(tmp_path / "Zed" / "detector_v10.pt")
    _put(tmp_path / "Zed" / "deep" / "detector_v10.pt")
    _put(tmp_path / "detector_v10.pt")
    real_scan = common.scan_model_dir

    results = []
    for order in (lambda ps: ps, lambda ps: list(reversed(ps))):
        monkeypatch.setattr(
            common, "scan_model_dir", lambda p, o=order: o(sorted(real_scan(p)))
        )
        results.append(list(_model_dict(monkeypatch, tmp_path).items()))

    assert results[0] == results[1]
    models = dict(results[0])
    # The file at the top of the folder first, then the subfolders A-Z with
    # capital letters or not ("alpha" before "Zed"), each before its own
    # subfolders.
    assert models["detector_v10.pt"] == str(tmp_path / "detector_v10.pt")
    assert models["detector_v10.pt (alpha)"] == str(alpha)
    assert models["detector_v10.pt (Zed)"] == str(zed)
    assert models["detector_v10.pt (Zed/deep)"] == str(
        tmp_path / "Zed" / "deep" / "detector_v10.pt"
    )
    assert list(models) == [
        "detector_v10.pt",
        "detector_v10.pt (alpha)",
        "detector_v10.pt (Zed)",
        "detector_v10.pt (Zed/deep)",
    ]


def test_duplicate_in_an_extra_folder_and_the_same_folder_listed_twice(
    tmp_path, monkeypatch
):
    main = tmp_path / "adetailer"
    extra = tmp_path / "more_detectors"
    first = _put(main / "hand.pt")
    nested = _put(main / "sub" / "hand.pt")
    second = _put(extra / "hand.pt")

    import adetailer.common as common

    compared = []
    real_same_content = common._same_content

    def same_content(a, b):
        compared.append({str(a), str(b)})
        return real_same_content(a, b)

    monkeypatch.setattr(common, "_same_content", same_content)

    # The extra folder, the main folder again and one of its subfolders: the
    # same file reached twice is still one model, not a duplicate, and it is
    # not read again to tell (only different files are compared).
    models = _model_dict(monkeypatch, main, extra, main, main / "sub")

    assert all(len(pair) == 2 for pair in compared)
    assert len(compared) == 3

    assert models == {
        "hand.pt": str(first),
        "hand.pt (sub)": str(nested),
        "hand.pt (more_detectors)": str(second),
    }


def test_duplicate_of_a_builtin_model_name_is_listed_too(tmp_path, monkeypatch):
    import adetailer.common as common

    top = _put(tmp_path / "face_yolov8n.pt")
    other = _put(tmp_path / "old" / "face_yolov8n.pt")
    asked = []

    def fake_download(*names, **_kwargs):
        asked.extend(names)
        return {}

    monkeypatch.setattr(common, "download_models", fake_download)
    models = common.get_models(tmp_path, huggingface=False)

    assert "face_yolov8n.pt" not in asked
    assert models["face_yolov8n.pt"] == str(top)
    assert models["face_yolov8n.pt (old)"] == str(other)
    # The built-in name stays in its usual place at the top of the menu.
    assert list(models)[0] == "face_yolov8n.pt"


def test_duplicate_name_never_looks_like_a_yolo_world_model(tmp_path, monkeypatch):
    # The panel takes any detector with "-world" in its name for YOLO-World.
    _put(tmp_path / "hand.pt")
    _put(tmp_path / "yolo-world" / "hand.pt")
    _put(tmp_path / "a" / "yolov8s-worldv2.pt")
    _put(tmp_path / "b" / "yolov8s-worldv2.pt")

    models = _model_dict(monkeypatch, tmp_path)

    assert "hand.pt (yolo_world)" in models
    assert not any("-world" in name for name in models if name.startswith("hand"))
    # A real YOLO-World duplicate keeps "-world" from its file name.
    assert "yolov8s-worldv2.pt (b)" in models


def test_two_duplicates_with_the_same_folder_label_get_different_names(
    tmp_path, monkeypatch
):
    main = tmp_path / "adetailer"
    extra = tmp_path / "feet"
    _put(main / "x.pt")
    _put(main / "feet" / "x.pt")
    last = _put(extra / "x.pt")

    models = _model_dict(monkeypatch, main, extra)

    assert models["x.pt (feet)"] == str(main / "feet" / "x.pt")
    assert models["x.pt (feet 2)"] == str(last)


def test_unique_names_are_unchanged(tmp_path, monkeypatch, capsys):
    # Presets, saved images and the API use these names: no duplicate, no
    # change, no message.
    _put(tmp_path / "face.pt")
    _put(tmp_path / "Hand" / "hand.pt")
    _put(tmp_path / "hand" / "deep" / "Hand.pt")

    models = _model_dict(monkeypatch, tmp_path)

    assert sorted(models) == ["Hand.pt", "face.pt", "hand.pt"]
    assert "detector files are named" not in capsys.readouterr().err


def test_the_same_detector_copied_in_two_folders_is_listed_once(
    tmp_path, monkeypatch, capsys
):
    # A shared folder added as an extra folder often holds copies of the
    # same files: one model, no extra menu entry and no message, as before.
    main = tmp_path / "adetailer"
    extra = tmp_path / "shared"
    first = _put(main / "hand.pt", b"same detector")
    _put(main / "copy" / "hand.pt", b"same detector")
    _put(extra / "hand.pt", b"same detector")
    _put(extra / "face_yolov8n.pt", b"built-in")
    _put(main / "face_yolov8n.pt", b"built-in")
    other = _put(extra / "old" / "hand.pt", b"other detector")

    models = _model_dict(monkeypatch, main, extra)

    assert models["hand.pt"] == str(first)
    assert models["hand.pt (old)"] == str(other)
    assert [k for k in models if k.startswith(("hand", "face"))] == [
        "face_yolov8n.pt",
        "hand.pt",
        "hand.pt (old)",
    ]
    err = capsys.readouterr().err
    assert err.count("detector files are named") == 1
    assert "2 detector files are named 'hand.pt'" in err
    assert "copy/hand.pt" not in err


def test_a_link_with_its_own_name_to_a_listed_file_stays_in_the_menu(
    tmp_path, monkeypatch, capsys
):
    # "hand_v2.pt" is a link to "shared/hand.pt": two names, two menu entries,
    # as before. Only the same file under the SAME name is one model. The
    # link is simulated (Windows needs a privilege to create one).
    import adetailer.common as common

    target = _put(tmp_path / "shared" / "hand.pt")
    link = _put(tmp_path / "hand_v2.pt", target.read_bytes())
    same_name_link = _put(tmp_path / "other" / "hand.pt", target.read_bytes())
    real_resolve = common.Path.resolve

    def resolve(self, *args, **kwargs):
        if self in (link, same_name_link):
            return real_resolve(target, *args, **kwargs)
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(common.Path, "resolve", resolve)

    models = _model_dict(monkeypatch, tmp_path)

    assert models == {"hand_v2.pt": str(link), "hand.pt": str(same_name_link)}
    assert "detector files are named" not in capsys.readouterr().err


def test_same_name_line_has_no_full_path_and_is_not_repeated(
    tmp_path, monkeypatch, capsys
):
    # The model list is read again at every refresh and for every detector
    # name it does not know: the line is said once, then only when the files
    # with the same name changed. It never holds a full path (the user name).
    import adetailer.common as common

    monkeypatch.setattr(common, "_DUPLICATES_SAID", {}, raising=False)
    main = tmp_path / "adetailer"
    extra = tmp_path / "more_detectors"
    _put(main / "hand.pt")
    _put(main / "sub" / "hand.pt")
    _put(extra / "hand.pt")

    _model_dict(monkeypatch, main, extra)
    err = capsys.readouterr().err
    assert err.count("detector files are named") == 1
    assert (
        "3 detector files are named 'hand.pt'. adetailer/hand.pt keeps the name"
        " 'hand.pt'; adetailer/sub/hand.pt is listed as 'hand.pt (sub)';"
        " more_detectors/hand.pt is listed as 'hand.pt (more_detectors)'."
    ) in err
    assert str(tmp_path) not in err
    # No path from the root: no word of the line starts with it. (On Linux
    # and macOS the root is "/", which the paths inside a folder hold too.)
    words = [word.strip("'\"();,.") for word in err.split()]
    assert not any(word.startswith(tmp_path.anchor) for word in words)
    assert ":" not in err.replace("[-] ADetailer:", "")
    assert "\\" not in err

    # Read again, nothing changed: no line.
    _model_dict(monkeypatch, main, extra)
    assert capsys.readouterr().err == ""

    # One more file with that name: said again.
    _put(main / "new" / "hand.pt")
    _model_dict(monkeypatch, main, extra)
    err = capsys.readouterr().err
    assert err.count("detector files are named") == 1
    assert "4 detector files are named 'hand.pt'" in err

    # The duplicates are gone, then back: said again.
    _model_dict(monkeypatch, main / "sub")
    assert capsys.readouterr().err == ""
    _model_dict(monkeypatch, main, extra)
    assert "4 detector files are named 'hand.pt'" in capsys.readouterr().err


def test_same_name_line_tells_apart_model_folders_with_the_same_name(
    tmp_path, monkeypatch, capsys
):
    # "models/adetailer" of one WebUI plus, as an extra folder, the one of
    # another WebUI: "adetailer/hand.pt keeps the name; adetailer/hand.pt is
    # listed as ..." did not say which file is which. The parent folders are
    # added, only as many as needed, and never the path above them.
    import adetailer.common as common

    monkeypatch.setattr(common, "_DUPLICATES_SAID", {}, raising=False)
    first = tmp_path / "webui_a" / "models" / "adetailer"
    second = tmp_path / "webui_b" / "models" / "adetailer"
    other = tmp_path / "more_detectors"
    _put(first / "hand.pt")
    _put(second / "hand.pt")
    _put(other / "hand.pt")

    _model_dict(monkeypatch, first, second, other)
    err = capsys.readouterr().err
    assert (
        "3 detector files are named 'hand.pt'. webui_a/models/adetailer/hand.pt"
        " keeps the name 'hand.pt'; webui_b/models/adetailer/hand.pt is listed"
        " as 'hand.pt (adetailer)'; more_detectors/hand.pt is listed as"
        " 'hand.pt (more_detectors)'."
    ) in err
    assert str(tmp_path) not in err
    assert tmp_path.name not in err


def test_short_paths_add_only_the_folders_needed_to_differ(tmp_path):
    from adetailer.common import _short_path, _short_paths

    deep = tmp_path / "a" / "x" / "adetailer"
    flat = tmp_path / "b" / "x" / "adetailer"
    alone = tmp_path / "c" / "adetailer"
    files = [(deep, deep / "hand.pt"), (flat, flat / "s" / "hand.pt")]
    # Different texts, but two model folders named the same way: the parent
    # folders are added here too.
    assert _short_paths(files) == [
        "a/x/adetailer/hand.pt",
        "b/x/adetailer/s/hand.pt",
    ]
    # Model folders with different names: the same text as _short_path.
    more = tmp_path / "more"
    files = [(deep, deep / "hand.pt"), (more, more / "hand.pt")]
    assert _short_paths(files) == [_short_path(*f) for f in files]
    assert _short_paths(files) == ["adetailer/hand.pt", "more/hand.pt"]
    # One parent folder is enough for the third file, two for the others.
    files = [(deep, deep / "hand.pt"), (flat, flat / "hand.pt")]
    files.append((alone, alone / "hand.pt"))
    assert _short_paths(files) == [
        "a/x/adetailer/hand.pt",
        "b/x/adetailer/hand.pt",
        "c/adetailer/hand.pt",
    ]


@pytest.mark.skipif(__import__("os").name != "nt", reason="drive letters and shares")
def test_short_paths_name_the_drive_or_share_when_folders_are_not_enough():
    from pathlib import Path

    from adetailer.common import _short_paths

    def short(*roots):
        return _short_paths([(Path(r), Path(r) / "hand.pt") for r in roots])

    # Next to the short text, never in front of it as a full path.
    assert short("D:/", "E:/") == ["hand.pt (on D:)", "hand.pt (on E:)"]
    assert short("D:/adetailer", "E:/adetailer") == [
        "adetailer/hand.pt (on D:)",
        "adetailer/hand.pt (on E:)",
    ]
    assert short("//srv/one", "//srv/two") == [
        "hand.pt (on //srv/one)",
        "hand.pt (on //srv/two)",
    ]
    # Only the drive differs: the parent folders say nothing, so they are not
    # shown at all.
    assert short("D:/x/models/adetailer", "E:/x/models/adetailer") == [
        "adetailer/hand.pt (on D:)",
        "adetailer/hand.pt (on E:)",
    ]
    assert short("D:/adetailer", "D:/x/adetailer", "E:/x/adetailer") == [
        "adetailer/hand.pt",
        "x/adetailer/hand.pt (on D:)",
        "x/adetailer/hand.pt (on E:)",
    ]
    # A drive root and a folder: nothing to add.
    assert short("D:/", "E:/adetailer") == ["hand.pt", "adetailer/hand.pt"]


def test_same_name_line_tells_apart_model_folders_when_one_file_is_in_a_subfolder(
    tmp_path, monkeypatch, capsys
):
    # Two model folders that end with the same name, the second file in a
    # subfolder: "adetailer/hand.pt ...; adetailer/feet/hand.pt ..." read as
    # one folder. The texts differ, but the model folders are described the
    # same way: that is what has to be told apart.
    import adetailer.common as common

    monkeypatch.setattr(common, "_DUPLICATES_SAID", {}, raising=False)
    first = tmp_path / "webui_a" / "models" / "adetailer"
    second = tmp_path / "webui_b" / "models" / "adetailer"
    _put(first / "hand.pt")
    _put(second / "feet" / "hand.pt")
    _put(second / "sub" / "hand.pt")

    _model_dict(monkeypatch, first, second)
    err = capsys.readouterr().err
    assert (
        "3 detector files are named 'hand.pt'. webui_a/models/adetailer/hand.pt"
        " keeps the name 'hand.pt'; webui_b/models/adetailer/feet/hand.pt is"
        " listed as 'hand.pt (feet)'; webui_b/models/adetailer/sub/hand.pt is"
        " listed as 'hand.pt (sub)'."
    ) in err
    assert str(tmp_path) not in err
    assert tmp_path.name not in err


def test_short_paths_of_one_model_folder_are_never_made_longer(tmp_path):
    from adetailer.common import _short_path, _short_paths

    root = tmp_path / "a" / "adetailer"
    files = [(root, root / "hand.pt"), (root, root / "feet" / "hand.pt")]
    assert _short_paths(files) == [_short_path(*f) for f in files]
    assert _short_paths(files) == ["adetailer/hand.pt", "adetailer/feet/hand.pt"]


def test_short_paths_tell_apart_files_that_read_the_same_in_nested_folders(tmp_path):
    # A model folder with the file in "feet", and two extra folders named
    # "feet": one of them inside a folder named as the first model folder.
    # The model folders are told apart, but two different files read
    # "adetailer/feet/hand.pt". One more parent folder each.
    from adetailer.common import _short_paths

    main = tmp_path / "d" / "adetailer"
    inner = tmp_path / "c" / "sd" / "adetailer" / "feet"
    other = tmp_path / "e" / "feet"
    files = [
        (main, main / "feet" / "hand.pt"),
        (inner, inner / "hand.pt"),
        (other, other / "hand.pt"),
    ]
    assert _short_paths(files) == [
        "d/adetailer/feet/hand.pt",
        "sd/adetailer/feet/hand.pt",
        "e/feet/hand.pt",
    ]


def test_short_paths_tell_apart_a_model_folder_that_is_a_drive_root(tmp_path):
    # A drive (or share) root as a model folder has no name, so it was never
    # compared with the others: its "adetailer/hand.pt" read as the one of
    # "models/adetailer".
    import os
    from pathlib import Path

    from adetailer.common import _short_paths

    top = Path(tmp_path.anchor)
    main = tmp_path / "models" / "adetailer"
    files = [(top, top / "adetailer" / "hand.pt"), (main, main / "hand.pt")]
    assert _short_paths(files) == ["adetailer/hand.pt", "models/adetailer/hand.pt"]
    if os.name != "nt":
        return

    def short(*pairs):
        return _short_paths([(Path(r), Path(r) / f) for r, f in pairs])

    assert short(
        ("C:/sd/models/adetailer", "hand.pt"), ("E:/", "adetailer/hand.pt")
    ) == ["models/adetailer/hand.pt", "adetailer/hand.pt"]
    assert short(
        ("C:/sd/models/adetailer", "hand.pt"), ("//nas/models/", "adetailer/hand.pt")
    ) == ["models/adetailer/hand.pt", "adetailer/hand.pt"]
    # No parent folder to show: the drive, as for two folders.
    assert short(("D:/", "adetailer/hand.pt"), ("E:/adetailer", "hand.pt")) == [
        "adetailer/hand.pt (on D:)",
        "adetailer/hand.pt (on E:)",
    ]


def _home_is(monkeypatch, path):
    monkeypatch.setenv("USERPROFILE", str(path))
    monkeypatch.setenv("HOME", str(path))


def test_same_name_line_never_names_the_home_folder(tmp_path, monkeypatch, capsys):
    # An extra folder right inside the home folder, same file as in
    # "models/adetailer": the parent folder that tells them apart is the home
    # folder, whose name is the user name. It is shown as "~".
    import adetailer.common as common

    monkeypatch.setattr(common, "_DUPLICATES_SAID", {}, raising=False)
    home = tmp_path / "Users" / "some_user"
    _home_is(monkeypatch, home)
    main = tmp_path / "webui" / "models" / "adetailer"
    _put(main / "hand.pt")
    _put(home / "adetailer" / "hand.pt")

    _model_dict(monkeypatch, main, home / "adetailer")
    err = capsys.readouterr().err
    assert (
        "models/adetailer/hand.pt keeps the name 'hand.pt';"
        " ~/adetailer/hand.pt is listed as 'hand.pt (adetailer)'."
    ) in err
    assert "some_user" not in err
    assert str(tmp_path) not in err


def test_short_paths_never_show_the_home_folder_or_what_is_above_it(
    tmp_path, monkeypatch
):
    from adetailer.common import _short_paths

    home = tmp_path / "Users" / "some_user"
    _home_is(monkeypatch, home)
    other = tmp_path / "other"

    def short(*roots, inside="hand.pt"):
        return _short_paths([(r, r / inside) for r in roots])

    # Under the home folder: "~" is the last folder added.
    assert short(home / "x" / "adetailer", other / "x" / "adetailer") == [
        "~/x/adetailer/hand.pt",
        "other/x/adetailer/hand.pt",
    ]
    assert short(home / "a" / "adetailer", home / "b" / "adetailer") == [
        "a/adetailer/hand.pt",
        "b/adetailer/hand.pt",
    ]
    # The home folder as a model folder, or a model folder above it.
    assert short(home, other) == ["~/hand.pt", "other/hand.pt"]
    assert short(home.parent, inside="some_user/feet/hand.pt") == ["~/feet/hand.pt"]
    # Only the home folder of this user is hidden.
    twin = home.parent / "another_user" / "adetailer"
    assert short(home / "adetailer", twin) == [
        "~/adetailer/hand.pt",
        "another_user/adetailer/hand.pt",
    ]


def test_short_paths_never_show_a_linked_home_folder(tmp_path, monkeypatch):
    # The home folder is a link to another disk ("/home/user" to
    # "/mnt/big/user"): the WebUIs give their folders with the links followed,
    # so the folder to hide is also the one the link leads to. The link is
    # only simulated (no rights to make one on Windows): os.path.realpath
    # answers as it would for a linked home folder.
    import os

    from adetailer.common import _short_paths

    link = tmp_path / "home" / "some_user"
    real = tmp_path / "mnt" / "big" / "some_user"
    _home_is(monkeypatch, link)
    realpath = os.path.realpath

    def linked(path, *args, **kwargs):
        if os.path.normcase(os.path.abspath(path)) == os.path.normcase(str(link)):
            return str(real)
        return realpath(path, *args, **kwargs)

    monkeypatch.setattr(os.path, "realpath", linked)
    other = tmp_path / "opt" / "models" / "adetailer"

    def short(*roots, inside="hand.pt"):
        return _short_paths([(r, r / inside) for r in roots])

    out = short(real / "models" / "adetailer", other)
    assert out == ["~/models/adetailer/hand.pt", "opt/models/adetailer/hand.pt"]
    assert "some_user" not in " ".join(out)
    assert short(real, other) == ["~/hand.pt", "adetailer/hand.pt"]
    assert short(real.parent, inside="some_user/feet/hand.pt") == ["~/feet/hand.pt"]
    # The home folder as written is still hidden.
    assert short(link / "models" / "adetailer", other) == out


def test_short_paths_never_show_a_folder_named_as_the_home_folder(
    tmp_path, monkeypatch
):
    # Models on a disk mounted in "/media/<user>" or "/run/media/<user>", or
    # in "D:/backup/<user>": outside the home folder, under a folder with the
    # user name. That folder is never shown, nor what is above it, and
    # without "~": it is not the home folder.
    import os

    from adetailer.common import _short_paths

    _home_is(monkeypatch, tmp_path / "home" / "some_user")
    media = tmp_path / "media" / "some_user"

    def short(*files):
        out = _short_paths([(r, r / inside) for r, inside in files])
        assert "some_user" not in " ".join(out).lower()
        assert "~" not in " ".join(out)
        return out

    # The parent folders, one at a time.
    assert short(
        (media / "x" / "models" / "adetailer", "hand.pt"),
        (tmp_path / "mnt" / "x" / "models" / "adetailer", "hand.pt"),
    ) == ["x/models/adetailer/hand.pt", "mnt/x/models/adetailer/hand.pt"]
    # The last pass, for texts that read the same.
    assert short(
        (media / "adetailer", "feet/hand.pt"),
        (tmp_path / "mnt" / "adetailer" / "feet", "hand.pt"),
        (tmp_path / "opt" / "feet", "hand.pt"),
    ) == [
        "adetailer/feet/hand.pt",
        "mnt/adetailer/feet/hand.pt",
        "opt/feet/hand.pt",
    ]
    assert short(
        (tmp_path / "run" / "media" / "some_user" / "models" / "adetailer", "hand.pt"),
        (tmp_path / "opt" / "x" / "models" / "adetailer", "hand.pt"),
        (tmp_path / "srv" / "sd" / "x" / "models" / "adetailer", "hand.pt"),
    ) == [
        "models/adetailer/hand.pt",
        "opt/x/models/adetailer/hand.pt",
        "sd/x/models/adetailer/hand.pt",
    ]
    # Compared as the system compares folder names.
    name = "Some_User" if os.path.normcase("A") == "a" else "some_user"
    assert short(
        (tmp_path / "backup" / name / "sd" / "models" / "adetailer", "hand.pt"),
        (tmp_path / "other" / "sd" / "models" / "adetailer", "hand.pt"),
    ) == ["sd/models/adetailer/hand.pt", "other/sd/models/adetailer/hand.pt"]
    # Two folders that only this name told apart are not told apart.
    assert short(
        (media / "adetailer", "hand.pt"),
        (tmp_path / "media" / "adetailer", "hand.pt"),
    ) == ["adetailer/hand.pt", "media/adetailer/hand.pt"]


@pytest.mark.skipif(__import__("os").name != "nt", reason="network shares")
def test_short_paths_never_show_a_share_named_as_the_home_folder(
    tmp_path, monkeypatch
):
    # A network share with the user name ("//server/<user>"): only the server
    # is shown after the text. Other shares stay whole.
    from pathlib import Path

    from adetailer.common import _short_paths

    _home_is(monkeypatch, tmp_path / "home" / "some_user")

    def short(*roots):
        return _short_paths([(Path(r), Path(r) / "hand.pt") for r in roots])

    assert short("//srv/Some_User/adetailer", "D:/adetailer") == [
        "adetailer/hand.pt (on //srv)",
        "adetailer/hand.pt (on D:)",
    ]
    # The hidden share of a user ("<user>$") as well.
    assert short("//srv/some_user$/adetailer", "D:/adetailer") == [
        "adetailer/hand.pt (on //srv)",
        "adetailer/hand.pt (on D:)",
    ]
    # A home folder whose own name ends with "$" (a service account).
    _home_is(monkeypatch, tmp_path / "home" / "svc$")
    for share in ("svc$", "Svc$$", "svc"):
        assert short(f"//srv/{share}/adetailer", "D:/adetailer") == [
            "adetailer/hand.pt (on //srv)",
            "adetailer/hand.pt (on D:)",
        ]
    # A share that is only "$", and the built-in ones, stay whole.
    for share in ("$", "C$", "ADMIN$"):
        assert short(f"//srv/{share}/adetailer", "D:/adetailer") == [
            f"adetailer/hand.pt (on //srv/{share})",
            "adetailer/hand.pt (on D:)",
        ]
    _home_is(monkeypatch, tmp_path / "home" / "some_user")
    assert short("//srv/another_user/adetailer", "D:/adetailer") == [
        "adetailer/hand.pt (on //srv/another_user)",
        "adetailer/hand.pt (on D:)",
    ]


def test_short_paths_without_the_home_folder_name_stay_the_same(
    tmp_path, monkeypatch
):
    import adetailer.common as common
    from adetailer.common import _short_paths

    _home_is(monkeypatch, tmp_path / "home" / "some_user")
    media = tmp_path / "media" / "another_user"

    def short(*files):
        return _short_paths([(r, r / inside) for r, inside in files])

    assert short(
        (media / "x" / "models" / "adetailer", "hand.pt"),
        (tmp_path / "mnt" / "x" / "models" / "adetailer", "hand.pt"),
    ) == [
        "another_user/x/models/adetailer/hand.pt",
        "mnt/x/models/adetailer/hand.pt",
    ]
    assert short(
        (media / "adetailer", "feet/hand.pt"),
        (tmp_path / "mnt" / "adetailer" / "feet", "hand.pt"),
        (tmp_path / "opt" / "feet", "hand.pt"),
    ) == [
        "another_user/adetailer/feet/hand.pt",
        "mnt/adetailer/feet/hand.pt",
        "opt/feet/hand.pt",
    ]
    # The name as the model folder itself, or as a folder inside it, is the
    # user's own choice of path and is shown as it is.
    assert short(
        (tmp_path / "a" / "some_user", "hand.pt"),
        (tmp_path / "b" / "some_user", "hand.pt"),
        (tmp_path / "c" / "adetailer", "some_user/hand.pt"),
    ) == ["a/some_user/hand.pt", "b/some_user/hand.pt", "adetailer/some_user/hand.pt"]
    # An unknown home folder: nothing is left out.
    monkeypatch.setattr(common, "_home_parts", lambda real=False: [])
    assert short(
        (tmp_path / "media" / "some_user" / "x" / "adetailer", "hand.pt"),
        (tmp_path / "mnt" / "x" / "adetailer", "hand.pt"),
    ) == ["some_user/x/adetailer/hand.pt", "mnt/x/adetailer/hand.pt"]


def test_short_paths_stop_after_three_parent_folders(tmp_path):
    from adetailer.common import _short_paths

    def short(*roots):
        return _short_paths([(r, r / "hand.pt") for r in roots])

    one = tmp_path / "one" / "p" / "q" / "adetailer"
    two = tmp_path / "two" / "p" / "q" / "adetailer"
    assert short(one, two) == [
        "one/p/q/adetailer/hand.pt",
        "two/p/q/adetailer/hand.pt",
    ]
    # Different only further up: never the whole path, a less precise line.
    one = tmp_path / "one" / "p" / "q" / "r" / "adetailer"
    two = tmp_path / "two" / "p" / "q" / "r" / "adetailer"
    assert short(one, two) == ["adetailer/hand.pt", "adetailer/hand.pt"]
