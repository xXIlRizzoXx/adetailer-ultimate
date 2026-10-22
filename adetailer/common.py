from __future__ import annotations

import builtins
import os
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

try:
    from huggingface_hub import hf_hub_download
except Exception:  # offline / stripped venv without huggingface_hub installed
    # Every call site lives inside a try/except Exception, so a None here
    # degrades gracefully to "INVALID" instead of crashing at import time.
    hf_hub_download = None
from PIL import Image, ImageDraw
from rich import print  # noqa: A004  Shadowing built-in 'print'
from rich.markup import escape
from torchvision.transforms.functional import to_pil_image

REPO_ID = "Bingsu/adetailer"

T = TypeVar("T", int, float)


@dataclass
class PredictOutput(Generic[T]):
    bboxes: list[list[T]] = field(default_factory=list)
    masks: list[Image.Image] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    preview: Optional[Image.Image] = None
    # Per-box class-name strings (parallel to `bboxes`), populated by
    # ultralytics_predict. Used by the combined multi-tab Detection Preview
    # to label each box like the single-tab plot. Empty for MediaPipe / when
    # unavailable. Added at the END so any positional construction is safe.
    class_names: list[str] = field(default_factory=list)


def _short_reason(error: BaseException | None) -> str:
    """The cause of a failed download in one short ASCII line, for the console:
    the exception name and the first line of its text, cut at 200 characters.
    Escaped for rich, which would read "[...]" in the text as markup."""
    if hf_hub_download is None:
        return "huggingface_hub is not installed"
    if error is None:
        return ""
    lines = str(error).strip().splitlines()
    text = f"{type(error).__name__}: {lines[0]}" if lines else type(error).__name__
    text = text.encode("ascii", "backslashreplace").decode("ascii")
    if len(text) > 200:
        text = text[:197] + "..."
    return escape(text)


def hf_download(file: str, repo_id: str = REPO_ID, check_remote: bool = True) -> str:
    # Same three tries as before (Hugging Face, its mirror, the local cache),
    # but the first remote error is kept, so the console can say WHY the
    # model could not be loaded (no connection, proxy, SSL, ...).
    error: BaseException | None = None
    if check_remote:
        try:
            return hf_hub_download(repo_id, file, etag_timeout=1)
        except Exception as e:  # noqa: BLE001
            error = e

        try:
            return hf_hub_download(
                repo_id, file, etag_timeout=1, endpoint="https://hf-mirror.com"
            )
        except Exception as e:  # noqa: BLE001
            error = error or e

    try:
        return hf_hub_download(repo_id, file, local_files_only=True)
    except Exception as e:  # noqa: BLE001
        error = error or e

    if check_remote:
        msg = f"[-] ADetailer: Failed to load model {file!r} from huggingface"
        reason = _short_reason(error)
        if reason:
            msg += f" ({reason})"
        print(msg)
    return "INVALID"


def safe_mkdir(path: str | os.PathLike[str]) -> None:
    path = Path(path)
    if not path.exists() and path.parent.exists() and os.access(path.parent, os.W_OK):
        path.mkdir()


def scan_model_dir(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return [p for p in path.rglob("*") if p.is_file() and p.suffix == ".pt"]


_ZIP_NAMES_SHOWN = 5


def warn_zip_models(*dirs: str | os.PathLike[str]) -> None:
    """Say once, when the model list is built, that a .zip in the detector
    folders is not read. Many detectors on Civitai download as a .zip, and the
    model then simply never showed up in the list. The list itself is
    unchanged; any error here is ignored."""
    try:
        # Keyed by the resolved path: the same folder listed twice (or inside
        # another listed folder) names each file once.
        found: dict[str, str] = {}
        for dir_ in dirs:
            if not dir_:
                continue
            path = Path(dir_)
            if path.is_dir():
                for p in sorted(path.rglob("*")):
                    if p.suffix.lower() == ".zip" and p.is_file():
                        found.setdefault(str(p.resolve()), p.name)
        zips = list(found.values())
        if not zips:
            return
        names = ", ".join(zips[:_ZIP_NAMES_SHOWN])
        if len(zips) > _ZIP_NAMES_SHOWN:
            names += f" and {len(zips) - _ZIP_NAMES_SHOWN} more"
        msg = (
            f"[-] ADetailer: found {len(zips)} .zip file(s) in the detector"
            f" model folders ({names}). ADetailer only reads .pt files: extract"
            " the .pt file from the .zip into the same folder and restart the"
            " WebUI."
        )
        # Plain print, not rich (a file name with [brackets] is not markup),
        # and ASCII only, so a console in a legacy code page can print it.
        builtins.print(
            msg.encode("ascii", "backslashreplace").decode("ascii"), file=sys.stderr
        )
    except Exception:  # noqa: BLE001
        pass


def download_models(*names: str, check_remote: bool = True) -> dict[str, str]:
    models = OrderedDict()
    with ThreadPoolExecutor() as executor:
        for name in names:
            if "-world" in name:
                models[name] = executor.submit(
                    hf_download,
                    name,
                    repo_id="Bingsu/yolo-world-mirror",
                    check_remote=check_remote,
                )
            else:
                models[name] = executor.submit(
                    hf_download,
                    name,
                    check_remote=check_remote,
                )
    return {name: future.result() for name, future in models.items()}


def get_models(
    *dirs: str | os.PathLike[str], huggingface: bool = True
) -> OrderedDict[str, str]:
    model_paths = []

    for dir_ in dirs:
        if not dir_:
            continue
        model_paths.extend(scan_model_dir(Path(dir_)))

    warn_zip_models(*dirs)

    models = OrderedDict()
    to_download = [
        "face_yolov8n.pt",
        "face_yolov8s.pt",
        "hand_yolov8n.pt",
        "person_yolov8n-seg.pt",
        "person_yolov8s-seg.pt",
        "yolov8x-worldv2.pt",
    ]
    models.update(download_models(*to_download, check_remote=huggingface))

    models.update(
        {
            "mediapipe_face_full": "mediapipe_face_full",
            "mediapipe_face_short": "mediapipe_face_short",
            "mediapipe_face_mesh": "mediapipe_face_mesh",
            "mediapipe_face_mesh_eyes_only": "mediapipe_face_mesh_eyes_only",
            "mediapipe_face_features": "mediapipe_face_features",
        }
    )

    invalid_keys = [k for k, v in models.items() if v == "INVALID"]
    for key in invalid_keys:
        models.pop(key)

    for path in model_paths:
        if path.name in models:
            continue
        models[path.name] = str(path)

    return models


def create_mask_from_bbox(
    bboxes: list[list[float]], shape: tuple[int, int]
) -> list[Image.Image]:
    """
    Parameters
    ----------
        bboxes: list[list[float]]
            list of [x1, y1, x2, y2]
            bounding boxes
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        masks: list[Image.Image]
        A list of masks

    """
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill=255)
        masks.append(mask)
    return masks


def create_bbox_from_mask(
    masks: list[Image.Image], shape: tuple[int, int]
) -> list[list[int]]:
    """
    Parameters
    ----------
        masks: list[Image.Image]
            A list of masks
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        bboxes: list[list[float]]
        A list of bounding boxes

    """
    bboxes = []
    for mask in masks:
        mask = mask.resize(shape)  # noqa: PLW2901
        bbox = mask.getbbox()
        if bbox is not None:
            bboxes.append(list(bbox))
    return bboxes


def ensure_pil_image(image: Any, mode: str = "RGB") -> Image.Image:
    if not isinstance(image, Image.Image):
        image = to_pil_image(image)
    if image.mode != mode:
        image = image.convert(mode)
    return image
