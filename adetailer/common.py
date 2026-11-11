from __future__ import annotations

import builtins
import filecmp
import inspect
import os
import sys
import tempfile
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar
from urllib.parse import urlparse

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
    # The same picture as `preview` with its boxes and masks but no class
    # names and scores, for the "Show labels" switch of the Detection preview.
    # Only the preview asks for it; None otherwise or when it can't be drawn.
    preview_plain: Optional[Image.Image] = None


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


def say_progress(message: str) -> None:
    """One console line about a slow step (a download or an install), so a
    first start that takes minutes does not look stuck. Plain ASCII on
    stdout, written in one piece so lines from parallel downloads do not
    mix; a console that cannot print it never stops the download."""
    try:
        text = message.encode("ascii", "backslashreplace").decode("ascii")
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
    except Exception:  # noqa: BLE001
        pass


def describe_file(path: str, started: float) -> str:
    """'6.2 MB, 3.1 s' for a file just downloaded; only the time when its
    size cannot be read."""
    elapsed = f"{time.monotonic() - started:.1f} s"
    try:
        return f"{os.path.getsize(path) / 1_000_000:.1f} MB, {elapsed}"
    except Exception:  # noqa: BLE001
        return elapsed


def _hub_offline() -> bool:
    """True when the Hugging Face library is in offline mode (HF_HUB_OFFLINE):
    it then refuses every download, so nothing is said to be downloading."""
    try:
        from huggingface_hub import constants

        check = getattr(constants, "is_offline_mode", None)
        if callable(check):
            return bool(check())
        return bool(getattr(constants, "HF_HUB_OFFLINE", False))
    except Exception:  # noqa: BLE001
        return False


def _hub_name() -> str:
    """Where the first download request goes: "Hugging Face", or the host of
    the HF_ENDPOINT variable when it points elsewhere (a mirror)."""
    try:
        from huggingface_hub import constants

        # "hf-mirror.com." (with the final dot) is the same host.
        host = (urlparse(str(constants.ENDPOINT)).hostname or "").rstrip(".")
    except Exception:  # noqa: BLE001
        return "Hugging Face"
    if not host or host == "huggingface.co" or host.endswith(".huggingface.co"):
        return "Hugging Face"
    return host


def _is_hf_mirror(host: str) -> bool:
    """hf-mirror.com or one of its subdomains ("www.hf-mirror.com"), not a
    host that only has the name in it ("hf-mirror.com.example.org")."""
    return host == "hf-mirror.com" or host.endswith(".hf-mirror.com")


# Errors of the request itself (no connection, proxy, SSL, a file that is not
# on the server) come from the Hugging Face library or the HTTP libraries
# under it.
_HUB_MODULES = ("huggingface_hub", "requests", "urllib3", "httpx", "httpcore")


def _model_folder(folder: str | os.PathLike[str] | None) -> dict[str, str]:
    """The hf_hub_download argument that saves a missing model as a real file
    in `folder` (models/adetailer, where the user sees it), or {} to keep it
    in the Hugging Face cache as before: no folder given, a folder that is
    not there or cannot be written, a huggingface_hub that does not know
    local_dir, or one older than 0.23. Up to 0.22 local_dir meant a link into
    the cache, or a move from another disk that an interrupted download could
    leave half done; from 0.23 it is a real file, and the unfinished download
    stays in the ".cache" subfolder the library keeps in that folder (locks
    and download state, never a .pt file) until it is complete."""
    try:
        if not folder or not Path(folder).is_dir():
            return {}
        params = inspect.signature(hf_hub_download).parameters
        if "local_dir" not in params and not any(
            p.kind is p.VAR_KEYWORD for p in params.values()
        ):
            return {}
        import huggingface_hub

        major, minor = (int(n) for n in huggingface_hub.__version__.split(".")[:2])
        if (major, minor) < (0, 23):
            return {}
        # A real write test: os.access does not see Windows permissions.
        with tempfile.TemporaryFile(dir=folder):
            pass
        return {"local_dir": str(folder)}
    except Exception:  # noqa: BLE001
        return {}


def _hub_fetch(repo_id: str, file: str, save_in: dict[str, str], **kwargs: Any) -> str:
    """hf_hub_download into the model folder of `save_in` (see _model_folder).
    When that fails for a reason that is not the request itself (the folder:
    a full disk, a file system without locks, ...), `save_in` is emptied and
    the model goes to the Hugging Face cache, as before; the error of the
    folder, which can hold its path, is dropped."""
    if save_in:
        try:
            return hf_hub_download(repo_id, file, **kwargs, **save_in)
        except Exception as e:
            if type(e).__module__.split(".")[0] in _HUB_MODULES:
                raise
            save_in.clear()
    return hf_hub_download(repo_id, file, **kwargs)


def _saved_in(path: str, save_in: dict[str, str]) -> str:
    """", saved in models/adetailer" for the console line of a download that
    did end up in the model folder, else "". Never the path of the folder: it
    holds the user name, and console lines end up pasted in public bug
    reports."""
    try:
        if save_in and Path(path).parent.samefile(save_in["local_dir"]):
            return ", saved in models/adetailer"
    except Exception:  # noqa: BLE001
        pass
    return ""


def hf_download(
    file: str,
    repo_id: str = REPO_ID,
    check_remote: bool = True,
    local_dir: str | os.PathLike[str] | None = None,
) -> str:
    # The local Hugging Face cache first: a model downloaded once loads
    # without contacting the network. Then Hugging Face and its mirror. The
    # first remote error is kept, so the console can say WHY the model could
    # not be loaded (no connection, proxy, SSL, ...).
    error: BaseException | None = None
    try:
        return hf_hub_download(repo_id, file, local_files_only=True)
    except Exception as e:  # noqa: BLE001
        cache_error = e

    if check_remote:
        # Only a real download is announced: not with --ad-no-huggingface or
        # the refresh button (check_remote=False), not in offline mode, not
        # without huggingface_hub. The failure line below stays as it was.
        announce = hf_hub_download is not None and not _hub_offline()
        hub = _hub_name()
        if announce:
            say_progress(f"[-] ADetailer: trying to download {file!r} from {hub}...")
        started = time.monotonic()
        # HF_ENDPOINT set to hf-mirror.com: this request goes to the mirror,
        # so it carries no Hugging Face token either (see token=False below).
        no_token = {"token": False} if _is_hf_mirror(hub) else {}
        # A model that is in no model folder and not in the cache is saved
        # in `local_dir` (models/adetailer), next to the user's own models.
        # Models already in the cache stay there and are used from there.
        save_in = _model_folder(local_dir) if announce else {}
        try:
            path = _hub_fetch(repo_id, file, save_in, etag_timeout=1, **no_token)
            if announce:
                say_progress(
                    f"[-] ADetailer: downloaded {file!r} from {hub}"
                    f" ({describe_file(path, started)}){_saved_in(path, save_in)}."
                )
            return path
        except Exception as e:  # noqa: BLE001
            error = e

        # token=False: the mirror is not Hugging Face, so the user's saved
        # Hugging Face token is not sent to it (the repos are public).
        # Not when HF_ENDPOINT is that mirror already: it is no other place,
        # and the request above was the only one.
        if not _is_hf_mirror(hub):
            if announce:
                say_progress(
                    f"[-] ADetailer: could not download {file!r} from {hub},"
                    " trying the hf-mirror.com mirror..."
                )
            started = time.monotonic()
            try:
                path = _hub_fetch(
                    repo_id,
                    file,
                    save_in,
                    etag_timeout=1,
                    endpoint="https://hf-mirror.com",
                    token=False,
                )
                if announce:
                    say_progress(
                        f"[-] ADetailer: downloaded {file!r} from hf-mirror.com"
                        f" ({describe_file(path, started)})"
                        f"{_saved_in(path, save_in)}."
                    )
                return path
            except Exception as e:  # noqa: BLE001
                error = error or e

    error = error or cache_error
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
    """Say, whenever the model list is built or read again, that a .zip in
    the detector folders is not read. Many detectors on Civitai download as a .zip, and the
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
            " the .pt file from the .zip into the same folder, then click the"
            " refresh button next to the detector menu or restart the WebUI."
        )
        # Plain print, not rich (a file name with [brackets] is not markup),
        # and ASCII only, so a console in a legacy code page can print it.
        builtins.print(
            msg.encode("ascii", "backslashreplace").decode("ascii"), file=sys.stderr
        )
    except Exception:  # noqa: BLE001
        pass


def download_models(
    *names: str,
    check_remote: bool = True,
    local_dir: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    models = OrderedDict()
    with ThreadPoolExecutor() as executor:
        for name in names:
            if "-world" in name:
                models[name] = executor.submit(
                    hf_download,
                    name,
                    repo_id="Bingsu/yolo-world-mirror",
                    check_remote=check_remote,
                    local_dir=local_dir,
                )
            else:
                models[name] = executor.submit(
                    hf_download,
                    name,
                    check_remote=check_remote,
                    local_dir=local_dir,
                )
    return {name: future.result() for name, future in models.items()}


def _reading_order(root: Path, path: Path) -> tuple:
    """Sort key of a detector file inside the folder it was found in: first
    the files of the folder itself, then each subfolder in A-Z order (capital
    letters or not), as AUTOMATIC1111 lists them on Windows. Fixed, so the
    same file keeps the plain name of a duplicate at every start, after the
    refresh button and on every WebUI (Python 3.13 walks subfolders in reverse
    order, and Linux in the order of the disk)."""
    rel = path.relative_to(root)
    parts = rel.parent.parts
    return (
        tuple(p.upper() for p in parts),
        rel.name.upper(),
        parts,
        rel.name,
    )


def _same_file_key(path: str | os.PathLike[str]) -> tuple[str, str]:
    # The same file reached twice (a folder listed twice, or an extra folder
    # inside models/adetailer) is one model, not a duplicate. Only under the
    # same file name: a link with a name of its own to a file listed elsewhere
    # is a model of its own, as before.
    name = os.path.basename(path)
    try:
        return os.path.normcase(str(Path(path).resolve())), name
    except Exception:  # noqa: BLE001
        return os.path.normcase(os.path.abspath(path)), name


def _same_content(a: str | os.PathLike[str], b: str | os.PathLike[str]) -> bool:
    # A copy of the same detector in two folders (a shared folder added as an
    # extra folder, say) is one model: skipped without a word, as before.
    # Different detectors almost always differ in size, so whole files are
    # read only for true copies.
    try:
        return os.path.getsize(a) == os.path.getsize(b) and filecmp.cmp(
            a, b, shallow=False
        )
    except Exception:  # noqa: BLE001
        return False


def _duplicate_label(root: Path, path: Path) -> str:
    """The folder of a duplicate as shown in its menu name: the subfolder
    inside the model folder ("feet", "pack/v2"), or the name of an extra
    folder for a file at its top. Never "-world": the panel would take any
    model with it in the name for a YOLO-World one."""
    rel = path.relative_to(root).parent.as_posix()
    if rel in ("", "."):
        rel = root.name
    return (rel or "models").replace("-world", "_world")


def _short_path(root: Path, path: Path) -> str:
    """A detector file as named in the console: the name of its model folder
    and the path inside it ("adetailer/feet/hand.pt"), as the menu names do.
    Never the full path: it holds the user name, and console lines end up
    pasted in public bug reports."""
    rel = path.relative_to(root).as_posix()
    return f"{root.name}/{rel}" if root.name else rel


def _home_parts(real: bool = False) -> list[str]:
    """The parts of the path of the user's home folder, or [] when unknown.
    With real, of the folder a linked home folder ("/home/user" as a link to
    another disk) leads to: the WebUIs give their folders that way."""
    try:
        home = os.path.expanduser("~")
        if home and not home.startswith("~"):
            if real:
                home = os.path.realpath(home)
            parts = [os.path.normcase(p) for p in Path(os.path.abspath(home)).parts]
            if len(parts) > 1:
                return parts
    except Exception:  # noqa: BLE001
        pass
    return []


def _short_paths(files: list[tuple[Path, Path]]) -> list[str]:
    """_short_path for the (model folder, file) pairs named in one console
    line. Two model folders can end with the same name ("models/adetailer" of
    two WebUIs, two drives): files of different model folders that are named
    the same way get the parent folders of their model folder in front, one
    at a time and three at most, and only until the folders differ. Folders
    that only a drive or network share tells apart get it after the text
    ("adetailer/hand.pt (on D:)"). Never a full path and never the name of
    the user's home folder ("~" instead, and nothing above it; a parent folder
    of the same name elsewhere is left out, with what is above it): a line
    that cannot tell two folders apart is better than a personal detail in
    it."""
    short = [_short_path(root, path) for root, path in files]
    try:
        homes = [h for h in (_home_parts(), _home_parts(real=True)) if h]
        users = {h[-1] for h in homes}
        count = range(len(files))
        # Which model folder, and how each one is named so far: two folders
        # named the same read as one, also when the files inside differ
        # ("adetailer/hand.pt", "adetailer/feet/hand.pt").
        which = [os.path.normcase(os.path.abspath(root)) for root, _ in files]
        named = [[root.name] if root.name else [] for root, _ in files]
        # Per file, the parent folders not shown yet, and the drive or share.
        spare = []
        drive = []
        more = []
        for i, (root, path) in enumerate(files):
            full = Path(os.path.abspath(root))
            names = list(full.parts[1:] if full.anchor else full.parts)
            if root.name:
                names = names[:-1]
            parts = [os.path.normcase(p) for p in full.parts]
            inside = Path(os.path.abspath(path)).parts
            for home in homes:
                if parts[: len(home)] == home and len(parts) > len(home):
                    # A model folder under the home folder.
                    names = ["~", *full.parts[len(home) : -1]]
                    break
                if [os.path.normcase(p) for p in inside[: len(home)]] == home:
                    # The home folder itself, or a folder above it.
                    short[i] = "/".join(["~", *inside[len(home) :]])
                    names, named[i] = [], ["~"]
                    break
            # A parent folder named as the home folder, outside it (a disk
            # mounted in "/media/<user>"): never shown, and nothing above it.
            # Without "~": it is not the home folder.
            for k in range(len(names) - 1, -1, -1):
                if os.path.normcase(names[k]) in users:
                    names = names[k + 1 :]
                    break
            spare.append(names[-3:])
            more.append(names[-3:])
            where = full.drive.replace("\\", "/")
            share = os.path.normcase(where.rsplit("/", 1)[-1]).rstrip("$")
            if share and share in {u.rstrip("$") for u in users}:
                # A share named as the home folder (also the hidden
                # "<user>$"): only its server.
                where = where.rsplit("/", 1)[0]
            drive.append(where)

        def twins(i: int) -> list[int]:
            return [j for j in count if which[j] != which[i] and named[j] == named[i]]

        while True:
            same = [i for i in count if spare[i] and twins(i)]
            if not same:
                break
            for i in same:
                named[i].insert(0, spare[i].pop())
        # Folders the parent folders did not tell apart: the drive or share
        # when it differs, and only the parent folders another folder of the
        # line makes necessary.
        left = [i for i in count if twins(i)]
        on = [
            drive[i]
            if i in left and any(drive[j] != drive[i] for j in twins(i))
            else ""
            for i in count
        ]
        for i in left:
            group = [j for j in left if named[j] == named[i]]
            keep = 1 if files[i][0].name or named[i][-1:] == ["~"] else 0
            while keep < len(named[i]) and any(
                named[j] == named[i][len(named[i]) - keep :]
                for j in count
                if j not in group
            ):
                keep += 1
            for j in group:
                named[j] = named[j][len(named[j]) - keep :]
        # Last, the texts themselves: two different files can still read the
        # same (a model folder inside a subfolder of another one, a model
        # folder that is a drive root). Parent folders not shown yet, within
        # the same three, then the drive or share; only when it tells apart.
        front = [named[i][:-1] for i in count]
        rest = [more[i][: max(0, len(more[i]) - len(front[i]))] for i in count]
        file = [os.path.normcase(os.path.abspath(path)) for _, path in files]

        def text(i: int, add: int = 0) -> str:
            head = rest[i][len(rest[i]) - add :] if add else []
            return "/".join([*head, *front[i], short[i]]) + (
                f" (on {on[i]})" if on[i] else ""
            )

        for _ in range(4 * len(files) + 1):
            groups: dict[str, list[int]] = {}
            for i in count:
                groups.setdefault(text(i), []).append(i)
            changed = False
            for group in groups.values():
                if len({file[i] for i in group}) < 2:
                    continue
                best, depth = 1, 0
                for add in (1, 2, 3):
                    told = len({text(i, add) for i in group})
                    if told > best:
                        best, depth = told, add
                if depth:
                    for i in group:
                        cut = max(0, len(rest[i]) - depth)
                        front[i][:0] = rest[i][cut:]
                        del rest[i][cut:]
                    changed = True
                elif len({drive[i] for i in group}) > 1 and not on[group[0]]:
                    for i in group:
                        on[i] = drive[i]
                    changed = True
            if not changed:
                break
        short = [text(i) for i in count]
    except Exception:  # noqa: BLE001
        pass
    return short


# The same-name files last told to the console, by file name: the model list
# is read again on every refresh and for every unknown detector name, and the
# line is only repeated when these files changed.
_DUPLICATES_SAID: dict[str, tuple] = {}


def _warn_duplicate(name: str, kept: str, listed: list[tuple[str, str]]) -> None:
    try:
        others = "; ".join(f"{where} is listed as {key!r}" for where, key in listed)
        msg = (
            f"[-] ADetailer: {len(listed) + 1} detector files are named {name!r}."
            f" {kept} keeps the name {name!r}; {others}. Rename a file to give"
            " it a name of your choice."
        )
        # Plain print and ASCII only, as warn_zip_models.
        builtins.print(
            msg.encode("ascii", "backslashreplace").decode("ascii"), file=sys.stderr
        )
    except Exception:  # noqa: BLE001
        pass


def get_models(
    *dirs: str | os.PathLike[str], huggingface: bool = True
) -> OrderedDict[str, str]:
    model_paths = []
    # The model folder each file was found in.
    found_in: dict[Path, Path] = {}

    for dir_ in dirs:
        if not dir_:
            continue
        root = Path(dir_)
        paths = sorted(scan_model_dir(root), key=lambda p, r=root: _reading_order(r, p))
        for p in paths:
            found_in.setdefault(p, root)
        model_paths.extend(paths)

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
    # A built-in model already in the model folders (models/adetailer or an
    # extra folder) is used from there, without contacting Hugging Face. The
    # first one found wins, as for the other models; the menu order stays.
    local = {}
    for path in model_paths:
        local.setdefault(path.name, str(path))
    missing = [name for name in to_download if name not in local]
    # A new download is saved in the first folder given: the WebUI's own
    # model folder (models/adetailer), the extra folders come after it.
    downloaded = (
        download_models(
            *missing, check_remote=huggingface, local_dir=dirs[0] if dirs else None
        )
        if missing
        else {}
    )
    for name in to_download:
        models[name] = local[name] if name in local else downloaded.get(name, "INVALID")

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

    # A second file with a name already taken (Civitai gives many detectors
    # the same file name) used to vanish from the menu without a word. The
    # first one found keeps the plain name, as before; each other one is
    # listed as "<name> (<folder>)" and the console says so.
    registered = {
        _same_file_key(local[name]) for name in to_download if name in local
    }
    # The files listed under each file name, the plain name's one first.
    listed_as: dict[str, list[str]] = {
        name: [local[name]] for name in to_download if name in local
    }
    duplicates: dict[str, list[tuple[str, str]]] = {}
    for path in model_paths:
        same = _same_file_key(path)
        if same in registered:
            continue
        registered.add(same)
        name = path.name
        if name not in models:
            models[name] = str(path)
            listed_as[name] = [str(path)]
            continue
        if any(_same_content(path, other) for other in listed_as.get(name, [])):
            continue
        listed_as.setdefault(name, []).append(str(path))
        key = f"{name} ({_duplicate_label(found_in[path], path)})"
        n = 2
        while key in models:
            key = f"{name} ({_duplicate_label(found_in[path], path)} {n})"
            n += 1
        models[key] = str(path)
        duplicates.setdefault(name, []).append((str(path), key))

    said = {}
    for name, listed in duplicates.items():
        kept = listed_as[name][0]
        said[name] = (kept, tuple(listed))
        if _DUPLICATES_SAID.get(name) == said[name]:
            continue
        try:
            short = _short_paths(
                [
                    (found_in[Path(p)], Path(p))
                    for p in (kept, *(where for where, _ in listed))
                ]
            )
        except Exception:  # noqa: BLE001
            continue
        _warn_duplicate(
            name, short[0], [(s, key) for s, (_, key) in zip(short[1:], listed)]
        )
    _DUPLICATES_SAID.clear()
    _DUPLICATES_SAID.update(said)

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
