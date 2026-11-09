from __future__ import annotations

import importlib.util
import subprocess
import sys
from importlib.metadata import version  # python >= 3.8

from packaging.version import parse

import_name = {"py-cpuinfo": "cpuinfo", "protobuf": "google.protobuf"}


def is_installed(
    package: str,
    min_version: str | None = None,
    max_version: str | None = None,
):
    name = import_name.get(package, package)
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return False

    if spec is None:
        return False

    if not min_version and not max_version:
        return True

    if not min_version:
        min_version = "0.0.0"
    if not max_version:
        max_version = "99999999.99999999.99999999"

    try:
        pkg_version = version(package)
        return parse(min_version) <= parse(pkg_version) <= parse(max_version)
    except Exception:
        return False


def run_pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", *args], check=True)


def numpy_pin() -> list[str]:
    """Keep the host's NumPy major version when installing our deps.

    On WebUIs pinned to NumPy 1.x (e.g. AUTOMATIC1111 — its gradio 3 / skimage /
    blendmodes stack requires numpy<2) a fresh install of recent
    ultralytics/mediapipe drags NumPy up to 2.x, which breaks the whole WebUI
    with a binary-incompatibility crash. Passing this constraint tells pip's
    resolver to pick dependency versions compatible with the existing NumPy.
    No constraint on NumPy 2.x hosts (Forge Neo) or when NumPy isn't importable.
    """
    try:
        import numpy

        if int(numpy.__version__.split(".")[0]) < 2:
            return ["numpy<2"]
    except Exception:
        pass
    return []


def install():
    deps = [
        # requirements
        ("ultralytics", "8.3.75", None),
        # Ceiling on purpose: mediapipe 1.0 is a ground-up rewrite whose top-level
        # module exposes only Image / ImageFormat / tasks, and 1.0.1 aborts the
        # whole process on macOS while creating a face detector (see
        # adetailer/mediapipe.py::_tasks_api_aborts). The 0.10.x line is the one
        # this extension is tested against, so keep users on it.
        ("mediapipe", "0.10.13", "0.10.99"),
        ("rich", "13.0.0", None),
    ]

    pkgs = []
    mediapipe_cmd = None
    for pkg, low, high in deps:
        if not is_installed(pkg, low, high):
            if low and high:
                cmd = f"{pkg}>={low},<={high}"
            elif low:
                cmd = f"{pkg}>={low}"
            elif high:
                cmd = f"{pkg}<={high}"
            else:
                cmd = pkg
            if pkg == "mediapipe":
                mediapipe_cmd = cmd
            else:
                pkgs.append(cmd)

    if pkgs:
        run_pip(*pkgs, *numpy_pin())

    # Separate command: some platforms have no mediapipe 0.10.x wheel at all
    # (Linux aarch64 on Python 3.13), and a failure there must not stop
    # ultralytics from installing. Only the mediapipe_* detectors need it.
    if mediapipe_cmd:
        try:
            run_pip(mediapipe_cmd, *numpy_pin())
        except subprocess.CalledProcessError:
            print(
                f"[-] ADetailer: could not install {mediapipe_cmd}; the"
                " mediapipe_* detectors will not work, the YOLO ones will.",
                file=sys.stderr,
            )


try:
    import launch

    skip_install = launch.args.skip_install
except Exception:
    skip_install = False

if not skip_install:
    install()
