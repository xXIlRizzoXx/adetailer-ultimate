import argparse


def preload(parser: argparse.ArgumentParser):
    # Run the nudge suppression FIRST and independently of the arg
    # registration: it has its own try/except and must not be skipped if
    # add_argument ever raises (e.g. a duplicate-option error from being
    # invoked twice). Forge loads this in the main process before the
    # extension scan, which is the only window early enough to suppress it.
    _suppress_forge_neo_outdated_nudge()

    # Guarded since 2026-06-10 (community issue #1): when another
    # ADetailer-family extension (the original Bing-su adetailer, a fork,
    # ADetailer-Neo, ...) is installed alongside this one, BOTH preloads
    # register --ad-no-huggingface and argparse raises ArgumentError on the
    # second registration — the webui then prints a scary
    # "Error running preload()" traceback at startup. The flag means the
    # same thing in every variant, so reusing the existing registration is
    # correct. (Running two ADetailers remains unsupported — they still
    # duplicate the UI — but it must not crash the startup.)
    try:
        parser.add_argument(
            "--ad-no-huggingface",
            action="store_true",
            help="Don't use adetailer models from huggingface",
        )
    except argparse.ArgumentError:
        pass


def _suppress_forge_neo_outdated_nudge() -> None:
    """Remove Forge Neo's "ADetailer might be outdated" startup nudge.

    Forge Neo hardcodes a `prefer_official_extensions` table in
    `modules_forge/config.py`:

        prefer_official_extensions = {
            "ADetailer": "https://github.com/Haoming02/ADetailer-Neo",
        }

    During `modules.extensions.list_extensions()` it prints, for any extension
    folder whose name contains "adetailer" (and does NOT contain "neo"):

        *** Extension "adetailer" might be outdated!
        *** > Recommended to install "https://github.com/Haoming02/ADetailer-Neo" instead

    For THIS fork the nudge is a false positive — it is an actively maintained
    ADetailer with full Forge Neo compatibility — so we drop the table entry
    before the scan runs.

    Why it lives here and not in `scripts/!adetailer.py`: Forge loads every
    extension's `preload.py` in the MAIN process, very early, from
    `modules/shared_cmd_options.py` while building the argparse parser — that
    is BEFORE `modules.initialize.initialize_rest()` calls
    `extensions.list_extensions()` (where the warning is printed). The script
    in `scripts/` is imported later by `scripts.load_scripts()`, too late to
    suppress it. `install.py` is no help either: Forge runs it in a throw-away
    subprocess, so an in-memory mutation there wouldn't reach the process that
    prints the warning.

    `prefer_official_extensions` is imported by reference into
    `modules.extensions` and is used for nothing except this nudge, so popping
    the key has no other effect. The change is in-memory only — it never edits
    a Forge file on disk, so it survives Forge Neo updates and is undone simply
    by removing this extension. On stock A1111 / classic Forge the import fails
    and we silently no-op (those builds don't print the warning anyway).
    """
    try:
        from modules_forge.config import prefer_official_extensions

        prefer_official_extensions.pop("ADetailer", None)
    except Exception:
        # Not Forge Neo, or the table shape changed in a future release.
        # A cosmetic tweak must never break startup — leave it as-is.
        pass
