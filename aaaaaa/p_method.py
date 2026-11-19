from __future__ import annotations

import hashlib
import os
import stat


def need_call_process(p) -> bool:
    if p.scripts is None:
        return False
    i = p.batch_index
    bs = p.batch_size
    return i == bs - 1


def need_call_postprocess(p) -> bool:
    if p.scripts is None:
        return False
    return p.batch_index == 0


def is_img2img_inpaint(p) -> bool:
    return hasattr(p, "image_mask") and p.image_mask is not None


def is_inpaint_only_masked(p) -> bool:
    return hasattr(p, "inpaint_full_res") and p.inpaint_full_res


def get_i(p) -> int:
    it = p.iteration
    bs = p.batch_size
    # The host sets batch_index only after the scripts' process() hook, which
    # reads the prompt for an XYZ Prompt S/R axis: that is the first image.
    i = getattr(p, "batch_index", 0)
    return it * bs + i


def is_skip_img2img(p) -> bool:
    return getattr(p, "_ad_skip_img2img", False)


# The infotext keys where the host lists the LoRAs and the embeddings it used,
# as "name: hash, name: hash" (some WebUIs write only "Lora hashes"). Sites
# such as Civitai read them to recognise the networks.
NETWORK_HASH_KEYS = ("Lora hashes", "TI hashes")


def _join_hashes(current, extra):
    """`current` with the entries of `extra` whose name it does not list."""
    if not isinstance(extra, str) or not (current is None or isinstance(current, str)):
        return current
    entries = [entry.strip() for entry in (current or "").split(",")]
    names = {entry.split(":", 1)[0].strip() for entry in entries if entry}
    added = []
    for entry in extra.split(","):
        entry = entry.strip()
        name = entry.split(":", 1)[0].strip()
        if entry and name not in names:
            names.add(name)
            added.append(entry)
    if not added:
        return current
    return ", ".join([current, *added]) if current else ", ".join(added)


def merge_network_hashes(params, pass_params, record=None):
    """Add to `params` the LoRA / embedding hashes of a detailer pass.

    The host lists only the networks of its own prompts, so one used only in
    an ADetailer prompt was missing from the saved image. The entries already
    there are kept and a name is never listed twice. Returns the record that
    `restore_network_hashes` reads to undo this for the next image.
    """
    if not isinstance(params, dict) or not isinstance(pass_params, dict):
        return record
    if record is None or record[0] is not params:
        record = (params, {})
    for key in NETWORK_HASH_KEYS:
        current = params.get(key)
        merged = _join_hashes(current, pass_params.get(key))
        if merged is current:
            continue
        base = record[1][key][0] if key in record[1] else current
        params[key] = merged
        record[1][key] = (base, merged)
    return record


def restore_network_hashes(params, record) -> None:
    """Undo `merge_network_hashes`, so the next image lists only its own.

    The next image of the batch, the next batch and the next X/Y/Z cell use
    the same dict. A key the host has written again since is left as it is.
    """
    if not record or record[0] is not params:
        return
    for key, (base, merged) in record[1].items():
        if params.get(key) != merged:
            continue
        if base is None:
            params.pop(key, None)
        else:
            params[key] = base
    record[1].clear()


# The infotext keys with the prompts a tab's detailer pass really used, one
# part per inpainted region, joined with [SEP]. Written only for a tab where a
# wildcard or variant was drawn: the tab's "ADetailer prompt" stays as typed,
# so pasted parameters draw again. Pasting ignores these keys.
RESOLVED_PROMPT_KEYS = (
    "ADetailer resolved prompt",
    "ADetailer resolved negative prompt",
)
# The most UTF-8 bytes of one such value, and of all those of an image
# together (every tab, prompt and negative prompt). A JPEG holds all its
# parameters in 64 KB, at most two bytes there for each of these, and the
# WebUI fails to save one that needs more: together the values take about
# 24 KB of them at the most, and the image's other parameters keep the rest.
RESOLVED_PROMPT_MAX = 4000
RESOLVED_PROMPTS_TOTAL_MAX = 12000
# The end of a region's prompt that was cut to fit.
RESOLVED_PROMPT_CUT = " ... (cut)"


def _tab_suffix(n: int) -> str:
    # The ending of a tab's infotext keys, as aaaaaa.ui.suffix gives it: none
    # for the 1st tab, then " 2nd", " 3rd", ...
    if n == 0:
        return ""
    k = n + 1
    ending = "th" if 11 <= k % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f" {k}{ending}"


# What a script of a detailer pass can draw in a prompt: the {a|b} variants
# and the __name__ wildcards of Dynamic Prompts and the "wildcards" extension.
RESOLVED_PROMPT_MARKS = ("{", "__")


def _prompt_differs(handed: str, used: str, strip=None) -> bool:
    # Not for other spaces or line breaks, nor for the prompt comments the
    # host removed (`strip`): those are not a wildcard drawn. Nor for a text
    # put before or after the one handed over, which is still there whole:
    # the host's "Add network to prompt" adds its LoRA tag in every pass.
    if handed == used or handed.split() == used.split():
        return False
    texts = [handed]
    if strip is not None:
        try:
            texts.append(strip(handed))
        except Exception:  # noqa: BLE001
            pass
    used = " ".join(used.split())
    return not any(" ".join(text.split()) in used for text in texts)


def note_used_prompts(  # noqa: PLR0913
    region, processed, styles=None, style_db=None, strip=None,
    marks=RESOLVED_PROMPT_MARKS,
) -> None:
    """Put in `region` the prompts the host really used for it.

    `region` is [prompt, negative prompt, False, False] as the pass handed
    them to the host. Each text becomes the one of `processed`, and its flag
    becomes True when the one handed over, with the host's `styles` applied,
    held something to draw (one of `marks`) and is no longer in the text
    used: a script of the pass (Dynamic Prompts, the "wildcards" extension)
    drew a wildcard or a variant. Any other change is not one: a LoRA tag the
    host added, a prompt another script rewrote. A text that cannot be
    compared is left as it is, not flagged.
    """
    sources = (
        ("all_prompts", "apply_styles_to_prompt"),
        ("all_negative_prompts", "apply_negative_styles_to_prompt"),
    )
    for k, (attr, apply) in enumerate(sources):
        try:
            used = (getattr(processed, attr, None) or [None])[0]
            handed = region[k]
            if not isinstance(used, str) or not isinstance(handed, str):
                continue
            expected = getattr(style_db, apply)(handed, styles) if styles else handed
            differs = any(mark in expected for mark in marks) and _prompt_differs(
                expected, used, strip
            )
            region[k] = used
            region[k + 2] = bool(region[k + 2]) or differs
        except Exception:  # noqa: BLE001, PERF203
            continue


def _utf8_size(text: str) -> int:
    return len(text.encode("utf-8", "replace"))


def _resolved_value(parts: list[str], room: int) -> str:
    """The regions' prompts joined with [SEP], in `room` UTF-8 bytes (the
    note that ends a cut value may go past it). When they do not all fit: the
    whole ones that do, then how many are left out. When not even the first
    one fits: its beginning, ended with RESOLVED_PROMPT_CUT."""
    value = " [SEP] ".join(parts)
    if not value.strip() or _utf8_size(value) <= room:
        return value
    kept: list[str] = []
    size = 0
    for part in parts:
        size += _utf8_size(part) + len(" [SEP] ")
        if size > room:
            break
        kept.append(part)
    if not kept:
        # Whole characters: one cut in the middle is left out.
        cut = max(room - len(RESOLVED_PROMPT_CUT), 0)
        head = parts[0].encode("utf-8", "replace")[:cut].decode("utf-8", "ignore")
        if head.strip():
            kept.append(head.rstrip() + RESOLVED_PROMPT_CUT)
    left = len(parts) - len(kept)
    return " [SEP] ".join([*kept, f"(+{left} more)"] if left else kept)


def _write_resolved_prompts(params, sfx: str, regions) -> None:
    own = [name + sfx for name in RESOLVED_PROMPT_KEYS]
    # What the image's other tabs left of the room for all its values.
    room = RESOLVED_PROMPTS_TOTAL_MAX - sum(
        _utf8_size(str(value))
        for key, value in params.items()
        if isinstance(key, str)
        and key.startswith(RESOLVED_PROMPT_KEYS)
        and key not in own
    )
    for k, key in enumerate(own):
        parts = [str(region[k]) for region in regions]
        # As in a prompt typed with [SEP]: the last part is also the one of
        # the regions after it.
        while len(parts) > 1 and parts[-1] == parts[-2]:
            parts.pop()
        # Too long for its own room or for the image's: cut, with a note
        # that says so, never left out without one.
        value = _resolved_value(parts, min(RESOLVED_PROMPT_MAX, room))
        # Never an empty value: AUTOMATIC1111's infotext parser fails on one.
        if any(region[k + 2] for region in regions) and value.strip():
            params[key] = value
            room -= _utf8_size(value)
        else:
            params.pop(key, None)


def start_resolved_prompts(p) -> None:
    """A new image: forget the regions of the one before and take its
    "ADetailer resolved prompt" keys out of the parameters. The next image of
    a batch, the next batch and the next X/Y/Z cell use the same dict."""
    try:
        p._ad_resolved_prompts = {}
    except Exception:  # noqa: BLE001
        pass
    params = getattr(p, "extra_generation_params", None)
    if not isinstance(params, dict):
        return
    for key in [
        k for k in params if isinstance(k, str) and k.startswith(RESOLVED_PROMPT_KEYS)
    ]:
        del params[key]


def end_resolved_prompts(p) -> None:
    """The job is over, and the keys still in the parameters are those of its
    last image. What reads the dict from here on gives them the prompt and
    seed of the first image (the X/Y/Z plot script, for its grid): after a
    job of several images they are taken out. The keys of a job's only image
    stay. The host's own grid of a batch is written before any hook an
    extension has, and is not reached."""
    try:
        if get_i(p) > 0:
            start_resolved_prompts(p)
    except Exception:  # noqa: BLE001
        pass


def keep_resolved_prompts(p, n: int, regions) -> None:
    """The regions tab `n` (0-based) has just inpainted and kept, in order:
    [prompt, negative prompt, prompt resolved, negative prompt resolved]
    each. Writes the tab's keys into p's parameters when a text was resolved;
    a tab where none was writes nothing, so its parameters stay as before."""
    params = getattr(p, "extra_generation_params", None)
    if not isinstance(params, dict) or not regions:
        return
    kept = getattr(p, "_ad_resolved_prompts", None)
    if not isinstance(kept, dict):
        kept = {}
        p._ad_resolved_prompts = kept
    # A sequential tab calls this once for each class pass.
    kept.setdefault(n, []).extend(regions)
    _write_resolved_prompts(params, _tab_suffix(n), kept[n])


def drop_resolved_prompts(p, n: int) -> None:
    """Tab `n`'s pass was rolled back (a sequential tab cancelled halfway):
    its regions are not in the image."""
    kept = getattr(p, "_ad_resolved_prompts", None)
    if isinstance(kept, dict):
        kept.pop(n, None)
    params = getattr(p, "extra_generation_params", None)
    if isinstance(params, dict):
        for name in RESOLVED_PROMPT_KEYS:
            params.pop(name + _tab_suffix(n), None)


# The infotext keys with the short hash of a tab's detector file and of the
# tab's own checkpoint ("Use separate checkpoint"): the first 10 characters of
# the file's sha256, as in the WebUI's "Model hash". Written only when there
# is one. Pasting ignores these keys.
MODEL_HASH_KEYS = ("ADetailer model hash", "ADetailer checkpoint hash")
# A detector file is a few MB to a few hundred: nothing larger is read.
DETECTOR_HASH_MAX_BYTES = 1 << 30
# path -> (modification time, size, short hash): each file is read once, and
# again only when it is replaced.
_DETECTOR_HASHES: dict[str, tuple[int, int, str]] = {}


def _hashing_off() -> bool:
    # The WebUI's --no-hashing: the user asked for no file to be hashed.
    try:
        from modules import shared

        return bool(getattr(shared.cmd_opts, "no_hashing", False))
    except Exception:  # noqa: BLE001 -- a WebUI without the flag
        return False


def detector_hash(path) -> str | None:
    """The short sha256 of a detector's file, or None when there is no such
    file (a MediaPipe detector has none), it cannot be read, or the WebUI
    runs with --no-hashing. Remembered by path, modification time and size."""
    try:
        if not path or _hashing_off():
            return None
        path = os.fspath(path)
        info = os.stat(path)
        # Nor an empty file, whose hash is the same for every one.
        if not stat.S_ISREG(info.st_mode) or not (
            0 < info.st_size <= DETECTOR_HASH_MAX_BYTES
        ):
            return None
        known = _DETECTOR_HASHES.get(path)
        if known is not None and known[:2] == (info.st_mtime_ns, info.st_size):
            return known[2]
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        short = digest.hexdigest()[:10]
        _DETECTOR_HASHES[path] = (info.st_mtime_ns, info.st_size, short)
    except Exception:  # noqa: BLE001 -- no hash, never an error
        return None
    return short


def known_checkpoint_hash(name) -> str | None:
    """The short hash the WebUI already holds for the checkpoint a tab names
    (its "Model hash"), or None when it has none yet. Only read from the
    WebUI's own record of the checkpoint: working it out reads a file of
    several GB, which the WebUI does itself when it loads the checkpoint."""
    try:
        from modules import sd_models

        # The same lookup the WebUI makes for a pass's own checkpoint.
        aliases = getattr(sd_models, "checkpoint_aliases", None)
        if isinstance(aliases, dict):
            info = aliases.get(name)
        else:
            info = sd_models.get_closet_checkpoint_match(name)
        short = getattr(info, "shorthash", None)
        if not short:
            sha256 = getattr(info, "sha256", None)
            short = sha256[:10] if isinstance(sha256, str) else None
        if isinstance(short, str) and short.strip():
            return short.strip()
    except Exception:  # noqa: BLE001 -- a WebUI without these: no hash
        pass
    return None


def add_model_hashes(params, n: int, detector=None) -> None:
    """Add to `params` the hash keys of tab `n` (0-based) that are missing:
    the detector's, for the file `detector` when the tab's "ADetailer model"
    is written, and the one of the tab's "ADetailer checkpoint" when the
    WebUI knows it. A key without a hash is not written."""
    if not isinstance(params, dict):
        return
    sfx = _tab_suffix(n)
    key = MODEL_HASH_KEYS[0] + sfx
    if detector and "ADetailer model" + sfx in params and key not in params:
        short = detector_hash(detector)
        if short:
            params[key] = short
    key = MODEL_HASH_KEYS[1] + sfx
    name = params.get("ADetailer checkpoint" + sfx)
    if (
        isinstance(name, str)
        and name not in ("", "None", "Use same checkpoint")
        and key not in params
    ):
        short = known_checkpoint_hash(name)
        if short:
            params[key] = short
