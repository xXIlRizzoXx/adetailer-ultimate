from __future__ import annotations


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
