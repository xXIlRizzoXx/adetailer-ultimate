import re

cn_model_module = {
    "inpaint": "inpaint_global_harmonious",
    "scribble": "t2ia_sketch_pidi",
    "lineart": "lineart_coarse",
    "openpose": "openpose_full",
    "tile": "tile_resample",
    "depth": "depth_midas",
}
_names = [*cn_model_module, "union"]
cn_model_regex = re.compile("|".join(_names), flags=re.IGNORECASE)


def default_cn_module(cn_type: str, choices: list[str]) -> str:
    """The preprocessor a ControlNet model of `cn_type` starts with.

    AUTOMATIC1111 lists only real preprocessors: the first one, the same the
    AUTOMATIC1111 backend picks for a model without one. Forge, Forge Neo and
    reForge list "None" (no preprocessor) first: the type's usual one from
    cn_model_module when they have it, else their first real one.
    """
    if not choices:
        return "None"
    if choices[0] != "None":
        return choices[0]
    preferred = cn_model_module.get(cn_type)
    if preferred in choices:
        return preferred
    return next((c for c in choices if c != "None"), "None")
