import re

cn_model_module = {
    "inpaint": "inpaint_global_harmonious",
    "scribble": "t2ia_sketch_pidi",
    "lineart": "lineart_coarse",
    "openpose": "openpose_full",
    "tile": "tile_resample",
    "depth": "depth_midas",
    # Anytest models take the crop as it is (AUTOMATIC1111's "none"). Last:
    # a name with another type word too (anytest_depth) is of that type.
    "anytest": "none",
}
_names = [*cn_model_module, "union", "any[-_]test"]
cn_model_regex = re.compile("|".join(_names), flags=re.IGNORECASE)


def anytest_name(name: str) -> str:
    """`name` in lower case, with any-test and any_test spelled anytest."""
    return name.lower().replace("any-test", "anytest").replace("any_test", "anytest")


def anytest_choices(union: list[str], fallback: str) -> list[str]:
    """The preprocessors an Anytest model lists: only "no preprocessor".

    This WebUI's module for no preprocessor (the crop as it is): Forge, Forge
    Neo and reForge list it as "None" in every preprocessor list,
    AUTOMATIC1111's sd-webui-controlnet names it "none". `fallback` when
    `union` has neither. Alone, so a module of the previous model (inpaint)
    is never kept for an Anytest model.
    """
    return [next((c for c in ("None", "none") if c in union), fallback)]


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
    if cn_type == "anytest":  # the crop as it is
        return "None"
    preferred = cn_model_module.get(cn_type)
    if preferred in choices:
        return preferred
    return next((c for c in choices if c != "None"), "None")
