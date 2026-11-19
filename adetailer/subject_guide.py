"""The guide "several subjects, one LoRA each": choices -> tab values + status.

The panel's guide asks for a LoRA and a few words per subject and fills one
tab (two for "person, then face") so that each subject gets its own part of
a [SEP] prompt. Everything it decides is here, as plain functions without
Gradio, torch or a WebUI module, so it is tested offline. aaaaaa/ui.py only
builds the components and turns the result into updates.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from adetailer.args import BBOX_SORTBY, BBOX_SORTBY_MORE

MODE_PERSON = "Whole person (outline)"
MODE_FACE = "Face only"
MODE_PERSON_FACE = "Whole person, then face (2 tabs)"
GUIDE_MODES = (MODE_PERSON, MODE_FACE, MODE_PERSON_FACE)

# The first one installed is used. The person ones give an outline (-seg).
PERSON_DETECTORS = ("person_yolov8s-seg.pt", "person_yolov8n-seg.pt")
FACE_DETECTORS = ("face_yolov8s.pt", "face_yolov8n.pt", "mediapipe_face_full")

GUIDE_SUBJECTS = 4
# The choice "People in the picture": how many people the guide counts. With
# "Auto", as many as the last filled row says.
PEOPLE_AUTO = "Auto"
PEOPLE_CHOICES = (PEOPLE_AUTO, "2", "3", "4")
NO_LORA = "None"
# Where the weight sliders start, and the weight of a cleared one: at 0.8
# a character was less recognisable than at 1.
DEFAULT_WEIGHT = 1.0
LEFT_TO_RIGHT = "Position (left to right)"
# "Inpaint denoising strength" of every tab the guide fills. With the tab's
# 0.4 a new character barely shows inside the outline (eyes, hair, colours);
# near 0.75 the bodies break, the new one does not fit the old outline.
GUIDE_DENOISE = 0.6

# Every setting the guide can write in a tab, in the order of its outputs.
# A mode that leaves one alone simply has no value for it.
GUIDE_ATTRS = (
    "ad_tab_enable",
    "ad_model",
    "ad_model_classes",
    "ad_model_classes_exclude",
    "ad_model_classes_excluded",
    "ad_classes_sequential",
    "ad_prompt",
    "ad_strip_loras",
    "ad_use_bbox_mask",
    "ad_mask_k",
    "ad_mask_filter_method",
    "ad_mask_merge_invert",
    "ad_inpaint_indices",
    "ad_inpaint_only_masked",
    "ad_denoising_strength",
)

# As scripts/!adetailer.py reads them (_LORA_TAG_RE, _merge_lora_tags): the
# name is the text after the first ":", case-sensitive.
_LORA_TAG_RE = re.compile(r"<(?:lora|lyco):[^>]+>", re.IGNORECASE)
# A "lyco" tag: AUTOMATIC1111 reads it as a LoRA tag, Forge and Forge Neo
# skip it (no such network there).
_LYCO_TAG_RE = re.compile(r"<lyco:[^>]+>", re.IGNORECASE)
# The tokens that split a tab's prompt or skip a region: the guide writes
# them itself, so a typed one would move every later subject by one part.
_SPLIT_TOKEN_RE = re.compile(r"[ \t]*\[(?:SEP|SKIP)\][ \t]*")
_BAD_NAME_RE = re.compile(r"[:<>\r\n]|\[SEP\]|\[SKIP\]|\[PROMPT")
# A LoRA is named in a tag without the ending of its file.
_LORA_FILE_ENDINGS = (".safetensors", ".pt", ".ckpt")


# What a WebUI can take for the start of a prompt comment: "#" on
# AUTOMATIC1111, all three on Forge Neo. As the status shows them: a "*"
# written as it is could start an emphasis in the Markdown of the status.
COMMENT_MARKS = ("#", "//", "/*")
_MARK_SHOWN = {"#": "#", "//": "//", "/*": "/&#42;"}
_MARK_PROBES = {"#": "a # b", "//": "a // b", "/*": "a /* b */ c"}


# What a line of the status starts with: done, mind this, by the way.
_DONE = chr(0x2705) + " "
_WARN = chr(0x26A0) + chr(0xFE0F) + " "
_INFO = chr(0x2139) + chr(0xFE0F) + " "


# The status is shown as Markdown (gr.Markdown, read by the browser on both
# Gradio 3 and 4). A LoRA name or a word of the user quoted in it can hold
# characters that mean something there: "_new_" would be shown in italics.
# Each is written as its HTML character reference, which Markdown shows as
# the character and never reads as markup. The same for "$": a Gradio that
# looks for formulas in the text it is given (4.40: "$$...$$") finds none.
# One that looks for them in the page it has drawn (3.41: "$...$") needs
# latex_delimiters=[] on the component as well (aaaaaa/ui.py).
_MD_MARKS = str.maketrans({
    **{mark: f"&#{ord(mark)};" for mark in "\\`*_[]|~$"},
    "<": "&lt;", ">": "&gt;", "\r": " ", "\n": " ",
})


def md_plain(text: Any) -> str:
    """`text` (a name, a word of the user) as the status shows it: every
    character as typed, none of them read as Markdown or HTML, on one line."""
    return str(text).replace("&", "&amp;").translate(_MD_MARKS)


class InvalidLoraNameError(ValueError):
    """A LoRA name that cannot be written inside a <lora:...> tag."""


@dataclass
class GuideResult:
    # Tab index -> {setting: value}. Empty: nothing is written, only the status.
    tabs: dict[int, dict[str, Any]] = field(default_factory=dict)
    status: str = ""


def _ordinal(n: int) -> str:
    d = {1: "st", 2: "nd", 3: "rd"}
    return str(n) + ("th" if 11 <= n % 100 <= 13 else d.get(n % 10, "th"))


def _and(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def lora_name(value: Any) -> str:
    """The LoRA a menu holds: "" for no LoRA ("None", blank, not a text)."""
    if not isinstance(value, str):
        return ""
    name = value.strip()
    return "" if name.casefold() == NO_LORA.casefold() else name


def lora_choices(names: Iterable[Any]) -> list[str]:
    """The menu's choices: "None" first, then the names a tag can hold."""
    usable = {
        n for n in names
        if isinstance(n, str) and n and lora_name(n) == n and not _BAD_NAME_RE.search(n)
    }
    return [NO_LORA, *sorted(usable, key=lambda n: (n.casefold(), n))]


def match_lora(name: str, names: Iterable[str], aliases: Iterable[str] = ()) -> str | None:
    """The name the WebUI knows `name` by, None when it knows none.

    `names` is the WebUI's LoRA list, `aliases` the other names a tag can
    take there. A name they hold is itself. Otherwise, first among `names`:
    the name without a file ending typed by mistake (".safetensors", ".pt",
    ".ckpt"), then the ONE name that differs only by the letter case, with
    or without that ending; with two of them (my_lora and My_Lora for "MY_LORA")
    none is chosen."""
    names, aliases = list(names), list(aliases)
    if name in names or name in aliases:
        return name
    folded = name.casefold()
    stems = [
        name[: -len(ending)]
        for ending in _LORA_FILE_ENDINGS
        if folded.endswith(ending) and len(name) > len(ending)
    ]
    wanted = {folded, *(stem.casefold() for stem in stems)}
    for pool in (names, aliases):
        for stem in stems:
            if stem in pool:
                return stem
        found = {known for known in pool if known.casefold() in wanted}
        if found:
            return found.pop() if len(found) == 1 else None
    return None


def _alias_files(aliases: Any) -> dict[str, str]:
    """{alias: the name of the LoRA file it stands for} when the WebUI told
    them (a dict, as scripts/!adetailer.py _lora_aliases gives it), so that an
    alias and the file name of one LoRA count as that one LoRA. {} otherwise:
    then each name is a LoRA of its own."""
    if not isinstance(aliases, dict):
        return {}
    try:
        return {
            alias: file
            for alias, file in list(aliases.items())
            if isinstance(alias, str) and alias and isinstance(file, str) and file
        }
    except Exception:  # noqa: BLE001 - then the names alone are compared
        return {}


def _listed_rows(
    rows: list[tuple], names: Any, aliases: Any
) -> tuple[
    list[tuple], list[tuple[int, str, str]], list[tuple[int, str]], list[tuple[int, str, str]]
]:
    """(the rows with each LoRA under the name the WebUI lists, the names
    changed for that as (row, typed, listed), the names it does not know as
    (row, name), the names left as a tag in the row's words holds them,
    which the WebUI will not find, as (row, typed, listed)). Without a list
    (a WebUI that has none) nothing is compared: every name is taken as
    typed. A tag of the words under another name the WebUI finds for the
    same LoRA file (its alias, its file name: _alias_files) is that LoRA."""
    def texts(values: Any) -> list[str]:
        # A text is not a list of names: its letters are not LoRAs.
        if isinstance(values, (str, bytes)):
            return []
        try:
            return [v for v in values or () if isinstance(v, str) and v]
        except TypeError:
            return []

    names = texts(names)
    if not names:
        return rows, [], [], []
    files = _alias_files(aliases)
    aliases = texts(aliases)
    out, renamed, unknown, kept = [], [], [], []
    for i, row in enumerate(rows, start=1):
        name = lora_name(row[0])
        # A name no tag can hold is refused later, whatever the list says.
        if name and lora_tag(name, row[1]) is not None:
            listed = match_lora(name, names, aliases)
            if listed is None:
                unknown.append((i, name))
            else:
                # The tags typed in the words that name the same LoRA: each
                # name is brought to the listed one as the menu's is, and an
                # alias to its file.
                lora = files.get(listed, listed)
                typed = []
                for tag in _tag_names(clean_words(row[2])[0]):
                    found = match_lora(tag, names, aliases)
                    if found is not None and files.get(found, found) == lora:
                        typed.append(tag)
                # A tag the WebUI finds as written: its alias or its file
                # name, the other one of the menu's.
                found_as_typed = [tag for tag in typed if tag in names or tag in aliases]
                if typed and listed not in typed and found_as_typed:
                    row = (found_as_typed[0], *row[1:])
                elif typed and listed not in typed:
                    # None of them holds the name as listed, and the WebUIs
                    # look a tag's name up as written, capitals included.
                    # The words are not rewritten, so the row takes the name
                    # of the tag that stays there: no second tag is written
                    # under the listed name, and the status says it.
                    kept.append((i, typed[0], listed))
                    row = (typed[0], *row[1:])
                elif listed != name:
                    renamed.append((i, name, listed))
                    row = (listed, *row[1:])
        out.append(row)
    return out, renamed, unknown, kept


def format_weight(weight: Any) -> str:
    """0.8 -> "0.8", 1.0 -> "1"; two decimals at most. A value that is not a
    number (a cleared slider) counts as the default."""
    try:
        number = float(weight)
    except (TypeError, ValueError):
        number = DEFAULT_WEIGHT
    if isinstance(weight, bool) or not math.isfinite(number):
        number = DEFAULT_WEIGHT
    text = f"{number:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def lora_tag(name: str, weight: Any) -> str | None:
    """<lora:name:weight>, or None when the name cannot be inside a tag: it
    is empty or holds ":", "<", ">", a line break or a prompt token."""
    if not isinstance(name, str):
        return None
    name = name.strip()
    if not name or _BAD_NAME_RE.search(name):
        return None
    return f"<lora:{name}:{format_weight(weight)}>"


def clean_words(words: Any) -> tuple[str, bool]:
    """The words of a subject as they go in its part, and whether a [SEP] or
    [SKIP] was taken out. Line breaks become a space, spaces and commas at
    the two ends go; the rest is left as typed."""
    if not isinstance(words, str):
        return "", False
    text = re.sub(r"\s*[\r\n]+\s*", " ", words)
    text, removed = _SPLIT_TOKEN_RE.subn(" ", text)
    return re.sub(r"^[\s,]+|[\s,]+$", "", text), bool(removed)


def people_count(value: Any) -> int:
    """The number "People in the picture" asks for (2 to 4), or 0 for Auto
    and for any other value (nothing picked, a translated text, a number):
    then the last filled row decides alone."""
    if isinstance(value, str) and value in PEOPLE_CHOICES[1:]:
        return int(value)
    return 0


def host_comment_marks(strip: Any = None) -> tuple[str, ...]:
    """The comment marks the WebUI in use takes out of a prompt, found by
    giving `strip` (the WebUI's own function, which follows its option) one
    short text per mark. None with prompt comments off. Without a function,
    or for a mark it fails on, the mark counts: one warning too many is
    better than a subject silently dropped."""
    if not callable(strip):
        return COMMENT_MARKS
    marks = []
    for mark in COMMENT_MARKS:
        try:
            taken = strip(_MARK_PROBES[mark]) != _MARK_PROBES[mark]
        except Exception:  # noqa: BLE001 - unknown: warn
            taken = True
        if taken:
            marks.append(mark)
    return tuple(marks)


def _tag_names(text: str) -> list[str]:
    names = []
    for tag in _LORA_TAG_RE.findall(text or ""):
        name = tag[1:-1].split(":")[1]
        if name not in names:
            names.append(name)
    return names


def subject_part(lora: Any, weight: Any, words: Any) -> str | None:
    """One subject's part of the prompt, None for an empty row.

    The words first, then the tag. Without words the part starts with
    [PROMPT] (the main prompt), or the region would be redrawn from the tag
    alone. A LoRA the words already name is not written twice: the tag typed
    there, with its weight, stays. Raises InvalidLoraNameError.
    """
    name = lora_name(lora)
    text, _ = clean_words(words)
    if not name and not text:
        return None
    tag = None
    if name:
        tag = lora_tag(name, weight)
        if tag is None:
            raise InvalidLoraNameError(name)
        if name in _tag_names(text):
            tag = None
    if not re.sub(r"[\s,]", "", _LORA_TAG_RE.sub("", text)):
        text = f"[PROMPT], {text}" if text else "[PROMPT]"
    return f"{text}, {tag}" if tag else text


def _rows(subjects: Sequence[Any]) -> list[tuple]:
    """Up to four (lora, weight, words) rows, a short or odd one padded."""
    rows = [tuple(row) if isinstance(row, (list, tuple)) else () for row in subjects]
    return [(*row, None, None, None)[:3] for row in rows[:GUIDE_SUBJECTS]]


def _parts(subjects: Sequence[Any]) -> list[str | None]:
    return [subject_part(*row) for row in _rows(subjects)]


def _counted(parts: list[str | None], people: Any) -> list[str | None]:
    """The parts of the subjects that count: up to the last filled row, or
    up to the number of "People in the picture" when that is higher (the
    rows in between are empty ones). None at all without a filled row."""
    last = max((i + 1 for i, part in enumerate(parts) if part is not None), default=0)
    if not last:
        return []
    count = max(last, people_count(people))
    return [*parts, *[None] * count][:count]


def build_prompt(subjects: Sequence[Any], people: Any = None) -> tuple[str, int]:
    """The tab's prompt and the number of subjects counted: the last filled
    row, or the number of "People in the picture" (`people`) when higher.

    One part per subject, an empty row as [SKIP] (that subject is left as it
    is), and a last [SKIP]: without it every person or face found beyond
    them would be redrawn with the last part.
    """
    parts = _counted(_parts(subjects), people)
    if not parts:
        return "", 0
    texts = [part if part is not None else "[SKIP]" for part in parts]
    return " [SEP] ".join([*texts, "[SKIP]"]), len(parts)


def effective_sort_order(sort_order: Any, more_sort_order: Any = None) -> tuple[str, Any]:
    """(the setting that decides the order, its value), read as the detailer
    pass does: "More sort orders" wins unless it is "Off" or unknown."""
    if isinstance(more_sort_order, str) and more_sort_order in BBOX_SORTBY_MORE[1:]:
        return "More sort orders", more_sort_order
    return "Sort bounding boxes by", sort_order


def _order_note(
    sort_order: Any, more_sort_order: Any, count: int, kind: str
) -> tuple[bool, str]:
    """(the parts go from left to right, what the status says about it).

    The tab keeps the `count` biggest people (faces with a face detector)
    and gives them the parts from left to right: "Subject 1 is the leftmost"
    alone would be wrong with one more person in the picture."""
    one, many = ("face", "faces") if kind == "face" else ("person", "people")
    if count == 1:
        # One region is kept, the biggest: no order to follow, so whatever
        # the Settings say there is nothing to warn about.
        return True, f"Subject 1 is the biggest {one}."
    setting, value = effective_sort_order(sort_order, more_sort_order)
    if setting == "Sort bounding boxes by" and value == LEFT_TO_RIGHT:
        return True, f"Subject 1 is the leftmost of the {count} biggest {many}."
    known = isinstance(value, str) and value in (*BBOX_SORTBY, *BBOX_SORTBY_MORE)
    shown = value if known else "not readable"
    if setting == "More sort orders":
        fix = (
            f"set it to {BBOX_SORTBY_MORE[0]}, set Sort bounding boxes by to "
            f"{LEFT_TO_RIGHT} and click Apply settings."
        )
    else:
        fix = f"set it to {LEFT_TO_RIGHT} and click Apply settings."
    return False, (
        f"{_WARN}The parts follow Settings, ADetailer, {setting}, now {shown}: {fix}"
    )


def _tab_values(kind: str, prompt: str, count: int, detector: str | None) -> dict:
    values: dict[str, Any] = {
        "ad_tab_enable": True,
        "ad_classes_sequential": False,
        "ad_prompt": prompt,
        # Or the pass would take the tags out again.
        "ad_strip_loras": False,
        # The `count` largest: a small figure in the background would
        # otherwise take a part, and every later subject the wrong one.
        "ad_mask_k": count,
        "ad_mask_filter_method": "Area",
        "ad_mask_merge_invert": "None",
        "ad_inpaint_indices": "",
        "ad_inpaint_only_masked": True,
        "ad_denoising_strength": GUIDE_DENOISE,
    }
    if kind == "person":
        values["ad_use_bbox_mask"] = False  # the outline, not its box
    if detector is not None:
        values.update(
            ad_model=detector,
            ad_model_classes="",
            ad_model_classes_exclude=False,
            ad_model_classes_excluded="",
        )
    return values


def _head(filled: list[tuple[str, str | None]], count: int, skipped: list[str]) -> str:
    """"Filled the 1st tab for 2 subjects with <detector>." and its variants."""
    subjects = f"{count} subject" + ("" if count == 1 else "s")
    tabs = [f"{tab} with {detector}" if detector else tab for tab, detector in filled]
    if len(filled) == 1:
        tab, detector = filled[0]
        head = f"Filled {tab} for {subjects}" + (f" with {detector}" if detector else "")
    else:
        head = f"Filled {_and(tabs)}, for {subjects}"
    if len(skipped) == 1:
        head += f" (subject {skipped[0]} stays as it is)"
    elif skipped:
        head += f" (subjects {_and(skipped)} stay as they are)"
    return head + "."


def _subject_notes(
    rows: list[tuple], parts: list[str | None], count: int, marks: Sequence[str]
) -> list[str]:
    notes = []
    for i, row in enumerate(rows[:count]):
        text, removed = clean_words(row[2])
        name = lora_name(row[0])
        if removed:
            notes.append(
                f"{_INFO}Subject {i + 1}: a [SEP] or [SKIP] was taken out of "
                "the words, the guide writes them itself."
            )
        if name and name in _tag_names(text):
            notes.append(
                f"{_INFO}Subject {i + 1}: the words already hold {md_plain(name)}, "
                "so the tag typed there is kept and no second one is added."
            )
        for mark in COMMENT_MARKS:
            if mark not in marks or mark not in (parts[i] or ""):
                continue
            shown = _MARK_SHOWN[mark]
            # "/*" hides up to its "*/", the other two the rest of the line:
            # in both cases the [SEP] of the next subjects goes with it.
            dropped = (
                "what follows it up to the next &#42;/"
                if mark == "/*"
                else "what follows it on the line"
            )
            notes.append(
                f"{_WARN}Subject {i + 1} holds a {shown}: with prompt comments "
                f"on, the WebUI drops {dropped}, the other subjects included. "
                f"Take the {shown} out."
            )
    return notes


def _panel_notes(
    prompt: str,
    written: Iterable[int],
    *,
    num_tabs: int,
    enabled_tabs: Sequence[Any],
    old_prompts: Sequence[Any],
    main_prompt: Any,
    files: dict[str, str] | None = None,
) -> list[str]:
    notes = []
    written = list(written)
    # An alias and the file name of one LoRA are that LoRA (_alias_files).
    files = files or {}
    used = {files.get(name, name) for name in _tag_names(prompt)}
    main = main_prompt if isinstance(main_prompt, str) else ""
    shared = [name for name in _tag_names(main) if files.get(name, name) in used]
    if shared:
        one = len(shared) == 1
        notes.append(
            f"{_WARN}{_and([md_plain(name) for name in shared])} "
            f"{'is' if one else 'are'} also in the main "
            "prompt: there a LoRA acts on the whole picture and, with Use LoRAs "
            f"from main prompt on, on every subject. Take {'it' if one else 'them'} "
            "out of the main prompt."
        )

    replaced = [
        _ordinal(index + 1)
        for index in written
        if index < len(old_prompts)
        and isinstance(old_prompts[index], str)
        and old_prompts[index].strip()
        and old_prompts[index].strip() != prompt
    ]
    if replaced:
        notes.append(
            f"{_INFO}The prompt the {_and(replaced)} "
            f"{'tab' if len(replaced) == 1 else 'tabs'} had was replaced."
        )

    on_too = [
        i for i, on in enumerate(enabled_tabs) if on and i not in written and i < num_tabs
    ]
    # A tab the guide filled before (the 2nd, after "person, then face") and
    # does not fill now: still on, with its parts and LoRAs, it would redraw
    # the same subjects again. The guide leaves it alone, and says so.
    left = [
        i for i in on_too
        if i < len(old_prompts)
        and isinstance(old_prompts[i], str)
        and ("[SEP]" in old_prompts[i] or _LORA_TAG_RE.search(old_prompts[i]))
    ]
    for i in left:
        notes.append(
            f"{_WARN}The {_ordinal(i + 1)} tab is still switched on and its "
            "prompt holds [SEP] or a LoRA, maybe from an earlier fill: when it "
            "has a detector it runs too, with that prompt. The guide did not "
            "change it. Switch it off or clear its prompt if you do not want that."
        )
    others = [_ordinal(i + 1) for i in on_too if i not in left]
    if others:
        one = len(others) == 1
        notes.append(
            f"{_INFO}The {_and(others)} {'tab is' if one else 'tabs are'} "
            f"switched on too: {'it runs' if one else 'they run'} as well when "
            f"{'it has' if one else 'they have'} a detector."
        )
    return notes


def _refusal(mode: Any, rows: list[tuple], count: int, num_tabs: int) -> str | None:
    """Why nothing can be written, None when the tabs can be filled."""
    if mode not in GUIDE_MODES:
        return _WARN + "Pick what to redraw first."
    bad = [
        (i + 1, lora_name(row[0]))
        for i, row in enumerate(rows)
        if lora_name(row[0]) and lora_tag(lora_name(row[0]), row[1]) is None
    ]
    if bad:
        return "\n\n".join(
            f'{_WARN}Subject {i}: "{md_plain(name)}" is not a LoRA name. Pick it from the '
            "menu or type the file name alone, without < > or : (for example "
            "my_lora). Nothing was changed."
            for i, name in bad
        )
    if not count:
        return _WARN + "Pick a LoRA or type some words for at least one subject."
    if mode == MODE_PERSON_FACE and num_tabs < len(_PLANS[mode]):
        return (
            f"{_WARN}This choice needs 2 tabs and the panel has 1: raise "
            "Settings, ADetailer, Max tabs to 2 or more and reload the UI, or "
            "pick another choice. Nothing was changed."
        )
    return None


# What each choice fills: per tab, the kind of detector and its candidates.
_PLANS = {
    MODE_PERSON: (("person", PERSON_DETECTORS),),
    MODE_FACE: (("face", FACE_DETECTORS),),
    MODE_PERSON_FACE: (("person", PERSON_DETECTORS), ("face", FACE_DETECTORS)),
}


def build_subject_guide(  # noqa: PLR0913
    mode: Any,
    subjects: Sequence[Any],
    *,
    installed_models: Iterable[str],
    num_tabs: int,
    sort_order: Any,
    more_sort_order: Any = None,
    enabled_tabs: Sequence[Any] = (),
    old_prompts: Sequence[Any] = (),
    main_prompt: Any = None,
    people: Any = None,
    comment_marks: Iterable[str] | None = None,
    lora_names: Iterable[Any] | None = None,
    lora_aliases: Iterable[Any] | None = None,
    reads_lyco: bool = False,
) -> GuideResult:
    """What "Fill the tabs" writes, and what it says.

    `subjects`: up to four (lora, weight, words) rows. `people`: the choice
    "People in the picture" ("Auto", "2", "3", "4"; anything else is Auto):
    the tab keeps that many of the biggest people, the last filled row at
    least, and a row left empty among them is a [SKIP]. `installed_models`:
    the detector names of the menu. `sort_order` / `more_sort_order`: the two
    Settings that order the detections (only read: the status tells when the
    parts will not go from left to right). `enabled_tabs`: "Enable this tab"
    of every tab. `old_prompts`: the prompt each tab holds now (the tabs it
    is not given for are not looked at). `main_prompt`:
    the main prompt box, None when it is not known. `comment_marks`: the
    marks the WebUI takes for a prompt comment (host_comment_marks), None
    when it is not known: then every mark is told. `lora_names`: the LoRA
    names the WebUI lists, `lora_aliases`: the other names a tag can take
    there (a dict when the WebUI tells the LoRA file of each: then an alias
    and the file name of one LoRA count as one). With a list, a name that
    differs from a listed one only by the letter case or by a typed file
    ending is written as listed, and one the
    WebUI does not know is told with its row: its part is still written as
    typed (a file added a moment ago is found at Generate, and a WebUI may
    take names its list does not show), so the warning says what happens
    when the name is wrong. Without a list no name is compared.
    `reads_lyco`: True when the WebUI surely reads a <lyco:...> tag as a
    LoRA tag (AUTOMATIC1111); otherwise a subject whose words hold one is
    told that Forge and Forge Neo skip it. The prompt is the same either way.

    With nothing to do, or a choice it cannot follow, `tabs` is empty and the
    status says why: nothing is written. A setting missing from a tab's
    values is left as the tab has it.
    """
    rows, renamed, unknown, kept = _listed_rows(_rows(subjects), lora_names, lora_aliases)
    # An unusable name is told by _refusal, before any prompt is built.
    try:
        prompt, count = build_prompt(rows, people)
    except InvalidLoraNameError:
        prompt, count = "", 0
    refusal = _refusal(mode, rows, count, num_tabs)
    if refusal is not None:
        return GuideResult(status=refusal)

    installed = {name for name in installed_models if isinstance(name, str)}
    tabs: dict[int, dict[str, Any]] = {}
    filled: list[tuple[str, str | None]] = []
    missing: list[str] = []
    for index, (kind, candidates) in enumerate(_PLANS[mode]):
        detector = next((name for name in candidates if name in installed), None)
        tab = f"the {_ordinal(index + 1)} tab"
        filled.append((tab, detector))
        if detector is None:
            # As Load does with a preset whose detector is not installed.
            names = ", ".join(candidates[:-1]) + " or " + candidates[-1]
            missing.append(
                f"{_WARN}No {kind} detector is installed ({names}): {tab} keeps "
                f"its detector and classes. Pick a detector that finds a {kind} "
                "there."
            )
        tabs[index] = _tab_values(kind, prompt, count, detector)

    parts = _counted(_parts(rows), people)
    skipped = [str(i + 1) for i, part in enumerate(parts) if part is None]
    in_order, order = _order_note(sort_order, more_sort_order, count, _PLANS[mode][0][0])
    # A lyco tag counts as a LoRA tag here, as AUTOMATIC1111 reads it; Forge
    # and Forge Neo skip it, and the subject is redrawn without its LoRA.
    lyco = [] if reads_lyco else [
        i + 1 for i, row in enumerate(rows[:count]) if _LYCO_TAG_RE.search(clean_words(row[2])[0])
    ]
    head = (_WARN if missing or unknown or kept or lyco else _DONE) + _head(filled, count, skipped)
    lines = [f"{head} {order}"] if in_order else [head, order]
    lines.extend(missing)
    lines.extend(
        f"{_WARN}Subject {i}: the tag typed in the words names {md_plain(typed)}, "
        f"but the WebUI lists that LoRA as {md_plain(listed)}: with the name as "
        f"typed that subject is redrawn without a LoRA. Write {md_plain(listed)} "
        "in the tag, or take the tag out of the words."
        for i, typed, listed in kept
    )
    lines.extend(
        f"{_WARN}Subject {i}: the words hold a lyco tag. Forge and Forge Neo do not "
        "read lyco tags, so there that subject is redrawn without the LoRA of that "
        "tag. Write lora instead of lyco in the tag."
        for i in lyco
    )
    lines.extend(
        f"{_WARN}Subject {i}: the WebUI lists no LoRA named {md_plain(name)}. Its part "
        "was written as typed: with a wrong name that subject is redrawn "
        "without a LoRA. Pick the LoRA from the menu; a file added just now is "
        "listed after a refresh of the WebUI's LoRA list."
        for i, name in unknown
    )
    lines.extend(
        f"{_INFO}Subject {i}: {md_plain(typed)} was written as {md_plain(listed)}, "
        "the name in the WebUI's LoRA list."
        for i, typed, listed in renamed
    )
    marks = COMMENT_MARKS if comment_marks is None else tuple(comment_marks)
    lines.extend(_subject_notes(rows, parts, count, marks))
    if mode == MODE_PERSON_FACE:
        # The 2nd tab keeps the biggest FACES, which no setting can tie to
        # the biggest people of the 1st tab: the status says so. Without a
        # face detector to give it, the tab keeps the detector it has: the
        # line then speaks of what that one finds, and of faces only for the
        # day it is a face detector.
        faces = filled[1][1] is not None
        if count == 1:
            lines.append(
                f"{_INFO}The 2nd tab counts on its own: it takes the biggest "
                + (
                    "face. It is usually the face"
                    if faces
                    else "region its detector finds. With a face detector that "
                    "is usually the face"
                )
                + " of the same person, but not when "
                "another person has a bigger face or that face is not found: "
                "check the result."
            )
        else:
            what = "faces" if faces else "regions its detector finds"
            # With another sort order the 2nd tab follows the same setting,
            # but on the faces: by area, the person with the bigger body can
            # have the smaller face, and the two tabs then swap the parts.
            how = (
                ", from left to right."
                if in_order
                else " and sorts them with the same setting as the 1st tab "
                "sorts the people. Their order can differ from the order of "
                f"the people, and a {'face' if faces else 'region'} then gets "
                "another person's part."
            )
            lines.append(
                f"{_INFO}The 2nd tab counts on its own: it takes the {count} "
                f"biggest {what}{how} "
                + ("They" if faces else "With a face detector they")
                + " are usually the faces of the same "
                "people, but not when another person has a bigger face, people "
                "overlap or a face is not found: check the result."
            )
    lines.extend(
        _panel_notes(
            prompt,
            tabs,
            num_tabs=num_tabs,
            enabled_tabs=enabled_tabs,
            old_prompts=old_prompts,
            main_prompt=main_prompt,
            files=_alias_files(lora_aliases),
        )
    )
    # One line, whatever denoise the tabs had: the value the guide writes,
    # and the main prompt that makes a 0.6 enough.
    lines.append(
        f"{_INFO}The guide set Inpaint denoising strength to {GUIDE_DENOISE:g} in "
        f"{_and([tab for tab, _ in filled])}. The subjects are easier to "
        "recognise when the main prompt already describes them (colours, "
        "clothes, who is on the left and who on the right) without their LoRAs."
    )
    return GuideResult(tabs=tabs, status="\n\n".join(lines))
