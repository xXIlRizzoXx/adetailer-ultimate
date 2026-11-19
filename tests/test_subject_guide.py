"""The guide "several subjects, one LoRA each": the pure choices -> values part."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from adetailer.args import ALL_ARGS, BBOX_SORTBY, BBOX_SORTBY_MORE, ADetailerArgs
from adetailer.subject_guide import (
    COMMENT_MARKS,
    FACE_DETECTORS,
    GUIDE_ATTRS,
    GUIDE_DENOISE,
    GUIDE_MODES,
    LEFT_TO_RIGHT,
    MODE_FACE,
    MODE_PERSON,
    MODE_PERSON_FACE,
    PEOPLE_CHOICES,
    PERSON_DETECTORS,
    GuideResult,
    InvalidLoraNameError,
    _ordinal,
    build_prompt,
    build_subject_guide,
    clean_words,
    effective_sort_order,
    format_weight,
    host_comment_marks,
    lora_choices,
    lora_name,
    lora_tag,
    match_lora,
    md_plain,
    people_count,
    subject_part,
)

_ROOT = Path(__file__).resolve().parents[1]

ALICE = ("alice", 0.8, "woman, red hair")
BOB = ("bob", 0.8, "man, grey beard")
CAROL = ("carol", 1, "")
OLD_MAN = ("None", 0.8, "old man")
EMPTY = ("None", 0.8, "")

TWO = "woman, red hair, <lora:alice:0.8> [SEP] man, grey beard, <lora:bob:0.8> [SEP] [SKIP]"
ALL_DETECTORS = [*PERSON_DETECTORS, *FACE_DETECTORS, "hand_yolov8n.pt"]

# The last line of every status that fills tabs (one tab, then two).
DENOISE_LINE = (
    "ℹ️ The guide set Inpaint denoising strength to 0.6 in the 1st tab. The "
    "subjects are easier to recognise when the main prompt already describes "
    "them (colours, clothes, who is on the left and who on the right) without "
    "their LoRAs."
)
DENOISE_LINE_2 = DENOISE_LINE.replace("the 1st tab", "the 1st tab and the 2nd tab")


def guide(mode=MODE_PERSON, subjects=(ALICE, BOB), **kwargs):
    kwargs.setdefault("installed_models", ALL_DETECTORS)
    kwargs.setdefault("num_tabs", 4)
    kwargs.setdefault("sort_order", LEFT_TO_RIGHT)
    return build_subject_guide(mode, list(subjects), **kwargs)


# ---------------------------------------------------------------- the prompt


def test_two_subjects_get_one_part_each_and_a_last_skip():
    assert build_prompt([ALICE, BOB]) == (TWO, 2)
    # The rows left empty after the last filled one do not count.
    assert build_prompt([ALICE, BOB, EMPTY, EMPTY]) == (TWO, 2)


def test_three_subjects_a_lora_without_words_keeps_the_main_prompt():
    assert build_prompt([ALICE, BOB, CAROL]) == (
        "woman, red hair, <lora:alice:0.8> [SEP] man, grey beard, <lora:bob:0.8> "
        "[SEP] [PROMPT], <lora:carol:1> [SEP] [SKIP]",
        3,
    )


def test_four_subjects_words_without_a_lora_are_a_part_too():
    assert build_prompt([ALICE, BOB, CAROL, OLD_MAN]) == (
        "woman, red hair, <lora:alice:0.8> [SEP] man, grey beard, <lora:bob:0.8> "
        "[SEP] [PROMPT], <lora:carol:1> [SEP] old man [SEP] [SKIP]",
        4,
    )


def test_an_empty_row_between_two_subjects_is_a_skip():
    # The subject of that row is left as it is, and the next one keeps its
    # own part: without the [SKIP] carol's LoRA would go to the 2nd subject.
    assert build_prompt([ALICE, EMPTY, CAROL]) == (
        "woman, red hair, <lora:alice:0.8> [SEP] [SKIP] [SEP] [PROMPT], "
        "<lora:carol:1> [SEP] [SKIP]",
        3,
    )


def test_one_subject_alone_and_nothing_at_all():
    assert build_prompt([ALICE]) == ("woman, red hair, <lora:alice:0.8> [SEP] [SKIP]", 1)
    assert build_prompt([]) == ("", 0)
    assert build_prompt([EMPTY, (None, None, None), ("", 0.8, " , ")]) == ("", 0)


def test_rows_beyond_the_fourth_are_not_read():
    assert build_prompt([EMPTY, EMPTY, EMPTY, EMPTY, ALICE]) == ("", 0)


@pytest.mark.parametrize(
    ("row", "part"),
    [
        (("alice", 0.8, "woman"), "woman, <lora:alice:0.8>"),
        # Only a LoRA: the main prompt, then the tag.
        (("alice", 0.8, ""), "[PROMPT], <lora:alice:0.8>"),
        (("alice", 0.8, None), "[PROMPT], <lora:alice:0.8>"),
        # Only words.
        (("None", 0.8, "old man"), "old man"),
        (("", 0.8, "old man"), "old man"),
        ((None, 0.8, "old man"), "old man"),
        (("none", 0.8, "old man"), "old man"),
        # Spaces and commas at the ends go, the rest stays as typed.
        (("alice", 0.8, "  , woman,  red  hair ,, "), "woman,  red  hair, <lora:alice:0.8>"),
        (("alice", 0.8, "woman\nred hair\r\n"), "woman red hair, <lora:alice:0.8>"),
        ((" alice ", 0.8, "woman"), "woman, <lora:alice:0.8>"),
        # [PROMPT] typed by the user is his own choice: nothing is added.
        (("alice", 0.8, "[PROMPT], smiling"), "[PROMPT], smiling, <lora:alice:0.8>"),
        (("alice", 1.25, "(woman:1.2), [CLASS=person]tall[/CLASS]"),
         "(woman:1.2), [CLASS=person]tall[/CLASS], <lora:alice:1.25>"),
    ],
)
def test_one_subject_part(row, part):
    assert subject_part(*row) == part


@pytest.mark.parametrize("row", [EMPTY, (None, None, None), ("None", 1, "  ,\n ")])
def test_an_empty_row_has_no_part(row):
    assert subject_part(*row) is None


@pytest.mark.parametrize(
    ("words", "part"),
    [
        # The same LoRA already typed in the words: its tag and weight stay,
        # no second tag (two tags would apply the LoRA twice).
        ("woman, <lora:alice:1.1>", "woman, <lora:alice:1.1>"),
        ("<lora:alice:1.1>, woman", "<lora:alice:1.1>, woman"),
        ("woman <LORA:alice:0.3>", "woman <LORA:alice:0.3>"),
        ("woman, <lyco:alice:1>", "woman, <lyco:alice:1>"),
        ("woman, <lora:alice>", "woman, <lora:alice>"),
        # Only the tag, no words: the main prompt comes first here too.
        ("<lora:alice:1.1>", "[PROMPT], <lora:alice:1.1>"),
        # Another LoRA, or the same name in another letter case (the WebUI
        # reads names as written): the subject's own tag is added.
        ("woman, <lora:bob:1>", "woman, <lora:bob:1>, <lora:alice:0.8>"),
        ("woman, <lora:Alice:1>", "woman, <lora:Alice:1>, <lora:alice:0.8>"),
        ("woman, <lora:alice2:1>", "woman, <lora:alice2:1>, <lora:alice:0.8>"),
        # The name as plain text is not a tag.
        ("alice, woman", "alice, woman, <lora:alice:0.8>"),
    ],
)
def test_the_tag_is_not_doubled_when_the_words_hold_it(words, part):
    assert subject_part("alice", 0.8, words) == part


@pytest.mark.parametrize(
    ("words", "cleaned"),
    [
        ("woman [SEP] man", "woman man"),
        ("[SEP] woman", "woman"),
        ("woman, [SEP]", "woman"),
        ("woman, [SEP] man [SEP] child", "woman, man child"),
        ("woman [SKIP]", "woman"),
        ("[SKIP]", ""),
        ("[SEP][SKIP][SEP]", ""),
    ],
)
def test_a_typed_sep_or_skip_is_taken_out_of_the_words(words, cleaned):
    # A typed [SEP] would add a part and give every later subject the part
    # of the one before; a typed [SKIP] inside a part reaches the model.
    assert clean_words(words) == (cleaned, True)
    part = subject_part("alice", 0.8, words)
    assert "[SEP]" not in part
    assert "[SKIP]" not in part


def test_the_words_keep_the_other_tokens_and_their_letter_case():
    for words in ("[PROMPT]", "[sep] woman", "[Skip]", "a [SEPARATE] b", "[PROMPT=hair]"):
        assert clean_words(words) == (words, False)
    assert clean_words(None) == ("", False)
    assert clean_words(3) == ("", False)


def test_the_parts_of_the_prompt_split_as_the_detailer_pass_splits_them():
    # scripts/!adetailer.py: re.split(r"\s*\[SEP\]\s*", ad_prompt), and a part
    # that is exactly [SKIP] skips the region.
    source = (_ROOT / "scripts" / "!adetailer.py").read_text(encoding="utf-8")
    assert 're.split(r"\\s*\\[SEP\\]\\s*", ad_prompt)' in source
    assert 're.match(r"^\\s*\\[SKIP\\]\\s*$", prompts[n])' in source
    rows = [("alice", 0.8, "woman [SEP] x"), EMPTY, ("bob", 2, "[SKIP] man"), OLD_MAN]
    prompt, count = build_prompt(rows)
    parts = re.split(r"\s*\[SEP\]\s*", prompt)
    assert count == 4
    assert parts == [
        "woman x, <lora:alice:0.8>", "[SKIP]", "man, <lora:bob:2>", "old man", "[SKIP]",
    ]


# ------------------------------------------------------------ names, weights


@pytest.mark.parametrize(
    "name",
    ["a:b", "<lora:alice:1>", "alice>", "<alice", "a\nb", "x[SEP]y", "[SKIP]", "a[PROMPT]"],
)
def test_a_name_that_cannot_be_inside_a_tag_is_refused(name):
    assert lora_tag(name, 1) is None
    with pytest.raises(InvalidLoraNameError):
        subject_part(name, 1, "woman")
    with pytest.raises(InvalidLoraNameError):
        build_prompt([ALICE, (name, 1, "")])
    result = guide(subjects=[ALICE, (name, 1, "man")])
    # Nothing is written, and the status names the row and the value.
    assert result.tabs == {}
    assert result.status.startswith("⚠️ Subject 2: ")
    # The value as typed, shown as plain text (see md_plain).
    assert f'"{md_plain(name.strip())}" is not a LoRA name' in result.status
    assert "\n" not in result.status
    assert "Nothing was changed." in result.status


def test_every_invalid_name_is_named():
    result = guide(subjects=[("a:b", 1, ""), ALICE, ("c<d", 1, "")])
    assert result.tabs == {}
    assert [line.split(" is not")[0] for line in result.status.split("\n\n")] == [
        '⚠️ Subject 1: "a:b"', '⚠️ Subject 3: "c&lt;d"',
    ]


@pytest.mark.parametrize(
    "name", ["alice", "my lora v2", "style (trigger words)", "char [v2]", "a,b", "名前"]
)
def test_an_ordinary_name_gives_a_tag(name):
    assert lora_tag(name, 0.8) == f"<lora:{name}:0.8>"


def test_no_name_no_tag():
    assert lora_tag("", 1) is None
    assert lora_tag("   ", 1) is None
    assert lora_tag(None, 1) is None


@pytest.mark.parametrize(
    ("weight", "text"),
    [
        (0.8, "0.8"), (1, "1"), (1.0, "1"), (2, "2"), (0, "0"), (0.05, "0.05"),
        (1.25, "1.25"), (0.30000000000000004, "0.3"), (0.7999999999, "0.8"),
        (-0.5, "-0.5"), (-0.001, "0"), ("0.6", "0.6"), (1000000, "1000000"),
        # Not a number: the slider's default, 1 (a live test, 19/11/2026: at
        # 0.8 the characters were less recognisable).
        (None, "1"), ("", "1"), ("abc", "1"), (float("nan"), "1"),
        (float("inf"), "1"), (True, "1"), ([], "1"),
    ],
)
def test_the_weight_is_written_short(weight, text):
    assert format_weight(weight) == text
    assert lora_tag("alice", weight) == f"<lora:alice:{text}>"


def test_the_menu_value_for_no_lora():
    for value in ("None", "none", " NONE ", "", "   ", None, 0, [], ["alice"]):
        assert lora_name(value) == ""
    assert lora_name(" alice ") == "alice"


def test_the_menu_lists_none_first_then_the_usable_names():
    names = ["zeta", "Alpha", "beta", "None", "none", "a:b", " padded", "", None, 3, "beta"]
    assert lora_choices(names) == ["None", "Alpha", "beta", "zeta"]
    assert lora_choices([]) == ["None"]
    assert lora_choices({"b": 1, "a": 2}) == ["None", "a", "b"]


# ------------------------------------------------------- the values per mode


COMMON = {
    "ad_tab_enable": True,
    "ad_model_classes": "",
    "ad_model_classes_exclude": False,
    "ad_model_classes_excluded": "",
    "ad_classes_sequential": False,
    "ad_prompt": TWO,
    "ad_strip_loras": False,
    "ad_mask_k": 2,
    "ad_mask_filter_method": "Area",
    "ad_mask_merge_invert": "None",
    "ad_inpaint_indices": "",
    "ad_inpaint_only_masked": True,
    "ad_denoising_strength": 0.6,
}
PERSON_TAB = {**COMMON, "ad_model": "person_yolov8s-seg.pt", "ad_use_bbox_mask": False}
# "Use bbox as mask" is left as the tab has it: a face detector gives boxes.
FACE_TAB = {**COMMON, "ad_model": "face_yolov8s.pt"}


def test_whole_person_fills_the_first_tab_with_the_outline_detector():
    result = guide(MODE_PERSON)
    assert result.tabs == {0: PERSON_TAB}
    assert result.status == (
        "✅ Filled the 1st tab for 2 subjects with person_yolov8s-seg.pt. "
        "Subject 1 is the leftmost of the 2 biggest people.\n\n" + DENOISE_LINE
    )


def test_face_only_fills_the_first_tab_with_the_face_detector():
    result = guide(MODE_FACE)
    assert result.tabs == {0: FACE_TAB}
    assert "ad_use_bbox_mask" not in result.tabs[0]
    assert result.status == (
        "✅ Filled the 1st tab for 2 subjects with face_yolov8s.pt. "
        "Subject 1 is the leftmost of the 2 biggest faces.\n\n" + DENOISE_LINE
    )


def test_person_then_face_fills_two_tabs_with_the_same_prompt():
    result = guide(MODE_PERSON_FACE)
    assert result.tabs == {0: PERSON_TAB, 1: FACE_TAB}
    assert result.tabs[0] is not result.tabs[1]
    assert result.status.split("\n\n") == [
        "✅ Filled the 1st tab with person_yolov8s-seg.pt and the 2nd tab with "
        "face_yolov8s.pt, for 2 subjects. Subject 1 is the leftmost of the 2 "
        "biggest people.",
        "ℹ️ The 2nd tab counts on its own: it takes the 2 biggest faces, from "
        "left to right. They are usually the faces of the same people, but not "
        "when another person has a bigger face, people overlap or a face is not "
        "found: check the result.",
        DENOISE_LINE_2,
    ]


def test_the_status_says_the_second_tab_counts_the_biggest_faces_on_its_own():
    # Three people, "People in the picture" = 2. The 1st tab keeps the two
    # biggest PEOPLE; the 2nd the two biggest FACES, here those of the 2nd
    # person and of a third one who sits near the camera. The status said
    # "by position too" and named only an overlap or a face not found.
    from PIL import Image

    from adetailer.common import PredictOutput
    from adetailer.mask import SortBy, filter_k_by, sort_bboxes

    def regions(values, boxes):
        pred = PredictOutput(
            bboxes=[list(box) for box in boxes.values()],
            masks=[Image.new("L", (512, 512), 0) for _ in boxes],
            confidences=[0.9] * len(boxes),
            preview=None,
        )
        pred = filter_k_by(pred, k=values["ad_mask_k"], by=values["ad_mask_filter_method"])
        pred = sort_bboxes(pred, SortBy.LEFT_TO_RIGHT)
        parts = re.split(r"\s*\[SEP\]\s*", values["ad_prompt"])
        names = {tuple(box): name for name, box in boxes.items()}
        return {names[tuple(box)]: parts[min(j, len(parts) - 1)] for j, box in enumerate(pred.bboxes)}

    people = {"1": (0, 50, 150, 500), "2": (180, 60, 330, 500), "3": (360, 250, 500, 500)}
    faces = {"1": (50, 60, 90, 110), "2": (230, 70, 280, 130), "3": (400, 260, 470, 350)}
    result = guide(MODE_PERSON_FACE, people="2")
    alice, bob = "woman, red hair, <lora:alice:0.8>", "man, grey beard, <lora:bob:0.8>"
    assert regions(result.tabs[0], people) == {"1": alice, "2": bob}
    assert regions(result.tabs[1], faces) == {"2": alice, "3": bob}
    assert result.status.split("\n\n")[1] == (
        "ℹ️ The 2nd tab counts on its own: it takes the 2 biggest faces, from "
        "left to right. They are usually the faces of the same people, but not "
        "when another person has a bigger face, people overlap or a face is not "
        "found: check the result."
    )
    assert "by position" not in result.status

    # One subject: no position at all, the biggest face whoever it belongs to.
    one = guide(MODE_PERSON_FACE, subjects=[ALICE]).status.split("\n\n")
    assert one[0].endswith("for 1 subject. Subject 1 is the biggest person.")
    assert one[1] == (
        "ℹ️ The 2nd tab counts on its own: it takes the biggest face. It is "
        "usually the face of the same person, but not when another person has a "
        "bigger face or that face is not found: check the result."
    )
    assert "left to right" not in one[1]

    # Another sort order in the Settings: not "from left to right" either.
    other = guide(MODE_PERSON_FACE, sort_order=BBOX_SORTBY[0]).status.split("\n\n")
    assert BBOX_SORTBY[0] != LEFT_TO_RIGHT
    assert other[1].startswith("⚠️ The parts follow Settings, ADetailer, Sort bounding boxes by")
    assert other[2].startswith(
        "ℹ️ The 2nd tab counts on its own: it takes the 2 biggest faces and "
        "sorts them with the same setting as the 1st tab sorts the people."
    )
    for line in (one[1], other[2], result.status.split("\n\n")[1]):
        assert "\n" not in line
        assert not any(mark in line for mark in ("**", "`", "[", "_", "<", "$"))


def test_the_status_does_not_say_the_second_tab_follows_the_order_of_the_first():
    # "Sort bounding boxes by" = Area (large to small). The two people are
    # the same on both tabs, but the one with the bigger body has the smaller
    # face: the 1st tab gives the first part to person 1, the 2nd tab gives
    # it to the face of person 2. The status said "in the same order as the
    # 1st tab".
    from PIL import Image

    from adetailer.common import PredictOutput
    from adetailer.mask import SortBy, filter_k_by, sort_bboxes

    def regions(values, boxes):
        pred = PredictOutput(
            bboxes=[list(box) for box in boxes.values()],
            masks=[Image.new("L", (512, 512), 0) for _ in boxes],
            confidences=[0.9] * len(boxes),
            preview=None,
        )
        pred = filter_k_by(pred, k=values["ad_mask_k"], by=values["ad_mask_filter_method"])
        pred = sort_bboxes(pred, SortBy.AREA)
        parts = re.split(r"\s*\[SEP\]\s*", values["ad_prompt"])
        names = {tuple(box): name for name, box in boxes.items()}
        return {names[tuple(box)]: parts[j] for j, box in enumerate(pred.bboxes)}

    people = {"1": (0, 20, 200, 500), "2": (260, 150, 420, 500)}
    faces = {"1": (70, 30, 110, 80), "2": (290, 160, 390, 290)}
    area = "Area (large to small)"
    assert area in BBOX_SORTBY
    result = guide(MODE_PERSON_FACE, sort_order=area)
    alice, bob = "woman, red hair, <lora:alice:0.8>", "man, grey beard, <lora:bob:0.8>"
    assert regions(result.tabs[0], people) == {"1": alice, "2": bob}
    assert regions(result.tabs[1], faces) == {"2": alice, "1": bob}  # swapped

    line = (
        "ℹ️ The 2nd tab counts on its own: it takes the 2 biggest faces and sorts "
        "them with the same setting as the 1st tab sorts the people. Their order "
        "can differ from the order of the people, and a face then gets another "
        "person's part. They are usually the faces of the same people, but not "
        "when another person has a bigger face, people overlap or a face is not "
        "found: check the result."
    )
    assert result.status.split("\n\n")[2] == line
    assert "same order" not in result.status
    # The same with every other order of the two Settings.
    for order in BBOX_SORTBY:
        status = guide(MODE_PERSON_FACE, sort_order=order).status
        assert (line in status) is (order != LEFT_TO_RIGHT), order
        assert "same order" not in status
    for more in BBOX_SORTBY_MORE[1:]:
        status = guide(MODE_PERSON_FACE, sort_order=LEFT_TO_RIGHT, more_sort_order=more).status
        assert line in status, more
    # One block of plain text, as every text of the panel that is translated.
    assert not any(mark in line for mark in ("**", "`", "[", "_", "<", "$", "\n"))


def test_the_second_tab_line_promises_no_faces_without_a_face_detector():
    # No face detector installed: the 2nd tab keeps the detector it has (a
    # hand one, or none at all), and the line still said what faces it takes.
    person_only = list(PERSON_DETECTORS)
    lines = guide(MODE_PERSON_FACE, installed_models=person_only).status.split("\n\n")
    assert lines[1] == (
        "⚠️ No face detector is installed (face_yolov8s.pt, face_yolov8n.pt or "
        "mediapipe_face_full): the 2nd tab keeps its detector and classes. Pick "
        "a detector that finds a face there."
    )
    assert lines[2] == (
        "ℹ️ The 2nd tab counts on its own: it takes the 2 biggest regions its "
        "detector finds, from left to right. With a face detector they are "
        "usually the faces of the same people, but not when another person has "
        "a bigger face, people overlap or a face is not found: check the result."
    )
    assert "biggest faces" not in "\n".join(lines)

    lines = guide(
        MODE_PERSON_FACE, installed_models=person_only, sort_order=BBOX_SORTBY[0]
    ).status.split("\n\n")
    assert lines[3] == (
        "ℹ️ The 2nd tab counts on its own: it takes the 2 biggest regions its "
        "detector finds and sorts them with the same setting as the 1st tab "
        "sorts the people. Their order can differ from the order of the people, "
        "and a region then gets another person's part. With a face detector "
        "they are usually the faces of the same people, but not when another "
        "person has a bigger face, people overlap or a face is not found: check "
        "the result."
    )

    lines = guide(
        MODE_PERSON_FACE, subjects=[ALICE], installed_models=person_only
    ).status.split("\n\n")
    assert lines[2] == (
        "ℹ️ The 2nd tab counts on its own: it takes the biggest region its "
        "detector finds. With a face detector that is usually the face of the "
        "same person, but not when another person has a bigger face or that "
        "face is not found: check the result."
    )
    assert "biggest face." not in "\n".join(lines)
    # With the person detector missing instead, the 2nd tab does take faces.
    lines = guide(MODE_PERSON_FACE, installed_models=list(FACE_DETECTORS)).status
    assert "it takes the 2 biggest faces, from left to right. They are usually" in lines


@pytest.mark.parametrize("mode", GUIDE_MODES)
@pytest.mark.parametrize(
    ("subjects", "count"),
    [([ALICE], 1), ([ALICE, BOB], 2), ([ALICE, BOB, CAROL], 3),
     ([ALICE, BOB, CAROL, OLD_MAN], 4), ([EMPTY, EMPTY, EMPTY, OLD_MAN], 4)],
)
def test_top_k_is_the_number_of_the_last_filled_row(mode, subjects, count):
    result = guide(mode, subjects)
    prompt, _ = build_prompt(subjects)
    for values in result.tabs.values():
        assert values["ad_mask_k"] == count
        assert values["ad_mask_filter_method"] == "Area"
        assert values["ad_prompt"] == prompt
    assert f"for {count} subject{'' if count == 1 else 's'}" in result.status


def test_the_status_names_the_rows_left_empty_in_between():
    assert "for 3 subjects with person_yolov8s-seg.pt (subject 2 stays as it is)." in (
        guide(subjects=[ALICE, EMPTY, CAROL]).status
    )
    assert "for 4 subjects (subjects 1, 2 and 3 stay as they are)." in (
        guide(subjects=[EMPTY, EMPTY, EMPTY, OLD_MAN], installed_models=[]).status
    )


# ------------------------------------------------- "People in the picture"


def test_people_in_the_picture_reads_its_four_choices_and_nothing_else():
    assert PEOPLE_CHOICES == ("Auto", "2", "3", "4")
    assert [people_count(choice) for choice in PEOPLE_CHOICES] == [0, 2, 3, 4]
    # Nothing picked, a translated text, a number, a choice that is not one.
    for value in (None, "", "auto", "Automatico", "Auto ", " 2", "1", "5", "2.0",
                  2, 3, 4, 2.0, True, ["2"], ("3",), {"4"}):
        assert people_count(value) == 0, value


ONE_OF_TWO = "woman, red hair, <lora:alice:0.8> [SEP] [SKIP] [SEP] [SKIP]"


def test_two_people_and_one_row_filled_keep_the_two_biggest_and_skip_the_other():
    # With "Auto" the tab kept the ONE biggest person: with two people and
    # the 1st row filled, the LoRA went to the bigger one, not to the one on
    # the left, and the status still said "Subject 1 is the leftmost".
    assert build_prompt([ALICE], "2") == (ONE_OF_TWO, 2)
    assert build_prompt([ALICE, EMPTY, EMPTY, EMPTY], "2") == (ONE_OF_TWO, 2)
    result = guide(subjects=[ALICE, EMPTY, EMPTY, EMPTY], people="2")
    assert result.tabs == {0: {**PERSON_TAB, "ad_prompt": ONE_OF_TWO, "ad_mask_k": 2}}
    assert result.status == (
        "✅ Filled the 1st tab for 2 subjects with person_yolov8s-seg.pt (subject "
        "2 stays as it is). Subject 1 is the leftmost of the 2 biggest people.\n\n"
        + DENOISE_LINE
    )


def test_one_row_and_auto_says_the_biggest_person_not_the_leftmost():
    for people in ("Auto", None, "Automatico", 2):
        result = guide(subjects=[ALICE, EMPTY, EMPTY, EMPTY], people=people)
        assert result.tabs[0]["ad_mask_k"] == 1
        assert result.tabs[0]["ad_prompt"] == "woman, red hair, <lora:alice:0.8> [SEP] [SKIP]"
        assert result.status == (
            "✅ Filled the 1st tab for 1 subject with person_yolov8s-seg.pt. "
            "Subject 1 is the biggest person.\n\n" + DENOISE_LINE
        )
        assert "leftmost" not in result.status
    face = guide(MODE_FACE, subjects=[ALICE], people="Auto").status
    assert face.split("\n\n")[0].endswith("Subject 1 is the biggest face.")


@pytest.mark.parametrize("mode", GUIDE_MODES)
@pytest.mark.parametrize(
    ("subjects", "people", "count", "skipped"),
    [
        # The higher of the last filled row and the number picked.
        ([ALICE], "2", 2, " (subject 2 stays as it is)"),
        ([ALICE], "3", 3, " (subjects 2 and 3 stay as they are)"),
        ([ALICE], "4", 4, " (subjects 2, 3 and 4 stay as they are)"),
        ([EMPTY, BOB], "4", 4, " (subjects 1, 3 and 4 stay as they are)"),
        ([ALICE, BOB], "2", 2, ""),
        ([ALICE, BOB, CAROL], "2", 3, ""),
        ([ALICE, BOB, CAROL, OLD_MAN], "3", 4, ""),
        ([ALICE, EMPTY, CAROL], "Auto", 3, " (subject 2 stays as it is)"),
        ([ALICE, BOB, EMPTY, EMPTY], "Auto", 2, ""),
    ],
)
def test_the_people_counted_are_the_most_of_the_rows_and_the_choice(
    mode, subjects, people, count, skipped
):
    result = guide(mode, subjects, people=people)
    prompt, counted = build_prompt(subjects, people)
    assert counted == count
    parts = prompt.split(" [SEP] ")
    # One part per person counted, and the last [SKIP] for anyone beyond.
    assert len(parts) == count + 1
    assert parts[-1] == "[SKIP]"
    for values in result.tabs.values():
        assert values["ad_mask_k"] == count
        assert values["ad_mask_filter_method"] == "Area"
        assert values["ad_prompt"] == prompt
        assert ADetailerArgs(**values).ad_mask_k == count
    head = result.status.split("\n\n")[0]
    assert f"for {count} subjects" in head
    if skipped:
        assert f"{skipped}. Subject 1 is" in head
    else:
        assert "stay" not in head
    many = "faces" if mode == MODE_FACE else "people"
    assert head.endswith(f"Subject 1 is the leftmost of the {count} biggest {many}.")


def test_people_in_the_picture_alone_fills_nothing():
    # A number without any row filled is not a subject.
    for people in PEOPLE_CHOICES:
        assert build_prompt([EMPTY] * 4, people) == ("", 0)
        result = guide(subjects=[EMPTY] * 4, people=people)
        assert result.tabs == {}
        assert result.status == "⚠️ Pick a LoRA or type some words for at least one subject."


def test_the_two_biggest_people_get_their_parts_from_left_to_right():
    # The picture of the finding: the person on the left is the smaller one.
    # With "2" the left one gets the LoRA and the right one is skipped; with
    # "Auto" and one row the only region kept is the bigger person's.
    from PIL import Image

    from adetailer.common import PredictOutput
    from adetailer.mask import SortBy, filter_k_by, sort_bboxes

    # As pred_preprocessing does: the biggest first, then the order.
    source = (_ROOT / "scripts" / "!adetailer.py").read_text(encoding="utf-8")
    assert source.index("pred = filter_k_by(pred, k=args.ad_mask_k") < source.index(
        "pred = self.sort_bboxes(pred)"
    )

    left_small, right_big = [20, 150, 180, 500], [260, 60, 500, 500]

    def regions(values):
        boxes = [right_big, left_small]
        pred = PredictOutput(
            bboxes=boxes,
            masks=[Image.new("L", (512, 512), 0) for _ in boxes],
            confidences=[0.9, 0.9],
            preview=None,
        )
        pred = filter_k_by(pred, k=values["ad_mask_k"], by=values["ad_mask_filter_method"])
        pred = sort_bboxes(pred, SortBy.LEFT_TO_RIGHT)
        parts = re.split(r"\s*\[SEP\]\s*", values["ad_prompt"])
        return [(box[0], parts[min(i, len(parts) - 1)]) for i, box in enumerate(pred.bboxes)]

    auto = guide(subjects=[ALICE, EMPTY], people="Auto").tabs[0]
    assert regions(auto) == [(260, "woman, red hair, <lora:alice:0.8>")]
    two = guide(subjects=[ALICE, EMPTY], people="2").tabs[0]
    assert regions(two) == [(20, "woman, red hair, <lora:alice:0.8>"), (260, "[SKIP]")]


@pytest.mark.parametrize("mode", GUIDE_MODES)
@pytest.mark.parametrize(
    "subjects", [[ALICE, BOB], [ALICE, EMPTY, CAROL, OLD_MAN], [("x y (z)", 2, "a [b] c")]]
)
def test_every_value_written_is_a_tab_setting_the_schema_accepts(mode, subjects):
    result = guide(mode, subjects)
    assert result.tabs
    for values in result.tabs.values():
        assert set(values) <= set(ALL_ARGS.attrs)
        assert set(values) <= set(GUIDE_ATTRS)
        args = ADetailerArgs(**values)
        for attr, value in values.items():
            assert getattr(args, attr) == value
            assert type(getattr(args, attr)) is type(value)
        assert not args.need_skip()


def test_the_outputs_of_the_guide_are_tab_settings_listed_once():
    assert set(GUIDE_ATTRS) <= set(ALL_ARGS.attrs)
    assert len(set(GUIDE_ATTRS)) == len(GUIDE_ATTRS)
    # Every setting a mode writes has an output, and every output is written
    # by a mode: one that is not would be a stale no-op.
    written = set()
    for mode in GUIDE_MODES:
        for values in guide(mode).tabs.values():
            written |= set(values)
    assert written == set(GUIDE_ATTRS)


def test_what_the_guide_leaves_alone():
    # The negative prompt, the append boxes, "Use LoRAs from main prompt",
    # sizes, padding, confidence, checkpoint, sampler, ControlNet.
    for attr in (
        "ad_negative_prompt", "ad_prompt_append", "ad_negative_prompt_append",
        "ad_use_main_loras", "ad_use_lora_triggers", "ad_class_prompts",
        "ad_confidence", "ad_inpaint_only_masked_padding",
        "ad_inpaint_width", "ad_inpaint_height", "ad_use_checkpoint", "ad_checkpoint",
        "ad_use_sampler", "ad_sampler", "ad_controlnet_model", "ad_mask_min_ratio",
        "ad_mask_max_ratio", "ad_dilate_erode", "ad_inpaint_masked_content",
    ):
        assert attr in ALL_ARGS.attrs
        assert attr not in GUIDE_ATTRS


@pytest.mark.parametrize("mode", GUIDE_MODES)
def test_every_tab_the_guide_fills_gets_a_denoise_of_0_6(mode):
    # A live test (19/11/2026, two characters, two LoRAs): with the tab's 0.4
    # the characters were not recognisable, at 0.75 the bodies broke. Every
    # tab filled, the 2nd of "person, then face" too, whatever it had.
    assert GUIDE_DENOISE == 0.6
    result = guide(mode, old_prompts=["a portrait", "a face"])
    assert set(result.tabs) == ({0, 1} if mode == MODE_PERSON_FACE else {0})
    for values in result.tabs.values():
        assert values["ad_denoising_strength"] == 0.6
        assert type(values["ad_denoising_strength"]) is float
    # One line says it, the last one, whatever the tabs had: no line per value.
    lines = result.status.split("\n\n")
    assert lines[-1] == (DENOISE_LINE_2 if mode == MODE_PERSON_FACE else DENOISE_LINE)
    assert sum("denoising" in line for line in lines) == 1
    # Nothing written, nothing said about it.
    for refused in (guide(mode, subjects=[]), guide("not a mode")):
        assert not refused.tabs
        assert "denoising" not in refused.status


def test_the_prompt_keeps_its_tags_through_the_detailer_pass_settings():
    # "Strip LoRAs from the detailer prompt" would take the tags out again.
    for mode in GUIDE_MODES:
        for values in guide(mode).tabs.values():
            assert values["ad_strip_loras"] is False
            assert values["ad_inpaint_indices"] == ""
            assert values["ad_mask_merge_invert"] == "None"
            assert values["ad_classes_sequential"] is False


# -------------------------------------------------------------- the detectors


@pytest.mark.parametrize(
    ("installed", "person", "face"),
    [
        (ALL_DETECTORS, "person_yolov8s-seg.pt", "face_yolov8s.pt"),
        (["person_yolov8n-seg.pt", "face_yolov8n.pt", "mediapipe_face_full"],
         "person_yolov8n-seg.pt", "face_yolov8n.pt"),
        (["mediapipe_face_full", "mediapipe_face_short"], None, "mediapipe_face_full"),
        (["hand_yolov8n.pt", "mediapipe_face_mesh"], None, None),
        ([], None, None),
        # A dict, as model_mapping is.
        ({"face_yolov8s.pt": "x", "person_yolov8s-seg.pt": "y"},
         "person_yolov8s-seg.pt", "face_yolov8s.pt"),
    ],
)
def test_the_first_installed_detector_of_each_kind_is_used(installed, person, face):
    result = guide(MODE_PERSON_FACE, installed_models=installed)
    assert set(result.tabs) == {0, 1}
    for index, kind, detector, names in (
        (0, "person", person, "person_yolov8s-seg.pt or person_yolov8n-seg.pt"),
        (1, "face", face, "face_yolov8s.pt, face_yolov8n.pt or mediapipe_face_full"),
    ):
        values = result.tabs[index]
        tab = "1st" if index == 0 else "2nd"
        note = (
            f"⚠️ No {kind} detector is installed ({names}): the {tab} tab "
            f"keeps its detector and classes. Pick a detector that finds a {kind} there."
        )
        if detector is None:
            # As Load does: the tab keeps its detector and its class filter,
            # and the status says so. The rest is still filled.
            assert "ad_model" not in values
            assert not any(attr.startswith("ad_model_classes") for attr in values)
            assert note in result.status.split("\n\n")
            assert f"the {tab} tab with " not in result.status
        else:
            assert values["ad_model"] == detector
            assert values["ad_model_classes"] == ""
            assert values["ad_model_classes_exclude"] is False
            assert values["ad_model_classes_excluded"] == ""
            assert note not in result.status
            assert f"the {tab} tab with {detector}" in result.status
        assert values["ad_prompt"] == TWO
        assert values["ad_tab_enable"] is True
    ok = person is not None and face is not None
    assert result.status.startswith("✅ " if ok else "⚠️ Filled ")


def test_a_missing_detector_in_a_one_tab_mode():
    result = guide(MODE_PERSON, installed_models=["face_yolov8s.pt"])
    assert "ad_model" not in result.tabs[0]
    assert result.tabs[0]["ad_use_bbox_mask"] is False
    assert result.status.split("\n\n")[0] == (
        "⚠️ Filled the 1st tab for 2 subjects. Subject 1 is the leftmost of the "
        "2 biggest people."
    )
    result = guide(MODE_FACE, installed_models=["person_yolov8s-seg.pt"])
    assert "ad_model" not in result.tabs[0]
    assert "No face detector is installed" in result.status


# ---------------------------------------------------------------- sort order


def test_left_to_right_needs_no_warning():
    for more in ("Off", None, "", "not a choice", 3):
        result = guide(sort_order=LEFT_TO_RIGHT, more_sort_order=more)
        assert result.status.split("\n\n")[0].endswith("Subject 1 is the leftmost of the 2 biggest people.")
        assert "⚠️" not in result.status


@pytest.mark.parametrize("value", [v for v in BBOX_SORTBY if v != LEFT_TO_RIGHT])
def test_another_sort_order_is_named_with_the_way_to_change_it(value):
    result = guide(sort_order=value, more_sort_order="Off")
    # The guide still fills the tab: it never changes the setting itself.
    assert result.tabs == {0: PERSON_TAB}
    assert result.status.split("\n\n") == [
        "✅ Filled the 1st tab for 2 subjects with person_yolov8s-seg.pt.",
        "⚠️ The parts follow Settings, ADetailer, Sort bounding boxes by, "
        f"now {value}: set it to Position (left to right) and click Apply settings.",
        DENOISE_LINE,
    ]
    assert "leftmost" not in result.status


@pytest.mark.parametrize("more", BBOX_SORTBY_MORE[1:])
@pytest.mark.parametrize("base", BBOX_SORTBY)
def test_more_sort_orders_wins_over_the_first_setting(more, base):
    # As sort_bboxes in scripts/!adetailer.py: any choice but "Off" is used
    # instead, also when the first setting says left to right.
    result = guide(sort_order=base, more_sort_order=more)
    assert result.tabs == {0: PERSON_TAB}
    assert result.status.split("\n\n")[1] == (
        f"⚠️ The parts follow Settings, ADetailer, More sort orders, now {more}: "
        "set it to Off, set Sort bounding boxes by to Position (left to right) and "
        "click Apply settings."
    )
    assert "leftmost" not in result.status


@pytest.mark.parametrize("value", [None, "", 3, "Position (diagonal)", ["None"]])
def test_a_sort_order_that_cannot_be_read_is_not_taken_for_left_to_right(value):
    result = guide(sort_order=value)
    assert result.tabs == {0: PERSON_TAB}
    assert "Sort bounding boxes by, now not readable: set it to" in result.status
    assert "leftmost" not in result.status


def test_the_sort_order_is_read_as_the_detailer_pass_reads_it():
    assert effective_sort_order("None", "Off") == ("Sort bounding boxes by", "None")
    assert effective_sort_order(LEFT_TO_RIGHT) == ("Sort bounding boxes by", LEFT_TO_RIGHT)
    assert effective_sort_order(LEFT_TO_RIGHT, "Area (small to large)") == (
        "More sort orders", "Area (small to large)",
    )
    assert LEFT_TO_RIGHT in BBOX_SORTBY
    assert BBOX_SORTBY_MORE[0] == "Off"
    # The same rule in the script: "Off" or an unknown value keeps the first.
    source = (_ROOT / "scripts" / "!adetailer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    sort = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "sort_bboxes"
        and any(arg.arg == "self" for arg in node.args.args)
    )
    assert "isinstance(more, str) and more in BBOX_SORTBY_MORE[1:]" in ast.unparse(sort)


# ------------------------------------------------------------- what stops it


def test_nothing_chosen_writes_nothing():
    for subjects in ([], [EMPTY] * 4, [("None", 1, " ,, ")], [("None", 1, "[SKIP]")]):
        result = guide(subjects=subjects)
        assert result == GuideResult(
            tabs={},
            status="⚠️ Pick a LoRA or type some words for at least one subject.",
        )


@pytest.mark.parametrize("mode", [None, "", "Whole person", 0, ["Face only"]])
def test_an_unknown_choice_of_what_to_redraw_writes_nothing(mode):
    result = guide(mode)
    assert result.tabs == {}
    assert result.status == "⚠️ Pick what to redraw first."


@pytest.mark.parametrize("num_tabs", [1, 0])
def test_two_tabs_with_a_single_tab_in_the_panel_writes_nothing(num_tabs):
    result = guide(MODE_PERSON_FACE, num_tabs=num_tabs)
    assert result.tabs == {}
    assert result.status == (
        "⚠️ This choice needs 2 tabs and the panel has 1: raise Settings, "
        "ADetailer, Max tabs to 2 or more and reload the UI, or pick another "
        "choice. Nothing was changed."
    )


def test_one_tab_is_enough_for_the_one_tab_modes():
    assert guide(MODE_PERSON, num_tabs=1).tabs == {0: PERSON_TAB}
    assert guide(MODE_FACE, num_tabs=1).tabs == {0: FACE_TAB}
    assert set(guide(MODE_PERSON_FACE, num_tabs=2).tabs) == {0, 1}


# ------------------------------------------------------------------ the notes


def test_a_previous_prompt_that_is_replaced_is_told():
    told = "ℹ️ The prompt the 1st tab had was replaced."
    assert told in guide(old_prompts=["a face", "hands"]).status.split("\n\n")
    # Not for an empty box, one of spaces, or the very prompt the guide writes.
    for old in ([], [""], ["  \n"], [TWO], [f" {TWO}\n"], [None], [3]):
        assert "was replaced" not in guide(old_prompts=old).status
    # Only the tabs the guide writes count.
    assert "was replaced" not in guide(old_prompts=["", "hands"]).status
    both = guide(MODE_PERSON_FACE, old_prompts=["a face", "hands", "feet"]).status
    assert "ℹ️ The prompt the 1st and 2nd tabs had was replaced." in both
    assert "3rd" not in both
    second = guide(MODE_PERSON_FACE, old_prompts=["", "hands"]).status
    assert "ℹ️ The prompt the 2nd tab had was replaced." in second


def test_a_lora_that_is_also_in_the_main_prompt_is_told():
    one = guide(main_prompt="a photo, <lora:bob:0.6>, <lora:other:1>").status
    assert (
        "⚠️ bob is also in the main prompt: there a LoRA acts on the whole "
        "picture and, with Use LoRAs from main prompt on, on every subject. Take "
        "it out of the main prompt."
    ) in one.split("\n\n")
    two = guide(main_prompt="<lora:bob:1> <LYCO:alice:1> <lora:bob:1>").status
    assert "⚠️ bob and alice are also in the main prompt" in two
    assert "Take them out of the main prompt." in two
    # A tag typed in the words counts, since it is in the tab's prompt.
    typed = guide(subjects=[("None", 1, "woman <lora:carol:1>")], main_prompt="<lora:carol:1>")
    assert "⚠️ carol is also in the main prompt" in typed.status
    # Another LoRA, another letter case, the bare name, or no main prompt.
    for main in ("<lora:other:1>", "<lora:Bob:1>", "bob, alice", "", None, 3, ["<lora:bob:1>"]):
        assert "main prompt" not in guide(main_prompt=main).status.replace(DENOISE_LINE, "")
    # The label the status names is the checkbox's own.
    ui = (_ROOT / "aaaaaa" / "ui.py").read_text(encoding="utf-8")
    assert 'label="Use LoRAs from main prompt" + suffix(n)' in ui


def test_the_other_tabs_that_are_switched_on_are_told():
    one = guide(enabled_tabs=[True, True, False, False]).status
    assert (
        "ℹ️ The 2nd tab is switched on too: it runs as well when it has a "
        "detector."
    ) in one.split("\n\n")
    more = guide(enabled_tabs=[False, True, False, True]).status
    assert (
        "ℹ️ The 2nd and 4th tabs are switched on too: they run as well when "
        "they have a detector."
    ) in more
    # The tabs the guide fills are not "other" tabs.
    assert "switched on too" not in guide(enabled_tabs=[True, False, False, False]).status
    both = guide(MODE_PERSON_FACE, enabled_tabs=[True, True, True, False]).status
    assert "ℹ️ The 3rd tab is switched on too" in both
    assert "The 2nd tab is switched on too" not in both
    assert "switched on too" not in guide(enabled_tabs=[]).status
    assert "switched on too" not in guide(enabled_tabs=[None, 0, "", False]).status
    # A value beyond the panel's tabs is not a tab.
    assert "switched on too" not in guide(num_tabs=1, enabled_tabs=[True, True]).status


def _left_on(tab="2nd"):
    return (
        f"⚠️ The {tab} tab is still switched on and its prompt holds [SEP] or a "
        "LoRA, maybe from an earlier fill: when it has a detector it runs too, "
        "with that prompt. The guide did not change it. Switch it off or clear "
        "its prompt if you do not want that."
    )


@pytest.mark.parametrize("mode", [MODE_PERSON, MODE_FACE])
@pytest.mark.parametrize(
    "second",
    [TWO, "woman [SEP] man", "a face, <lora:alice:0.8>", "<LYCO:bob:1>", "[SEP]"],
)
def test_a_second_tab_left_on_with_the_parts_of_an_earlier_fill_is_told(mode, second):
    # After "Whole person, then face (2 tabs)", a choice with one tab leaves
    # the 2nd tab on with the prompt and the LoRAs of before: it ran again
    # on the same subjects, and the status only said "switched on too".
    result = guide(mode, enabled_tabs=[True, True, False], old_prompts=["", second])
    lines = result.status.split("\n\n")
    assert _left_on() in lines
    assert "switched on too" not in result.status
    # The guide still writes the 1st tab only.
    assert set(result.tabs) == {0}


def test_a_tab_left_on_is_told_only_when_it_is_on_and_not_filled_now():
    told = lambda *a, **kw: "still switched on" in guide(*a, **kw).status  # noqa: E731
    assert told(enabled_tabs=[True, True], old_prompts=["", TWO])
    # Off, or filled by this very choice: nothing to tell.
    assert not told(enabled_tabs=[True, False], old_prompts=["", TWO])
    assert not told(MODE_PERSON_FACE, enabled_tabs=[True, True], old_prompts=[TWO, TWO])
    # The 1st tab is always filled: its old prompt is "replaced", not "left".
    assert not told(enabled_tabs=[True, False], old_prompts=[TWO, ""])
    # An ordinary prompt, an empty one, one that is not a text, a lowercase
    # [sep] (the pass splits on [SEP] only): the plain note.
    for second in ("a detailed face", "", None, 3, "woman [sep] man", "lora:alice"):
        status = guide(enabled_tabs=[True, True], old_prompts=["", second]).status
        assert "still switched on" not in status
        assert "ℹ️ The 2nd tab is switched on too" in status
    # A tab whose prompt was not given (the panel reads the first two only).
    status = guide(enabled_tabs=[True, True, True], old_prompts=["", TWO]).status
    assert _left_on() in status.split("\n\n")
    assert "ℹ️ The 3rd tab is switched on too: it runs as well" in status
    assert "The 2nd tab is switched on too" not in status
    # Beyond the panel's tabs there is no tab.
    assert not told(num_tabs=1, enabled_tabs=[True, True], old_prompts=["", TWO])


# The interface always sends four rows: the ones left alone hold the menu's
# "None", the slider's weight and an empty box (or what a cleared field gives).
_UNTOUCHED = [("None", 0.8, ""), ("None", 0.8, None), ("", None, " , "), (None, None, None)]


@pytest.mark.parametrize("empty", _UNTOUCHED)
@pytest.mark.parametrize(
    ("mode", "filled", "many"),
    [
        (MODE_PERSON, "the 1st tab for 2 subjects with person_yolov8s-seg.pt", "people"),
        (MODE_FACE, "the 1st tab for 2 subjects with face_yolov8s.pt", "faces"),
        (MODE_PERSON_FACE,
         "the 1st tab with person_yolov8s-seg.pt and the 2nd tab with face_yolov8s.pt, "
         "for 2 subjects", "people"),
    ],
)
def test_the_status_with_four_rows_whose_last_ones_are_empty(empty, mode, filled, many):
    result = guide(mode, [ALICE, BOB, empty, empty])
    assert result.status.split("\n\n")[0] == (
        f"✅ Filled {filled}. Subject 1 is the leftmost of the 2 biggest {many}."
    )
    # The empty rows after the last filled one are not subjects: no word
    # about them, no part for them, and the same answer as with two rows.
    assert "stay" not in result.status
    assert result == guide(mode, [ALICE, BOB])
    for values in result.tabs.values():
        assert values["ad_prompt"] == TWO
        assert values["ad_mask_k"] == 2


@pytest.mark.parametrize("empty", _UNTOUCHED)
def test_the_status_with_four_rows_names_only_the_empty_ones_in_between(empty):
    one = guide(subjects=[ALICE, empty, empty, empty])
    assert one.status == (
        "✅ Filled the 1st tab for 1 subject with person_yolov8s-seg.pt. "
        "Subject 1 is the biggest person.\n\n" + DENOISE_LINE
    )
    second = guide(subjects=[empty, BOB, empty, empty])
    assert second.status == (
        "✅ Filled the 1st tab for 2 subjects with person_yolov8s-seg.pt (subject "
        "1 stays as it is). Subject 1 is the leftmost of the 2 biggest people.\n\n"
        + DENOISE_LINE
    )
    assert second.tabs[0]["ad_prompt"] == "[SKIP] [SEP] man, grey beard, <lora:bob:0.8> [SEP] [SKIP]"
    third = guide(subjects=[ALICE, empty, CAROL, empty])
    assert third.status == (
        "✅ Filled the 1st tab for 3 subjects with person_yolov8s-seg.pt (subject "
        "2 stays as it is). Subject 1 is the leftmost of the 3 biggest people.\n\n"
        + DENOISE_LINE
    )
    last = guide(subjects=[empty, empty, empty, OLD_MAN])
    assert last.status == (
        "✅ Filled the 1st tab for 4 subjects with person_yolov8s-seg.pt (subjects "
        "1, 2 and 3 stay as they are). Subject 1 is the leftmost of the 4 biggest "
        "people.\n\n" + DENOISE_LINE
    )
    nothing = guide(subjects=[empty] * 4)
    assert nothing.tabs == {}
    assert nothing.status == "⚠️ Pick a LoRA or type some words for at least one subject."


def test_the_ordinals_of_the_tabs_up_to_the_last_one():
    # "Max tabs" goes up to 15: 11th, 12th and 13th are not 11st, 12nd, 13rd.
    assert [_ordinal(n) for n in range(1, 16)] == [
        "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th",
        "11th", "12th", "13th", "14th", "15th",
    ]
    assert [_ordinal(n) for n in (21, 22, 23, 24, 101, 111, 112, 113, 121)] == [
        "21st", "22nd", "23rd", "24th", "101st", "111th", "112th", "113th", "121st",
    ]
    # The same words as the tabs of the panel carry (aaaaaa/ui.py, ordinal).
    source = (_ROOT / "aaaaaa" / "ui.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    panel = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "ordinal"
    )
    namespace: dict = {}
    exec(compile(ast.Module(body=[panel], type_ignores=[]), "ui.py", "exec"), namespace)
    assert [_ordinal(n) for n in range(1, 16)] == [namespace["ordinal"](n) for n in range(1, 16)]


def test_the_status_names_the_tabs_beyond_the_tenth():
    enabled = [True] + [False] * 9 + [True, True, True, False, False]
    status = guide(num_tabs=15, enabled_tabs=enabled).status
    assert (
        "ℹ️ The 11th, 12th and 13th tabs are switched on too: they run as well "
        "when they have a detector."
    ) in status.split("\n\n")
    one = guide(num_tabs=15, enabled_tabs=[False] * 11 + [True]).status
    assert "ℹ️ The 12th tab is switched on too: it runs as well when it has a detector." in one
    for wrong in ("11st", "12nd", "13rd"):
        assert wrong not in status + one


def test_a_removed_token_and_a_kept_tag_are_told_with_their_row():
    result = guide(subjects=[ALICE, ("bob", 0.8, "man [SEP] <lora:bob:1.2>")])
    lines = result.status.split("\n\n")
    assert (
        "ℹ️ Subject 2: a [SEP] or [SKIP] was taken out of the words, the "
        "guide writes them itself."
    ) in lines
    assert (
        "ℹ️ Subject 2: the words already hold bob, so the tag typed there is "
        "kept and no second one is added."
    ) in lines
    assert not any(line.startswith("ℹ️ Subject 1") for line in lines)
    assert result.tabs[0]["ad_prompt"] == (
        "woman, red hair, <lora:alice:0.8> [SEP] man <lora:bob:1.2> [SEP] [SKIP]"
    )


def test_a_hash_in_a_subject_is_told():
    # With the WebUI's prompt comments on, "#" hides the rest of the line:
    # the subject's own tag and every later part.
    result = guide(subjects=[("alice", 0.8, "woman #1"), BOB])
    assert (
        "⚠️ Subject 1 holds a #: with prompt comments on, the WebUI drops what "
        "follows it on the line, the other subjects included. Take the # out."
    ) in result.status.split("\n\n")
    assert "Subject 2 holds" not in result.status
    assert "holds a #" in guide(subjects=[("style#2", 0.8, "")]).status
    assert "holds a #" not in guide().status


# ------------------------------------------- one subject, and quoted values


@pytest.mark.parametrize("mode", GUIDE_MODES)
@pytest.mark.parametrize(
    ("sort", "more"),
    [("None", "Off"), ("Position (center to edge)", "Off"), ("Area", "Off"),
     (LEFT_TO_RIGHT, BBOX_SORTBY_MORE[1]), (None, None), (LEFT_TO_RIGHT, "Off")],
)
def test_one_subject_needs_no_word_about_the_sort_order(mode, sort, more):
    # One region is kept, the biggest: whatever order the Settings give, it
    # is the same one. The status used to tell to change the setting.
    result = guide(mode, [ALICE, EMPTY, EMPTY, EMPTY], sort_order=sort, more_sort_order=more)
    for values in result.tabs.values():
        assert values["ad_mask_k"] == 1
    one = "face" if mode == MODE_FACE else "person"
    head = result.status.split("\n\n")[0]
    assert head.startswith("✅ Filled the 1st tab ")
    assert " for 1 subject" in head
    assert head.endswith(f". Subject 1 is the biggest {one}.")
    assert "The parts follow Settings" not in result.status
    assert "Sort bounding boxes by" not in result.status
    assert "⚠️" not in result.status


def test_two_subjects_still_get_the_word_about_the_sort_order():
    for people, subjects in (("Auto", [ALICE, BOB]), ("2", [ALICE]), ("4", [ALICE])):
        status = guide(subjects=subjects, people=people, sort_order="None").status
        assert "⚠️ The parts follow Settings, ADetailer, Sort bounding boxes by, now None" in status
        assert "biggest" not in status


# What marked, the Markdown reader of Gradio 3.41 and 4.40, escapes in a text
# (its escape without "encode"): a character reference is left as it is, so
# the browser shows the character.
_MARKED_ESCAPES = re.compile(r"[<>]|&(?!(#\d{1,7}|#[Xx][a-fA-F0-9]{1,6}|\w+);)")


@pytest.mark.parametrize(
    "value",
    ["_new_", "*b*", "**bold**", "`code`", "a|b", "[link](http://example.com)",
     "![x](http://example.com/x.png)", "~~gone~~", "back\\_slash", "<b>bold</b>",
     "R&D &lt; &#95;", "a\nb\r\n# title", "plain name (v2) 1.5, ok!", "名前", "",
     "a$b$c", "$$x^2$$ and $y$", "cost: 5$"],
)
def test_a_quoted_value_is_shown_as_typed_not_read_as_markdown(value):
    shown = md_plain(value)
    # Nothing left that Markdown reads as markup (or Gradio as a formula),
    # and nothing marked would change: every "&" starts a character reference.
    assert not re.search(r"[\\`*_\[\]|~<>\r\n$]", shown)
    assert not _MARKED_ESCAPES.search(shown)
    # The browser shows the value itself (line breaks as spaces).
    import html

    assert html.unescape(shown) == value.replace("\r", " ").replace("\n", " ")
    assert md_plain("plain name (v2) 1.5, ok!") == "plain name (v2) 1.5, ok!"
    assert md_plain(None) == "None"
    assert md_plain(3) == "3"


def test_the_names_and_words_quoted_in_the_status_are_shown_as_typed():
    # "_new_" was shown in italics, "*b*" too, and so on for every name the
    # status quotes: only "<", ">" and "![" were taken care of.
    status = guide(
        subjects=[("_new_", 1, "w <lora:_new_:1>"), ("*b*", 1, "x"), ("`c`|[d]", 1, "")],
        main_prompt="<lora:*b*:1> <lora:_new_:1>",
        lora_names=["_NEW_", "other"],
    ).status
    lines = status.split("\n\n")
    assert (
        "⚠️ Subject 1: the tag typed in the words names &#95;new&#95;, but the "
        "WebUI lists that LoRA as &#95;NEW&#95;: with the name as typed that "
        "subject is redrawn without a LoRA. Write &#95;NEW&#95; in the tag, or "
        "take the tag out of the words."
    ) in lines
    renamed = guide(subjects=[("_new_", 1, "w")], lora_names=["_NEW_", "other"]).status
    assert (
        "ℹ️ Subject 1: &#95;new&#95; was written as &#95;NEW&#95;, the name in the "
        "WebUI's LoRA list."
    ) in renamed.split("\n\n")
    assert (
        "⚠️ Subject 2: the WebUI lists no LoRA named &#42;b&#42;. Its part was "
        "written as typed: with a wrong name that subject is redrawn without a "
        "LoRA. Pick the LoRA from the menu; a file added just now is listed "
        "after a refresh of the WebUI's LoRA list."
    ) in lines
    assert any(
        line.startswith("⚠️ Subject 3: the WebUI lists no LoRA named &#96;c&#96;&#124;&#91;d&#93;.")
        for line in lines
    )
    assert any(
        line.startswith("⚠️ &#42;b&#42; and &#95;new&#95; are also in the main prompt")
        for line in lines
    )
    kept = guide(subjects=[("_new_", 1, "w <lora:_new_:1>")]).status
    assert "the words already hold &#95;new&#95;, so the tag typed there is kept" in kept
    both = guide(
        subjects=[("_new_", 1, ""), ("*b*", 1, "")], main_prompt="<lora:*b*:1> <lora:_new_:1>"
    ).status
    assert "⚠️ &#42;b&#42; and &#95;new&#95; are also in the main prompt" in both
    refused = guide(subjects=[("_a_:*b*", 1, "")]).status
    assert refused.startswith('⚠️ Subject 1: "&#95;a&#95;:&#42;b&#42;" is not a LoRA name.')
    # A "$" too: Gradio's Markdown would read "$b$" or "$$b$$" as a formula.
    dollars = guide(subjects=[("a$b$c", 1, "x")], lora_names=["other"]).status
    assert "lists no LoRA named a&#36;b&#36;c." in dollars
    assert "$" not in dollars
    assert guide(subjects=[("a$b$c", 1, "x")]).tabs[0]["ad_prompt"] == (
        "x, <lora:a$b$c:1> [SEP] [SKIP]"
    )
    # No name of the user is left with a mark of its own. (The detector's
    # file name is the guide's own text: one "_" inside a word is no markup.)
    for text in (status, renamed, kept, both, refused):
        assert "*" not in text
        assert "`" not in text
        assert "_new_" not in text
    # The prompt is written as typed.
    assert guide(subjects=[("_new_", 1, "x")]).tabs[0]["ad_prompt"] == (
        "x, <lora:_new_:1> [SEP] [SKIP]"
    )


# --------------------------------------------- names the WebUI does not know

LISTED = ["alice", "bob", "Carol_v2", "detail.pt", "twin", "Twin"]


def _unknown(row, name):
    return (
        f"⚠️ Subject {row}: the WebUI lists no LoRA named {name}. Its part was "
        "written as typed: with a wrong name that subject is redrawn without a "
        "LoRA. Pick the LoRA from the menu; a file added just now is listed "
        "after a refresh of the WebUI's LoRA list."
    )


@pytest.mark.parametrize("name", ["Nessuno", "Aucun", "alcie", "alice2", "TWIN", "TWIN.pt"])
def test_a_name_the_webui_does_not_list_is_told_with_its_row(name):
    # The menus take typed text: with a translated interface the "None" of a
    # menu can come back as its translation, and the guide wrote
    # <lora:Nessuno:0.8> without a word. The same for a typing mistake.
    result = guide(subjects=[ALICE, (name, 0.8, "man")], lora_names=LISTED)
    lines = result.status.split("\n\n")
    assert lines[0].startswith("⚠️ Filled the 1st tab for 2 subjects")
    assert lines[1] == _unknown(2, name)
    assert not any(line.startswith("⚠️ Subject 1") for line in lines)
    # Written as typed, as the status says: a file added a moment ago is
    # found by the WebUI at Generate.
    assert result.tabs[0]["ad_prompt"] == (
        f"woman, red hair, <lora:alice:0.8> [SEP] man, <lora:{name}:0.8> [SEP] [SKIP]"
    )


@pytest.mark.parametrize(
    ("typed", "listed"),
    [
        # Another letter case, a file ending typed by mistake, both.
        ("ALICE", "alice"), ("carol_V2", "Carol_v2"),
        ("alice.safetensors", "alice"), ("bob.pt", "bob"), ("bob.ckpt", "bob"),
        ("Carol_v2.SafeTensors", "Carol_v2"), ("DETAIL.PT", "detail.pt"),
        # A name that ends like a file and is listed with it: itself first.
        ("detail.pt.safetensors", "detail.pt"),
        # The name without its ending as listed, also with a twin in
        # another letter case.
        ("twin.pt", "twin"), ("Twin.safetensors", "Twin"),
    ],
)
def test_a_name_in_another_letter_case_or_with_its_file_ending_is_written_as_listed(
    typed, listed
):
    assert match_lora(typed, LISTED) == listed
    result = guide(subjects=[(typed, 0.8, "woman")], lora_names=LISTED)
    assert result.tabs[0]["ad_prompt"] == f"woman, <lora:{listed}:0.8> [SEP] [SKIP]"
    lines = result.status.split("\n\n")
    assert lines[0].startswith("✅ ")
    assert lines[1] == (
        f"ℹ️ Subject 1: {md_plain(typed)} was written as {md_plain(listed)}, the "
        "name in the WebUI's LoRA list."
    )
    assert "lists no LoRA" not in result.status


@pytest.mark.parametrize(
    ("typed", "listed"),
    [("ALICE", "alice"), ("carol_V2", "Carol_v2"), ("bob.safetensors", "bob")],
)
def test_a_name_the_words_already_hold_as_typed_gets_no_second_tag(typed, listed):
    # The menu's name was written as listed and the tag typed in the words,
    # under the name as typed, stayed beside it: two tags for one subject
    # ("woman <lora:ALICE:0.5>, <lora:alice:0.8>"), and not a word about it.
    # A WebUI looks a tag's name up as written, capitals included.
    words = f"woman <lora:{typed}:0.5>"
    result = guide(subjects=[(typed, 0.8, words), BOB], lora_names=LISTED)
    assert result.tabs[0]["ad_prompt"] == (
        f"{words} [SEP] man, grey beard, <lora:bob:0.8> [SEP] [SKIP]"
    )
    lines = result.status.split("\n\n")
    assert lines[0].startswith("⚠️ Filled the 1st tab for 2 subjects")
    assert lines[1:] == [
        f"⚠️ Subject 1: the tag typed in the words names {md_plain(typed)}, but "
        f"the WebUI lists that LoRA as {md_plain(listed)}: with the name as typed "
        f"that subject is redrawn without a LoRA. Write {md_plain(listed)} in the "
        "tag, or take the tag out of the words.",
        f"ℹ️ Subject 1: the words already hold {md_plain(typed)}, so the tag typed "
        "there is kept and no second one is added.",
        DENOISE_LINE,
    ]
    # Nothing was written as listed, so that is not said.
    assert "was written as" not in result.status

    # The words hold the name as LISTED: the menu's name is written as
    # listed, as before, and the tag of the words is the only one.
    words = f"woman <lora:{listed}:0.5>"
    result = guide(subjects=[(typed, 0.8, words)], lora_names=LISTED)
    assert result.tabs[0]["ad_prompt"] == f"{words} [SEP] [SKIP]"
    assert result.status.split("\n\n")[1:] == [
        f"ℹ️ Subject 1: {md_plain(typed)} was written as {md_plain(listed)}, the "
        "name in the WebUI's LoRA list.",
        f"ℹ️ Subject 1: the words already hold {md_plain(listed)}, so the tag typed "
        "there is kept and no second one is added.",
        DENOISE_LINE,
    ]
    # A tag of the words under another name still leaves the menu's its own.
    result = guide(subjects=[(typed, 0.8, "woman <lora:detail.pt:0.5>")], lora_names=LISTED)
    assert result.tabs[0]["ad_prompt"] == (
        f"woman <lora:detail.pt:0.5>, <lora:{listed}:0.8> [SEP] [SKIP]"
    )


@pytest.mark.parametrize(
    ("typed", "listed"),
    [
        ("ALICE", "alice"), ("alice.safetensors", "alice"), ("carol_V2", "Carol_v2"),
        ("BOB.pt", "bob"),
    ],
)
def test_a_listed_name_the_words_hold_in_another_form_gets_no_second_tag(typed, listed):
    # The other way round, and the usual one: the LoRA is picked in the menu
    # (so under its listed name) and the tag typed by hand in the words has
    # other capitals or the file ending. Two tags came out for one subject
    # ("woman <lora:ALICE:0.5>, <lora:alice:0.8>") with a green status and not
    # a word: the typed tag is not found by the WebUI, so its weight was lost.
    words = f"woman <lora:{typed}:0.5>"
    result = guide(subjects=[(listed, 0.8, words)], lora_names=LISTED)
    # The words are not rewritten, and no second tag is added.
    assert result.tabs[0]["ad_prompt"] == f"{words} [SEP] [SKIP]"
    assert result.status.split("\n\n") == [
        "⚠️ Filled the 1st tab for 1 subject with person_yolov8s-seg.pt. "
        "Subject 1 is the biggest person.",
        f"⚠️ Subject 1: the tag typed in the words names {md_plain(typed)}, but "
        f"the WebUI lists that LoRA as {md_plain(listed)}: with the name as typed "
        f"that subject is redrawn without a LoRA. Write {md_plain(listed)} in the "
        "tag, or take the tag out of the words.",
        f"ℹ️ Subject 1: the words already hold {md_plain(typed)}, so the tag typed "
        "there is kept and no second one is added.",
        DENOISE_LINE,
    ]
    # The same for a name the WebUI knows as an alias.
    result = guide(
        subjects=[("AliceXL", 0.8, "woman <lora:alicexl:0.5>")],
        lora_names=LISTED, lora_aliases=["AliceXL"],
    )
    assert result.tabs[0]["ad_prompt"] == "woman <lora:alicexl:0.5> [SEP] [SKIP]"
    assert "names alicexl, but the WebUI lists that LoRA as AliceXL" in result.status


@pytest.mark.parametrize("menu", ["ALICE", "alice", "alice.safetensors"])
def test_no_lora_is_not_said_while_a_tag_of_the_words_is_found_by_the_webui(menu):
    # Both forms typed in the words: <lora:alice:0.7> is found by the WebUI,
    # so the subject IS redrawn with the LoRA. With ALICE in the menu the
    # status said "redrawn without a LoRA".
    words = "woman <lora:ALICE:0.5> <lora:alice:0.7>"
    result = guide(subjects=[(menu, 0.8, words)], lora_names=LISTED)
    assert result.tabs[0]["ad_prompt"] == f"{words} [SEP] [SKIP]"
    lines = result.status.split("\n\n")
    assert lines[0].startswith("✅ Filled the 1st tab for 1 subject")
    assert "without a LoRA" not in result.status
    assert lines[-2:] == [
        "ℹ️ Subject 1: the words already hold alice, so the tag typed there is "
        "kept and no second one is added.",
        DENOISE_LINE,
    ]
    # Two names of the list for one letter case: none is chosen for the tag
    # of the words either, and the menu's own LoRA is still written.
    result = guide(subjects=[("twin", 0.8, "woman <lora:TWIN:0.5>")], lora_names=LISTED)
    assert result.tabs[0]["ad_prompt"] == (
        "woman <lora:TWIN:0.5>, <lora:twin:0.8> [SEP] [SKIP]"
    )


def test_a_listed_name_and_an_alias_are_taken_as_they_are():
    for name in LISTED:
        assert match_lora(name, LISTED) == name
    # Exact first: "twin" and "Twin" are two LoRAs, each itself.
    assert match_lora("twin", LISTED) == "twin"
    assert match_lora("Twin", LISTED) == "Twin"
    # Two names for another letter case: none is chosen.
    assert match_lora("TWIN", LISTED) is None
    assert match_lora("nobody", LISTED) is None
    assert match_lora(".pt", LISTED) is None
    # AUTOMATIC1111 takes a LoRA's alias in a tag as well: not told. The
    # list comes first when both hold a name in another letter case.
    aliases = ["Alice (the alias)", "ALICE", "bob"]
    assert match_lora("Alice (the alias)", LISTED, aliases) == "Alice (the alias)"
    assert match_lora("alice (THE alias)", LISTED, aliases) == "Alice (the alias)"
    assert match_lora("Alice", LISTED, aliases) == "alice"
    assert match_lora("ALICE", LISTED, aliases) == "ALICE"
    known = guide(
        subjects=[("Alice (the alias)", 0.8, "woman")], lora_names=LISTED, lora_aliases=aliases
    )
    assert known.status.startswith("✅ ")
    assert "Subject 1:" not in known.status
    assert known.tabs[0]["ad_prompt"] == "woman, <lora:Alice (the alias):0.8> [SEP] [SKIP]"
    # Without the aliases the same name is told.
    told = guide(subjects=[("Alice (the alias)", 0.8, "woman")], lora_names=LISTED)
    assert _unknown(1, "Alice (the alias)") in told.status.split("\n\n")


# As networks.available_network_aliases gives them (scripts/!adetailer.py
# _lora_aliases): every file name, and the alias written in each file
# ("Alice" in alice.safetensors, "bobby_v2" in bob.safetensors), each with
# the file it stands for.
FILES = {"alice": "alice", "Alice": "alice", "bob": "bob", "bobby_v2": "bob"}


@pytest.mark.parametrize(
    ("menu", "tag"),
    [
        # The menu holds the file name, the words the alias (what a click on
        # the LoRA's card writes by default on AUTOMATIC1111).
        ("alice", "Alice"),
        ("bob", "bobby_v2"),
        # The other way round: the alias typed in the menu.
        ("bobby_v2", "bob"),
        ("Alice", "alice"),
    ],
)
def test_an_alias_and_the_file_name_of_one_lora_give_one_tag(menu, tag):
    # Both tags were written: the WebUI found both and applied the LoRA
    # twice (0.5 + 0.8), with a green status that said nothing.
    result = guide(
        subjects=[(menu, 0.8, f"woman <lora:{tag}:0.5>")],
        lora_names=["alice", "bob"], lora_aliases=FILES,
    )
    # The words are not rewritten: the tag typed there stays, alone.
    assert result.tabs[0]["ad_prompt"] == f"woman <lora:{tag}:0.5> [SEP] [SKIP]"
    assert result.status.split("\n\n") == [
        "✅ Filled the 1st tab for 1 subject with person_yolov8s-seg.pt. "
        "Subject 1 is the biggest person.",
        f"ℹ️ Subject 1: the words already hold {md_plain(tag)}, so the tag typed there is "
        "kept and no second one is added.",
        DENOISE_LINE,
    ]
    # A WebUI that does not tell the file of each alias (a list): as before,
    # each name is a LoRA of its own.
    result = guide(
        subjects=[(menu, 0.8, f"woman <lora:{tag}:0.5>")],
        lora_names=["alice", "bob"], lora_aliases=list(FILES),
    )
    assert result.tabs[0]["ad_prompt"] == (
        f"woman <lora:{tag}:0.5>, <lora:{menu}:0.8> [SEP] [SKIP]"
    )


def test_an_alias_in_the_main_prompt_is_the_lora_of_a_subject():
    # The main prompt holds the LoRA under its alias, a subject under its
    # file name: the warning that it acts on the whole picture was not given.
    result = guide(
        subjects=[("bob", 0.8, "man")], main_prompt="photo, <lora:bobby_v2:0.4>",
        lora_names=["alice", "bob"], lora_aliases=FILES,
    )
    assert f"⚠️ {md_plain('bobby_v2')} is also in the main prompt" in result.status
    # Another LoRA's alias is not.
    result = guide(
        subjects=[("bob", 0.8, "man")], main_prompt="photo, <lora:Alice:0.4>",
        lora_names=["alice", "bob"], lora_aliases=FILES,
    )
    assert "also in the main prompt" not in result.status


LYCO_NAMES = ["alice", "bob", "carol"]
_LYCO_LINE = (
    "⚠️ Subject {}: the words hold a lyco tag. Forge and Forge Neo do not read "
    "lyco tags, so there that subject is redrawn without the LoRA of that tag. "
    "Write lora instead of lyco in the tag."
)


@pytest.mark.parametrize(
    ("words", "prompt"),
    [
        # The row's own LoRA, as a lyco tag: it counts as the row's tag.
        ("woman <lyco:alice:0.5>", "woman <lyco:alice:0.5> [SEP] [SKIP]"),
        # Another LoRA, any letter case of "lyco".
        ("woman <LyCo:carol:0.5>", "woman <LyCo:carol:0.5>, <lora:alice:0.8> [SEP] [SKIP]"),
    ],
)
def test_a_lyco_tag_is_told_where_forge_does_not_read_it(words, prompt):
    # Forge and Forge Neo register no "lyco" network (their
    # modules/extra_networks.py skips an unknown tag): the subject was
    # redrawn without that LoRA and the status was green.
    result = guide(subjects=[("alice", 0.8, words)], lora_names=LYCO_NAMES)
    # The prompt as before.
    assert result.tabs[0]["ad_prompt"] == prompt
    lines = result.status.split("\n\n")
    assert lines[0].startswith("⚠️ Filled the 1st tab for 1 subject")
    assert _LYCO_LINE.format(1) in lines
    # Where the WebUI surely reads it (AUTOMATIC1111), nothing is said.
    read = guide(subjects=[("alice", 0.8, words)], lora_names=LYCO_NAMES, reads_lyco=True)
    assert read.tabs == result.tabs
    assert read.status.startswith("✅ ")
    assert "lyco" not in read.status
    # A lora tag is never told; each subject with a lyco tag is.
    plain = guide(subjects=[("alice", 0.8, "woman <lora:alice:0.5>")], lora_names=LYCO_NAMES)
    assert "lyco" not in plain.status
    two = guide(
        subjects=[("alice", 0.8, words), ("bob", 1, "man <lyco:bob:1>")], lora_names=LYCO_NAMES
    )
    assert [line for line in two.status.split("\n\n") if "lyco tag" in line] == [
        _LYCO_LINE.format(1), _LYCO_LINE.format(2),
    ]


@pytest.mark.parametrize("names", [None, [], (), [None, 3, ""], 7, "alice"])
def test_without_a_lora_list_no_name_is_compared(names):
    # A WebUI without a LoRA list still takes a typed name: nothing to tell.
    # (A text is not a list of names either: its letters are not LoRAs.)
    result = guide(subjects=[("Nessuno", 0.8, "woman"), ("ALICE", 1, "")], lora_names=names)
    assert result.status == (
        "✅ Filled the 1st tab for 2 subjects with person_yolov8s-seg.pt. "
        "Subject 1 is the leftmost of the 2 biggest people.\n\n" + DENOISE_LINE
    )
    assert result.tabs[0]["ad_prompt"] == (
        "woman, <lora:Nessuno:0.8> [SEP] [PROMPT], <lora:ALICE:1> [SEP] [SKIP]"
    )


def test_what_is_not_a_menu_name_is_not_compared_with_the_list():
    # "None", an empty row, words alone, a tag typed in the words.
    result = guide(
        subjects=[("None", 0.8, "woman <lora:typed_in_words:1>"), EMPTY, ("", 1, "man")],
        lora_names=LISTED,
    )
    assert "lists no LoRA" not in result.status
    assert result.status.startswith("✅ ")
    # A name no tag can hold is refused as before, whatever the list holds.
    refused = guide(subjects=[("a:b", 1, "")], lora_names=[*LISTED, "a:b"])
    assert refused.tabs == {}
    assert '"a:b" is not a LoRA name' in refused.status
    # The name as listed is the one compared with the tags typed in the words.
    kept = guide(subjects=[("ALICE", 0.8, "woman <lora:alice:1.1>")], lora_names=LISTED)
    assert kept.tabs[0]["ad_prompt"] == "woman <lora:alice:1.1> [SEP] [SKIP]"


# The comment rules of the two WebUIs (modules/processing_scripts/comments.py).
def _a1111_strip(text):
    text = re.sub("(^|\n)#[^\n]*(\n|$)", "\n", text)
    return re.sub("#[^\n]*(\n|$)", "\n", text)


def _neo_strip(text):
    text = re.sub(r"\/\*.*?\*\/", "", text, flags=re.DOTALL)
    return re.sub(r"[^\S\n]*(\#|\/\/).*", "", text)


def _broken_strip(text):
    raise RuntimeError("no such option")


def test_the_comment_marks_are_those_of_the_webui_in_use():
    assert COMMENT_MARKS == ("#", "//", "/*")
    assert host_comment_marks(_a1111_strip) == ("#",)
    assert host_comment_marks(_neo_strip) == ("#", "//", "/*")
    # Prompt comments off, or a WebUI without them: the text comes back as
    # it is, and nothing is a comment.
    assert host_comment_marks(lambda text: text) == ()
    # Not known: every mark is told.
    for unknown in (None, "strip", 3, _broken_strip):
        assert host_comment_marks(unknown) == COMMENT_MARKS


@pytest.mark.parametrize(
    ("words", "mark", "shown", "dropped"),
    [
        ("blush, ///", "//", "//", "what follows it on the line"),
        ("see http://example.com", "//", "//", "what follows it on the line"),
        ("a /* b", "/*", "/&#42;", "what follows it up to the next &#42;/"),
        ("woman #1", "#", "#", "what follows it on the line"),
    ],
)
def test_a_comment_mark_of_forge_neo_in_a_subject_is_told(words, mark, shown, dropped):
    # On Forge Neo "//" and "/* */" are prompt comments too: with "blush, ///"
    # in the words only one part was left, so no LoRA and no [SKIP], and the
    # status said nothing (only "#" was told).
    result = guide(subjects=[("alice", 0.8, words), ("bob", 0.8, "man /* x */")],
                   comment_marks=host_comment_marks(_neo_strip))
    prompt = result.tabs[0]["ad_prompt"]
    # What the finding was: the other subjects go with the comment.
    assert len(re.split(r"\s*\[SEP\]\s*", _neo_strip(prompt))) < 3
    note = (
        f"⚠️ Subject 1 holds a {shown}: with prompt comments on, the WebUI drops "
        f"{dropped}, the other subjects included. Take the {shown} out."
    )
    lines = result.status.split("\n\n")
    assert note in lines
    assert sum(line.startswith("⚠️ Subject 1 holds") for line in lines) == 1
    # Each subject is told for its own marks.
    assert sum(line.startswith("⚠️ Subject 2 holds a /&#42;:") for line in lines) == 1
    # A "*" as it is could open an emphasis in the status.
    assert "*" not in "".join(line for line in lines if " holds a " in line)
    # Without the WebUI's rules every mark is told as well.
    assert note in guide(subjects=[("alice", 0.8, words)]).status.split("\n\n")


def test_only_the_comment_marks_of_the_webui_in_use_are_told():
    rows = [("alice", 0.8, "a // b /* c */ # d"), BOB]
    told = lambda marks: [  # noqa: E731
        line.split(":")[0] for line in guide(subjects=rows, comment_marks=marks).status.split("\n\n")
        if " holds a " in line
    ]
    # AUTOMATIC1111: "//" and "/* */" are plain text there.
    assert told(host_comment_marks(_a1111_strip)) == ["⚠️ Subject 1 holds a #"]
    assert told(host_comment_marks(_neo_strip)) == [
        "⚠️ Subject 1 holds a #", "⚠️ Subject 1 holds a //", "⚠️ Subject 1 holds a /&#42;",
    ]
    # Prompt comments off: nothing is dropped, nothing is told.
    assert told(()) == []
    assert told(None) == told(COMMENT_MARKS) == told(host_comment_marks(_neo_strip))
    # The prompt is written as typed in every case.
    assert guide(subjects=rows, comment_marks=()).tabs[0]["ad_prompt"].startswith(
        "a // b /* c */ # d, <lora:alice:0.8> [SEP] "
    )


def test_the_guide_is_pure():
    # No Gradio, torch or WebUI module: the whole logic runs offline, and a
    # WebUI without one of them cannot break it.
    tree = ast.parse((_ROOT / "adetailer" / "subject_guide.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {
        "__future__", "collections", "dataclasses", "math", "re", "typing", "adetailer",
    }
    assert not {"gradio", "torch", "modules"} & imported
    # And of the package, only the args module (pydantic, no detector).
    assert [
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("adetailer")
    ] == ["adetailer.args"]


def test_the_same_choices_give_the_same_answer_twice():
    first = guide(MODE_PERSON_FACE, old_prompts=["x"], enabled_tabs=[True] * 4)
    second = guide(MODE_PERSON_FACE, old_prompts=["x"], enabled_tabs=[True] * 4)
    assert first == second
    # Changing an answer does not change the next one.
    first.tabs[0]["ad_prompt"] = "changed"
    assert guide(MODE_PERSON_FACE).tabs[0]["ad_prompt"] == TWO
