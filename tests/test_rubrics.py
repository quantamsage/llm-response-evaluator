import json

import pytest

from judge import MAX_CRITERIA
from rubrics import PRESETS, RubricError, clean_criteria, rubric_from_json, rubric_to_json


def test_round_trip():
    criteria = {"Tone": "Friendly", "Accuracy": "Correct facts"}
    name, loaded = rubric_from_json(rubric_to_json("Support", criteria))
    assert name == "Support" and loaded == criteria
    assert list(loaded) == ["Tone", "Accuracy"]  # order kept


def test_clean_criteria_skips_blank_names_and_fills_descriptions():
    assert clean_criteria([("Tone", ""), ("", "ignored"), (None, None)]) == {"Tone": "Tone"}


@pytest.mark.parametrize("pairs", [
    [("Tone", "a"), ("tone", "b")],                          # duplicate names
    [(f"C{i}", "d") for i in range(MAX_CRITERIA + 1)],      # too many
    [("x" * 61, "d")],                                       # name too long
])
def test_clean_criteria_rejects_bad_rubrics(pairs):
    with pytest.raises(RubricError):
        clean_criteria(pairs)


@pytest.mark.parametrize("text", ["not json", json.dumps({"name": "x"}), json.dumps({"criteria": []}),
                                  json.dumps({"criteria": ["just a string"]})])
def test_rubric_from_json_rejects_bad_files(text):
    with pytest.raises(RubricError):
        rubric_from_json(text)


def test_presets_are_valid():
    for criteria in PRESETS.values():
        assert clean_criteria(criteria.items()) == criteria
