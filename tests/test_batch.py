import csv
import io

import pandas as pd
import pytest

from batch import TEMPLATE_CSV, load_batch, results_csv, run_batch, summarize
from gemini_client import AIError

CRITERIA = {"Accuracy": "a", "Clarity": "c"}


def verdict(winner, a=(5, 4), b=(2, 3)):
    return {
        "criteria": [
            {"name": "Accuracy", "score_a": a[0], "score_b": b[0], "reasoning": "r"},
            {"name": "Clarity", "score_a": a[1], "score_b": b[1], "reasoning": "r"},
        ],
        "total_a": sum(a), "total_b": sum(b), "winner": winner, "summary": "", "consistent": None,
    }


def test_template_loads_cleanly():
    rows, notes = load_batch(pd.read_csv(io.StringIO(TEMPLATE_CSV), dtype=str, keep_default_na=False))
    assert len(rows) == 2 and not notes
    assert rows[0]["row"] == 2 and rows[1]["reference"] == ""


def test_load_batch_matches_columns_case_insensitively_and_skips_blanks():
    df = pd.DataFrame({"Prompt": ["q1", "", "q3"], "Response_A": ["a", "a", "a"], "RESPONSE_B": ["b", "b", " "]})
    rows, notes = load_batch(df)
    assert [r["row"] for r in rows] == [2]
    assert "Skipped 2 row(s)" in notes[0]


def test_load_batch_requires_columns():
    with pytest.raises(AIError, match="response_b"):
        load_batch(pd.DataFrame({"prompt": ["q"], "response_a": ["a"]}))


def test_run_batch_records_failures_and_continues():
    rows = [{"row": 2, "prompt": "q", "response_a": "a", "response_b": "b", "reference": ""},
            {"row": 3, "prompt": "boom", "response_a": "a", "response_b": "b", "reference": "ref"}]
    seen = []

    def fake_evaluate(prompt, a, b, criteria, reference=None, bias_check=False):
        seen.append(reference)
        if prompt == "boom":
            raise AIError("rate limited")
        return verdict("A")

    results = run_batch(rows, CRITERIA, evaluate_fn=fake_evaluate)
    assert seen == [None, "ref"]
    assert results[0]["verdict"]["winner"] == "A" and results[1]["error"] == "rate limited"


def test_summarize_and_export():
    results = [
        {"row": 2, "prompt": "q", "verdict": verdict("A"), "error": None},
        {"row": 3, "prompt": "q", "verdict": verdict("B", a=(1, 2), b=(4, 5)), "error": None},
        {"row": 4, "prompt": "q", "verdict": verdict("A", a=(3, 3), b=(3, 3)), "error": None},
        {"row": 5, "prompt": "q", "verdict": None, "error": "failed"},
    ]
    s = summarize(results, list(CRITERIA))
    assert s["evaluated"] == 3 and s["failed"] == 1
    assert s["counts"]["A"] == 2 and round(s["win_rate_a"], 2) == 0.67
    assert s["averages"][0] == {"criterion": "Accuracy", "avg_a": 3.0, "avg_b": 3.0}

    rows = list(csv.DictReader(io.StringIO(results_csv(results))))
    assert len(rows) == 7                       # 3 rows x 2 criteria + 1 failed row
    assert rows[-1]["error"] == "failed"
