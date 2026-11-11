"""Batch mode: evaluate many prompt/response pairs from a CSV and summarise them."""

import csv
import io

from gemini_client import AIError
from judge import evaluate

REQUIRED_COLUMNS = ["prompt", "response_a", "response_b"]
OPTIONAL_COLUMNS = ["reference"]
MAX_BATCH_ROWS = 50
OUTCOMES = ["A", "B", "tie", "inconsistent"]

TEMPLATE_CSV = (
    "prompt,response_a,response_b,reference\n"
    '"What is 2 + 2?","4","2 + 2 is 5.","4"\n'
    '"Name the capital of Kenya.","Nairobi is the capital of Kenya.","Mombasa.",""\n'
)


def _is_blank(value):
    return value is None or str(value).strip() == "" or str(value).strip().lower() == "nan"


def load_batch(df):
    """Check an uploaded DataFrame and return (rows, notes).

    Column names are matched case-insensitively. Rows missing a prompt or either
    response are skipped and reported in `notes`.
    """
    lookup = {str(column).strip().lower(): column for column in df.columns}
    missing = [column for column in REQUIRED_COLUMNS if column not in lookup]
    if missing:
        raise AIError(
            "The CSV is missing these columns: " + ", ".join(missing)
            + ". Download the template to see the expected layout."
        )

    rows, skipped = [], []
    for position, (_, record) in enumerate(df.iterrows(), start=2):  # row 1 is the header
        values = {}
        for column in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
            raw = record[lookup[column]] if column in lookup else ""
            values[column] = "" if _is_blank(raw) else str(raw)
        if any(not values[column].strip() for column in REQUIRED_COLUMNS):
            skipped.append(position)
            continue
        rows.append({"row": position, **values})

    notes = []
    if skipped:
        shown = ", ".join(str(r) for r in skipped[:10]) + ("..." if len(skipped) > 10 else "")
        notes.append(f"Skipped {len(skipped)} row(s) with a blank prompt or response: {shown}.")
    if len(rows) > MAX_BATCH_ROWS:
        notes.append(f"Only the first {MAX_BATCH_ROWS} usable rows will be evaluated.")
        rows = rows[:MAX_BATCH_ROWS]
    if not rows:
        raise AIError("No usable rows found. Each row needs a prompt, response_a and response_b.")
    return rows, notes


def run_batch(rows, criteria, bias_check=False, progress=None, evaluate_fn=evaluate):
    """Evaluate every row. A failed row is recorded with its error and the batch carries on."""
    results = []
    for number, item in enumerate(rows, start=1):
        try:
            verdict = evaluate_fn(
                item["prompt"], item["response_a"], item["response_b"], criteria,
                reference=item.get("reference") or None, bias_check=bias_check,
            )
            results.append({**item, "verdict": verdict, "error": None})
        except AIError as error:
            results.append({**item, "verdict": None, "error": str(error)})
        if progress:
            progress(number / len(rows))
    return results


def summarize(results, criteria_names):
    """Win counts, win rates and per-criterion averages across a batch."""
    judged = [r["verdict"] for r in results if r["verdict"]]
    counts = {outcome: sum(v["winner"] == outcome for v in judged) for outcome in OUTCOMES}
    total = len(judged)

    averages = []
    for name in criteria_names:
        scores = [c for v in judged for c in v["criteria"] if c["name"] == name]
        if scores:
            averages.append({
                "criterion": name,
                "avg_a": round(sum(c["score_a"] for c in scores) / len(scores), 2),
                "avg_b": round(sum(c["score_b"] for c in scores) / len(scores), 2),
            })

    return {
        "evaluated": total,
        "failed": len(results) - total,
        "counts": counts,
        "win_rate_a": counts["A"] / total if total else 0.0,
        "win_rate_b": counts["B"] / total if total else 0.0,
        "averages": averages,
    }


def results_csv(results):
    """One line per row and criterion, plus a line for rows that failed."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["row", "prompt", "winner", "total_a", "total_b",
                     "criterion", "score_a", "score_b", "reasoning", "error"])
    for r in results:
        verdict = r["verdict"]
        if not verdict:
            writer.writerow([r["row"], r["prompt"][:200], "", "", "", "", "", "", "", r["error"]])
            continue
        for c in verdict["criteria"]:
            writer.writerow([
                r["row"], r["prompt"][:200], verdict["winner"], verdict["total_a"], verdict["total_b"],
                c["name"], c["score_a"], c["score_b"], c["reasoning"], "",
            ])
    return buffer.getvalue()
