"""Streamlit interface for the LLM Response Evaluator."""

import csv
import datetime
import io
import logging

import pandas as pd
import streamlit as st

from batch import TEMPLATE_CSV, load_batch, results_csv, run_batch, summarize
from gemini_client import AIError
from judge import (
    MAX_CRITERIA,
    MAX_PROMPT_CHARS,
    MAX_REFERENCE_CHARS,
    MAX_RESPONSE_CHARS,
    evaluate,
)
from rubrics import PRESETS, RubricError, clean_criteria, rubric_from_json, rubric_to_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

EXAMPLE = {
    "prompt": "Explain the difference between a list and a tuple in Python, with a short example.",
    "response_a": (
        "Lists are mutable: you can add, remove, or change items after creation. "
        "Tuples are immutable: once created, they can't be changed. Tuples are often "
        "used for fixed collections and can be dictionary keys.\n\n"
        "colors = ['red', 'green']\ncolors.append('blue')   # works\n\n"
        "point = (3, 4)\npoint[0] = 5   # TypeError: tuples don't support item assignment"
    ),
    "response_b": (
        "Both lists and tuples store items, and both can be changed after you create them. "
        "The only difference is the brackets: lists use [] and tuples use ().\n\n"
        "my_tuple = (1, 2)\nmy_tuple.append(3)"
    ),
    "reference": "Lists are mutable; tuples are immutable. Tuples are hashable if their items are.",
}

WINNER_TEXT = {"A": "Response A wins", "B": "Response B wins", "tie": "It's a tie"}


def load_example():
    for key, value in EXAMPLE.items():
        st.session_state[key] = value


def criteria_frame(criteria):
    return pd.DataFrame(
        [{"Criterion": n, "Description": d} for n, d in criteria.items()],
        columns=["Criterion", "Description"],
    )


def set_rubric(name, criteria):
    """Replace the rubric being edited. Bumping the version resets the editor widget."""
    st.session_state.rubric_name = name
    st.session_state.rubric_frame = criteria_frame(criteria)
    st.session_state.rubric_version = st.session_state.get("rubric_version", 0) + 1


def render_sidebar():
    """Rubric editor and options. Returns (criteria dict, bias_check flag)."""
    if "rubric_frame" not in st.session_state:
        first = next(iter(PRESETS))
        set_rubric(first, PRESETS[first])

    st.sidebar.header("Scoring rubric")
    preset = st.sidebar.selectbox("Start from a preset", ["—", *PRESETS], key="preset_choice")
    if preset != "—" and st.sidebar.button("Load preset", width="stretch"):
        set_rubric(preset, PRESETS[preset])
        st.rerun()

    uploaded = st.sidebar.file_uploader("Or load a saved rubric (.json)", type="json")
    if uploaded is not None and st.session_state.get("loaded_rubric_id") != uploaded.file_id:
        st.session_state.loaded_rubric_id = uploaded.file_id
        try:
            name, criteria = rubric_from_json(uploaded.getvalue().decode("utf-8", errors="replace"))
            set_rubric(name, criteria)
            st.rerun()
        except RubricError as error:
            st.sidebar.error(str(error))

    st.sidebar.caption(f"Edit, add or delete rows. Up to {MAX_CRITERIA} criteria.")
    edited = st.sidebar.data_editor(
        st.session_state.rubric_frame,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key=f"rubric_editor_{st.session_state.rubric_version}",
        column_config={
            "Criterion": st.column_config.TextColumn(required=True, max_chars=60),
            "Description": st.column_config.TextColumn(max_chars=300),
        },
    )
    try:
        criteria = clean_criteria(zip(edited["Criterion"], edited["Description"]))
    except RubricError as error:
        st.sidebar.error(str(error))
        criteria = {}

    name = st.sidebar.text_input("Rubric name", key="rubric_name", max_chars=60)
    st.sidebar.download_button(
        "Save rubric (.json)", rubric_to_json(name, criteria) if criteria else "{}",
        file_name=f"{(name or 'rubric').strip().replace(' ', '_').lower()}.json",
        mime="application/json", disabled=not criteria, width="stretch",
    )

    st.sidebar.divider()
    bias_check = st.sidebar.checkbox(
        "Position-bias check",
        help="Runs the judge twice with A and B swapped and flags it if the winner changes. Uses 2 API calls per comparison.",
    )
    st.sidebar.caption("Scores run from 1 (very poor) to 5 (excellent).")
    return criteria, bias_check


def render_inputs():
    st.button("Load example", on_click=load_example)
    st.text_area("Prompt", key="prompt", height=100, max_chars=MAX_PROMPT_CHARS, placeholder="The prompt both responses were answering")
    col_a, col_b = st.columns(2)
    with col_a:
        st.text_area("Response A", key="response_a", height=240, max_chars=MAX_RESPONSE_CHARS)
    with col_b:
        st.text_area("Response B", key="response_b", height=240, max_chars=MAX_RESPONSE_CHARS)
    with st.expander("Reference answer (optional)"):
        st.text_area("Reference", key="reference", height=100, label_visibility="collapsed", max_chars=MAX_REFERENCE_CHARS,
                     placeholder="A known-good answer the judge can check correctness against")


def render_verdict(verdict):
    if verdict["winner"] == "inconsistent":
        a_win, b_win = verdict["winners"]
        st.warning(
            f"Position bias detected: the judge picked {a_win} in the first run and {b_win} "
            "with the order swapped. Treat this comparison as too close to call."
        )
    else:
        st.success(f"**{WINNER_TEXT[verdict['winner']]}**")
        if verdict.get("consistent"):
            st.caption("Position-bias check passed: same winner with the order swapped.")

    col_a, col_b = st.columns(2)
    max_total = len(verdict["criteria"]) * 5
    col_a.metric("Response A total", f"{verdict['total_a']} / {max_total}")
    col_b.metric("Response B total", f"{verdict['total_b']} / {max_total}")

    st.dataframe(
        [{"Criterion": r["name"], "A": r["score_a"], "B": r["score_b"]} for r in verdict["criteria"]],
        hide_index=True, width="stretch",
    )
    if verdict["summary"]:
        st.markdown(f"**Judge's summary:** {verdict['summary']}")
    with st.expander("Reasoning per criterion"):
        for r in verdict["criteria"]:
            st.markdown(f"**{r['name']}** (A {r['score_a']}, B {r['score_b']}): {r['reasoning']}")


def history_csv(history):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["time", "prompt", "winner", "total_a", "total_b", "criterion", "score_a", "score_b", "reasoning"])
    for entry in history:
        for r in entry["verdict"]["criteria"]:
            writer.writerow([
                entry["time"], entry["prompt"], entry["verdict"]["winner"],
                entry["verdict"]["total_a"], entry["verdict"]["total_b"],
                r["name"], r["score_a"], r["score_b"], r["reasoning"],
            ])
    return buffer.getvalue()


def render_single(criteria, bias_check):
    render_inputs()
    if st.button("Evaluate", type="primary", width="stretch"):
        with st.spinner("Judging responses..."):
            try:
                verdict = evaluate(
                    st.session_state.get("prompt", ""),
                    st.session_state.get("response_a", ""),
                    st.session_state.get("response_b", ""),
                    criteria,
                    reference=st.session_state.get("reference", ""),
                    bias_check=bias_check,
                )
                st.session_state.verdict, st.session_state.error = verdict, None
                st.session_state.history.append({
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "prompt": st.session_state.get("prompt", "")[:200],
                    "verdict": verdict,
                })
            except AIError as error:
                st.session_state.verdict, st.session_state.error = None, str(error)
            except Exception:
                logger.exception("Unexpected error during evaluation")
                st.session_state.verdict = None
                st.session_state.error = "Something unexpected went wrong. Try again."

    if st.session_state.get("error"):
        st.error(st.session_state.error)
    if st.session_state.get("verdict"):
        st.subheader("Verdict")
        render_verdict(st.session_state.verdict)

    if st.session_state.history:
        st.divider()
        st.subheader(f"Session history ({len(st.session_state.history)})")
        st.download_button(
            "Download all evaluations (CSV)", history_csv(st.session_state.history),
            file_name="evaluations.csv", mime="text/csv",
        )


def render_batch_summary(summary, results):
    counts = summary["counts"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evaluated", summary["evaluated"], help=f"{summary['failed']} row(s) failed" if summary["failed"] else None)
    c2.metric("A win rate", f"{summary['win_rate_a']:.0%}", f"{counts['A']} wins", delta_color="off")
    c3.metric("B win rate", f"{summary['win_rate_b']:.0%}", f"{counts['B']} wins", delta_color="off")
    c4.metric("Ties / inconsistent", f"{counts['tie']} / {counts['inconsistent']}")

    if summary["averages"]:
        st.markdown("**Average score per criterion** (1–5)")
        averages = pd.DataFrame(summary["averages"]).set_index("criterion")
        averages.columns = ["Response A", "Response B"]
        st.bar_chart(averages, stack=False, y_label="Average score", height=280)

    table = [{
        "Row": r["row"],
        "Prompt": r["prompt"][:120],
        "Winner": r["verdict"]["winner"] if r["verdict"] else "—",
        "A total": r["verdict"]["total_a"] if r["verdict"] else None,
        "B total": r["verdict"]["total_b"] if r["verdict"] else None,
        "Error": r["error"] or "",
    } for r in results]
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("Download batch results (CSV)", results_csv(results),
                       file_name="batch_evaluations.csv", mime="text/csv")


def render_batch(criteria, bias_check):
    st.markdown(
        "Upload a CSV with `prompt`, `response_a` and `response_b` columns (plus an optional "
        "`reference`). Every row is judged with the current rubric."
    )
    st.download_button("Download CSV template", TEMPLATE_CSV, file_name="batch_template.csv", mime="text/csv")
    uploaded = st.file_uploader("Batch CSV", type="csv", label_visibility="collapsed")
    if uploaded is None:
        return
    try:
        rows, notes = load_batch(pd.read_csv(uploaded, dtype=str, keep_default_na=False))
    except AIError as error:
        st.error(str(error))
        return
    except Exception:
        logger.exception("Failed to read batch CSV")
        st.error("That file couldn't be read as a CSV. Check it opens in a spreadsheet app.")
        return
    for note in notes:
        st.info(note)

    calls = len(rows) * (2 if bias_check else 1)
    st.caption(f"{len(rows)} row(s) ready. This uses about {calls} API call(s); free-tier rate limits may apply.")
    if st.button(f"Evaluate {len(rows)} row(s)", type="primary", disabled=not criteria):
        bar = st.progress(0.0, text="Judging batch...")
        results = run_batch(rows, criteria, bias_check,
                            progress=lambda p: bar.progress(p, text="Judging batch..."))
        bar.empty()
        st.session_state.batch = {"results": results, "summary": summarize(results, list(criteria))}

    if st.session_state.get("batch"):
        render_batch_summary(st.session_state.batch["summary"], st.session_state.batch["results"])


def main():
    st.set_page_config(page_title="LLM Response Evaluator", page_icon="⚖️", layout="wide")
    st.title("⚖️ LLM Response Evaluator")
    st.markdown("Compare AI responses against a rubric with an AI judge: one pair at a time, or a whole CSV at once.")

    criteria, bias_check = render_sidebar()
    st.session_state.setdefault("history", [])
    if not criteria:
        st.warning("Add at least one criterion to the rubric in the sidebar.")

    single, batch = st.tabs(["Single comparison", "Batch (CSV)"])
    with single:
        render_single(criteria, bias_check)
    with batch:
        render_batch(criteria, bias_check)

    st.caption("AI judges can be wrong. Use these scores to support human review, not replace it.")


if __name__ == "__main__":
    main()
