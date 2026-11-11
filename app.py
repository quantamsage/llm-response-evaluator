"""Streamlit interface for the LLM Response Evaluator."""

import csv
import datetime
import io
import logging

import streamlit as st

from gemini_client import AIError
from judge import (
    DEFAULT_CRITERIA,
    MAX_PROMPT_CHARS,
    MAX_REFERENCE_CHARS,
    MAX_RESPONSE_CHARS,
    evaluate,
)

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


def render_sidebar():
    """Criteria and options. Returns (criteria dict, bias_check flag)."""
    st.sidebar.header("Scoring setup")
    chosen = st.sidebar.multiselect(
        "Criteria", list(DEFAULT_CRITERIA), default=list(DEFAULT_CRITERIA)
    )
    criteria = {name: DEFAULT_CRITERIA[name] for name in chosen}

    custom = st.sidebar.text_input(
        "Custom criterion (optional)", max_chars=200, placeholder="e.g. Tone: friendly and professional"
    )
    if custom.strip():
        name, _, desc = custom.partition(":")
        if name.strip():
            criteria[name.strip()] = desc.strip() or name.strip()

    bias_check = st.sidebar.checkbox(
        "Position-bias check",
        help="Runs the judge twice with A and B swapped and flags it if the winner changes. Uses 2 API calls.",
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


def main():
    st.set_page_config(page_title="LLM Response Evaluator", page_icon="⚖️", layout="wide")
    st.title("⚖️ LLM Response Evaluator")
    st.markdown("Compare two AI responses side by side, scored against a rubric by an AI judge.")

    criteria, bias_check = render_sidebar()
    render_inputs()
    st.session_state.setdefault("history", [])

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
        st.caption("AI judges can be wrong. Use these scores to support human review, not replace it.")


if __name__ == "__main__":
    main()
