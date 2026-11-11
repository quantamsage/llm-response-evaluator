# ⚖️ LLM Response Evaluator

![Tests](https://github.com/quantamsage/llm-response-evaluator/actions/workflows/tests.yml/badge.svg)

A Streamlit app that compares two AI responses to the same prompt and scores them against a rubric, using Google Gemini as the judge. Compare one pair at a time, or upload a CSV and score a whole batch. It includes an optional position-bias check that re-runs the judgement with the responses swapped and flags it when the verdict changes.

Built to show structured LLM evaluation in practice: rubric-based scoring, schema-validated JSON output, and a simple guard against one of the best-known weaknesses of LLM judges.

## Features

- Paste a prompt and two responses, and get a 1–5 score per criterion for each response, a winner, and the judge's reasoning
- **Editable rubric:** criteria and descriptions are an editable table in the sidebar (add, rename, delete), with presets for general answers, code answers and customer support
- **Saved rubrics:** download the current rubric as JSON and load it again later, so the same criteria are reused across sessions and shared with teammates
- **Batch mode:** upload a CSV of `prompt`, `response_a`, `response_b` (and optional `reference`), score every row, and get win rates, ties, a per-criterion average chart and a full CSV of results. A row that fails is recorded and the batch carries on
- Optional reference answer the judge uses to check correctness
- **Position-bias check:** runs the judge twice with A and B swapped, averages the scores, and warns if the winner flips
- Every reply from the model is validated: all criteria must be scored, scores are clamped to 1–5, and the winner must be A, B or tie
- Session history with CSV export of every evaluation
- "Load example" button and a downloadable batch CSV template for a quick demo
- Friendly error messages for a missing or invalid API key, rate limits, network problems and malformed AI output

## Screenshots

<img width="1600" height="793" alt="WhatsApp Image 2026-10-06 at 18 13 57" src="https://github.com/user-attachments/assets/d6073d1e-d4c2-4d99-8ca7-2efea1d5fbef" />
<img width="1600" height="797" alt="WhatsApp Image 2026-10-06 at 18 15 51" src="https://github.com/user-attachments/assets/31689d36-a1f5-44d6-ad1c-024b644eb4d0" />
<img width="1600" height="792" alt="WhatsApp Image 2026-10-06 at 18 16 10" src="https://github.com/user-attachments/assets/a9ecd927-88db-4831-971c-d7046ab2de9a" />

## Technologies used

- Python 3.10+
- [Streamlit](https://streamlit.io/) for the interface
- [Google Gemini API](https://ai.google.dev/) via the official `google-genai` SDK
- pandas for batch CSV handling
- `python-dotenv` for configuration
- `pytest` for tests (development only)

## Project structure

```
llm-response-evaluator/
├── app.py              # Streamlit UI: rubric editor, single and batch tabs, exports
├── judge.py            # Default rubric, judge prompt, verdict validation, position-bias logic
├── batch.py            # Batch CSV loading, running, summary statistics and export
├── rubrics.py          # Rubric presets, validation, save/load as JSON
├── gemini_client.py    # Gemini API call, JSON parsing, friendly error handling
├── tests/
│   ├── test_judge.py   # Validation and bias-check logic
│   ├── test_batch.py   # CSV loading, failure handling, win rates and averages
│   └── test_rubrics.py # Rubric round-trip and validation
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

## Setup

```bash
git clone https://github.com/quantamsage/llm-response-evaluator.git
cd llm-response-evaluator
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Create a free Gemini API key at [Google AI Studio](https://aistudio.google.com/apikey), then copy `.env.example` to `.env` and set:

```
AI_API_KEY=your_real_key_here
```

The default model is `gemini-3.1-flash-lite`. Set `AI_MODEL` in `.env` to use another one.

## Run

```bash
streamlit run app.py
```

## Run the tests

```bash
pip install pytest
python -m pytest
```

The tests cover verdict validation, the position-bias logic, batch processing and rubric files, and don't call the API.

## How it works

1. `judge.py` builds a grading prompt containing the user's prompt, both responses, the optional reference answer, and the chosen criteria with their descriptions.
2. Gemini is asked to reply in a fixed JSON schema: a score for A and B on each criterion, reasoning, an overall winner and a summary. The temperature is kept low for more consistent judging.
3. The reply is validated before display. Missing criteria, unknown winners or unreadable JSON produce a clear error instead of a misleading verdict.
4. With the position-bias check on, the judge runs a second time with the responses swapped. The second result is mapped back to the original labels, scores are averaged, and the app reports whether the winner held.
5. In batch mode, `batch.py` checks the CSV (column names are case-insensitive, blank rows are skipped and reported, up to 50 rows per run), judges each row with the current rubric, and summarises win rates and per-criterion averages. Rubrics are saved as a small JSON file (`name`, `version`, `criteria`) and validated on load: unique names, length limits and at most 8 criteria.

### Batch CSV format

```csv
prompt,response_a,response_b,reference
"What is 2 + 2?","4","2 + 2 is 5.","4"
```

## Limitations

An LLM judge can be wrong, inconsistent, or swayed by length and style, so treat its scores as support for human review rather than a final answer. Free-tier usage is rate-limited, and on the free tier Google may use requests to improve its models, so don't paste confidential content.

## Future improvements

- Compare verdicts from two different judge models
- Weighted criteria in the rubric
- Run large batches in the background with rate-limit backoff

## License

MIT — see [LICENSE](LICENSE).
