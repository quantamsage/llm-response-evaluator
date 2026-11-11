"""Save and load scoring rubrics (sets of criteria) as JSON."""

import json

from judge import DEFAULT_CRITERIA, MAX_CRITERIA

RUBRIC_VERSION = 1
MAX_NAME_CHARS = 60
MAX_DESCRIPTION_CHARS = 300

PRESETS = {
    "General (default)": DEFAULT_CRITERIA,
    "Code answers": {
        "Correctness": "The code runs and does what was asked. No invented APIs or functions.",
        "Explanation": "Explains what the code does and why, at the level the user needs.",
        "Best practice": "Readable, idiomatic, handles obvious edge cases and errors.",
        "Instruction following": "Uses the language, libraries and constraints the prompt asked for.",
    },
    "Customer support": {
        "Resolution": "Actually solves the customer's issue or gives a clear next step.",
        "Accuracy": "Policies, steps and facts are correct. Nothing promised that can't be delivered.",
        "Tone": "Polite, calm and empathetic without being wordy.",
        "Conciseness": "Gets to the point; no filler or repeated apologies.",
    },
}


class RubricError(Exception):
    """Raised when a rubric file can't be used. The message is safe to show to users."""


def clean_criteria(pairs):
    """Validate (name, description) pairs and return an ordered criteria dict."""
    criteria = {}
    seen = set()
    for name, description in pairs:
        name = str(name or "").strip()
        description = str(description or "").strip()
        if not name:
            continue
        if len(name) > MAX_NAME_CHARS:
            raise RubricError(f'Criterion name "{name[:30]}..." is longer than {MAX_NAME_CHARS} characters.')
        if len(description) > MAX_DESCRIPTION_CHARS:
            raise RubricError(f'The description for "{name}" is longer than {MAX_DESCRIPTION_CHARS} characters.')
        if name.casefold() in seen:
            raise RubricError(f'"{name}" appears more than once. Criterion names must be unique.')
        seen.add(name.casefold())
        criteria[name] = description or name
    if len(criteria) > MAX_CRITERIA:
        raise RubricError(f"A rubric can have at most {MAX_CRITERIA} criteria; this one has {len(criteria)}.")
    return criteria


def rubric_to_json(name, criteria):
    """Serialise a rubric for download."""
    return json.dumps(
        {
            "name": name.strip() or "My rubric",
            "version": RUBRIC_VERSION,
            "criteria": [{"name": n, "description": d} for n, d in criteria.items()],
        },
        indent=2,
    )


def rubric_from_json(text):
    """Parse an uploaded rubric. Returns (name, criteria dict)."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        raise RubricError("That file isn't valid JSON. Upload a rubric saved from this app.")
    if not isinstance(data, dict) or not isinstance(data.get("criteria"), list):
        raise RubricError('A rubric file needs a "criteria" list. Upload a rubric saved from this app.')

    pairs = []
    for item in data["criteria"]:
        if not isinstance(item, dict):
            raise RubricError("Each criterion must have a name and a description.")
        pairs.append((item.get("name"), item.get("description")))
    criteria = clean_criteria(pairs)
    if not criteria:
        raise RubricError("This rubric has no criteria.")
    return str(data.get("name") or "Uploaded rubric").strip()[:MAX_NAME_CHARS], criteria
