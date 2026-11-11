"""Small wrapper around the Google Gemini API.

Sends a prompt, asks for JSON back, and turns every API or network problem
into an AIError whose message is safe to show to the user.
"""

import json
import logging
import os

import httpx  # Installed with google-genai; used to catch network errors.
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

logger = logging.getLogger(__name__)

# Gemini 3.1 Flash-Lite is on the Gemini API free tier.
# Set AI_MODEL in .env to use a different model without editing code.
DEFAULT_MODEL = "gemini-3.1-flash-lite"
REQUEST_TIMEOUT_MS = 90_000


class AIError(Exception):
    """An error with a message that is safe to show to the user."""


def get_api_key():
    """Read the API key from the environment, or raise a friendly error."""
    api_key = os.getenv("AI_API_KEY", "").strip()
    if not api_key or api_key == "your_api_key_here":
        raise AIError(
            "No API key found. Copy .env.example to .env, add your Gemini API "
            "key as AI_API_KEY, then restart the app."
        )
    return api_key


def get_model_name():
    return os.getenv("AI_MODEL", "").strip() or DEFAULT_MODEL


def generate_json(prompt, system_instruction, schema, temperature=0.2):
    """Call Gemini and return the reply parsed as a Python object."""
    api_key = get_api_key()
    model = get_model_name()
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        response_mime_type="application/json",
        response_json_schema=schema,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    try:
        response = client.models.generate_content(model=model, contents=prompt, config=config)
    except errors.ClientError as error:
        logger.warning("Gemini client error %s: %s", error.code, error.message)
        raise AIError(_client_error_message(error, model)) from error
    except errors.ServerError as error:
        logger.warning("Gemini server error %s: %s", error.code, error.message)
        raise AIError("The AI service is having trouble right now. Wait a minute and try again.") from error
    except errors.APIError as error:
        logger.warning("Gemini API error %s: %s", error.code, error.message)
        raise AIError(f"The AI service returned an unexpected error (code {error.code}). Try again.") from error
    except httpx.TimeoutException as error:
        logger.warning("Gemini request timed out: %s", error)
        raise AIError("The AI service took too long to answer. Try again in a moment.") from error
    except httpx.TransportError as error:
        logger.warning("Network error calling Gemini: %s", error)
        raise AIError("Couldn't reach the AI service. Check your internet connection and try again.") from error

    if not response.text:
        logger.warning("Gemini returned an empty reply (possibly blocked by safety filters)")
        raise AIError(
            "The AI returned an empty reply, which can happen if a request is "
            "blocked by its safety filters. Try different input."
        )
    return parse_json(response.text)


def _client_error_message(error, model):
    details = f"{error.status or ''} {error.message or ''}".lower()
    if error.code == 429:
        return "You've hit the API rate limit (common on the free tier). Wait a minute and try again."
    if error.code in (401, 403) or "api key" in details or "api_key" in details:
        return (
            "Your API key was rejected. Check that AI_API_KEY in your .env file "
            "is a valid Gemini API key, then restart the app."
        )
    if error.code == 404:
        return (
            f'The model "{model}" was not found. Set AI_MODEL in your .env file '
            "to a model your key can use, then restart the app."
        )
    return f"The AI service rejected the request (code {error.code})."


def parse_json(raw_text):
    """Parse JSON, tolerating Markdown fences or stray text around it."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    logger.warning("Unparseable AI reply (first 300 chars): %r", raw_text[:300])
    raise AIError("The AI's reply wasn't valid JSON, so it couldn't be read. Try again.")


def wrap_untrusted(tag, text):
    """Wrap user-supplied text in <tag>...</tag> so the model treats it as data.

    Any copies of the tag inside the text are removed, so the content can't
    close the wrapper early and smuggle in instructions.
    """
    cleaned = str(text).replace(f"<{tag}", "").replace(f"</{tag}>", "")
    return f"<{tag}>\n{cleaned.strip()}\n</{tag}>"
