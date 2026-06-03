"""0-9 nonlinear calibration reward for base-version SEE GRPO.

Scale convention:
- Model SELF_EVAL scores must be integer 0-9 values.
- Judge scores are requested as integer 0-9 values.
- HelpSteer2 reference labels in extra_info are original 0-4 annotations and
  are resized by literal doubling when used as judge calibration anchors.

Reward:
    quality = mean(judge helpfulness/correctness/coherence) / 9
    mae = mean(abs(self_score - judge_score))
    calibration_linear = 1 - mae / 9
    calibration = calibration_linear ** CALIBRATION_GAMMA
    score = W_QUALITY * quality + W_CALIBRATION * calibration
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from typing import Optional

import aiohttp


SCORE_DIMS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
QUALITY_DIMS = ["helpfulness", "correctness", "coherence"]
SCALE_MAX = 9
ALLOWED_ROLES = {"system", "user", "assistant"}

logger = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _require_float_env(
    name: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    value = _require_env(name)
    try:
        parsed = float(value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be a float, got {value!r}") from exc
    if min_value is not None and parsed < min_value:
        raise RuntimeError(f"Environment variable {name} must be >= {min_value}, got {parsed}")
    if max_value is not None and parsed > max_value:
        raise RuntimeError(f"Environment variable {name} must be <= {max_value}, got {parsed}")
    return parsed


def _require_int_env(
    name: str,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    value = _require_env(name)
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an int, got {value!r}") from exc
    if min_value is not None and parsed < min_value:
        raise RuntimeError(f"Environment variable {name} must be >= {min_value}, got {parsed}")
    if max_value is not None and parsed > max_value:
        raise RuntimeError(f"Environment variable {name} must be <= {max_value}, got {parsed}")
    return parsed


OPENAI_API_KEY = _require_env("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_MODEL = _require_env("OPENAI_MODEL")

W_QUALITY = _require_float_env("W_QUALITY", min_value=0.0)
W_CALIBRATION = _require_float_env("W_CALIBRATION", min_value=0.0)
FORMAT_PENALTY = _require_float_env("FORMAT_PENALTY")
CALIBRATION_GAMMA = _require_float_env("CALIBRATION_GAMMA", min_value=0.0)
MAX_RETRIES = _require_int_env("JUDGE_MAX_RETRIES", min_value=1)
BACKOFF_BASE = _require_float_env("JUDGE_BACKOFF_BASE", min_value=0.0)
BACKOFF_MAX = _require_float_env("JUDGE_BACKOFF_MAX", min_value=0.0)
JUDGE_TIMEOUT = _require_int_env("JUDGE_TIMEOUT", min_value=1)
JUDGE_MAX_CONCURRENT = _require_int_env("JUDGE_MAX_CONCURRENT", min_value=1)

if BACKOFF_MAX < BACKOFF_BASE:
    raise RuntimeError(
        f"JUDGE_BACKOFF_MAX must be >= JUDGE_BACKOFF_BASE, got {BACKOFF_MAX} < {BACKOFF_BASE}"
    )
if CALIBRATION_GAMMA <= 1.0:
    raise RuntimeError(
        f"CALIBRATION_GAMMA must be > 1.0 for nonlinear error amplification, got {CALIBRATION_GAMMA}"
    )
if W_QUALITY + W_CALIBRATION <= 0.0:
    raise RuntimeError("W_QUALITY + W_CALIBRATION must be positive")

_aio_session: Optional[aiohttp.ClientSession] = None
_aio_semaphore: Optional[asyncio.Semaphore] = None


async def _get_aio_session() -> aiohttp.ClientSession:
    global _aio_session
    if _aio_session is None or _aio_session.closed:
        _aio_session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            connector=aiohttp.TCPConnector(
                limit=JUDGE_MAX_CONCURRENT,
                limit_per_host=JUDGE_MAX_CONCURRENT,
            ),
            timeout=aiohttp.ClientTimeout(total=JUDGE_TIMEOUT),
        )
    return _aio_session


async def _get_semaphore() -> asyncio.Semaphore:
    global _aio_semaphore
    if _aio_semaphore is None:
        _aio_semaphore = asyncio.Semaphore(JUDGE_MAX_CONCURRENT)
    return _aio_semaphore


JUDGE_SYSTEM = """\
You are an expert evaluator of AI assistant responses. Rate the response on five dimensions using integer scores from 0 to 9.

Dimensions:
- helpfulness (0-9): overall helpfulness of the response to the prompt.
- correctness (0-9): inclusion of all pertinent facts without errors.
- coherence (0-9): consistency and clarity of expression.
- complexity (0-9): intellectual depth required to write the response, i.e. whether it can be written with basic language competency or requires deep domain expertise.
- verbosity (0-9): amount of detail in the response, relative to what is asked for in the prompt.

Important rating guidance:
- Higher is not always better for every attribute. In particular, complexity and verbosity are descriptive properties, not always targets to maximize.
- For correctness, missing important facts should lower the score even if the response contains no obvious false statement.
- For verbosity, judge the amount of detail relative to the user's request, not whether the answer is good or bad overall.
- For multi-turn conversations, judge the final answer with respect to the latest user turn while considering the conversation history as context.
- Reference response labels use a resized 0-9 scale obtained by doubling the original 0-4 human annotations.

When reference responses with human-annotated scores are provided, use them as calibration anchors to understand the scoring scale for this specific prompt. Compare the target response against these references to produce well-calibrated scores.

Output ONLY a JSON object with five integer scores, nothing else."""


def _scale_ref_score(value: object) -> int:
    if type(value) is not int or not (0 <= value <= 4):
        raise ValueError(f"Reference score must be an integer in [0, 4], got {value!r}")
    return value * 2


def _scale_ref_scores(scores: dict) -> dict[str, int]:
    if not isinstance(scores, dict) or set(scores.keys()) != set(SCORE_DIMS):
        raise ValueError(f"Reference scores must contain exactly {SCORE_DIMS}, got {scores!r}")
    return {dim: _scale_ref_score(scores[dim]) for dim in SCORE_DIMS}


def _validate_conversation(conversation: object) -> list[dict[str, str]]:
    if not isinstance(conversation, list) or not conversation:
        raise ValueError(f"conversation_json must decode to a non-empty list, got {conversation!r}")
    validated = []
    for idx, msg in enumerate(conversation):
        if not isinstance(msg, dict) or set(msg.keys()) != {"role", "content"}:
            raise ValueError(f"Malformed conversation message at index {idx}: {msg!r}")
        role = msg["role"]
        content = msg["content"]
        if role not in ALLOWED_ROLES:
            raise ValueError(f"Invalid conversation role at index {idx}: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Conversation content at index {idx} must be a non-empty string")
        validated.append({"role": role, "content": content})
    return validated


def _validate_ref_responses(ref_responses: object) -> list[dict]:
    if not isinstance(ref_responses, list) or not ref_responses:
        raise ValueError(f"ref_responses_json must decode to a non-empty list, got {ref_responses!r}")
    for idx, ref in enumerate(ref_responses):
        if not isinstance(ref, dict) or set(ref.keys()) != {"response", "scores"}:
            raise ValueError(f"Malformed reference response at index {idx}: {ref!r}")
        response = ref["response"]
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"Reference response at index {idx} must be a non-empty string")
        _scale_ref_scores(ref["scores"])
    return ref_responses


def _format_ref_block(ref_responses: list[dict]) -> str:
    ref_responses = _validate_ref_responses(ref_responses)

    parts = []
    for i, ref in enumerate(ref_responses, 1):
        scaled_scores = _scale_ref_scores(ref["scores"])
        parts.append(
            f"--- Reference Response {i} ---\n"
            f"{ref['response']}\n"
            f"Human-Annotated Scores on resized 0-9 scale: "
            f"{json.dumps(scaled_scores, ensure_ascii=False)}"
        )

    return (
        "Below are reference responses to the same prompt, each with human-annotated scores. "
        "Use them as calibration anchors.\n\n"
        + "\n\n".join(parts)
        + "\n\n"
    )


def _format_judge_input(
    conversation: list[dict],
    response_text: str,
    ref_responses: list[dict],
) -> str:
    conversation = _validate_conversation(conversation)
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("response_text must be a non-empty string")

    parts = []
    for msg in conversation:
        role = msg["role"].capitalize()
        parts.append(f"[{role}]\n{msg['content']}")
    conv_block = "\n\n".join(parts)

    return (
        f"{conv_block}\n\n"
        f"{_format_ref_block(ref_responses)}"
        f"[AI Assistant's Response to Evaluate]\n{response_text}\n\n"
        "Rate the response on the resized 0-9 scale. Output ONLY JSON:\n"
        '{"helpfulness": <int>, "correctness": <int>, "coherence": <int>, '
        '"complexity": <int>, "verbosity": <int>}'
    )


def _backoff_delay(attempt: int) -> float:
    delay = min(BACKOFF_BASE * (2 ** min(attempt, 30)), BACKOFF_MAX)
    return delay * (0.5 + 0.5 * random.random())


class JudgeAPIError(RuntimeError):
    pass


async def _read_chat_completion_json(resp: aiohttp.ClientResponse) -> dict:
    text = await resp.text()
    content_type = resp.headers.get("Content-Type", "")
    if "text/event-stream" not in content_type and not text.lstrip().startswith("data:"):
        try:
            body = json.loads(text)
        except json.JSONDecodeError as exc:
            raise JudgeAPIError(f"Judge API returned invalid JSON body: {text[:200]}") from exc
        if not isinstance(body, dict):
            raise JudgeAPIError(f"Judge API JSON body must be an object, got {type(body).__name__}")
        return body

    chunks: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError as exc:
            raise JudgeAPIError(f"Judge API returned malformed SSE event: {data[:200]}") from exc
        if not isinstance(event, dict):
            raise JudgeAPIError(f"Judge API SSE event must be an object, got {event!r}")
        choices = event.get("choices")
        if not isinstance(choices, list):
            raise JudgeAPIError(f"Judge API SSE event missing choices list: {event!r}")
        if not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            raise JudgeAPIError(f"Judge API SSE choice must be an object, got {choice!r}")

        content = None
        delta = choice.get("delta")
        if delta is not None:
            if not isinstance(delta, dict):
                raise JudgeAPIError(f"Judge API SSE delta must be an object, got {delta!r}")
            if "content" in delta:
                content = delta["content"]
        message = choice.get("message")
        if content is None and message is not None:
            if not isinstance(message, dict):
                raise JudgeAPIError(f"Judge API SSE message must be an object, got {message!r}")
            if "content" in message:
                content = message["content"]
        if content is None:
            continue
        if not isinstance(content, str):
            raise JudgeAPIError(f"Judge API SSE content must be a string, got {content!r}")
        if content:
            chunks.append(content)

    if not chunks:
        raise JudgeAPIError(f"empty SSE response: {text[:200]}")
    return {"choices": [{"message": {"content": "".join(chunks)}}]}


def _extract_chat_content(body: object) -> str:
    if not isinstance(body, dict):
        raise JudgeAPIError(f"Judge response body must be an object, got {type(body).__name__}")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise JudgeAPIError(f"Judge response must contain a non-empty choices list: {body!r}")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise JudgeAPIError(f"Judge response choice must be an object, got {choice!r}")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise JudgeAPIError(f"Judge response choice.message must be an object, got {choice!r}")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise JudgeAPIError(f"Judge response message.content must be a non-empty string, got {content!r}")
    return content.strip()


def _validate_scores(
    scores: object,
    *,
    require_order: bool = False,
    require_exact_keys: bool = False,
) -> Optional[dict[str, int]]:
    if not isinstance(scores, dict):
        return None
    if require_order and list(scores.keys()) != SCORE_DIMS:
        return None
    if require_exact_keys and set(scores.keys()) != set(SCORE_DIMS):
        return None

    validated = {}
    for dim in SCORE_DIMS:
        value = scores.get(dim)
        if type(value) is not int or not (0 <= value <= SCALE_MAX):
            return None
        validated[dim] = value
    return validated


def _parse_judge_scores(content: str) -> Optional[dict[str, int]]:
    stripped = content.strip()
    if not stripped:
        return None
    try:
        scores = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return _validate_scores(scores, require_exact_keys=True)


async def _call_judge(
    conversation: list[dict],
    response_text: str,
    ref_responses: list[dict],
) -> dict[str, int]:
    session = await _get_aio_session()
    semaphore = await _get_semaphore()
    user_prompt = _format_judge_input(conversation, response_text, ref_responses)

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 128,
        "stream": True,
    }
    url = f"{OPENAI_BASE_URL}/chat/completions"
    last_error = "unknown"
    attempts = 0
    api_failures = 0
    format_failures = 0

    while True:
        try:
            async with semaphore:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        last_error = f"HTTP {resp.status}: {body[:200]}"
                        attempts += 1
                        api_failures += 1
                        logger.warning("Judge API attempt %d/%d %s", attempts, MAX_RETRIES, last_error)
                        if attempts >= MAX_RETRIES:
                            break
                        await asyncio.sleep(_backoff_delay(attempts - 1))
                        continue
                    body = await _read_chat_completion_json(resp)

            content = _extract_chat_content(body)
            validated = _parse_judge_scores(content)
            if validated is None:
                last_error = f"malformed judge JSON or score schema: {content[:200]}"
                attempts += 1
                format_failures += 1
                logger.warning("Judge returned %s", last_error)
                if attempts >= MAX_RETRIES:
                    break
                await asyncio.sleep(_backoff_delay(attempts - 1))
                continue

            return validated
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_error = str(exc)
            attempts += 1
            api_failures += 1
            logger.warning("Judge API attempt %d/%d failed: %s", attempts, MAX_RETRIES, last_error)
            if attempts >= MAX_RETRIES:
                break

        await asyncio.sleep(_backoff_delay(attempts - 1))
    raise JudgeAPIError(f"Judge failed after {MAX_RETRIES} retries. Last error: {last_error}")


def _parse_strict_self_eval(text: str) -> Optional[tuple[str, dict[str, int]]]:
    open_tags = re.findall(r"\[SELF_EVAL\]", text, flags=re.IGNORECASE)
    close_tags = re.findall(r"\[/SELF_EVAL\]", text, flags=re.IGNORECASE)
    if len(open_tags) != 1 or len(close_tags) != 1:
        return None

    match = re.search(
        r"(?s)^(?P<answer>.*?)(?:\s*)\[SELF_EVAL\]\s*(?P<json>\{[^{}]*\})\s*\[/SELF_EVAL\]\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None

    answer_text = match.group("answer").strip()
    if not answer_text:
        return None

    try:
        scores = json.loads(match.group("json"))
    except json.JSONDecodeError:
        return None

    validated = _validate_scores(scores, require_order=True)
    if validated is None:
        return None
    return answer_text, validated


def _format_failure_result() -> dict:
    result = {
        "score": FORMAT_PENALTY,
        "quality": 0.0,
        "calibration": 0.0,
        "calibration_linear": 0.0,
        "calibration_gamma": CALIBRATION_GAMMA,
        "mae": 0.0,
        "format_ok": 0.0,
        "judge_failed": 0.0,
    }
    for dim in SCORE_DIMS:
        result[f"judge_{dim}"] = 0.0
        result[f"self_{dim}"] = 0.0
    return result


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
) -> dict:
    if not isinstance(extra_info, dict):
        raise ValueError(f"extra_info must be a dict, got {type(extra_info).__name__}")

    if "conversation_json" not in extra_info:
        raise ValueError("extra_info is missing required field conversation_json")
    conv_json = extra_info["conversation_json"]
    try:
        conversation = json.loads(conv_json) if isinstance(conv_json, str) else conv_json
    except (json.JSONDecodeError, TypeError):
        raise ValueError(f"Invalid conversation_json: {conv_json!r}")
    conversation = _validate_conversation(conversation)

    if "ref_responses_json" not in extra_info:
        raise ValueError("extra_info is missing required field ref_responses_json")
    ref_json = extra_info["ref_responses_json"]
    try:
        ref_responses = json.loads(ref_json) if isinstance(ref_json, str) else ref_json
    except (json.JSONDecodeError, TypeError):
        raise ValueError(f"Invalid ref_responses_json: {ref_json!r}")
    ref_responses = _validate_ref_responses(ref_responses)

    parsed_self_eval = _parse_strict_self_eval(solution_str)
    if parsed_self_eval is None:
        return _format_failure_result()

    answer_text, self_scores = parsed_self_eval

    judge_scores = await _call_judge(conversation, answer_text, ref_responses)

    quality = sum(judge_scores[d] for d in QUALITY_DIMS) / (len(QUALITY_DIMS) * SCALE_MAX)
    abs_errors = [abs(judge_scores[d] - self_scores[d]) for d in SCORE_DIMS]
    mae = sum(abs_errors) / len(SCORE_DIMS)
    error_norm = mae / SCALE_MAX
    if not (0.0 <= error_norm <= 1.0):
        raise RuntimeError(f"Normalized calibration error out of range: {error_norm}")
    calibration_linear = 1.0 - error_norm
    calibration = calibration_linear ** CALIBRATION_GAMMA
    reward = W_QUALITY * quality + W_CALIBRATION * calibration

    result = {
        "score": reward,
        "quality": round(quality, 4),
        "calibration": round(calibration, 4),
        "calibration_linear": round(calibration_linear, 4),
        "calibration_gamma": CALIBRATION_GAMMA,
        "mae": round(mae, 4),
        "format_ok": 1.0,
        "judge_failed": 0.0,
    }
    for dim in SCORE_DIMS:
        result[f"judge_{dim}"] = float(judge_scores[dim])
        result[f"self_{dim}"] = float(self_scores[dim])
    return result
