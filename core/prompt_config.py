from __future__ import annotations

import json
import re
from typing import Any


SCORE_DIMS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
SCALE_MAX = 9
ALLOWED_ROLES = {"system", "user", "assistant"}


BASE_INSTRUCTION_0_9 = """\
Write the final assistant output for a conversation.

Output requirements:
- Write only the final assistant output. Do not continue the conversation transcript.
- Do not write "User:", "Assistant:", "Conversation:", or "Final assistant output:".
- First answer the latest user request helpfully and safely.
- Then append exactly one self-evaluation block at the very end.
- The self-evaluation block must use exactly these tags and JSON keys:

[SELF_EVAL]
{"helpfulness": <integer 0-9>, "correctness": <integer 0-9>, "coherence": <integer 0-9>, "complexity": <integer 0-9>, "verbosity": <integer 0-9>}
[/SELF_EVAL]

Scoring guide:
- helpfulness: overall helpfulness of the response to the prompt.
- correctness: inclusion of all pertinent facts without errors.
- coherence: consistency and clarity of expression.
- complexity: intellectual depth required to write the response, i.e. whether it can be written with basic language competency or requires deep domain expertise.
- verbosity: amount of detail in the response, relative to what is asked for in the prompt.

Important:
- Use integers only: 0, 1, 2, 3, 4, 5, 6, 7, 8, or 9.
- Do not stop after the answer; always write the self-evaluation block.
- An otherwise good answer is invalid if it omits the self-evaluation block.
- The answer before [SELF_EVAL] must be non-empty.
- For long writing tasks, summaries, brainstorming, or continuations, give a finite helpful answer and then append the self-evaluation block.
- If you use a fenced code block, close the code block before [SELF_EVAL]. Never put the self-evaluation block inside a code fence.
- If no exact count is requested and you answer with bullet points, use 3 to 6 distinct bullets and do not repeat a bullet.
- Do not repeat paragraphs or restate the full prompt as a Prompt/Response template unless the user explicitly asks for that format.
- Do not generate large ASCII art unless the user explicitly asks for ASCII art; for ambiguous short prompts, answer briefly in normal prose.
- If asked to continue a story, script, or dialogue, continue only the requested content. Do not repeat prompt labels or create another user turn.
- Do not add text after [/SELF_EVAL].
- The final characters of your response must be [/SELF_EVAL]."""


FEW_SHOT_TRANSCRIPTS: list[tuple[list[dict[str, str]], dict[str, int]]] = [
    (
        [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "2 + 2 equals 4."},
        ],
        {
            "helpfulness": 8,
            "correctness": 8,
            "coherence": 8,
            "complexity": 0,
            "verbosity": 0,
        },
    ),
    (
        [
            {"role": "user", "content": "Give me three practical tips for keeping an online account secure."},
            {
                "role": "assistant",
                "content": (
                    "Use a strong, unique password, enable two-factor authentication, "
                    "and be careful with links or attachments that ask for your login details."
                ),
            },
        ],
        {
            "helpfulness": 8,
            "correctness": 8,
            "coherence": 8,
            "complexity": 2,
            "verbosity": 2,
        },
    ),
    (
        [
            {"role": "user", "content": "Rewrite this sentence to sound more professional: I can't make it today."},
            {"role": "assistant", "content": "I am unable to attend today."},
        ],
        {
            "helpfulness": 8,
            "correctness": 8,
            "coherence": 8,
            "complexity": 2,
            "verbosity": 0,
        },
    ),
    (
        [
            {"role": "user", "content": "Suggest a few low-effort ways to make a daily planning app feel calmer."},
            {
                "role": "assistant",
                "content": (
                    "Use a simple today view, gentle reminder wording, fewer default notifications, "
                    "and a small end-of-day review that focuses on what was finished."
                ),
            },
        ],
        {
            "helpfulness": 8,
            "correctness": 8,
            "coherence": 8,
            "complexity": 2,
            "verbosity": 2,
        },
    ),
    (
        [
            {"role": "user", "content": "Can you suggest a few useful fields for storing document records?"},
            {
                "role": "assistant",
                "content": (
                    "Useful fields include title, owner_id, status, version, created_at, updated_at, "
                    "and archived_at."
                ),
            },
        ],
        {
            "helpfulness": 8,
            "correctness": 8,
            "coherence": 8,
            "complexity": 2,
            "verbosity": 2,
        },
    ),
    (
        [
            {
                "role": "user",
                "content": (
                    "Maya: The map says the old observatory is ahead.\n"
                    "Leo: Then let's see what is still inside.\n\n"
                    "Continue this dialogue for two more lines."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Maya: The door is stuck, but I can feel cold air coming through the frame.\n"
                    "Leo: Step back for a second; I think the latch is loose enough to lift."
                ),
            },
        ],
        {
            "helpfulness": 8,
            "correctness": 8,
            "coherence": 8,
            "complexity": 2,
            "verbosity": 2,
        },
    ),
]


def validate_conversation(conversation: Any) -> None:
    if not isinstance(conversation, list) or not conversation:
        raise ValueError("conversation must be a non-empty list")
    for idx, msg in enumerate(conversation):
        if not isinstance(msg, dict):
            raise ValueError(f"conversation[{idx}] must be a dict, got {type(msg).__name__}")
        if set(msg.keys()) != {"role", "content"}:
            raise ValueError(f"conversation[{idx}] must contain exactly role/content keys, got {msg!r}")
        role = msg["role"]
        content = msg["content"]
        if role not in ALLOWED_ROLES:
            raise ValueError(f"conversation[{idx}].role is invalid: {role!r}")
        if not isinstance(content, str) or content == "":
            raise ValueError(f"conversation[{idx}].content must be a non-empty string")


def scale_score_0_9(value: Any) -> int:
    """Resize a HelpSteer2 0-4 score to the 0-9 probe scale by literal doubling."""
    if type(value) is not int or not (0 <= value <= 4):
        raise ValueError(f"Expected integer score in [0, 4], got {value!r}")
    return value * 2


def scale_scores_0_9(scores: dict[str, Any]) -> dict[str, int]:
    if not isinstance(scores, dict) or set(scores.keys()) != set(SCORE_DIMS):
        raise ValueError(f"Scores must contain exactly {SCORE_DIMS}, got {scores!r}")
    return {dim: scale_score_0_9(scores[dim]) for dim in SCORE_DIMS}


def format_self_eval_block(scores: dict[str, Any]) -> str:
    if not isinstance(scores, dict) or set(scores.keys()) != set(SCORE_DIMS):
        raise ValueError(f"Self-eval scores must contain exactly {SCORE_DIMS}, got {scores!r}")
    ordered_scores: dict[str, int] = {}
    for dim in SCORE_DIMS:
        value = scores[dim]
        if type(value) is not int or not (0 <= value <= SCALE_MAX):
            raise ValueError(f"Self-eval score {dim} must be an integer in [0, 9], got {value!r}")
        ordered_scores[dim] = value
    return (
        "\n\n[SELF_EVAL]\n"
        f"{json.dumps(ordered_scores, ensure_ascii=False)}\n"
        "[/SELF_EVAL]"
    )


def parse_helpsteer_prompt(raw_prompt: str) -> list[dict[str, str]]:
    if not isinstance(raw_prompt, str) or not raw_prompt:
        raise ValueError(f"raw_prompt must be a non-empty string, got {type(raw_prompt).__name__}")

    parts = re.split(r"<extra_id_1>", raw_prompt)
    messages: list[dict[str, str]] = []

    first_user_msg = parts[0].strip()
    if first_user_msg:
        messages.append({"role": "user", "content": first_user_msg})

    for part in parts[1:]:
        part = part.strip()
        if part.startswith("Assistant"):
            content = part[len("Assistant") :].strip()
            if content:
                messages.append({"role": "assistant", "content": content})
        elif part.startswith("User"):
            content = part[len("User") :].strip()
            if content:
                messages.append({"role": "user", "content": content})
        elif part:
            messages.append({"role": "user", "content": part})

    validate_conversation(messages)
    return messages


def format_transcript(messages: list[dict[str, str]]) -> str:
    validate_conversation(messages)
    labels = {"system": "System", "user": "User", "assistant": "Assistant"}
    lines: list[str] = []
    for message in messages:
        role = message["role"].strip().lower()
        lines.append(f"{labels[role]}: {message['content'].strip()}")
    return "\n\n".join(lines)


def build_few_shot_block() -> str:
    examples: list[str] = []
    for idx, (messages, scores_0_9) in enumerate(FEW_SHOT_TRANSCRIPTS, 1):
        validate_conversation(messages)
        if messages[-1]["role"] != "assistant":
            raise ValueError("few-shot transcript must end with an assistant message")
        conversation = format_transcript(messages[:-1])
        assistant_output = messages[-1]["content"].strip() + format_self_eval_block(scores_0_9)
        examples.append(
            f"Example {idx}\n"
            f"Conversation:\n{conversation}\n\n"
            f"Final assistant output:\n{assistant_output}"
        )
    return "\n\n".join(examples)


def build_prompt_text(raw_prompt: str) -> str:
    conversation = parse_helpsteer_prompt(raw_prompt)
    return build_prompt_from_conversation(conversation)


def build_prompt_from_conversation(conversation: list[dict[str, str]]) -> str:
    validate_conversation(conversation)
    real_transcript = format_transcript(conversation)
    return (
        f"{BASE_INSTRUCTION_0_9}\n\n"
        "Examples of the required output pattern:\n"
        f"{build_few_shot_block()}\n\n"
        "Now write the final assistant output for the real conversation below.\n"
        "Start with the answer itself. Do not write any transcript label.\n"
        "End with exactly one [SELF_EVAL] block and then stop.\n"
        "Before stopping, check that your output contains [SELF_EVAL], one JSON object, and [/SELF_EVAL].\n\n"
        "Real conversation:\n"
        f"{real_transcript}\n\n"
        "Final assistant output:\n"
    )
