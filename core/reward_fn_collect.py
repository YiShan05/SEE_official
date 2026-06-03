"""Collecting variant of the strict 0-9 nonlinear reward.

This module keeps the same reward semantics as reward_fn.py, but
appends one JSONL record per rollout to ROLLOUT_LOG_PATH. It is intended for the
first closed-loop experiment:

    short RL -> collect model answers and judge scores -> score-token SFT.

Judge failures still raise after the retry limit. No fallback judge score is
ever returned or logged as a valid reward.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from core.reward_fn import (
    CALIBRATION_GAMMA,
    FORMAT_PENALTY,
    QUALITY_DIMS,
    SCALE_MAX,
    SCORE_DIMS,
    W_CALIBRATION,
    W_QUALITY,
    _call_judge,
    _format_failure_result,
    _parse_strict_self_eval,
    _validate_conversation,
    _validate_ref_responses,
)


ROLLOUT_LOG_PATH = os.environ.get("ROLLOUT_LOG_PATH", "")
ROLLOUT_LOG_FORMAT_FAILURES = os.environ.get("ROLLOUT_LOG_FORMAT_FAILURES", "1") == "1"

if not ROLLOUT_LOG_PATH:
    raise RuntimeError(
        "reward_fn_collect.py requires ROLLOUT_LOG_PATH. "
        "Use the normal reward_fn.py when rollout collection is not needed."
    )


def _json_load_field(extra_info: dict[str, Any], key: str) -> Any:
    if key not in extra_info:
        raise ValueError(f"extra_info is missing required field {key}")
    value = extra_info[key]
    try:
        return json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Invalid {key}: {value!r}") from exc


def _append_jsonl(record: dict[str, Any]) -> None:
    path = Path(ROLLOUT_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            handle.write(line + "\n")
            handle.flush()
            raise


def _base_record(
    *,
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
    conversation: list[dict[str, str]],
    ref_responses: list[dict[str, Any]],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp": time.time(),
        "data_source": data_source,
        "ground_truth": ground_truth,
        "solution_str": solution_str,
        "solution_chars": len(solution_str) if isinstance(solution_str, str) else None,
        "extra_info": {
            "index": extra_info.get("index"),
            "split": extra_info.get("split"),
            "source": extra_info.get("source"),
            "score_scale": extra_info.get("score_scale"),
            "prompt_preview": extra_info.get("prompt_preview"),
            "source_row_indices_json": extra_info.get("source_row_indices_json"),
        },
        "conversation": conversation,
        "ref_response_count": len(ref_responses),
        "reward_kwargs_keys": sorted(kwargs.keys()),
    }


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
) -> dict:
    if not isinstance(extra_info, dict):
        raise ValueError(f"extra_info must be a dict, got {type(extra_info).__name__}")

    conversation = _validate_conversation(_json_load_field(extra_info, "conversation_json"))
    ref_responses = _validate_ref_responses(_json_load_field(extra_info, "ref_responses_json"))

    record = _base_record(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        conversation=conversation,
        ref_responses=ref_responses,
        kwargs=kwargs,
    )

    parsed_self_eval = _parse_strict_self_eval(solution_str)
    if parsed_self_eval is None:
        result = _format_failure_result()
        if ROLLOUT_LOG_FORMAT_FAILURES:
            record.update(
                {
                    "format_ok": False,
                    "answer_text": None,
                    "self_scores": None,
                    "judge_scores": None,
                    "reward": result,
                    "collection_status": "format_invalid",
                }
            )
            _append_jsonl(record)
        return result

    answer_text, self_scores = parsed_self_eval

    # This call raises JudgeAPIError after the configured retry limit. That is
    # intentional: dirty judge outputs must not be converted into training
    # rewards or calibration-SFT labels.
    judge_scores = await _call_judge(conversation, answer_text, ref_responses)

    quality = sum(judge_scores[d] for d in QUALITY_DIMS) / (len(QUALITY_DIMS) * SCALE_MAX)
    abs_errors = [abs(judge_scores[d] - self_scores[d]) for d in SCORE_DIMS]
    mae = sum(abs_errors) / len(SCORE_DIMS)
    error_norm = mae / SCALE_MAX
    if not (0.0 <= error_norm <= 1.0):
        raise RuntimeError(f"Normalized calibration error out of range: {error_norm}")
    calibration_linear = 1.0 - error_norm
    calibration = calibration_linear**CALIBRATION_GAMMA
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

    record.update(
        {
            "format_ok": True,
            "answer_text": answer_text,
            "self_scores": self_scores,
            "judge_scores": judge_scores,
            "abs_errors": {dim: abs(judge_scores[dim] - self_scores[dim]) for dim in SCORE_DIMS},
            "reward": result,
            "collection_status": "judge_ok",
        }
    )
    _append_jsonl(record)
    return result
