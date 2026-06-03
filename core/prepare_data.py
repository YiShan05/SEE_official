#!/usr/bin/env python3
"""Prepare 0-9 base-version RL parquet files directly from HelpSteer2.

Input files are the original HelpSteer2 jsonl splits. Rows sharing the same
prompt are grouped into one RL item so the reward function can use all annotated
responses for judge calibration references.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from prompt_config import (  # noqa: E402
    SCORE_DIMS,
    build_prompt_from_conversation,
    parse_helpsteer_prompt,
    scale_scores_0_9,
    validate_conversation,
)


DEFAULT_HELPSTEER2_DIR = Path("/path/to/HelpSteer2")
DEFAULT_TRAIN_PATH = DEFAULT_HELPSTEER2_DIR / "train.jsonl"
DEFAULT_VAL_PATH = DEFAULT_HELPSTEER2_DIR / "validation.jsonl"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "rl"
RANDOM_SEED = 42


def normalize_obj(value: Any) -> Any:
    if hasattr(value, "item") and callable(value.item):
        try:
            return normalize_obj(value.item())
        except ValueError:
            if not (hasattr(value, "tolist") and callable(value.tolist)):
                raise
    if hasattr(value, "tolist") and callable(value.tolist):
        return normalize_obj(value.tolist())
    if isinstance(value, list):
        return [normalize_obj(v) for v in value]
    if isinstance(value, tuple):
        return [normalize_obj(v) for v in value]
    if isinstance(value, dict):
        return {str(k): normalize_obj(v) for k, v in value.items()}
    return value


def validate_score_dict(scores: dict[str, Any], *, context: str) -> dict[str, int]:
    if not isinstance(scores, dict) or set(scores.keys()) != set(SCORE_DIMS):
        raise ValueError(f"{context} scores must contain exactly {SCORE_DIMS}, got {scores!r}")
    validated: dict[str, int] = {}
    for dim in SCORE_DIMS:
        value = scores[dim]
        if type(value) is not int or not (0 <= value <= 4):
            raise ValueError(f"{context}.{dim} must be an integer in [0, 4], got {value!r}")
        validated[dim] = value
    return validated


def validate_helpsteer_row(row: dict[str, Any], *, row_idx: int, path: Path) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"{path}:{row_idx} must decode to a dict, got {type(row).__name__}")
    required = {"prompt", "response", *SCORE_DIMS}
    missing = sorted(required - set(row.keys()))
    if missing:
        raise ValueError(f"{path}:{row_idx} missing required keys: {missing}")
    extra = sorted(set(row.keys()) - required)
    if extra:
        raise ValueError(f"{path}:{row_idx} has unexpected keys: {extra}")
    if not isinstance(row["prompt"], str) or row["prompt"] == "":
        raise ValueError(f"{path}:{row_idx}.prompt must be a non-empty string")
    if not isinstance(row["response"], str) or row["response"] == "":
        raise ValueError(f"{path}:{row_idx}.response must be a non-empty string")
    scores = validate_score_dict({dim: row[dim] for dim in SCORE_DIMS}, context=f"{path}:{row_idx}")
    return {
        "prompt": row["prompt"],
        "response": row["response"],
        **scores,
        "_row_idx": row_idx,
    }


def load_helpsteer_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing HelpSteer2 jsonl: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{row_idx} is not valid JSON") from exc
            rows.append(validate_helpsteer_row(raw, row_idx=row_idx, path=path))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def group_by_prompt(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        groups.setdefault(row["prompt"], []).append(row)
    grouped = list(groups.values())
    if not grouped:
        raise ValueError("No prompt groups were produced")
    return grouped


def select_group_indices(total: int, max_samples: int, seed: int) -> list[int]:
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if max_samples < 0 or max_samples >= total:
        return list(range(total))
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    return sorted(indices[:max_samples])


def build_ref_responses(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for sample in samples:
        scores = validate_score_dict({dim: sample[dim] for dim in SCORE_DIMS}, context="reference")
        refs.append({"response": sample["response"], "scores": scores})
    if not refs:
        raise ValueError("Each prompt group must contain at least one reference response")
    return refs


def last_user_preview(conversation: list[dict[str, str]]) -> str:
    for message in reversed(conversation):
        if message["role"] == "user":
            return message["content"][:200]
    return ""


def convert_group(samples: list[dict[str, Any]], prompt_idx: int, split_name: str) -> dict[str, Any]:
    if not samples:
        raise ValueError(f"{split_name}:{prompt_idx} group is empty")

    raw_prompt = samples[0]["prompt"]
    if any(sample["prompt"] != raw_prompt for sample in samples):
        raise ValueError(f"{split_name}:{prompt_idx} group contains mixed prompts")

    conversation = parse_helpsteer_prompt(raw_prompt)
    validate_conversation(conversation)

    original_scores = validate_score_dict(
        {dim: samples[0][dim] for dim in SCORE_DIMS},
        context=f"{split_name}:{prompt_idx}.primary",
    )
    scaled_scores = scale_scores_0_9(original_scores)
    ref_responses = build_ref_responses(samples)

    extra_info = {
        "index": str(prompt_idx),
        "split": split_name,
        "source": "HelpSteer2",
        "score_scale": "0_9",
        "original_score_scale": "0_4",
        "conversation_json": json.dumps(conversation, ensure_ascii=False),
        "original_scores_json": json.dumps(original_scores, ensure_ascii=False),
        "scaled_scores_0_9_json": json.dumps(scaled_scores, ensure_ascii=False),
        "ref_responses_json": json.dumps(ref_responses, ensure_ascii=False),
        "source_row_indices_json": json.dumps([sample["_row_idx"] for sample in samples]),
        "prompt_preview": last_user_preview(conversation),
    }

    return {
        "data_source": "helpsteer2_see_base_0_9",
        "prompt": build_prompt_from_conversation(conversation),
        "ability": "self_eval_base_completion_0_9",
        "reward_model": {
            "style": "llm_judge_0_9",
            "ground_truth": json.dumps(scaled_scores, ensure_ascii=False),
        },
        "extra_info": extra_info,
    }


def convert_split(input_path: Path, output_path: Path, split_name: str, max_samples: int, seed: int) -> None:
    rows = load_helpsteer_jsonl(input_path)
    grouped = group_by_prompt(rows)
    selected_indices = select_group_indices(len(grouped), max_samples, seed)

    converted_rows = [
        normalize_obj(convert_group(grouped[prompt_idx], prompt_idx, split_name))
        for prompt_idx in selected_indices
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(converted_rows).to_parquet(output_path, index=False)
    print(
        f"[{split_name}] rows={len(rows)} unique_prompts={len(grouped)} "
        f"selected={len(converted_rows)} -> {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 0-9 base-version RL parquet files from HelpSteer2 jsonl.")
    parser.add_argument("--train_path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val_path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train_max_samples", type=int, default=-1, help="-1 keeps all unique train prompts.")
    parser.add_argument("--val_max_samples", type=int, default=-1, help="Number of unique validation prompts to sample; -1 keeps all.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    convert_split(
        args.train_path,
        args.output_dir / "train.parquet",
        "train",
        args.train_max_samples,
        args.seed,
    )
    convert_split(
        args.val_path,
        args.output_dir / "val.parquet",
        "val",
        args.val_max_samples,
        args.seed,
    )


if __name__ == "__main__":
    main()
