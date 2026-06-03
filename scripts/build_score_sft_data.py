#!/usr/bin/env python3
"""Build score-token SFT data from collected strict RL rollouts."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from core.prompt_config import (  # noqa: E402
    SCORE_DIMS,
    build_prompt_from_conversation,
    format_self_eval_block,
)


BINS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} must decode to an object")
            rows.append(obj)
    return rows


def validate_scores(scores: Any, *, context: str) -> dict[str, int]:
    if not isinstance(scores, dict) or set(scores.keys()) != set(SCORE_DIMS):
        raise ValueError(f"{context} scores must contain exactly {SCORE_DIMS}, got {scores!r}")
    validated: dict[str, int] = {}
    for dim in SCORE_DIMS:
        value = scores[dim]
        if type(value) is not int or not (0 <= value <= 9):
            raise ValueError(f"{context}.{dim} must be an integer in [0, 9], got {value!r}")
        validated[dim] = value
    return validated


def score_bin(value: int) -> str:
    for low, high in BINS:
        if low <= value <= high:
            return f"{low}_{high}"
    raise ValueError(f"score out of range: {value}")


def score_block_digit_positions(score_block: str, scores: dict[str, int]) -> list[dict[str, Any]]:
    positions: list[dict[str, Any]] = []
    for dim in SCORE_DIMS:
        pattern = rf'"{re.escape(dim)}":\s*{scores[dim]}'
        match = re.search(pattern, score_block)
        if match is None:
            raise ValueError(f"Could not locate {dim}={scores[dim]} in score block: {score_block!r}")
        digit_text = str(scores[dim])
        digit_start = score_block.index(digit_text, match.start())
        positions.append(
            {
                "dim": dim,
                "value": scores[dim],
                "bin": score_bin(scores[dim]),
                "start": digit_start,
                "end": digit_start + len(digit_text),
            }
        )
    return positions


def convert_record(record: dict[str, Any]) -> dict[str, Any] | None:
    if record.get("collection_status") != "judge_ok":
        return None
    if record.get("format_ok") is not True:
        return None
    answer_text = record.get("answer_text")
    if not isinstance(answer_text, str) or not answer_text.strip():
        return None
    judge_scores = validate_scores(record.get("judge_scores"), context="judge")
    self_scores = validate_scores(record.get("self_scores"), context="self")
    abs_errors = {dim: abs(self_scores[dim] - judge_scores[dim]) for dim in SCORE_DIMS}
    self_judge_mae = sum(abs_errors.values()) / len(SCORE_DIMS)

    conversation = record.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        raise ValueError("collected record is missing a non-empty conversation")

    prompt = build_prompt_from_conversation(conversation)
    answer = answer_text.strip()
    score_block = format_self_eval_block(judge_scores)
    text = prompt + answer + score_block
    offset = len(prompt) + len(answer)
    digit_positions = score_block_digit_positions(score_block, judge_scores)
    for item in digit_positions:
        item["start"] += offset
        item["end"] += offset

    extra = record.get("extra_info") if isinstance(record.get("extra_info"), dict) else {}
    reward = record.get("reward") if isinstance(record.get("reward"), dict) else {}
    return {
        "text": text,
        "prompt": prompt,
        "answer_text": answer,
        "score_block": score_block,
        "score_digit_positions": digit_positions,
        "judge_scores": judge_scores,
        "self_scores_before_sft": self_scores,
        "source_rollout": {
            "data_source": record.get("data_source"),
            "index": extra.get("index"),
            "split": extra.get("split"),
            "source": extra.get("source"),
            "prompt_preview": extra.get("prompt_preview"),
            "collection_status": record.get("collection_status"),
            "reward": reward,
        },
        "quality": reward.get("quality"),
        "calibration": reward.get("calibration"),
        "calibration_linear": reward.get("calibration_linear"),
        "mae": self_judge_mae,
        "self_judge_abs_errors": abs_errors,
        "self_judge_max_abs_error": max(abs_errors.values()),
    }


def choose_balanced(
    rows: list[dict[str, Any]],
    *,
    max_samples: int,
    per_prompt_limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    if max_samples <= 0 or max_samples >= len(rows):
        return rows

    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        for dim in SCORE_DIMS:
            buckets[(dim, score_bin(row["judge_scores"][dim]))].append(idx)

    for indices in buckets.values():
        rng.shuffle(indices)

    selected: list[int] = []
    selected_set: set[int] = set()
    prompt_counts: Counter[str] = Counter()
    keys = [(dim, f"{low}_{high}") for low, high in BINS for dim in SCORE_DIMS]

    while len(selected) < max_samples:
        made_progress = False
        rng.shuffle(keys)
        for key in keys:
            bucket = buckets.get(key, [])
            while bucket:
                idx = bucket.pop()
                if idx in selected_set:
                    continue
                prompt_id = str(rows[idx]["source_rollout"].get("index"))
                if per_prompt_limit > 0 and prompt_counts[prompt_id] >= per_prompt_limit:
                    continue
                selected.append(idx)
                selected_set.add(idx)
                prompt_counts[prompt_id] += 1
                made_progress = True
                break
            if len(selected) >= max_samples:
                break
        if not made_progress:
            break

    if len(selected) < max_samples:
        remaining = [idx for idx in range(len(rows)) if idx not in selected_set]
        rng.shuffle(remaining)
        for idx in remaining:
            prompt_id = str(rows[idx]["source_rollout"].get("index"))
            if per_prompt_limit > 0 and prompt_counts[prompt_id] >= per_prompt_limit:
                continue
            selected.append(idx)
            selected_set.add(idx)
            prompt_counts[prompt_id] += 1
            if len(selected) >= max_samples:
                break

    return [rows[idx] for idx in selected]


def choose_largest_mae(
    rows: list[dict[str, Any]],
    *,
    max_samples: int,
    per_prompt_limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    if max_samples <= 0 or max_samples >= len(rows):
        return rows

    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    order.sort(
        key=lambda idx: (
            float(rows[idx].get("mae") or 0.0),
            int(rows[idx].get("self_judge_max_abs_error") or 0),
        ),
        reverse=True,
    )

    selected: list[int] = []
    selected_set: set[int] = set()
    prompt_counts: Counter[str] = Counter()
    for idx in order:
        prompt_id = str(rows[idx]["source_rollout"].get("index"))
        if per_prompt_limit > 0 and prompt_counts[prompt_id] >= per_prompt_limit:
            continue
        selected.append(idx)
        selected_set.add(idx)
        prompt_counts[prompt_id] += 1
        if len(selected) >= max_samples:
            break

    # Always fill the requested sample count if enough usable rollouts exist.
    if len(selected) < max_samples:
        for idx in order:
            if idx in selected_set:
                continue
            selected.append(idx)
            selected_set.add(idx)
            if len(selected) >= max_samples:
                break

    return [rows[idx] for idx in selected]


def select_rows(
    rows: list[dict[str, Any]],
    *,
    strategy: str,
    max_samples: int,
    per_prompt_limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    if strategy == "balanced":
        return choose_balanced(rows, max_samples=max_samples, per_prompt_limit=per_prompt_limit, seed=seed)
    if strategy == "largest_mae":
        return choose_largest_mae(rows, max_samples=max_samples, per_prompt_limit=per_prompt_limit, seed=seed)
    raise ValueError(f"Unknown selection strategy: {strategy}")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dim_bins: dict[str, dict[str, int]] = {}
    dim_means: dict[str, float] = {}
    for dim in SCORE_DIMS:
        values = [row["judge_scores"][dim] for row in rows]
        counts = Counter(score_bin(value) for value in values)
        dim_bins[dim] = {f"{low}_{high}": counts.get(f"{low}_{high}", 0) for low, high in BINS}
        dim_means[dim] = round(sum(values) / len(values), 4) if values else 0.0
    mae_values = [float(row.get("mae") or 0.0) for row in rows]
    max_error_values = [int(row.get("self_judge_max_abs_error") or 0) for row in rows]
    return {
        "num_examples": len(rows),
        "judge_score_bin_counts": dim_bins,
        "judge_score_means": dim_means,
        "avg_mae_before_sft": round(sum(mae_values) / len(rows), 4) if rows else 0.0,
        "min_mae_before_sft": round(min(mae_values), 4) if rows else 0.0,
        "max_mae_before_sft": round(max(mae_values), 4) if rows else 0.0,
        "avg_max_dim_abs_error_before_sft": round(sum(max_error_values) / len(rows), 4) if rows else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout_jsonl", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--output_parquet", type=Path, default=None)
    parser.add_argument("--summary_path", type=Path, default=None)
    parser.add_argument("--max_samples", type=int, default=4000)
    parser.add_argument("--per_prompt_limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection_strategy",
        choices=("balanced", "largest_mae"),
        default="balanced",
        help="How to choose SFT examples when usable rollouts exceed max_samples.",
    )
    parser.add_argument("--split", default="train", help="Only keep rollout records whose extra_info.split matches this value.")
    args = parser.parse_args()

    raw_records = load_jsonl(args.rollout_jsonl)
    converted: list[dict[str, Any]] = []
    skipped = Counter()
    for record in raw_records:
        extra = record.get("extra_info") if isinstance(record.get("extra_info"), dict) else {}
        if args.split and extra.get("split") != args.split:
            skipped["split"] += 1
            continue
        item = convert_record(record)
        if item is None:
            skipped["not_judge_ok"] += 1
            continue
        converted.append(item)

    if not converted:
        raise RuntimeError(f"No usable judge-ok rollout records found in {args.rollout_jsonl}")

    selected = select_rows(
        converted,
        strategy=args.selection_strategy,
        max_samples=args.max_samples,
        per_prompt_limit=args.per_prompt_limit,
        seed=args.seed,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.output_parquet is not None:
        args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(selected).to_parquet(args.output_parquet, index=False)

    summary = {
        "input_path": str(args.rollout_jsonl),
        "raw_records": len(raw_records),
        "usable_records": len(converted),
        "selected_records": len(selected),
        "selection_strategy": args.selection_strategy,
        "skipped": dict(skipped),
        **summarize(selected),
    }
    if args.summary_path is not None:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
