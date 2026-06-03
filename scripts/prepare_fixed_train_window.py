#!/usr/bin/env python3
"""Create one fixed RL train window for all closed-loop cycles.

The input parquet is the full HelpSteer2-derived train set. This script draws a
seeded subset of exactly window_size rows once; every RL+SFT cycle reuses this
same parquet so the training window never slides.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


def sample_fixed_window(df: pd.DataFrame, *, window_size: int, seed: int) -> pd.DataFrame:
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if len(df) == 0:
        raise ValueError("input train parquet is empty")
    if window_size > len(df):
        raise ValueError(
            f"window_size={window_size} exceeds available train rows={len(df)}; "
            "reduce train_batch_size or rl_steps"
        )
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(df)), window_size))
    return df.iloc[indices].reset_index(drop=True)


def preview_indices(window: pd.DataFrame, limit: int = 10) -> list[Any]:
    values: list[Any] = []
    if "extra_info" not in window.columns:
        return values
    for item in window["extra_info"].head(limit):
        if isinstance(item, dict):
            values.append(item.get("index"))
        else:
            values.append(None)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_parquet", type=Path, required=True)
    parser.add_argument("--output_parquet", type=Path, required=True)
    parser.add_argument("--summary_path", type=Path, default=None)
    parser.add_argument("--window_size", type=int, required=True)
    parser.add_argument("--train_batch_size", type=int, required=True)
    parser.add_argument("--rl_steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    expected_window_size = args.train_batch_size * args.rl_steps
    if args.window_size != expected_window_size:
        raise ValueError(
            f"window_size must equal train_batch_size * rl_steps: "
            f"{args.window_size} != {args.train_batch_size} * {args.rl_steps}"
        )

    df = pd.read_parquet(args.input_parquet)
    window = sample_fixed_window(df, window_size=args.window_size, seed=args.seed)

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    window.to_parquet(args.output_parquet, index=False)

    summary = {
        "input_parquet": str(args.input_parquet),
        "output_parquet": str(args.output_parquet),
        "full_rows": int(len(df)),
        "window_rows": int(len(window)),
        "window_size": int(args.window_size),
        "train_batch_size": int(args.train_batch_size),
        "rl_steps": int(args.rl_steps),
        "seed": int(args.seed),
        "sampling": "seeded_random_without_replacement",
        "preview_extra_info_indices": preview_indices(window),
    }
    if args.summary_path is not None:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
