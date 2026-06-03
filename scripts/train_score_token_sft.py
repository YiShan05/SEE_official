#!/usr/bin/env python3
"""Full-model SFT where loss is applied only to five self-eval score tokens."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} must decode to an object")
            rows.append(obj)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


class ScoreTokenDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int, skip_overlong: bool):
        self.rows = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.skip_overlong = skip_overlong
        for idx, row in enumerate(rows):
            item = self._encode(row, idx)
            if item is not None:
                self.rows.append(item)
        if not self.rows:
            raise RuntimeError("No usable SFT examples after tokenization")

    def _encode(self, row: dict[str, Any], idx: int) -> dict[str, Any] | None:
        text = row.get("text")
        positions = row.get("score_digit_positions")
        if not isinstance(text, str) or not text:
            raise ValueError(f"row {idx} has empty text")
        if not isinstance(positions, list) or len(positions) != 5:
            raise ValueError(f"row {idx} must contain five score_digit_positions")

        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if len(input_ids) > self.max_length:
            if self.skip_overlong:
                return None
            raise ValueError(f"row {idx} token length {len(input_ids)} exceeds max_length={self.max_length}")

        labels = [-100] * len(input_ids)
        weights = [0.0] * len(input_ids)

        for pos in positions:
            start = int(pos["start"])
            end = int(pos["end"])
            matched = []
            for token_idx, (tok_start, tok_end) in enumerate(offsets):
                if tok_end <= start or tok_start >= end:
                    continue
                matched.append(token_idx)
            if not matched:
                snippet = text[max(0, start - 20) : min(len(text), end + 20)]
                raise ValueError(f"row {idx} score position did not match any token: {pos!r}, {snippet!r}")
            for token_idx in matched:
                labels[token_idx] = input_ids[token_idx]
                weights[token_idx] = 1.0

        if sum(1 for label in labels if label != -100) < 5:
            raise ValueError(f"row {idx} produced fewer than five supervised score tokens")

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "token_weights": torch.tensor(weights, dtype=torch.float32),
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.rows[idx]


def collate_batch(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    output: dict[str, list[torch.Tensor]] = {key: [] for key in batch[0].keys()}
    for item in batch:
        pad_len = max_len - item["input_ids"].shape[0]
        output["input_ids"].append(F.pad(item["input_ids"], (0, pad_len), value=pad_token_id))
        output["attention_mask"].append(F.pad(item["attention_mask"], (0, pad_len), value=0))
        output["labels"].append(F.pad(item["labels"], (0, pad_len), value=-100))
        output["token_weights"].append(F.pad(item["token_weights"], (0, pad_len), value=0.0))
    return {key: torch.stack(values, dim=0) for key, values in output.items()}


def score_token_loss(logits: torch.Tensor, labels: torch.Tensor, token_weights: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = token_weights[:, 1:].contiguous()
    valid = shift_labels != -100
    if not valid.any():
        raise RuntimeError("Batch contains no supervised score tokens after causal shift")
    per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    weights = shift_weights * valid.float()
    return (per_token * weights).sum() / weights.sum().clamp_min(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--train_jsonl", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_length", type=int, default=12288)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_overlong", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_jsonl(args.train_jsonl)
    random.shuffle(rows)
    dataset = ScoreTokenDataset(rows, tokenizer, max_length=args.max_length, skip_overlong=args.skip_overlong)
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        if is_distributed
        else None
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.to(device)
    model.train()
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_update_steps = math.ceil(len(dataloader) * args.epochs / args.grad_accum)
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)

    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=total_update_steps, desc="score-token-sft", disable=not is_main)
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(dataloader):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            should_step = (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(dataloader)
            sync_context = (
                model.no_sync()
                if is_distributed and not should_step
                else nullcontext()
            )
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                    loss = score_token_loss(outputs.logits, batch["labels"], batch["token_weights"])
                    scaled_loss = loss / args.grad_accum
                scaled_loss.backward()
            running_loss += float(loss.detach().cpu())

            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if is_main:
                    pbar.update(1)
                    pbar.set_postfix(loss=running_loss / max(global_step, 1), lr=scheduler.get_last_lr()[0])
    if is_main:
        pbar.close()

    if is_distributed:
        dist.barrier()

    if is_main:
        save_model = model.module if isinstance(model, DDP) else model
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_model.save_pretrained(str(args.output_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(args.output_dir))
        metrics = {
            "train_examples": len(dataset),
            "epochs": args.epochs,
            "per_gpu_batch_size": args.batch_size,
            "world_size": world_size,
            "grad_accum": args.grad_accum,
            "effective_batch_size": args.batch_size * world_size * args.grad_accum,
            "update_steps_per_rank": global_step,
            "mean_loss_per_microbatch_rank0": running_loss / max(len(dataloader) * args.epochs, 1),
            "lr": args.lr,
            "max_length": args.max_length,
            "loss_scope": "five self-eval score tokens only",
        }
        (args.output_dir / "score_token_sft_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
