# SEE: Self-Evaluation Elicitation

[![arXiv](https://img.shields.io/badge/arXiv-2606.05122-b31b1b.svg)](https://arxiv.org/abs/2606.05122)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository is the official implementation of **SEE** for the paper:

**[Self-Evaluation Is Already There: Eliciting Latent Judge Calibration in Base LLMs with Minimal Data](https://arxiv.org/abs/2606.05122)**.

SEE studies whether a base language model can predict how an external judge will score its own open-ended response. The central finding is that this ability is largely already present in the base model. SEE therefore treats self-evaluation as an elicitation problem rather than as a capability that must be learned from scratch.

## Method

SEE trains a base model to produce both an answer and a strict self-evaluation block:

```text
[SELF_EVAL]
{"helpfulness": <0-9>, "correctness": <0-9>, "coherence": <0-9>, "complexity": <0-9>, "verbosity": <0-9>}
[/SELF_EVAL]
```

Training alternates two phases on one fixed data window:

1. **Calibration-Coupled RL**
   The model answers a prompt and predicts five judge scores. An external judge scores the same response. The RL reward combines answer quality with nonlinear calibration between the model's self-scores and the judge's scores.
2. **Masked Judge Distillation**
   Rollouts from the RL phase are converted into SFT examples by replacing the self-evaluation scores with the judge's scores. The SFT loss is applied only to the five score tokens, leaving the answer tokens untouched.

These two phases are repeated for multiple cycles. In the paper setting, SEE uses a small fixed window of 160 unique training prompts and repeatedly re-grounds self-evaluation to the judge while preserving answer quality.

The reward is:

```text
quality = mean(judge helpfulness, correctness, coherence) / 9
calibration_linear = 1 - MAE(self_scores, judge_scores) / 9
calibration = calibration_linear ** CALIBRATION_GAMMA
reward = W_QUALITY * quality + W_CALIBRATION * calibration
```

Invalid policy format, invalid data, invalid judge format, or judge retry exhaustion raises an error instead of returning fallback rewards.

## Repository Structure

```text
core/
  prompt_config.py                 Prompt template and strict self-eval parsing
  prepare_data.py                  Convert HelpSteer2 jsonl into RL parquet data
  plain_completion_agent_loop.py   Plain-completion verl agent loop with stop handling
  plain_completion_agent_loop.yaml Agent-loop registration config
  reward_fn.py                     Strict nonlinear 0-9 reward
  reward_fn_collect.py             Reward function with rollout logging for SFT

scripts/
  run_closed_loop_rl_sft.sh        Main repeated-cycle RL+SFT entry point
  prepare_fixed_train_window.py    Sample the fixed RL window once per run
  train_grpo_base_parallel4.sh     4-GPU GRPO training configuration
  train_grpo_collect_round1.sh     RL wrapper that enables rollout collection
  build_score_sft_data.py          Build score-token SFT data from rollouts
  train_score_token_sft.py         SFT with loss only on self-eval score tokens
  merge_actor_hf.sh                Merge verl actor shards into HF format
```

## Setup

Install the required environment:

```bash
pip install -r requirements.txt
```

The default training script expects 4 visible GPUs. The code uses a base model such as `Qwen3-4B-Base`, HelpSteer2-format jsonl data, and an OpenAI-compatible judge API.

## Prepare Data

Generate RL parquet files from HelpSteer2:

```bash
python core/prepare_data.py \
  --train_path /path/to/HelpSteer2/train.jsonl \
  --val_path /path/to/HelpSteer2/validation.jsonl \
  --output_dir core/data/rl
```

This creates:

```text
core/data/rl/train.parquet
core/data/rl/val.parquet
```

## Run Training

Set the judge API and model paths:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-5.4"

export BASE_MODEL="/path/to/Qwen3-4B-Base"
export PYTHON_BIN="$(which python)"
chmod +x scripts/*.sh
```

Launch SEE:

```bash
CYCLES=15 \
RL_STEPS=10 \
TRAIN_BATCH_SIZE=16 \
SFT_MAX_SAMPLES=400 \
SFT_SELECTION_STRATEGY=balanced \
bash scripts/run_closed_loop_rl_sft.sh
```

The fixed RL window size is:

```text
window_size = TRAIN_BATCH_SIZE * RL_STEPS
```

All cycles reuse the same fixed window. Checkpoints, rollouts, SFT data, and manifests are written by default to:

```text
outputs/closed_loop/<run_name>
```

## Frameworks

This implementation is built on:

- [verl](https://github.com/verl-project/verl) for distributed GRPO/RL post-training.
- [vLLM](https://github.com/vllm-project/vllm) for efficient rollout generation.

## Citation

If you find SEE useful in your research, please consider citing our paper:

```bibtex
@misc{zhang2026see,
      title={Self-Evaluation Is Already There: Eliciting Latent Judge Calibration in Base LLMs with Minimal Data},
      author={XiuYu Zhang and Yi Shan and Junfeng Fang and Zhenkai Liang},
      year={2026},
      eprint={2606.05122},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.05122}
}
```

## License

This project is released under the [Apache 2.0 License](LICENSE).
