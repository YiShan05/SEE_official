Generated 0-9 RL parquet files are written here.

By default, `prepare_data.py` keeps all unique HelpSteer2 train prompts and the full HelpSteer2 validation unique-prompt set.

Expected full data files:

```text
core/data/rl/train.parquet
core/data/rl/val.parquet
```

The closed-loop training script does not slide over this full train file. At run start it samples one fixed train window with:

```text
window_size = train_batch_size * rl_steps
```

and reuses that same window for every RL+SFT cycle.

Generate the full data directly from HelpSteer2 jsonl with:

```bash
python core/prepare_data.py
```
