#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
BASE_DIR="${BASE_DIR:-${PROJECT_DIR}/core}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:-/path/to/Qwen3-4B-Base}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PROJECT_DIR}/outputs/closed_loop}"
RUN_NAME="${RUN_NAME:-closed_loop_rl_sft_fixed_window_${TIMESTAMP}}"
RUN_DIR="${RUN_DIR:-${EXPERIMENT_ROOT}/${RUN_NAME}}"
DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/data}"
VAL_DATA="${VAL_DATA:-${DATA_ROOT}/rl/val.parquet}"
FULL_TRAIN_DATA="${FULL_TRAIN_DATA:-${DATA_ROOT}/rl/train.parquet}"

CYCLES="${CYCLES:-${NUM_CYCLES:-3}}"
RL_STEPS="${RL_STEPS:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
WINDOW_SIZE=$((TRAIN_BATCH_SIZE * RL_STEPS))
RL_SAVE_FREQ="${RL_SAVE_FREQ:-5}"
SFT_MAX_SAMPLES="${SFT_MAX_SAMPLES:-400}"
SFT_SELECTION_STRATEGY="${SFT_SELECTION_STRATEGY:-balanced}"
SFT_PER_PROMPT_LIMIT="${SFT_PER_PROMPT_LIMIT:-4}"
SEED="${SEED:-42}"
ALLOW_CKPT_CLEANUP="${ALLOW_CKPT_CLEANUP:-1}"
KEEP_LAST_N_CYCLES="${KEEP_LAST_N_CYCLES:-1}"

if [ -z "${SEED_MODEL:-}" ]; then
    SEED_MODEL="${BASE_MODEL}"
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[ERROR] Please export OPENAI_API_KEY." >&2
    exit 1
fi
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o}"

cleanup_ray() {
    mkdir -p "${RUN_DIR}"
    "${PYTHON_BIN}" -m ray stop --force >"${RUN_DIR}/ray_stop.log" 2>&1 || true
}
trap cleanup_ray ERR

require_dir() {
    local path="$1"
    if [ ! -d "${path}" ]; then
        echo "[ERROR] Missing required directory: ${path}" >&2
        exit 1
    fi
}

safe_rm_dir() {
    local path="$1"
    case "${path}" in
        "${RUN_DIR}"/*) ;;
        *) echo "[ERROR] Refusing to delete path outside RUN_DIR: ${path}" >&2; exit 1 ;;
    esac
    if [ "${path}" = "${SEED_MODEL}" ]; then
        echo "[INFO] Keeping seed model ${path}"
        return 0
    fi
    if [ -e "${path}" ]; then
        du -sh "${path}" || true
        rm -rf --one-file-system "${path}"
    fi
}

cleanup_rl_training_shards_after_merge() {
    local cycle="$1"
    if [ "${ALLOW_CKPT_CLEANUP}" != "1" ]; then
        return 0
    fi
    local cycle_rl="${RUN_DIR}/cycle_${cycle}/rl"
    local final_step_dir="${cycle_rl}/global_step_${RL_STEPS}"
    local merged_dir="${final_step_dir}/actor_hf_merged"
    require_dir "${merged_dir}"
    echo "[INFO] Cycle ${cycle}: cleanup redundant RL shards; keep merged actor ${merged_dir}"
    for step_dir in "${cycle_rl}"/global_step_*; do
        [ -e "${step_dir}" ] || continue
        if [ "${step_dir}" != "${final_step_dir}" ]; then
            safe_rm_dir "${step_dir}"
        fi
    done
    safe_rm_dir "${final_step_dir}/actor"
}

cleanup_old_cycles() {
    local current_cycle="$1"
    if [ "${ALLOW_CKPT_CLEANUP}" != "1" ]; then
        return 0
    fi
    local cutoff=$((current_cycle - KEEP_LAST_N_CYCLES))
    if [ "${cutoff}" -lt 1 ]; then
        return 0
    fi
    echo "[INFO] Cleanup cycles <= ${cutoff}; keep last ${KEEP_LAST_N_CYCLES} cycle(s)."
    for cycle in $(seq 1 "${cutoff}"); do
        safe_rm_dir "${RUN_DIR}/cycle_${cycle}/rl"
        safe_rm_dir "${RUN_DIR}/cycle_${cycle}/sft_score_tokens"
    done
    df -h "${RUN_DIR}" || true
}

if [ "${CYCLES}" -le 0 ]; then
    echo "[ERROR] CYCLES must be positive, got ${CYCLES}" >&2
    exit 1
fi
require_dir "${SEED_MODEL}"
mkdir -p "${RUN_DIR}/rl_data"

FIXED_TRAIN_DATA="${RUN_DIR}/rl_data/fixed_train_window_bs${TRAIN_BATCH_SIZE}_steps${RL_STEPS}_seed${SEED}.parquet"
FIXED_TRAIN_SUMMARY="${RUN_DIR}/rl_data/fixed_train_window_summary.json"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/prepare_fixed_train_window.py" \
    --input_parquet "${FULL_TRAIN_DATA}" \
    --output_parquet "${FIXED_TRAIN_DATA}" \
    --summary_path "${FIXED_TRAIN_SUMMARY}" \
    --window_size "${WINDOW_SIZE}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --rl_steps "${RL_STEPS}" \
    --seed "${SEED}"

current_model="${SEED_MODEL}"
echo "[INFO] RUN_DIR=${RUN_DIR}"
echo "[INFO] Fixed train window: ${FIXED_TRAIN_DATA}"
echo "[INFO] Run from ${current_model}; cycles=${CYCLES}; RL_STEPS=${RL_STEPS}; TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}; WINDOW_SIZE=${WINDOW_SIZE}; SFT_MAX_SAMPLES=${SFT_MAX_SAMPLES}; SFT_SELECTION_STRATEGY=${SFT_SELECTION_STRATEGY}"

for cycle in $(seq 1 "${CYCLES}"); do
    cycle_dir="${RUN_DIR}/cycle_${cycle}"
    final_ckpt="${cycle_dir}/rl/global_step_${RL_STEPS}"
    rl_model="${final_ckpt}/actor_hf_merged"
    sft_model="${cycle_dir}/sft_score_tokens"

    if [ -d "${sft_model}" ] && [ -d "${rl_model}" ]; then
        echo "[INFO] Cycle ${cycle}: existing complete RL+SFT found; skipping training."
        current_model="${sft_model}"
        cleanup_old_cycles "${cycle}"
        continue
    fi

    if [ -d "${cycle_dir}" ]; then
        echo "[ERROR] Incomplete cycle directory already exists: ${cycle_dir}" >&2
        echo "[ERROR] Move or remove it explicitly before restarting to avoid mixing partial data." >&2
        exit 1
    fi
    mkdir -p "${cycle_dir}/rollouts" "${cycle_dir}/sft_data"

    export PROJECT_DIR BASE_DIR
    export MODEL_PATH="${current_model}"
    export TRAIN_DATA="${FIXED_TRAIN_DATA}"
    export CHECKPOINT_DIR="${cycle_dir}/rl"
    export ROLLOUT_LOG_PATH="${cycle_dir}/rollouts/rl_rollouts.jsonl"
    export TOTAL_TRAINING_STEPS="${RL_STEPS}"
    export SAVE_FREQ="${RL_SAVE_FREQ}"
    export TEST_FREQ=-1
    export TOTAL_EPOCHS=1
    export ROLLOUT_N="${ROLLOUT_N:-8}"
    export TRAIN_BATCH_SIZE
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
    export ACTOR_MAX_TOKEN_LEN_PER_GPU="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-16384}"
    export ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU="${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU:-16384}"
    export REF_LOGPROB_MAX_TOKEN_LEN_PER_GPU="${REF_LOGPROB_MAX_TOKEN_LEN_PER_GPU:-16384}"
    export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"
    export ROLLOUT_LOGPROB_USE_DYNAMIC_BSZ="${ROLLOUT_LOGPROB_USE_DYNAMIC_BSZ:-False}"
    export REF_LOGPROB_USE_DYNAMIC_BSZ="${REF_LOGPROB_USE_DYNAMIC_BSZ:-False}"
    export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
    export ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
    export REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
    export ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-65536}"
    export ROLLOUT_MAX_SEQS="${ROLLOUT_MAX_SEQS:-128}"
    export JUDGE_MAX_CONCURRENT="${JUDGE_MAX_CONCURRENT:-24}"
    export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-3}"

    echo "[INFO] ===== Cycle ${cycle}/${CYCLES}: RL from ${current_model} ====="
    "${PROJECT_DIR}/scripts/train_grpo_collect_round1.sh" > "${cycle_dir}/rl_phase.log" 2>&1
    cleanup_ray

    require_dir "${final_ckpt}"
    echo "[INFO] Cycle ${cycle}: merge RL actor ${final_ckpt}"
    "${PROJECT_DIR}/scripts/merge_actor_hf.sh" "${final_ckpt}" > "${cycle_dir}/merge_actor.log" 2>&1
    require_dir "${rl_model}"
    cleanup_rl_training_shards_after_merge "${cycle}"

    echo "[INFO] Cycle ${cycle}: build ${SFT_MAX_SAMPLES} ${SFT_SELECTION_STRATEGY} score-token SFT rows"
    sft_jsonl="${cycle_dir}/sft_data/score_token_sft_train.jsonl"
    "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/build_score_sft_data.py" \
        --rollout_jsonl "${ROLLOUT_LOG_PATH}" \
        --output_jsonl "${sft_jsonl}" \
        --output_parquet "${cycle_dir}/sft_data/score_token_sft_train.parquet" \
        --summary_path "${cycle_dir}/sft_data/score_token_sft_summary.json" \
        --max_samples "${SFT_MAX_SAMPLES}" \
        --per_prompt_limit "${SFT_PER_PROMPT_LIMIT}" \
        --selection_strategy "${SFT_SELECTION_STRATEGY}" \
        --seed "$((SEED + cycle))"

    selected_count="$("${PYTHON_BIN}" -c 'import sys; print(sum(1 for _ in open(sys.argv[1], encoding="utf-8")))' "${sft_jsonl}")"
    if [ "${selected_count}" -ne "${SFT_MAX_SAMPLES}" ]; then
        echo "[ERROR] Expected ${SFT_MAX_SAMPLES} SFT rows, got ${selected_count}." >&2
        exit 1
    fi

    echo "[INFO] Cycle ${cycle}: score-token-only SFT"
    CUDA_VISIBLE_DEVICES="${SFT_CUDA_VISIBLE_DEVICES:-0,1,2,3}" "${PYTHON_BIN}" -m torch.distributed.run \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${SFT_NUM_GPUS:-4}" \
        "${PROJECT_DIR}/scripts/train_score_token_sft.py" \
        --model_path "${rl_model}" \
        --train_jsonl "${sft_jsonl}" \
        --output_dir "${sft_model}" \
        --epochs "${SFT_EPOCHS:-1}" \
        --batch_size "${SFT_BATCH_SIZE:-2}" \
        --grad_accum "${SFT_GRAD_ACCUM:-4}" \
        --lr "${SFT_LR:-2e-6}" \
        --max_length "${SFT_MAX_LENGTH:-12288}" \
        --seed "$((SEED + cycle))" \
        > "${cycle_dir}/sft_train.log" 2>&1

    current_model="${sft_model}"
    cat > "${cycle_dir}/cycle_manifest.json" <<MANIFEST
{
  "cycle": ${cycle},
  "input_model": "${MODEL_PATH}",
  "rl_steps": ${RL_STEPS},
  "train_batch_size": ${TRAIN_BATCH_SIZE},
  "fixed_train_window": "${FIXED_TRAIN_DATA}",
  "fixed_train_window_summary": "${FIXED_TRAIN_SUMMARY}",
  "rl_checkpoint": "${final_ckpt}",
  "rl_model": "${rl_model}",
  "rollout_log": "${ROLLOUT_LOG_PATH}",
  "sft_data": "${sft_jsonl}",
  "sft_model": "${sft_model}",
  "sft_max_samples": ${SFT_MAX_SAMPLES},
  "sft_selection_strategy": "${SFT_SELECTION_STRATEGY}",
  "verl_internal_validation": false
}
MANIFEST
    cleanup_old_cycles "${cycle}"
done

cat > "${RUN_DIR}/run_manifest.json" <<MANIFEST
{
  "project_dir": "${PROJECT_DIR}",
  "run_dir": "${RUN_DIR}",
  "seed_model": "${SEED_MODEL}",
  "cycles": ${CYCLES},
  "rl_steps_per_cycle": ${RL_STEPS},
  "train_batch_size": ${TRAIN_BATCH_SIZE},
  "fixed_window_size": ${WINDOW_SIZE},
  "fixed_train_window": "${FIXED_TRAIN_DATA}",
  "fixed_train_window_summary": "${FIXED_TRAIN_SUMMARY}",
  "sft_max_samples_per_cycle": ${SFT_MAX_SAMPLES},
  "sft_selection_strategy": "${SFT_SELECTION_STRATEGY}",
  "final_model": "${current_model}",
  "full_train_data": "${FULL_TRAIN_DATA}",
  "val_data": "${VAL_DATA}"
}
MANIFEST

echo "[INFO] Closed-loop fixed-window RL+SFT run complete. Final model: ${current_model}"
