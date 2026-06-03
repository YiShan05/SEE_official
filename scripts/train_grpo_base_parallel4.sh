#!/usr/bin/env bash
set -euo pipefail

if [ "${DEBUG_SH:-0}" = "1" ]; then
    set -x
fi

# GRPO training for this repository's core with:
# - Qwen3-4B-Base plain completion prompts
# - 0-9 self-eval scores
# - nonlinear calibration reward: (1 - MAE / 9) ** CALIBRATION_GAMMA

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
BASE_DIR="${BASE_DIR:-${PROJECT_DIR}/core}"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-4B-Base}"
DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/data}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_ROOT}/rl/train.parquet}"
VAL_DATA="${VAL_DATA:-${DATA_ROOT}/rl/val.parquet}"
REWARD_FN="${REWARD_FN:-${BASE_DIR}/reward_fn.py}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_DIR}/outputs/grpo_parallel4}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[ERROR] Please export OPENAI_API_KEY before running this script."
    exit 1
fi

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o}"
export W_QUALITY="${W_QUALITY:-0.7}"
export W_CALIBRATION="${W_CALIBRATION:-0.3}"
export CALIBRATION_GAMMA="${CALIBRATION_GAMMA:-2.0}"
export FORMAT_PENALTY="${FORMAT_PENALTY:--1.0}"
export JUDGE_MAX_CONCURRENT="${JUDGE_MAX_CONCURRENT:-48}"
export JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-90}"
export JUDGE_MAX_RETRIES="${JUDGE_MAX_RETRIES:-8}"
export JUDGE_BACKOFF_BASE="${JUDGE_BACKOFF_BASE:-2.0}"
export JUDGE_BACKOFF_MAX="${JUDGE_BACKOFF_MAX:-120.0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [ "${#GPU_IDS[@]}" -ne 4 ]; then
    echo "[ERROR] This script expects exactly 4 visible GPUs."
    echo "[ERROR] Current CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    exit 1
fi

"${PYTHON_BIN}" -c "import verl" >/dev/null 2>&1 || {
    echo "[ERROR] 'verl' is not available from PYTHON_BIN=${PYTHON_BIN}."
    exit 1
}

if [ ! -f "${TRAIN_DATA}" ] || [ ! -f "${VAL_DATA}" ]; then
    echo "[ERROR] Required 0-9 training data is missing."
    echo "[ERROR] TRAIN_DATA=${TRAIN_DATA}"
    echo "[ERROR] VAL_DATA=${VAL_DATA}"
    echo "[ERROR] Run ${BASE_DIR}/prepare_data.py explicitly after validating the source data."
    exit 1
fi

mkdir -p "${CHECKPOINT_DIR}"

# Keep the generation budget at 8k to avoid excessive truncation in long-form
# HelpSteer2 prompts. The model context is prompt 4k + response 8k.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-48}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-48}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-48}"
ROLLOUT_N="${ROLLOUT_N:-8}"
FILTER_OVERLONG_PROMPTS_WORKERS="${FILTER_OVERLONG_PROMPTS_WORKERS:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-12288}"

ACTOR_MAX_TOKEN_LEN_PER_GPU="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-49152}"
ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU="${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU:-49152}"
REF_LOGPROB_MAX_TOKEN_LEN_PER_GPU="${REF_LOGPROB_MAX_TOKEN_LEN_PER_GPU:-49152}"
normalize_bool() {
    case "${1,,}" in
        1|true|yes|y) echo "True" ;;
        0|false|no|n) echo "False" ;;
        *) echo "$1" ;;
    esac
}
USE_DYNAMIC_BSZ="$(normalize_bool "${USE_DYNAMIC_BSZ:-True}")"
ROLLOUT_LOGPROB_USE_DYNAMIC_BSZ="$(normalize_bool "${ROLLOUT_LOGPROB_USE_DYNAMIC_BSZ:-${USE_DYNAMIC_BSZ}}")"
REF_LOGPROB_USE_DYNAMIC_BSZ="$(normalize_bool "${REF_LOGPROB_USE_DYNAMIC_BSZ:-${USE_DYNAMIC_BSZ}}")"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"

ROLLOUT_TP="${ROLLOUT_TP:-2}"
ROLLOUT_GPU_MEM="${ROLLOUT_GPU_MEM:-0.82}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-262144}"
ROLLOUT_MAX_SEQS="${ROLLOUT_MAX_SEQS:-512}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
AGENT_LOOP_NAME="${AGENT_LOOP_NAME:-plain_completion_agent}"
AGENT_LOOP_CONFIG="${AGENT_LOOP_CONFIG:-${BASE_DIR}/plain_completion_agent_loop.yaml}"

LR="${LR:-1e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:--1}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-3}"
ENABLE_RESUME="${ENABLE_RESUME:-0}"
RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"

RESUME_ARGS=(trainer.resume_mode=disable)
if [ "${ENABLE_RESUME}" = "1" ]; then
    if [ -z "${RESUME_CKPT_PATH}" ] || [ ! -d "${RESUME_CKPT_PATH}" ]; then
        echo "[ERROR] RESUME_CKPT_PATH must point to an existing checkpoint when ENABLE_RESUME=1."
        exit 1
    fi
    RESUME_ARGS=(
        trainer.resume_mode=resume_path
        trainer.resume_from_path="${RESUME_CKPT_PATH}"
    )
fi

LIMIT_ARGS=()
if [ -n "${TOTAL_TRAINING_STEPS}" ]; then
    LIMIT_ARGS=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")
fi

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=${FILTER_OVERLONG_PROMPTS_WORKERS} \
    data.dataloader_num_workers=${DATALOADER_NUM_WORKERS} \
    data.truncation=error \
    \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}" \
    \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.top_k=25 \
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${ROLLOUT_LOGPROB_USE_DYNAMIC_BSZ} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LENGTH} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_SEQS} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.agent.default_agent_loop="${AGENT_LOOP_NAME}" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}" \
    \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${REF_LOGPROB_USE_DYNAMIC_BSZ} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${REF_LOGPROB_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    \
    reward.custom_reward_function.path="${REWARD_FN}" \
    reward.custom_reward_function.name=compute_score \
    reward.reward_model.enable=False \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=see_grpo_base \
    trainer.experiment_name=qwen3_4b_base_see_grpo_parallel4 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.val_before_train=False \
    "${RESUME_ARGS[@]}" \
    "${LIMIT_ARGS[@]}" \
    "$@" 2>&1 | tee "${CHECKPOINT_DIR}/training_grpo_base_parallel4.log"
