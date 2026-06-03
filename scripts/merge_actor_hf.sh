#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 /path/to/global_step_N [target_dir]" >&2
    exit 2
fi

CKPT_DIR="$1"
TARGET_DIR="${2:-${CKPT_DIR}/actor_hf_merged}"

if [ ! -d "${CKPT_DIR}/actor" ]; then
    echo "[ERROR] Missing actor checkpoint directory: ${CKPT_DIR}/actor" >&2
    exit 1
fi

if [ -f "${TARGET_DIR}/model.safetensors" ]; then
    echo "[INFO] merged HF actor already exists: ${TARGET_DIR}"
    exit 0
fi

rm -rf "${TARGET_DIR}"
"${PYTHON_BIN}" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${CKPT_DIR}/actor" \
    --target_dir "${TARGET_DIR}" \
    --trust-remote-code \
    --use_cpu_initialization

test -f "${TARGET_DIR}/model.safetensors"
echo "[INFO] merged HF actor: ${TARGET_DIR}"
