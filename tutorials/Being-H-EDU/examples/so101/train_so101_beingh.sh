#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export NCCL_IB_DISABLE=0
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Override these with environment variables or command-line arguments on a new machine.
PRETRAIN_MODEL="${PRETRAIN_MODEL:-/path/to/InternVL3_5-2B}"
EXPERT_MODEL="${EXPERT_MODEL:-/path/to/Qwen3-0.6B}"
RESUME_PATH="${RESUME_PATH:-/path/to/being-h05-checkpoint}"
DATASET_CONFIG_FILE="${DATASET_CONFIG_FILE:-configs/posttrain/so101/so101_example.yaml}"

# ============ 训练参数 ============
NUM_GPUS=8
MASTER_PORT=29107
MAX_STEPS=15000
SAVE_STEPS=5000
SAVE_STEPS_START=0
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-5
WARMUP_RATIO=0.05
NUM_WORKERS=4
PREFETCH_FACTOR=4
ACTION_CHUNK_LENGTH=16

# ============ W&B 参数 ============
USE_WANDB=False
WANDB_PROJECT="beingh-so101"
WANDB_ENTITY=""
WANDB_NAME=""
WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/logs/wandb}"
WANDB_MODE="online"

# ============ 路径参数 ============
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/tensorboard}"
RUN_NAME=""
EXTRA_TRAIN_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash train_so101_beingh.sh [options] [-- extra train.py args]

Common options:
  --gpus N                    Number of GPUs, e.g. 4
  --steps N                   Max training steps, e.g. 20000
  --lr LR                     Learning rate, e.g. 5e-5
  --save-steps N              Checkpoint save interval
  --save-start N              Start saving after this step
  --weight-decay VALUE        Weight decay
  --warmup-ratio VALUE        Warmup ratio
  --workers N                 DataLoader workers
  --prefetch-factor N         DataLoader prefetch factor
  --action-chunk-length N     Action chunk length
  --dataset-config PATH       Dataset YAML config
  --pretrain-model PATH       InternVL/Qwen VLM checkpoint path
  --expert-model PATH         Qwen expert checkpoint path
  --resume-from PATH          Resume/pretrained checkpoint path
  --run-name NAME             Output folder name
  --output-root PATH          Output root directory
  --log-dir PATH              TensorBoard log directory
  --master-port PORT          torch.distributed master port

W&B options (not supported by BeingH/train/train.py):
  These are still accepted so existing commands do not break on an unknown option,
  but anything that enables W&B exits with an error instead of training silently
  without logging. Training metrics go to TensorBoard in --log-dir.
  --wandb, --wandb-project NAME, --wandb-entity NAME, --wandb-name NAME,
  --wandb-dir PATH, --wandb-mode MODE
  --no-wandb                  Disable W&B logging (the default)

Example:
  bash train_so101_beingh.sh --gpus 4 --steps 20000 --lr 5e-5
EOF
}

need_value() {
  if [[ $# -lt 2 || "$2" == --* ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

require_existing_path() {
  local name="$1"
  local path="$2"
  if [[ "${path}" == /path/to/* || ! -e "${path}" ]]; then
    echo "${name} is not set to an existing path: ${path}" >&2
    echo "Set ${name} with an environment variable or the matching command-line option." >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --gpus)
      need_value "$@"; NUM_GPUS="$2"; shift 2
      ;;
    --steps|--max-steps)
      need_value "$@"; MAX_STEPS="$2"; shift 2
      ;;
    --lr|--learning-rate)
      need_value "$@"; LEARNING_RATE="$2"; shift 2
      ;;
    --save-steps)
      need_value "$@"; SAVE_STEPS="$2"; shift 2
      ;;
    --save-start|--save-steps-start)
      need_value "$@"; SAVE_STEPS_START="$2"; shift 2
      ;;
    --weight-decay)
      need_value "$@"; WEIGHT_DECAY="$2"; shift 2
      ;;
    --warmup-ratio)
      need_value "$@"; WARMUP_RATIO="$2"; shift 2
      ;;
    --workers|--num-workers)
      need_value "$@"; NUM_WORKERS="$2"; shift 2
      ;;
    --prefetch-factor)
      need_value "$@"; PREFETCH_FACTOR="$2"; shift 2
      ;;
    --action-chunk-length)
      need_value "$@"; ACTION_CHUNK_LENGTH="$2"; shift 2
      ;;
    --dataset-config|--dataset-config-file)
      need_value "$@"; DATASET_CONFIG_FILE="$2"; shift 2
      ;;
    --pretrain-model)
      need_value "$@"; PRETRAIN_MODEL="$2"; shift 2
      ;;
    --expert-model)
      need_value "$@"; EXPERT_MODEL="$2"; shift 2
      ;;
    --resume-from)
      need_value "$@"; RESUME_PATH="$2"; shift 2
      ;;
    --run-name)
      need_value "$@"; RUN_NAME="$2"; shift 2
      ;;
    --output-root)
      need_value "$@"; OUTPUT_ROOT="$2"; shift 2
      ;;
    --log-dir)
      need_value "$@"; LOG_DIR="$2"; shift 2
      ;;
    --master-port)
      need_value "$@"; MASTER_PORT="$2"; shift 2
      ;;
    --wandb)
      USE_WANDB=True; shift
      ;;
    --no-wandb)
      USE_WANDB=False; shift
      ;;
    --wandb-project)
      need_value "$@"; USE_WANDB=True; WANDB_PROJECT="$2"; shift 2
      ;;
    --wandb-entity)
      need_value "$@"; USE_WANDB=True; WANDB_ENTITY="$2"; shift 2
      ;;
    --wandb-name)
      need_value "$@"; USE_WANDB=True; WANDB_NAME="$2"; shift 2
      ;;
    --wandb-dir)
      need_value "$@"; WANDB_DIR="$2"; shift 2
      ;;
    --wandb-mode)
      need_value "$@"; WANDB_MODE="$2"; shift 2
      ;;
    --)
      shift
      EXTRA_TRAIN_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Use --help to see supported options." >&2
      exit 2
      ;;
  esac
done

# ============ 输出路径 ============
MODEL_NAME="${RUN_NAME:-so101_beingh_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}"
if [[ -z "${WANDB_NAME}" ]]; then
  WANDB_NAME="${MODEL_NAME}"
fi

# BeingH/train/train.py accepts no W&B arguments, so they are not forwarded below.
# Fail loudly rather than accepting --wandb and logging nothing.
if [[ "${USE_WANDB}" == "True" ]]; then
  echo "W&B logging was requested, but BeingH/train/train.py does not support it." >&2
  echo "Re-run without --wandb / --wandb-project; training logs to TensorBoard in ${LOG_DIR}." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
echo "================================================================"
echo "Starting SO101 training..."
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NUM_GPUS} | steps: ${MAX_STEPS} | lr: ${LEARNING_RATE}"
echo "Dataset config: ${DATASET_CONFIG_FILE}"
echo "TensorBoard log dir: ${LOG_DIR}"
echo "================================================================"

cd "${PROJECT_ROOT}"

require_existing_path "PRETRAIN_MODEL" "${PRETRAIN_MODEL}"
require_existing_path "EXPERT_MODEL" "${EXPERT_MODEL}"
require_existing_path "RESUME_PATH" "${RESUME_PATH}"
require_existing_path "DATASET_CONFIG_FILE" "${DATASET_CONFIG_FILE}"

python -m torch.distributed.run \
  --nnodes=1 --node_rank=0 \
  --nproc_per_node=${NUM_GPUS} \
  --master_port=${MASTER_PORT} \
  BeingH/train/train.py \
  --mllm_path "${PRETRAIN_MODEL}" \
  --expert_path "${EXPERT_MODEL}" \
  --resume_from "${RESUME_PATH}" \
  --resume_model_only True \
  --layer_module Qwen3MoTDecoderLayer \
  --use_expert True \
  --use_flow_matching True \
  --llm_qk_norm True \
  --freeze_mllm False \
  --freeze_vit_mlp False \
  --action_chunk_length ${ACTION_CHUNK_LENGTH} \
  --max_num_tokens 8704 \
  --max_num_tokens_per_sample 8704 \
  --expected_num_tokens 8192 \
  --prefer_buffer_before 4096 \
  --max_buffer_size 4 \
  --attn_mode causal \
  --max_view_num -1 \
  --use_fixed_view False \
  --force_image_size 224 \
  --down_sample_ratio 0.5 \
  --dataset_config_file "${DATASET_CONFIG_FILE}" \
  --save_merged_metadata True \
  --conv_style "being_h0" \
  --vision_select_layer -1 \
  --prompt_template long \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${LOG_DIR}" \
  --num_workers ${NUM_WORKERS} \
  --prefetch_factor ${PREFETCH_FACTOR} \
  --max_steps ${MAX_STEPS} \
  --save_model_only False \
  --save_steps ${SAVE_STEPS} \
  --save_steps_start ${SAVE_STEPS_START} \
  --logging_steps 10 \
  --learning_rate ${LEARNING_RATE} \
  --weight_decay ${WEIGHT_DECAY} \
  --warmup_ratio ${WARMUP_RATIO} \
  --lr_scheduler cosine \
  --grad_checkpoint False \
  --gradient_accumulation_steps 2 \
  --use_mpg True \
  --mpg_lambda 0.1 \
  --mpg_num_projections 32 \
  --mpg_refinement_iters 1 \
  --mpg_gate_temperature 1.0 \
  --mpg_use_stop_gradient True \
  --use_training_time_rtc False \
  --simulated_delay 0 \
  --rtc_delay_exp_weight True \
  --use_inference_prefix_overwrite True \
  "${EXTRA_TRAIN_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/training.log"
