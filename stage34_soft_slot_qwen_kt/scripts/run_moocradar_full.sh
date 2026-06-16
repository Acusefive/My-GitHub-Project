#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/home/xiaoyao/code/Work4.1}"
PROJECT_ROOT="${WORKSPACE}/stage34_soft_slot_qwen_kt"
FEATURE_DIR="${PROJECT_ROOT}/artifacts/moocradar/features"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/moocradar/context_target_slots_only"
RESULT_DIR="${PROJECT_ROOT}/results/moocradar/context_target_slots_only"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${WORKSPACE}"

echo "[CONFIG] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} train_batch=${BATCH_SIZE} grad_accum=${GRAD_ACCUM} effective_batch=$((BATCH_SIZE * GRAD_ACCUM)) eval_batch=${EVAL_BATCH_SIZE} attention=${ATTN_IMPLEMENTATION}"

if [[ ! -f "${FEATURE_DIR}/feature_manifest.json" ]]; then
  echo "Missing ${FEATURE_DIR}/feature_manifest.json. Run run_moocradar_smoke.sh first." >&2
  exit 1
fi

python "${PROJECT_ROOT}/scripts/train_soft_slot_kt.py" \
  --feature_dir "${FEATURE_DIR}" \
  --model_name_or_path qwen/Qwen3-8B \
  --output_dir "${CHECKPOINT_DIR}" \
  --slot_mode context_target \
  --context_fields llm_embeddings,llm_struct_features,stage34_numeric \
  --target_fields hqtext,hqid,semantic,collaborative \
  --context_soft_tokens 4 \
  --target_soft_tokens 2 \
  --no-include_context_text \
  --epochs 40 \
  --save_epochs 10,20,30,40 \
  --validation_disabled \
  --dtype bfloat16 \
  --attn_implementation "${ATTN_IMPLEMENTATION}" \
  --batch_size "${BATCH_SIZE}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --resume auto \
  --seed 42

python "${PROJECT_ROOT}/scripts/select_and_evaluate_soft_slot_kt.py" \
  --feature_dir "${FEATURE_DIR}" \
  --model_name_or_path qwen/Qwen3-8B \
  --checkpoints "${CHECKPOINT_DIR}/checkpoint_epoch_10.pt,${CHECKPOINT_DIR}/checkpoint_epoch_20.pt,${CHECKPOINT_DIR}/checkpoint_epoch_30.pt,${CHECKPOINT_DIR}/checkpoint_epoch_40.pt" \
  --output_dir "${RESULT_DIR}" \
  --slot_mode context_target \
  --context_fields llm_embeddings,llm_struct_features,stage34_numeric \
  --target_fields hqtext,hqid,semantic,collaborative \
  --context_soft_tokens 4 \
  --target_soft_tokens 2 \
  --no-include_context_text \
  --selection_limit 50000 \
  --dtype bfloat16 \
  --attn_implementation "${ATTN_IMPLEMENTATION}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --seed 42
