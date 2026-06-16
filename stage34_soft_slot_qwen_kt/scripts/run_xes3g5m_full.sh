#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/home/xiaoyao/code/Work4.1}"
PROJECT_ROOT="${WORKSPACE}/stage34_soft_slot_qwen_kt"
FEATURE_DIR="${PROJECT_ROOT}/artifacts/xes3g5m/features"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/xes3g5m/context_target_slots_only"
RESULT_DIR="${PROJECT_ROOT}/results/xes3g5m/context_target_slots_only"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
EPOCHS="${EPOCHS:-40}"
SAVE_EPOCHS="${SAVE_EPOCHS:-10,20,30,40}"
CANDIDATE_EPOCHS="${CANDIDATE_EPOCHS:-${SAVE_EPOCHS}}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
SELECTION_ONLY="${SELECTION_ONLY:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${WORKSPACE}"

GRADIENT_CHECKPOINTING_ARGS=(--gradient_checkpointing)
if [[ "${GRADIENT_CHECKPOINTING}" == "0" ]]; then
  GRADIENT_CHECKPOINTING_ARGS=(--no-gradient_checkpointing)
fi
SELECTION_ONLY_ARGS=()
if [[ "${SELECTION_ONLY}" == "1" ]]; then
  SELECTION_ONLY_ARGS=(--selection_only)
fi
CHECKPOINTS=""
IFS=',' read -ra CANDIDATES <<< "${CANDIDATE_EPOCHS}"
for EPOCH in "${CANDIDATES[@]}"; do
  CHECKPOINT="${CHECKPOINT_DIR}/checkpoint_epoch_${EPOCH}.pt"
  CHECKPOINTS="${CHECKPOINTS:+${CHECKPOINTS},}${CHECKPOINT}"
done

echo "[CONFIG] dataset=xes3g5m CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} train_batch=${BATCH_SIZE} grad_accum=${GRAD_ACCUM} effective_batch=$((BATCH_SIZE * GRAD_ACCUM)) eval_batch=${EVAL_BATCH_SIZE} epochs=${EPOCHS} save_epochs=${SAVE_EPOCHS} candidates=${CANDIDATE_EPOCHS} gradient_checkpointing=${GRADIENT_CHECKPOINTING} selection_only=${SELECTION_ONLY} attention=${ATTN_IMPLEMENTATION}"

if [[ ! -f "${FEATURE_DIR}/feature_manifest.json" ]]; then
  echo "Missing ${FEATURE_DIR}/feature_manifest.json. Run run_xes3g5m_smoke.sh first." >&2
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
  --epochs "${EPOCHS}" \
  --save_epochs "${SAVE_EPOCHS}" \
  --validation_disabled \
  --dtype bfloat16 \
  --attn_implementation "${ATTN_IMPLEMENTATION}" \
  --batch_size "${BATCH_SIZE}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  "${GRADIENT_CHECKPOINTING_ARGS[@]}" \
  --resume auto \
  --seed 42

python "${PROJECT_ROOT}/scripts/select_and_evaluate_soft_slot_kt.py" \
  --feature_dir "${FEATURE_DIR}" \
  --model_name_or_path qwen/Qwen3-8B \
  --checkpoints "${CHECKPOINTS}" \
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
  "${SELECTION_ONLY_ARGS[@]}" \
  --seed 42
