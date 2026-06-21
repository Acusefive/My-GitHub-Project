#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SAMPLE_SIZE="${SAMPLE_SIZE:-50000}"
SEED="${SEED:-42}"
K="${K:-4}"
WORKERS="${WORKERS:-4}"
MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-0}"
FLUSH_EVERY="${FLUSH_EVERY:-100}"
MAX_TOKENS="${MAX_TOKENS:-32}"
SUMMARY_MAX_TOKENS="${SUMMARY_MAX_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0}"
PREDICTION_MODE="${PREDICTION_MODE:-token_logprob}"
LOGPROB_TOP_K="${LOGPROB_TOP_K:-20}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8002/v1}"
LLM_MODEL="${LLM_MODEL:-qwen3-8b}"
LLM_TIMEOUT_SEC="${LLM_TIMEOUT_SEC:-60}"
RETRIES="${RETRIES:-1}"
LLM_USE_CHAT_TEMPLATE_KWARGS="${LLM_USE_CHAT_TEMPLATE_KWARGS:-0}"
SAVE_PROMPTS="${SAVE_PROMPTS:-0}"
DATASETS="${DATASETS:-moocradar xes3g5m foundationalassist}"
VARIANTS="${VARIANTS:-full_cognitive_rag_llm,wo_cognitive_retrieval_recent,wo_llm_summary,wo_structured_evidence}"
OUT_PREFIX="${OUT_PREFIX:-out/paper_llm_ablation_tokenlogprob_sample${SAMPLE_SIZE}}"
MANIFEST_DIR="${MANIFEST_DIR:-out/paper_llm_ablation_eval_manifests}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing file: $1" >&2
    exit 1
  fi
}

dataset_paths() {
  local dataset="$1"
  case "$dataset" in
    moocradar)
      echo "datalocal/problem.json|datalocal/student-problem-fine.json|out/strict_common_pipeline/contexts/contexts.jsonl|stage34_soft_slot_qwen_kt/results/moocradar/context_target_slots_only_lr5e5/predictions.test.full.jsonl|${OUT_PREFIX}_moocradar"
      ;;
    xes3g5m)
      echo "datalocal/xes3g5m_strict_core20/problem.json|datalocal/xes3g5m_strict_core20/student-problem-fine.json|out/xes3g5m_strict_core20_common_pipeline/contexts/contexts.jsonl|stage34_soft_slot_qwen_kt/results/xes3g5m/context_target_slots_only/predictions.test.full.jsonl|${OUT_PREFIX}_xes3g5m"
      ;;
    foundationalassist)
      echo "datalocal/foundationalassist_text_only_core200_contextcomplete/problem.json|datalocal/foundationalassist_text_only_core200_contextcomplete/student-problem-fine.json|out/foundationalassist_text_only_core200_contextcomplete_common_pipeline/contexts/contexts.jsonl|stage34_soft_slot_qwen_kt/results/foundationalassist/context_target_slots_only/predictions.test.full.jsonl|${OUT_PREFIX}_foundationalassist"
      ;;
    *)
      echo "Unknown dataset: ${dataset}" >&2
      exit 2
      ;;
  esac
}

run_one_dataset() {
  local dataset="$1"
  local spec problem_json student_json contexts_jsonl test_predictions out_dir eval_manifest recent_summaries
  local -a template_args=()
  local -a prompt_audit_args=()
  if [[ "$LLM_USE_CHAT_TEMPLATE_KWARGS" == "1" ]]; then
    template_args+=(--llm_use_chat_template_kwargs)
  fi
  if [[ "$SAVE_PROMPTS" == "1" ]]; then
    prompt_audit_args+=(--save_prompts)
  fi
  spec="$(dataset_paths "$dataset")"
  IFS='|' read -r problem_json student_json contexts_jsonl test_predictions out_dir <<< "$spec"
  eval_manifest="${MANIFEST_DIR}/${dataset}.test.sample${SAMPLE_SIZE}.seed${SEED}.jsonl"
  recent_summaries="${out_dir}/recent_evidence_llm_summaries.jsonl"

  require_file "$problem_json"
  require_file "$student_json"
  require_file "$contexts_jsonl"
  require_file "$test_predictions"

  if [[ ! -f "$eval_manifest" ]]; then
    python scripts/build_llm_ablation_eval_manifest.py \
      --test_predictions_jsonl "$test_predictions" \
      --dataset_name "$dataset" \
      --out_manifest_jsonl "$eval_manifest" \
      --sample_size "$SAMPLE_SIZE" \
      --seed "$SEED"
  else
    echo "[RESUME] frozen manifest: ${eval_manifest}"
  fi

  echo "===== PAPER LLM ABLATION START: ${dataset} ====="
  python scripts/build_recent_evidence_llm_summaries.py \
    --problem_json "$problem_json" \
    --student_json "$student_json" \
    --contexts_jsonl "$contexts_jsonl" \
    --eval_manifest_jsonl "$eval_manifest" \
    --out_summary_jsonl "$recent_summaries" \
    --k "$K" \
    --llm_base_url "$LLM_BASE_URL" \
    --llm_model "$LLM_MODEL" \
    --workers "$WORKERS" \
    --max_in_flight "$MAX_IN_FLIGHT" \
    --llm_timeout_sec "$LLM_TIMEOUT_SEC" \
    --summary_max_tokens "$SUMMARY_MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --flush_every "$FLUSH_EVERY" \
    --llm_disable_thinking \
    "${template_args[@]}" \
    --resume

  python scripts/run_llm_ablation_experiments.py \
    --problem_json "$problem_json" \
    --student_json "$student_json" \
    --contexts_jsonl "$contexts_jsonl" \
    --out_dir "$out_dir" \
    --variants "$VARIANTS" \
    --k "$K" \
    --eval_manifest_jsonl "$eval_manifest" \
    --recent_summary_jsonl "$recent_summaries" \
    --llm_base_url "$LLM_BASE_URL" \
    --llm_model "$LLM_MODEL" \
    --workers "$WORKERS" \
    --max_in_flight "$MAX_IN_FLIGHT" \
    --temperature "$TEMPERATURE" \
    --max_tokens "$MAX_TOKENS" \
    --prediction_mode "$PREDICTION_MODE" \
    --logprob_top_k "$LOGPROB_TOP_K" \
    --seed "$SEED" \
    --llm_timeout_sec "$LLM_TIMEOUT_SEC" \
    --retries "$RETRIES" \
    --flush_every "$FLUSH_EVERY" \
    --llm_disable_thinking \
    "${template_args[@]}" \
    "${prompt_audit_args[@]}" \
    --resume

  python scripts/summarize_llm_ablation_results.py \
    --ablation_dir "$out_dir" \
    --out_csv "${out_dir}/ablation_summary.csv" \
    --out_md "${out_dir}/ablation_summary.md"
  echo "===== PAPER LLM ABLATION DONE: ${dataset} ====="
}

for dataset in ${DATASETS}; do
  run_one_dataset "$dataset"
done

echo "[OK] paper LLM ablations completed for: ${DATASETS}"
