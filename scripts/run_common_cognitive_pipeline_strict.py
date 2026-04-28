from __future__ import annotations

import argparse
from pathlib import Path

from common_pipeline_strict.constants import (
    DEFAULT_OUT_ROOT,
    LLM_SUMMARY_CHUNK_SIZE,
    LLM_SUMMARY_WORKERS,
    RERANK_TOPK,
    RERANK_WEIGHT,
    TEXT_EMBED_BATCH_SIZE,
    TEXT_EMBED_MAX_LENGTH,
    TEXT_EMBED_MODEL_NAME,
    TEXT_RERANK_BATCH_SIZE,
    TEXT_RERANK_MODEL_NAME,
    TRAIN_SEED,
    USE_QWEN_RERANKER,
)
from common_pipeline_strict.io_utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    workspace = script_dir.parent
    parser.add_argument("--problem_json", default=str(workspace / "datalocal" / "problem.json"))
    parser.add_argument("--student_json", default=str(workspace / "datalocal" / "student-problem-fine.json"))
    parser.add_argument("--out_root", default=str(workspace / DEFAULT_OUT_ROOT))
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--preview_limit", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip_stage32", action="store_true")
    parser.add_argument("--stage32_core_artifacts_only", action="store_true")
    parser.add_argument("--text_embed_model", type=str, default=TEXT_EMBED_MODEL_NAME)
    parser.add_argument("--text_embed_batch_size", type=int, default=TEXT_EMBED_BATCH_SIZE)
    parser.add_argument("--text_embed_max_length", type=int, default=TEXT_EMBED_MAX_LENGTH)
    parser.add_argument("--text_rerank_model", type=str, default=TEXT_RERANK_MODEL_NAME)
    parser.add_argument("--text_rerank_batch_size", type=int, default=TEXT_RERANK_BATCH_SIZE)
    parser.add_argument("--use_qwen_reranker", action="store_true", default=USE_QWEN_RERANKER)
    parser.add_argument("--disable_qwen_reranker", action="store_true")
    parser.add_argument("--rerank_topk", type=int, default=RERANK_TOPK)
    parser.add_argument("--rerank_weight", type=float, default=RERANK_WEIGHT)
    parser.add_argument("--enable_llm_graph_completion", action="store_true")
    parser.add_argument("--enable_llm_summary", action="store_true")
    parser.add_argument("--reuse_existing_contexts", action="store_true")
    parser.add_argument("--llm_base_url", type=str, default="")
    parser.add_argument("--llm_model", type=str, default="")
    parser.add_argument("--llm_api_key", type=str, default="")
    parser.add_argument("--llm_timeout_sec", type=int, default=120)
    parser.add_argument("--llm_max_tokens", type=int, default=160)
    parser.add_argument("--llm_temperature", type=float, default=0.1)
    parser.add_argument("--llm_summary_workers", type=int, default=LLM_SUMMARY_WORKERS)
    parser.add_argument("--llm_summary_chunk_size", type=int, default=LLM_SUMMARY_CHUNK_SIZE)
    parser.add_argument("--context_shard_index", type=int, default=0)
    parser.add_argument("--context_num_shards", type=int, default=1)
    parser.add_argument("--merge_context_shards", action="store_true")
    args = parser.parse_args()

    out_root = ensure_dir(Path(args.out_root).resolve())
    priors_dir = ensure_dir(out_root / "priors")
    contexts_dir = ensure_dir(out_root / "contexts")
    reports_dir = ensure_dir(out_root / "reports")
    cache_dir = ensure_dir(out_root / "cache")

    if args.stage32_core_artifacts_only:
        from common_pipeline_strict.stage32 import run_stage32_core_artifacts

        result = run_stage32_core_artifacts(
            problem_json=Path(args.problem_json).resolve(),
            student_json=Path(args.student_json).resolve(),
            priors_dir=priors_dir,
            smoke=bool(args.smoke),
            text_embed_model=str(args.text_embed_model),
            text_embed_batch_size=int(args.text_embed_batch_size),
            text_embed_max_length=int(args.text_embed_max_length),
        )
        print("[OK] strict stage32 core artifacts finished")
        print("[SEMANTIC_IDS]", result.semantic_ids_path)
        print("[SEMANTIC_AUDIT]", result.semantic_id_audit_path)
        print("[PROBLEM_CATALOG]", result.problem_catalog_path)
        print("[ITEM_COLLABORATIVE]", result.item_collaborative_path)
        print("[MANIFEST]", result.manifest_path)
        return

    if args.skip_stage32:
        stage32_manifest = priors_dir / "stage32_manifest.json"
        if not stage32_manifest.exists():
            raise FileNotFoundError(f"--skip_stage32 was set but {stage32_manifest} does not exist")
        stage32 = type("Stage32ResultStub", (), {"manifest_path": str(stage32_manifest)})()
    else:
        from common_pipeline_strict.stage32 import run_stage32

        stage32 = run_stage32(
            problem_json=Path(args.problem_json).resolve(),
            student_json=Path(args.student_json).resolve(),
            priors_dir=priors_dir,
            seed=int(args.seed),
            smoke=bool(args.smoke),
            text_embed_model=str(args.text_embed_model),
            text_embed_batch_size=int(args.text_embed_batch_size),
            text_embed_max_length=int(args.text_embed_max_length),
            enable_llm_graph_completion=bool(args.enable_llm_graph_completion),
            llm_base_url=str(args.llm_base_url),
            llm_model=str(args.llm_model),
            llm_api_key=str(args.llm_api_key),
            llm_timeout_sec=int(args.llm_timeout_sec),
            llm_max_tokens=int(args.llm_max_tokens),
            llm_temperature=float(args.llm_temperature),
        )
    from common_pipeline_strict.stage34 import run_stage34

    stage34 = run_stage34(
        problem_json=Path(args.problem_json).resolve(),
        student_json=Path(args.student_json).resolve(),
        priors_dir=priors_dir,
        contexts_dir=contexts_dir,
        reports_dir=reports_dir,
        cache_dir=cache_dir,
        preview_limit=int(args.preview_limit),
        dry_run=bool(args.dry_run),
        smoke=bool(args.smoke),
        text_embed_model=str(args.text_embed_model),
        text_embed_batch_size=int(args.text_embed_batch_size),
        text_embed_max_length=int(args.text_embed_max_length),
        text_rerank_model=str(args.text_rerank_model),
        text_rerank_batch_size=int(args.text_rerank_batch_size),
        use_qwen_reranker=bool(args.use_qwen_reranker and not args.disable_qwen_reranker),
        rerank_topk=int(args.rerank_topk),
        rerank_weight=float(args.rerank_weight),
        enable_llm_summary=bool(args.enable_llm_summary),
        llm_base_url=str(args.llm_base_url),
        llm_model=str(args.llm_model),
        llm_api_key=str(args.llm_api_key),
        llm_timeout_sec=int(args.llm_timeout_sec),
        llm_max_tokens=int(args.llm_max_tokens),
        llm_temperature=float(args.llm_temperature),
        llm_summary_workers=int(args.llm_summary_workers),
        llm_summary_chunk_size=int(args.llm_summary_chunk_size),
        reuse_existing_contexts=bool(args.reuse_existing_contexts),
        context_shard_index=int(args.context_shard_index),
        context_num_shards=int(args.context_num_shards),
        merge_context_shards=bool(args.merge_context_shards),
    )

    print("[OK] strict common pipeline finished")
    print("[PRIORS]", stage32.manifest_path)
    if getattr(stage32, "semantic_id_audit_path", None):
        print("[SEMANTIC_AUDIT]", stage32.semantic_id_audit_path)
    print("[CONTEXTS]", stage34.contexts_path)
    print("[PREVIEW]", stage34.preview_path)
    if stage34.embeddings_path:
        print("[EMBEDDINGS]", stage34.embeddings_path)


if __name__ == "__main__":
    main()
