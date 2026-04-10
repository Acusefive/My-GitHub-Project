"""
Unified pipeline for the common, model-agnostic part of the paper:
  3.2 multi-source cognitive priors
  3.3 two-stage cognitive evidence retrieval
  3.4 hierarchical cognitive context construction

Outputs are organized into two shared directories:
  - datalocal/: reusable structured priors
  - cachelocal/: reusable context/embedding caches

All downstream baselines should consume these shared artifacts instead of
re-implementing retrieval or context building inside each baseline.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def run_step(cmd: List[str], cwd: Path) -> None:
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    workspace = script_dir.parent
    default_data_dir = workspace / "datalocal"
    default_cache_dir = workspace / "cachelocal"

    parser.add_argument("--problem_json", default=str(default_data_dir / "problem.json"))
    parser.add_argument("--student_json", default=str(default_data_dir / "student-problem-fine.json"))
    parser.add_argument("--artifacts_dir", default=str(default_data_dir))
    parser.add_argument("--cache_dir", default=str(default_cache_dir))

    parser.add_argument("--skip_stage32", action="store_true", help="skip 3.2 prior construction")
    parser.add_argument("--skip_stage33_34", action="store_true", help="skip 3.3-3.4 retrieval/context build")

    parser.add_argument("--do_llm_graph", action="store_true", help="enable LLM graph-edge completion")
    parser.add_argument("--do_llm_summary", action="store_true", help="enable LLM summary generation")
    parser.add_argument("--api_key", default="", help="shared API key for optional LLM stages")
    parser.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1/")
    parser.add_argument("--model_name", default="deepseek-v3.2-exp")

    parser.add_argument("--top_k", type=int, default=50, help="top-level KMeans cluster count")
    parser.add_argument("--sub_k", type=int, default=5, help="second-level KMeans cluster count")
    parser.add_argument("--collab_topk", type=int, default=5)
    parser.add_argument("--collab_vec_dim", type=int, default=64)

    parser.add_argument("--topk1", type=int, default=10, help="stage-1 retrieval candidate size")
    parser.add_argument("--K", type=int, default=4, help="stage-2 final evidence size")
    parser.add_argument("--preview_limit", type=int, default=50)
    parser.add_argument("--dry_run_embeddings", action="store_true", help="build texts only, skip BGE encoding")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    problem_json = Path(args.problem_json).resolve()
    student_json = Path(args.student_json).resolve()

    if not problem_json.exists():
        raise FileNotFoundError(f"problem json not found: {problem_json}")
    if not student_json.exists():
        raise FileNotFoundError(f"student json not found: {student_json}")

    preprocess_gram = script_dir / "preprocess_gram.py"
    build_graph = script_dir / "build_concept_graph_edges.py"
    build_context = script_dir / "preprocess_embeddings_cognitive_rag.py"

    semantic_ids = artifacts_dir / "item_semantic_ids.json"
    semantic_vectors = artifacts_dir / "item_semantic_vectors.pkl"
    mu_q = artifacts_dir / "problem_mu_q.json"
    concept_pc1 = artifacts_dir / "concept_pc1_dirs.pkl"
    lr_params = artifacts_dir / "e_diag_lr_params.json"
    collab_json = artifacts_dir / "item_collaborative.json"
    collab_vecs = artifacts_dir / "item_collaborative_embeddings.pkl"
    concept_edges = artifacts_dir / "concept_graph_edges.csv"

    context_embeddings = cache_dir / "cognitive_embeddings.pkl"
    context_texts = cache_dir / "cognitive_context_texts.pkl"
    preview_txt = cache_dir / "context_text_preview.txt"
    summary_cache = cache_dir / "summary_cache.json"

    if not args.skip_stage32:
        run_step(
            [
                sys.executable,
                str(preprocess_gram),
                "--problem_json", str(problem_json),
                "--student_json", str(student_json),
                "--out_semantic_ids", str(semantic_ids),
                "--out_semantic_vectors", str(semantic_vectors),
                "--out_mu_q", str(mu_q),
                "--out_concept_pc1", str(concept_pc1),
                "--out_lr_params", str(lr_params),
                "--out_collab", str(collab_json),
                "--out_collab_vecs", str(collab_vecs),
                "--top_k", str(args.top_k),
                "--sub_k", str(args.sub_k),
                "--collab_topk", str(args.collab_topk),
                "--collab_vec_dim", str(args.collab_vec_dim),
            ],
            workspace,
        )

        graph_cmd = [
            sys.executable,
            str(build_graph),
            "--problem_json", str(problem_json),
            "--student_json", str(student_json),
            "--out_csv", str(concept_edges),
            "--base_url", args.base_url,
            "--model_name", args.model_name,
        ]
        if args.do_llm_graph:
            graph_cmd.extend(["--do_llm", "--api_key", args.api_key])
        run_step(graph_cmd, workspace)

    if not args.skip_stage33_34:
        context_cmd = [
            sys.executable,
            str(build_context),
            "--problem_json", str(problem_json),
            "--student_json", str(student_json),
            "--semantic_ids_json", str(semantic_ids),
            "--semantic_vectors_pkl", str(semantic_vectors),
            "--collab_json", str(collab_json),
            "--collab_vecs_pkl", str(collab_vecs),
            "--lr_params_json", str(lr_params),
            "--concept_edges_csv", str(concept_edges),
            "--out_pkl", str(context_embeddings),
            "--out_text_pkl", str(context_texts),
            "--out_preview_txt", str(preview_txt),
            "--summary_cache_path", str(summary_cache),
            "--topk1", str(args.topk1),
            "--K", str(args.K),
            "--preview_limit", str(args.preview_limit),
        ]
        if args.dry_run_embeddings:
            context_cmd.append("--dry_run")
        if args.do_llm_summary:
            context_cmd.extend([
                "--do_llm",
                "--api_key", args.api_key,
                "--base_url", args.base_url,
                "--llm_model", args.model_name,
            ])
        run_step(context_cmd, workspace)

    print("\n[OK] Common pipeline finished.")
    print("[DATA]")
    print(" ", semantic_ids)
    print(" ", semantic_vectors)
    print(" ", mu_q)
    print(" ", lr_params)
    print(" ", collab_json)
    print(" ", collab_vecs)
    print(" ", concept_edges)
    print("[CACHE]")
    print(" ", context_embeddings)
    print(" ", context_texts)
    print(" ", preview_txt)


if __name__ == "__main__":
    main()
