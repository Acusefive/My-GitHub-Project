from __future__ import annotations

import argparse
from pathlib import Path

from common_pipeline_strict.constants import DEFAULT_OUT_ROOT, TRAIN_SEED
from common_pipeline_strict.io_utils import ensure_dir
from common_pipeline_strict.stage32 import run_stage32
from common_pipeline_strict.stage34 import run_stage34


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
    parser.add_argument("--enable_llm_summary", action="store_true")
    parser.add_argument("--reuse_existing_contexts", action="store_true")
    parser.add_argument("--llm_base_url", type=str, default="")
    parser.add_argument("--llm_model", type=str, default="")
    parser.add_argument("--llm_api_key", type=str, default="")
    parser.add_argument("--llm_timeout_sec", type=int, default=120)
    parser.add_argument("--llm_max_tokens", type=int, default=160)
    parser.add_argument("--llm_temperature", type=float, default=0.1)
    args = parser.parse_args()

    out_root = ensure_dir(Path(args.out_root).resolve())
    priors_dir = ensure_dir(out_root / "priors")
    contexts_dir = ensure_dir(out_root / "contexts")
    reports_dir = ensure_dir(out_root / "reports")
    cache_dir = ensure_dir(out_root / "cache")

    if args.skip_stage32:
        stage32_manifest = priors_dir / "stage32_manifest.json"
        if not stage32_manifest.exists():
            raise FileNotFoundError(f"--skip_stage32 was set but {stage32_manifest} does not exist")
        stage32 = type("Stage32ResultStub", (), {"manifest_path": str(stage32_manifest)})()
    else:
        stage32 = run_stage32(
            problem_json=Path(args.problem_json).resolve(),
            student_json=Path(args.student_json).resolve(),
            priors_dir=priors_dir,
            seed=int(args.seed),
            smoke=bool(args.smoke),
        )
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
        enable_llm_summary=bool(args.enable_llm_summary),
        llm_base_url=str(args.llm_base_url),
        llm_model=str(args.llm_model),
        llm_api_key=str(args.llm_api_key),
        llm_timeout_sec=int(args.llm_timeout_sec),
        llm_max_tokens=int(args.llm_max_tokens),
        llm_temperature=float(args.llm_temperature),
        reuse_existing_contexts=bool(args.reuse_existing_contexts),
    )

    print("[OK] strict common pipeline finished")
    print("[PRIORS]", stage32.manifest_path)
    print("[CONTEXTS]", stage34.contexts_path)
    print("[PREVIEW]", stage34.preview_path)
    if stage34.embeddings_path:
        print("[EMBEDDINGS]", stage34.embeddings_path)


if __name__ == "__main__":
    main()
