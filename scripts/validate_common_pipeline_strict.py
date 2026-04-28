from __future__ import annotations

import argparse
from pathlib import Path

from common_pipeline_strict.constants import DEFAULT_OUT_ROOT
from common_pipeline_strict.validation import run_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    workspace = script_dir.parent
    parser.add_argument("--problem_json", default=str(workspace / "datalocal" / "problem.json"))
    parser.add_argument("--out_root", default=str(workspace / DEFAULT_OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    result = run_validation(
        out_root=Path(args.out_root).resolve(),
        problem_json=Path(args.problem_json).resolve(),
        smoke=bool(args.smoke),
    )
    print("[OK] validation finished")
    print("[REPORT]", result.report_path)
    print("[SEMANTIC_STABLE]", result.semantic_ids_stable)
    print("[SEMANTIC_QUALITY_OK]", result.semantic_id_quality_ok)
    print("[CONTEXTS_CHECKED]", result.context_records_checked)
    print("[FAILURES]", len(result.failures))
    for failure in result.failures:
        print(" -", failure)


if __name__ == "__main__":
    main()
