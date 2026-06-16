from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from soft_slot_kt.prepare import prepare_existing_stage34_features
from soft_slot_kt.utils import parse_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert existing Stage32/Stage34 artifacts into a Soft-Slot feature store.")
    parser.add_argument("--context_embeddings_path", type=Path, required=True)
    parser.add_argument("--contexts_path", type=Path, required=True)
    parser.add_argument("--priors_dir", type=Path, required=True)
    parser.add_argument("--student_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--context_fields", default="llm_embeddings,llm_struct_features,stage34_numeric")
    parser.add_argument("--target_fields", default="hqtext,hqid,semantic,collaborative")
    parser.add_argument(
        "--prompt_context_field",
        choices=["auto", "main_context_text", "template_context_text", "llm_context_text"],
        default="auto",
    )
    parser.add_argument("--max_context_chars", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_concept_ratio", type=float, default=0.8)
    parser.add_argument("--valid_concept_ratio", type=float, default=0.0)
    parser.add_argument("--chunk_rows", type=int, default=2048)
    args = parser.parse_args()

    manifest = prepare_existing_stage34_features(
        context_embeddings_path=args.context_embeddings_path.resolve(),
        contexts_path=args.contexts_path.resolve(),
        priors_dir=args.priors_dir.resolve(),
        student_json=args.student_json.resolve(),
        output_dir=args.output_dir.resolve(),
        context_fields=parse_csv(args.context_fields),
        target_fields=parse_csv(args.target_fields),
        prompt_context_field=str(args.prompt_context_field),
        max_context_chars=int(args.max_context_chars),
        seed=int(args.seed),
        test_concept_ratio=float(args.test_concept_ratio),
        valid_concept_ratio=float(args.valid_concept_ratio),
        chunk_rows=int(args.chunk_rows),
    )
    print("[OK] Soft-Slot feature store prepared")
    print("[MANIFEST]", args.output_dir.resolve() / "feature_manifest.json")
    print(json.dumps(manifest["audit"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
