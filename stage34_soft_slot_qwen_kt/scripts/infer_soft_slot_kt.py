from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from soft_slot_kt.cli import add_model_arguments
from soft_slot_kt.prompts import PROMPT_VERSION
from soft_slot_kt.runtime import build_model_and_data, evaluate, load_checkpoint
from soft_slot_kt.utils import compute_metrics, ensure_dir, set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Soft-Slot Qwen KT inference.")
    add_model_arguments(parser)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--resume_predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    set_seed(int(args.seed))
    output_dir = ensure_dir(args.output_dir.resolve())
    device, _tokenizer, model, label_spec, dataset, _collator, loader = build_model_and_data(args, args.split)
    if args.checkpoint is not None:
        load_checkpoint(args.checkpoint.resolve(), model=model)
    elif model.trainable_parameters():
        raise ValueError("--checkpoint is required when the selected slot mode contains trainable projectors")

    predictions_path = output_dir / f"predictions.{args.split}.jsonl"
    done_rows = set()
    if predictions_path.exists() and args.resume_predictions and not args.force:
        with predictions_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    done_rows.add(int(json.loads(line)["row"]))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
        if done_rows:
            dataset.indices = np.asarray([row for row in dataset.indices if int(row) not in done_rows], dtype=np.int64)
            print(f"[RESUME] completed_predictions={len(done_rows)} pending={len(dataset)}")
    evaluate(
        model,
        loader,
        label_spec,
        device,
        predictions_path=predictions_path,
        append_predictions=bool(done_rows),
        desc=f"infer {args.split}",
    )
    all_labels = []
    all_probabilities = []
    with predictions_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                all_labels.append(int(row["label"]))
                all_probabilities.append(float(row["probability"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    metrics = compute_metrics(all_labels, all_probabilities)
    metrics.update(
        {
            "split": args.split,
            "slot_mode": args.slot_mode,
            "prompt_version": PROMPT_VERSION,
            "feature_protocol": dataset.feature_store.manifest["protocol"],
            "feature_audit": dataset.feature_store.manifest["audit"],
            "final_inference_uses_downstream_kt_baseline": False,
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint is not None else None,
            "label_spec": label_spec.to_dict(),
            "predictions_path": str(predictions_path),
        }
    )
    metrics_path = output_dir / f"metrics.{args.split}.json"
    write_json(metrics, metrics_path)
    print("[OK] Soft-Slot inference finished")
    print("[METRICS]", metrics_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
