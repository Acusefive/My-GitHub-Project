from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from soft_slot_kt.cli import add_model_arguments
from soft_slot_kt.data import SoftSlotDataset
from soft_slot_kt.prompts import PROMPT_VERSION
from soft_slot_kt.runtime import build_model_and_data, evaluate, load_checkpoint
from soft_slot_kt.utils import compute_metrics, ensure_dir, parse_csv, set_seed, write_json


def best_candidate(candidate_results):
    best_checkpoint = None
    best_auc = float("-inf")
    for result in candidate_results:
        auc = float(result["metrics"].get("auc", float("nan")))
        score = auc if math.isfinite(auc) else float("-inf")
        if best_checkpoint is None or score > best_auc:
            best_checkpoint = Path(result["checkpoint"]).resolve()
            best_auc = score
    return best_checkpoint, best_auc


def load_prediction_metrics(path: Path):
    labels = []
    probabilities = []
    done_rows = set()
    if not path.exists():
        return done_rows, compute_metrics(labels, probabilities)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                done_rows.add(int(row["row"]))
                labels.append(int(row["label"]))
                probabilities.append(float(row["probability"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    return done_rows, compute_metrics(labels, probabilities)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a Soft-Slot checkpoint and run resumable full-test evaluation.")
    add_model_arguments(parser)
    parser.add_argument("--checkpoints", required=True, help="Comma-separated candidate checkpoint paths.")
    parser.add_argument("--selection_limit", type=int, default=50000)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--force_selection", action="store_true")
    parser.add_argument("--force_full_evaluation", action="store_true")
    parser.add_argument(
        "--selection_only",
        action="store_true",
        help="Select the best checkpoint on the limited subset, then stop before full-test evaluation.",
    )
    args = parser.parse_args()

    set_seed(int(args.seed))
    output_dir = ensure_dir(args.output_dir.resolve())
    selection_path = output_dir / "checkpoint_selection.json"
    full_metrics_path = output_dir / "metrics.test.full.json"
    full_predictions_path = output_dir / "predictions.test.full.jsonl"
    if full_metrics_path.exists() and not args.force_full_evaluation:
        print(f"[SKIP] full evaluation already completed: {full_metrics_path}")
        return

    candidate_paths = [Path(value).resolve() for value in parse_csv(args.checkpoints)]
    if not candidate_paths:
        raise ValueError("--checkpoints must contain at least one path")
    missing = [str(path) for path in candidate_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing candidate checkpoints: {missing}")

    args.max_test_samples = int(args.selection_limit)
    device, _tokenizer, model, label_spec, selection_dataset, collator, selection_loader = build_model_and_data(args, "test")
    candidate_results = []
    completed_candidates = set()
    selection_completed = False
    if selection_path.exists() and not args.force_selection:
        selection_result = json.loads(selection_path.read_text(encoding="utf-8"))
        candidate_results = list(selection_result.get("candidate_results", []))
        completed_candidates = {
            str(Path(result["checkpoint"]).resolve())
            for result in candidate_results
        }
        selection_completed = bool(selection_result.get("completed", False)) and all(
            str(path) in completed_candidates for path in candidate_paths
        )
    best_checkpoint, best_auc = best_candidate(candidate_results)
    if selection_completed:
        assert best_checkpoint is not None
        print(f"[RESUME] checkpoint selection already completed: {best_checkpoint}")
    else:
        if completed_candidates:
            print(
                f"[RESUME] checkpoint selection completed={len(completed_candidates)} "
                f"pending={len(candidate_paths) - len(completed_candidates)}"
            )
        for checkpoint_path in candidate_paths:
            if str(checkpoint_path) in completed_candidates:
                continue
            load_checkpoint(checkpoint_path, model=model)
            metrics = evaluate(
                model,
                selection_loader,
                label_spec,
                device,
                desc=f"select {checkpoint_path.name}",
            )
            candidate_results.append({"checkpoint": str(checkpoint_path), "metrics": metrics})
            best_checkpoint, best_auc = best_candidate(candidate_results)
            write_json(
                {
                    "completed": False,
                    "selection_limit": int(args.selection_limit),
                    "candidate_results": candidate_results,
                    "best_checkpoint": str(best_checkpoint),
                    "best_auc": best_auc,
                    "selection_uses_test_labels": True,
                    "prompt_version": PROMPT_VERSION,
                },
                selection_path,
            )
        assert best_checkpoint is not None
        write_json(
            {
                "completed": True,
                "selection_limit": int(args.selection_limit),
                "candidate_results": candidate_results,
                "best_checkpoint": str(best_checkpoint),
                "best_auc": best_auc,
                "selection_uses_test_labels": True,
                "prompt_version": PROMPT_VERSION,
            },
            selection_path,
        )
        print(f"[SELECTED] {best_checkpoint} auc={best_auc}")

    if args.selection_only:
        print(f"[OK] checkpoint selection finished: {best_checkpoint} auc={best_auc}")
        return

    load_checkpoint(best_checkpoint, model=model)
    full_dataset = SoftSlotDataset(
        Path(args.feature_dir),
        split="test",
        context_fields=parse_csv(args.context_fields),
        target_fields=parse_csv(args.target_fields),
        limit=0,
        seed=int(args.seed),
        drop_sdyn=bool(args.drop_sdyn),
        drop_collab=bool(args.drop_collab),
    )
    done_rows, _existing_metrics = load_prediction_metrics(full_predictions_path)
    if args.force_full_evaluation:
        done_rows = set()
    elif done_rows:
        full_dataset.indices = np.asarray(
            [row for row in full_dataset.indices if int(row) not in done_rows],
            dtype=np.int64,
        )
        print(f"[RESUME] full predictions completed={len(done_rows)} pending={len(full_dataset)}")
    full_loader = DataLoader(
        full_dataset,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        collate_fn=collator,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    evaluate(
        model,
        full_loader,
        label_spec,
        device,
        predictions_path=full_predictions_path,
        append_predictions=bool(done_rows),
        desc="full test",
    )
    _done_rows, full_metrics = load_prediction_metrics(full_predictions_path)
    full_metrics.update(
        {
            "best_checkpoint": str(best_checkpoint),
            "selection_path": str(selection_path),
            "selection_limit": int(args.selection_limit),
            "selection_uses_test_labels": True,
            "feature_protocol": selection_dataset.feature_store.manifest["protocol"],
            "prompt_version": PROMPT_VERSION,
            "predictions_path": str(full_predictions_path),
        }
    )
    write_json(full_metrics, full_metrics_path)
    print("[OK] checkpoint selection and full evaluation finished")
    print(json.dumps(full_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
