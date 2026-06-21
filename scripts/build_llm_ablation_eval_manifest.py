from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _sample_id(row: Dict[str, Any]) -> str:
    return f"{str(row['user_id'])}\t{int(row['target_t'])}\t{str(row['target_pid'])}"


def _read_test_rows(path: Path, dataset_name: str) -> List[Dict[str, Any]]:
    required = {"user_id", "target_t", "target_pid", "label"}
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"Expected object in {path} line {line_no}")
            missing = sorted(required - set(raw))
            if missing:
                raise ValueError(f"Missing {missing} in {path} line {line_no}")
            label = int(raw["label"])
            if label not in {0, 1}:
                raise ValueError(f"Invalid binary label {label!r} in {path} line {line_no}")
            split = str(raw.get("split") or "test")
            if split != "test":
                raise ValueError(f"Expected split=test in {path} line {line_no}, got {split!r}")
            sample_id = _sample_id(raw)
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id in {path} line {line_no}: {sample_id}")
            seen.add(sample_id)
            rows.append(
                {
                    "sample_id": sample_id,
                    "dataset": dataset_name,
                    "row": int(raw.get("row", line_no - 1)),
                    "user_id": str(raw["user_id"]),
                    "target_t": int(raw["target_t"]),
                    "target_pid": str(raw["target_pid"]),
                    "y_true": label,
                    "split": "test",
                }
            )
    if not rows:
        raise ValueError(f"No prediction rows found in {path}")
    return rows


def _allocate_stratified_counts(group_sizes: Dict[int, int], sample_size: int) -> Dict[int, int]:
    total = sum(group_sizes.values())
    target = min(int(sample_size), total)
    raw = {label: float(size) * float(target) / float(total) for label, size in group_sizes.items() if size > 0}
    alloc = {label: min(group_sizes[label], int(math.floor(value))) for label, value in raw.items()}
    for label, size in group_sizes.items():
        if target >= len(raw) and size > 0 and alloc.get(label, 0) == 0:
            alloc[label] = 1
    remaining = target - sum(alloc.values())
    order = sorted(raw, key=lambda label: (raw[label] - math.floor(raw[label]), group_sizes[label]), reverse=True)
    while remaining > 0:
        progressed = False
        for label in order:
            if remaining <= 0:
                break
            if alloc[label] >= group_sizes[label]:
                continue
            alloc[label] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    return alloc


def _select_rows(rows: List[Dict[str, Any]], sample_size: int, seed: int) -> List[Dict[str, Any]]:
    if int(sample_size) <= 0 or int(sample_size) >= len(rows):
        return rows
    groups: Dict[int, List[Dict[str, Any]]] = {0: [], 1: []}
    for row in rows:
        groups[int(row["y_true"])].append(row)
    allocation = _allocate_stratified_counts({label: len(items) for label, items in groups.items()}, int(sample_size))
    rng = random.Random(int(seed))
    selected_ids: set[str] = set()
    for label, items in groups.items():
        take = int(allocation.get(label, 0))
        selected_ids.update(row["sample_id"] for row in rng.sample(items, k=take))
    return [row for row in rows if row["sample_id"] in selected_ids]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a frozen held-out evaluation manifest for LLM ablations from main-test predictions."
    )
    parser.add_argument("--test_predictions_jsonl", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--out_manifest_jsonl", required=True)
    parser.add_argument("--sample_size", type=int, default=0, help="Label-stratified sample size; 0 keeps all test rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = Path(args.test_predictions_jsonl).resolve()
    output = Path(args.out_manifest_jsonl).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Missing main-test predictions: {source}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest already exists: {output}. Use --overwrite only when intentionally rebuilding it.")

    source_rows = _read_test_rows(source, str(args.dataset_name))
    selected_rows = _select_rows(source_rows, int(args.sample_size), int(args.seed))
    _write_jsonl(selected_rows, output)
    metadata = {
        "dataset": str(args.dataset_name),
        "source_predictions": str(source),
        "source_row_count": len(source_rows),
        "selected_row_count": len(selected_rows),
        "sample_size_requested": int(args.sample_size),
        "sample_strategy": "label_stratified" if int(args.sample_size) > 0 else "all_test_rows",
        "seed": int(args.seed),
        "label_counts": {
            "0": sum(int(row["y_true"]) == 0 for row in selected_rows),
            "1": sum(int(row["y_true"]) == 1 for row in selected_rows),
        },
        "manifest_sha256": _sha256(output),
    }
    metadata_path = output.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[OK] held-out evaluation manifest written")
    print(f"[MANIFEST] {output}")
    print(f"[METADATA] {metadata_path}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
