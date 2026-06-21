"""Aggregate per-variant LLM direct-prediction ablation metrics into CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


SUMMARY_FIELDS = [
    "variant",
    "sample_count",
    "acc",
    "precision",
    "recall",
    "f1",
    "positive_rate_true",
    "positive_rate_pred",
    "prediction_mode",
    "probability_sample_count",
    "probability_coverage",
    "failed_count",
    "tp",
    "fp",
    "tn",
    "fn",
    "auc",
    "bce",
    "rmse",
    "note",
]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _normalise_metric_row(path: Path) -> Dict[str, Any]:
    metrics = _read_json(path)
    variant = str(metrics.get("variant") or path.parent.name)
    row: Dict[str, Any] = {field: metrics.get(field) for field in SUMMARY_FIELDS}
    row["variant"] = variant
    if row.get("auc") is None and row.get("bce") is None and row.get("rmse") is None:
        row["auc"] = None
        row["bce"] = None
        row["rmse"] = None
        row["note"] = row.get("note") or "hard-label predictions only"
    return row


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SUMMARY_FIELDS})


def _write_markdown(rows: List[Dict[str, Any]], out_md: Path, *, ablation_dir: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# LLM Direct Prediction Ablation Summary")
    lines.append("")
    lines.append(f"- ablation_dir: `{ablation_dir}`")
    has_probability_metrics = any(
        row.get("auc") is not None or row.get("bce") is not None or row.get("rmse") is not None
        for row in rows
    )
    if has_probability_metrics:
        lines.append("- probability metrics: computed per variant from valid `prob_correct` values; see table.")
    else:
        lines.append("- auc: null")
        lines.append("- bce: null")
        lines.append("- rmse: null")
        lines.append("- note: hard-label predictions only")
    lines.append("")
    lines.append("| " + " | ".join(SUMMARY_FIELDS) + " |")
    lines.append("| " + " | ".join(["---"] * len(SUMMARY_FIELDS)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(field)) for field in SUMMARY_FIELDS) + " |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_metrics(ablation_dir: Path) -> List[Dict[str, Any]]:
    """Collect one metrics.json row from each variant directory."""
    if not ablation_dir.exists():
        raise FileNotFoundError(f"Missing ablation_dir: {ablation_dir}")
    rows: List[Dict[str, Any]] = []
    for metrics_path in sorted(ablation_dir.glob("*/metrics.json")):
        rows.append(_normalise_metric_row(metrics_path))
    if not rows:
        raise FileNotFoundError(f"No */metrics.json files found under {ablation_dir}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LLM direct-prediction ablation metrics.")
    parser.add_argument("--ablation_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_md = Path(args.out_md).resolve()
    rows = collect_metrics(ablation_dir)
    _write_csv(rows, out_csv)
    _write_markdown(rows, out_md, ablation_dir=ablation_dir)
    print("[OK] llm ablation summary written")
    print(f"[CSV] {out_csv}")
    print(f"[MD] {out_md}")


if __name__ == "__main__":
    main()
