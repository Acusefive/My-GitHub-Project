from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from scripts.common_pipeline_strict.io_utils import load_problem_records


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def review_row(record: Dict[str, Any], sample_id: int, problem_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    target_pid = str(record.get("target_pid", ""))
    target_meta = problem_map.get(target_pid, {})
    summary_fields = record.get("summary_fields") or {}
    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "user_id": record.get("user_id", ""),
        "target_t": record.get("target_t", ""),
        "target_pid": target_pid,
        "target_semantic_id": record.get("target_semantic_id", ""),
        "target_question_text": target_meta.get("text", ""),
        "target_concepts": "、".join(target_meta.get("concepts") or []),
        "target_cognitive_dimension": target_meta.get("cognitive_dimension", ""),
        "stage1_candidate_count": record.get("stage1_candidate_count", ""),
        "selected_count": record.get("selected_count", ""),
        "summary_text": summary_fields.get("summary_text", ""),
        "llm_summary_text": summary_fields.get("llm_summary_text", ""),
        "main_context_text": record.get("main_context_text", ""),
        "template_context_text": record.get("template_context_text", ""),
        "llm_context_text": record.get("llm_context_text", ""),
        "relevance_score_1to5": "",
        "coverage_score_1to5": "",
        "llm_quality_score_1to5": "",
        "interpretability_score_1to5": "",
        "redundancy_score_1to5": "",
        "overall_judgment": "",
        "notes": "",
    }

    evidence_list = list(record.get("evidence_list") or [])
    for idx in range(6):
        evidence = evidence_list[idx] if idx < len(evidence_list) else {}
        prefix = f"evidence_{idx + 1}"
        row[f"{prefix}_problem_id"] = evidence.get("problem_id", "")
        row[f"{prefix}_role"] = evidence.get("role", "")
        row[f"{prefix}_knowledge_overlap"] = evidence.get("knowledge_overlap", "")
        row[f"{prefix}_level_diff"] = evidence.get("level_diff", "")
        row[f"{prefix}_answer_result"] = evidence.get("answer_result", "")
        row[f"{prefix}_support_score"] = evidence.get("support_score", "")
        row[f"{prefix}_question_text"] = evidence.get("question_text", "")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    workspace = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--contexts_jsonl",
        default=str(workspace / "out" / "strict_common_pipeline" / "contexts" / "contexts.jsonl"),
    )
    parser.add_argument(
        "--output_csv",
        default=str(workspace / "out" / "strict_common_pipeline" / "reports" / "manual_review_sample_100.csv"),
    )
    parser.add_argument(
        "--output_jsonl",
        default=str(workspace / "out" / "strict_common_pipeline" / "reports" / "manual_review_sample_100.jsonl"),
    )
    parser.add_argument(
        "--problem_json",
        default=str(workspace / "datalocal" / "problem.json"),
    )
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    contexts_path = Path(args.contexts_jsonl).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    problem_json = Path(args.problem_json).resolve()

    records = load_jsonl(contexts_path)
    if not records:
        raise ValueError(f"No records found in {contexts_path}")
    problem_map = {
        record.problem_id: {
            "text": record.text,
            "concepts": list(record.concepts),
            "cognitive_dimension": record.cognitive_dimension,
        }
        for record in load_problem_records(problem_json)
    }

    sample_size = min(int(args.sample_size), len(records))
    rng = random.Random(int(args.seed))
    sampled_records = rng.sample(records, sample_size)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows = [review_row(record, sample_id=idx + 1, problem_map=problem_map) for idx, record in enumerate(sampled_records)]
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with output_jsonl.open("w", encoding="utf-8") as f:
        for record in sampled_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("[OK] manual review sample created")
    print("[CSV]", output_csv)
    print("[JSONL]", output_jsonl)
    print("[SAMPLE_SIZE]", sample_size)
    print("[SEED]", int(args.seed))


if __name__ == "__main__":
    main()
