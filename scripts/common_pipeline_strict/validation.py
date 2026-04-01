from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .constants import K1_DEFAULT, K2_DEFAULT, QUESTION_TEXT_ELLIPSIS, QUESTION_TEXT_LIMIT, ROLE_LABELS, SUPPORT_SCORE_DECIMALS
from .io_utils import ProblemRecord, write_json
from .stage32 import build_semantic_ids


@dataclass
class ValidationResult:
    report_path: str
    semantic_ids_stable: bool
    context_records_checked: int
    failures: List[str]


def _load_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def run_validation(
    *,
    out_root: Path,
    problem_json: Path,
    smoke: bool,
) -> ValidationResult:
    priors_dir = out_root / "priors"
    contexts_dir = out_root / "contexts"
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    stage32_manifest = json.loads((priors_dir / "stage32_manifest.json").read_text(encoding="utf-8"))
    stage34_manifest = json.loads((contexts_dir / "stage34_manifest.json").read_text(encoding="utf-8"))
    defaults = json.loads((priors_dir / "implementation_defaults.json").read_text(encoding="utf-8"))
    training_report = json.loads((priors_dir / "training_report.json").read_text(encoding="utf-8"))
    graph_bundle = json.loads((priors_dir / "concept_graph_bundle.json").read_text(encoding="utf-8"))
    semantic_ids = json.loads((priors_dir / "semantic_ids.json").read_text(encoding="utf-8"))

    failures: List[str] = []

    problem_catalog = _load_jsonl(priors_dir / "problem_catalog.jsonl")
    contexts = _load_jsonl(contexts_dir / "contexts.jsonl", limit=256 if smoke else None)
    with (priors_dir / "hqtext_vectors.pkl").open("rb") as f:
        hqtext_map = pickle.load(f)

    # semantic id determinism on saved hqtext vectors
    recompute_problems = [
        ProblemRecord(
            problem_id=str(record["problem_id"]),
            text=str(record["text"]),
            title=str(record["title"]),
            chapter=str(record["chapter"]),
            location=str(record["location"]),
            cognitive_dimension=int(record["cognitive_dimension"]),
            concepts=[str(c) for c in record["concepts"]],
        )
        for record in problem_catalog
    ]
    ordered_hqtext = [hqtext_map[problem.problem_id] for problem in recompute_problems]
    recomputed_path = reports_dir / "semantic_ids_recomputed.json"
    recomputed_ids, _semantic_texts = build_semantic_ids(
        recompute_problems,
        text_vectors=np.stack(ordered_hqtext, axis=0),
        semantic_ids_path=recomputed_path,
    )
    semantic_ids_stable = recomputed_ids == semantic_ids
    if not semantic_ids_stable:
        failures.append("semantic_ids_not_stable")

    # stage 3.2 checks
    if len(semantic_ids) != len(problem_catalog):
        failures.append("semantic_id_coverage_mismatch")
    if defaults.get("kglobal") != 50 or defaults.get("klocal") != 5:
        failures.append("semantic_id_defaults_mismatch")
    if defaults.get("ctfidf_max_features") != 5000:
        failures.append("ctfidf_default_mismatch")
    if not graph_bundle.get("has_explicit_prerequisite") is False:
        failures.append("graph_has_unexpected_prerequisite")
    if graph_bundle.get("e_pre") != []:
        failures.append("graph_prerequisite_not_empty")
    if not training_report.get("history"):
        failures.append("training_history_missing")
    else:
        val_losses = [float(item["val_loss"]) for item in training_report["history"]]
        if min(val_losses) > val_losses[0] + 1e-8:
            failures.append("training_val_not_improved")

    # stage 3.3 / 3.4 checks
    non_time_sorted_records = 0
    for record in contexts:
        target_t = int(record["target_t"])
        candidate_count = int(record["stage1_candidate_count"])
        selected_count = int(record["selected_count"])
        if candidate_count != min(K1_DEFAULT, target_t):
            failures.append(f"stage1_count_invalid:{record['user_id']}:{target_t}")
            break
        if selected_count != min(K2_DEFAULT, candidate_count):
            failures.append(f"stage2_count_invalid:{record['user_id']}:{target_t}")
            break
        if "[GLOBAL SUMMARY]" in str(record["main_context_text"]):
            failures.append(f"main_contains_summary_header:{record['user_id']}:{target_t}")
            break
        if record["summary_fields"]["summary_text"] not in str(record["template_context_text"]):
            failures.append(f"template_missing_summary_text:{record['user_id']}:{target_t}")
            break

        history_positions: List[int] = []
        for idx, evidence in enumerate(record["evidence_list"], start=1):
            if int(evidence["rank"]) != idx:
                failures.append(f"evidence_rank_invalid:{record['user_id']}:{target_t}")
                break
            if str(evidence["role"]) not in set(ROLE_LABELS.values()):
                failures.append(f"evidence_role_invalid:{record['user_id']}:{target_t}")
                break
            support_score = str(evidence["support_score"])
            if "." not in support_score or len(support_score.rsplit(".", 1)[1]) != SUPPORT_SCORE_DECIMALS:
                failures.append(f"support_score_precision_invalid:{record['user_id']}:{target_t}")
                break
            q_text = str(evidence["question_text"])
            if len(q_text) > QUESTION_TEXT_LIMIT + len(QUESTION_TEXT_ELLIPSIS):
                failures.append(f"question_text_too_long:{record['user_id']}:{target_t}")
                break
            if len(q_text) > QUESTION_TEXT_LIMIT and not q_text.endswith(QUESTION_TEXT_ELLIPSIS):
                failures.append(f"question_text_missing_ellipsis:{record['user_id']}:{target_t}")
                break
            history_positions.append(int(evidence["history_pos"]))
        if failures:
            break
        if history_positions != sorted(history_positions):
            non_time_sorted_records += 1

    sample_audit = contexts[:5]
    report = {
        "stage32_manifest": stage32_manifest,
        "stage34_manifest": stage34_manifest,
        "semantic_ids_stable": semantic_ids_stable,
        "problem_count": len(problem_catalog),
        "context_records_checked": len(contexts),
        "graph_constraints": {
            "has_explicit_prerequisite": graph_bundle.get("has_explicit_prerequisite"),
            "e_pre_size": len(graph_bundle.get("e_pre") or []),
        },
        "training": {
            "epochs_ran": training_report.get("epochs_ran"),
            "best_val_loss": min(float(item["val_loss"]) for item in training_report.get("history") or [dict(val_loss=0.0)]),
        },
        "context_checks": {
            "non_time_sorted_records": non_time_sorted_records,
            "main_without_summary_header": all("[GLOBAL SUMMARY]" not in str(record["main_context_text"]) for record in contexts),
            "template_with_summary_text": all(record["summary_fields"]["summary_text"] in str(record["template_context_text"]) for record in contexts),
        },
        "sample_audit_records": sample_audit,
        "failures": failures,
    }
    report_path = reports_dir / "validation_report.json"
    write_json(report, report_path)
    return ValidationResult(
        report_path=str(report_path),
        semantic_ids_stable=semantic_ids_stable,
        context_records_checked=len(contexts),
        failures=failures,
    )
