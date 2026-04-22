from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .constants import K1_DEFAULT, K2_DEFAULT, QUESTION_TEXT_ELLIPSIS, QUESTION_TEXT_LIMIT, ROLE_LABELS, SUPPORT_SCORE_DECIMALS
from .io_utils import ProblemRecord, write_json
from .llm_utils import parse_llm_summary_json
from .stage32 import build_semantic_ids


@dataclass
class ValidationResult:
    report_path: str
    semantic_ids_stable: bool
    context_records_checked: int
    failures: List[str]


def _load_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
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
    problem_mu_q = json.loads((priors_dir / "problem_mu_q.json").read_text(encoding="utf-8"))
    embeddings_path = out_root / "cache" / "context_embeddings.pkl"

    failures: List[str] = []

    problem_catalog = _load_jsonl(priors_dir / "problem_catalog.jsonl")
    contexts = _load_jsonl(contexts_dir / "contexts.jsonl", limit=256 if smoke else None)
    with (priors_dir / "hqtext_vectors.pkl").open("rb") as f:
        hqtext_map = pickle.load(f)

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

    if len(semantic_ids) != len(problem_catalog):
        failures.append("semantic_id_coverage_mismatch")
    if len(problem_mu_q) != len(problem_catalog):
        failures.append("problem_mu_q_coverage_mismatch")
    if not str(defaults.get("text_embed_model_name") or "").strip():
        failures.append("text_embed_model_name_missing")
    if str(stage34_manifest.get("text_embed_model") or "").strip() != str(defaults.get("text_embed_model_name") or "").strip():
        failures.append("text_embed_model_mismatch_between_stage32_stage34")
    if int(defaults.get("text_embed_batch_size") or 0) <= 0:
        failures.append("text_embed_batch_size_invalid")
    if int(defaults.get("text_embed_max_length") or 0) <= 0:
        failures.append("text_embed_max_length_invalid")
    if int(stage34_manifest.get("text_embed_max_length") or 0) != int(defaults.get("text_embed_max_length") or 0):
        failures.append("text_embed_max_length_mismatch_between_stage32_stage34")
    if defaults.get("kglobal") != 50 or defaults.get("klocal") != 5:
        failures.append("semantic_id_defaults_mismatch")
    if defaults.get("ctfidf_max_features") != 5000:
        failures.append("ctfidf_default_mismatch")

    if defaults.get("use_rasch_enhancement"):
        mu_values = np.asarray([float(problem_mu_q.get(str(record["problem_id"]), 0.0)) for record in problem_catalog], dtype=np.float32)
        if not np.all(np.isfinite(mu_values)):
            failures.append("problem_mu_q_not_finite")
        elif np.max(np.abs(mu_values)) <= 1e-8:
            failures.append("problem_mu_q_all_zero")

    if defaults.get("enable_llm_graph_completion"):
        llm_graph_completion = graph_bundle.get("llm_graph_completion")
        if not llm_graph_completion:
            failures.append("graph_completion_missing")
        else:
            allowed_confidence = {"低", "中", "高"}
            required_graph_keys = {"prerequisite_candidates", "related_candidates", "confidence"}
            optional_graph_keys = {"chapters", "candidate_concepts"}
            for concept, payload in llm_graph_completion.items():
                if not isinstance(payload, dict):
                    failures.append(f"graph_completion_payload_invalid:{concept}")
                    break
                payload_keys = set(payload.keys())
                if not required_graph_keys.issubset(payload_keys):
                    failures.append(f"graph_completion_schema_invalid:{concept}")
                    break
                if not payload_keys.issubset(required_graph_keys | optional_graph_keys):
                    failures.append(f"graph_completion_schema_invalid:{concept}")
                    break
                prereq = payload.get("prerequisite_candidates")
                related = payload.get("related_candidates")
                confidence = payload.get("confidence")
                if not isinstance(prereq, list) or not isinstance(related, list):
                    failures.append(f"graph_completion_candidates_invalid:{concept}")
                    break
                if confidence not in allowed_confidence:
                    failures.append(f"graph_completion_confidence_invalid:{concept}")
                    break
    else:
        if graph_bundle.get("has_explicit_prerequisite") is not False:
            failures.append("graph_has_unexpected_prerequisite")
        if graph_bundle.get("e_pre") != []:
            failures.append("graph_prerequisite_not_empty")

    if not training_report.get("history"):
        failures.append("training_history_missing")
    else:
        val_losses = [float(item["val_loss"]) for item in training_report["history"]]
        if min(val_losses) > val_losses[0] + 1e-8:
            failures.append("training_val_not_improved")

    non_time_sorted_records = 0
    llm_summary_present = 0
    llm_summary_struct_valid = 0
    context_stage_failed = False
    for record in contexts:
        target_t = int(record["target_t"])
        candidate_count = int(record["stage1_candidate_count"])
        selected_count = int(record["selected_count"])
        stage1_limit = int(stage34_manifest.get("rerank_topk") or K1_DEFAULT)
        if candidate_count != min(stage1_limit, K1_DEFAULT, target_t):
            failures.append(f"stage1_count_invalid:{record['user_id']}:{target_t}")
            context_stage_failed = True
            break
        if selected_count != min(K2_DEFAULT, candidate_count):
            failures.append(f"stage2_count_invalid:{record['user_id']}:{target_t}")
            context_stage_failed = True
            break
        if "[GLOBAL SUMMARY]" in str(record["main_context_text"]):
            failures.append(f"main_contains_summary_header:{record['user_id']}:{target_t}")
            context_stage_failed = True
            break
        if record["summary_fields"]["summary_text"] not in str(record["template_context_text"]):
            failures.append(f"template_missing_summary_text:{record['user_id']}:{target_t}")
            context_stage_failed = True
            break

        llm_summary_text = str(record.get("summary_fields", {}).get("llm_summary_text") or "").strip()
        llm_summary_struct = record.get("summary_fields", {}).get("llm_summary_struct")
        llm_context_text = str(record.get("llm_context_text") or "").strip()
        if llm_summary_text:
            llm_summary_present += 1
            try:
                parsed_llm_struct = parse_llm_summary_json(llm_summary_text)
            except Exception:
                failures.append(f"llm_summary_json_invalid:{record['user_id']}:{target_t}")
                context_stage_failed = True
                break
            if llm_summary_struct != parsed_llm_struct:
                failures.append(f"llm_summary_struct_mismatch:{record['user_id']}:{target_t}")
                context_stage_failed = True
                break
            if not llm_context_text:
                failures.append(f"llm_context_missing:{record['user_id']}:{target_t}")
                context_stage_failed = True
                break
            llm_summary_struct_valid += 1

        history_positions: List[int] = []
        for idx, evidence in enumerate(record["evidence_list"], start=1):
            if int(evidence["rank"]) != idx:
                failures.append(f"evidence_rank_invalid:{record['user_id']}:{target_t}")
                context_stage_failed = True
                break
            if str(evidence["role"]) not in set(ROLE_LABELS.values()):
                failures.append(f"evidence_role_invalid:{record['user_id']}:{target_t}")
                context_stage_failed = True
                break
            if stage34_manifest.get("use_qwen_reranker"):
                raw_scores = evidence.get("raw_scores") or {}
                if "rerank" not in raw_scores:
                    failures.append(f"rerank_score_missing:{record['user_id']}:{target_t}")
                    context_stage_failed = True
                    break
            support_score = str(evidence["support_score"])
            if "." not in support_score or len(support_score.rsplit(".", 1)[1]) != SUPPORT_SCORE_DECIMALS:
                failures.append(f"support_score_precision_invalid:{record['user_id']}:{target_t}")
                context_stage_failed = True
                break
            q_text = str(evidence["question_text"])
            if len(q_text) > QUESTION_TEXT_LIMIT + len(QUESTION_TEXT_ELLIPSIS):
                failures.append(f"question_text_too_long:{record['user_id']}:{target_t}")
                context_stage_failed = True
                break
            if len(q_text) > QUESTION_TEXT_LIMIT and not q_text.endswith(QUESTION_TEXT_ELLIPSIS):
                failures.append(f"question_text_missing_ellipsis:{record['user_id']}:{target_t}")
                context_stage_failed = True
                break
            history_positions.append(int(evidence["history_pos"]))
        if context_stage_failed:
            break
        if history_positions != sorted(history_positions):
            non_time_sorted_records += 1

    sample_audit = contexts[:5]
    llm_embedding_checks: Dict[str, Any] = {
        "embeddings_path": str(embeddings_path),
        "has_llm_embeddings": False,
        "has_llm_struct_embeddings": False,
        "has_llm_struct_features": False,
    }
    if embeddings_path.exists():
        with embeddings_path.open("rb") as f:
            embedding_payload = pickle.load(f)
        index_len = len(embedding_payload.get("index") or [])
        llm_embedding_checks["index_len"] = index_len
        llm_embedding_checks["has_llm_embeddings"] = "llm_embeddings" in embedding_payload
        llm_embedding_checks["has_llm_struct_embeddings"] = "llm_struct_embeddings" in embedding_payload
        llm_embedding_checks["has_llm_struct_features"] = "llm_struct_features" in embedding_payload
        if index_len != stage34_manifest.get("record_count"):
            failures.append("context_embedding_index_count_mismatch")
        if "llm_embeddings" in embedding_payload:
            llm_shape = list(np.asarray(embedding_payload["llm_embeddings"]).shape)
            llm_embedding_checks["llm_embeddings_shape"] = llm_shape
            if llm_shape[0] != index_len:
                failures.append("llm_embeddings_count_mismatch")
        if "llm_struct_embeddings" in embedding_payload:
            llm_struct_shape = list(np.asarray(embedding_payload["llm_struct_embeddings"]).shape)
            llm_embedding_checks["llm_struct_embeddings_shape"] = llm_struct_shape
            if llm_struct_shape[0] != index_len:
                failures.append("llm_struct_embeddings_count_mismatch")
        if "llm_struct_features" in embedding_payload:
            llm_struct_feature_shape = list(np.asarray(embedding_payload["llm_struct_features"]).shape)
            llm_embedding_checks["llm_struct_features_shape"] = llm_struct_feature_shape
            if llm_struct_feature_shape[0] != index_len:
                failures.append("llm_struct_features_count_mismatch")

    reranker_checks: Dict[str, Any] = {
        "enabled": bool(stage34_manifest.get("use_qwen_reranker")),
        "text_rerank_model": stage34_manifest.get("text_rerank_model"),
        "rerank_topk": stage34_manifest.get("rerank_topk"),
        "rerank_weight": stage34_manifest.get("rerank_weight"),
        "reranker_cache_exists": False,
    }
    reranker_cache_path = stage34_manifest.get("reranker_cache_path")
    if stage34_manifest.get("use_qwen_reranker"):
        if not str(stage34_manifest.get("text_rerank_model") or "").strip():
            failures.append("text_rerank_model_missing")
        if int(stage34_manifest.get("rerank_topk") or 0) <= 0:
            failures.append("rerank_topk_invalid")
        if int(stage34_manifest.get("text_rerank_batch_size") or 0) <= 0:
            failures.append("text_rerank_batch_size_invalid")
        reranker_cache = Path(str(reranker_cache_path)) if reranker_cache_path else None
        if reranker_cache is None or not reranker_cache.exists():
            failures.append("reranker_cache_missing")
        else:
            reranker_checks["reranker_cache_exists"] = True

    if llm_summary_present > 0 and not context_stage_failed:
        if llm_summary_present != len(contexts):
            failures.append("llm_summary_partial_coverage")
        if llm_summary_struct_valid != len(contexts):
            failures.append("llm_summary_struct_partial_coverage")
        if not llm_embedding_checks.get("has_llm_embeddings"):
            failures.append("llm_embeddings_missing")
        if not llm_embedding_checks.get("has_llm_struct_embeddings"):
            failures.append("llm_struct_embeddings_missing")
        if not llm_embedding_checks.get("has_llm_struct_features"):
            failures.append("llm_struct_features_missing")

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
        "rasch_checks": {
            "enabled": bool(defaults.get("use_rasch_enhancement")),
            "problem_mu_q_count": len(problem_mu_q),
            "problem_mu_q_nonzero_count": int(sum(abs(float(v)) > 1e-8 for v in problem_mu_q.values())),
        },
        "llm_graph_checks": {
            "enabled": bool(defaults.get("enable_llm_graph_completion")),
            "completion_count": len(graph_bundle.get("llm_graph_completion") or {}),
        },
        "retrieval_checks": {
            "text_embed_model_name": defaults.get("text_embed_model_name"),
            "text_embed_batch_size": defaults.get("text_embed_batch_size"),
            "text_embed_max_length": defaults.get("text_embed_max_length"),
            "stage34_text_embed_model_name": stage34_manifest.get("text_embed_model"),
            "stage34_text_embed_max_length": stage34_manifest.get("text_embed_max_length"),
            "text_rerank_model_name": stage34_manifest.get("text_rerank_model"),
            "text_rerank_batch_size": stage34_manifest.get("text_rerank_batch_size"),
            "use_qwen_reranker": bool(stage34_manifest.get("use_qwen_reranker")),
        },
        "training": {
            "epochs_ran": training_report.get("epochs_ran"),
            "best_val_loss": min(float(item["val_loss"]) for item in training_report.get("history") or [dict(val_loss=0.0)]),
        },
        "context_checks": {
            "non_time_sorted_records": non_time_sorted_records,
            "main_without_summary_header": all("[GLOBAL SUMMARY]" not in str(record["main_context_text"]) for record in contexts),
            "template_with_summary_text": all(record["summary_fields"]["summary_text"] in str(record["template_context_text"]) for record in contexts),
            "llm_summary_present_count": llm_summary_present,
            "llm_summary_struct_valid_count": llm_summary_struct_valid,
        },
        "llm_embedding_checks": llm_embedding_checks,
        "reranker_checks": reranker_checks,
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
