from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from common_pipeline_strict.io_utils import ProblemRecord, write_json
from common_pipeline_strict.stage32 import build_semantic_ids


@dataclass
class Stage32CheckResult:
    report_path: str
    passed: bool
    failures: List[str]
    warnings: List[str]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _add_failure(failures: List[str], key: str) -> None:
    if key not in failures:
        failures.append(key)


def _add_warning(warnings: List[str], key: str) -> None:
    if key not in warnings:
        warnings.append(key)


def _check_required_files(priors_dir: Path, manifest: Dict[str, Any], failures: List[str]) -> None:
    required_manifest_keys = [
        "semantic_ids_path",
        "semantic_id_audit_path",
        "semantic_vectors_path",
        "hqtext_vectors_path",
        "hqid_vectors_path",
        "eqbase_vectors_path",
        "problem_mu_q_path",
        "concept_pc1_path",
        "item_collaborative_path",
        "item_collaborative_vectors_path",
        "graph_bundle_path",
        "problem_catalog_path",
        "model_state_path",
        "training_report_path",
        "implementation_defaults_path",
        "manifest_path",
    ]
    for key in required_manifest_keys:
        value = str(manifest.get(key) or "").strip()
        if not value:
            _add_failure(failures, f"manifest_missing:{key}")
            continue
        path = Path(value)
        if not path.exists():
            _add_failure(failures, f"missing_file:{key}")
            continue
        if path.parent.resolve() != priors_dir.resolve():
            _add_failure(failures, f"manifest_path_outside_priors:{key}")


def _build_recompute_problems(problem_catalog: Sequence[Dict[str, Any]]) -> List[ProblemRecord]:
    return [
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


def _vector_shape_summary(name: str, payload: Dict[str, Any], expected_ids: Sequence[str], failures: List[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"name": name, "count": len(payload)}
    missing_ids = [pid for pid in expected_ids if pid not in payload]
    extra_ids = [pid for pid in payload.keys() if pid not in set(expected_ids)]
    if missing_ids:
        _add_failure(failures, f"{name}_missing_ids")
    if extra_ids:
        _add_failure(failures, f"{name}_extra_ids")

    shapes = {}
    dtypes = {}
    for pid in expected_ids[: min(32, len(expected_ids))]:
        if pid not in payload:
            continue
        arr = np.asarray(payload[pid])
        shapes[str(arr.shape)] = shapes.get(str(arr.shape), 0) + 1
        dtypes[str(arr.dtype)] = dtypes.get(str(arr.dtype), 0) + 1
        if not np.all(np.isfinite(arr)):
            _add_failure(failures, f"{name}_non_finite")
            break
    summary["sample_shapes"] = shapes
    summary["sample_dtypes"] = dtypes
    return summary


def _vector_shape_summary_subset(name: str, payload: Dict[str, Any], failures: List[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"name": name, "count": len(payload)}
    shapes = {}
    dtypes = {}
    for _pid, value in list(payload.items())[: min(32, len(payload))]:
        arr = np.asarray(value)
        shapes[str(arr.shape)] = shapes.get(str(arr.shape), 0) + 1
        dtypes[str(arr.dtype)] = dtypes.get(str(arr.dtype), 0) + 1
        if not np.all(np.isfinite(arr)):
            _add_failure(failures, f"{name}_non_finite")
            break
    summary["sample_shapes"] = shapes
    summary["sample_dtypes"] = dtypes
    return summary


def run_stage32_checks(
    *,
    out_root: Path,
    semantic_flagged_ratio_threshold: float,
    semantic_fallback_ratio_threshold: float,
    collab_vector_missing_ratio_threshold: float,
) -> Stage32CheckResult:
    priors_dir = out_root / "priors"
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    failures: List[str] = []
    warnings: List[str] = []

    manifest_path = priors_dir / "stage32_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = _load_json(manifest_path)
    _check_required_files(priors_dir, manifest, failures)

    semantic_ids = _load_json(priors_dir / "semantic_ids.json")
    semantic_id_audit = _load_json(priors_dir / "semantic_id_audit.json")
    problem_catalog = _load_jsonl(priors_dir / "problem_catalog.jsonl")
    defaults = _load_json(priors_dir / "implementation_defaults.json")
    training_report = _load_json(priors_dir / "training_report.json")
    graph_bundle = _load_json(priors_dir / "concept_graph_bundle.json")
    problem_mu_q = _load_json(priors_dir / "problem_mu_q.json")

    expected_ids = [str(record["problem_id"]) for record in problem_catalog]
    expected_id_set = set(expected_ids)

    if int(manifest.get("problem_count") or -1) != len(problem_catalog):
        _add_failure(failures, "manifest_problem_count_mismatch")
    if len(semantic_ids) != len(problem_catalog):
        _add_failure(failures, "semantic_id_coverage_mismatch")
    if set(semantic_ids.keys()) != expected_id_set:
        _add_failure(failures, "semantic_id_problem_set_mismatch")
    if len(problem_mu_q) != len(problem_catalog):
        _add_failure(failures, "problem_mu_q_coverage_mismatch")

    hqtext_map = _load_pickle(priors_dir / "hqtext_vectors.pkl")
    hqid_map = _load_pickle(priors_dir / "hqid_vectors.pkl")
    eqbase_map = _load_pickle(priors_dir / "eqbase_vectors.pkl")
    semantic_vectors = _load_pickle(priors_dir / "semantic_vectors.pkl")
    collab_vectors = _load_pickle(priors_dir / "item_collaborative_embeddings.pkl")
    concept_pc1_dirs = _load_pickle(priors_dir / "concept_pc1_dirs.pkl")
    collab_neighbors = _load_json(priors_dir / "item_collaborative.json")

    vector_checks = {
        "hqtext_vectors": _vector_shape_summary("hqtext_vectors", hqtext_map, expected_ids, failures),
        "hqid_vectors": _vector_shape_summary("hqid_vectors", hqid_map, expected_ids, failures),
        "eqbase_vectors": _vector_shape_summary("eqbase_vectors", eqbase_map, expected_ids, failures),
        "semantic_vectors": _vector_shape_summary("semantic_vectors", semantic_vectors, expected_ids, failures),
        "collab_vectors": _vector_shape_summary_subset("collab_vectors", collab_vectors, failures),
    }

    if not isinstance(concept_pc1_dirs, dict) or not concept_pc1_dirs:
        _add_failure(failures, "concept_pc1_dirs_invalid")
    else:
        sample_concept_dirs = list(concept_pc1_dirs.items())[:10]
        for _concept, vec in sample_concept_dirs:
            arr = np.asarray(vec)
            if arr.ndim != 1 or not np.all(np.isfinite(arr)):
                _add_failure(failures, "concept_pc1_dirs_non_finite")
                break

    if not isinstance(collab_neighbors, dict):
        _add_failure(failures, "collab_neighbors_coverage_mismatch")
    else:
        collab_neighbor_keys = set(collab_neighbors.keys())
        collab_vector_keys = set(collab_vectors.keys())
        if collab_neighbor_keys != expected_id_set:
            _add_failure(failures, "collab_neighbors_problem_set_mismatch")
        unknown_vector_keys = collab_vector_keys - expected_id_set
        if unknown_vector_keys:
            _add_failure(failures, "collab_vectors_unknown_problem_ids")
        missing_vector_keys = expected_id_set - collab_vector_keys
        missing_vector_ratio = float(len(missing_vector_keys)) / float(max(len(problem_catalog), 1))
        if missing_vector_ratio > collab_vector_missing_ratio_threshold:
            _add_failure(failures, "collab_vectors_missing_ratio_too_high")
        elif missing_vector_keys:
            _add_warning(warnings, "collab_vectors_missing_for_unobserved_or_short_sequence_items")
        if not collab_neighbor_keys.issubset(expected_id_set):
            _add_failure(failures, "collab_neighbors_unknown_problem_ids")
        collab_coverage_ratio = float(len(collab_neighbor_keys)) / float(max(len(problem_catalog), 1))
        if collab_coverage_ratio < 0.05:
            _add_warning(warnings, "collab_coverage_too_low")

    semantic_flagged_ratio = float(semantic_id_audit.get("flagged_ratio") or 0.0)
    semantic_category_counts = semantic_id_audit.get("category_counts") or {}
    semantic_generation_stats = semantic_id_audit.get("generation_stats") or {}
    semantic_problem_count = int(semantic_id_audit.get("total_ids") or len(problem_catalog) or 1)
    macro_fallback_ratio = float(semantic_generation_stats.get("macro_fallback_count") or 0) / max(semantic_problem_count, 1)
    micro_fallback_ratio = float(semantic_generation_stats.get("micro_fallback_count") or 0) / max(semantic_problem_count, 1)

    if semantic_flagged_ratio > semantic_flagged_ratio_threshold:
        _add_failure(failures, "semantic_id_flagged_ratio_too_high")
    if int(semantic_category_counts.get("contains_noise") or 0) > 0:
        _add_failure(failures, "semantic_id_contains_noise_tokens")
    if int(semantic_category_counts.get("empty") or 0) > 0:
        _add_failure(failures, "semantic_id_empty_present")
    if macro_fallback_ratio > semantic_fallback_ratio_threshold:
        _add_failure(failures, "semantic_id_macro_fallback_ratio_too_high")
    if micro_fallback_ratio > semantic_fallback_ratio_threshold:
        _add_failure(failures, "semantic_id_micro_fallback_ratio_too_high")
    if int(semantic_category_counts.get("duplicate_tokens") or 0) > 0:
        _add_warning(warnings, "semantic_id_duplicate_tokens_present")

    recompute_problems = _build_recompute_problems(problem_catalog)
    ordered_hqtext = [np.asarray(hqtext_map[problem.problem_id]) for problem in recompute_problems]
    recomputed_path = reports_dir / "semantic_ids_recomputed_stage32.json"
    semantic_id_source = (
        defaults.get("semantic_id_cleaning", {}).get("semantic_id_source")
        if isinstance(defaults.get("semantic_id_cleaning"), dict)
        else None
    )
    recomputed_ids, _semantic_texts, _semantic_audit = build_semantic_ids(
        recompute_problems,
        text_vectors=np.stack(ordered_hqtext, axis=0),
        semantic_ids_path=recomputed_path,
        semantic_id_source=str(semantic_id_source or "cluster"),
    )
    semantic_ids_stable = recomputed_ids == semantic_ids
    if not semantic_ids_stable:
        _add_failure(failures, "semantic_ids_not_stable")

    if not str(defaults.get("text_embed_model_name") or "").strip():
        _add_failure(failures, "text_embed_model_name_missing")
    if int(defaults.get("text_embed_batch_size") or 0) <= 0:
        _add_failure(failures, "text_embed_batch_size_invalid")
    if int(defaults.get("text_embed_max_length") or 0) <= 0:
        _add_failure(failures, "text_embed_max_length_invalid")
    if defaults.get("kglobal") != 50 or defaults.get("klocal") != 5:
        _add_failure(failures, "semantic_id_defaults_mismatch")
    if defaults.get("ctfidf_max_features") != 5000:
        _add_failure(failures, "ctfidf_default_mismatch")

    mu_values = np.asarray([float(problem_mu_q.get(pid, 0.0)) for pid in expected_ids], dtype=np.float32)
    if not np.all(np.isfinite(mu_values)):
        _add_failure(failures, "problem_mu_q_not_finite")
    if defaults.get("use_rasch_enhancement"):
        if np.max(np.abs(mu_values)) <= 1e-8:
            _add_failure(failures, "problem_mu_q_all_zero")
        if float(np.std(mu_values)) <= 1e-8:
            _add_warning(warnings, "problem_mu_q_variance_too_small")

    if not training_report.get("history"):
        _add_failure(failures, "training_history_missing")
        best_val_loss = None
    else:
        val_losses = [float(item["val_loss"]) for item in training_report["history"]]
        train_losses = [float(item["train_loss"]) for item in training_report["history"]]
        best_val_loss = min(val_losses)
        if best_val_loss > val_losses[0] + 1e-8:
            _add_failure(failures, "training_val_not_improved")
        if min(train_losses) >= train_losses[0] - 1e-8:
            _add_warning(warnings, "training_loss_not_reduced")

    if not isinstance(graph_bundle, dict):
        _add_failure(failures, "graph_bundle_invalid")
    else:
        if graph_bundle.get("has_explicit_prerequisite") is not False and not defaults.get("enable_llm_graph_completion"):
            _add_failure(failures, "graph_has_unexpected_prerequisite")
        if not isinstance(graph_bundle.get("e_pre"), list):
            _add_failure(failures, "graph_e_pre_invalid")
        if not isinstance(graph_bundle.get("local_edges"), list):
            _add_failure(failures, "graph_local_edges_invalid")
        if not isinstance(graph_bundle.get("concept_neighbors"), dict):
            _add_failure(failures, "graph_concept_neighbors_invalid")
        if not isinstance(graph_bundle.get("problem_neighbor_concepts"), dict):
            _add_failure(failures, "graph_problem_neighbor_concepts_invalid")
        if defaults.get("enable_llm_graph_completion"):
            llm_completion = graph_bundle.get("llm_graph_completion")
            if not isinstance(llm_completion, dict) or not llm_completion:
                _add_failure(failures, "graph_completion_missing")

    report = {
        "passed": len(failures) == 0,
        "failures": failures,
        "warnings": warnings,
        "manifest": {
            "problem_count": manifest.get("problem_count"),
            "student_count": manifest.get("student_count"),
        },
        "semantic_id_checks": {
            "stable": semantic_ids_stable,
            "source": semantic_id_source or "cluster",
            "effective_source": semantic_generation_stats.get("semantic_id_source_effective"),
            "flagged_ratio": semantic_flagged_ratio,
            "flagged_ratio_threshold": semantic_flagged_ratio_threshold,
            "macro_fallback_ratio": macro_fallback_ratio,
            "micro_fallback_ratio": micro_fallback_ratio,
            "fallback_ratio_threshold": semantic_fallback_ratio_threshold,
            "category_counts": semantic_category_counts,
            "generation_stats": semantic_generation_stats,
            "top_suspicious_tokens": semantic_id_audit.get("top_suspicious_tokens") or [],
        },
        "vector_checks": vector_checks,
        "rasch_checks": {
            "enabled": bool(defaults.get("use_rasch_enhancement")),
            "mu_q_count": len(problem_mu_q),
            "mu_q_nonzero_count": int(np.sum(np.abs(mu_values) > 1e-8)),
            "mu_q_mean": float(np.mean(mu_values)) if mu_values.size else 0.0,
            "mu_q_std": float(np.std(mu_values)) if mu_values.size else 0.0,
        },
        "training_checks": {
            "epochs_ran": training_report.get("epochs_ran"),
            "best_val_loss": best_val_loss,
            "history_length": len(training_report.get("history") or []),
        },
        "graph_checks": {
            "has_explicit_prerequisite": graph_bundle.get("has_explicit_prerequisite") if isinstance(graph_bundle, dict) else None,
            "e_pre_size": len(graph_bundle.get("e_pre") or []) if isinstance(graph_bundle, dict) else None,
            "e_high_size": len(graph_bundle.get("e_high") or []) if isinstance(graph_bundle, dict) else None,
            "e_peer_size": len(graph_bundle.get("e_peer") or []) if isinstance(graph_bundle, dict) else None,
        },
        "collab_checks": {
            "neighbor_problem_count": len(collab_neighbors) if isinstance(collab_neighbors, dict) else 0,
            "coverage_ratio": float(len(collab_neighbors) if isinstance(collab_neighbors, dict) else 0) / float(max(len(problem_catalog), 1)),
            "vector_problem_count": len(collab_vectors) if isinstance(collab_vectors, dict) else 0,
            "vector_missing_count": len(expected_id_set - set(collab_vectors.keys())) if isinstance(collab_vectors, dict) else len(problem_catalog),
            "vector_missing_ratio": float(len(expected_id_set - set(collab_vectors.keys())) if isinstance(collab_vectors, dict) else len(problem_catalog)) / float(max(len(problem_catalog), 1)),
            "vector_missing_ratio_threshold": collab_vector_missing_ratio_threshold,
        },
    }
    report_path = reports_dir / "stage32_check_report.json"
    write_json(report, report_path)
    return Stage32CheckResult(
        report_path=str(report_path),
        passed=len(failures) == 0,
        failures=failures,
        warnings=warnings,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    workspace = script_dir.parent
    parser.add_argument("--out_root", default=str(workspace / "out" / "strict_common_pipeline"))
    parser.add_argument("--semantic_flagged_ratio_threshold", type=float, default=0.20)
    parser.add_argument("--semantic_fallback_ratio_threshold", type=float, default=0.35)
    parser.add_argument("--collab_vector_missing_ratio_threshold", type=float, default=0.05)
    args = parser.parse_args()

    result = run_stage32_checks(
        out_root=Path(args.out_root).resolve(),
        semantic_flagged_ratio_threshold=float(args.semantic_flagged_ratio_threshold),
        semantic_fallback_ratio_threshold=float(args.semantic_fallback_ratio_threshold),
        collab_vector_missing_ratio_threshold=float(args.collab_vector_missing_ratio_threshold),
    )
    print("[OK] stage32 check finished")
    print("[REPORT]", result.report_path)
    print("[PASSED]", result.passed)
    print("[FAILURES]", len(result.failures))
    for failure in result.failures:
        print(" -", failure)
    print("[WARNINGS]", len(result.warnings))
    for warning in result.warnings:
        print(" -", warning)


if __name__ == "__main__":
    main()
