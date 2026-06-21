"""Run resumable LLM-direct-prediction ablations on frozen cognitive-RAG test contexts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib import error, request

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime.
    tqdm = None

from common_pipeline_strict.io_utils import (
    ensure_dir,
    load_problem_records,
    load_student_sequences,
    parse_detail_field,
    read_json_any,
    write_json,
)


SUPPORTED_VARIANTS = {
    "full_cognitive_rag_llm",
    "wo_cognitive_retrieval_recent",
    "wo_cognitive_retrieval_random",
    "wo_llm_summary",
    "wo_structured_evidence",
}
DEFAULT_VARIANTS = (
    "full_cognitive_rag_llm,"
    "wo_cognitive_retrieval_recent,"
    "wo_llm_summary,"
    "wo_structured_evidence"
)
PROMPT_VERSION = "llm_ablation_hard_label_v2_token_logprob"
PREDICTION_MODES = {"hard_label", "token_logprob"}
TOKEN_LOGPROB_METHOD = "binary_conditional_token_logprob"


class MissingSummaryError(RuntimeError):
    """Raised when a strict full variant lacks an existing LLM summary."""


@dataclass(frozen=True)
class PromptBundle:
    """System/user prompt plus structured prompt input for traceability."""

    system_prompt: str
    user_prompt: str
    prompt_input: Dict[str, Any]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _compact_text(text: Any, limit: int = 260) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _context_key(user_id: str, target_t: int, target_pid: str) -> str:
    return f"{user_id}\t{int(target_t)}\t{target_pid}"


def _prompt_hash(system_prompt: str, user_prompt: str) -> str:
    payload = (system_prompt + "\n\n" + user_prompt).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _stable_int_seed(seed: int, *parts: Any) -> int:
    text = "\t".join([str(seed)] + [str(part) for part in parts])
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _answer_to_int(value: Any) -> Optional[int]:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "correct", "right", "yes", "y", "correct_answer"}:
        return 1
    if text in {"0", "false", "incorrect", "wrong", "no", "n", "incorrect_answer"}:
        return 0
    if str(value).strip() in {"正确", "答对"}:
        return 1
    if str(value).strip() in {"错误", "答错"}:
        return 0
    try:
        return 1 if int(value) else 0
    except Exception:
        return None


def _answer_label(value: Any) -> str:
    parsed = _answer_to_int(value)
    if parsed is None:
        return str(value or "")
    return "correct" if parsed else "incorrect"


def _format_rate(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def inspect_jsonl_schema(path: Path, *, limit: int = 3) -> List[Dict[str, Any]]:
    """Return a compact schema sketch for the first JSONL rows."""
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f):
            if idx >= int(limit):
                break
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {idx + 1}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object rows in {path}, got {type(obj).__name__} at line {idx + 1}")
            evidence = obj.get("evidence_list")
            summary_fields = obj.get("summary_fields")
            rows.append(
                {
                    "line": idx + 1,
                    "keys": sorted(obj.keys()),
                    "evidence_keys": sorted(evidence[0].keys()) if isinstance(evidence, list) and evidence and isinstance(evidence[0], dict) else [],
                    "summary_keys": sorted(summary_fields.keys()) if isinstance(summary_fields, dict) else [],
                }
            )
    return rows


def _schema_error(path: Path, schema: List[Dict[str, Any]], message: str) -> ValueError:
    rendered = json.dumps(schema, ensure_ascii=False, indent=2)
    return ValueError(f"{message}\nSchema preview for {path}:\n{rendered}")


def validate_context_schema(path: Path) -> List[Dict[str, Any]]:
    """Check required context fields and return the schema preview."""
    schema = inspect_jsonl_schema(path, limit=3)
    if not schema:
        raise ValueError(f"No non-empty JSONL rows found in contexts file: {path}")
    required = {"user_id", "target_t", "target_pid", "evidence_list", "summary_fields"}
    missing = required - set(schema[0]["keys"])
    if missing:
        raise _schema_error(path, schema, f"contexts.jsonl is missing required fields: {sorted(missing)}")
    return schema


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
            if isinstance(row, dict):
                yield row


def _load_problem_catalog(problem_json: Path, contexts_jsonl: Path) -> Dict[str, Dict[str, Any]]:
    """Load problem metadata, preferring the strict pipeline catalog when present."""
    inferred_catalog_path = contexts_jsonl.resolve().parent.parent / "priors" / "problem_catalog.jsonl"
    catalog: Dict[str, Dict[str, Any]] = {}
    if inferred_catalog_path.exists():
        for row in _load_jsonl(inferred_catalog_path):
            pid = str(row.get("problem_id") or "").strip()
            if pid:
                catalog[pid] = dict(row)
    else:
        for record in load_problem_records(problem_json):
            catalog[record.problem_id] = {
                "problem_id": record.problem_id,
                "text": record.text,
                "title": record.title,
                "chapter": record.chapter,
                "location": record.location,
                "cognitive_dimension": record.cognitive_dimension,
                "concepts": record.concepts,
                "semantic_id": "",
            }

    raw = read_json_any(problem_json)
    if isinstance(raw, dict):
        raw = [raw]
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("problem_id") or item.get("id") or "").strip()
            if not pid:
                continue
            detail = parse_detail_field(item.get("detail"))
            raw_options = detail.get("option") or detail.get("options") or {}
            options: List[Dict[str, str]] = []
            if isinstance(raw_options, dict):
                for label in sorted(raw_options):
                    text = _compact_text(raw_options[label], 180)
                    if text:
                        options.append({"label": str(label), "text": text})
            elif isinstance(raw_options, list):
                for idx, option_text in enumerate(raw_options):
                    text = _compact_text(option_text, 180)
                    if text:
                        options.append({"label": str(idx), "text": text})
            if pid not in catalog:
                catalog[pid] = {
                    "problem_id": pid,
                    "text": _compact_text(detail.get("content") or detail.get("title") or "", 1200),
                    "title": str(detail.get("title") or ""),
                    "chapter": "",
                    "location": str(detail.get("location") or ""),
                    "cognitive_dimension": item.get("cognitive_dimension"),
                    "concepts": [str(x) for x in (item.get("concepts") or [])],
                    "semantic_id": "",
                }
            if options:
                catalog[pid]["options"] = options
            problem_type = str(detail.get("typetext") or detail.get("type") or "").strip()
            if problem_type:
                catalog[pid]["problem_type"] = problem_type
    if not catalog:
        raise ValueError(f"No problem metadata could be loaded from {problem_json}")
    return catalog


def _target_payload(meta: Dict[str, Any], target_pid: str, target_semantic_id: str, max_text_chars: int) -> Dict[str, Any]:
    return {
        "problem_id": target_pid,
        "semantic_id": target_semantic_id or str(meta.get("semantic_id") or ""),
        "concepts": [str(x) for x in (meta.get("concepts") or [])],
        "cognitive_dimension": meta.get("cognitive_dimension"),
        "problem_type": meta.get("problem_type"),
        "title": _compact_text(meta.get("title"), 120),
        "text": _compact_text(meta.get("text"), max_text_chars),
        "options": meta.get("options") or [],
    }


def _history_item_from_log(
    *,
    log: Dict[str, Any],
    history_pos: int,
    target_t: int,
    catalog: Dict[str, Dict[str, Any]],
    max_text_chars: int,
) -> Dict[str, Any]:
    pid = str(log.get("problem_id") or "")
    meta = catalog.get(pid, {})
    answer = _answer_to_int(log.get("is_correct"))
    return {
        "rank": None,
        "problem_id": pid,
        "semantic_id": str(meta.get("semantic_id") or ""),
        "concepts": [str(x) for x in (meta.get("concepts") or [])],
        "cognitive_dimension": meta.get("cognitive_dimension"),
        "answer": answer,
        "answer_text": _answer_label(answer),
        "history_pos": int(history_pos),
        "steps_before_target": max(0, int(target_t) - int(history_pos)),
        "text": _compact_text(meta.get("text"), max_text_chars),
        "source": "student_history",
    }


def _load_histories_and_labels(
    student_json: Path,
    catalog: Dict[str, Dict[str, Any]],
    needed_keys: set[str],
    *,
    max_text_chars: int,
) -> Dict[str, Dict[str, Any]]:
    """Build target labels and pre-target history for context rows."""
    out: Dict[str, Dict[str, Any]] = {}
    if not needed_keys:
        return out
    allowed_pids = set(catalog)
    for sequence in load_student_sequences(student_json):
        filtered = [log for log in sequence.seq if str(log.get("problem_id") or "") in allowed_pids]
        for target_t, log in enumerate(filtered):
            pid = str(log.get("problem_id") or "")
            key = _context_key(sequence.user_id, target_t, pid)
            if key not in needed_keys:
                continue
            history = [
                _history_item_from_log(
                    log=hist_log,
                    history_pos=hist_pos,
                    target_t=target_t,
                    catalog=catalog,
                    max_text_chars=max_text_chars,
                )
                for hist_pos, hist_log in enumerate(filtered[:target_t])
            ]
            label = _answer_to_int(log.get("is_correct"))
            out[key] = {
                "sample_id": key,
                "user_id": sequence.user_id,
                "target_t": int(target_t),
                "target_pid": pid,
                "y_true": label,
                "history": history,
            }
        if len(out) >= len(needed_keys):
            break
    return out


def _load_context_records(contexts_jsonl: Path, *, max_samples: int = 0, offset: int = 0) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    skipped = 0
    for row in _load_jsonl(contexts_jsonl):
        if skipped < int(offset):
            skipped += 1
            continue
        records.append(row)
        if int(max_samples) > 0 and len(records) >= int(max_samples):
            break
    return records


def _record_key(record: Dict[str, Any]) -> str:
    return _context_key(
        str(record.get("user_id") or ""),
        int(record.get("target_t") or 0),
        str(record.get("target_pid") or ""),
    )


def _load_eval_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load a frozen held-out evaluation manifest keyed by sample_id."""
    manifest: Dict[str, Dict[str, Any]] = {}
    required = {"sample_id", "user_id", "target_t", "target_pid", "y_true", "split"}
    for line_no, row in enumerate(_load_jsonl(path), start=1):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Evaluation manifest {path} line {line_no} is missing {missing}")
        sample_id = str(row.get("sample_id") or "").strip()
        expected_id = _context_key(str(row["user_id"]), int(row["target_t"]), str(row["target_pid"]))
        if not sample_id or sample_id != expected_id:
            raise ValueError(f"Invalid sample_id in evaluation manifest {path} line {line_no}")
        if str(row.get("split")) != "test":
            raise ValueError(f"Evaluation manifest {path} line {line_no} is not split=test")
        if int(row["y_true"]) not in {0, 1}:
            raise ValueError(f"Evaluation manifest {path} line {line_no} has invalid y_true")
        if sample_id in manifest:
            raise ValueError(f"Duplicate sample_id in evaluation manifest {path}: {sample_id}")
        manifest[sample_id] = dict(row)
    if not manifest:
        raise ValueError(f"Evaluation manifest is empty: {path}")
    return manifest


def _load_labels_for_keys(
    student_json: Path,
    catalog: Dict[str, Dict[str, Any]],
    needed_keys: set[str],
) -> Dict[str, int]:
    """Load only target labels for candidate context keys."""
    labels: Dict[str, int] = {}
    if not needed_keys:
        return labels
    allowed_pids = set(catalog)
    for sequence in load_student_sequences(student_json):
        filtered = [log for log in sequence.seq if str(log.get("problem_id") or "") in allowed_pids]
        for target_t, log in enumerate(filtered):
            pid = str(log.get("problem_id") or "")
            key = _context_key(sequence.user_id, target_t, pid)
            if key not in needed_keys:
                continue
            label = _answer_to_int(log.get("is_correct"))
            if label in {0, 1}:
                labels[key] = int(label)
        if len(labels) >= len(needed_keys):
            break
    return labels


def _allocate_stratified_counts(group_sizes: Dict[int, int], sample_size: int) -> Dict[int, int]:
    total = sum(group_sizes.values())
    if total <= 0 or sample_size <= 0:
        return {}
    target = min(int(sample_size), total)
    raw_alloc = {
        label: (float(size) / float(total)) * float(target)
        for label, size in group_sizes.items()
        if size > 0
    }
    alloc = {label: min(group_sizes[label], int(math.floor(value))) for label, value in raw_alloc.items()}
    for label, size in group_sizes.items():
        if size > 0 and alloc.get(label, 0) == 0 and target >= len(group_sizes):
            alloc[label] = 1
    remaining = target - sum(alloc.values())
    ranked = sorted(
        raw_alloc,
        key=lambda label: (raw_alloc[label] - math.floor(raw_alloc[label]), group_sizes[label]),
        reverse=True,
    )
    while remaining > 0 and ranked:
        progressed = False
        for label in ranked:
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


def _select_records_for_run(
    records: List[Dict[str, Any]],
    *,
    sample_size: int,
    sample_strategy: str,
    seed: int,
    label_by_key: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Select a deterministic subset of context records for sampled ablations."""
    if int(sample_size) <= 0 or int(sample_size) >= len(records):
        return records
    strategy = str(sample_strategy)
    target = int(sample_size)
    if strategy == "first":
        return records[:target]
    rng = random.Random(_stable_int_seed(seed, "sample_records", strategy, target, len(records)))
    if strategy == "random":
        selected_keys = {_record_key(record) for record in rng.sample(records, k=target)}
        return [record for record in records if _record_key(record) in selected_keys]
    if strategy not in {"stratified_label", "balanced_label"}:
        raise ValueError(f"Unsupported --sample_strategy: {strategy}")
    if label_by_key is None:
        raise ValueError(f"--sample_strategy {strategy} requires label_by_key")
    groups: Dict[int, List[Dict[str, Any]]] = {0: [], 1: []}
    for record in records:
        label = label_by_key.get(_record_key(record))
        if label in {0, 1}:
            groups[int(label)].append(record)
    if not groups[0] and not groups[1]:
        raise ValueError("No labeled context records available for stratified sampling")
    if strategy == "balanced_label":
        half = target // 2
        alloc = {0: min(len(groups[0]), half), 1: min(len(groups[1]), target - min(len(groups[0]), half))}
        if sum(alloc.values()) < target:
            for label in sorted(groups, key=lambda x: len(groups[x]), reverse=True):
                take = min(len(groups[label]) - alloc[label], target - sum(alloc.values()))
                alloc[label] += max(0, take)
    else:
        alloc = _allocate_stratified_counts({label: len(items) for label, items in groups.items()}, target)
    selected_keys: set[str] = set()
    for label, items in groups.items():
        take = min(int(alloc.get(label, 0)), len(items))
        if take <= 0:
            continue
        for item in rng.sample(items, k=take):
            selected_keys.add(_record_key(item))
    return [record for record in records if _record_key(record) in selected_keys]


def _summary_value_from_row(row: Dict[str, Any]) -> Optional[str]:
    summary_fields = row.get("summary_fields") if isinstance(row.get("summary_fields"), dict) else {}
    candidates = [
        summary_fields.get("llm_summary_text") if isinstance(summary_fields, dict) else None,
        row.get("llm_summary_text"),
        row.get("summary_text"),
        row.get("diagnosis"),
    ]
    for value in candidates:
        text = str(value or "").strip()
        if text:
            return text
    payload = row.get("payload")
    if isinstance(payload, dict):
        return _summary_value_from_row(payload)
    return None


def _summary_key_from_row(row: Dict[str, Any]) -> Optional[str]:
    explicit = str(row.get("sample_id") or row.get("key") or "").strip()
    if explicit:
        return explicit
    user_id = str(row.get("user_id") or "").strip()
    target_pid = str(row.get("target_pid") or row.get("target_question_id") or "").strip()
    if user_id and target_pid and row.get("target_t") is not None:
        try:
            return _context_key(user_id, int(row.get("target_t")), target_pid)
        except Exception:
            return None
    return None


def _load_summary_maps(paths: Sequence[Path]) -> Dict[str, str]:
    summary_by_key: Dict[str, str] = {}
    for path in paths:
        if not path or not path.exists():
            continue
        for row in _load_jsonl(path):
            key = _summary_key_from_row(row)
            text = _summary_value_from_row(row)
            if key and text and not str(key).startswith("prompt-signature\t"):
                summary_by_key[key] = text
    return summary_by_key


def _extract_llm_summary(record: Dict[str, Any], summary_by_key: Dict[str, str]) -> Tuple[Optional[str], str]:
    summary_fields = record.get("summary_fields") if isinstance(record.get("summary_fields"), dict) else {}
    text = str(summary_fields.get("llm_summary_text") or "").strip() if isinstance(summary_fields, dict) else ""
    if text:
        return text, "contexts.summary_fields.llm_summary_text"
    key = _context_key(str(record.get("user_id") or ""), int(record.get("target_t") or 0), str(record.get("target_pid") or ""))
    if key in summary_by_key and str(summary_by_key[key]).strip():
        return str(summary_by_key[key]).strip(), "summary_jsonl_or_cache"
    return None, "missing"


def _build_cognitive_evidence(record: Dict[str, Any], *, k: int, max_text_chars: int) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    raw_evidence = record.get("evidence_list") or []
    if not isinstance(raw_evidence, list):
        raise ValueError("context record field evidence_list must be a list")
    target_t = int(record.get("target_t") or 0)
    for idx, ev in enumerate(raw_evidence[: max(0, int(k))], start=1):
        if not isinstance(ev, dict):
            continue
        history_pos = ev.get("history_pos")
        try:
            history_pos_int = int(history_pos)
        except Exception:
            history_pos_int = -1
        answer = _answer_to_int(ev.get("answer_result"))
        evidence.append(
            {
                "rank": ev.get("rank") or idx,
                "problem_id": str(ev.get("problem_id") or ""),
                "semantic_id": str(ev.get("semantic_id") or ""),
                "role": ev.get("role"),
                "knowledge_overlap": ev.get("knowledge_overlap"),
                "level_diff": ev.get("level_diff"),
                "support_score": _safe_float(ev.get("support_score")),
                "answer": answer,
                "answer_text": _answer_label(answer),
                "history_pos": history_pos_int if history_pos_int >= 0 else None,
                "steps_before_target": max(0, target_t - history_pos_int) if history_pos_int >= 0 else None,
                "text": _compact_text(ev.get("question_text") or ev.get("text"), max_text_chars),
                "source": "cognitive_retrieval",
            }
        )
    return evidence


def _build_recent_evidence(history: List[Dict[str, Any]], *, k: int) -> List[Dict[str, Any]]:
    selected = history[-max(0, int(k)) :] if int(k) > 0 else []
    evidence: List[Dict[str, Any]] = []
    for idx, item in enumerate(selected, start=1):
        copied = dict(item)
        copied.update({"rank": idx, "role": "recent_history", "source": "recent_history"})
        evidence.append(copied)
    return evidence


def _build_random_evidence(history: List[Dict[str, Any]], *, k: int, seed: int, sample_id: str) -> List[Dict[str, Any]]:
    if int(k) <= 0 or not history:
        return []
    rng = random.Random(_stable_int_seed(seed, sample_id, "wo_cognitive_retrieval_random"))
    selected = rng.sample(history, k=min(int(k), len(history)))
    selected = sorted(selected, key=lambda item: int(item.get("history_pos") or 0))
    evidence: List[Dict[str, Any]] = []
    for idx, item in enumerate(selected, start=1):
        copied = dict(item)
        copied.update({"rank": idx, "role": "random_history", "source": "random_history"})
        evidence.append(copied)
    return evidence


def _history_stats(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    answers = [int(item["answer"]) for item in history if item.get("answer") in {0, 1}]
    recent_5 = answers[-5:]
    recent_10 = answers[-10:]
    return {
        "history_count": len(history),
        "history_correct_count": int(sum(answers)),
        "history_wrong_count": int(len(answers) - sum(answers)),
        "history_correct_rate": (float(sum(answers)) / float(len(answers))) if answers else None,
        "recent_5_correct_rate": (float(sum(recent_5)) / float(len(recent_5))) if recent_5 else None,
        "recent_10_correct_rate": (float(sum(recent_10)) / float(len(recent_10))) if recent_10 else None,
        "last_answer": answers[-1] if answers else None,
    }


def _evidence_stats(evidence_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    answers = [int(item["answer"]) for item in evidence_items if item.get("answer") in {0, 1}]
    supports: List[float] = []
    weighted_num = 0.0
    weighted_den = 0.0
    for item in evidence_items:
        answer = item.get("answer")
        support = item.get("support_score")
        if answer in {0, 1} and isinstance(support, (int, float)) and math.isfinite(float(support)):
            weight = max(0.0, float(support))
            weighted_num += float(answer) * weight
            weighted_den += weight
            supports.append(float(support))
    return {
        "evidence_count": len(evidence_items),
        "evidence_correct_count": int(sum(answers)),
        "evidence_wrong_count": int(len(answers) - sum(answers)),
        "evidence_correct_rate": (float(sum(answers)) / float(len(answers))) if answers else None,
        "weighted_evidence_correct_rate": (weighted_num / weighted_den) if weighted_den > 0 else None,
        "support_mean": (float(sum(supports)) / float(len(supports))) if supports else None,
    }


def _coarse_trajectory(history: List[Dict[str, Any]], *, max_history_items: int) -> List[Dict[str, Any]]:
    window = history[-max(0, int(max_history_items)) :] if int(max_history_items) > 0 else history
    out: List[Dict[str, Any]] = []
    for item in window:
        out.append(
            {
                "history_pos": item.get("history_pos"),
                "problem_id": item.get("problem_id"),
                "semantic_id": item.get("semantic_id"),
                "concepts": item.get("concepts") or [],
                "cognitive_dimension": item.get("cognitive_dimension"),
                "answer": item.get("answer_text"),
            }
        )
    return out


def _prompt_variant_inputs(
    sample: Dict[str, Any],
    variant: str,
    evidence_items: List[Dict[str, Any]],
    llm_summary: Optional[str],
    *,
    max_history_items: int,
) -> Dict[str, Any]:
    target = sample["target"]
    history = sample["history"]
    prompt_input: Dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "variant": variant,
        "target": target,
        "student_trajectory_coarse": _coarse_trajectory(history, max_history_items=max_history_items),
        "history_stats_before_target": _history_stats(history),
    }
    if variant != "wo_structured_evidence":
        prompt_input["structured_evidence"] = evidence_items
        prompt_input["evidence_stats"] = _evidence_stats(evidence_items)
    if variant in {"full_cognitive_rag_llm", "wo_cognitive_retrieval_recent"}:
        if not llm_summary:
            raise MissingSummaryError(
                f"{variant} requires its evidence-specific LLM summary, but none was found for this sample"
            )
        prompt_input["llm_summary_or_diagnosis"] = llm_summary
    return prompt_input


def build_prediction_prompt(
    sample: Dict[str, Any],
    variant: str,
    evidence_items: List[Dict[str, Any]],
    llm_summary: Optional[str] = None,
    *,
    max_history_items: int = 50,
) -> PromptBundle:
    """Build the hard-label prediction prompt for one ablation variant."""
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported variant: {variant}")
    prompt_input = _prompt_variant_inputs(
        sample,
        variant,
        evidence_items,
        llm_summary,
        max_history_items=max_history_items,
    )
    variant_rules = {
        "full_cognitive_rag_llm": "Use cognitive retrieval evidence and the provided LLM summary/diagnosis.",
        "wo_cognitive_retrieval_recent": "Do not use cognitive retrieval. Use recent-history evidence and its LLM summary instead.",
        "wo_cognitive_retrieval_random": "Do not use cognitive retrieval. Use the deterministic random history evidence instead.",
        "wo_llm_summary": "Use cognitive retrieval evidence directly. Do not use any LLM summary or diagnosis.",
        "wo_structured_evidence": "Do not use fine-grained retrieved evidence or any summary. Use only the coarse trajectory and target.",
    }
    system_prompt = (
        "You are an educational assessment and knowledge tracing expert. "
        "Predict whether the student will answer the target problem correctly."
    )
    user_prompt = (
        "Task: predict the target response as a hard label.\n"
        f"Ablation variant: {variant}.\n"
        f"Variant rule: {variant_rules[variant]}\n"
        "The input never contains the target's true answer. Use only pre-target history and the target metadata.\n"
        "Return exactly one character: 0 for incorrect, or 1 for correct. Do not output explanations, JSON, markdown, or extra text.\n"
        "Input:\n"
        + json.dumps(prompt_input, ensure_ascii=False, separators=(",", ":"))
    )
    return PromptBundle(system_prompt=system_prompt, user_prompt=user_prompt, prompt_input=prompt_input)


class OpenAICompatibleLabelClient:
    """OpenAI-compatible chat client for 0/1 labels and optional token logprobs."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout_sec: int,
        max_tokens: int,
        temperature: float,
        retries: int,
        seed: int,
        disable_thinking: bool = False,
        use_chat_template_kwargs: bool = False,
        request_logprobs: bool = False,
        logprob_top_k: int = 20,
        verbose: bool = False,
    ) -> None:
        if not str(base_url or "").strip():
            raise ValueError("--llm_base_url is required")
        if not str(model or "").strip():
            raise ValueError("--llm_model is required")
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self.api_key = str(api_key or "").strip()
        self.timeout_sec = int(timeout_sec)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.retries = max(1, int(retries))
        self.seed = int(seed)
        self.disable_thinking = bool(disable_thinking)
        self._allow_chat_template_kwargs = bool(disable_thinking and use_chat_template_kwargs)
        self.request_logprobs = bool(request_logprobs)
        self.logprob_top_k = int(logprob_top_k)
        self.verbose = bool(verbose)
        if self.request_logprobs and not 2 <= self.logprob_top_k <= 20:
            raise ValueError("--logprob_top_k must be in [2, 20] when --prediction_mode token_logprob")

    def complete(self, *, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        user_content = self._maybe_disable_thinking(user_prompt)
        last_error: Optional[Exception] = None
        request_attempt = 0
        retry_count = 0
        while retry_count < self.retries:
            request_attempt += 1
            body = self._request_body(
                system_prompt=system_prompt,
                user_content=user_content,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            if self.verbose:
                print(
                    json.dumps(
                        {
                            "event": "llm_request_start",
                            "model": self.model,
                            "attempt": request_attempt,
                            "timeout_sec": self.timeout_sec,
                            "prompt_chars": len(system_prompt) + len(user_content),
                            "disable_thinking": self.disable_thinking,
                            "chat_template_kwargs_used": self._allow_chat_template_kwargs,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            req = request.Request(
                url=self.base_url + "/chat/completions",
                data=payload,
                headers=headers,
                method="POST",
            )
            try:
                started = time.perf_counter()
                with request.urlopen(req, timeout=self.timeout_sec) as resp:
                    raw_response = resp.read().decode("utf-8", errors="replace")
                data = json.loads(raw_response)
                choice = data["choices"][0]
                content = self._flatten_content(choice["message"]["content"])
                if self.verbose:
                    print(
                        json.dumps(
                            {
                                "event": "llm_request_ok",
                                "model": self.model,
                                "attempt": request_attempt,
                                "duration_sec": time.perf_counter() - started,
                                "content_preview": content[:80],
                                "logprobs_returned": isinstance(choice.get("logprobs"), dict),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                return {
                    "raw_response": raw_response,
                    "content": content,
                    "logprobs": choice.get("logprobs"),
                    "attempts": request_attempt,
                }
            except error.HTTPError as exc:
                body_text = ""
                try:
                    body_text = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                last_error = RuntimeError(f"HTTP {exc.code}: {body_text or exc.reason}")
                if self._disable_unsupported_chat_template_kwargs(exc):
                    continue
            except (error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
                last_error = exc
            retry_count += 1
            if self.verbose:
                print(
                    json.dumps(
                        {
                            "event": "llm_request_error",
                            "model": self.model,
                            "attempt": request_attempt,
                            "error": f"{type(last_error).__name__}: {last_error}",
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if retry_count < self.retries:
                time.sleep(min(8.0, 0.5 * (2 ** (retry_count - 1))))
        raise RuntimeError(f"LLM request failed after {self.retries} attempts: {last_error}")

    def preflight(self, *, timeout_sec: int) -> Dict[str, Any]:
        """Send a tiny chat request so API or model routing failures surface before workers start."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        while True:
            body = self._request_body(
                system_prompt="You are a connectivity checker.",
                user_content=self._maybe_disable_thinking("Return exactly 0."),
                temperature=0.0,
                max_tokens=8,
            )
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            req = request.Request(
                url=self.base_url + "/chat/completions",
                data=payload,
                headers=headers,
                method="POST",
            )
            started = time.perf_counter()
            try:
                with request.urlopen(req, timeout=max(1, int(timeout_sec))) as resp:
                    raw_response = resp.read().decode("utf-8", errors="replace")
                break
            except error.HTTPError as exc:
                if self._disable_unsupported_chat_template_kwargs(exc):
                    continue
                raise
        data = json.loads(raw_response)
        choice = data["choices"][0]
        content = self._flatten_content(choice["message"]["content"])
        return {
            "content": content,
            "logprobs": choice.get("logprobs"),
            "duration_sec": time.perf_counter() - started,
        }

    def _request_body(
        self,
        *,
        system_prompt: str,
        user_content: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": self.seed,
        }
        if self._allow_chat_template_kwargs:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if self.request_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = self.logprob_top_k
        return body

    def _disable_unsupported_chat_template_kwargs(self, exc: error.HTTPError) -> bool:
        if not self._allow_chat_template_kwargs or int(exc.code) not in {400, 422}:
            return False
        self._allow_chat_template_kwargs = False
        if self.verbose:
            print(
                json.dumps(
                    {
                        "event": "llm_chat_template_kwargs_fallback",
                        "model": self.model,
                        "http_status": int(exc.code),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        return True

    def _maybe_disable_thinking(self, user_prompt: str) -> str:
        text = str(user_prompt)
        if self.disable_thinking and "/no_think" not in text:
            return text.rstrip() + "\n/no_think"
        return text

    @staticmethod
    def _flatten_content(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content or "")


def parse_hard_label(text: str) -> int:
    """Parse an LLM response into 0 or 1, rejecting ambiguous outputs."""
    raw = str(text or "").strip()
    if raw in {"0", "1"}:
        return int(raw)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            for field in ("prediction", "y_pred", "label", "correct"):
                if field in obj:
                    parsed = _answer_to_int(obj[field])
                    if parsed in {0, 1}:
                        return int(parsed)
    except Exception:
        pass
    matches = re.findall(r"(?<![\w.])([01])(?![\w.])", raw)
    unique = sorted(set(matches))
    if len(unique) == 1:
        return int(unique[0])
    if not unique:
        raise ValueError(f"Could not parse a standalone 0/1 label from response: {raw[:160]}")
    raise ValueError(f"Ambiguous 0/1 labels in response: {raw[:160]}")


def _binary_label_token(value: Any) -> Optional[int]:
    """Map a generated token to a binary label when it is exactly 0 or 1 after whitespace stripping."""
    text = str(value or "").strip()
    if text in {"0", "1"}:
        return int(text)
    return None


def extract_binary_token_probability(logprobs: Any, *, y_pred: int) -> Dict[str, Any]:
    """Return P(correct) from the final 0/1 token's binary-renormalized logprobs.

    The score is exp(log P(1)) / (exp(log P(0)) + exp(log P(1))) at the
    generated label position. Both candidates must be returned by the server;
    otherwise a probability metric would be unsupported and the sample fails.
    """
    if y_pred not in {0, 1}:
        raise ValueError(f"Expected y_pred in {{0, 1}}, got {y_pred!r}")
    if not isinstance(logprobs, dict):
        raise ValueError("Response has no choices[0].logprobs object")
    content = logprobs.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("Response logprobs has no non-empty content list")

    matching_label_positions = 0
    for token_index in range(len(content) - 1, -1, -1):
        entry = content[token_index]
        if not isinstance(entry, dict):
            continue
        emitted_label = _binary_label_token(entry.get("token"))
        if emitted_label != y_pred:
            continue
        matching_label_positions += 1
        label_logprobs: Dict[int, float] = {}
        emitted_logprob = _safe_float(entry.get("logprob"))
        if emitted_logprob is not None:
            label_logprobs[emitted_label] = emitted_logprob
        top_logprobs = entry.get("top_logprobs")
        if isinstance(top_logprobs, list):
            for candidate in top_logprobs:
                if not isinstance(candidate, dict):
                    continue
                candidate_label = _binary_label_token(candidate.get("token"))
                candidate_logprob = _safe_float(candidate.get("logprob"))
                if candidate_label in {0, 1} and candidate_logprob is not None:
                    old_value = label_logprobs.get(candidate_label)
                    if old_value is None or candidate_logprob > old_value:
                        label_logprobs[candidate_label] = candidate_logprob
        if 0 not in label_logprobs or 1 not in label_logprobs:
            continue
        logprob_0 = label_logprobs[0]
        logprob_1 = label_logprobs[1]
        max_logprob = max(logprob_0, logprob_1)
        prob_1 = math.exp(logprob_1 - max_logprob)
        prob_0 = math.exp(logprob_0 - max_logprob)
        prob_correct = prob_1 / (prob_0 + prob_1)
        if not math.isfinite(prob_correct) or not 0.0 <= prob_correct <= 1.0:
            raise ValueError("Binary-renormalized token probability is invalid")
        return {
            "prob_correct": float(prob_correct),
            "probability_method": TOKEN_LOGPROB_METHOD,
            "label_logprob_0": float(logprob_0),
            "label_logprob_1": float(logprob_1),
            "label_token_index": int(token_index),
        }
    raise ValueError(
        "Could not find a generated final-label token whose top_logprobs contains both 0 and 1; "
        f"matching_label_positions={matching_label_positions}. Increase --logprob_top_k up to 20, "
        "or use a vLLM server that returns token logprobs."
    )


def _valid_variants(value: str) -> List[str]:
    variants = [part.strip() for part in str(value or "").split(",") if part.strip()]
    if not variants:
        raise ValueError("--variants must contain at least one variant")
    unknown = [variant for variant in variants if variant not in SUPPORTED_VARIANTS]
    if unknown:
        raise ValueError(f"Unsupported variants {unknown}. Supported variants: {sorted(SUPPORTED_VARIANTS)}")
    return variants


def _has_valid_probability(row: Dict[str, Any]) -> bool:
    probability = _safe_float(row.get("prob_correct"))
    return probability is not None and 0.0 <= probability <= 1.0


def _read_completed_predictions(path: Path, *, prediction_mode: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return done
    for row in _load_jsonl(path):
        sample_id = str(row.get("sample_id") or row.get("key") or "").strip()
        if prediction_mode == "token_logprob" and not _has_valid_probability(row):
            continue
        if sample_id:
            done[sample_id] = row
    return done


def _count_failed(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


class BufferedJsonlWriter:
    """Append JSONL rows and flush periodically without sacrificing resumability."""

    def __init__(self, handle: Any, *, flush_every: int) -> None:
        self.handle = handle
        self.flush_every = max(1, int(flush_every))
        self.pending_rows = 0

    def write(self, row: Dict[str, Any]) -> None:
        self.handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.pending_rows += 1
        if self.pending_rows >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        self.handle.flush()
        self.pending_rows = 0


def _iter_bounded_futures(
    executor: ThreadPoolExecutor,
    records: Iterable[Dict[str, Any]],
    worker_fn: Callable[[Dict[str, Any]], Tuple[str, Dict[str, Any]]],
    *,
    max_in_flight: int,
) -> Iterator[Tuple[Any, Dict[str, Any]]]:
    """Yield completed futures while keeping only a bounded request backlog in memory."""
    record_iter = iter(records)
    futures: Dict[Any, Dict[str, Any]] = {}

    def submit_one() -> bool:
        try:
            record = next(record_iter)
        except StopIteration:
            return False
        futures[executor.submit(worker_fn, record)] = record
        return True

    for _ in range(max(1, int(max_in_flight))):
        if not submit_one():
            break
    while futures:
        completed, _ = wait(futures, return_when=FIRST_COMPLETED)
        for future in completed:
            record = futures.pop(future)
            yield future, record
            submit_one()


def _binary_auc(y_true: Sequence[int], probabilities: Sequence[float]) -> Optional[float]:
    """Compute ROC-AUC using average ranks, without requiring sklearn."""
    if len(y_true) != len(probabilities) or not y_true:
        return None
    positive_count = sum(1 for value in y_true if int(value) == 1)
    negative_count = len(y_true) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    ordered = sorted(enumerate(probabilities), key=lambda item: item[1])
    ranks = [0.0] * len(probabilities)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (float(start + 1) + float(end)) / 2.0
        for index in range(start, end):
            ranks[ordered[index][0]] = average_rank
        start = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, y_true) if int(label) == 1)
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / float(positive_count * negative_count)


def _compute_metrics(
    prediction_rows: Iterable[Dict[str, Any]],
    *,
    failed_count: int,
    variant: str,
    prediction_mode: str,
) -> Dict[str, Any]:
    rows = [row for row in prediction_rows if row.get("y_true") in {0, 1} and row.get("y_pred") in {0, 1}]
    tp = sum(1 for row in rows if int(row["y_true"]) == 1 and int(row["y_pred"]) == 1)
    fp = sum(1 for row in rows if int(row["y_true"]) == 0 and int(row["y_pred"]) == 1)
    tn = sum(1 for row in rows if int(row["y_true"]) == 0 and int(row["y_pred"]) == 0)
    fn = sum(1 for row in rows if int(row["y_true"]) == 1 and int(row["y_pred"]) == 0)
    n = len(rows)
    precision = float(tp) / float(tp + fp) if tp + fp > 0 else 0.0
    recall = float(tp) / float(tp + fn) if tp + fn > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    y_true_sum = sum(int(row["y_true"]) for row in rows)
    y_pred_sum = sum(int(row["y_pred"]) for row in rows)
    probability_rows = [row for row in rows if _has_valid_probability(row)]
    probability_y_true = [int(row["y_true"]) for row in probability_rows]
    probabilities = [float(row["prob_correct"]) for row in probability_rows]
    probability_count = len(probability_rows)
    if probability_count:
        clipped = [min(1.0 - 1e-15, max(1e-15, probability)) for probability in probabilities]
        bce = -sum(
            label * math.log(probability) + (1 - label) * math.log(1.0 - probability)
            for label, probability in zip(probability_y_true, clipped)
        ) / float(probability_count)
        rmse = math.sqrt(
            sum((probability - label) ** 2 for label, probability in zip(probability_y_true, probabilities))
            / float(probability_count)
        )
        methods = sorted({str(row.get("probability_method") or "unknown") for row in probability_rows})
        note = (
            f"probability metrics computed from {probability_count}/{n} valid prob_correct values; "
            f"method={','.join(methods)}"
        )
    else:
        bce = None
        rmse = None
        note = "hard-label predictions only"
    return {
        "variant": variant,
        "prediction_mode": prediction_mode,
        "sample_count": n,
        "failed_count": int(failed_count),
        "acc": (float(tp + tn) / float(n)) if n else None,
        "precision": precision if n else None,
        "recall": recall if n else None,
        "f1": f1 if n else None,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "positive_rate_true": (float(y_true_sum) / float(n)) if n else None,
        "positive_rate_pred": (float(y_pred_sum) / float(n)) if n else None,
        "probability_sample_count": probability_count,
        "probability_coverage": (float(probability_count) / float(n)) if n else None,
        "auc": _binary_auc(probability_y_true, probabilities) if probability_count else None,
        "bce": bce,
        "rmse": rmse,
        "note": note,
    }


def _sample_from_record(
    record: Dict[str, Any],
    history_info: Dict[str, Any],
    catalog: Dict[str, Dict[str, Any]],
    *,
    max_text_chars: int,
) -> Dict[str, Any]:
    target_pid = str(record.get("target_pid") or "")
    target_t = int(record.get("target_t") or 0)
    user_id = str(record.get("user_id") or "")
    sample_id = _context_key(user_id, target_t, target_pid)
    if not user_id or not target_pid:
        raise ValueError(f"Invalid context identity for sample {sample_id!r}")
    if target_pid not in catalog:
        raise ValueError(f"Target problem {target_pid} is absent from problem catalog")
    if not history_info:
        raise ValueError(f"Missing label/history for sample {sample_id}; check student_json filtering and context target_t")
    y_true = history_info.get("y_true")
    if y_true not in {0, 1}:
        raise ValueError(f"Missing or invalid y_true for sample {sample_id}: {y_true}")
    target = _target_payload(
        catalog[target_pid],
        target_pid,
        str(record.get("target_semantic_id") or ""),
        max_text_chars,
    )
    return {
        "sample_id": sample_id,
        "user_id": user_id,
        "target_t": target_t,
        "target_pid": target_pid,
        "target_semantic_id": str(record.get("target_semantic_id") or ""),
        "y_true": int(y_true),
        "history": history_info.get("history") or [],
        "record": record,
        "target": target,
    }


def _evidence_for_variant(
    *,
    sample: Dict[str, Any],
    variant: str,
    k: int,
    seed: int,
    max_text_chars: int,
) -> List[Dict[str, Any]]:
    if variant in {"full_cognitive_rag_llm", "wo_llm_summary"}:
        return _build_cognitive_evidence(sample["record"], k=k, max_text_chars=max_text_chars)
    if variant == "wo_cognitive_retrieval_recent":
        return _build_recent_evidence(sample["history"], k=k)
    if variant == "wo_cognitive_retrieval_random":
        return _build_random_evidence(sample["history"], k=k, seed=seed, sample_id=sample["sample_id"])
    if variant == "wo_structured_evidence":
        return []
    raise ValueError(f"Unsupported variant: {variant}")


def _llm_summary_for_variant(
    sample: Dict[str, Any],
    variant: str,
    full_summary_by_key: Dict[str, str],
    recent_summary_by_key: Dict[str, str],
) -> Tuple[Optional[str], str]:
    if variant == "full_cognitive_rag_llm":
        return _extract_llm_summary(sample["record"], full_summary_by_key)
    if variant == "wo_cognitive_retrieval_recent":
        text = str(recent_summary_by_key.get(sample["sample_id"]) or "").strip()
        return (text or None), "recent_evidence_llm_summary"
    return None, "not_used"


def _failure_row(
    *,
    sample: Optional[Dict[str, Any]],
    record: Dict[str, Any],
    variant: str,
    error_message: str,
    prediction_mode: str,
    prompt_hash: Optional[str] = None,
    evidence_items: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    user_id = str(record.get("user_id") or (sample or {}).get("user_id") or "")
    target_t = record.get("target_t", (sample or {}).get("target_t"))
    target_pid = str(record.get("target_pid") or (sample or {}).get("target_pid") or "")
    sample_id = (sample or {}).get("sample_id") or (
        _context_key(user_id, int(target_t or 0), target_pid) if user_id and target_pid else ""
    )
    return {
        "sample_id": sample_id,
        "user_id": user_id,
        "target_t": target_t,
        "target_question_id": target_pid,
        "variant": variant,
        "prediction_mode": prediction_mode,
        "y_true": (sample or {}).get("y_true"),
        "prompt_hash": prompt_hash,
        "evidence_question_ids": [str(item.get("problem_id") or "") for item in (evidence_items or [])],
        "num_evidence": len(evidence_items or []),
        "failed_at": _now_iso(),
        "error": error_message,
    }


def _predict_sample(
    *,
    client: OpenAICompatibleLabelClient,
    sample: Dict[str, Any],
    variant: str,
    summary_by_key: Dict[str, str],
    recent_summary_by_key: Dict[str, str],
    k: int,
    seed: int,
    max_text_chars: int,
    max_history_items: int,
) -> Dict[str, Any]:
    evidence_items = _evidence_for_variant(
        sample=sample,
        variant=variant,
        k=k,
        seed=seed,
        max_text_chars=max_text_chars,
    )
    llm_summary, llm_summary_source = _llm_summary_for_variant(
        sample,
        variant,
        summary_by_key,
        recent_summary_by_key,
    )
    bundle = build_prediction_prompt(
        sample,
        variant,
        evidence_items,
        llm_summary=llm_summary,
        max_history_items=max_history_items,
    )
    prompt_hash = _prompt_hash(bundle.system_prompt, bundle.user_prompt)
    started_perf = time.perf_counter()
    started_at = _now_iso()
    response = client.complete(system_prompt=bundle.system_prompt, user_prompt=bundle.user_prompt)
    duration_sec = time.perf_counter() - started_perf
    y_pred = parse_hard_label(response["content"])
    probability_info: Dict[str, Any] = {}
    prediction_mode = "token_logprob" if client.request_logprobs else "hard_label"
    if client.request_logprobs:
        probability_info = extract_binary_token_probability(response.get("logprobs"), y_pred=y_pred)
    return {
        "sample_id": sample["sample_id"],
        "key": sample["sample_id"],
        "user_id": sample["user_id"],
        "target_t": sample["target_t"],
        "target_question_id": sample["target_pid"],
        "target_pid": sample["target_pid"],
        "target_semantic_id": sample["target_semantic_id"],
        "variant": variant,
        "prediction_mode": prediction_mode,
        "y_true": sample["y_true"],
        "y_pred": int(y_pred),
        "raw_response": response["content"],
        "prompt_hash": prompt_hash,
        "evidence_question_ids": [str(item.get("problem_id") or "") for item in evidence_items],
        "num_evidence": len(evidence_items),
        "llm_summary_source": llm_summary_source,
        "prompt_version": PROMPT_VERSION,
        "started_at": started_at,
        "duration_sec": duration_sec,
        "llm_attempts": int(response.get("attempts") or 1),
        **probability_info,
    }


def _prepare_variant_dir(out_dir: Path, variant: str, *, resume: bool, overwrite: bool) -> Dict[str, Path]:
    variant_dir = ensure_dir(out_dir / variant)
    paths = {
        "dir": variant_dir,
        "predictions": variant_dir / "predictions.jsonl",
        "metrics": variant_dir / "metrics.json",
        "run_config": variant_dir / "run_config.json",
        "failed": variant_dir / "failed.jsonl",
        "prompts": variant_dir / "prompts.jsonl",
    }
    if overwrite:
        for name in ("predictions", "metrics", "run_config", "failed", "prompts"):
            paths[name].unlink(missing_ok=True)
    elif not resume and (paths["predictions"].exists() or paths["failed"].exists()):
        raise FileExistsError(f"{variant_dir} already has outputs. Use --resume or --overwrite.")
    return paths


def _existing_prediction_mode(run_config_path: Path) -> Optional[str]:
    if not run_config_path.exists():
        return None
    try:
        with run_config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise ValueError(f"Cannot read existing run config {run_config_path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Existing run config must be a JSON object: {run_config_path}")
    mode = str(data.get("prediction_mode") or "hard_label").strip()
    if mode not in PREDICTION_MODES:
        raise ValueError(f"Existing run config has unsupported prediction_mode {mode!r}: {run_config_path}")
    return mode


def run_variant(
    *,
    variant: str,
    records: List[Dict[str, Any]],
    histories_by_key: Dict[str, Dict[str, Any]],
    catalog: Dict[str, Dict[str, Any]],
    summary_by_key: Dict[str, str],
    recent_summary_by_key: Dict[str, str],
    args: argparse.Namespace,
    schema_preview: List[Dict[str, Any]],
) -> Dict[str, Any]:
    paths = _prepare_variant_dir(Path(args.out_dir), variant, resume=bool(args.resume), overwrite=bool(args.overwrite))
    if bool(args.resume) and paths["predictions"].exists():
        existing_mode = _existing_prediction_mode(paths["run_config"])
        if existing_mode is None:
            raise ValueError(
                f"Cannot safely resume {paths['dir']}: predictions exist but run_config.json is missing. "
                "Use a new --out_dir or --overwrite."
            )
        if existing_mode != str(args.prediction_mode):
            raise ValueError(
                f"Cannot resume {paths['dir']} with --prediction_mode {args.prediction_mode!r}; "
                f"existing predictions use {existing_mode!r}. Use a new --out_dir or --overwrite."
            )
    done = _read_completed_predictions(paths["predictions"], prediction_mode=str(args.prediction_mode)) if args.resume else {}
    workers = max(1, int(args.workers))
    max_in_flight = int(args.max_in_flight) if int(args.max_in_flight) > 0 else workers * 8
    save_prompt_rows = bool(args.save_prompts) or bool(args.dry_run_prompts)
    client: Optional[OpenAICompatibleLabelClient] = None
    if not bool(args.dry_run_prompts):
        client = OpenAICompatibleLabelClient(
            base_url=str(args.llm_base_url),
            model=str(args.llm_model),
            api_key=str(args.llm_api_key),
            timeout_sec=int(args.llm_timeout_sec),
            max_tokens=int(args.max_tokens),
            temperature=float(args.temperature),
            retries=int(args.retries),
            seed=int(args.seed),
            disable_thinking=bool(args.llm_disable_thinking),
            use_chat_template_kwargs=bool(args.llm_use_chat_template_kwargs),
            request_logprobs=str(args.prediction_mode) == "token_logprob",
            logprob_top_k=int(args.logprob_top_k),
            verbose=bool(args.verbose_requests),
        )
    pending_records = []
    for record in records:
        key = _context_key(str(record.get("user_id") or ""), int(record.get("target_t") or 0), str(record.get("target_pid") or ""))
        if key in done:
            continue
        pending_records.append(record)

    run_config = {
        "variant": variant,
        "prompt_version": PROMPT_VERSION,
        "started_at": _now_iso(),
        "problem_json": str(args.problem_json),
        "student_json": str(args.student_json),
        "contexts_jsonl": str(args.contexts_jsonl),
        "eval_manifest_jsonl": str(args.eval_manifest_jsonl or ""),
        "out_dir": str(args.out_dir),
        "k": int(args.k),
        "max_samples": int(args.max_samples),
        "sample_size": int(args.sample_size),
        "sample_strategy": str(args.sample_strategy),
        "offset": int(args.offset),
        "workers": workers,
        "max_in_flight": max_in_flight,
        "flush_every": int(args.flush_every),
        "save_prompts": bool(args.save_prompts),
        "temperature": float(args.temperature),
        "max_tokens": int(args.max_tokens),
        "prediction_mode": str(args.prediction_mode),
        "logprob_top_k": int(args.logprob_top_k),
        "seed": int(args.seed),
        "resume": bool(args.resume),
        "overwrite": bool(args.overwrite),
        "dry_run_prompts": bool(args.dry_run_prompts),
        "llm_base_url": str(args.llm_base_url),
        "llm_model": str(args.llm_model),
        "llm_timeout_sec": int(args.llm_timeout_sec),
        "retries": int(args.retries),
        "llm_disable_thinking": bool(args.llm_disable_thinking),
        "llm_use_chat_template_kwargs": bool(args.llm_use_chat_template_kwargs),
        "input_record_count": len(records),
        "existing_done_count": len(done),
        "pending_count": len(pending_records),
        "available_llm_summary_count": len(summary_by_key),
        "available_recent_llm_summary_count": len(recent_summary_by_key),
        "schema_preview": schema_preview,
    }
    write_json(run_config, paths["run_config"])

    failures_this_run = 0
    prompts_this_run = 0

    def _make_sample(record: Dict[str, Any]) -> Dict[str, Any]:
        key = _context_key(str(record.get("user_id") or ""), int(record.get("target_t") or 0), str(record.get("target_pid") or ""))
        return _sample_from_record(
            record,
            histories_by_key.get(key, {}),
            catalog,
            max_text_chars=int(args.max_text_chars),
        )

    def _run(record: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        sample: Optional[Dict[str, Any]] = None
        evidence_items: List[Dict[str, Any]] = []
        prompt_hash: Optional[str] = None
        try:
            sample = _make_sample(record)
            evidence_items = _evidence_for_variant(
                sample=sample,
                variant=variant,
                k=int(args.k),
                seed=int(args.seed),
                max_text_chars=int(args.max_text_chars),
            )
            llm_summary, llm_summary_source = _llm_summary_for_variant(
                sample,
                variant,
                summary_by_key,
                recent_summary_by_key,
            )
            bundle = build_prediction_prompt(
                sample,
                variant,
                evidence_items,
                llm_summary=llm_summary,
                max_history_items=int(args.max_history_items),
            )
            prompt_hash = _prompt_hash(bundle.system_prompt, bundle.user_prompt)
            prompt_row: Optional[Dict[str, Any]] = None
            if save_prompt_rows:
                prompt_row = {
                    "sample_id": sample["sample_id"],
                    "variant": variant,
                    "prompt_hash": prompt_hash,
                    "system_prompt": bundle.system_prompt,
                    "user_prompt": bundle.user_prompt,
                    "prompt_input": bundle.prompt_input,
                    "llm_summary_source": llm_summary_source,
                }
            if bool(args.dry_run_prompts):
                assert prompt_row is not None
                return "prompt", prompt_row
            assert client is not None
            prediction = _predict_sample(
                client=client,
                sample=sample,
                variant=variant,
                summary_by_key=summary_by_key,
                recent_summary_by_key=recent_summary_by_key,
                k=int(args.k),
                seed=int(args.seed),
                max_text_chars=int(args.max_text_chars),
                max_history_items=int(args.max_history_items),
            )
            prediction["llm_summary_source"] = llm_summary_source
            return "prediction", {"prediction": prediction, "prompt": prompt_row}
        except Exception as exc:
            item = _failure_row(
                sample=sample,
                record=record,
                variant=variant,
                error_message=f"{type(exc).__name__}: {exc}",
                prediction_mode=str(args.prediction_mode),
                prompt_hash=prompt_hash,
                evidence_items=evidence_items,
            )
            return "failure", item

    total = len(pending_records)
    progress = tqdm(total=total, desc=f"ablation:{variant}", unit="sample", dynamic_ncols=True) if tqdm else None
    prompt_context = paths["prompts"].open("a", encoding="utf-8") if save_prompt_rows else nullcontext(None)
    with (
        paths["predictions"].open("a", encoding="utf-8") as pred_f,
        paths["failed"].open("a", encoding="utf-8") as fail_f,
        prompt_context as prompt_f,
    ):
        prediction_writer = BufferedJsonlWriter(pred_f, flush_every=int(args.flush_every))
        failure_writer = BufferedJsonlWriter(fail_f, flush_every=int(args.flush_every))
        prompt_writer = BufferedJsonlWriter(prompt_f, flush_every=int(args.flush_every)) if prompt_f is not None else None

        def persist_result(status: str, payload: Dict[str, Any]) -> None:
            nonlocal failures_this_run, prompts_this_run
            if status == "prediction":
                prediction_writer.write(payload["prediction"])
                if prompt_writer is not None and payload.get("prompt") is not None:
                    prompt_writer.write(payload["prompt"])
                    prompts_this_run += 1
            elif status == "prompt":
                assert prompt_writer is not None
                prompt_writer.write(payload)
                prompts_this_run += 1
            else:
                failures_this_run += 1
                failure_writer.write(payload)

        if workers == 1:
            for record in pending_records:
                status, payload = _run(record)
                persist_result(status, payload)
                if progress:
                    progress.update(1)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for future, _record in _iter_bounded_futures(
                    executor,
                    pending_records,
                    _run,
                    max_in_flight=max_in_flight,
                ):
                    status, payload = future.result()
                    persist_result(status, payload)
                    if progress:
                        progress.update(1)
        prediction_writer.flush()
        failure_writer.flush()
        if prompt_writer is not None:
            prompt_writer.flush()
    if progress:
        progress.close()

    all_predictions = list(
        _read_completed_predictions(paths["predictions"], prediction_mode=str(args.prediction_mode)).values()
    )
    failed_count = _count_failed(paths["failed"])
    metrics_out = _compute_metrics(
        all_predictions,
        failed_count=failed_count,
        variant=variant,
        prediction_mode=str(args.prediction_mode),
    )
    metrics_out.update(
        {
            "finished_at": _now_iso(),
            "predictions_path": str(paths["predictions"]),
            "failed_path": str(paths["failed"]),
            "run_config_path": str(paths["run_config"]),
            "failure_count_this_run": failures_this_run,
            "prompt_count_this_run": prompts_this_run,
            "pending_count_this_run": total,
            "dry_run_prompts": bool(args.dry_run_prompts),
        }
    )
    write_json(metrics_out, paths["metrics"])
    print(
        json.dumps(
            {
                "event": "variant_finished",
                "variant": variant,
                "sample_count": metrics_out["sample_count"],
                "failed_count": metrics_out["failed_count"],
                "metrics_path": str(paths["metrics"]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return metrics_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM-direct-prediction ablations from existing strict cognitive contexts."
    )
    parser.add_argument("--problem_json", required=True)
    parser.add_argument("--student_json", required=True)
    parser.add_argument("--contexts_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument(
        "--eval_manifest_jsonl",
        default="",
        help="Frozen held-out test manifest. When set, only its sample_id rows are evaluated.",
    )
    parser.add_argument(
        "--recent_summary_jsonl",
        default="",
        help="Required for wo_cognitive_retrieval_recent: LLM summaries generated only from recent-K evidence.",
    )
    parser.add_argument("--summary_jsonl", default="", help="Optional contexts/cache JSONL containing existing LLM summaries.")
    parser.add_argument("--summary_cache_jsonl", default="", help="Optional llm_summary_cache.jsonl with legacy sample keys.")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--llm_base_url", required=True)
    parser.add_argument("--llm_model", required=True)
    parser.add_argument("--llm_api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--max_in_flight",
        type=int,
        default=0,
        help="Maximum concurrent submitted requests; 0 uses workers * 8.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument(
        "--prediction_mode",
        choices=sorted(PREDICTION_MODES),
        default="hard_label",
        help="hard_label keeps the prior protocol; token_logprob requests vLLM token logprobs and writes prob_correct.",
    )
    parser.add_argument(
        "--logprob_top_k",
        type=int,
        default=20,
        help="Number of top completion-token logprobs requested in token_logprob mode; valid range is 2-20.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0, help="Limit context rows; 0 means no limit/full run.")
    parser.add_argument("--sample_size", type=int, default=0, help="Deterministically sample this many context rows after loading candidates; 0 disables sampling.")
    parser.add_argument(
        "--sample_strategy",
        choices=["first", "random", "stratified_label", "balanced_label"],
        default="stratified_label",
        help="Sampling strategy used when --sample_size is positive.",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--flush_every",
        type=int,
        default=100,
        help="Flush JSONL outputs every N rows; interrupted runs safely replay at most one unflushed batch.",
    )
    parser.add_argument(
        "--save_prompts",
        action="store_true",
        help="Persist full per-sample prompts for audit/debug. Disabled by default to avoid large duplicate I/O.",
    )
    parser.add_argument("--llm_timeout_sec", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--llm_disable_thinking", action="store_true")
    parser.add_argument("--llm_use_chat_template_kwargs", action="store_true")
    parser.add_argument("--api_preflight_timeout_sec", type=int, default=15)
    parser.add_argument("--skip_api_preflight", action="store_true")
    parser.add_argument("--verbose_requests", action="store_true")
    parser.add_argument("--max_text_chars", type=int, default=260)
    parser.add_argument("--max_history_items", type=int, default=50)
    parser.add_argument(
        "--dry_run_prompts",
        action="store_true",
        help="Build prompts and run configs without calling the LLM; useful for local smoke tests.",
    )
    parser.add_argument("--inspect_schema_only", action="store_true")
    args = parser.parse_args()

    args.problem_json = str(Path(args.problem_json).resolve())
    args.student_json = str(Path(args.student_json).resolve())
    args.contexts_jsonl = str(Path(args.contexts_jsonl).resolve())
    args.out_dir = str(Path(args.out_dir).resolve())
    if str(args.eval_manifest_jsonl or "").strip():
        args.eval_manifest_jsonl = str(Path(args.eval_manifest_jsonl).resolve())
    if str(args.recent_summary_jsonl or "").strip():
        args.recent_summary_jsonl = str(Path(args.recent_summary_jsonl).resolve())

    variants = _valid_variants(str(args.variants))
    if str(args.prediction_mode) == "token_logprob" and not 2 <= int(args.logprob_top_k) <= 20:
        raise ValueError("--logprob_top_k must be in [2, 20] when --prediction_mode token_logprob")
    if int(args.max_in_flight) < 0:
        raise ValueError("--max_in_flight must be non-negative")
    if int(args.flush_every) < 1:
        raise ValueError("--flush_every must be positive")
    contexts_jsonl = Path(args.contexts_jsonl)
    schema_preview = validate_context_schema(contexts_jsonl)
    print(json.dumps({"event": "context_schema", "preview": schema_preview}, ensure_ascii=False), flush=True)
    if args.inspect_schema_only:
        return
    if (
        not str(args.eval_manifest_jsonl or "").strip()
        and int(args.max_samples) == 0
        and int(args.sample_size) == 0
        and not bool(args.dry_run_prompts)
    ):
        print("[WARN] --max_samples 0 means no limit; this will run all context rows.", flush=True)
    if not bool(args.dry_run_prompts) and not bool(args.skip_api_preflight):
        preflight_client = OpenAICompatibleLabelClient(
            base_url=str(args.llm_base_url),
            model=str(args.llm_model),
            api_key=str(args.llm_api_key),
            timeout_sec=int(args.llm_timeout_sec),
            max_tokens=int(args.max_tokens),
            temperature=float(args.temperature),
            retries=1,
            seed=int(args.seed),
            disable_thinking=bool(args.llm_disable_thinking),
            use_chat_template_kwargs=bool(args.llm_use_chat_template_kwargs),
            request_logprobs=str(args.prediction_mode) == "token_logprob",
            logprob_top_k=int(args.logprob_top_k),
            verbose=False,
        )
        print(
            json.dumps(
                {
                    "event": "llm_preflight_start",
                    "base_url": str(args.llm_base_url),
                    "model": str(args.llm_model),
                    "timeout_sec": int(args.api_preflight_timeout_sec),
                    "disable_thinking": bool(args.llm_disable_thinking),
                    "use_chat_template_kwargs": bool(args.llm_use_chat_template_kwargs),
                    "prediction_mode": str(args.prediction_mode),
                    "logprob_top_k": int(args.logprob_top_k),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        try:
            preflight_result = preflight_client.preflight(timeout_sec=int(args.api_preflight_timeout_sec))
            preflight_probability: Optional[Dict[str, Any]] = None
            if str(args.prediction_mode) == "token_logprob":
                preflight_label = parse_hard_label(str(preflight_result.get("content") or ""))
                preflight_probability = extract_binary_token_probability(
                    preflight_result.get("logprobs"),
                    y_pred=preflight_label,
                )
        except Exception as exc:
            raise RuntimeError(
                "LLM API preflight failed before running ablations. "
                "Check that the vLLM OpenAI-compatible server is reachable, the model name is correct, "
                f"and a tiny /chat/completions request returns within {int(args.api_preflight_timeout_sec)} seconds. "
                "In token_logprob mode, the response must expose choices[0].logprobs.content with both 0 and 1 "
                "candidates for the generated label token. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        print(
            json.dumps(
                {
                    "event": "llm_preflight_ok",
                    "duration_sec": preflight_result["duration_sec"],
                    "content_preview": str(preflight_result.get("content") or "")[:80],
                    "probability_preview": (
                        preflight_probability.get("prob_correct") if preflight_probability is not None else None
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    catalog = _load_problem_catalog(Path(args.problem_json), contexts_jsonl)
    label_by_key: Dict[str, int] = {}
    eval_manifest_by_key: Dict[str, Dict[str, Any]] = {}
    if str(args.eval_manifest_jsonl or "").strip():
        if int(args.max_samples) > 0 or int(args.sample_size) > 0 or int(args.offset) > 0:
            raise ValueError(
                "--eval_manifest_jsonl fixes the evaluation cohort; do not combine it with "
                "--max_samples, --sample_size, or --offset"
            )
        eval_manifest_by_key = _load_eval_manifest(Path(args.eval_manifest_jsonl))
        all_context_records = _load_context_records(contexts_jsonl)
        candidate_record_count = len(all_context_records)
        records = [record for record in all_context_records if _record_key(record) in eval_manifest_by_key]
        matched_keys = {_record_key(record) for record in records}
        missing_keys = sorted(set(eval_manifest_by_key) - matched_keys)
        if missing_keys:
            preview = missing_keys[:3]
            raise ValueError(
                f"Evaluation manifest has {len(missing_keys)} samples absent from contexts_jsonl; examples={preview}"
            )
        label_by_key = {key: int(row["y_true"]) for key, row in eval_manifest_by_key.items()}
    else:
        records = _load_context_records(contexts_jsonl, max_samples=int(args.max_samples), offset=int(args.offset))
        candidate_record_count = len(records)
        if int(args.sample_size) > 0 and str(args.sample_strategy) in {"stratified_label", "balanced_label"}:
            candidate_keys = {_record_key(record) for record in records}
            print(
                json.dumps(
                    {
                        "event": "label_scan_start",
                        "candidate_record_count": candidate_record_count,
                        "sample_size": int(args.sample_size),
                        "sample_strategy": str(args.sample_strategy),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            label_by_key = _load_labels_for_keys(Path(args.student_json), catalog, candidate_keys)
            label_counts = {
                "0": sum(1 for value in label_by_key.values() if value == 0),
                "1": sum(1 for value in label_by_key.values() if value == 1),
            }
            print(
                json.dumps(
                    {
                        "event": "label_scan_done",
                        "labeled_count": len(label_by_key),
                        "label_counts": label_counts,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        records = _select_records_for_run(
            records,
            sample_size=int(args.sample_size),
            sample_strategy=str(args.sample_strategy),
            seed=int(args.seed),
            label_by_key=label_by_key if label_by_key else None,
        )
    selected_keys_for_log = {_record_key(record) for record in records}
    selected_label_counts = {
        "0": sum(1 for key in selected_keys_for_log if label_by_key.get(key) == 0),
        "1": sum(1 for key in selected_keys_for_log if label_by_key.get(key) == 1),
    } if label_by_key else {}
    print(
        json.dumps(
            {
                "event": "context_records_selected",
                "candidate_record_count": candidate_record_count,
                "selected_record_count": len(records),
                "eval_manifest_jsonl": str(args.eval_manifest_jsonl or ""),
                "sample_size": int(args.sample_size),
                "sample_strategy": str(args.sample_strategy),
                "selected_label_counts": selected_label_counts,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    needed_keys = {
        _record_key(record)
        for record in records
    }
    histories_by_key = _load_histories_and_labels(
        Path(args.student_json),
        catalog,
        needed_keys,
        max_text_chars=int(args.max_text_chars),
    )
    if eval_manifest_by_key:
        missing_history_keys = sorted(set(eval_manifest_by_key) - set(histories_by_key))
        if missing_history_keys:
            raise ValueError(
                f"Evaluation manifest has {len(missing_history_keys)} samples absent from student_json; "
                f"examples={missing_history_keys[:3]}"
            )
        mismatched_labels = [
            key
            for key, manifest_row in eval_manifest_by_key.items()
            if int(histories_by_key[key]["y_true"]) != int(manifest_row["y_true"])
        ]
        if mismatched_labels:
            raise ValueError(
                f"Evaluation manifest label mismatch for {len(mismatched_labels)} samples; "
                f"examples={mismatched_labels[:3]}"
            )

    summary_paths: List[Path] = []
    if str(args.summary_jsonl or "").strip():
        summary_paths.append(Path(args.summary_jsonl).resolve())
    if str(args.summary_cache_jsonl or "").strip():
        summary_paths.append(Path(args.summary_cache_jsonl).resolve())
    auto_cache = contexts_jsonl.resolve().parent.parent / "cache" / "llm_summary_cache.jsonl"
    if auto_cache.exists():
        summary_paths.append(auto_cache)
    summary_by_key = _load_summary_maps(summary_paths)
    for record in records:
        key = _context_key(str(record.get("user_id") or ""), int(record.get("target_t") or 0), str(record.get("target_pid") or ""))
        text, source = _extract_llm_summary(record, {})
        if text and source != "missing":
            summary_by_key[key] = text
    recent_summary_by_key: Dict[str, str] = {}
    if str(args.recent_summary_jsonl or "").strip():
        recent_summary_by_key = _load_summary_maps([Path(args.recent_summary_jsonl)])
    if "wo_cognitive_retrieval_recent" in variants:
        if not str(args.recent_summary_jsonl or "").strip():
            raise ValueError(
                "wo_cognitive_retrieval_recent requires --recent_summary_jsonl generated from the same eval manifest"
            )
        missing_recent_summaries = sorted({_record_key(record) for record in records} - set(recent_summary_by_key))
        if missing_recent_summaries:
            raise ValueError(
                f"Recent-summary file is missing {len(missing_recent_summaries)} selected samples; "
                f"examples={missing_recent_summaries[:3]}"
            )

    ensure_dir(Path(args.out_dir))
    top_config = {
        "started_at": _now_iso(),
        "problem_json": args.problem_json,
        "student_json": args.student_json,
        "contexts_jsonl": args.contexts_jsonl,
        "eval_manifest_jsonl": str(args.eval_manifest_jsonl or ""),
        "recent_summary_jsonl": str(args.recent_summary_jsonl or ""),
        "out_dir": args.out_dir,
        "variants": variants,
        "k": int(args.k),
        "max_samples": int(args.max_samples),
        "sample_size": int(args.sample_size),
        "sample_strategy": str(args.sample_strategy),
        "offset": int(args.offset),
        "seed": int(args.seed),
        "dry_run_prompts": bool(args.dry_run_prompts),
        "workers": int(args.workers),
        "max_in_flight": int(args.max_in_flight),
        "flush_every": int(args.flush_every),
        "save_prompts": bool(args.save_prompts),
        "prediction_mode": str(args.prediction_mode),
        "logprob_top_k": int(args.logprob_top_k),
        "skip_api_preflight": bool(args.skip_api_preflight),
        "api_preflight_timeout_sec": int(args.api_preflight_timeout_sec),
        "llm_disable_thinking": bool(args.llm_disable_thinking),
        "llm_use_chat_template_kwargs": bool(args.llm_use_chat_template_kwargs),
        "verbose_requests": bool(args.verbose_requests),
        "summary_paths": [str(path) for path in summary_paths],
        "candidate_record_count": candidate_record_count,
        "loaded_record_count": len(records),
        "selected_label_counts": selected_label_counts,
        "history_label_count": len(histories_by_key),
        "available_llm_summary_count": len(summary_by_key),
        "available_recent_llm_summary_count": len(recent_summary_by_key),
        "schema_preview": schema_preview,
    }
    write_json(top_config, Path(args.out_dir) / "run_config.json")
    if "full_cognitive_rag_llm" in variants and not summary_by_key:
        print(
            "[WARN] full_cognitive_rag_llm was requested but no existing LLM summary file/field was found; "
            "that variant will record missing-summary failures instead of fabricating a replacement summary.",
            flush=True,
        )

    all_metrics: List[Dict[str, Any]] = []
    for variant in variants:
        metrics_out = run_variant(
            variant=variant,
            records=records,
            histories_by_key=histories_by_key,
            catalog=catalog,
            summary_by_key=summary_by_key,
            recent_summary_by_key=recent_summary_by_key,
            args=args,
            schema_preview=schema_preview,
        )
        all_metrics.append(metrics_out)

    summary_path = Path(args.out_dir) / "metrics_overview.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "variant",
            "sample_count",
            "failed_count",
            "acc",
            "precision",
            "recall",
            "f1",
            "positive_rate_true",
            "positive_rate_pred",
            "probability_sample_count",
            "probability_coverage",
            "auc",
            "bce",
            "rmse",
            "note",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_metrics:
            writer.writerow({field: row.get(field) for field in fieldnames})
    print("[OK] llm ablation run finished", flush=True)
    print(f"[OUT_DIR] {Path(args.out_dir)}", flush=True)
    print(f"[OVERVIEW] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
