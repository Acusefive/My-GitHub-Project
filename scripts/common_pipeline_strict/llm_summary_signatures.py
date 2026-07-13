from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

from tqdm import tqdm

from .llm_utils import (
    OpenAICompatibleSummarizer,
    load_summary_cache,
    parse_llm_summary_json,
    summary_cache_key,
)


LLM_SUMMARY_SIGNATURE_PREFIX = "prompt-signature\t"


def build_llm_summary_prompts_for_record(
    *,
    record: Dict[str, Any],
    problem_catalog_records: Dict[str, Dict[str, Any]],
    summarizer: OpenAICompatibleSummarizer,
) -> Tuple[str, str]:
    target_pid = str(record["target_pid"])
    target_meta = problem_catalog_records[target_pid]
    return summarizer.build_summary_prompts(
        target_pid=target_pid,
        target_question_text=str(target_meta["text"]),
        target_semantic_id=str(record.get("target_semantic_id") or target_meta["semantic_id"]),
        target_concepts=record.get("summary_fields", {}).get("target_concepts") or target_meta["concepts"],
        evidence_list=record.get("evidence_list") or [],
        template_summary_text=str(record.get("summary_fields", {}).get("summary_text") or ""),
    )


def llm_summary_signature_key(
    *,
    record: Dict[str, Any],
    problem_catalog_records: Dict[str, Dict[str, Any]],
    summarizer: OpenAICompatibleSummarizer,
) -> str:
    system_prompt, user_prompt = build_llm_summary_prompts_for_record(
        record=record,
        problem_catalog_records=problem_catalog_records,
        summarizer=summarizer,
    )
    return summarizer.prompt_signature(system_prompt=system_prompt, user_prompt=user_prompt)


def summarize_llm_record(
    *,
    record: Dict[str, Any],
    problem_catalog_records: Dict[str, Dict[str, Any]],
    summarizer: OpenAICompatibleSummarizer,
) -> str:
    target_pid = str(record["target_pid"])
    target_meta = problem_catalog_records[target_pid]
    system_prompt, user_prompt = build_llm_summary_prompts_for_record(
        record=record,
        problem_catalog_records=problem_catalog_records,
        summarizer=summarizer,
    )
    return summarizer.summarize_from_prompts(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        target_pid=target_pid,
        target_semantic_id=str(record.get("target_semantic_id") or target_meta["semantic_id"]),
    )


def build_llm_summary_case_for_record(
    *,
    case_id: str,
    record: Dict[str, Any],
    problem_catalog_records: Dict[str, Dict[str, Any]],
) -> Dict[str, object]:
    target_pid = str(record["target_pid"])
    target_meta = problem_catalog_records[target_pid]
    return {
        "case_id": str(case_id),
        "target_pid": target_pid,
        "target_question_text": str(target_meta["text"]),
        "target_semantic_id": str(record.get("target_semantic_id") or target_meta["semantic_id"]),
        "target_concepts": record.get("summary_fields", {}).get("target_concepts") or target_meta["concepts"],
        "evidence_list": record.get("evidence_list") or [],
        "template_summary_text": str(record.get("summary_fields", {}).get("summary_text") or ""),
    }


def summarize_llm_records_batch(
    *,
    record_items: Iterable[Tuple[str, Dict[str, Any]]],
    problem_catalog_records: Dict[str, Dict[str, Any]],
    summarizer: OpenAICompatibleSummarizer,
) -> Dict[str, str]:
    cases = [
        build_llm_summary_case_for_record(
            case_id=case_id,
            record=record,
            problem_catalog_records=problem_catalog_records,
        )
        for case_id, record in record_items
    ]
    return summarizer.summarize_batch(cases=cases)


def _valid_llm_summary_text(summary_text: str, *, validate: bool) -> bool:
    text = str(summary_text or "").strip()
    if not text:
        return False
    if not validate:
        return True
    try:
        parse_llm_summary_json(text)
        return True
    except Exception:
        return False


def collect_llm_summary_signature_stats(
    *,
    contexts_path: Path,
    cache_dir: Path,
    problem_catalog_records: Dict[str, Dict[str, Any]],
    summarizer: OpenAICompatibleSummarizer,
    validate_cache: bool = False,
    sample_limit: Optional[int] = None,
    total_records_hint: Optional[int] = None,
) -> Dict[str, Any]:
    """Scan contexts and estimate real LLM request count under signature caching.

    This function does not call the LLM and does not write context artifacts.
    """
    llm_cache_path = cache_dir / "llm_summary_cache.jsonl"
    raw_llm_cache = load_summary_cache(llm_cache_path)
    cache_valid_memo: Dict[str, bool] = {}

    def _cache_has_valid_summary(key: str) -> bool:
        if key in cache_valid_memo:
            return cache_valid_memo[key]
        ok = _valid_llm_summary_text(raw_llm_cache.get(key, ""), validate=validate_cache)
        cache_valid_memo[key] = ok
        return ok

    initial_signature_keys: Set[str] = {
        key
        for key in raw_llm_cache
        if str(key).startswith(LLM_SUMMARY_SIGNATURE_PREFIX) and _cache_has_valid_summary(str(key))
    }
    available_signature_keys: Set[str] = set(initial_signature_keys)
    streaming_request_signatures: Set[str] = set()
    signature_flags: Dict[str, int] = {}

    flag_signature_cache = 1
    flag_legacy_cache = 2
    flag_context_summary = 4
    flag_streaming_request = 8

    total_records = 0
    selected_count_histogram: Dict[str, int] = {}
    signature_cache_hit_records = 0
    same_run_signature_reuse_records = 0
    legacy_cache_hit_records = 0
    context_summary_hit_records = 0
    streaming_request_records = 0

    with contexts_path.open("r", encoding="utf-8", errors="replace") as src:
        progress_total = int(total_records_hint or 0) if sample_limit is None else int(sample_limit)
        progress = tqdm(total=progress_total if progress_total > 0 else None, desc="strict llm signature scan")
        try:
            for line in src:
                if not line.strip():
                    continue
                record = json.loads(line)
                total_records += 1
                selected_count = str(record.get("selected_count", len(record.get("evidence_list") or [])))
                selected_count_histogram[selected_count] = selected_count_histogram.get(selected_count, 0) + 1
                signature_key = llm_summary_signature_key(
                    record=record,
                    problem_catalog_records=problem_catalog_records,
                    summarizer=summarizer,
                )
                state = signature_flags.get(signature_key, 0)

                legacy_key = summary_cache_key(record["user_id"], int(record["target_t"]), record["target_pid"])
                record_summary_text = str(record.get("summary_fields", {}).get("llm_summary_text") or "").strip()
                record_llm_context_text = str(record.get("llm_context_text") or "").strip()
                has_valid_context_summary = bool(record_llm_context_text) and _valid_llm_summary_text(
                    record_summary_text,
                    validate=validate_cache,
                )

                if signature_key in initial_signature_keys:
                    signature_cache_hit_records += 1
                    state |= flag_signature_cache
                elif signature_key in available_signature_keys:
                    same_run_signature_reuse_records += 1
                elif _cache_has_valid_summary(legacy_key):
                    legacy_cache_hit_records += 1
                    available_signature_keys.add(signature_key)
                    state |= flag_legacy_cache
                elif has_valid_context_summary:
                    context_summary_hit_records += 1
                    available_signature_keys.add(signature_key)
                    state |= flag_context_summary
                else:
                    if signature_key not in streaming_request_signatures:
                        streaming_request_signatures.add(signature_key)
                        available_signature_keys.add(signature_key)
                    streaming_request_records += 1
                    state |= flag_streaming_request

                signature_flags[signature_key] = state
                progress.update(1)
                if sample_limit is not None and total_records >= int(sample_limit):
                    break
        finally:
            progress.close()

    unique_signatures = len(signature_flags)
    signatures_with_signature_cache = sum(1 for flags in signature_flags.values() if flags & flag_signature_cache)
    signatures_with_legacy_cache = sum(1 for flags in signature_flags.values() if flags & flag_legacy_cache)
    signatures_with_context_summary = sum(1 for flags in signature_flags.values() if flags & flag_context_summary)
    signatures_requiring_request_after_full_scan = sum(
        1
        for flags in signature_flags.values()
        if not (flags & (flag_signature_cache | flag_legacy_cache | flag_context_summary))
    )
    duplicate_records_by_signature = total_records - unique_signatures
    records_covered_without_new_llm_after_streaming = (
        signature_cache_hit_records
        + same_run_signature_reuse_records
        + legacy_cache_hit_records
        + context_summary_hit_records
    )

    return {
        "contexts_path": str(contexts_path),
        "llm_cache_path": str(llm_cache_path),
        "sample_limit": int(sample_limit) if sample_limit is not None else None,
        "total_records_scanned": total_records,
        "total_records_hint": int(total_records_hint or 0),
        "raw_cache_entries": len(raw_llm_cache),
        "valid_signature_cache_entries": len(initial_signature_keys),
        "unique_signatures": unique_signatures,
        "duplicate_records_by_signature": duplicate_records_by_signature,
        "selected_count_histogram": selected_count_histogram,
        "signature_cache_hit_records": signature_cache_hit_records,
        "same_run_signature_reuse_records": same_run_signature_reuse_records,
        "legacy_cache_hit_records": legacy_cache_hit_records,
        "context_summary_hit_records": context_summary_hit_records,
        "streaming_request_records": streaming_request_records,
        "expected_llm_requests_streaming": len(streaming_request_signatures),
        "expected_llm_requests_after_full_prescan": signatures_requiring_request_after_full_scan,
        "avoidable_requests_by_full_prescan": max(
            0,
            len(streaming_request_signatures) - signatures_requiring_request_after_full_scan,
        ),
        "records_covered_without_new_llm_after_streaming": records_covered_without_new_llm_after_streaming,
        "signatures_with_signature_cache": signatures_with_signature_cache,
        "signatures_with_legacy_cache": signatures_with_legacy_cache,
        "signatures_with_context_summary": signatures_with_context_summary,
        "validate_cache": bool(validate_cache),
    }
