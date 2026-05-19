from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from common_pipeline_strict.llm_utils import OpenAICompatibleSummarizer, load_summary_cache


def load_problem_catalog(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            pid = str(record.get("problem_id") or "")
            if pid:
                records[pid] = record
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contexts_path",
        type=Path,
        default=Path("out/xes3g5m_text_only_strict_common_pipeline/contexts/contexts.jsonl"),
    )
    parser.add_argument("--problem_catalog_path", type=Path, default=None)
    parser.add_argument("--cache_path", type=Path, default=None)
    parser.add_argument("--llm_model", type=str, default="Qwen3-8B")
    parser.add_argument("--llm_max_tokens", type=int, default=256)
    parser.add_argument("--llm_temperature", type=float, default=0.0)
    parser.add_argument("--progress_every", type=int, default=500000)
    args = parser.parse_args()

    contexts_path = args.contexts_path
    root = contexts_path.parent.parent
    problem_catalog_path = args.problem_catalog_path or (root / "priors" / "problem_catalog.jsonl")
    cache_path = args.cache_path or (root / "cache" / "llm_summary_cache.jsonl")
    problem_catalog = load_problem_catalog(problem_catalog_path)
    cache = {
        key: value
        for key, value in load_summary_cache(cache_path).items()
        if str(key).startswith("prompt-signature\t")
    }

    summarizer = OpenAICompatibleSummarizer(
        base_url="http://127.0.0.1:8000/v1",
        model=args.llm_model,
        api_key="",
        timeout_sec=1,
        max_tokens=args.llm_max_tokens,
        temperature=args.llm_temperature,
    )

    rows = 0
    seen = set()
    cache_hits = 0
    target_semantic = Counter()
    selected_count = Counter()
    dominant_role = Counter()

    with contexts_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            record = json.loads(line)
            target_pid = str(record["target_pid"])
            target_meta = problem_catalog[target_pid]
            system_prompt, user_prompt = summarizer.build_summary_prompts(
                target_pid=target_pid,
                target_question_text=str(target_meta["text"]),
                target_semantic_id=str(record.get("target_semantic_id") or target_meta["semantic_id"]),
                target_concepts=(record.get("summary_fields") or {}).get("target_concepts") or target_meta["concepts"],
                evidence_list=record.get("evidence_list") or [],
                template_summary_text=str((record.get("summary_fields") or {}).get("summary_text") or ""),
            )
            signature = summarizer.prompt_signature(system_prompt=system_prompt, user_prompt=user_prompt)
            seen.add(signature)
            if signature in cache:
                cache_hits += 1
            target_semantic[str(record.get("target_semantic_id") or "")] += 1
            selected_count[len(record.get("evidence_list") or [])] += 1
            dominant_role[str((record.get("summary_fields") or {}).get("dominant_role") or "")] += 1
            if args.progress_every > 0 and rows % args.progress_every == 0:
                unique_count = len(seen)
                print(
                    "[progress]",
                    rows,
                    "unique_prompt_signatures",
                    unique_count,
                    "reuse_ratio",
                    f"{1 - unique_count / rows:.2%}",
                    "cache_hits",
                    cache_hits,
                    flush=True,
                )

    unique_count = len(seen)
    unique_cache_hits = len(seen.intersection(cache))
    expected_llm_requests = unique_count - unique_cache_hits
    duplicate_miss_records = max(0, rows - cache_hits - expected_llm_requests)
    print("[total_records]", rows)
    print("[unique_prompt_signatures]", unique_count)
    print("[reuse_ratio]", f"{1 - unique_count / max(rows, 1):.2%}")
    print("[prompt_cache_entries]", len(cache))
    print("[cache_hits]", cache_hits)
    print("[unique_cached_signatures]", unique_cache_hits)
    print("[unique_miss_signatures]", expected_llm_requests)
    print("[duplicate_miss_records]", duplicate_miss_records)
    print("[expected_llm_requests]", expected_llm_requests)
    print("[selected_count]", selected_count.most_common())
    print("[dominant_role]", dominant_role.most_common())
    print("[target_semantic_top]", target_semantic.most_common(20))


if __name__ == "__main__":
    main()
