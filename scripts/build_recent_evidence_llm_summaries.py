"""Build resumable LLM summaries using only recent-K history evidence for the retrieval ablation."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple

from common_pipeline_strict.llm_utils import OpenAICompatibleSummarizer
from run_llm_ablation_experiments import (
    _build_recent_evidence,
    _compact_text,
    _context_key,
    _load_context_records,
    _load_eval_manifest,
    _load_histories_and_labels,
    _load_problem_catalog,
    _record_key,
    _target_payload,
)


RECENT_EVIDENCE_ONLY_CONTEXT = (
    "No auxiliary statistical or template summary is provided. "
    "Base the diagnosis only on the listed recent-history evidence."
)


def _read_done(path: Path) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = str(row.get("sample_id") or "").strip()
            summary_text = str(row.get("summary_text") or "").strip()
            if sample_id and summary_text:
                done[sample_id] = row
    return done


class BufferedJsonlWriter:
    """Append JSONL rows and flush periodically while retaining restart safety."""

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
    worker_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    *,
    max_in_flight: int,
) -> Iterator[Tuple[Any, Dict[str, Any]]]:
    """Yield completed summary tasks without enqueuing the full manifest at once."""
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


def _summary_evidence_items(evidence_items: List[Dict[str, Any]], target: Dict[str, Any]) -> List[Dict[str, Any]]:
    target_concepts = {str(value) for value in target.get("concepts") or []}
    target_level = target.get("cognitive_dimension")
    out: List[Dict[str, Any]] = []
    for item in evidence_items:
        item_concepts = {str(value) for value in item.get("concepts") or []}
        overlap = sorted(target_concepts & item_concepts)
        history_level = item.get("cognitive_dimension")
        try:
            level_diff: Any = int(target_level) - int(history_level)
        except Exception:
            level_diff = ""
        out.append(
            {
                "problem_id": str(item.get("problem_id") or ""),
                "semantic_id": str(item.get("semantic_id") or ""),
                "history_pos": item.get("history_pos"),
                "role": "recent_history",
                "knowledge_overlap": ",".join(overlap) if overlap else "none",
                "level_diff": level_diff,
                "answer_result": str(item.get("answer_text") or ""),
                "support_score": "",
                "question_text": _compact_text(item.get("text"), 260),
                "activation": {},
                "raw_scores": {},
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate resumable LLM summaries for recent-K evidence on a frozen held-out test manifest."
    )
    parser.add_argument("--problem_json", required=True)
    parser.add_argument("--student_json", required=True)
    parser.add_argument("--contexts_jsonl", required=True)
    parser.add_argument("--eval_manifest_jsonl", required=True)
    parser.add_argument("--out_summary_jsonl", required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--llm_base_url", required=True)
    parser.add_argument("--llm_model", required=True)
    parser.add_argument("--llm_api_key", default="EMPTY")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max_in_flight",
        type=int,
        default=0,
        help="Maximum submitted summary requests; 0 uses workers * 8.",
    )
    parser.add_argument("--llm_timeout_sec", type=int, default=120)
    parser.add_argument("--summary_max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--flush_every",
        type=int,
        default=100,
        help="Flush output JSONL every N rows; resume safely replays at most one unflushed batch.",
    )
    parser.add_argument("--llm_disable_thinking", action="store_true")
    parser.add_argument("--llm_use_chat_template_kwargs", action="store_true")
    args = parser.parse_args()
    if int(args.max_in_flight) < 0:
        raise ValueError("--max_in_flight must be non-negative")
    if int(args.flush_every) < 1:
        raise ValueError("--flush_every must be positive")

    problem_json = Path(args.problem_json).resolve()
    student_json = Path(args.student_json).resolve()
    contexts_jsonl = Path(args.contexts_jsonl).resolve()
    eval_manifest_jsonl = Path(args.eval_manifest_jsonl).resolve()
    output = Path(args.out_summary_jsonl).resolve()
    if output.exists() and args.overwrite:
        output.unlink()
    elif output.exists() and not args.resume:
        raise FileExistsError(f"Summary output exists: {output}. Use --resume or --overwrite.")

    manifest = _load_eval_manifest(eval_manifest_jsonl)
    catalog = _load_problem_catalog(problem_json, contexts_jsonl)
    contexts = _load_context_records(contexts_jsonl)
    records = [record for record in contexts if _record_key(record) in manifest]
    found = {_record_key(record) for record in records}
    missing = sorted(set(manifest) - found)
    if missing:
        raise ValueError(f"Manifest samples absent from contexts: count={len(missing)}, examples={missing[:3]}")
    histories = _load_histories_and_labels(
        student_json,
        catalog,
        found,
        max_text_chars=260,
    )
    missing_history = sorted(found - set(histories))
    if missing_history:
        raise ValueError(f"Manifest samples absent from student history: count={len(missing_history)}, examples={missing_history[:3]}")
    mismatched = [key for key in found if int(histories[key]["y_true"]) != int(manifest[key]["y_true"])]
    if mismatched:
        raise ValueError(f"Manifest label mismatch: count={len(mismatched)}, examples={mismatched[:3]}")

    done = _read_done(output) if args.resume else {}
    pending = [record for record in records if _record_key(record) not in done]
    workers = max(1, int(args.workers))
    max_in_flight = int(args.max_in_flight) if int(args.max_in_flight) > 0 else workers * 8
    config_path = output.with_suffix(".config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "problem_json": str(problem_json),
                "student_json": str(student_json),
                "contexts_jsonl": str(contexts_jsonl),
                "eval_manifest_jsonl": str(eval_manifest_jsonl),
                "out_summary_jsonl": str(output),
                "k": int(args.k),
                "llm_model": str(args.llm_model),
                "llm_base_url": str(args.llm_base_url),
                "summary_max_tokens": int(args.summary_max_tokens),
                "temperature": float(args.temperature),
                "workers": workers,
                "max_in_flight": max_in_flight,
                "flush_every": int(args.flush_every),
                "summary_context_source": "recent_evidence_only_no_template_summary",
                "selected_count": len(records),
                "existing_summary_count": len(done),
                "pending_count": len(pending),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summarizer = OpenAICompatibleSummarizer(
        base_url=str(args.llm_base_url),
        model=str(args.llm_model),
        api_key=str(args.llm_api_key),
        timeout_sec=int(args.llm_timeout_sec),
        max_tokens=int(args.summary_max_tokens),
        temperature=float(args.temperature),
        disable_thinking=bool(args.llm_disable_thinking),
        use_chat_template_kwargs=bool(args.llm_use_chat_template_kwargs),
        compact_prompt=True,
    )

    def run_one(record: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = _record_key(record)
        history_info = histories[sample_id]
        target_pid = str(record["target_pid"])
        target = _target_payload(
            catalog[target_pid],
            target_pid,
            str(record.get("target_semantic_id") or ""),
            260,
        )
        evidence = _build_recent_evidence(history_info["history"], k=int(args.k))
        summary_text = summarizer.summarize(
            target_pid=target_pid,
            target_question_text=str(target.get("text") or ""),
            target_semantic_id=str(target.get("semantic_id") or ""),
            target_concepts=target.get("concepts") or [],
            evidence_list=_summary_evidence_items(evidence, target),
            template_summary_text=RECENT_EVIDENCE_ONLY_CONTEXT,
        )
        return {
            "sample_id": sample_id,
            "user_id": str(record["user_id"]),
            "target_t": int(record["target_t"]),
            "target_pid": target_pid,
            "y_true": int(manifest[sample_id]["y_true"]),
            "summary_text": summary_text,
            "summary_source": "llm_summary_over_recent_k_evidence",
            "evidence_question_ids": [str(item.get("problem_id") or "") for item in evidence],
            "k": int(args.k),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        }

    failures: List[Dict[str, Any]] = []
    failure_path = output.with_suffix(".failed.jsonl")
    print(f"[recent_summary] selected={len(records)} done={len(done)} pending={len(pending)}", flush=True)
    with output.open("a", encoding="utf-8") as summary_handle, failure_path.open("a", encoding="utf-8") as failure_handle:
        summary_writer = BufferedJsonlWriter(summary_handle, flush_every=int(args.flush_every))
        failure_writer = BufferedJsonlWriter(failure_handle, flush_every=int(args.flush_every))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for index, (future, record) in enumerate(
                _iter_bounded_futures(executor, pending, run_one, max_in_flight=max_in_flight),
                start=1,
            ):
                try:
                    summary_writer.write(future.result())
                except Exception as exc:
                    failure = {
                        "sample_id": _context_key(str(record["user_id"]), int(record["target_t"]), str(record["target_pid"])),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failures.append(failure)
                    failure_writer.write(failure)
                if index % 100 == 0 or index == len(pending):
                    print(f"[recent_summary] completed={index}/{len(pending)} failures={len(failures)}", flush=True)
        summary_writer.flush()
        failure_writer.flush()
    if failures:
        raise RuntimeError(f"Recent-summary generation completed with {len(failures)} failures; inspect {failure_path}")
    print("[OK] recent-evidence LLM summaries finished")
    print(f"[SUMMARIES] {output}")


if __name__ == "__main__":
    main()
