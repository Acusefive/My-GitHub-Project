from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


QUESTION_IMAGE_RE = re.compile(r"\bquestion_\d+-image_\d+\b")
ANALYSIS_IMAGE_RE = re.compile(r"\banalysis_\d+-image_\d+\b")
ANY_IMAGE_RE = re.compile(r"\b(?:question|analysis)_\d+-image_\d+\b")


def default_dataset_root() -> Path:
    workspace_root = Path(__file__).resolve().parents[1]
    workspace_dataset = workspace_root / "datalocal" / "XES3G5M"
    if workspace_dataset.exists():
        return workspace_dataset
    windows_dataset = Path(r"D:\Dataset\XES3G5M")
    if windows_dataset.exists():
        return windows_dataset
    return workspace_dataset


def split_csv_list(value: str | None) -> List[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",")]


def compact_spaces(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def route_leaf(route: str) -> str:
    parts = [part.strip() for part in str(route or "").split("----") if part.strip()]
    return parts[-1] if parts else ""


def has_question_image(record: Dict[str, Any]) -> bool:
    return bool(QUESTION_IMAGE_RE.search(image_scan_text(record)))


def has_any_image(record: Dict[str, Any]) -> bool:
    return bool(ANY_IMAGE_RE.search(image_scan_text(record)))


def image_scan_text(record: Dict[str, Any]) -> str:
    payload = {
        "content": record.get("content") or "",
        "analysis": record.get("analysis") or "",
        "options": record.get("options") or {},
    }
    return json.dumps(payload, ensure_ascii=False)


def include_record(record: Dict[str, Any], image_filter: str) -> bool:
    if image_filter == "all":
        return True
    if image_filter == "no_question_image":
        return not has_question_image(record)
    if image_filter == "strict_text_only":
        return not has_any_image(record)
    raise ValueError(f"Unsupported image_filter: {image_filter}")


def load_questions(path: Path, image_filter: str) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"questions.json must be a JSON object, got {type(raw)}")
    return {
        str(qid): record
        for qid, record in raw.items()
        if isinstance(record, dict) and include_record(record, image_filter)
    }


def load_cognitive_dimensions(path: Optional[Path]) -> Dict[str, int]:
    if path is None or not path.exists():
        return {}
    dims: Dict[str, int] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(item.get("question_id") or item.get("problem_id") or "").strip()
            if qid.startswith("Q_"):
                qid = qid[2:]
            try:
                dim = int(float(item.get("cognitive_dimension")))
            except Exception:
                continue
            if qid and 1 <= dim <= 4:
                dims[qid] = dim
    return dims


def sort_qids(qids: Iterable[str]) -> List[str]:
    return sorted(qids, key=lambda item: (int(item), item) if item.isdigit() else (10**12, item))


def question_concepts(record: Dict[str, Any]) -> List[str]:
    concepts: List[str] = []
    seen: set[str] = set()
    for route in record.get("kc_routes") or []:
        concept = route_leaf(str(route))
        if concept and concept not in seen:
            concepts.append(concept)
            seen.add(concept)
    return concepts


def format_options(options: Any) -> str:
    if not isinstance(options, dict) or not options:
        return ""
    parts = []
    for key in sorted(options):
        value = compact_spaces(options[key])
        if value:
            parts.append(f"{key}. {value}")
    return " 选项：" + "；".join(parts) if parts else ""


def build_problem_text(record: Dict[str, Any]) -> str:
    content = compact_spaces(record.get("content"))
    options = format_options(record.get("options"))
    return compact_spaces(content + options)


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def build_problem_records(
    questions: Dict[str, Dict[str, Any]],
    cognitive_dimensions: Dict[str, int],
    *,
    default_cognitive_dimension: int,
    require_cognitive_dimension: bool,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    missing_dims: List[str] = []
    for qid in sort_qids(questions):
        record = questions[qid]
        concepts = question_concepts(record)
        dim = cognitive_dimensions.get(qid)
        if dim is None:
            missing_dims.append(qid)
            dim = int(default_cognitive_dimension)
        first_route = str((record.get("kc_routes") or [""])[0])
        detail = {
            "problem_id": qid,
            "title": f"XES3G5M Q{qid}",
            "content": build_problem_text(record),
            "type": record.get("type") or "",
            "location": first_route,
            "kc_routes": record.get("kc_routes") or [],
        }
        records.append(
            {
                "problem_id": f"Q_{qid}",
                "exercise_id": f"XES3G5M_{qid}",
                "course_id": "XES3G5M",
                "detail": json.dumps(detail, ensure_ascii=False, separators=(",", ":")),
                "knowledge_type": 0,
                "cognitive_dimension": int(dim),
                "concepts": concepts,
            }
        )
    if require_cognitive_dimension and missing_dims:
        preview = ", ".join(missing_dims[:10])
        raise ValueError(f"Missing cognitive_dimension for {len(missing_dims)} questions, first ids: {preview}")
    return records


def ms_to_submit_time(value: str, fallback_index: int) -> str:
    try:
        millis = int(float(value))
    except Exception:
        millis = 0
    if millis <= 0:
        millis = int(fallback_index)
    seconds = millis / 1000.0 if millis > 10_000_000_000 else float(millis)
    try:
        return datetime.fromtimestamp(seconds).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "1970-01-01 00:00:00"


def iter_student_sequences(
    source_csvs: List[Path],
    allowed_qids: set[str],
    *,
    min_seq_len: int,
    max_students: int,
) -> Iterable[Dict[str, Any]]:
    emitted = 0
    for path in source_csvs:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            has_selectmasks = bool(reader.fieldnames and "selectmasks" in reader.fieldnames)
            for row in reader:
                uid = str(row.get("uid") or "").strip()
                if not uid:
                    continue
                questions = split_csv_list(row.get("questions"))
                responses = split_csv_list(row.get("responses"))
                timestamps = split_csv_list(row.get("timestamps"))
                masks = split_csv_list(row.get("selectmasks")) if has_selectmasks else []
                seq: List[Dict[str, Any]] = []
                for pos, qid in enumerate(questions):
                    if qid == "-1" or qid not in allowed_qids:
                        continue
                    if masks and (pos >= len(masks) or masks[pos] != "1"):
                        continue
                    if pos >= len(responses) or responses[pos] not in {"0", "1"}:
                        continue
                    timestamp_raw = timestamps[pos] if pos < len(timestamps) else ""
                    seq.append(
                        {
                            "log_id": f"XES3G5M_{uid}_{len(seq):06d}",
                            "problem_id": f"Q_{qid}",
                            "user_id": f"U_{uid}",
                            "is_correct": int(responses[pos]),
                            "attempts": 1,
                            "score": float(responses[pos]),
                            "submit_time": ms_to_submit_time(timestamp_raw, pos),
                        }
                    )
                if len(seq) < int(min_seq_len):
                    continue
                emitted += 1
                yield {"seq": seq}
                if max_students > 0 and emitted >= max_students:
                    return


def summarize_outputs(problem_records: List[Dict[str, Any]], student_jsonl: Path) -> Dict[str, Any]:
    students = 0
    interactions = 0
    used_questions: set[str] = set()
    positive = 0
    with student_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            seq = item.get("seq") or []
            if not seq:
                continue
            students += 1
            interactions += len(seq)
            for log in seq:
                used_questions.add(str(log.get("problem_id") or ""))
                positive += int(log.get("is_correct") or 0)
    dim_counter = Counter(int(record["cognitive_dimension"]) for record in problem_records)
    concept_count = len({concept for record in problem_records for concept in record.get("concepts") or []})
    return {
        "students": students,
        "questions_in_catalog": len(problem_records),
        "questions_observed": len(used_questions),
        "interactions": interactions,
        "positive_rate": float(positive / interactions) if interactions else 0.0,
        "unique_concepts": concept_count,
        "cognitive_dimension_distribution": dict(sorted(dim_counter.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert XES3G5M question-level data to this repo's strict pipeline format.")
    parser.add_argument("--dataset_root", type=Path, default=default_dataset_root())
    parser.add_argument("--out_dir", type=Path, default=Path("datalocal/xes3g5m_text_only"))
    parser.add_argument("--image_filter", choices=["all", "no_question_image", "strict_text_only"], default="strict_text_only")
    parser.add_argument("--cognitive_dimensions_jsonl", type=Path, default=Path("out/xes3g5m_cognitive/cognitive_dimensions.jsonl"))
    parser.add_argument("--default_cognitive_dimension", type=int, default=0)
    parser.add_argument("--require_cognitive_dimension", action="store_true")
    parser.add_argument(
        "--drop_missing_cognitive_dimension",
        action="store_true",
        help="Drop questions and interactions whose LLM-generated cognitive_dimension is missing.",
    )
    parser.add_argument("--min_seq_len", type=int, default=2)
    parser.add_argument("--max_students", type=int, default=0)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    out_dir = args.out_dir.resolve()
    questions_path = root / "metadata" / "questions.json"
    source_csvs = [
        root / "question_level" / "train_valid_sequences_quelevel.csv",
        root / "question_level" / "test_quelevel.csv",
    ]
    for path in [questions_path, *source_csvs]:
        if not path.exists():
            raise FileNotFoundError(path)

    questions = load_questions(questions_path, args.image_filter)
    cognitive_dimensions = load_cognitive_dimensions(args.cognitive_dimensions_jsonl.resolve())
    missing_cognitive_qids = sort_qids(set(questions) - set(cognitive_dimensions))
    if args.drop_missing_cognitive_dimension:
        questions = {qid: record for qid, record in questions.items() if qid in cognitive_dimensions}
    problem_records = build_problem_records(
        questions,
        cognitive_dimensions,
        default_cognitive_dimension=int(args.default_cognitive_dimension),
        require_cognitive_dimension=bool(args.require_cognitive_dimension),
    )
    allowed_qids = set(questions)

    problem_json = out_dir / "problem.json"
    student_json = out_dir / "student-problem-fine.json"
    report_json = out_dir / "prepare_report.json"
    problem_count = write_jsonl(problem_records, problem_json)
    student_count = write_jsonl(
        iter_student_sequences(
            source_csvs,
            allowed_qids,
            min_seq_len=int(args.min_seq_len),
            max_students=int(args.max_students),
        ),
        student_json,
    )
    report = {
        "dataset_root": str(root),
        "image_filter": args.image_filter,
        "source_csvs": [str(path) for path in source_csvs],
        "problem_json": str(problem_json),
        "student_json": str(student_json),
        "problem_records_written": problem_count,
        "student_sequences_written": student_count,
        "cognitive_dimensions_loaded": len(cognitive_dimensions),
        "questions_dropped_missing_cognitive_dimension": len(missing_cognitive_qids)
        if args.drop_missing_cognitive_dimension
        else 0,
        "dropped_missing_cognitive_question_ids": missing_cognitive_qids
        if args.drop_missing_cognitive_dimension
        else [],
        "summary": summarize_outputs(problem_records, student_json),
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] XES3G5M strict-format data prepared")
    print("[PROBLEM_JSON]", problem_json)
    print("[STUDENT_JSON]", student_json)
    print("[REPORT]", report_json)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
