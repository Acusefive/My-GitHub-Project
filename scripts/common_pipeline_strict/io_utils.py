from __future__ import annotations

import ast
import datetime as dt
import json
import os
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


@dataclass
class ProblemRecord:
    problem_id: str
    text: str
    title: str
    chapter: str
    location: str
    cognitive_dimension: int
    concepts: List[str]


@dataclass
class StudentSequence:
    user_id: str
    seq: List[Dict[str, Any]]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json_any(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]


def write_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_detail_field(detail: Any) -> Dict[str, Any]:
    if isinstance(detail, dict):
        return detail
    if detail is None:
        return {}
    if not isinstance(detail, str):
        return {}
    detail = detail.strip()
    if not detail:
        return {}
    parsers = (json.loads, ast.literal_eval)
    for parser in parsers:
        try:
            value = parser(detail)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return {}


def extract_problem_text(detail_dict: Dict[str, Any]) -> Tuple[str, str]:
    title = str(detail_dict.get("title") or "").strip()
    text = str(
        detail_dict.get("content")
        or detail_dict.get("title")
        or detail_dict.get("body")
        or ""
    ).strip()
    return title, " ".join(text.split())


def extract_main_chapter(location: str) -> str:
    location = str(location or "").strip()
    if not location:
        return ""
    if "." in location:
        return location.split(".", 1)[0].strip() or location
    return location


def safe_level(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return 0
    return int(float(text))


def parse_submit_time(value: Any) -> Tuple[int, str]:
    text = str(value or "").strip()
    if not text:
        return (0, "")
    formats = ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S")
    for fmt in formats:
        try:
            parsed = dt.datetime.strptime(text, fmt)
            return (int(parsed.timestamp()), text)
        except ValueError:
            continue
    return (0, text)


def load_problem_records(path: Path, *, max_problems: Optional[int] = None) -> List[ProblemRecord]:
    raw = read_json_any(path)
    if isinstance(raw, dict):
        raw = [raw]

    records: List[ProblemRecord] = []
    for item in raw:
        if max_problems is not None and len(records) >= max_problems:
            break
        if not isinstance(item, dict):
            continue
        pid = str(item.get("problem_id") or item.get("id") or "").strip()
        if not pid:
            continue
        detail_dict = parse_detail_field(item.get("detail"))
        title, text = extract_problem_text(detail_dict)
        location = str(detail_dict.get("location") or "").strip()
        chapter = extract_main_chapter(location)
        concepts = [str(c) for c in (item.get("concepts") or []) if c is not None and str(c)]
        records.append(
            ProblemRecord(
                problem_id=pid,
                text=text,
                title=title,
                chapter=chapter,
                location=location,
                cognitive_dimension=safe_level(item.get("cognitive_dimension")),
                concepts=concepts,
            )
        )
    return records


def load_student_sequences(
    path: Path,
    *,
    max_students: Optional[int] = None,
    max_targets_per_student: Optional[int] = None,
) -> List[StudentSequence]:
    raw = read_json_any(path)
    if isinstance(raw, dict):
        for key in ("data", "students", "records", "logs"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise ValueError(f"Unsupported student record structure: {type(raw)}")

    sequences: List[StudentSequence] = []
    for item in raw:
        if max_students is not None and len(sequences) >= max_students:
            break
        if not isinstance(item, dict):
            continue
        seq = item.get("seq") or []
        if not isinstance(seq, list):
            continue
        user_id = str(item.get("user_id") or "").strip()
        if not user_id:
            for log in seq:
                if isinstance(log, dict) and log.get("user_id"):
                    user_id = str(log["user_id"]).strip()
                    break
        if not user_id:
            continue

        sorted_seq = sorted(
            [log for log in seq if isinstance(log, dict)],
            key=lambda log: (
                parse_submit_time(log.get("submit_time"))[0],
                parse_submit_time(log.get("submit_time"))[1],
                str(log.get("log_id") or ""),
            ),
        )
        if max_targets_per_student is not None and len(sorted_seq) > max_targets_per_student + 1:
            sorted_seq = sorted_seq[: max_targets_per_student + 1]
        sequences.append(StudentSequence(user_id=user_id, seq=sorted_seq))
    return sequences


def user_hash_bucket(user_id: str, mod: int) -> int:
    import hashlib

    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % mod


def dataclass_list_to_jsonl(records: Sequence[Any]) -> Iterator[Dict[str, Any]]:
    for record in records:
        yield asdict(record)


def format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def atomic_save_text(text: str, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def pick_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def resolve_local_sentence_transformer_path(model_name: str) -> Optional[Path]:
    override = os.environ.get("STRICT_BGE_LOCAL_PATH", "").strip()
    if override:
        path = Path(override)
        if path.exists():
            return path

    model_dir_name = "models--" + model_name.replace("/", "--")
    search_roots = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path(os.environ.get("HF_HOME", "")).expanduser() / "hub" if os.environ.get("HF_HOME") else None,
    ]
    for root in search_roots:
        if root is None:
            continue
        candidate_root = root / model_dir_name / "snapshots"
        if not candidate_root.exists():
            continue
        snapshots = sorted([path for path in candidate_root.iterdir() if path.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        for snapshot in snapshots:
            if (snapshot / "modules.json").exists():
                return snapshot
    return None
