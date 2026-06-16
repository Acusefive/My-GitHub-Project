from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class StudentSequence:
    user_id: str
    seq: List[Dict[str, Any]]


def read_json_any(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError:
            handle.seek(0)
            return [json.loads(line) for line in handle if line.strip()]


def parse_submit_time(value: Any) -> Tuple[int, str]:
    text = str(value or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return int(dt.datetime.strptime(text, fmt).timestamp()), text
        except ValueError:
            continue
    return 0, text


def load_student_sequences(path: Path) -> List[StudentSequence]:
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
        if not isinstance(item, dict) or not isinstance(item.get("seq"), list):
            continue
        seq = [log for log in item["seq"] if isinstance(log, dict)]
        user_id = str(item.get("user_id") or "").strip()
        if not user_id:
            user_id = next((str(log["user_id"]).strip() for log in seq if log.get("user_id")), "")
        if not user_id:
            continue
        seq.sort(
            key=lambda log: (
                parse_submit_time(log.get("submit_time"))[0],
                parse_submit_time(log.get("submit_time"))[1],
                str(log.get("log_id") or ""),
            )
        )
        sequences.append(StudentSequence(user_id=user_id, seq=seq))
    return sequences
