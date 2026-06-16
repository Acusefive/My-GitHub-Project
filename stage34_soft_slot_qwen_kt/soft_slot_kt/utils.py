from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from sklearn import metrics


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv(value: str) -> List[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def resolve_torch_dtype(value: str) -> torch.dtype:
    normalized = str(value).lower()
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float16":
        return torch.float16
    if normalized == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def compute_metrics(labels: Sequence[int], probabilities: Sequence[float]) -> Dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(probabilities, dtype=np.float64)
    if y_true.size == 0:
        return {"count": 0}
    y_score = np.clip(y_score, 1e-7, 1.0 - 1e-7)
    y_pred = (y_score >= 0.5).astype(np.int64)
    result: Dict[str, Any] = {
        "count": int(y_true.size),
        "positive_rate": float(y_true.mean()),
        "auc": float(metrics.roc_auc_score(y_true, y_score)) if len(set(y_true.tolist())) >= 2 else float("nan"),
        "acc": float(metrics.accuracy_score(y_true, y_pred)),
        "f1": float(metrics.f1_score(y_true, y_pred, zero_division=0)),
        "bce": float(metrics.log_loss(y_true, y_score, labels=[0, 1])),
        "rmse": float(math.sqrt(np.mean((y_score - y_true) ** 2))),
    }
    return result


def atomic_torch_save(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp_path)
    os.replace(tmp_path, path)
