from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


ContextKey = Tuple[str, int, str]


class ContextEmbeddingMap:
    def __init__(self, embeddings_path: str | Path) -> None:
        self.embeddings_path = Path(embeddings_path).resolve()
        with self.embeddings_path.open("rb") as f:
            payload = pickle.load(f)

        index = list(payload["index"])
        self.main_embeddings = np.asarray(payload["main_embeddings"], dtype=np.float32)
        self.template_embeddings = np.asarray(payload["template_embeddings"], dtype=np.float32)
        self.llm_embeddings = None
        if "llm_embeddings" in payload:
            self.llm_embeddings = np.asarray(payload["llm_embeddings"], dtype=np.float32)
        if len(index) != self.main_embeddings.shape[0] or len(index) != self.template_embeddings.shape[0]:
            raise ValueError("context_embeddings.pkl index and embedding arrays have inconsistent lengths")
        if self.llm_embeddings is not None and len(index) != self.llm_embeddings.shape[0]:
            raise ValueError("context_embeddings.pkl llm embedding array has inconsistent length")

        self.key_to_row: Dict[ContextKey, int] = {}
        for row_idx, item in enumerate(index):
            key = (str(item["user_id"]), int(item["target_t"]), str(item["target_pid"]))
            self.key_to_row[key] = row_idx

        self.context_dim = int(self.main_embeddings.shape[1])
        self.has_llm = self.llm_embeddings is not None

    def get_main(self, key: ContextKey) -> np.ndarray | None:
        row_idx = self.key_to_row.get(key)
        if row_idx is None:
            return None
        return self.main_embeddings[row_idx]

    def get_template(self, key: ContextKey) -> np.ndarray | None:
        row_idx = self.key_to_row.get(key)
        if row_idx is None:
            return None
        return self.template_embeddings[row_idx]

    def get_llm(self, key: ContextKey) -> np.ndarray | None:
        if self.llm_embeddings is None:
            return None
        row_idx = self.key_to_row.get(key)
        if row_idx is None:
            return None
        return self.llm_embeddings[row_idx]
