from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .prompts import EncodedPrompt, encode_prompt, left_pad_batch
from .utils import parse_csv, read_json


class IndexedJsonl:
    def __init__(self, path: Path, offsets_path: Path) -> None:
        self.path = path
        self.offsets = np.load(offsets_path, mmap_mode="r")
        self._handle = None

    def __len__(self) -> int:
        return int(len(self.offsets))

    def _get_handle(self):
        if self._handle is None:
            self._handle = self.path.open("rb")
        return self._handle

    def __getitem__(self, index: int) -> Dict[str, Any]:
        handle = self._get_handle()
        handle.seek(int(self.offsets[index]))
        line = handle.readline()
        return json.loads(line.decode("utf-8"))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()


class FeatureStore:
    def __init__(
        self,
        feature_dir: Path,
        *,
        context_fields: Sequence[str],
        target_fields: Sequence[str],
        drop_sdyn: bool = False,
        drop_collab: bool = False,
    ) -> None:
        self.feature_dir = feature_dir
        self.manifest = read_json(feature_dir / "feature_manifest.json")
        self.context_fields = list(context_fields)
        self.target_fields = [field for field in target_fields if not (drop_collab and field == "collaborative")]
        self.drop_sdyn = bool(drop_sdyn)
        available_context = self.manifest["context_fields"]
        available_target = self.manifest["target_fields"]
        missing_context = [field for field in self.context_fields if field not in available_context]
        missing_target = [field for field in self.target_fields if field not in available_target]
        if missing_context or missing_target:
            raise ValueError(f"Missing feature fields: context={missing_context}, target={missing_target}")
        self.context_arrays = {
            field: np.load(feature_dir / available_context[field]["path"], mmap_mode="r")
            for field in self.context_fields
        }
        self.target_arrays = {
            field: np.load(feature_dir / available_target[field]["path"], mmap_mode="r")
            for field in self.target_fields
        }
        self.problem_ids = list(self.manifest["problem_ids"])
        self.pid_to_row = {pid: row for row, pid in enumerate(self.problem_ids)}
        self.context_dim = sum(int(available_context[field]["shape"][1]) for field in self.context_fields)
        self.target_dim = sum(int(available_target[field]["shape"][1]) for field in self.target_fields)

    def context(self, row: int) -> np.ndarray:
        arrays: List[np.ndarray] = []
        for field in self.context_fields:
            value = np.asarray(self.context_arrays[field][row], dtype=np.float32)
            if self.drop_sdyn and field == "stage34_numeric" and value.size:
                value = value.copy()
                value[0] = 0.0
            arrays.append(value)
        return np.concatenate(arrays, axis=0) if arrays else np.zeros((0,), dtype=np.float32)

    def target(self, pid: str) -> np.ndarray:
        row = self.pid_to_row[pid]
        arrays = [np.asarray(self.target_arrays[field][row], dtype=np.float32) for field in self.target_fields]
        return np.concatenate(arrays, axis=0) if arrays else np.zeros((0,), dtype=np.float32)


class SoftSlotDataset(Dataset):
    def __init__(
        self,
        feature_dir: Path,
        *,
        split: str,
        context_fields: Sequence[str],
        target_fields: Sequence[str],
        limit: int = 0,
        seed: int = 42,
        drop_sdyn: bool = False,
        drop_collab: bool = False,
    ) -> None:
        self.feature_dir = feature_dir
        self.feature_store = FeatureStore(
            feature_dir,
            context_fields=context_fields,
            target_fields=target_fields,
            drop_sdyn=drop_sdyn,
            drop_collab=drop_collab,
        )
        manifest = self.feature_store.manifest
        self.samples = IndexedJsonl(feature_dir / manifest["samples_path"], feature_dir / manifest["sample_offsets_path"])
        split_codes = np.load(feature_dir / manifest["split_codes_path"], mmap_mode="r")
        split_code = int(manifest["split_code_map"][split])
        indices = np.flatnonzero(split_codes == split_code).astype(np.int64)
        if int(limit) > 0 and len(indices) > int(limit):
            rng = np.random.default_rng(int(seed))
            indices = np.sort(rng.choice(indices, size=int(limit), replace=False))
        self.indices = indices

    @property
    def context_dim(self) -> int:
        return self.feature_store.context_dim

    @property
    def target_dim(self) -> int:
        return self.feature_store.target_dim

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = int(self.indices[index])
        sample = self.samples[row]
        return {
            **sample,
            "row": row,
            "context_features": self.feature_store.context(row),
            "target_features": self.feature_store.target(str(sample["target_pid"])),
        }


class SoftSlotCollator:
    def __init__(
        self,
        tokenizer: Any,
        problem_catalog: Dict[str, Dict[str, Any]],
        *,
        context_soft_tokens: int,
        target_soft_tokens: int,
        include_context_text: bool,
    ) -> None:
        self.tokenizer = tokenizer
        self.problem_catalog = problem_catalog
        self.context_soft_tokens = int(context_soft_tokens)
        self.target_soft_tokens = int(target_soft_tokens)
        self.include_context_text = bool(include_context_text)
        self.pad_token_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        self.placeholder_token_id = self.pad_token_id

    def __call__(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        encoded: List[EncodedPrompt] = []
        for sample in samples:
            pid = str(sample["target_pid"])
            encoded.append(
                encode_prompt(
                    self.tokenizer,
                    sample,
                    self.problem_catalog[pid],
                    context_soft_tokens=self.context_soft_tokens,
                    target_soft_tokens=self.target_soft_tokens,
                    include_context_text=self.include_context_text,
                    placeholder_token_id=self.placeholder_token_id,
                )
            )
        batch = left_pad_batch(encoded, pad_token_id=self.pad_token_id)
        batch.update(
            {
                "context_features": torch.tensor(
                    np.stack([sample["context_features"] for sample in samples], axis=0),
                    dtype=torch.float32,
                ),
                "target_features": torch.tensor(
                    np.stack([sample["target_features"] for sample in samples], axis=0),
                    dtype=torch.float32,
                ),
                "labels": torch.tensor([int(sample["label"]) for sample in samples], dtype=torch.long),
                "metadata": [
                    {
                        "row": int(sample["row"]),
                        "user_id": str(sample["user_id"]),
                        "target_t": int(sample["target_t"]),
                        "target_pid": str(sample["target_pid"]),
                        "label": int(sample["label"]),
                        "split": str(sample["split"]),
                    }
                    for sample in samples
                ],
            }
        )
        return batch


def load_problem_catalog(path: Path) -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                catalog[str(row["problem_id"])] = row
    return catalog


def dataset_from_args(args: Any, split: str) -> SoftSlotDataset:
    return SoftSlotDataset(
        Path(args.feature_dir),
        split=split,
        context_fields=parse_csv(args.context_fields),
        target_fields=parse_csv(args.target_fields),
        limit=int(getattr(args, f"max_{split}_samples", 0)),
        seed=int(args.seed),
        drop_sdyn=bool(args.drop_sdyn),
        drop_collab=bool(args.drop_collab),
    )
