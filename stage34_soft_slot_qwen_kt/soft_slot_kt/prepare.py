from __future__ import annotations

import json
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .data import load_problem_catalog
from .source_io import load_student_sequences
from .utils import ensure_dir, write_json


CONTEXT_ARRAY_FIELDS = (
    "main_embeddings",
    "template_embeddings",
    "llm_embeddings",
    "llm_struct_embeddings",
    "llm_struct_features",
)
TARGET_SOURCE_FILES = {
    "hqtext": "hqtext_vectors.pkl",
    "hqid": "hqid_vectors.pkl",
    "semantic": "semantic_vectors.pkl",
    "collaborative": "item_collaborative_embeddings.pkl",
}
SPLIT_CODE_MAP = {"train": 0, "valid": 1, "test": 2, "excluded": 3}


def file_signature(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def split_concepts(
    all_concepts: Sequence[str],
    *,
    seed: int,
    test_concept_ratio: float,
    valid_concept_ratio: float,
) -> Dict[str, set[str]]:
    concepts = np.asarray(sorted(set(str(x) for x in all_concepts)), dtype=object)
    if len(concepts):
        rng = np.random.default_rng(int(seed))
        concepts = concepts[rng.permutation(len(concepts))]
    test_count = max(1, int(len(concepts) * float(test_concept_ratio))) if len(concepts) and test_concept_ratio > 0 else 0
    valid_count = max(1, int(len(concepts) * float(valid_concept_ratio))) if len(concepts) and valid_concept_ratio > 0 else 0
    valid_count = min(valid_count, max(0, len(concepts) - test_count))
    test = set(concepts[:test_count].tolist())
    valid = set(concepts[test_count : test_count + valid_count].tolist())
    train = set(concepts.tolist()) - test - valid
    return {"train": train, "valid": valid, "test": test}


def target_split(concepts: Iterable[str], concept_splits: Dict[str, set[str]]) -> str:
    values = set(str(x) for x in concepts)
    if values & concept_splits["test"]:
        return "test"
    if values & concept_splits["valid"]:
        return "valid"
    if values and values <= concept_splits["train"]:
        return "train"
    return "excluded"


def build_label_map(student_json: Path, allowed_pids: set[str]) -> Dict[Tuple[str, int, str], int]:
    labels: Dict[Tuple[str, int, str], int] = {}
    for sequence in tqdm(load_student_sequences(student_json), desc="soft-slot labels"):
        filtered = [log for log in sequence.seq if str(log.get("problem_id") or "") in allowed_pids]
        for target_t, log in enumerate(filtered):
            pid = str(log.get("problem_id") or "")
            labels[(sequence.user_id, target_t, pid)] = int(log.get("is_correct") or 0)
    return labels


def copy_array_to_npy(array: np.ndarray, path: Path, *, chunk_rows: int) -> Dict[str, Any]:
    if path.exists():
        existing = np.load(path, mmap_mode="r")
        if tuple(existing.shape) == tuple(array.shape) and existing.dtype == array.dtype:
            return {"path": path.name, "shape": list(existing.shape), "dtype": str(existing.dtype), "reused": True}
        raise ValueError(f"Existing array has incompatible shape or dtype: {path}")
    partial_path = path.with_suffix(path.suffix + ".partial")
    if partial_path.exists():
        partial_path.unlink()
    target = np.lib.format.open_memmap(partial_path, mode="w+", dtype=array.dtype, shape=array.shape)
    for start in tqdm(range(0, len(array), chunk_rows), desc=f"copy {path.name}"):
        target[start : start + chunk_rows] = array[start : start + chunk_rows]
    target.flush()
    del target
    os.replace(partial_path, path)
    return {"path": path.name, "shape": list(array.shape), "dtype": str(array.dtype), "reused": False}


def map_to_aligned_npy(
    values: Dict[str, Any],
    problem_ids: Sequence[str],
    path: Path,
    *,
    chunk_rows: int,
) -> Dict[str, Any]:
    first = next((np.asarray(value) for value in values.values() if np.asarray(value).ndim == 1), None)
    if first is None:
        raise ValueError(f"Could not infer target feature dimension for {path.name}")
    dim = int(first.shape[0])
    if path.exists():
        existing = np.load(path, mmap_mode="r")
        if tuple(existing.shape) == (len(problem_ids), dim):
            return {"path": path.name, "shape": list(existing.shape), "dtype": str(existing.dtype), "reused": True}
        raise ValueError(f"Existing array has incompatible shape: {path}")
    partial_path = path.with_suffix(path.suffix + ".partial")
    if partial_path.exists():
        partial_path.unlink()
    target = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32, shape=(len(problem_ids), dim))
    for start in tqdm(range(0, len(problem_ids), chunk_rows), desc=f"align {path.name}"):
        rows = []
        for pid in problem_ids[start : start + chunk_rows]:
            value = values.get(pid)
            rows.append(np.asarray(value, dtype=np.float32) if value is not None else np.zeros((dim,), dtype=np.float32))
        target[start : start + len(rows)] = np.stack(rows, axis=0)
    target.flush()
    del target
    os.replace(partial_path, path)
    return {"path": path.name, "shape": [len(problem_ids), dim], "dtype": "float32", "reused": False}


def _pick_context_text(record: Dict[str, Any], field: str) -> str:
    if field == "auto":
        return str(record.get("llm_context_text") or record.get("template_context_text") or record.get("main_context_text") or "")
    return str(record.get(field) or "")


def prepare_existing_stage34_features(
    *,
    context_embeddings_path: Path,
    contexts_path: Path,
    priors_dir: Path,
    student_json: Path,
    output_dir: Path,
    context_fields: Sequence[str],
    target_fields: Sequence[str],
    prompt_context_field: str,
    max_context_chars: int,
    seed: int,
    test_concept_ratio: float,
    valid_concept_ratio: float,
    chunk_rows: int,
) -> Dict[str, Any]:
    ensure_dir(output_dir)
    catalog_path = priors_dir / "problem_catalog.jsonl"
    source_signatures = {
        "context_embeddings": file_signature(context_embeddings_path),
        "contexts": file_signature(contexts_path),
        "student_json": file_signature(student_json),
        "problem_catalog": file_signature(catalog_path),
    }
    existing_manifest_path = output_dir / "feature_manifest.json"
    if existing_manifest_path.exists():
        existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("source_signatures") != source_signatures:
            raise ValueError(
                "Existing feature store was built from different source files. "
                "Use a new --output_dir to avoid mixing incompatible artifacts."
            )
    catalog = load_problem_catalog(catalog_path)
    problem_ids = list(catalog)
    allowed_pids = set(problem_ids)
    concept_splits = split_concepts(
        [concept for item in catalog.values() for concept in item.get("concepts") or []],
        seed=seed,
        test_concept_ratio=test_concept_ratio,
        valid_concept_ratio=valid_concept_ratio,
    )
    labels = build_label_map(student_json, allowed_pids)

    print(f"[soft-slot prepare] loading existing embedding pickle: {context_embeddings_path}", flush=True)
    with context_embeddings_path.open("rb") as handle:
        payload = pickle.load(handle)
    index_records = payload.get("index")
    if not isinstance(index_records, list) or not index_records:
        raise ValueError("context_embeddings.pkl does not contain a non-empty index list")
    record_count = len(index_records)

    requested_context = list(context_fields)
    if "all" in requested_context:
        requested_context = [field for field in CONTEXT_ARRAY_FIELDS if field in payload] + ["stage34_numeric"]
    requested_target = list(target_fields)
    if "all" in requested_target:
        requested_target = list(TARGET_SOURCE_FILES)

    samples_path = output_dir / "samples.jsonl"
    offsets_path = output_dir / "sample_offsets.npy"
    split_codes_path = output_dir / "split_codes.npy"
    numeric_path = output_dir / "stage34_numeric.npy"
    offsets = np.lib.format.open_memmap(offsets_path, mode="w+", dtype=np.int64, shape=(record_count,))
    split_codes = np.lib.format.open_memmap(split_codes_path, mode="w+", dtype=np.int8, shape=(record_count,))
    numeric = (
        np.lib.format.open_memmap(numeric_path, mode="w+", dtype=np.float32, shape=(record_count, 3))
        if "stage34_numeric" in requested_context
        else None
    )
    split_counts: Counter = Counter()
    audit_counts: Counter = Counter()
    with (
        contexts_path.open("r", encoding="utf-8", errors="replace") as contexts_handle,
        samples_path.open("wb") as samples_handle,
    ):
        for row, line in enumerate(tqdm(contexts_handle, total=record_count, desc="soft-slot samples")):
            if row >= record_count:
                raise ValueError("contexts.jsonl has more records than embedding index")
            record = json.loads(line)
            index = index_records[row]
            identity = (str(record["user_id"]), int(record["target_t"]), str(record["target_pid"]))
            expected = (str(index["user_id"]), int(index["target_t"]), str(index["target_pid"]))
            if identity != expected:
                raise ValueError(f"Context/index mismatch at row {row}: {identity} != {expected}")
            label = labels.get(identity)
            if label is None:
                raise ValueError(f"Missing target label for context row {row}: {identity}")
            split = target_split(catalog[identity[2]].get("concepts") or [], concept_splits)
            split_counts[split] += 1
            for evidence in record.get("evidence_list") or []:
                if int(evidence.get("history_pos") or 0) >= identity[1]:
                    audit_counts["noncausal_evidence"] += 1
            context_text = " ".join(_pick_context_text(record, prompt_context_field).split())
            if len(context_text) > max_context_chars:
                context_text = context_text[: max(0, max_context_chars - 3)] + "..."
            sample = {
                "user_id": identity[0],
                "target_t": identity[1],
                "target_pid": identity[2],
                "label": int(label),
                "split": split,
                "context_text": context_text,
                "max_context_chars": int(max_context_chars),
            }
            offsets[row] = samples_handle.tell()
            samples_handle.write((json.dumps(sample, ensure_ascii=False) + "\n").encode("utf-8"))
            split_codes[row] = SPLIT_CODE_MAP[split]
            if numeric is not None:
                summary = record.get("summary_fields") or {}
                numeric[row] = np.asarray(
                    [
                        float(summary.get("sdyn") or 0.0),
                        float(record.get("stage1_candidate_count") or 0.0) / 20.0,
                        float(record.get("selected_count") or 0.0) / 10.0,
                    ],
                    dtype=np.float32,
                )
        if row + 1 != record_count:
            raise ValueError(f"contexts.jsonl has {row + 1} records but embedding index has {record_count}")
    offsets.flush()
    split_codes.flush()
    if numeric is not None:
        numeric.flush()

    context_manifest: Dict[str, Any] = {}
    for field in requested_context:
        if field == "stage34_numeric":
            context_manifest[field] = {"path": numeric_path.name, "shape": [record_count, 3], "dtype": "float32"}
            continue
        if field not in payload:
            raise ValueError(f"Requested context field is missing from pickle: {field}")
        context_manifest[field] = copy_array_to_npy(
            np.asarray(payload[field]),
            output_dir / f"{field}.npy",
            chunk_rows=chunk_rows,
        )

    target_manifest: Dict[str, Any] = {}
    for field in requested_target:
        source_name = TARGET_SOURCE_FILES.get(field)
        if source_name is None:
            raise ValueError(f"Unsupported target field: {field}")
        source_path = priors_dir / source_name
        with source_path.open("rb") as handle:
            values = pickle.load(handle)
        target_manifest[field] = map_to_aligned_npy(
            values,
            problem_ids,
            output_dir / f"target_{field}.npy",
            chunk_rows=chunk_rows,
        )

    audit = {
        "protocol": "existing_stage34_transductive",
        "strict_new_concept_leakage_free": False,
        "warning": (
            "Existing Stage32/Stage34 features may contain labels from held-out concepts. "
            "This feature store is suitable for transductive Soft-Slot experiments, not strict leakage-free claims."
        ),
        "noncausal_evidence_count": int(audit_counts["noncausal_evidence"]),
        "target_label_stored_only_in_sample_label_field": True,
        "record_count": record_count,
        "split_counts": dict(split_counts),
    }
    manifest = {
        "format_version": 1,
        "protocol": "existing_stage34_transductive",
        "record_count": record_count,
        "context_embeddings_path": str(context_embeddings_path),
        "contexts_path": str(contexts_path),
        "priors_dir": str(priors_dir),
        "student_json": str(student_json),
        "problem_catalog_path": str(catalog_path),
        "source_signatures": source_signatures,
        "samples_path": samples_path.name,
        "sample_offsets_path": offsets_path.name,
        "split_codes_path": split_codes_path.name,
        "split_code_map": SPLIT_CODE_MAP,
        "seed": int(seed),
        "test_concept_ratio": float(test_concept_ratio),
        "valid_concept_ratio": float(valid_concept_ratio),
        "prompt_context_field": prompt_context_field,
        "max_context_chars": int(max_context_chars),
        "problem_ids": problem_ids,
        "context_fields": context_manifest,
        "target_fields": target_manifest,
        "audit": audit,
    }
    write_json(audit, output_dir / "leakage_audit.json")
    write_json(manifest, output_dir / "feature_manifest.json")
    return manifest
