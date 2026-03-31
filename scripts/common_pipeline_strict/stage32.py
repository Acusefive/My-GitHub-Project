from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import jieba
import numpy as np
import torch
import torch.nn.functional as F
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from .constants import (
    A_SCALE,
    BGE_MODEL_NAME,
    COLLAB_WINDOW,
    CTFIDF_MAX_FEATURES,
    DS,
    EPS,
    HISTORY_WINDOW,
    KGLOBAL,
    KLOCAL,
    KMEANS_N_INIT,
    KMEANS_RANDOM_STATE,
    LAMBDA_L,
    LAMBDA_T,
    LOCAL_COOCCUR_THRESHOLD,
    NEGATIVE_SAMPLES,
    RASCH_A,
    RASCH_EPOCHS,
    RASCH_LAMBDA_MU,
    RASCH_LAMBDA_THETA,
    RASCH_LR,
    SMOKE_MAX_PROBLEMS,
    SMOKE_MAX_STUDENTS,
    SMOKE_MAX_TARGETS_PER_STUDENT,
    SMOKE_TRAIN_MAX_SAMPLES,
    STOP_WORDS,
    TAU,
    TRAIN_BATCH_SIZE,
    TRAIN_EARLY_STOP_PATIENCE,
    TRAIN_LR,
    TRAIN_MAX_EPOCHS,
    TRAIN_SEED,
    TRAIN_VAL_MOD,
)
from .io_utils import (
    ProblemRecord,
    StudentSequence,
    atomic_save_text,
    dataclass_list_to_jsonl,
    ensure_dir,
    load_problem_records,
    load_student_sequences,
    pick_device,
    resolve_local_sentence_transformer_path,
    user_hash_bucket,
    write_json,
    write_jsonl,
)
from .models import StrictPriorModel


@dataclass
class Stage32Result:
    priors_dir: str
    semantic_ids_path: str
    semantic_vectors_path: str
    hqtext_vectors_path: str
    hqid_vectors_path: str
    eqbase_vectors_path: str
    problem_mu_q_path: str
    concept_pc1_path: str
    item_collaborative_path: str
    item_collaborative_vectors_path: str
    graph_bundle_path: str
    problem_catalog_path: str
    model_state_path: str
    training_report_path: str
    implementation_defaults_path: str
    manifest_path: str
    problem_count: int
    student_count: int


@dataclass
class ProblemCatalogRecord:
    problem_id: str
    semantic_id: str
    text: str
    title: str
    chapter: str
    location: str
    cognitive_dimension: int
    concepts: List[str]


def _jieba_tokenizer(text: str) -> List[str]:
    return [token for token in jieba.lcut(text) if len(token) > 1 and token not in STOP_WORDS]


def normalize_vecs(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms <= 0, 1.0, norms)
    return (matrix / norms).astype(np.float32, copy=False)


def compute_ctfidf_labels(
    texts: Sequence[str],
    labels: np.ndarray,
    *,
    max_features: int,
) -> Dict[int, str]:
    vectorizer = CountVectorizer(
        tokenizer=_jieba_tokenizer,
        max_features=max_features,
        token_pattern=None,
    )
    dtm_all = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    k = int(labels.max()) + 1 if labels.size else 0
    label_names: Dict[int, str] = {}
    if k <= 0:
        return label_names

    token_sums = np.zeros((k, len(vocab)), dtype=np.float64)
    doc_counts = np.zeros((k,), dtype=np.float64)
    for cid in range(k):
        mask = labels == cid
        if not np.any(mask):
            label_names[cid] = "misc"
            continue
        doc_counts[cid] = float(mask.sum())
        token_sums[cid] = np.asarray(dtm_all[mask].sum(axis=0)).ravel().astype(np.float64)
    token_df = (token_sums > 0).sum(axis=0).astype(np.float64)
    idf = np.log(1.0 + float(k) / np.maximum(token_df, 1.0))

    for cid in range(k):
        if doc_counts[cid] <= 0:
            label_names[cid] = "misc"
            continue
        tf = token_sums[cid] / doc_counts[cid]
        ctfidf = tf * idf
        label_names[cid] = pick_top_token_from_ctfidf(ctfidf, vocab)
    return label_names


def pick_top_token_from_ctfidf(ctfidf: np.ndarray, vocab: np.ndarray) -> str:
    indices = np.argsort(ctfidf)[::-1]
    for idx in indices[:20]:
        token = str(vocab[int(idx)])
        if token and token not in STOP_WORDS:
            return token
    return "misc"


def build_semantic_ids(
    problem_records: Sequence[ProblemRecord],
    text_vectors: np.ndarray,
    *,
    semantic_ids_path: Path,
) -> Tuple[Dict[str, str], List[str]]:
    problem_ids = [problem.problem_id for problem in problem_records]
    texts = [problem.text or problem.title or problem.problem_id for problem in problem_records]
    k1 = min(KGLOBAL, len(problem_ids))
    km1 = KMeans(n_clusters=k1, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT)
    labels1 = km1.fit_predict(text_vectors)
    macro_labels = compute_ctfidf_labels(texts, labels1, max_features=CTFIDF_MAX_FEATURES)

    semantic_ids: Dict[str, str] = {}
    semantic_texts: List[str] = []
    for cid in tqdm(range(k1), desc="strict semantic ids"):
        idxs = np.where(labels1 == cid)[0]
        if len(idxs) == 0:
            continue
        k2 = min(KLOCAL, len(idxs))
        sub_vectors = text_vectors[idxs]
        if k2 <= 1:
            labels2 = np.zeros((len(idxs),), dtype=np.int64)
        else:
            km2 = KMeans(n_clusters=k2, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT)
            labels2 = km2.fit_predict(sub_vectors)
        sub_texts = [texts[int(idx)] for idx in idxs]
        micro_labels = compute_ctfidf_labels(sub_texts, labels2, max_features=CTFIDF_MAX_FEATURES)
        for local_pos, global_idx in enumerate(idxs):
            micro_label = micro_labels[int(labels2[local_pos])]
            semantic_id = f"{macro_labels[cid]}-{micro_label}"
            semantic_ids[problem_ids[int(global_idx)]] = semantic_id
            semantic_texts.append(semantic_id)

    write_json(semantic_ids, semantic_ids_path)
    return semantic_ids, [semantic_ids[pid] for pid in problem_ids]


def estimate_rasch_mu_q(
    student_sequences: Sequence[StudentSequence],
    pid_to_idx: Dict[str, int],
    *,
    seed: int,
) -> Dict[str, float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    user_to_idx: Dict[str, int] = {}
    records: List[Tuple[int, int, float]] = []
    for sequence in student_sequences:
        if sequence.user_id not in user_to_idx:
            user_to_idx[sequence.user_id] = len(user_to_idx)
        user_idx = user_to_idx[sequence.user_id]
        for log in sequence.seq:
            pid = str(log.get("problem_id") or "")
            if pid not in pid_to_idx:
                continue
            result = log.get("is_correct")
            if result is None:
                continue
            records.append((user_idx, pid_to_idx[pid], float(int(result))))

    if not records:
        return {}

    u_idx = torch.tensor([row[0] for row in records], dtype=torch.long)
    q_idx = torch.tensor([row[1] for row in records], dtype=torch.long)
    y = torch.tensor([row[2] for row in records], dtype=torch.float32)

    theta = torch.zeros((len(user_to_idx),), dtype=torch.float32, requires_grad=True)
    b_q = torch.zeros((len(pid_to_idx),), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([theta, b_q], lr=RASCH_LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(RASCH_EPOCHS):
        optimizer.zero_grad()
        logits = theta[u_idx] - b_q[q_idx]
        loss_nll = loss_fn(logits, y)
        loss_reg = RASCH_LAMBDA_THETA * torch.mean(theta.pow(2)) + RASCH_LAMBDA_MU * torch.mean(b_q.pow(2))
        loss = loss_nll + loss_reg
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            b_q -= torch.mean(b_q)

    idx_to_pid = {idx: pid for pid, idx in pid_to_idx.items()}
    out: Dict[str, float] = {}
    for idx, pid in idx_to_pid.items():
        out[pid] = float(np.tanh(RASCH_A * float(b_q[idx].detach().cpu().item())))
    return out


def build_concept_pc1_directions(
    problem_records: Sequence[ProblemRecord],
    eqbase_vectors: np.ndarray,
) -> Dict[str, np.ndarray]:
    concept_to_positions: Dict[str, List[int]] = defaultdict(list)
    concept_to_levels: Dict[str, List[int]] = defaultdict(list)
    for pos, problem in enumerate(problem_records):
        for concept in problem.concepts:
            concept_to_positions[concept].append(pos)
            concept_to_levels[concept].append(problem.cognitive_dimension)

    directions: Dict[str, np.ndarray] = {}
    for concept, positions in tqdm(concept_to_positions.items(), desc="strict concept pc1"):
        if len(positions) == 1:
            vec = eqbase_vectors[positions[0]]
            norm = float(np.linalg.norm(vec))
            directions[concept] = (vec / norm).astype(np.float32) if norm > 0 else np.zeros((DS,), dtype=np.float32)
            continue
        matrix = torch.tensor(eqbase_vectors[positions], dtype=torch.float32)
        centered = matrix - torch.mean(matrix, dim=0, keepdim=True)
        _, _, v_h = torch.linalg.svd(centered, full_matrices=False)
        pc1 = v_h[0].to(torch.float32)
        levels_t = torch.tensor(concept_to_levels[concept], dtype=torch.int64)
        proj = centered @ pc1
        unique_levels = sorted(set(levels_t.tolist()))
        if len(unique_levels) >= 2:
            low = unique_levels[0]
            high = unique_levels[-1]
            low_mask = levels_t == low
            high_mask = levels_t == high
            low_mean = float(torch.mean(proj[low_mask]).item()) if bool(torch.any(low_mask)) else 0.0
            high_mean = float(torch.mean(proj[high_mask]).item()) if bool(torch.any(high_mask)) else 0.0
            if high_mean < low_mean:
                pc1 = -pc1
        directions[concept] = np.asarray(pc1.detach().cpu().tolist(), dtype=np.float32)
    return directions


def build_problem_concept_directions(
    problem_records: Sequence[ProblemRecord],
    concept_dirs: Dict[str, np.ndarray],
) -> np.ndarray:
    out = np.zeros((len(problem_records), DS), dtype=np.float32)
    for pos, problem in enumerate(problem_records):
        vectors = [concept_dirs[concept] for concept in problem.concepts if concept in concept_dirs]
        if not vectors:
            continue
        vec = np.mean(np.stack(vectors, axis=0), axis=0)
        norm = float(np.linalg.norm(vec))
        out[pos] = (vec / norm).astype(np.float32) if norm > 0 else np.zeros((DS,), dtype=np.float32)
    return out


def build_collaborative_vectors(
    student_sequences: Sequence[StudentSequence],
    valid_problem_ids: Sequence[str],
    *,
    seed: int,
) -> Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]:
    valid_set = set(valid_problem_ids)
    sentences: List[List[str]] = []
    for sequence in student_sequences:
        tokens = [str(log.get("problem_id") or "") for log in sequence.seq if str(log.get("problem_id") or "") in valid_set]
        if len(tokens) >= 2:
            sentences.append(tokens)
    if not sentences:
        return {}, {}

    model = Word2Vec(
        sentences=sentences,
        vector_size=64,
        window=COLLAB_WINDOW,
        min_count=1,
        sg=1,
        negative=NEGATIVE_SAMPLES,
        workers=1,
        epochs=5,
        seed=seed,
    )
    neighbors: Dict[str, List[str]] = {}
    vectors: Dict[str, np.ndarray] = {}
    for pid in model.wv.index_to_key:
        vectors[pid] = model.wv[pid].astype(np.float32)
        neighbors[pid] = [item for item, _score in model.wv.most_similar(pid, topn=5) if item != pid][:5]
    return neighbors, vectors


def build_graph_bundle(problem_records: Sequence[ProblemRecord]) -> Dict[str, Any]:
    concept_to_chapters: Dict[str, set[str]] = defaultdict(set)
    local_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for problem in problem_records:
        unique_concepts = sorted(set(problem.concepts))
        if problem.chapter:
            for concept in unique_concepts:
                concept_to_chapters[concept].add(problem.chapter)
        for i, left in enumerate(unique_concepts):
            for right in unique_concepts[i + 1 :]:
                pair = (left, right) if left < right else (right, left)
                local_counts[pair] += 1

    concept_neighbors: Dict[str, set[str]] = defaultdict(set)
    local_edges: List[Dict[str, Any]] = []

    chapter_to_concepts: Dict[str, set[str]] = defaultdict(set)
    for concept, chapters in concept_to_chapters.items():
        for chapter in chapters:
            chapter_to_concepts[chapter].add(concept)

    for chapter, concepts in chapter_to_concepts.items():
        ordered = sorted(concepts)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                concept_neighbors[left].add(right)
                concept_neighbors[right].add(left)
                local_edges.append({"src": left, "dst": right, "source": "chapter", "chapter": chapter})

    for (left, right), count in sorted(local_counts.items()):
        if count < LOCAL_COOCCUR_THRESHOLD:
            continue
        concept_neighbors[left].add(right)
        concept_neighbors[right].add(left)
        local_edges.append({"src": left, "dst": right, "source": "cofreq", "count": count})

    problem_neighbor_concepts: Dict[str, List[str]] = {}
    for problem in problem_records:
        neighbor_set = set()
        for concept in problem.concepts:
            neighbor_set.update(concept_neighbors.get(concept, set()))
        neighbor_set.difference_update(problem.concepts)
        problem_neighbor_concepts[problem.problem_id] = sorted(neighbor_set)

    return {
        "has_explicit_prerequisite": False,
        "e_pre": [],
        "tau_localco": LOCAL_COOCCUR_THRESHOLD,
        "concept_neighbors": {concept: sorted(neighbors) for concept, neighbors in sorted(concept_neighbors.items())},
        "problem_neighbor_concepts": problem_neighbor_concepts,
        "local_edges": local_edges,
    }


def build_training_sequences(
    student_sequences: Sequence[StudentSequence],
    pid_to_idx: Dict[str, int],
    problem_records: Sequence[ProblemRecord],
) -> List[Dict[str, Any]]:
    levels = [problem.cognitive_dimension for problem in problem_records]
    cached: List[Dict[str, Any]] = []
    for sequence in student_sequences:
        pid_indices: List[int] = []
        results: List[int] = []
        item_levels: List[int] = []
        for log in sequence.seq:
            pid = str(log.get("problem_id") or "")
            if pid not in pid_to_idx:
                continue
            pid_idx = pid_to_idx[pid]
            pid_indices.append(pid_idx)
            results.append(int(log.get("is_correct") or 0))
            item_levels.append(levels[pid_idx])
        if len(pid_indices) >= 2:
            cached.append(
                {
                    "user_id": sequence.user_id,
                    "problem_indices": pid_indices,
                    "results": results,
                    "levels": item_levels,
                }
            )
    return cached


def build_target_samples(
    sequence_cache: Sequence[Dict[str, Any]],
    *,
    smoke: bool,
    seed: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    train_samples: List[Tuple[int, int]] = []
    val_samples: List[Tuple[int, int]] = []
    for seq_idx, sequence in enumerate(sequence_cache):
        user_id = str(sequence["user_id"])
        bucket = user_hash_bucket(user_id, TRAIN_VAL_MOD)
        target_list = val_samples if bucket == 0 else train_samples
        for t in range(1, len(sequence["problem_indices"])):
            target_list.append((seq_idx, t))

    rng = random.Random(seed)
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    if smoke:
        train_samples = train_samples[:SMOKE_TRAIN_MAX_SAMPLES]
        val_samples = val_samples[: max(1, SMOKE_TRAIN_MAX_SAMPLES // 8)]
    return train_samples, val_samples


def _batch_loss(
    model: StrictPriorModel,
    batch_samples: Sequence[Tuple[int, int]],
    sequence_cache: Sequence[Dict[str, Any]],
    hqtext_tensor: torch.Tensor,
    hqid_tensor: torch.Tensor,
    *,
    device: str,
) -> torch.Tensor:
    unique_problem_indices: set[int] = set()
    for seq_idx, t in batch_samples:
        sequence = sequence_cache[seq_idx]
        unique_problem_indices.add(int(sequence["problem_indices"][t]))
        history_start = max(0, t - HISTORY_WINDOW)
        for idx in sequence["problem_indices"][history_start:t]:
            unique_problem_indices.add(int(idx))

    sorted_unique = sorted(unique_problem_indices)
    local_pos = {pid_idx: pos for pos, pid_idx in enumerate(sorted_unique)}
    eqbase_local = model.eqbase(
        hqtext_tensor[sorted_unique].to(device),
        hqid_tensor[sorted_unique].to(device),
    )

    z_rows: List[torch.Tensor] = []
    target_eq_rows: List[torch.Tensor] = []
    labels: List[float] = []

    for seq_idx, t in batch_samples:
        sequence = sequence_cache[seq_idx]
        target_idx = int(sequence["problem_indices"][t])
        target_eq = eqbase_local[local_pos[target_idx]]
        target_level = int(sequence["levels"][t])

        history_start = max(0, t - HISTORY_WINDOW)
        hist_problem_indices = sequence["problem_indices"][history_start:t]
        hist_results = sequence["results"][history_start:t]
        hist_levels = sequence["levels"][history_start:t]
        if not hist_problem_indices:
            z_rows.append(torch.zeros((260,), dtype=torch.float32, device=device))
            target_eq_rows.append(target_eq)
            labels.append(float(sequence["results"][t]))
            continue

        hist_eq = torch.stack([eqbase_local[local_pos[int(pid_idx)]] for pid_idx in hist_problem_indices], dim=0)
        hist_eq_norm = F.normalize(hist_eq, dim=-1)
        target_eq_norm = F.normalize(target_eq.unsqueeze(0), dim=-1).squeeze(0)
        cos_scores = torch.sum(hist_eq_norm * target_eq_norm.unsqueeze(0), dim=-1)

        history_positions = list(range(history_start, t))
        score_terms: List[torch.Tensor] = []
        for pos, cos_score, level in zip(history_positions, cos_scores, hist_levels):
            lag = t - pos
            score_terms.append(
                cos_score
                - float(LAMBDA_L) * abs(float(level - target_level))
                - float(LAMBDA_T) * math.log1p(float(lag))
            )
        score_tensor = torch.stack(score_terms, dim=0) / float(TAU)
        alpha = torch.softmax(score_tensor, dim=0)

        result_tensor = torch.tensor(hist_results, dtype=torch.float32, device=device).unsqueeze(-1)
        relation_features = torch.tensor(
            [
                [1.0 if level < target_level else 0.0, 1.0 if level == target_level else 0.0, 1.0 if level > target_level else 0.0]
                for level in hist_levels
            ],
            dtype=torch.float32,
            device=device,
        )
        x_i = torch.cat([hist_eq, result_tensor, relation_features], dim=-1)
        z = torch.sum(alpha.unsqueeze(-1) * x_i, dim=0)

        z_rows.append(z)
        target_eq_rows.append(target_eq)
        labels.append(float(sequence["results"][t]))

    z_batch = torch.stack(z_rows, dim=0)
    target_eq_batch = torch.stack(target_eq_rows, dim=0)
    y = torch.tensor(labels, dtype=torch.float32, device=device)

    d_batch = model.dynamic(z_batch)
    diag_logits = model.diag_logits(target_eq_batch, d_batch)
    summary_logits = model.summary_logits(d_batch)
    loss_diag = F.binary_cross_entropy_with_logits(diag_logits, y)
    loss_summary = F.binary_cross_entropy_with_logits(summary_logits, y)
    return 0.5 * (loss_diag + loss_summary)


def train_strict_model(
    problem_records: Sequence[ProblemRecord],
    student_sequences: Sequence[StudentSequence],
    hqtext_vectors: np.ndarray,
    hqid_vectors: np.ndarray,
    *,
    priors_dir: Path,
    smoke: bool,
    seed: int,
) -> Tuple[StrictPriorModel, Dict[str, Any]]:
    pid_to_idx = {problem.problem_id: idx for idx, problem in enumerate(problem_records)}
    sequence_cache = build_training_sequences(student_sequences, pid_to_idx, problem_records)
    train_samples, val_samples = build_target_samples(sequence_cache, smoke=smoke, seed=seed)

    device = pick_device()
    model = StrictPriorModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)

    hqtext_tensor = torch.tensor(hqtext_vectors.astype(np.float32), dtype=torch.float32)
    hqid_tensor = torch.tensor(hqid_vectors.astype(np.float32), dtype=torch.float32)

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience_left = TRAIN_EARLY_STOP_PATIENCE
    history: List[Dict[str, float]] = []

    for epoch in range(1, TRAIN_MAX_EPOCHS + 1):
        model.train()
        random.Random(seed + epoch).shuffle(train_samples)
        train_losses: List[float] = []
        for start in tqdm(range(0, len(train_samples), TRAIN_BATCH_SIZE), desc=f"strict train epoch {epoch}"):
            batch = train_samples[start : start + TRAIN_BATCH_SIZE]
            if not batch:
                continue
            optimizer.zero_grad()
            loss = _batch_loss(
                model,
                batch,
                sequence_cache,
                hqtext_tensor,
                hqid_tensor,
                device=device,
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for start in range(0, len(val_samples), TRAIN_BATCH_SIZE):
                batch = val_samples[start : start + TRAIN_BATCH_SIZE]
                if not batch:
                    continue
                loss = _batch_loss(
                    model,
                    batch,
                    sequence_cache,
                    hqtext_tensor,
                    hqid_tensor,
                    device=device,
                )
                val_losses.append(float(loss.detach().cpu().item()))
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = TRAIN_EARLY_STOP_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()

    training_report = {
        "seed": seed,
        "device": device,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "epochs_ran": len(history),
        "history": history,
        "assumptions": {
            "joint_loss": "0.5 * BCE(diag_head, y) + 0.5 * BCE(summary_head, y)",
            "eqsem_training_phase": "train eqbase/dynamic heads on eqbase, then derive concept directions in eqbase latent space",
        },
    }
    model_state_path = priors_dir / "model_state.pt"
    training_report_path = priors_dir / "training_report.json"
    torch.save({"state_dict": model.state_dict()}, model_state_path)
    write_json(training_report, training_report_path)
    return model.cpu(), training_report


def save_pickle(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_stage32(
    *,
    problem_json: Path,
    student_json: Path,
    priors_dir: Path,
    seed: int = TRAIN_SEED,
    smoke: bool = False,
) -> Stage32Result:
    ensure_dir(priors_dir)
    problem_records = load_problem_records(
        problem_json,
        max_problems=SMOKE_MAX_PROBLEMS if smoke else None,
    )
    student_sequences = load_student_sequences(
        student_json,
        max_students=SMOKE_MAX_STUDENTS if smoke else None,
        max_targets_per_student=SMOKE_MAX_TARGETS_PER_STUDENT if smoke else None,
    )
    if not problem_records:
        raise ValueError("No problems loaded for strict stage 3.2")
    if not student_sequences:
        raise ValueError("No student sequences loaded for strict stage 3.2")

    local_model_path = resolve_local_sentence_transformer_path(BGE_MODEL_NAME)
    model = SentenceTransformer(str(local_model_path) if local_model_path else BGE_MODEL_NAME, local_files_only=bool(local_model_path))
    raw_texts = [problem.text or problem.title or problem.problem_id for problem in problem_records]
    hqtext_vectors = model.encode(
        raw_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    hqtext_vectors = np.asarray(hqtext_vectors.tolist(), dtype=np.float32)
    semantic_ids_path = priors_dir / "semantic_ids.json"
    semantic_ids, semantic_id_texts = build_semantic_ids(problem_records, hqtext_vectors, semantic_ids_path=semantic_ids_path)
    hqid_vectors = model.encode(
        semantic_id_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    hqid_vectors = np.asarray(hqid_vectors.tolist(), dtype=np.float32)

    pid_to_idx = {problem.problem_id: idx for idx, problem in enumerate(problem_records)}
    pid_to_mu_q = estimate_rasch_mu_q(student_sequences, pid_to_idx, seed=seed)
    model_torch, training_report = train_strict_model(
        problem_records,
        student_sequences,
        hqtext_vectors,
        hqid_vectors,
        priors_dir=priors_dir,
        smoke=smoke,
        seed=seed,
    )

    with torch.no_grad():
        hqtext_tensor = torch.tensor(hqtext_vectors.astype(np.float32), dtype=torch.float32)
        hqid_tensor = torch.tensor(hqid_vectors.astype(np.float32), dtype=torch.float32)
        eqbase_vectors = model_torch.eqbase(hqtext_tensor, hqid_tensor).detach().cpu().tolist()
        eqbase_vectors = np.asarray(eqbase_vectors, dtype=np.float32)

    concept_pc1_dirs = build_concept_pc1_directions(problem_records, eqbase_vectors)
    problem_dcq = build_problem_concept_directions(problem_records, concept_pc1_dirs)
    mu_arr = np.asarray([pid_to_mu_q.get(problem.problem_id, 0.0) for problem in problem_records], dtype=np.float32).reshape(-1, 1)
    eqsem_vectors = eqbase_vectors + mu_arr * problem_dcq
    eqsem_vectors = eqsem_vectors.astype(np.float32)

    collab_neighbors, collab_vectors = build_collaborative_vectors(
        student_sequences,
        [problem.problem_id for problem in problem_records],
        seed=seed,
    )
    graph_bundle = build_graph_bundle(problem_records)

    semantic_vectors = {problem.problem_id: eqsem_vectors[idx] for idx, problem in enumerate(problem_records)}
    hqtext_map = {problem.problem_id: hqtext_vectors[idx] for idx, problem in enumerate(problem_records)}
    hqid_map = {problem.problem_id: hqid_vectors[idx] for idx, problem in enumerate(problem_records)}
    eqbase_map = {problem.problem_id: eqbase_vectors[idx] for idx, problem in enumerate(problem_records)}

    save_pickle(semantic_vectors, priors_dir / "semantic_vectors.pkl")
    save_pickle(hqtext_map, priors_dir / "hqtext_vectors.pkl")
    save_pickle(hqid_map, priors_dir / "hqid_vectors.pkl")
    save_pickle(eqbase_map, priors_dir / "eqbase_vectors.pkl")
    save_pickle(concept_pc1_dirs, priors_dir / "concept_pc1_dirs.pkl")
    save_pickle(collab_vectors, priors_dir / "item_collaborative_embeddings.pkl")
    write_json(pid_to_mu_q, priors_dir / "problem_mu_q.json")
    write_json(collab_neighbors, priors_dir / "item_collaborative.json")
    write_json(graph_bundle, priors_dir / "concept_graph_bundle.json")

    catalog_records = [
        ProblemCatalogRecord(
            problem_id=problem.problem_id,
            semantic_id=semantic_ids[problem.problem_id],
            text=problem.text,
            title=problem.title,
            chapter=problem.chapter,
            location=problem.location,
            cognitive_dimension=problem.cognitive_dimension,
            concepts=problem.concepts,
        )
        for problem in problem_records
    ]
    problem_catalog_path = priors_dir / "problem_catalog.jsonl"
    write_jsonl(dataclass_list_to_jsonl(catalog_records), problem_catalog_path)

    implementation_defaults = {
        "bge_model_name": BGE_MODEL_NAME,
        "kglobal": KGLOBAL,
        "klocal": KLOCAL,
        "random_state": KMEANS_RANDOM_STATE,
        "n_init": KMEANS_N_INIT,
        "ctfidf_max_features": CTFIDF_MAX_FEATURES,
        "history_window": HISTORY_WINDOW,
        "collab_window": COLLAB_WINDOW,
        "negative_samples": NEGATIVE_SAMPLES,
        "local_cooccur_threshold": LOCAL_COOCCUR_THRESHOLD,
        "rasch_a": A_SCALE,
        "lambda_l": LAMBDA_L,
        "lambda_t": LAMBDA_T,
        "tau": TAU,
        "epsilon": EPS,
        "smoke": smoke,
    }
    implementation_defaults_path = priors_dir / "implementation_defaults.json"
    write_json(implementation_defaults, implementation_defaults_path)

    manifest = Stage32Result(
        priors_dir=str(priors_dir),
        semantic_ids_path=str(priors_dir / "semantic_ids.json"),
        semantic_vectors_path=str(priors_dir / "semantic_vectors.pkl"),
        hqtext_vectors_path=str(priors_dir / "hqtext_vectors.pkl"),
        hqid_vectors_path=str(priors_dir / "hqid_vectors.pkl"),
        eqbase_vectors_path=str(priors_dir / "eqbase_vectors.pkl"),
        problem_mu_q_path=str(priors_dir / "problem_mu_q.json"),
        concept_pc1_path=str(priors_dir / "concept_pc1_dirs.pkl"),
        item_collaborative_path=str(priors_dir / "item_collaborative.json"),
        item_collaborative_vectors_path=str(priors_dir / "item_collaborative_embeddings.pkl"),
        graph_bundle_path=str(priors_dir / "concept_graph_bundle.json"),
        problem_catalog_path=str(problem_catalog_path),
        model_state_path=str(priors_dir / "model_state.pt"),
        training_report_path=str(priors_dir / "training_report.json"),
        implementation_defaults_path=str(implementation_defaults_path),
        manifest_path=str(priors_dir / "stage32_manifest.json"),
        problem_count=len(problem_records),
        student_count=len(student_sequences),
    )
    write_json(asdict(manifest), Path(manifest.manifest_path))
    return manifest
