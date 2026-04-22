from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .constants import (
    ALPHA_TIME,
    BETA_NEG,
    BETA_POS,
    COVERAGE_WEIGHTS,
    DELTA_GRAPH,
    EPS,
    EXPLICIT_MATCH_WEIGHTS,
    GAMMA_HIGH,
    GAMMA_PRE,
    HISTORY_WINDOW,
    K1_DEFAULT,
    K2_DEFAULT,
    LLM_SUMMARY_CHUNK_SIZE,
    LLM_SUMMARY_WORKERS,
    LAMBDA_COV,
    LAMBDA_RED,
    QUESTION_TEXT_ELLIPSIS,
    QUESTION_TEXT_LIMIT,
    REDUNDANCY_WEIGHTS,
    RERANK_TOPK,
    RERANK_WEIGHT,
    ROLE_LABELS,
    ROLE_ORDER,
    ROLE_PRIORITY,
    ROLE_THRESHOLDS,
    RHO,
    SUMMARY_TEMPLATE,
    SUPPORT_SCORE_DECIMALS,
    TEXT_EMBED_BATCH_SIZE,
    TEXT_EMBED_MAX_LENGTH,
    TEXT_EMBED_MODEL_NAME,
    TEXT_RERANK_BATCH_SIZE,
    TEXT_RERANK_MAX_LENGTH,
    TEXT_RERANK_MODEL_NAME,
    USE_QWEN_RERANKER,
    WEIGHT_STAGE1,
    WEIGHT_STAGE2,
)
from .io_utils import (
    atomic_save_text,
    ensure_dir,
    format_float,
    load_problem_records,
    load_student_sequences,
    pick_device,
    write_json,
)
from .llm_utils import (
    OpenAICompatibleSummarizer,
    append_summary_cache_entries,
    load_json_cache,
    load_summary_cache,
    parse_llm_summary_json,
    summary_cache_key,
)
from .models import load_strict_prior_model
from .retrieval_models import QwenEmbeddingEncoder, QwenReranker


@dataclass
class Stage34Result:
    contexts_path: str
    preview_path: str
    embeddings_path: Optional[str]
    manifest_path: str
    record_count: int
    text_embed_model: str
    text_embed_batch_size: int
    text_embed_max_length: int
    text_rerank_model: str
    text_rerank_batch_size: int
    use_qwen_reranker: bool
    rerank_topk: int
    rerank_weight: float
    reranker_cache_path: Optional[str]
    llm_summary_workers: int
    llm_summary_chunk_size: int
    context_shard_index: int
    context_num_shards: int
    merge_context_shards: bool


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms <= 0, 1.0, norms)
    return (matrix / norms).astype(np.float32, copy=False)


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    set_left = set(left)
    set_right = set(right)
    if not set_left or not set_right:
        return 0.0
    return float(len(set_left & set_right)) / float(len(set_left | set_right))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _scaled_cosine(a: np.ndarray, b: np.ndarray) -> float:
    return _cosine(a, b)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _question_text(text: str) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= QUESTION_TEXT_LIMIT:
        return text
    return text[:QUESTION_TEXT_LIMIT] + QUESTION_TEXT_ELLIPSIS


def _role_from_candidate(candidate: Dict[str, Any]) -> str:
    scores = candidate["raw_scores"]
    active_roles = [role for role in ROLE_ORDER if candidate["activation"][role] == 1]
    role_pool = active_roles or list(ROLE_ORDER)
    best_role = None
    best_key = None
    for role in role_pool:
        key = (scores[role], -ROLE_PRIORITY[role])
        if best_key is None or key > best_key:
            best_key = key
            best_role = role
    assert best_role is not None
    return best_role


class GraphAccessor:
    def __init__(self, graph_bundle: Dict[str, Any]) -> None:
        self.concept_neighbors = {
            str(concept): set(neighbors)
            for concept, neighbors in (graph_bundle.get("concept_neighbors") or {}).items()
        }
        self.prerequisite_map: Dict[str, set[str]] = {}
        for edge in graph_bundle.get("e_pre") or []:
            src = str(edge.get("src") or "").strip()
            dst = str(edge.get("dst") or "").strip()
            if src and dst:
                self.prerequisite_map.setdefault(dst, set()).add(src)
        self.problem_neighbor_concepts = {
            str(pid): list(neighbors)
            for pid, neighbors in (graph_bundle.get("problem_neighbor_concepts") or {}).items()
        }

    def problem_neighbors(self, pid: str) -> List[str]:
        return list(self.problem_neighbor_concepts.get(pid, []))

    def structural_bonus(self, concepts_i: Sequence[str], concepts_t: Sequence[str]) -> float:
        best = 0.0
        target_set = set(concepts_t)
        for concept_i in concepts_i:
            if any(concept_i in self.prerequisite_map.get(target, set()) for target in target_set):
                best = max(best, 0.75 * math.exp(-DELTA_GRAPH * 1.0))
            neigh = self.concept_neighbors.get(concept_i, set())
            if neigh & target_set:
                best = max(best, 0.5 * math.exp(-DELTA_GRAPH * 1.0))
        return best


def _dtc(
    seq_problem_indices: Sequence[int],
    current_t: int,
    hist_i: int,
    eqsem_norm: np.ndarray,
) -> float:
    value = 1.0
    qt_vec = eqsem_norm[seq_problem_indices[current_t]]
    for j in range(hist_i + 1, current_t):
        j_vec = eqsem_norm[seq_problem_indices[j]]
        value += 1.0 - _scaled_cosine(j_vec, qt_vec)
    return value


def _dtc_values(
    seq_problem_indices: Sequence[int],
    current_t: int,
    eq_cos_matrix: np.ndarray,
) -> np.ndarray:
    if current_t <= 0:
        return np.zeros((0,), dtype=np.float32)
    one_minus = 1.0 - np.clip(eq_cos_matrix[:current_t, current_t], -1.0, 1.0)
    suffix = np.cumsum(one_minus[::-1], dtype=np.float32)[::-1]
    shifted = np.concatenate([suffix[1:], np.zeros((1,), dtype=np.float32)], axis=0)
    return (1.0 + shifted).astype(np.float32)


def _build_sequence_cache(
    seq_problem_indices: Sequence[int],
    seq_levels: Sequence[int],
    pid_lookup: Sequence[str],
    eqsem_norm: np.ndarray,
    collab_norm: Dict[int, np.ndarray],
    graph_accessor: GraphAccessor,
    problem_catalog: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    seq_indices = np.asarray(seq_problem_indices, dtype=np.int64)
    seq_levels_arr = np.asarray(seq_levels, dtype=np.int32)
    seq_pids = [pid_lookup[int(idx)] for idx in seq_indices.tolist()]
    seq_concepts = [problem_catalog[pid]["concepts"] for pid in seq_pids]
    seq_concept_sets = [set(concepts) for concepts in seq_concepts]
    seq_neighbors = [set(graph_accessor.problem_neighbors(pid)) for pid in seq_pids]

    seq_eq_norm = eqsem_norm[seq_indices]
    eq_cos = np.clip(seq_eq_norm @ seq_eq_norm.T, -1.0, 1.0).astype(np.float32)

    collab_cos = np.zeros((len(seq_indices), len(seq_indices)), dtype=np.float32)
    available_rows: List[int] = []
    collab_rows: List[np.ndarray] = []
    for pos, global_idx in enumerate(seq_indices.tolist()):
        vector = collab_norm.get(int(global_idx))
        if vector is None:
            continue
        available_rows.append(pos)
        collab_rows.append(vector)
    if collab_rows:
        collab_matrix = np.stack(collab_rows, axis=0).astype(np.float32)
        collab_pairwise = np.clip(collab_matrix @ collab_matrix.T, -1.0, 1.0).astype(np.float32)
        for left_pos, row_pos in enumerate(available_rows):
            collab_cos[row_pos, available_rows] = collab_pairwise[left_pos]

    jaccard = np.zeros((len(seq_indices), len(seq_indices)), dtype=np.float32)
    graph_bonus = np.zeros((len(seq_indices), len(seq_indices)), dtype=np.float32)
    overlap_lists: List[List[List[str]]] = [[[] for _ in range(len(seq_indices))] for _ in range(len(seq_indices))]
    for hist_pos in range(len(seq_indices)):
        left_set = seq_concept_sets[hist_pos]
        left_concepts = seq_concepts[hist_pos]
        for target_t in range(hist_pos + 1, len(seq_indices)):
            right_set = seq_concept_sets[target_t]
            inter = sorted(left_set & right_set)
            union_size = len(left_set | right_set)
            jaccard_value = float(len(inter)) / float(union_size) if union_size > 0 else 0.0
            jaccard[hist_pos, target_t] = jaccard_value
            overlap_lists[hist_pos][target_t] = inter
            graph_bonus[hist_pos, target_t] = graph_accessor.structural_bonus(left_concepts, seq_concepts[target_t])

    return {
        "seq_indices": seq_indices,
        "seq_levels": seq_levels_arr,
        "seq_pids": seq_pids,
        "seq_neighbors": seq_neighbors,
        "eq_cos": eq_cos,
        "collab_cos": collab_cos,
        "jaccard": jaccard,
        "graph_bonus": graph_bonus,
        "overlap_lists": overlap_lists,
    }


def _compute_dynamic_prior(
    seq_problem_indices: Sequence[int],
    seq_results: Sequence[int],
    seq_levels: Sequence[int],
    target_t: int,
    eqsem: np.ndarray,
    eqsem_norm: np.ndarray,
    model: Any,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    history_start = max(0, target_t - HISTORY_WINDOW)
    hist_positions = list(range(history_start, target_t))
    if not hist_positions:
        zero_d = np.zeros((128,), dtype=np.float32)
        return np.zeros((260,), dtype=np.float32), zero_d, 0.5

    qt_idx = seq_problem_indices[target_t]
    qt_level = seq_levels[target_t]
    qt_vec = eqsem[qt_idx]
    qt_vec_norm = eqsem_norm[qt_idx]

    s_values: List[float] = []
    x_rows: List[np.ndarray] = []
    for pos in hist_positions:
        qi_idx = seq_problem_indices[pos]
        qi_vec = eqsem[qi_idx]
        qi_vec_norm = eqsem_norm[qi_idx]
        level_i = seq_levels[pos]
        s_i = _cosine(qi_vec_norm, qt_vec_norm) - abs(level_i - qt_level) - math.log1p(target_t - pos)
        s_values.append(s_i)
        x_rows.append(
            np.concatenate(
                [
                    qi_vec.astype(np.float32),
                    np.asarray([float(seq_results[pos])], dtype=np.float32),
                    np.asarray(
                        [
                            1.0 if level_i < qt_level else 0.0,
                            1.0 if level_i == qt_level else 0.0,
                            1.0 if level_i > qt_level else 0.0,
                        ],
                        dtype=np.float32,
                    ),
                ],
                axis=0,
            )
        )

    s_arr = np.asarray(s_values, dtype=np.float32)
    s_arr = s_arr - np.max(s_arr)
    alpha = np.exp(s_arr)
    alpha = alpha / max(float(alpha.sum()), EPS)
    z = np.sum(alpha.reshape(-1, 1) * np.stack(x_rows, axis=0), axis=0).astype(np.float32)

    with torch.no_grad():
        z_t = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
        d_tensor = model.dynamic(z_t).squeeze(0).to(torch.float32)
        qt_tensor = torch.tensor(qt_vec, dtype=torch.float32, device=device).unsqueeze(0)
        sdyn = torch.sigmoid(model.diag_logits(qt_tensor, d_tensor.unsqueeze(0))).cpu().item()
        d_vec = np.asarray(d_tensor.cpu().tolist(), dtype=np.float32)
    return z, d_vec, float(sdyn)


def _history_diag_probs(
    hist_problem_indices: Sequence[int],
    eqsem: np.ndarray,
    d_vec: np.ndarray,
    model: Any,
    device: str,
) -> np.ndarray:
    if not hist_problem_indices:
        return np.zeros((0,), dtype=np.float32)
    eq_batch = torch.tensor(eqsem[list(hist_problem_indices)], dtype=torch.float32, device=device)
    d_batch = torch.tensor(np.repeat(d_vec[None, :], len(hist_problem_indices), axis=0), dtype=torch.float32, device=device)
    with torch.no_grad():
        probs = torch.sigmoid(model.diag_logits(eq_batch, d_batch)).detach().cpu().numpy()
    return np.asarray(probs, dtype=np.float32)


def _candidate_scores(
    *,
    hist_pos: int,
    current_t: int,
    seq_problem_indices: Sequence[int],
    seq_results: Sequence[int],
    seq_levels: Sequence[int],
    eqsem: np.ndarray,
    problem_catalog: Dict[str, Dict[str, Any]],
    pid_lookup: Sequence[str],
    p_diag: float,
    dtc_value: float,
    seq_cache: Dict[str, Any],
) -> Dict[str, Any]:
    pid_i = pid_lookup[seq_problem_indices[hist_pos]]
    pid_t = pid_lookup[seq_problem_indices[current_t]]
    meta_i = problem_catalog[pid_i]
    meta_t = problem_catalog[pid_t]
    concepts_i = meta_i["concepts"]
    concepts_t = meta_t["concepts"]
    ki = float(seq_cache["jaccard"][hist_pos, current_t])
    delta_l = int(meta_t["cognitive_dimension"]) - int(meta_i["cognitive_dimension"])

    eq_i = eqsem[seq_problem_indices[hist_pos]]
    seq_eq_cos = seq_cache["eq_cos"]

    mi = RHO * float(seq_results[hist_pos]) + (1.0 - RHO) * float(p_diag)

    spre = 0.0
    speer = 0.0
    shigh = 0.0
    if delta_l > 0:
        spre = ki * mi * math.exp(-GAMMA_PRE * abs(delta_l))
    elif delta_l == 0:
        speer = float(seq_eq_cos[hist_pos, current_t])
    else:
        shigh = ki * ((BETA_POS * float(seq_results[hist_pos])) - (BETA_NEG * (1.0 - float(seq_results[hist_pos])))) * math.exp(
            -GAMMA_HIGH * abs(delta_l)
        )

    graph_bonus = float(seq_cache["graph_bonus"][hist_pos, current_t])
    graw = ki + graph_bonus
    gcomp = max(0.0, graw - ki)
    explicit_match = _clip01(
        EXPLICIT_MATCH_WEIGHTS["K"] * ki
        + EXPLICIT_MATCH_WEIGHTS["L"] * math.exp(-1.0 * abs(delta_l))
        + EXPLICIT_MATCH_WEIGHTS["G"] * gcomp
    )
    collab_sim = float(seq_cache["collab_cos"][hist_pos, current_t])
    scollab = (1.0 - explicit_match) * collab_sim

    ti = math.exp(-ALPHA_TIME * dtc_value)

    bi = (
        WEIGHT_STAGE1["K"] * ki
        + WEIGHT_STAGE1["pre"] * spre
        + WEIGHT_STAGE1["peer"] * speer
        + WEIGHT_STAGE1["high"] * shigh
        + WEIGHT_STAGE1["graph"] * gcomp
        + WEIGHT_STAGE1["collab"] * scollab
    )
    ri = ti * bi
    ui = (
        WEIGHT_STAGE2["K"] * ki
        + WEIGHT_STAGE2["pre"] * spre
        + WEIGHT_STAGE2["peer"] * speer
        + WEIGHT_STAGE2["high"] * shigh
        + WEIGHT_STAGE2["graph"] * gcomp
        + WEIGHT_STAGE2["collab"] * scollab
    )

    activation = {
        "pre": int(spre > ROLE_THRESHOLDS["pre"]),
        "peer": int(speer > ROLE_THRESHOLDS["peer"]),
        "high": int(shigh > ROLE_THRESHOLDS["high"]),
        "graph": int(gcomp > ROLE_THRESHOLDS["graph"]),
        "collab": int(scollab > ROLE_THRESHOLDS["collab"]),
    }
    return {
        "history_pos": hist_pos,
        "problem_id": pid_i,
        "knowledge_overlap_concepts": list(seq_cache["overlap_lists"][hist_pos][current_t]),
        "raw_scores": {
            "pre": float(spre),
            "peer": float(speer),
            "high": float(shigh),
            "graph": float(gcomp),
            "collab": float(scollab),
        },
        "activation": activation,
        "Ki": float(ki),
        "Mi": float(mi),
        "dtc": float(dtc_value),
        "Ti": float(ti),
        "Ui": float(ui),
        "Ri": float(ri),
        "level_diff": int(delta_l),
        "answer_result": "正确" if int(seq_results[hist_pos]) == 1 else "错误",
    }


def _sim_a(left: Dict[str, int], right: Dict[str, int]) -> float:
    dot = sum(float(left[role]) * float(right[role]) for role in ROLE_ORDER)
    left_sum = sum(left.values())
    right_sum = sum(right.values())
    denom = left_sum + right_sum - dot + EPS
    return dot / denom


def _redundancy(
    candidate: Dict[str, Any],
    selected: Sequence[Dict[str, Any]],
    eqsem_norm: np.ndarray,
    pid_to_idx: Dict[str, int],
    catalog: Dict[str, Dict[str, Any]],
) -> float:
    if not selected:
        return 0.0
    pid_i = candidate["problem_id"]
    concepts_i = catalog[pid_i]["concepts"]
    best = 0.0
    for other in selected:
        pid_j = other["problem_id"]
        sim_k = _jaccard(concepts_i, catalog[pid_j]["concepts"])
        sim_a = _sim_a(candidate["activation"], other["activation"])
        sim_e = _scaled_cosine(eqsem_norm[pid_to_idx[pid_i]], eqsem_norm[pid_to_idx[pid_j]])
        value = (
            REDUNDANCY_WEIGHTS["K"] * sim_k
            + REDUNDANCY_WEIGHTS["A"] * sim_a
            + REDUNDANCY_WEIGHTS["E"] * sim_e
        )
        best = max(best, value)
    return best


def _coverage_gain(
    candidate: Dict[str, Any],
    selected: Sequence[Dict[str, Any]],
    target_pid: str,
    graph_accessor: GraphAccessor,
    catalog: Dict[str, Dict[str, Any]],
) -> float:
    target_concepts = catalog[target_pid]["concepts"]
    target_neighbor_set = set(graph_accessor.problem_neighbors(target_pid))
    covered_roles = {role: 0 for role in ROLE_ORDER}
    covered_concepts: set[str] = set()
    covered_neighbors: set[str] = set()
    for other in selected:
        for role in ROLE_ORDER:
            covered_roles[role] = max(covered_roles[role], int(other["activation"][role]))
        covered_concepts.update(other["knowledge_overlap_concepts"])
        covered_neighbors.update(set(graph_accessor.problem_neighbors(other["problem_id"])) & target_neighbor_set)

    role_gain = 0.0
    for role in ROLE_ORDER:
        role_gain += float(candidate["activation"][role]) * (1.0 - float(covered_roles[role]))

    new_concepts = set(candidate["knowledge_overlap_concepts"]) - covered_concepts
    new_neighbors = (set(graph_accessor.problem_neighbors(candidate["problem_id"])) & target_neighbor_set) - covered_neighbors
    knowledge_gain = len(new_concepts) / float(len(target_concepts) + EPS)
    neighbor_gain = len(new_neighbors) / float(len(target_neighbor_set) + EPS)
    return (
        COVERAGE_WEIGHTS["role"] * role_gain
        + COVERAGE_WEIGHTS["knowledge"] * knowledge_gain
        + COVERAGE_WEIGHTS["neighbor"] * neighbor_gain
    )


def _dominant_role(selected: Sequence[Dict[str, Any]]) -> str:
    role_scores = {role: 0.0 for role in ROLE_ORDER}
    for candidate in selected:
        for role in ROLE_ORDER:
            role_scores[role] += float(candidate["activation"][role]) * float(candidate["Ui"])
    best_role = max(role_scores.items(), key=lambda item: (item[1], -ROLE_PRIORITY[item[0]]))[0]
    return ROLE_LABELS[best_role]


def _summary_fields(
    target_pid: str,
    selected: Sequence[Dict[str, Any]],
    catalog: Dict[str, Dict[str, Any]],
    sdyn: float,
) -> Dict[str, Any]:
    target_concepts = catalog[target_pid]["concepts"]
    freq: Dict[str, int] = {concept: 0 for concept in target_concepts}
    for candidate in selected:
        for concept in candidate["knowledge_overlap_concepts"]:
            if concept in freq:
                freq[concept] += 1
    ordered = sorted(target_concepts, key=lambda concept: (-freq.get(concept, 0), concept))
    target_concepts_out = ordered[: min(2, len(target_concepts))]

    total_ui = sum(float(candidate["Ui"]) for candidate in selected)
    r_e = (
        sum(float(candidate["Ui"]) * (1.0 if candidate["answer_result"] == "正确" else 0.0) for candidate in selected)
        / float(total_ui + EPS)
        if selected
        else 0.0
    )
    ztrend = 0.5 * r_e + 0.5 * float(sdyn)
    if ztrend >= 0.67:
        recent_trend = "近期表现稳定偏强"
    elif ztrend >= 0.33:
        recent_trend = "近期表现波动"
    else:
        recent_trend = "近期表现偏弱"

    zrisk = 1.0 - ztrend
    if zrisk < 0.33:
        risk_level = "低"
    elif zrisk < 0.67:
        risk_level = "中"
    else:
        risk_level = "高"

    dominant_role = _dominant_role(selected)
    target_concepts_text = "、".join(target_concepts_out) if target_concepts_out else "无"
    summary_text = SUMMARY_TEMPLATE.format(
        target_concepts=target_concepts_text,
        recent_trend=recent_trend,
        dominant_role=dominant_role,
        risk_level=risk_level,
    )
    return {
        "target_concepts": target_concepts_out,
        "dominant_role": dominant_role,
        "recent_trend": recent_trend,
        "risk_level": risk_level,
        "sdyn": float(sdyn),
        "summary_text": summary_text,
    }


def _build_llm_context_text(llm_summary_text: str, main_context_text: str) -> str:
    parsed = parse_llm_summary_json(str(llm_summary_text or ""))
    parts: List[str] = []
    if parsed["summary"]:
        parts.append(parsed["summary"])
    if parsed["stable_points"]:
        parts.append("稳定掌握点:" + "；".join(parsed["stable_points"]))
    if parsed["weak_points"]:
        parts.append("持续薄弱点:" + "；".join(parsed["weak_points"]))
    parts.append(f"近期波动性:{parsed['volatility']}")
    parts.append(f"摘要置信度:{parsed['confidence']}")
    summary = " ".join(part.strip() for part in parts if part.strip())
    main = str(main_context_text or "").strip()
    if summary and main:
        return summary + "\n" + main
    return summary or main


def _build_llm_struct_texts(record: Dict[str, Any]) -> Dict[str, str]:
    llm_summary_text = str(record.get("summary_fields", {}).get("llm_summary_text") or "").strip()
    parsed = parse_llm_summary_json(llm_summary_text)
    stable_text = "；".join(parsed["stable_points"])
    weak_text = "；".join(parsed["weak_points"])
    summary_text = parsed["summary"]
    volatility = parsed["volatility"]
    confidence = parsed["confidence"]
    aux_text = f"波动性:{volatility} 置信度:{confidence}"
    return {
        "stable_text": stable_text,
        "weak_text": weak_text,
        "summary_text": summary_text,
        "aux_text": aux_text,
    }


def _build_llm_struct_feature_vector(record: Dict[str, Any]) -> np.ndarray:
    llm_summary_text = str(record.get("summary_fields", {}).get("llm_summary_text") or "").strip()
    parsed = parse_llm_summary_json(llm_summary_text)
    stable_points = [str(item).strip() for item in parsed["stable_points"] if str(item).strip()]
    weak_points = [str(item).strip() for item in parsed["weak_points"] if str(item).strip()]
    summary = str(parsed["summary"]).strip()

    volatility = str(parsed["volatility"])
    confidence = str(parsed["confidence"])
    levels = ("低", "中", "高")

    volatility_onehot = [1.0 if volatility == level else 0.0 for level in levels]
    confidence_onehot = [1.0 if confidence == level else 0.0 for level in levels]

    features = np.asarray(
        [
            float(len(stable_points)) / 2.0,
            float(len(weak_points)) / 2.0,
            1.0 if stable_points else 0.0,
            1.0 if weak_points else 0.0,
            min(float(len(summary)), 60.0) / 60.0,
            *volatility_onehot,
            *confidence_onehot,
        ],
        dtype=np.float32,
    )
    return features


def _unique_texts_with_inverse(texts: Sequence[str]) -> Tuple[List[str], np.ndarray]:
    unique_texts: List[str] = []
    inverse_indices = np.zeros((len(texts),), dtype=np.int64)
    text_to_unique_idx: Dict[str, int] = {}
    for idx, text in enumerate(texts):
        normalized = str(text or "")
        unique_idx = text_to_unique_idx.get(normalized)
        if unique_idx is None:
            unique_idx = len(unique_texts)
            unique_texts.append(normalized)
            text_to_unique_idx[normalized] = unique_idx
        inverse_indices[idx] = unique_idx
    return unique_texts, inverse_indices


def _encode_dedup_texts_resumable(
    *,
    encoder: QwenEmbeddingEncoder,
    texts: Sequence[str],
    instruction: str,
    desc: str,
    cache_prefix: Path,
) -> np.ndarray:
    unique_texts, inverse_indices = _unique_texts_with_inverse(texts)
    total_count = len(texts)
    unique_count = len(unique_texts)
    reuse_ratio = 1.0 - (float(unique_count) / float(total_count)) if total_count > 0 else 0.0
    print(
        f"[stage34] {desc}: total={total_count}, unique={unique_count}, reuse_ratio={reuse_ratio:.2%}",
        flush=True,
    )
    unique_embeddings = encoder.encode_texts_resumable(
        unique_texts,
        instruction=instruction,
        desc=desc,
        cache_prefix=cache_prefix,
    )
    unique_embeddings = np.asarray(unique_embeddings, dtype=np.float32)
    return unique_embeddings[inverse_indices]


def _load_index_records_from_contexts(contexts_path: Path) -> Tuple[List[Dict[str, Any]], int]:
    index_records: List[Dict[str, Any]] = []
    record_count = 0
    with contexts_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            index_records.append(
                {
                    "user_id": record["user_id"],
                    "target_t": record["target_t"],
                    "target_pid": record["target_pid"],
                }
            )
            record_count += 1
    return index_records, record_count


def _shard_suffix(shard_index: int, shard_count: int) -> str:
    return f"shard_{int(shard_index):02d}_of_{int(shard_count):02d}"


def _merge_context_shards(
    *,
    contexts_dir: Path,
    reports_dir: Path,
    shard_count: int,
    preview_limit: int,
    output_contexts_path: Path,
    output_preview_path: Path,
) -> int:
    shard_dir = contexts_dir / "shards"
    record_count = 0
    preview_lines: List[str] = []
    with output_contexts_path.open("w", encoding="utf-8") as dst:
        for shard_index in range(int(shard_count)):
            shard_path = shard_dir / f"contexts.{_shard_suffix(shard_index, shard_count)}.jsonl"
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing shard contexts file: {shard_path}")
            with shard_path.open("r", encoding="utf-8", errors="replace") as src:
                for line in tqdm(src, desc=f"merge shard {shard_index + 1}/{int(shard_count)}"):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if len(preview_lines) < max(0, int(preview_limit)):
                        preview_lines.append(
                            f"--- {record['user_id']} @ t={int(record['target_t'])} / {record['target_pid']} ---\n"
                            f"[MAIN]\n{str(record.get('main_context_text') or '').strip()}\n\n"
                            f"[TEMPLATE]\n{str(record.get('template_context_text') or '').strip()}\n"
                        )
                    record_count += 1
    atomic_save_text("\n\n".join(preview_lines), output_preview_path)
    return record_count


def _build_context_embeddings(
    *,
    contexts_path: Path,
    cache_dir: Path,
    device: str,
    text_embed_model: str,
    text_embed_batch_size: int,
    text_embed_max_length: int,
) -> Path:
    print("[stage34] loading text embedding model for context embeddings", flush=True)
    encoder = QwenEmbeddingEncoder(
        model_name_or_path=str(text_embed_model or TEXT_EMBED_MODEL_NAME),
        device=device,
        max_length=int(text_embed_max_length),
        batch_size=int(text_embed_batch_size),
    )
    parts_dir = ensure_dir(cache_dir / "context_embedding_parts")
    print("[stage34] scanning contexts for embedding inputs", flush=True)
    index_records, _record_count = _load_index_records_from_contexts(contexts_path)
    main_texts: List[str] = []
    template_texts: List[str] = []
    llm_texts: List[str] = []
    stable_texts: List[str] = []
    weak_texts: List[str] = []
    struct_summary_texts: List[str] = []
    aux_texts: List[str] = []
    struct_feature_rows: List[np.ndarray] = []
    has_any_llm_summary = False
    with contexts_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            main_texts.append(record["main_context_text"])
            template_texts.append(record["template_context_text"])
            llm_summary_text = str(record.get("summary_fields", {}).get("llm_summary_text") or "").strip()
            llm_context_text = str(record.get("llm_context_text") or "").strip()
            if llm_summary_text:
                has_any_llm_summary = True
                if not llm_context_text:
                    raise ValueError("Found llm_summary_text but llm_context_text is empty")
                llm_texts.append(llm_context_text)
                struct_texts = _build_llm_struct_texts(record)
                stable_texts.append(struct_texts["stable_text"])
                weak_texts.append(struct_texts["weak_text"])
                struct_summary_texts.append(struct_texts["summary_text"])
                aux_texts.append(struct_texts["aux_text"])
                struct_feature_rows.append(_build_llm_struct_feature_vector(record))
            else:
                llm_texts.append("")
                stable_texts.append("")
                weak_texts.append("")
                struct_summary_texts.append("")
                aux_texts.append("")
                struct_feature_rows.append(np.zeros((11,), dtype=np.float32))
    print(f"[stage34] encoding main context texts ({len(main_texts)})", flush=True)
    main_embeddings = _encode_dedup_texts_resumable(
        encoder=encoder,
        texts=main_texts,
        instruction="Encode retrieved educational evidence context for downstream knowledge tracing.",
        desc="strict main embeddings",
        cache_prefix=parts_dir / "main_embeddings_unique",
    )
    print(f"[stage34] encoding template context texts ({len(template_texts)})", flush=True)
    template_embeddings = _encode_dedup_texts_resumable(
        encoder=encoder,
        texts=template_texts,
        instruction="Encode summary-augmented educational evidence context for downstream knowledge tracing.",
        desc="strict template embeddings",
        cache_prefix=parts_dir / "template_embeddings_unique",
    )
    llm_embeddings = None
    if has_any_llm_summary:
        if not all(text.strip() for text in llm_texts):
            raise ValueError("LLM summaries are partially missing; strict full-system mode requires full llm coverage")
        print(f"[stage34] encoding llm context texts ({len(llm_texts)})", flush=True)
        llm_embeddings = _encode_dedup_texts_resumable(
            encoder=encoder,
            texts=llm_texts,
            instruction="Encode LLM-enhanced cognitive context for downstream knowledge tracing.",
            desc="strict llm embeddings",
            cache_prefix=parts_dir / "llm_embeddings_unique",
        )
    llm_struct_embeddings = None
    llm_struct_features = None
    if has_any_llm_summary:
        print(f"[stage34] encoding llm stable point texts ({len(stable_texts)})", flush=True)
        stable_embeddings = _encode_dedup_texts_resumable(
            encoder=encoder,
            texts=stable_texts,
            instruction="Encode stable cognitive mastery points extracted from LLM structured summaries.",
            desc="strict llm stable embeddings",
            cache_prefix=parts_dir / "llm_stable_embeddings_unique",
        )
        print(f"[stage34] encoding llm weak point texts ({len(weak_texts)})", flush=True)
        weak_embeddings = _encode_dedup_texts_resumable(
            encoder=encoder,
            texts=weak_texts,
            instruction="Encode weak cognitive points extracted from LLM structured summaries.",
            desc="strict llm weak embeddings",
            cache_prefix=parts_dir / "llm_weak_embeddings_unique",
        )
        print(f"[stage34] encoding llm summary texts ({len(struct_summary_texts)})", flush=True)
        summary_embeddings = _encode_dedup_texts_resumable(
            encoder=encoder,
            texts=struct_summary_texts,
            instruction="Encode concise cognitive state summaries extracted from LLM structured summaries.",
            desc="strict llm summary embeddings",
            cache_prefix=parts_dir / "llm_summary_embeddings_unique",
        )
        print(f"[stage34] encoding llm aux texts ({len(aux_texts)})", flush=True)
        aux_embeddings = _encode_dedup_texts_resumable(
            encoder=encoder,
            texts=aux_texts,
            instruction="Encode volatility and confidence descriptors extracted from LLM structured summaries.",
            desc="strict llm aux embeddings",
            cache_prefix=parts_dir / "llm_aux_embeddings_unique",
        )
        llm_struct_embeddings = np.concatenate(
            [
                np.asarray(stable_embeddings, dtype=np.float32),
                np.asarray(weak_embeddings, dtype=np.float32),
                np.asarray(summary_embeddings, dtype=np.float32),
                np.asarray(aux_embeddings, dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        llm_struct_features = np.stack(struct_feature_rows, axis=0).astype(np.float32)
        print("[stage34] built llm_struct_embeddings and llm_struct_features", flush=True)
    main_embeddings = np.asarray(main_embeddings, dtype=np.float32)
    template_embeddings = np.asarray(template_embeddings, dtype=np.float32)
    if llm_embeddings is not None:
        llm_embeddings = np.asarray(llm_embeddings, dtype=np.float32)
    embeddings_path = cache_dir / "context_embeddings.pkl"
    print(f"[stage34] embedding stage checkpoints are stored under {parts_dir}", flush=True)
    print(f"[stage34] writing context embeddings to {embeddings_path}", flush=True)
    with embeddings_path.open("wb") as f:
        payload = {
            "index": index_records,
            "main_embeddings": main_embeddings,
            "template_embeddings": template_embeddings,
        }
        if llm_embeddings is not None:
            payload["llm_embeddings"] = llm_embeddings
        if llm_struct_embeddings is not None:
            payload["llm_struct_embeddings"] = llm_struct_embeddings
        if llm_struct_features is not None:
            payload["llm_struct_features"] = llm_struct_features
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return embeddings_path


def _enrich_contexts_with_llm_summary(
    *,
    contexts_path: Path,
    cache_dir: Path,
    problem_catalog_records: Dict[str, Dict[str, Any]],
    summarizer: OpenAICompatibleSummarizer,
    llm_summary_workers: int,
    llm_summary_chunk_size: int,
) -> None:
    def _summarize_record(record: Dict[str, Any]) -> str:
        target_pid = str(record["target_pid"])
        target_meta = problem_catalog_records[target_pid]
        return summarizer.summarize(
            target_pid=target_pid,
            target_question_text=str(target_meta["text"]),
            target_semantic_id=str(record.get("target_semantic_id") or target_meta["semantic_id"]),
            target_concepts=record.get("summary_fields", {}).get("target_concepts") or target_meta["concepts"],
            evidence_list=record.get("evidence_list") or [],
            template_summary_text=str(record.get("summary_fields", {}).get("summary_text") or ""),
        )

    llm_cache_path = cache_dir / "llm_summary_cache.jsonl"
    llm_failure_path = cache_dir / "llm_summary_failures.jsonl"
    llm_cache = load_summary_cache(llm_cache_path)
    temp_path = contexts_path.with_suffix(".llm.tmp")
    workers = max(1, int(llm_summary_workers))
    chunk_size = max(workers, int(llm_summary_chunk_size))
    total_records = 0
    with contexts_path.open("r", encoding="utf-8", errors="replace") as probe_f:
        for line in probe_f:
            if line.strip():
                total_records += 1

    failures: List[Dict[str, Any]] = []

    def _rewrite_summary_cache() -> None:
        temp_cache_path = llm_cache_path.with_suffix(".rewrite.tmp")
        with temp_cache_path.open("w", encoding="utf-8") as f:
            for key, summary_text in llm_cache.items():
                f.write(json.dumps({"key": key, "summary_text": summary_text}, ensure_ascii=False) + "\n")
        temp_cache_path.replace(llm_cache_path)

    def _record_failure(record: Dict[str, Any], *, key: str, exc: Exception) -> None:
        failures.append(
            {
                "key": key,
                "user_id": str(record.get("user_id") or ""),
                "target_t": int(record.get("target_t") or 0),
                "target_pid": str(record.get("target_pid") or ""),
                "target_semantic_id": str(record.get("target_semantic_id") or ""),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    def _flush_chunk(chunk_records: List[Dict[str, Any]], dst: Any, executor: Optional[ThreadPoolExecutor]) -> None:
        if not chunk_records:
            return
        missing_items: List[Tuple[int, str, Dict[str, Any]]] = []
        new_cache_entries: List[Tuple[str, str]] = []
        for idx, record in enumerate(chunk_records):
            key = summary_cache_key(record["user_id"], int(record["target_t"]), record["target_pid"])
            llm_summary_text = llm_cache.get(key, "").strip()
            record["_llm_key"] = key
            if llm_summary_text:
                record["_llm_summary_text"] = llm_summary_text
            else:
                missing_items.append((idx, key, record))

        if missing_items:
            if executor is None or len(missing_items) == 1:
                for idx, key, record in missing_items:
                    try:
                        llm_summary_text = _summarize_record(record)
                    except Exception as exc:
                        _record_failure(record, key=key, exc=exc)
                        continue
                    record["_llm_summary_text"] = llm_summary_text
                    llm_cache[key] = llm_summary_text
                    new_cache_entries.append((key, llm_summary_text))
            else:
                future_map = {
                    executor.submit(_summarize_record, record): (idx, key, record)
                    for idx, key, record in missing_items
                }
                for future in as_completed(future_map):
                    idx, key, record = future_map[future]
                    try:
                        llm_summary_text = future.result()
                    except Exception as exc:
                        _record_failure(record, key=key, exc=exc)
                        continue
                    record["_llm_summary_text"] = llm_summary_text
                    llm_cache[key] = llm_summary_text
                    new_cache_entries.append((key, llm_summary_text))

        if new_cache_entries:
            append_summary_cache_entries(llm_cache_path, new_cache_entries)

        for record in chunk_records:
            key = str(record.pop("_llm_key"))
            llm_summary_text = str(record.pop("_llm_summary_text", "") or "")
            if not llm_summary_text:
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            record["summary_fields"]["llm_summary_text"] = llm_summary_text
            try:
                record["summary_fields"]["llm_summary_struct"] = parse_llm_summary_json(llm_summary_text)
            except Exception as exc:
                llm_cache.pop(key, None)
                _record_failure(record, key=key, exc=exc)
                record["summary_fields"].pop("llm_summary_text", None)
                record["summary_fields"].pop("llm_summary_struct", None)
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            record["llm_context_text"] = _build_llm_context_text(llm_summary_text, record.get("main_context_text", ""))
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    executor: Optional[ThreadPoolExecutor] = None
    if workers > 1:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="llm-summary")
    try:
        with contexts_path.open("r", encoding="utf-8", errors="replace") as src, temp_path.open("w", encoding="utf-8") as dst:
            chunk_records: List[Dict[str, Any]] = []
            progress = tqdm(total=total_records, desc="strict llm summaries")
            try:
                for line in src:
                    if not line.strip():
                        continue
                    chunk_records.append(json.loads(line))
                    if len(chunk_records) >= chunk_size:
                        processed = len(chunk_records)
                        _flush_chunk(chunk_records, dst, executor)
                        progress.update(processed)
                        chunk_records = []
                if chunk_records:
                    processed = len(chunk_records)
                    _flush_chunk(chunk_records, dst, executor)
                    progress.update(processed)
            finally:
                progress.close()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    contexts_path.unlink(missing_ok=True)
    temp_path.replace(contexts_path)
    _rewrite_summary_cache()
    if failures:
        with llm_failure_path.open("w", encoding="utf-8") as f:
            for item in failures:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        raise RuntimeError(
            f"LLM summary completed with {len(failures)} unresolved failures. "
            f"See {llm_failure_path}"
        )
    llm_failure_path.unlink(missing_ok=True)


def _reranker_score_cache_key(user_id: str, target_t: int, target_pid: str, hist_pid: str) -> str:
    return f"{user_id}\t{int(target_t)}\t{target_pid}\t{hist_pid}"


def _append_reranker_cache_entries(cache_path: Path, entries: List[Tuple[str, float]]) -> None:
    if not entries:
        return
    ensure_dir(cache_path.parent)
    with cache_path.open("a", encoding="utf-8") as f:
        for key, score in entries:
            f.write(json.dumps({"key": key, "payload": {"score": float(score)}}, ensure_ascii=False) + "\n")


def _apply_qwen_reranker(
    *,
    user_id: str,
    target_t: int,
    target_pid: str,
    target_meta: Dict[str, Any],
    stage1_candidates: List[Dict[str, Any]],
    problem_catalog_records: Dict[str, Dict[str, Any]],
    reranker: QwenReranker,
    rerank_cache: Dict[str, Dict[str, Any]],
    reranker_cache_path: Path,
    rerank_weight: float,
) -> None:
    if not stage1_candidates:
        return
    query = (
        f"目标题目: {str(target_meta.get('text') or '').strip()}\n"
        f"目标知识点: {'、'.join(str(item) for item in target_meta.get('concepts') or [])}\n"
        f"目标语义ID: {str(target_meta.get('semantic_id') or '').strip()}"
    )
    instruction = "Judge whether the historical problem is semantically useful evidence for predicting the target educational problem."
    docs: List[str] = []
    missing_indices: List[int] = []
    missing_keys: List[str] = []
    new_cache_entries: List[Tuple[str, float]] = []
    for idx, candidate in enumerate(stage1_candidates):
        hist_pid = str(candidate["problem_id"])
        cache_key = _reranker_score_cache_key(user_id, target_t, target_pid, hist_pid)
        payload = rerank_cache.get(cache_key)
        if payload is not None:
            candidate["raw_scores"]["rerank"] = float(payload["score"])
            continue
        hist_meta = problem_catalog_records[hist_pid]
        docs.append(
            f"历史题目: {str(hist_meta.get('text') or '').strip()}\n"
            f"知识点: {'、'.join(str(item) for item in hist_meta.get('concepts') or [])}\n"
            f"层级差: {candidate['level_diff']}\n"
            f"历史结果: {candidate['answer_result']}\n"
            f"知识重合: {'、'.join(candidate.get('knowledge_overlap_concepts') or []) or '无'}"
        )
        missing_indices.append(idx)
        missing_keys.append(cache_key)

    if docs:
        scores = reranker.score(query=query, docs=docs, instruction=instruction)
        if len(scores) != len(missing_indices):
            raise ValueError("Reranker returned inconsistent score count")
        for idx, cache_key, score in zip(missing_indices, missing_keys, scores):
            stage1_candidates[idx]["raw_scores"]["rerank"] = float(score)
            rerank_cache[cache_key] = {"score": float(score)}
            new_cache_entries.append((cache_key, float(score)))
    if new_cache_entries:
        _append_reranker_cache_entries(reranker_cache_path, new_cache_entries)

    for candidate in stage1_candidates:
        rerank_score = float(candidate["raw_scores"].get("rerank", 0.0))
        candidate["Ui"] = float(candidate["Ui"]) + float(rerank_weight) * rerank_score


def _evidence_record(candidate: Dict[str, Any], rank: int, catalog: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    pid = candidate["problem_id"]
    role_key = _role_from_candidate(candidate)
    overlap = candidate["knowledge_overlap_concepts"]
    knowledge_overlap = "、".join(sorted(overlap)) if overlap else "无"
    text = _question_text(catalog[pid]["text"])
    support_score = format_float(float(candidate["support_score"]), SUPPORT_SCORE_DECIMALS)
    evidence_text = (
        f"[证据#{rank}]关系角色：{ROLE_LABELS[role_key]}"
        f"知识点重合：{knowledge_overlap}"
        f"层级差：{candidate['level_diff']}"
        f"历史作答：{candidate['answer_result']}"
        f"支撑分数：{support_score}"
        f"题目内容：{text}"
    )
    return {
        "rank": rank,
        "problem_id": pid,
        "semantic_id": catalog[pid]["semantic_id"],
        "role": ROLE_LABELS[role_key],
        "knowledge_overlap": knowledge_overlap,
        "level_diff": int(candidate["level_diff"]),
        "answer_result": candidate["answer_result"],
        "support_score": support_score,
        "question_text": text,
        "activation": candidate["activation"],
        "raw_scores": candidate["raw_scores"],
        "Ui": float(candidate["Ui"]),
        "Ri": float(candidate["Ri"]),
        "history_pos": int(candidate["history_pos"]),
        "text": evidence_text,
    }


def run_stage34(
    *,
    problem_json: Path,
    student_json: Path,
    priors_dir: Path,
    contexts_dir: Path,
    reports_dir: Path,
    cache_dir: Path,
    preview_limit: int,
    dry_run: bool,
    smoke: bool,
    text_embed_model: str = TEXT_EMBED_MODEL_NAME,
    text_embed_batch_size: int = TEXT_EMBED_BATCH_SIZE,
    text_embed_max_length: int = TEXT_EMBED_MAX_LENGTH,
    text_rerank_model: str = TEXT_RERANK_MODEL_NAME,
    text_rerank_batch_size: int = TEXT_RERANK_BATCH_SIZE,
    use_qwen_reranker: bool = USE_QWEN_RERANKER,
    rerank_topk: int = RERANK_TOPK,
    rerank_weight: float = RERANK_WEIGHT,
    enable_llm_summary: bool = False,
    llm_base_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_timeout_sec: int = 120,
    llm_max_tokens: int = 160,
    llm_temperature: float = 0.1,
    llm_summary_workers: int = LLM_SUMMARY_WORKERS,
    llm_summary_chunk_size: int = LLM_SUMMARY_CHUNK_SIZE,
    reuse_existing_contexts: bool = False,
    context_shard_index: int = 0,
    context_num_shards: int = 1,
    merge_context_shards: bool = False,
) -> Stage34Result:
    ensure_dir(contexts_dir)
    ensure_dir(reports_dir)
    ensure_dir(cache_dir)
    context_num_shards = int(context_num_shards)
    context_shard_index = int(context_shard_index)
    if context_num_shards <= 0:
        raise ValueError("context_num_shards must be positive")
    if context_shard_index < 0 or context_shard_index >= context_num_shards:
        raise ValueError("context_shard_index must be in [0, context_num_shards)")
    if merge_context_shards and context_num_shards <= 1:
        raise ValueError("--merge_context_shards requires context_num_shards > 1")
    if merge_context_shards and reuse_existing_contexts:
        raise ValueError("--merge_context_shards cannot be combined with --reuse_existing_contexts")
    shard_mode = context_num_shards > 1 and not merge_context_shards
    if shard_mode and enable_llm_summary:
        raise ValueError("LLM summary must be run after shard merge; do not pass --enable_llm_summary in shard mode")

    problem_catalog_records: Dict[str, Dict[str, Any]] = {}
    with (priors_dir / "problem_catalog.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            problem_catalog_records[str(record["problem_id"])] = record

    shard_dir = ensure_dir(contexts_dir / "shards")
    report_shard_dir = ensure_dir(reports_dir / "shards")
    if shard_mode:
        suffix = _shard_suffix(context_shard_index, context_num_shards)
        contexts_path = shard_dir / f"contexts.{suffix}.jsonl"
        preview_path = report_shard_dir / f"context_preview.{suffix}.txt"
        manifest_path = shard_dir / f"stage34_manifest.{suffix}.json"
    else:
        contexts_path = contexts_dir / "contexts.jsonl"
        preview_path = reports_dir / "context_preview.txt"
        manifest_path = contexts_dir / "stage34_manifest.json"
    device = pick_device()
    reranker: Optional[QwenReranker] = None
    reranker_cache_path = (
        cache_dir / f"reranker_cache.{_shard_suffix(context_shard_index, context_num_shards)}.jsonl"
        if shard_mode
        else cache_dir / "reranker_cache.jsonl"
    )
    rerank_cache: Dict[str, Dict[str, Any]] = {}
    summarizer: Optional[OpenAICompatibleSummarizer] = None
    if use_qwen_reranker and not reuse_existing_contexts and not merge_context_shards:
        reranker = QwenReranker(
            model_name_or_path=str(text_rerank_model or TEXT_RERANK_MODEL_NAME),
            device=device,
            max_length=TEXT_RERANK_MAX_LENGTH,
            batch_size=int(text_rerank_batch_size),
        )
        rerank_cache = load_json_cache(reranker_cache_path)
    if enable_llm_summary:
        summarizer = OpenAICompatibleSummarizer(
            base_url=str(llm_base_url or ""),
            model=str(llm_model or ""),
            api_key=llm_api_key,
            timeout_sec=int(llm_timeout_sec),
            max_tokens=int(llm_max_tokens),
            temperature=float(llm_temperature),
        )

    if merge_context_shards:
        record_count = _merge_context_shards(
            contexts_dir=contexts_dir,
            reports_dir=reports_dir,
            shard_count=context_num_shards,
            preview_limit=preview_limit,
            output_contexts_path=contexts_path,
            output_preview_path=preview_path,
        )
        index_records, _ = _load_index_records_from_contexts(contexts_path)
        if summarizer is not None:
            _enrich_contexts_with_llm_summary(
                contexts_path=contexts_path,
                cache_dir=cache_dir,
                problem_catalog_records=problem_catalog_records,
                summarizer=summarizer,
                llm_summary_workers=int(llm_summary_workers),
                llm_summary_chunk_size=int(llm_summary_chunk_size),
            )
        embeddings_path: Optional[Path] = None
        if not dry_run:
            embeddings_path = _build_context_embeddings(
                contexts_path=contexts_path,
                cache_dir=cache_dir,
                device=device,
                text_embed_model=str(text_embed_model or TEXT_EMBED_MODEL_NAME),
                text_embed_batch_size=int(text_embed_batch_size),
                text_embed_max_length=int(text_embed_max_length),
            )
        result = Stage34Result(
            contexts_path=str(contexts_path),
            preview_path=str(preview_path),
            embeddings_path=str(embeddings_path) if embeddings_path is not None else None,
            manifest_path=str(manifest_path),
            record_count=record_count,
            text_embed_model=str(text_embed_model or TEXT_EMBED_MODEL_NAME),
            text_embed_batch_size=int(text_embed_batch_size),
            text_embed_max_length=int(text_embed_max_length),
            text_rerank_model=str(text_rerank_model or TEXT_RERANK_MODEL_NAME),
            text_rerank_batch_size=int(text_rerank_batch_size),
            use_qwen_reranker=bool(use_qwen_reranker),
            rerank_topk=int(rerank_topk),
            rerank_weight=float(rerank_weight),
            reranker_cache_path=None,
            llm_summary_workers=int(llm_summary_workers),
            llm_summary_chunk_size=int(llm_summary_chunk_size),
            context_shard_index=context_shard_index,
            context_num_shards=context_num_shards,
            merge_context_shards=True,
        )
        write_json(asdict(result), Path(result.manifest_path))
        return result

    if reuse_existing_contexts:
        if not contexts_path.exists():
            raise FileNotFoundError(f"--reuse_existing_contexts was set but {contexts_path} does not exist")
        index_records, record_count = _load_index_records_from_contexts(contexts_path)
    else:
        problem_records = load_problem_records(problem_json)
        student_sequences = load_student_sequences(student_json)

        if smoke:
            allowed_problem_ids = set(problem_catalog_records.keys())
            problem_records = [problem for problem in problem_records if problem.problem_id in allowed_problem_ids]
            student_sequences = student_sequences[:128]
        if context_num_shards > 1:
            student_sequences = [
                student for idx, student in enumerate(student_sequences) if idx % context_num_shards == context_shard_index
            ]

        with (priors_dir / "semantic_vectors.pkl").open("rb") as f:
            eqsem_map = pickle.load(f)
        with (priors_dir / "item_collaborative_embeddings.pkl").open("rb") as f:
            collab_map = pickle.load(f)
        graph_bundle = json.loads((priors_dir / "concept_graph_bundle.json").read_text(encoding="utf-8"))
        graph_accessor = GraphAccessor(graph_bundle)
        model = load_strict_prior_model(str(priors_dir / "model_state.pt"), map_location=device).to(device)
        model.eval()

        pid_lookup = list(problem_catalog_records.keys())
        pid_to_idx = {pid: idx for idx, pid in enumerate(pid_lookup)}
        eqsem = np.stack([eqsem_map[pid] for pid in pid_lookup], axis=0).astype(np.float32)
        eqsem_norm = _normalize_matrix(eqsem)
        collab_norm: Dict[int, np.ndarray] = {}
        for pid, vector in collab_map.items():
            if pid in pid_to_idx:
                value = np.asarray(vector, dtype=np.float32)
                norm = float(np.linalg.norm(value))
                collab_norm[pid_to_idx[pid]] = (value / norm).astype(np.float32) if norm > 0 else value

        preview_lines: List[str] = []
        index_records = []
        with contexts_path.open("w", encoding="utf-8") as out_f:
            record_count = 0
            for student in tqdm(student_sequences, desc="strict contexts"):
                seq_pids = [str(log.get("problem_id") or "") for log in student.seq if str(log.get("problem_id") or "") in pid_to_idx]
                seq_results = [int(log.get("is_correct") or 0) for log in student.seq if str(log.get("problem_id") or "") in pid_to_idx]
                seq_levels = [
                    int(problem_catalog_records[str(log.get("problem_id"))]["cognitive_dimension"])
                    for log in student.seq
                    if str(log.get("problem_id") or "") in pid_to_idx
                ]
                seq_problem_indices = [pid_to_idx[pid] for pid in seq_pids]
                if len(seq_problem_indices) < 2:
                    continue
                seq_cache = _build_sequence_cache(
                    seq_problem_indices,
                    seq_levels,
                    pid_lookup,
                    eqsem_norm,
                    collab_norm,
                    graph_accessor,
                    problem_catalog_records,
                )

                for target_t in range(1, len(seq_problem_indices)):
                    target_pid = pid_lookup[seq_problem_indices[target_t]]
                    _z, d_vec, sdyn = _compute_dynamic_prior(
                        seq_problem_indices,
                        seq_results,
                        seq_levels,
                        target_t,
                        eqsem,
                        eqsem_norm,
                        model,
                        device,
                    )
                    hist_problem_indices = seq_problem_indices[:target_t]
                    hist_diag_probs = _history_diag_probs(hist_problem_indices, eqsem, d_vec, model, device)
                    dtc_values = _dtc_values(seq_problem_indices, target_t, seq_cache["eq_cos"])

                    candidates: List[Dict[str, Any]] = []
                    for hist_pos in range(0, target_t):
                        candidate = _candidate_scores(
                            hist_pos=hist_pos,
                            current_t=target_t,
                            seq_problem_indices=seq_problem_indices,
                            seq_results=seq_results,
                            seq_levels=seq_levels,
                            eqsem=eqsem,
                            problem_catalog=problem_catalog_records,
                            pid_lookup=pid_lookup,
                            p_diag=float(hist_diag_probs[hist_pos]),
                            dtc_value=float(dtc_values[hist_pos]),
                            seq_cache=seq_cache,
                        )
                        candidates.append(candidate)

                    candidates.sort(key=lambda item: (item["Ri"], item["history_pos"]), reverse=True)
                    stage1_candidates = candidates[: min(int(rerank_topk), K1_DEFAULT, len(candidates))]
                    if reranker is not None and stage1_candidates:
                        _apply_qwen_reranker(
                            user_id=student.user_id,
                            target_t=target_t,
                            target_pid=target_pid,
                            target_meta=problem_catalog_records[target_pid],
                            stage1_candidates=stage1_candidates,
                            problem_catalog_records=problem_catalog_records,
                            reranker=reranker,
                            rerank_cache=rerank_cache,
                            reranker_cache_path=reranker_cache_path,
                            rerank_weight=float(rerank_weight),
                        )

                    selected: List[Dict[str, Any]] = []
                    remaining = list(stage1_candidates)
                    if remaining:
                        remaining.sort(key=lambda item: (item["Ui"], item["history_pos"]), reverse=True)
                        first = remaining.pop(0)
                        first["support_score"] = float(first["Ui"])
                        selected.append(first)

                    while remaining and len(selected) < K2_DEFAULT:
                        best_item = None
                        best_key = None
                        for candidate in remaining:
                            cov_gain = _coverage_gain(candidate, selected, target_pid, graph_accessor, problem_catalog_records)
                            red = _redundancy(candidate, selected, eqsem_norm, pid_to_idx, problem_catalog_records)
                            f_score = float(candidate["Ui"]) + LAMBDA_COV * cov_gain - LAMBDA_RED * red
                            sort_key = (f_score, candidate["Ui"], candidate["history_pos"])
                            if best_key is None or sort_key > best_key:
                                best_key = sort_key
                                best_item = candidate
                                best_item["support_score"] = float(f_score)
                        assert best_item is not None
                        selected.append(best_item)
                        remaining.remove(best_item)

                    evidence_list = [_evidence_record(candidate, rank, problem_catalog_records) for rank, candidate in enumerate(selected, start=1)]
                    summary_fields = _summary_fields(target_pid, selected, problem_catalog_records, sdyn)
                    main_context_text = "\n".join(evidence["text"] for evidence in evidence_list).strip()
                    template_context_text = (summary_fields["summary_text"] + ("\n" + main_context_text if main_context_text else "")).strip()

                    record = {
                        "user_id": student.user_id,
                        "target_t": target_t,
                        "target_pid": target_pid,
                        "target_semantic_id": problem_catalog_records[target_pid]["semantic_id"],
                        "stage1_candidate_count": len(stage1_candidates),
                        "selected_count": len(evidence_list),
                        "main_context_text": main_context_text,
                        "template_context_text": template_context_text,
                        "llm_context_text": "",
                        "summary_fields": summary_fields,
                        "evidence_list": evidence_list,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    index_records.append(
                        {
                            "user_id": student.user_id,
                            "target_t": target_t,
                            "target_pid": target_pid,
                        }
                    )
                    if len(preview_lines) < preview_limit:
                        preview_lines.append(
                            f"--- {student.user_id} @ t={target_t} / {target_pid} ---\n"
                            f"[MAIN]\n{main_context_text}\n\n[TEMPLATE]\n{template_context_text}\n"
                        )
                    record_count += 1

        atomic_save_text("\n\n".join(preview_lines), preview_path)

    if (not shard_mode) and summarizer is not None:
        _enrich_contexts_with_llm_summary(
            contexts_path=contexts_path,
            cache_dir=cache_dir,
            problem_catalog_records=problem_catalog_records,
            summarizer=summarizer,
            llm_summary_workers=int(llm_summary_workers),
            llm_summary_chunk_size=int(llm_summary_chunk_size),
        )

    embeddings_path: Optional[Path] = None
    if (not shard_mode) and (not dry_run):
        embeddings_path = _build_context_embeddings(
            contexts_path=contexts_path,
            cache_dir=cache_dir,
            device=device,
            text_embed_model=str(text_embed_model or TEXT_EMBED_MODEL_NAME),
            text_embed_batch_size=int(text_embed_batch_size),
            text_embed_max_length=int(text_embed_max_length),
        )

    result = Stage34Result(
        contexts_path=str(contexts_path),
        preview_path=str(preview_path),
        embeddings_path=str(embeddings_path) if embeddings_path is not None else None,
        manifest_path=str(manifest_path),
        record_count=record_count,
        text_embed_model=str(text_embed_model or TEXT_EMBED_MODEL_NAME),
        text_embed_max_length=int(text_embed_max_length),
        text_rerank_model=str(text_rerank_model or TEXT_RERANK_MODEL_NAME),
        use_qwen_reranker=bool(use_qwen_reranker),
        rerank_topk=int(rerank_topk),
        rerank_weight=float(rerank_weight),
        reranker_cache_path=str(reranker_cache_path.resolve()) if use_qwen_reranker else None,
        text_embed_batch_size=int(text_embed_batch_size),
        text_rerank_batch_size=int(text_rerank_batch_size),
        llm_summary_workers=int(llm_summary_workers),
        llm_summary_chunk_size=int(llm_summary_chunk_size),
        context_shard_index=context_shard_index,
        context_num_shards=context_num_shards,
        merge_context_shards=False,
    )
    write_json(asdict(result), Path(result.manifest_path))
    return result
