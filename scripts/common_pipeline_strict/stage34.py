"""Stage 3.4：为每个预测目标构造认知 Context。

本阶段读取 Stage 3.2 的题目语义向量、协同信号、知识图和动态先验模型，
针对学生序列中的每个目标交互执行以下流程：
1. 计算学生在当前目标时刻的动态认知先验；
2. 为历史交互计算知识重合、层级关系、语义、图结构、协同和时间分数；
3. 第一阶段召回高相关候选，可选用 Qwen reranker 再打分；
4. 第二阶段加入覆盖收益和冗余惩罚，选择少量互补证据；
5. 生成 main/template/LLM Context，并编码成知识追踪模型可读取的向量。

文件还包含分片生成、缓存预热、LLM 总结和断点续跑等工程支持逻辑。
"""

from __future__ import annotations

import json
import math
import pickle
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .constants import (ALPHA_TIME, BETA_NEG, BETA_POS, COVERAGE_WEIGHTS,
                        DELTA_GRAPH, EPS, EXPLICIT_MATCH_WEIGHTS, GAMMA_HIGH,
                        GAMMA_PRE, HISTORY_WINDOW, K1_DEFAULT, K2_DEFAULT,
                        LAMBDA_COV, LAMBDA_RED, LLM_SUMMARY_CHUNK_SIZE,
                        LLM_SUMMARY_WORKERS, QUESTION_TEXT_ELLIPSIS,
                        QUESTION_TEXT_LIMIT, REDUNDANCY_WEIGHTS, RERANK_TOPK,
                        RERANK_WEIGHT, RHO, ROLE_LABELS, ROLE_ORDER,
                        ROLE_PRIORITY, ROLE_THRESHOLDS, SUMMARY_TEMPLATE,
                        SUPPORT_SCORE_DECIMALS, TEXT_EMBED_BATCH_SIZE,
                        TEXT_EMBED_MAX_LENGTH, TEXT_EMBED_MODEL_NAME,
                        TEXT_RERANK_BATCH_SIZE, TEXT_RERANK_MAX_LENGTH,
                        TEXT_RERANK_MODEL_NAME, USE_QWEN_RERANKER,
                        WEIGHT_STAGE1, WEIGHT_STAGE2)
from .io_utils import (atomic_save_text, ensure_dir, format_float,
                       load_problem_records, load_student_sequences,
                       pick_device, write_json)
from .llm_summary_signatures import (LLM_SUMMARY_SIGNATURE_PREFIX,
                                     llm_summary_signature_key,
                                     summarize_llm_record,
                                     summarize_llm_records_batch)
from .llm_utils import (OpenAICompatibleSummarizer,
                        append_summary_cache_entries, load_json_cache,
                        load_summary_cache, parse_llm_summary_json,
                        summary_cache_key)
from .models import load_strict_prior_model
from .retrieval_models import QwenEmbeddingEncoder, QwenReranker

JSON_COLLAB_NEIGHBOR_WEIGHT = 0.35
COLLAB_ISOLATED_GATE_WEIGHT = 0.35
COLLAB_PEER_GATE_THRESHOLD = 0.65


@dataclass
class Stage34Result:
    """Stage 3.4 输出文件、运行模式和关键参数的清单。"""

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
    text_rerank_max_length: int
    use_qwen_reranker: bool
    rerank_topk: int
    rerank_weight: float
    rerank_cache_scope: str
    reranker_cache_path: Optional[str]
    llm_summary_workers: int
    llm_summary_chunk_size: int
    llm_summary_batch_size: int
    context_shard_index: int
    context_num_shards: int
    merge_context_shards: bool
    llm_summary_compact_prompt: bool = False
    mode: str = "contexts"
    warmup_stats: Optional[Dict[str, Any]] = None


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
    """把 Stage 3.2 的知识图整理为适合候选打分的快速查询结构。"""

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
        """返回历史知识点与目标知识点之间的前置/邻接结构奖励。"""
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
    """计算从某条历史到目标之间累积的语义变化距离。"""
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
    """向量化计算目标之前所有历史位置的语义变化距离。"""
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
    collab_neighbors: Dict[str, List[str]],
    graph_accessor: GraphAccessor,
    problem_catalog: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """预计算一个学生序列内反复使用的两两相似度和图结构特征。

    缓存包括题目语义余弦、协同相似度、知识点 Jaccard、知识重合列表和图奖励，
    避免为序列中的每个目标重复进行相同计算。
    """
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
    for target_pos, target_pid in enumerate(seq_pids):
        ranked_neighbors = collab_neighbors.get(target_pid, []) or []
        if not ranked_neighbors:
            continue
        denom = float(max(len(ranked_neighbors), 1))
        neighbor_scores = {
            str(pid): JSON_COLLAB_NEIGHBOR_WEIGHT * max(0.0, 1.0 - (float(rank) / denom))
            for rank, pid in enumerate(ranked_neighbors)
        }
        for hist_pos in range(target_pos):
            hist_pid = seq_pids[hist_pos]
            if hist_pid in neighbor_scores:
                collab_cos[hist_pos, target_pos] = max(
                    float(collab_cos[hist_pos, target_pos]),
                    float(neighbor_scores[hist_pid]),
                )

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
    """根据目标之前的历史计算学生动态状态及目标答对先验概率。

    历史权重同时考虑题目语义相似度、认知层级差和时间距离；加权历史 ``z``
    经 Stage 3.2 训练的 dynamic 网络得到动态状态 ``d_vec``。
    """
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
    """在当前动态学生状态下，估计每道历史题的诊断答对概率。"""
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
    """计算一条历史交互对当前目标的多路证据分数。

    主要分量：
    - ``Ki``：知识点重合；
    - ``pre/peer/high``：低阶前置、同阶迁移、高阶反馈关系；
    - ``semantic``：无显式知识关系时的语义同类关系；
    - ``graph/collab``：知识图和协同信号；
    - ``Ti``：随语义变化距离衰减的时间权重。

    ``Ri`` 用于第一阶段召回，``Ui`` 用于第二阶段选择。
    """
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
    ssemantic = 0.0
    shigh = 0.0
    peer_similarity = 0.0
    if delta_l > 0:
        spre = ki * mi * math.exp(-GAMMA_PRE * abs(delta_l))
    elif delta_l == 0:
        peer_similarity = float(seq_eq_cos[hist_pos, current_t])
    else:
        shigh = ki * ((BETA_POS * float(seq_results[hist_pos])) - (BETA_NEG * (1.0 - float(seq_results[hist_pos])))) * math.exp(
            -GAMMA_HIGH * abs(delta_l)
        )

    graph_bonus = float(seq_cache["graph_bonus"][hist_pos, current_t])
    graw = ki + graph_bonus
    gcomp = max(0.0, graw - ki)
    if delta_l == 0:
        if ki > 0.0 or gcomp > 0.0:
            speer = peer_similarity
        else:
            ssemantic = peer_similarity
    explicit_match = _clip01(
        EXPLICIT_MATCH_WEIGHTS["K"] * ki
        + EXPLICIT_MATCH_WEIGHTS["L"] * math.exp(-1.0 * abs(delta_l))
        + EXPLICIT_MATCH_WEIGHTS["G"] * gcomp
    )
    collab_sim = float(seq_cache["collab_cos"][hist_pos, current_t])
    scollab = (1.0 - explicit_match) * collab_sim
    if ki <= 0.0 and gcomp <= 0.0 and speer < COLLAB_PEER_GATE_THRESHOLD:
        scollab *= COLLAB_ISOLATED_GATE_WEIGHT

    ti = math.exp(-ALPHA_TIME * dtc_value)

    bi = (
        WEIGHT_STAGE1["K"] * ki
        + WEIGHT_STAGE1["pre"] * spre
        + WEIGHT_STAGE1["peer"] * speer
        + WEIGHT_STAGE1.get("semantic", 1.0) * ssemantic
        + WEIGHT_STAGE1["high"] * shigh
        + WEIGHT_STAGE1["graph"] * gcomp
        + WEIGHT_STAGE1["collab"] * scollab
    )
    ri = ti * bi
    ui = (
        WEIGHT_STAGE2["K"] * ki
        + WEIGHT_STAGE2["pre"] * spre
        + WEIGHT_STAGE2["peer"] * speer
        + WEIGHT_STAGE2.get("semantic", 1.0) * ssemantic
        + WEIGHT_STAGE2["high"] * shigh
        + WEIGHT_STAGE2["graph"] * gcomp
        + WEIGHT_STAGE2["collab"] * scollab
    )

    activation = {
        "pre": int(spre > ROLE_THRESHOLDS["pre"]),
        "peer": int(speer > ROLE_THRESHOLDS["peer"]),
        "semantic": int(ssemantic > ROLE_THRESHOLDS.get("semantic", ROLE_THRESHOLDS["peer"])),
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
            "semantic": float(ssemantic),
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
    """衡量候选与已选证据的重复程度，避免最终证据表达相同信息。"""
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
    """衡量候选新增的关系角色、知识点和图邻居覆盖量。"""
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
    """汇总最终证据，确定最主要的认知关系角色。"""
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
    """根据已选证据和动态先验生成无需 LLM 的模板化认知摘要字段。"""
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
    """将结构化 LLM 总结转为下游模型使用的自然语言 Context。"""
    parsed = parse_llm_summary_json(str(llm_summary_text or ""))
    parts: List[str] = []
    diagnosis = str(parsed.get("diagnosis") or "").strip()
    if diagnosis:
        parts.append(diagnosis)
    mastered = [str(item.get("concept") or "").strip() for item in parsed.get("mastered_concepts", []) if str(item.get("concept") or "").strip()]
    weak = [str(item.get("concept") or "").strip() for item in parsed.get("weak_concepts", []) if str(item.get("concept") or "").strip()]
    if mastered:
        parts.append("稳定掌握点：" + "、".join(mastered))
    if weak:
        parts.append("持续薄弱点：" + "、".join(weak))
    parts.append(f"迁移状态：{parsed['transfer_state']}")
    parts.append(f"风险等级：{parsed['risk_level']}")
    parts.append(f"证据质量：{parsed['evidence_quality']}")
    summary = " ".join(part.strip() for part in parts if part.strip())
    main = str(main_context_text or "").strip()
    if summary and main:
        return summary + "\n" + main
    return summary or main


def _build_llm_struct_texts(record: Dict[str, Any]) -> Dict[str, str]:
    """拆分 LLM 结构化总结，分别编码掌握点、薄弱点、诊断和辅助描述。"""
    llm_summary_text = str(record.get("summary_fields", {}).get("llm_summary_text") or "").strip()
    parsed = parse_llm_summary_json(llm_summary_text)
    stable_text = "、".join(str(item.get("concept") or "").strip() for item in parsed["mastered_concepts"] if str(item.get("concept") or "").strip())
    weak_text = "、".join(str(item.get("concept") or "").strip() for item in parsed["weak_concepts"] if str(item.get("concept") or "").strip())
    summary_text = str(parsed["diagnosis"])
    aux_text = (
        f"迁移状态 {parsed['transfer_state']} "
        f"风险等级 {parsed['risk_level']} "
        f"证据质量 {parsed['evidence_quality']}"
    )
    return {
        "stable_text": stable_text,
        "weak_text": weak_text,
        "summary_text": summary_text,
        "aux_text": aux_text,
    }


def _build_llm_struct_feature_vector(record: Dict[str, Any]) -> np.ndarray:
    """把 LLM 总结中的计数、风险和证据质量转为固定长度数值特征。"""
    llm_summary_text = str(record.get("summary_fields", {}).get("llm_summary_text") or "").strip()
    parsed = parse_llm_summary_json(llm_summary_text)
    mastered_points = [str(item.get("concept") or "").strip() for item in parsed["mastered_concepts"] if str(item.get("concept") or "").strip()]
    weak_points = [str(item.get("concept") or "").strip() for item in parsed["weak_concepts"] if str(item.get("concept") or "").strip()]
    diagnosis = str(parsed["diagnosis"]).strip()
    levels = ("低", "中", "高")
    risk_onehot = [1.0 if parsed["risk_level"] == level else 0.0 for level in levels]
    quality_onehot = [1.0 if parsed["evidence_quality"] == level else 0.0 for level in levels]
    return np.asarray(
        [
            float(len(mastered_points)) / 3.0,
            float(len(weak_points)) / 3.0,
            1.0 if mastered_points else 0.0,
            1.0 if weak_points else 0.0,
            min(float(len(diagnosis)), 80.0) / 80.0,
            *risk_onehot,
            *quality_onehot,
        ],
        dtype=np.float32,
    )


def _unique_texts_with_inverse(texts: Sequence[str]) -> Tuple[List[str], np.ndarray]:
    """对重复文本去重，并记录恢复原顺序所需的反向索引。"""
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
    """去重并可断点续跑地编码文本，最后恢复到原记录顺序。"""
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


def _context_record_count_hint(manifest_path: Path, contexts_path: Path) -> Optional[int]:
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            record_count = int(manifest.get("record_count") or 0)
            if record_count > 0:
                return record_count
        except Exception:
            pass
    return None


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
    """按分片编号顺序合并 Context JSONL，并重新生成预览文件。"""
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
    """把 Context 文本与 LLM 结构字段编码为训练阶段读取的向量文件。

    输出文件中的 ``index`` 与各向量矩阵严格按行对齐。文本会先去重并使用分块
    缓存编码，以便大规模任务断点续跑。
    """
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
    llm_summary_batch_size: int,
    total_records_hint: Optional[int] = None,
    output_contexts_path: Optional[Path] = None,
    line_start: int = 0,
    line_end: Optional[int] = None,
    failure_path: Optional[Path] = None,
    rewrite_summary_cache_at_end: bool = True,
) -> None:
    """为已有 Context 追加严格校验的 LLM 结构化认知总结。

    函数按块读取 JSONL，优先复用 prompt-signature 缓存；缺失项支持批量请求、
    逐条回退、多线程和分片范围处理。任意无法解决的失败都会写入失败文件并报错。
    """
    ensure_dir(cache_dir)
    cache_prefix = LLM_SUMMARY_SIGNATURE_PREFIX

    def _summarize_record(record: Dict[str, Any]) -> str:
        return summarize_llm_record(
            record=record,
            problem_catalog_records=problem_catalog_records,
            summarizer=summarizer,
        )

    llm_cache_path = cache_dir / "llm_summary_cache.jsonl"
    llm_failure_path = failure_path or (cache_dir / "llm_summary_failures.jsonl")
    raw_llm_cache = load_summary_cache(llm_cache_path)
    llm_cache = {
        key: value
        for key, value in raw_llm_cache.items()
        if str(key).startswith(cache_prefix)
    }
    # 分片范围使用 JSONL 物理行号，并采用左闭右开区间 [start_line, end_line)，
    # 因此多个独立分片可以无重叠、无缺口地合并。
    start_line = max(0, int(line_start or 0))
    end_line = int(line_end) if line_end is not None else None
    if end_line is not None and end_line < start_line:
        raise ValueError(f"Invalid LLM summary line range: start={start_line} end={end_line}")
    final_output_path = output_contexts_path
    if final_output_path is None:
        temp_path = contexts_path.with_suffix(".llm.tmp")
    else:
        final_output_path = Path(final_output_path)
        ensure_dir(final_output_path.parent)
        temp_path = final_output_path.with_suffix(final_output_path.suffix + ".tmp")
    workers = max(1, int(llm_summary_workers))
    chunk_size = max(workers, int(llm_summary_chunk_size))
    batch_size = max(1, int(llm_summary_batch_size))
    total_records = int(total_records_hint or 0)
    if end_line is not None:
        total_records = max(0, end_line - start_line)
    if total_records <= 0:
        with contexts_path.open("r", encoding="utf-8", errors="replace") as probe_f:
            for line_no, line in enumerate(probe_f):
                if line_no < start_line:
                    continue
                if end_line is not None and line_no >= end_line:
                    break
                if line.strip():
                    total_records += 1
    print(
        f"[stage34] llm summary start total_records={total_records} "
        f"workers={workers} chunk_size={chunk_size} batch_size={batch_size} "
        f"cache_scope=prompt-signature compact_prompt={bool(getattr(summarizer, 'compact_prompt', False))} "
        f"line_start={start_line} line_end={end_line if end_line is not None else 'EOF'} "
        f"semantic_cache_entries={len(llm_cache)} raw_cache_entries={len(raw_llm_cache)}",
        flush=True,
    )

    failures: List[Dict[str, Any]] = []
    runtime_stats: Dict[str, int] = {
        "records": 0,
        "signature_cache_hit_records": 0,
        "legacy_cache_hit_records": 0,
        "existing_context_hit_records": 0,
        "llm_request_count": 0,
        "llm_summary_records_requested": 0,
        "llm_batch_fallback_count": 0,
        "same_chunk_duplicate_records": 0,
        "new_signature_cache_entries": 0,
        "failure_count": 0,
    }

    def _rewrite_summary_cache() -> None:
        temp_cache_path = llm_cache_path.with_suffix(".rewrite.tmp")
        with temp_cache_path.open("w", encoding="utf-8") as f:
            for key, summary_text in llm_cache.items():
                if not str(key).startswith(cache_prefix):
                    continue
                f.write(json.dumps({"key": key, "summary_text": summary_text}, ensure_ascii=False) + "\n")
        temp_cache_path.replace(llm_cache_path)

    def _record_failure(record: Dict[str, Any], *, key: str, exc: Exception) -> None:
        runtime_stats["failure_count"] += 1
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
            runtime_stats["records"] += 1
            key = summary_cache_key(record["user_id"], int(record["target_t"]), record["target_pid"])
            signature_key = llm_summary_signature_key(
                record=record,
                problem_catalog_records=problem_catalog_records,
                summarizer=summarizer,
            )
            signature_summary_text = str(llm_cache.get(signature_key, "") or "").strip()
            legacy_summary_text = str(raw_llm_cache.get(key, "") or "").strip()
            existing_summary_text = str(record.get("summary_fields", {}).get("llm_summary_text") or "").strip()
            existing_llm_context_text = str(record.get("llm_context_text") or "").strip()
            # 缓存优先级：精确 prompt 签名 > 旧版交互键 > Context 中已有总结。
            # 后两者特异性较低，仅作为兼容和恢复手段。
            cache_candidates = [
                ("signature", signature_summary_text),
                ("legacy", legacy_summary_text),
                ("context", existing_summary_text if existing_llm_context_text else ""),
            ]
            record["_llm_key"] = key
            record["_llm_signature_key"] = signature_key
            llm_summary_text = ""
            cache_source = ""
            for candidate_source, candidate_text in cache_candidates:
                candidate_text = str(candidate_text or "").strip()
                if not candidate_text:
                    continue
                try:
                    parse_llm_summary_json(candidate_text)
                except Exception:
                    if candidate_source == "signature":
                        llm_cache.pop(signature_key, None)
                    elif candidate_source == "legacy":
                        raw_llm_cache.pop(key, None)
                    continue
                llm_summary_text = candidate_text
                cache_source = candidate_source
                break
            if llm_summary_text:
                record["_llm_summary_text"] = llm_summary_text
                if cache_source == "signature":
                    runtime_stats["signature_cache_hit_records"] += 1
                elif cache_source == "legacy":
                    runtime_stats["legacy_cache_hit_records"] += 1
                elif cache_source == "context":
                    runtime_stats["existing_context_hit_records"] += 1
                if signature_key and signature_key not in llm_cache:
                    llm_cache[signature_key] = llm_summary_text
                    new_cache_entries.append((signature_key, llm_summary_text))
            else:
                missing_items.append((idx, key, record))

        if missing_items:
            # 相同 prompt 签名对应完全相同的请求；块内只请求一次，再复用给重复记录。
            signature_owner: Dict[str, Tuple[int, str, Dict[str, Any]]] = {}
            duplicate_items: List[Tuple[int, str, str, Dict[str, Any]]] = []
            for idx, key, record in missing_items:
                signature_key = str(record.get("_llm_signature_key") or "")
                if signature_key in signature_owner:
                    duplicate_items.append((idx, key, signature_key, record))
                else:
                    signature_owner[signature_key] = (idx, key, record)
            unique_missing_items = list(signature_owner.values())
            runtime_stats["llm_summary_records_requested"] += len(unique_missing_items)
            runtime_stats["same_chunk_duplicate_records"] += len(duplicate_items)

            def _summarize_item_batch(
                batch_items: List[Tuple[int, str, Dict[str, Any]]]
            ) -> Tuple[List[Tuple[int, str, Dict[str, Any], str]], int, int, List[Tuple[str, Dict[str, Any], Exception]]]:
                if not batch_items:
                    return [], 0, 0, []
                if batch_size <= 1 or len(batch_items) == 1:
                    results: List[Tuple[int, str, Dict[str, Any], str]] = []
                    failures_local: List[Tuple[str, Dict[str, Any], Exception]] = []
                    for idx, key, record in batch_items:
                        try:
                            results.append((idx, key, record, _summarize_record(record)))
                        except Exception as exc:
                            failures_local.append((key, record, exc))
                    return results, len(batch_items), 0, failures_local

                record_items = [
                    (f"case_{pos}", record)
                    for pos, (_idx, _key, record) in enumerate(batch_items)
                ]
                try:
                    batch_summaries = summarize_llm_records_batch(
                        record_items=record_items,
                        problem_catalog_records=problem_catalog_records,
                        summarizer=summarizer,
                    )
                    results = []
                    for pos, (idx, key, record) in enumerate(batch_items):
                        results.append((idx, key, record, batch_summaries[f"case_{pos}"]))
                    return results, 1, 0, []
                except Exception:
                    # 批量响应格式错误时逐条重试，避免一个坏样本导致整批结果丢失。
                    results = []
                    failures_local = []
                    for idx, key, record in batch_items:
                        try:
                            results.append((idx, key, record, _summarize_record(record)))
                        except Exception as exc:
                            failures_local.append((key, record, exc))
                    return results, 1 + len(batch_items), 1, failures_local

            item_batches = [
                unique_missing_items[start : start + batch_size]
                for start in range(0, len(unique_missing_items), batch_size)
            ]
            if executor is None or len(item_batches) == 1:
                for item_batch in item_batches:
                    results, request_count, fallback_count, batch_failures = _summarize_item_batch(item_batch)
                    runtime_stats["llm_request_count"] += request_count
                    runtime_stats["llm_batch_fallback_count"] += fallback_count
                    for key, record, exc in batch_failures:
                        _record_failure(record, key=key, exc=exc)
                    for idx, key, record, llm_summary_text in results:
                        signature_key = str(record.get("_llm_signature_key") or "")
                        record["_llm_summary_text"] = llm_summary_text
                        if signature_key and signature_key not in llm_cache:
                            llm_cache[signature_key] = llm_summary_text
                            new_cache_entries.append((signature_key, llm_summary_text))
            else:
                future_map = {
                    executor.submit(_summarize_item_batch, item_batch): item_batch
                    for item_batch in item_batches
                }
                for future in as_completed(future_map):
                    try:
                        results, request_count, fallback_count, batch_failures = future.result()
                    except Exception as exc:
                        for _idx, key, record in future_map[future]:
                            _record_failure(record, key=key, exc=exc)
                        continue
                    runtime_stats["llm_request_count"] += request_count
                    runtime_stats["llm_batch_fallback_count"] += fallback_count
                    for key, record, exc in batch_failures:
                        _record_failure(record, key=key, exc=exc)
                    for idx, key, record, llm_summary_text in results:
                        signature_key = str(record.get("_llm_signature_key") or "")
                        record["_llm_summary_text"] = llm_summary_text
                        if signature_key and signature_key not in llm_cache:
                            llm_cache[signature_key] = llm_summary_text
                            new_cache_entries.append((signature_key, llm_summary_text))

            for _idx, key, signature_key, record in duplicate_items:
                llm_summary_text = str(llm_cache.get(signature_key, "") or "").strip()
                if not llm_summary_text:
                    continue
                record["_llm_summary_text"] = llm_summary_text

        if new_cache_entries:
            runtime_stats["new_signature_cache_entries"] += len(new_cache_entries)
            append_summary_cache_entries(llm_cache_path, new_cache_entries)

        for record in chunk_records:
            key = str(record.pop("_llm_key"))
            signature_key = str(record.pop("_llm_signature_key", "") or "")
            llm_summary_text = str(record.pop("_llm_summary_text", "") or "")
            if not llm_summary_text:
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            record["summary_fields"]["llm_summary_text"] = llm_summary_text
            try:
                record["summary_fields"]["llm_summary_struct"] = parse_llm_summary_json(llm_summary_text)
            except Exception as exc:
                if signature_key:
                    llm_cache.pop(signature_key, None)
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
                for line_no, line in enumerate(src):
                    if line_no < start_line:
                        continue
                    if end_line is not None and line_no >= end_line:
                        break
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

    # 原地模式仅在处理成功后替换源文件；分片模式写入独立文件，不修改共享源 JSONL。
    if final_output_path is None:
        contexts_path.unlink(missing_ok=True)
        temp_path.replace(contexts_path)
    else:
        temp_path.replace(final_output_path)
    # 并发分片会共同追加缓存；只有单一所有者可以压缩重写，否则可能覆盖其他分片结果。
    if rewrite_summary_cache_at_end:
        _rewrite_summary_cache()
    print(
        "[stage34] llm summary runtime stats "
        + json.dumps(runtime_stats, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    if failures:
        with llm_failure_path.open("w", encoding="utf-8") as f:
            for item in failures:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        raise RuntimeError(
            f"LLM summary completed with {len(failures)} unresolved failures. "
            f"See {llm_failure_path}"
        )
    llm_failure_path.unlink(missing_ok=True)


def _reranker_score_cache_key(
    *,
    cache_scope: str,
    user_id: str,
    target_t: int,
    target_pid: str,
    hist_pid: str,
    answer_result: str = "",
) -> str:
    """根据缓存作用域构造 reranker 分数键。"""
    scope = str(cache_scope or "pair_result").strip().lower()
    if scope == "pair":
        return f"pair\t{target_pid}\t{hist_pid}"
    if scope == "pair_result":
        return f"pair_result\t{target_pid}\t{hist_pid}\t{str(answer_result or '').strip()}"
    if scope == "interaction":
        return f"interaction\t{user_id}\t{int(target_t)}\t{target_pid}\t{hist_pid}"
    raise ValueError(f"Unsupported rerank_cache_scope: {cache_scope}")


def _safe_cache_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in str(value or ""))
    token = "_".join(part for part in token.split("_") if part)
    return token[-80:] or "default"


def _reranker_cache_stem(*, model_name_or_path: str, cache_scope: str, max_length: int) -> str:
    model_token = _safe_cache_token(Path(str(model_name_or_path or "reranker")).name)
    return f"reranker_cache.{str(cache_scope)}.{model_token}.max{int(max_length)}"


def _reranker_query_text(target_meta: Dict[str, Any]) -> str:
    return (
        f"目标题目: {str(target_meta.get('text') or '').strip()}\n"
        f"目标知识点: {'、'.join(str(item) for item in target_meta.get('concepts') or [])}\n"
        f"目标语义ID: {str(target_meta.get('semantic_id') or '').strip()}"
    )


def _reranker_doc_text(
    *,
    hist_meta: Dict[str, Any],
    level_diff: int,
    answer_result: str,
    overlap_concepts: Sequence[str],
    include_answer_result: bool,
) -> str:
    doc_parts = [
        f"历史题目: {str(hist_meta.get('text') or '').strip()}",
        f"知识点: {'、'.join(str(item) for item in hist_meta.get('concepts') or [])}",
        f"层级差: {int(level_diff)}",
        f"知识重合: {'、'.join(str(item) for item in overlap_concepts) or '无'}",
    ]
    if include_answer_result:
        doc_parts.append(f"历史结果: {str(answer_result or '').strip()}")
    return "\n".join(doc_parts)


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
    rerank_cache_scope: str,
) -> None:
    """对第一阶段候选执行语义重排，并把 rerank 分数加入 ``Ui``。"""
    if not stage1_candidates:
        return
    query = _reranker_query_text(target_meta)
    instruction = "Judge whether the historical problem is semantically useful evidence for predicting the target educational problem."
    docs: List[str] = []
    missing_indices: List[int] = []
    missing_keys: List[str] = []
    new_cache_entries: List[Tuple[str, float]] = []
    cache_scope = str(rerank_cache_scope or "pair_result").strip().lower()
    for idx, candidate in enumerate(stage1_candidates):
        hist_pid = str(candidate["problem_id"])
        cache_key = _reranker_score_cache_key(
            cache_scope=cache_scope,
            user_id=user_id,
            target_t=target_t,
            target_pid=target_pid,
            hist_pid=hist_pid,
            answer_result=str(candidate.get("answer_result") or ""),
        )
        payload = rerank_cache.get(cache_key)
        if payload is not None:
            candidate["raw_scores"]["rerank"] = float(payload["score"])
            continue
        hist_meta = problem_catalog_records[hist_pid]
        docs.append(
            _reranker_doc_text(
                hist_meta=hist_meta,
                level_diff=int(candidate["level_diff"]),
                answer_result=str(candidate["answer_result"]),
                overlap_concepts=candidate.get("knowledge_overlap_concepts") or [],
                include_answer_result=cache_scope in {"pair_result", "interaction"},
            )
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


def _flush_rerank_job_rows(conn: sqlite3.Connection, rows: List[Tuple[str, str, str, str, int, str]]) -> int:
    if not rows:
        return 0
    cursor = conn.executemany(
        """
        INSERT OR IGNORE INTO rerank_jobs
        (cache_key, target_pid, hist_pid, answer_result, level_diff, overlap_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    inserted = int(cursor.rowcount if cursor.rowcount is not None and cursor.rowcount >= 0 else 0)
    rows.clear()
    return inserted


def _score_rerank_job_rows(
    *,
    rows: List[Tuple[str, str, str, str, int, str]],
    problem_catalog_records: Dict[str, Dict[str, Any]],
    reranker: QwenReranker,
    reranker_cache_path: Path,
    rerank_cache_scope: str,
) -> int:
    if not rows:
        return 0
    target_pid = rows[0][1]
    query = _reranker_query_text(problem_catalog_records[target_pid])
    include_answer_result = rerank_cache_scope in {"pair_result", "interaction"}
    docs: List[str] = []
    keys: List[str] = []
    for cache_key, _target_pid, hist_pid, answer_result, level_diff, overlap_json in rows:
        try:
            overlap_concepts = json.loads(overlap_json)
        except Exception:
            overlap_concepts = []
        docs.append(
            _reranker_doc_text(
                hist_meta=problem_catalog_records[hist_pid],
                level_diff=int(level_diff),
                answer_result=str(answer_result),
                overlap_concepts=overlap_concepts,
                include_answer_result=include_answer_result,
            )
        )
        keys.append(cache_key)
    instruction = "Judge whether the historical problem is semantically useful evidence for predicting the target educational problem."
    scores = reranker.score(query=query, docs=docs, instruction=instruction)
    if len(scores) != len(keys):
        raise ValueError("Reranker returned inconsistent score count during cache warmup")
    _append_reranker_cache_entries(reranker_cache_path, list(zip(keys, scores)))
    return len(keys)


def _precompute_reranker_cache(
    *,
    problem_json: Path,
    student_json: Path,
    priors_dir: Path,
    cache_dir: Path,
    problem_catalog_records: Dict[str, Dict[str, Any]],
    device: str,
    reranker: QwenReranker,
    reranker_cache_path: Path,
    rerank_cache_scope: str,
    rerank_topk: int,
    smoke: bool,
    context_shard_index: int,
    context_num_shards: int,
) -> Dict[str, Any]:
    """预扫描全部第一阶段候选并批量填充 reranker 缓存。

    待评分任务先持久化到 SQLite，评分成功后删除，因此任务可中断后继续。
    """
    if rerank_cache_scope == "interaction":
        raise ValueError("Rerank cache warmup is only useful for pair_result or pair scope, not interaction scope")

    existing_cache = load_json_cache(reranker_cache_path)
    job_suffix = (
        f"{rerank_cache_scope}.{_shard_suffix(context_shard_index, context_num_shards)}"
        if context_num_shards > 1
        else rerank_cache_scope
    )
    job_db_path = cache_dir / f"reranker_jobs.{job_suffix}.sqlite"
    conn = sqlite3.connect(str(job_db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rerank_jobs (
            cache_key TEXT PRIMARY KEY,
            target_pid TEXT NOT NULL,
            hist_pid TEXT NOT NULL,
            answer_result TEXT NOT NULL,
            level_diff INTEGER NOT NULL,
            overlap_json TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rerank_jobs_target ON rerank_jobs(target_pid)")
    conn.commit()
    if existing_cache:
        existing_keys = list(existing_cache.keys())
        for start in range(0, len(existing_keys), 5000):
            conn.executemany(
                "DELETE FROM rerank_jobs WHERE cache_key = ?",
                [(key,) for key in existing_keys[start : start + 5000]],
            )
        conn.commit()

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
    collab_neighbors_path = priors_dir / "item_collaborative.json"
    collab_neighbors = (
        json.loads(collab_neighbors_path.read_text(encoding="utf-8"))
        if collab_neighbors_path.exists()
        else {}
    )
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

    pending_rows: List[Tuple[str, str, str, str, int, str]] = []
    inserted_total = 0
    seen_in_run: set[str] = set()
    for student in tqdm(student_sequences, desc="strict rerank cache collect"):
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
            collab_neighbors,
            graph_accessor,
            problem_catalog_records,
        )
        for target_t in range(1, len(seq_problem_indices)):
            target_pid = pid_lookup[seq_problem_indices[target_t]]
            _z, d_vec, _sdyn = _compute_dynamic_prior(
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
            history_start = max(0, target_t - HISTORY_WINDOW)
            for hist_pos in range(history_start, target_t):
                candidates.append(
                    _candidate_scores(
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
                )
            candidates.sort(key=lambda item: (item["Ri"], item["history_pos"]), reverse=True)
            stage1_candidates = candidates[: min(int(rerank_topk), K1_DEFAULT, len(candidates))]
            for candidate in stage1_candidates:
                hist_pid = str(candidate["problem_id"])
                cache_key = _reranker_score_cache_key(
                    cache_scope=rerank_cache_scope,
                    user_id=student.user_id,
                    target_t=target_t,
                    target_pid=target_pid,
                    hist_pid=hist_pid,
                    answer_result=str(candidate.get("answer_result") or ""),
                )
                if cache_key in existing_cache or cache_key in seen_in_run:
                    continue
                seen_in_run.add(cache_key)
                pending_rows.append(
                    (
                        cache_key,
                        target_pid,
                        hist_pid,
                        str(candidate.get("answer_result") or ""),
                        int(candidate["level_diff"]),
                        json.dumps(candidate.get("knowledge_overlap_concepts") or [], ensure_ascii=False),
                    )
                )
                if len(pending_rows) >= 20000:
                    inserted_total += _flush_rerank_job_rows(conn, pending_rows)
    inserted_total += _flush_rerank_job_rows(conn, pending_rows)

    total_jobs = int(conn.execute("SELECT COUNT(*) FROM rerank_jobs").fetchone()[0])
    print(
        f"[stage34] rerank cache warmup jobs={total_jobs} newly_inserted={inserted_total} "
        f"existing_cache={len(existing_cache)} db={job_db_path}",
        flush=True,
    )

    score_batch_size = max(128, int(reranker.batch_size) * 32)
    scored_total = 0
    current_target = ""
    rows_for_target: List[Tuple[str, str, str, str, int, str]] = []

    def flush_target_rows() -> None:
        nonlocal scored_total, rows_for_target
        while rows_for_target:
            batch = rows_for_target[:score_batch_size]
            del rows_for_target[:score_batch_size]
            scored = _score_rerank_job_rows(
                rows=batch,
                problem_catalog_records=problem_catalog_records,
                reranker=reranker,
                reranker_cache_path=reranker_cache_path,
                rerank_cache_scope=rerank_cache_scope,
            )
            conn.executemany("DELETE FROM rerank_jobs WHERE cache_key = ?", [(row[0],) for row in batch])
            conn.commit()
            scored_total += scored
            progress.update(scored)

    cursor = conn.execute(
        "SELECT cache_key, target_pid, hist_pid, answer_result, level_diff, overlap_json "
        "FROM rerank_jobs ORDER BY target_pid"
    )
    with tqdm(total=total_jobs, desc="strict rerank cache score") as progress:
        for row in cursor:
            row_tuple = (
                str(row[0]),
                str(row[1]),
                str(row[2]),
                str(row[3]),
                int(row[4]),
                str(row[5]),
            )
            if current_target and row_tuple[1] != current_target:
                flush_target_rows()
            current_target = row_tuple[1]
            rows_for_target.append(row_tuple)
        flush_target_rows()
    conn.close()
    return {
        "reranker_cache_path": str(reranker_cache_path),
        "reranker_job_db_path": str(job_db_path),
        "existing_cache_entries": len(existing_cache),
        "job_count": total_jobs,
        "newly_inserted_jobs": inserted_total,
        "scored_jobs": scored_total,
    }


def _evidence_record(candidate: Dict[str, Any], rank: int, catalog: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """把内部候选分数整理为可保存、可展示、可供 LLM 使用的证据记录。"""
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
    text_rerank_max_length: int = TEXT_RERANK_MAX_LENGTH,
    use_qwen_reranker: bool = USE_QWEN_RERANKER,
    rerank_topk: int = RERANK_TOPK,
    rerank_weight: float = RERANK_WEIGHT,
    rerank_cache_scope: str = "pair_result",
    rerank_cache_warmup_only: bool = False,
    enable_llm_summary: bool = False,
    llm_base_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_timeout_sec: int = 120,
    llm_max_tokens: int = 160,
    llm_temperature: float = 0.1,
    llm_disable_thinking: bool = False,
    llm_use_chat_template_kwargs: bool = False,
    llm_summary_compact_prompt: bool = False,
    llm_summary_workers: int = LLM_SUMMARY_WORKERS,
    llm_summary_chunk_size: int = LLM_SUMMARY_CHUNK_SIZE,
    llm_summary_batch_size: int = 1,
    reuse_existing_contexts: bool = False,
    context_shard_index: int = 0,
    context_num_shards: int = 1,
    merge_context_shards: bool = False,
) -> Stage34Result:
    """执行完整 Stage 3.4 Context 构建流程。

    支持四类运行方式：普通生成、按学生分片生成、合并分片、复用已有 Context；
    还可单独预热 reranker 缓存。普通流程完成后可继续添加 LLM 总结并编码向量。
    """
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
    if rerank_cache_warmup_only and (merge_context_shards or reuse_existing_contexts or enable_llm_summary):
        raise ValueError("--rerank_cache_warmup_only cannot be combined with merge/reuse/LLM summary modes")
    rerank_cache_scope = str(rerank_cache_scope or "pair_result").strip().lower()
    if rerank_cache_scope not in {"pair_result", "pair", "interaction"}:
        raise ValueError("--rerank_cache_scope must be one of: pair_result, pair, interaction")

    # 初始化题目目录、分片路径以及可选 reranker/LLM 客户端。
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
        cache_dir
        / f"{_reranker_cache_stem(model_name_or_path=str(text_rerank_model or TEXT_RERANK_MODEL_NAME), cache_scope=rerank_cache_scope, max_length=int(text_rerank_max_length))}.{_shard_suffix(context_shard_index, context_num_shards)}.jsonl"
        if shard_mode
        else cache_dir
        / f"{_reranker_cache_stem(model_name_or_path=str(text_rerank_model or TEXT_RERANK_MODEL_NAME), cache_scope=rerank_cache_scope, max_length=int(text_rerank_max_length))}.jsonl"
    )
    rerank_cache: Dict[str, Dict[str, Any]] = {}
    summarizer: Optional[OpenAICompatibleSummarizer] = None
    if use_qwen_reranker and not reuse_existing_contexts and not merge_context_shards:
        reranker = QwenReranker(
            model_name_or_path=str(text_rerank_model or TEXT_RERANK_MODEL_NAME),
            device=device,
            max_length=int(text_rerank_max_length),
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
            disable_thinking=bool(llm_disable_thinking),
            use_chat_template_kwargs=bool(llm_use_chat_template_kwargs),
            compact_prompt=bool(llm_summary_compact_prompt),
        )

    # 缓存预热模式只计算并保存 reranker 分数，不生成 Context。
    if rerank_cache_warmup_only:
        if reranker is None:
            raise ValueError("--rerank_cache_warmup_only requires Qwen reranker to be enabled")
        warmup_stats = _precompute_reranker_cache(
            problem_json=problem_json,
            student_json=student_json,
            priors_dir=priors_dir,
            cache_dir=cache_dir,
            problem_catalog_records=problem_catalog_records,
            device=device,
            reranker=reranker,
            reranker_cache_path=reranker_cache_path,
            rerank_cache_scope=rerank_cache_scope,
            rerank_topk=int(rerank_topk),
            smoke=bool(smoke),
            context_shard_index=context_shard_index,
            context_num_shards=context_num_shards,
        )
        manifest_path = cache_dir / f"reranker_cache_warmup.{rerank_cache_scope}.manifest.json"
        result = Stage34Result(
            contexts_path=str(contexts_path),
            preview_path=str(preview_path),
            embeddings_path=None,
            manifest_path=str(manifest_path),
            record_count=int(warmup_stats.get("scored_jobs") or 0),
            text_embed_model=str(text_embed_model or TEXT_EMBED_MODEL_NAME),
            text_embed_batch_size=int(text_embed_batch_size),
            text_embed_max_length=int(text_embed_max_length),
            text_rerank_model=str(text_rerank_model or TEXT_RERANK_MODEL_NAME),
            text_rerank_batch_size=int(text_rerank_batch_size),
            text_rerank_max_length=int(text_rerank_max_length),
            use_qwen_reranker=bool(use_qwen_reranker),
            rerank_topk=int(rerank_topk),
            rerank_weight=float(rerank_weight),
            rerank_cache_scope=rerank_cache_scope,
            reranker_cache_path=str(reranker_cache_path.resolve()),
            llm_summary_workers=int(llm_summary_workers),
            llm_summary_chunk_size=int(llm_summary_chunk_size),
            llm_summary_batch_size=int(llm_summary_batch_size),
            context_shard_index=context_shard_index,
            context_num_shards=context_num_shards,
            merge_context_shards=False,
            llm_summary_compact_prompt=bool(llm_summary_compact_prompt),
            mode="rerank_cache_warmup",
            warmup_stats=warmup_stats,
        )
        write_json(asdict(result), Path(result.manifest_path))
        return result

    # 合并模式负责拼接已完成分片，之后再统一执行 LLM 总结和向量编码。
    if merge_context_shards:
        record_count = _merge_context_shards(
            contexts_dir=contexts_dir,
            reports_dir=reports_dir,
            shard_count=context_num_shards,
            preview_limit=preview_limit,
            output_contexts_path=contexts_path,
            output_preview_path=preview_path,
        )
        if summarizer is not None:
            _enrich_contexts_with_llm_summary(
                contexts_path=contexts_path,
                cache_dir=cache_dir,
                problem_catalog_records=problem_catalog_records,
                summarizer=summarizer,
                llm_summary_workers=int(llm_summary_workers),
                llm_summary_chunk_size=int(llm_summary_chunk_size),
                llm_summary_batch_size=int(llm_summary_batch_size),
                total_records_hint=record_count,
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
            text_rerank_max_length=int(text_rerank_max_length),
            use_qwen_reranker=bool(use_qwen_reranker),
            rerank_topk=int(rerank_topk),
            rerank_weight=float(rerank_weight),
            rerank_cache_scope=rerank_cache_scope,
            reranker_cache_path=None,
            llm_summary_workers=int(llm_summary_workers),
            llm_summary_chunk_size=int(llm_summary_chunk_size),
            llm_summary_batch_size=int(llm_summary_batch_size),
            context_shard_index=context_shard_index,
            context_num_shards=context_num_shards,
            merge_context_shards=True,
            llm_summary_compact_prompt=bool(llm_summary_compact_prompt),
        )
        write_json(asdict(result), Path(result.manifest_path))
        return result

    # 复用模式跳过证据检索；普通/分片模式则从学生历史重新生成 Context JSONL。
    if reuse_existing_contexts:
        if not contexts_path.exists():
            raise FileNotFoundError(f"--reuse_existing_contexts was set but {contexts_path} does not exist")
        record_count = _context_record_count_hint(manifest_path, contexts_path) or 0
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
        collab_neighbors_path = priors_dir / "item_collaborative.json"
        collab_neighbors = (
            json.loads(collab_neighbors_path.read_text(encoding="utf-8"))
            if collab_neighbors_path.exists()
            else {}
        )
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
                    collab_neighbors,
                    graph_accessor,
                    problem_catalog_records,
                )

                # 每个目标只能使用其之前的交互。先计算动态先验，再对有限历史窗口打分。
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
                    history_start = max(0, target_t - HISTORY_WINDOW)
                    for hist_pos in range(history_start, target_t):
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

                    # 第一阶段按带时间衰减的 Ri 召回候选，可选 reranker 只作用于召回结果。
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
                            rerank_cache_scope=rerank_cache_scope,
                        )

                    # 第二阶段先选 Ui 最高证据，再迭代加入“高支持、增覆盖、低冗余”的证据。
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

                    # 将选中证据同时保存为结构化列表、纯证据文本和模板增强文本。
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

    # LLM 总结和最终向量编码只在非分片输出上执行，确保每条记录只处理一次。
    if (not shard_mode) and summarizer is not None:
        _enrich_contexts_with_llm_summary(
            contexts_path=contexts_path,
            cache_dir=cache_dir,
            problem_catalog_records=problem_catalog_records,
            summarizer=summarizer,
            llm_summary_workers=int(llm_summary_workers),
            llm_summary_chunk_size=int(llm_summary_chunk_size),
            llm_summary_batch_size=int(llm_summary_batch_size),
            total_records_hint=record_count,
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
        rerank_cache_scope=rerank_cache_scope,
        reranker_cache_path=str(reranker_cache_path.resolve()) if use_qwen_reranker else None,
        text_embed_batch_size=int(text_embed_batch_size),
        text_rerank_batch_size=int(text_rerank_batch_size),
        text_rerank_max_length=int(text_rerank_max_length),
        llm_summary_workers=int(llm_summary_workers),
        llm_summary_chunk_size=int(llm_summary_chunk_size),
        llm_summary_batch_size=int(llm_summary_batch_size),
        context_shard_index=context_shard_index,
        context_num_shards=context_num_shards,
        merge_context_shards=False,
        llm_summary_compact_prompt=bool(llm_summary_compact_prompt),
    )
    write_json(asdict(result), Path(result.manifest_path))
    return result
