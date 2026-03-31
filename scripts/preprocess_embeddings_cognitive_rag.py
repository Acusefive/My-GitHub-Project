"""
preprocess_embeddings_cognitive_rag.py

基于 Cognitive-RAG 方案的 embeddings 预处理：
1) 预计算每次交互的 e_diag -> diag_prob -> mastery M_i
2) 对每个目标时刻 t 执行两阶段检索：
   - 第一阶段：按 R_recall(h_i,q_t) 选择候选 K1
   - 第二阶段：覆盖式贪心选择最终证据 K
3) 构造分层认知上下文：
   T_ctx = Concat(D_u^t, E_i1..E_iK)
   - D_u^t：LLM 生成（带本地 cache；失败则返回空串）
   - E_ik：局部证据块（语义ID/协同/认知层级/题干内容/作答结果）
4) 用 BGE embedding 将 T_ctx 编码为 cognitive_embeddings.pkl

输出：
  cognitive_embeddings.pkl
  cognitive_context_texts.pkl
  context_text_preview.txt

备注：
- 该脚本假定 preprocess_gram.py 已输出：
  item_semantic_ids.json
  item_semantic_vectors.pkl
  problem_mu_q.json
  item_collaborative.json
  item_collaborative_embeddings.pkl
  e_diag_lr_params.json
  concept_graph_edges.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise SystemExit("缺少依赖：sentence-transformers。请先执行：pip install sentence-transformers") from e

"""
注意：只有在 --do_llm 且提供 --api_key 时才需要 requests。
为避免环境缺少 requests 导致整体脚本不可运行，本脚本会在运行时延迟导入 requests。
"""


MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def load_json_any(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]


def _extract_content_from_problem_record(problem_record: Dict[str, Any]) -> str:
    detail = problem_record.get("detail", "")
    if not detail:
        return ""
    if isinstance(detail, str):
        d = ast.literal_eval(detail)
        if isinstance(d, dict):
            return d.get("content", "") or d.get("title", "") or d.get("body", "") or ""
    if isinstance(detail, dict):
        return detail.get("content", "") or detail.get("title", "") or detail.get("body", "") or ""
    return ""


def iter_student_records(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ["data", "students", "records", "logs"]:
            if isinstance(data.get(k), list):
                return data[k]
    return []


def safe_level(x: Any) -> int:
    if x is None:
        return 0
    if isinstance(x, (int, float)):
        return int(x)
    s = str(x).strip()
    if not s:
        return 0
    return int(float(s))


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    s = float(ex.sum())
    if s <= 0:
        return np.ones_like(x) / max(1, x.shape[0])
    return ex / s


def sigmoid(x: float) -> float:
    # stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def cosine_from_normed(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def normalize_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0:
        return v.astype(np.float32, copy=False)
    return (v / n).astype(np.float32, copy=False)


class ConceptGraphScorer:
    """
    用 concept_graph_edges.csv 构建有向图（pre 有向；same/adj 已双向输出）。
    并提供：
      phi(a,b) = w_last_hop(a->...->b) * exp(-beta_g*(d-1))
    其中 d 是最短路径长度（边数）。

    工程近似：
    - w_last_hop 取 BFS 抵达 b 时“最后一条边”的 omega_rel 最大值
    """

    def __init__(
        self,
        edges_csv: str,
        *,
        W_pre: float = 1.0,
        W_same: float = 0.8,
        W_adj: float = 0.5,
        beta_g: float = 1.0,
        max_path_len: int = 3,
    ) -> None:
        self.edges_csv = edges_csv
        self.W_pre = float(W_pre)
        self.W_same = float(W_same)
        self.W_adj = float(W_adj)
        self.beta_g = float(beta_g)
        self.max_path_len = int(max_path_len)

        self.adj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)  # src -> list(dst, omega_rel)
        self._phi_cache: Dict[str, Dict[str, float]] = {}

        self._load_edges()

    def _base_w_rel(self, rel_type: str) -> float:
        if rel_type == "pre":
            return self.W_pre
        if rel_type == "same":
            return self.W_same
        if rel_type == "adj":
            return self.W_adj
        return 0.0

    def _load_edges(self) -> None:
        if not os.path.exists(self.edges_csv):
            raise FileNotFoundError(f"concept edges csv not found: {self.edges_csv}")
        with open(self.edges_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = str(row.get("src_concept", "")).strip()
                dst = str(row.get("dst_concept", "")).strip()
                rel_type = str(row.get("rel_type", "")).strip()
                weight = float(row.get("weight", 0.0))
                if not src or not dst or weight <= 0:
                    continue
                omega = self._base_w_rel(rel_type) * weight
                if omega <= 0:
                    continue
                self.adj[src].append((dst, omega))

    def phi_map_from_src(self, src: str) -> Dict[str, float]:
        if src in self._phi_cache:
            return self._phi_cache[src]

        # BFS up to max_path_len, 维护 dist 与 last-hop best omega
        dist: Dict[str, int] = {src: 0}
        last_best_omega: Dict[str, float] = {src: 0.0}
        # queue of (node)
        q: List[str] = [src]
        qi = 0
        while qi < len(q):
            u = q[qi]
            qi += 1
            du = dist[u]
            if du >= self.max_path_len:
                continue
            for v, omega in self.adj.get(u, []):
                dv = du + 1
                if dv > self.max_path_len:
                    continue
                # 只保留最短路径距离（但 last_best_omega 取最大）
                if v not in dist or dv < dist[v]:
                    dist[v] = dv
                    last_best_omega[v] = float(omega)
                    q.append(v)
                elif dv == dist[v]:
                    last_best_omega[v] = max(last_best_omega.get(v, 0.0), float(omega))
                else:
                    # dv > dist[v] ：不是最短路径，不进入 phi
                    pass

        # 计算 phi
        phi: Dict[str, float] = {}
        for b, d in dist.items():
            if b == src:
                continue
            omega_last = last_best_omega.get(b, 0.0)
            if omega_last <= 0:
                continue
            phi[b] = float(omega_last) * math.exp(-self.beta_g * float(d - 1))

        self._phi_cache[src] = phi
        return phi

    def Sg(self, concepts_i: Sequence[str], concepts_t: Sequence[str]) -> float:
        if not concepts_i or not concepts_t:
            return 0.0
        # S_bar_g = max_{a in ci, b in ct} phi(a,b)
        best = 0.0
        concepts_t_set = set(concepts_t)
        for a in concepts_i:
            phi = self.phi_map_from_src(a)
            for b in concepts_t_set:
                v = phi.get(b, 0.0)
                if v > best:
                    best = v
        return float(best)


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    if not a or not b:
        return 0.0
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def build_pid2meta(problem_json: str) -> Dict[str, Dict[str, Any]]:
    problems_raw = load_json_any(problem_json)
    if isinstance(problems_raw, dict):
        problems_raw = [problems_raw]
    out: Dict[str, Dict[str, Any]] = {}
    for p in problems_raw:
        pid = str(p.get("problem_id", p.get("id", "")))
        if pid:
            out[pid] = p
    return out


def compute_e_diag_and_mastery_for_student(
    *,
    seq: Sequence[Dict[str, Any]],
    pid2vec_sem: Dict[str, np.ndarray],
    w_d: np.ndarray,
    b_d: float,
    W: int,
    beta_d: float,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    对一个学生序列的每个时间步 t 计算：
      e_diag[t]（attention pooling）
      mastery[t] = (1-beta_d)*r_t + beta_d*sigmoid(w_d^T e_diag[t] + b_d)
    """
    D = next(iter(pid2vec_sem.values())).shape[0] if pid2vec_sem else 0
    e_diags: List[np.ndarray] = []
    mastery: List[float] = []

    n = len(seq)
    # 为了速度：提前把 pid 映射到向量与 r
    pids: List[str] = []
    rs: List[int] = []
    vecs: List[Optional[np.ndarray]] = []
    for t in range(n):
        log = seq[t]
        pid = str(log.get("problem_id", ""))
        r_val = log.get("is_correct", 0)
        r_i = int(r_val)
        pids.append(pid)
        rs.append(r_i)
        vecs.append(pid2vec_sem.get(pid))

    for t in range(n):
        window_start = max(0, t - W)
        # history: [window_start..t-1]
        if t == 0 or window_start >= t:
            # no history
            any_vec = next(iter(pid2vec_sem.values()))
            e_diag = np.zeros((D,), dtype=np.float32)
        else:
            keys_list: List[np.ndarray] = []
            values_list: List[np.ndarray] = []
            for k in range(window_start, t):
                v_k = vecs[k]
                if v_k is None:
                    continue
                keys_list.append(v_k)
                sign = 2 * int(rs[k]) - 1
                values_list.append(v_k * float(sign))

            q_vec = vecs[t]
            if q_vec is None or not keys_list:
                e_diag = np.zeros((D,), dtype=np.float32)
            else:
                K = np.stack(keys_list, axis=0).astype(np.float32)  # [L,D]
                V = np.stack(values_list, axis=0).astype(np.float32)  # [L,D]
                scores = K @ q_vec.astype(np.float32)
                alpha = softmax(scores)
                e_diag = (alpha.reshape(-1, 1) * V).sum(axis=0).astype(np.float32)

        # M_i
        r_t = float(rs[t])
        z = float(np.dot(w_d.astype(np.float32), e_diag.astype(np.float32)) + float(b_d))
        diag_prob = sigmoid(z)
        M = (1.0 - float(beta_d)) * r_t + float(beta_d) * diag_prob
        e_diags.append(e_diag)
        mastery.append(float(M))
    return e_diags, mastery


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem_json", default="problem.json")
    ap.add_argument("--student_json", default="student-problem-fine.json")

    ap.add_argument("--semantic_ids_json", default="item_semantic_ids.json")
    # 兼容旧脚本参数名
    ap.add_argument("--semantic_json", dest="semantic_ids_json", default="item_semantic_ids.json", help="语义ID映射（别名兼容）")
    ap.add_argument("--semantic_vectors_pkl", default="item_semantic_vectors.pkl")
    ap.add_argument("--collab_json", default="item_collaborative.json")
    ap.add_argument("--collab_vecs_pkl", default="item_collaborative_embeddings.pkl")
    ap.add_argument("--lr_params_json", default="e_diag_lr_params.json")

    ap.add_argument("--concept_edges_csv", default="concept_graph_edges.csv")

    ap.add_argument("--out_pkl", default="cognitive_embeddings.pkl")
    ap.add_argument("--out_text_pkl", default="cognitive_context_texts.pkl")
    ap.add_argument("--out_preview_txt", default="context_text_preview.txt")
    ap.add_argument("--preview_limit", type=int, default=50)

    ap.add_argument("--dry_run", action="store_true", help="只生成文本，不跑 embedding")

    ap.add_argument("--topk1", type=int, default=10)  # K1
    # 兼容旧脚本参数：旧脚本用 --topk 表示最终证据数
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--topk", dest="K", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_content_chars", type=int, default=200)

    # 检索参数（默认建议值；全部可调）
    ap.add_argument("--theta_time_decay", type=float, default=0.05)
    ap.add_argument("--W", type=int, default=10)
    ap.add_argument("--alpha_p", type=float, default=1.0)
    ap.add_argument("--alpha_h", type=float, default=1.0)
    ap.add_argument("--eta_k", type=float, default=0.5)
    ap.add_argument("--rho_h", type=float, default=0.5)

    ap.add_argument("--w_k", type=float, default=10.0)
    ap.add_argument("--w_p", type=float, default=5.0)
    ap.add_argument("--w_m", type=float, default=2.0)
    ap.add_argument("--w_h", type=float, default=2.0)
    ap.add_argument("--w_g", type=float, default=1.0)

    # dynamic cf 用的参数
    ap.add_argument("--eta1_exp", type=float, default=0.33)
    ap.add_argument("--eta2_exp", type=float, default=0.33)
    ap.add_argument("--eta3_exp", type=float, default=0.34)
    ap.add_argument("--beta_L", type=float, default=1.0)

    # stage2 贪心覆盖/冗余
    ap.add_argument("--alpha_k_cov", type=float, default=10.0)
    ap.add_argument("--alpha_p_cov", type=float, default=5.0)
    ap.add_argument("--alpha_m_cov", type=float, default=2.0)
    ap.add_argument("--alpha_h_cov", type=float, default=2.0)
    ap.add_argument("--alpha_g_cov", type=float, default=1.0)
    ap.add_argument("--alpha_c_cov", type=float, default=1.0)
    ap.add_argument("--beta_cov", type=float, default=1.0)
    ap.add_argument("--gamma_red", type=float, default=0.5)
    ap.add_argument("--mu_k_red", type=float, default=1.0)
    ap.add_argument("--mu_r_red", type=float, default=1.0)
    ap.add_argument("--mu_s_red", type=float, default=0.2)
    ap.add_argument("--eps", type=float, default=1e-6)

    # LLM 摘要缓存
    ap.add_argument("--do_llm", action="store_true", help="是否调用 LLM 生成全局摘要 D_u^t")
    ap.add_argument("--api_key", default="", help="LLM API Key（空则不会调用）")
    ap.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1/")
    ap.add_argument("--llm_model", default="deepseek-v3.2-exp")
    ap.add_argument("--summary_cache_path", default="summary_cache.json")

    args = ap.parse_args()

    # Load artifacts
    print("Loading problems and student records ...")
    problems = build_pid2meta(args.problem_json)
    student_records = iter_student_records(load_json_any(args.student_json))

    print("Loading semantic ids/vectors ...")
    semantic_ids: Dict[str, str] = load_json_any(args.semantic_ids_json)
    with open(args.semantic_vectors_pkl, "rb") as f:
        pid2vec_sem: Dict[str, np.ndarray] = pickle.load(f)

    # normalized sem vectors for cosine
    pid2vec_sem_norm: Dict[str, np.ndarray] = {pid: normalize_vec(v) for pid, v in pid2vec_sem.items()}

    print("Loading collaborative embeddings ...")
    collab_neighbors: Dict[str, List[str]] = load_json_any(args.collab_json)
    with open(args.collab_vecs_pkl, "rb") as f:
        pid2vec_collab: Dict[str, np.ndarray] = pickle.load(f)
    pid2vec_collab_norm: Dict[str, np.ndarray] = {pid: normalize_vec(v) for pid, v in pid2vec_collab.items()}

    print("Loading e_diag LR params ...")
    lr_params = load_json_any(args.lr_params_json)
    w_d = np.asarray(lr_params["w_d"], dtype=np.float32)
    b_d = float(lr_params["b_d"])
    beta_d = float(lr_params["beta_d"])
    W_diag = int(lr_params.get("W", args.W))

    # Concept graph scorer
    print("Loading concept graph scorer ...")
    graph_scorer = ConceptGraphScorer(
        args.concept_edges_csv,
        W_pre=1.0,
        W_same=0.8,
        W_adj=0.5,
        beta_g=1.0,
        max_path_len=3,
    )

    # Precompute question meta
    pid2meta = {}
    for pid, p in problems.items():
        concepts = p.get("concepts", []) or []
        concepts = [str(c) for c in concepts if c is not None and str(c)]
        pid2meta[pid] = {
            "concepts": concepts,
            "level": safe_level(p.get("cognitive_dimension", 0)),
            "content": _extract_content_from_problem_record(p),
        }

    # summary cache
    cache: Dict[str, str] = {}
    if os.path.exists(args.summary_cache_path):
        cache = load_json_any(args.summary_cache_path)
        if not isinstance(cache, dict):
            raise ValueError(f"summary_cache_path must contain a dict, got: {type(cache)}")

    def get_summary_with_cache(
        uid: str,
        evidence_pids: List[str],
        recent_hist_items: List[Tuple[str, int]],
        target_pid: str,
    ) -> str:
        if not args.do_llm:
            return ""
        if not args.api_key:
            raise RuntimeError("S3 summary generation enabled but args.api_key is empty.")
        # hash key
        h = hashlib.sha256()
        h.update(str(uid).encode("utf-8"))
        h.update(b"|")
        h.update(str(target_pid).encode("utf-8"))
        h.update(b"|")
        h.update(",".join(evidence_pids).encode("utf-8"))
        h.update(b"|")
        recent_correct_seq = [rc for _, rc in recent_hist_items]
        h.update(",".join(str(x) for x in recent_correct_seq).encode("utf-8"))
        key = h.hexdigest()
        if key in cache:
            return cache[key]

        # build prompt (1-2句中文摘要）
        # 只塞语义ID和正确性，避免超长上下文
        semantic_seq_lines = []
        for pid, rc in recent_hist_items:
            semantic_id = semantic_ids.get(pid, "Unknown")
            semantic_seq_lines.append(f"{semantic_id}:{rc}")

        evidence_lines = []
        for pid in evidence_pids:
            sem_id = semantic_ids.get(pid, "Unknown")
            # evidence 的 r 严格由 recent_correct_seq 不一定包含，这里只用语义ID
            evidence_lines.append(f"{sem_id}")
        prompt = (
            "你是知识追踪助手。请根据最近作答与检索证据，输出 1-2 句中文摘要，"
            "概括学生在目标题相关概念上的掌握状态（是否偏对/偏错是否集中、同层是否波动）。"
            "不要输出任何多余内容。\n\n"
            f"[TARGET]\n{semantic_ids.get(target_pid,'Unknown')}\n\n"
            f"[RECENT]\n" + " ".join(semantic_seq_lines) + "\n\n"
            f"[EVIDENCE]\n" + " ".join(evidence_lines) + "\n"
        )
        system_role = "你必须遵循用户要求，输出严格 1-2 句中文。"

        headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
        body = {
            "model": args.llm_model,
            "messages": [{"role": "system", "content": system_role}, {"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 64,
        }
        import requests  # lazy import
        resp = requests.post(args.base_url + "chat/completions", headers=headers, json=body, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        summary = str(data["choices"][0]["message"]["content"]).strip()

        cache[key] = summary
        with open(args.summary_cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        return summary

    # stage1/2 helper
    def compute_Sk(qi_pid: str, qt_pid: str) -> float:
        mi = pid2meta[qi_pid]["concepts"]
        mt = pid2meta[qt_pid]["concepts"]
        return jaccard(mi, mt)

    def compute_Sg(qi_pid: str, qt_pid: str) -> float:
        mi = pid2meta[qi_pid]["concepts"]
        mt = pid2meta[qt_pid]["concepts"]
        # 先算 S_bar_g
        sbar = graph_scorer.Sg(mi, mt)
        sk = jaccard(mi, mt)
        return (1.0 - sk) * sbar

    def compute_Scf(qi_pid: str, qt_pid: str) -> float:
        vi = pid2vec_collab_norm.get(qi_pid)
        vt = pid2vec_collab_norm.get(qt_pid)
        if vi is None or vt is None:
            return 0.0
        cos = float(np.dot(vi, vt))
        return float((1.0 + cos * cos) / 2.0)

    def retrieve_for_target(
        seq: Sequence[Dict[str, Any]],
        target_t: int,
        student_uid: str,
        mastery: Sequence[float],
    ) -> Tuple[List[int], List[str]]:
        """
        返回：
          - selected evidence indices (history indices in [0..target_t-1]) length K
          - evidence pids in the chosen order
        """
        qt_log = seq[target_t]
        qt_pid = str(qt_log.get("problem_id", ""))
        if qt_pid not in pid2meta:
            return [], []

        L_t = int(pid2meta[qt_pid]["level"])
        concepts_t = pid2meta[qt_pid]["concepts"]

        history_indices = list(range(0, target_t))

        # stage1 scoring for all i
        scores1: List[Tuple[float, int]] = []

        # Precompute cosine semantic vectors for time decay sim
        qt_vec = pid2vec_sem_norm.get(qt_pid)
        if qt_vec is None:
            # no semantics => skip
            return [], []

        # sims and gamma weights for time decay
        hist_pids = [str(seq[i].get("problem_id", "")) for i in history_indices]
        sims = []
        for pid in hist_pids:
            v = pid2vec_sem_norm.get(pid)
            sims.append(float(np.dot(qt_vec, v)) if v is not None else -1e9)
        sims_arr = np.asarray(sims, dtype=np.float64)
        # gamma_j softmax over history
        # use exp(sims) stable
        sims_arr = sims_arr - np.max(sims_arr)
        exp_s = np.exp(sims_arr)
        denom = float(exp_s.sum())
        if denom <= 0:
            gamma = np.zeros_like(exp_s)
        else:
            gamma = exp_s / denom

        # suffix sums for d(t,i)
        # index mapping: history_indices[k] == i, where k==i
        # gamma_j is for j in [0..target_t-1]
        suffix = np.zeros(target_t + 1, dtype=np.float64)
        for i in range(target_t - 1, -1, -1):
            suffix[i] = suffix[i + 1] + gamma[i]

        alpha_p = float(args.alpha_p)
        alpha_h = float(args.alpha_h)
        eta_k = float(args.eta_k)
        rho_h = float(args.rho_h)

        w_k = float(args.w_k)
        w_p = float(args.w_p)
        w_m = float(args.w_m)
        w_h = float(args.w_h)
        w_g = float(args.w_g)

        eta1 = float(args.eta1_exp)
        eta2 = float(args.eta2_exp)
        eta3 = float(args.eta3_exp)
        beta_L = float(args.beta_L)
        theta_td = float(args.theta_time_decay)

        # mastery M_i precomputed outside
        # compute diag mastery for each i quickly by looking at mastery_arr

        # We compute mastery array once per student outside, attach to closure.
        for i in history_indices:
            hi_pid = str(seq[i].get("problem_id", ""))
            if hi_pid not in pid2meta:
                continue
            sk = compute_Sk(hi_pid, qt_pid)
            if sk <= 0 and beta_L <= 0:
                # still might get from other terms; but cheap pruning
                pass
            Li = int(pid2meta[hi_pid]["level"])
            dL = L_t - Li
            r_i = int(seq[i].get("is_correct", 0))
            M_i = float(mastery[i])

            Sp = 0.0
            Sm = 0.0
            Sh = 0.0
            if dL > 0:
                # front support
                # exp(-alpha_p*(ΔL-1))
                Sp = sk * math.exp(-alpha_p * (float(dL) - 1.0)) * M_i
            elif dL == 0:
                vi = pid2vec_sem_norm.get(hi_pid)
                if vi is not None:
                    cos = float(np.dot(vi, qt_vec))
                    Ai = float((1.0 + cos) / 2.0)
                else:
                    Ai = 0.0
                Sm = (eta_k * sk + (1.0 - eta_k) * Ai) * M_i
            else:
                # higher-level evidence
                abs_dL = abs(float(dL))
                last = float(abs_dL) - 1.0
                Sh = sk * math.exp(-alpha_h * last) * (float(r_i) - rho_h * (1.0 - float(r_i)))

            Sg = compute_Sg(hi_pid, qt_pid) if w_g > 0 else 0.0
            Scf = compute_Scf(hi_pid, qt_pid)

            # E_exp and lambda_i
            # E_exp = eta1*S_k + eta2*exp(-beta_L*|L(qi)-L(qt)|) + eta3*S_g
            exp_level = math.exp(-beta_L * abs(float(Li - L_t)))
            E_exp = eta1 * sk + eta2 * exp_level + eta3 * Sg
            E_exp = max(0.0, min(1.0, float(E_exp)))
            lambda_i = 1.0 - E_exp

            R_base = w_k * sk + w_p * Sp + w_m * Sm + w_h * Sh + w_g * Sg + lambda_i * Scf

            # time decay: d(t,i) = (t-i) * sum_{j=i+1}^{t-1} gamma_j = (t-i)*suffix[i+1]
            # since gamma indexed by j in [0..t-1]
            if target_t - i <= 0:
                T_i = 1.0
            else:
                d = float(target_t - i) * float(suffix[i + 1])
                T_i = math.exp(-theta_td * d)

            R_recall = float(R_base * T_i)
            if R_recall > 0:
                scores1.append((R_recall, i))

        scores1.sort(key=lambda x: x[0], reverse=True)
        cand = [i for _, i in scores1[: int(args.topk1)]]
        if not cand:
            return [], []

        # stage2 greedy selection
        # precompute needed S values for candidates for speed
        cand_data: Dict[int, Dict[str, float]] = {}
        for i in cand:
            hi_pid = str(seq[i].get("problem_id", ""))
            sk = compute_Sk(hi_pid, qt_pid)
            Li = int(pid2meta[hi_pid]["level"])
            dL = L_t - Li
            r_i = int(seq[i].get("is_correct", 0))
            M_i = float(mastery[i])

            Sp = 0.0
            Sm = 0.0
            Sh = 0.0
            if dL > 0:
                Sp = sk * math.exp(-alpha_p * (float(dL) - 1.0)) * M_i
            elif dL == 0:
                vi = pid2vec_sem_norm.get(hi_pid)
                cos = float(np.dot(vi, qt_vec)) if vi is not None else 0.0
                Ai = float((1.0 + cos) / 2.0)
                Sm = (eta_k * sk + (1.0 - eta_k) * Ai) * M_i
            else:
                abs_dL = abs(float(dL))
                last = float(abs_dL) - 1.0
                Sh = sk * math.exp(-alpha_h * last) * (float(r_i) - rho_h * (1.0 - float(r_i))) if r_i == 0 else sk * math.exp(-alpha_h * last) * 1.0

            Sg = compute_Sg(hi_pid, qt_pid) if w_g > 0 else 0.0
            Scf = compute_Scf(hi_pid, qt_pid)

            exp_level = math.exp(-beta_L * abs(float(Li - L_t)))
            E_exp = eta1 * sk + eta2 * exp_level + eta3 * Sg
            E_exp = max(0.0, min(1.0, float(E_exp)))
            lambda_i = 1.0 - E_exp

            cand_data[i] = {
                "sk": float(sk),
                "Sp": float(Sp),
                "Sm": float(Sm),
                "Sh": float(Sh),
                "Sg": float(Sg),
                "Scf": float(Scf),
                "lambda": float(lambda_i),
            }

        def activation_vec(i: int) -> np.ndarray:
            dat = cand_data[i]
            # binary activation by sign>0
            ap = 1.0 if dat["Sp"] > 0 else 0.0
            am = 1.0 if dat["Sm"] > 0 else 0.0
            ah = 1.0 if dat["Sh"] > 0 else 0.0
            ag = 1.0 if dat["Sg"] > 0 else 0.0
            return np.asarray([ap, am, ah, ag], dtype=np.float32)

        def compute_role_sim(a: np.ndarray, b: np.ndarray) -> float:
            denom = float(np.sum(a) * np.sum(b) + args.eps)
            return float(np.dot(a, b)) / denom

        def compute_k_jac(i: int, j: int) -> float:
            hi_pid = str(seq[i].get("problem_id", ""))
            hj_pid = str(seq[j].get("problem_id", ""))
            return jaccard(pid2meta[hi_pid]["concepts"], pid2meta[hj_pid]["concepts"])

        def compute_sim_for_red(i: int, j: int) -> float:
            a_i = activation_vec(i)
            a_j = activation_vec(j)
            kij = compute_k_jac(i, j)
            role_sim = compute_role_sim(a_i, a_j)
            # semantic cosine
            pi = str(seq[i].get("problem_id", ""))
            pj = str(seq[j].get("problem_id", ""))
            vi = pid2vec_sem_norm.get(pi)
            vj = pid2vec_sem_norm.get(pj)
            cos_sem = float(np.dot(vi, vj)) if vi is not None and vj is not None else 0.0
            return float(args.mu_k_red * kij + args.mu_r_red * role_sim + args.mu_s_red * cos_sem)

        # U_i
        def compute_U(i: int) -> float:
            dat = cand_data[i]
            return (
                float(args.alpha_k_cov) * dat["sk"]
                + float(args.alpha_p_cov) * dat["Sp"]
                + float(args.alpha_m_cov) * dat["Sm"]
                + float(args.alpha_h_cov) * dat["Sh"]
                + float(args.alpha_g_cov) * dat["Sg"]
                + float(args.alpha_c_cov) * dat["lambda"] * dat["Scf"]
            )

        # coverage increments
        Kt = pid2meta[qt_pid]["concepts"]
        Kt_set = set(Kt)

        chosen: List[int] = []
        # initialize with max U
        cand_rem = list(cand)
        best_i = max(cand_rem, key=lambda x: compute_U(x))
        chosen.append(best_i)
        cand_rem.remove(best_i)

        # maintain union concepts for Δ_k
        union_concepts = set(pid2meta[str(seq[best_i].get("problem_id", ""))]["concepts"])

        def compute_Cov_add(i: int) -> float:
            nonlocal union_concepts
            # relation coverage: binary activation increment
            # we approximate with whether each role already activated
            # maintain role coverage flags
            a_i = activation_vec(i)
            # current role coverage flags
            # for simplicity, if any chosen item activates a role, it is considered covered
            covered = np.zeros((4,), dtype=np.float32)
            for j in chosen:
                aj = activation_vec(j)
                covered = np.maximum(covered, aj)
            # Δ relation part: whether new roles appear
            delta_role = np.minimum(1.0, covered + a_i) - np.minimum(1.0, covered)

            # knowledge coverage
            hi_pid = str(seq[i].get("problem_id", ""))
            concepts_i = pid2meta[hi_pid]["concepts"]
            new_union = union_concepts | set(concepts_i)
            cur_cnt = len(union_concepts & Kt_set)
            new_cnt = len(new_union & Kt_set)
            delta_k = (new_cnt - cur_cnt) / float(len(Kt_set) + args.eps)

            # graph coverage: approximate by knowledge coverage (strict graph neighborhood requires KG neighborhood sets)
            delta_g = delta_k

            # weights η_* 这里默认 1.0（与先前默认一致）
            eta_p = 1.0
            eta_m = 1.0
            eta_h = 1.0
            eta_k = 1.0
            eta_g = 1.0
            # delta_role[0]=p, [1]=m, [2]=h, [3]=g
            return float(eta_p * delta_role[0] + eta_m * delta_role[1] + eta_h * delta_role[2] + eta_k * delta_k + eta_g * delta_g)

        # Greedy iterations
        while len(chosen) < int(args.K) and cand_rem:
            best_score = -1e18
            best_candidate = None
            best_update_union = None
            for i in cand_rem:
                cov_inc = compute_Cov_add(i)
                # redundancy penalty
                if chosen:
                    red = max(compute_sim_for_red(i, j) for j in chosen)
                else:
                    red = 0.0
                score = compute_U(i) + float(args.beta_cov) * cov_inc - float(args.gamma_red) * red
                if score > best_score:
                    best_score = score
                    best_candidate = i
            assert best_candidate is not None

            # finalize pick
            chosen.append(best_candidate)
            cand_rem.remove(best_candidate)
            hi_pid = str(seq[best_candidate].get("problem_id", ""))
            union_concepts |= set(pid2meta[hi_pid]["concepts"])

        # chosen order: keep time order ascending
        chosen_sorted = sorted(chosen)
        evidence_pids = [str(seq[i].get("problem_id", "")) for i in chosen_sorted]
        return chosen_sorted, evidence_pids

    # Iterate students and build flat texts
    flat_texts: List[str] = []
    index_map: List[Tuple[int, int]] = []
    per_student_len: List[int] = []

    kept_student_idx = 0
    for s_idx, rec in enumerate(tqdm(student_records, desc="Building context texts")):
        uid = str(rec.get("user_id", ""))
        seq = rec.get("seq", []) or []
        if not isinstance(seq, list):
            continue
        s_real_idx = kept_student_idx
        kept_student_idx += 1
        per_student_len.append(len(seq))

        # precompute mastery for all t in this student
        _, mastery = compute_e_diag_and_mastery_for_student(
            seq=seq,
            pid2vec_sem=pid2vec_sem,
            w_d=w_d,
            b_d=b_d,
            W=W_diag,
            beta_d=beta_d,
        )

        # local reference to mastery for retrieve_for_target
        # store in closure variable
        # noinspection PyUnboundLocalVariable
        for t in range(len(seq)):
            index_map.append((s_real_idx, t))
            if t == 0:
                flat_texts.append("")
                continue

            chosen_idx, evidence_pids = retrieve_for_target(seq, t, uid, mastery)
            if not evidence_pids:
                flat_texts.append("")
                continue

            target_pid = str(seq[t].get("problem_id", ""))
            evidence_blocks: List[str] = []
            for k, i in enumerate(chosen_idx):
                h_pid = str(seq[i].get("problem_id", ""))
                sem_id = semantic_ids.get(h_pid, "Unknown")
                collab_nei = collab_neighbors.get(h_pid, [])[:3]
                collab_sem_ids = [semantic_ids.get(x, "Unknown") for x in collab_nei]
                collab_str = "、".join(collab_sem_ids) if collab_sem_ids else "无"
                lvl = pid2meta.get(h_pid, {}).get("level", "未知")
                content_str = pid2meta.get(h_pid, {}).get("content", "")
                content_str = (content_str[: int(args.max_content_chars)] if content_str else "无可用内容").replace("\n", " ")
                r_i = int(seq[i].get("is_correct", 0))
                perf = "正确" if r_i == 1 else "错误"

                evidence_blocks.append(
                    f"[EVIDENCE {k+1}]\n"
                    f"Question ID: {sem_id}\n"
                    f"Collaborative: {collab_str}\n"
                    f"Cognitive: Level={lvl}\n"
                    f"Content: {content_str}\n"
                    f"Result: {perf}"
                )

            # global summary (LLM cached)
            recent_hist = seq[max(0, t - W_diag):t]
            recent_recent_seq: List[Tuple[str, int]] = []
            for log in recent_hist:
                pid = str(log.get("problem_id", ""))
                rc = int(log.get("is_correct", 0))
                recent_recent_seq.append((pid, rc))
            D_u_t = get_summary_with_cache(
                uid=uid,
                evidence_pids=evidence_pids,
                recent_hist_items=recent_recent_seq,
                target_pid=target_pid,
            )

            text = "\n".join(
                [f"[GLOBAL SUMMARY]\n{D_u_t}".strip(), "\n".join(evidence_blocks)]
            ).strip()
            flat_texts.append(text)

    # Save aligned context texts
    print(f"[SAVE] context_texts -> {args.out_text_pkl}")
    texts: List[List[str]] = [[""] * int(per_student_len[i]) for i in range(len(per_student_len))]
    for idx, (s_i, t_i) in enumerate(index_map):
        texts[s_i][t_i] = flat_texts[idx]

    with open(args.out_text_pkl, "wb") as f:
        pickle.dump(texts, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[OK] Saved {args.out_text_pkl}")

    # preview
    with open(args.out_preview_txt, "w", encoding="utf-8") as f:
        f.write(f"=== Context Preview (Total records: {len(flat_texts)}) ===\n\n")
        for i, text in enumerate(flat_texts[: int(args.preview_limit)]):
            s_i, t_i = index_map[i]
            f.write(f"--- Record {i} (Student {s_i}, Time {t_i}) ---\n")
            f.write(text)
            f.write("\n" + "=" * 50 + "\n")
    print(f"[AUDIT] preview saved to {args.out_preview_txt}")

    if args.dry_run:
        print("[INFO] dry_run enabled; skip embedding generation.")
        return

    # Encode texts -> embeddings.pkl
    model = SentenceTransformer(MODEL_NAME)
    total = len(flat_texts)
    print(f"Encoding {total} texts ...")

    flat_embs: List[np.ndarray] = []
    for start in tqdm(range(0, total, int(args.batch_size)), desc="Encoding batches"):
        batch_texts = flat_texts[start:start + int(args.batch_size)]
        batch_texts = [bt if bt is not None else "" for bt in batch_texts]
        embs = model.encode(batch_texts, batch_size=int(args.batch_size), show_progress_bar=False, convert_to_numpy=True)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        for j in range(embs.shape[0]):
            flat_embs.append(embs[j].astype(np.float32, copy=False))

    # rehydrate into per-student sequences
    semantic_dim = int(flat_embs[0].shape[0]) if flat_embs else 0
    emb_seqs: List[List[np.ndarray]] = [[None] * int(per_student_len[i]) for i in range(len(per_student_len))]  # type: ignore[list-item]
    for idx, (s_i, t_i) in enumerate(index_map):
        emb_seqs[s_i][t_i] = flat_embs[idx]
    zero = np.zeros((semantic_dim,), dtype=np.float32)
    for s_i in range(len(emb_seqs)):
        for t_i in range(len(emb_seqs[s_i])):
            if emb_seqs[s_i][t_i] is None:
                emb_seqs[s_i][t_i] = zero.copy()

    out_dir = os.path.dirname(os.path.abspath(args.out_pkl)) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out_pkl, "wb") as f:
        pickle.dump(emb_seqs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[OK] Saved embeddings to {args.out_pkl} (dim={semantic_dim})")


if __name__ == "__main__":
    main()

