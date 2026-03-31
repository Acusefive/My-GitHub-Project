"""
基于 Cognitive-RAG 方案的“分层语义ID/语义注入/诊断LR/协同信号”离线预处理：

输出（默认文件名，可用参数覆盖）：
1) item_semantic_ids.json
   - {problem_id: hierarchical_semantic_id}

2) item_semantic_vectors.pkl
   - {problem_id: np.ndarray(shape=(D,), dtype=float32)}
   - 用于 e_diag attention pooling / 检索中的语义相似度等

3) problem_mu_q.json
   - {problem_id: float}，Rasch 难度参数（已做去均值平移）

4) item_collaborative.json
   - {problem_id: [neighbor_problem_id, ...]}，用于文本中的“协同邻接展示”（兼容现有脚本）

5) item_collaborative_embeddings.pkl（新增）
   - {problem_id: np.ndarray(shape=(C,), dtype=float32)}，用于 S_cf 的 cosine 相似度

6) e_diag_lr_params.json（新增）
   - LR 的 w_d / b_d（用于 M_i）

依赖：sentence_transformers, sklearn, numpy, tqdm, jieba, gensim, torch, tqdm
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import jieba
import numpy as np
import torch
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

# 尝试导入 sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("错误: 请先安装依赖 -> pip install sentence-transformers")
    sys.exit(1)

# === 配置 ===
# 使用 BGE Small 中文模型，体积小(约100MB)，中文语义理解能力极强，同时也支持英文
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 停用词表：用于最后的关键词提取（Labeling），不再影响聚类结果
CUSTOM_STOP_WORDS = set([
    "填空", "选择", "下列", "属于", "关于", "正确", "错误", "选项", "描述", "答案", 
    "一个", "什么", "可以", "的是", "以及", "因为", "所以", "如图", "所示", "包括", 
    "内容", "题目", "问题", "我们", "你们", "它们", "这个", "那个", "部分", 
    "与其", "不如", "虽然", "但是", "不仅", "而且", "其中", "或者", "如果",
    "ABCD", "A.", "B.", "C.", "D.", "1.", "2.", "3.", "4.", "000","100",
    "主要", "根据", "特征", "具有", "一种", "一般", "通常", "使用", "利用"," ", "\n", "\t", "br", "&nbsp;", 
    "的", "了", "和", "是", "在", "与", "及", "或", 'nbsp','the','以下','说法','下面','进行',"单选",
    "分析", "解答", "解析", "答案", "详解", "过程", "步骤", "结论",  # 题目结构词
    "如图", "所示", "下列", "选项", "已知", "求证", "计算", "求解", "关于", # 题目引导词
    "正确", "错误", "属于", "可以", "怎么", "什么", "为什么", "结构", "特征", "性质","一项","两项","三项","四项","____","_____", # 泛泛的名词
    "系统", "理论", "曲线", "模型", "定义", "原理", "方法", "基础", "基本", 
    "概念", "应用", "技术", "作业", "习题", "第一章", "第二章", "第三章", 
    "第四章", "第五章", "第六章", "第七章", "第八章", "测试", "练习题",
    "企业", "公司", "质量", "方式", "研究", "设计", "应用", "管理", 
    "不同", "影响", "作用", "相关", "主要", "手段", "目的", "特点",# === 第三轮新增：清洗代码残留和提问词 ===
    "frac", "amp", "div", "span", "class", "style", "width", "height","ldquo","rdquo","sumlimits","&#","pmatrix","begin","end","matrix","left","right","left[","right]","rm","mu","theta", # HTML/LaTeX 垃圾
    "of", "to", "in", "for", "with", "on", "at", "by", "from","is","mathjaxinline","mathbf","red","black","blue","green","yellow","purple","orange","brown","gray","pink","lime","teal","indigo","violet","maroon","navy","olive","crimson","azure","fuchsia","chartreuse","coral","cyan","gold","silver","plum","salmon","tan","turquoise","wheat","ivory","khaki","lavender","magenta","mistyrose","navajowhite","oldlace","papayawhip","peachpuff","peru","pink","plum","powderblue","rosybrown","royalblue","sienna","skyblue","slateblue","slategray","snow","springgreen","steelblue","tan","thistle","tomato","turquoise","violet","wheat","whitesmoke","yellowgreen","black","blue","green","red","yellow","purple","orange","brown","gray","pink","lime","teal","indigo","violet","maroon","navy","olive","crimson","azure","fuchsia","chartreuse","coral","cyan","gold","silver","plum","salmon","tan","turquoise","wheat","ivory","khaki","lavender","magenta","mistyrose","navajowhite","oldlace","papayawhip","peachpuff","peru","pink","plum","powderblue","rosybrown","royalblue","sienna","skyblue","slateblue","slategray","snow","springgreen","steelblue","tan","thistle","tomato","turquoise","violet","wheat","whitesmoke","yellowgreen", # 常见英文停用词
    "哪些", "哪个", "什么", "怎样", "如何", # 疑问词
    "关系", "区别", "联系", "比较" # 逻辑连接词（非实体）
])

_RE_WS = re.compile(r"\s+")

def _safe_json_dump(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json_any(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_clean_content(item: Dict[str, Any]) -> str:
    """从 problem.json 的 detail/content 字段提取用于向量化的文本。"""
    content = item.get("content")
    if content:
        return str(content)

    detail = item.get("detail", "")
    if isinstance(detail, str):
        detail = detail.strip()
        if not detail:
            return ""
        detail_dict = json.loads(detail)
        detail = detail_dict
    if isinstance(detail, dict):
        text = detail.get("content") or detail.get("title") or detail.get("body") or ""
        return str(text)
    return ""


def load_problems(path: str) -> Tuple[List[str], List[str], Dict[str, Dict[str, Any]]]:
    """读取 problem.json 并返回：problem_ids, raw_texts, problems_map。"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    data = load_json_any(path)
    if isinstance(data, dict):
        data = [data]

    pids: List[str] = []
    texts: List[str] = []
    problems: Dict[str, Dict[str, Any]] = {}
    for p in data:
        pid = str(p.get("problem_id", p.get("id", "")))
        if not pid:
            continue
        content = extract_clean_content(p)
        concepts = p.get("concepts", []) or []
        if concepts:
            valid_concepts = [str(c) for c in concepts if c and str(c) not in CUSTOM_STOP_WORDS]
            if valid_concepts:
                content = content + " " + " ".join(valid_concepts)
        pids.append(pid)
        texts.append(content)
        problems[pid] = p

    print(f"Loaded {len(pids)} problems.")
    return pids, texts, problems


def _jieba_tokenizer(text: str) -> List[str]:
    return [w for w in jieba.lcut(text) if len(w) > 1 and w not in CUSTOM_STOP_WORDS]


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    s = float(ex.sum())
    if s <= 0:
        return np.ones_like(x) / max(1, x.shape[0])
    return ex / s


def build_ctfidf_top_token(
    *,
    cluster_texts: Sequence[str],
    all_texts: Sequence[str],
    cluster_labels: np.ndarray,
    cluster_id: int,
    stopwords: set,
    max_features: int = 5000,
) -> str:
    """
    近似实现论文 c-TF-IDF：对每个“类/簇”独立选择权重最高的 token 作为 label。
    采用：
      tf_c(w) = f_{w,c} / |c|
      idf(w) = log(1 + N / df(w))
      ctfidf(w,c) = tf_c(w) * idf(w)
    """
    # 需要在全量上拟合一次 vocab；这里用简化实现：直接用所有文本拟合
    vectorizer = CountVectorizer(
        tokenizer=_jieba_tokenizer,
        max_features=max_features,
        token_pattern=None,
    )
    dtm_all = vectorizer.fit_transform(all_texts)
    vocab = vectorizer.get_feature_names_out()

    N = int(np.max(cluster_labels) + 1)
    # cluster token sum
    mask = cluster_labels == cluster_id
    if not np.any(mask):
        return "misc"

    dtm_c = dtm_all[mask]
    f_w_c = np.asarray(dtm_c.sum(axis=0)).ravel()
    n_c_docs = max(1, int(mask.sum()))
    tf = f_w_c / float(n_c_docs)

    # df(w): token 出现在多少簇
    df = np.zeros_like(f_w_c, dtype=np.float64)
    for cid in range(N):
        m2 = cluster_labels == cid
        if not np.any(m2):
            continue
        f_w_cd = np.asarray(dtm_all[m2].sum(axis=0)).ravel()
        df += (f_w_cd > 0).astype(np.float64)

    idf = np.log(1.0 + float(N) / np.maximum(df, 1.0))
    ctfidf = tf * idf

    # 选 top token
    idx_sorted = np.argsort(ctfidf)[::-1]
    for idx in idx_sorted[:20]:
        tok = str(vocab[idx])
        if tok and tok not in stopwords:
            if len(tok) > 0:
                return tok
    return "misc"


def pick_top_token_from_ctfidf(ctfidf: np.ndarray, vocab: np.ndarray, stopwords: set, topn: int = 1) -> str:
    """从 c-TF-IDF 权重中取最高权重 token（做停用词过滤）。"""
    idx_sorted = np.argsort(ctfidf)[::-1]
    for idx in idx_sorted[: max(1, int(topn) * 20)]:
        tok = str(vocab[int(idx)])
        if tok and tok not in stopwords:
            return tok
    return "misc"


def estimate_rasch_mu_q(
    *,
    student_json: str,
    problems: Dict[str, Dict[str, Any]],
    lr_mu_lambda: float,
    lr_theta_lambda: float,
    epochs: int = 6,
    lr: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """
    正则化 Rasch：P(r=1|u,q)=sigmoid(theta_u - mu_q)
    返回 mu_q（困难度），并做去均值平移以去除可辨识性问题。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    raw = load_json_any(student_json)
    if isinstance(raw, dict):
        # 常见包裹
        for k in ["data", "students", "records", "logs"]:
            if isinstance(raw.get(k), list):
                raw = raw[k]
                break
    if not isinstance(raw, list):
        raise ValueError("student_json 解析失败：期望 list 结构。")

    # 收集 (u,q,y)
    user_ids: List[str] = []
    u2idx: Dict[str, int] = {}
    q2idx: Dict[str, int] = {}
    records: List[Tuple[int, int, float]] = []

    for stu in raw:
        seq = stu.get("seq", []) or []
        uid = stu.get("user_id")
        if not uid and seq and isinstance(seq[0], dict):
            uid = seq[0].get("user_id")
        if not uid or not isinstance(seq, list):
            continue
        uid = str(uid)
        if uid not in u2idx:
            u2idx[uid] = len(u2idx)
            user_ids.append(uid)

        for log in seq:
            if not isinstance(log, dict):
                continue
            pid = str(log.get("problem_id", ""))
            if not pid or pid not in problems:
                continue
            if pid not in q2idx:
                q2idx[pid] = len(q2idx)
            u_idx = u2idx[uid]
            q_idx = q2idx[pid]
            y = log.get("is_correct", None)
            if y is None:
                continue
            y_f = float(int(y))
            records.append((u_idx, q_idx, y_f))

    if not records:
        raise ValueError("student_json 中没有可用于 Rasch 的交互记录。")

    U = len(u2idx)
    Q = len(q2idx)
    print(f"[Rasch] samples={len(records)} | users={U} | questions={Q}")

    u_idx = torch.tensor([r[0] for r in records], dtype=torch.long)
    q_idx = torch.tensor([r[1] for r in records], dtype=torch.long)
    y = torch.tensor([r[2] for r in records], dtype=torch.float32)

    theta = torch.zeros((U,), dtype=torch.float32, requires_grad=True)
    mu = torch.zeros((Q,), dtype=torch.float32, requires_grad=True)

    opt = torch.optim.Adam([theta, mu], lr=lr)
    bce = torch.nn.BCEWithLogitsLoss(reduction="mean")

    for ep in range(epochs):
        opt.zero_grad()
        logits = theta[u_idx] - mu[q_idx]
        loss_nll = bce(logits, y)
        loss_reg = lr_theta_lambda * (theta * theta).mean() + lr_mu_lambda * (mu * mu).mean()
        loss = loss_nll + loss_reg
        loss.backward()
        opt.step()

        # 去均值平移：保证 mu 的均值为 0（避免可辨识性漂移）
        with torch.no_grad():
            mu_mean = mu.mean()
            mu -= mu_mean

        print(f"[Rasch] epoch={ep+1}/{epochs} loss={float(loss):.6f}")

    # 映射回 pid
    idx2q = {idx: pid for pid, idx in q2idx.items()}
    out: Dict[str, float] = {}
    for q_i in range(Q):
        out[idx2q[q_i]] = float(mu[q_i].detach().cpu().item())
    return out


def build_concept_pc1_directions(
    *,
    problems: Dict[str, Dict[str, Any]],
    pids: List[str],
    embeddings: np.ndarray,
    global_mean: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    对每个 concept 计算 PC1 作为 d_c，返回向量（未注入 mu 的基方向）。
    """
    concept2idxs: Dict[str, List[int]] = defaultdict(list)
    concept2levels: Dict[str, List[int]] = defaultdict(list)

    pid2pos = {pid: i for i, pid in enumerate(pids)}
    for pid, p in problems.items():
        if pid not in pid2pos:
            continue
        i = pid2pos[pid]
        lvl = p.get("cognitive_dimension", 0)
        lvl_i = int(float(lvl))
        concepts = p.get("concepts", []) or []
        for c in concepts:
            if c is None:
                continue
            cs = str(c)
            if not cs:
                continue
            concept2idxs[cs].append(i)
            concept2levels[cs].append(lvl_i)

    out: Dict[str, np.ndarray] = {}
    for c, idxs in tqdm(concept2idxs.items(), desc="PCA d_c(q) directions"):
        if len(idxs) == 1:
            vec = embeddings[idxs[0]] - global_mean
            norm = float(np.linalg.norm(vec))
            out[c] = (vec / norm).astype(np.float32) if norm > 0 else np.zeros_like(global_mean, dtype=np.float32)
            continue

        X = embeddings[idxs]
        # PCA 默认会做中心化；我们也显式做防止数值漂移
        Xc = X - X.mean(axis=0, keepdims=True)
        pca = PCA(n_components=1, random_state=42)
        pca.fit(Xc)
        pc1 = pca.components_[0].astype(np.float32)

        # 方向对齐：让“高认知层级平均投影 > 低层级平均投影”
        lvls = concept2levels[c]
        lvls_arr = np.asarray(lvls, dtype=int)
        proj = Xc @ pc1
        uniq_lvls = sorted(set(lvls_arr.tolist()))
        if len(uniq_lvls) >= 2:
            low = uniq_lvls[0]
            high = uniq_lvls[-1]
            mean_low = float(proj[lvls_arr == low].mean()) if np.any(lvls_arr == low) else 0.0
            mean_high = float(proj[lvls_arr == high].mean()) if np.any(lvls_arr == high) else 0.0
            if mean_high < mean_low:
                pc1 = -pc1

        out[c] = pc1

    return out


def build_d_c_q_for_pids(
    *,
    problems: Dict[str, Dict[str, Any]],
    pids: List[str],
    concept_pc1: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    对每个题目 q：d_c(q) = normalize(平均_{c in concepts(q)} d_c )
    """
    D = next(iter(concept_pc1.values())).shape[0] if concept_pc1 else 0
    out = np.zeros((len(pids), D), dtype=np.float32)
    pid2pos = {pid: i for i, pid in enumerate(pids)}

    for pid, p in problems.items():
        if pid not in pid2pos:
            continue
        i = pid2pos[pid]
        concepts = p.get("concepts", []) or []
        vecs = []
        for c in concepts:
            if c is None:
                continue
            cs = str(c)
            if cs in concept_pc1:
                vecs.append(concept_pc1[cs])
        if not vecs:
            continue
        v = np.mean(np.stack(vecs, axis=0), axis=0)
        norm = float(np.linalg.norm(v))
        out[i] = (v / norm).astype(np.float32) if norm > 0 else np.zeros((D,), dtype=np.float32)
    return out


def load_student_records_list(path: str) -> List[Dict[str, Any]]:
    raw = load_json_any(path)
    if isinstance(raw, dict):
        for k in ["data", "students", "records", "logs"]:
            if isinstance(raw.get(k), list):
                raw = raw[k]
                break
    if not isinstance(raw, list):
        raise ValueError("student_json 解析失败：期望 list 结构。")
    return raw


def reservoir_sample_student_time(
    records: List[Dict[str, Any]],
    problems: Dict[str, Dict[str, Any]],
    *,
    lr_max_pairs: int,
    seed: int,
) -> List[Tuple[int, int]]:
    """
    从所有 (student_idx, t) 中均匀抽样至 lr_max_pairs。
    只抽 t>=1 的交互，保证 history 非空（否则 e_diag 退化为常数）。
    """
    rng = random.Random(seed)
    reservoir: List[Tuple[int, int]] = []
    n_seen = 0

    for s_idx, stu in enumerate(records):
        seq = stu.get("seq", []) or []
        if not isinstance(seq, list):
            continue
        # t in [1..len(seq)-1]
        for t in range(1, len(seq)):
            log = seq[t]
            if not isinstance(log, dict):
                continue
            pid = str(log.get("problem_id", ""))
            if not pid or pid not in problems:
                continue

            n_seen += 1
            if len(reservoir) < lr_max_pairs:
                reservoir.append((s_idx, t))
            else:
                j = rng.randrange(0, n_seen)
                if j < lr_max_pairs:
                    reservoir[j] = (s_idx, t)

    print(f"[LR sample] seen={n_seen} sampled={len(reservoir)}")
    return reservoir


def compute_e_diag_for_sample(
    *,
    stu_seq: Sequence[Dict[str, Any]],
    t: int,
    W: int,
    pid2vec: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    e_diag：Target Attention Pooling
      query = e_t^sem
      keys = {e_k^sem | k in [t-W, t-1]}
      values = (2*r_k-1) * e_k^sem
      alpha = softmax(dot(query, keys))
      e_diag = Σ alpha_k * value_k
    """
    window_start = max(0, t - W)
    hist = stu_seq[window_start:t]
    if not hist:
        # 维度取决于向量表
        # 这里使用第一个可用向量长度兜底
        any_vec = next(iter(pid2vec.values()))
        return np.zeros_like(any_vec, dtype=np.float32)

    target_log = stu_seq[t]
    target_pid = str(target_log.get("problem_id", ""))
    q = pid2vec.get(target_pid)
    if q is None:
        any_vec = next(iter(pid2vec.values()))
        return np.zeros_like(any_vec, dtype=np.float32)

    keys: List[np.ndarray] = []
    values: List[np.ndarray] = []
    for lk in hist:
        if not isinstance(lk, dict):
            continue
        pid = str(lk.get("problem_id", ""))
        if pid not in pid2vec:
            continue
        vec = pid2vec[pid]
        r = lk.get("is_correct", 0)
        r_i = int(r)
        sign = 2 * r_i - 1  # 0 -> -1, 1 -> +1
        keys.append(vec)
        values.append(vec * float(sign))

    if not keys:
        return np.zeros_like(q, dtype=np.float32)

    K = np.stack(keys, axis=0)  # [L, D]
    V = np.stack(values, axis=0)  # [L, D]

    scores = K @ q.astype(np.float32)  # dot
    alpha = softmax(scores)
    e_diag = alpha.reshape(-1, 1) * V
    return e_diag.sum(axis=0).astype(np.float32)


def train_e_diag_lr(
    *,
    student_records: List[Dict[str, Any]],
    problems: Dict[str, Dict[str, Any]],
    pid2vec_sem: Dict[str, np.ndarray],
    W: int,
    beta_d: float,
    lr_max_pairs: int,
    lr_seed: int,
    batch_size: int,
    out_path: str,
) -> Dict[str, Any]:
    """
    用 e_diag 作为特征训练极轻量 LR：sigmoid(w_d^T e_diag + b_d)
    不参与后续端到端训练（只是用于计算 M_i）。
    """
    # 抽样 (student_idx, t)
    samples = reservoir_sample_student_time(
        student_records,
        problems,
        lr_max_pairs=lr_max_pairs,
        seed=lr_seed,
    )

    # SGD 版本的逻辑回归（log loss）
    # 目标：fit sigmoid，得到 w_d/b_d
    D = next(iter(pid2vec_sem.values())).shape[0]
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=1,
        tol=None,
        learning_rate="optimal",
        random_state=lr_seed,
        fit_intercept=True,
    )

    classes = np.array([0, 1], dtype=np.int64)
    first = True

    # 为了避免一次性存 20 万 * D 的特征，按 batch 逐步训练
    rng = random.Random(lr_seed)
    rng.shuffle(samples)

    for start in tqdm(range(0, len(samples), batch_size), desc="Training e_diag LR"):
        batch = samples[start:start + batch_size]
        X = np.zeros((len(batch), D), dtype=np.float32)
        y = np.zeros((len(batch),), dtype=np.int64)

        for i, (s_idx, t) in enumerate(batch):
            stu_seq = student_records[s_idx].get("seq", []) or []
            if not isinstance(stu_seq, list) or t >= len(stu_seq):
                continue
            e_diag = compute_e_diag_for_sample(
                stu_seq=stu_seq,
                t=t,
                W=W,
                pid2vec=pid2vec_sem,
            )
            X[i] = e_diag
            y_val = stu_seq[t].get("is_correct", 0)
            y[i] = int(y_val)

        if first:
            clf.partial_fit(X, y, classes=classes)
            first = False
        else:
            clf.partial_fit(X, y)

    w_d = clf.coef_[0].astype(np.float32).tolist()
    b_d = float(clf.intercept_[0])

    params = {
        "W": int(W),
        "beta_d": float(beta_d),
        "embedding_dim": int(D),
        "w_d": w_d,
        "b_d": b_d,
        "lr_max_pairs": int(lr_max_pairs),
        "lr_seed": int(lr_seed),
        "model": "SGDClassifier(log_loss)",
    }
    _safe_json_dump(params, out_path)
    print(f"[OK] e_diag LR params saved to {out_path}")
    return params


def build_collaborative_signals(
    *,
    student_json: str,
    problems: Dict[str, Dict[str, Any]],
    topk: int,
    vector_size: int,
) -> Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]:
    """
    Item2Vec/Word2Vec：以学生作答序列作为句子训练。
    返回：
      - item_collaborative.json: pid -> neighbor pid list
      - item_collaborative_embeddings.pkl: pid -> vector
    """
    records = load_student_records_list(student_json)
    sentences: List[List[str]] = []
    for rec in records:
        seq = rec.get("seq", []) or []
        if not isinstance(seq, list) or len(seq) < 2:
            continue
        tokens: List[str] = []
        for log in seq:
            if not isinstance(log, dict):
                continue
            pid = str(log.get("problem_id", ""))
            if pid and pid in problems:
                tokens.append(pid)
        if len(tokens) >= 2:
            sentences.append(tokens)

    if not sentences:
        return {}, {}

    print(f"Training Word2Vec for collaborative signals: sentences={len(sentences)}")
    model = Word2Vec(
        sentences=sentences,
        vector_size=int(vector_size),
        window=5,
        min_count=1,
        sg=1,
        workers=4,
        epochs=5,
    )

    collab_neighbors: Dict[str, List[str]] = {}
    collab_vecs: Dict[str, np.ndarray] = {}
    for pid in tqdm(list(model.wv.index_to_key), desc="Building collaborative maps"):
        collab_vecs[pid] = model.wv[pid].astype(np.float32)
        try:
            similar_items = model.wv.most_similar(pid, topn=topk)
        except KeyError:
            collab_neighbors[pid] = []
            continue
        neighbors = [item for item, _ in similar_items if item != pid][:topk]
        collab_neighbors[pid] = neighbors
    return collab_neighbors, collab_vecs


def build_hierarchical_semantic_ids_and_vectors(
    *,
    problem_ids: List[str],
    problem_texts: List[str],
    problems: Dict[str, Dict[str, Any]],
    top_k: int,
    sub_k: int,
    random_state: int,
    pid2mu_q: Dict[str, float],
    concept_pc1: Dict[str, np.ndarray],
    pid2d_c_q: np.ndarray,
) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """
    对调整后的语义向量 z'_q 进行两层 KMeans + c-TF-IDF 命名，输出：
      - pid -> semantic_id
      - pid -> z'_q（adjusted semantic vector）
    """
    pid2pos = {pid: i for i, pid in enumerate(problem_ids)}
    D = int(pid2d_c_q.shape[-1])

    # base embeddings = 从概念注入前由调用者缓存；此处用 d_c_q 注入需由外部提供 adjusted_sem
    # 由于我们在主流程里已计算 base embeddings，这里直接通过参数重建 adjusted 语义向量：
    # 为避免重复计算，我们把 base embeddings 也当作在外部完成的 step：调整后语义向量由外部构建并传入 d_c_q + mu
    # 这里按约定 pid2d_c_q 是 normalize(average d_c)；我们在外部将 base embeddings 与 mu 注入组合后再传给此函数。
    # 所以本函数只做聚类与命名。
    raise NotImplementedError("此函数在主流程中实现。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem_json", default="problem.json")
    ap.add_argument("--student_json", default="student-problem-fine.json")

    ap.add_argument("--out_semantic_ids", default="item_semantic_ids.json")
    ap.add_argument("--out_semantic_vectors", default="item_semantic_vectors.pkl")
    ap.add_argument("--out_mu_q", default="problem_mu_q.json")
    ap.add_argument("--out_concept_pc1", default="concept_pc1_dirs.pkl")
    ap.add_argument("--out_lr_params", default="e_diag_lr_params.json")

    ap.add_argument("--out_collab", default="item_collaborative.json")
    ap.add_argument("--out_collab_vecs", default="item_collaborative_embeddings.pkl")

    # 聚类参数
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--sub_k", type=int, default=5)

    # c-TF-IDF tokenizer vocab 大小（离线一次拟合 vocab）
    ap.add_argument("--ctfidf_max_features", type=int, default=5000)

    # Rasch 参数（稀疏鲁棒）
    ap.add_argument("--rasch_epochs", type=int, default=6)
    ap.add_argument("--rasch_lr", type=float, default=0.05)
    ap.add_argument("--rasch_lambda_mu", type=float, default=1.0)
    ap.add_argument("--rasch_lambda_theta", type=float, default=0.1)

    # d_c(q) direction 注入与诊断LR参数
    ap.add_argument("--d_pca_seed", type=int, default=42)
    ap.add_argument("--W", type=int, default=10)
    ap.add_argument("--beta_d", type=float, default=0.5)
    ap.add_argument("--lr_max_pairs", type=int, default=200000)
    ap.add_argument("--lr_seed", type=int, default=42)
    ap.add_argument("--lr_batch_size", type=int, default=2048)

    # 协同信号
    ap.add_argument("--collab_topk", type=int, default=5)
    ap.add_argument("--collab_vec_dim", type=int, default=64)

    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    # 1) load problems
    pids, texts, problems = load_problems(args.problem_json)
    if not pids:
        raise SystemExit("problem_json 中未加载到任何题目。")

    # 2) embed base semantic vectors z_q^sem
    print("Loading BGE model ...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Encoding problems: n={len(texts)}")
    base_embs = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    global_mean = base_embs.mean(axis=0)

    # 3) estimate Rasch mu_q
    pid2mu = estimate_rasch_mu_q(
        student_json=args.student_json,
        problems=problems,
        lr_mu_lambda=float(args.rasch_lambda_mu),
        lr_theta_lambda=float(args.rasch_lambda_theta),
        epochs=int(args.rasch_epochs),
        lr=float(args.rasch_lr),
        seed=int(args.random_state),
    )

    # 4) build concept PC1 directions d_c
    concept_pc1 = build_concept_pc1_directions(
        problems=problems,
        pids=pids,
        embeddings=base_embs,
        global_mean=global_mean,
    )

    # 5) build d_c(q) for each question
    pid2d_c_q = build_d_c_q_for_pids(
        problems=problems,
        pids=pids,
        concept_pc1=concept_pc1,
    )

    # 6) inject: z'_q = z_q + mu_q * d_c(q)
    adjusted_embs = np.zeros_like(base_embs, dtype=np.float32)
    for i, pid in enumerate(pids):
        mu = float(pid2mu.get(pid, 0.0))
        adjusted_embs[i] = base_embs[i] + mu * pid2d_c_q[i]

    # save semantic vectors for downstream (e_diag / A_i cos / etc.)
    pid2sem_vec: Dict[str, np.ndarray] = {pid: adjusted_embs[i].astype(np.float32) for i, pid in enumerate(pids)}
    import pickle
    with open(args.out_semantic_vectors, "wb") as f:
        pickle.dump(pid2sem_vec, f, protocol=pickle.HIGHEST_PROTOCOL)
    _safe_json_dump(pid2mu, args.out_mu_q)
    with open(args.out_concept_pc1, "wb") as f:
        pickle.dump(concept_pc1, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("[OK] semantic vectors / mu_q / concept_pc1 saved.")

    # 7) hierarchical semantic IDs with 2-layer KMeans + c-TF-IDF naming
    D = adjusted_embs.shape[-1]
    k1 = min(int(args.top_k), adjusted_embs.shape[0])
    kmeans1 = KMeans(n_clusters=k1, random_state=int(args.random_state), n_init=10)
    print("Running top-level KMeans ...")
    labels1 = kmeans1.fit_predict(adjusted_embs)

    # c-TF-IDF vocab + dtm 只拟合一次，然后在宏/子簇内复用
    print("Prepare c-TF-IDF vocab/dtm ...")
    vectorizer = CountVectorizer(
        tokenizer=_jieba_tokenizer,
        max_features=int(args.ctfidf_max_features),
        token_pattern=None,
    )
    dtm_all = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()

    # macro clusters: 预计算每个簇的 token sums 与 df
    print("Naming macro clusters with c-TF-IDF ...")
    V = int(len(vocab))
    macro_token_sums = np.zeros((k1, V), dtype=np.float64)
    macro_n_docs = np.zeros((k1,), dtype=np.float64)
    for cid in range(k1):
        mask = labels1 == cid
        if not np.any(mask):
            continue
        macro_n_docs[cid] = float(mask.sum())
        macro_token_sums[cid] = np.asarray(dtm_all[mask].sum(axis=0)).ravel().astype(np.float64)

    macro_token_df = (macro_token_sums > 0).sum(axis=0).astype(np.float64)
    macro_N = float(k1)

    macro_labels: Dict[int, str] = {}
    idf_macro = np.log(1.0 + macro_N / np.maximum(macro_token_df, 1.0))
    for cid in tqdm(range(k1), desc="macro labels"):
        if macro_n_docs[cid] <= 0:
            macro_labels[cid] = "misc"
            continue
        tf = macro_token_sums[cid] / macro_n_docs[cid]
        ctfidf = tf * idf_macro
        macro_labels[cid] = pick_top_token_from_ctfidf(
            ctfidf=ctfidf,
            vocab=vocab,
            stopwords=CUSTOM_STOP_WORDS,
            topn=1,
        )

    pid2semantic_id: Dict[str, str] = {}
    print("Running sub-cluster KMeans + naming ...")
    for cid in tqdm(range(k1), desc="sub clusters"):
        idxs = np.where(labels1 == cid)[0]
        if len(idxs) == 0:
            continue
        sub_embs = adjusted_embs[idxs]
        sub_texts = [texts[i] for i in idxs]

        k2 = min(int(args.sub_k), sub_embs.shape[0])
        if k2 <= 1:
            sub_labels = np.zeros(len(idxs), dtype=int)
        else:
            kmeans2 = KMeans(n_clusters=k2, random_state=int(args.random_state), n_init=10)
            sub_labels = kmeans2.fit_predict(sub_embs)

        # 子簇命名：在当前宏簇内部，对 token df 进行重估（只统计该宏簇下的子簇集合）
        sub_label_tokens: Dict[int, str] = {}
        sub_dtm = dtm_all[idxs]  # 行子集
        sub_V = V

        sub_token_sums = np.zeros((k2, sub_V), dtype=np.float64)
        sub_n_docs = np.zeros((k2,), dtype=np.float64)
        for sc in range(k2):
            m2 = sub_labels == sc
            if not np.any(m2):
                continue
            sub_n_docs[sc] = float(m2.sum())
            sub_token_sums[sc] = np.asarray(sub_dtm[m2].sum(axis=0)).ravel().astype(np.float64)

        sub_token_df = (sub_token_sums > 0).sum(axis=0).astype(np.float64)
        sub_N = float(k2)
        idf_sub = np.log(1.0 + sub_N / np.maximum(sub_token_df, 1.0))

        for sc in range(k2):
            if sub_n_docs[sc] <= 0:
                sub_label_tokens[sc] = "misc"
                continue
            tf = sub_token_sums[sc] / sub_n_docs[sc]
            ctfidf = tf * idf_sub
            sub_label_tokens[sc] = pick_top_token_from_ctfidf(
                ctfidf=ctfidf,
                vocab=vocab,
                stopwords=CUSTOM_STOP_WORDS,
                topn=1,
            )

        top_label = macro_labels[cid]
        for local_pos, original_pos in enumerate(idxs):
            sc = int(sub_labels[local_pos])
            pid = pids[int(original_pos)]
            pid2semantic_id[pid] = f"{top_label}-{sub_label_tokens[sc]}"

    _safe_json_dump(pid2semantic_id, args.out_semantic_ids)
    print(f"[OK] item_semantic_ids saved to {args.out_semantic_ids}")

    # 8) train e_diag LR (w_d, b_d)
    student_records = load_student_records_list(args.student_json)
    # 仅对需要的样本做 e_diag 计算
    train_e_diag_lr(
        student_records=student_records,
        problems=problems,
        pid2vec_sem=pid2sem_vec,
        W=int(args.W),
        beta_d=float(args.beta_d),
        lr_max_pairs=int(args.lr_max_pairs),
        lr_seed=int(args.lr_seed),
        batch_size=int(args.lr_batch_size),
        out_path=args.out_lr_params,
    )

    # 9) collaborative signals
    collab_neighbors, collab_vecs = build_collaborative_signals(
        student_json=args.student_json,
        problems=problems,
        topk=int(args.collab_topk),
        vector_size=int(args.collab_vec_dim),
    )
    _safe_json_dump(collab_neighbors, args.out_collab)
    with open(args.out_collab_vecs, "wb") as f:
        import pickle
        pickle.dump(collab_vecs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[OK] item_collaborative saved to {args.out_collab}")


if __name__ == "__main__":
    main()