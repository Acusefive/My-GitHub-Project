from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (Any, Dict, Iterable, Iterator, List, Optional, Sequence,
                    Tuple)

import jieba
import numpy as np
import torch
import torch.nn.functional as F
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from .constants import (A_SCALE, COLLAB_WINDOW, CTFIDF_MAX_FEATURES, DS, EPS,
                        GRAPH_LLM_CANDIDATE_LIMIT, GRAPH_LLM_MAX_PREREQ,
                        GRAPH_LLM_MAX_RELATED, HISTORY_WINDOW, KGLOBAL, KLOCAL,
                        KMEANS_N_INIT, KMEANS_RANDOM_STATE, LAMBDA_L, LAMBDA_T,
                        LOCAL_COOCCUR_THRESHOLD, NEGATIVE_SAMPLES, RASCH_A,
                        RASCH_EPOCHS, RASCH_LAMBDA_MU, RASCH_LAMBDA_THETA,
                        RASCH_LR, SMOKE_MAX_PROBLEMS, SMOKE_MAX_STUDENTS,
                        SMOKE_MAX_TARGETS_PER_STUDENT, SMOKE_TRAIN_MAX_SAMPLES,
                        STOP_WORDS, TAU, TEXT_EMBED_BATCH_SIZE,
                        TEXT_EMBED_MAX_LENGTH, TEXT_EMBED_MODEL_NAME,
                        TRAIN_BATCH_SIZE, TRAIN_EARLY_STOP_PATIENCE,
                        TRAIN_GRAD_CLIP, TRAIN_LR, TRAIN_MAX_EPOCHS,
                        TRAIN_SEED, TRAIN_VAL_MOD, USE_RASCH_ENHANCEMENT)
from .io_utils import (ProblemRecord, StudentSequence, atomic_save_text,
                       dataclass_list_to_jsonl, ensure_dir,
                       load_problem_records, load_student_sequences,
                       pick_device, user_hash_bucket, write_json, write_jsonl)
from .llm_utils import (OpenAICompatibleGraphCompleter, append_json_cache,
                        load_json_cache)
from .models import StrictPriorModel
from .retrieval_models import QwenEmbeddingEncoder


@dataclass
class Stage32Result:
    priors_dir: str
    semantic_ids_path: str
    semantic_id_audit_path: str
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
class Stage32CoreArtifactsResult:
    priors_dir: str
    semantic_ids_path: str
    semantic_id_audit_path: str
    problem_catalog_path: str
    item_collaborative_path: str
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


SEMANTIC_EXTRA_STOP_WORDS = {
    "the",
    "of",
    "to",
    "in",
    "for",
    "with",
    "on",
    "at",
    "by",
    "from",
    "is",
    "nbsp",
    "amp",
    "lt",
    "gt",
    "div",
    "span",
    "class",
    "style",
    "width",
    "height",
    "mathjaxinline",
    "mathjaxdisplay",
    "pmatrix",
    "bmatrix",
    "vmatrix",
    "matrix",
    "begin",
    "end",
    "left",
    "right",
    "frac",
    "sqrt",
    "overline",
    "underline",
    "mathbf",
    "mathrm",
    "sumlimits",
    "myclass",
    "cookie",
    "blank",
    "填空",    "填空", "选择", "下列", "属于", "关于", "正确", "错误", "选项", "描述", "答案",
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
    "不同", "影响", "作用", "相关", "主要", "手段", "目的", "特点",
    "frac", "amp", "div", "span", "class", "style", "width", "height","ldquo","rdquo","sumlimits","&#","pmatrix","begin","end","matrix","left","right","left[","right]","rm","mu","theta", # HTML/LaTeX 垃圾
    "of", "to", "in", "for", "with", "on", "at", "by", "from","is","mathjaxinline","mathbf","red","black","blue","green","yellow","purple","orange","brown","gray","pink","lime","teal","indigo","violet","maroon","navy","olive","crimson","azure","fuchsia","chartreuse","coral","cyan","gold","silver","plum","salmon","tan","turquoise","wheat","ivory","khaki","lavender","magenta","mistyrose","navajowhite","oldlace","papayawhip","peachpuff","peru","pink","plum","powderblue","rosybrown","royalblue","sienna","skyblue","slateblue","slategray","snow","springgreen","steelblue","tan","thistle","tomato","turquoise","violet","wheat","whitesmoke","yellowgreen","black","blue","green","red","yellow","purple","orange","brown","gray","pink","lime","teal","indigo","violet","maroon","navy","olive","crimson","azure","fuchsia","chartreuse","coral","cyan","gold","silver","plum","salmon","tan","turquoise","wheat","ivory","khaki","lavender","magenta","mistyrose","navajowhite","oldlace","papayawhip","peachpuff","peru","pink","plum","powderblue","rosybrown","royalblue","sienna","skyblue","slateblue","slategray","snow","springgreen","steelblue","tan","thistle","tomato","turquoise","violet","wheat","whitesmoke","yellowgreen", # 常见英文停用词
    "哪些", "哪个", "什么", "怎样", "如何","哪项", # 疑问词
    "关系", "区别", "联系", "比较", "自己" , "吴起", "沈博阳", "地区", "恐龙", "next", "php", "完全", "第三", "万元", "eps", "noteexpress","ag","对否","结果","tree","quot","00","lambda","and","circuit","multiple","port","figure","dx","int","lim","infty",""# 逻辑连接词（非实体）
}

SEMANTIC_PLACEHOLDER_RE = re.compile(r"^_+$")
SEMANTIC_ALNUM_TAG_RE = re.compile(r"^[a-z]{1,3}\d{1,4}$", re.IGNORECASE)
SEMANTIC_STRIP_RE = re.compile(r"^[\s\-_`~!@#$%^&*+=|\\/:;\"'<>,.?()\[\]{}]+|[\s\-_`~!@#$%^&*+=|\\/:;\"'<>,.?()\[\]{}]+$")


def _normalize_semantic_token(token: str) -> str:
    token = str(token or "").strip()
    if not token:
        return ""
    token = token.replace("&nbsp;", "nbsp").replace("&#", "").strip()
    token = SEMANTIC_STRIP_RE.sub("", token)
    return token.strip()


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _is_semantic_noise(token: str) -> bool:
    normalized = _normalize_semantic_token(token)
    if not normalized:
        return True
    lowered = normalized.lower()
    if lowered == "misc":
        return True
    if normalized in STOP_WORDS or lowered in STOP_WORDS:
        return True
    if lowered in SEMANTIC_EXTRA_STOP_WORDS:
        return True
    if "____" in normalized or SEMANTIC_PLACEHOLDER_RE.fullmatch(normalized):
        return True
    if lowered.startswith("mathjax"):
        return True
    if lowered.endswith("inline") and lowered.startswith("math"):
        return True
    if SEMANTIC_ALNUM_TAG_RE.fullmatch(normalized) and not _has_cjk(normalized):
        return True
    return False


def _sanitize_semantic_token(token: str) -> str:
    normalized = _normalize_semantic_token(token)
    if _is_semantic_noise(normalized):
        return ""
    return normalized


def _extract_fallback_token(text: str) -> str:
    if not text:
        return ""
    for token in jieba.lcut(str(text)):
        cleaned = _sanitize_semantic_token(token)
        if cleaned:
            return cleaned
    cleaned = _sanitize_semantic_token(text)
    if cleaned:
        return cleaned
    return ""


def _semantic_fallback_token(problem: ProblemRecord, *, prefer_chapter: bool) -> str:
    candidates: List[str] = []
    if prefer_chapter:
        candidates.extend([problem.chapter, *problem.concepts, problem.title, problem.text])
    else:
        candidates.extend([*problem.concepts, problem.title, problem.text, problem.chapter])
    for candidate in candidates:
        fallback = _extract_fallback_token(candidate)
        if fallback:
            return fallback
    return "misc"


def _join_semantic_id_parts(macro: str, micro: str) -> str:
    parts: List[str] = []
    for token in [*str(macro).split("-"), *str(micro).split("-")]:
        cleaned = _sanitize_semantic_token(token)
        if not cleaned:
            continue
        if cleaned in parts:
            continue
        parts.append(cleaned)
    return "-".join(parts) if parts else "misc"


def _compose_semantic_id(problem: ProblemRecord, macro_label: str, micro_label: str) -> str:
    macro = _sanitize_semantic_token(macro_label) or _semantic_fallback_token(problem, prefer_chapter=True)
    micro = _sanitize_semantic_token(micro_label) or _semantic_fallback_token(problem, prefer_chapter=False)
    return _join_semantic_id_parts(macro, micro)


def _compose_semantic_id_with_stats(
    problem: ProblemRecord,
    macro_label: str,
    micro_label: str,
    stats: Counter,
) -> str:
    macro = _sanitize_semantic_token(macro_label)
    if not macro:
        stats["macro_fallback_count"] += 1
        macro = _semantic_fallback_token(problem, prefer_chapter=True)

    micro = _sanitize_semantic_token(micro_label)
    if not micro:
        stats["micro_fallback_count"] += 1
        micro = _semantic_fallback_token(problem, prefer_chapter=False)

    semantic_id = _join_semantic_id_parts(macro, micro)
    if semantic_id == macro:
        stats["duplicate_collapsed_count"] += 1
    return semantic_id


def _jieba_tokenizer(text: str) -> List[str]:
    tokens: List[str] = []
    for token in jieba.lcut(text):
        cleaned = _sanitize_semantic_token(token)
        if len(cleaned) > 1:
            tokens.append(cleaned)
    return tokens


def _semantic_cluster_text(problem: ProblemRecord) -> str:
    parts = [problem.text or problem.title or problem.problem_id]
    clean_concepts = [concept for concept in problem.concepts if _sanitize_semantic_token(concept)]
    if clean_concepts:
        parts.append(" ".join(clean_concepts))
    return " ".join(part for part in parts if part).strip()


def canonicalize_cluster_labels(labels: np.ndarray) -> np.ndarray:
    unique_labels = sorted(int(label) for label in np.unique(labels))
    ordered_groups: List[Tuple[Tuple[int, ...], int]] = []
    for old_label in unique_labels:
        members = tuple(int(idx) for idx in np.where(labels == old_label)[0].tolist())
        ordered_groups.append((members, old_label))
    ordered_groups.sort()
    label_map = {old_label: new_label for new_label, (_members, old_label) in enumerate(ordered_groups)}
    return np.asarray([label_map[int(label)] for label in labels], dtype=np.int64)


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
    k = int(labels.max()) + 1 if labels.size else 0
    label_names: Dict[int, str] = {cid: "misc" for cid in range(k)}
    if k <= 0:
        return label_names
    vectorizer = CountVectorizer(
        tokenizer=_jieba_tokenizer,
        max_features=max_features,
        token_pattern=None,
    )
    try:
        dtm_all = vectorizer.fit_transform(texts)
    except ValueError:
        return label_names
    vocab = vectorizer.get_feature_names_out()

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
    candidates: List[Tuple[float, str]] = []
    for idx, score in enumerate(ctfidf.tolist()):
        token = str(vocab[int(idx)])
        token = _sanitize_semantic_token(token)
        if not token:
            continue
        candidates.append((float(score), token))
    if candidates:
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][1]
    return "misc"


def extract_cluster_keywords(texts: Sequence[str], *, top_n: int = 2, max_features: int = 50) -> str:
    if not texts:
        return "misc"
    vectorizer = CountVectorizer(
        tokenizer=_jieba_tokenizer,
        max_features=max_features,
        token_pattern=None,
    )
    try:
        dtm = vectorizer.fit_transform(texts)
    except ValueError:
        return "misc"
    vocab = vectorizer.get_feature_names_out()
    freqs = np.asarray(dtm.sum(axis=0)).ravel()
    keywords: List[str] = []
    for idx in freqs.argsort()[::-1]:
        token = _sanitize_semantic_token(str(vocab[int(idx)]))
        if token and token not in keywords:
            keywords.append(token)
        if len(keywords) >= top_n:
            break
    return "-".join(keywords) if keywords else "misc"


def compute_cluster_frequency_labels(
    texts: Sequence[str],
    labels: np.ndarray,
    *,
    top_n: int,
) -> Dict[int, str]:
    k = int(labels.max()) + 1 if labels.size else 0
    label_names: Dict[int, str] = {cid: "misc" for cid in range(k)}
    for cid in range(k):
        idxs = np.where(labels == cid)[0]
        if len(idxs) == 0:
            continue
        label_names[cid] = extract_cluster_keywords(
            [texts[int(idx)] for idx in idxs],
            top_n=top_n,
        )
    return label_names


def build_semantic_id_audit_report(
    problem_records: Sequence[ProblemRecord],
    semantic_ids: Dict[str, str],
    generation_stats: Counter,
) -> Dict[str, Any]:
    problem_map = {problem.problem_id: problem for problem in problem_records}
    category_counts: Counter = Counter()
    suspicious_tokens: Counter = Counter()
    flagged_samples: List[Dict[str, Any]] = []
    flagged_problem_count = 0

    for pid, semantic_id in semantic_ids.items():
        token_parts = [part.strip() for part in str(semantic_id).split("-") if part.strip()]
        flags: List[str] = []
        if not token_parts:
            flags.append("empty")
        if len(token_parts) == 1:
            category_counts["single_token_ids"] += 1
        else:
            category_counts["pair_ids"] += 1

        if len(token_parts) >= 2 and len({part.lower() for part in token_parts}) == 1:
            flags.append("duplicate_tokens")

        for token in token_parts:
            if _is_semantic_noise(token):
                flags.append("contains_noise")
                suspicious_tokens[token] += 1

        if flags:
            flagged_problem_count += 1
            unique_flags = sorted(set(flags))
            category_counts.update(unique_flags)
            if len(flagged_samples) < 50:
                problem = problem_map.get(pid)
                flagged_samples.append(
                    {
                        "problem_id": pid,
                        "semantic_id": semantic_id,
                        "flags": unique_flags,
                        "chapter": problem.chapter if problem else "",
                        "concepts": problem.concepts if problem else [],
                    }
                )

    total_ids = len(semantic_ids)
    return {
        "total_ids": total_ids,
        "flagged_ids": int(flagged_problem_count),
        "flagged_ratio": round(float(flagged_problem_count) / float(max(total_ids, 1)), 4),
        "category_counts": dict(category_counts),
        "generation_stats": dict(generation_stats),
        "top_suspicious_tokens": suspicious_tokens.most_common(50),
        "sample_flagged": flagged_samples,
    }


def build_semantic_ids(
    problem_records: Sequence[ProblemRecord],
    text_vectors: np.ndarray,
    *,
    semantic_ids_path: Path,
) -> Tuple[Dict[str, str], List[str], Dict[str, Any]]:
    problem_ids = [problem.problem_id for problem in problem_records]
    texts = [_semantic_cluster_text(problem) for problem in problem_records]
    k1 = min(KGLOBAL, len(problem_ids))
    km1 = KMeans(n_clusters=k1, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT)
    labels1 = canonicalize_cluster_labels(km1.fit_predict(text_vectors))
    macro_labels = compute_cluster_frequency_labels(texts, labels1, top_n=1)

    semantic_ids: Dict[str, str] = {}
    semantic_texts: List[str] = []
    generation_stats: Counter = Counter()
    generation_stats["problem_count"] = len(problem_records)
    generation_stats["macro_cluster_count"] = k1
    for cid in tqdm(range(k1), desc="strict semantic ids"):
        idxs = np.where(labels1 == cid)[0]
        if len(idxs) == 0:
            continue
        k2 = min(KLOCAL, len(idxs))
        generation_stats["micro_cluster_total"] += k2
        sub_vectors = text_vectors[idxs]
        if k2 <= 1:
            labels2 = np.zeros((len(idxs),), dtype=np.int64)
        else:
            km2 = KMeans(n_clusters=k2, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT)
            labels2 = canonicalize_cluster_labels(km2.fit_predict(sub_vectors))
        sub_texts = [texts[int(idx)] for idx in idxs]
        micro_labels = compute_cluster_frequency_labels(sub_texts, labels2, top_n=2)
        for local_pos, global_idx in enumerate(idxs):
            micro_label = micro_labels[int(labels2[local_pos])]
            problem = problem_records[int(global_idx)]
            semantic_id = _compose_semantic_id_with_stats(problem, macro_labels[cid], micro_label, generation_stats)
            semantic_ids[problem_ids[int(global_idx)]] = semantic_id
            semantic_texts.append(semantic_id)

    write_json(semantic_ids, semantic_ids_path)
    audit_report = build_semantic_id_audit_report(problem_records, semantic_ids, generation_stats)
    return semantic_ids, [semantic_ids[pid] for pid in problem_ids], audit_report


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
    *,
    concept_groups: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> Dict[str, np.ndarray]:
    if concept_groups is None:
        concept_groups = build_concept_groups(problem_records)

    directions: Dict[str, np.ndarray] = {}
    for concept, group in tqdm(concept_groups.items(), desc="strict concept pc1"):
        positions = group["positions"]
        concept_levels = group["levels"]
        if len(positions) == 1:
            vec = eqbase_vectors[positions[0]]
            norm = float(np.linalg.norm(vec))
            directions[concept] = (vec / norm).astype(np.float32) if norm > 0 else np.zeros((DS,), dtype=np.float32)
            continue
        matrix = torch.tensor(eqbase_vectors[positions], dtype=torch.float32)
        centered = matrix - torch.mean(matrix, dim=0, keepdim=True)
        _, _, v_h = torch.linalg.svd(centered, full_matrices=False)
        pc1 = v_h[0].to(torch.float32)
        levels_t = torch.tensor(concept_levels.tolist(), dtype=torch.int64)
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


def build_concept_groups(problem_records: Sequence[ProblemRecord]) -> Dict[str, Dict[str, np.ndarray]]:
    concept_to_positions: Dict[str, List[int]] = defaultdict(list)
    concept_to_levels: Dict[str, List[int]] = defaultdict(list)
    for pos, problem in enumerate(problem_records):
        for concept in problem.concepts:
            concept_to_positions[concept].append(pos)
            concept_to_levels[concept].append(problem.cognitive_dimension)
    groups: Dict[str, Dict[str, np.ndarray]] = {}
    for concept in sorted(concept_to_positions):
        groups[concept] = {
            "positions": np.asarray(concept_to_positions[concept], dtype=np.int64),
            "levels": np.asarray(concept_to_levels[concept], dtype=np.int64),
        }
    return groups


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
    semantic_ids: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]:
    valid_set = set(valid_problem_ids)
    neighbors = build_jaccard_collaborative_neighbors(
        student_sequences,
        valid_problem_ids,
        topk=5,
        semantic_ids=semantic_ids,
    )
    sentences: List[List[str]] = []
    for sequence in student_sequences:
        tokens = [str(log.get("problem_id") or "") for log in sequence.seq if str(log.get("problem_id") or "") in valid_set]
        if len(tokens) >= 2:
            sentences.append(tokens)
    if not sentences:
        return neighbors, {}

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
    vectors: Dict[str, np.ndarray] = {}
    for pid in model.wv.index_to_key:
        vectors[pid] = model.wv[pid].astype(np.float32)
    return neighbors, vectors


def build_jaccard_collaborative_neighbors(
    student_sequences: Sequence[StudentSequence],
    valid_problem_ids: Sequence[str],
    *,
    topk: int = 5,
    max_items_per_user: int = 200,
    semantic_ids: Optional[Dict[str, str]] = None,
) -> Dict[str, List[str]]:
    valid_set = set(valid_problem_ids)
    item_user_cnt: Counter = Counter()
    neigh_inter_cnt: Dict[str, Counter] = defaultdict(Counter)

    for sequence in tqdm(student_sequences, desc="strict collaborative jaccard"):
        problems: List[str] = []
        seen: set[str] = set()
        for log in sequence.seq:
            pid = str(log.get("problem_id") or "")
            if not pid or pid not in valid_set or pid in seen:
                continue
            seen.add(pid)
            problems.append(pid)
            if len(problems) >= max_items_per_user:
                break

        for pid in problems:
            item_user_cnt[pid] += 1
        for i in range(len(problems)):
            for j in range(i + 1, len(problems)):
                left, right = problems[i], problems[j]
                neigh_inter_cnt[left][right] += 1
                neigh_inter_cnt[right][left] += 1

    out: Dict[str, List[str]] = {}
    for pid in valid_problem_ids:
        scores: List[Tuple[float, str]] = []
        for other, inter in neigh_inter_cnt.get(pid, {}).items():
            union = item_user_cnt[pid] + item_user_cnt[other] - inter
            if union <= 0:
                continue
            scores.append((float(inter) / float(union), other))
        scores.sort(key=lambda item: (-item[0], item[1]))
        out[pid] = [other for _score, other in scores[:topk]]
    if semantic_ids:
        out = fill_collaborative_neighbors_with_semantic_fallback(
            out,
            valid_problem_ids,
            semantic_ids,
            topk=topk,
        )
    return out


def fill_collaborative_neighbors_with_semantic_fallback(
    neighbors: Dict[str, List[str]],
    valid_problem_ids: Sequence[str],
    semantic_ids: Dict[str, str],
    *,
    topk: int,
) -> Dict[str, List[str]]:
    semantic_groups: Dict[str, List[str]] = defaultdict(list)
    valid_set = set(valid_problem_ids)
    for pid in valid_problem_ids:
        semantic_id = str(semantic_ids.get(pid, "")).strip()
        if semantic_id:
            semantic_groups[semantic_id].append(pid)

    filled: Dict[str, List[str]] = {}
    for pid in valid_problem_ids:
        current: List[str] = []
        seen: set[str] = set()
        for neighbor in neighbors.get(pid, []) or []:
            neighbor = str(neighbor)
            if neighbor == pid or neighbor not in valid_set or neighbor in seen:
                continue
            current.append(neighbor)
            seen.add(neighbor)
            if len(current) >= topk:
                break

        semantic_id = str(semantic_ids.get(pid, "")).strip()
        if len(current) < topk and semantic_id:
            for candidate in semantic_groups.get(semantic_id, []):
                if candidate == pid or candidate in seen:
                    continue
                current.append(candidate)
                seen.add(candidate)
                if len(current) >= topk:
                    break
        filled[pid] = current
    return filled


def _collect_concept_stats(problem_records: Sequence[ProblemRecord]) -> Tuple[Dict[str, set[str]], Dict[Tuple[str, str], int]]:
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
    return concept_to_chapters, local_counts


def _llm_graph_completion(
    *,
    problem_records: Sequence[ProblemRecord],
    priors_dir: Path,
    completer: Optional[OpenAICompatibleGraphCompleter],
) -> Dict[str, Dict[str, Any]]:
    if completer is None:
        return {}

    cache_path = priors_dir / "llm_graph_completion_cache.jsonl"
    llm_cache = load_json_cache(cache_path)
    concept_to_chapters, local_counts = _collect_concept_stats(problem_records)
    chapter_to_concepts: Dict[str, set[str]] = defaultdict(set)
    for concept, chapters in concept_to_chapters.items():
        for chapter in chapters:
            chapter_to_concepts[chapter].add(concept)

    concept_candidates: Dict[str, List[str]] = {}
    for concept in sorted(concept_to_chapters):
        candidates: set[str] = set()
        for chapter in concept_to_chapters[concept]:
            candidates.update(chapter_to_concepts.get(chapter, set()))
        pair_scores: List[Tuple[int, str]] = []
        for (left, right), count in local_counts.items():
            if left == concept:
                pair_scores.append((count, right))
            elif right == concept:
                pair_scores.append((count, left))
        pair_scores.sort(key=lambda item: (-item[0], item[1]))
        for _count, other in pair_scores[:GRAPH_LLM_CANDIDATE_LIMIT]:
            candidates.add(other)
        candidates.discard(concept)
        concept_candidates[concept] = sorted(candidates)[:GRAPH_LLM_CANDIDATE_LIMIT]

    out: Dict[str, Dict[str, Any]] = {}
    for concept, candidates in tqdm(concept_candidates.items(), desc="strict llm graph"):
        if not candidates:
            continue
        payload = llm_cache.get(concept)
        if payload is None:
            payload = completer.complete(
                concept=concept,
                chapters=sorted(concept_to_chapters.get(concept, set())),
                candidate_concepts=candidates,
            )
            llm_cache[concept] = payload
            append_json_cache(cache_path, concept, payload)
        prereq = [item for item in payload.get("prerequisite_candidates", []) if item in candidates][:GRAPH_LLM_MAX_PREREQ]
        related = [item for item in payload.get("related_candidates", []) if item in candidates][:GRAPH_LLM_MAX_RELATED]
        out[concept] = {
            "prerequisite_candidates": prereq,
            "related_candidates": related,
            "confidence": payload.get("confidence", "低"),
            "chapters": sorted(concept_to_chapters.get(concept, set())),
            "candidate_concepts": candidates,
        }
    return out


def build_graph_bundle(
    problem_records: Sequence[ProblemRecord],
    *,
    llm_graph_completion: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    _concept_to_chapters, local_counts = _collect_concept_stats(problem_records)

    adjacency_scores: Dict[str, Counter] = defaultdict(Counter)
    local_edges: List[Dict[str, Any]] = []
    prerequisite_edges: List[Dict[str, Any]] = []

    low_count_candidates: Dict[str, Counter] = defaultdict(Counter)
    for (left, right), count in sorted(local_counts.items()):
        low_count_candidates[left][right] = max(float(low_count_candidates[left][right]), float(count))
        low_count_candidates[right][left] = max(float(low_count_candidates[right][left]), float(count))
        if count < LOCAL_COOCCUR_THRESHOLD:
            continue
        adjacency_scores[left][right] = max(float(adjacency_scores[left][right]), float(count))
        adjacency_scores[right][left] = max(float(adjacency_scores[right][left]), float(count))
        local_edges.append({"src": left, "dst": right, "source": "cofreq", "count": count})

    min_cofreq_neighbors = 3
    for concept, candidates in sorted(low_count_candidates.items()):
        if len(adjacency_scores[concept]) >= min_cofreq_neighbors:
            continue
        ranked = sorted(candidates.items(), key=lambda item: (-float(item[1]), item[0]))
        for neighbor, count in ranked:
            if neighbor in adjacency_scores[concept]:
                continue
            adjacency_scores[concept][neighbor] = max(float(adjacency_scores[concept][neighbor]), float(count))
            adjacency_scores[neighbor][concept] = max(float(adjacency_scores[neighbor][concept]), float(count))
            local_edges.append({"src": concept, "dst": neighbor, "source": "cofreq_topk", "count": int(count)})
            if len(adjacency_scores[concept]) >= min_cofreq_neighbors:
                break

    llm_graph_completion = llm_graph_completion or {}
    for concept, payload in sorted(llm_graph_completion.items()):
        confidence = str(payload.get("confidence") or "低")
        for prereq in payload.get("prerequisite_candidates", []) or []:
            adjacency_scores[concept][prereq] = max(float(adjacency_scores[concept][prereq]), 1_000_000.0)
            adjacency_scores[prereq][concept] = max(float(adjacency_scores[prereq][concept]), 1_000_000.0)
            edge = {
                "src": prereq,
                "dst": concept,
                "source": "llm_prerequisite",
                "confidence": confidence,
            }
            prerequisite_edges.append(edge)
            local_edges.append(edge)
        for related in payload.get("related_candidates", []) or []:
            adjacency_scores[concept][related] = max(float(adjacency_scores[concept][related]), 500_000.0)
            adjacency_scores[related][concept] = max(float(adjacency_scores[related][concept]), 500_000.0)
            local_edges.append(
                {
                    "src": concept,
                    "dst": related,
                    "source": "llm_related",
                    "confidence": confidence,
                }
            )

    max_concept_neighbors = 64
    concept_neighbors: Dict[str, List[str]] = {}
    for concept, scores in sorted(adjacency_scores.items()):
        ranked = sorted(scores.items(), key=lambda item: (-float(item[1]), item[0]))
        concept_neighbors[concept] = [neighbor for neighbor, _score in ranked[:max_concept_neighbors]]

    max_problem_neighbor_concepts = 128
    problem_neighbor_concepts: Dict[str, List[str]] = {}
    for problem in problem_records:
        neighbor_scores: Counter = Counter()
        for concept in problem.concepts:
            for rank, neighbor in enumerate(concept_neighbors.get(concept, [])):
                if neighbor in problem.concepts:
                    continue
                neighbor_scores[neighbor] = max(float(neighbor_scores[neighbor]), 1.0 / float(rank + 1))
        ranked_neighbors = sorted(neighbor_scores.items(), key=lambda item: (-float(item[1]), item[0]))
        problem_neighbor_concepts[problem.problem_id] = [
            neighbor for neighbor, _score in ranked_neighbors[:max_problem_neighbor_concepts]
        ]

    return {
        "has_explicit_prerequisite": bool(prerequisite_edges),
        "e_pre": prerequisite_edges,
        "tau_localco": LOCAL_COOCCUR_THRESHOLD,
        "graph_policy": {
            "chapter_clique_edges": False,
            "min_cofreq_neighbors": min_cofreq_neighbors,
            "max_concept_neighbors": max_concept_neighbors,
            "max_problem_neighbor_concepts": max_problem_neighbor_concepts,
        },
        "concept_neighbors": concept_neighbors,
        "problem_neighbor_concepts": problem_neighbor_concepts,
        "local_edges": local_edges,
        "llm_graph_completion": llm_graph_completion,
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
                    "problem_indices": np.asarray(pid_indices, dtype=np.int64),
                    "results": np.asarray(results, dtype=np.int64),
                    "levels": np.asarray(item_levels, dtype=np.int64),
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
    if not val_samples and train_samples:
        raise ValueError(
            "Validation sample set is empty under strict full-system mode; "
            "refusing to synthesize fallback validation samples."
        )
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
    problem_dcq_tensor: torch.Tensor,
    mu_values_tensor: torch.Tensor,
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
        hqtext_tensor[sorted_unique],
        hqid_tensor[sorted_unique],
    )
    dcq_local = problem_dcq_tensor[sorted_unique]
    mu_local = mu_values_tensor[sorted_unique].unsqueeze(-1)
    eqsem_local = eqbase_local + mu_local * dcq_local

    z_rows: List[torch.Tensor] = []
    target_eq_rows: List[torch.Tensor] = []
    labels: List[float] = []

    for seq_idx, t in batch_samples:
        sequence = sequence_cache[seq_idx]
        target_idx = int(sequence["problem_indices"][t])
        target_eq = eqsem_local[local_pos[target_idx]]
        target_level = int(sequence["levels"][t])

        history_start = max(0, t - HISTORY_WINDOW)
        hist_problem_indices = sequence["problem_indices"][history_start:t]
        hist_results = sequence["results"][history_start:t]
        hist_levels = sequence["levels"][history_start:t]
        if len(hist_problem_indices) == 0:
            z_rows.append(torch.zeros((260,), dtype=torch.float32, device=device))
            target_eq_rows.append(target_eq)
            labels.append(float(sequence["results"][t]))
            continue

        hist_local_indices = [local_pos[int(pid_idx)] for pid_idx in hist_problem_indices]
        hist_eq = eqsem_local[hist_local_indices]
        hist_eq_norm = F.normalize(hist_eq, dim=-1)
        target_eq_norm = F.normalize(target_eq.unsqueeze(0), dim=-1).squeeze(0)
        cos_scores = torch.sum(hist_eq_norm * target_eq_norm.unsqueeze(0), dim=-1)

        hist_levels_tensor = torch.tensor(hist_levels, dtype=torch.float32, device=device)
        lags_tensor = torch.arange(len(hist_levels), 0, -1, dtype=torch.float32, device=device)
        score_tensor = (
            cos_scores
            - float(LAMBDA_L) * torch.abs(hist_levels_tensor - float(target_level))
            - float(LAMBDA_T) * torch.log1p(lags_tensor)
        ) / float(TAU)
        alpha = torch.softmax(score_tensor, dim=0)

        result_tensor = torch.tensor(hist_results, dtype=torch.float32, device=device).unsqueeze(-1)
        relation_features = torch.stack(
            [
                (hist_levels_tensor < float(target_level)).to(torch.float32),
                (hist_levels_tensor == float(target_level)).to(torch.float32),
                (hist_levels_tensor > float(target_level)).to(torch.float32),
            ],
            dim=-1,
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
    loss_diag = F.binary_cross_entropy_with_logits(diag_logits, y)
    return loss_diag


def train_strict_model(
    problem_records: Sequence[ProblemRecord],
    student_sequences: Sequence[StudentSequence],
    hqtext_vectors: np.ndarray,
    hqid_vectors: np.ndarray,
    mu_values: np.ndarray,
    concept_groups: Dict[str, Dict[str, np.ndarray]],
    *,
    priors_dir: Path,
    smoke: bool,
    seed: int,
) -> Tuple[StrictPriorModel, Dict[str, Any]]:
    pid_to_idx = {problem.problem_id: idx for idx, problem in enumerate(problem_records)}
    sequence_cache = build_training_sequences(student_sequences, pid_to_idx, problem_records)
    train_samples, val_samples = build_target_samples(sequence_cache, smoke=smoke, seed=seed)

    device = pick_device()
    eq_input_dim = int(hqtext_vectors.shape[1] + hqid_vectors.shape[1])
    model = StrictPriorModel(eq_input_dim=eq_input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)

    hqtext_tensor = torch.tensor(hqtext_vectors.astype(np.float32), dtype=torch.float32, device=device)
    hqid_tensor = torch.tensor(hqid_vectors.astype(np.float32), dtype=torch.float32, device=device)
    mu_values_tensor = torch.tensor(mu_values.astype(np.float32), dtype=torch.float32, device=device)

    def refresh_problem_dcq() -> np.ndarray:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            eqbase_all = model.eqbase(
                hqtext_tensor,
                hqid_tensor,
            ).detach().cpu().tolist()
        if was_training:
            model.train()
        eqbase_all_np = np.asarray(eqbase_all, dtype=np.float32)
        concept_pc1_dirs = build_concept_pc1_directions(problem_records, eqbase_all_np, concept_groups=concept_groups)
        return build_problem_concept_directions(problem_records, concept_pc1_dirs)

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience_left = TRAIN_EARLY_STOP_PATIENCE
    history: List[Dict[str, float]] = []

    for epoch in range(1, TRAIN_MAX_EPOCHS + 1):
        model.train()
        problem_dcq_train = refresh_problem_dcq()
        problem_dcq_train_tensor = torch.tensor(problem_dcq_train, dtype=torch.float32, device=device)
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
                problem_dcq_train_tensor,
                mu_values_tensor,
                device=device,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_GRAD_CLIP)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        problem_dcq_val = refresh_problem_dcq()
        problem_dcq_val_tensor = torch.tensor(problem_dcq_val, dtype=torch.float32, device=device)
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
                    problem_dcq_val_tensor,
                    mu_values_tensor,
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
            "joint_loss": "BCE(diag_head(eqsem, d), y)",
            "eqsem_training_phase": "refresh concept directions from current eqbase each epoch and train with eqsem = eqbase + mu_q * d_c(q)",
        },
    }
    model_state_path = priors_dir / "model_state.pt"
    training_report_path = priors_dir / "training_report.json"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_meta": {
                "eq_input_dim": eq_input_dim,
                "hqtext_dim": int(hqtext_vectors.shape[1]),
                "hqid_dim": int(hqid_vectors.shape[1]),
            },
        },
        model_state_path,
    )
    write_json(training_report, training_report_path)
    return model.cpu(), training_report


def save_pickle(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def build_problem_catalog_records(
    problem_records: Sequence[ProblemRecord],
    semantic_ids: Dict[str, str],
) -> List[ProblemCatalogRecord]:
    return [
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


def run_stage32_core_artifacts(
    *,
    problem_json: Path,
    student_json: Path,
    priors_dir: Path,
    smoke: bool = False,
    text_embed_model: str = TEXT_EMBED_MODEL_NAME,
    text_embed_batch_size: int = TEXT_EMBED_BATCH_SIZE,
    text_embed_max_length: int = TEXT_EMBED_MAX_LENGTH,
) -> Stage32CoreArtifactsResult:
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
        raise ValueError("No problems loaded for strict stage 3.2 core artifacts")
    if not student_sequences:
        raise ValueError("No student sequences loaded for strict stage 3.2 core artifacts")

    encoder = QwenEmbeddingEncoder(
        model_name_or_path=str(text_embed_model or TEXT_EMBED_MODEL_NAME),
        device=pick_device(),
        max_length=int(text_embed_max_length),
        batch_size=int(text_embed_batch_size),
    )
    raw_texts = [problem.text or problem.title or problem.problem_id for problem in problem_records]
    hqtext_vectors = encoder.encode_texts(
        raw_texts,
        instruction="Encode educational problem text for semantic clustering and downstream retrieval.",
    )

    semantic_ids_path = priors_dir / "semantic_ids.json"
    semantic_ids, _semantic_id_texts, semantic_id_audit = build_semantic_ids(
        problem_records,
        hqtext_vectors,
        semantic_ids_path=semantic_ids_path,
    )
    semantic_id_audit_path = priors_dir / "semantic_id_audit.json"
    write_json(semantic_id_audit, semantic_id_audit_path)

    collab_neighbors = build_jaccard_collaborative_neighbors(
        student_sequences,
        [problem.problem_id for problem in problem_records],
        topk=5,
        semantic_ids=semantic_ids,
    )
    item_collaborative_path = priors_dir / "item_collaborative.json"
    write_json(collab_neighbors, item_collaborative_path)

    problem_catalog_path = priors_dir / "problem_catalog.jsonl"
    write_jsonl(
        dataclass_list_to_jsonl(build_problem_catalog_records(problem_records, semantic_ids)),
        problem_catalog_path,
    )

    manifest = Stage32CoreArtifactsResult(
        priors_dir=str(priors_dir),
        semantic_ids_path=str(semantic_ids_path),
        semantic_id_audit_path=str(semantic_id_audit_path),
        problem_catalog_path=str(problem_catalog_path),
        item_collaborative_path=str(item_collaborative_path),
        manifest_path=str(priors_dir / "stage32_core_artifacts_manifest.json"),
        problem_count=len(problem_records),
        student_count=len(student_sequences),
    )
    write_json(asdict(manifest), Path(manifest.manifest_path))
    return manifest


def run_stage32(
    *,
    problem_json: Path,
    student_json: Path,
    priors_dir: Path,
    seed: int = TRAIN_SEED,
    smoke: bool = False,
    text_embed_model: str = TEXT_EMBED_MODEL_NAME,
    text_embed_batch_size: int = TEXT_EMBED_BATCH_SIZE,
    text_embed_max_length: int = TEXT_EMBED_MAX_LENGTH,
    enable_llm_graph_completion: bool = False,
    llm_base_url: str = "",
    llm_model: str = "",
    llm_api_key: str = "",
    llm_timeout_sec: int = 120,
    llm_max_tokens: int = 160,
    llm_temperature: float = 0.1,
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

    encoder = QwenEmbeddingEncoder(
        model_name_or_path=str(text_embed_model or TEXT_EMBED_MODEL_NAME),
        device=pick_device(),
        max_length=int(text_embed_max_length),
        batch_size=int(text_embed_batch_size),
    )
    raw_texts = [problem.text or problem.title or problem.problem_id for problem in problem_records]
    concept_groups = build_concept_groups(problem_records)
    hqtext_vectors = encoder.encode_texts(raw_texts, instruction="Encode educational problem text for semantic clustering and downstream retrieval.")
    semantic_ids_path = priors_dir / "semantic_ids.json"
    semantic_ids, semantic_id_texts, semantic_id_audit = build_semantic_ids(
        problem_records,
        hqtext_vectors,
        semantic_ids_path=semantic_ids_path,
    )
    semantic_id_audit_path = priors_dir / "semantic_id_audit.json"
    write_json(semantic_id_audit, semantic_id_audit_path)
    hqid_vectors = encoder.encode_texts(
        semantic_id_texts,
        instruction="Encode hierarchical semantic identifiers for educational problem semantics.",
    )

    pid_to_idx = {problem.problem_id: idx for idx, problem in enumerate(problem_records)}
    pid_to_mu_q = (
        estimate_rasch_mu_q(student_sequences, pid_to_idx, seed=seed)
        if USE_RASCH_ENHANCEMENT
        else {problem.problem_id: 0.0 for problem in problem_records}
    )
    mu_values = np.asarray(
        [pid_to_mu_q.get(problem.problem_id, 0.0) for problem in problem_records],
        dtype=np.float32,
    )
    model_torch, training_report = train_strict_model(
        problem_records,
        student_sequences,
        hqtext_vectors,
        hqid_vectors,
        mu_values,
        concept_groups,
        priors_dir=priors_dir,
        smoke=smoke,
        seed=seed,
    )

    with torch.no_grad():
        hqtext_tensor = torch.tensor(hqtext_vectors.astype(np.float32), dtype=torch.float32)
        hqid_tensor = torch.tensor(hqid_vectors.astype(np.float32), dtype=torch.float32)
        eqbase_vectors = model_torch.eqbase(hqtext_tensor, hqid_tensor).detach().cpu().tolist()
        eqbase_vectors = np.asarray(eqbase_vectors, dtype=np.float32)

    concept_pc1_dirs = build_concept_pc1_directions(problem_records, eqbase_vectors, concept_groups=concept_groups)
    problem_dcq = build_problem_concept_directions(problem_records, concept_pc1_dirs)
    mu_arr = mu_values.reshape(-1, 1)
    eqsem_vectors = eqbase_vectors + mu_arr * problem_dcq
    eqsem_vectors = eqsem_vectors.astype(np.float32)

    collab_neighbors, collab_vectors = build_collaborative_vectors(
        student_sequences,
        [problem.problem_id for problem in problem_records],
        seed=seed,
        semantic_ids=semantic_ids,
    )
    graph_completer = None
    if enable_llm_graph_completion:
        graph_completer = OpenAICompatibleGraphCompleter(
            base_url=str(llm_base_url or ""),
            model=str(llm_model or ""),
            api_key=llm_api_key,
            timeout_sec=int(llm_timeout_sec),
            max_tokens=int(llm_max_tokens),
            temperature=float(llm_temperature),
        )
    llm_graph_completion = _llm_graph_completion(
        problem_records=problem_records,
        priors_dir=priors_dir,
        completer=graph_completer,
    )
    graph_bundle = build_graph_bundle(problem_records, llm_graph_completion=llm_graph_completion)

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

    catalog_records = build_problem_catalog_records(problem_records, semantic_ids)
    problem_catalog_path = priors_dir / "problem_catalog.jsonl"
    write_jsonl(dataclass_list_to_jsonl(catalog_records), problem_catalog_path)

    implementation_defaults = {
        "text_embed_model_name": str(text_embed_model or TEXT_EMBED_MODEL_NAME),
        "text_embed_batch_size": int(text_embed_batch_size),
        "text_embed_max_length": int(text_embed_max_length),
        "kglobal": KGLOBAL,
        "klocal": KLOCAL,
        "random_state": KMEANS_RANDOM_STATE,
        "n_init": KMEANS_N_INIT,
        "ctfidf_max_features": CTFIDF_MAX_FEATURES,
        "history_window": HISTORY_WINDOW,
        "collab_window": COLLAB_WINDOW,
        "negative_samples": NEGATIVE_SAMPLES,
        "local_cooccur_threshold": LOCAL_COOCCUR_THRESHOLD,
        "rasch_a": RASCH_A,
        "use_rasch_enhancement": USE_RASCH_ENHANCEMENT,
        "enable_llm_graph_completion": bool(enable_llm_graph_completion),
        "train_lr": TRAIN_LR,
        "train_grad_clip": TRAIN_GRAD_CLIP,
        "lambda_l": LAMBDA_L,
        "lambda_t": LAMBDA_T,
        "tau": TAU,
        "epsilon": EPS,
        "smoke": smoke,
        "semantic_id_cleaning": {
            "extra_stopword_count": len(SEMANTIC_EXTRA_STOP_WORDS),
            "placeholder_rule": "drop underscore-only and blank-like tokens",
            "duplicate_collapse": True,
            "fallback_order_macro": "chapter -> concepts -> title -> text",
            "fallback_order_micro": "concepts -> title -> text -> chapter",
        },
    }
    implementation_defaults_path = priors_dir / "implementation_defaults.json"
    write_json(implementation_defaults, implementation_defaults_path)

    manifest = Stage32Result(
        priors_dir=str(priors_dir),
        semantic_ids_path=str(priors_dir / "semantic_ids.json"),
        semantic_id_audit_path=str(semantic_id_audit_path),
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
