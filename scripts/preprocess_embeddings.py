"""
preprocess_embeddings.py

目标：
基于已有的 Cognitive-RAG Prompt 构建逻辑，为 student-problem-fine.json 中每一次交互生成
“检索上下文纯文本”，并使用 BGE embedding 模型生成向量，最终保存为 pkl 文件：

  embeddings: List[List[np.ndarray]]
    - 外层 list：学生维度（与原始 records 顺序一致）
    - 内层 list：该学生 seq 维度（与原始 seq 索引严格对齐）
    - 每个元素：np.ndarray(shape=(512,), dtype=float32) 语义向量

说明：
- 复用/改写 run_ablation_experiment-GRAM.py 中的 CognitiveRetriever（保留动态 wcf 权重逻辑）
- 使用 preprocess_gram.py 中相同的 SentenceTransformer 加载逻辑与模型名：
  "BAAI/bge-small-zh-v1.5"
- generate_context_text 参考 generate_prompt，但只保留 [RETRIEVED CONTEXT] 部分：
  Question ID / Collaborative / Cognitive / Content
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import numpy as np
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise SystemExit(
        "缺少依赖：sentence-transformers。请先执行：pip install sentence-transformers"
    ) from e


MODEL_NAME = "BAAI/bge-small-zh-v1.5"


# ========= 复用的辅助函数（来自 run_ablation_experiment-GRAM.py）=========
def extract_content(problem_record: Dict[str, Any]) -> str:
    """从问题记录中提取 content 文本（兼容 detail 为 str / dict）。"""
    detail = problem_record.get("detail", "")
    if not detail:
        return ""
    if isinstance(detail, str):
        d = ast.literal_eval(detail)
        if isinstance(d, dict):
            return d.get("content", "") or ""
    elif isinstance(detail, dict):
        return detail.get("content", "") or ""
    return ""


class CognitiveRetriever:
    """
    复用 run_ablation_experiment-GRAM.py 的思路：
    - 显式认知检索（concept overlap + cognitive level）
    - 动态协同权重 wcf：当认知得分低时提升协同信号权重
    """

    def __init__(
        self,
        problems: Dict[str, Dict[str, Any]],
        collaborative_data: Optional[Dict[str, List[str]]] = None,
        collaborative_file: str = "item_collaborative.json",
        w_cf_high: float = 8.0,
        w_cf_low: float = 2.0,
        lambda_time_decay: float = 0.05,
    ) -> None:
        self.problems = problems
        self.W_CF_HIGH = w_cf_high
        self.W_CF_LOW = w_cf_low
        self.lambda_time_decay = float(lambda_time_decay)

        if collaborative_data is not None:
            self.collaborative_similar = collaborative_data
        else:
            self.collaborative_similar = {}
            if not os.path.exists(collaborative_file):
                raise FileNotFoundError(collaborative_file)
            with open(collaborative_file, "r", encoding="utf-8") as f:
                self.collaborative_similar = json.load(f)

    @staticmethod
    def _safe_level(x: Any) -> int:
        if x is None:
            return 0
        if isinstance(x, (int, float)):
            return int(x)
        s = str(x).strip()
        if not s:
            return 0
        return int(float(s))

    def retrieve(self, history: Sequence[Dict[str, Any]], target_prob: Dict[str, Any], k: int = 4) -> List[Dict[str, Any]]:
        target_id = str(target_prob.get("problem_id", ""))
        if not target_id:
            return []

        target_concepts, target_level = self._get_target_cog_info(target_id, target_prob)
        target_similar_items = set(self.collaborative_similar.get(target_id, []) or [])

        candidates: List[Tuple[float, Dict[str, Any]]] = []
        history_len = len(history)
        for idx, h in enumerate(history):
            delta_t = history_len - idx
            score = self._score_history_item(
                h=h,
                target_id=target_id,
                target_concepts=target_concepts,
                target_level=target_level,
                target_similar_items=target_similar_items,
                delta_t=delta_t,
            )
            if score is not None:
                candidates.append(score)

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in candidates[:k]]

    def _get_target_cog_info(self, target_id: str, target_prob: Dict[str, Any]) -> Tuple[set, int]:
        target_meta = self.problems.get(target_id, target_prob)
        target_concepts = set(target_meta.get("concepts", []) or [])
        target_level = self._safe_level(target_meta.get("cognitive_dimension", 0))
        return target_concepts, target_level

    def _score_history_item(
        self,
        h: Dict[str, Any],
        target_id: str,
        target_concepts: set,
        target_level: int,
        target_similar_items: set,
        delta_t: int,
    ) -> Optional[Tuple[float, Dict[str, Any]]]:
        h_id = str(h.get("problem_id", ""))
        if not h_id or h_id == target_id or (h_id not in self.problems):
            return None

        h_meta = self.problems[h_id]
        cog_score = self._cognitive_score(h_meta, target_concepts, target_level)
        final_score = self._apply_dynamic_cf(h_id, cog_score, target_similar_items)

        # 时间衰减：越久远的题目权重越低
        if delta_t is not None and delta_t > 0 and self.lambda_time_decay > 0:
            final_score *= math.exp(-self.lambda_time_decay * float(delta_t))

        if final_score <= 0:
            return None
        return final_score, h

    def _cognitive_score(self, h_meta: Dict[str, Any], target_concepts: set, target_level: int) -> float:
        score = 0.0

        # 知识一致性：使用 Jaccard 相似度并乘以基础权重 (wc=10)
        h_concepts = set(h_meta.get("concepts", []) or [])
        if target_concepts and h_concepts:
            inter = target_concepts.intersection(h_concepts)
            union = target_concepts.union(h_concepts)
            if union:
                jaccard = len(inter) / len(union)
                score += 10.0 * jaccard

        # 认知层级 (wp=5, wm=2)
        h_level = self._safe_level(h_meta.get("cognitive_dimension", 0))
        if h_level > 0 and target_level > 0:
            if h_level < target_level:
                score += 5.0
            if h_level == target_level:
                score += 2.0

        return score

    def _apply_dynamic_cf(self, h_id: str, cog_score: float, target_similar_items: set) -> float:
        # 动态协同权重 wcf（保留原始逻辑）
        final_score = cog_score
        if h_id in target_similar_items:
            if cog_score < 5.0:
                final_score += self.W_CF_HIGH
            else:
                final_score += self.W_CF_LOW
        return final_score


# ========= 新函数：生成自然语言 [RETRIEVED CONTEXT] 纯文本 =========
def generate_context_text(
    retrieved: Sequence[Dict[str, Any]],
    problems: Dict[str, Dict[str, Any]],
    semantic_map: Dict[str, str],
    collab_map: Dict[str, List[str]],
    max_content_chars: int = 200,
) -> str:
    """
    将检索到的题目历史记录转换为自然语言描述的纯文本上下文。

    模板：
    “这道题目的核心语义属于[{semantic_id}]类别，涉及[{concepts_str}]等知识点，认知难度层级为[{cognitive_level}]。
    常与[{collaborative_str}]等题目一起被练习。题目内容为：[{content_str}]。学生对该题的作答结果是：[{performance_str}]。”
    """
    sentences: List[str] = []

    for r in retrieved:
        pid = str(r.get("problem_id", ""))
        if not pid or pid not in problems:
            continue

        meta = problems[pid]

        # 核心语义 ID
        semantic_id = semantic_map.get(pid, "未知")

        # 协同题的语义 ID（最多 3 个）
        similar_pids = (collab_map.get(pid, []) or [])[:3]
        similar_semantic_ids = [semantic_map.get(sid, "未知") for sid in similar_pids]
        collaborative_str = "、".join(similar_semantic_ids) if similar_semantic_ids else "无"

        # 知识点与认知层级
        concepts_list = meta.get("concepts", []) or []
        concepts_str = "、".join(map(str, concepts_list)) if concepts_list else "无"
        cognitive_level = meta.get("cognitive_dimension", "未知")

        # 题目内容摘要
        content = extract_content(meta)
        content_str = (content[:max_content_chars] if content else "无可用内容").replace("\n", " ")

        # 学生作答结果 is_correct
        is_correct = r.get("is_correct", None)
        if is_correct == 1 or is_correct == "1" or is_correct is True:
            performance_str = "正确"
        elif is_correct == 0 or is_correct == "0" or is_correct is False:
            performance_str = "错误"
        else:
            performance_str = "未知"

        sentence = (
            f"这道题目的核心语义属于[{semantic_id}]类别，涉及[{concepts_str}]等知识点，"
            f"认知难度层级为[{cognitive_level}]。常与[{collaborative_str}]等题目一起被练习。"
            f"题目内容为：[{content_str}]。学生对该题的作答结果是：[{performance_str}]。"
        )
        sentences.append(sentence)

    return "\n".join(sentences).strip()


# ========= I/O 工具 =========
def _load_json_any(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]


def load_problems_map(problem_file: str) -> Dict[str, Dict[str, Any]]:
    data = _load_json_any(problem_file)
    problems: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, dict):
        data = [data]
    for p in data:
        pid = str(p.get("problem_id", p.get("id", "")))
        if pid:
            problems[pid] = p
    return problems


def load_student_records(student_file: str) -> List[Dict[str, Any]]:
    data = _load_json_any(student_file)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # 兼容常见包裹结构
        for k in ["data", "students", "records", "logs"]:
            if isinstance(data.get(k), list):
                return data[k]
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem_json", default="problem.json")
    ap.add_argument("--student_json", default="student-problem-fine.json")
    ap.add_argument("--semantic_json", default="item_semantic_ids.json")
    ap.add_argument("--collab_json", default="item_collaborative.json")
    ap.add_argument("--out_pkl", default="cognitive_embeddings.pkl")
    # 保存完整文本（与 embeddings 对齐）：List[List[str]]
    ap.add_argument("--out_text_pkl", default="cognitive_context_texts.pkl")
    # 仍然保留一个可读的预览 txt（默认只写前 N 条，避免文件过大）
    ap.add_argument("--out_preview_txt", default="context_text_preview.txt")
    ap.add_argument("--preview_limit", type=int, default=50)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_content_chars", type=int, default=200)
    # 新增参数：是否只生成文本供审核（不跑 embedding）
    ap.add_argument("--dry_run", action="store_true", help="只生成文本进行审核，不运行模型")
    args = ap.parse_args()

    print("Loading problems / students / semantic ids / collaborative map ...")
    problems = load_problems_map(args.problem_json)
    student_records = load_student_records(args.student_json)
    semantic_map = _load_json_any(args.semantic_json) if os.path.exists(args.semantic_json) else {}
    collab_map = _load_json_any(args.collab_json) if os.path.exists(args.collab_json) else {}

    print(f"Problems: {len(problems)} | Students(records): {len(student_records)}")

    retriever = CognitiveRetriever(problems=problems, collaborative_data=collab_map)

    # 1. 生成所有文本 (这里是您最关心的部分)
    flat_texts, index_map, per_student_len = build_all_context_texts(
        student_records=student_records,
        problems=problems,
        semantic_map=semantic_map,
        collab_map=collab_map,
        retriever=retriever,
        topk=args.topk,
        max_content_chars=args.max_content_chars,
    )

    # === 新增：保存“完整对齐文本”到 pkl ===
    print(f"\n[SAVE] 正在保存完整对齐文本到 {args.out_text_pkl} ...")
    context_texts = rehydrate_texts(flat_texts, index_map, per_student_len)
    with open(args.out_text_pkl, "wb") as f:
        pickle.dump(context_texts, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[OK] Saved context texts to: {args.out_text_pkl}")

    # === 仍保留：保存少量文本供人工快速检查 ===
    preview_n = max(0, int(args.preview_limit))
    print(f"\n[AUDIT] 正在保存预览文本(前 {preview_n} 条)到 {args.out_preview_txt} ...")
    with open(args.out_preview_txt, "w", encoding="utf-8") as f:
        f.write(f"=== Context Preview (Total: {len(flat_texts)}) ===\n\n")
        for i, text in enumerate(flat_texts[:preview_n]):
            s_idx, t_idx = index_map[i]
            f.write(f"--- Record {i} (Student {s_idx}, Time {t_idx}) ---\n")
            f.write(text)
            f.write("\n" + "="*50 + "\n")
    
    print(f"[AUDIT] 预览文本已保存。请打开 {args.out_preview_txt} 检查检索内容是否符合预期。")
    
    # 如果开启了 dry_run，在这里就停止，不跑 embedding
    if args.dry_run:
        print("[INFO] Dry run mode enabled. Exiting before embedding generation.")
        return
    # ==========================

    # 2) 批量 embedding (审核无误后，去掉 --dry_run 即可运行这一步)
    print("Loading embedding model (首次运行会自动下载)...")
    model = SentenceTransformer(MODEL_NAME)

    total = len(flat_texts)
    if total == 0:
        raise SystemExit("student_json 中没有可处理的交互记录。")

    print(f"Encoding {total} texts ...")
    flat_embs = encode_texts_in_batches(
        model=model,
        texts=flat_texts,
        # 移除内部硬编码限制，使用命令行参数
        batch_size=args.batch_size, 
    )

    # 3) 回填
    embeddings, dim = rehydrate_embeddings(flat_embs, index_map, per_student_len)

    # 4) 保存
    out_dir = os.path.dirname(os.path.abspath(args.out_pkl)) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out_pkl, "wb") as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[OK] Saved embeddings to: {args.out_pkl} (dim={dim})")

def build_all_context_texts(
    student_records: List[Dict[str, Any]],
    problems: Dict[str, Dict[str, Any]],
    semantic_map: Dict[str, str],
    collab_map: Dict[str, List[str]],
    retriever: CognitiveRetriever,
    topk: int,
    max_content_chars: int,
) -> Tuple[List[str], List[Tuple[int, int]], List[int]]:
    flat_texts: List[str] = []
    index_map: List[Tuple[int, int]] = []
    per_student_len: List[int] = []

    for s_idx, rec in enumerate(tqdm(student_records, desc="Building context texts (per interaction)")):
        seq = rec.get("seq", []) or []
        per_student_len.append(len(seq))
        for t in range(len(seq)):
            index_map.append((s_idx, t))
            flat_texts.append(
                build_single_interaction_text(
                    seq=seq,
                    t=t,
                    problems=problems,
                    semantic_map=semantic_map,
                    collab_map=collab_map,
                    retriever=retriever,
                    topk=topk,
                    max_content_chars=max_content_chars,
                )
            )

    return flat_texts, index_map, per_student_len


def build_single_interaction_text(
    seq: Sequence[Dict[str, Any]],
    t: int,
    problems: Dict[str, Dict[str, Any]],
    semantic_map: Dict[str, str],
    collab_map: Dict[str, List[str]],
    retriever: CognitiveRetriever,
    topk: int,
    max_content_chars: int,
) -> str:
    if t == 0:
        return ""

    target = seq[t]
    target_id = str(target.get("problem_id", ""))
    if not target_id or target_id not in problems:
        return ""

    history = seq[:t]
    target_meta = problems[target_id]
    retrieved = retriever.retrieve(history, target_meta, k=topk)
    return generate_context_text(
        retrieved=retrieved,
        problems=problems,
        semantic_map=semantic_map,
        collab_map=collab_map,
        max_content_chars=max_content_chars,
    )


def encode_texts_in_batches(
    model: SentenceTransformer,
    texts: Sequence[str],
    batch_size: int,
) -> List[np.ndarray]:
    total = len(texts)
    out: List[np.ndarray] = []
    for start in tqdm(range(0, total, batch_size), desc="Encoding batches"):
        batch = list(texts[start : start + batch_size])
        embs = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        for i in range(embs.shape[0]):
            out.append(embs[i].astype(np.float32, copy=False))
    return out


def rehydrate_embeddings(
    flat_embs: Sequence[np.ndarray],
    index_map: Sequence[Tuple[int, int]],
    per_student_len: Sequence[int],
) -> Tuple[List[List[np.ndarray]], int]:
    if not flat_embs:
        raise ValueError("flat_embs 为空，无法回填。")
    dim = int(flat_embs[0].shape[0])

    embeddings: List[List[np.ndarray]] = [
        [None] * int(per_student_len[i]) for i in range(len(per_student_len))  # type: ignore[list-item]
    ]

    for idx, (s_idx, t_idx) in enumerate(index_map):
        embeddings[s_idx][t_idx] = flat_embs[idx]

    zero = np.zeros((dim,), dtype=np.float32)
    for s_idx in range(len(embeddings)):
        for t_idx in range(len(embeddings[s_idx])):
            if embeddings[s_idx][t_idx] is None:
                embeddings[s_idx][t_idx] = zero.copy()

    return embeddings, dim


def rehydrate_texts(
    flat_texts: Sequence[str],
    index_map: Sequence[Tuple[int, int]],
    per_student_len: Sequence[int],
) -> List[List[str]]:
    texts: List[List[str]] = [
        [""] * int(per_student_len[i]) for i in range(len(per_student_len))
    ]
    for idx, (s_idx, t_idx) in enumerate(index_map):
        texts[s_idx][t_idx] = flat_texts[idx]
    return texts


if __name__ == "__main__":
    from preprocess_embeddings_cognitive_rag import main as rag_main
    rag_main()


