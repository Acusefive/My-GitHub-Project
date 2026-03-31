"""
validate_retrieval.py

用“统计级指标”验证检索上下文是否真的有用。

对比三种检索策略：
  1) cog_only  : 只用认知/概念重合得分（Jaccard + level）
  2) cf_only   : 只用协同信号（item_collaborative.json 召回集合）
  3) hybrid    : cog + dynamic cf（与 preprocess_embeddings.py 的逻辑一致）

对比两种基线：
  a) history_random : 从当前交互的历史题中随机抽 k 个
  b) global_random  : 从全题库（有 semantic_id 的题）里随机抽 k 个

指标：
  - same_cluster_rate：retrieved 的 semantic_id == target semantic_id 的比例
  - same_top_rate    ：retrieved 的 top label（semantic_id 第一段）== target top 的比例

用法示例：
  python3 validate_retrieval.py --samples 5000 --k 4 --seed 0
  python3 validate_retrieval.py --samples 20000 --k 4 --seed 0 --no_cf
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def load_json_any(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]


def iter_student_records(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ["data", "students", "records", "logs"]:
            v = data.get(k)
            if isinstance(v, list):
                return v
    return []


def safe_level(x: Any) -> int:
    try:
        if x is None:
            return 0
        if isinstance(x, (int, float)):
            return int(x)
        s = str(x).strip()
        if not s:
            return 0
        return int(float(s))
    except Exception:
        return 0


def top_label(semantic_id: str) -> str:
    if not semantic_id:
        return ""
    return semantic_id.split("-", 1)[0]


@dataclass(frozen=True)
class ProblemMeta:
    concepts: Tuple[str, ...]
    level: int


def build_problem_meta(problems: Dict[str, Dict[str, Any]]) -> Dict[str, ProblemMeta]:
    out: Dict[str, ProblemMeta] = {}
    for pid, p in problems.items():
        concepts = tuple(str(c) for c in (p.get("concepts", []) or []) if c)
        lvl = safe_level(p.get("cognitive_dimension", 0))
        out[pid] = ProblemMeta(concepts=concepts, level=lvl)
    return out


class Retriever:
    """
    与 preprocess_embeddings.py 对齐的检索打分逻辑（可开关 cog / cf）。
    """

    def __init__(
        self,
        meta: Dict[str, ProblemMeta],
        collab: Dict[str, List[str]],
        w_cf_high: float = 8.0,
        w_cf_low: float = 2.0,
        lambda_time_decay: float = 0.05,
        use_cog: bool = True,
        use_cf: bool = True,
    ) -> None:
        self.meta = meta
        self.collab = collab
        self.w_cf_high = float(w_cf_high)
        self.w_cf_low = float(w_cf_low)
        self.lambda_time_decay = float(lambda_time_decay)
        self.use_cog = bool(use_cog)
        self.use_cf = bool(use_cf)

    def retrieve(self, history: Sequence[Dict[str, Any]], target_id: str, k: int = 4) -> List[str]:
        if not target_id or target_id not in self.meta:
            return []

        target_meta = self.meta[target_id]
        target_concepts = set(target_meta.concepts)
        target_level = target_meta.level
        target_sim_set = set(self.collab.get(target_id, []) or [])

        candidates: List[Tuple[float, str]] = []
        L = len(history)
        for idx, h in enumerate(history):
            h_id = str(h.get("problem_id", ""))
            if (not h_id) or (h_id == target_id) or (h_id not in self.meta):
                continue

            delta_t = L - idx
            s = self._score(
                h_id=h_id,
                target_concepts=target_concepts,
                target_level=target_level,
                target_sim_set=target_sim_set,
                delta_t=delta_t,
            )
            if s is None:
                continue
            candidates.append((s, h_id))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [pid for _, pid in candidates[:k]]

    def _score(
        self,
        h_id: str,
        target_concepts: set,
        target_level: int,
        target_sim_set: set,
        delta_t: int,
    ) -> Optional[float]:
        base = 0.0
        if self.use_cog:
            base += self._cognitive_score(h_id=h_id, target_concepts=target_concepts, target_level=target_level)

        if self.use_cf and (h_id in target_sim_set):
            # 与原逻辑一致：当 cog_score < 5 时给更强的协同加分
            cog_score = base if self.use_cog else 0.0
            base += (self.w_cf_high if cog_score < 5.0 else self.w_cf_low)

        if base <= 0.0:
            return None

        if delta_t is not None and delta_t > 0 and self.lambda_time_decay > 0:
            base *= math.exp(-self.lambda_time_decay * float(delta_t))

        return base if base > 0.0 else None

    def _cognitive_score(self, h_id: str, target_concepts: set, target_level: int) -> float:
        h_meta = self.meta[h_id]
        h_concepts = set(h_meta.concepts)
        h_level = h_meta.level

        score = 0.0
        # 知识一致性：Jaccard * 10
        if target_concepts and h_concepts:
            inter = target_concepts.intersection(h_concepts)
            union = target_concepts.union(h_concepts)
            if union:
                score += 10.0 * (len(inter) / len(union))

        # 认知层级：wp=5, wm=2
        if h_level > 0 and target_level > 0:
            if h_level < target_level:
                score += 5.0
            if h_level == target_level:
                score += 2.0

        return score


def sample_interactions(
    records: Sequence[Dict[str, Any]],
    meta: Dict[str, ProblemMeta],
    semantic_map: Dict[str, str],
    n_samples: int,
    seed: int,
) -> List[Tuple[Sequence[Dict[str, Any]], int, str]]:
    rng = random.Random(seed)
    out: List[Tuple[Sequence[Dict[str, Any]], int, str]] = []

    # 尝试多采几次，避免 seq 太短或缺字段
    max_tries = max(10000, n_samples * 50)
    tries = 0
    while len(out) < n_samples and tries < max_tries:
        tries += 1
        rec = rng.choice(records)
        seq = rec.get("seq", []) or []
        if not isinstance(seq, list) or len(seq) < 2:
            continue
        t = rng.randrange(1, len(seq))
        target_id = str(seq[t].get("problem_id", ""))
        if not target_id:
            continue
        if target_id not in meta:
            continue
        if target_id not in semantic_map:
            continue
        out.append((seq, t, target_id))

    return out


def evaluate_one(
    pairs: Sequence[Tuple[Sequence[Dict[str, Any]], int, str]],
    retriever: Retriever,
    semantic_map: Dict[str, str],
    k: int,
    baseline: str,
    global_pool: Sequence[str],
    seed: int,
) -> Dict[str, float]:
    rng = random.Random(seed)

    same_cluster = 0
    same_top = 0
    total = 0

    for seq, t, target_id in pairs:
        tgt_sem = semantic_map.get(target_id, "")
        if not tgt_sem:
            continue

        if baseline == "retriever":
            retrieved = retriever.retrieve(seq[:t], target_id=target_id, k=k)
        elif baseline == "history_random":
            hist_ids = [str(h.get("problem_id", "")) for h in (seq[:t] or [])]
            hist_ids = [pid for pid in hist_ids if pid in semantic_map and pid != target_id]
            if not hist_ids:
                continue
            retrieved = rng.sample(hist_ids, min(k, len(hist_ids)))
        elif baseline == "global_random":
            # 从全题库随机抽（排除 target）
            if not global_pool:
                continue
            retrieved = []
            # 允许一定次数重试，避免刚好抽到 target 或空
            for _ in range(k * 5):
                if len(retrieved) >= k:
                    break
                pid = rng.choice(global_pool)
                if pid and pid != target_id:
                    retrieved.append(pid)
            if not retrieved:
                continue
        else:
            raise ValueError(f"Unknown baseline: {baseline}")

        tgt_top = top_label(tgt_sem)
        for pid in retrieved:
            sem = semantic_map.get(pid, "")
            if not sem:
                continue
            total += 1
            if sem == tgt_sem:
                same_cluster += 1
            if top_label(sem) == tgt_top:
                same_top += 1

    return {
        "pairs_used": float(len(pairs)),
        "retrieved_pairs": float(total),
        "same_cluster_rate": (same_cluster / total) if total else 0.0,
        "same_top_rate": (same_top / total) if total else 0.0,
    }


def fmt(x: float) -> str:
    return f"{x:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem_json", default="problem.json")
    ap.add_argument("--student_json", default="student-problem-fine.json")
    ap.add_argument("--semantic_json", default="item_semantic_ids.json")
    ap.add_argument("--collab_json", default="item_collaborative.json")
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambda_time_decay", type=float, default=0.05)
    ap.add_argument("--w_cf_high", type=float, default=8.0)
    ap.add_argument("--w_cf_low", type=float, default=2.0)
    ap.add_argument("--no_cog", action="store_true", help="检索时禁用认知/概念得分")
    ap.add_argument("--no_cf", action="store_true", help="检索时禁用协同加分")
    args = ap.parse_args()

    print("[LOAD] problems / records / semantic / collab ...")
    problems_raw = load_json_any(args.problem_json)
    problems_list = problems_raw if isinstance(problems_raw, list) else [problems_raw]
    problems: Dict[str, Dict[str, Any]] = {}
    for p in problems_list:
        pid = str(p.get("problem_id", p.get("id", "")))
        if pid:
            problems[pid] = p

    records_raw = load_json_any(args.student_json)
    records = iter_student_records(records_raw)
    semantic_map: Dict[str, str] = load_json_any(args.semantic_json) if args.semantic_json else {}
    collab_map: Dict[str, List[str]] = load_json_any(args.collab_json) if args.collab_json else {}

    print(f"[STAT] Problems: {len(problems)} | Records: {len(records)} | Semantic items: {len(semantic_map)}")

    meta = build_problem_meta(problems)
    global_pool = [pid for pid in semantic_map.keys() if pid in meta]

    pairs = sample_interactions(
        records=records,
        meta=meta,
        semantic_map=semantic_map,
        n_samples=max(0, int(args.samples)),
        seed=int(args.seed),
    )
    print(f"[SAMPLE] interactions: {len(pairs)} (requested={args.samples})")
    if not pairs:
        raise SystemExit("No valid interactions sampled. Check your input files.")

    # 组装三种策略
    strategies = [
        ("cog_only", True, False),
        ("cf_only", False, True),
        ("hybrid", True, True),
    ]

    # 支持用户全局开关（便于快速 ablation）
    if args.no_cog:
        strategies = [(name, False, use_cf) for (name, _use_cog, use_cf) in strategies]
    if args.no_cf:
        strategies = [(name, use_cog, False) for (name, use_cog, _use_cf) in strategies]

    print("\n[RUN] retrieval strategies (vs baselines)\n")

    # 先跑两种基线
    base_hist = evaluate_one(
        pairs=pairs,
        retriever=Retriever(meta, collab_map, args.w_cf_high, args.w_cf_low, args.lambda_time_decay, True, True),
        semantic_map=semantic_map,
        k=int(args.k),
        baseline="history_random",
        global_pool=global_pool,
        seed=int(args.seed),
    )
    base_global = evaluate_one(
        pairs=pairs,
        retriever=Retriever(meta, collab_map, args.w_cf_high, args.w_cf_low, args.lambda_time_decay, True, True),
        semantic_map=semantic_map,
        k=int(args.k),
        baseline="global_random",
        global_pool=global_pool,
        seed=int(args.seed),
    )

    print(f"Baseline history_random : same_cluster={fmt(base_hist['same_cluster_rate'])} same_top={fmt(base_hist['same_top_rate'])} retrieved_pairs={int(base_hist['retrieved_pairs'])}")
    print(f"Baseline global_random  : same_cluster={fmt(base_global['same_cluster_rate'])} same_top={fmt(base_global['same_top_rate'])} retrieved_pairs={int(base_global['retrieved_pairs'])}")
    print("")

    # 再跑策略
    for name, use_cog, use_cf in strategies:
        retriever = Retriever(
            meta=meta,
            collab=collab_map,
            w_cf_high=float(args.w_cf_high),
            w_cf_low=float(args.w_cf_low),
            lambda_time_decay=float(args.lambda_time_decay),
            use_cog=bool(use_cog),
            use_cf=bool(use_cf),
        )
        res = evaluate_one(
            pairs=pairs,
            retriever=retriever,
            semantic_map=semantic_map,
            k=int(args.k),
            baseline="retriever",
            global_pool=global_pool,
            seed=int(args.seed),
        )

        # 提升量（对比两种基线）
        d_hist_cluster = res["same_cluster_rate"] - base_hist["same_cluster_rate"]
        d_hist_top = res["same_top_rate"] - base_hist["same_top_rate"]
        d_glb_cluster = res["same_cluster_rate"] - base_global["same_cluster_rate"]
        d_glb_top = res["same_top_rate"] - base_global["same_top_rate"]

        print(
            f"{name:8s} (use_cog={int(use_cog)} use_cf={int(use_cf)}): "
            f"same_cluster={fmt(res['same_cluster_rate'])} (Δhist={fmt(d_hist_cluster)} Δglb={fmt(d_glb_cluster)}) | "
            f"same_top={fmt(res['same_top_rate'])} (Δhist={fmt(d_hist_top)} Δglb={fmt(d_glb_top)}) | "
            f"retrieved_pairs={int(res['retrieved_pairs'])}"
        )

    print("\n[NOTE] 如果 global_random 基线很低，而检索显著更高，说明检索确实带来语义相关性提升。")


if __name__ == "__main__":
    main()

