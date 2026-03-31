"""
构建概念图边表 concept_graph_edges.csv

输出字段（CSV）：
  src_concept, dst_concept, rel_type(pre|same|adj), weight(0,1], source

rel_type 说明：
  pre: 有向（src 是 dst 的前置依赖）。CSV 中按需输出单向边
  same: 无向（建议导入时双向建边；本脚本直接输出双向）
  adj: 无向（建议导入时双向建边；本脚本直接输出双向）
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def load_json_any(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]


def extract_problem_content(detail: Any) -> str:
    """仅用于 LLM prompt 时的概念上下文（可选）。"""
    if isinstance(detail, str):
        d = json.loads(detail)
        if isinstance(d, dict):
            return str(d.get("content") or d.get("title") or d.get("body") or "")
    if isinstance(detail, dict):
        return str(detail.get("content") or detail.get("title") or detail.get("body") or "")
    return ""


def extract_problems(path: str) -> Dict[str, Dict[str, Any]]:
    data = load_json_any(path)
    if isinstance(data, dict):
        data = [data]
    out: Dict[str, Dict[str, Any]] = {}
    for p in data:
        pid = str(p.get("problem_id", p.get("id", "")))
        if pid:
            out[pid] = p
    return out


def iter_student_records(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ["data", "students", "records", "logs"]:
            v = data.get(k)
            if isinstance(v, list):
                return v
    return []


def parse_location_main_chapter(location: str) -> str:
    """
    例：'4.6' -> '4'
    兜底：无法解析就返回原字符串
    """
    if not location:
        return ""
    s = str(location).strip()
    if not s:
        return s
    if "." in s:
        head = s.split(".", 1)[0]
        return head.strip() or s
    return s


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    rel_type: str  # pre|same|adj
    weight: float
    source: str


def mode_int(values: Sequence[int]) -> int:
    if not values:
        return 0
    return Counter(values).most_common(1)[0][0]


def build_L_concept_mode(
    problems: Dict[str, Dict[str, Any]],
) -> Dict[str, int]:
    concept_levels: Dict[str, List[int]] = defaultdict(list)
    for _, p in problems.items():
        lvl = p.get("cognitive_dimension", 0)
        lvl_i = int(float(lvl))
        concepts = p.get("concepts", []) or []
        for c in concepts:
            if c is None:
                continue
            cs = str(c)
            if not cs:
                continue
            concept_levels[cs].append(lvl_i)

    out: Dict[str, int] = {}
    for c, lv_list in concept_levels.items():
        out[c] = mode_int(lv_list)
    return out


def build_stat_edges(
    problems: Dict[str, Dict[str, Any]],
    *,
    L_concept: Dict[str, int],
    min_cooc_count: int,
    some_scale: float,
) -> Dict[Tuple[str, str, str], Edge]:
    """
    统计共现边：同一题目的概念两两共现，频次归一化后产生 pre/same。
    """
    cooc: Dict[Tuple[str, str], int] = defaultdict(int)  # unordered pair -> count
    for _, p in problems.items():
        concepts = p.get("concepts", []) or []
        concepts = [str(c) for c in concepts if c is not None and str(c)]
        if len(concepts) < 2:
            continue
        for a, b in combinations(sorted(set(concepts)), 2):
            cooc[(a, b)] += 1

    edges: Dict[Tuple[str, str, str], Edge] = {}
    for (a, b), freq in cooc.items():
        if freq < min_cooc_count:
            continue
        w = min(1.0, float(freq) / float(some_scale))
        La = int(L_concept.get(a, 0))
        Lb = int(L_concept.get(b, 0))
        if La < Lb:
            edges[(a, b, "pre")] = Edge(src=a, dst=b, rel_type="pre", weight=w, source="stat")
        elif La > Lb:
            edges[(b, a, "pre")] = Edge(src=b, dst=a, rel_type="pre", weight=w, source="stat")
        else:
            edges[(a, b, "same")] = Edge(src=a, dst=b, rel_type="same", weight=w, source="stat")
            edges[(b, a, "same")] = Edge(src=b, dst=a, rel_type="same", weight=w, source="stat")
    return edges


def build_sequence_adj_edges(
    students: List[Dict[str, Any]],
    problems: Dict[str, Dict[str, Any]],
    *,
    W_cooc: int,
    min_cooc_count: int,
    some_scale: float,
) -> Dict[Tuple[str, str, str], Edge]:
    """
    序列邻接/共现：在 student 做题轨迹上，取时间窗口内的任意两条交互 (i, j), 0<j-i<=W_cooc
    对应的题目 concept 两两对（概念集合内）计数，最后统一标记为 adj 并输出双向边。
    """
    adj_cnt: Dict[Tuple[str, str], int] = defaultdict(int)  # unordered concept pair -> count

    # 为了加速：把每个 problem_id 映射到其 concept list
    pid2concepts: Dict[str, List[str]] = {}
    for pid, p in problems.items():
        concepts = p.get("concepts", []) or []
        pid2concepts[pid] = [str(c) for c in concepts if c is not None and str(c)]

    for rec in students:
        seq = rec.get("seq", []) or []
        if not isinstance(seq, list) or len(seq) < 2:
            continue
        # 预先拿到该序列每步的 concepts
        seq_pids: List[str] = []
        seq_concepts: List[List[str]] = []
        for log in seq:
            if not isinstance(log, dict):
                continue
            pid = str(log.get("problem_id", ""))
            if pid and pid in problems:
                seq_pids.append(pid)
                seq_concepts.append(pid2concepts.get(pid, []))

        n = len(seq_pids)
        if n < 2:
            continue

        for i in range(n):
            ci = seq_concepts[i]
            if not ci:
                continue
            # window: (i, i+1..i+W_cooc)
            j_max = min(n, i + W_cooc + 1)
            for j in range(i + 1, j_max):
                cj = seq_concepts[j]
                if not cj:
                    continue
                # concepts 内两两计数（无向）
                # 这里按 set 去重，避免同一概念在一个题目内部重复计数
                for a in set(ci):
                    for b in set(cj):
                        if a == b:
                            continue
                        x, y = (a, b) if a < b else (b, a)
                        adj_cnt[(x, y)] += 1

    edges: Dict[Tuple[str, str, str], Edge] = {}
    for (a, b), freq in adj_cnt.items():
        if freq < min_cooc_count:
            continue
        w = min(1.0, float(freq) / float(some_scale))
        edges[(a, b, "adj")] = Edge(src=a, dst=b, rel_type="adj", weight=w, source="stat_seq")
        edges[(b, a, "adj")] = Edge(src=b, dst=a, rel_type="adj", weight=w, source="stat_seq")
    return edges


def compute_degree_undirected(edges: Dict[Tuple[str, str, str], Edge]) -> Dict[str, int]:
    neigh: Dict[str, set] = defaultdict(set)
    for e in edges.values():
        if not e.src or not e.dst:
            continue
        # 对 pre/same/adj 都视为无向连边用于孤立点检测（更符合“度”的直觉）
        neigh[e.src].add(e.dst)
        neigh[e.dst].add(e.src)
    return {k: len(v) for k, v in neigh.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem_json", default="problem.json")
    ap.add_argument("--student_json", default="student-problem-fine.json")
    ap.add_argument("--out_csv", default="concept_graph_edges.csv")

    # 工程默认（来自你确认）
    ap.add_argument("--min_cooc_count", type=int, default=50)
    ap.add_argument("--some_scale", type=float, default=200.0)
    ap.add_argument("--W_cooc", type=int, default=10)

    # LLM 仅作为可选开关（默认不跑，避免误触发 API 成本）
    ap.add_argument("--do_llm", action="store_true", help="是否对孤立节点调用 LLM 补全边")
    ap.add_argument("--max_llm_edges", type=int, default=3)
    ap.add_argument("--llm_conf_threshold", type=float, default=0.7)

    # deepseek 配置（沿用 run_ablation_experiment-GRAM.py 的风格）
    ap.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1/")
    ap.add_argument("--model_name", default="deepseek-v3.2-exp")
    ap.add_argument("--api_key", default="", help="LLM API Key；不提供则无法调用 LLM")

    args = ap.parse_args()

    problems = extract_problems(args.problem_json)
    students_raw = load_json_any(args.student_json)
    students = iter_student_records(students_raw)

    L_concept = build_L_concept_mode(problems)
    print(f"[L] Concepts={len(L_concept)}")

    edges: Dict[Tuple[str, str, str], Edge] = {}
    edges.update(
        build_stat_edges(
            problems,
            L_concept=L_concept,
            min_cooc_count=args.min_cooc_count,
            some_scale=args.some_scale,
        )
    )
    print(f"[STAT] edges={len(edges)} (after stat)")

    seq_edges = build_sequence_adj_edges(
        students,
        problems,
        W_cooc=args.W_cooc,
        min_cooc_count=args.min_cooc_count,
        some_scale=args.some_scale,
    )
    # 合并（同一 src/dst/rel_type 的边取更大 weight）
    for k, e in seq_edges.items():
        if k not in edges or edges[k].weight < e.weight:
            edges[k] = e
    print(f"[SEQ/ADJ] edges={len(edges)} (after seq)")

    # 写出先验图（不含 tree/llm）
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)

    # LLM 补全孤立点
    if args.do_llm:
        degree = compute_degree_undirected(edges)
        isolated = [c for c in L_concept.keys() if degree.get(c, 0) < 2]
        print(f"[LLM] isolated nodes={len(isolated)} (degree<2)")

        if not args.api_key:
            raise SystemExit("[LLM] do_llm=true 但 api_key 为空，无法调用 LLM")

        # 这里对 LLM 调用做最小实现：使用 HTTP 请求
        import requests

        headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}

        rel_allowed = ["pre", "same", "adj"]

        # 固定系统提示，限制输出结构
        system_prompt = (
            "你是图谱构建助手。你只能输出预定义的 rel_type: pre|same|adj。"
            "你必须以 JSON 数组输出，每个元素包含 neighbor_concept, rel_type, confidence。"
            f"rel_type 必须且只能在 {rel_allowed} 中取值。confidence 取 0-1 的浮点数。"
            "如果不确定，请输出空数组 []。"
        )

        # 逐点调用（可缓存；这里用简单 on-disk cache）
        cache_path = os.path.splitext(args.out_csv)[0] + "_llm_cache.json"
        if os.path.exists(cache_path):
            cache = load_json_any(cache_path)
            if not isinstance(cache, dict):
                raise ValueError(f"LLM cache must be a dict, got: {type(cache)}")
        else:
            cache = {}

        def llm_key(concept: str) -> str:
            return hashlib.sha256(concept.encode("utf-8")).hexdigest()

        for c in isolated:
            k = llm_key(c)
            if k in cache:
                pred = cache[k]
            else:
                Lc = L_concept.get(c, 0)
                prompt = (
                    f"当前孤立概念：{c}，其认知层级 L(c)={Lc}。"
                    f"请为它补全最多 {args.max_llm_edges} 条与图中概念相关的邻接边，"
                    "并为每条边给出 rel_type 和 confidence。"
                    "输出 neighbor_concept 必须是数据集中存在的概念字符串。"
                    f"当无法确定时输出 []。"
                )
                body = {
                    "model": args.model_name,
                    "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 512,
                }
                resp = requests.post(args.base_url + "chat/completions", headers=headers, json=body, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                pred = json.loads(content)
                if not isinstance(pred, list):
                    raise ValueError(f"[LLM] invalid response type for concept={c}: expected list, got {type(pred)}")
                cache[k] = pred

                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)

            # 解析候选并写入（配额按 neighbor_concept 数计：Top-args.max_llm_edges）
            l_c = int(L_concept.get(c, 0))

            candidates: List[Dict[str, Any]] = []
            if isinstance(pred, list):
                for item in pred:
                    if not isinstance(item, dict):
                        raise ValueError(f"[LLM] invalid item type for concept={c}: {type(item)}")
                    neigh = str(item.get("neighbor_concept", "")).strip()
                    rel_type = str(item.get("rel_type", "")).strip()
                    conf = item.get("confidence", None)
                    if rel_type not in rel_allowed:
                        raise ValueError(f"[LLM] invalid rel_type for concept={c}: {rel_type}")
                    conf_f = float(conf)
                    if conf_f < args.llm_conf_threshold:
                        continue  # thresholding is intended behavior
                    if not neigh:
                        raise ValueError(f"[LLM] empty neighbor_concept for concept={c}")
                    if neigh not in L_concept:
                        raise ValueError(f"[LLM] neighbor_concept not in dataset for concept={c}: {neigh}")
                    candidates.append({"neigh": neigh, "rel_type": rel_type, "conf": conf_f})

            # confidence 降序，然后按 neighbor_concept 去重后截断 Top-3（邻居数配额）
            candidates.sort(key=lambda x: float(x["conf"]), reverse=True)
            selected: List[Dict[str, Any]] = []
            seen_neighbors: set = set()
            for cand in candidates:
                neigh = str(cand["neigh"])
                if neigh in seen_neighbors:
                    continue
                selected.append(cand)
                seen_neighbors.add(neigh)
                if len(selected) >= int(args.max_llm_edges):
                    break

            # 将 LLM 的语义关系强制约束到本地先验（pre/same 只由 L(c) 规则决定）
            for cand in selected:
                neigh = str(cand["neigh"])
                llm_rel_type = str(cand["rel_type"])
                conf_f = float(cand["conf"])
                l_n = int(L_concept.get(neigh, 0))

                if llm_rel_type in {"pre", "same"}:
                    if l_c == l_n:
                        # 同层：强制 same（双向输出）
                        for src, dst in [(c, neigh), (neigh, c)]:
                            key = (src, dst, "same")
                            if key not in edges or edges[key].weight < conf_f:
                                edges[key] = Edge(src=src, dst=dst, rel_type="same", weight=conf_f, source="llm")
                    else:
                        # 不同层：强制 pre（方向由 L 大小决定：src(L更小) -> dst(L更大)）
                        if l_c < l_n:
                            src, dst = c, neigh
                        else:
                            src, dst = neigh, c
                        key = (src, dst, "pre")
                        if key not in edges or edges[key].weight < conf_f:
                            edges[key] = Edge(src=src, dst=dst, rel_type="pre", weight=conf_f, source="llm")
                elif llm_rel_type == "adj":
                    # adj：保持 LLM 输出的 adj（双向输出），不再额外改写
                    for src, dst in [(c, neigh), (neigh, c)]:
                        key = (src, dst, "adj")
                        if key not in edges or edges[key].weight < conf_f:
                            edges[key] = Edge(src=src, dst=dst, rel_type="adj", weight=conf_f, source="llm")

        print(f"[LLM] edges={len(edges)} (after llm)")

    # 写 CSV
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_concept", "dst_concept", "rel_type", "weight", "source"])
        for e in edges.values():
            if e.weight <= 0:
                continue
            w.writerow([e.src, e.dst, e.rel_type, float(e.weight), e.source])

    print(f"[OK] Wrote {len(edges)} edges -> {args.out_csv}")


if __name__ == "__main__":
    main()

