"""
为 AKT-master 生成与 PID_DATA.load_data 完全对齐的语义向量切块文件。

关键点：
- AKT-master 的 PID_DATA 会把每个学生的长序列按 seqlen(默认200)切成多个 chunk；
  因此语义向量也必须按同样规则切块，并对齐到最终的 q_dataArray.shape[0]（chunk 数）。
- 本脚本复刻 convert_moocradar_to_akt.py 的“学生过滤 + 固定随机种子 shuffle + train/valid/test 划分”，
  保证生成的 *_sem.pkl 与 moocradar_pid_{train,valid,test}1.csv 的写入顺序一致。

输出（默认路径）：
  AKT-master/data/MOOCRadar/akt_format/
    - moocradar_pid_train_sem.pkl   # np.ndarray, shape [num_chunks, seqlen, 512]
    - moocradar_pid_valid_sem.pkl
    - moocradar_pid_test_sem.pkl
"""

import argparse
import json
import os
import pickle
from typing import List, Tuple

import numpy as np


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_students_and_semantic(
    raw_students: List[dict],
    semantic_by_student: List[List[np.ndarray]],
) -> List[Tuple[List[dict], List[np.ndarray]]]:
    """
    返回与 convert_moocradar_to_akt.py 同样的学生过滤规则下的列表：
      items[i] = (student_seq_items, student_sem_vectors)
    其中 student_seq_items 只保留 problem_id 存在的交互；语义也同步过滤。
    """
    if len(raw_students) != len(semantic_by_student):
        raise ValueError(
            f"语义 pkl 外层长度 {len(semantic_by_student)} 与 JSON 学生数 {len(raw_students)} 不一致。"
        )

    items: List[Tuple[List[dict], List[np.ndarray]]] = []
    for stu_idx, (stu, sem_seq) in enumerate(zip(raw_students, semantic_by_student)):
        seq = stu.get("seq", [])
        if not seq:
            continue

        filtered_seq: List[dict] = []
        filtered_sem: List[np.ndarray] = []

        # 假设 semantic_by_student[stu_idx] 与 seq 的时间步一一对应
        if len(sem_seq) != len(seq):
            raise ValueError(
                f"学生 {stu_idx} 的语义长度与 seq 不一致：len(sem)={len(sem_seq)} len(seq)={len(seq)}"
            )

        for t, item in enumerate(seq):
            problem_id = item.get("problem_id")
            if not problem_id:
                continue
            filtered_seq.append(
                {
                    "problem_id": problem_id,
                    "is_correct": int(item.get("is_correct", 0)),
                }
            )
            filtered_sem.append(np.asarray(sem_seq[t], dtype=np.float32))

        if filtered_seq:
            if len(filtered_seq) != len(filtered_sem):
                raise ValueError(
                    f"学生 {stu_idx} 过滤后长度不一致：len(seq)={len(filtered_seq)} len(sem)={len(filtered_sem)}"
                )
            items.append((filtered_seq, filtered_sem))

    return items


def split_train_valid_test(items, train_ratio=0.8, valid_ratio=0.1, seed=224):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(items))
    rng.shuffle(idx)
    items = [items[i] for i in idx.tolist()]

    total = len(items)
    train_end = int(total * train_ratio)
    valid_end = int(total * (train_ratio + valid_ratio))
    return items[:train_end], items[train_end:valid_end], items[valid_end:]


def chunk_and_pad_semantic(items, seqlen: int, semantic_dim: int) -> np.ndarray:
    """
    将学生级语义序列切成 PID_DATA.load_data 同款 chunk，并 pad 到 [seqlen, semantic_dim]。
    返回 shape [num_chunks, seqlen, semantic_dim] 的 float32 数组。
    """
    chunks: List[np.ndarray] = []

    for _, sem_seq in items:
        length = len(sem_seq)
        n_split = 1
        if length > seqlen:
            n_split = int(np.floor(length / seqlen))
            if length % seqlen:
                n_split += 1
        for k in range(n_split):
            start = k * seqlen
            end = length if k == n_split - 1 else (k + 1) * seqlen
            seg = sem_seq[start:end]
            mat = np.zeros((seqlen, semantic_dim), dtype=np.float32)
            for i, v in enumerate(seg):
                if v.shape != (semantic_dim,):
                    v = np.asarray(v, dtype=np.float32).reshape(-1)
                    if v.shape[0] != semantic_dim:
                        raise ValueError(f"语义维度不匹配：got {v.shape}, expected ({semantic_dim},)")
                mat[i] = v
            # padding 已经是 0
            chunks.append(mat)

    if len(chunks) == 0:
        return np.zeros((0, seqlen, semantic_dim), dtype=np.float32)
    return np.stack(chunks, axis=0).astype(np.float32, copy=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="student-problem-fine.json 路径")
    parser.add_argument("--semantic_pkl", type=str, required=True, help="全量语义向量 pkl（按学生、按交互）")
    parser.add_argument("--out_dir", type=str, required=True, help="输出目录（通常是 data/MOOCRadar/akt_format）")
    parser.add_argument("--seqlen", type=int, default=200)
    parser.add_argument("--semantic_dim", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=224)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    raw_students = load_json(args.json_path)
    semantic_by_student = load_pickle(args.semantic_pkl)

    items = build_students_and_semantic(raw_students, semantic_by_student)
    train_items, valid_items, test_items = split_train_valid_test(
        items, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio, seed=args.seed
    )

    train_sem = chunk_and_pad_semantic(train_items, seqlen=args.seqlen, semantic_dim=args.semantic_dim)
    valid_sem = chunk_and_pad_semantic(valid_items, seqlen=args.seqlen, semantic_dim=args.semantic_dim)
    test_sem = chunk_and_pad_semantic(test_items, seqlen=args.seqlen, semantic_dim=args.semantic_dim)

    train_path = os.path.join(args.out_dir, "moocradar_pid_train_sem.pkl")
    valid_path = os.path.join(args.out_dir, "moocradar_pid_valid_sem.pkl")
    test_path = os.path.join(args.out_dir, "moocradar_pid_test_sem.pkl")

    with open(train_path, "wb") as f:
        pickle.dump(train_sem, f)
    with open(valid_path, "wb") as f:
        pickle.dump(valid_sem, f)
    with open(test_path, "wb") as f:
        pickle.dump(test_sem, f)

    print("Saved:")
    print(" ", train_path, train_sem.shape)
    print(" ", valid_path, valid_sem.shape)
    print(" ", test_path, test_sem.shape)


if __name__ == "__main__":
    main()

