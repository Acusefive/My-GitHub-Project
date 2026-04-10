from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset

from dataloader.context_map import ContextEmbeddingMap
from scripts.common_pipeline_strict.io_utils import load_problem_records, load_student_sequences


def match_seq_len_with_context(
    q_seqs: Sequence[np.ndarray],
    r_seqs: Sequence[np.ndarray],
    ctx_main_seqs: Sequence[np.ndarray],
    ctx_tpl_seqs: Sequence[np.ndarray],
    ctx_llm_seqs: Sequence[np.ndarray],
    ctx_llm_struct_seqs: Sequence[np.ndarray],
    ctx_llm_struct_feature_seqs: Sequence[np.ndarray],
    seq_user_ids: Sequence[str],
    seq_len: int,
    *,
    pad_val: int = -1,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[str]]:
    proc_q: List[np.ndarray] = []
    proc_r: List[np.ndarray] = []
    proc_ctx_main: List[np.ndarray] = []
    proc_ctx_tpl: List[np.ndarray] = []
    proc_ctx_llm: List[np.ndarray] = []
    proc_user_ids: List[str] = []
    proc_ctx_llm_struct: List[np.ndarray] = []
    proc_ctx_llm_struct_features: List[np.ndarray] = []

    for q_seq, r_seq, ctx_main_seq, ctx_tpl_seq, ctx_llm_seq, ctx_llm_struct_seq, ctx_llm_struct_feature_seq, user_id in zip(
        q_seqs, r_seqs, ctx_main_seqs, ctx_tpl_seqs, ctx_llm_seqs, ctx_llm_struct_seqs, ctx_llm_struct_feature_seqs, seq_user_ids
    ):
        i = 0
        total_len = len(q_seq)
        while i + seq_len + 1 < total_len:
            proc_q.append(q_seq[i : i + seq_len + 1])
            proc_r.append(r_seq[i : i + seq_len + 1])
            proc_ctx_main.append(ctx_main_seq[i : i + seq_len + 1])
            proc_ctx_tpl.append(ctx_tpl_seq[i : i + seq_len + 1])
            proc_ctx_llm.append(ctx_llm_seq[i : i + seq_len + 1])
            proc_ctx_llm_struct.append(ctx_llm_struct_seq[i : i + seq_len + 1])
            proc_ctx_llm_struct_features.append(ctx_llm_struct_feature_seq[i : i + seq_len + 1])
            proc_user_ids.append(user_id)
            i += seq_len + 1

        pad_count = i + seq_len + 1 - total_len
        proc_q.append(np.concatenate([q_seq[i:], np.full((pad_count,), pad_val, dtype=np.int64)], axis=0))
        proc_r.append(np.concatenate([r_seq[i:], np.full((pad_count,), pad_val, dtype=np.int64)], axis=0))
        proc_ctx_main.append(
            np.concatenate([ctx_main_seq[i:], np.zeros((pad_count, ctx_main_seq.shape[1]), dtype=np.float32)], axis=0)
        )
        proc_ctx_tpl.append(
            np.concatenate([ctx_tpl_seq[i:], np.zeros((pad_count, ctx_tpl_seq.shape[1]), dtype=np.float32)], axis=0)
        )
        proc_ctx_llm.append(
            np.concatenate([ctx_llm_seq[i:], np.zeros((pad_count, ctx_llm_seq.shape[1]), dtype=np.float32)], axis=0)
        )
        proc_ctx_llm_struct.append(
            np.concatenate(
                [ctx_llm_struct_seq[i:], np.zeros((pad_count, ctx_llm_struct_seq.shape[1]), dtype=np.float32)],
                axis=0,
            )
        )
        proc_ctx_llm_struct_features.append(
            np.concatenate(
                [ctx_llm_struct_feature_seq[i:], np.zeros((pad_count, ctx_llm_struct_feature_seq.shape[1]), dtype=np.float32)],
                axis=0,
            )
        )
        proc_user_ids.append(user_id)

    return proc_q, proc_r, proc_ctx_main, proc_ctx_tpl, proc_ctx_llm, proc_ctx_llm_struct, proc_ctx_llm_struct_features, proc_user_ids


class MOOCRadarStrict(Dataset):
    def __init__(
        self,
        seq_len: int,
        *,
        problem_json: str | Path,
        student_json: str | Path,
        context_embeddings_path: str | Path,
        dataset_dir: str | Path,
        require_llm_context: bool = False,
        require_llm_struct_context: bool = False,
        require_llm_struct_feature_context: bool = False,
    ) -> None:
        super().__init__()

        self.seq_len = int(seq_len)
        self.problem_json = Path(problem_json).resolve()
        self.student_json = Path(student_json).resolve()
        self.context_embeddings_path = Path(context_embeddings_path).resolve()
        self.dataset_dir = Path(dataset_dir).resolve()
        self.require_llm_context = bool(require_llm_context)
        self.require_llm_struct_context = bool(require_llm_struct_context)
        self.require_llm_struct_feature_context = bool(require_llm_struct_feature_context)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        context_tag = f"{self.context_embeddings_path.stem}_{self.context_embeddings_path.stat().st_size}"
        cache_prefix = self.dataset_dir / f"moocradar_strict_{context_tag}"
        base_cache = {
            "q_seqs": cache_prefix.with_name(cache_prefix.name + "_q_seqs.pkl"),
            "r_seqs": cache_prefix.with_name(cache_prefix.name + "_r_seqs.pkl"),
            "ctx_main_seqs": cache_prefix.with_name(cache_prefix.name + "_ctx_main_seqs.pkl"),
            "ctx_tpl_seqs": cache_prefix.with_name(cache_prefix.name + "_ctx_tpl_seqs.pkl"),
            "ctx_llm_seqs": cache_prefix.with_name(cache_prefix.name + "_ctx_llm_seqs.pkl"),
            "ctx_llm_struct_seqs": cache_prefix.with_name(cache_prefix.name + "_ctx_llm_struct_seqs.pkl"),
            "ctx_llm_struct_feature_seqs": cache_prefix.with_name(cache_prefix.name + "_ctx_llm_struct_feature_seqs.pkl"),
            "q_list": cache_prefix.with_name(cache_prefix.name + "_q_list.pkl"),
            "u_list": cache_prefix.with_name(cache_prefix.name + "_u_list.pkl"),
            "q2idx": cache_prefix.with_name(cache_prefix.name + "_q2idx.pkl"),
            "u2idx": cache_prefix.with_name(cache_prefix.name + "_u2idx.pkl"),
            "context_dim": cache_prefix.with_name(cache_prefix.name + "_context_dim.pkl"),
            "has_llm_context": cache_prefix.with_name(cache_prefix.name + "_has_llm_context.pkl"),
            "llm_struct_dim": cache_prefix.with_name(cache_prefix.name + "_llm_struct_dim.pkl"),
            "has_llm_struct_context": cache_prefix.with_name(cache_prefix.name + "_has_llm_struct_context.pkl"),
            "llm_struct_feature_dim": cache_prefix.with_name(cache_prefix.name + "_llm_struct_feature_dim.pkl"),
            "has_llm_struct_feature_context": cache_prefix.with_name(cache_prefix.name + "_has_llm_struct_feature_context.pkl"),
            "seq_user_ids": cache_prefix.with_name(cache_prefix.name + "_seq_user_ids.pkl"),
        }

        if all(path.exists() for path in base_cache.values()):
            self.q_seqs = pickle.loads(base_cache["q_seqs"].read_bytes())
            self.r_seqs = pickle.loads(base_cache["r_seqs"].read_bytes())
            self.ctx_main_seqs = pickle.loads(base_cache["ctx_main_seqs"].read_bytes())
            self.ctx_tpl_seqs = pickle.loads(base_cache["ctx_tpl_seqs"].read_bytes())
            self.ctx_llm_seqs = pickle.loads(base_cache["ctx_llm_seqs"].read_bytes())
            self.ctx_llm_struct_seqs = pickle.loads(base_cache["ctx_llm_struct_seqs"].read_bytes())
            self.ctx_llm_struct_feature_seqs = pickle.loads(base_cache["ctx_llm_struct_feature_seqs"].read_bytes())
            self.q_list = pickle.loads(base_cache["q_list"].read_bytes())
            self.u_list = pickle.loads(base_cache["u_list"].read_bytes())
            self.q2idx = pickle.loads(base_cache["q2idx"].read_bytes())
            self.u2idx = pickle.loads(base_cache["u2idx"].read_bytes())
            self.context_dim = pickle.loads(base_cache["context_dim"].read_bytes())
            self.has_llm_context = pickle.loads(base_cache["has_llm_context"].read_bytes())
            self.llm_struct_dim = pickle.loads(base_cache["llm_struct_dim"].read_bytes())
            self.has_llm_struct_context = pickle.loads(base_cache["has_llm_struct_context"].read_bytes())
            self.llm_struct_feature_dim = pickle.loads(base_cache["llm_struct_feature_dim"].read_bytes())
            self.has_llm_struct_feature_context = pickle.loads(base_cache["has_llm_struct_feature_context"].read_bytes())
            self.seq_user_ids = pickle.loads(base_cache["seq_user_ids"].read_bytes())
        else:
            (
                self.q_seqs,
                self.r_seqs,
                self.ctx_main_seqs,
                self.ctx_tpl_seqs,
                self.ctx_llm_seqs,
                self.ctx_llm_struct_seqs,
                self.ctx_llm_struct_feature_seqs,
                self.q_list,
                self.u_list,
                self.q2idx,
                self.u2idx,
                self.context_dim,
                self.has_llm_context,
                self.llm_struct_dim,
                self.has_llm_struct_context,
                self.llm_struct_feature_dim,
                self.has_llm_struct_feature_context,
                self.seq_user_ids,
            ) = self.preprocess()
            for name, path in base_cache.items():
                path.write_bytes(pickle.dumps(getattr(self, name if name != "context_dim" else "context_dim")))

        self.num_u = int(len(self.u_list))
        self.num_q = int(len(self.q_list))

        if self.seq_len:
            (
                self.q_seqs,
                self.r_seqs,
                self.ctx_main_seqs,
                self.ctx_tpl_seqs,
                self.ctx_llm_seqs,
                self.ctx_llm_struct_seqs,
                self.ctx_llm_struct_feature_seqs,
                self.sample_user_ids,
            ) = match_seq_len_with_context(
                self.q_seqs,
                self.r_seqs,
                self.ctx_main_seqs,
                self.ctx_tpl_seqs,
                self.ctx_llm_seqs,
                self.ctx_llm_struct_seqs,
                self.ctx_llm_struct_feature_seqs,
                self.seq_user_ids,
                self.seq_len,
            )
        else:
            self.sample_user_ids = list(self.seq_user_ids)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return (
            self.q_seqs[index],
            self.r_seqs[index],
            self.ctx_main_seqs[index],
            self.ctx_tpl_seqs[index],
            self.ctx_llm_seqs[index],
            self.ctx_llm_struct_seqs[index],
            self.ctx_llm_struct_feature_seqs[index],
        )

    def __len__(self):
        return self.len

    def preprocess(self):
        problem_records = load_problem_records(self.problem_json)
        student_sequences = load_student_sequences(self.student_json)
        context_map = ContextEmbeddingMap(self.context_embeddings_path)

        q_list = np.asarray(sorted({problem.problem_id for problem in problem_records}))
        u_list = np.asarray([sequence.user_id for sequence in student_sequences])
        q2idx = {pid: idx for idx, pid in enumerate(q_list.tolist())}
        u2idx = {uid: idx for idx, uid in enumerate(u_list.tolist())}

        q_seqs: List[np.ndarray] = []
        r_seqs: List[np.ndarray] = []
        ctx_main_seqs: List[np.ndarray] = []
        ctx_tpl_seqs: List[np.ndarray] = []
        ctx_llm_seqs: List[np.ndarray] = []
        ctx_llm_struct_seqs: List[np.ndarray] = []
        ctx_llm_struct_feature_seqs: List[np.ndarray] = []
        seq_user_ids: List[str] = []
        zero_ctx = np.zeros((context_map.context_dim,), dtype=np.float32)
        zero_struct_ctx = np.zeros((context_map.llm_struct_dim,), dtype=np.float32)
        zero_struct_feature_ctx = np.zeros((context_map.llm_struct_feature_dim,), dtype=np.float32)

        for sequence in student_sequences:
            filtered_logs = [log for log in sequence.seq if str(log.get("problem_id") or "") in q2idx]
            if len(filtered_logs) < 2:
                continue

            q_seq = np.asarray([q2idx[str(log["problem_id"])] for log in filtered_logs], dtype=np.int64)
            r_seq = np.asarray([int(log.get("is_correct") or 0) for log in filtered_logs], dtype=np.int64)
            ctx_main_seq = np.zeros((len(filtered_logs), context_map.context_dim), dtype=np.float32)
            ctx_tpl_seq = np.zeros((len(filtered_logs), context_map.context_dim), dtype=np.float32)
            ctx_llm_seq = np.zeros((len(filtered_logs), context_map.context_dim), dtype=np.float32)
            ctx_llm_struct_seq = np.zeros((len(filtered_logs), context_map.llm_struct_dim), dtype=np.float32)
            ctx_llm_struct_feature_seq = np.zeros((len(filtered_logs), context_map.llm_struct_feature_dim), dtype=np.float32)

            for target_t in range(1, len(filtered_logs)):
                pid = str(filtered_logs[target_t]["problem_id"])
                key = (sequence.user_id, target_t, pid)
                main_value = context_map.get_main(key)
                tpl_value = context_map.get_template(key)
                llm_value = context_map.get_llm(key)
                llm_struct_value = context_map.get_llm_struct(key)
                llm_struct_feature_value = context_map.get_llm_struct_features(key)
                ctx_main_seq[target_t] = main_value if main_value is not None else zero_ctx
                ctx_tpl_seq[target_t] = tpl_value if tpl_value is not None else zero_ctx
                if self.require_llm_context and llm_value is None:
                    raise ValueError(f"Missing llm context embedding for {key}")
                if self.require_llm_struct_context and llm_struct_value is None:
                    raise ValueError(f"Missing llm structured embedding for {key}")
                if self.require_llm_struct_feature_context and llm_struct_feature_value is None:
                    raise ValueError(f"Missing llm structured feature vector for {key}")
                ctx_llm_seq[target_t] = llm_value if llm_value is not None else zero_ctx
                ctx_llm_struct_seq[target_t] = llm_struct_value if llm_struct_value is not None else zero_struct_ctx
                ctx_llm_struct_feature_seq[target_t] = (
                    llm_struct_feature_value if llm_struct_feature_value is not None else zero_struct_feature_ctx
                )

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)
            ctx_main_seqs.append(ctx_main_seq)
            ctx_tpl_seqs.append(ctx_tpl_seq)
            ctx_llm_seqs.append(ctx_llm_seq)
            ctx_llm_struct_seqs.append(ctx_llm_struct_seq)
            ctx_llm_struct_feature_seqs.append(ctx_llm_struct_feature_seq)
            seq_user_ids.append(sequence.user_id)

        return (
            q_seqs,
            r_seqs,
            ctx_main_seqs,
            ctx_tpl_seqs,
            ctx_llm_seqs,
            ctx_llm_struct_seqs,
            ctx_llm_struct_feature_seqs,
            q_list,
            u_list,
            q2idx,
            u2idx,
            context_map.context_dim,
            context_map.has_llm,
            context_map.llm_struct_dim,
            context_map.has_llm_struct,
            context_map.llm_struct_feature_dim,
            context_map.has_llm_struct_features,
            seq_user_ids,
        )
