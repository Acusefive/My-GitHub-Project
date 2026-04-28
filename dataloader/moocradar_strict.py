from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset

from dataloader.context_map import ContextEmbeddingMap
from scripts.common_pipeline_strict.io_utils import load_problem_records, load_student_sequences


def match_seq_len_with_context(
    q_seqs: Sequence[np.ndarray],
    r_seqs: Sequence[np.ndarray],
    eval_mask_seqs: Sequence[np.ndarray],
    ctx_main_seqs: Sequence[np.ndarray],
    ctx_tpl_seqs: Sequence[np.ndarray],
    ctx_llm_seqs: Sequence[np.ndarray],
    ctx_llm_struct_seqs: Sequence[np.ndarray],
    ctx_llm_struct_feature_seqs: Sequence[np.ndarray],
    seq_user_ids: Sequence[str],
    seq_len: int,
    *,
    pad_val: int = -1,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[str],
]:
    proc_q: List[np.ndarray] = []
    proc_r: List[np.ndarray] = []
    proc_eval_masks: List[np.ndarray] = []
    proc_ctx_main: List[np.ndarray] = []
    proc_ctx_tpl: List[np.ndarray] = []
    proc_ctx_llm: List[np.ndarray] = []
    proc_user_ids: List[str] = []
    proc_ctx_llm_struct: List[np.ndarray] = []
    proc_ctx_llm_struct_features: List[np.ndarray] = []

    for q_seq, r_seq, eval_mask_seq, ctx_main_seq, ctx_tpl_seq, ctx_llm_seq, ctx_llm_struct_seq, ctx_llm_struct_feature_seq, user_id in zip(
        q_seqs, r_seqs, eval_mask_seqs, ctx_main_seqs, ctx_tpl_seqs, ctx_llm_seqs, ctx_llm_struct_seqs, ctx_llm_struct_feature_seqs, seq_user_ids
    ):
        i = 0
        total_len = len(q_seq)
        while i + seq_len + 1 < total_len:
            proc_q.append(q_seq[i : i + seq_len + 1])
            proc_r.append(r_seq[i : i + seq_len + 1])
            proc_eval_masks.append(eval_mask_seq[i : i + seq_len + 1])
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
        proc_eval_masks.append(np.concatenate([eval_mask_seq[i:], np.zeros((pad_count,), dtype=np.int64)], axis=0))
        proc_ctx_main.append(
            np.concatenate([ctx_main_seq[i:], np.zeros((pad_count, ctx_main_seq.shape[1]), dtype=ctx_main_seq.dtype)], axis=0)
        )
        proc_ctx_tpl.append(
            np.concatenate([ctx_tpl_seq[i:], np.zeros((pad_count, ctx_tpl_seq.shape[1]), dtype=ctx_tpl_seq.dtype)], axis=0)
        )
        proc_ctx_llm.append(
            np.concatenate([ctx_llm_seq[i:], np.zeros((pad_count, ctx_llm_seq.shape[1]), dtype=ctx_llm_seq.dtype)], axis=0)
        )
        proc_ctx_llm_struct.append(
            np.concatenate(
                [ctx_llm_struct_seq[i:], np.zeros((pad_count, ctx_llm_struct_seq.shape[1]), dtype=ctx_llm_struct_seq.dtype)],
                axis=0,
            )
        )
        proc_ctx_llm_struct_features.append(
            np.concatenate(
                [ctx_llm_struct_feature_seq[i:], np.zeros((pad_count, ctx_llm_struct_feature_seq.shape[1]), dtype=ctx_llm_struct_feature_seq.dtype)],
                axis=0,
            )
        )
        proc_user_ids.append(user_id)

    return (
        proc_q,
        proc_r,
        proc_eval_masks,
        proc_ctx_main,
        proc_ctx_tpl,
        proc_ctx_llm,
        proc_ctx_llm_struct,
        proc_ctx_llm_struct_features,
        proc_user_ids,
    )


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
        split_mode: str = "user",
        split_role: str = "all",
        seed: int = 42,
        test_concept_ratio: float = 0.2,
        cache_preprocessed: bool = True,
        context_storage_dtype: str = "float32",
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
        self.split_mode = str(split_mode)
        self.split_role = str(split_role)
        self.seed = int(seed)
        self.test_concept_ratio = float(test_concept_ratio)
        self.cache_preprocessed = bool(cache_preprocessed)
        if context_storage_dtype not in {"float32", "float16"}:
            raise ValueError(f"Unsupported context_storage_dtype: {context_storage_dtype}")
        self.context_storage_dtype = np.float16 if context_storage_dtype == "float16" else np.float32
        if self.split_mode not in {"user", "new_concept"}:
            raise ValueError(f"Unsupported split_mode: {self.split_mode}")
        if self.split_mode == "new_concept" and self.split_role not in {"train_valid", "test"}:
            raise ValueError("split_role must be train_valid or test when split_mode=new_concept")
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        context_tag = f"{self.context_embeddings_path.stem}_{self.context_embeddings_path.stat().st_size}"
        split_tag = (
            f"{self.split_mode}_{self.split_role}_seq{self.seq_len}_seed{self.seed}_"
            f"ratio{str(self.test_concept_ratio).replace('.', 'p')}_ctx{context_storage_dtype}"
        )
        cache_prefix = self.dataset_dir / f"moocradar_strict_{context_tag}_{split_tag}"
        base_cache = {
            "q_seqs": cache_prefix.with_name(cache_prefix.name + "_q_seqs.pkl"),
            "r_seqs": cache_prefix.with_name(cache_prefix.name + "_r_seqs.pkl"),
            "eval_mask_seqs": cache_prefix.with_name(cache_prefix.name + "_eval_mask_seqs.pkl"),
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
            "split_stats": cache_prefix.with_name(cache_prefix.name + "_split_stats.pkl"),
        }

        self.lazy_test = self.split_mode == "new_concept" and self.split_role == "test"
        if self.lazy_test:
            self._preprocess_lazy_test()
        elif self.cache_preprocessed and all(path.exists() for path in base_cache.values()):
            self.q_seqs = pickle.loads(base_cache["q_seqs"].read_bytes())
            self.r_seqs = pickle.loads(base_cache["r_seqs"].read_bytes())
            self.eval_mask_seqs = pickle.loads(base_cache["eval_mask_seqs"].read_bytes())
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
            self.split_stats = pickle.loads(base_cache["split_stats"].read_bytes())
        else:
            (
                self.q_seqs,
                self.r_seqs,
                self.eval_mask_seqs,
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
                self.split_stats,
            ) = self.preprocess()
            if self.cache_preprocessed:
                for name, path in base_cache.items():
                    path.write_bytes(pickle.dumps(getattr(self, name if name != "context_dim" else "context_dim")))

        self.num_u = int(len(self.u_list))
        self.num_q = int(len(self.q_list))

        if self.lazy_test:
            self.sample_user_ids = list(self.lazy_sample_user_ids)
            self.len = len(self.test_samples)
            return

        if self.seq_len:
            (
                self.q_seqs,
                self.r_seqs,
                self.eval_mask_seqs,
                self.ctx_main_seqs,
                self.ctx_tpl_seqs,
                self.ctx_llm_seqs,
                self.ctx_llm_struct_seqs,
                self.ctx_llm_struct_feature_seqs,
                self.sample_user_ids,
            ) = match_seq_len_with_context(
                self.q_seqs,
                self.r_seqs,
                self.eval_mask_seqs,
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
        if getattr(self, "lazy_test", False):
            return self._get_lazy_test_item(index)
        return (
            self.q_seqs[index],
            self.r_seqs[index],
            self.eval_mask_seqs[index],
            self.ctx_main_seqs[index],
            self.ctx_tpl_seqs[index],
            self.ctx_llm_seqs[index],
            self.ctx_llm_struct_seqs[index],
            self.ctx_llm_struct_feature_seqs[index],
        )

    def __len__(self):
        return self.len

    def _build_sequence_arrays(
        self,
        log_items,
        q2idx: Dict[str, int],
        context_map: ContextEmbeddingMap,
        user_id: str,
        zero_ctx: np.ndarray,
        zero_struct_ctx: np.ndarray,
        zero_struct_feature_ctx: np.ndarray,
    ):
        q_seq = np.asarray([q2idx[str(log["problem_id"])] for _, log in log_items], dtype=np.int64)
        r_seq = np.asarray([int(log.get("is_correct") or 0) for _, log in log_items], dtype=np.int64)
        ctx_main_seq = np.zeros((len(log_items), context_map.context_dim), dtype=self.context_storage_dtype)
        ctx_tpl_seq = np.zeros((len(log_items), context_map.context_dim), dtype=self.context_storage_dtype)
        ctx_llm_seq = np.zeros((len(log_items), context_map.context_dim), dtype=self.context_storage_dtype)
        ctx_llm_struct_seq = np.zeros((len(log_items), context_map.llm_struct_dim), dtype=self.context_storage_dtype)
        ctx_llm_struct_feature_seq = np.zeros((len(log_items), context_map.llm_struct_feature_dim), dtype=self.context_storage_dtype)

        for pos, (target_t, log) in enumerate(log_items):
            if pos == 0:
                continue
            pid = str(log["problem_id"])
            key = (user_id, target_t, pid)
            main_value = context_map.get_main(key)
            tpl_value = context_map.get_template(key)
            llm_value = context_map.get_llm(key)
            llm_struct_value = context_map.get_llm_struct(key)
            llm_struct_feature_value = context_map.get_llm_struct_features(key)
            ctx_main_seq[pos] = main_value if main_value is not None else zero_ctx
            ctx_tpl_seq[pos] = tpl_value if tpl_value is not None else zero_ctx
            if self.require_llm_context and llm_value is None:
                raise ValueError(f"Missing llm context embedding for {key}")
            if self.require_llm_struct_context and llm_struct_value is None:
                raise ValueError(f"Missing llm structured embedding for {key}")
            if self.require_llm_struct_feature_context and llm_struct_feature_value is None:
                raise ValueError(f"Missing llm structured feature vector for {key}")
            ctx_llm_seq[pos] = llm_value if llm_value is not None else zero_ctx
            ctx_llm_struct_seq[pos] = llm_struct_value if llm_struct_value is not None else zero_struct_ctx
            ctx_llm_struct_feature_seq[pos] = (
                llm_struct_feature_value if llm_struct_feature_value is not None else zero_struct_feature_ctx
            )

        return q_seq, r_seq, ctx_main_seq, ctx_tpl_seq, ctx_llm_seq, ctx_llm_struct_seq, ctx_llm_struct_feature_seq

    @staticmethod
    def _append_arrays(
        q_seqs,
        r_seqs,
        eval_mask_seqs,
        ctx_main_seqs,
        ctx_tpl_seqs,
        ctx_llm_seqs,
        ctx_llm_struct_seqs,
        ctx_llm_struct_feature_seqs,
        seq_user_ids,
        user_id,
        arrays,
        eval_mask,
    ) -> None:
        q_seq, r_seq, ctx_main_seq, ctx_tpl_seq, ctx_llm_seq, ctx_llm_struct_seq, ctx_llm_struct_feature_seq = arrays
        q_seqs.append(q_seq)
        r_seqs.append(r_seq)
        eval_mask_seqs.append(np.asarray(eval_mask, dtype=np.int64))
        ctx_main_seqs.append(ctx_main_seq)
        ctx_tpl_seqs.append(ctx_tpl_seq)
        ctx_llm_seqs.append(ctx_llm_seq)
        ctx_llm_struct_seqs.append(ctx_llm_struct_seq)
        ctx_llm_struct_feature_seqs.append(ctx_llm_struct_feature_seq)
        seq_user_ids.append(user_id)

    def _preprocess_lazy_test(self) -> None:
        problem_records = load_problem_records(self.problem_json)
        student_sequences = load_student_sequences(self.student_json)
        self.context_map = ContextEmbeddingMap(self.context_embeddings_path)

        self.q_list = np.asarray(sorted({problem.problem_id for problem in problem_records}))
        self.u_list = np.asarray([sequence.user_id for sequence in student_sequences])
        self.q2idx = {pid: idx for idx, pid in enumerate(self.q_list.tolist())}
        self.u2idx = {uid: idx for idx, uid in enumerate(self.u_list.tolist())}
        pid2concepts = {problem.problem_id: set(problem.concepts) for problem in problem_records}

        self.context_dim = self.context_map.context_dim
        self.has_llm_context = self.context_map.has_llm
        self.llm_struct_dim = self.context_map.llm_struct_dim
        self.has_llm_struct_context = self.context_map.has_llm_struct
        self.llm_struct_feature_dim = self.context_map.llm_struct_feature_dim
        self.has_llm_struct_feature_context = self.context_map.has_llm_struct_features
        self.zero_ctx = np.zeros((self.context_dim,), dtype=self.context_storage_dtype)
        self.zero_struct_ctx = np.zeros((self.llm_struct_dim,), dtype=self.context_storage_dtype)
        self.zero_struct_feature_ctx = np.zeros((self.llm_struct_feature_dim,), dtype=self.context_storage_dtype)

        all_concepts = sorted({concept for concepts in pid2concepts.values() for concept in concepts})
        rng = np.random.default_rng(self.seed)
        shuffled_concepts = np.asarray(all_concepts, dtype=object)
        if len(shuffled_concepts) > 0:
            shuffled_concepts = shuffled_concepts[rng.permutation(len(shuffled_concepts))]
        test_concept_count = max(1, int(len(shuffled_concepts) * self.test_concept_ratio)) if len(shuffled_concepts) else 0
        test_concepts = set(shuffled_concepts[:test_concept_count].tolist())
        train_concepts = set(all_concepts) - test_concepts

        self.lazy_sequences = []
        self.test_samples = []
        self.lazy_sample_user_ids = []
        train_target_concepts = set()
        test_target_concepts = set()
        skipped_test_targets_no_old_history = 0
        train_target_interactions = 0
        test_target_interactions = 0

        for sequence in student_sequences:
            raw_filtered_logs = [log for log in sequence.seq if str(log.get("problem_id") or "") in self.q2idx]
            if len(raw_filtered_logs) < 2:
                continue

            pids = [str(log["problem_id"]) for log in raw_filtered_logs]
            qids = np.asarray([self.q2idx[pid] for pid in pids], dtype=np.int64)
            responses = np.asarray([int(log.get("is_correct") or 0) for log in raw_filtered_logs], dtype=np.int64)
            target_ts = np.arange(len(raw_filtered_logs), dtype=np.int64)
            is_test_target = np.zeros((len(raw_filtered_logs),), dtype=bool)
            old_mask = np.zeros((len(raw_filtered_logs),), dtype=bool)
            has_old_history = False
            user_idx = len(self.lazy_sequences)

            for pos, pid in enumerate(pids):
                concepts = pid2concepts.get(pid, set())
                if concepts & test_concepts:
                    is_test_target[pos] = True
                    test_target_concepts.update(concepts & test_concepts)
                    test_target_interactions += 1
                    if not has_old_history:
                        skipped_test_targets_no_old_history += 1
                        continue
                    self.test_samples.append((user_idx, pos))
                    self.lazy_sample_user_ids.append(sequence.user_id)
                else:
                    old_mask[pos] = True
                    has_old_history = True
                    train_target_concepts.update(concepts & train_concepts)
                    train_target_interactions += 1

            self.lazy_sequences.append(
                {
                    "user_id": sequence.user_id,
                    "pids": pids,
                    "qids": qids,
                    "responses": responses,
                    "target_ts": target_ts,
                    "old_mask": old_mask,
                    "is_test_target": is_test_target,
                }
            )

        self.split_stats = {
            "split_mode": self.split_mode,
            "split_role": self.split_role,
            "lazy_test": True,
            "seed": self.seed,
            "test_concept_ratio": self.test_concept_ratio,
            "total_concepts": len(all_concepts),
            "configured_train_concepts": len(train_concepts),
            "configured_test_concepts": len(test_concepts),
            "train_target_concepts": len(train_target_concepts),
            "test_target_concepts": len(test_target_concepts),
            "train_test_concept_overlap": len(train_target_concepts & test_target_concepts),
            "train_target_interactions": train_target_interactions,
            "test_target_interactions": test_target_interactions,
            "skipped_test_targets_no_old_history": skipped_test_targets_no_old_history,
            "sequence_count": len(self.test_samples),
        }

    def _get_lazy_test_item(self, index):
        user_idx, target_pos = self.test_samples[index]
        record = self.lazy_sequences[user_idx]
        positions = []
        pos = int(target_pos) - 1
        while pos >= 0 and len(positions) < self.seq_len:
            if bool(record["old_mask"][pos]):
                positions.append(pos)
            pos -= 1
        positions.reverse()
        positions.append(int(target_pos))

        q_seq = np.asarray([record["qids"][pos] for pos in positions], dtype=np.int64)
        r_seq = np.asarray([record["responses"][pos] for pos in positions], dtype=np.int64)
        eval_mask = np.zeros((len(positions),), dtype=np.int64)
        eval_mask[-1] = 1

        ctx_main_seq = np.zeros((len(positions), self.context_dim), dtype=self.context_storage_dtype)
        ctx_tpl_seq = np.zeros((len(positions), self.context_dim), dtype=self.context_storage_dtype)
        ctx_llm_seq = np.zeros((len(positions), self.context_dim), dtype=self.context_storage_dtype)
        ctx_llm_struct_seq = np.zeros((len(positions), self.llm_struct_dim), dtype=self.context_storage_dtype)
        ctx_llm_struct_feature_seq = np.zeros((len(positions), self.llm_struct_feature_dim), dtype=self.context_storage_dtype)

        user_id = record["user_id"]
        for out_pos, src_pos in enumerate(positions):
            if out_pos == 0:
                continue
            pid = record["pids"][src_pos]
            key = (user_id, int(record["target_ts"][src_pos]), pid)
            main_value = self.context_map.get_main(key)
            tpl_value = self.context_map.get_template(key)
            llm_value = self.context_map.get_llm(key)
            llm_struct_value = self.context_map.get_llm_struct(key)
            llm_struct_feature_value = self.context_map.get_llm_struct_features(key)
            ctx_main_seq[out_pos] = main_value if main_value is not None else self.zero_ctx
            ctx_tpl_seq[out_pos] = tpl_value if tpl_value is not None else self.zero_ctx
            if self.require_llm_context and llm_value is None:
                raise ValueError(f"Missing llm context embedding for {key}")
            if self.require_llm_struct_context and llm_struct_value is None:
                raise ValueError(f"Missing llm structured embedding for {key}")
            if self.require_llm_struct_feature_context and llm_struct_feature_value is None:
                raise ValueError(f"Missing llm structured feature vector for {key}")
            ctx_llm_seq[out_pos] = llm_value if llm_value is not None else self.zero_ctx
            ctx_llm_struct_seq[out_pos] = llm_struct_value if llm_struct_value is not None else self.zero_struct_ctx
            ctx_llm_struct_feature_seq[out_pos] = (
                llm_struct_feature_value if llm_struct_feature_value is not None else self.zero_struct_feature_ctx
            )

        return (
            q_seq,
            r_seq,
            eval_mask,
            ctx_main_seq,
            ctx_tpl_seq,
            ctx_llm_seq,
            ctx_llm_struct_seq,
            ctx_llm_struct_feature_seq,
        )

    def preprocess(self):
        problem_records = load_problem_records(self.problem_json)
        student_sequences = load_student_sequences(self.student_json)
        context_map = ContextEmbeddingMap(self.context_embeddings_path)

        q_list = np.asarray(sorted({problem.problem_id for problem in problem_records}))
        u_list = np.asarray([sequence.user_id for sequence in student_sequences])
        q2idx = {pid: idx for idx, pid in enumerate(q_list.tolist())}
        u2idx = {uid: idx for idx, uid in enumerate(u_list.tolist())}
        pid2concepts = {problem.problem_id: set(problem.concepts) for problem in problem_records}

        q_seqs: List[np.ndarray] = []
        r_seqs: List[np.ndarray] = []
        eval_mask_seqs: List[np.ndarray] = []
        ctx_main_seqs: List[np.ndarray] = []
        ctx_tpl_seqs: List[np.ndarray] = []
        ctx_llm_seqs: List[np.ndarray] = []
        ctx_llm_struct_seqs: List[np.ndarray] = []
        ctx_llm_struct_feature_seqs: List[np.ndarray] = []
        seq_user_ids: List[str] = []
        zero_ctx = np.zeros((context_map.context_dim,), dtype=self.context_storage_dtype)
        zero_struct_ctx = np.zeros((context_map.llm_struct_dim,), dtype=self.context_storage_dtype)
        zero_struct_feature_ctx = np.zeros((context_map.llm_struct_feature_dim,), dtype=self.context_storage_dtype)

        all_concepts = sorted({concept for concepts in pid2concepts.values() for concept in concepts})
        rng = np.random.default_rng(self.seed)
        shuffled_concepts = np.asarray(all_concepts, dtype=object)
        if len(shuffled_concepts) > 0:
            shuffled_concepts = shuffled_concepts[rng.permutation(len(shuffled_concepts))]
        test_concept_count = max(1, int(len(shuffled_concepts) * self.test_concept_ratio)) if len(shuffled_concepts) else 0
        test_concepts = set(shuffled_concepts[:test_concept_count].tolist())
        train_concepts = set(all_concepts) - test_concepts

        train_target_concepts = set()
        test_target_concepts = set()
        skipped_test_targets_no_old_history = 0
        train_target_interactions = 0
        test_target_interactions = 0

        for sequence in student_sequences:
            raw_filtered_logs = [log for log in sequence.seq if str(log.get("problem_id") or "") in q2idx]
            filtered_logs = list(enumerate(raw_filtered_logs))
            if len(filtered_logs) < 2:
                continue

            if self.split_mode == "user":
                arrays = self._build_sequence_arrays(
                    filtered_logs,
                    q2idx,
                    context_map,
                    sequence.user_id,
                    zero_ctx,
                    zero_struct_ctx,
                    zero_struct_feature_ctx,
                )
                self._append_arrays(
                    q_seqs,
                    r_seqs,
                    eval_mask_seqs,
                    ctx_main_seqs,
                    ctx_tpl_seqs,
                    ctx_llm_seqs,
                    ctx_llm_struct_seqs,
                    ctx_llm_struct_feature_seqs,
                    seq_user_ids,
                    sequence.user_id,
                    arrays,
                    np.ones((len(filtered_logs),), dtype=np.int64),
                )
                continue

            old_history = []
            train_logs = []
            for target_t, log in filtered_logs:
                pid = str(log["problem_id"])
                concepts = pid2concepts.get(pid, set())
                is_test_target = bool(concepts & test_concepts)
                if is_test_target:
                    test_target_concepts.update(concepts & test_concepts)
                    test_target_interactions += 1
                    if len(old_history) == 0:
                        skipped_test_targets_no_old_history += 1
                        continue
                    if self.split_role == "test":
                        target_logs = (old_history + [(target_t, log)])[-(self.seq_len + 1) :]
                        arrays = self._build_sequence_arrays(
                            target_logs,
                            q2idx,
                            context_map,
                            sequence.user_id,
                            zero_ctx,
                            zero_struct_ctx,
                            zero_struct_feature_ctx,
                        )
                        eval_mask = np.zeros((len(target_logs),), dtype=np.int64)
                        eval_mask[-1] = 1
                        self._append_arrays(
                            q_seqs,
                            r_seqs,
                            eval_mask_seqs,
                            ctx_main_seqs,
                            ctx_tpl_seqs,
                            ctx_llm_seqs,
                            ctx_llm_struct_seqs,
                            ctx_llm_struct_feature_seqs,
                            seq_user_ids,
                            sequence.user_id,
                            arrays,
                            eval_mask,
                        )
                else:
                    train_logs.append((target_t, log))
                    old_history.append((target_t, log))
                    train_target_concepts.update(concepts & train_concepts)
                    train_target_interactions += 1

            if self.split_role == "train_valid" and len(train_logs) >= 2:
                arrays = self._build_sequence_arrays(
                    train_logs,
                    q2idx,
                    context_map,
                    sequence.user_id,
                    zero_ctx,
                    zero_struct_ctx,
                    zero_struct_feature_ctx,
                )
                self._append_arrays(
                    q_seqs,
                    r_seqs,
                    eval_mask_seqs,
                    ctx_main_seqs,
                    ctx_tpl_seqs,
                    ctx_llm_seqs,
                    ctx_llm_struct_seqs,
                    ctx_llm_struct_feature_seqs,
                    seq_user_ids,
                    sequence.user_id,
                    arrays,
                    np.ones((len(train_logs),), dtype=np.int64),
                )

        split_stats = {
            "split_mode": self.split_mode,
            "split_role": self.split_role,
            "seed": self.seed,
            "test_concept_ratio": self.test_concept_ratio,
            "total_concepts": len(all_concepts),
            "configured_train_concepts": len(train_concepts),
            "configured_test_concepts": len(test_concepts),
            "train_target_concepts": len(train_target_concepts),
            "test_target_concepts": len(test_target_concepts),
            "train_test_concept_overlap": len(train_target_concepts & test_target_concepts),
            "train_target_interactions": train_target_interactions,
            "test_target_interactions": test_target_interactions,
            "skipped_test_targets_no_old_history": skipped_test_targets_no_old_history,
            "sequence_count": len(q_seqs),
        }

        return (
            q_seqs,
            r_seqs,
            eval_mask_seqs,
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
            split_stats,
        )
