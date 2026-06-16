"""Context 增强知识追踪模型的统一训练、验证和测试入口。

主流程：
1. 根据 ``context_type`` 选择并拼接需要的 Context；
2. 根据实验划分构造训练、验证和测试数据集；
3. 创建指定知识追踪模型及 Context 融合模块；
4. 训练并按验证 AUC 保存检查点；
5. 在最终测试集计算指标并保存完整实验记录。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from sklearn import metrics
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataloader.context_collate import collate_fn_with_context
from dataloader.moocradar_strict import MOOCRadarStrict
from models.akt_context import AKTContext
from models.denoisekt_context import DenoiseKTContext
from models.dimkt_context import DIMKTContext
from models.dkt_context import DKTContext
from models.keenkt_context import KeenKTContext
from models.qikt_context import QIKTContext
from models.robustkt_context import RobustKTContext
from models.saint_context import SAINTContext
from models.sakt_context import SAKTContext
from models.simplekt_context import SimpleKTContext
from models.sparsekt_context import SparseKTContext
from models.tckt_context import TCKTContext
from scripts.common_pipeline_strict.io_utils import load_problem_records


def unpack_context_batch(batch):
    """校验 collate 输出契约，尽早发现数据加载器与训练代码不一致。"""
    if len(batch) != 11:
        raise ValueError(
            "Unexpected batch size from collate_fn_with_context: "
            f"expected 11 tensors (q, r, qshft, rshft, mask, eval_mask, ctx_main, ctx_tpl, ctx_llm, "
            f"ctx_llm_struct, ctx_llm_struct_features), got {len(batch)}. "
            "This usually means train_context.py and dataloader files are out of sync."
        )
    return batch


def reset_context_fusion_stats(model) -> None:
    """在一个训练或评估阶段开始前清空 Context 门控统计。"""
    fusion = getattr(model, "context_fusion", None)
    if fusion is not None and hasattr(fusion, "reset_usage_stats"):
        fusion.reset_usage_stats()


def get_context_fusion_stats(model) -> Dict[str, float]:
    """收集 Context 融合和 Context logit 分支的诊断指标。"""
    stats: Dict[str, float] = {}
    fusion = getattr(model, "context_fusion", None)
    if fusion is not None and hasattr(fusion, "get_usage_stats"):
        stats.update(dict(fusion.get_usage_stats()))
    context_logit_head = getattr(model, "context_logit_head", None)
    if context_logit_head is not None and hasattr(context_logit_head, "get_usage_stats"):
        stats.update(dict(context_logit_head.get_usage_stats()))
    else:
        context_logit_scale = getattr(context_logit_head, "scale", None)
        if context_logit_scale is not None:
            stats["context_logit_scale"] = float(context_logit_scale.detach().cpu().item())
    return stats


def select_context(
    context_type: str,
    ctx_main: torch.Tensor | None,
    ctx_tpl: torch.Tensor | None,
    ctx_llm: torch.Tensor | None,
    ctx_llm_struct: torch.Tensor | None,
    ctx_llm_struct_features: torch.Tensor | None,
) -> torch.Tensor | None:
    """根据实验配置选取 Context，并按固定顺序拼接多个 Context 分组。"""
    if context_type == "none":
        return None
    if context_type == "main":
        return ctx_main
    if context_type == "template":
        return ctx_tpl
    if context_type == "llm":
        if ctx_llm is None:
            raise ValueError("Requested llm context but llm text embedding tensor is missing")
        if ctx_llm_struct is None or ctx_llm_struct.shape[-1] == 0:
            raise ValueError("Requested llm context but llm structured embedding tensor is missing")
        if ctx_llm_struct_features is None or ctx_llm_struct_features.shape[-1] == 0:
            raise ValueError("Requested llm context but llm structured feature tensor is missing")
        return torch.cat([ctx_llm, ctx_llm_struct, ctx_llm_struct_features], dim=-1)
    if context_type == "all":
        if ctx_main is None or ctx_tpl is None:
            raise ValueError("Requested all context but main/template context tensors are missing")
        if ctx_llm is None:
            raise ValueError("Requested all context but llm text embedding tensor is missing")
        if ctx_llm_struct is None or ctx_llm_struct.shape[-1] == 0:
            raise ValueError("Requested all context but llm structured embedding tensor is missing")
        if ctx_llm_struct_features is None or ctx_llm_struct_features.shape[-1] == 0:
            raise ValueError("Requested all context but llm structured feature tensor is missing")
        return torch.cat([ctx_main, ctx_tpl, ctx_llm, ctx_llm_struct, ctx_llm_struct_features], dim=-1)
    raise ValueError(f"Unsupported context_type: {context_type}")


def build_problem_metadata(dataset: MOOCRadarStrict, difficult_levels: int = 10) -> Dict[str, Any]:
    """为需要题目-知识点关系的模型构造统一元数据及经验难度。"""
    problem_records = load_problem_records(Path(dataset.problem_json))
    pid2concepts = {str(record.problem_id): [str(c) for c in record.concepts if str(c)] for record in problem_records}
    concepts = sorted({concept for concept_list in pid2concepts.values() for concept in concept_list})
    if not concepts:
        concepts = ["__unknown_concept__"]
    concept2idx = {concept: idx for idx, concept in enumerate(concepts)}
    num_c = len(concepts)

    # DIMKT/QIKT 每题只接收一个从 1 开始编号的主知识点；TCKT 的 q_matrix
    # 则保留全部知识点关系。编号 0 留给 padding，缺失元数据时回退到首个知识点。
    q_to_concept = np.ones(int(dataset.num_q), dtype=np.int64)
    q_matrix = np.zeros((int(dataset.num_q), num_c), dtype=np.float32)
    for q_idx, raw_pid in enumerate(getattr(dataset, "q_list", [])):
        pid = str(raw_pid)
        concept_ids = [concept2idx[c] for c in pid2concepts.get(pid, []) if c in concept2idx]
        if not concept_ids:
            concept_ids = [0]
        q_to_concept[q_idx] = int(concept_ids[0]) + 1
        q_matrix[q_idx, concept_ids] = 1.0

    difficult_levels = max(1, int(difficult_levels))
    q_count = np.zeros(int(dataset.num_q), dtype=np.float64)
    q_correct = np.zeros(int(dataset.num_q), dtype=np.float64)
    if hasattr(dataset, "q_seqs") and hasattr(dataset, "r_seqs"):
        for q_seq, r_seq in zip(dataset.q_seqs, dataset.r_seqs):
            q_arr = np.asarray(q_seq, dtype=np.int64)
            r_arr = np.asarray(r_seq, dtype=np.int64)
            valid = (q_arr >= 0) & (q_arr < int(dataset.num_q)) & ((r_arr == 0) | (r_arr == 1))
            if not np.any(valid):
                continue
            np.add.at(q_count, q_arr[valid], 1.0)
            np.add.at(q_correct, q_arr[valid], r_arr[valid].astype(np.float64))

    global_rate = float(q_correct.sum() / q_count.sum()) if q_count.sum() > 0 else 0.5
    q_rate = np.divide(q_correct, q_count, out=np.full_like(q_correct, global_rate), where=q_count > 0)
    # 难度由经验错误率分箱得到；未出现题目继承全局正确率，+1 为 padding 保留 0。
    q_difficulty = np.floor((1.0 - q_rate) * difficult_levels).clip(0, difficult_levels - 1).astype(np.int64) + 1

    # 知识点难度按 DIMKT 所需的主知识点映射汇总；多知识点关系仍保存在 q_matrix。
    c_count = np.zeros(num_c + 1, dtype=np.float64)
    c_correct = np.zeros(num_c + 1, dtype=np.float64)
    for q_idx in range(int(dataset.num_q)):
        c_idx = int(q_to_concept[q_idx])
        c_count[c_idx] += q_count[q_idx]
        c_correct[c_idx] += q_correct[q_idx]
    c_rate = np.divide(c_correct, c_count, out=np.full_like(c_correct, global_rate), where=c_count > 0)
    concept_difficulty = np.floor((1.0 - c_rate) * difficult_levels).clip(0, difficult_levels - 1).astype(np.int64) + 1
    concept_difficulty[0] = 0

    return {
        "num_c": int(num_c),
        "q_to_concept": q_to_concept,
        "q_matrix": q_matrix,
        "q_difficulty": q_difficulty,
        "concept_difficulty": concept_difficulty,
        "difficulty_global_rate": global_rate,
    }


def build_model(
    model_name: str,
    dataset: MOOCRadarStrict,
    model_config: Dict[str, object],
    fusion_type: str,
    context_type: str,
    *,
    ctx_encoder_dim: int = 256,
    ctx_logit_hidden_dim: int = 128,
    ctx_logit_mode: str = "scaled",
    ctx_logit_init: float = -3.0,
    gate_bias_init: float = -2.0,
):
    """统一创建基础知识追踪模型，并配置其 Context 输入维度和分组方式。"""
    config = dict(model_config)
    ctx_dim = dataset.context_dim
    ctx_group_dims = None
    if context_type == "none":
        ctx_dim = max(1, int(ctx_dim or 0))
    elif context_type == "llm":
        llm_struct_dim = int(getattr(dataset, "llm_struct_dim", 0))
        llm_struct_feature_dim = int(getattr(dataset, "llm_struct_feature_dim", 0))
        ctx_dim = int(
            dataset.context_dim
            + llm_struct_dim
            + llm_struct_feature_dim
        )
        ctx_group_dims = (int(dataset.context_dim), llm_struct_dim, llm_struct_feature_dim)
    elif context_type == "all":
        llm_struct_dim = int(getattr(dataset, "llm_struct_dim", 0))
        llm_struct_feature_dim = int(getattr(dataset, "llm_struct_feature_dim", 0))
        ctx_dim = int(
            dataset.context_dim * 3
            + llm_struct_dim
            + llm_struct_feature_dim
        )
        ctx_group_dims = (
            int(dataset.context_dim),
            int(dataset.context_dim),
            int(dataset.context_dim),
            llm_struct_dim,
            llm_struct_feature_dim,
        )
    config["ctx_encoder_dim"] = int(ctx_encoder_dim)
    config["ctx_group_dims"] = ctx_group_dims
    config["ctx_logit_hidden_dim"] = int(ctx_logit_hidden_dim)
    config["ctx_logit_mode"] = str(ctx_logit_mode)
    config["ctx_logit_init"] = float(ctx_logit_init)
    config["gate_bias_init"] = float(gate_bias_init)
    if model_name == "dkt":
        return DKTContext(dataset.num_q, ctx_dim=ctx_dim, fusion_type=fusion_type, **config)
    if model_name == "sakt":
        return SAKTContext(dataset.num_q, ctx_dim=ctx_dim, fusion_type=fusion_type, **config)
    if model_name == "saint":
        return SAINTContext(dataset.num_q, ctx_dim=ctx_dim, fusion_type=fusion_type, **config)
    if model_name == "akt":
        return AKTContext(dataset.num_q, ctx_dim=ctx_dim, fusion_type=fusion_type, **config)
    if model_name in {"dimkt", "qikt", "tckt", "simplekt", "sparsekt", "robustkt", "denoisekt", "keenkt"}:
        metadata = build_problem_metadata(dataset, int(config.get("difficult_levels", 10)))
        if model_name == "dimkt":
            return DIMKTContext(
                dataset.num_q,
                metadata["num_c"],
                q_to_concept=metadata["q_to_concept"],
                q_difficulty=metadata["q_difficulty"],
                concept_difficulty=metadata["concept_difficulty"],
                ctx_dim=ctx_dim,
                fusion_type=fusion_type,
                **config,
            )
        if model_name == "qikt":
            return QIKTContext(
                dataset.num_q,
                metadata["num_c"],
                q_to_concept=metadata["q_to_concept"],
                ctx_dim=ctx_dim,
                fusion_type=fusion_type,
                **config,
            )
        if model_name == "simplekt":
            return SimpleKTContext(
                dataset.num_q,
                metadata["num_c"],
                q_to_concept=metadata["q_to_concept"],
                ctx_dim=ctx_dim,
                fusion_type=fusion_type,
                **config,
            )
        if model_name == "sparsekt":
            return SparseKTContext(
                dataset.num_q,
                metadata["num_c"],
                q_to_concept=metadata["q_to_concept"],
                ctx_dim=ctx_dim,
                fusion_type=fusion_type,
                **config,
            )
        if model_name == "robustkt":
            return RobustKTContext(
                dataset.num_q,
                metadata["num_c"],
                q_to_concept=metadata["q_to_concept"],
                ctx_dim=ctx_dim,
                fusion_type=fusion_type,
                **config,
            )
        if model_name == "denoisekt":
            return DenoiseKTContext(
                dataset.num_q,
                metadata["num_c"],
                q_matrix=metadata["q_matrix"],
                ctx_dim=ctx_dim,
                fusion_type=fusion_type,
                **config,
            )
        if model_name == "keenkt":
            return KeenKTContext(
                dataset.num_q,
                metadata["num_c"],
                q_to_concept=metadata["q_to_concept"],
                ctx_dim=ctx_dim,
                fusion_type=fusion_type,
                **config,
            )
        return TCKTContext(
            dataset.num_q,
            q_matrix=metadata["q_matrix"],
            ctx_dim=ctx_dim,
            fusion_type=fusion_type,
            **config,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def load_model_state(
    model: torch.nn.Module,
    ckpt_path: Path,
    device: str,
    *,
    allow_unused_context_mismatch: bool = False,
) -> None:
    """加载检查点；无 Context 实验可忽略未使用 Context 分支的形状差异。"""
    state = torch.load(ckpt_path, map_location=device)
    if not allow_unused_context_mismatch:
        model.load_state_dict(state)
        return

    model_state = model.state_dict()
    filtered_state = {}
    dropped_keys = []
    for key, value in state.items():
        expected = model_state.get(key)
        if expected is None:
            filtered_state[key] = value
            continue
        if tuple(expected.shape) == tuple(value.shape):
            filtered_state[key] = value
            continue
        if key.startswith(("context_fusion.", "context_logit_head.")):
            dropped_keys.append(key)
            continue
        filtered_state[key] = value

    if dropped_keys:
        print(
            f"[train_context] skipped {len(dropped_keys)} unused context checkpoint tensors "
            f"for context_type=none",
            flush=True,
        )
    model.load_state_dict(filtered_state, strict=False)


def compute_eval_metrics(preds_np: np.ndarray, targets_np: np.ndarray) -> Dict[str, float]:
    """根据预测概率和二分类标签计算完整评估指标。"""
    preds_np = np.asarray(preds_np, dtype=np.float64)
    targets_np = np.asarray(targets_np, dtype=np.float64)
    preds_np = np.clip(preds_np, 1e-7, 1.0 - 1e-7)
    binary_preds = (preds_np >= 0.5).astype(np.int64)
    targets_int = targets_np.astype(np.int64)

    metrics_out: Dict[str, float] = {}
    if len(np.unique(targets_int)) >= 2:
        metrics_out["auc"] = float(metrics.roc_auc_score(y_true=targets_int, y_score=preds_np))
        metrics_out["pr_auc"] = float(metrics.average_precision_score(y_true=targets_int, y_score=preds_np))
    else:
        metrics_out["auc"] = float("nan")
        metrics_out["pr_auc"] = float("nan")

    metrics_out["acc"] = float(metrics.accuracy_score(targets_int, binary_preds))
    metrics_out["precision"] = float(metrics.precision_score(targets_int, binary_preds, zero_division=0))
    metrics_out["recall"] = float(metrics.recall_score(targets_int, binary_preds, zero_division=0))
    metrics_out["f1"] = float(metrics.f1_score(targets_int, binary_preds, zero_division=0))
    metrics_out["bce"] = float(metrics.log_loss(targets_int, preds_np, labels=[0, 1]))
    metrics_out["rmse"] = float(math.sqrt(np.mean((preds_np - targets_np) ** 2)))
    metrics_out["sample_count"] = int(targets_np.shape[0])
    metrics_out["positive_rate"] = float(np.mean(targets_np))
    return metrics_out


def split_dataset(dataset, train_ratio: float, seed: int, split_dir: Path) -> Tuple[Subset, Subset, Dict[str, int]]:
    """按学生而非按序列随机切分，避免同一学生同时出现在两侧。"""
    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / "train_indices.pkl"
    test_path = split_dir / "test_indices.pkl"
    train_users_path = split_dir / "train_users.pkl"
    test_users_path = split_dir / "test_users.pkl"

    if train_path.exists() and test_path.exists() and train_users_path.exists() and test_users_path.exists():
        train_indices = torch.load(train_path)
        test_indices = torch.load(test_path)
        train_users = torch.load(train_users_path)
        test_users = torch.load(test_users_path)
    else:
        rng = np.random.default_rng(seed)
        unique_users = np.asarray(sorted(set(dataset.sample_user_ids)))
        permuted_users = unique_users[rng.permutation(len(unique_users))].tolist()
        train_user_size = int(len(permuted_users) * train_ratio)
        train_users = set(permuted_users[:train_user_size])
        test_users = set(permuted_users[train_user_size:])
        train_indices = [idx for idx, user_id in enumerate(dataset.sample_user_ids) if user_id in train_users]
        test_indices = [idx for idx, user_id in enumerate(dataset.sample_user_ids) if user_id in test_users]
        torch.save(train_indices, train_path)
        torch.save(test_indices, test_path)
        torch.save(sorted(train_users), train_users_path)
        torch.save(sorted(test_users), test_users_path)

    split_stats = {
        "train_user_count": len(train_users),
        "test_user_count": len(test_users),
    }
    return Subset(dataset, train_indices), Subset(dataset, test_indices), split_stats


def limit_dataset(dataset, limit: int, seed: int):
    """为快速评估可重复地抽取固定数量样本；limit<=0 表示不限制。"""
    limit = int(limit or 0)
    if limit <= 0 or limit >= len(dataset):
        return dataset
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(dataset), size=limit, replace=False)
    return Subset(dataset, np.sort(indices).astype(np.int64).tolist())


def resolve_num_workers(args) -> int:
    """选择 DataLoader worker 数；大 Context 和懒加载新知识点评估默认单进程。"""
    if args.num_workers is not None:
        return int(args.num_workers)
    if args.split_mode == "new_concept":
        return 0
    if args.context_type in {"llm", "all"}:
        return 0
    return 4


def build_optimizer_with_optional_context_lr(
    model: torch.nn.Module,
    optimizer_name: str,
    lr: float,
    *,
    context_lr_scale: float = 1.0,
    use_context: bool = True,
    weight_decay: float = 0.0,
):
    """创建优化器，并可为 Context 分支设置独立学习率倍率。"""
    optimizer_name = str(optimizer_name).lower()
    lr = float(lr)
    weight_decay = max(0.0, float(weight_decay))
    context_lr_scale = float(context_lr_scale)
    if use_context and context_lr_scale > 0.0 and abs(context_lr_scale - 1.0) > 1e-12:
        context_prefixes = ("context_fusion.", "context_logit_head.")
        base_params = []
        context_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(context_prefixes):
                context_params.append(param)
            else:
                base_params.append(param)
        param_groups = [
            {"params": base_params, "lr": lr},
            {"params": context_params, "lr": lr * context_lr_scale},
        ]
    else:
        param_groups = model.parameters()

    if optimizer_name == "sgd":
        return SGD(param_groups, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return Adam(param_groups, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def evaluate(model, loader, device: str, model_name: str, context_type: str, amp_enabled: bool = False) -> Dict[str, float]:
    """在 eval_mask 指定的位置评估模型，并汇总预测与 Context 使用统计。"""
    model.eval()
    preds = []
    targets = []
    losses = []
    reset_context_fusion_stats(model)
    try:
        batch_count = len(loader)
    except TypeError:
        batch_count = "unknown"
    print(f"[train_context] eval start model={model_name} context={context_type} batches={batch_count}", flush=True)
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=True):
            q, r, qshft, rshft, mask, eval_mask, ctx_main, ctx_tpl, ctx_llm, ctx_llm_struct, ctx_llm_struct_features = unpack_context_batch(batch)
            q = q.to(device)
            r = r.to(device)
            qshft = qshft.to(device)
            rshft = rshft.to(device)
            mask = mask.to(device) & eval_mask.to(device)
            ctx = select_context(context_type, ctx_main, ctx_tpl, ctx_llm, ctx_llm_struct, ctx_llm_struct_features)
            if ctx is not None:
                ctx = ctx.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                p = model(q.long(), r.long(), qshft.long(), ctx)
                p = torch.masked_select(p, mask)
                t = torch.masked_select(rshft.float(), mask)
                if p.numel() == 0:
                    continue
            loss = binary_cross_entropy(p.float(), t.float())
            losses.append(float(loss.detach().cpu().item()))
            preds.append(p.detach().cpu())
            targets.append(t.detach().cpu())

    if not preds:
        return {
            "auc": 0.0,
            "pr_auc": 0.0,
            "acc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "bce": 0.0,
            "rmse": 0.0,
            "sample_count": 0,
            "positive_rate": 0.0,
            "loss_mean": 0.0,
        }

    preds_np = torch.cat(preds).numpy()
    targets_np = torch.cat(targets).numpy()
    metrics_out = compute_eval_metrics(preds_np, targets_np)
    metrics_out["loss_mean"] = float(np.mean(losses)) if losses else 0.0
    metrics_out["context_fusion"] = get_context_fusion_stats(model)
    print(
        f"[train_context] eval done samples={metrics_out.get('sample_count')} "
        f"auc={metrics_out.get('auc')} acc={metrics_out.get('acc')}",
        flush=True,
    )
    return metrics_out


def train(
    model,
    train_loader,
    valid_loader,
    optimizer,
    num_epochs: int,
    device: str,
    model_name: str,
    context_type: str,
    ckpt_dir: Path,
    patience: int = 0,
    eval_interval: int = 1,
    amp_enabled: bool = False,
    context_warmup_epochs: int = 0,
    max_grad_norm: float = 0.0,
    test_selection_interval: int = 0,
    test_selection_start_epoch: int = 1,
):
    """执行训练循环，负责 warmup、验证、早停和候选检查点保存。"""
    history = []
    no_validation = valid_loader is None
    best_auc = float("nan") if no_validation else -1.0
    best_epoch = -1
    best_metrics: Dict[str, float] | None = None
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "model.ckpt"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    eval_interval = max(1, int(eval_interval))
    patience = max(0, int(patience))
    context_warmup_epochs = max(0, int(context_warmup_epochs))
    max_grad_norm = max(0.0, float(max_grad_norm))
    test_selection_interval = max(0, int(test_selection_interval))
    test_selection_start_epoch = max(1, int(test_selection_start_epoch))
    test_candidate_paths: list[dict[str, Any]] = []
    if test_selection_interval > 0:
        for stale_path in ckpt_dir.glob("test_candidate_epoch_*.ckpt"):
            stale_path.unlink()

    for epoch in range(1, num_epochs + 1):
        model.train()
        batch_losses = []
        reset_context_fusion_stats(model)
        # warmup 阶段只训练基础模型路径：传入 None 后，Context 融合与 logit 分支
        # 都不会参与前向计算，也不会收到梯度。
        warmup_context_disabled = context_type != "none" and epoch <= context_warmup_epochs
        effective_context_type = "none" if warmup_context_disabled else context_type
        train_bar = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)
        for batch in train_bar:
            q, r, qshft, rshft, mask, eval_mask, ctx_main, ctx_tpl, ctx_llm, ctx_llm_struct, ctx_llm_struct_features = unpack_context_batch(batch)
            q = q.to(device)
            r = r.to(device)
            qshft = qshft.to(device)
            rshft = rshft.to(device)
            mask = mask.to(device) & eval_mask.to(device)
            ctx = select_context(effective_context_type, ctx_main, ctx_tpl, ctx_llm, ctx_llm_struct, ctx_llm_struct_features)
            if ctx is not None:
                ctx = ctx.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                p = model(q.long(), r.long(), qshft.long(), ctx)
                p = torch.masked_select(p, mask)
                t = torch.masked_select(rshft.float(), mask)
                if p.numel() == 0:
                    continue
            loss = binary_cross_entropy(p.float(), t.float())
            get_auxiliary_loss = getattr(model, "get_training_auxiliary_loss", None)
            if callable(get_auxiliary_loss):
                auxiliary_loss = get_auxiliary_loss()
                if auxiliary_loss is not None:
                    loss = loss + auxiliary_loss.float()
            scaler.scale(loss).backward()
            if max_grad_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            loss_value = float(loss.detach().cpu().item())
            batch_losses.append(loss_value)
            train_bar.set_postfix(loss=f"{loss_value:.4f}")

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        train_context_fusion = get_context_fusion_stats(model)
        should_save_test_candidate = (
            test_selection_interval > 0
            and epoch >= test_selection_start_epoch
            and (epoch % test_selection_interval == 0 or epoch == num_epochs)
        )
        if should_save_test_candidate:
            candidate_path = ckpt_dir / f"test_candidate_epoch_{epoch:04d}.ckpt"
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), candidate_path)
            test_candidate_paths.append({"epoch": int(epoch), "path": str(candidate_path)})
            print(f"[train_context] saved test-selection candidate epoch={epoch} path={candidate_path}", flush=True)
        if no_validation:
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_path)
            best_epoch = epoch
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_context_fusion": train_context_fusion,
                    "context_warmup_active": warmup_context_disabled,
                    "eval_metrics": None,
                    "skipped_eval": True,
                    "no_validation": True,
                }
            )
            train_logit_summary = ""
            train_gate_summary = ""
            if context_type != "none":
                if "ctx_logit_scale" in train_context_fusion:
                    train_logit_summary = (
                        ", Train CtxLogitMode: {mode}, Train CtxLogitScale: {scale:.4f}"
                    ).format(
                        mode=str(train_context_fusion.get("ctx_logit_mode", "unknown")),
                        scale=float(train_context_fusion["ctx_logit_scale"]),
                    )
                if train_context_fusion.get("fusion_mode") in {"gate", "residual_gate"} and int(train_context_fusion.get("usage_steps", 0)) > 0:
                    train_gate_summary = (
                        ", Train Gate Mean: {gate_mean:.4f}, Train Ctx Weight: {ctx_weight_mean:.4f}, "
                        "Train Gate<0.1: {gate_low:.4f}, Train Gate>0.9: {gate_high:.4f}"
                    ).format(
                        gate_mean=float(train_context_fusion["gate_mean"]),
                        ctx_weight_mean=float(train_context_fusion["ctx_weight_mean"]),
                        gate_low=float(train_context_fusion["gate_lt_0_1_frac"]),
                        gate_high=float(train_context_fusion["gate_gt_0_9_frac"]),
                    )
            print(
                f"Epoch: {epoch}, Train Loss: {train_loss:.6f}, Valid: disabled"
                f"{', ContextWarmup: active' if warmup_context_disabled else ''}"
                f"{train_gate_summary}{train_logit_summary}"
            )
            continue

        should_eval = (epoch % eval_interval == 0) or (epoch == num_epochs)
        if not should_eval:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_context_fusion": train_context_fusion,
                    "context_warmup_active": warmup_context_disabled,
                    "eval_metrics": None,
                    "skipped_eval": True,
                }
            )
            print(
                f"Epoch: {epoch}, Train Loss: {train_loss:.6f}, Valid: skipped"
                f"{', ContextWarmup: active' if warmup_context_disabled else ''}"
            )
            continue

        # 验证始终使用正式推理配置；因此 warmup 期间测到的是尚未开始训练的 Context 分支。
        eval_metrics = evaluate(model, valid_loader, device, model_name, context_type, amp_enabled=amp_enabled)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_context_fusion": train_context_fusion,
                "context_warmup_active": warmup_context_disabled,
                "eval_metrics": eval_metrics,
                "skipped_eval": False,
            }
        )
        train_gate_summary = ""
        eval_gate_summary = ""
        train_logit_summary = ""
        eval_logit_summary = ""
        if context_type != "none":
            if "ctx_logit_scale" in train_context_fusion:
                train_logit_summary = (
                    ", Train CtxLogitMode: {mode}, Train CtxLogitScale: {scale:.4f}"
                ).format(
                    mode=str(train_context_fusion.get("ctx_logit_mode", "unknown")),
                    scale=float(train_context_fusion["ctx_logit_scale"]),
                )
            if train_context_fusion.get("fusion_mode") in {"gate", "residual_gate"} and int(train_context_fusion.get("usage_steps", 0)) > 0:
                train_gate_summary = (
                    ", Train Gate Mean: {gate_mean:.4f}, Train Ctx Weight: {ctx_weight_mean:.4f}, "
                    "Train Gate<0.1: {gate_low:.4f}, Train Gate>0.9: {gate_high:.4f}"
                ).format(
                    gate_mean=float(train_context_fusion["gate_mean"]),
                    ctx_weight_mean=float(train_context_fusion["ctx_weight_mean"]),
                    gate_low=float(train_context_fusion["gate_lt_0_1_frac"]),
                    gate_high=float(train_context_fusion["gate_gt_0_9_frac"]),
                )
            eval_context_fusion = eval_metrics.get("context_fusion") or {}
            if "ctx_logit_scale" in eval_context_fusion:
                eval_logit_summary = (
                    ", Eval CtxLogitMode: {mode}, Eval CtxLogitScale: {scale:.4f}"
                ).format(
                    mode=str(eval_context_fusion.get("ctx_logit_mode", "unknown")),
                    scale=float(eval_context_fusion["ctx_logit_scale"]),
                )
            if eval_context_fusion.get("fusion_mode") in {"gate", "residual_gate"} and int(eval_context_fusion.get("usage_steps", 0)) > 0:
                eval_gate_summary = (
                    ", Eval Gate Mean: {gate_mean:.4f}, Eval Ctx Weight: {ctx_weight_mean:.4f}, "
                    "Eval Gate<0.1: {gate_low:.4f}, Eval Gate>0.9: {gate_high:.4f}"
                ).format(
                    gate_mean=float(eval_context_fusion["gate_mean"]),
                    ctx_weight_mean=float(eval_context_fusion["ctx_weight_mean"]),
                    gate_low=float(eval_context_fusion["gate_lt_0_1_frac"]),
                    gate_high=float(eval_context_fusion["gate_gt_0_9_frac"]),
                )
        print(
            "Epoch: {epoch}, Valid AUC: {auc:.6f}, Valid ACC: {acc:.6f}, Valid F1: {f1:.6f}, "
            "BCE: {bce:.6f}, RMSE: {rmse:.6f}, Train Loss: {train_loss:.6f}, Valid Loss: {eval_loss:.6f}"
            "{train_gate_summary}{eval_gate_summary}{train_logit_summary}{eval_logit_summary}".format(
                epoch=epoch,
                auc=float(eval_metrics["auc"]) if not math.isnan(float(eval_metrics["auc"])) else float("nan"),
                acc=float(eval_metrics["acc"]),
                f1=float(eval_metrics["f1"]),
                bce=float(eval_metrics["bce"]),
                rmse=float(eval_metrics["rmse"]),
                train_loss=train_loss,
                eval_loss=float(eval_metrics["loss_mean"]),
                train_gate_summary=train_gate_summary,
                eval_gate_summary=(
                    f", ContextWarmup: active{eval_gate_summary}" if warmup_context_disabled else eval_gate_summary
                ),
                train_logit_summary=train_logit_summary,
                eval_logit_summary=eval_logit_summary,
            )
        )

        eval_auc = float(eval_metrics["auc"])
        if not math.isnan(eval_auc) and eval_auc > best_auc:
            best_auc = eval_auc
            best_epoch = epoch
            best_metrics = dict(eval_metrics)
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_path)
        if patience > 0 and best_epoch > 0 and epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}, best_valid_auc={best_auc:.6f}")
            break

    if best_metrics is None:
        best_metrics = {
            "auc": best_auc if best_auc >= 0 else float("nan"),
            "pr_auc": float("nan"),
            "acc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "bce": float("nan"),
            "rmse": float("nan"),
            "sample_count": 0,
            "positive_rate": float("nan"),
            "loss_mean": float("nan"),
            "selection": "last_epoch_no_validation" if no_validation else "none",
            "best_epoch": int(best_epoch),
        }

    return history, best_auc, best_path, best_metrics, test_candidate_paths


def main() -> None:
    """解析命令行参数并串联数据集、模型训练、最终测试和结果保存。"""
    parser = argparse.ArgumentParser()
    workspace = Path(__file__).resolve().parent
    parser.add_argument("--model_name", type=str, default="dkt", choices=["dkt", "sakt", "saint", "akt", "dimkt", "qikt", "tckt", "simplekt", "sparsekt", "robustkt", "denoisekt", "keenkt"])
    parser.add_argument("--context_type", type=str, default="llm", choices=["none", "main", "template", "llm", "all"])
    parser.add_argument(
        "--fusion_type",
        "--fusion_mode",
        dest="fusion_type",
        type=str,
        default="residual_gate",
        choices=["add", "concat", "gate", "residual_gate"],
    )
    parser.add_argument("--problem_json", type=str, default=str(workspace / "datalocal" / "problem.json"))
    parser.add_argument("--student_json", type=str, default=str(workspace / "datalocal" / "student-problem-fine.json"))
    parser.add_argument(
        "--context_embeddings_path",
        type=str,
        default=str(workspace / "out" / "strict_common_pipeline" / "cache" / "context_embeddings.pkl"),
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(workspace / "datasets" / "MOOCRadarStrict"),
    )
    parser.add_argument(
        "--ckpt_root",
        type=str,
        default=str(workspace / "ckpts_context"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_mode", type=str, default="user", choices=["user", "new_concept"])
    parser.add_argument("--test_concept_ratio", type=float, default=0.2)
    parser.add_argument("--valid_concept_ratio", type=float, default=0.0)
    parser.add_argument("--valid_ratio", type=float, default=0.0)
    parser.add_argument("--cache_dataset", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_ckpt", type=str, default=None)
    parser.add_argument("--cpu_threads", type=int, default=None)
    parser.add_argument("--context_storage_dtype", type=str, default="float32", choices=["float32", "float16"])
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--valid_limit", type=int, default=0)
    parser.add_argument("--test_limit", type=int, default=0)
    parser.add_argument("--ctx_encoder_dim", type=int, default=256)
    parser.add_argument("--ctx_logit_hidden_dim", type=int, default=128)
    parser.add_argument("--ctx_logit_mode", type=str, default="scaled", choices=["none", "raw", "scaled"])
    parser.add_argument("--ctx_logit_init", type=float, default=-3.0)
    parser.add_argument("--gate_bias_init", type=float, default=-2.0)
    parser.add_argument("--context_warmup_epochs", type=int, default=0)
    parser.add_argument("--context_lr_scale", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--select_by_test_auc", action="store_true")
    parser.add_argument("--test_eval_interval", type=int, default=5)
    parser.add_argument("--test_eval_start_epoch", type=int, default=1)
    args = parser.parse_args()

    if args.cpu_threads is not None:
        cpu_threads = max(1, int(args.cpu_threads))
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = str(cpu_threads)
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(cpu_threads)
        except RuntimeError:
            pass

    with (workspace / "models" / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    train_config = dict(config["train_config"])
    model_config = dict(config[args.model_name])

    if args.batch_size is not None:
        train_config["batch_size"] = int(args.batch_size)
    if args.num_epochs is not None:
        train_config["num_epochs"] = int(args.num_epochs)

    seq_len = int(train_config["seq_len"])
    # 所有数据集角色共享同一组构造参数，区别仅由后续传入的 split_role 决定。
    dataset_kwargs = {
        "seq_len": seq_len,
        "problem_json": args.problem_json,
        "student_json": args.student_json,
        "context_embeddings_path": args.context_embeddings_path,
        "dataset_dir": args.dataset_dir,
        "require_llm_context": (args.context_type in {"llm", "all"}),
        "require_llm_struct_context": (args.context_type in {"llm", "all"}),
        "require_llm_struct_feature_context": (args.context_type in {"llm", "all"}),
        "split_mode": args.split_mode,
        "seed": int(args.seed),
        "test_concept_ratio": float(args.test_concept_ratio),
        "valid_concept_ratio": float(args.valid_concept_ratio),
        "cache_preprocessed": (args.split_mode == "user" or args.cache_dataset),
        "context_storage_dtype": args.context_storage_dtype,
        "load_context_embeddings": (args.context_type != "none"),
    }

    # eval_only 路径跳过训练，直接加载指定检查点并在完整测试角色上评估。
    if args.eval_only:
        eval_role = "test" if args.split_mode == "new_concept" else "all"
        print(
            f"[train_context] building eval dataset split_role={eval_role} "
            f"context_type={args.context_type} load_context_embeddings={args.context_type != 'none'}",
            flush=True,
        )
        dataset = MOOCRadarStrict(**dataset_kwargs, split_role=eval_role)
        print(
            f"[train_context] eval dataset ready len={len(dataset)} "
            f"context_dim={dataset.context_dim} llm_struct_dim={getattr(dataset, 'llm_struct_dim', 0)}",
            flush=True,
        )
        if args.context_type in {"llm", "all"} and not dataset.has_llm_context:
            raise ValueError(f"Requested context_type={args.context_type} but context_embeddings.pkl does not contain llm_embeddings")
        if args.context_type in {"llm", "all"} and not getattr(dataset, "has_llm_struct_context", False):
            raise ValueError(f"Requested context_type={args.context_type} but context_embeddings.pkl does not contain llm_struct_embeddings")
        if args.context_type in {"llm", "all"} and not getattr(dataset, "has_llm_struct_feature_context", False):
            raise ValueError(f"Requested context_type={args.context_type} but context_embeddings.pkl does not contain llm_struct_features")

        eval_split_stats = getattr(dataset, "split_stats", {})
        device = "cuda" if torch.cuda.is_available() else "cpu"
        amp_enabled = bool(args.amp and device == "cuda")
        context_tensor_dtype = "float16" if amp_enabled and args.context_storage_dtype == "float16" else "float32"
        effective_num_workers = resolve_num_workers(args)
        effective_batch_size = int(train_config["batch_size"])
        effective_eval_batch_size = int(args.eval_batch_size) if args.eval_batch_size is not None else effective_batch_size
        full_eval_len = len(dataset)
        dataset = limit_dataset(dataset, int(args.test_limit), int(args.seed) + 29)
        print(f"[train_context] eval limit applied full_len={full_eval_len} eval_len={len(dataset)}", flush=True)
        if args.model_name in ("sakt", "saint"):
            model_config["n"] = seq_len
        if args.model_name in ("tckt", "simplekt", "sparsekt", "denoisekt", "keenkt"):
            model_config["max_seq_len"] = max(int(model_config.get("max_seq_len", 0)), seq_len)
        ckpt_dir = Path(args.ckpt_root).resolve() / args.split_mode / args.model_name / args.context_type / args.fusion_type
        ckpt_path = Path(args.eval_ckpt).resolve() if args.eval_ckpt else ckpt_dir / "model.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)

        model = build_model(
            args.model_name,
            dataset,
            model_config,
            args.fusion_type,
            args.context_type,
            ctx_encoder_dim=int(args.ctx_encoder_dim),
            ctx_logit_hidden_dim=int(args.ctx_logit_hidden_dim),
            ctx_logit_mode=args.ctx_logit_mode,
            ctx_logit_init=float(args.ctx_logit_init),
            gate_bias_init=float(args.gate_bias_init),
        ).to(device)
        load_model_state(
            model,
            ckpt_path,
            device,
            allow_unused_context_mismatch=(args.context_type == "none"),
        )
        eval_loader = DataLoader(
            dataset,
            batch_size=effective_eval_batch_size,
            shuffle=False,
            collate_fn=partial(
                collate_fn_with_context,
                context_type=args.context_type,
                context_tensor_dtype=context_tensor_dtype,
            ),
            num_workers=max(0, effective_num_workers),
            pin_memory=(device == "cuda"),
            persistent_workers=(effective_num_workers > 0),
        )
        print(
            f"[train_context] eval loader ready batch_size={effective_eval_batch_size} "
            f"num_workers={effective_num_workers}",
            flush=True,
        )
        test_metrics = evaluate(model, eval_loader, device, args.model_name, args.context_type, amp_enabled=amp_enabled)
        metrics_path = ckpt_dir / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": args.model_name,
                    "context_type": args.context_type,
                    "split_mode": args.split_mode,
                    "eval_only": True,
                    "device": device,
                    "cpu_threads": args.cpu_threads,
                    "valid_concept_ratio": float(args.valid_concept_ratio),
                    "context_storage_dtype": args.context_storage_dtype,
                    "eval_batch_size": int(effective_eval_batch_size),
                    "context_tensor_dtype": context_tensor_dtype,
                    "ctx_encoder_dim": int(args.ctx_encoder_dim),
                    "ctx_logit_hidden_dim": int(args.ctx_logit_hidden_dim),
                    "ctx_logit_mode": args.ctx_logit_mode,
                    "ctx_logit_init": float(args.ctx_logit_init),
                    "gate_bias_init": float(args.gate_bias_init),
                    "context_warmup_epochs": int(args.context_warmup_epochs),
                    "context_lr_scale": float(args.context_lr_scale),
                    "max_grad_norm": float(args.max_grad_norm),
                    "amp": amp_enabled,
                    "full_test_len": int(full_eval_len),
                    "test_len": len(dataset),
                    "test_limit": int(args.test_limit),
                    "split_stats": {
                        "split_mode": args.split_mode,
                        "test_dataset_stats": eval_split_stats,
                    },
                    "context_dim": dataset.context_dim,
                    "has_llm_context": dataset.has_llm_context,
                    "has_llm_struct_context": getattr(dataset, "has_llm_struct_context", False),
                    "llm_struct_dim": getattr(dataset, "llm_struct_dim", 0),
                    "has_llm_struct_feature_context": getattr(dataset, "has_llm_struct_feature_context", False),
                    "llm_struct_feature_dim": getattr(dataset, "llm_struct_feature_dim", 0),
                    "fusion_type": args.fusion_type,
                    "test_metrics": test_metrics,
                    "best_ckpt": str(ckpt_path),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print("[OK] eval-only finished")
        print("[MODEL]", args.model_name)
        print("[CONTEXT]", args.context_type)
        print("[FUSION]", args.fusion_type)
        print("[CTX_LOGIT_MODE]", args.ctx_logit_mode)
        print("[SPLIT_MODE]", args.split_mode)
        print("[TEST_AUC]", test_metrics.get("auc"))
        print("[TEST_ACC]", test_metrics.get("acc"))
        print("[METRICS]", metrics_path)
        return

    # 新知识点实验可使用独立验证知识点；若未配置任何验证划分，则按最后一轮模型测试。
    use_concept_valid = args.split_mode == "new_concept" and float(args.valid_concept_ratio) > 0.0
    use_no_validation = (
        args.split_mode == "new_concept"
        and float(args.valid_concept_ratio) <= 0.0
        and float(args.valid_ratio) <= 0.0
    )
    concept_valid_dataset = None
    print(
        f"[train_context] building train dataset context_type={args.context_type} "
        f"load_context_embeddings={args.context_type != 'none'} split_mode={args.split_mode}",
        flush=True,
    )
    # 先构造与实验协议匹配的基础数据集，再在下方决定是否按学生进一步切分。
    if use_concept_valid:
        dataset = MOOCRadarStrict(**dataset_kwargs, split_role="train")
        print(f"[train_context] train dataset ready len={len(dataset)}", flush=True)
        print("[train_context] building validation dataset split_role=valid", flush=True)
        concept_valid_dataset = MOOCRadarStrict(**dataset_kwargs, split_role="valid")
        print(f"[train_context] validation dataset ready len={len(concept_valid_dataset)}", flush=True)
        if len(concept_valid_dataset) == 0:
            raise ValueError(
                "valid_concept_ratio produced an empty cold-start validation set; "
                "increase --valid_concept_ratio or use --valid_concept_ratio 0.0"
            )
        final_test_dataset = None
    elif use_no_validation:
        dataset = MOOCRadarStrict(**dataset_kwargs, split_role="train")
        print(f"[train_context] train dataset ready len={len(dataset)}", flush=True)
        final_test_dataset = None
    elif args.split_mode == "new_concept":
        dataset = MOOCRadarStrict(**dataset_kwargs, split_role="train_valid")
        print(f"[train_context] train_valid dataset ready len={len(dataset)}", flush=True)
        final_test_dataset = None
    else:
        dataset = MOOCRadarStrict(**dataset_kwargs, split_role="all")
        print(f"[train_context] all dataset ready len={len(dataset)}", flush=True)
        final_test_dataset = None
    if args.context_type in {"llm", "all"} and not dataset.has_llm_context:
        raise ValueError(f"Requested context_type={args.context_type} but context_embeddings.pkl does not contain llm_embeddings")
    if args.context_type in {"llm", "all"} and not getattr(dataset, "has_llm_struct_context", False):
        raise ValueError(f"Requested context_type={args.context_type} but context_embeddings.pkl does not contain llm_struct_embeddings")
    if args.context_type in {"llm", "all"} and not getattr(dataset, "has_llm_struct_feature_context", False):
        raise ValueError(f"Requested context_type={args.context_type} but context_embeddings.pkl does not contain llm_struct_features")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = bool(args.amp and device == "cuda")
    context_tensor_dtype = "float16" if amp_enabled and args.context_storage_dtype == "float16" else "float32"
    effective_num_workers = resolve_num_workers(args)
    effective_batch_size = int(train_config["batch_size"])
    effective_eval_batch_size = int(args.eval_batch_size) if args.eval_batch_size is not None else effective_batch_size
    print(
        f"[train_context] model={args.model_name} context_type={args.context_type} "
        f"split_mode={args.split_mode} "
        f"test_concept_ratio={float(args.test_concept_ratio)} valid_ratio={float(args.valid_ratio)} "
        f"valid_concept_ratio={float(args.valid_concept_ratio)} validation_disabled={use_no_validation} "
        f"batch_size={effective_batch_size} eval_batch_size={effective_eval_batch_size} num_workers={effective_num_workers} "
        f"cpu_threads={args.cpu_threads} context_storage_dtype={args.context_storage_dtype} "
        f"context_tensor_dtype={context_tensor_dtype} valid_limit={int(args.valid_limit)} test_limit={int(args.test_limit)} "
        f"ctx_encoder_dim={int(args.ctx_encoder_dim)} ctx_logit_hidden_dim={int(args.ctx_logit_hidden_dim)} "
        f"ctx_logit_mode={args.ctx_logit_mode} ctx_logit_init={float(args.ctx_logit_init)} "
        f"gate_bias_init={float(args.gate_bias_init)} "
        f"context_warmup_epochs={int(args.context_warmup_epochs)} "
        f"context_lr_scale={float(args.context_lr_scale)} max_grad_norm={float(args.max_grad_norm)} "
        f"select_by_test_auc={bool(args.select_by_test_auc)} "
        f"test_eval_interval={int(args.test_eval_interval)} test_eval_start_epoch={int(args.test_eval_start_epoch)} "
        f"patience={args.patience} eval_interval={args.eval_interval} amp={amp_enabled} "
        f"context_dim={dataset.context_dim} llm_struct_dim={getattr(dataset, 'llm_struct_dim', 0)} "
        f"llm_struct_feature_dim={getattr(dataset, 'llm_struct_feature_dim', 0)}",
        flush=True,
    )
    # 三种验证策略：独立验证知识点、禁用验证、按学生划分验证集。
    if use_concept_valid:
        train_dataset = dataset
        valid_dataset = concept_valid_dataset
        split_stats = {
            "split_mode": args.split_mode,
            "validation_disabled": False,
            "valid_ratio": None,
            "valid_concept_ratio": float(args.valid_concept_ratio),
            "train_dataset_stats": getattr(dataset, "split_stats", {}),
            "valid_dataset_stats": getattr(valid_dataset, "split_stats", {}),
        }
    elif use_no_validation:
        train_dataset = dataset
        valid_dataset = None
        split_stats = {
            "split_mode": args.split_mode,
            "validation_disabled": True,
            "valid_ratio": 0.0,
            "valid_concept_ratio": 0.0,
            "train_dataset_stats": getattr(dataset, "split_stats", {}),
        }
    else:
        valid_tag = str(float(args.valid_ratio)).replace(".", "p")
        split_dir = Path(args.dataset_dir).resolve() / f"splits_{args.split_mode}_seq{seq_len}_seed{int(args.seed)}_valid{valid_tag}"
        if args.split_mode == "new_concept":
            train_ratio = 1.0 - float(args.valid_ratio)
        else:
            train_ratio = float(train_config["train_ratio"])
        train_dataset, valid_dataset, split_stats = split_dataset(dataset, train_ratio, int(args.seed), split_dir)
        split_stats.update(
            {
                "split_mode": args.split_mode,
                "validation_disabled": False,
                "valid_ratio": float(args.valid_ratio),
                "valid_concept_ratio": float(args.valid_concept_ratio),
                "train_valid_dataset_stats": getattr(dataset, "split_stats", {}),
            }
        )
    full_valid_len = len(valid_dataset) if valid_dataset is not None else 0
    if valid_dataset is not None:
        valid_dataset = limit_dataset(valid_dataset, int(args.valid_limit), int(args.seed) + 17)
    split_stats["full_valid_len"] = int(full_valid_len)
    split_stats["valid_eval_len"] = int(len(valid_dataset)) if valid_dataset is not None else 0
    split_stats["valid_limit"] = int(args.valid_limit)

    # collate_fn 在这里完成序列移位、padding、mask 以及 Context 与预测目标的对齐。
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        collate_fn=partial(
            collate_fn_with_context,
            context_type=args.context_type,
            context_tensor_dtype=context_tensor_dtype,
        ),
        num_workers=max(0, effective_num_workers),
        pin_memory=(device == "cuda"),
        persistent_workers=(effective_num_workers > 0),
    )
    print(
        f"[train_context] train loader ready train_len={len(train_dataset)} "
        f"batch_size={effective_batch_size} num_workers={effective_num_workers}",
        flush=True,
    )
    valid_loader = None
    if valid_dataset is not None:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=effective_eval_batch_size,
            shuffle=False,
            collate_fn=partial(
                collate_fn_with_context,
                context_type=args.context_type,
                context_tensor_dtype=context_tensor_dtype,
            ),
            num_workers=max(0, effective_num_workers),
            pin_memory=(device == "cuda"),
            persistent_workers=(effective_num_workers > 0),
        )
        print(
            f"[train_context] valid loader ready valid_len={len(valid_dataset)} "
            f"batch_size={effective_eval_batch_size} num_workers={effective_num_workers}",
            flush=True,
        )
    if args.model_name in ("sakt", "saint"):
        model_config["n"] = seq_len
    if args.model_name in ("tckt", "simplekt", "sparsekt", "denoisekt", "keenkt"):
        model_config["max_seq_len"] = max(int(model_config.get("max_seq_len", 0)), seq_len)

    # 基础知识追踪模型和 Context 模块共同创建，随后可对 Context 参数使用独立学习率。
    model = build_model(
        args.model_name,
        dataset,
        model_config,
        args.fusion_type,
        args.context_type,
        ctx_encoder_dim=int(args.ctx_encoder_dim),
        ctx_logit_hidden_dim=int(args.ctx_logit_hidden_dim),
        ctx_logit_mode=args.ctx_logit_mode,
        ctx_logit_init=float(args.ctx_logit_init),
        gate_bias_init=float(args.gate_bias_init),
    ).to(device)

    optimizer_name = str(train_config["optimizer"]).lower()
    lr = float(model_config.get("learning_rate", train_config["learning_rate"]))
    weight_decay = float(model_config.get("weight_decay", 0.0))
    optimizer = build_optimizer_with_optional_context_lr(
        model,
        optimizer_name,
        lr,
        context_lr_scale=float(args.context_lr_scale),
        use_context=(args.context_type != "none"),
        weight_decay=weight_decay,
    )
    if args.context_type != "none" and abs(float(args.context_lr_scale) - 1.0) > 1e-12:
        print(
            f"[train_context] optimizer context_lr_scale={float(args.context_lr_scale)} "
            f"base_lr={lr} context_lr={lr * float(args.context_lr_scale)} "
            f"weight_decay={weight_decay}",
            flush=True,
        )

    ckpt_dir = Path(args.ckpt_root).resolve() / args.split_mode / args.model_name / args.context_type / args.fusion_type
    print(
        f"[train_context] training start epochs={int(train_config['num_epochs'])} "
        f"learning_rate={lr} weight_decay={weight_decay} ckpt_dir={ckpt_dir}",
        flush=True,
    )
    # 正式训练阶段按验证 AUC 保存最佳模型；可选保存若干测试选择候选检查点。
    history, best_auc, best_path, best_metrics, test_candidate_paths = train(
        model,
        train_loader,
        valid_loader,
        optimizer,
        int(train_config["num_epochs"]),
        device,
        args.model_name,
        args.context_type,
        ckpt_dir,
        patience=int(args.patience),
        eval_interval=int(args.eval_interval),
        amp_enabled=amp_enabled,
        context_warmup_epochs=int(args.context_warmup_epochs),
        max_grad_norm=float(args.max_grad_norm),
        test_selection_interval=(int(args.test_eval_interval) if args.select_by_test_auc else 0),
        test_selection_start_epoch=int(args.test_eval_start_epoch),
    )
    train_valid_dataset_len = len(train_dataset) + int(full_valid_len)
    train_len = len(train_dataset)
    valid_len = len(valid_dataset) if valid_dataset is not None else 0
    context_info = {
        "context_dim": dataset.context_dim,
        "has_llm_context": dataset.has_llm_context,
        "has_llm_struct_context": getattr(dataset, "has_llm_struct_context", False),
        "llm_struct_dim": getattr(dataset, "llm_struct_dim", 0),
        "has_llm_struct_feature_context": getattr(dataset, "has_llm_struct_feature_context", False),
        "llm_struct_feature_dim": getattr(dataset, "llm_struct_feature_dim", 0),
    }
    if best_path.exists():
        load_model_state(
            model,
            best_path,
            device,
            allow_unused_context_mismatch=(args.context_type == "none"),
        )
    # user 模式沿用按学生切出的验证侧作为最终评估侧；new_concept 模式必须重新
    # 构造严格排除测试知识点历史的独立 test 角色。
    reuse_valid_as_test = args.split_mode != "new_concept"
    if reuse_valid_as_test:
        final_test_dataset = valid_dataset
        del train_loader, valid_loader, train_dataset, dataset, optimizer
    else:
        del train_loader, valid_loader, train_dataset, valid_dataset, dataset, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if args.split_mode == "new_concept":
        print("[train_context] building final test dataset split_role=test", flush=True)
        final_test_dataset = MOOCRadarStrict(**dataset_kwargs, split_role="test")
        print(f"[train_context] final test dataset ready len={len(final_test_dataset)}", flush=True)
    split_stats["test_dataset_stats"] = getattr(final_test_dataset, "split_stats", {})
    full_test_len = len(final_test_dataset)
    final_test_dataset = limit_dataset(final_test_dataset, int(args.test_limit), int(args.seed) + 29)
    split_stats["full_test_len"] = int(full_test_len)
    split_stats["test_eval_len"] = int(len(final_test_dataset))
    split_stats["test_limit"] = int(args.test_limit)
    test_loader = DataLoader(
        final_test_dataset,
        batch_size=effective_eval_batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_fn_with_context,
            context_type=args.context_type,
            context_tensor_dtype=context_tensor_dtype,
        ),
        num_workers=max(0, effective_num_workers),
        pin_memory=(device == "cuda"),
        persistent_workers=(effective_num_workers > 0),
    )
    print(
        f"[train_context] final test loader ready full_len={full_test_len} eval_len={len(final_test_dataset)} "
        f"batch_size={effective_eval_batch_size} num_workers={effective_num_workers}",
        flush=True,
    )
    test_selection_history: list[dict[str, Any]] = []
    best_test_epoch = None
    best_test_auc = None
    best_test_metrics = None
    # 注意：按测试 AUC 选检查点属于 oracle/上界诊断，会使用测试标签，不能作为
    # 无偏泛化结果报告。默认路径只评估按验证集或最后一轮选出的检查点。
    if args.select_by_test_auc:
        if not test_candidate_paths:
            raise RuntimeError(
                "No test-selection candidate checkpoints were saved. "
                "Check --test_eval_interval and --test_eval_start_epoch."
            )
        print(
            f"[train_context] test-best selection start candidates={len(test_candidate_paths)} "
            f"interval={int(args.test_eval_interval)} start_epoch={int(args.test_eval_start_epoch)}",
            flush=True,
        )
        best_test_path: Path | None = None
        best_test_auc_value = -float("inf")
        for candidate in test_candidate_paths:
            epoch = int(candidate["epoch"])
            candidate_path = Path(candidate["path"])
            if not candidate_path.exists():
                raise FileNotFoundError(candidate_path)
            load_model_state(
                model,
                candidate_path,
                device,
                allow_unused_context_mismatch=(args.context_type == "none"),
            )
            candidate_metrics = evaluate(
                model,
                test_loader,
                device,
                args.model_name,
                args.context_type,
                amp_enabled=amp_enabled,
            )
            candidate_auc = float(candidate_metrics.get("auc", float("nan")))
            test_selection_history.append(
                {
                    "epoch": epoch,
                    "checkpoint": str(candidate_path),
                    "test_metrics": candidate_metrics,
                }
            )
            print(
                f"[train_context] test-selection epoch={epoch} "
                f"auc={candidate_metrics.get('auc')} acc={candidate_metrics.get('acc')}",
                flush=True,
            )
            if not math.isnan(candidate_auc) and candidate_auc > best_test_auc_value:
                best_test_auc_value = candidate_auc
                best_test_epoch = epoch
                best_test_metrics = candidate_metrics
                best_test_path = candidate_path

        if best_test_path is None or best_test_metrics is None:
            raise RuntimeError("Unable to select best test checkpoint because all candidate AUC values were NaN")
        if best_test_path.resolve() != best_path.resolve():
            shutil.copy2(best_test_path, best_path)
        load_model_state(
            model,
            best_path,
            device,
            allow_unused_context_mismatch=(args.context_type == "none"),
        )
        best_test_auc = best_test_auc_value
        test_metrics = dict(best_test_metrics)
        print(
            f"[train_context] test-best selected epoch={best_test_epoch} "
            f"auc={best_test_auc} ckpt={best_test_path}",
            flush=True,
        )
    else:
        test_metrics = evaluate(model, test_loader, device, args.model_name, args.context_type, amp_enabled=amp_enabled)

    # 保存完整实验协议和历史，确保最终指标可以追溯到数据划分及超参数。
    metrics_path = ckpt_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "context_type": args.context_type,
                "split_mode": args.split_mode,
                "device": device,
                "cpu_threads": args.cpu_threads,
                "valid_concept_ratio": float(args.valid_concept_ratio),
                "valid_ratio": float(args.valid_ratio),
                "validation_disabled": bool(use_no_validation),
                "checkpoint_selection": (
                    "best_test_auc_oracle"
                    if args.select_by_test_auc
                    else ("last_epoch" if use_no_validation else "best_valid_auc")
                ),
                "context_storage_dtype": args.context_storage_dtype,
                "patience": int(args.patience),
                "eval_interval": int(args.eval_interval),
                "select_by_test_auc": bool(args.select_by_test_auc),
                "test_eval_interval": int(args.test_eval_interval),
                "test_eval_start_epoch": int(args.test_eval_start_epoch),
                "eval_batch_size": int(effective_eval_batch_size),
                "context_tensor_dtype": context_tensor_dtype,
                "ctx_encoder_dim": int(args.ctx_encoder_dim),
                "ctx_logit_hidden_dim": int(args.ctx_logit_hidden_dim),
                "ctx_logit_mode": args.ctx_logit_mode,
                "ctx_logit_init": float(args.ctx_logit_init),
                "gate_bias_init": float(args.gate_bias_init),
                "context_warmup_epochs": int(args.context_warmup_epochs),
                "context_lr_scale": float(args.context_lr_scale),
                "max_grad_norm": float(args.max_grad_norm),
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "amp": amp_enabled,
                "train_valid_dataset_len": train_valid_dataset_len,
                "train_len": train_len,
                "valid_len": valid_len,
                "full_valid_len": int(full_valid_len),
                "valid_limit": int(args.valid_limit),
                "test_len": len(final_test_dataset),
                "full_test_len": int(full_test_len),
                "test_limit": int(args.test_limit),
                "split_stats": split_stats,
                **context_info,
                "fusion_type": args.fusion_type,
                "best_valid_auc": best_auc,
                "best_valid_metrics": best_metrics,
                "best_test_epoch": best_test_epoch,
                "best_test_auc": best_test_auc,
                "best_test_metrics": best_test_metrics,
                "test_selection_history": test_selection_history,
                "test_candidate_paths": test_candidate_paths,
                "test_metrics": test_metrics,
                "best_ckpt": str(best_path),
                "history": history,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[OK] context training finished")
    print("[MODEL]", args.model_name)
    print("[CONTEXT]", args.context_type)
    print("[FUSION]", args.fusion_type)
    print("[CTX_LOGIT_MODE]", args.ctx_logit_mode)
    print("[SPLIT_MODE]", args.split_mode)
    print("[BEST_VALID_AUC]", "disabled" if use_no_validation else best_auc)
    print("[TEST_AUC]", test_metrics.get("auc"))
    print("[TEST_ACC]", test_metrics.get("acc"))
    print("[METRICS]", metrics_path)


if __name__ == "__main__":
    main()
