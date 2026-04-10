from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn import metrics
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataloader.context_collate import collate_fn_with_context
from dataloader.moocradar_strict import MOOCRadarStrict
from models.dkt_context import DKTContext
from models.saint_context import SAINTContext
from models.sakt_context import SAKTContext


def select_context(
    context_type: str,
    ctx_main: torch.Tensor | None,
    ctx_tpl: torch.Tensor | None,
    ctx_llm: torch.Tensor | None,
    ctx_llm_struct: torch.Tensor | None,
    ctx_llm_struct_features: torch.Tensor | None,
) -> torch.Tensor | None:
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
    raise ValueError(f"Unsupported context_type: {context_type}")


def build_model(model_name: str, dataset: MOOCRadarStrict, model_config: Dict[str, object], fusion_type: str, context_type: str):
    config = dict(model_config)
    ctx_dim = dataset.context_dim
    if context_type == "llm":
        ctx_dim = int(
            dataset.context_dim
            + getattr(dataset, "llm_struct_dim", 0)
            + getattr(dataset, "llm_struct_feature_dim", 0)
        )
    if model_name == "dkt":
        return DKTContext(dataset.num_q, ctx_dim=ctx_dim, fusion_type=fusion_type, **config)
    if model_name == "sakt":
        return SAKTContext(dataset.num_q, ctx_dim=ctx_dim, fusion_type=fusion_type, **config)
    if model_name == "saint":
        return SAINTContext(dataset.num_q, ctx_dim=ctx_dim, fusion_type=fusion_type, **config)
    raise ValueError(f"Unsupported model_name: {model_name}")


def compute_eval_metrics(preds_np: np.ndarray, targets_np: np.ndarray) -> Dict[str, float]:
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


def evaluate(model, loader, device: str, model_name: str, context_type: str) -> Dict[str, float]:
    model.eval()
    preds = []
    targets = []
    losses = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            q, r, qshft, rshft, mask, ctx_main, ctx_tpl, ctx_llm, ctx_llm_struct, ctx_llm_struct_features = batch
            q = q.to(device)
            r = r.to(device)
            qshft = qshft.to(device)
            rshft = rshft.to(device)
            mask = mask.to(device)
            ctx = select_context(context_type, ctx_main, ctx_tpl, ctx_llm, ctx_llm_struct, ctx_llm_struct_features)
            if ctx is not None:
                ctx = ctx.to(device, non_blocking=True)

            p = model(q.long(), r.long(), qshft.long(), ctx)
            p = torch.masked_select(p, mask)
            t = torch.masked_select(rshft.float(), mask)
            if p.numel() == 0:
                continue
            loss = binary_cross_entropy(p, t)
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
    return metrics_out


def train(model, train_loader, test_loader, optimizer, num_epochs: int, device: str, model_name: str, context_type: str, ckpt_dir: Path):
    history = []
    best_auc = -1.0
    best_metrics: Dict[str, float] | None = None
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "model.ckpt"

    for epoch in range(1, num_epochs + 1):
        model.train()
        batch_losses = []
        train_bar = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)
        for batch in train_bar:
            q, r, qshft, rshft, mask, ctx_main, ctx_tpl, ctx_llm, ctx_llm_struct, ctx_llm_struct_features = batch
            q = q.to(device)
            r = r.to(device)
            qshft = qshft.to(device)
            rshft = rshft.to(device)
            mask = mask.to(device)
            ctx = select_context(context_type, ctx_main, ctx_tpl, ctx_llm, ctx_llm_struct, ctx_llm_struct_features)
            if ctx is not None:
                ctx = ctx.to(device, non_blocking=True)

            p = model(q.long(), r.long(), qshft.long(), ctx)
            p = torch.masked_select(p, mask)
            t = torch.masked_select(rshft.float(), mask)
            if p.numel() == 0:
                continue

            optimizer.zero_grad()
            loss = binary_cross_entropy(p, t)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            batch_losses.append(loss_value)
            train_bar.set_postfix(loss=f"{loss_value:.4f}")

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        eval_metrics = evaluate(model, test_loader, device, model_name, context_type)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "eval_metrics": eval_metrics,
            }
        )
        print(
            "Epoch: {epoch}, AUC: {auc:.6f}, ACC: {acc:.6f}, F1: {f1:.6f}, "
            "BCE: {bce:.6f}, RMSE: {rmse:.6f}, Train Loss: {train_loss:.6f}, Eval Loss: {eval_loss:.6f}".format(
                epoch=epoch,
                auc=float(eval_metrics["auc"]) if not math.isnan(float(eval_metrics["auc"])) else float("nan"),
                acc=float(eval_metrics["acc"]),
                f1=float(eval_metrics["f1"]),
                bce=float(eval_metrics["bce"]),
                rmse=float(eval_metrics["rmse"]),
                train_loss=train_loss,
                eval_loss=float(eval_metrics["loss_mean"]),
            )
        )

        eval_auc = float(eval_metrics["auc"])
        if not math.isnan(eval_auc) and eval_auc > best_auc:
            best_auc = eval_auc
            best_metrics = dict(eval_metrics)
            torch.save(model.state_dict(), best_path)

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
        }

    return history, best_auc, best_path, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    workspace = Path(__file__).resolve().parent
    parser.add_argument("--model_name", type=str, default="dkt", choices=["dkt", "sakt", "saint"])
    parser.add_argument("--context_type", type=str, default="llm", choices=["none", "main", "template", "llm"])
    parser.add_argument("--fusion_type", type=str, default="gate", choices=["add", "concat", "gate"])
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
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    with (workspace / "models" / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    train_config = dict(config["train_config"])
    model_config = dict(config[args.model_name])

    seq_len = int(train_config["seq_len"])
    dataset = MOOCRadarStrict(
        seq_len=seq_len,
        problem_json=args.problem_json,
        student_json=args.student_json,
        context_embeddings_path=args.context_embeddings_path,
        dataset_dir=args.dataset_dir,
        require_llm_context=(args.context_type == "llm"),
        require_llm_struct_context=(args.context_type == "llm"),
        require_llm_struct_feature_context=(args.context_type == "llm"),
    )
    if args.context_type == "llm" and not dataset.has_llm_context:
        raise ValueError("Requested context_type=llm but context_embeddings.pkl does not contain llm_embeddings")
    if args.context_type == "llm" and not getattr(dataset, "has_llm_struct_context", False):
        raise ValueError("Requested context_type=llm but context_embeddings.pkl does not contain llm_struct_embeddings")
    if args.context_type == "llm" and not getattr(dataset, "has_llm_struct_feature_context", False):
        raise ValueError("Requested context_type=llm but context_embeddings.pkl does not contain llm_struct_features")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    split_dir = Path(args.dataset_dir).resolve() / f"splits_seq{seq_len}_seed{int(args.seed)}"
    train_dataset, test_dataset, split_stats = split_dataset(dataset, float(train_config["train_ratio"]), int(args.seed), split_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        collate_fn=collate_fn_with_context,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device == "cuda"),
        persistent_workers=(int(args.num_workers) > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=False,
        collate_fn=collate_fn_with_context,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device == "cuda"),
        persistent_workers=(int(args.num_workers) > 0),
    )

    if args.model_name in ("sakt", "saint"):
        model_config["n"] = seq_len

    model = build_model(args.model_name, dataset, model_config, args.fusion_type, args.context_type).to(device)

    optimizer_name = str(train_config["optimizer"]).lower()
    lr = float(train_config["learning_rate"])
    if optimizer_name == "sgd":
        optimizer = SGD(model.parameters(), lr, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = Adam(model.parameters(), lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    ckpt_dir = Path(args.ckpt_root).resolve() / args.model_name / args.context_type / args.fusion_type
    history, best_auc, best_path, best_metrics = train(
        model,
        train_loader,
        test_loader,
        optimizer,
        int(train_config["num_epochs"]),
        device,
        args.model_name,
        args.context_type,
        ckpt_dir,
    )

    metrics_path = ckpt_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "context_type": args.context_type,
                "device": device,
                "dataset_len": len(dataset),
                "train_len": len(train_dataset),
                "test_len": len(test_dataset),
                "split_stats": split_stats,
                "context_dim": dataset.context_dim,
                "has_llm_context": dataset.has_llm_context,
                "has_llm_struct_context": getattr(dataset, "has_llm_struct_context", False),
                "llm_struct_dim": getattr(dataset, "llm_struct_dim", 0),
                "has_llm_struct_feature_context": getattr(dataset, "has_llm_struct_feature_context", False),
                "llm_struct_feature_dim": getattr(dataset, "llm_struct_feature_dim", 0),
                "fusion_type": args.fusion_type,
                "best_auc": best_auc,
                "best_metrics": best_metrics,
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
    print("[BEST_AUC]", best_auc)
    print("[METRICS]", metrics_path)


if __name__ == "__main__":
    main()
