from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn import metrics
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Subset

from dataloader.context_collate import collate_fn_with_context
from dataloader.moocradar_strict import MOOCRadarStrict
from models.dkt_context import DKTContext
from models.saint_context import SAINTContext
from models.sakt_context import SAKTContext


def select_context(
    context_type: str,
    ctx_main: torch.Tensor,
    ctx_tpl: torch.Tensor,
    ctx_llm: torch.Tensor,
) -> torch.Tensor | None:
    if context_type == "none":
        return None
    if context_type == "main":
        return ctx_main
    if context_type == "template":
        return ctx_tpl
    if context_type == "llm":
        return ctx_llm
    raise ValueError(f"Unsupported context_type: {context_type}")


def build_model(model_name: str, dataset: MOOCRadarStrict, model_config: Dict[str, object], fusion_type: str):
    config = dict(model_config)
    if model_name == "dkt":
        return DKTContext(dataset.num_q, ctx_dim=dataset.context_dim, fusion_type=fusion_type, **config)
    if model_name == "sakt":
        return SAKTContext(dataset.num_q, ctx_dim=dataset.context_dim, fusion_type=fusion_type, **config)
    if model_name == "saint":
        return SAINTContext(dataset.num_q, ctx_dim=dataset.context_dim, fusion_type=fusion_type, **config)
    raise ValueError(f"Unsupported model_name: {model_name}")


def split_dataset(dataset, train_ratio: float, seed: int, split_dir: Path) -> Tuple[Subset, Subset]:
    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / "train_indices.pkl"
    test_path = split_dir / "test_indices.pkl"

    if train_path.exists() and test_path.exists():
        train_indices = torch.load(train_path)
        test_indices = torch.load(test_path)
    else:
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(dataset), generator=generator).tolist()
        train_size = int(len(dataset) * train_ratio)
        train_indices = perm[:train_size]
        test_indices = perm[train_size:]
        torch.save(train_indices, train_path)
        torch.save(test_indices, test_path)

    return Subset(dataset, train_indices), Subset(dataset, test_indices)


def evaluate(model, loader, device: str, model_name: str, context_type: str) -> Tuple[float, float]:
    model.eval()
    preds = []
    targets = []
    losses = []
    with torch.no_grad():
        for batch in loader:
            q, r, qshft, rshft, mask, ctx_main, ctx_tpl, ctx_llm = batch
            q = q.to(device)
            r = r.to(device)
            qshft = qshft.to(device)
            rshft = rshft.to(device)
            mask = mask.to(device)
            ctx_main = ctx_main.to(device)
            ctx_tpl = ctx_tpl.to(device)
            ctx_llm = ctx_llm.to(device)
            ctx = select_context(context_type, ctx_main, ctx_tpl, ctx_llm)

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
        return 0.0, 0.0

    preds_np = torch.cat(preds).numpy()
    targets_np = torch.cat(targets).numpy()
    auc = metrics.roc_auc_score(y_true=targets_np, y_score=preds_np)
    loss_mean = float(np.mean(losses)) if losses else 0.0
    return auc, loss_mean


def train(model, train_loader, test_loader, optimizer, num_epochs: int, device: str, model_name: str, context_type: str, ckpt_dir: Path):
    history = []
    best_auc = -1.0
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "model.ckpt"

    for epoch in range(1, num_epochs + 1):
        model.train()
        batch_losses = []
        for batch in train_loader:
            q, r, qshft, rshft, mask, ctx_main, ctx_tpl, ctx_llm = batch
            q = q.to(device)
            r = r.to(device)
            qshft = qshft.to(device)
            rshft = rshft.to(device)
            mask = mask.to(device)
            ctx_main = ctx_main.to(device)
            ctx_tpl = ctx_tpl.to(device)
            ctx_llm = ctx_llm.to(device)
            ctx = select_context(context_type, ctx_main, ctx_tpl, ctx_llm)

            p = model(q.long(), r.long(), qshft.long(), ctx)
            p = torch.masked_select(p, mask)
            t = torch.masked_select(rshft.float(), mask)
            if p.numel() == 0:
                continue

            optimizer.zero_grad()
            loss = binary_cross_entropy(p, t)
            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        auc, eval_loss = evaluate(model, test_loader, device, model_name, context_type)
        history.append({"epoch": epoch, "train_loss": train_loss, "eval_loss": eval_loss, "eval_auc": float(auc)})
        print(f"Epoch: {epoch}, AUC: {auc:.6f}, Train Loss: {train_loss:.6f}, Eval Loss: {eval_loss:.6f}")

        if auc > best_auc:
            best_auc = float(auc)
            torch.save(model.state_dict(), best_path)

    return history, best_auc, best_path


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
    )
    if args.context_type == "llm" and not dataset.has_llm_context:
        raise ValueError("Requested context_type=llm but context_embeddings.pkl does not contain llm_embeddings")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    split_dir = Path(args.dataset_dir).resolve() / f"splits_seq{seq_len}_seed{int(args.seed)}"
    train_dataset, test_dataset = split_dataset(dataset, float(train_config["train_ratio"]), int(args.seed), split_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        collate_fn=collate_fn_with_context,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=False,
        collate_fn=collate_fn_with_context,
    )

    if args.model_name in ("sakt", "saint"):
        model_config["n"] = seq_len

    model = build_model(args.model_name, dataset, model_config, args.fusion_type).to(device)

    optimizer_name = str(train_config["optimizer"]).lower()
    lr = float(train_config["learning_rate"])
    if optimizer_name == "sgd":
        optimizer = SGD(model.parameters(), lr, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = Adam(model.parameters(), lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    ckpt_dir = Path(args.ckpt_root).resolve() / args.model_name / args.context_type / args.fusion_type
    history, best_auc, best_path = train(
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
                "context_dim": dataset.context_dim,
                "has_llm_context": dataset.has_llm_context,
                "fusion_type": args.fusion_type,
                "best_auc": best_auc,
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
