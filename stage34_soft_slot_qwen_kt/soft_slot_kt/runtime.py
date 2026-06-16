from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import SoftSlotCollator, SoftSlotDataset, load_problem_catalog
from .model import LabelSpec, SoftSlotQwenKT, resolve_label_spec
from .utils import atomic_torch_save, compute_metrics, ensure_dir, read_json, resolve_device, resolve_torch_dtype, write_json


def slot_counts(slot_mode: str, context_soft_tokens: int, target_soft_tokens: int) -> Tuple[int, int, bool]:
    if slot_mode == "text_only":
        return 0, 0, False
    if slot_mode == "context":
        return int(context_soft_tokens), 0, False
    if slot_mode == "target":
        return 0, int(target_soft_tokens), False
    if slot_mode == "context_target":
        return int(context_soft_tokens), int(target_soft_tokens), False
    if slot_mode == "random":
        return int(context_soft_tokens), int(target_soft_tokens), True
    raise ValueError(f"Unsupported slot mode: {slot_mode}")


def load_frozen_llm(
    model_name_or_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str = "sdpa",
    gradient_checkpointing: bool = False,
):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Loading Qwen requires compatible transformers and huggingface-hub packages. "
            "Install the versions declared by the server environment before training or inference."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if str(attn_implementation).lower() not in {"", "auto"}:
        model_kwargs["attn_implementation"] = str(attn_implementation)
    llm = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    llm.to(device)
    llm.config.use_cache = False
    if gradient_checkpointing:
        try:
            llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            llm.gradient_checkpointing_enable()
    llm.eval()
    return tokenizer, llm


def build_model_and_data(args: Any, split: str):
    device = resolve_device(args.device)
    dtype = resolve_torch_dtype(args.dtype)
    tokenizer, llm = load_frozen_llm(
        args.model_name_or_path,
        device=device,
        dtype=dtype,
        attn_implementation=str(args.attn_implementation),
        gradient_checkpointing=bool(getattr(args, "gradient_checkpointing", False)),
    )
    context_tokens, target_tokens, random_slots = slot_counts(args.slot_mode, args.context_soft_tokens, args.target_soft_tokens)
    dataset = SoftSlotDataset(
        Path(args.feature_dir),
        split=split,
        context_fields=[x.strip() for x in args.context_fields.split(",") if x.strip()],
        target_fields=[x.strip() for x in args.target_fields.split(",") if x.strip()],
        limit=int(getattr(args, f"max_{split}_samples", 0)),
        seed=int(args.seed),
        drop_sdyn=bool(args.drop_sdyn),
        drop_collab=bool(args.drop_collab),
    )
    feature_manifest = dataset.feature_store.manifest
    problem_catalog = load_problem_catalog(Path(feature_manifest["problem_catalog_path"]))
    collator = SoftSlotCollator(
        tokenizer,
        problem_catalog,
        context_soft_tokens=context_tokens,
        target_soft_tokens=target_tokens,
        include_context_text=bool(args.include_context_text),
    )
    model = SoftSlotQwenKT(
        llm,
        context_dim=dataset.context_dim if context_tokens else 0,
        target_dim=dataset.target_dim if target_tokens else 0,
        context_soft_tokens=context_tokens,
        target_soft_tokens=target_tokens,
        projector_hidden_dim=int(args.projector_hidden_dim),
        projector_dropout=float(args.projector_dropout),
        random_slots=random_slots,
        random_seed=int(args.seed),
        llm_gradient_checkpointing=bool(getattr(args, "gradient_checkpointing", False)),
    ).to(device)
    label_spec = resolve_label_spec(tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=int(getattr(args, "eval_batch_size", args.batch_size)),
        shuffle=False,
        collate_fn=collator,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    return device, tokenizer, model, label_spec, dataset, collator, loader


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def model_batch_args(batch: Dict[str, Any], label_spec: LabelSpec, *, with_labels: bool) -> Dict[str, Any]:
    result = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "context_mask": batch["context_mask"],
        "target_mask": batch["target_mask"],
        "context_features": batch["context_features"],
        "target_features": batch["target_features"],
        "label_spec": label_spec,
    }
    if with_labels:
        result["labels"] = batch["labels"]
    return result


@torch.no_grad()
def evaluate(
    model: SoftSlotQwenKT,
    loader: DataLoader,
    label_spec: LabelSpec,
    device: torch.device,
    *,
    predictions_path: Optional[Path] = None,
    append_predictions: bool = False,
    desc: str = "evaluate",
) -> Dict[str, Any]:
    model.eval()
    labels: List[int] = []
    probabilities: List[float] = []
    output_handle = None
    if predictions_path is not None:
        ensure_dir(predictions_path.parent)
        output_handle = predictions_path.open("a" if append_predictions else "w", encoding="utf-8")
    try:
        for batch in tqdm(loader, desc=desc):
            metadata = batch["metadata"]
            batch = move_batch(batch, device)
            result = model(**model_batch_args(batch, label_spec, with_labels=False))
            probs = result["probabilities"].detach().cpu().tolist()
            batch_labels = batch["labels"].detach().cpu().tolist()
            labels.extend(int(x) for x in batch_labels)
            probabilities.extend(float(x) for x in probs)
            if output_handle is not None:
                for meta, probability in zip(metadata, probs):
                    output_handle.write(
                        json.dumps(
                            {
                                **meta,
                                "probability": float(probability),
                                "prediction": int(float(probability) >= 0.5),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    finally:
        if output_handle is not None:
            output_handle.close()
    return compute_metrics(labels, probabilities)


def save_checkpoint(
    path: Path,
    *,
    model: SoftSlotQwenKT,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[Any],
    epoch: int,
    global_step: int,
    args: Any,
    label_spec: LabelSpec,
) -> None:
    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    trainable_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if not name.startswith("llm.")
    }
    payload = {
        "trainable_state_dict": trainable_state,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "args": serialized_args,
        "label_spec": label_spec.to_dict(),
    }
    atomic_torch_save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    model: SoftSlotQwenKT,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Backward compatibility for trusted checkpoints written before Path
        # arguments were normalized to strings.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    unexpected = [name for name in unexpected if not name.startswith("llm.")]
    missing = [name for name in missing if not name.startswith("llm.")]
    if missing or unexpected:
        raise ValueError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint
