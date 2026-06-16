from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from soft_slot_kt.cli import add_model_arguments
from soft_slot_kt.data import SoftSlotDataset
from soft_slot_kt.prompts import PROMPT_VERSION
from soft_slot_kt.runtime import (
    build_model_and_data,
    evaluate,
    load_checkpoint,
    model_batch_args,
    move_batch,
    save_checkpoint,
)
from soft_slot_kt.utils import ensure_dir, parse_csv, set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage34-informed Soft-Slot Qwen KT projectors.")
    add_model_arguments(parser)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--save_epochs", default="10,20,30,40")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--validation_disabled", action="store_true")
    parser.add_argument("--evaluate_test_after_train", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    set_seed(int(args.seed))
    output_dir = ensure_dir(args.output_dir.resolve())
    complete_path = output_dir / "training_complete.json"
    if complete_path.exists() and not args.force:
        completed_run = json.loads(complete_path.read_text(encoding="utf-8"))
        completed_epochs = int(completed_run.get("epochs", 0))
        if completed_epochs >= int(args.epochs):
            print(f"[SKIP] completed run exists: {complete_path}")
            return
        print(f"[EXTEND] completed_epochs={completed_epochs} requested_epochs={args.epochs}")

    device, _tokenizer, model, label_spec, train_dataset, collator, _unused_loader = build_model_and_data(args, "train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=collator,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    trainable = model.trainable_parameters()
    if not trainable:
        raise ValueError("This slot mode has no trainable projector. Use the isolated infer_soft_slot_kt.py entrypoint directly.")
    write_json(
        {
            "feature_protocol": train_dataset.feature_store.manifest["protocol"],
            "feature_audit": train_dataset.feature_store.manifest["audit"],
            "slot_mode": args.slot_mode,
            "context_fields": parse_csv(args.context_fields),
            "target_fields": parse_csv(args.target_fields),
            "context_soft_tokens": int(model.context_soft_tokens),
            "target_soft_tokens": int(model.target_soft_tokens),
            "prompt_version": PROMPT_VERSION,
            "label_spec": label_spec.to_dict(),
            "llm_parameter_count": int(sum(parameter.numel() for parameter in model.llm.parameters())),
            "llm_trainable_parameter_count": int(sum(parameter.numel() for parameter in model.llm.parameters() if parameter.requires_grad)),
            "projector_trainable_parameter_count": int(sum(parameter.numel() for parameter in trainable)),
            "final_inference_uses_downstream_kt_baseline": False,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
        output_dir / "run_manifest.json",
    )
    optimizer = torch.optim.AdamW(trainable, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    amp_enabled = device.type == "cuda" and args.dtype in {"float16", "bfloat16"}
    amp_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    start_epoch = 1
    global_step = 0
    resume_path = output_dir / "checkpoint_last.pt" if args.resume == "auto" else Path(args.resume)
    if str(args.resume).lower() not in {"", "none"} and resume_path.exists():
        checkpoint = load_checkpoint(resume_path, model=model, optimizer=optimizer, scaler=scaler)
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        print(f"[RESUME] {resume_path} epoch={start_epoch - 1} global_step={global_step}")

    valid_loader = None
    if not args.validation_disabled:
        valid_dataset = SoftSlotDataset(
            Path(args.feature_dir),
            split="valid",
            context_fields=parse_csv(args.context_fields),
            target_fields=parse_csv(args.target_fields),
            limit=int(args.max_valid_samples),
            seed=int(args.seed),
            drop_sdyn=bool(args.drop_sdyn),
            drop_collab=bool(args.drop_collab),
        )
        if len(valid_dataset):
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=int(args.eval_batch_size),
                shuffle=False,
                collate_fn=collator,
                num_workers=int(args.num_workers),
                pin_memory=(device.type == "cuda"),
            )

    save_epochs = {int(x) for x in parse_csv(args.save_epochs)}
    history = []
    grad_accum = max(1, int(args.gradient_accumulation_steps))
    for epoch in range(start_epoch, int(args.epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        update_count = 0
        progress = tqdm(train_loader, desc=f"soft-slot train epoch {epoch}")
        for batch_index, batch in enumerate(progress, start=1):
            batch = move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                result = model(**model_batch_args(batch, label_spec, with_labels=True))
                loss = result["loss"] / grad_accum
            scaler.scale(loss).backward()
            running_loss += float(loss.detach().cpu().item()) * grad_accum
            should_step = batch_index % grad_accum == 0 or batch_index == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, float(args.max_grad_norm))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                update_count += 1
            progress.set_postfix(loss=f"{running_loss / max(1, batch_index):.4f}", step=global_step)

        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": running_loss / max(1, len(train_loader)),
        }
        if valid_loader is not None:
            epoch_record["valid_metrics"] = evaluate(model, valid_loader, label_spec, device, desc=f"valid epoch {epoch}")
        history.append(epoch_record)
        write_json({"history": history}, output_dir / "training_history.json")
        save_checkpoint(
            output_dir / "checkpoint_last.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            args=args,
            label_spec=label_spec,
        )
        if epoch in save_epochs:
            save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch}.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                args=args,
                label_spec=label_spec,
            )

    final: dict = {
        "completed": True,
        "epochs": int(args.epochs),
        "global_step": global_step,
        "label_spec": label_spec.to_dict(),
        "feature_protocol": train_dataset.feature_store.manifest["protocol"],
        "checkpoint": str(output_dir / "checkpoint_last.pt"),
    }
    if args.evaluate_test_after_train:
        test_dataset = SoftSlotDataset(
            Path(args.feature_dir),
            split="test",
            context_fields=parse_csv(args.context_fields),
            target_fields=parse_csv(args.target_fields),
            limit=int(args.max_test_samples),
            seed=int(args.seed),
            drop_sdyn=bool(args.drop_sdyn),
            drop_collab=bool(args.drop_collab),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(args.eval_batch_size),
            shuffle=False,
            collate_fn=collator,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
        )
        final["test_metrics"] = evaluate(
            model,
            test_loader,
            label_spec,
            device,
            predictions_path=output_dir / "predictions.test.jsonl",
            desc="final test",
        )
    write_json(final, complete_path)
    print("[OK] Soft-Slot training finished")
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
