from __future__ import annotations

import argparse


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument(
        "--slot_mode",
        choices=["text_only", "context", "target", "context_target", "random"],
        default="context_target",
    )
    parser.add_argument("--context_fields", default="llm_embeddings,llm_struct_features,stage34_numeric")
    parser.add_argument("--target_fields", default="hqtext,hqid,semantic,collaborative")
    parser.add_argument("--context_soft_tokens", type=int, default=4)
    parser.add_argument("--target_soft_tokens", type=int, default=2)
    parser.add_argument("--projector_hidden_dim", type=int, default=512)
    parser.add_argument("--projector_dropout", type=float, default=0.1)
    parser.add_argument("--drop_sdyn", action="store_true")
    parser.add_argument("--drop_collab", action="store_true")
    parser.add_argument(
        "--include_context_text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include raw Stage3.4 history text. Disabled by default; enable only for the full-history ablation.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_valid_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
