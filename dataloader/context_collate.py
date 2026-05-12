from __future__ import annotations

from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn_with_context(
    batch,
    pad_val: int = -1,
    context_type: str = "all",
    context_tensor_dtype: str = "float32",
):
    context_type = str(context_type or "all")
    if context_tensor_dtype not in {"float32", "float16"}:
        raise ValueError(f"Unsupported context_tensor_dtype: {context_tensor_dtype}")
    ctx_dtype = torch.float16 if context_tensor_dtype == "float16" else torch.float32
    use_main = context_type in {"all", "main"}
    use_template = context_type in {"all", "template"}
    use_llm = context_type in {"all", "llm"}
    q_seqs: List[torch.Tensor] = []
    r_seqs: List[torch.Tensor] = []
    qshft_seqs: List[torch.Tensor] = []
    rshft_seqs: List[torch.Tensor] = []
    eval_mask_seqs: List[torch.Tensor] = []
    ctx_main_seqs: List[torch.Tensor] = []
    ctx_tpl_seqs: List[torch.Tensor] = []
    ctx_llm_seqs: List[torch.Tensor] = []
    ctx_llm_struct_seqs: List[torch.Tensor] = []
    ctx_llm_struct_feature_seqs: List[torch.Tensor] = []

    for item in batch:
        if len(item) == 8:
            q_seq, r_seq, eval_mask_seq, ctx_main_seq, ctx_tpl_seq, ctx_llm_seq, ctx_llm_struct_seq, ctx_llm_struct_feature_seq = item
        else:
            q_seq, r_seq, ctx_main_seq, ctx_tpl_seq, ctx_llm_seq, ctx_llm_struct_seq, ctx_llm_struct_feature_seq = item
            eval_mask_seq = torch.ones(len(q_seq), dtype=torch.long)
        q_seqs.append(torch.tensor(q_seq[:-1], dtype=torch.long))
        r_seqs.append(torch.tensor(r_seq[:-1], dtype=torch.long))
        qshft_seqs.append(torch.tensor(q_seq[1:], dtype=torch.long))
        rshft_seqs.append(torch.tensor(r_seq[1:], dtype=torch.long))
        eval_mask_seqs.append(torch.tensor(eval_mask_seq[1:], dtype=torch.bool))
        if use_main:
            ctx_main_seqs.append(torch.as_tensor(ctx_main_seq[1:], dtype=ctx_dtype))
        if use_template:
            ctx_tpl_seqs.append(torch.as_tensor(ctx_tpl_seq[1:], dtype=ctx_dtype))
        if use_llm:
            ctx_llm_seqs.append(torch.as_tensor(ctx_llm_seq[1:], dtype=ctx_dtype))
            ctx_llm_struct_seqs.append(torch.as_tensor(ctx_llm_struct_seq[1:], dtype=ctx_dtype))
            ctx_llm_struct_feature_seqs.append(torch.as_tensor(ctx_llm_struct_feature_seq[1:], dtype=ctx_dtype))

    q_seqs = pad_sequence(q_seqs, batch_first=True, padding_value=pad_val)
    r_seqs = pad_sequence(r_seqs, batch_first=True, padding_value=pad_val)
    qshft_seqs = pad_sequence(qshft_seqs, batch_first=True, padding_value=pad_val)
    rshft_seqs = pad_sequence(rshft_seqs, batch_first=True, padding_value=pad_val)
    eval_mask_seqs = pad_sequence(eval_mask_seqs, batch_first=True, padding_value=0)
    ctx_main_seqs = pad_sequence(ctx_main_seqs, batch_first=True, padding_value=0.0) if use_main else None
    ctx_tpl_seqs = pad_sequence(ctx_tpl_seqs, batch_first=True, padding_value=0.0) if use_template else None
    ctx_llm_seqs = pad_sequence(ctx_llm_seqs, batch_first=True, padding_value=0.0) if use_llm else None
    ctx_llm_struct_seqs = pad_sequence(ctx_llm_struct_seqs, batch_first=True, padding_value=0.0) if use_llm else None
    ctx_llm_struct_feature_seqs = (
        pad_sequence(ctx_llm_struct_feature_seqs, batch_first=True, padding_value=0.0) if use_llm else None
    )

    mask_seqs = (q_seqs != pad_val) & (qshft_seqs != pad_val)
    eval_mask_seqs = eval_mask_seqs & mask_seqs
    q_seqs = q_seqs.masked_fill(~mask_seqs, 0)
    r_seqs = r_seqs.masked_fill(~mask_seqs, 0)
    qshft_seqs = qshft_seqs.masked_fill(~mask_seqs, 0)
    rshft_seqs = rshft_seqs.masked_fill(~mask_seqs, 0)
    if ctx_main_seqs is not None:
        ctx_main_seqs = ctx_main_seqs * mask_seqs.unsqueeze(-1)
    if ctx_tpl_seqs is not None:
        ctx_tpl_seqs = ctx_tpl_seqs * mask_seqs.unsqueeze(-1)
    if ctx_llm_seqs is not None:
        ctx_llm_seqs = ctx_llm_seqs * mask_seqs.unsqueeze(-1)
    if ctx_llm_struct_seqs is not None:
        ctx_llm_struct_seqs = ctx_llm_struct_seqs * mask_seqs.unsqueeze(-1)
    if ctx_llm_struct_feature_seqs is not None:
        ctx_llm_struct_feature_seqs = ctx_llm_struct_feature_seqs * mask_seqs.unsqueeze(-1)

    return (
        q_seqs,
        r_seqs,
        qshft_seqs,
        rshft_seqs,
        mask_seqs,
        eval_mask_seqs,
        ctx_main_seqs,
        ctx_tpl_seqs,
        ctx_llm_seqs,
        ctx_llm_struct_seqs,
        ctx_llm_struct_feature_seqs,
    )
