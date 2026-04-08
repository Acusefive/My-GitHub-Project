from __future__ import annotations

from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn_with_context(batch, pad_val: int = -1):
    q_seqs: List[torch.Tensor] = []
    r_seqs: List[torch.Tensor] = []
    qshft_seqs: List[torch.Tensor] = []
    rshft_seqs: List[torch.Tensor] = []
    ctx_main_seqs: List[torch.Tensor] = []
    ctx_tpl_seqs: List[torch.Tensor] = []
    ctx_llm_seqs: List[torch.Tensor] = []

    for q_seq, r_seq, ctx_main_seq, ctx_tpl_seq, ctx_llm_seq in batch:
        q_seqs.append(torch.tensor(q_seq[:-1], dtype=torch.long))
        r_seqs.append(torch.tensor(r_seq[:-1], dtype=torch.long))
        qshft_seqs.append(torch.tensor(q_seq[1:], dtype=torch.long))
        rshft_seqs.append(torch.tensor(r_seq[1:], dtype=torch.long))
        ctx_main_seqs.append(torch.tensor(ctx_main_seq[1:], dtype=torch.float32))
        ctx_tpl_seqs.append(torch.tensor(ctx_tpl_seq[1:], dtype=torch.float32))
        ctx_llm_seqs.append(torch.tensor(ctx_llm_seq[1:], dtype=torch.float32))

    q_seqs = pad_sequence(q_seqs, batch_first=True, padding_value=pad_val)
    r_seqs = pad_sequence(r_seqs, batch_first=True, padding_value=pad_val)
    qshft_seqs = pad_sequence(qshft_seqs, batch_first=True, padding_value=pad_val)
    rshft_seqs = pad_sequence(rshft_seqs, batch_first=True, padding_value=pad_val)
    ctx_main_seqs = pad_sequence(ctx_main_seqs, batch_first=True, padding_value=0.0)
    ctx_tpl_seqs = pad_sequence(ctx_tpl_seqs, batch_first=True, padding_value=0.0)
    ctx_llm_seqs = pad_sequence(ctx_llm_seqs, batch_first=True, padding_value=0.0)

    mask_seqs = (q_seqs != pad_val) & (qshft_seqs != pad_val)
    q_seqs = q_seqs.masked_fill(~mask_seqs, 0)
    r_seqs = r_seqs.masked_fill(~mask_seqs, 0)
    qshft_seqs = qshft_seqs.masked_fill(~mask_seqs, 0)
    rshft_seqs = rshft_seqs.masked_fill(~mask_seqs, 0)
    ctx_main_seqs = ctx_main_seqs * mask_seqs.unsqueeze(-1)
    ctx_tpl_seqs = ctx_tpl_seqs * mask_seqs.unsqueeze(-1)
    ctx_llm_seqs = ctx_llm_seqs * mask_seqs.unsqueeze(-1)

    return q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs, ctx_main_seqs, ctx_tpl_seqs, ctx_llm_seqs
