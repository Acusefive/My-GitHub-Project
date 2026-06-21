"""Core SparseKT architecture adapted from pyKT for the unified trainer."""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

from models.pykt_simplekt_base import CosinePositionalEmbedding, SimpleKT


class SparseKT(SimpleKT):
    def __init__(
        self,
        n_question,
        n_pid,
        d_model,
        n_blocks,
        dropout,
        d_ff=256,
        kq_same=1,
        final_fc_dim=512,
        final_fc_dim2=256,
        num_attn_heads=5,
        separate_qa=False,
        l2=1e-5,
        max_seq_len=512,
        sparse_ratio=0.8,
        k_index=5,
        stride=1,
    ):
        super().__init__(
            n_question,
            n_pid,
            d_model=d_model,
            n_blocks=n_blocks,
            dropout=dropout,
            d_ff=d_ff,
            kq_same=kq_same,
            final_fc_dim=final_fc_dim,
            final_fc_dim2=final_fc_dim2,
            num_attn_heads=num_attn_heads,
            separate_qa=separate_qa,
            l2=l2,
            max_seq_len=max_seq_len,
        )
        self.sparse_ratio = float(sparse_ratio)
        self.k_index = int(k_index)
        self.stride = int(stride)
        self.model = SparseArchitecture(
            n_blocks=int(n_blocks),
            d_model=self.d_model,
            d_ff=int(d_ff),
            n_heads=int(num_attn_heads),
            dropout=float(dropout),
            kq_same=int(kq_same),
            max_seq_len=int(max_seq_len),
            k_index=self.k_index,
        )


class SparseArchitecture(nn.Module):
    def __init__(
        self,
        n_blocks,
        d_model,
        d_ff,
        n_heads,
        dropout,
        kq_same,
        max_seq_len,
        k_index,
    ):
        super().__init__()
        self.position_emb = CosinePositionalEmbedding(d_model, max_len=max_seq_len)
        self.blocks = nn.ModuleList(
            [
                SparseTransformerLayer(
                    d_model=d_model,
                    d_feature=d_model // n_heads,
                    d_ff=d_ff,
                    n_heads=n_heads,
                    dropout=dropout,
                    kq_same=(kq_same == 1),
                    k_index=k_index,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, q_embed_data, qa_embed_data):
        q_embed_data = q_embed_data + self.position_emb(q_embed_data)
        qa_embed_data = qa_embed_data + self.position_emb(qa_embed_data)
        x = q_embed_data
        for block in self.blocks:
            x = block(query=x, key=x, values=qa_embed_data)
        return x


class SparseTransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same, k_index):
        super().__init__()
        self.masked_attn_head = SparseMultiHeadAttention(
            d_model,
            d_feature,
            n_heads,
            dropout,
            kq_same=kq_same,
            k_index=k_index,
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, values):
        seq_len = query.size(1)
        mask = torch.tril(
            torch.ones((1, 1, seq_len, seq_len), device=query.device, dtype=torch.bool),
            diagonal=-1,
        )
        query2 = self.masked_attn_head(query, key, values, mask=mask, zero_pad=True)
        query = self.layer_norm1(query + self.dropout1(query2))
        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        return self.layer_norm2(query + self.dropout2(query2))


class SparseMultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, k_index, bias=True):
        super().__init__()
        self.d_model = int(d_model)
        self.d_k = int(d_feature)
        self.h = int(n_heads)
        self.kq_same = bool(kq_same)
        self.k_index = max(1, int(k_index))
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if not self.kq_same:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if not self.kq_same:
            xavier_uniform_(self.q_linear.weight)
        constant_(self.k_linear.bias, 0.0)
        constant_(self.v_linear.bias, 0.0)
        if not self.kq_same:
            constant_(self.q_linear.bias, 0.0)
        constant_(self.out_proj.bias, 0.0)

    def forward(self, q, k, v, mask, zero_pad):
        batch_size = q.size(0)
        k = self.k_linear(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        if self.kq_same:
            q = self.k_linear(q)
        else:
            q = self.q_linear(q)
        q = q.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask_fill_value = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(mask == 0, mask_fill_value)

        # SparseKT keeps normal causal attention while fewer than K historical
        # interactions exist, then retains only the K strongest past logits.
        seq_len = logits.size(-1)
        if self.k_index < seq_len:
            topk_indices = torch.topk(logits, k=self.k_index, dim=-1).indices
            sparse_mask = torch.zeros_like(mask.expand_as(logits))
            sparse_mask.scatter_(-1, topk_indices, True)
            row_ids = torch.arange(seq_len, device=logits.device).view(1, 1, seq_len, 1)
            use_sparse = row_ids > self.k_index
            effective_mask = torch.where(use_sparse, sparse_mask, mask.expand_as(logits))
            logits = logits.masked_fill(~effective_mask, mask_fill_value)

        scores = F.softmax(logits, dim=-1)
        if zero_pad:
            zero = torch.zeros(
                batch_size,
                self.h,
                1,
                seq_len,
                device=scores.device,
                dtype=scores.dtype,
            )
            scores = torch.cat([zero, scores[:, :, 1:, :]], dim=2)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(output)
