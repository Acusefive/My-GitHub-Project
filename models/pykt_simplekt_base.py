"""Core SimpleKT architecture adapted from pyKT for the unified trainer."""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


class SimpleKT(nn.Module):
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
    ):
        super().__init__()
        self.n_question = int(n_question)
        self.n_pid = int(n_pid)
        self.d_model = int(d_model)
        self.separate_qa = bool(separate_qa)
        self.l2 = float(l2)

        self.difficult_param = nn.Embedding(self.n_pid + 1, self.d_model)
        self.q_embed_diff = nn.Embedding(self.n_question, self.d_model)
        self.q_embed = nn.Embedding(self.n_question, self.d_model)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2 * self.n_question + 1, self.d_model)
        else:
            self.qa_embed = nn.Embedding(2, self.d_model)

        self.model = Architecture(
            n_blocks=int(n_blocks),
            d_model=self.d_model,
            d_ff=int(d_ff),
            n_heads=int(num_attn_heads),
            dropout=float(dropout),
            kq_same=int(kq_same),
            max_seq_len=int(max_seq_len),
        )
        self.out = nn.Sequential(
            nn.Linear(self.d_model * 2, int(final_fc_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(final_fc_dim), int(final_fc_dim2)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(final_fc_dim2), 1),
        )
        nn.init.constant_(self.difficult_param.weight, 0.0)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

    def forward(self, q_data, target, pid_data, return_hidden=False):
        q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        # SimpleKT's Rasch branch models problem-specific difficulty on top of
        # the concept and interaction embeddings.
        q_embed_diff_data = self.q_embed_diff(q_data)
        pid_embed_data = self.difficult_param(pid_data)
        q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

        d_output = self.model(q_embed_data, qa_embed_data)
        hidden = torch.cat([d_output, q_embed_data], dim=-1)
        preds = torch.sigmoid(self.out(hidden).squeeze(-1))
        if return_hidden:
            return preds, hidden
        return preds


class Architecture(nn.Module):
    def __init__(self, n_blocks, d_model, d_ff, n_heads, dropout, kq_same, max_seq_len):
        super().__init__()
        self.position_emb = CosinePositionalEmbedding(d_model, max_len=max_seq_len)
        self.blocks = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    d_feature=d_model // n_heads,
                    d_ff=d_ff,
                    n_heads=n_heads,
                    dropout=dropout,
                    kq_same=(kq_same == 1),
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


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same
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
        # Strictly lower triangular masking prevents the current response from
        # leaking into its own prediction. The first attention row is zeroed.
        mask = torch.tril(
            torch.ones((1, 1, seq_len, seq_len), device=query.device, dtype=torch.bool),
            diagonal=-1,
        )
        query2 = self.masked_attn_head(query, key, values, mask=mask, zero_pad=True)
        query = self.layer_norm1(query + self.dropout1(query2))
        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        return self.layer_norm2(query + self.dropout2(query2))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        self.d_model = int(d_model)
        self.d_k = int(d_feature)
        self.h = int(n_heads)
        self.kq_same = bool(kq_same)
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

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        scores = F.softmax(scores, dim=-1)
        if zero_pad:
            zero = torch.zeros(
                batch_size,
                self.h,
                1,
                scores.size(-1),
                device=scores.device,
                dtype=scores.dtype,
            )
            scores = torch.cat([zero, scores[:, :, 1:, :]], dim=2)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(output)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("weight", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        if x.size(1) > self.weight.size(1):
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds SimpleKT max_seq_len {self.weight.size(1)}"
            )
        return self.weight[:, : x.size(1), :].to(dtype=x.dtype)
