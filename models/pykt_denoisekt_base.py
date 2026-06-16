"""DenoiseKT core adapted to use strict-data question-concept metadata."""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

from models.pykt_simplekt_base import CosinePositionalEmbedding


class DenoiseKT(nn.Module):
    def __init__(
        self,
        num_q,
        num_c,
        q_matrix,
        d_model=100,
        n_blocks=1,
        dropout=0.2,
        dropout1=0.1,
        bf=0.9,
        d_ff=256,
        final_fc_dim=512,
        final_fc_dim2=256,
        num_attn_heads=5,
        kq_same=1,
        max_seq_len=512,
    ):
        super().__init__()
        self.num_q = int(num_q)
        self.num_c = int(num_c)
        self.d_model = int(d_model)
        self.bf = float(bf)
        self.num_attn_heads = int(num_attn_heads)
        matrix = torch.as_tensor(q_matrix, dtype=torch.float32)
        edge_indices = matrix.nonzero(as_tuple=False)
        edge_values = torch.ones(edge_indices.size(0), dtype=torch.float32)
        graph = torch.sparse_coo_tensor(
            edge_indices.t(),
            edge_values,
            size=(self.num_q, self.num_c),
        ).coalesce()
        self.register_buffer("q_graph", graph)
        self.register_buffer("q_degree", matrix.sum(dim=-1, keepdim=True).clamp_min(1.0))
        self.register_buffer("c_degree", matrix.sum(dim=0, keepdim=True).t().clamp_min(1.0))
        concept_counts = matrix.count_nonzero(dim=-1)
        max_concepts = max(1, int(concept_counts.max().item()))
        q_concepts = torch.zeros(self.num_q, max_concepts, dtype=torch.long)
        write_positions = torch.zeros(self.num_q, dtype=torch.long)
        for q_idx, c_idx in edge_indices.tolist():
            position = int(write_positions[q_idx].item())
            q_concepts[q_idx, position] = int(c_idx) + 1
            write_positions[q_idx] += 1
        self.register_buffer("q_concepts", q_concepts)

        self.answer_embed = nn.Embedding(2, self.d_model)
        self.skill_embed = nn.Parameter(torch.empty(self.num_c, self.d_model))
        self.problem_embed = nn.Parameter(torch.empty(self.num_q, self.d_model))
        self.difficult_param = nn.Embedding(self.num_q, self.d_model)
        self.graph_linear = nn.Linear(self.d_model, self.d_model)
        self.graph_dropout = nn.Dropout(float(dropout1))
        xavier_uniform_(self.skill_embed)
        xavier_uniform_(self.problem_embed)
        nn.init.constant_(self.difficult_param.weight, 0.0)

        self.model = Architecture(
            n_blocks=int(n_blocks),
            d_model=self.d_model,
            d_ff=int(d_ff),
            n_heads=self.num_attn_heads,
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

    def _graph_problem_embedding(self):
        # Equivalent to aggregating a question graph induced by shared
        # concepts, without materializing a potentially huge Q x Q matrix.
        problem = self.graph_dropout(self.problem_embed)
        # CUDA sparse matrix multiplication does not implement FP16. Explicitly
        # disable the outer AMP context for graph propagation, then hand the
        # float32 result back to the normal autocast-enabled dense layers.
        with torch.autocast(device_type=problem.device.type, enabled=False):
            graph = self.q_graph.float()
            problem_fp32 = problem.float()
            concept_mean = (
                torch.sparse.mm(graph.transpose(0, 1), problem_fp32)
                / self.c_degree.float()
            )
            graph_problem = (
                torch.sparse.mm(graph, concept_mean) / self.q_degree.float()
            )
        return self.graph_linear(graph_problem.to(dtype=problem.dtype))

    def forward(self, qids, responses, return_hidden=False):
        qids = qids.long().clamp(min=0, max=self.num_q - 1)
        responses = responses.long().clamp(min=0, max=1)
        graph_problem = self._graph_problem_embedding()
        q_embed = F.embedding(qids, graph_problem)
        qa_embed = q_embed + self.answer_embed(responses)
        concept_ids = self.q_concepts[qids]
        skill_with_padding = torch.cat(
            [self.skill_embed.new_zeros(1, self.d_model), self.skill_embed],
            dim=0,
        )
        avg_skill = F.embedding(concept_ids, skill_with_padding).sum(dim=-2)
        avg_skill = avg_skill / concept_ids.ne(0).sum(dim=-1, keepdim=True).clamp_min(1)
        q_embed = q_embed + self.difficult_param(qids) * avg_skill

        left = concept_ids.unsqueeze(2).unsqueeze(-1)
        right = concept_ids.unsqueeze(1).unsqueeze(-2)
        shared_concept = ((left == right) & (left != 0)).any(dim=-1).any(dim=-1)
        seq_len = qids.size(1)
        positions = torch.arange(seq_len, device=qids.device)
        distance = (positions[:, None] - positions[None, :]).abs().to(q_embed.dtype)
        boost = torch.pow(q_embed.new_tensor(self.bf), distance)
        boost = boost * shared_concept.to(q_embed.dtype)
        boost = boost * (~torch.eye(seq_len, device=qids.device, dtype=torch.bool)).to(q_embed.dtype)
        boost = boost.unsqueeze(1).expand(-1, self.num_attn_heads, -1, -1)

        d_output = self.model(q_embed, qa_embed, boost)
        hidden = torch.cat([d_output, q_embed], dim=-1)
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
                    d_model,
                    d_model // n_heads,
                    d_ff,
                    n_heads,
                    dropout,
                    kq_same == 1,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, q_embed, qa_embed, boost):
        x = q_embed + self.position_emb(q_embed)
        y = qa_embed + self.position_emb(qa_embed)
        for block in self.blocks:
            x = block(x, x, y, boost)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, d_feature, n_heads, dropout, kq_same)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, key, values, boost):
        seq_len = query.size(1)
        mask = torch.tril(
            torch.ones((1, 1, seq_len, seq_len), device=query.device, dtype=torch.bool),
            diagonal=-1,
        )
        query = self.norm1(query + self.dropout1(self.attn(query, key, values, mask, boost)))
        update = self.linear2(self.dropout(F.relu(self.linear1(query))))
        return self.norm2(query + self.dropout2(update))


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

    def forward(self, q, k, v, mask, boost):
        batch = q.size(0)
        k = self.k_linear(k).view(batch, -1, self.h, self.d_k).transpose(1, 2)
        q = (self.k_linear(q) if self.kq_same else self.q_linear(q))
        q = q.view(batch, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch, -1, self.h, self.d_k).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        logits = logits * (1.0 + boost)
        logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)
        scores = F.softmax(logits, dim=-1)
        zero = torch.zeros(
            batch, self.h, 1, scores.size(-1), device=scores.device, dtype=scores.dtype
        )
        scores = torch.cat([zero, scores[:, :, 1:, :]], dim=2)
        output = torch.matmul(self.dropout(scores), v)
        output = output.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        return self.out_proj(output)
