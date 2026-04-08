import torch

from torch.nn import Module, Parameter, Embedding, Sequential, Linear, ReLU, MultiheadAttention, LayerNorm, Dropout
from torch.nn.init import kaiming_normal_

from models.context_fusion import ContextFusion


class SAKTContext(Module):
    def __init__(self, num_q, n, d, num_attn_heads, dropout, ctx_dim, fusion_type="gate"):
        super().__init__()
        self.num_q = num_q
        self.n = n
        self.d = d
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.ctx_dim = ctx_dim
        self.fusion_type = fusion_type

        self.M = Embedding(self.num_q * 2, self.d)
        self.E = Embedding(self.num_q, d)
        self.P = Parameter(torch.Tensor(self.n, self.d))
        kaiming_normal_(self.P)

        self.attn = MultiheadAttention(self.d, self.num_attn_heads, dropout=self.dropout)
        self.attn_dropout = Dropout(self.dropout)
        self.attn_layer_norm = LayerNorm(self.d)

        self.FFN = Sequential(
            Linear(self.d, self.d),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.d, self.d),
            Dropout(self.dropout),
        )
        self.FFN_layer_norm = LayerNorm(self.d)
        self.context_fusion = ContextFusion(self.d, self.ctx_dim, mode=self.fusion_type)
        self.pred = Linear(self.d, 1)

    def forward(self, q, r, qry, ctx=None):
        x = q + self.num_q * r

        M = self.M(x).permute(1, 0, 2)
        E = self.E(qry).permute(1, 0, 2)
        pos = self.P[: M.shape[0]].unsqueeze(1)

        causal_mask = torch.triu(torch.ones([E.shape[0], M.shape[0]], device=q.device), diagonal=1).bool()

        M = M + pos
        E = E + pos[: E.shape[0]]

        S, _ = self.attn(E, M, M, attn_mask=causal_mask)
        S = self.attn_dropout(S)
        S = S.permute(1, 0, 2)
        E = E.permute(1, 0, 2)

        S = self.attn_layer_norm(S + E)
        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)

        if ctx is not None:
            F = self.context_fusion(F, ctx)

        p = torch.sigmoid(self.pred(F)).squeeze(-1)
        return p
