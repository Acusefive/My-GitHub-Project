import torch

from torch.nn import Module, Parameter, Embedding, Sequential, Linear, ReLU, MultiheadAttention, LayerNorm, Dropout
from torch.nn.init import kaiming_normal_

from models.context_fusion import ContextFusion, ContextLogitHead


class SAKTContext(Module):
    def __init__(
        self,
        num_q,
        n,
        d,
        num_attn_heads,
        dropout,
        ctx_dim,
        fusion_type="residual_gate",
        ctx_encoder_dim=256,
        ctx_group_dims=None,
        ctx_logit_hidden_dim=128,
        ctx_logit_mode="scaled",
        ctx_logit_init=-3.0,
        gate_bias_init=-2.0,
    ):
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
        self.context_fusion = ContextFusion(
            self.d,
            self.ctx_dim,
            mode=self.fusion_type,
            dropout=self.dropout,
            ctx_encoder_dim=ctx_encoder_dim,
            ctx_group_dims=ctx_group_dims,
            gate_bias_init=gate_bias_init,
        )
        self.context_logit_head = ContextLogitHead(
            self.context_fusion.ctx_encoder_dim,
            ctx_logit_hidden_dim,
            dropout=self.dropout,
            ctx_logit_mode=ctx_logit_mode,
            ctx_logit_init=ctx_logit_init,
        )
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

        ctx_logits = None
        if ctx is not None:
            ctx_encoded = self.context_fusion.encode_context(ctx)
            ctx_logits = self.context_logit_head(ctx_encoded)
            F = self.context_fusion(F, ctx, ctx_encoded=ctx_encoded)

        logits = self.pred(F).squeeze(-1)
        logits = self.context_logit_head.apply_to_logits(logits, ctx_logits)
        p = torch.sigmoid(logits)
        return p
