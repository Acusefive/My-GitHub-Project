import torch

from torch.nn import Module, Parameter, Embedding, Linear, Transformer, LayerNorm, Dropout
from torch.nn.init import normal_

from models.context_fusion import ContextFusion, ContextLogitHead


class SAINTContext(Module):
    def __init__(
        self,
        num_q,
        n,
        d,
        num_attn_heads,
        dropout,
        ctx_dim,
        num_tr_layers=1,
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
        self.num_tr_layers = num_tr_layers
        self.fusion_type = fusion_type

        self.E = Embedding(self.num_q, self.d)
        self.R = Embedding(2, self.d)
        self.P = Parameter(torch.Tensor(self.n, self.d))
        self.S = Parameter(torch.Tensor(1, self.d))
        normal_(self.P)
        normal_(self.S)

        self.transformer = Transformer(
            d_model=self.d,
            nhead=self.num_attn_heads,
            num_encoder_layers=self.num_tr_layers,
            num_decoder_layers=self.num_tr_layers,
            dropout=self.dropout,
            batch_first=True,
        )
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
        self.fuse_norm = LayerNorm(self.d)
        self.fuse_dropout = Dropout(self.dropout)
        self.pred = Linear(self.d, 1)

    def forward(self, q, r, qry, ctx=None):
        batch_size, seq_len = q.shape
        pos = self.P[:seq_len].unsqueeze(0)

        src = self.E(qry) + pos
        hist_r = self.R(r)
        start = self.S.unsqueeze(0).expand(batch_size, 1, self.d)
        tgt = torch.cat([start, hist_r[:, :-1, :]], dim=1) + pos

        mask = self.transformer.generate_square_subsequent_mask(seq_len).to(q.device)
        out = self.transformer(src=src, tgt=tgt, tgt_mask=mask)

        ctx_logits = None
        if ctx is not None:
            ctx_encoded = self.context_fusion.encode_context(ctx)
            ctx_logits = self.context_logit_head(ctx_encoded)
            out = self.fuse_norm(self.context_fusion(out, ctx, ctx_encoded=ctx_encoded))
            out = self.fuse_dropout(out)

        logits = self.pred(out).squeeze(-1)
        logits = self.context_logit_head.apply_to_logits(logits, ctx_logits)
        p = torch.sigmoid(logits)
        return p
