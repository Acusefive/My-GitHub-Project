import torch
from torch.nn import Module

from models.context_fusion import ContextFusion, ContextLogitHead
from models.pykt_denoisekt_base import DenoiseKT


class DenoiseKTContext(Module):
    def __init__(
        self,
        num_q,
        num_c,
        q_matrix,
        ctx_dim,
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
        fusion_type="residual_gate",
        ctx_encoder_dim=256,
        ctx_group_dims=None,
        ctx_logit_hidden_dim=128,
        ctx_logit_mode="scaled",
        ctx_logit_init=-3.0,
        gate_bias_init=-2.0,
        **_,
    ):
        super().__init__()
        self.num_q = int(num_q)
        self.hidden_dim = int(d_model) * 2
        self.ctx_dim = int(ctx_dim)
        self.fusion_type = fusion_type
        self.base = DenoiseKT(
            num_q,
            num_c,
            q_matrix,
            d_model=d_model,
            n_blocks=n_blocks,
            dropout=dropout,
            dropout1=dropout1,
            bf=bf,
            d_ff=d_ff,
            final_fc_dim=final_fc_dim,
            final_fc_dim2=final_fc_dim2,
            num_attn_heads=num_attn_heads,
            kq_same=kq_same,
            max_seq_len=max_seq_len,
        )
        self.context_fusion = ContextFusion(
            self.hidden_dim,
            self.ctx_dim,
            mode=self.fusion_type,
            dropout=float(dropout),
            ctx_encoder_dim=ctx_encoder_dim,
            ctx_group_dims=ctx_group_dims,
            gate_bias_init=gate_bias_init,
        )
        self.context_logit_head = ContextLogitHead(
            self.context_fusion.ctx_encoder_dim,
            ctx_logit_hidden_dim,
            dropout=float(dropout),
            ctx_logit_mode=ctx_logit_mode,
            ctx_logit_init=ctx_logit_init,
        )

    def forward(self, q, r, qry, ctx=None):
        if q.size(1) == 0:
            return q.float()
        q = q.long().clamp(min=0, max=self.num_q - 1)
        qry = qry.long().clamp(min=0, max=self.num_q - 1)
        r = r.long().clamp(min=0, max=1)
        qids = torch.cat([q[:, :1], qry], dim=1)
        responses = torch.cat([r, r[:, -1:]], dim=1)
        preds, hidden = self.base(qids, responses, return_hidden=True)
        hidden = hidden[:, 1:, :]
        if ctx is None:
            return preds[:, 1:]
        ctx_encoded = self.context_fusion.encode_context(ctx)
        ctx_logits = self.context_logit_head(ctx_encoded)
        hidden = self.context_fusion(hidden, ctx, ctx_encoded=ctx_encoded)
        logits = self.base.out(hidden).squeeze(-1)
        logits = self.context_logit_head.apply_to_logits(logits, ctx_logits)
        return torch.sigmoid(logits)
