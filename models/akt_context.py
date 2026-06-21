import torch

from torch.nn import Module

from models.context_fusion import ContextFusion, ContextLogitHead
from models.pykt_akt_base import AKT


class AKTContext(Module):
    def __init__(
        self,
        num_q,
        ctx_dim,
        d_model,
        n_blocks,
        dropout,
        d_ff=256,
        kq_same=1,
        final_fc_dim=512,
        num_attn_heads=5,
        separate_qa=False,
        l2=1e-5,
        fusion_type="residual_gate",
        ctx_encoder_dim=256,
        ctx_group_dims=None,
        ctx_logit_hidden_dim=128,
        ctx_logit_mode="scaled",
        ctx_logit_init=-3.0,
        gate_bias_init=-2.0,
    ):
        super().__init__()
        self.num_q = int(num_q)
        self.ctx_dim = int(ctx_dim)
        self.d_model = int(d_model)
        self.hidden_dim = self.d_model * 2
        self.fusion_type = fusion_type

        self.base = AKT(
            self.num_q,
            0,
            self.d_model,
            int(n_blocks),
            float(dropout),
            d_ff=int(d_ff),
            kq_same=int(kq_same),
            final_fc_dim=int(final_fc_dim),
            num_attn_heads=int(num_attn_heads),
            separate_qa=bool(separate_qa),
            l2=float(l2),
            emb_type="qid",
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

        # pykt AKT predicts y[:, 1:] from cq=[q0, q1..qT] and historical responses.
        dummy_last_response = r[:, -1:]
        cq = torch.cat([q[:, :1], qry], dim=1)
        cr = torch.cat([r, dummy_last_response], dim=1)
        preds, _, hidden = self.base(cq.long(), cr.long(), qtest=True)
        hidden = hidden[:, 1:, :]

        if ctx is None:
            return preds[:, 1:]

        ctx_encoded = self.context_fusion.encode_context(ctx)
        ctx_logits = self.context_logit_head(ctx_encoded)
        hidden = self.context_fusion(hidden, ctx, ctx_encoded=ctx_encoded)
        logits = self.base.out(hidden).squeeze(-1)
        logits = self.context_logit_head.apply_to_logits(logits, ctx_logits)
        return torch.sigmoid(logits)
