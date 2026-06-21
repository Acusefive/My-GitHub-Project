import torch
from torch.nn import Module

from models.context_fusion import ContextFusion, ContextLogitHead
from models.pykt_robustkt_base import RobustKT


class RobustKTContext(Module):
    def __init__(
        self,
        num_q,
        num_c,
        q_to_concept,
        ctx_dim,
        d_model=100,
        n_blocks=1,
        dropout=0.2,
        ks=5,
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
        **_,
    ):
        super().__init__()
        self.num_q = int(num_q)
        self.num_c = int(num_c)
        self.hidden_dim = int(d_model) * 2
        self.ctx_dim = int(ctx_dim)
        self.fusion_type = fusion_type
        self.register_buffer("q_to_concept", torch.as_tensor(q_to_concept, dtype=torch.long))
        self._training_auxiliary_loss = None

        self.base = RobustKT(
            self.num_c + 1,
            self.num_q,
            int(d_model),
            int(n_blocks),
            float(dropout),
            ks=int(ks),
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
        q = q.long().clamp(min=0, max=self.num_q - 1)
        qry = qry.long().clamp(min=0, max=self.num_q - 1)
        r = r.long().clamp(min=0, max=1)
        pid_data = torch.cat([q[:, :1], qry], dim=1)
        concept_data = self.q_to_concept[pid_data]
        response_data = torch.cat([r, r[:, -1:]], dim=1)
        preds, reg_loss, hidden = self.base(
            concept_data,
            response_data,
            pid_data=pid_data,
            qtest=True,
        )
        self._training_auxiliary_loss = reg_loss
        hidden = hidden[:, 1:, :]
        if ctx is None:
            return preds[:, 1:]
        ctx_encoded = self.context_fusion.encode_context(ctx)
        ctx_logits = self.context_logit_head(ctx_encoded)
        hidden = self.context_fusion(hidden, ctx, ctx_encoded=ctx_encoded)
        logits = self.base.out(hidden).squeeze(-1)
        logits = self.context_logit_head.apply_to_logits(logits, ctx_logits)
        return torch.sigmoid(logits)

    def get_training_auxiliary_loss(self):
        """Return the Rasch difficulty regularizer used by original RobustKT."""
        return self._training_auxiliary_loss
