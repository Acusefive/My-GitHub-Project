"""KeenKT wrapper with the shared context-fusion interface."""

import torch
from torch.nn import Module

from models.context_fusion import ContextFusion, ContextLogitHead
from models.pykt_keenkt_base import KeenKT


class KeenKTContext(Module):
    def __init__(
        self,
        num_q,
        num_c,
        q_to_concept,
        ctx_dim,
        d_model=256,
        n_blocks=4,
        dropout=0.2,
        d_ff=512,
        num_attn_heads=8,
        final_fc_dim=256,
        final_fc_dim2=256,
        max_seq_len=512,
        separate_qa=False,
        use_diffusion=True,
        diffusion_weight=0.08,
        noise_level=0.3,
        use_cl=True,
        cl_weight=0.02,
        se_ratio=16,
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
        self.hidden_dim = int(d_model) * 4
        self.ctx_dim = int(ctx_dim)
        self.fusion_type = str(fusion_type)
        self.register_buffer("q_to_concept", torch.as_tensor(q_to_concept, dtype=torch.long))
        self._training_auxiliary_loss = None

        self.base = KeenKT(
            self.num_c + 1,
            self.num_q,
            d_model=int(d_model),
            n_blocks=int(n_blocks),
            dropout=float(dropout),
            d_ff=int(d_ff),
            num_attn_heads=int(num_attn_heads),
            final_fc_dim=int(final_fc_dim),
            final_fc_dim2=int(final_fc_dim2),
            max_seq_len=int(max_seq_len),
            separate_qa=bool(separate_qa),
            use_diffusion=bool(use_diffusion),
            diffusion_weight=float(diffusion_weight),
            noise_level=float(noise_level),
            use_cl=bool(use_cl),
            cl_weight=float(cl_weight),
            se_ratio=int(se_ratio),
        )
        self.context_fusion = ContextFusion(
            self.hidden_dim,
            self.ctx_dim,
            mode=self.fusion_type,
            dropout=float(dropout),
            ctx_encoder_dim=int(ctx_encoder_dim),
            ctx_group_dims=ctx_group_dims,
            gate_bias_init=float(gate_bias_init),
        )
        self.context_logit_head = ContextLogitHead(
            self.context_fusion.ctx_encoder_dim,
            int(ctx_logit_hidden_dim),
            dropout=float(dropout),
            ctx_logit_mode=str(ctx_logit_mode),
            ctx_logit_init=float(ctx_logit_init),
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
        preds, hidden, auxiliary_loss = self.base(
            concept_data,
            response_data,
            pid_data,
            return_hidden=True,
        )
        self._training_auxiliary_loss = auxiliary_loss
        hidden = hidden[:, 1:]
        if ctx is None:
            return preds[:, 1:]

        ctx_encoded = self.context_fusion.encode_context(ctx)
        ctx_logits = self.context_logit_head(ctx_encoded)
        hidden = self.context_fusion(hidden, ctx, ctx_encoded=ctx_encoded)
        logits = self.base.out(hidden).squeeze(-1)
        logits = self.context_logit_head.apply_to_logits(logits, ctx_logits)
        return torch.sigmoid(logits)

    def get_training_auxiliary_loss(self):
        """Return KeenKT's diffusion and NIG contrastive objectives."""
        return self._training_auxiliary_loss
