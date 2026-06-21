import torch
from torch.nn import Module

from models.context_fusion import ContextFusion, ContextLogitHead
from models.pykt_simplekt_base import SimpleKT


class SimpleKTContext(Module):
    def __init__(
        self,
        num_q,
        num_c,
        q_to_concept,
        ctx_dim,
        d_model=100,
        n_blocks=1,
        dropout=0.2,
        d_ff=256,
        kq_same=1,
        final_fc_dim=512,
        final_fc_dim2=256,
        num_attn_heads=5,
        separate_qa=False,
        l2=1e-5,
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
        self.num_c = int(num_c)
        self.hidden_dim = int(d_model) * 2
        self.ctx_dim = int(ctx_dim)
        self.fusion_type = fusion_type
        self.register_buffer("q_to_concept", torch.as_tensor(q_to_concept, dtype=torch.long))

        # Concept IDs are one-based because zero is reserved for missing/padded
        # values in the shared strict-data metadata.
        self.base = SimpleKT(
            self.num_c + 1,
            self.num_q,
            d_model=int(d_model),
            n_blocks=int(n_blocks),
            dropout=float(dropout),
            d_ff=int(d_ff),
            kq_same=int(kq_same),
            final_fc_dim=int(final_fc_dim),
            final_fc_dim2=int(final_fc_dim2),
            num_attn_heads=int(num_attn_heads),
            separate_qa=bool(separate_qa),
            l2=float(l2),
            max_seq_len=int(max_seq_len),
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

        # Reconstruct the full sequence expected by pyKT SimpleKT. Outputs at
        # positions 1..T align with qry and never attend to their own response.
        pid_data = torch.cat([q[:, :1], qry], dim=1)
        concept_data = self.q_to_concept[pid_data]
        response_data = torch.cat([r, r[:, -1:]], dim=1)
        preds, hidden = self.base(
            concept_data,
            response_data,
            pid_data,
            return_hidden=True,
        )
        hidden = hidden[:, 1:, :]

        if ctx is None:
            return preds[:, 1:]

        ctx_encoded = self.context_fusion.encode_context(ctx)
        ctx_logits = self.context_logit_head(ctx_encoded)
        hidden = self.context_fusion(hidden, ctx, ctx_encoded=ctx_encoded)
        logits = self.base.out(hidden).squeeze(-1)
        logits = self.context_logit_head.apply_to_logits(logits, ctx_logits)
        return torch.sigmoid(logits)
