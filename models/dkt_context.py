import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import embedding

from models.context_fusion import ContextFusion, ContextLogitHead


class DKTContext(Module):
    def __init__(
        self,
        num_q,
        emb_size,
        hidden_size,
        ctx_dim,
        dropout=0.1,
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
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.ctx_dim = ctx_dim
        self.fusion_type = fusion_type

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)
        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.context_fusion = ContextFusion(
            self.hidden_size,
            self.ctx_dim,
            mode=self.fusion_type,
            dropout=dropout,
            ctx_encoder_dim=ctx_encoder_dim,
            ctx_group_dims=ctx_group_dims,
            gate_bias_init=gate_bias_init,
        )
        self.context_logit_head = ContextLogitHead(
            self.context_fusion.ctx_encoder_dim,
            ctx_logit_hidden_dim,
            dropout=dropout,
            ctx_logit_mode=ctx_logit_mode,
            ctx_logit_init=ctx_logit_init,
        )
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_q)

    def forward(self, q, r, qry, ctx=None):
        x = q + self.num_q * r
        h, _ = self.lstm_layer(self.interaction_emb(x))

        ctx_logits = None
        if ctx is not None:
            ctx_encoded = self.context_fusion.encode_context(ctx)
            ctx_logits = self.context_logit_head(ctx_encoded)
            h = self.context_fusion(h, ctx, ctx_encoded=ctx_encoded)

        h = self.dropout_layer(h)
        target_weight = embedding(qry.long(), self.out_layer.weight)
        target_bias = embedding(qry.long(), self.out_layer.bias.unsqueeze(-1)).squeeze(-1)
        logits = (h * target_weight).sum(-1) + target_bias
        logits = self.context_logit_head.apply_to_logits(logits, ctx_logits)
        p = torch.sigmoid(logits)
        return p
