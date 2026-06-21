import torch

from torch.nn import Dropout, Embedding, GRU, LayerNorm, Linear, Module, MultiheadAttention

from models.context_fusion import ContextFusion, ContextLogitHead


class TCKTContext(Module):
    def __init__(
        self,
        num_q,
        q_matrix,
        ctx_dim,
        d_model=100,
        num_attn_heads=5,
        dropout=0.2,
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
        self.d_model = int(d_model)
        self.ctx_dim = int(ctx_dim)
        self.max_seq_len = int(max_seq_len)
        self.fusion_type = fusion_type
        q_matrix = torch.as_tensor(q_matrix, dtype=torch.float32)
        if q_matrix.ndim != 2 or q_matrix.shape[0] != self.num_q:
            raise ValueError(f"q_matrix must have shape [num_q, num_c], got {tuple(q_matrix.shape)}")
        self.num_c = int(q_matrix.shape[1])
        self.register_buffer("q_matrix", q_matrix)

        self.exercise_emb = Embedding(self.num_q, self.d_model)
        self.concept_emb = Embedding(self.num_c, self.d_model)
        self.response_emb = Embedding(2, self.d_model)
        self.position_emb = Embedding(self.max_seq_len, self.d_model)
        self.attn = MultiheadAttention(self.d_model, int(num_attn_heads), dropout=float(dropout), batch_first=True)
        self.gru = GRU(self.d_model, self.d_model, batch_first=True)
        self.input_norm = LayerNorm(self.d_model)
        self.output_norm = LayerNorm(self.d_model)
        self.dropout = Dropout(float(dropout))
        self.pred = Linear(self.d_model, 1)

        self.context_fusion = ContextFusion(
            self.d_model,
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

    def _concept_embedding(self, qids):
        qids = qids.long().clamp(min=0, max=self.num_q - 1)
        flat = qids.reshape(-1)
        weights = self.q_matrix[flat]
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        emb = weights @ self.concept_emb.weight / denom
        return emb.view(*qids.shape, self.d_model)

    def forward(self, q, r, qry, ctx=None):
        q = q.long().clamp(min=0, max=self.num_q - 1)
        qry = qry.long().clamp(min=0, max=self.num_q - 1)
        r = r.long().clamp(min=0, max=1)
        batch_size, seq_len = q.shape
        pos = torch.arange(seq_len, device=q.device).clamp(max=self.max_seq_len - 1).unsqueeze(0).expand(batch_size, -1)

        hist = (
            self.exercise_emb(q)
            + self._concept_embedding(q)
            + self.response_emb(r)
            + self.position_emb(pos)
        )
        hist = self.input_norm(hist)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        attn_out, _ = self.attn(hist, hist, hist, attn_mask=causal_mask)
        attn_out = self.dropout(attn_out)
        gru_out = self.gru(self.input_norm(hist + attn_out))[0]

        target = self.exercise_emb(qry) + self._concept_embedding(qry) + self.position_emb(pos)
        hidden = self.output_norm(gru_out + target)
        ctx_logits = None
        if ctx is not None:
            ctx_encoded = self.context_fusion.encode_context(ctx)
            ctx_logits = self.context_logit_head(ctx_encoded)
            hidden = self.context_fusion(hidden, ctx, ctx_encoded=ctx_encoded)

        logits = self.pred(hidden).squeeze(-1)
        logits = self.context_logit_head.apply_to_logits(logits, ctx_logits)
        return torch.sigmoid(logits)
