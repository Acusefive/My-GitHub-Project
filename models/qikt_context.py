import torch

from torch.nn import Dropout, Embedding, LSTM, Linear, Module, ReLU, Sequential

from models.context_fusion import ContextFusion, ContextLogitHead


class QIKTContext(Module):
    def __init__(
        self,
        num_q,
        num_c,
        q_to_concept,
        ctx_dim,
        emb_size=100,
        dropout=0.2,
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
        self.emb_size = int(emb_size)
        self.hidden_dim = self.emb_size * 3
        self.ctx_dim = int(ctx_dim)
        self.fusion_type = fusion_type
        self.register_buffer("q_to_concept", torch.as_tensor(q_to_concept, dtype=torch.long))

        self.q_emb = Embedding(self.num_q, self.emb_size)
        self.c_emb = Embedding(self.num_c + 1, self.emb_size, padding_idx=0)
        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)
        self.qc_proj = Sequential(
            Linear(self.emb_size * 2, self.emb_size),
            ReLU(),
            Dropout(float(dropout)),
        )
        self.question_lstm = LSTM(self.emb_size * 2, self.emb_size, batch_first=True)
        self.concept_lstm = LSTM(self.emb_size * 2, self.emb_size, batch_first=True)
        self.dropout = Dropout(float(dropout))
        self.pred = Sequential(
            Linear(self.hidden_dim, self.emb_size),
            ReLU(),
            Dropout(float(dropout)),
            Linear(self.emb_size, 1),
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

    def _concepts(self, qids):
        qids = qids.clamp(min=0, max=self.num_q - 1)
        return self.q_to_concept[qids]

    def _qc_embedding(self, qids):
        qids = qids.clamp(min=0, max=self.num_q - 1)
        cids = self._concepts(qids)
        return self.qc_proj(torch.cat([self.q_emb(qids), self.c_emb(cids)], dim=-1))

    def forward(self, q, r, qry, ctx=None):
        q = q.long().clamp(min=0, max=self.num_q - 1)
        qry = qry.long().clamp(min=0, max=self.num_q - 1)
        r = r.long().clamp(min=0, max=1)
        c_emb = self.c_emb(self._concepts(q))
        qc_emb = self._qc_embedding(q)
        target_qc = self._qc_embedding(qry)
        interaction = self.interaction_emb(q + self.num_q * r)

        question_input = torch.cat([qc_emb, interaction], dim=-1)
        question_hidden = self.dropout(self.question_lstm(question_input)[0])

        r_float = r.unsqueeze(-1).to(dtype=c_emb.dtype)
        concept_input = torch.cat([c_emb * (1.0 - r_float), c_emb * r_float], dim=-1)
        concept_hidden = self.dropout(self.concept_lstm(concept_input)[0])

        hidden = torch.cat([target_qc, question_hidden, concept_hidden], dim=-1)
        ctx_logits = None
        if ctx is not None:
            ctx_encoded = self.context_fusion.encode_context(ctx)
            ctx_logits = self.context_logit_head(ctx_encoded)
            hidden = self.context_fusion(hidden, ctx, ctx_encoded=ctx_encoded)

        logits = self.pred(hidden).squeeze(-1)
        logits = self.context_logit_head.apply_to_logits(logits, ctx_logits)
        return torch.sigmoid(logits)
