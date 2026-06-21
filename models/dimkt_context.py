import torch

from torch.nn import Dropout, Embedding, Linear, Module, Parameter, Sigmoid, Tanh

from models.context_fusion import ContextFusion, ContextLogitHead


class DIMKTContext(Module):
    def __init__(
        self,
        num_q,
        num_c,
        q_to_concept,
        q_difficulty,
        concept_difficulty,
        ctx_dim,
        dropout=0.2,
        emb_size=100,
        difficult_levels=10,
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
        self.difficult_levels = int(difficult_levels)
        self.ctx_dim = int(ctx_dim)
        self.fusion_type = fusion_type
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        self.dropout = Dropout(float(dropout))

        self.register_buffer("q_to_concept", torch.as_tensor(q_to_concept, dtype=torch.long))
        self.register_buffer("q_difficulty", torch.as_tensor(q_difficulty, dtype=torch.long))
        self.register_buffer("concept_difficulty", torch.as_tensor(concept_difficulty, dtype=torch.long))

        self.knowledge = Parameter(torch.empty(1, self.emb_size))
        torch.nn.init.xavier_uniform_(self.knowledge)

        self.q_emb = Embedding(self.num_q + 1, self.emb_size, padding_idx=0)
        self.c_emb = Embedding(self.num_c + 1, self.emb_size, padding_idx=0)
        self.sd_emb = Embedding(self.difficult_levels + 2, self.emb_size, padding_idx=0)
        self.qd_emb = Embedding(self.difficult_levels + 2, self.emb_size, padding_idx=0)
        self.a_emb = Embedding(2, self.emb_size)

        self.linear_1 = Linear(4 * self.emb_size, self.emb_size)
        self.linear_2 = Linear(self.emb_size, self.emb_size)
        self.linear_3 = Linear(self.emb_size, self.emb_size)
        self.linear_4 = Linear(2 * self.emb_size, self.emb_size)
        self.linear_5 = Linear(2 * self.emb_size, self.emb_size)
        self.linear_6 = Linear(4 * self.emb_size, self.emb_size)

        self.context_fusion = ContextFusion(
            self.emb_size,
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

    def _features(self, q, qry):
        q_safe = q.clamp(min=0, max=self.num_q - 1)
        qry_safe = qry.clamp(min=0, max=self.num_q - 1)
        c = self.q_to_concept[q_safe]
        cshft = self.q_to_concept[qry_safe]
        sd = self.concept_difficulty[c]
        sdshft = self.concept_difficulty[cshft]
        qd = self.q_difficulty[q_safe]
        qdshft = self.q_difficulty[qry_safe]
        return q_safe + 1, c, sd, qd, qry_safe + 1, cshft, sdshft, qdshft

    def forward(self, q, r, qry, ctx=None):
        q, c, sd, qd, qshft, cshft, sdshft, qdshft = self._features(q.long(), qry.long())
        q_emb = self.q_emb(q)
        c_emb = self.c_emb(c)
        sd_emb = self.sd_emb(sd)
        qd_emb = self.qd_emb(qd)
        a_emb = self.a_emb(r.long())

        target_q = self.q_emb(qshft)
        target_c = self.c_emb(cshft)
        target_sd = self.sd_emb(sdshft)
        target_qd = self.qd_emb(qdshft)

        input_data = self.linear_1(torch.cat((q_emb, c_emb, sd_emb, qd_emb), dim=-1))
        target_data = self.linear_1(torch.cat((target_q, target_c, target_sd, target_qd), dim=-1))

        batch_size, seq_len, _ = input_data.shape
        zero = torch.zeros(batch_size, 1, self.emb_size, device=q.device, dtype=input_data.dtype)
        sd_steps = torch.cat((zero, sd_emb), dim=1).split(1, dim=1)
        a_steps = torch.cat((zero, a_emb), dim=1).split(1, dim=1)
        qd_steps = torch.cat((zero, qd_emb), dim=1).split(1, dim=1)
        input_steps = torch.cat((zero, input_data), dim=1).split(1, dim=1)

        k = self.knowledge.repeat(batch_size, 1).to(device=q.device, dtype=input_data.dtype)
        hidden_steps = []
        for idx in range(1, seq_len + 1):
            sd_t = sd_steps[idx].squeeze(1)
            a_t = a_steps[idx].squeeze(1)
            qd_t = qd_steps[idx].squeeze(1)
            input_t = input_steps[idx].squeeze(1)

            diff = k - input_t
            gates_sdf = self.sigmoid(self.linear_2(diff))
            sdf_t = self.dropout(self.tanh(self.linear_3(diff)))
            sdf_t = gates_sdf * sdf_t

            pka_input = torch.cat((sdf_t, a_t), dim=-1)
            gates_pka = self.sigmoid(self.linear_4(pka_input))
            pka_t = gates_pka * self.tanh(self.linear_5(pka_input))

            ksu_input = torch.cat((k, a_t, sd_t, qd_t), dim=-1)
            gates_ksu = self.sigmoid(self.linear_6(ksu_input))
            k = gates_ksu * k + (1.0 - gates_ksu) * pka_t
            hidden_steps.append(k.unsqueeze(1))

        hidden = torch.cat(hidden_steps, dim=1)
        ctx_logits = None
        if ctx is not None:
            ctx_encoded = self.context_fusion.encode_context(ctx)
            ctx_logits = self.context_logit_head(ctx_encoded)
            hidden = self.context_fusion(hidden, ctx, ctx_encoded=ctx_encoded)

        logits = torch.sum(target_data * hidden, dim=-1)
        logits = self.context_logit_head.apply_to_logits(logits, ctx_logits)
        return torch.sigmoid(logits)
