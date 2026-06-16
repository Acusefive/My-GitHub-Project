"""KeenKT core adapted for the unified strict-data trainer."""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


def nig_distance_matmul(mean1, cov1, mean2, cov2):
    """Pairwise distance used by KeenKT's distributional attention."""
    mean1 = mean1.float()
    cov1 = cov1.float()
    mean2 = mean2.float()
    cov2 = cov2.float()
    mean_diff = (
        torch.sum(mean1**2, dim=-1, keepdim=True)
        + torch.sum(mean2**2, dim=-1, keepdim=True).transpose(-2, -1)
        - 2 * torch.matmul(mean1, mean2.transpose(-2, -1))
    )
    cov_diff = (
        torch.sum(cov1**2, dim=-1, keepdim=True)
        + torch.sum(cov2**2, dim=-1, keepdim=True).transpose(-2, -1)
        - 2
        * torch.matmul(
            torch.sqrt(torch.clamp(cov1, min=1e-24)),
            torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-2, -1),
        )
    )
    return mean_diff + cov_diff


class NIGNCELoss(nn.Module):
    """KeenKT contrastive objective over pooled distributional states."""

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, mean1, cov1, mean2, cov2):
        cov1 = F.elu(cov1.float()) + 1
        cov2 = F.elu(cov2.float()) + 1
        sim11 = 1 / (1 + nig_distance_matmul(mean1, cov1, mean1, cov1))
        sim22 = 1 / (1 + nig_distance_matmul(mean2, cov2, mean2, cov2))
        sim12 = -1 / (1 + nig_distance_matmul(mean1, cov1, mean2, cov2))
        sim11 = sim11 / self.temperature
        sim22 = sim22 / self.temperature
        sim12 = sim12 / self.temperature
        batch_size = sim12.shape[-1]
        diagonal = torch.arange(batch_size, device=sim12.device)
        sim11[..., diagonal, diagonal] = torch.finfo(sim11.dtype).min
        sim22[..., diagonal, diagonal] = torch.finfo(sim22.dtype).min
        logits = torch.cat(
            [
                torch.cat([sim12, sim11], dim=-1),
                torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1),
            ],
            dim=-2,
        )
        labels = torch.arange(2 * batch_size, device=logits.device)
        return F.cross_entropy(logits, labels)


class DiffusionModule(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        return x + self.net(x)


class SEBlock(nn.Module):
    def __init__(self, hidden_dim, reduction=16):
        super().__init__()
        reduced_dim = max(1, int(hidden_dim) // int(reduction))
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, reduced_dim, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_dim, hidden_dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # The released implementation averages the complete sequence, which
        # leaks future responses into earlier predictions. A cumulative mean
        # preserves channel recalibration while keeping every step causal.
        counts = torch.arange(
            1, x.size(1) + 1, device=x.device, dtype=x.dtype
        ).view(1, -1, 1)
        cumulative_mean = x.cumsum(dim=1) / counts
        return x * self.net(cumulative_mean)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("weight", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        if x.size(1) > self.weight.size(1):
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds KeenKT max_seq_len "
                f"{self.weight.size(1)}"
            )
        return self.weight[:, : x.size(1)].to(dtype=x.dtype)


class KeenKT(nn.Module):
    """Distributional knowledge tracing model from KeenKT."""

    def __init__(
        self,
        n_concept,
        n_pid,
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
    ):
        super().__init__()
        if int(d_model) % int(num_attn_heads) != 0:
            raise ValueError("d_model must be divisible by num_attn_heads")
        self.n_concept = int(n_concept)
        self.n_pid = int(n_pid)
        self.d_model = int(d_model)
        self.separate_qa = bool(separate_qa)
        self.use_diffusion = bool(use_diffusion)
        self.diffusion_weight = float(diffusion_weight)
        self.noise_level = float(noise_level)
        self.use_cl = bool(use_cl)
        self.cl_weight = float(cl_weight)

        self.mu_q_embed = nn.Embedding(self.n_concept, self.d_model)
        self.alpha_q_embed = nn.Embedding(self.n_concept, self.d_model)
        self.beta_q_embed = nn.Embedding(self.n_concept, self.d_model)
        self.delta_q_embed = nn.Embedding(self.n_concept, self.d_model)
        qa_count = 2 * self.n_concept + 1 if self.separate_qa else 2
        self.mu_qa_embed = nn.Embedding(qa_count, self.d_model)
        self.alpha_qa_embed = nn.Embedding(qa_count, self.d_model)
        self.beta_qa_embed = nn.Embedding(qa_count, self.d_model)
        self.delta_qa_embed = nn.Embedding(qa_count, self.d_model)

        self.q_embed_diff = nn.Embedding(self.n_concept, self.d_model)
        self.difficult_param = nn.Embedding(self.n_pid, self.d_model)
        nn.init.constant_(self.difficult_param.weight, 0.0)

        self.architecture = DistributionArchitecture(
            n_blocks=int(n_blocks),
            d_model=self.d_model,
            d_ff=int(d_ff),
            n_heads=int(num_attn_heads),
            dropout=float(dropout),
            max_seq_len=int(max_seq_len),
        )
        self.se_gate = SEBlock(self.d_model, reduction=int(se_ratio))
        self.out = nn.Sequential(
            nn.Linear(self.d_model * 4, int(final_fc_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(final_fc_dim), int(final_fc_dim2)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(final_fc_dim2), 1),
        )
        self.diffusion_module = DiffusionModule(self.d_model)
        self.contrastive_loss = NIGNCELoss(temperature=1.0)

    @staticmethod
    def _nig_parameters(mean_embed, alpha_embed, beta_embed, delta_embed):
        alpha = F.softplus(alpha_embed) + 1e-8
        beta = torch.tanh(beta_embed) * alpha * 0.999
        delta = F.elu(delta_embed) + 1
        gamma = torch.sqrt(torch.clamp(alpha**2 - beta**2, min=1e-8))
        mean = mean_embed + delta * beta / torch.clamp(gamma, min=1e-8)
        sqrt_var = torch.sqrt(delta) * alpha / torch.clamp(gamma, min=1e-8) ** 1.5
        return mean, sqrt_var

    def base_emb(self, concept_data, target):
        q_params = (
            self.mu_q_embed(concept_data),
            self.alpha_q_embed(concept_data),
            self.beta_q_embed(concept_data),
            self.delta_q_embed(concept_data),
        )
        if self.separate_qa:
            qa_data = concept_data + self.n_concept * target
            qa_params = (
                self.mu_qa_embed(qa_data),
                self.alpha_qa_embed(qa_data),
                self.beta_qa_embed(qa_data),
                self.delta_qa_embed(qa_data),
            )
        else:
            qa_params = (
                self.mu_qa_embed(target) + q_params[0],
                self.alpha_qa_embed(target) + q_params[1],
                self.beta_qa_embed(target) + q_params[2],
                self.delta_qa_embed(target) + q_params[3],
            )
        return (*self._nig_parameters(*q_params), *self._nig_parameters(*qa_params))

    def _encode(self, concept_data, target, pid_data):
        q_mean, q_cov, qa_mean, qa_cov = self.base_emb(concept_data, target)
        difficulty = self.difficult_param(pid_data)
        q_diff = self.q_embed_diff(concept_data)
        q_mean = q_mean + difficulty * q_diff
        q_cov = q_cov + difficulty * q_diff
        mean_state, cov_state = self.architecture(q_mean, q_cov, qa_mean, qa_cov)
        return self.se_gate(mean_state), self.se_gate(cov_state), q_mean, q_cov

    def forward(self, concept_data, target, pid_data, return_hidden=False):
        mean_state, cov_state, q_mean, q_cov = self._encode(
            concept_data, target, pid_data
        )
        hidden = torch.cat([mean_state, cov_state, q_mean, q_cov], dim=-1)
        preds = torch.sigmoid(self.out(hidden).squeeze(-1))

        auxiliary_loss = preds.new_zeros(())
        if self.training and self.use_diffusion:
            noisy = mean_state + torch.randn_like(mean_state) * self.noise_level
            denoised = self.diffusion_module(noisy)
            auxiliary_loss = auxiliary_loss + self.diffusion_weight * F.mse_loss(
                denoised.float(), mean_state.float()
            )
        if self.training and self.use_cl:
            # The released KeenKT loader perturbs the first response to form a
            # second uncertainty view. Dropout supplies additional stochasticity.
            target_aug = target.clone()
            target_aug[:, 0] = 1 - target_aug[:, 0]
            mean_aug, cov_aug, _, _ = self._encode(concept_data, target_aug, pid_data)
            auxiliary_loss = auxiliary_loss + self.cl_weight * self.contrastive_loss(
                mean_state.mean(dim=1),
                cov_state.mean(dim=1),
                mean_aug.mean(dim=1),
                cov_aug.mean(dim=1),
            )
        if return_hidden:
            return preds, hidden, auxiliary_loss
        return preds, auxiliary_loss


class DistributionArchitecture(nn.Module):
    def __init__(self, n_blocks, d_model, d_ff, n_heads, dropout, max_seq_len):
        super().__init__()
        self.mean_position = CosinePositionalEmbedding(d_model, max_len=max_seq_len)
        self.cov_position = CosinePositionalEmbedding(d_model, max_len=max_seq_len)
        self.blocks = nn.ModuleList(
            [
                DistributionTransformerLayer(
                    d_model=d_model,
                    d_feature=d_model // n_heads,
                    d_ff=d_ff,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, q_mean, q_cov, qa_mean, qa_cov):
        x_mean = q_mean + self.mean_position(q_mean)
        x_cov = F.elu(q_cov + self.cov_position(q_cov)) + 1
        y_mean = qa_mean + self.mean_position(qa_mean)
        y_cov = F.elu(qa_cov + self.cov_position(qa_cov)) + 1
        for block in self.blocks:
            x_mean, x_cov = block(x_mean, x_cov, x_mean, x_cov, y_mean, y_cov)
        return x_mean, x_cov


class DistributionTransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout):
        super().__init__()
        self.attention = DistributionMultiHeadAttention(
            d_model, d_feature, n_heads, dropout
        )
        self.mean_norm1 = nn.LayerNorm(d_model)
        self.cov_norm1 = nn.LayerNorm(d_model)
        self.mean_norm2 = nn.LayerNorm(d_model)
        self.cov_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mean_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.cov_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )

    def forward(self, query_mean, query_cov, key_mean, key_cov, value_mean, value_cov):
        seq_len = query_mean.size(1)
        mask = torch.tril(
            torch.ones(
                (1, 1, seq_len, seq_len),
                device=query_mean.device,
                dtype=torch.bool,
            ),
            diagonal=-1,
        )
        update_mean, update_cov = self.attention(
            query_mean,
            query_cov,
            key_mean,
            key_cov,
            value_mean,
            value_cov,
            mask,
        )
        query_mean = self.mean_norm1(query_mean + self.dropout1(update_mean))
        query_cov = self.cov_norm1(F.elu(query_cov + self.dropout1(update_cov)) + 1)
        query_mean = self.mean_norm2(query_mean + self.dropout2(self.mean_ffn(query_mean)))
        query_cov = self.cov_norm2(
            F.elu(query_cov + self.dropout2(self.cov_ffn(query_cov))) + 1
        )
        return query_mean, query_cov


class DistributionMultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, bias=True):
        super().__init__()
        self.d_model = int(d_model)
        self.d_k = int(d_feature)
        self.h = int(n_heads)
        self.mean_value = nn.Linear(d_model, d_model, bias=bias)
        self.cov_value = nn.Linear(d_model, d_model, bias=bias)
        self.mean_key = nn.Linear(d_model, d_model, bias=bias)
        self.cov_key = nn.Linear(d_model, d_model, bias=bias)
        self.mean_out = nn.Linear(d_model, d_model, bias=bias)
        self.cov_out = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for layer in (
            self.mean_value,
            self.cov_value,
            self.mean_key,
            self.cov_key,
            self.mean_out,
            self.cov_out,
        ):
            xavier_uniform_(layer.weight)
            if layer.bias is not None:
                constant_(layer.bias, 0.0)

    def _split(self, layer, value):
        return layer(value).view(value.size(0), -1, self.h, self.d_k).transpose(1, 2)

    def forward(self, q_mean, q_cov, k_mean, k_cov, v_mean, v_cov, mask):
        value_mean = self._split(self.mean_value, v_mean)
        value_cov = self._split(self.cov_value, v_cov)
        key_mean = self._split(self.mean_key, k_mean)
        key_cov = self._split(self.cov_key, k_cov)
        query_mean = self._split(self.mean_key, q_mean)
        query_cov = self._split(self.cov_key, q_cov)

        logits = -nig_distance_matmul(
            query_mean, query_cov, key_mean, key_cov
        ) / math.sqrt(self.d_k)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        scores = self.dropout(F.softmax(logits, dim=-1))
        scores = scores.to(dtype=value_mean.dtype)
        zero = torch.zeros(
            scores.size(0),
            scores.size(1),
            1,
            scores.size(-1),
            device=scores.device,
            dtype=scores.dtype,
        )
        scores = torch.cat([zero, scores[:, :, 1:]], dim=2)
        output_mean = torch.matmul(scores, value_mean)
        output_cov = torch.matmul(scores, value_cov)
        output_mean = output_mean.transpose(1, 2).contiguous().view(
            q_mean.size(0), -1, self.d_model
        )
        output_cov = output_cov.transpose(1, 2).contiguous().view(
            q_cov.size(0), -1, self.d_model
        )
        return self.mean_out(output_mean), self.cov_out(output_cov)
