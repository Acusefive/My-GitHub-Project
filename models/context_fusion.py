import torch

from torch.nn import Dropout, GELU, LayerNorm, Linear, Module, ModuleList, Parameter, ReLU, Sequential


class ContextEncoder(Module):
    def __init__(
        self,
        ctx_dim: int,
        encoder_dim: int = 256,
        group_dims: tuple[int, ...] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ctx_dim = int(ctx_dim)
        self.encoder_dim = int(encoder_dim)
        self.group_dims = tuple(int(dim) for dim in (group_dims or ()) if int(dim) > 0)
        if sum(self.group_dims) != self.ctx_dim or len(self.group_dims) < 2:
            self.group_dims = ()

        if self.group_dims:
            group_out_dims = []
            for dim in self.group_dims:
                if dim <= 64:
                    group_out_dims.append(min(64, max(16, self.encoder_dim // 4)))
                else:
                    group_out_dims.append(max(64, self.encoder_dim // 2))
            self.group_out_dims = tuple(group_out_dims)
            self.group_nets = ModuleList(
                [
                    Sequential(
                        LayerNorm(group_dim),
                        Linear(group_dim, group_out_dim),
                        GELU(),
                        Dropout(float(dropout)),
                    )
                    for group_dim, group_out_dim in zip(self.group_dims, self.group_out_dims)
                ]
            )
            concat_dim = int(sum(self.group_out_dims))
            self.out = Sequential(
                LayerNorm(concat_dim),
                Linear(concat_dim, self.encoder_dim),
                GELU(),
                Dropout(float(dropout)),
            )
        else:
            self.group_out_dims = ()
            self.group_nets = ModuleList()
            self.out = Sequential(
                LayerNorm(self.ctx_dim),
                Linear(self.ctx_dim, self.encoder_dim),
                GELU(),
                Dropout(float(dropout)),
            )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        if self.group_dims:
            chunks = torch.split(ctx, self.group_dims, dim=-1)
            encoded = [net(chunk) for net, chunk in zip(self.group_nets, chunks)]
            return self.out(torch.cat(encoded, dim=-1))
        return self.out(ctx)


class ContextFusion(Module):
    def __init__(
        self,
        hidden_dim: int,
        ctx_dim: int,
        mode: str = "gate",
        dropout: float = 0.1,
        ctx_encoder_dim: int = 256,
        ctx_group_dims: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.ctx_dim = int(ctx_dim)
        self.ctx_encoder_dim = int(ctx_encoder_dim)
        self.mode = str(mode)
        self.ctx_encoder = ContextEncoder(
            self.ctx_dim,
            encoder_dim=self.ctx_encoder_dim,
            group_dims=ctx_group_dims,
            dropout=dropout,
        )
        self.ctx_proj = Linear(self.ctx_encoder_dim, self.hidden_dim)
        self.out_norm = LayerNorm(self.hidden_dim)
        if self.mode == "concat":
            self.concat_proj = Linear(self.hidden_dim + self.ctx_encoder_dim, self.hidden_dim)
        elif self.mode in {"gate", "residual_gate"}:
            self.gate_proj = Linear(self.hidden_dim * 2, self.hidden_dim)
            if self.mode == "residual_gate":
                torch.nn.init.constant_(self.gate_proj.bias, -1.0)
        elif self.mode == "add":
            pass
        else:
            raise ValueError(f"Unsupported fusion mode: {self.mode}")
        self.reset_usage_stats()

    def reset_usage_stats(self) -> None:
        self._usage_steps = 0
        self._gate_mean_sum = 0.0
        self._ctx_weight_mean_sum = 0.0
        self._gate_low_frac_sum = 0.0
        self._gate_high_frac_sum = 0.0

    def get_usage_stats(self) -> dict[str, float]:
        if self.mode not in {"gate", "residual_gate"} or self._usage_steps <= 0:
            return {
                "fusion_mode": self.mode,
                "usage_steps": int(self._usage_steps),
                "ctx_encoder_dim": int(self.ctx_encoder_dim),
            }
        denom = float(self._usage_steps)
        return {
            "fusion_mode": self.mode,
            "usage_steps": int(self._usage_steps),
            "ctx_encoder_dim": int(self.ctx_encoder_dim),
            "gate_mean": float(self._gate_mean_sum / denom),
            "ctx_weight_mean": float(self._ctx_weight_mean_sum / denom),
            "gate_lt_0_1_frac": float(self._gate_low_frac_sum / denom),
            "gate_gt_0_9_frac": float(self._gate_high_frac_sum / denom),
        }

    def _record_gate_stats(self, gate: torch.Tensor, ctx_weight: torch.Tensor) -> None:
        with torch.no_grad():
            gate_detached = gate.detach()
            ctx_weight_detached = ctx_weight.detach()
            self._usage_steps += 1
            self._gate_mean_sum += float(gate_detached.mean().item())
            self._ctx_weight_mean_sum += float(ctx_weight_detached.mean().item())
            self._gate_low_frac_sum += float((gate_detached < 0.1).float().mean().item())
            self._gate_high_frac_sum += float((gate_detached > 0.9).float().mean().item())

    def encode_context(self, ctx: torch.Tensor | None) -> torch.Tensor | None:
        if ctx is None:
            return None
        return self.ctx_encoder(ctx)

    def forward(
        self,
        hidden: torch.Tensor,
        ctx: torch.Tensor | None,
        ctx_encoded: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if ctx is None and ctx_encoded is None:
            return hidden
        if ctx_encoded is None:
            ctx_encoded = self.encode_context(ctx)
        if ctx_encoded is None:
            return hidden
        ctx_hidden = self.ctx_proj(ctx_encoded)
        if self.mode == "add":
            fused = hidden + ctx_hidden
        elif self.mode == "concat":
            fused = self.concat_proj(torch.cat([hidden, ctx_encoded], dim=-1))
        elif self.mode == "gate":
            gate = torch.sigmoid(self.gate_proj(torch.cat([hidden, ctx_hidden], dim=-1)))
            self._record_gate_stats(gate, 1.0 - gate)
            fused = gate * hidden + (1.0 - gate) * ctx_hidden
        else:
            gate = torch.sigmoid(self.gate_proj(torch.cat([hidden, ctx_hidden], dim=-1)))
            self._record_gate_stats(gate, gate)
            fused = hidden + gate * ctx_hidden
        return self.out_norm(fused)


class ContextLogitHead(Module):
    def __init__(self, ctx_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ctx_dim = int(ctx_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = Sequential(
            LayerNorm(self.ctx_dim),
            Linear(self.ctx_dim, self.hidden_dim),
            ReLU(),
            Dropout(float(dropout)),
            Linear(self.hidden_dim, 1),
        )
        self.scale = Parameter(torch.tensor(0.1))

    def forward(self, ctx: torch.Tensor | None) -> torch.Tensor | None:
        if ctx is None:
            return None
        return self.scale * self.net(ctx).squeeze(-1)
