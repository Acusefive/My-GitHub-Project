import torch

from torch.nn import LayerNorm, Linear, Module


class ContextFusion(Module):
    def __init__(self, hidden_dim: int, ctx_dim: int, mode: str = "gate") -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.ctx_dim = int(ctx_dim)
        self.mode = str(mode)
        self.ctx_proj = Linear(self.ctx_dim, self.hidden_dim)
        self.out_norm = LayerNorm(self.hidden_dim)
        if self.mode == "concat":
            self.concat_proj = Linear(self.hidden_dim + self.ctx_dim, self.hidden_dim)
        elif self.mode == "gate":
            self.gate_proj = Linear(self.hidden_dim * 2, self.hidden_dim)
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
        if self.mode != "gate" or self._usage_steps <= 0:
            return {
                "fusion_mode": self.mode,
                "usage_steps": int(self._usage_steps),
            }
        denom = float(self._usage_steps)
        return {
            "fusion_mode": self.mode,
            "usage_steps": int(self._usage_steps),
            "gate_mean": float(self._gate_mean_sum / denom),
            "ctx_weight_mean": float(self._ctx_weight_mean_sum / denom),
            "gate_lt_0_1_frac": float(self._gate_low_frac_sum / denom),
            "gate_gt_0_9_frac": float(self._gate_high_frac_sum / denom),
        }

    def forward(self, hidden: torch.Tensor, ctx: torch.Tensor | None) -> torch.Tensor:
        if ctx is None:
            return hidden
        ctx_hidden = self.ctx_proj(ctx)
        if self.mode == "add":
            fused = hidden + ctx_hidden
        elif self.mode == "concat":
            fused = self.concat_proj(torch.cat([hidden, ctx], dim=-1))
        else:
            gate = torch.sigmoid(self.gate_proj(torch.cat([hidden, ctx_hidden], dim=-1)))
            with torch.no_grad():
                gate_detached = gate.detach()
                self._usage_steps += 1
                self._gate_mean_sum += float(gate_detached.mean().item())
                self._ctx_weight_mean_sum += float((1.0 - gate_detached).mean().item())
                self._gate_low_frac_sum += float((gate_detached < 0.1).float().mean().item())
                self._gate_high_frac_sum += float((gate_detached > 0.9).float().mean().item())
            fused = gate * hidden + (1.0 - gate) * ctx_hidden
        return self.out_norm(fused)
