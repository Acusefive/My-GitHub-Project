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
            fused = gate * hidden + (1.0 - gate) * ctx_hidden
        return self.out_norm(fused)
