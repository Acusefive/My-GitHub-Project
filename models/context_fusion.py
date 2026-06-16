"""Context 编码、隐藏状态融合与预测 logit 修正模块。

基础知识追踪模型负责从答题历史生成隐藏状态；本文件负责把 Stage 3.4
生成的认知上下文接入基础模型。上下文有两条作用路径：
1. ContextFusion：把上下文表示融合进基础模型隐藏状态；
2. ContextLogitHead：直接从上下文产生一个附加预测 logit。
"""

import torch

from torch.nn import Dropout, GELU, LayerNorm, Linear, Module, ModuleList, Parameter, ReLU, Sequential, Sigmoid


class ContextEncoder(Module):
    """把一种或多种原始 Context 特征编码到统一的低维表示空间。

    当 ``group_dims`` 有效时，各组特征会先独立编码，再拼接并投影。
    这样可以避免文本向量、结构化向量和数值特征因维度差异互相淹没。
    """

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
        """编码形状为 ``[..., ctx_dim]`` 的上下文张量。"""
        if self.group_dims:
            chunks = torch.split(ctx, self.group_dims, dim=-1)
            encoded = [net(chunk) for net, chunk in zip(self.group_nets, chunks)]
            return self.out(torch.cat(encoded, dim=-1))
        return self.out(ctx)


class ContextFusion(Module):
    """将编码后的 Context 融合进基础知识追踪模型的隐藏状态。

    支持四种模式：
    - ``add``：直接相加；
    - ``concat``：拼接后线性投影；
    - ``gate``：在基础状态和 Context 状态之间插值；
    - ``residual_gate``：保留基础状态，再叠加门控 Context 残差。
    """

    def __init__(
        self,
        hidden_dim: int,
        ctx_dim: int,
        mode: str = "gate",
        dropout: float = 0.1,
        ctx_encoder_dim: int = 256,
        ctx_group_dims: tuple[int, ...] | None = None,
        gate_bias_init: float = -2.0,
        fusion_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.ctx_dim = int(ctx_dim)
        self.ctx_encoder_dim = int(ctx_encoder_dim)
        if fusion_mode is not None:
            if mode != "gate" and str(mode) != str(fusion_mode):
                raise ValueError(f"Conflicting fusion modes: mode={mode}, fusion_mode={fusion_mode}")
            mode = str(fusion_mode)
        self.mode = str(mode)
        self.gate_bias_init = float(gate_bias_init)
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
        elif self.mode == "gate":
            self.gate_proj = Linear(self.hidden_dim * 2, self.hidden_dim)
        elif self.mode == "residual_gate":
            self.gate_net = Sequential(
                Linear(self.hidden_dim * 2, self.hidden_dim),
                ReLU(),
                Dropout(float(dropout)),
                Linear(self.hidden_dim, self.hidden_dim),
                Sigmoid(),
            )
            final_gate_linear = self.gate_net[-2]
            torch.nn.init.constant_(final_gate_linear.bias, self.gate_bias_init)
        elif self.mode == "add":
            pass
        else:
            raise ValueError(f"Unsupported fusion mode: {self.mode}")
        self.reset_usage_stats()

    def reset_usage_stats(self) -> None:
        """清空一个训练或评估阶段内累计的门控使用统计。"""
        self._usage_steps = 0
        self._gate_mean_sum = 0.0
        self._ctx_weight_mean_sum = 0.0
        self._gate_low_frac_sum = 0.0
        self._gate_high_frac_sum = 0.0

    def get_usage_stats(self) -> dict[str, float]:
        """返回门控均值、Context 权重等诊断指标。"""
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
        """记录门控统计，但不让统计计算进入反向传播图。"""
        with torch.no_grad():
            gate_detached = gate.detach()
            ctx_weight_detached = ctx_weight.detach()
            self._usage_steps += 1
            self._gate_mean_sum += float(gate_detached.mean().item())
            self._ctx_weight_mean_sum += float(ctx_weight_detached.mean().item())
            self._gate_low_frac_sum += float((gate_detached < 0.1).float().mean().item())
            self._gate_high_frac_sum += float((gate_detached > 0.9).float().mean().item())

    def encode_context(self, ctx: torch.Tensor | None) -> torch.Tensor | None:
        """统一处理无 Context 场景，便于各基础模型复用。"""
        if ctx is None:
            return None
        return self.ctx_encoder(ctx)

    def forward(
        self,
        hidden: torch.Tensor,
        ctx: torch.Tensor | None,
        ctx_encoded: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """把 Context 融合到与其时间位置对齐的基础隐藏状态。"""
        if ctx is None and ctx_encoded is None:
            return hidden
        if ctx_encoded is None:
            ctx_encoded = self.encode_context(ctx)
        if ctx_encoded is None:
            return hidden
        ctx_hidden = self.ctx_proj(ctx_encoded)
        if hidden.shape[-1] != self.hidden_dim:
            raise ValueError(f"hidden last dim must be {self.hidden_dim}, got {hidden.shape[-1]}")
        if hidden.shape[:-1] != ctx_hidden.shape[:-1]:
            raise ValueError(f"hidden and context leading dims must match, got {hidden.shape} and {ctx_hidden.shape}")
        # gate 模式在基础状态和 Context 状态之间插值，因此 gate 越大越依赖基础模型；
        # residual_gate 始终保留基础状态，gate 表示额外加入多少 Context 修正量。
        if self.mode == "add":
            fused = hidden + ctx_hidden
        elif self.mode == "concat":
            fused = self.concat_proj(torch.cat([hidden, ctx_encoded], dim=-1))
        elif self.mode == "gate":
            gate = torch.sigmoid(self.gate_proj(torch.cat([hidden, ctx_hidden], dim=-1)))
            self._record_gate_stats(gate, 1.0 - gate)
            fused = gate * hidden + (1.0 - gate) * ctx_hidden
            return self.out_norm(fused)
        else:
            gate = self.gate_net(torch.cat([hidden, ctx_hidden], dim=-1))
            self._record_gate_stats(gate, gate)
            fused = hidden + gate * ctx_hidden
            return fused
        return self.out_norm(fused)


class ContextLogitHead(Module):
    """根据 Context 单独预测一个 logit，并将其作为基础预测的修正项。

    这条路径允许 Context 直接影响最终正确率，而不必完全依赖隐藏状态融合。
    """

    def __init__(
        self,
        ctx_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        ctx_logit_mode: str = "scaled",
        ctx_logit_init: float = -3.0,
    ) -> None:
        super().__init__()
        self.ctx_dim = int(ctx_dim)
        self.hidden_dim = int(hidden_dim)
        self.ctx_logit_mode = str(ctx_logit_mode)
        if self.ctx_logit_mode not in {"none", "raw", "scaled"}:
            raise ValueError(f"Unsupported ctx_logit_mode: {self.ctx_logit_mode}")
        self.net = Sequential(
            LayerNorm(self.ctx_dim),
            Linear(self.ctx_dim, self.hidden_dim),
            ReLU(),
            Dropout(float(dropout)),
            Linear(self.hidden_dim, 1),
        )
        self.scale = Parameter(torch.tensor(float(ctx_logit_init)))

    @property
    def ctx_logit_scale(self) -> Parameter:
        return self.scale

    def effective_scale(self) -> torch.Tensor:
        """返回 Context logit 实际使用的缩放系数。"""
        if self.ctx_logit_mode == "none":
            return self.scale.new_tensor(0.0)
        if self.ctx_logit_mode == "raw":
            return self.scale.new_tensor(1.0)
        # scaled 模式保存无约束参数，但通过 sigmoid 将实际贡献限制在 [0, 1]；
        # 因而可以用较小的初始值让模型先主要依赖基础预测。
        return torch.sigmoid(self.ctx_logit_scale)

    def get_usage_stats(self) -> dict[str, float | str]:
        """返回 Context logit 模式、实际缩放值和原始参数值。"""
        scale = self.effective_scale()
        return {
            "ctx_logit_mode": self.ctx_logit_mode,
            "ctx_logit_scale": float(scale.detach().cpu().item()),
            "ctx_logit_scale_logit": float(self.ctx_logit_scale.detach().cpu().item()),
        }

    def forward(self, ctx: torch.Tensor | None) -> torch.Tensor | None:
        """从已编码的 Context 生成逐时间步的附加 logit。"""
        if ctx is None or self.ctx_logit_mode == "none":
            return None
        return self.net(ctx).squeeze(-1)

    def apply_to_logits(self, logits: torch.Tensor, ctx_logits: torch.Tensor | None) -> torch.Tensor:
        """按配置把 Context logit 加到基础模型 logit 上。"""
        if ctx_logits is None or self.ctx_logit_mode == "none":
            return logits
        if self.ctx_logit_mode == "raw":
            return logits + ctx_logits
        return logits + self.effective_scale() * ctx_logits
