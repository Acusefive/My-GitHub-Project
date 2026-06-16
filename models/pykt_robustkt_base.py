"""Core RobustKT architecture adapted from pyKT for the unified trainer."""

import torch
from torch import nn

from models.pykt_akt_base import AKT, TransformerLayer


class RobustKT(AKT):
    def __init__(self, *args, ks=5, **kwargs):
        super().__init__(*args, **kwargs)
        d_model = int(kwargs.get("d_model", args[2] if len(args) > 2 else 100))
        n_blocks = int(kwargs.get("n_blocks", args[3] if len(args) > 3 else 1))
        dropout = float(kwargs.get("dropout", args[4] if len(args) > 4 else 0.2))
        d_ff = int(kwargs.get("d_ff", 256))
        kq_same = int(kwargs.get("kq_same", 1))
        num_attn_heads = int(kwargs.get("num_attn_heads", 5))
        self.model = RobustArchitecture(
            n_blocks=n_blocks,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=num_attn_heads,
            dropout=dropout,
            kq_same=kq_same,
            ks=int(ks),
        )


class RobustArchitecture(nn.Module):
    def __init__(self, n_blocks, d_model, d_ff, n_heads, dropout, kq_same, ks):
        super().__init__()
        self.blocks_1 = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    d_feature=d_model // n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                    kq_same=kq_same,
                    emb_type="qid",
                )
                for _ in range(n_blocks)
            ]
        )
        self.blocks_2 = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    d_feature=d_model // n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                    kq_same=kq_same,
                    emb_type="qid",
                )
                for _ in range(n_blocks * 2)
            ]
        )
        self.smooth = Smooth(dropout, d_model, ks)

    def forward(self, q_embed_data, qa_embed_data, pid_embed_data):
        x = self.smooth(q_embed_data)
        y = self.smooth(qa_embed_data)
        for block in self.blocks_1:
            y = block(mask=1, query=y, key=y, values=y, pdiff=pid_embed_data)
        first = True
        for block in self.blocks_2:
            if first:
                x = block(
                    mask=1,
                    query=x,
                    key=x,
                    values=x,
                    apply_pos=False,
                    pdiff=pid_embed_data,
                )
            else:
                x = block(
                    mask=0,
                    query=x,
                    key=x,
                    values=y,
                    apply_pos=True,
                    pdiff=pid_embed_data,
                )
            first = not first
        return x


class CausalTemporalConv(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.crop = max(0, int(kernel_size) - 1)
        self.conv = nn.Conv1d(channels, channels, int(kernel_size), padding=self.crop)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.crop] if self.crop > 0 else out


class Smooth(nn.Module):
    def __init__(self, dropout, hidden_size, kernel_size):
        super().__init__()
        self.out_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.causal_conv = CausalTemporalConv(hidden_size, kernel_size)
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, input_tensor):
        trend = self.causal_conv(input_tensor.transpose(1, 2)).transpose(1, 2)
        random_component = input_tensor - trend
        denoised = trend + self.sqrt_beta.square() * random_component
        return self.layer_norm(self.out_dropout(denoised) + input_tensor)
