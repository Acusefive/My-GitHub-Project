from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from .constants import DD, DS


class EqBaseMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1024, 512, bias=True)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, DS, bias=True)

    def forward(self, hqtext: torch.Tensor, hqid: torch.Tensor) -> torch.Tensor:
        x = torch.cat([hqtext, hqid], dim=-1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DynamicPriorMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(260, 256, bias=True)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, DD, bias=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc1(z)
        x = self.act(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class StrictPriorModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eqbase = EqBaseMLP()
        self.dynamic = DynamicPriorMLP()
        self.W_diag = nn.Parameter(torch.empty(DS, DD))
        self.b_diag = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.W_diag)

    def diag_logits(self, eq_vec: torch.Tensor, d_vec: torch.Tensor) -> torch.Tensor:
        return torch.sum(eq_vec @ self.W_diag * d_vec, dim=-1) + self.b_diag


@dataclass
class ModelArtifacts:
    model_state_path: str
    training_report_path: str
    implementation_defaults_path: str


def load_strict_prior_model(model_state_path: str, map_location: str = "cpu") -> StrictPriorModel:
    model = StrictPriorModel()
    state = torch.load(model_state_path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model
