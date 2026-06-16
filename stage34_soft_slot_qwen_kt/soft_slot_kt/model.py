from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LabelSpec:
    correct_text: str
    incorrect_text: str
    correct_ids: Tuple[int, ...]
    incorrect_ids: Tuple[int, ...]

    @property
    def is_single_token(self) -> bool:
        return len(self.correct_ids) == 1 and len(self.incorrect_ids) == 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_label_spec(
    tokenizer: Any,
    candidates: Optional[Sequence[Tuple[str, str]]] = None,
) -> LabelSpec:
    candidates = candidates or (("A", "B"), ("正确", "错误"), ("Yes", "No"), ("1", "0"))
    fallback: Optional[LabelSpec] = None
    for correct_text, incorrect_text in candidates:
        correct_ids = tuple(int(x) for x in tokenizer.encode(correct_text, add_special_tokens=False))
        incorrect_ids = tuple(int(x) for x in tokenizer.encode(incorrect_text, add_special_tokens=False))
        if not correct_ids or not incorrect_ids or correct_ids == incorrect_ids:
            continue
        spec = LabelSpec(correct_text, incorrect_text, correct_ids, incorrect_ids)
        if spec.is_single_token:
            return spec
        if fallback is None:
            fallback = spec
    if fallback is None:
        raise ValueError("Could not resolve distinct non-empty classification labels")
    return fallback


class SoftSlotProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, llm_hidden_dim: int, num_soft_tokens: int, dropout: float) -> None:
        super().__init__()
        if input_dim <= 0 or num_soft_tokens <= 0:
            raise ValueError("Projector dimensions and token count must be positive")
        self.input_dim = int(input_dim)
        self.num_soft_tokens = int(num_soft_tokens)
        self.llm_hidden_dim = int(llm_hidden_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_soft_tokens * llm_hidden_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.network(features.to(dtype=self.network[1].weight.dtype))
        return projected.view(features.shape[0], self.num_soft_tokens, self.llm_hidden_dim)


class SoftSlotQwenKT(nn.Module):
    """Frozen causal LLM with trainable external-embedding projectors."""

    def __init__(
        self,
        llm: nn.Module,
        *,
        context_dim: int,
        target_dim: int,
        context_soft_tokens: int,
        target_soft_tokens: int,
        projector_hidden_dim: int = 512,
        projector_dropout: float = 0.1,
        random_slots: bool = False,
        random_seed: int = 42,
        llm_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.llm = llm
        for parameter in self.llm.parameters():
            parameter.requires_grad = False

        llm_hidden_dim = int(getattr(self.llm.config, "hidden_size"))
        self.context_soft_tokens = int(context_soft_tokens)
        self.target_soft_tokens = int(target_soft_tokens)
        self.random_slots = bool(random_slots)
        self.llm_gradient_checkpointing = bool(llm_gradient_checkpointing)
        if not self.random_slots and self.context_soft_tokens > 0 and context_dim <= 0:
            raise ValueError("Context soft tokens require non-empty context features")
        if not self.random_slots and self.target_soft_tokens > 0 and target_dim <= 0:
            raise ValueError("Target soft tokens require non-empty target features")
        self.context_projector = (
            SoftSlotProjector(context_dim, projector_hidden_dim, llm_hidden_dim, context_soft_tokens, projector_dropout)
            if context_dim > 0 and context_soft_tokens > 0 and not random_slots
            else None
        )
        self.target_projector = (
            SoftSlotProjector(target_dim, projector_hidden_dim, llm_hidden_dim, target_soft_tokens, projector_dropout)
            if target_dim > 0 and target_soft_tokens > 0 and not random_slots
            else None
        )

        generator = torch.Generator(device="cpu").manual_seed(int(random_seed))
        context_random = torch.randn((context_soft_tokens, llm_hidden_dim), generator=generator) * 0.02
        target_random = torch.randn((target_soft_tokens, llm_hidden_dim), generator=generator) * 0.02
        self.register_buffer("random_context_slots", context_random, persistent=True)
        self.register_buffer("random_target_slots", target_random, persistent=True)

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def train(self, mode: bool = True) -> "SoftSlotQwenKT":
        super().train(mode)
        if mode and self.llm_gradient_checkpointing:
            # Hugging Face decoder layers activate gradient checkpointing only in training mode.
            self.llm.train()
        else:
            self.llm.eval()
        return self

    def _replace_slots(
        self,
        inputs_embeds: torch.Tensor,
        mask: torch.Tensor,
        values: Optional[torch.Tensor],
        *,
        expected_tokens: int,
        random_values: torch.Tensor,
    ) -> torch.Tensor:
        if expected_tokens <= 0:
            return inputs_embeds
        counts = mask.sum(dim=1)
        if not bool(torch.all(counts == expected_tokens)):
            raise ValueError(f"Expected {expected_tokens} soft-slot positions per sample, got {counts.tolist()}")
        output = inputs_embeds
        for row in range(output.shape[0]):
            positions = torch.nonzero(mask[row], as_tuple=False).squeeze(-1)
            replacement = (
                random_values.to(device=output.device, dtype=output.dtype)
                if self.random_slots
                else values[row].to(device=output.device, dtype=output.dtype)
            )
            output[row, positions] = replacement
        return output

    def build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
        context_features: Optional[torch.Tensor],
        target_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        embedding_layer = self.llm.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids).clone()
        context_values = self.context_projector(context_features) if self.context_projector is not None else None
        target_values = self.target_projector(target_features) if self.target_projector is not None else None
        inputs_embeds = self._replace_slots(
            inputs_embeds,
            context_mask,
            context_values,
            expected_tokens=self.context_soft_tokens,
            random_values=self.random_context_slots,
        )
        inputs_embeds = self._replace_slots(
            inputs_embeds,
            target_mask,
            target_values,
            expected_tokens=self.target_soft_tokens,
            random_values=self.random_target_slots,
        )
        return inputs_embeds

    def _candidate_sequence_score(
        self,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        token_ids: Sequence[int],
    ) -> torch.Tensor:
        if not token_ids:
            raise ValueError("Classification label cannot be empty")
        batch_size = prompt_embeds.shape[0]
        device = prompt_embeds.device
        if len(token_ids) > 1:
            previous_ids = torch.tensor(token_ids[:-1], dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
            previous_embeds = self.llm.get_input_embeddings()(previous_ids)
            inputs_embeds = torch.cat([prompt_embeds, previous_embeds], dim=1)
            attention_mask = torch.cat(
                [
                    prompt_attention_mask,
                    torch.ones((batch_size, len(token_ids) - 1), dtype=prompt_attention_mask.dtype, device=device),
                ],
                dim=1,
            )
        else:
            inputs_embeds = prompt_embeds
            attention_mask = prompt_attention_mask
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
        start_position = prompt_embeds.shape[1] - 1
        score = torch.zeros((batch_size,), dtype=outputs.logits.dtype, device=device)
        for offset, token_id in enumerate(token_ids):
            token_log_probs = F.log_softmax(outputs.logits[:, start_position + offset, :].float(), dim=-1)
            score = score + token_log_probs[:, int(token_id)]
        return score / float(len(token_ids))

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
        context_features: Optional[torch.Tensor],
        target_features: Optional[torch.Tensor],
        label_spec: LabelSpec,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        prompt_embeds = self.build_inputs_embeds(
            input_ids,
            context_mask,
            target_mask,
            context_features,
            target_features,
        )
        if label_spec.is_single_token:
            outputs = self.llm(inputs_embeds=prompt_embeds, attention_mask=attention_mask, use_cache=False)
            next_token_logits = outputs.logits[:, -1, :].float()
            correct_score = next_token_logits[:, int(label_spec.correct_ids[0])]
            incorrect_score = next_token_logits[:, int(label_spec.incorrect_ids[0])]
        else:
            correct_score = self._candidate_sequence_score(prompt_embeds, attention_mask, label_spec.correct_ids)
            incorrect_score = self._candidate_sequence_score(prompt_embeds, attention_mask, label_spec.incorrect_ids)
        class_logits = torch.stack([incorrect_score, correct_score], dim=-1)
        probabilities = torch.softmax(class_logits.float(), dim=-1)[:, 1]
        result = {"class_logits": class_logits, "probabilities": probabilities}
        if labels is not None:
            result["loss"] = F.cross_entropy(class_logits.float(), labels.long())
        return result
