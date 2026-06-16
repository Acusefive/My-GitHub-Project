from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch


PROMPT_VERSION = "compact_state_target_match_v1"


@dataclass
class EncodedPrompt:
    input_ids: List[int]
    context_positions: List[int]
    target_positions: List[int]


def compact_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def build_prompt_segments(sample: Dict[str, Any], problem: Dict[str, Any], *, include_context_text: bool) -> Dict[str, str]:
    concepts = "、".join(str(item) for item in problem.get("concepts") or []) or "未知"
    context_text = compact_text(sample.get("context_text"), int(sample.get("max_context_chars") or 2400))
    prefix = (
        "知识追踪任务：结合学生当前认知状态与目标题要求，判断该学生本次作答更可能正确还是错误。\n"
        "学生状态表示："
    )
    if include_context_text:
        between = "\n历史证据：" + (context_text or "无可用历史证据")
    else:
        between = ""
    between += (
        "\n目标题："
        f"知识点={concepts}；认知层级={problem.get('cognitive_dimension', 0)}；"
        f"题干={compact_text(problem.get('text') or problem.get('title') or sample.get('target_pid'), 1200)}\n"
        "目标题表示："
    )
    suffix = "\n输出标签：A=正确，B=错误。仅输出一个标签。\n标签："
    return {"prefix": prefix, "between": between, "suffix": suffix}


def encode_prompt(
    tokenizer: Any,
    sample: Dict[str, Any],
    problem: Dict[str, Any],
    *,
    context_soft_tokens: int,
    target_soft_tokens: int,
    include_context_text: bool,
    placeholder_token_id: int,
) -> EncodedPrompt:
    segments = build_prompt_segments(sample, problem, include_context_text=include_context_text)

    def encode(text: str) -> List[int]:
        return list(tokenizer.encode(text, add_special_tokens=False))

    input_ids = encode(segments["prefix"])
    context_positions = list(range(len(input_ids), len(input_ids) + context_soft_tokens))
    input_ids.extend([placeholder_token_id] * context_soft_tokens)
    input_ids.extend(encode(segments["between"]))
    target_positions = list(range(len(input_ids), len(input_ids) + target_soft_tokens))
    input_ids.extend([placeholder_token_id] * target_soft_tokens)
    input_ids.extend(encode(segments["suffix"]))
    if not input_ids:
        raise ValueError("Encoded prompt is empty")
    return EncodedPrompt(input_ids=input_ids, context_positions=context_positions, target_positions=target_positions)


def left_pad_batch(
    encoded: Sequence[EncodedPrompt],
    *,
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    max_len = max(len(item.input_ids) for item in encoded)
    batch_size = len(encoded)
    input_ids = torch.full((batch_size, max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    context_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    target_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for row, item in enumerate(encoded):
        offset = max_len - len(item.input_ids)
        input_ids[row, offset:] = torch.tensor(item.input_ids, dtype=torch.long)
        attention_mask[row, offset:] = 1
        if item.context_positions:
            context_mask[row, torch.tensor([offset + pos for pos in item.context_positions], dtype=torch.long)] = True
        if item.target_positions:
            target_mask[row, torch.tensor([offset + pos for pos in item.target_positions], dtype=torch.long)] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "context_mask": context_mask,
        "target_mask": target_mask,
    }
