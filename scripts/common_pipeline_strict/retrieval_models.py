from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def _resolve_model_source(model_name_or_path: str) -> tuple[str, bool]:
    source = str(model_name_or_path or "").strip()
    if not source:
        raise ValueError("Model source is empty")
    path = Path(source)
    if path.exists():
        return str(path), True
    return source, False


def _last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.sum(dim=1) - 1
    lengths = lengths.clamp(min=0)
    batch_indices = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[batch_indices, lengths]


def _preferred_torch_dtype(device: str) -> Optional[torch.dtype]:
    device_type = torch.device(device).type
    if device_type == "cuda":
        return torch.bfloat16
    return None


class QwenEmbeddingEncoder:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        device: str,
        max_length: int,
        batch_size: int = 16,
    ) -> None:
        source, local_only = _resolve_model_source(model_name_or_path)
        self.model_name_or_path = source
        self.device = device
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        torch_dtype = _preferred_torch_dtype(device)
        self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_only, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            source,
            local_files_only=local_only,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(device).eval()

    def encode_texts(self, texts: Sequence[str], *, instruction: Optional[str] = None) -> np.ndarray:
        all_vectors: List[np.ndarray] = []
        normalized_instruction = str(instruction or "").strip()
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = [str(text or "").strip() for text in texts[start : start + self.batch_size]]
                if normalized_instruction:
                    batch = [f"Instruct: {normalized_instruction}\nQuery: {text}" for text in batch]
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                outputs = self.model(**tokens)
                pooled = _last_token_pool(outputs.last_hidden_state, tokens["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_vectors.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(all_vectors, axis=0) if all_vectors else np.zeros((0, 0), dtype=np.float32)


class QwenReranker:
    SYSTEM_PROMPT = (
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        'Note that the answer can only be "yes" or "no".'
    )

    def __init__(
        self,
        *,
        model_name_or_path: str,
        device: str,
        max_length: int,
        batch_size: int = 8,
    ) -> None:
        source, local_only = _resolve_model_source(model_name_or_path)
        self.model_name_or_path = source
        self.device = device
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        torch_dtype = _preferred_torch_dtype(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            source,
            local_files_only=local_only,
            trust_remote_code=True,
            padding_side="left",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            source,
            local_files_only=local_only,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        if self.token_false_id is None or self.token_true_id is None:
            raise ValueError("Qwen reranker tokenizer does not expose yes/no token ids")

    def _format_prompt(self, *, query: str, doc: str, instruction: str) -> str:
        return (
            "<|im_start|>system\n"
            f"{self.SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n"
            f"Instruct: {instruction}\n"
            f"Query: {query}\n"
            f"Document: {doc}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def score(self, *, query: str, docs: Sequence[str], instruction: str) -> List[float]:
        prompts = [
            self._format_prompt(query=str(query or "").strip(), doc=str(doc or "").strip(), instruction=str(instruction or "").strip())
            for doc in docs
        ]
        scores: List[float] = []
        with torch.inference_mode():
            for start in range(0, len(prompts), self.batch_size):
                batch = prompts[start : start + self.batch_size]
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                outputs = self.model(**tokens, logits_to_keep=1)
                lengths = tokens["attention_mask"].sum(dim=1) - 1
                batch_indices = torch.arange(lengths.shape[0], device=self.device)
                logits = outputs.logits[:, -1, :]
                false_scores = logits[:, self.token_false_id]
                true_scores = logits[:, self.token_true_id]
                pair_scores = torch.stack([false_scores, true_scores], dim=1)
                pair_probs = torch.nn.functional.softmax(pair_scores, dim=1)[:, 1]
                scores.extend(pair_probs.detach().cpu().tolist())
        return [float(score) for score in scores]
