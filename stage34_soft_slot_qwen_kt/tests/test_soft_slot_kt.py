from __future__ import annotations

import json
import os
import pickle
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from soft_slot_kt.data import SoftSlotCollator, SoftSlotDataset, load_problem_catalog
from soft_slot_kt.model import SoftSlotQwenKT, resolve_label_spec
from soft_slot_kt.prepare import prepare_existing_stage34_features
from soft_slot_kt.prompts import PROMPT_VERSION, build_prompt_segments
from soft_slot_kt.runtime import load_checkpoint, save_checkpoint


class MockTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self, *, multi_token_labels: bool = False) -> None:
        self.multi_token_labels = multi_token_labels

    def encode(self, text: str, add_special_tokens: bool = False):
        if text == "A":
            return [3, 4] if self.multi_token_labels else [3]
        if text == "B":
            return [5, 6] if self.multi_token_labels else [5]
        return [7 + (ord(char) % 40) for char in str(text)]


class MockCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 12) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds, attention_mask, use_cache=False):
        return SimpleNamespace(logits=self.head(inputs_embeds))


class SoftSlotModelTest(unittest.TestCase):
    def _batch(self):
        return {
            "input_ids": torch.tensor([[1, 0, 2, 0, 8], [1, 0, 2, 0, 9]], dtype=torch.long),
            "attention_mask": torch.ones((2, 5), dtype=torch.long),
            "context_mask": torch.tensor([[False, True, False, False, False]] * 2),
            "target_mask": torch.tensor([[False, False, False, True, False]] * 2),
            "context_features": torch.randn((2, 4)),
            "target_features": torch.randn((2, 5)),
            "labels": torch.tensor([0, 1]),
        }

    def test_frozen_llm_projector_receives_gradient(self):
        tokenizer = MockTokenizer()
        llm = MockCausalLM()
        model = SoftSlotQwenKT(
            llm,
            context_dim=4,
            target_dim=5,
            context_soft_tokens=1,
            target_soft_tokens=1,
            projector_hidden_dim=8,
        )
        batch = self._batch()
        result = model(**batch, label_spec=resolve_label_spec(tokenizer))
        result["loss"].backward()
        self.assertTrue(all(parameter.grad is None for parameter in llm.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in model.trainable_parameters()))
        self.assertEqual(tuple(result["probabilities"].shape), (2,))

    def test_multi_token_label_fallback(self):
        tokenizer = MockTokenizer(multi_token_labels=True)
        spec = resolve_label_spec(tokenizer, candidates=(("A", "B"),))
        self.assertFalse(spec.is_single_token)
        model = SoftSlotQwenKT(
            MockCausalLM(),
            context_dim=4,
            target_dim=5,
            context_soft_tokens=1,
            target_soft_tokens=1,
            projector_hidden_dim=8,
        )
        result = model(**self._batch(), label_spec=spec)
        self.assertTrue(torch.isfinite(result["loss"]))

    def test_slots_only_prompt_omits_raw_history(self):
        sample = {"context_text": "RAW_HISTORY_MUST_NOT_APPEAR", "target_pid": "p1"}
        problem = {"problem_id": "p1", "text": "target", "concepts": ["c1"], "cognitive_dimension": 1}
        slots_only = build_prompt_segments(sample, problem, include_context_text=False)
        full_history = build_prompt_segments(sample, problem, include_context_text=True)
        self.assertNotIn("RAW_HISTORY_MUST_NOT_APPEAR", "".join(slots_only.values()))
        self.assertIn("RAW_HISTORY_MUST_NOT_APPEAR", "".join(full_history.values()))
        self.assertIn("更可能正确还是错误", slots_only["prefix"])
        self.assertEqual("compact_state_target_match_v1", PROMPT_VERSION)

    def test_checkpoint_roundtrip_normalizes_path_args(self):
        root = PROJECT_ROOT / "artifacts" / "test_runs" / f"checkpoint_test_{os.getpid()}"
        checkpoint_path = root / "checkpoint.pt"
        model = SoftSlotQwenKT(
            MockCausalLM(),
            context_dim=4,
            target_dim=5,
            context_soft_tokens=1,
            target_soft_tokens=1,
            projector_hidden_dim=8,
        )
        optimizer = torch.optim.AdamW(model.trainable_parameters())
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=None,
            epoch=3,
            global_step=7,
            args=SimpleNamespace(output_dir=checkpoint_path.parent, seed=42),
            label_spec=resolve_label_spec(MockTokenizer()),
        )
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.assertIsInstance(raw["args"]["output_dir"], str)
        loaded = load_checkpoint(checkpoint_path, model=model, optimizer=optimizer)
        self.assertEqual(3, loaded["epoch"])

        legacy_path = root / "legacy_checkpoint.pt"
        raw["args"]["output_dir"] = checkpoint_path.parent
        torch.save(raw, legacy_path)
        legacy_loaded = load_checkpoint(legacy_path, model=model, optimizer=optimizer)
        self.assertEqual(3, legacy_loaded["epoch"])


class PrepareAndDatasetTest(unittest.TestCase):
    def test_prepare_existing_features_and_load_dataset(self):
        root = PROJECT_ROOT / "artifacts" / "test_runs" / f"unit_test_{os.getpid()}"
        root.mkdir(parents=True, exist_ok=True)
        with self.subTest("workspace fixture"):
            priors = root / "priors"
            priors.mkdir(exist_ok=True)
            contexts_path = root / "contexts.jsonl"
            student_path = root / "student.json"
            embedding_path = root / "context_embeddings.pkl"
            output_dir = root / "features"

            catalog = [
                {"problem_id": "p1", "text": "one", "cognitive_dimension": 1, "concepts": ["c1"]},
                {"problem_id": "p2", "text": "two", "cognitive_dimension": 2, "concepts": ["c2"]},
                {"problem_id": "p3", "text": "three", "cognitive_dimension": 3, "concepts": ["c3"]},
            ]
            (priors / "problem_catalog.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in catalog),
                encoding="utf-8",
            )
            student_path.write_text(
                json.dumps(
                    [
                        {
                            "user_id": "u1",
                            "seq": [
                                {"problem_id": "p1", "is_correct": 1, "submit_time": "2020-01-01 00:00:01"},
                                {"problem_id": "p2", "is_correct": 0, "submit_time": "2020-01-01 00:00:02"},
                                {"problem_id": "p3", "is_correct": 1, "submit_time": "2020-01-01 00:00:03"},
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            contexts = [
                {
                    "user_id": "u1",
                    "target_t": 1,
                    "target_pid": "p2",
                    "llm_context_text": "history one",
                    "stage1_candidate_count": 1,
                    "selected_count": 1,
                    "summary_fields": {"sdyn": 0.4},
                    "evidence_list": [{"history_pos": 0}],
                },
                {
                    "user_id": "u1",
                    "target_t": 2,
                    "target_pid": "p3",
                    "llm_context_text": "history two",
                    "stage1_candidate_count": 2,
                    "selected_count": 1,
                    "summary_fields": {"sdyn": 0.7},
                    "evidence_list": [{"history_pos": 1}],
                },
            ]
            contexts_path.write_text("".join(json.dumps(row) + "\n" for row in contexts), encoding="utf-8")
            with embedding_path.open("wb") as handle:
                pickle.dump(
                    {
                        "index": [
                            {"user_id": "u1", "target_t": 1, "target_pid": "p2"},
                            {"user_id": "u1", "target_t": 2, "target_pid": "p3"},
                        ],
                        "llm_embeddings": np.ones((2, 6), dtype=np.float32),
                        "llm_struct_features": np.ones((2, 3), dtype=np.float32),
                    },
                    handle,
                )
            for name, dim in (
                ("hqtext_vectors.pkl", 4),
                ("hqid_vectors.pkl", 4),
                ("semantic_vectors.pkl", 2),
                ("item_collaborative_embeddings.pkl", 2),
            ):
                with (priors / name).open("wb") as handle:
                    pickle.dump({row["problem_id"]: np.ones((dim,), dtype=np.float32) for row in catalog}, handle)

            manifest = prepare_existing_stage34_features(
                context_embeddings_path=embedding_path,
                contexts_path=contexts_path,
                priors_dir=priors,
                student_json=student_path,
                output_dir=output_dir,
                context_fields=["llm_embeddings", "llm_struct_features", "stage34_numeric"],
                target_fields=["hqtext", "hqid", "semantic", "collaborative"],
                prompt_context_field="auto",
                max_context_chars=100,
                seed=42,
                test_concept_ratio=0.34,
                valid_concept_ratio=0.0,
                chunk_rows=1,
            )
            self.assertEqual(manifest["record_count"], 2)
            split_counts = manifest["audit"]["split_counts"]
            split = "test" if split_counts.get("test") else "train"
            dataset = SoftSlotDataset(
                output_dir,
                split=split,
                context_fields=["llm_embeddings", "llm_struct_features", "stage34_numeric"],
                target_fields=["hqtext", "hqid", "semantic", "collaborative"],
            )
            self.assertGreater(len(dataset), 0)
            sample = dataset[0]
            self.assertEqual(sample["context_features"].shape[0], 12)
            self.assertEqual(sample["target_features"].shape[0], 12)

            collator = SoftSlotCollator(
                MockTokenizer(),
                load_problem_catalog(priors / "problem_catalog.jsonl"),
                context_soft_tokens=2,
                target_soft_tokens=1,
                include_context_text=True,
            )
            batch = collator([sample])
            self.assertEqual(int(batch["context_mask"].sum()), 2)
            self.assertEqual(int(batch["target_mask"].sum()), 1)


if __name__ == "__main__":
    unittest.main()
