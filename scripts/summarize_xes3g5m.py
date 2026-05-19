from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np


QUESTION_IMAGE_RE = re.compile(r"\bquestion_\d+-image_\d+\b")
ANALYSIS_IMAGE_RE = re.compile(r"\banalysis_\d+-image_\d+\b")
ANY_IMAGE_RE = re.compile(r"\b(?:question|analysis)_\d+-image_\d+\b")


def default_dataset_root() -> Path:
    workspace_root = Path(__file__).resolve().parents[1]
    workspace_dataset = workspace_root / "datalocal" / "XES3G5M"
    if workspace_dataset.exists():
        return workspace_dataset
    windows_dataset = Path(r"D:\Dataset\XES3G5M")
    if windows_dataset.exists():
        return windows_dataset
    return workspace_dataset


Interaction = Tuple[int, int, str, int]


def split_csv_list(value: str | None) -> List[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",")]


def route_leaf(route: str) -> str:
    parts = [part.strip() for part in str(route or "").split("----") if part.strip()]
    return parts[-1] if parts else ""


def load_questions(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"questions.json must be a dict, got {type(raw)}")
    questions: Dict[str, Dict[str, Any]] = {}
    for qid, record in raw.items():
        if isinstance(record, dict):
            questions[str(qid)] = record
    return questions


def question_concepts(record: Dict[str, Any]) -> List[str]:
    concepts: List[str] = []
    seen: set[str] = set()
    for route in record.get("kc_routes") or []:
        concept = route_leaf(str(route))
        if concept and concept not in seen:
            concepts.append(concept)
            seen.add(concept)
    return concepts


def image_scan_text(record: Dict[str, Any]) -> str:
    payload = {
        "content": record.get("content") or "",
        "analysis": record.get("analysis") or "",
        "options": record.get("options") or {},
    }
    return json.dumps(payload, ensure_ascii=False)


def build_question_meta(questions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    for qid, record in questions.items():
        content = str(record.get("content") or "")
        analysis = str(record.get("analysis") or "")
        options = json.dumps(record.get("options") or {}, ensure_ascii=False)
        scan_text = image_scan_text(record)
        content_has_image = bool(QUESTION_IMAGE_RE.search(content))
        analysis_has_image = bool(ANALYSIS_IMAGE_RE.search(analysis))
        options_has_image = bool(QUESTION_IMAGE_RE.search(options) or ANALYSIS_IMAGE_RE.search(options))
        meta[qid] = {
            "concepts": question_concepts(record),
            "content_has_image": content_has_image,
            "analysis_has_image": analysis_has_image,
            "options_has_image": options_has_image,
            "any_image": bool(ANY_IMAGE_RE.search(scan_text)),
            "type": str(record.get("type") or ""),
            "kc_count": len(record.get("kc_routes") or []),
        }
    return meta


def iter_interactions(paths: Sequence[Path]) -> Iterator[Tuple[str, str, int, int, str]]:
    order = 0
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            has_selectmasks = bool(reader.fieldnames and "selectmasks" in reader.fieldnames)
            for row in reader:
                uid = str(row.get("uid") or "").strip()
                questions = split_csv_list(row.get("questions"))
                concepts = split_csv_list(row.get("concepts"))
                responses = split_csv_list(row.get("responses"))
                timestamps = split_csv_list(row.get("timestamps"))
                masks = split_csv_list(row.get("selectmasks")) if has_selectmasks else []
                for idx, qid in enumerate(questions):
                    order += 1
                    if not uid or qid == "-1" or not qid:
                        continue
                    if masks and (idx >= len(masks) or masks[idx] != "1"):
                        continue
                    if idx >= len(responses) or responses[idx] not in {"0", "1"}:
                        continue
                    timestamp = 0
                    if idx < len(timestamps):
                        try:
                            timestamp = int(float(timestamps[idx]))
                        except ValueError:
                            timestamp = 0
                    concept_field = concepts[idx] if idx < len(concepts) else ""
                    yield uid, qid, int(responses[idx]), timestamp, concept_field


def split_new_concepts(
    concepts: Iterable[str],
    *,
    seed: int,
    test_concept_ratio: float,
    valid_concept_ratio: float,
) -> Tuple[set[str], set[str], set[str]]:
    concept_list = sorted(str(concept) for concept in concepts if str(concept))
    shuffled = np.asarray(concept_list, dtype=object)
    if len(shuffled) > 0:
        rng = np.random.default_rng(int(seed))
        shuffled = shuffled[rng.permutation(len(shuffled))]
    total = len(shuffled)
    test_count = max(1, int(total * float(test_concept_ratio))) if total and test_concept_ratio > 0 else 0
    remaining_after_test = max(0, total - test_count)
    valid_count = (
        max(1, int(total * float(valid_concept_ratio)))
        if remaining_after_test and valid_concept_ratio > 0
        else 0
    )
    valid_count = min(valid_count, remaining_after_test)
    test_concepts = set(shuffled[:test_count].tolist())
    valid_concepts = set(shuffled[test_count : test_count + valid_count].tolist())
    train_concepts = set(concept_list) - test_concepts - valid_concepts
    return train_concepts, valid_concepts, test_concepts


def summarize_metadata(question_meta: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    type_counter = Counter(meta["type"] for meta in question_meta.values())
    kc_count_counter = Counter(int(meta["kc_count"]) for meta in question_meta.values())
    concepts = sorted({concept for meta in question_meta.values() for concept in meta["concepts"]})
    return {
        "questions_total": len(question_meta),
        "question_content_image_questions": sum(1 for meta in question_meta.values() if meta["content_has_image"]),
        "analysis_image_questions": sum(1 for meta in question_meta.values() if meta["analysis_has_image"]),
        "any_image_questions": sum(1 for meta in question_meta.values() if meta["any_image"]),
        "text_only_by_question_content": sum(1 for meta in question_meta.values() if not meta["content_has_image"]),
        "text_only_strict": sum(1 for meta in question_meta.values() if not meta["any_image"]),
        "types": dict(sorted(type_counter.items())),
        "kc_count_distribution": dict(sorted(kc_count_counter.items())),
        "unique_leaf_concepts": len(concepts),
    }


def allowed_qids_for_variant(question_meta: Dict[str, Dict[str, Any]], variant: str) -> set[str]:
    if variant == "all":
        return set(question_meta)
    if variant == "no_question_image":
        return {qid for qid, meta in question_meta.items() if not meta["content_has_image"]}
    if variant == "strict_text_only":
        return {qid for qid, meta in question_meta.items() if not meta["any_image"]}
    raise ValueError(f"Unknown variant: {variant}")


def summarize_interactions(
    *,
    question_meta: Dict[str, Dict[str, Any]],
    interaction_paths: Sequence[Path],
) -> Tuple[Dict[str, Any], Dict[str, List[Interaction]]]:
    variants = {
        "all": allowed_qids_for_variant(question_meta, "all"),
        "no_question_image": allowed_qids_for_variant(question_meta, "no_question_image"),
        "strict_text_only": allowed_qids_for_variant(question_meta, "strict_text_only"),
    }
    stats = {
        name: {
            "students": set(),
            "questions": set(),
            "interactions": 0,
            "correct": 0,
            "incorrect": 0,
            "csv_concepts": set(),
            "metadata_leaf_concepts": set(),
        }
        for name in variants
    }
    grouped: Dict[str, List[Interaction]] = defaultdict(list)
    missing_qids = Counter()

    for uid, qid, response, timestamp, concept_field in iter_interactions(interaction_paths):
        if qid not in question_meta:
            missing_qids[qid] += 1
            continue
        grouped[uid].append((timestamp, len(grouped[uid]), qid, response))
        csv_concepts = [part for part in concept_field.split("_") if part and part != "-1"]
        metadata_concepts = question_meta[qid]["concepts"]
        for name, allowed in variants.items():
            if qid not in allowed:
                continue
            stats[name]["students"].add(uid)
            stats[name]["questions"].add(qid)
            stats[name]["interactions"] += 1
            if response == 1:
                stats[name]["correct"] += 1
            else:
                stats[name]["incorrect"] += 1
            stats[name]["csv_concepts"].update(csv_concepts)
            stats[name]["metadata_leaf_concepts"].update(metadata_concepts)

    materialized: Dict[str, Any] = {}
    for name, payload in stats.items():
        interactions = int(payload["interactions"])
        materialized[name] = {
            "students": len(payload["students"]),
            "questions": len(payload["questions"]),
            "interactions": interactions,
            "correct": int(payload["correct"]),
            "incorrect": int(payload["incorrect"]),
            "positive_rate": float(payload["correct"] / interactions) if interactions else 0.0,
            "csv_concepts": len(payload["csv_concepts"]),
            "metadata_leaf_concepts": len(payload["metadata_leaf_concepts"]),
        }
    materialized["missing_question_interactions"] = int(sum(missing_qids.values()))
    materialized["missing_question_ids"] = len(missing_qids)
    return materialized, grouped


def summarize_sequence_lengths(
    grouped: Dict[str, List[Interaction]],
    *,
    question_meta: Dict[str, Dict[str, Any]],
    allowed_qids: set[str],
) -> Dict[str, Any]:
    lengths: List[int] = []
    for interactions in grouped.values():
        count = sum(1 for _ts, _order, qid, _response in interactions if qid in allowed_qids)
        if count > 0:
            lengths.append(count)
    if not lengths:
        return {"min": 0, "p50": 0, "p90": 0, "max": 0, "mean": 0.0}
    arr = np.asarray(lengths, dtype=np.float64)
    return {
        "min": int(np.min(arr)),
        "p50": int(np.percentile(arr, 50)),
        "p90": int(np.percentile(arr, 90)),
        "max": int(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def summarize_cold_start(
    *,
    grouped: Dict[str, List[Interaction]],
    question_meta: Dict[str, Dict[str, Any]],
    variant: str,
    seed: int,
    test_concept_ratio: float,
    valid_concept_ratio: float,
) -> Dict[str, Any]:
    allowed_qids = allowed_qids_for_variant(question_meta, variant)
    qid_to_concepts = {
        qid: set(meta["concepts"])
        for qid, meta in question_meta.items()
        if qid in allowed_qids and meta["concepts"]
    }
    all_concepts = sorted({concept for concepts in qid_to_concepts.values() for concept in concepts})
    train_concepts, valid_concepts, test_concepts = split_new_concepts(
        all_concepts,
        seed=seed,
        test_concept_ratio=test_concept_ratio,
        valid_concept_ratio=valid_concept_ratio,
    )
    holdout_concepts = valid_concepts | test_concepts

    catalog_train_questions = set()
    catalog_test_questions = set()
    catalog_mixed_questions = set()
    for qid, concepts in qid_to_concepts.items():
        has_train = bool(concepts & train_concepts)
        has_holdout = bool(concepts & holdout_concepts)
        if has_holdout:
            catalog_test_questions.add(qid)
        else:
            catalog_train_questions.add(qid)
        if has_train and has_holdout:
            catalog_mixed_questions.add(qid)

    train_interactions = 0
    test_target_interactions = 0
    test_target_with_old_history = 0
    skipped_test_targets_no_old_history = 0
    train_users: set[str] = set()
    test_users: set[str] = set()
    train_questions: set[str] = set()
    test_questions: set[str] = set()

    for uid, interactions in grouped.items():
        sorted_interactions = sorted(interactions, key=lambda item: (item[0], item[1]))
        has_old_history = False
        for _timestamp, _order, qid, _response in sorted_interactions:
            if qid not in allowed_qids:
                continue
            concepts = qid_to_concepts.get(qid, set())
            if not concepts:
                continue
            if concepts & test_concepts:
                test_target_interactions += 1
                test_questions.add(qid)
                if has_old_history:
                    test_target_with_old_history += 1
                    test_users.add(uid)
                else:
                    skipped_test_targets_no_old_history += 1
            elif not (concepts & holdout_concepts):
                train_interactions += 1
                train_users.add(uid)
                train_questions.add(qid)
                has_old_history = True

    return {
        "variant": variant,
        "seed": seed,
        "test_concept_ratio": test_concept_ratio,
        "valid_concept_ratio": valid_concept_ratio,
        "total_concepts": len(all_concepts),
        "configured_train_concepts": len(train_concepts),
        "configured_valid_concepts": len(valid_concepts),
        "configured_test_concepts": len(test_concepts),
        "catalog_train_questions": len(catalog_train_questions),
        "catalog_test_questions": len(catalog_test_questions),
        "catalog_mixed_questions": len(catalog_mixed_questions),
        "observed_train_questions": len(train_questions),
        "observed_test_questions": len(test_questions),
        "train_users": len(train_users),
        "test_users_with_old_history": len(test_users),
        "train_interactions": int(train_interactions),
        "test_target_interactions": int(test_target_interactions),
        "test_target_with_old_history": int(test_target_with_old_history),
        "skipped_test_targets_no_old_history": int(skipped_test_targets_no_old_history),
    }


def write_report(summary: Dict[str, Any], path: Path) -> None:
    lines = [
        "# XES3G5M Dataset Statistics",
        "",
        f"Generated at: {summary['generated_at']}",
        f"Dataset root: `{summary['dataset_root']}`",
        "",
        "## Metadata",
        "",
    ]
    metadata = summary["metadata"]
    for key in [
        "questions_total",
        "question_content_image_questions",
        "analysis_image_questions",
        "any_image_questions",
        "text_only_by_question_content",
        "text_only_strict",
        "unique_leaf_concepts",
    ]:
        lines.append(f"- {key}: {metadata[key]}")
    lines.extend(["", "## Interaction Variants", ""])
    lines.append("| variant | students | questions | interactions | positive_rate | csv_concepts | leaf_concepts |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for variant in ["all", "no_question_image", "strict_text_only"]:
        item = summary["interactions"][variant]
        lines.append(
            f"| {variant} | {item['students']} | {item['questions']} | {item['interactions']} | "
            f"{item['positive_rate']:.4f} | {item['csv_concepts']} | {item['metadata_leaf_concepts']} |"
        )
    lines.extend(["", "## Sequence Lengths", ""])
    lines.append("| variant | min | p50 | p90 | max | mean |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for variant in ["all", "no_question_image", "strict_text_only"]:
        item = summary["sequence_lengths"][variant]
        lines.append(
            f"| {variant} | {item['min']} | {item['p50']} | {item['p90']} | {item['max']} | {item['mean']:.2f} |"
        )
    lines.extend(["", "## Concept Cold-Start Split", ""])
    lines.append(
        "| variant | concepts train/test | catalog q train/test/mixed | train interactions | "
        "test targets | test targets with old history | train users | test users |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for variant in ["all", "no_question_image", "strict_text_only"]:
        item = summary["cold_start"][variant]
        lines.append(
            f"| {variant} | {item['configured_train_concepts']}/{item['configured_test_concepts']} | "
            f"{item['catalog_train_questions']}/{item['catalog_test_questions']}/{item['catalog_mixed_questions']} | "
            f"{item['train_interactions']} | {item['test_target_interactions']} | "
            f"{item['test_target_with_old_history']} | {item['train_users']} | {item['test_users_with_old_history']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize local XES3G5M question-level data.")
    parser.add_argument("--dataset_root", type=Path, default=default_dataset_root())
    parser.add_argument("--out_json", type=Path, default=Path("out/xes3g5m_stats/xes3g5m_stats.json"))
    parser.add_argument("--out_md", type=Path, default=Path("out/xes3g5m_stats/xes3g5m_stats.md"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_concept_ratio", type=float, default=0.8)
    parser.add_argument("--valid_concept_ratio", type=float, default=0.0)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    question_paths = [
        root / "question_level" / "train_valid_sequences_quelevel.csv",
        root / "question_level" / "test_quelevel.csv",
    ]
    questions_path = root / "metadata" / "questions.json"
    for path in [questions_path, *question_paths]:
        if not path.exists():
            raise FileNotFoundError(path)

    questions = load_questions(questions_path)
    question_meta = build_question_meta(questions)
    interaction_stats, grouped = summarize_interactions(question_meta=question_meta, interaction_paths=question_paths)

    sequence_lengths = {}
    cold_start = {}
    for variant in ["all", "no_question_image", "strict_text_only"]:
        allowed_qids = allowed_qids_for_variant(question_meta, variant)
        sequence_lengths[variant] = summarize_sequence_lengths(
            grouped,
            question_meta=question_meta,
            allowed_qids=allowed_qids,
        )
        cold_start[variant] = summarize_cold_start(
            grouped=grouped,
            question_meta=question_meta,
            variant=variant,
            seed=int(args.seed),
            test_concept_ratio=float(args.test_concept_ratio),
            valid_concept_ratio=float(args.valid_concept_ratio),
        )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(root),
        "source_files": [str(path) for path in question_paths],
        "metadata": summarize_metadata(question_meta),
        "interactions": interaction_stats,
        "sequence_lengths": sequence_lengths,
        "cold_start": cold_start,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, args.out_md)

    print("[OK] XES3G5M stats written")
    print("[JSON]", args.out_json.resolve())
    print("[MD]", args.out_md.resolve())
    print(json.dumps(
        {
            "metadata": summary["metadata"],
            "interactions": summary["interactions"],
            "cold_start": summary["cold_start"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
