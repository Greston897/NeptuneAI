#!/usr/bin/env python3
"""Build one deduplicated Russian toxicity classifier from verified HF corpora."""

from __future__ import annotations

import argparse
import json
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import ClassLabel, Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .threshold import binary_metrics, select_threshold

BASE_MODEL = "cointegrated/rubert-tiny2"


def _download(repo: str, split: str, config: str = "default") -> Dataset:
    print(f"Loading {repo} [{config}/{split}]", flush=True)
    return load_dataset(repo, config, split=split)


def _label_name(dataset: Dataset, column: str, value: int) -> str:
    feature = dataset.features[column]
    if not isinstance(feature, ClassLabel):
        raise ValueError(f"{column!r} is not a ClassLabel; refusing to guess label mapping")
    return feature.int2str(int(value)).casefold().strip()


def _reservoir_cap(dataset: Dataset, convert, cap: int, seed: int) -> list[dict]:
    """Bound memory while sampling real examples uniformly within each class."""
    rng = random.Random(seed)
    pools: dict[int, list[dict]] = {0: [], 1: []}
    seen = {0: 0, 1: 0}
    for item in dataset:
        row = convert(item)
        text = row.get("text_input")
        label = int(row.get("labels", -1))
        if label not in (0, 1) or not isinstance(text, str) or not text.strip():
            continue
        seen[label] += 1
        pool = pools[label]
        if len(pool) < cap:
            pool.append(row)
        else:
            slot = rng.randrange(seen[label])
            if slot < cap:
                pool[slot] = row
    result = pools[0] + pools[1]
    rng.shuffle(result)
    return result


def _load_sources(max_per_class: int, seed: int) -> tuple[Dataset, dict[str, int]]:
    rows: list[dict] = []
    source_counts: dict[str, int] = {}

    def add(repo: str, dataset: Dataset, convert) -> None:
        sampled = _reservoir_cap(dataset, convert, max_per_class, seed)
        rows.extend(sampled)
        source_counts[repo] = len(sampled)
        print(f"  retained {len(sampled):,} samples from {repo}", flush=True)

    # Mnwa uses ClassLabel names; the actual corpus examples show `negative` is toxic
    # and `positive` is clean. Fail loudly if the upstream label schema changes.
    mnwa = _download("Mnwa/russian-toxic", "train")
    if not {"text", "label"}.issubset(mnwa.column_names):
        raise ValueError(f"Unexpected Mnwa columns: {mnwa.column_names}")
    def convert_mnwa(item: dict) -> dict:
        name = _label_name(mnwa, "label", item["label"])
        if name not in {"negative", "positive", "toxic", "clean", "non-toxic", "nontoxic"}:
            raise ValueError(f"Unrecognized Mnwa label {name!r}; refusing to guess")
        violation = name in {"negative", "toxic"}
        return {"text_input": item["text"], "labels": int(violation)}

    add("Mnwa/russian-toxic", mnwa, convert_mnwa)

    # These two repositories have the same 248k rows/schema and repeated examples;
    # AlexSham is used once rather than downloading the near-identical ilyiniv copy.
    comments = _download("AlexSham/Toxic_Russian_Comments", "train")
    if not {"text", "label"}.issubset(comments.column_names):
        raise ValueError(f"Unexpected AlexSham columns: {comments.column_names}")
    add(
        "AlexSham/Toxic_Russian_Comments",
        comments,
        lambda item: {"text_input": item["text"], "labels": int(item["label"])},
    )

    # Multi-label labels are positive when any explicit profanity/threat/illegal flag
    # is set; remaining noise is handled by cross-corpus conflict removal below.
    multi = _download("dbrovkin/toxic-russian-comments-multilabel", "train")
    required = {"clean_text", "profanity", "threat", "illegal"}
    if not required.issubset(multi.column_names):
        raise ValueError(f"Unexpected dbrovkin columns: {multi.column_names}")
    add(
        "dbrovkin/toxic-russian-comments-multilabel",
        multi,
        lambda item: {
            "text_input": item["clean_text"],
            "labels": int(any(int(item[key]) > 0 for key in ("profanity", "threat", "illegal"))),
        },
    )

    # Toxic -> VIOLATION, detoxified counterpart -> SAFE. The same full sentence is
    # deduplicated against all other corpora, including cases with conflicting labels.
    pairs = _download("s-nlp/ru_paradetox", "train")
    pair_columns = {"ru_toxic_comment", "ru_neutral_comment"}
    if not pair_columns.issubset(pairs.column_names):
        raise ValueError(f"Unexpected ru_paradetox columns: {pairs.column_names}")
    pair_rows = []
    for item in pairs:
        pair_rows.append({"text_input": item["ru_toxic_comment"], "labels": 1})
        pair_rows.append({"text_input": item["ru_neutral_comment"], "labels": 0})
    add("s-nlp/ru_paradetox", Dataset.from_list(pair_rows), lambda item: item)

    toxicity = _download("textdetox/multilingual_toxicity_dataset", "ru")
    if not {"text", "toxic"}.issubset(toxicity.column_names):
        raise ValueError(f"Unexpected textdetox columns: {toxicity.column_names}")
    add(
        "textdetox/multilingual_toxicity_dataset",
        toxicity,
        lambda item: {"text_input": item["text"], "labels": int(item["toxic"])},
    )

    # Canonicalize Unicode/case/whitespace only, so spelling and punctuation are not
    # silently rewritten. Conflicting duplicate labels are excluded, not arbitrated.
    labels_by_text: dict[str, set[int]] = defaultdict(set)
    original_text_by_key: dict[str, str] = {}
    for row in rows:
        text = row.get("text_input")
        if isinstance(text, str) and text.strip() and len(text) <= 20_000:
            key = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
            labels_by_text[key].add(int(row["labels"]))
            original_text_by_key.setdefault(key, text.strip())
    kept = [
        {"text_input": original_text_by_key[key], "labels": next(iter(labels))}
        for key, labels in labels_by_text.items()
        if len(labels) == 1
    ]
    if not kept or {row["labels"] for row in kept} != {0, 1}:
        raise ValueError("After deduplication both SAFE and VIOLATION examples must remain")
    print(
        f"After cross-dataset deduplication: {len(kept):,} unique texts "
        f"({sum(row['labels'] == 1 for row in kept):,} VIOLATION, "
        f"{sum(row['labels'] == 0 for row in kept):,} SAFE)",
        flush=True,
    )
    dataset = Dataset.from_list(kept)
    dataset = dataset.cast_column("labels", ClassLabel(num_classes=2, names=["safe", "violation"]))
    return dataset, source_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/neptune-ru-toxic")
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-per-class", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.5 <= args.target_precision <= 1:
        parser.error("--target-precision must be between 0.5 and 1")
    if args.batch_size < 1 or args.max_length < 8 or args.max_per_class < 1:
        parser.error("batch size, max length and per-class cap must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data, source_counts = _load_sources(args.max_per_class, args.seed)
    train_validation_test = data.train_test_split(
        test_size=0.15, seed=args.seed, stratify_by_column="labels"
    )
    train_validation = train_validation_test["train"].train_test_split(
        test_size=0.14, seed=args.seed + 1, stratify_by_column="labels"
    )
    splits = DatasetDict(
        {
            "train": train_validation["train"],
            "validation": train_validation["test"],
            "test": train_validation_test["test"],
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch: dict[str, list[str]]) -> dict:
        return tokenizer(batch["text_input"], truncation=True, max_length=args.max_length)

    encoded = DatasetDict(
        {
            name: subset.map(tokenize, batched=True, remove_columns=["text_input"])
            for name, subset in splits.items()
        }
    )
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=100,
        report_to="none",
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    trainer.train()

    calibration = trainer.predict(encoded["validation"])
    calibration_probs = torch.softmax(torch.as_tensor(calibration.predictions), dim=-1)[:, 1].numpy()
    threshold, calibration_metrics = select_threshold(
        calibration_probs, np.asarray(calibration.label_ids), args.target_precision
    )
    final = trainer.predict(encoded["test"])
    final_probs = torch.softmax(torch.as_tensor(final.predictions), dim=-1)[:, 1].numpy()
    final_metrics = binary_metrics(
        (final_probs >= threshold).astype(np.int64), np.asarray(final.label_ids)
    )

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metadata = {
        "threshold": threshold,
        "base_model": BASE_MODEL,
        "labels": {"0": "SAFE", "1": "VIOLATION"},
        "datasets": source_counts,
        "deduplication": "Unicode NFKC + casefold + whitespace; conflicting labels dropped",
        "metrics": final_metrics,
        "calibration_metrics": calibration_metrics,
        "max_length": args.max_length,
        "seed": args.seed,
    }
    (output_dir / "threshold.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
