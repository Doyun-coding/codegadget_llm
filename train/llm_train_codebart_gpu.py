#!/usr/bin/env python3
# train_codebert_llm.py
"""
Train CodeBERT (or other HF seq-classif model) on an LLM-generated CodeBERT dataset.

Usage:
  python train_codebert_llm.py --data-dir ./data/llm/juliet_codebert_dataset --output-dir ./llm-codebert \
      --model microsoft/codebert-base --per-device-batch-size 8 --epochs 3

Requires:
  pip install transformers datasets torch scikit-learn
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# -------------------
# Weighted Trainer (applies class weights in CrossEntropy)
# -------------------
class WeightedTrainer(Trainer):
    def __init__(self, class_weights_tensor=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # class_weights_tensor should be a torch.Tensor on CPU (we'll move to device in compute_loss)
        self.class_weights = class_weights_tensor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the loss using CrossEntropyLoss with optional class weights.
        Accepts **kwargs to remain compatible across HF Trainer versions.
        """
        labels = inputs.get("labels")
        # Forward pass (model(**inputs) expects input_ids, attention_mask, labels optional)
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs.get("logits")

        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=cw)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# -------------------
# Utilities
# -------------------
def tokenize_batch(tokenizer, batch, max_length):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }

# -------------------
# Main
# -------------------
def main():
    parser = argparse.ArgumentParser(description="Train CodeBERT on LLM-generated dataset")
    parser.add_argument("--data-dir", required=True, help="Directory containing train.jsonl/validation.jsonl/test.jsonl")
    parser.add_argument("--model", default="microsoft/codebert-base", help="HuggingFace model name")
    parser.add_argument("--output-dir", default="./llm-codebert", help="Output directory for checkpoints")
    parser.add_argument("--per-device-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Force fp16 (if GPU available)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"data dir not found: {data_dir}")

    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "validation.jsonl"
    test_file = data_dir / "test.jsonl"

    for p in (train_file, val_file, test_file):
        if not p.exists():
            raise SystemExit(f"Missing dataset file: {p}")

    device = "cuda" if torch.cuda.is_available() else "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
    use_cuda = (device == "cuda")
    print("[INFO] Device:", device)
    if use_cuda:
        print("[INFO] CUDA device name:", torch.cuda.get_device_name(0))

    print("[INFO] Loading tokenizer and model:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    print("[INFO] Loading datasets from JSONL")
    ds = load_dataset("json", data_files={
        "train": str(train_file),
        "validation": str(val_file),
        "test": str(test_file)
    })
    # Ensure label column is present & integer
    def fix_label(x):
        x["label"] = int(x["label"])
        return x
    ds = ds.map(fix_label)

    print("[INFO] Tokenizing dataset (max_length=%d)..." % args.max_length)
    ds = ds.map(lambda b: tokenize_batch(tokenizer, b, args.max_length), batched=True, batch_size=256)
    # Trainer expects 'labels' column (not 'label') for some versions; create both safely
    ds = ds.map(lambda x: {"labels": x["label"]})

    # set format for PyTorch
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # compute class weights from training set
    train_labels = np.array(ds["train"]["labels"])
    n0 = int((train_labels == 0).sum())
    n1 = int((train_labels == 1).sum())
    print(f"[INFO] train label counts -> 0: {n0}, 1: {n1}")
    class_weights = None
    if n0 == 0 or n1 == 0:
        print("[WARN] One class has zero samples - class weighting disabled.")
    else:
        w0 = (n0 + n1) / (2.0 * n0)
        w1 = (n0 + n1) / (2.0 * n1)
        class_weights = torch.tensor([w0, w1], dtype=torch.float32)
        print(f"[INFO] class weights -> {class_weights.tolist()}")

    # Training arguments: note use 'eval_strategy' for older transformers compatibility
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_strategy="steps",
        logging_steps=100,
        eval_strategy="epoch",   # older HF versions use eval_strategy
        save_strategy="epoch",
        save_total_limit=2,
        fp16=(args.fp16 and use_cuda),
        remove_unused_columns=False,  # keep label column available
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        class_weights_tensor=class_weights
    )

    # Optionally disable tokenizers parallelism env
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("[INFO] Starting training...")
    trainer.train()

    print("[INFO] Training finished. Evaluating on test set...")
    metrics = trainer.evaluate(ds["test"])
    print("Test metrics:", metrics)

    print("[INFO] Saving final model to:", os.path.join(args.output_dir, "final"))
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

if __name__ == "__main__":
    main()
