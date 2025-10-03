#!/usr/bin/env python3
"""
Evaluate a HuggingFace sequence-classification checkpoint on a JSONL test set.

Saves:
 - outdir/predictions.csv (id,file,true_label,pred_label,prob_0,prob_1,...)
 - outdir/metrics.json (metrics + bootstrap CIs)
 - outdir/confusion_matrix.png
 - outdir/roc_pr_curve.png

Required packages:
  pip install transformers datasets torch scikit-learn matplotlib seaborn pandas numpy

"""
import argparse, json, os, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random

def safe_load_dataset(test_json):
    ds = load_dataset("json", data_files={"test": test_json})["test"]
    if "text" not in ds.column_names or "label" not in ds.column_names:
        raise ValueError("test jsonl must have 'text' and 'label' fields.")
    return ds

def tokenize_dataset(ds, tokenizer, batch_size, max_length=512):
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)
    ds = ds.map(tok, batched=True, batch_size=batch_size)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return ds

def predict_with_trainer(model, tokenizer, ds_test, device, batch_size):
    # Use Trainer.predict for convenience (works if transformers & accelerate installed)
    trainer = Trainer(model=model, tokenizer=tokenizer)
    pred_output = trainer.predict(ds_test, max_length=tokenizer.model_max_length, batch_size=batch_size)
    logits = pred_output.predictions
    labels = pred_output.label_ids
    if logits.ndim == 1:
        probs = np.vstack([1 - logits, logits]).T
    else:
        # softmax
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    return preds, probs, labels

def compute_metrics_all(y_true, y_pred, probs):
    res = {}
    res["accuracy"] = float(accuracy_score(y_true, y_pred))
    res["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    res["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    res["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    # per-class
    res["precision_per_class"] = dict(zip(map(str, sorted(set(y_true))), precision_score(y_true, y_pred, average=None, zero_division=0).tolist()))
    res["recall_per_class"] = dict(zip(map(str, sorted(set(y_true))), recall_score(y_true, y_pred, average=None, zero_division=0).tolist()))
    res["f1_per_class"] = dict(zip(map(str, sorted(set(y_true))), f1_score(y_true, y_pred, average=None, zero_division=0).tolist()))
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    res["confusion_matrix"] = cm.tolist()
    # ROC AUC / PR AUC (binary only)
    classes = sorted(set(y_true))
    if len(classes) == 2:
        # assume class labels are 0/1; get probs[:,1]
        try:
            y_score = probs[:,1]
            res["roc_auc"] = float(roc_auc_score(y_true, y_score))
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            res["pr_auc"] = float(auc(rec, prec))
        except Exception as e:
            res["roc_auc"] = None
            res["pr_auc"] = None
    else:
        res["roc_auc"] = None
        res["pr_auc"] = None
    return res

def bootstrap_ci(y_true, probs, n_boot=1000, seed=42):
    """
    Bootstrap 95% CI for accuracy and f1_macro.
    Returns dict: {'accuracy':(mean, lower, upper), 'f1_macro':(...)}
    """
    rng = random.Random(seed)
    n = len(y_true)
    accs = []
    f1s = []
    for _ in tqdm(range(n_boot), desc="bootstrap"):
        idxs = [rng.randrange(0, n) for _ in range(n)]
        y_t = [y_true[i] for i in idxs]
        # take predicted labels from probs
        p = probs[idxs]
        y_p = list(np.argmax(p, axis=1))
        accs.append(accuracy_score(y_t, y_p))
        f1s.append(f1_score(y_t, y_p, average="macro", zero_division=0))
    def summarize(arr):
        a = np.array(arr)
        mean = float(a.mean())
        lo = float(np.percentile(a, 2.5))
        hi = float(np.percentile(a, 97.5))
        return {"mean": mean, "2.5%": lo, "97.5%": hi}
    return {"accuracy": summarize(accs), "f1_macro": summarize(f1s)}

def plot_confusion_matrix(cm, outpath, class_names=None):
    plt.figure(figsize=(4,4))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_roc_pr(y_true, probs, outpath):
    # only binary
    if probs.shape[1] < 2:
        return
    y_score = probs[:,1]
    # ROC
    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc_score(y_true,y_score):.4f})")
    plt.plot([0,1],[0,1],"k--", linewidth=0.6)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curve"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(rec, prec, label=f"PR (AUC={auc(rec,prec):.4f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall"); plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint folder (hf format) to load model from")
    parser.add_argument("--test-json", required=True, help="Test jsonl (fields: id?, file?, text, label)")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap iterations for CI (set 0 to skip)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    print("Loading model and tokenizer...")
    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, use_fast=True)
    model.to(device)

    print("Loading test dataset...")
    ds_test = safe_load_dataset(args.test_json)
    ds_test_tokenized = tokenize_dataset(ds_test, tokenizer, batch_size=args.batch_size)

    print("Predicting...")
    preds, probs, labels = predict_with_trainer(model, tokenizer, ds_test_tokenized, device=device, batch_size=args.batch_size)
    preds = np.array(preds); probs = np.array(probs); labels = np.array(labels)

    print("Saving predictions CSV...")
    rows=[]
    for i in range(len(labels)):
        row = {
            "id": ds_test[i].get("id", i),
            "file": ds_test[i].get("file", ""),
            "true_label": int(labels[i]),
            "pred_label": int(preds[i])
        }
        # add prob columns
        for c in range(probs.shape[1]):
            row[f"prob_{c}"] = float(probs[i,c])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(Path(outdir)/"predictions.csv", index=False)

    print("Computing metrics...")
    metrics = compute_metrics_all(labels, preds, probs)

    # bootstrap CIs
    if args.bootstrap > 0:
        print(f"Running bootstrap ({args.bootstrap}) to compute 95% CI for accuracy & f1_macro...")
        ci = bootstrap_ci(labels, probs, n_boot=args.bootstrap, seed=args.seed)
    else:
        ci = None

    out_metrics = {"metrics": metrics, "bootstrap": ci}
    with open(Path(outdir)/"metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2)

    # plots
    print("Plotting confusion matrix and ROC/PR...")
    class_names = [str(c) for c in sorted(set(labels))]
    plot_confusion_matrix(metrics["confusion_matrix"], Path(outdir)/"confusion_matrix.png", class_names)
    if probs.shape[1] >= 2:
        plot_roc_pr(labels, probs, Path(outdir)/"roc_pr.png")

    print("Done. Outputs in:", outdir)
    print(json.dumps({"summary": {"accuracy": metrics["accuracy"], "f1_macro": metrics["f1_macro"], "roc_auc": metrics.get("roc_auc")} }, indent=2))

if __name__ == "__main__":
    main()
