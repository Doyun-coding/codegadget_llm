#!/usr/bin/env python3
"""
Evaluate a HuggingFace sequence-classification checkpoint on a JSONL test set.

Saves:
 - outdir/predictions.csv (id,file,true_label,pred_label,prob_0,prob_1,...)
 - outdir/metrics.json (metrics + bootstrap CIs + optional LLM-as-Eval)
 - outdir/confusion_matrix.png
 - outdir/roc_pr_curve.png
 - outdir/llm_eval/llm_judgments.jsonl (if --llm-eval)

Usage example:
python  ./evaluation/evaluate_checkpoint_llm_as_eval.py \
  --ckpt ./static-codebert-model \
  --test-json ./data/juliet_eval_from_sources/test.jsonl \
  --outdir ./eval/with_llm_eval \
  --batch-size 32 --bootstrap 0 --max-length 256 \
  --llm-eval --llm-model gpt-4o-mini --llm-sample 400

 python evaluate_checkpoint.py \
  --ckpt static-codebert/checkpoint-14148_1 \
  --test-json ./data/static/juliet_codebert_dataset/test.jsonl \
  --outdir ./eval/out \
  --batch-size 32 \
  --bootstrap 1000 \
  --max-length 512 \
  --llm-eval --llm-model gpt-4o-mini --llm-sample 300
"""
import argparse, json, os, math, re, time, random
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --------------------------
# Core dataset / model eval
# --------------------------
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
    """
    Try HF Trainer.predict first (simple). If it fails due to version mismatch,
    fall back to manual DataLoader loop.
    Returns: preds (np.array), probs (np.array, N x C), labels (np.array)
    """
    model.to(device)
    # Try trainer.predict path first
    try:
        trainer = Trainer(model=model, tokenizer=tokenizer)
        pred_output = trainer.predict(ds_test, batch_size=batch_size)
        logits = pred_output.predictions
        labels = pred_output.label_ids
        if logits is None:
            raise RuntimeError("Trainer.predict returned no predictions")
        logits = np.asarray(logits)
        if logits.ndim == 1:
            probs = np.vstack([1 - logits, logits]).T
        elif logits.ndim == 2:
            logits_t = torch.from_numpy(logits)
            probs = F.softmax(logits_t, dim=-1).cpu().numpy()
        elif logits.ndim == 3:
            logits2 = logits[:,0,:] if logits.shape[1] >= 1 else logits.mean(axis=1)
            logits_t = torch.from_numpy(logits2)
            probs = F.softmax(logits_t, dim=-1).cpu().numpy()
        else:
            raise RuntimeError(f"Unexpected logits ndim: {logits.ndim}")
        preds = np.argmax(probs, axis=1)
        labels = np.asarray(labels) if labels is not None else None
        return preds, probs, labels
    except TypeError as e:
        print(f"[WARN] Trainer.predict approach failed ({e}), falling back to manual DataLoader inference...", flush=True)
    except Exception as e:
        print(f"[WARN] Trainer.predict raised exception ({e}), falling back to manual inference...", flush=True)

    # Manual inference fallback
    loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="manual predict"):
            inputs = {}
            if "input_ids" in batch: inputs["input_ids"] = batch["input_ids"].to(device)
            if "attention_mask" in batch: inputs["attention_mask"] = batch["attention_mask"].to(device)
            if "token_type_ids" in batch: inputs["token_type_ids"] = batch["token_type_ids"].to(device)
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            all_logits.append(logits.detach().cpu().numpy())
            if "label" in batch:
                all_labels.append(batch["label"].cpu().numpy())
    if not all_logits:
        raise RuntimeError("No predictions from manual inference (empty loader?).")
    logits = np.vstack(all_logits)
    labels = np.concatenate(all_labels) if all_labels else None

    if logits.ndim == 1:
        probs = np.vstack([1 - logits, logits]).T
    elif logits.ndim == 2:
        probs = F.softmax(torch.from_numpy(logits), dim=-1).cpu().numpy()
    elif logits.ndim == 3:
        logits2 = logits[:,0,:] if logits.shape[1] >= 1 else logits.mean(axis=1)
        probs = F.softmax(torch.from_numpy(logits2), dim=-1).cpu().numpy()
    else:
        raise RuntimeError(f"Unexpected logits ndim (manual): {logits.ndim}")
    preds = np.argmax(probs, axis=1)
    labels = np.asarray(labels) if labels is not None else None
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
        try:
            y_score = probs[:,1]
            res["roc_auc"] = float(roc_auc_score(y_true, y_score))
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            res["pr_auc"] = float(auc(rec, prec))
        except Exception:
            res["roc_auc"] = None
            res["pr_auc"] = None
    else:
        res["roc_auc"] = None
        res["pr_auc"] = None
    return res

def bootstrap_ci(y_true, probs, n_boot=1000, seed=42):
    """Bootstrap 95% CI for accuracy and f1_macro."""
    rng = random.Random(seed)
    n = len(y_true)
    accs, f1s = [], []
    for _ in tqdm(range(n_boot), desc="bootstrap"):
        idxs = [rng.randrange(0, n) for _ in range(n)]
        y_t = [y_true[i] for i in idxs]
        p = probs[idxs]
        y_p = list(np.argmax(p, axis=1))
        accs.append(accuracy_score(y_t, y_p))
        f1s.append(f1_score(y_t, y_p, average="macro", zero_division=0))
    def summarize(arr):
        a = np.array(arr)
        return {"mean": float(a.mean()),
                "2.5%": float(np.percentile(a, 2.5)),
                "97.5%": float(np.percentile(a, 97.5))}
    return {"accuracy": summarize(accs), "f1_macro": summarize(f1s)}

def plot_confusion_matrix(cm, outpath, class_names=None):
    plt.figure(figsize=(4,4))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_roc_pr(y_true, probs, outpath):
    if probs.shape[1] < 2:
        return
    y_score = probs[:,1]
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

# --------------------------
# LLM-as-Eval (optional)
# --------------------------
def _get_openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    try:
        from openai import OpenAI
        try:
            client = OpenAI(api_key=key)
        except Exception:
            client = OpenAI()
        return client, "openai_new"
    except Exception as e:
        raise RuntimeError(f"OpenAI client import failed: {e}")

_PROMPT_EVAL = r"""
You are a security expert judging C/C++ code snippets for vulnerability.
Return STRICT JSON ONLY with keys:
  llm_label  : 1 if vulnerable, else 0 (integer)
  confidence : 0.0-1.0 (float)
  reason     : short string (<=20 words)

Definitions:
- "vulnerable (1)" means the snippet exhibits a concrete vulnerability pattern
  (e.g., unchecked buffer copy, format string injection, command injection, path traversal).
- "non-vulnerable (0)" means no concrete exploit pattern is present in the shown context.

Now judge the following snippet.
Model's predicted label (0=non-vuln, 1=vuln): {model_pred}

-----CODE-BEGIN-----
{code}
-----CODE-END-----
"""

def _safe_parse_json_only(text: str) -> dict:
    s = text.find("{"); e = text.rfind("}")
    if s == -1 or e == -1 or s > e:
        raise ValueError("No JSON braces found.")
    raw = text[s:e+1]
    raw = raw.replace("`","")
    raw = re.sub(r",\s*}", "}", raw)
    raw = re.sub(r",\s*]", "]", raw)
    return json.loads(raw)

def _llm_judge_one(client_tuple, model:str, code:str, model_pred:int, max_tokens:int=200, temperature:float=0.0):
    client, ctype = client_tuple
    system = "You answer with JSON ONLY."
    prompt = _PROMPT_EVAL.format(model_pred=model_pred, code=code)
    # single attempt; you can add retry if you want
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type":"json_object"},
    )
    msg = resp.choices[0].message
    content = getattr(msg, "content", "") or ""
    obj = _safe_parse_json_only(content)
    llm_label = int(obj.get("llm_label", 0))
    conf = float(obj.get("confidence", 0.0))
    reason = str(obj.get("reason",""))[:200]
    return {"llm_label": llm_label, "confidence": conf, "reason": reason}

def _balanced_sample_indices(y_true, k):
    """Label-balanced sampling of indices (binary)."""
    n = len(y_true)
    if k <= 0 or k >= n:
        return list(range(n))
    idx0 = [i for i, y in enumerate(y_true) if y == 0]
    idx1 = [i for i, y in enumerate(y_true) if y == 1]
    random.shuffle(idx0); random.shuffle(idx1)
    half = k // 2
    take0 = idx0[:min(len(idx0), half)]
    take1 = idx1[:min(len(idx1), k - len(take0))]
    # if still short, fill from whichever remains
    remain = [i for i in range(n) if i not in set(take0) | set(take1)]
    random.shuffle(remain)
    need = k - (len(take0) + len(take1))
    take = take0 + take1 + remain[:max(0, need)]
    return take

def run_llm_as_eval(outdir: Path, rows_df: pd.DataFrame, test_texts, seed:int,
                    llm_model:str, llm_sample:int, llm_temp:float, llm_max_tokens:int):
    random.seed(seed)
    # Prepare subset indices (balanced on gold labels)
    y_true = rows_df["true_label"].astype(int).tolist()
    sample_idx = _balanced_sample_indices(y_true, llm_sample)
    sub = rows_df.iloc[sample_idx].copy()

    # Prepare OpenAI client
    client_tuple = _get_openai_client()

    # Output dir
    lev_dir = outdir / "llm_eval"
    lev_dir.mkdir(parents=True, exist_ok=True)
    judg_path = lev_dir / "llm_judgments.jsonl"

    # Judge
    judg_rows = []
    with open(judg_path, "w", encoding="utf-8") as w:
        for _, r in tqdm(sub.iterrows(), total=len(sub), desc="LLM-as-Eval"):
            i = int(r.name)
            code = str(test_texts[i])
            # cap very long snippets (optional)
            if len(code) > 4000:
                code = code[:4000] + "\n/* …truncated… */"
            jp = _llm_judge_one(client_tuple, llm_model, code, int(r["pred_label"]),
                                max_tokens=llm_max_tokens, temperature=llm_temp)
            row = {
                "row_index": int(i),
                "id": r["id"],
                "file": r["file"],
                "true_label": int(r["true_label"]),
                "pred_label": int(r["pred_label"]),
                "llm_label": int(jp["llm_label"]),
                "llm_confidence": float(jp["confidence"]),
                "llm_reason": jp["reason"],
            }
            judg_rows.append(row)
            w.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Compute aggregate metrics
    jf = pd.DataFrame(judg_rows)
    agreement = float((jf["pred_label"] == jf["llm_label"]).mean())
    llm_acc = float((jf["llm_label"] == jf["true_label"]).mean())
    model_acc_sub = float((jf["pred_label"] == jf["true_label"]).mean())
    llm_f1_macro = float(f1_score(jf["true_label"], jf["llm_label"], average="macro", zero_division=0))
    conf_mat = confusion_matrix(jf["pred_label"], jf["llm_label"]).tolist()

    summary = {
        "n_judged": int(len(jf)),
        "agreement_model_vs_llm": agreement,
        "llm_accuracy_vs_gold": llm_acc,
        "llm_f1_macro_vs_gold": llm_f1_macro,
        "model_accuracy_on_judged_vs_gold": model_acc_sub,
        "confusion_model_vs_llm": conf_mat,
        "llm_model": llm_model,
    }
    with open(lev_dir/"llm_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint folder (hf format)")
    parser.add_argument("--test-json", required=True, help="Test jsonl (fields: id?, file?, text, label)")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap iterations for CI (set 0 to skip)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)

    # LLM-as-Eval opts
    parser.add_argument("--llm-eval", action="store_true", help="Enable LLM-as-judge evaluation")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--llm-sample", type=int, default=300, help="Max judged examples (balanced by gold labels)")
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-max-tokens", type=int, default=200)
    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Device:", device)

    print("Loading model and tokenizer...")
    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, use_fast=True)
    model.to(device)

    print("Loading test dataset...")
    ds_test = safe_load_dataset(args.test_json)
    ds_test_tokenized = tokenize_dataset(ds_test, tokenizer, batch_size=args.batch_size, max_length=args.max_length)

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
        for c in range(probs.shape[1]):
            row[f"prob_{c}"] = float(probs[i,c])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(Path(outdir)/"predictions.csv", index=False)

    print("Computing metrics...")
    metrics = compute_metrics_all(labels, preds, probs)

    if args.bootstrap > 0:
        print(f"Running bootstrap ({args.bootstrap}) to compute 95% CI for accuracy & f1_macro...")
        ci = bootstrap_ci(labels, probs, n_boot=args.bootstrap, seed=args.seed)
    else:
        ci = None

    # Optional: LLM-as-Eval
    llm_eval_summary = None
    if args.llm_eval:
        try:
            print("Running LLM-as-Eval (judge)...")
            # pass raw texts for judged items
            test_texts = [ds_test[i]["text"] for i in range(len(ds_test))]
            llm_eval_summary = run_llm_as_eval(
                outdir=outdir,
                rows_df=df[["id","file","true_label","pred_label"]].copy(),
                test_texts=test_texts,
                seed=args.seed,
                llm_model=args.llm_model,
                llm_sample=args.llm_sample,
                llm_temp=args.llm_temperature,
                llm_max_tokens=args.llm_max_tokens,
            )
        except Exception as e:
            print(f"[WARN] LLM-as-Eval failed: {e}")

    out_metrics = {"metrics": metrics, "bootstrap": ci}
    if llm_eval_summary is not None:
        out_metrics["llm_eval"] = llm_eval_summary

    with open(Path(outdir)/"metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2, ensure_ascii=False)

    print("Plotting confusion matrix and ROC/PR...")
    class_names = [str(c) for c in sorted(set(labels))]
    plot_confusion_matrix(metrics["confusion_matrix"], Path(outdir)/"confusion_matrix.png", class_names)
    if probs.shape[1] >= 2:
        plot_roc_pr(labels, probs, Path(outdir)/"roc_pr_curve.png")

    print("Done. Outputs in:", outdir)
    summary_payload = {
        "summary": {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "roc_auc": metrics.get("roc_auc"),
        }
    }
    if llm_eval_summary is not None:
        summary_payload["summary"]["llm_eval_agreement"] = llm_eval_summary["agreement_model_vs_llm"]
        summary_payload["summary"]["llm_acc_vs_gold"] = llm_eval_summary["llm_accuracy_vs_gold"]
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
