# train_llm_codebert.py
"""
python train_llm_codebert.py \
  --dataset_dir ./data/llm/juliet_codebert_dataset \
  --output_dir ./llm-codebert
"""
import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import argparse
import inspect

MODEL_NAME = "microsoft/codebert-base"

def parse_args():
    ap = argparse.ArgumentParser(description="Train CodeBERT on LLM-extracted Juliet dataset")
    ap.add_argument("--dataset_dir", type=str, required=True,
                    help="Directory containing train.jsonl/validation.jsonl/test.jsonl")
    ap.add_argument("--output_dir", type=str, default="./llm-codebert",
                    help="Where to save the trained model")

    # ★ static과 동일한 기본값
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--train_bs", type=int, default=16)
    ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    # 추가 옵션(필요시 조정)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--logging_steps", type=int, default=100)
    ap.add_argument("--pad_to_multiple_of", type=int, default=8)
    return ap.parse_args()

def main():
    args = parse_args()

    # 안전 옵션
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print(f"[INFO] CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] No CUDA detected — training on CPU.")

    dataset_dir = Path(args.dataset_dir)
    train_p = dataset_dir / "train.jsonl"
    val_p   = dataset_dir / "validation.jsonl"
    test_p  = dataset_dir / "test.jsonl"

    if not train_p.exists():
        print(f"[ERROR] train.jsonl not found in {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    # 데이터 로드
    data_files = {"train": str(train_p)}
    if val_p.exists(): data_files["validation"] = str(val_p)
    if test_p.exists(): data_files["test"] = str(test_p)

    dataset = load_dataset("json", data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def fix_label(example):
        example["label"] = int(example["label"])
        return example

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    # 전처리
    dataset = dataset.map(fix_label)
    dataset = dataset.map(tokenize_fn, batched=True)
    dataset = dataset.map(lambda x: {"labels": x["label"]})

    # 텐서 포맷
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 동적 패딩(collator)
    collator = DataCollatorWithPadding(tokenizer=tokenizer,
                                       pad_to_multiple_of=args.pad_to_multiple_of)

    # 라벨 매핑(저장/추론 시 유용)
    id2label = {0: "nonvuln", 1: "vuln"}
    label2id = {"nonvuln": 0, "vuln": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    # 클래스 가중치 (train split 기준)
    if "train" in dataset:
        train_labels = np.array(dataset["train"]["labels"])
        n0 = int((train_labels == 0).sum())
        n1 = int((train_labels == 1).sum())
        if n0 > 0 and n1 > 0:
            w0 = (n0 + n1) / (2.0 * n0)
            w1 = (n0 + n1) / (2.0 * n1)
            class_weights = torch.tensor([w0, w1])
        else:
            print("[WARN] One class has zero samples; class weighting disabled.")
            class_weights = None
    else:
        class_weights = None

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall":    recall_score(labels, preds, zero_division=0),
            "f1":        f1_score(labels, preds, zero_division=0),
        }

    # Windows 안전/정지 회피 + static과 같은 전략(훈련 중 평가는 끔)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="no",                 # 학습 중 평가 off (static과 동일)
        save_strategy="no",                 # 중간 체크포인트 저장 off
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,

        logging_dir=str(Path(args.output_dir) / "logs"),
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        disable_tqdm=False,
        report_to="none",

        dataloader_num_workers=0,           # Windows 멀티프로세싱 이슈 회피
        dataloader_pin_memory=False,        # 일부 환경에서 멈춤 방지

        optim="adamw_torch",                # fused 옵티마이저는 특정 드라이버/윈도우에서 이슈 가능
        fp16=True if use_cuda else False,
        seed=args.seed,
    )

    class WeightedTrainer(Trainer):
        def __init__(self, class_weights_tensor=None, *targs, **tkwargs):
            super().__init__(*targs, **tkwargs)
            self.class_weights = class_weights_tensor

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs.get("logits")
            if self.class_weights is not None:
                cw = self.class_weights.to(logits.device)
                loss_fct = torch.nn.CrossEntropyLoss(weight=cw)
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    # HF 버전 경고 제거(tokenizer vs processing_class)
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        compute_metrics=compute_metrics,
        data_collator=collator,
        class_weights_tensor=class_weights,
    )
    if "validation" in dataset:
        trainer_kwargs["eval_dataset"] = dataset["validation"]
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedTrainer(**trainer_kwargs)

    print(">>> start training")
    trainer.train()

    # 모델/토크나이저/상태 저장 (체크포인트 전략 off여도 수동 저장)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(outdir))
    tokenizer.save_pretrained(str(outdir))
    trainer.save_state()
    print(f"[SAVED] Model & tokenizer saved to: {outdir}")

    # 최종 평가(Validation, Test)
    if "validation" in dataset:
        print(">>> evaluate on validation")
        val_res = trainer.evaluate(dataset["validation"])
        print("Validation:", val_res)

    if "test" in dataset:
        print(">>> evaluate on test")
        test_res = trainer.evaluate(dataset["test"])
        print("Test:", test_res)

if __name__ == "__main__":
    main()
