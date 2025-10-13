# static_train_codebart_gpu_v3.py
import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import inspect

MODEL = "microsoft/codebert-base"

def main():
    # 안전 옵션: 토크나이저 멀티스레드 경고/교착 방지
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # 디버그용(멈춤 추적) — 느려질 수 있으니 필요 시만 주석 해제
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print(f"[INFO] CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] No CUDA detected — training on CPU.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    def tokenize_fn(batch):
        # 1660 SUPER: 256 토큰이 속도/품질 균형. 더 빠르게 하려면 192/128로 줄여도 됨
        return tokenizer(batch["text"], truncation=True, max_length=256)

    dataset = load_dataset(
        "json",
        data_files={
            "train": "./data/static/v3/juliet_codebert_dataset/train.jsonl",
            "validation": "./data/static/v3/juliet_codebert_dataset/validation.jsonl",
            "test": "./data/static/v3/juliet_codebert_dataset/test.jsonl",
        },
    )

    def fix_label(example):
        example["label"] = int(example["label"])
        return example

    # 사전 가공(단일 프로세스: 윈도우 안전)
    dataset = dataset.map(fix_label)
    dataset = dataset.map(tokenize_fn, batched=True)
    dataset = dataset.map(lambda x: {"labels": x["label"]})
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 동적 패딩(텐서코어 정렬용 pad_to_multiple_of=8)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    # 라벨 맵핑(저장/추론 시 유용)
    id2label = {0: "nonvuln", 1: "vuln"}
    label2id = {"nonvuln": 0, "vuln": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    # 클래스 불균형 가중치
    train_labels = np.array(dataset["train"]["labels"])
    n0 = int((train_labels == 0).sum())
    n1 = int((train_labels == 1).sum())
    class_weights = None
    if n0 > 0 and n1 > 0:
        w0 = (n0 + n1) / (2.0 * n0)
        w1 = (n0 + n1) / (2.0 * n1)
        class_weights = torch.tensor([w0, w1])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
        }

    # ⚠ 윈도우 안전 + 정지 회피 세팅
    training_args = TrainingArguments(
        output_dir="./static-codebert",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,  # 체크포인트 2개만 보관
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        num_train_epochs=4,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        optim="adamw_torch",
        report_to="none",

        # Windows 안정성용(윈도우면 그대로 두세요; 맥/리눅스는 생략해도 무방)
        dataloader_num_workers=0,
        dataloader_pin_memory=False,

        # 혼합정밀: CUDA만 True로, MPS/CPU는 False
        fp16=torch.cuda.is_available(),
        # Ampere 이상이면 bf16도 고려 (원하면 활성화)
        # bf16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
    )

    class WeightedTrainer(Trainer):
        def __init__(self, class_weights_tensor=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
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

    # HF 버전 호환(토크나이저 전달 경고 제거)
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],   # eval_strategy="no"라 학습 중엔 실행 안 됨
        compute_metrics=compute_metrics,
        data_collator=collator,
        class_weights_tensor=class_weights,
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedTrainer(**trainer_kwargs)

    print(">>> start training")
    trainer.train()

    # ====== 👇 학습 결과 저장 (모델 + 토크나이저 + 상태) ======
    save_dir = training_args.output_dir  # "./static-codebert"
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)               # 모델 가중치/구성 저장
    tokenizer.save_pretrained(save_dir)        # 토크나이저 파일 저장
    trainer.save_state()                       # (선택) optimizer/scheduler state 등 저장
    print(f"[SAVED] Model & tokenizer saved to: {save_dir}")

    print(">>> evaluate on test")
    res = trainer.evaluate(dataset["test"])
    print("Test eval:", res)

if __name__ == "__main__":
    main()
