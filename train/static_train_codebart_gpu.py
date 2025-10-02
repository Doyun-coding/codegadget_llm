# train_codebert.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

MODEL = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize_fn(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=512)

dataset = load_dataset('json', data_files={
    'train': './data/static/juliet_codebert_dataset/train.jsonl',
    'validation': './data/static/juliet_codebert_dataset/validation.jsonl',
    'test': './data/static/juliet_codebert_dataset/test.jsonl'
})

# ensure label field exists as int
def fix_label(example):
    example['label'] = int(example['label'])
    return example

dataset = dataset.map(fix_label)
dataset = dataset.map(tokenize_fn, batched=True, remove_columns=[c for c in dataset['train'].column_names if c not in ['text', 'label']])
# make sure labels column exists for Trainer as 'labels'
dataset = dataset.map(lambda x: {'labels': x['label']})
dataset.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

# load model
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

# compute class weights (cpu tensor; we'll move to device in compute_loss)
train_labels = np.array(dataset['train']['labels'])
n0 = (train_labels==0).sum()
n1 = (train_labels==1).sum()
if n0 == 0 or n1 == 0:
    print("[WARN] One of the classes has zero samples. Class weights set to None.")
    class_weights = None
else:
    w0 = (n0 + n1) / (2.0 * n0)
    w1 = (n0 + n1) / (2.0 * n1)
    class_weights = torch.tensor([w0, w1])  # keep on cpu; moved to device later

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
    }

# Detect CUDA (GPU) availability
use_cuda = torch.cuda.is_available()
if use_cuda:
    print(f"[INFO] CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    # Check for MPS on macOS
    use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if use_mps:
        print("[INFO] MPS (Apple Silicon) is available; Trainer will try to use MPS device.")
    else:
        print("[INFO] CUDA not available. Training will run on CPU.")

# TrainingArguments: enable fp16 if CUDA available (or you can keep False)
training_args = TrainingArguments(
    output_dir="./static-codebert",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,   # adjust if OOM on your GPU
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True if use_cuda else False,   # use mixed precision on CUDA for speed/memory
    fp16_opt_level="O1",                # optional, HF will accept True/False; opt_level used in some envs
)

class WeightedTrainer(Trainer):
    """
    WeightedTrainer overrides compute_loss to apply class weights.
    Accepts extra kwargs forwarded by Trainer.train() (keeps compatibility).
    """
    def __init__(self, class_weights_tensor=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # class_weights_tensor should be a torch.Tensor (cpu) or None
        self.class_weights = class_weights_tensor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # labels might be under "labels"
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs.get("logits")
        # build loss
        if self.class_weights is not None:
            # move class weights to same device as logits
            cw = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=cw)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Build trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    class_weights_tensor=class_weights
)

# Optional: pin memory advice
if use_cuda:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Start training
trainer.train()

# Evaluate on test set
res = trainer.evaluate(dataset['test'])
print("Test eval:", res)
