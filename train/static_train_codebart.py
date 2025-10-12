# train_codebert.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

MODEL = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize_fn(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=512)

dataset = load_dataset('json', data_files={
    'train': './data/static/juliet_codebert_dataset/train.jsonl',
    'validation': './data/static/juliet_codebert_dataset/validation.jsonl',
    'test': './data/static/juliet_codebert_dataset/test.jsonl'
})

dataset = dataset.map(lambda x: {'label': x['label']})
dataset = dataset.map(tokenize_fn, batched=True)
dataset.set_format(type='torch', columns=['input_ids','attention_mask','label'])

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

# compute class weights
train_labels = np.array(dataset['train']['label'])
n0 = (train_labels==0).sum()
n1 = (train_labels==1).sum()
w0 = (n0 + n1) / (2.0 * n0)
w1 = (n0 + n1) / (2.0 * n1)
class_weights = torch.tensor([w0,w1]).to('cpu')  # pass to compute_loss if needed

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'f1': f1_score(labels, preds),
    }

training_args = TrainingArguments(
    output_dir="./static-codebert",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=False,
)


class WeightedTrainer(Trainer):
    """
    WeightedTrainer overrides compute_loss to apply class weights.
    Accepts extra kwargs forwarded by Trainer.train() (e.g., num_items_in_batch).
    """
    def __init__(self, class_weights_tensor=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # class_weights_tensor should be a torch.Tensor (cpu); can be None
        self.class_weights = class_weights_tensor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss using CrossEntropy with optional class weights.
        Accepts **kwargs to be compatible with HF Trainer changes.
        """
        labels = inputs.get("labels")
        # forward
        outputs = model(**inputs)
        # outputs can be ModelOutput with .logits or a dict
        logits = outputs.logits if hasattr(outputs, "logits") else outputs.get("logits")
        # ensure weights on same device as logits
        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=cw)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

trainer.train()
trainer.evaluate(dataset['test'])
