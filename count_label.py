import json

file_path = "./data/static/juliet_codebert_dataset/train.jsonl"
count_label0 = 0

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if int(data.get("label", -1)) == 0:  # "label"이 0이면 카운트
            count_label0 += 1

print(f'Number of samples with label 0: {count_label0}')