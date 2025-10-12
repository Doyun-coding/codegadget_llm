# prepare_dataset.py

"""
 python ./static_extractor/prepare_dataset.py \
./data/static/juliet_gadgets_normalized.jsonl \
--outdir \
./data/static/juliet_codebert_dataset
"""

import json, hashlib, argparse, random
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit

def hash_text(t):
    import hashlib
    return hashlib.sha256(t.encode('utf-8')).hexdigest() if t else "EMPTY"

ap = argparse.ArgumentParser()
ap.add_argument("infile")
ap.add_argument("--outdir", default="./data/codebert_dataset")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--test_size", type=float, default=0.1)
ap.add_argument("--val_size", type=float, default=0.1)
args = ap.parse_args()

inp = Path(args.infile)
outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

# read and dedupe by text
seen = set()
records = []
with inp.open('r',encoding='utf-8') as f:
    for l in f:
        obj = json.loads(l)
        t = obj.get('text','')
        h = hash_text(t)
        if h in seen:
            continue
        seen.add(h)
        records.append(obj)

print("After dedupe:", len(records))

# prepare groups (use file path as group)
groups = [r.get('file','') for r in records]
labels = [r.get('label',0) for r in records]

gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size + args.val_size, random_state=args.seed)
train_idx, hold_idx = next(gss.split(records, labels, groups))
# split hold into val/test
hold_records = [records[i] for i in hold_idx]
hold_groups = [groups[i] for i in hold_idx]
hold_labels = [labels[i] for i in hold_idx]
gss2 = GroupShuffleSplit(n_splits=1, test_size=args.test_size/(args.test_size+args.val_size), random_state=args.seed)
val_idx_rel, test_idx_rel = next(gss2.split(hold_records, hold_labels, hold_groups))
val_idx = [hold_idx[i] for i in val_idx_rel]
test_idx = [hold_idx[i] for i in test_idx_rel]

def write_idxs(name, idxs):
    p = outdir / f"{name}.jsonl"
    with p.open('w',encoding='utf-8') as w:
        for i in idxs:
            w.write(json.dumps(records[i], ensure_ascii=False)+"\n")
    print("Wrote", p, "count", len(idxs))

write_idxs("train", train_idx)
write_idxs("validation", val_idx)
write_idxs("test", test_idx)
