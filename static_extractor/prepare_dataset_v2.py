#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# prepare_dataset_v2.py — 텍스트 중복 제거, 그룹 홀드아웃 분할, split 후 과다 클래스 다운샘플

"""
python prepare_dataset_v2.py \
  ./data/static/strict/juliet_gadgets_normalized.jsonl \
  --outdir ./data/static/strict/juliet_codebert_dataset \
  --seed 42 --test_size 0.1 --val_size 0.1 \
  --balance-target 0.5 --balance-tol 0.1
"""

import json, argparse, random, hashlib
from pathlib import Path
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit

def hash_text(t: str) -> str:
    return hashlib.sha256((t or "").encode("utf-8")).hexdigest()

def template_key(file_path: str) -> str:
    p = Path(file_path or "")
    name = p.stem
    parent = p.parent.name
    return f"{parent}/{name}"

def dedupe_by_text(records):
    seen = set(); out = []
    for r in records:
        h = hash_text(r.get("text", ""))
        if h in seen:
            continue
        seen.add(h); out.append(r)
    return out

def group_split(records, groups, labels, seed, test_size, val_size):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size + val_size, random_state=seed)
    train_idx_np, hold_idx_np = next(gss.split(records, labels, groups))
    train_idx = train_idx_np.tolist()
    hold_idx  = hold_idx_np.tolist()

    hold_records = [records[i] for i in hold_idx]
    hold_groups  = [groups[i]  for i in hold_idx]
    hold_labels  = [labels[i]  for i in hold_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_size/(test_size+val_size), random_state=seed)
    v_rel_np, t_rel_np = next(gss2.split(hold_records, hold_labels, hold_groups))
    val_idx  = [hold_idx[i] for i in v_rel_np.tolist()]
    test_idx = [hold_idx[i] for i in t_rel_np.tolist()]
    return train_idx, val_idx, test_idx

def rebalance_split(records, idxs, target=0.5, tol=0.1, seed=42):
    """과다 클래스 다운샘플 (split 내부에서만). idxs는 list 가정."""
    if idxs is None or len(idxs) == 0:
        return idxs
    rnd = random.Random(seed)
    ones  = [i for i in idxs if int(records[i]["label"]) == 1]
    zeros = [i for i in idxs if int(records[i]["label"]) == 0]
    n = len(idxs)
    cur_ratio = len(ones)/n
    low, high = target - tol, target + tol
    if low <= cur_ratio <= high:
        return idxs  # 이미 균형

    tgt_ones  = int(round(target * n))
    tgt_zeros = n - tgt_ones

    if len(ones) > tgt_ones:
        rnd.shuffle(ones);  ones  = ones[:max(1, tgt_ones)]
    if len(zeros) > tgt_zeros:
        rnd.shuffle(zeros); zeros = zeros[:max(1, tgt_zeros)]

    new = ones + zeros
    rnd.shuffle(new)
    return new

def write_split(name, records, idxs, outdir: Path):
    p = outdir / f"{name}.jsonl"
    with p.open("w", encoding="utf-8") as w:
        for i in idxs:
            w.write(json.dumps(records[i], ensure_ascii=False) + "\n")
    cnt = Counter(int(records[i]["label"]) for i in idxs)
    print(f"Wrote {p} count={len(idxs)}  label1={cnt.get(1,0)}  label0={cnt.get(0,0)}")
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--outdir", default="./data/codebert_dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--balance-target", type=float, default=0.5)
    ap.add_argument("--balance-tol", type=float, default=0.1)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    inp = Path(args.infile)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) 로드 + 유효 라벨만
    raw_lines = inp.read_text(encoding="utf-8").splitlines()
    raw = [json.loads(l) for l in raw_lines]
    records = [r for r in raw if int(r.get("label", -1)) in (0, 1)]
    print(f"Loaded: {len(records)} (valid labels)")

    # 2) 텍스트 중복 제거
    records = dedupe_by_text(records)
    print(f"After dedupe: {len(records)}")

    # 3) 그룹 키(템플릿 유사 단위) 생성
    groups = [template_key(r.get("file","")) for r in records]
    labels = [int(r.get("label",0)) for r in records]

    # 4) 그룹 홀드아웃 분할
    train_idx, val_idx, test_idx = group_split(records, groups, labels,
                                               seed=args.seed,
                                               test_size=args.test_size,
                                               val_size=args.val_size)

    print(f"Split sizes (pre-balance): train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # 5) 각 split 내부 라벨 균형 보정(다운샘플)
    train_idx = rebalance_split(records, train_idx, args.balance_target, args.balance_tol, seed=args.seed)
    val_idx   = rebalance_split(records, val_idx,   args.balance_target, args.balance_tol, seed=args.seed)
    test_idx  = rebalance_split(records, test_idx,  args.balance_target, args.balance_tol, seed=args.seed)

    print(f"Split sizes (post-balance): train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # 6) 저장
    write_split("train", records, train_idx, outdir)
    write_split("validation", records, val_idx, outdir)
    write_split("test", records, test_idx, outdir)

    print("[DONE]", outdir)

if __name__ == "__main__":
    main()
