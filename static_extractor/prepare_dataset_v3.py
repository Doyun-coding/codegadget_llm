#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# prepare_dataset_v3.py — 그룹키 옵션 + 스플릿 리포트/가중치 저장

import json, argparse, random, hashlib
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.model_selection import GroupShuffleSplit

def hash_text(t: str) -> str:
    import re
    t = re.sub(r'\s+', ' ', (t or '')).strip()
    return hashlib.sha256(t.encode('utf-8')).hexdigest()

def template_key(file_path: str) -> str:
    p = Path(file_path or "")
    return f"{p.parent.name}/{p.stem}"

def group_key(rec, mode: str):
    if mode == "template": return template_key(rec.get("file",""))
    if mode == "dir":      return Path(rec.get("file","")).parent.as_posix()
    if mode == "file":     return rec.get("file","")
    return template_key(rec.get("file",""))  # default

def dedupe_by_text(records):
    seen = set(); out = []
    for r in records:
        h = hash_text(r.get("text",""))
        if h in seen: continue
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
    if not idxs: return idxs
    rnd = random.Random(seed)
    ones  = [i for i in idxs if int(records[i]["label"]) == 1]
    zeros = [i for i in idxs if int(records[i]["label"]) == 0]
    n = len(idxs)
    cur_ratio = len(ones)/n
    low, high = target - tol, target + tol
    if low <= cur_ratio <= high: return idxs
    tgt_ones  = int(round(target * n))
    tgt_zeros = n - tgt_ones
    if len(ones) > tgt_ones:  rnd.shuffle(ones);  ones  = ones[:max(1, tgt_ones)]
    if len(zeros) > tgt_zeros: rnd.shuffle(zeros); zeros = zeros[:max(1, tgt_zeros)]
    new = ones + zeros
    rnd.shuffle(new)
    return new

def write_split(name, records, idxs, outdir: Path):
    p = outdir / f"{name}.jsonl"
    with p.open("w", encoding="utf-8") as w:
        for i in idxs:
            w.write(json.dumps(records[i], ensure_ascii=False) + "\n")
    cnt = Counter(int(records[i]["label"]) for i in idxs)
    print(f"Wrote {p}  n={len(idxs)}  label1={cnt.get(1,0)}  label0={cnt.get(0,0)}")
    return p, cnt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--outdir", default="./data/codebert_dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--balance-target", type=float, default=0.5)
    ap.add_argument("--balance-tol", type=float, default=0.1)
    ap.add_argument("--group-by", choices=["template","dir","file"], default="template")
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    inp = Path(args.infile)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    raw = [json.loads(l) for l in inp.read_text(encoding="utf-8").splitlines()]
    records = [r for r in raw if int(r.get("label",-1)) in (0,1)]
    print(f"Loaded(valid): {len(records)}")

    records = dedupe_by_text(records)
    print(f"After dedupe: {len(records)}")

    groups = [group_key(r, args.group_by) for r in records]
    labels = [int(r.get("label",0)) for r in records]

    train_idx, val_idx, test_idx = group_split(records, groups, labels,
                                               seed=args.seed,
                                               test_size=args.test_size,
                                               val_size=args.val_size)
    print(f"Split (pre-balance): train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_idx = rebalance_split(records, train_idx, args.balance_target, args.balance_tol, seed=args.seed)
    val_idx   = rebalance_split(records, val_idx,   args.balance_target, args.balance_tol, seed=args.seed)
    test_idx  = rebalance_split(records, test_idx,  args.balance_target, args.balance_tol, seed=args.seed)
    print(f"Split (post-balance): train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # 저장 + 통계
    tr_p, tr_cnt = write_split("train", records, train_idx, outdir)
    va_p, va_cnt = write_split("validation", records, val_idx, outdir)
    te_p, te_cnt = write_split("test", records, test_idx, outdir)

    # 클래스 가중치/싱크 히스토그램(학습 분석용)
    total = len(train_idx)
    w0 = total / (2 * (tr_cnt.get(0,1)))
    w1 = total / (2 * (tr_cnt.get(1,1)))
    (outdir / "class_weights.json").write_text(json.dumps({"0": w0, "1": w1}, indent=2), encoding="utf-8")

    print("[DONE]", outdir)

if __name__ == "__main__":
    main()
