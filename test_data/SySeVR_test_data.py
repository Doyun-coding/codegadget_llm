#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_inline_labeled_sysevr.py

형식:
  [헤더 라인]
  [코드 ... 여러 줄]
  [라벨: 0 또는 1]
  ------------------------------
  [헤더 라인]
  [코드 ...]
  [라벨]
  ...

헤더 라인은 대략 아래 둘 중 하나처럼 가정(둘 다 지원):
  1) "88227/CWE195_..._32.c dataBuffer 46"
  2) "1 88227/CWE195_..._32.c dataBuffer 45"   # 맨 앞에 인덱스가 하나 더 있을 수 있음

출력:
  outdir/train.jsonl, validation.jsonl, test.jsonl
  각 라인은 {"id","file","text","label"} (label: 1=vuln, 0=non-vuln)

사용 예:
  python convert_inline_labeled_sysevr.py \
    --inputs /path/to/inline_labeled.txt /more/*.txt /a/folder \
    --outdir ./data/sysevr_test_data \
    --train-frac 0.8 --val-frac 0.1 --test-frac 0.1 --seed 42
"""

import argparse, sys, re, json, random
from pathlib import Path
from typing import List, Dict, Tuple

SEP_RE = re.compile(r"\n-{6,}\n")  # '------' 이상인 구분선
HEADER_RE_1 = re.compile(r'^(?P<path>\S+\.c)\s+(?P<var>\S+)\s+(?P<line>\d+)\s*$')
HEADER_RE_2 = re.compile(r'^\d+\s+(?P<path>\S+\.c)\s+(?P<var>\S+)\s+(?P<line>\d+)\s*$')

def find_input_files(inputs: List[Path]) -> List[Path]:
    out = []
    for p in inputs:
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out.extend([q for q in p.rglob("*") if q.is_file() and q.suffix.lower() in {".txt", ".svr", ".data"}])
    return sorted(out)

def parse_segment(seg_text: str) -> Dict:
    """
    세그먼트: [헤더] + [코드 ...] + [라벨]
    마지막 비어있지 않은 줄이 '0' 또는 '1'이어야 함.
    """
    lines = [ln for ln in seg_text.strip().splitlines() if ln.strip() != ""]
    if len(lines) < 2:
        return {}
    label_raw = lines[-1].strip()
    if label_raw not in {"0", "1"}:
        return {}

    label = int(label_raw)
    header = lines[0].strip()
    m = HEADER_RE_1.match(header) or HEADER_RE_2.match(header)
    if not m:
        # 헤더 파싱 실패 → 파일경로를 알 수 없으면 file은 빈 값으로 둠
        path = ""
        var = ""
        line = 0
        code_lines = lines[1:-1]
    else:
        path = m.group("path")
        var = m.group("var")
        line = int(m.group("line"))
        code_lines = lines[1:-1]

    text = "\n".join(code_lines).strip()
    rec = {
        "id": f"{path}:{line}" if path else f"inline:{hash(text) & 0xffffffff}",
        "file": path if path else "",
        "text": text,
        "label": label,
    }
    return rec

def parse_file(fp: Path) -> List[Dict]:
    text = fp.read_text(encoding="utf-8", errors="ignore")
    # 구분선으로 쪼개기. 구분선 없으면 파일 전체를 하나의 세그먼트로 처리 시도
    parts = SEP_RE.split(text) if SEP_RE.search(text) else [text]
    out = []
    for seg in parts:
        rec = parse_segment(seg)
        if rec and rec["text"]:
            out.append(rec)
    return out

def split_train_val_test(items: List[dict], train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    random.seed(seed)
    items = items[:]
    random.shuffle(items)
    n = len(items)
    n_train = int(round(n*train_frac))
    n_val = int(round(n*val_frac))
    if n_train + n_val > n - 1:
        n_val = max(0, n - n_train - 1)
    n_test = n - n_train - n_val
    return items[:n_train], items[n_train:n_train+n_val], items[n_train+n_val:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="인라인 라벨(.txt 등) 파일 또는 폴더(재귀)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    inputs = [Path(p) for p in args.inputs]
    files = find_input_files(inputs)
    if not files:
        print("[ERROR] no input files found", file=sys.stderr)
        sys.exit(2)

    records = []
    for f in files:
        recs = parse_file(f)
        records.extend(recs)

    if not records:
        print("[ERROR] no records parsed. 구분선/헤더/라벨 형식을 확인하세요.", file=sys.stderr)
        sys.exit(3)

    train, val, test = split_train_val_test(records, args.train_frac, args.val_frac, args.test_frac, args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    def dump(lst, path):
        with open(path, "w", encoding="utf-8") as w:
            for r in lst:
                # CodeBERT 학습 스크립트 호환 필드
                w.write(json.dumps({
                    "id": r["id"],
                    "file": r.get("file",""),
                    "text": r["text"],
                    "label": int(r["label"])
                }, ensure_ascii=False) + "\n")

    dump(train, outdir/"train.jsonl")
    dump(val, outdir/"validation.jsonl")
    dump(test, outdir/"test.jsonl")
    print(f"[DONE] train={len(train)} val={len(val)} test={len(test)} -> {outdir}")

if __name__ == "__main__":
    main()
