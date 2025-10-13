#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# relabel_smart.py — 함수 범위/로컬 윈도우 기반의 보수적 재라벨링

"""
사용 예:
python relabel_smart.py \
  ./static_extractor/juliet_gadgets.jsonl \
  ./data/juliet_gadgets_smart_labeled.jsonl \
  --juliet-root ../C/testcases \
  --win-lines 64 --local-only \
  --prefer-bad
"""

import argparse, json, re, os
from pathlib import Path
from collections import Counter

# ---- 라벨링 패턴 (주석/함수명/식별자) ----
BAD_PATTERNS = [
    re.compile(r'\bvoid\s+bad\b', re.I),
    re.compile(r'\bbadSink\b', re.I),
    re.compile(r'\bbad\s*\(', re.I),
    re.compile(r'\bPOTENTIAL\s+FL?AW\b', re.I),
    re.compile(r'\bBadSink\b', re.I),
]
GOOD_PATTERNS = [
    re.compile(r'\bvoid\s+good\b', re.I),
    re.compile(r'\bgoodB2G\b', re.I),
    re.compile(r'\bgoodG2B\b', re.I),
    re.compile(r'\bgood\s*\(', re.I),
    # Juliet에서 자주 등장하는 스위치 매크로(있다고 무조건 good은 아님)
    re.compile(r'\bOMITBAD\b', re.I),
    re.compile(r'\bOMITGOOD\b', re.I),
]

FUNC_SIG = re.compile(
    r'^\s*(?:[A-Za-z_][\w\s\*\(\)]*?)\s+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{',
    re.M
)

def check_patterns(text: str, patterns) -> bool:
    if not text:
        return False
    for p in patterns:
        if p.search(text):
            return True
    return False

def resolve_path(fp: str, juliet_root: str | None) -> str | None:
    if not fp:
        return None
    if os.path.isabs(fp):
        return fp if os.path.exists(fp) else None
    if juliet_root:
        cand = os.path.join(juliet_root, fp)
        if os.path.exists(cand):
            return cand
    # CWD 기준
    if os.path.exists(fp):
        return fp
    # normalize
    cand = os.path.normpath(fp)
    return cand if os.path.exists(cand) else None

def find_enclosing_function(full_text: str, line_no: int) -> tuple[str, int, int] | None:
    """
    line_no(1-based)가 포함된 함수 블록 텍스트와 시작/끝 라인을 추정해 반환.
    실패하면 None.
    """
    lines = full_text.splitlines()
    if not (1 <= line_no <= len(lines)):
        return None
    upto = "\n".join(lines[:line_no])
    matches = list(FUNC_SIG.finditer(upto))
    if not matches:
        return None
    m = matches[-1]
    # 함수 시작 라인
    start_line = upto[:m.start()].count("\n") + 1

    # 간단한 중괄호 카운팅으로 함수 끝 추정
    depth = 0
    start_idx = m.start()
    end_line = len(lines)
    for i, ch in enumerate(full_text[start_idx:], start_idx):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_line = full_text[:i].count("\n") + 1
                break

    func_text = "\n".join(lines[start_line-1:end_line])
    return func_text, start_line, end_line

def slice_local_window(full_text: str, begin_line: int, end_line: int, win_lines: int) -> str:
    lines = full_text.splitlines()
    n = len(lines)
    mid = max(1, min((begin_line + end_line) // 2, n))
    half = max(1, win_lines // 2)
    s = max(1, mid - half)
    e = min(n, mid + half)
    return "\n".join(lines[s-1:e])

def decide_label(code: str, prefer_bad: bool) -> int | None:
    is_bad = check_patterns(code, BAD_PATTERNS)
    is_good = check_patterns(code, GOOD_PATTERNS)
    if is_bad and not is_good:
        return 1
    if is_good and not is_bad:
        return 0
    if is_bad and is_good:
        return 1 if prefer_bad else 0
    return None  # 결정 불가

def main():
    ap = argparse.ArgumentParser(description="Re-label gadgets using local/function scope only")
    ap.add_argument("input", help="input jsonl")
    ap.add_argument("output", nargs="?", default=None, help="output jsonl")
    ap.add_argument("--juliet-root", default=None, help="root dir to resolve relative file paths")
    ap.add_argument("--force", action="store_true", help="overwrite existing label or label=-1")
    ap.add_argument("--prefer-bad", action="store_true", help="when ambiguous, prefer label=1")
    ap.add_argument("--win-lines", type=int, default=64, help="local window +/- lines used when function parse fails")
    ap.add_argument("--local-only", action="store_true", help="do NOT fall back to file-wide search")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output) if args.output else inp.with_name(inp.stem + "_smart_labeled.jsonl")

    cnt = Counter()
    wrote = 0

    with inp.open("r", encoding="utf-8") as inf, out.open("w", encoding="utf-8") as outf:
        for i, line in enumerate(inf, 1):
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] skip malformed line {i}: {e}")
                continue

            orig_label = obj.get("label", None)
            label = orig_label

            if orig_label is None or args.force or int(orig_label) == -1:
                # 1) 가젯 텍스트 우선
                gadget = obj.get("gadget_raw") or obj.get("gadget_code") or obj.get("text") or ""
                label = decide_label(gadget, args.prefer_bad)
                # 2) 함수 범위 / 로컬 윈도우에서 보조 판단
                if label is None:
                    fp = obj.get("file", "")
                    sb = int(obj.get("slice_begin", 1))
                    se = int(obj.get("slice_end", sb))
                    realp = resolve_path(fp, args.juliet_root)
                    if realp and os.path.exists(realp):
                        try:
                            full = Path(realp).read_text(encoding="utf-8", errors="ignore")
                            found = find_enclosing_function(full, sb)
                            if found:
                                func_text, _, _ = found
                                label = decide_label(func_text, args.prefer_bad)
                            if label is None:  # 함수에서 못 찾으면 로컬 윈도우
                                local = slice_local_window(full, sb, se, args.win_lines)
                                label = decide_label(local, args.prefer_bad)
                            if label is None and not args.local_only:
                                # 최후 보루: 파일 전체 검색 (기본 off)
                                label = decide_label(full, args.prefer_bad)
                            if label is None:
                                label = -1
                        except Exception:
                            label = -1
                    else:
                        label = -1

            obj["label"] = int(label) if label is not None else -1
            cnt[obj["label"]] += 1
            outf.write(json.dumps(obj, ensure_ascii=False) + "\n")
            wrote += 1

    print(f"[DONE] wrote: {wrote} -> {out}")
    print("Label counts:", dict(cnt))

if __name__ == "__main__":
    main()
