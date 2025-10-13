#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# relabel_smart_plus.py — 함수/로컬 윈도우 + 싱크/가드 휴리스틱 기반의 보수적 재라벨링

import argparse, json, re, os
from pathlib import Path
from collections import Counter

# ---- Juliet 힌트 패턴 ----
BAD_PATTERNS = [
    re.compile(r'\bvoid\s+bad\b', re.I),
    re.compile(r'\bbadSink\b', re.I),
    re.compile(r'\bbad\s*\(', re.I),
    re.compile(r'\bPOTENTIAL\s+FL?AW\b', re.I),
]
GOOD_PATTERNS = [
    re.compile(r'\bvoid\s+good\b', re.I),
    re.compile(r'\bgoodB2G\b', re.I),
    re.compile(r'\bgoodG2B\b', re.I),
    re.compile(r'\bgood\s*\(', re.I),
]

# ---- 싱크/가드 휴리스틱 ----
SINKS = {
    "strcpy","strncpy","strcat","strncat","memcpy","memmove",
    "printf","fprintf","sprintf","snprintf","asprintf","vprintf","vfprintf","vsprintf","vsnprintf","vasprintf",
    "gets","fgets","scanf","sscanf","fscanf",
    "recv","read","send","write","system","popen","execl","execlp","execle","execv","execvp","execve"
}
WORD = r'[A-Za-z_]\w*'
STR_LIT = r'"[^"\\]*(?:\\.[^"\\]*)*"'
SAFE_CONST_FORMAT = re.compile(rf'\b(?:v?f?printf|printf)\s*\(\s{STR_LIT}', re.M)
SAFE_SNPRINTF = re.compile(rf'\bsn?printf\s*\(\s*{WORD}\s*,\s*(?:sizeof\s*\(\s*{WORD}\s*\)|\d+|{WORD})\s*,', re.M)
SAFE_STRNCPY = re.compile(rf'\bstrncpy\s*\(\s*{WORD}\s*,\s*{WORD}\s*,\s*(?:sizeof\s*\(\s*{WORD}\s*\)|\d+|{WORD})\s*\)', re.M)
LEN_GUARD_IF = re.compile(rf'\bif\s*\([^)]*(<=|<|>=|>)\s*(?:sizeof\s*\(\s*{WORD}\s*\)|{WORD})[^)]*\)', re.M)

FUNC_SIG = re.compile(
    r'^\s*(?:[A-Za-z_][\w\s\*\(\)]*?)\s+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{',
    re.M
)

COMMENT_BLOCK = re.compile(r'/\*.*?\*/', re.S)
COMMENT_LINE  = re.compile(r'//.*?$', re.M)

def strip_comments(s: str) -> str:
    if not s: return s
    s = COMMENT_BLOCK.sub(' ', s)
    s = COMMENT_LINE.sub(' ', s)
    return s

def contains_sink(s: str) -> bool:
    if not s: return False
    for name in SINKS:
        if re.search(rf'\b{name}\b', s):
            return True
    return False

def looks_guarded(s: str) -> bool:
    # 대표적 ‘안전 패턴’ 탐지
    if SAFE_CONST_FORMAT.search(s): return True
    if SAFE_SNPRINTF.search(s):     return True
    if SAFE_STRNCPY.search(s):      return True
    if LEN_GUARD_IF.search(s):      return True
    return False

def vuln_heuristic(code: str) -> int | None:
    """
    휴리스틱: 싱크가 있고 가드가 없다 → 1, 싱크가 없거나 충분히 가드 → 0, 결정불가 None
    """
    if not code: return None
    has = contains_sink(code)
    if not has:
        return 0  # 싱크 자체가 없으면 비취약으로 가정
    # 싱크가 있는데 ‘명백한’ 가드가 보이면 0, 아니면 1로 보수적 분류
    return 0 if looks_guarded(code) else 1

def check_patterns(text: str, patterns) -> bool:
    if not text: return False
    for p in patterns:
        if p.search(text):
            return True
    return False

def resolve_path(fp: str, juliet_root: str | None) -> str | None:
    if not fp: return None
    if os.path.isabs(fp): return fp if os.path.exists(fp) else None
    if juliet_root:
        cand = os.path.join(juliet_root, fp)
        if os.path.exists(cand): return cand
    if os.path.exists(fp): return fp
    cand = os.path.normpath(fp)
    return cand if os.path.exists(cand) else None

def find_enclosing_function(full_text: str, line_no: int):
    lines = full_text.splitlines()
    if not (1 <= line_no <= len(lines)): return None
    upto = "\n".join(lines[:line_no])
    matches = list(FUNC_SIG.finditer(upto))
    if not matches: return None
    m = matches[-1]
    start_line = upto[:m.start()].count("\n") + 1
    depth = 0
    start_idx = m.start()
    end_line = len(lines)
    for i, ch in enumerate(full_text[start_idx:], start_idx):
        if ch == "{": depth += 1
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

def decide_label(code_raw: str, prefer_bad: bool, use_comments: bool, use_guards: bool) -> int | None:
    code = code_raw if use_comments else strip_comments(code_raw)

    # 1) 휴리스틱(싱크/가드)로 먼저 시도
    if use_guards:
        h = vuln_heuristic(code)
        if h is not None:
            return h

    # 2) Juliet 힌트 보조
    is_bad = check_patterns(code, BAD_PATTERNS)
    is_good = check_patterns(code, GOOD_PATTERNS)
    if is_bad and not is_good:  return 1
    if is_good and not is_bad:  return 0
    if is_bad and is_good:      return 1 if prefer_bad else 0

    return None  # 결정 불가

def main():
    ap = argparse.ArgumentParser(description="Re-label gadgets with local/function scope + sink/guard heuristics")
    ap.add_argument("input", help="input jsonl")
    ap.add_argument("output", nargs="?", default=None, help="output jsonl")
    ap.add_argument("--juliet-root", default=None)
    ap.add_argument("--force", action="store_true", help="overwrite existing label or label=-1")
    ap.add_argument("--prefer-bad", action="store_true")
    ap.add_argument("--win-lines", type=int, default=64)
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--keep-comments", action="store_true", help="주석을 보존하고 판단(기본은 제거)")
    ap.add_argument("--no-guards", action="store_true", help="싱크/가드 휴리스틱 비활성화")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output) if args.output else inp.with_name(inp.stem + "_smart_plus.jsonl")

    cnt = Counter(); wrote = 0
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
                # (a) 가젯 우선
                gadget = obj.get("gadget_raw") or obj.get("gadget_code") or obj.get("text") or ""
                label = decide_label(gadget, args.prefer_bad, args.keep_comments, not args.no_guards)

                # (b) 함수/로컬 윈도우 보조
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
                                label = decide_label(func_text, args.prefer_bad, args.keep_comments, not args.no_guards)
                            if label is None:
                                local = slice_local_window(full, sb, se, args.win_lines)
                                label = decide_label(local, args.prefer_bad, args.keep_comments, not args.no_guards)
                            if label is None and not args.local_only:
                                label = decide_label(full, args.prefer_bad, args.keep_comments, not args.no_guards)
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
