#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# normalize_gadgets.py — 싱크 중심 크롭 + 주석/공백 정리

"""
python normalize_gadgets.py \
  ./data/static/juliet_gadgets_smart_labeled.jsonl \
  ./data/static/juliet_gadgets_normalized.jsonl \
  --max-chars 2000 --center --sinks sinks.txt
"""

import json, re, sys, argparse
from pathlib import Path

DEFAULT_SINKS = {
    "strcpy","strncpy","strcat","strncat","stpcpy","strlcpy","strlcat",
    "wcscpy","wcsncpy","wcscat","wcsncat","lstrcpy","lstrcpyn","lstrcat","lstrncat",
    "memcpy","memmove","bcopy",
    "printf","fprintf","sprintf","snprintf","asprintf","vprintf","vfprintf","vsprintf","vsnprintf","vasprintf",
    "gets","fgets","scanf","sscanf","fscanf","vscanf","vfscanf","vsscanf",
    "recv","recvfrom","recvmsg","read","readv","send","sendto","sendmsg","write","writev",
    "system","popen","execl","execlp","execle","execv","execvp","execve"
}

def load_sinks(path: str | None):
    if not path:
        return DEFAULT_SINKS
    s = set(DEFAULT_SINKS)
    try:
        for ln in Path(path).read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                s.add(ln)
    except Exception:
        pass
    return s

def strip_and_collapse(code: str) -> str:
    if not code: return ""
    # 주석 제거
    code = re.sub(r'/\*.*?\*/', ' ', code, flags=re.S)
    code = re.sub(r'//.*?$', ' ', code, flags=re.M)
    # 개행/공백 정리
    code = re.sub(r'\r\n?', '\n', code)
    code = re.sub(r'\n\s*\n+', '\n', code)
    return code.strip()

def center_crop_around_sink(code: str, sinks: set[str], max_chars=2000) -> str:
    """싱크 등장 위치를 중심으로 max_chars 크롭 (guard + sink를 함께 남기기 위함)"""
    if len(code) <= max_chars:
        return code
    # 가장 앞쪽에 등장하는 싱크 오프셋 찾기
    positions = []
    for s in sinks:
        idx = code.find(s + "(") if "(" not in s else code.find(s)
        if idx >= 0:
            positions.append(idx)
    if not positions:
        return code[:max_chars]  # 싱크가 없으면 앞부분
    center = min(positions)
    half = max_chars // 2
    start = max(0, center - half)
    end = min(len(code), start + max_chars)
    return code[start:end]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--max-chars", type=int, default=2000)
    ap.add_argument("--center", action="store_true", help="sink-centered cropping")
    ap.add_argument("--sinks", default=None, help="extra sink list file")
    args = ap.parse_args()

    sinks = load_sinks(args.sinks)

    inp = Path(args.infile)
    out = Path(args.outfile)
    wrote = 0
    with inp.open("r", encoding="utf-8") as inf, out.open("w", encoding="utf-8") as outf:
        for line in inf:
            obj = json.loads(line)
            lab = obj.get("label", -1)
            if lab not in (0, 1):
                continue
            code = obj.get("gadget_raw") or obj.get("gadget_code") or obj.get("text") or ""
            code = strip_and_collapse(code)
            if args.center:
                code = center_crop_around_sink(code, sinks, max_chars=args.max_chars)
            else:
                if len(code) > args.max_chars:
                    code = code[:args.max_chars]
            out_obj = {
                "id": obj.get("id"),
                "file": obj.get("file"),
                "text": code,
                "label": int(lab),
            }
            outf.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            wrote += 1
    print(f"[DONE] normalized: {wrote} -> {out}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
    main()
