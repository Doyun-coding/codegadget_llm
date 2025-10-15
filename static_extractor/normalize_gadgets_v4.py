#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# normalize_gadgets_plus.py — 싱크 중심 라인 윈도우 + 주석/공백 정리

import json, re, sys, argparse
from pathlib import Path

DEFAULT_SINKS = {
    "strcpy","strncpy","strcat","strncat","memcpy","memmove",
    "printf","fprintf","sprintf","snprintf","asprintf","vprintf","vfprintf","vsprintf","vsnprintf","vasprintf",
    "gets","fgets","scanf","sscanf","fscanf","recv","read","send","write","system","popen",
    "execl","execlp","execle","execv","execvp","execve"
}
COMMENT_BLOCK = re.compile(r'/\*.*?\*/', re.S)
COMMENT_LINE  = re.compile(r'//.*?$', re.M)

def load_sinks(path: str | None):
    if not path: return set(DEFAULT_SINKS)
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
    code = COMMENT_BLOCK.sub(" ", code)
    code = COMMENT_LINE.sub(" ", code)
    code = re.sub(r'\r\n?', '\n', code)
    code = re.sub(r'\n\s*\n+', '\n', code)
    return code.strip()

def line_window_around_sink(code: str, sinks: set[str], keep_lines=32) -> str:
    lines = code.splitlines()
    if not lines: return code
    idx = None
    joined = "\n".join(lines)
    # 첫 싱크 라인 찾기
    for i, ln in enumerate(lines):
        for s in sinks:
            if re.search(rf'\b{s}\b', ln):
                idx = i; break
        if idx is not None: break
    if idx is None:
        return "\n".join(lines[:keep_lines]) if len(lines) > keep_lines else code
    half = max(1, keep_lines // 2)
    s = max(0, idx - half)
    e = min(len(lines), idx + half + 1)
    return "\n".join(lines[s:e])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--keep-lines", type=int, default=256, help="sink 중심 ±N 라인")
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
            # code = line_window_around_sink(code, sinks, keep_lines=args.keep_lines)
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
