#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Juliet /C/testcases 소스에서 바로 평가용 JSONL(test.jsonl 또는 train/val/test) 생성

기본 동작:
  - C/C++ 파일 재귀 탐색
  - 정규식 기반 싱크 호출라인 찾기(printf/scanf/memcpy/recv/system 등)
  - 싱크 주변 컨텍스트를 "코드 가젯"으로 추출
  - 경로 휴리스틱으로 라벨링: (_bad -> 1, _good -> 0; 그 외는 스킵)
  - 중복 텍스트 제거(옵션)
  - 결과를 JSONL로 저장 (id, file, text, label)

사용 예:
  # 1) 전체를 평가용 test.jsonl 하나로 생성 (가장 간단)
  python build_juliet_evalset.py \
    --src ../C/testcases \
    --outdir ./data/juliet_eval_from_sources \
    --only-test

  # 2) 템플릿 홀드아웃(train/val/test) 생성 (누수 방지)
  python build_juliet_evalset.py \
    --src ../C/testcases \
    --outdir ./data/juliet_tpl_holdout \
    --split template --train-frac 0.8 --val-frac 0.1

  # 3) 랜덤 분할(train/val/test)
  python build_juliet_evalset.py \
    --src ../C/testcases \
    --outdir ./data/juliet_random_split \
    --split random --train-frac 0.8 --val-frac 0.1

주의:
  - CodeBERT 평가 스크립트는 {"text","label"}만 필수라 이 산출물 그대로 넣으면 됩니다.
  - 싱크가 하나도 없는 파일은 예제가 생기지 않을 수 있습니다.
"""

from __future__ import annotations
from pathlib import Path
import argparse, sys, os, re, json, random
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

# ---------- 탐색 설정 ----------
EXTS = {".c", ".cpp", ".cc", ".h", ".hpp"}

# ---------- 싱크 함수/패밀리 ----------
SINK_FUNCS = {
    "strcpy","strncpy","strcat","strncat","stpcpy","strlcpy","strlcat",
    "wcscpy","wcsncpy","wcscat","wcsncat","lstrcpy","lstrcpyn","lstrcat","lstrncat",
    "memcpy","memmove","bcopy",

    "printf","fprintf","sprintf","snprintf","asprintf",
    "vprintf","vfprintf","vsprintf","vsnprintf","vasprintf",
    "wprintf","fwprintf","swprintf","vwprintf","vswprintf",
    "_snprintf","_snwprintf","wsprintfA","wsprintfW",

    "gets","fgets","scanf","sscanf","fscanf","vscanf","vfscanf","vsscanf",
    "scanf_s","sscanf_s","fscanf_s","getenv",

    "open","openat","creat","fopen","freopen","_wfopen",
    "CreateFileA","CreateFileW","DeleteFileA","DeleteFileW","MoveFileA","MoveFileW",
    "remove","unlink","rename","chmod","chown","fchmod","fchown","mkdir","rmdir",
    "realpath","PathCanonicalizeA","PathCanonicalizeW","PathCombineA","PathCombineW",
    "tmpnam","tempnam","mktemp","_mktemp","mkstemp","access","stat","lstat","fstat",

    "recv","recvfrom","recvmsg","read","readv",
    "send","sendto","sendmsg","write","writev",

    "system","popen","_popen","_wsystem",
    "execl","execlp","execle","execv","execvp","execve",
    "CreateProcessA","CreateProcessW","ShellExecuteA","ShellExecuteW","WinExec",

    "dlopen","dlsym","LoadLibraryA","LoadLibraryW","GetProcAddress",

    "RegOpenKeyExA","RegOpenKeyExW","RegQueryValueExA","RegQueryValueExW",
    "RegSetValueExA","RegSetValueExW","RegCreateKeyExA","RegCreateKeyExW",

    "gets_s","strtok","strtok_r",

    "SetComputerNameA","SetComputerNameW",
}

SINK_FAMILY_REGEX = [
    r"^v?s?n?printf$", r"^v?f?scanf$",
    r"^str(n)?(cpy|cat)$", r"^wcs(n)?(cpy|cat)$", r"^lstr(cpy|cpyn|cat|ncat)$",
    r"^mem(cpy|move)$",
    r"^recv(from|msg)?$", r"^send(to|msg)?$", r"^read(v)?$", r"^write(v)?$",
    r"^exec([lvpe]{1,2})$", r"^CreateProcess(A|W)$", r"^ShellExecute(A|W)$",
    r"^CreateFile(A|W)$", r"^DeleteFile(A|W)$", r"^MoveFile(A|W)$",
    r"^Path(Canonicalize|Combine)(A|W)$",
    r"^Reg(OpenKeyEx|QueryValueEx|SetValueEx|CreateKeyEx)(A|W)$",
    r"^LoadLibrary(A|W)$", r"^GetProcAddress$",
    r"^_?s?n?w?printf$", r"^wsprintf(A|W)$",
    r"^(tmpnam|tempnam|mktemp|_mktemp|mkstemp)$",
    r"^SetComputerName(A|W)$",
]
_SINK_RXES = [re.compile(p, re.I) for p in SINK_FAMILY_REGEX]

# ---------- 유틸 ----------
def is_sink_name(name: str) -> bool:
    if name in SINK_FUNCS:
        return True
    for rx in _SINK_RXES:
        if rx.match(name):
            return True
    return False

def list_code_files(src: Path) -> List[Path]:
    return [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]

def infer_label_from_path(path: str) -> int | None:
    p = path.replace("\\","/").lower()
    if "/bad" in p or "_bad" in p:
        return 1
    if "/good" in p or "_good" in p:
        return 0
    return None

def find_sinks_regex(code: str) -> List[Tuple[str, int, str]]:
    """
    싱크 호출 1줄을 찾는다.
    return: list of (func_name, start_pos, line_text)
    """
    out=[]
    # 정확 토큰
    for fn in SINK_FUNCS:
        for m in re.finditer(r'\b' + re.escape(fn) + r'\s*\(', code):
            start = code.rfind('\n', 0, m.start()) + 1
            end = code.find('\n', m.end())
            if end == -1: end = len(code)
            out.append((fn, m.start(), code[start:end].strip()))
    # 패밀리 패턴
    for m in re.finditer(r'\b([A-Za-z_]\w*)\s*\(', code):
        name = m.group(1)
        if is_sink_name(name):
            start = code.rfind('\n', 0, m.start()) + 1
            end = code.find('\n', m.end())
            if end == -1: end = len(code)
            out.append((name, m.start(), code[start:end].strip()))
    return out

def bytepos_to_line_index(code: str, byte_pos: int) -> int:
    cb = code.encode('utf-8', errors='ignore')
    byte_pos = max(0, min(byte_pos, len(cb)))
    return cb[:byte_pos].count(b'\n')

def take_context(code: str, sink_pos: int, before=20, after=2) -> Tuple[str,int,int]:
    """
    싱크 주변 컨텍스트를 간단히 추출 (before/after 줄)
    반환: (text, slice_begin_line(1-based), slice_end_line)
    """
    lines = code.splitlines()
    sink_li = bytepos_to_line_index(code, sink_pos)
    s = max(0, sink_li - before)
    e = min(len(lines)-1, sink_li + after)
    snippet = "\n".join(lines[s:e+1]).strip()
    return snippet, s+1, e+1

def clean_text_for_model(code: str) -> str:
    # 너무 공격적이지 않게 여백 정리
    code = code.replace("\r\n","\n").replace("\r","\n")
    code = "\n".join(ln.rstrip() for ln in code.splitlines())
    code = re.sub(r"\n\s*\n{2,}", "\n\n", code)
    return code.strip()

def template_key(path: str) -> str:
    p = Path(path)
    name = re.sub(r"(_\d+)(\.[ch](pp)?|\.cc)$", r"\2", p.name, flags=re.I)
    return p.parent.name + "/" + name

def write_jsonl(items: List[dict], outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as w:
        for r in items:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- 메인 파이프라인 ----------
def build_examples(src_dir: Path, min_len=1, max_len=0, dedupe=True) -> List[dict]:
    files = list_code_files(src_dir)
    if not files:
        print("[ERROR] no C/C++ files under", src_dir, file=sys.stderr)
        return []
    print(f"[INFO] scanning files: {len(files)}")

    out=[]
    for fp in files:
        lbl = infer_label_from_path(str(fp))
        if lbl is None:
            continue  # 라벨 알 수 없으면 스킵(평가셋 일관성)
        try:
            code = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        sinks = find_sinks_regex(code)
        if not sinks:
            continue
        for (fn, pos, line_txt) in sinks:
            text, b, e = take_context(code, pos, before=20, after=2)
            text = clean_text_for_model(text)
            if len(text) < min_len:
                continue
            if max_len>0 and len(text)>max_len:
                text = text[:max_len]
            rec = {
                "id": f"{fp}:{b}",
                "file": str(fp),
                "text": text,
                "label": int(lbl),
            }
            out.append(rec)

    if dedupe:
        uniq=[]
        seen=set()
        for r in out:
            key = r["text"]
            if key in seen:
                continue
            seen.add(key); uniq.append(r)
        print(f"[INFO] dedupe: {len(out)} -> {len(uniq)}")
        out = uniq

    print(f"[INFO] examples: {len(out)}")
    return out

def split_template(items: List[dict], train_frac=0.8, val_frac=0.1, seed=42):
    groups = defaultdict(list)
    for r in items:
        groups[template_key(r["file"])].append(r)
    keys = list(groups.keys())
    random.seed(seed); random.shuffle(keys)
    n=len(keys)
    n_train=int(round(n*train_frac))
    n_val=int(round(n*val_frac))
    if n_train+n_val>n-1:
        n_val=max(0, n-n_train-1)
    k_train=set(keys[:n_train]); k_val=set(keys[n_train:n_train+n_val]); k_test=set(keys[n_train+n_val:])
    train=[]; val=[]; test=[]
    for k in k_train: train.extend(groups[k])
    for k in k_val:   val.extend(groups[k])
    for k in k_test:  test.extend(groups[k])
    return train, val, test

def split_random(items: List[dict], train_frac=0.8, val_frac=0.1, seed=42):
    items = items[:]
    random.seed(seed); random.shuffle(items)
    n=len(items)
    n_train=int(round(n*train_frac))
    n_val=int(round(n*val_frac))
    if n_train+n_val>n-1:
        n_val=max(0, n-n_train-1)
    train=items[:n_train]; val=items[n_train:n_train+n_val]; test=items[n_train+n_train and (n_train+n_val):]
    return train, val, test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Juliet 소스 루트 (예: ../C/testcases)")
    ap.add_argument("--outdir", required=True, help="출력 디렉터리")
    ap.add_argument("--only-test", action="store_true", help="train/val 없이 test.jsonl 하나만 생성")
    ap.add_argument("--split", choices=["template","random"], default="template", help="train/val/test 분할 방식")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-len", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=0, help=">0이면 텍스트를 이 길이로 자름")
    ap.add_argument("--no-dedupe", action="store_true", help="텍스트 중복 제거 비활성화")
    args = ap.parse_args()

    src_dir = Path(args.src)
    outdir  = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    items = build_examples(src_dir, min_len=args.min_len, max_len=args.max_len, dedupe=(not args.no_dedupe))
    if not items:
        print("[ERROR] no examples built — 소스 경로/라벨 규칙/싱크 탐지 확인", file=sys.stderr)
        sys.exit(2)

    # 간단 통계 출력
    lbl_cnt = Counter(int(r["label"]) for r in items)
    print("[INFO] label distribution:", dict(lbl_cnt))

    if args.only-test:
        write_jsonl(items, outdir/"test.jsonl")
        print(f"[DONE] test={len(items)} -> {outdir/'test.jsonl'}")
        return

    if args.split == "template":
        train, val, test = split_template(items, args.train_frac, args.val_frac, args.seed)
    else:
        train, val, test = split_random(items, args.train_frac, args.val_frac, args.seed)

    write_jsonl(train, outdir/"train.jsonl")
    write_jsonl(val,   outdir/"validation.jsonl")
    write_jsonl(test,  outdir/"test.jsonl")
    print(f"[DONE] train={len(train)} val={len(val)} test={len(test)} -> {outdir}")

if __name__ == "__main__":
    main()
