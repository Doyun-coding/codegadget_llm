#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
label_gadgets_with_llm.py

정적 추출기(static_extractor.py)로 만든 C/C++ 코드 가젯(JSONL)을
OpenAI GPT로 라벨링(취약=1/비취약=0)하고, 필요시 스니펫을 정제하여
CodeBERT 학습용 데이터셋(train/validation/test)으로 저장합니다.

- 입력(JSONL, per line): {file, lang, sink_name, cwe_candidates, sink_line, slice_begin, slice_end, gadget_raw, ...}
- 출력(JSONL, per line): {id, file, text, label}

요구사항:
  pip install openai tqdm scikit-learn python-dotenv

환경변수:
  export OPENAI_API_KEY="sk-..."

예시:
  python label_gadgets_with_llm.py \
    --input ./static_extractor/v3/juliet_gadgets.jsonl \
    --outdir ./data/llm_from_static/codebert_dataset \
    --model gpt-4o-mini \
    --workers 6 \
    --min-conf 0.70 \
    --clean-level basic
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import sys
import textwrap
import threading
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

# optional sklearn GroupShuffleSplit
try:
    from sklearn.model_selection import GroupShuffleSplit
except Exception:
    GroupShuffleSplit = None

# optional dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------------------
# Config
# -------------------
MAX_RETRIES = 4
RETRY_BACKOFF = 2.0
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMP = 0.0

# -------------------
# OpenAI client (new/legacy) + JSON mode
# -------------------
def get_openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    # prefer new client
    try:
        from openai import OpenAI
        try:
            client = OpenAI(api_key=key)
        except Exception:
            client = OpenAI()
        return client, "openai_new"
    except Exception:
        pass
    try:
        import openai as legacy
        legacy.api_key = key
        return legacy, "openai_legacy"
    except Exception:
        raise RuntimeError("Cannot import OpenAI client. pip install openai")

def call_llm_json(client_tuple, model: str, system: str, user: str, max_tokens:int=256, temperature:float=0.0) -> str:
    client, ctype = client_tuple
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if ctype == "openai_new":
                # 새 클라이언트: JSON response_format 시도
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role":"system","content":system},{"role":"user","content":user}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        response_format={"type": "json_object"},
                    )
                except Exception:
                    # 일부 런타임/모델에서 response_format 미지원일 수 있음 → 일반 모드로 재시도
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role":"system","content":system},{"role":"user","content":user}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                c0 = resp.choices[0]
                msg = getattr(c0, "message", None)
                if msg and getattr(msg, "content", None):
                    return msg.content.strip()
                # fallback
                tx = getattr(c0, "text", None)
                if tx:
                    return tx.strip()
                return str(resp).strip()
            else:
                # 레거시
                resp = client.ChatCompletion.create(
                    model=model,
                    messages=[{"role":"system","content":system},{"role":"user","content":user}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = RETRY_BACKOFF ** attempt + random.uniform(0, 0.5)
            print(f"[WARN] LLM request failed (attempt {attempt}/{MAX_RETRIES}): {e}. Retry in {wait:.1f}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("LLM request failed after retries.")

# -------------------
# Prompt (JSON-only)
# -------------------
PROMPT_TEMPLATE = r"""
You are a STRICT JSON-ONLY vulnerability classifier/editor for C/C++ "gadgets".
Return exactly ONE JSON object with keys:
  id, label, score, reason, gadget_code

- label: 1 if vulnerable, otherwise 0 (integer).
- score: 0.0-1.0 confidence (float).
- gadget_code: minimal contiguous snippet (may be SAME as input if already minimal),
  but MUST be derived from the input gadget, with "\n" for line breaks.
- reason: one short sentence.

If you are uncertain, still pick a label but lower score (e.g., 0.4-0.6).
Output JSON ONLY. No explanations.

# Gadget metadata (hints)
sink_name: {sink_name}
cwe_candidates: {cwe_candidates}
file: {file}

# INPUT GADGET (between markers)
-----GADGET-BEGIN-----
{gadget}
-----GADGET-END-----
"""

# -------------------
# Cleaning helpers
# -------------------
_JULIET_HINT_PATTERNS = [
    r"/\*\s*(?:FLAW|FIX|POTENTIAL FLAW|INCIDENTAL|NOTE)[^*]*\*/",
    r"//\s*(?:FLAW|FIX|POTENTIAL FLAW|INCIDENTAL|NOTE).*$",
]
_PRESERVE_PATTERNS = [
    r"^v?s?n?printf$", r"^v?f?scanf$",
    r"^str(n)?(cpy|cat)$", r"^wcs(n)?(cpy|cat)$", r"^lstr(cpy|cpyn|cat|ncat)$",
    r"^mem(cpy|move)$", r"^recv(from|msg)?$", r"^send(to|msg)?$", r"^read(v)?$", r"^write(v)?$",
    r"^exec([lvpe]{1,2})$", r"^CreateProcess(A|W)$", r"^ShellExecute(A|W)$",
    r"^CreateFile(A|W)$", r"^DeleteFile(A|W)$", r"^MoveFile(A|W)$",
    r"^Path(Canonicalize|Combine)(A|W)$",
    r"^Reg(OpenKeyEx|QueryValueEx|SetValueEx|CreateKeyEx)(A|W)$",
    r"^LoadLibrary(A|W)$", r"^GetProcAddress$",
    r"^_?s?n?w?printf$", r"^wsprintf(A|W)$",
    r"^(tmpnam|tempnam|mktemp|_mktemp|mkstemp)$",
    r"^SetComputerName(A|W)$",
]
_PRESERVE_REGEX = [re.compile(p, re.IGNORECASE) for p in _PRESERVE_PATTERNS]
_SINK_TOKENS: Set[str] = {
    "strcpy","strncpy","strcat","strncat","stpcpy","strlcpy","strlcat",
    "wcscpy","wcsncpy","wcscat","wcsncat","lstrcpy","lstrcpyn","lstrcat","lstrncat",
    "memcpy","memmove","bcopy",
    "printf","fprintf","sprintf","snprintf","asprintf","vprintf","vfprintf","vsprintf","vsnprintf","vasprintf",
    "wprintf","fwprintf","swprintf","vwprintf","vswprintf","_snprintf","_snwprintf","wsprintfA","wsprintfW",
    "gets","fgets","scanf","sscanf","fscanf","vscanf","vfscanf","vsscanf","scanf_s","sscanf_s","fscanf_s","getenv",
    "open","openat","creat","fopen","freopen","_wfopen","CreateFileA","CreateFileW","DeleteFileA","DeleteFileW","MoveFileA","MoveFileW",
    "remove","unlink","rename","chmod","chown","fchmod","fchown","mkdir","rmdir","realpath","PathCanonicalizeA","PathCanonicalizeW","PathCombineA","PathCombineW",
    "tmpnam","tempnam","mktemp","_mktemp","mkstemp","access","stat","lstat","fstat",
    "recv","recvfrom","recvmsg","read","readv","send","sendto","sendmsg","write","writev",
    "system","popen","_popen","_wsystem",
    "execl","execlp","execle","execv","execvp","execve","CreateProcessA","CreateProcessW","ShellExecuteA","ShellExecuteW","WinExec",
    "dlopen","dlsym","LoadLibraryA","LoadLibraryW","GetProcAddress",
    "RegOpenKeyExA","RegOpenKeyExW","RegQueryValueExA","RegQueryValueExW","RegSetValueExA","RegSetValueExW","RegCreateKeyExA","RegCreateKeyExW",
    "gets_s","strtok","strtok_r",
    "SetComputerNameA","SetComputerNameW",
}
_IDENTIFIER_RE = re.compile(r"\b([A-Za-z_]\w*)\b")
_MULTI_BLANK_RE = re.compile(r"\n\s*\n{2,}")  # 2개 초과 빈 줄 축소

def _strip_juliet_hints(code: str) -> str:
    c = code
    for pat in _JULIET_HINT_PATTERNS:
        c = re.sub(pat, "", c, flags=re.IGNORECASE | re.MULTILINE)
    return c

def _remove_comments_but_keep_strings(code: str) -> str:
    out = []
    i, n = 0, len(code)
    in_sl = in_bl = False
    in_sq = in_dq = False
    while i < n:
        ch = code[i]
        nx = code[i+1] if i+1 < n else ""

        if in_sl:
            if ch == "\n": in_sl = False; out.append(ch)
            i += 1; continue
        if in_bl:
            if ch == "*" and nx == "/": in_bl = False; i += 2
            else: i += 1
            continue

        if not in_sq and not in_dq and ch == "/" and nx == "/":
            in_sl = True; i += 2; continue
        if not in_sq and not in_dq and ch == "/" and nx == "*":
            in_bl = True; i += 2; continue

        if ch == "'" and not in_dq:
            bs = 0; j = i-1
            while j >= 0 and code[j] == "\\": bs += 1; j -= 1
            if bs % 2 == 0: in_sq = not in_sq
        elif ch == '"' and not in_sq:
            bs = 0; j = i-1
            while j >= 0 and code[j] == "\\": bs += 1; j -= 1
            if bs % 2 == 0: in_dq = not in_dq

        out.append(ch); i += 1
    return "".join(out)

def _dedent_and_trim(code: str) -> str:
    c = code.replace("\r\n","\n").replace("\r","\n")
    c = "\n".join(ln.rstrip() for ln in c.splitlines())
    c = textwrap.dedent(c)
    c = _MULTI_BLANK_RE.sub("\n\n", c)
    return c.strip()

def _should_preserve_token(tok: str, preserve_tokens: Set[str]) -> bool:
    if tok in preserve_tokens:
        return True
    for rx in _PRESERVE_REGEX:
        if rx.match(tok): return True
    if re.search(r"(len|size|count|cap|capacity|idx|index|off|offset)$", tok, re.I):
        return True
    return False

def _simplify_identifiers(code: str, preserve_tokens: Set[str], max_len: int = 60) -> str:
    mapping = {}; counter = 0
    def repl(m):
        nonlocal counter
        tok = m.group(1)
        if _should_preserve_token(tok, preserve_tokens): return tok
        if tok in {"if","for","while","do","switch","case","break","continue","return","sizeof",
                   "int","char","long","short","float","double","void","const","volatile","static",
                   "struct","union","enum","typedef","unsigned","signed","bool","true","false","NULL",
                   "exit","goto"}:
            return tok
        if len(tok) <= 3: return tok
        if tok not in mapping:
            mapping[tok] = f"VAR_{counter}"; counter += 1
        return mapping[tok]
    lines = []
    for ln in code.splitlines():
        if len(ln) > max_len: ln = ln[:max_len] + " /*…*/"
        lines.append(ln)
    code = "\n".join(lines)
    return _IDENTIFIER_RE.sub(repl, code)

def clean_gadget_code(code: str, level: str="basic", preserve_sinks: bool=True) -> str:
    if not code: return code
    if level == "none": return _dedent_and_trim(code)
    c = _strip_juliet_hints(code)
    c = _remove_comments_but_keep_strings(c)
    c = _dedent_and_trim(c)
    if level == "aggressive":
        preserve_set = _SINK_TOKENS if preserve_sinks else set()
        c = _simplify_identifiers(c, preserve_tokens=preserve_set)
        c = _dedent_and_trim(c)
    return c

# -------------------
# Utils
# -------------------
def safe_parse_json_only(text: str) -> dict:
    # JSON 모드여도 방어적으로 파싱
    s = text.find("{"); e = text.rfind("}")
    if s == -1 or e == -1 or s > e:
        raise ValueError("No JSON braces found.")
    raw = text[s:e+1]
    raw = raw.replace("`","")
    raw = re.sub(r",\s*}", "}", raw)
    raw = re.sub(r",\s*]", "]", raw)
    return json.loads(raw)

def dedupe_by_text(records: List[dict]) -> List[dict]:
    seen = set(); out = []
    for r in records:
        t = r.get("gadget_code") or r.get("text") or ""
        h = hashlib.sha256(t.encode("utf-8")).hexdigest()
        if h in seen: continue
        seen.add(h); out.append(r)
    return out

def group_key(rec: dict, mode: str) -> str:
    if mode == "cwe":
        cwes = rec.get("cwe_candidates") or []
        if isinstance(cwes, list) and cwes:
            return ",".join(sorted(cwes))
        return "CWE-NA"
    if mode == "sink":
        return str(rec.get("sink_name","NA"))
    # default: file
    return str(rec.get("file","NA"))

def safe_group_split(records: List[dict], train_frac:float, val_frac:float, test_frac:float, seed:int, group_by:str):
    n = len(records)
    if n == 0: return [], [], []
    random.seed(seed)
    if GroupShuffleSplit is None or len(set(group_key(r, group_by) for r in records)) < 3 or n < 6:
        ntrain = max(1, int(round(train_frac*n)))
        nval = int(round(val_frac*n))
        if ntrain + nval >= n:
            ntrain = max(1, n - max(1, int(round(test_frac*n))))
            nval = max(0, n - ntrain - 1)
        random.shuffle(records)
        return records[:ntrain], records[ntrain:ntrain+nval], records[ntrain+nval:]

    groups = [group_key(r, group_by) for r in records]
    labels = [r.get("label",-1) for r in records]
    idx = list(range(n))

    gss = GroupShuffleSplit(n_splits=1, test_size=(val_frac+test_frac), random_state=seed)
    tr_idx, hold_idx = next(gss.split(idx, labels, groups))
    hold_ids = [idx[i] for i in hold_idx]
    hold_groups = [groups[i] for i in hold_idx]
    hold_labels = [labels[i] for i in hold_idx]

    rel_test = test_frac/(test_frac+val_frac) if (test_frac+val_frac)>0 else 0.5
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_test, random_state=seed)
    v_rel, t_rel = next(gss2.split(hold_ids, hold_labels, hold_groups))
    v_idx = [hold_idx[i] for i in v_rel]
    t_idx = [hold_idx[i] for i in t_rel]

    train = [records[i] for i in tr_idx]
    val = [records[i] for i in v_idx]
    test = [records[i] for i in t_idx]
    return train, val, test

# -------------------
# Core LLM worker
# -------------------
def llm_label_one(client_tuple, model: str, rec: dict, max_tokens:int, temp:float) -> dict:
    # 입력 가젯
    gid = f"{rec.get('file','')}:L{rec.get('sink_line','')}"
    gadget_text = str(rec.get("gadget_raw","")).strip()
    if not gadget_text:
        # 빈 가젯은 바로 비취약 0, score 0
        return {
            "id": gid, "file": rec.get("file",""),
            "gadget_code": "", "label": 0, "score": 0.0,
            "reason": "empty_gadget", "cwe_candidates": rec.get("cwe_candidates",[])
        }

    prompt = PROMPT_TEMPLATE.format(
        sink_name=rec.get("sink_name",""),
        cwe_candidates=rec.get("cwe_candidates",[]),
        file=rec.get("file",""),
        gadget=gadget_text
    )
    system = "You output STRICT JSON for vulnerability gadgets."

    resp = call_llm_json(client_tuple, model, system, prompt, max_tokens=max_tokens, temperature=temp)
    try:
        obj = safe_parse_json_only(resp)
    except Exception as e:
        # 파싱 실패 → 보수적으로 음성/낮은 점수
        return {
            "id": gid, "file": rec.get("file",""),
            "gadget_code": gadget_text, "label": -1, "score": 0.0,
            "reason": f"json_parse_fail:{e}", "raw": resp,
            "cwe_candidates": rec.get("cwe_candidates",[])
        }

    # 필드 보정
    if "id" not in obj: obj["id"] = gid
    if "file" not in obj: obj["file"] = rec.get("file","")
    if "gadget_code" not in obj: obj["gadget_code"] = gadget_text
    try:
        obj["label"] = int(obj.get("label", -1))
    except:
        obj["label"] = -1
    try:
        obj["score"] = float(obj.get("score", 0.0))
    except:
        obj["score"] = 0.0
    if "reason" not in obj: obj["reason"] = ""

    # 입력 메타 유지(나중 분할 옵션용)
    obj["sink_name"] = rec.get("sink_name","")
    obj["cwe_candidates"] = rec.get("cwe_candidates",[])
    return obj

# -------------------
# Main
# -------------------
def main():
    ap = argparse.ArgumentParser(description="Label static gadgets with GPT and build CodeBERT dataset.")
    ap.add_argument("--input", required=True, help="static_extractor JSONL path")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    ap.add_argument("--max-tokens", type=int, default=256, help="LLM OUTPUT token cap (JSON)")
    ap.add_argument("--workers", type=int, default=16, help="Concurrent LLM calls")
    ap.add_argument("--min-conf", type=float, default=0.70)
    ap.add_argument("--pseudo-conf", type=float, default=0.50)
    ap.add_argument("--include-pseudo", action="store_true")
    ap.add_argument("--dedupe", action="store_true")
    ap.add_argument("--cap-per-file", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--group-by", choices=["file","cwe","sink"], default="file", help="Split grouping key")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clean-level", choices=["none","basic","aggressive"], default="basic")
    ap.add_argument("--no-preserve-sinks", action="store_true", help="aggressive mode: do NOT preserve sink tokens")
    ap.add_argument("--resume", action="store_true", help="Reuse llm_raw.jsonl in outdir if exists (skip done)")
    args = ap.parse_args()

    random.seed(args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    inp = Path(args.input)
    if not inp.exists():
        print(f"[ERROR] input not found: {inp}", file=sys.stderr); sys.exit(1)

    # 입력 로드
    src_records: List[dict] = []
    with open(inp, "r", encoding="utf-8") as f:
        for ln in f:
            try:
                obj = json.loads(ln)
                # 최소 필드만 보장
                if "file" in obj and "gadget_raw" in obj:
                    src_records.append(obj)
            except Exception:
                continue
    if not src_records:
        print("[ERROR] empty input records", file=sys.stderr); sys.exit(1)

    # LLM 준비
    client_tuple = get_openai_client()
    print(f"[INFO] Found {len(src_records)} gadgets")
    print("[INFO] OpenAI client ready:", client_tuple[1], "model:", args.model)
    print(f"[INFO] running with {args.workers} workers")

    # resume: 기존 raw 불러와서 스킵
    done_map: Dict[str, dict] = {}
    raw_path = outdir/"llm_raw.jsonl"
    if args.resume and raw_path.exists():
        with open(raw_path, "r", encoding="utf-8") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                    if "id" in r: done_map[r["id"]] = r
                except:
                    pass
        print(f"[INFO] resume: loaded {len(done_map)} already processed gadgets")

    # 처리
    lock = threading.Lock()
    raw_out = open(raw_path, "a", encoding="utf-8")
    stats = Counter()

    def work(rec):
        gid = f"{rec.get('file','')}:L{rec.get('sink_line','')}"
        if gid in done_map:
            r = done_map[gid]
        else:
            r = llm_label_one(client_tuple, args.model, rec, args.max_tokens, args.temp)
        # cleaning & projection은 후단에서 일괄
        with lock:
            raw_out.write(json.dumps(r, ensure_ascii=False) + "\n")
        return r

    results: List[dict] = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in tqdm(ex.map(work, src_records), total=len(src_records), desc=f"LLM label (workers={args.workers})"):
            results.append(r)
    raw_out.close()
    print("[INFO] wrote raw:", raw_path, "count:", len(results))

    # dedupe (옵션)
    final = results
    if args.dedupe:
        before = len(final)
        final = dedupe_by_text(final)
        print(f"[INFO] deduped {before} -> {len(final)}")

    # 신뢰도 필터링
    accepted, pseudo, uncertain = [], [], []
    for r in final:
        sc = float(r.get("score", 0.0))
        if sc >= args.min_conf:
            accepted.append(r)
        elif sc >= args.pseudo_conf:
            pseudo.append(r)
        else:
            uncertain.append(r)
    print(f"[INFO] accepted>={args.min_conf}: {len(accepted)}, pseudo: {len(pseudo)}, uncertain: {len(uncertain)}")

    used = accepted + (pseudo if args.include_pseudo else [])
    # label ∈ {0,1}만 유지
    used = [r for r in used if int(r.get("label",-1)) in (0,1)]

    # 최종 필드 및 클리닝
    preserve = not args.no_preserve_sinks
    for r in used:
        r.setdefault("file","")
        r.setdefault("id", r.get("file",""))
        txt = r.get("gadget_code") or ""
        r["text"] = clean_gadget_code(txt, level=args.clean_level, preserve_sinks=preserve)
        r["label"] = int(r.get("label",0))

    # cap-per-file (옵션)
    if args.cap_per_file > 0:
        cap = args.cap_per_file
        counts = defaultdict(int)
        kept = []
        for r in used:
            key = (r.get("file"), r.get("label", -1))
            if counts[key] < cap:
                kept.append(r); counts[key]+=1
        used = kept
        print(f"[INFO] cap_per_file applied -> {len(used)}")

    if len(used) == 0:
        print("[WARN] No confident records after filtering.", file=sys.stderr)
        (outdir/"uncertain.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in uncertain), encoding="utf-8")
        sys.exit(0)

    # 그룹 분할
    train, val, test = safe_group_split(used, args.train_frac, args.val_frac, args.test_frac, seed=args.seed, group_by=args.group_by)
    print(f"[INFO] splits({args.group_by}) -> train:{len(train)} val:{len(val)} test:{len(test)}")

    # 저장
    def write_split(lst, path):
        with open(path, "w", encoding="utf-8") as w:
            for r in lst:
                out = {"id": r.get("id"), "file": r.get("file"), "text": r.get("text",""), "label": int(r.get("label",0))}
                w.write(json.dumps(out, ensure_ascii=False) + "\n")

    write_split(train, outdir/"train.jsonl")
    write_split(val,   outdir/"validation.jsonl")
    write_split(test,  outdir/"test.jsonl")

    with open(outdir/"llm_filtered.jsonl", "w", encoding="utf-8") as w:
        for r in used:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(outdir/"uncertain.jsonl", "w", encoding="utf-8") as w:
        for r in uncertain:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("[DONE] dataset ready in", outdir)
    print("Train/Val/Test sizes:", len(train), len(val), len(test))

if __name__ == "__main__":
    main()
