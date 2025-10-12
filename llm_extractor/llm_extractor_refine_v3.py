#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_codebert_from_juliet_llm.py (concurrent version)

Usage example:
  python prepare_codebert_from_juliet_llm.py \
    --juliet-root ../C/testcases \
    --outdir ./data/llm/juliet_codebert_dataset \
    --model gpt-4o-mini \
    --temp 0.0 \
    --max-tokens 256 \
    --chunk-lines 400 \
    --min-conf 0.70 \
    --clean-level basic \
    --workers 8

Requirements:
  pip install openai tqdm scikit-learn python-dotenv
Environment:
  export OPENAI_API_KEY="sk-..."
"""
import argparse
import json
import os
import sys
import time
import re
import hashlib
import random
import textwrap
from pathlib import Path
from typing import List, Tuple, Optional, Set
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# OpenAI client compatibility (new/legacy)
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

def call_llm(client_tuple, model: str, system: str, user: str, max_tokens:int=512, temperature:float=0.0) -> str:
    client, ctype = client_tuple
    for attempt in range(1, MAX_RETRIES+1):
        try:
            if ctype == "openai_new":
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role":"system","content":system},{"role":"user","content":user}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    if hasattr(resp, "choices") and len(resp.choices) > 0:
                        c0 = resp.choices[0]
                        msg = getattr(c0, "message", None)
                        if msg:
                            text = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
                            if text:
                                return text.strip()
                        text = getattr(c0, "text", None)
                        if text:
                            return text.strip()
                    return str(resp).strip()
                except Exception:
                    # fallback to older method signature
                    resp = client.chat.create(
                        model=model,
                        messages=[{"role":"system","content":system},{"role":"user","content":user}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    if hasattr(resp, "choices") and len(resp.choices) > 0:
                        c0 = resp.choices[0]
                        msg = getattr(c0, "message", None)
                        if msg:
                            return getattr(msg, "content", None) or str(resp)
                    return str(resp).strip()
            else:
                resp = client.ChatCompletion.create(
                    model=model,
                    messages=[{"role":"system","content":system},{"role":"user","content":user}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = RETRY_BACKOFF ** attempt
            print(f"[WARN] LLM request failed (attempt {attempt}/{MAX_RETRIES}): {e}. Retrying in {wait:.1f}s...", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("LLM request failed after retries.")

# -------------------
# Prompt: extract gadget + label in one shot (JSON-only)
# -------------------
PROMPT_TEMPLATE = r"""
You are a strict JSON-only extractor + vulnerability grader. INPUT is a single C/C++ source file (or chunk).
You must RETURN ONLY one JSON object (no extra text). Keys MUST be exactly:
  id, file, slice_begin, slice_end, gadget_code, label, score, reason

Definitions:
- gadget_code: smallest contiguous snippet (with necessary context lines) that contains the suspicious sink or critical statements useful for vulnerability classification. Use "\n" for line breaks.
- label: integer 1 if the gadget is vulnerable, 0 if not.
- score: float 0.0-1.0 representing confidence (higher is more confident).
- reason: a short sentence explaining the decision.

Rules:
1) If you cannot find any suspicious sink, return gadget_code as "" and set slice_begin/slice_end to 1, label to 0, score 0.0, reason brief.
2) Use low verbosity; output only the JSON object.
3) Try to be conservative when unsure: if uncertain give label and a modest score (e.g., 0.4-0.6).
4) Keep gadget_code as short as possible but include variable declarations or adjacent lines needed for comprehension.

INPUT (between markers):
-----SOURCE-BEGIN-----
{source}
-----SOURCE-END-----
Return the JSON now.
"""

# -------------------
# Helpers
# -------------------
def gather_source_files(root: Path, exts=(".c",".cpp",".cc",".h",".hpp")) -> List[Path]:
    out = []
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)

def chunk_text_lines(text: str, max_lines: int) -> List[Tuple[str,int,int]]:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return [(text, 1, len(lines))]
    chunks = []
    i = 0
    n = len(lines)
    while i < n:
        chunk_lines = lines[i:i+max_lines]
        start = i + 1
        end = i + len(chunk_lines)
        chunks.append(("\n".join(chunk_lines), start, end))
        i += max_lines
    return chunks

def safe_parse_json_only(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start > end:
        raise ValueError("No JSON braces found.")
    raw = text[start:end+1]
    raw = raw.replace("`","")
    raw = re.sub(r",\s*}", "}", raw)
    raw = re.sub(r",\s*]", "]", raw)
    return json.loads(raw)

def normalize_code(code: str) -> str:
    if not code:
        return ""
    code = code.replace("\r\n","\n").replace("\r","\n")
    code = "\n".join([ln.rstrip() for ln in code.splitlines()])
    code = re.sub(r"\n\s*\n+", "\n", code)
    return code.strip()

def infer_label_from_path(path: str) -> int:
    p = path.replace("\\","/").lower()
    if "/bad" in p or "_bad" in p:
        return 1
    if "/good" in p or "_good" in p:
        return 0
    return -1

def dedupe_by_text(records: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for r in records:
        t = r.get("gadget_code","")
        h = hashlib.sha256(t.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h); out.append(r)
    return out

def safe_group_split(records: List[dict], train_frac:float, val_frac:float, test_frac:float, seed:int=42):
    n = len(records)
    if n == 0:
        return [],[],[]
    random.seed(seed)
    if GroupShuffleSplit is None or len(set([r['file'] for r in records])) < 3 or n < 6:
        ntrain = max(1, int(round(train_frac*n)))
        nval = int(round(val_frac*n))
        if ntrain + nval >= n:
            ntrain = max(1, n - max(1, int(round(test_frac*n))))
            nval = max(0, n - ntrain - 1)
        items = records[:]
        random.shuffle(items)
        return items[:ntrain], items[ntrain:ntrain+nval], items[ntrain+nval:]
    groups = [r['file'] for r in records]
    labels = [r.get('label', -1) for r in records]
    indices = list(range(n))
    gss = GroupShuffleSplit(n_splits=1, test_size=(val_frac+test_frac), random_state=seed)
    train_idx, hold_idx = next(gss.split(indices, labels, groups))
    hold_indices = [indices[i] for i in hold_idx]
    hold_groups = [groups[i] for i in hold_idx]
    hold_labels = [labels[i] for i in hold_idx]
    rel_test_frac = test_frac / (test_frac + val_frac) if (test_frac + val_frac)>0 else 0.5
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_test_frac, random_state=seed)
    val_rel_idx, test_rel_idx = next(gss2.split(hold_indices, hold_labels, hold_groups))
    val_idx = [hold_idx[i] for i in val_rel_idx]
    test_idx = [hold_idx[i] for i in test_rel_idx]
    train = [records[i] for i in train_idx]
    val = [records[i] for i in val_idx]
    test = [records[i] for i in test_idx]
    return train, val, test

# -------------------
# Cleaning helpers
# -------------------
_JULIET_HINT_PATTERNS = [
    r"/\*\s*(?:FLAW|FIX|POTENTIAL FLAW|INCIDENTAL|NOTE)[^*]*\*/",
    r"//\s*(?:FLAW|FIX|POTENTIAL FLAW|INCIDENTAL|NOTE).*$",
]

_SINK_TOKENS: Set[str] = {
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

_PRESERVE_PATTERNS = [
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
_PRESERVE_REGEX = [re.compile(p, re.IGNORECASE) for p in _PRESERVE_PATTERNS]

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
    in_sl_comment = in_bl_comment = False
    in_sq = in_dq = False
    while i < n:
        ch = code[i]
        nx = code[i+1] if i+1 < n else ""

        if in_sl_comment:
            if ch == "\n":
                in_sl_comment = False
                out.append(ch)
            i += 1
            continue
        if in_bl_comment:
            if ch == "*" and nx == "/":
                in_bl_comment = False
                i += 2
            else:
                i += 1
            continue

        if not in_sq and not in_dq and ch == "/" and nx == "/":
            in_sl_comment = True
            i += 2
            continue
        if not in_sq and not in_dq and ch == "/" and nx == "*":
            in_bl_comment = True
            i += 2
            continue

        # 문자열 리터럴 토글(이스케이프 고려)
        if ch == "'" and not in_dq:
            bs = 0; j = i-1
            while j >=0 and code[j] == "\\":
                bs += 1; j -= 1
            if bs % 2 == 0:
                in_sq = not in_sq
        elif ch == '"' and not in_sq:
            bs = 0; j = i-1
            while j >=0 and code[j] == "\\":
                bs += 1; j -= 1
            if bs % 2 == 0:
                in_dq = not in_dq

        out.append(ch)
        i += 1

    return "".join(out)

def _dedent_and_trim(code: str) -> str:
    c = code.replace("\r\n", "\n").replace("\r", "\n")
    c = "\n".join(ln.rstrip() for ln in c.splitlines())
    c = textwrap.dedent(c)
    c = _MULTI_BLANK_RE.sub("\n\n", c)
    return c.strip()

def _should_preserve_token(tok: str, preserve_tokens: Set[str]) -> bool:
    if tok in preserve_tokens:
        return True
    for rx in _PRESERVE_REGEX:
        if rx.match(tok):
            return True
    # 길이/인덱스 휴리스틱(정수 취약 관련 신호 보호)
    if re.search(r"(len|size|count|cap|capacity|idx|index|off|offset)$", tok, re.I):
        return True
    return False

def _simplify_identifiers(code: str, preserve_tokens: Set[str], max_len: int = 60) -> str:
    mapping = {}
    counter = 0

    def repl(m):
        nonlocal counter
        tok = m.group(1)
        if _should_preserve_token(tok, preserve_tokens):
            return tok
        if tok in {"if","for","while","do","switch","case","break","continue","return","sizeof",
                   "int","char","long","short","float","double","void","const","volatile","static",
                   "struct","union","enum","typedef","unsigned","signed","bool","true","false","NULL",
                   "exit","goto"}:
            return tok
        if len(tok) <= 3:
            return tok
        if tok not in mapping:
            mapping[tok] = f"VAR_{counter}"
            counter += 1
        return mapping[tok]

    lines = []
    for ln in code.splitlines():
        if len(ln) > max_len:
            ln = ln[:max_len] + " /*…*/"
        lines.append(ln)
    code = "\n".join(lines)

    return _IDENTIFIER_RE.sub(repl, code)

def clean_gadget_code(
    code: str,
    level: str = "basic",
    preserve_sinks: bool = True,
) -> str:
    if not code:
        return code
    if level == "none":
        return _dedent_and_trim(code)

    c = _strip_juliet_hints(code)
    c = _remove_comments_but_keep_strings(c)
    c = _dedent_and_trim(c)

    if level == "aggressive":
        preserve_set = _SINK_TOKENS if preserve_sinks else set()
        c = _simplify_identifiers(c, preserve_tokens=preserve_set)
        c = _dedent_and_trim(c)

    return c

def load_extra_sinks(path: Optional[str]) -> Set[str]:
    s: Set[str] = set()
    if not path:
        return s
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith('#'):
                    continue
                s.add(ln)
    except Exception as e:
        print(f"[WARN] cannot load extra sinks from {path}: {e}", file=sys.stderr)
    return s

# -------------------
# Core: single-file LLM extract (one-shot JSON)
# -------------------
def extract_and_label_file(client_tuple, model, file_path: Path, chunk_lines:int=400, temp:float=DEFAULT_TEMP, max_tokens:int=512, sleep_per_call:float=0.15) -> Optional[dict]:
    try:
        src = file_path.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        print(f"[WARN] cannot read {file_path}: {e}", file=sys.stderr)
        return None
    chunks = chunk_text_lines(src, chunk_lines)
    system = "You are a strict JSON-only extractor + vulnerability grader."
    chosen = None
    for chunk_text, start_line, end_line in chunks:
        prompt = PROMPT_TEMPLATE.format(source=chunk_text)
        try:
            resp_text = call_llm(client_tuple, model, system, prompt, max_tokens=max_tokens, temperature=temp)
        except Exception as e:
            print(f"[WARN] LLM failed for {file_path} chunk({start_line}-{end_line}): {e}", file=sys.stderr)
            continue
        try:
            obj = safe_parse_json_only(resp_text)
        except Exception as e:
            print(f"[WARN] JSON parse failed for {file_path} chunk({start_line}-{end_line}): {e}", file=sys.stderr)
            if chosen is None:
                chosen = {"id": f"{file_path}:{start_line}", "file": str(file_path), "slice_begin": start_line, "slice_end": end_line, "gadget_code": "", "label": -1, "score": 0.0, "reason": f"parse_failed:{e}", "raw": resp_text}
            continue
        for k in ("id","file","slice_begin","slice_end","gadget_code","label","score","reason"):
            if k not in obj:
                if k == "gadget_code" and "gadget_raw" in obj:
                    obj["gadget_code"] = obj.pop("gadget_raw")
                else:
                    if k == "id":
                        obj["id"] = f"{file_path}:{start_line}"
                    elif k == "file":
                        obj["file"] = str(file_path)
                    elif k in ("slice_begin","slice_end"):
                        obj[k] = start_line if k=="slice_begin" else end_line
                    elif k == "gadget_code":
                        obj[k] = ""
                    elif k == "label":
                        obj[k] = -1
                    elif k == "score":
                        obj[k] = 0.0
                    elif k == "reason":
                        obj[k] = ""
        obj["gadget_code"] = normalize_code(obj.get("gadget_code",""))
        obj["file"] = str(file_path)
        try:
            obj["slice_begin"] = int(obj.get("slice_begin", start_line))
            obj["slice_end"] = int(obj.get("slice_end", obj["slice_begin"]))
        except:
            obj["slice_begin"], obj["slice_end"] = start_line, end_line
        try:
            obj["label"] = int(obj.get("label", -1))
        except:
            obj["label"] = -1
        try:
            obj["score"] = float(obj.get("score", 0.0))
        except:
            obj["score"] = 0.0
        if obj["gadget_code"].strip():
            chosen = obj
            break
        else:
            if chosen is None:
                chosen = obj
        time.sleep(sleep_per_call)
    if chosen is None:
        return {"id": f"{file_path}:0", "file": str(file_path), "slice_begin":1, "slice_end":1, "gadget_code":"", "label":-1, "score":0.0, "reason":"no_response"}
    return chosen

# -------------------
# CLI / main pipeline
# -------------------
def main():
    parser = argparse.ArgumentParser(description="Create CodeBERT dataset from Juliet using an LLM.")
    parser.add_argument("--juliet-root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--chunk-lines", type=int, default=400)
    parser.add_argument("--min-conf", type=float, default=0.70, help="Minimum score to accept LLM label as confident")
    parser.add_argument("--pseudo-conf", type=float, default=0.50, help=">= pseudo-conf and <min-conf => pseudo-label")
    parser.add_argument("--include-pseudo", action="store_true", help="Include pseudo-labeled samples into final dataset")
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--cap-per-file", type=int, default=0, help="If >0, keep at most N recs per file")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-level", choices=["none","basic","aggressive"], default="basic",
                        help="Gadget cleaning level: none/basic/aggressive")
    parser.add_argument("--no-preserve-sinks", action="store_true",
                        help="Aggressive mode: do NOT preserve sink tokens/patterns")
    parser.add_argument("--extra-sinks", help="Path to newline-separated sink tokens to preserve (optional)")
    parser.add_argument("--workers", type=int, default=6,
                        help="Number of concurrent LLM calls (threads). Default: 6")
    args = parser.parse_args()

    # load extra sinks (optional)
    extra = load_extra_sinks(args.extra_sinks)
    if extra:
        _SINK_TOKENS.update(extra)
        print(f"[INFO] extra sinks loaded: +{len(extra)} tokens")

    random.seed(args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    juliet_root = Path(args.juliet_root)

    # gather files
    files = gather_source_files(juliet_root)
    if not files:
        print(f"[ERROR] No source files found at {juliet_root}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Found {len(files)} source files under {juliet_root}")

    # init LLM client
    client_tuple = get_openai_client()
    print("[INFO] OpenAI client ready:", client_tuple[1], "model:", args.model)

    # --------------- concurrent generation ---------------
    def _job(fpath: Path):
        try:
            rec = extract_and_label_file(
                client_tuple, args.model, fpath,
                chunk_lines=args.chunk_lines, temp=args.temp, max_tokens=args.max_tokens
            )
            if rec is None:
                return None
            # undecided → path heuristic
            if rec.get("label", -1) == -1:
                infer = infer_label_from_path(str(fpath))
                if infer != -1:
                    rec["label"]  = infer
                    rec["score"]  = 0.95
                    rec["reason"] = "path_heuristic"
            return rec
        except Exception as e:
            return {
                "id": f"{fpath}:0", "file": str(fpath),
                "slice_begin": 1, "slice_end": 1,
                "gadget_code": "", "label": -1, "score": 0.0,
                "reason": f"exception:{e}"
            }

    records = []
    print(f"[INFO] running with {args.workers} workers")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_job, f): f for f in files}
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc=f"LLM generate (workers={args.workers})"):
            rec = fut.result()
            if rec is not None:
                records.append(rec)

    # write raw llm responses
    raw_out = outdir / "llm_raw.jsonl"
    with raw_out.open('w', encoding='utf-8') as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[INFO] wrote raw llm file:", raw_out, "count:", len(records))

    # dedupe if asked
    final_recs = records
    if args.dedupe:
        before = len(final_recs)
        final_recs = dedupe_by_text(final_recs)
        print(f"[INFO] deduped {before} -> {len(final_recs)}")

    # cap per file
    if args.cap_per_file > 0:
        cap = args.cap_per_file
        counts = defaultdict(int)
        kept = []
        for r in final_recs:
            key = (r.get("file"), r.get("label", -1))
            if counts[key] < cap:
                kept.append(r); counts[key]+=1
        final_recs = kept
        print(f"[INFO] cap_per_file applied -> {len(final_recs)}")

    # filter by confidence thresholds
    accepted, pseudo, uncertain = [], [], []
    for r in final_recs:
        sc = float(r.get("score", 0.0))
        if sc >= args.min_conf:
            accepted.append(r)
        elif sc >= args.pseudo_conf:
            pseudo.append(r)
        else:
            uncertain.append(r)

    print(f"[INFO] accepted(conf>={args.min_conf}): {len(accepted)}, pseudo: {len(pseudo)}, uncertain: {len(uncertain)}")

    # final set
    used = accepted + (pseudo if args.include_pseudo else [])
    used = [r for r in used if int(r.get("label", -1)) in (0,1)]

    # ensure fields & apply cleaning
    for r in used:
        r.setdefault("id", f"{r.get('file')}:{r.get('slice_begin',1)}")
        r.setdefault("text", r.get("gadget_code",""))
        r["text"] = r.get("text","")
        r["label"] = int(r.get("label", 0))
        r["text"] = clean_gadget_code(
            r["text"],
            level=args.clean_level,
            preserve_sinks=not args.no_preserve_sinks
        )

    if len(used) == 0:
        print("[WARN] No confident records selected. Try lowering --min-conf or include pseudo labels.", file=sys.stderr)
        (outdir/"uncertain.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in uncertain), encoding='utf-8')
        sys.exit(0)

    # group-aware split
    train, val, test = safe_group_split(used, args.train_frac, args.val_frac, args.test_frac, seed=args.seed)
    print("[INFO] splits -> train:", len(train), "val:", len(val), "test:", len(test))

    # write outputs
    def write_split(lst, path):
        with open(path, 'w', encoding='utf-8') as w:
            for r in lst:
                out = {"id": r.get("id"), "file": r.get("file"), "text": r.get("text"), "label": int(r.get("label"))}
                w.write(json.dumps(out, ensure_ascii=False) + "\n")
    write_split(train, outdir/"train.jsonl")
    write_split(val, outdir/"validation.jsonl")
    write_split(test, outdir/"test.jsonl")

    with (outdir/"llm_filtered.jsonl").open('w', encoding='utf-8') as w:
        for r in used:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    with (outdir/"uncertain.jsonl").open('w', encoding='utf-8') as w:
        for r in uncertain:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("[DONE] dataset ready in", outdir)
    print("Train/Val/Test sizes:", len(train), len(val), len(test))

if __name__ == "__main__":
    main()
