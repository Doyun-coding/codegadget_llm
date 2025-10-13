#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_codebert_from_juliet_llm_strict.py

Juliet 소스에서 LLM으로 가젯+라벨을 추출해 CodeBERT 학습셋으로 변환 (품질/균형 강화판)

예)
python prepare_codebert_from_juliet_llm_strict.py \
  --juliet-root ../C/testcases \
  --outdir ./data/llm/strict-codebert-model \
  --model gpt-4o-mini --temp 0.0 --max-tokens 320 \
  --chunk-lines 300 --max-gadget-lines 30 \
  --min-conf-pos 0.72 --min-conf-neg 0.55 \
  --require-sink-for-pos \
  --few-shot \
  --dedupe \
  --balance-target 0.5 --balance-tolerance 0.15 \
  --max-per-file 2 \
  --workers 8
"""
import argparse, json, os, sys, time, re, hashlib, random, textwrap
from pathlib import Path
from typing import List, Tuple, Optional, Set, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
from tqdm import tqdm

# ----------------- optional deps -----------------
try:
    from sklearn.model_selection import GroupShuffleSplit
except Exception:
    GroupShuffleSplit = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ----------------- config -----------------
MAX_RETRIES = 4
RETRY_BACKOFF = 2.0
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMP = 0.0

# ----------------- sinks & regex -----------------
BASE_SINK_TOKENS: Set[str] = {
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
SINK_WORD_RE = re.compile(r"\b([A-Za-z_]\w*)\b")

# ----------------- OpenAI client -----------------
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

def call_llm(client_tuple, model: str, system: str, user: str, max_tokens:int=320, temperature:float=0.0) -> str:
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
                        response_format={"type":"json_object"},
                    )
                except Exception:
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
                tx = getattr(c0, "text", None)
                if tx:
                    return tx.strip()
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
            print(f"[WARN] LLM request failed ({attempt}/{MAX_RETRIES}): {e}. retry in {wait:.1f}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("LLM request failed after retries.")

# ----------------- Prompt -----------------
STRICT_POLICY = """
Decision policy (CRITICAL):
- Output label=1 ONLY with CLEAR evidence of unsafe dataflow
  from untrusted/unchecked input → dangerous sink, AND missing/insufficient guards
  (bounds checks, fixed format string, sanitization/validation, allowlist, privilege checks).
- Mere presence of a dangerous API is NOT enough.
- If evidence is insufficient or context is ambiguous, output label=0
  with a moderate score (~0.45–0.60).
"""

FEW_SHOT = r"""
# EXAMPLE-OK (label 0)
{"id":"ex0","file":"ex.c","slice_begin":1,"slice_end":4,"gadget_code":"int x; scanf(\"%d\", &x);\nprintf(\"x=%d\", x);\n","label":0,"score":0.52,"reason":"fixed format string and bounded read"}

# EXAMPLE-VULN (label 1)
{"id":"ex1","file":"ex.c","slice_begin":1,"slice_end":3,"gadget_code":"char buf[64]; gets(buf);\nprintf(buf);\n","label":1,"score":0.92,"reason":"unchecked input flows to format string sink"}
"""

PROMPT_TEMPLATE = r"""
You are a STRICT JSON-only vulnerability grader for C/C++ snippets.
Return ONE JSON object with keys (and types):
  id (string), file (string),
  slice_begin (int, 1-based in INPUT), slice_end (int, >= slice_begin),
  gadget_code (string, newline='\n', ≤ {max_gadget_lines} lines),
  label (int in {{0,1}}), score (float 0.0–1.0),
  reason (string, ≤ 25 words, no newlines)

{strict_policy}

Gadget rules:
- gadget_code MUST be a contiguous snippet from INPUT that includes the sink
  AND any nearby guards needed for your decision. Keep it ≤ {max_gadget_lines} lines.
- slice_* MUST match gadget_code's line range in INPUT (1-based).
- If no suspicious sink is present, set gadget_code="", slice_begin=1, slice_end=1, label=0, score=0.0.

STRICTNESS:
- Return ONLY the JSON object (no prose).
- Use integers for label/slice_*; float for score (e.g., 0.53).

{few_shot}

-----SOURCE-BEGIN-----
{source}
-----SOURCE-END-----
"""

# ----------------- helpers -----------------
def gather_source_files(root: Path, exts=(".c",".cpp",".cc",".h",".hpp")) -> List[Path]:
    out = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)

def chunk_text_lines(text: str, max_lines: int) -> List[Tuple[str,int,int]]:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return [(text, 1, len(lines))]
    chunks = []
    i = 0; n = len(lines)
    while i < n:
        chunk_lines = lines[i:i+max_lines]
        start = i + 1
        end = i + len(chunk_lines)
        chunks.append(("\n".join(chunk_lines), start, end))
        i += max_lines
    return chunks

def add_line_numbers(text: str) -> str:
    out = []
    for i, ln in enumerate(text.splitlines(), start=1):
        out.append(f"{i:04d}: {ln}")
    return "\n".join(out)

def safe_parse_json_only(text: str) -> dict:
    s = text.find("{"); e = text.rfind("}")
    if s == -1 or e == -1 or s > e:
        raise ValueError("No JSON braces")
    raw = text[s:e+1].replace("`","")
    raw = re.sub(r",\s*}", "}", raw)
    raw = re.sub(r",\s*]", "]", raw)
    return json.loads(raw)

def normalize_code(code: str) -> str:
    if not code: return ""
    code = code.replace("\r\n","\n").replace("\r","\n")
    code = "\n".join(ln.rstrip() for ln in code.splitlines())
    code = re.sub(r"\n\s*\n+", "\n", code)
    return code.strip()

def infer_label_from_path(path: str) -> int:
    p = path.replace("\\","/").lower()
    if "/bad" in p or "_bad" in p:  return 1
    if "/good" in p or "_good" in p: return 0
    return -1

def contains_sink(text: str, sink_vocab: Set[str]) -> bool:
    hits = set()
    for m in SINK_WORD_RE.finditer(text or ""):
        tok = m.group(1)
        if tok in sink_vocab:
            hits.add(tok)
            if len(hits) >= 1:
                return True
    return False

def enforce_schema(obj: dict, file_path: Path, chunk_len: int, max_gadget_lines:int) -> dict:
    obj = dict(obj)
    obj["id"] = str(obj.get("id", f"{file_path}:1"))
    obj["file"] = str(file_path)
    try:
        obj["slice_begin"] = int(obj.get("slice_begin", 1))
        obj["slice_end"] = int(obj.get("slice_end", obj["slice_begin"]))
    except Exception:
        obj["slice_begin"], obj["slice_end"] = 1, 1
    sb = max(1, min(int(obj["slice_begin"]), chunk_len))
    se = max(sb, min(int(obj["slice_end"]), chunk_len))
    obj["slice_begin"], obj["slice_end"] = sb, se
    obj["gadget_code"] = normalize_code(obj.get("gadget_code",""))
    # 제한: 최대 줄수
    if obj["gadget_code"]:
        glines = obj["gadget_code"].splitlines()
        if len(glines) > max_gadget_lines:
            obj["gadget_code"] = "\n".join(glines[:max_gadget_lines])
    try:
        obj["label"] = int(obj.get("label", 0))
    except Exception:
        obj["label"] = 0
    obj["label"] = 1 if obj["label"] == 1 else 0
    try:
        obj["score"] = float(obj.get("score", 0.0))
    except Exception:
        obj["score"] = 0.0
    obj["reason"] = str(obj.get("reason",""))[:200].replace("\n"," ")
    return obj

def dedupe_by_text(records: List[dict]) -> List[dict]:
    seen = set(); out = []
    for r in records:
        t = r.get("gadget_code") or r.get("text") or ""
        h = hashlib.sha256(t.encode("utf-8")).hexdigest()
        if h in seen: continue
        seen.add(h); out.append(r)
    return out

def safe_group_split(records: List[dict], train_frac:float, val_frac:float, test_frac:float, seed:int=42):
    n = len(records)
    if n == 0: return [],[],[]
    random.seed(seed)
    if GroupShuffleSplit is None or len(set([r['file'] for r in records])) < 3 or n < 6:
        items = records[:]
        random.shuffle(items)
        ntrain = max(1, int(round(train_frac*n)))
        nval = int(round(val_frac*n))
        if ntrain + nval >= n:
            ntrain = max(1, n - max(1, int(round(test_frac*n))))
            nval = max(0, n - ntrain - 1)
        return items[:ntrain], items[ntrain:ntrain+nval], items[ntrain+ntrain:]
    groups = [r['file'] for r in records]
    labels = [r.get('label', -1) for r in records]
    idx = list(range(n))
    gss = GroupShuffleSplit(n_splits=1, test_size=(val_frac+test_frac), random_state=seed)
    tr, hold = next(gss.split(idx, labels, groups))
    hold_idx = [idx[i] for i in hold]
    hold_groups = [groups[i] for i in hold]
    hold_labels = [labels[i] for i in hold]
    rel_test = test_frac/(test_frac+val_frac) if (test_frac+val_frac)>0 else 0.5
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_test, random_state=seed)
    v_rel, t_rel = next(gss2.split(hold_idx, hold_labels, hold_groups))
    v_idx = [hold[i] for i in v_rel]
    t_idx = [hold[i] for i in t_rel]
    train = [records[i] for i in tr]
    val = [records[i] for i in v_idx]
    test = [records[i] for i in t_idx]
    return train, val, test

# ----------------- core per-file -----------------
def extract_candidates_for_file(
    client_tuple, model, file_path: Path, chunk_lines:int, temp:float, max_tokens:int,
    max_gadget_lines:int, few_shot:bool
) -> List[dict]:
    try:
        src = file_path.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        print(f"[WARN] cannot read {file_path}: {e}", file=sys.stderr)
        return []
    chunks = chunk_text_lines(src, chunk_lines)
    system = "You output STRICT JSON for vulnerability gadgets."
    cands: List[dict] = []

    for chunk_text, start_line, end_line in chunks:
        # 번호를 붙여 slice 동기화 정확도 향상
        numbered = add_line_numbers(chunk_text)
        prompt = PROMPT_TEMPLATE.format(
            strict_policy=STRICT_POLICY,
            few_shot=(FEW_SHOT if few_shot else ""),
            source=numbered,
            max_gadget_lines=max_gadget_lines
        )
        try:
            resp_text = call_llm(client_tuple, model, system, prompt, max_tokens=max_tokens, temperature=temp)
        except Exception as e:
            print(f"[WARN] LLM failed {file_path}({start_line}-{end_line}): {e}", file=sys.stderr)
            continue
        try:
            obj = safe_parse_json_only(resp_text)
        except Exception as e:
            # 보수적 후보 (불확실)
            cands.append({
                "id": f"{file_path}:{start_line}", "file": str(file_path),
                "slice_begin": start_line, "slice_end": end_line,
                "gadget_code":"", "label": 0, "score": 0.0,
                "reason": f"parse_fail:{e}"
            })
            continue
        obj = enforce_schema(obj, file_path, chunk_len=(end_line-start_line+1), max_gadget_lines=max_gadget_lines)
        # chunk 로컬 인덱스 → 원본 라인으로 보정
        if obj["slice_begin"]>=1 and obj["slice_end"]>=obj["slice_begin"]:
            obj["slice_begin"] = start_line + (obj["slice_begin"]-1)
            obj["slice_end"]   = start_line + (obj["slice_end"]-1)
        obj["id"] = f"{file_path}:{obj['slice_begin']}"
        cands.append(obj)
        time.sleep(0.08)  # rate friendly
    return cands

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser(description="Create CodeBERT dataset from Juliet using an LLM (strict).")
    ap.add_argument("--juliet-root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--chunk-lines", type=int, default=300)
    ap.add_argument("--max-gadget-lines", type=int, default=30)

    # 신뢰도 임계치(양/음 분리)
    ap.add_argument("--min-conf-pos", type=float, default=0.72)
    ap.add_argument("--min-conf-neg", type=float, default=0.55)
    ap.add_argument("--pseudo-conf", type=float, default=0.50)
    ap.add_argument("--include-pseudo", action="store_true")

    # 품질 필터
    ap.add_argument("--require-sink-for-pos", action="store_true", help="label=1일 때 gadget_code에 싱크 토큰이 없으면 폐기")
    ap.add_argument("--extra-sinks", help="추가 싱크 토큰 파일(개행구분)")

    # 데이터 균형/제한
    ap.add_argument("--max-per-file", type=int, default=2, help="파일당 최대 샘플 수(예: 2=양1/음1 권장)")
    ap.add_argument("--balance-target", type=float, default=0.5, help="최종 라벨1 비율 목표(0~1)")
    ap.add_argument("--balance-tolerance", type=float, default=0.15, help="허용 편차 (예: 0.15 → 0.35~0.65)")

    # 분할/클린업
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dedupe", action="store_true")
    ap.add_argument("--few-shot", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    random.seed(args.seed)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    root = Path(args.juliet_root)
    files = gather_source_files(root)
    if not files:
        print(f"[ERROR] no source files under {root}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] files: {len(files)}")

    # sink vocab
    sink_vocab = set(BASE_SINK_TOKENS)
    if args.extra_sinks:
        try:
            with open(args.extra_sinks, "r", encoding="utf-8") as f:
                extra = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
            sink_vocab.update(extra)
            print(f"[INFO] extra sinks +{len(extra)}")
        except Exception as e:
            print(f"[WARN] cannot load extra sinks: {e}", file=sys.stderr)

    client_tuple = get_openai_client()
    print("[INFO] OpenAI ready:", client_tuple[1], "model:", args.model)

    # ---- parallel LLM ----
    def _job(fpath: Path):
        try:
            cands = extract_candidates_for_file(
                client_tuple, args.model, fpath, args.chunk_lines, args.temp, args.max_tokens,
                args.max_gadget_lines, args.few_shot
            )
            if not cands:
                # 경로 휴리스틱
                lab = infer_label_from_path(str(fpath))
                return [{"id": f"{fpath}:1","file": str(fpath),
                         "slice_begin":1,"slice_end":1,"gadget_code":"",
                         "label": (lab if lab in (0,1) else 0),
                         "score": 0.0,"reason":"no_candidate"}]
            return cands
        except Exception as e:
            return [{"id": f"{fpath}:1","file": str(fpath),
                     "slice_begin":1,"slice_end":1,"gadget_code":"",
                     "label":0,"score":0.0,"reason":f"exception:{e}"}]

    records_raw: List[dict] = []
    print(f"[INFO] run with {args.workers} workers")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_job, f) for f in files]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="LLM"):
            for rec in fut.result():
                records_raw.append(rec)

    # 저장(raw)
    raw_path = outdir/"llm_raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as w:
        for r in records_raw:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[INFO] wrote raw:", raw_path, "count:", len(records_raw))

    # ---- 파일당 후보 선택: best pos(옵션) + best neg ----
    by_file: Dict[str, List[dict]] = defaultdict(list)
    for r in records_raw:
        by_file[r["file"]].append(r)

    selected: List[dict] = []
    for f, lst in by_file.items():
        # 정렬: 점수 내림차순
        lst_sorted = sorted(lst, key=lambda x: float(x.get("score",0.0)), reverse=True)
        pos_pick = None; neg_pick = None
        for r in lst_sorted:
            lab = int(r.get("label",0)); sc = float(r.get("score",0.0))
            gtxt = r.get("gadget_code","")
            if lab == 1:
                if sc >= args.min_conf_pos:
                    if (not args.require_sink_for_pos) or contains_sink(gtxt, sink_vocab):
                        pos_pick = r; break
        for r in lst_sorted:
            lab = int(r.get("label",0)); sc = float(r.get("score",0.0))
            if lab == 0 and sc >= args.min_conf_neg:
                neg_pick = r; break

        bucket = []
        if pos_pick: bucket.append(pos_pick)
        if neg_pick: bucket.append(neg_pick)
        if not bucket:
            # fallback: 최고점 1개라도 보관(데이터 부족 방지)
            bucket = lst_sorted[:1]
        # 파일당 최대 개수 제한
        if args.max_per_file > 0:
            bucket = bucket[:args.max_per_file]
        selected.extend(bucket)

    print(f"[INFO] selected after per-file picking: {len(selected)}")

    # ---- dedupe (텍스트 기준) ----
    final = selected
    if args.dedupe:
        before = len(final)
        final = dedupe_by_text(final)
        print(f"[INFO] dedupe {before} -> {len(final)}")

    # ---- pseudo 포함/제외 ----
    confident, pseudo, uncertain = [], [], []
    for r in final:
        sc = float(r.get("score",0.0))
        lab = int(r.get("label",0))
        if (lab==1 and sc>=args.min_conf_pos) or (lab==0 and sc>=args.min_conf_neg):
            confident.append(r)
        elif sc >= args.pseudo_conf:
            pseudo.append(r)
        else:
            uncertain.append(r)
    used = confident + (pseudo if args.include_pseudo else [])
    used = [r for r in used if int(r.get("label",0)) in (0,1)]

    # ---- 라벨 균형 다운샘플링 ----
    if used:
        p_ratio = sum(1 for r in used if r["label"]==1)/len(used)
        lo = args.balance_target - args.balance_tolerance
        hi = args.balance_target + args.balance_tolerance
        if not (lo <= p_ratio <= hi):
            # 과다 클래스 다운샘플링
            pos = [r for r in used if r["label"]==1]
            neg = [r for r in used if r["label"]==0]
            target_pos = int(round(args.balance_target * len(used)))
            target_neg = len(used) - target_pos
            if len(pos) > target_pos:
                random.shuffle(pos); pos = pos[:max(1,target_pos)]
            if len(neg) > target_neg:
                random.shuffle(neg); neg = neg[:max(1,target_neg)]
            used = pos + neg
            random.shuffle(used)
            print(f"[INFO] rebalanced to ~{args.balance_target:.2f} (size={len(used)})")

    if not used:
        print("[WARN] no samples after filtering; try lowering thresholds or enabling --include-pseudo", file=sys.stderr)
        (outdir/"uncertain.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in uncertain), encoding="utf-8")
        sys.exit(0)

    # ---- CodeBERT 필드 정리 ----
    for r in used:
        r.setdefault("id", f"{r.get('file')}:{r.get('slice_begin',1)}")
        r["text"] = r.get("gadget_code","")
        r["label"] = int(r.get("label",0))

    # ---- split ----
    train, val, test = safe_group_split(used, args.train_frac, args.val_frac, args.test_frac, seed=args.seed)
    print(f"[INFO] split -> train:{len(train)} val:{len(val)} test:{len(test)}")

    def _write(lst, path: Path):
        with path.open("w", encoding="utf-8") as w:
            for r in lst:
                out = {"id": r.get("id"), "file": r.get("file"), "text": r.get("text",""), "label": int(r.get("label",0))}
                w.write(json.dumps(out, ensure_ascii=False) + "\n")

    _write(train, outdir/"train.jsonl")
    _write(val,   outdir/"validation.jsonl")
    _write(test,  outdir/"test.jsonl")

    with (outdir/"llm_filtered.jsonl").open("w", encoding="utf-8") as w:
        for r in used:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (outdir/"uncertain.jsonl").open("w", encoding="utf-8") as w:
        for r in uncertain:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 통계
    labcnt = Counter(r["label"] for r in used)
    print("[DONE]", outdir)
    print(f"  used={len(used)} (label1={labcnt.get(1,0)}, label0={labcnt.get(0,0)})")
    print(f"  confident={len(confident)}, pseudo={len(pseudo)}, uncertain={len(uncertain)}")

if __name__ == "__main__":
    main()
