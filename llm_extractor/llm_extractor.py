#!/usr/bin/env python3
"""
prepare_llm_gadgets_for_codebert.py

Usage (example):
  # generate LLM gadgets for Juliet tree and prepare datasets (merged mode)
   python llm_extractor.py \
    --juliet-root ../../C/testcases/CWE90_LDAP_Injection/CWE90_LDAP_Injection__w32_char_connect_socket_01 \
    --outdir ./data/llm/juliet_codebert_dataset \
    --mode merged \
    --model gpt-4o-mini \
    --temp 0.0 \
    --max-tokens 512 \
    --seed 42

If you already have static extractor output:
  --static-jsonl /path/to/static_gadgets.jsonl

Requirements:
  pip install -U openai tqdm scikit-learn python-dotenv
Environment:
  export OPENAI_API_KEY="sk-..."
"""
import argparse
import json
import os
import sys
import time
import random
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from collections import defaultdict, Counter

# optional sklearn for group split
try:
    from sklearn.model_selection import GroupShuffleSplit
except Exception:
    GroupShuffleSplit = None

# optional: load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------------------
# Config / defaults
# -------------------
DEFAULT_MODEL = "gpt-4o-mini"           # default LLM model (change if needed)
DEFAULT_TEMP = 0.0
MAX_RETRIES = 5
RETRY_BACKOFF = 2.0

# -------------------
# LLM client compatibility (legacy openai or new openai.OpenAI client)
# -------------------
def get_openai_client():
    """
    Return a tuple (client_obj, client_type) where client_type is 'openai_new' or 'openai_legacy'.
    Tries new interface first, falls back to legacy package.
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set. export OPENAI_API_KEY=...")

    # prefer new client (openai>=1.0.0)
    try:
        from openai import OpenAI
        # Some installations accept api_key argument, others expect env var
        try:
            client = OpenAI(api_key=key)
        except Exception:
            client = OpenAI()
        return client, "openai_new"
    except Exception:
        pass

    # fallback to legacy openai package
    try:
        import openai as legacy_openai
        legacy_openai.api_key = key
        return legacy_openai, "openai_legacy"
    except Exception:
        raise RuntimeError("Could not import OpenAI client. pip install openai")


def call_llm(client_tuple, model: str, prompt_system: str, prompt_user: str,
             max_tokens:int=512, temperature:float=0.0) -> str:
    """
    Calls the LLM and returns text output.
    client_tuple: (client, client_type) where client_type is 'openai_new' or 'openai_legacy'
    Robust to several openai python client versions.
    """
    client, ctype = client_tuple
    for attempt in range(1, MAX_RETRIES+1):
        try:
            if ctype == "openai_new":
                # modern interface: try common variants
                # 1) client.chat.completions.create(...)
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role":"system", "content": prompt_system},
                            {"role":"user", "content": prompt_user}
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    # extract content
                    if hasattr(resp, "choices") and len(resp.choices) > 0:
                        c0 = resp.choices[0]
                        # try nested .message.content
                        msg = getattr(c0, "message", None)
                        if msg:
                            text = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
                            if text:
                                return text.strip()
                        # try .text
                        text = getattr(c0, "text", None)
                        if text:
                            return text.strip()
                    # fallback stringified response
                    return str(resp).strip()
                except Exception:
                    # 2) client.chat.create(...)
                    try:
                        resp = client.chat.create(
                            model=model,
                            messages=[{"role":"system", "content": prompt_system},
                                      {"role":"user", "content": prompt_user}],
                            max_tokens=max_tokens,
                            temperature=temperature
                        )
                        if hasattr(resp, "choices") and len(resp.choices) > 0:
                            c0 = resp.choices[0]
                            msg = getattr(c0, "message", None)
                            if msg:
                                return getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else str(resp))
                        return str(resp).strip()
                    except Exception as e_inner:
                        raise e_inner

            else:
                # legacy openai package
                resp = client.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role":"system", "content": prompt_system},
                        {"role":"user", "content": prompt_user}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                # typical shape: resp.choices[0].message.content
                return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = RETRY_BACKOFF ** attempt
            print(f"[WARN] LLM request failed (attempt {attempt}/{MAX_RETRIES}): {e}. Retrying in {wait:.1f}s...", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("LLM request failed after retries.")


# -------------------
# Prompt template
# -------------------
PROMPT_TEMPLATE = """
You are an automated extractor. Input is a single C/C++ source file (or a chunk). Your job: RETURN ONLY a single JSON object (no surrounding text) describing a minimal code gadget extracted from the input.

Rules:
1) "code gadget" = the smallest self-contained contiguous snippet that contains a suspicious sink or the statement most likely interesting for vulnerability classification (e.g., strcpy/memcpy/sprintf/recv/read/exec). Include small surrounding lines necessary to understand variables (keep minimal).
2) The JSON keys MUST be exactly: id, file, slice_begin, slice_end, gadget_code
   - id: unique id string (e.g., path:slice_begin)
   - file: original file path (as provided)
   - slice_begin: 1-based line number where gadget begins
   - slice_end: 1-based line number where gadget ends (inclusive)
   - gadget_code: the literal code snippet; preserve newlines as \\n in JSON.
3) If you cannot find any suspicious sink, return JSON object with slice_begin and slice_end set to 1 and gadget_code set to "" (empty string).
4) Produce only ONE JSON object, no arrays, no commentary.

INPUT (below) is delimited by lines:
-----SOURCE-BEGIN-----
{source}
-----SOURCE-END-----

Analyze and output the JSON object now.
"""

# -------------------
# Helpers: files, chunking, JSON parse
# -------------------
def gather_source_files(juliet_root: Path, exts=(".c", ".cpp", ".cc", ".h", ".hpp")) -> List[Path]:
    out = []
    if not juliet_root.exists():
        return []
    for p in juliet_root.rglob("*"):
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
        raise ValueError("Response does not contain a JSON object.")
    try:
        return json.loads(text[start:end+1])
    except Exception as e:
        # try to clean trailing commas, stray backticks, etc.
        cleaned = text[start:end+1]
        cleaned = cleaned.replace("`", "")
        cleaned = re.sub(r",\s*}", "}", cleaned)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        return json.loads(cleaned)

def normalize_code_for_model(code: str) -> str:
    if not code:
        return ""
    code = re.sub(r'/\*.*?\*/', ' ', code, flags=re.S)
    code = re.sub(r'//.*?$', ' ', code, flags=re.M)
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = re.sub(r'\n\s*\n+', '\n', code)
    code = "\n".join([ln.rstrip() for ln in code.splitlines() if ln.strip() != ""])
    return code.strip()

def dedupe_by_text(records: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for r in records:
        t = r.get("gadget_code") or r.get("gadget_raw") or r.get("text") or ""
        h = hashlib.sha256(t.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(r)
    return out

# -------------------
# Label heuristics (Juliet): path contains '/bad' or '/good'
# -------------------
def infer_label_from_path(path: str) -> int:
    """
    Enhanced label inference:
    - path/filename heuristics (existing)
    - then file content heuristics:
        - look for 'OMITBAD' / 'OMITGOOD' macros
        - look for 'void ... bad(' or 'void ... good(' patterns
        - look for comment blocks mentioning 'BadSource' / 'GoodSource' or 'BadSink' / 'GoodSink'
        - fallback: -1
    """
    lowerp = path.replace("\\","/").lower()
    # quick path-based checks (existing)
    if "/bad" in lowerp or "/bad." in lowerp or lowerp.endswith("bad.c") or lowerp.endswith("bad.cpp") or "/omittedbad" in lowerp:
        return 1
    if "/good" in lowerp or "/good." in lowerp or lowerp.endswith("good.c") or lowerp.endswith("good.cpp") or "/omitgood" in lowerp:
        return 0

    fname = Path(path).name.lower()
    if fname.endswith("_bad.c") or fname.endswith("_bad.cpp") or "_bad" in fname:
        return 1
    if fname.endswith("_good.c") or fname.endswith("_good.cpp") or "_good" in fname:
        return 0

    # Try to inspect file content for stronger hints
    try:
        txt = Path(path).read_text(encoding='utf-8', errors='ignore')
        ltxt = txt.lower()
    except Exception:
        ltxt = ""

    # macro-based explicit markers
    if "omitbad" in ltxt:
        return 1
    if "omitgood" in ltxt:
        return 0

    # look for 'void ... bad(' or 'void ... good(' or 'cwe..._bad' patterns
    if re.search(r'\bvoid\b\s+[a-z0-9_]*\b(?:bad|badsink)\s*\(', ltxt) or re.search(r'\b_bad\b', fname):
        return 1
    if re.search(r'\bvoid\b\s+[a-z0-9_]*\b(?:good|goodsink)\s*\(', ltxt) or re.search(r'\b_good\b', fname):
        return 0

    # look into the header comment/template block for "BadSource" / "GoodSource" / "BadSink" / "GoodSink"
    if re.search(r'badsource', ltxt) or re.search(r'badsink', ltxt) or re.search(r'@description.*bad', ltxt):
        return 1
    if re.search(r'goodsource', ltxt) or re.search(r'goodsink', ltxt) or re.search(r'@description.*good', ltxt):
        return 0

    # another cue: presence of 'fgets' or 'recv' etc isn't decisive; but presence of free-not-at-start pattern?
    # (We avoid overfitting — only use high confidence signals above.)

    return -1


# -------------------
# Core: generate LLM gadget for a single file (uses chunking, returns first non-empty)
# -------------------
def generate_gadget_for_file(client_tuple, model, file_path: Path, chunk_lines=400, temp=DEFAULT_TEMP, max_tokens=512) -> Optional[dict]:
    try:
        src = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARN] could not read {file_path}: {e}", file=sys.stderr)
        return None
    chunks = chunk_text_lines(src, chunk_lines)
    system = "You are a strict JSON-only code gadget extractor."
    chosen = None
    for chunk_text, start_line, end_line in chunks:
        prompt = PROMPT_TEMPLATE.format(source=chunk_text)
        try:
            resp_text = call_llm(client_tuple, model, system, prompt, max_tokens=max_tokens, temperature=temp)
        except Exception as e:
            print(f"[WARN] LLM failed for {file_path} chunk ({start_line}-{end_line}): {e}", file=sys.stderr)
            continue
        try:
            obj = safe_parse_json_only(resp_text)
        except Exception as e:
            print(f"[WARN] JSON parse failed for {file_path} chunk ({start_line}-{end_line}): {e}", file=sys.stderr)
            continue

        # validate keys
        required = {"id","file","slice_begin","slice_end","gadget_code"}
        if not required.issubset(obj.keys()):
            print(f"[WARN] LLM returned missing keys for {file_path} chunk ({start_line}-{end_line}): {list(obj.keys())}", file=sys.stderr)
            # attempt best-effort mapping if possible
            if "gadget_raw" in obj:
                obj["gadget_code"] = obj.pop("gadget_raw")
            else:
                continue

        # normalize code text
        obj["gadget_code"] = normalize_code_for_model(obj.get("gadget_code",""))
        # ensure file field refers to original path
        obj["file"] = str(file_path)
        # ensure numeric slice values
        try:
            obj["slice_begin"] = int(obj.get("slice_begin",1))
            obj["slice_end"]   = int(obj.get("slice_end", obj["slice_begin"]))
        except Exception:
            obj["slice_begin"], obj["slice_end"] = 1, 1

        # if gadget_code not empty accept, else keep as fallback (first)
        if obj.get("gadget_code","").strip():
            chosen = obj
            break
        else:
            if chosen is None:
                chosen = obj
    # fallback if none
    if chosen is None:
        chosen = {"id": f"{file_path}:0", "file": str(file_path), "slice_begin":1, "slice_end":1, "gadget_code": ""}
    return chosen

# -------------------
# Safe splitting logic (group-aware fallback)
# -------------------
def safe_split_records(records: List[dict], train_frac: float, val_frac: float, test_frac: float, seed: int = 42) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Try group-aware split (GroupShuffleSplit). If dataset too small or groups too few,
    fallback to a deterministic safe random split that guarantees at least one training sample when possible.
    """
    random.seed(seed)
    n = len(records)
    if n == 0:
        return [], [], []

    # Build groups (file path)
    groups = [r.get("file","") for r in records]
    unique_groups = len(set(groups))

    # If group splitting not possible, use safe random split
    if GroupShuffleSplit is None or unique_groups < 3 or n < 6:
        # compute counts but guarantee at least 1 train
        ntrain = max(1, int(round(train_frac * n)))
        nval = int(round(val_frac * n))
        if ntrain + nval >= n:
            # reduce nval if necessary, ensure at least 1 test if fractions asked for
            ntrain = max(1, n - max(1, int(round(test_frac * n))))
            nval = max(0, n - ntrain - 1)
        ntest = n - ntrain - nval
        if ntest < 0:
            ntest = 0
            if ntrain + nval > n:
                nval = max(0, n - ntrain)
        items = list(records)
        random.shuffle(items)
        train = items[:ntrain]
        val = items[ntrain:ntrain+nval]
        test = items[ntrain+nval:]
        return train, val, test

    # group-aware splitting
    indices = list(range(n))
    labels = [r.get("label") for r in records]
    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=(val_frac + test_frac), random_state=seed)
        train_idx, hold_idx = next(gss.split(indices, labels, groups))
    except Exception:
        # fallback to random deterministic split
        items = list(records)
        random.shuffle(items)
        ntrain = max(1, int(round(train_frac * n)))
        nval = int(round(val_frac * n))
        train = items[:ntrain]
        val = items[ntrain:ntrain+nval]
        test = items[ntrain+nval:]
        return train, val, test

    hold_indices = [indices[i] for i in hold_idx]
    hold_groups = [groups[i] for i in hold_idx]
    hold_labels = [labels[i] for i in hold_idx]

    # If hold set too small or has too few groups to split, split hold randomly
    if len(hold_indices) < 2 or len(set(hold_groups)) < 2:
        hold_items = [records[i] for i in hold_indices]
        random.shuffle(hold_items)
        nhold = len(hold_items)
        rel_test_frac = test_frac / (test_frac + val_frac) if (test_frac + val_frac) > 0 else 0.5
        ntest = int(round(rel_test_frac * nhold))
        nval = nhold - ntest
        val_items = hold_items[:nval]
        test_items = hold_items[nval:]
        train = [records[i] for i in train_idx]
        return train, val_items, test_items

    # Normal group-aware split on hold
    rel_test_frac = test_frac / (test_frac + val_frac) if (test_frac + val_frac) > 0 else 0.5
    try:
        gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_test_frac, random_state=seed)
        val_rel_idx, test_rel_idx = next(gss2.split(hold_indices, hold_labels, hold_groups))
    except Exception:
        hold_items = [records[i] for i in hold_indices]
        random.shuffle(hold_items)
        nhold = len(hold_items)
        ntest = int(round(rel_test_frac * nhold))
        nval = nhold - ntest
        val_items = hold_items[:nval]
        test_items = hold_items[nval:]
        train = [records[i] for i in train_idx]
        return train, val_items, test_items

    val_idx = [hold_idx[i] for i in val_rel_idx]
    test_idx = [hold_idx[i] for i in test_rel_idx]
    train = [records[i] for i in train_idx]
    val = [records[i] for i in val_idx]
    test = [records[i] for i in test_idx]
    return train, val, test

# -------------------
# High-level pipeline
# -------------------
def main():
    parser = argparse.ArgumentParser(description="Generate LLM gadgets from Juliet and prepare CodeBERT dataset.")
    parser.add_argument("--juliet-root", required=True, help="Path to Juliet C testcases root (e.g., .../C/testcases)")
    parser.add_argument("--outdir", required=True, help="Output directory for datasets")
    parser.add_argument("--static-jsonl", default=None, help="Optional precomputed static gadgets JSONL")
    parser.add_argument("--mode", choices=("static","llm","merged"), default="merged", help="Which dataset to prepare")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model to call")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--chunk-lines", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dedupe", action="store_true", help="Deduplicate gadget texts")
    parser.add_argument("--cap-per-file", type=int, default=0, help="If >0, keep at most N gadgets per file (per label)")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM generation (use only static if provided)")
    args = parser.parse_args()

    random.seed(args.seed)

    juliet_root = Path(args.juliet_root)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # load static gadgets if provided
    static_gadgets = []
    if args.static_jsonl:
        sp = Path(args.static_jsonl)
        if sp.exists():
            with sp.open('r',encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    if "gadget_raw" in obj and "gadget_code" not in obj:
                        obj["gadget_code"] = normalize_code_for_model(obj.get("gadget_raw",""))
                    static_gadgets.append(obj)
            print(f"[INFO] Loaded {len(static_gadgets)} static gadgets from {sp}")

    # gather source files
    print("[INFO] gathering source files...")
    files = gather_source_files(juliet_root)
    print(f"[INFO] found {len(files)} candidate source files")
    if len(files) == 0 and not args.static_jsonl:
        print(f"[ERROR] No source files found under: {juliet_root}. Exiting.", file=sys.stderr)
        sys.exit(1)

    # prepare LLM client
    client_tuple = None
    if not args.skip_llm and args.mode in ("llm","merged"):
        try:
            client_tuple = get_openai_client()
            print("[INFO] OpenAI client ready:", client_tuple[1])
        except Exception as e:
            print("[ERROR] OpenAI client init failed:", e, file=sys.stderr)
            sys.exit(2)

    llm_gadgets = []
    if args.mode in ("llm","merged") and not args.skip_llm:
        print("[INFO] Generating LLM gadgets (this may take a while)...")
        for f in tqdm(files, desc="LLM generate"):
            try:
                g = generate_gadget_for_file(client_tuple, args.model, f, chunk_lines=args.chunk_lines, temp=args.temp, max_tokens=args.max_tokens)
                if g:
                    g["label"] = infer_label_from_path(g["file"])
                    llm_gadgets.append(g)
            except KeyboardInterrupt:
                print("Interrupted by user"); break
            except Exception as e:
                print(f"[WARN] generator failed for {f}: {e}", file=sys.stderr)
        print(f"[INFO] generated {len(llm_gadgets)} LLM gadgets")

        # write raw
        out_llm = outdir / "llm_gadgets.jsonl"
        with out_llm.open('w',encoding='utf-8') as w:
            for r in llm_gadgets:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("[INFO] Wrote LLM gadgets ->", out_llm)

    # pick final dataset
    final_records = []
    if args.mode == "static":
        final_records = static_gadgets
    elif args.mode == "llm":
        final_records = llm_gadgets
    else:  # merged
        final_records = list(static_gadgets)
        texts = set()
        for r in final_records:
            texts.add(hashlib.sha256((r.get("gadget_code","") or "").encode("utf-8")).hexdigest())
        for r in llm_gadgets:
            h = hashlib.sha256((r.get("gadget_code","") or "").encode("utf-8")).hexdigest()
            if h not in texts:
                final_records.append(r)
                texts.add(h)
    print(f"[INFO] initial combined records: {len(final_records)}")

    # dedupe if requested
    if args.dedupe:
        before = len(final_records)
        final_records = dedupe_by_text(final_records)
        print(f"[INFO] deduped: {before} -> {len(final_records)}")

    # cap per file if requested
    if args.cap_per_file > 0:
        cap = args.cap_per_file
        kept = []
        counts = defaultdict(int)
        for r in final_records:
            key = (r.get("file"), r.get("label", -1))
            if counts[key] < cap:
                kept.append(r)
                counts[key] += 1
        final_records = kept
        print(f"[INFO] applied cap_per_file={cap}, now {len(final_records)} records")

    # ensure label populated (infer if missing)
    for r in final_records:
        if "label" not in r or r.get("label") is None:
            r["label"] = infer_label_from_path(r.get("file",""))

    # filter out ambiguous label -1
    ambiguous = [r for r in final_records if r.get("label") == -1]
    if ambiguous:
        print(f"[WARN] {len(ambiguous)} records have ambiguous label -1; they will be dropped from dataset")
        final_records = [r for r in final_records if r.get("label") in (0,1)]

    # final stats
    cnt = Counter([r.get("label") for r in final_records])
    print("[INFO] final label counts:", dict(cnt), "total:", len(final_records))

    # if nothing to split -> safely write empty outputs and exit
    if len(final_records) == 0:
        print("[WARN] No final records to split; writing empty dataset files and exiting.")
        for name in ("train.jsonl","validation.jsonl","test.jsonl","llm_gadgets.jsonl"):
            p = outdir / name
            if not p.exists():
                p.write_text("", encoding="utf-8")
        sys.exit(0)

    # use safe splitter
    train, val, test = safe_split_records(final_records, args.train_frac, args.val_frac, args.test_frac, seed=args.seed)

    # write outputs
    out_train = outdir / "train.jsonl"
    out_val   = outdir / "validation.jsonl"
    out_test  = outdir / "test.jsonl"

    for p, recs in [(out_train, train), (out_val, val), (out_test, test)]:
        with p.open('w', encoding='utf-8') as w:
            for r in recs:
                text = r.get("gadget_code") or r.get("gadget_raw") or ""
                outobj = {
                    "id": r.get("id", f"{r.get('file')}:{r.get('slice_begin',1)}"),
                    "file": r.get("file"),
                    "text": text,
                    "label": int(r.get("label"))
                }
                w.write(json.dumps(outobj, ensure_ascii=False) + "\n")
        print("[INFO] wrote", p, "count:", len(recs))

    print("[DONE] dataset ready in", outdir)

if __name__ == "__main__":
    main()
