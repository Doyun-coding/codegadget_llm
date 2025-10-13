#!/usr/bin/env python3
# relabel_smart.py

"""
python relabel_smart.py \
./static_extractor/juliet_gadgets.jsonl \
./data/juliet_gadgets_smart_labeled.jsonl \
"""
import argparse, json, re, os
from pathlib import Path
from collections import Counter

BAD_PATTERNS = [
    re.compile(r'\bvoid\s+bad\b', re.I),
    re.compile(r'\bbadSink\b', re.I),
    re.compile(r'\bbad\s*\(', re.I),
    re.compile(r'\bPOTENTIAL FL?AW\b', re.I),
    re.compile(r'\bBadSink\b', re.I)
]
GOOD_PATTERNS = [
    re.compile(r'\bvoid\s+good\b', re.I),
    re.compile(r'\bgoodB2G\b', re.I),
    re.compile(r'\bgoodG2B\b', re.I),
    re.compile(r'\bgood\s*\(', re.I),
    re.compile(r'OMITBAD', re.I),  # presence of OMITBAD often means GOOD included
    re.compile(r'OMITGOOD', re.I)
]

def check_patterns(text, patterns):
    for p in patterns:
        if p.search(text):
            return True
    return False

def resolve_path(fp, juliet_root):
    if not fp:
        return None
    if os.path.isabs(fp):
        return fp if os.path.exists(fp) else None
    if juliet_root:
        cand = os.path.join(juliet_root, fp)
        if os.path.exists(cand):
            return cand
    # try relative to cwd
    if os.path.exists(fp):
        return fp
    # try normalized ../ style
    try:
        cand = os.path.normpath(fp)
        if os.path.exists(cand):
            return cand
    except:
        pass
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input', help='input jsonl')
    ap.add_argument('output', nargs='?', help='output jsonl', default=None)
    ap.add_argument('--juliet-root', help='root dir to resolve relative file paths', default=None)
    ap.add_argument('--force', action='store_true', help='force overwrite existing label (default: keep existing)')
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output) if args.output else inp.with_name(inp.stem + "_smart_labeled.jsonl")
    julroot = args.juliet_root

    cnt = Counter()
    wrote = 0
    with inp.open('r', encoding='utf-8') as inf, out.open('w', encoding='utf-8') as outf:
        for i, line in enumerate(inf,1):
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] skip malformed line {i}: {e}")
                continue
            orig_label = obj.get('label', None)
            label = orig_label
            # if label exists and not forced, leave it
            if orig_label is None or args.force or int(orig_label) == -1:
                # 1) check gadget text first
                code = obj.get('gadget_raw') or obj.get('gadget_code') or ''
                if code:
                    is_bad = check_patterns(code, BAD_PATTERNS)
                    is_good = check_patterns(code, GOOD_PATTERNS)
                    if is_bad and not is_good:
                        label = 1
                    elif is_good and not is_bad:
                        label = 0
                    elif is_bad and is_good:
                        label = 1  # policy: prefer bad when ambiguous
                    else:
                        label = None
                # 2) if not decided, try resolve file path and read original
                if label is None:
                    fp = obj.get('file','')
                    realp = resolve_path(fp, julroot)
                    if realp:
                        try:
                            text = Path(realp).read_text(encoding='utf-8', errors='ignore')
                            is_bad = check_patterns(text, BAD_PATTERNS)
                            is_good = check_patterns(text, GOOD_PATTERNS)
                            if is_bad and not is_good:
                                label = 1
                            elif is_good and not is_bad:
                                label = 0
                            elif is_bad and is_good:
                                label = 1
                            else:
                                label = -1
                        except Exception:
                            label = -1
                    else:
                        label = -1
            # set and count
            obj['label'] = int(label) if label is not None else -1
            cnt[obj['label']] += 1
            outf.write(json.dumps(obj, ensure_ascii=False) + '\n')
            wrote += 1

    print(f"Wrote {wrote} records to {out}")
    print("Label counts:", dict(cnt))

if __name__ == '__main__':
    main()
