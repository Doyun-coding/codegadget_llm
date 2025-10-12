# normalize_gadgets.py
"""
python ./static_extractor/normalize_gadgets.py \
./data/static/juliet_gadgets_smart_labeled.jsonl \
./data/static/juliet_gadgets_normalized.jsonl
"""
import json, re, sys
from pathlib import Path

def normalize_code(code, max_chars=2000):
    if not code:
        return ""
    # remove C-style block comments and // comments
    code = re.sub(r'/\*.*?\*/', ' ', code, flags=re.S)
    code = re.sub(r'//.*?$', ' ', code, flags=re.M)
    # collapse multiple whitespace into single space but keep newlines for readability
    code = re.sub(r'\r\n?', '\n', code)
    # remove excessive newlines
    code = re.sub(r'\n\s*\n+', '\n', code)
    # strip leading/trailing spaces
    code = code.strip()
    # optional: if extremely long, truncate keeping sink line centered
    if len(code) > max_chars:
        return code[:max_chars]
    return code

if __name__ == "__main__":
    if len(sys.argv)<3:
        print("Usage: python normalize_gadgets.py in.jsonl out.jsonl")
        sys.exit(1)
    inp = Path(sys.argv[1])
    out = Path(sys.argv[2])
    with inp.open('r',encoding='utf-8') as inf, out.open('w',encoding='utf-8') as outf:
        for line in inf:
            obj = json.loads(line)
            code = obj.get('gadget_raw') or obj.get('gadget_code') or ""
            text = normalize_code(code)
            # keep existing label
            lab = obj.get('label', -1)
            if lab not in (0,1):
                # skip uncertain by default (or write with label -1)
                continue
            out_obj = {"id": obj.get("id"), "file": obj.get("file"), "text": text, "label": int(lab)}
            outf.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
    print("Wrote normalized:", out)
