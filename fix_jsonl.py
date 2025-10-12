# fix_jsonl_schema.py
import json, argparse, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Normalize CodeBERT JSONL schema")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    n_in = n_out = 0
    with open(args.inp, "r", encoding="utf-8") as r, open(args.out, "w", encoding="utf-8") as w:
        for ln_no, line in enumerate(r, 1):
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] skip line {ln_no}: invalid JSON: {e}", file=sys.stderr)
                continue

            # normalize types
            obj_id = obj.get("id", "")
            if not isinstance(obj_id, str):
                obj_id = str(obj_id)

            obj_file = str(obj.get("file", ""))
            obj_text = str(obj.get("text", ""))
            try:
                obj_label = int(obj.get("label", 0))
            except Exception:
                obj_label = 0

            fixed = {"id": obj_id, "file": obj_file, "text": obj_text, "label": obj_label}
            w.write(json.dumps(fixed, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[OK] {args.inp} -> {args.out}  ({n_out}/{n_in} lines kept)")

if __name__ == "__main__":
    main()
