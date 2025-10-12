# convert_juliet_jsonl_label.py
import json
from pathlib import Path
import sys

"""
python convert_juliet_jsonl_label.py juliet_gadgets.jsonl juliet_gadgets_labeled.jsonl
"""

infile = Path(sys.argv[1])
outfile = Path(sys.argv[2]) if len(sys.argv)>2 else infile.with_name(infile.stem + "_labeled.jsonl")

with infile.open('r', encoding='utf-8') as r, outfile.open('w', encoding='utf-8') as w:
    for line in r:
        obj = json.loads(line)
        p = obj.get('file','')
        # 경로에 'bad'가 포함되면 취약, 'good' 포함하면 비취약
        lab = None
        lowerp = p.lower()
        if '/bad' in lowerp or '\\bad' in lowerp or '/bad.' in lowerp or 'bad.c' in lowerp:
            lab = 1
        elif '/good' in lowerp or '\\good' in lowerp or '/good.' in lowerp or 'good.c' in lowerp:
            lab = 0
        else:
            # 못 판별하면 -1 (나중에 수동검토)
            lab = -1

        # 필드명 변환 (원하시면 다른 mapping으로 변경)
        new = {
            "id": f"{obj.get('file','')}: {obj.get('slice_begin',1)}",
            "file": obj.get('file'),
            "slice_begin": obj.get('slice_begin'),
            "slice_end": obj.get('slice_end'),
            "gadget_code": obj.get('gadget_raw') or obj.get('gadget_code') or "",
            "sink_name": obj.get('sink_name'),
            "cwe_candidates": obj.get('cwe_candidates', []),
            "label": lab
        }
        w.write(json.dumps(new, ensure_ascii=False) + "\n")

print("Wrote:", outfile)
