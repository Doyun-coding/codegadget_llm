#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
static_extractor.py
Tree-sitter 기반의 정적 가젯 추출 PoC (C/C++)

기능:
- tree-sitter 파서가 있으면 AST(call_expression)로 sink 함수 탐지
- 없으면 정규식 기반 fallback
- sink 함수의 인수에서 식별자를 추출하고, 파일 내에서 backward context를 수집하여 gadget_raw 생성
- 결과를 JSONL로 저장

Usage:
  python static_extractor.py /path/to/code_dir out_gadgets.jsonl \
        [--use-prebuilt] [--ts-lib /abs/path/to/my-languages.so] \
        [--build-grammar /path/to/grammar1 /path/to/grammar2 --out-lib build/my-languages.so]

Notes:
- To use tree-sitter AST path, install tree_sitter:
    pip install tree_sitter
  or install tree_sitter_languages:
    pip install tree_sitter_languages
- To build a language bundle (.so), you need the grammar repo(s) cloned (e.g. tree-sitter-c).
  Then call:
    python static_extractor.py --build-grammar path/to/tree-sitter-c --out-lib build/my-languages.so
  (This only builds lib and exits.)
"""

from pathlib import Path
import argparse
import os
import sys
import json
import re
import traceback

# Try to import tree_sitter
_HAS_TS = False
_HAS_TS_LANGS = False
try:
    from tree_sitter import Language, Parser
    _HAS_TS = True
except Exception:
    _HAS_TS = False

# Optional helper package providing prebuilt parsers
try:
    # tree_sitter_languages provides get_parser("c") convenience
    from tree_sitter_languages import get_parser
    _HAS_TS_LANGS = True
except Exception:
    _HAS_TS_LANGS = False

# ----------------- Configuration -----------------
# Default sink functions to detect (expand as needed)
SINK_FUNCS = ['strcpy', 'strcat', 'sprintf', 'vsprintf', 'gets', 'memcpy', 'strncpy', 'system']
# How many lines backwards to search for context
BACKWARD_LINES_WINDOW = 120
PRE_CONTEXT_LINES = 5
FALLBACK_CONTEXT_LINES = 12
# File extensions to include
CODE_EXTS = ('.c', '.cpp', '.cc', '.h', '.hpp')

# ----------------- Utilities -----------------
def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

def write_jsonl(records, outpath: str):
    outp = Path(outpath)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w', encoding='utf-8') as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + '\n')

# ----------------- Tree-sitter build helper -----------------
def build_tree_sitter_lib(grammar_dirs, out_lib_path):
    """
    Build a tree-sitter languages shared library from a list of grammar directories.
    grammar_dirs: list of paths to grammar repos (e.g. tree-sitter-c)
    out_lib_path: path to output .so/.dylib/.dll
    """
    if not _HAS_TS:
        raise RuntimeError("tree_sitter Python package is not installed; cannot build.")
    from tree_sitter import Language
    grammar_dirs = [str(Path(g).resolve()) for g in grammar_dirs]
    print(f"[build] Building tree-sitter library: {out_lib_path} from grammars: {grammar_dirs}")
    Language.build_library(out_lib_path, grammar_dirs)
    print(f"[build] Built: {out_lib_path}")
    return out_lib_path

# ----------------- Parser initialization -----------------
def init_parser(ts_lib: str = None, use_prebuilt: bool = False):
    """
    Initialize and return a tree-sitter Parser instance or None.
    - If use_prebuilt and tree_sitter_languages is installed, returns get_parser('c') parser.
    - Else if ts_lib provided and exists, loads Language(ts_lib, 'c') and returns Parser.
    - Else returns None (fallback to regex).
    """
    if use_prebuilt:
        if _HAS_TS_LANGS:
            try:
                print("[init] Using prebuilt parser from tree_sitter_languages.get_parser('c').")
                return get_parser("c")
            except Exception as e:
                print("[init] Failed to get prebuilt parser:", e, file=sys.stderr)
        else:
            print("[init] use_prebuilt requested but tree_sitter_languages not installed.", file=sys.stderr)

    if ts_lib:
        libp = Path(ts_lib)
        if libp.exists():
            if not _HAS_TS:
                print("[init] tree_sitter package not installed; cannot load ts lib.", file=sys.stderr)
                return None
            try:
                C_LANG = Language(str(libp), 'c')
                parser = Parser()
                parser.set_language(C_LANG)
                print(f"[init] Loaded tree-sitter lib: {libp}")
                return parser
            except Exception as e:
                print("[init] Failed to initialize parser from ts_lib:", e, file=sys.stderr)
                traceback.print_exc()
        else:
            print(f"[init] ts_lib path not found: {ts_lib}", file=sys.stderr)
    # fallback
    return None

# ----------------- Tree-sitter helpers -----------------
def _walk_nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        for ch in reversed(node.children):
            stack.append(ch)

def _node_text(node, code_bytes: bytes) -> str:
    return code_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')

def _find_call_expressions(parser, code: str):
    """
    Yield (call_node, start_byte) for each call_expression in the parsed AST.
    """
    if parser is None:
        return
    try:
        code_bytes = code.encode('utf-8')
        tree = parser.parse(code_bytes)
        root = tree.root_node
        for node in _walk_nodes(root):
            if node.type == 'call_expression':
                yield node, node.start_byte
    except Exception:
        # parser error - yield nothing
        return

def _extract_function_name_from_call(node, code_bytes: bytes):
    """
    Given a call_expression node, try to find the function identifier name.
    Returns the name (str) or None.
    """
    # Typical shapes:
    # call_expression
    #   function: identifier      -> simple
    #   function: field_expression (like obj.method) -> dive
    # So search within first child(s) for identifier
    for child in node.children:
        if child.type == 'identifier':
            return _node_text(child, code_bytes)
        # dive one level
        for ch2 in child.children:
            if ch2.type == 'identifier':
                return _node_text(ch2, code_bytes)
    # fallback: extract via snippet regex
    snippet = _node_text(node, code_bytes)
    m = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(', snippet)
    if m:
        return m.group(1)
    return None

def _extract_arg_text_from_call(node, code_bytes: bytes):
    """
    Return the text inside the parentheses of a call_expression (naive).
    """
    text = _node_text(node, code_bytes)
    m = re.search(r'\((.*)\)', text, flags=re.S)
    if m:
        return m.group(1).strip()
    return ""

# ----------------- Regex fallback sink detection -----------------
def regex_find_sinks(code: str):
    sinks = []
    for fn in SINK_FUNCS:
        for m in re.finditer(r'\b' + re.escape(fn) + r'\s*\(', code):
            # get surrounding line(s)
            start = code.rfind('\n', 0, m.start()) + 1
            end = code.find('\n', m.end())
            if end == -1:
                end = len(code)
            sinks.append((fn, code[start:end].strip(), m.start()))
    return sinks

# ----------------- Identifier extraction / backward context -----------------
IDENT_PATTERN = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\b')

def extract_identifiers_from_arg_text(arg_text: str):
    # remove string/char literals to avoid extracting tokens inside them
    no_lit = re.sub(r'".*?"', '""', arg_text, flags=re.S)
    no_lit = re.sub(r"'.*?'", "''", no_lit, flags=re.S)
    toks = IDENT_PATTERN.findall(no_lit)
    filtered = [t for t in toks if t not in SINK_FUNCS and not re.match(r'^(if|for|while|return|sizeof|int|char)$', t)]
    # dedupe preserving order
    seen = set(); out = []
    for t in filtered:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def bytepos_to_line_index(code: str, byte_pos: int):
    """
    Convert byte position (0-based) into line index in code.splitlines()
    (counts bytes in utf-8 encoding).
    """
    # compute number of bytes until each newline
    # faster: slice code bytes and count newlines
    code_bytes = code.encode('utf-8')
    if byte_pos <= 0:
        return 0
    # clamp
    if byte_pos >= len(code_bytes):
        return len(code.splitlines()) - 1
    # count newlines in prefix
    prefix = code_bytes[:byte_pos]
    # number of '\n' bytes
    ln = prefix.count(b'\n')
    return ln  # zero-based line index

def backward_context_search(code: str, sink_byte_pos: int, idents):
    """
    Search backward up to BACKWARD_LINES_WINDOW lines for lines that refer to idents or keywords.
    Return concatenated context (sorted by original order) plus the sink line appended.
    """
    lines = code.splitlines()
    sink_line_idx = bytepos_to_line_index(code, sink_byte_pos)
    start_scan = max(0, sink_line_idx - BACKWARD_LINES_WINDOW)
    context_indices = set()
    keywords = ['argv', 'fgets', 'scanf', 'read(', 'gets', 'malloc', 'calloc', 'realloc']

    # backward scan
    for i in range(sink_line_idx - 1, start_scan - 1, -1):
        ln = lines[i]
        hit = False
        for ident in idents:
            if re.search(r'\b' + re.escape(ident) + r'\b', ln):
                hit = True
                break
        if not hit:
            for kw in keywords:
                if kw in ln:
                    hit = True
                    break
        if hit:
            s = max(0, i - PRE_CONTEXT_LINES)
            for j in range(s, min(len(lines), i + 1)):
                context_indices.add(j)
        if len(context_indices) >= 20:
            break

    if not context_indices:
        s = max(0, sink_line_idx - FALLBACK_CONTEXT_LINES)
        for j in range(s, sink_line_idx):
            context_indices.add(j)

    sorted_idx = sorted(list(context_indices))
    ctx_lines = [lines[i] for i in sorted_idx] if sorted_idx else []
    sink_line = lines[sink_line_idx] if 0 <= sink_line_idx < len(lines) else ""
    full_ctx = "\n".join(ctx_lines).strip()
    if full_ctx:
        return full_ctx + "\n" + sink_line
    else:
        return sink_line

# ----------------- High-level extraction from a single file -----------------
def extract_from_file(path: Path, parser=None):
    """
    Return list of gadget dicts: {"file": str, "sink": str, "gadget_raw": str}
    """
    code = read_text_file(path)
    gadgets = []

    # 1) AST-based detection (if parser available)
    if parser is not None:
        try:
            code_bytes = code.encode('utf-8')
            for call_node, pos in _find_call_expressions(parser, code):
                func_name = _extract_function_name_from_call(call_node, code_bytes)
                if func_name and func_name in SINK_FUNCS:
                    sink_text = _node_text(call_node, code_bytes).strip()
                    arg_text = _extract_arg_text_from_call(call_node, code_bytes)
                    idents = extract_identifiers_from_arg_text(arg_text)
                    ctx = backward_context_search(code, pos, idents)
                    gadgets.append({
                        "file": str(path),
                        "sink": sink_text,
                        "gadget_raw": ctx
                    })
        except Exception:
            # if AST parsing fails, fall through to regex approach
            traceback.print_exc(file=sys.stderr)

    # 2) Regex fallback (also add those missed by parser)
    try:
        regex_sinks = regex_find_sinks(code)
        for fn, sink_line, pos in regex_sinks:
            # avoid duplicate if sink_line already found
            if any(g['sink'] == sink_line for g in gadgets):
                continue
            # extract idents from sink_line
            m = re.search(r'\((.*)\)', sink_line, flags=re.S)
            args = m.group(1) if m else ""
            idents = extract_identifiers_from_arg_text(args)
            ctx = backward_context_search(code, pos, idents)
            gadgets.append({
                "file": str(path),
                "sink": sink_line,
                "gadget_raw": ctx
            })
    except Exception:
        traceback.print_exc(file=sys.stderr)

    return gadgets

# ----------------- Directory extraction -----------------
def extract_from_dir(indir: str, parser=None):
    indir_path = Path(indir)
    if not indir_path.exists():
        raise FileNotFoundError(f"Input dir not found: {indir}")
    out = []
    file_count = 0
    for root, dirs, files in os.walk(indir):
        for f in files:
            if f.endswith(CODE_EXTS):
                file_count += 1
                p = Path(root) / f
                try:
                    gs = extract_from_file(p, parser)
                    if gs:
                        out.extend(gs)
                except Exception:
                    print(f"[warn] error processing {p}", file=sys.stderr)
    return out

# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser(description="static_extractor (tree-sitter based gadget extractor)")
    p.add_argument("code_dir", nargs='?', help="Directory containing C/C++ source files")
    p.add_argument("out_jsonl", nargs='?', help="Output JSONL path")
    p.add_argument("--use-prebuilt", action="store_true", help="Use tree_sitter_languages.get_parser('c') if available")
    p.add_argument("--ts-lib", help="Path to tree-sitter shared library (.so/.dylib/.dll)")
    p.add_argument("--build-grammar", nargs='+', help="If set, build a tree-sitter lib from given grammar directories and exit. Provide one or more grammar dirs.")
    p.add_argument("--out-lib", default="build/my-languages.so", help="Output path for built tree-sitter lib (used with --build-grammar)")
    p.add_argument("--quiet", action="store_true", help="Less verbose")
    return p.parse_args()

def main():
    args = parse_args()

    # If requested, build grammar and exit
    if args.build_grammar:
        try:
            lib_path = build_tree_sitter_lib(args.build_grammar, args.out_lib)
            print(f"[main] Built tree-sitter lib at: {lib_path}")
            return
        except Exception as e:
            print("[main] Failed to build tree-sitter lib:", e, file=sys.stderr)
            sys.exit(2)

    if not args.code_dir or not args.out_jsonl:
        print("Usage: python static_extractor.py <code_dir> <out_jsonl> [--ts-lib path | --use-prebuilt]", file=sys.stderr)
        sys.exit(1)

    # init parser
    parser = None
    if args.use_prebuilt:
        parser = init_parser(use_prebuilt=True)
        if parser is None:
            print("[main] Warning: prebuilt parser requested but not available. Falling back to regex.", file=sys.stderr)
    elif args.ts_lib:
        parser = init_parser(ts_lib=args.ts_lib)
        if parser is None:
            print("[main] Warning: ts_lib provided but parser init failed. Falling back to regex.", file=sys.stderr)
    else:
        # try environment var
        env_lib = os.environ.get('TREE_SITTER_LIB')
        if env_lib:
            parser = init_parser(ts_lib=env_lib)
            if parser is None:
                print("[main] Warning: TREE_SITTER_LIB set but parser init failed. Falling back to regex.", file=sys.stderr)

    if parser is None:
        if _HAS_TS_LANGS:
            # try prebuilt silently
            try:
                parser = init_parser(use_prebuilt=True)
                if parser is not None and not args.quiet:
                    print("[main] Using prebuilt parser from tree_sitter_languages.")
            except Exception:
                parser = None

    if parser is None and not args.quiet:
        print("[main] Note: tree-sitter parser not initialized. Using regex fallback.", file=sys.stderr)

    # run extraction
    print("[main] Starting extraction from:", args.code_dir)
    gadgets = extract_from_dir(args.code_dir, parser)
    write_jsonl(gadgets, args.out_jsonl)
    print(f"[main] Saved {len(gadgets)} gadgets -> {args.out_jsonl}")

if __name__ == "__main__":
    main()
