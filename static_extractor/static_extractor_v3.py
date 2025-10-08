#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
static_extractor.py
C/C++ 정적 가젯 추출기 (Tree-sitter 기반, intra-procedural backward slice 포함)

기능 요약:
- Tree-sitter가 있으면 AST로 싱크 함수 호출(call_expression) 탐지
- 싱크 인자에서 식별자(변수명) 추출 후, 같은 함수 내부에서 def-use(선언/할당/초기화) 라인들을
  거꾸로 수집하여 "프로그램 슬라이스"에 가까운 가젯 컨텍스트 생성
- Tree-sitter가 없으면 정규식 기반 fallback으로 동작
- 결과를 JSONL로 저장 (파일/라인/싱크명/슬라이스 범위/가젯 원문 등)
- 선택적으로 --normalize 로 모델 입력용 가젯 텍스트를 정규화
- 선택적으로 --cwe-rules 로 간단한 CWE 후보 매핑(싱크→CWE) 제공

사용법:
  python static_extractor.py /path/to/code_dir out_gadgets.jsonl \
    [--use-prebuilt | --ts-lib build/my-languages.so] \
    [--build-grammar path/to/tree-sitter-c ... --out-lib build/my-languages.so] \
    [--normalize] [--cwe-rules] [--ext .c .cpp .cc .h .hpp]

  python ./static_extractor_v2.py \
  ../../VulDeePecker/VulDeePecker/CWE-119/source_files \
  ../../VulDeePecker/CWE119_gadgets.jsonl \
  --use-prebuilt --normalize --cwe-rules

  python static_extractor_v2.py \
  ../../C/testcases \
  ./juliet_gadgets.jsonl \
  --use-prebuilt --normalize --cwe-rules

  python ./static_extractor/static_extractor_v3.py \
  ../C/testcases \
  ./static_extractor/v3/juliet_gadgets.jsonl \
  --use-prebuilt --normalize --cwe-rules


사전 준비:
- pip install tree_sitter tree_sitter_languages (prebuilt 사용 시)
- 혹은 grammar 직접 빌드:
    python static_extractor.py --build-grammar ./tree-sitter-c --out-lib build/my-languages.so
    python static_extractor.py src/ out.jsonl --ts-lib build/my-languages.so
"""

from __future__ import annotations
from pathlib import Path
import argparse
import os
import sys
import json
import re
import traceback
from typing import List, Tuple, Dict, Optional, Iterable, Set

# ----------------- Optional: Tree-sitter imports -----------------
_HAS_TS = False
_HAS_TS_LANGS = False
try:
    from tree_sitter import Language, Parser
    _HAS_TS = True
except Exception:
    _HAS_TS = False

try:
    from tree_sitter_languages import get_parser
    _HAS_TS_LANGS = True
except Exception:
    _HAS_TS_LANGS = False

# ----------------- Config -----------------
# 확장 가능한 싱크 함수 목록 (C/C++)
# (정확 토큰 일치; 아래 _SINK_FAMILY_PATTERNS 로 변종을 함께 커버)
SINK_FUNCS: Set[str] = {
    # 문자열/버퍼 복사·결합
    "strcpy","strncpy","strcat","strncat","stpcpy","strlcpy","strlcat",
    "wcscpy","wcsncpy","wcscat","wcsncat","lstrcpy","lstrcpyn","lstrcat","lstrncat",
    "memcpy","memmove","bcopy",

    # 포맷 스트링(가변인자 포함)
    "printf","fprintf","sprintf","snprintf","asprintf",
    "vprintf","vfprintf","vsprintf","vsnprintf","vasprintf",
    "wprintf","fwprintf","swprintf","vwprintf","vswprintf",
    "_snprintf","_snwprintf","wsprintfA","wsprintfW",

    # 입력(경계 미검증)
    "gets","fgets","scanf","sscanf","fscanf","vscanf","vfscanf","vsscanf",
    "scanf_s","sscanf_s","fscanf_s",
    "getenv",

    # 파일/경로 (경로 조작·TOCTOU)
    "open","openat","creat","fopen","freopen","_wfopen",
    "CreateFileA","CreateFileW","DeleteFileA","DeleteFileW","MoveFileA","MoveFileW",
    "remove","unlink","rename","chmod","chown","fchmod","fchown","mkdir","rmdir",
    "realpath","PathCanonicalizeA","PathCanonicalizeW","PathCombineA","PathCombineW",
    "tmpnam","tempnam","mktemp","_mktemp","mkstemp","access","stat","lstat","fstat",

    # 네트워크/소켓
    "recv","recvfrom","recvmsg","read","readv",
    "send","sendto","sendmsg","write","writev",

    # 커맨드 실행/프로세스 생성
    "system","popen","_popen","_wsystem",
    "execl","execlp","execle","execv","execvp","execve",
    "CreateProcessA","CreateProcessW","ShellExecuteA","ShellExecuteW","WinExec",

    # 동적 로딩/심볼
    "dlopen","dlsym","LoadLibraryA","LoadLibraryW","GetProcAddress",

    # 레지스트리/설정
    "RegOpenKeyExA","RegOpenKeyExW","RegQueryValueExA","RegQueryValueExW",
    "RegSetValueExA","RegSetValueExW","RegCreateKeyExA","RegCreateKeyExW",

    # 위험/레거시
    "gets_s","strtok","strtok_r",

    # 예시에서 중요
    "SetComputerNameA","SetComputerNameW",
}

# 패밀리 정규식 (대소문자/A/W/가변인자/변종 흡수)
_SINK_FAMILY_PATTERNS = [
    r"^v?s?n?printf$",          # printf/sprintf/snprintf + v-접두
    r"^v?f?scanf$",             # scanf/fscanf/vscanf 등
    r"^str(n)?(cpy|cat)$",      # strcpy/strncpy/strcat/strncat
    r"^wcs(n)?(cpy|cat)$",      # wide char
    r"^lstr(cpy|cpyn|cat|ncat)$",
    r"^mem(cpy|move)$",
    r"^recv(from|msg)?$",       # recv/recvfrom/recvmsg
    r"^send(to|msg)?$",         # send/sendto/sendmsg
    r"^read(v)?$", r"^write(v)?$",
    r"^exec([lvpe]{1,2})$",     # execl/execlp/execle/execv/execvp/execve
    r"^CreateProcess(A|W)$", r"^ShellExecute(A|W)$",
    r"^CreateFile(A|W)$", r"^DeleteFile(A|W)$", r"^MoveFile(A|W)$",
    r"^Path(Canonicalize|Combine)(A|W)$",
    r"^Reg(OpenKeyEx|QueryValueEx|SetValueEx|CreateKeyEx)(A|W)$",
    r"^LoadLibrary(A|W)$", r"^GetProcAddress$",
    r"^_?s?n?w?printf$", r"^wsprintf(A|W)$",
    r"^(tmpnam|tempnam|mktemp|_mktemp|mkstemp)$",
    r"^SetComputerName(A|W)$",
]
_SINK_FAMILY_REGEX = [re.compile(p, re.IGNORECASE) for p in _SINK_FAMILY_PATTERNS]

# 싱크→CWE 후보(보강 매핑; 힌트 수준)
SINK_TO_CWE = {
    # BOF/메모리
    "strcpy": ["CWE-120","CWE-119"], "strncpy": ["CWE-120"], "strcat": ["CWE-120","CWE-119"],
    "strncat": ["CWE-120"], "memcpy": ["CWE-119","CWE-120"], "memmove": ["CWE-119","CWE-120"],
    "gets": ["CWE-242","CWE-676","CWE-120"], "fgets": ["CWE-120"],

    # 포맷 스트링
    "printf": ["CWE-134"], "sprintf": ["CWE-120","CWE-134"], "snprintf": ["CWE-134"],
    "vsprintf": ["CWE-120","CWE-134"], "vsnprintf": ["CWE-134"],

    # 입력 스캐너
    "scanf": ["CWE-120"], "sscanf": ["CWE-120"], "fscanf": ["CWE-120"],

    # 네트워크 입력
    "recv": ["CWE-120"], "read": ["CWE-120"],

    # 커맨드 실행
    "system": ["CWE-78"], "popen": ["CWE-78"], "execve": ["CWE-78"], "CreateProcessA": ["CWE-78"], "CreateProcessW": ["CWE-78"],

    # 경로/파일
    "open": ["CWE-23","CWE-36"], "fopen": ["CWE-23","CWE-36"], "realpath": ["CWE-23","CWE-22"],
    "CreateFileA": ["CWE-23","CWE-36"], "CreateFileW": ["CWE-23","CWE-36"],

    # 레지스트리/설정
    "RegSetValueExA": ["CWE-269","CWE-250"], "RegSetValueExW": ["CWE-269","CWE-250"],

    # 예시
    "SetComputerNameA": ["CWE-269","CWE-250"], "SetComputerNameW": ["CWE-269","CWE-250"],
}

# backward 윈도우 (라인 수)
BACKWARD_LINES_WINDOW = 160
# 정규식 fallback 시 최소 컨텍스트 확보 라인 수
FALLBACK_CONTEXT_LINES = 16
# intra-proc def 수집 시, sink 이전에서 최대 몇 개의 정의를 포함할지
MAX_DEFS_PER_IDENT = 6

DEFAULT_EXTS = ('.c', '.cpp', '.cc', '.h', '.hpp')

# ----------------- Utils -----------------
IDENT_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\b')
WS_MANY = re.compile(r'\s+')
COMMENT_C_BLOCK = re.compile(r'/\*.*?\*/', re.S)
COMMENT_CPP_LINE = re.compile(r'//.*?$' , re.M)

KEYWORDS_SKIP = {
    'if','for','while','return','sizeof','int','char','float','double','void','short','long',
    'unsigned','signed','static','const','volatile','struct','union','enum','typedef','register','goto'
}

INPUT_HINTS = {'argv', 'fgets', 'gets', 'scanf', 'read', 'recv', 'getenv'}

def is_sink_name(name: str) -> bool:
    """정확 토큰 또는 패밀리 정규식에 해당하면 싱크로 간주."""
    if name in SINK_FUNCS:
        return True
    for rx in _SINK_FAMILY_REGEX:
        if rx.match(name):
            return True
    return False

def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

def write_jsonl(records: List[dict], outpath: str) -> None:
    outp = Path(outpath)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w', encoding='utf-8') as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + '\n')

def normalize_code_for_model(code: str) -> str:
    # 주석 제거 + 공백 축소 + 빈 줄 제거
    code = COMMENT_C_BLOCK.sub(' ', code)
    code = COMMENT_CPP_LINE.sub(' ', code)
    code = '\n'.join([ln.rstrip() for ln in code.splitlines() if ln.strip()])
    code = WS_MANY.sub(' ', code).strip()
    return code

# ----------------- Tree-sitter helpers -----------------
def build_tree_sitter_lib(grammar_dirs: List[str], out_lib_path: str) -> str:
    if not _HAS_TS:
        raise RuntimeError("tree_sitter not installed")
    gds = [str(Path(g).resolve()) for g in grammar_dirs]
    print(f"[build] build -> {out_lib_path} from: {gds}")
    Language.build_library(out_lib_path, gds)
    print(f"[build] done: {out_lib_path}")
    return out_lib_path

def init_parser(ts_lib: Optional[str] = None, use_prebuilt: bool = False):
    if use_prebuilt and _HAS_TS_LANGS:
        try:
            print("[init] using prebuilt parser (tree_sitter_languages: c)")
            return get_parser('c')
        except Exception as e:
            print("[init] prebuilt get_parser('c') failed:", e, file=sys.stderr)

    if ts_lib:
        libp = Path(ts_lib)
        if libp.exists() and _HAS_TS:
            try:
                C_LANG = Language(str(libp), 'c')
                p = Parser(); p.set_language(C_LANG)
                print(f"[init] loaded ts lib: {libp}")
                return p
            except Exception as e:
                print("[init] ts-lib load failed:", e, file=sys.stderr)
                traceback.print_exc()

    if _HAS_TS_LANGS:
        try:
            return get_parser('c')
        except Exception:
            pass
    return None

def _walk_nodes(root):
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        for ch in reversed(n.children):
            stack.append(ch)

def _node_text(node, code_bytes: bytes) -> str:
    return code_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')

def _find_call_expressions(parser, code: str):
    try:
        code_bytes = code.encode('utf-8')
        tree = parser.parse(code_bytes)
        root = tree.root_node
        for node in _walk_nodes(root):
            if node.type == 'call_expression':
                yield node, node.start_byte
    except Exception:
        return

def _extract_function_name_from_call(node, code_bytes: bytes) -> Optional[str]:
    for ch in node.children:
        if ch.type == 'identifier':
            return _node_text(ch, code_bytes)
        for ch2 in ch.children:
            if ch2.type == 'identifier':
                return _node_text(ch2, code_bytes)
    snippet = _node_text(node, code_bytes)
    m = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(', snippet)
    return m.group(1) if m else None

def _extract_arg_text_from_call(node, code_bytes: bytes) -> str:
    txt = _node_text(node, code_bytes)
    m = re.search(r'\((.*)\)', txt, flags=re.S)
    return m.group(1).strip() if m else ''

def _find_enclosing_function(node):
    cur = node
    while cur is not None:
        if cur.type in ('function_definition',):
            return cur
        cur = cur.parent
    return None

# ----------------- Regex fallback sink detection -----------------
def regex_find_sinks(code: str) -> List[Tuple[str, str, int]]:
    sinks: List[Tuple[str, str, int]] = []

    # 1) 정확 토큰 집합 기반
    for fn in SINK_FUNCS:
        for m in re.finditer(r'\b' + re.escape(fn) + r'\s*\(', code):
            start = code.rfind('\n', 0, m.start()) + 1
            end = code.find('\n', m.end())
            if end == -1:
                end = len(code)
            sinks.append((fn, code[start:end].strip(), m.start()))

    # 2) 패밀리 패턴 기반 (이름 캡처)
    for m in re.finditer(r'\b([A-Za-z_]\w*)\s*\(', code):
        name = m.group(1)
        for rx in _SINK_FAMILY_REGEX:
            if rx.match(name):
                start = code.rfind('\n', 0, m.start()) + 1
                end = code.find('\n', m.end())
                if end == -1:
                    end = len(code)
                sinks.append((name, code[start:end].strip(), m.start()))
                break

    return sinks

# ----------------- Identifier & context -----------------
def extract_identifiers_from_arg_text(arg_text: str) -> List[str]:
    """문자열/문자 리터럴 제거 후 토큰화. 싱크 이름 및 키워드는 식별자에서 제외."""
    no_lit = re.sub(r'".*?"', '""', arg_text, flags=re.S)
    no_lit = re.sub(r"'.*?'", "''", no_lit, flags=re.S)
    toks = IDENT_RE.findall(no_lit)
    out: List[str] = []
    seen: Set[str] = set()
    for t in toks:
        if t in KEYWORDS_SKIP:
            continue
        # 싱크 토큰/패밀리 패턴은 변수로 취급하지 않음
        if t in SINK_FUNCS or any(rx.match(t) for rx in _SINK_FAMILY_REGEX):
            continue
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def bytepos_to_line_index(code: str, byte_pos: int) -> int:
    cb = code.encode('utf-8')
    if byte_pos <= 0:
        return 0
    if byte_pos >= len(cb):
        return len(code.splitlines()) - 1
    ln = cb[:byte_pos].count(b'\n')
    return ln

def backward_context_search(code: str, sink_byte_pos: int, idents: List[str]) -> Tuple[str, int, int]:
    lines = code.splitlines()
    sink_li = bytepos_to_line_index(code, sink_byte_pos)
    start_scan = max(0, sink_li - BACKWARD_LINES_WINDOW)
    context_idx = set()

    for i in range(sink_li - 1, start_scan - 1, -1):
        ln = lines[i]
        hit = any(re.search(r'\b' + re.escape(v) + r'\b', ln) for v in idents)
        if not hit:
            hit = any(kw in ln for kw in INPUT_HINTS)
        if hit:
            s = max(0, i - 3)
            for j in range(s, i + 1):
                context_idx.add(j)
        if len(context_idx) >= 40:
            break

    if not context_idx:
        s = max(0, sink_li - FALLBACK_CONTEXT_LINES)
        for j in range(s, sink_li):
            context_idx.add(j)

    sorted_idx = sorted(list(context_idx))
    ctx_lines = [lines[i] for i in sorted_idx] if sorted_idx else []
    sink_line = lines[sink_li] if 0 <= sink_li < len(lines) else ''
    full_ctx = ("\n".join(ctx_lines).strip() + ("\n" if ctx_lines else "") + sink_line).strip()
    begin = sorted_idx[0] if sorted_idx else sink_li
    end = sink_li
    return full_ctx, begin + 1, end + 1  # 1-based

# ----------------- Intra-procedural DEF collection -----------------
def collect_defs_in_function(func_node, code: str, code_bytes: bytes, idents: List[str], sink_li: int) -> List[Tuple[int,str]]:
    """함수 내부에서 해당 식별자와 관련된 선언/할당/초기화 노드를 sink 이전 라인에서 수집."""
    found: List[Tuple[int,str]] = []
    max_per_ident: Dict[str,int] = {i:0 for i in idents}

    def add_if_hit(n) -> None:
        txt = _node_text(n, code_bytes)
        li = bytepos_to_line_index(code, n.start_byte)
        if li >= sink_li:
            return
        for ident in idents:
            if max_per_ident.get(ident,0) >= MAX_DEFS_PER_IDENT:
                continue
            if re.search(r'\b' + re.escape(ident) + r'\b', txt):
                found.append((li, txt.strip()))
                max_per_ident[ident] = max_per_ident.get(ident,0) + 1
                break

    C_DEF_TYPES = {'declaration', 'init_declarator', 'assignment_expression', 'declaration_list', 'parameter_declaration'}
    for n in _walk_nodes(func_node):
        if n.type in C_DEF_TYPES:
            add_if_hit(n)
    found.sort(key=lambda x: x[0])
    dedup = []
    seen_li = set()
    for li, txt in found:
        if li not in seen_li:
            seen_li.add(li); dedup.append((li, txt))
    return dedup

# ----------------- Single-file extraction -----------------
def extract_from_file(path: Path, parser=None, normalize: bool=False, cwe_rules: bool=False) -> List[dict]:
    code = read_text_file(path)
    gadgets: List[dict] = []

    # 1) AST 기반 탐지
    if parser is not None:
        try:
            code_bytes = code.encode('utf-8')
            for call_node, pos in _find_call_expressions(parser, code):
                fn = _extract_function_name_from_call(call_node, code_bytes)
                if not fn or not is_sink_name(fn):
                    continue

                sink_text = _node_text(call_node, code_bytes).strip()
                arg_text = _extract_arg_text_from_call(call_node, code_bytes)
                idents = extract_identifiers_from_arg_text(arg_text)

                sink_li = bytepos_to_line_index(code, pos)
                func_node = _find_enclosing_function(call_node)

                defs_texts: List[str] = []
                defs_lines: List[int] = []
                if func_node is not None and idents:
                    defs = collect_defs_in_function(func_node, code, code_bytes, idents, sink_li)
                    if defs:
                        defs_lines = [li for li,_ in defs]
                        defs_texts = [txt for _,txt in defs]

                ctx_raw, ctx_begin, ctx_end = backward_context_search(code, pos, idents)

                merged_lines: List[Tuple[int,str]] = []
                seen = set()
                for li, txt in zip(defs_lines, defs_texts):
                    key = (li, txt)
                    if key not in seen:
                        seen.add(key); merged_lines.append((li, txt))

                ctx_lines = ctx_raw.splitlines()
                ctx_start_li = ctx_begin - 1
                for off, ln in enumerate(ctx_lines):
                    li = ctx_start_li + off
                    key = (li, ln)
                    if key not in seen:
                        seen.add(key); merged_lines.append((li, ln))

                merged_lines.sort(key=lambda x: x[0])
                gadget_raw = "\n".join(t for _,t in merged_lines).strip()
                out_txt = normalize_code_for_model(gadget_raw) if normalize else gadget_raw
                cwe = SINK_TO_CWE.get(fn, []) if cwe_rules else []

                gadgets.append({
                    'file': str(path),
                    'lang': 'c',
                    'sink_name': fn,
                    'cwe_candidates': cwe,
                    'sink_line': sink_li + 1,
                    'slice_begin': (merged_lines[0][0] + 1) if merged_lines else (ctx_begin),
                    'slice_end': (merged_lines[-1][0] + 1) if merged_lines else (ctx_end),
                    'gadget_raw': out_txt,
                })
        except Exception:
            traceback.print_exc(file=sys.stderr)

    # 2) 정규식 fallback (parser 미사용/누락된 경우 보완)
    try:
        regex_sinks = regex_find_sinks(code)
        for fn, sink_line_text, pos in regex_sinks:
            # 중복 제거(동일 sink명+라인)
            if any(g.get('sink_name') == fn and g.get('sink_line') == bytepos_to_line_index(code, pos) + 1 for g in gadgets):
                continue

            m = re.search(r'\((.*)\)', sink_line_text, flags=re.S)
            args = m.group(1) if m else ''
            idents = extract_identifiers_from_arg_text(args)

            ctx_raw, ctx_begin, ctx_end = backward_context_search(code, pos, idents)
            out_txt = normalize_code_for_model(ctx_raw) if normalize else ctx_raw
            cwe = SINK_TO_CWE.get(fn, []) if cwe_rules else []

            gadgets.append({
                'file': str(path),
                'lang': 'c',
                'sink_name': fn,
                'cwe_candidates': cwe,
                'sink_line': bytepos_to_line_index(code, pos) + 1,
                'slice_begin': ctx_begin,
                'slice_end': ctx_end,
                'gadget_raw': out_txt,
            })
    except Exception:
        traceback.print_exc(file=sys.stderr)

    return gadgets

# ----------------- Directory extraction -----------------
def iter_code_files(indir: str, exts: Tuple[str,...]) -> List[Path]:
    out: List[Path] = []
    for root, _, files in os.walk(indir):
        for f in files:
            if f.endswith(exts):
                out.append(Path(root) / f)
    return out

def extract_from_dir(indir: str, parser=None, normalize: bool=False, cwe_rules: bool=False, exts: Tuple[str,...]=DEFAULT_EXTS) -> List[dict]:
    pdir = Path(indir)
    if not pdir.exists():
        raise FileNotFoundError(f"Input dir not found: {indir}")
    files = iter_code_files(indir, exts)
    out: List[dict] = []
    for p in files:
        try:
            gs = extract_from_file(p, parser=parser, normalize=normalize, cwe_rules=cwe_rules)
            if gs:
                out.extend(gs)
        except Exception:
            print(f"[warn] error processing {p}", file=sys.stderr)
    return out

# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser(description='C/C++ static gadget extractor (tree-sitter + intra-proc slice)')
    ap.add_argument('code_dir', nargs='?', help='Directory containing C/C++ sources')
    ap.add_argument('out_jsonl', nargs='?', help='Output JSONL path')

    ap.add_argument('--use-prebuilt', action='store_true', help="Use tree_sitter_languages prebuilt parser for 'c'")
    ap.add_argument('--ts-lib', help='Path to tree-sitter shared library (.so/.dylib/.dll) built with C grammar')

    ap.add_argument('--build-grammar', nargs='+', help='Build a tree-sitter lib from given grammar directories and exit')
    ap.add_argument('--out-lib', default='build/my-languages.so', help='Output path for built tree-sitter lib (used with --build-grammar)')

    ap.add_argument('--normalize', action='store_true', help='Normalize gadget text for model input')
    ap.add_argument('--cwe-rules', action='store_true', help='Attach simple CWE candidate list from sink name')

    ap.add_argument('--ext', nargs='*', default=list(DEFAULT_EXTS), help='File extensions to include (e.g., --ext .c .h .cpp)')
    ap.add_argument('--quiet', action='store_true')
    return ap.parse_args()

def main():
    args = parse_args()

    # 1) Grammar build mode
    if args.build_grammar:
        try:
            path = build_tree_sitter_lib(args.build_grammar, args.out_lib)
            print(f"[main] built tree-sitter lib: {path}")
            return
        except Exception as e:
            print('[main] build failed:', e, file=sys.stderr)
            sys.exit(2)

    if not args.code_dir or not args.out_jsonl:
        print('Usage: python static_extractor.py <code_dir> <out_jsonl> [--use-prebuilt | --ts-lib PATH] [--normalize] [--cwe-rules]', file=sys.stderr)
        sys.exit(1)

    # 2) Parser init
    parser = None
    if args.use_prebuilt:
        parser = init_parser(use_prebuilt=True)
        if parser is None and not args.quiet:
            print('[main] prebuilt parser not available. Fallback to regex.', file=sys.stderr)
    elif args.ts_lib:
        parser = init_parser(ts_lib=args.ts_lib)
        if parser is None and not args.quiet:
            print('[main] ts-lib provided but init failed. Fallback to regex.', file=sys.stderr)
    else:
        env_lib = os.environ.get('TREE_SITTER_LIB')
        if env_lib:
            parser = init_parser(ts_lib=env_lib)
            if parser is None and not args.quiet:
                print('[main] TREE_SITTER_LIB init failed. Fallback to regex.', file=sys.stderr)
        if parser is None:
            parser = init_parser(use_prebuilt=True)

    if parser is None and not args.quiet:
        print('[main] note: tree-sitter parser not initialized; regex fallback only', file=sys.stderr)

    # 3) Extract
    print('[main] extracting from:', args.code_dir)
    gadgets = extract_from_dir(args.code_dir, parser=parser, normalize=args.normalize, cwe_rules=args.cwe_rules, exts=tuple(args.ext))

    # 4) Save
    write_jsonl(gadgets, args.out_jsonl)
    print(f"[main] saved {len(gadgets)} gadgets -> {args.out_jsonl}")

if __name__ == '__main__':
    main()
