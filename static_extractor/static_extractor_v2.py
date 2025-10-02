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
from typing import List, Tuple, Dict, Optional

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
SINK_FUNCS = {
    'strcpy', 'strcat', 'sprintf', 'vsprintf', 'gets', 'memcpy', 'strncpy', 'strncat',
    'system', 'popen', 'execl', 'execv', 'execve', 'execvp', 'scanf', 'sscanf', 'fscanf',
    'read', 'recv', 'gets_s', 'sprintf_s', 'strcpy_s', 'strcat_s', 'memmove', 'memmove_s'
}
# 싱크→CWE 후보(간단 매핑)
SINK_TO_CWE = {
    'strcpy': ['CWE-120', 'CWE-119'],
    'strcat': ['CWE-120', 'CWE-119'],
    'sprintf': ['CWE-120', 'CWE-134'],
    'vsprintf': ['CWE-120', 'CWE-134'],
    'gets': ['CWE-242', 'CWE-676', 'CWE-120'],
    'memcpy': ['CWE-120', 'CWE-119'],
    'strncpy': ['CWE-120'],
    'system': ['CWE-78'],
    'popen': ['CWE-78'],
    'scanf': ['CWE-120'],
    'sscanf': ['CWE-120'],
    'fscanf': ['CWE-120'],
    'read': ['CWE-120'],
    'recv': ['CWE-120'],
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
# 이 함수는 Tree-sitter grammar 디렉토리에서 공유 라이브러리(.so/.dylib/.dll)를 빌드하는 역할이다
# grammar_dirs은 Tree-sitter grammar 소스 디렉토리들의 리스트이다 ("./tree-sitter-c")
# out_lib_path는 빌드된 공유 라이브러리를 저장할 경로이다 ("build/my-languages.so")
def build_tree_sitter_lib(grammar_dirs: List[str], out_lib_path: str) -> str:
    # _HAS_TS는 전역 플래그로 tree_sitter 패키지가 설치되어 있는지 확인한다
    if not _HAS_TS:
        # 설치 안 되어 있으면 에러
        raise RuntimeError("tree_sitter not installed")

    # grammar_dirs에 들어온 상대 결로들을 Path().resolve()로 절대 경로로 변환한다
    # gds는 최종적으로 문자열 경로 리스트이다
    gds = [str(Path(g).resolve()) for g in grammar_dirs]
    print(f"[build] build -> {out_lib_path} from: {gds}")

    # Tree-sitter의 Language.build_library 함수를 호출
    # out_lib_path 경로와 gds grammar 소스 디렉토리 목록
    # 이 결과로 grammar에 있는 grammar.json + C 소스들이 컴파일되어 .so/.dylib/.dll 파일이 생성된다
    Language.build_library(out_lib_path, gds)
    print(f"[build] done: {out_lib_path}")

    # 빌드된 .so 파일의 경로를 문자열로 반환한다
    return out_lib_path

"""
# 파서의 일반적인 역할은 strcpy(buf, user); 이렇게 있으면 밑에 처럼 나누는 것
# call_expression
#  ├── function: identifier ("strcpy")
#  └── arguments
#       ├── identifier ("buf")
#       └── identifier ("user")

1. Tree-sitter 파서의 역할은 AST를 순회하면서 call_expression을 찾는다
2. 호출된 함수 이름이 위험 함수 목록에 있으면 sink 표시
3. 그 함수 호출 노드 안에서 인자 노드를 찾아 어떤 변수가 전달됐는지 추출한다
4. 인자 변수명을 기준으로, 해당 변수가 어디서 선언/할당됐는지 같은 내부 함수를 다시 뒤져서 관련 라인을 수집한

여기서 AST(Abstract Syntax Tree, 추상 구문 트리)란?
원시 코드를 트리 구조로 바꾼 표현이다
문자열로 된 코드를 의미 단위로 쪼갠 구조라고 생각하면 된다
위의 call_expression 구조가 AST 구조라고 보면 된다

"""

# init_parser 함수는 Tree-sitter 파서를 초기화해서 돌려주는 역할을 한다
# 어떤 방식으로든 C 언어 파서를 준비해서 parser 객체를 반환하는 함수이다
# ts_lib는 직접 빌드한 Tree-sitter 공유 라이브러리 경로이다
# use_prebuilt가 True이면 tree_sitter_languages 패키지의 미리 빌드된 C 파서를 쓰려고 시도한다
def init_parser(ts_lib: Optional[str] = None, use_prebuilt: bool = False):
    # 만약 --use-prebuilt 옵션이 켜져 있고, tree_sitter_languages 패키지가 설치되어 있으면 get_parser('c')를 호출하여 C 언어용 prebuilt parser 객체를 반환한다
    if use_prebuilt and _HAS_TS_LANGS:
        try:
            print("[init] using prebuilt parser (tree_sitter_languages: c)")
            return get_parser('c')
        except Exception as e:
            print("[init] prebuilt get_parser('c') failed:", e, file=sys.stderr)

    # --ts-lib로 경로가 들어온 경우
    if ts_lib:
        libp = Path(ts_lib)

        # 해당 libp 경로가 있는 지 확인
        if libp.exists() and _HAS_TS:
            try:
                # C 언어로 grammar를 로드한다
                C_LANG = Language(str(libp), 'c')
                # 파서 객체를 만들고 C 파서로 설정한다
                p = Parser(); p.set_language(C_LANG)
                print(f"[init] loaded ts lib: {libp}")
                return p
            except Exception as e:
                print("[init] ts-lib load failed:", e, file=sys.stderr)
                traceback.print_exc()

    # fallback: try prebuilt silently
    # 위의 단계들을 모두 실패한 경우 한 번 더 파서로 시도해보고 실패하면 pass
    if _HAS_TS_LANGS:
        try:
            return get_parser('c')
        except Exception:
            pass

    # 실패하면 fallback 모드
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


# AST에서 함수 호출(call expressions)을 찾고, 그 안에서 함수 이름을 추출하는 함수이다
def _find_call_expressions(parser, code: str):
    try:
        # code는 소스 코드 문자열로 code_bytes에 저장
        code_bytes = code.encode('utf-8')
        # parser.parse() 호출하여 AST로 만든다
        tree = parser.parse(code_bytes)
        # AST 트리의 root node
        root = tree.root_node

        # 트리의 모든 노드를 DFS 방식으로 순회한다
        for node in _walk_nodes(root):
            # 함수 호출 구문인 경우에만 선택한다
            if node.type == 'call_expression':
                # generator로 (노드, 노드 시작 바이트 위치)를 하나씩 반환한다
                yield node, node.start_byte

    except Exception:
        return


# 함수 호출 노드를 찾아서 하나씩 내보내는 Generator 함수이다
# 노드는 call_expression 노드, code_bytes는 원본 코드의 바이트
def _extract_function_name_from_call(node, code_bytes: bytes) -> Optional[str]:
    # identifier or field_expression 내부 identifier 추출
    # call_expression의 자식들을 보고 함수 이름이 identifier로 나타나는 경우
    for ch in node.children:
        if ch.type == 'identifier':
            return _node_text(ch, code_bytes)

        # 함수 이름이 field_expression 과 같이 더 복잡한 구조 안에 들어 있는 경우
        for ch2 in ch.children:
            if ch2.type == 'identifier':
                return _node_text(ch2, code_bytes)

    # fallback regex
    # 만약 AST 탐색에서 함수 이름을 찾지 못한 경우
    # 노드의 전체 텍스트 snippet을 꺼내서 정규식으로 함수 이름을 추출한다
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
    sinks = []
    for fn in SINK_FUNCS:
        for m in re.finditer(r'\b' + re.escape(fn) + r'\s*\(', code):
            start = code.rfind('\n', 0, m.start()) + 1
            end = code.find('\n', m.end())
            if end == -1:
                end = len(code)
            sinks.append((fn, code[start:end].strip(), m.start()))
    return sinks

# ----------------- Identifier & context -----------------

def extract_identifiers_from_arg_text(arg_text: str) -> List[str]:
    no_lit = re.sub(r'".*?"', '""', arg_text, flags=re.S)
    no_lit = re.sub(r"'.*?'", "''", no_lit, flags=re.S)
    toks = IDENT_RE.findall(no_lit)
    out = []
    seen = set()
    for t in toks:
        if t in SINK_FUNCS or t in KEYWORDS_SKIP:
            continue
        if t not in seen:
            seen.add(t); out.append(t)
    return out


# 소스코드에서 바이트 위치를 라인 번호로 변환하는 유틸리티이다
# code는 전체 소스 코드 문자열이고 byte_pos는 소스 코드 안에서 특정 토큰이 시작하는 바이트 위치이다
def bytepos_to_line_index(code: str, byte_pos: int) -> int:
    # 소스 코드를 UTF-8로 인코딩해서 바이트열로 변환한다
    cb = code.encode('utf-8')

    # byte_pos가 0 이하라면 그냥 첫 번째 줄을 반환한다
    if byte_pos <= 0:
        return 0

    # 바이트 위치가 코드 길이를 넘어가면 마지막 줄을 반환한다
    if byte_pos >= len(cb):
        return len(code.splitlines()) - 1

    # 코드에서 byte_pos 앞쪽까지 잘라서 개행 개수를 센다 (줄 번호 = 개행 수)
    ln = cb[:byte_pos].count(b'\n')

    return ln


# sink 함수 호출이 있는 위치를 기준으로 과거 코드를 거슬러 올라가면서 관련 문맥을 수집하는 함수이다
# code는 전체 소스 코드, sink_byte_pos는 sink 함수 호출이 시작된 바이트 위치, idents는 sink 함수 인자에서 뽑은 변수 이름 리스트이다
def backward_context_search(code: str, sink_byte_pos: int, idents: List[str]) -> Tuple[str, int, int]:
    # 코드 전체를 줄 단위 리스트로 분할한다
    lines = code.splitlines()
    # sink 함수가 위치한 줄 번호이다
    sink_li = bytepos_to_line_index(code, sink_byte_pos)
    start_scan = max(0, sink_li - BACKWARD_LINES_WINDOW)
    context_idx = set()

    # 뒤로 거슬러 올라가며 관련 라인 수집
    # sink 위로 거슬러 올라가며 문맥을 찾는다
    for i in range(sink_li - 1, start_scan - 1, -1):
        ln = lines[i]
        hit = any(re.search(r'\b' + re.escape(v) + r'\b', ln) for v in idents)

        # 만약 인자 변수와 매칭이 안 되면, 입력 관련 힌트가 포함돼 있는지 검사한다
        if not hit:
            hit = any(kw in ln for kw in INPUT_HINTS)

        # 관련 있는 줄을 발견하면 그 줄만 추가하는 게 아니라 그 위쪽 3줄까지(i-3) 같이 포함해서 맥락을 보존한다
        if hit:
            s = max(0, i - 3)  # 약간의 pre-context 포함
            for j in range(s, i + 1):
                context_idx.add(j)

        # 너무 많은 라인을 담지 않도록 최대 40줄까지만 수집한다.
        if len(context_idx) >= 40:
            break

    # 아무것도 못 찾으면 fallback
    if not context_idx:
        s = max(0, sink_li - FALLBACK_CONTEXT_LINES)
        for j in range(s, sink_li):
            context_idx.add(j)

    # 모은 라인 번호를 정렬한다
    sorted_idx = sorted(list(context_idx))
    # 실제 코드 줄 텍스트 가져오기
    ctx_lines = [lines[i] for i in sorted_idx] if sorted_idx else []
    # sink가 있는 라인도 별도로 저장한다
    sink_line = lines[sink_li] if 0 <= sink_li < len(lines) else ''
    # 최종 문맥 = 관련 라인들 + 마지막에 sink 라인 붙인다
    # 중간에 빈 줄을 제거한다
    full_ctx = ("\n".join(ctx_lines).strip() + ("\n" if ctx_lines else "") + sink_line).strip()
    # 문맥 시작 줄과 sink 줄을 반환한다
    begin = sorted_idx[0] if sorted_idx else sink_li
    end = sink_li

    # 여기에서 +1을 한 이유가 일반적으로 라인 번호를 1-based로 맞추기 위해서이다
    return full_ctx, begin + 1, end + 1  # 1-based line numbers

# ----------------- Intra-procedural DEF collection -----------------

def collect_defs_in_function(func_node, code: str, code_bytes: bytes, idents: List[str], sink_li: int) -> List[Tuple[int,str]]:
    """함수 내부에서 해당 식별자와 관련된 선언/할당/초기화 노드를 sink 이전 라인에서 수집."""
    found: List[Tuple[int,str]] = []
    max_per_ident: Dict[str,int] = {i:0 for i in idents}

    def add_if_hit(n) -> None:
        txt = _node_text(n, code_bytes)
        li = bytepos_to_line_index(code, n.start_byte)
        if li >= sink_li:  # sink 이후는 제외
            return
        for ident in idents:
            if max_per_ident.get(ident,0) >= MAX_DEFS_PER_IDENT:
                continue
            if re.search(r'\b' + re.escape(ident) + r'\b', txt):
                found.append((li, txt.strip()))
                max_per_ident[ident] = max_per_ident.get(ident,0) + 1
                break

    # 선언/초기화/대입에 해당할 법한 노드 유형들을 넓게 커버 (C grammar 기반 보수적 처리)
    C_DEF_TYPES = {
        'declaration', 'init_declarator', 'assignment_expression', 'declaration_list',
        'parameter_declaration'
    }
    for n in _walk_nodes(func_node):
        if n.type in C_DEF_TYPES:
            add_if_hit(n)
    # 라인 기준 정렬 후 dedup
    found.sort(key=lambda x: x[0])
    dedup = []
    seen_li = set()
    for li, txt in found:
        if li not in seen_li:
            seen_li.add(li); dedup.append((li, txt))
    return dedup

# ----------------- Single-file extraction -----------------
# 소스 코드 파일 하나에서 코드 가젯을 추출하는 함수이다
# 파일 하나를 연다 -> sink 함수 호출을 찾는다 -> 문맥을 수집한다 -> dict 형태로 정리한다
# path는 분석할 파일의 경로, parser는 Tree-sitter, normalize는 주석, 공백 제거, cwe_rules는 True면 sink 함수 -> CWE 매핑 추가
def extract_from_file(path: Path, parser=None, normalize: bool=False, cwe_rules: bool=False) -> List[dict]:
    # 소스 코드 파일의 내용을 text로 읽는다
    code = read_text_file(path)
    # 최종 결과를 담을 코드 가젯 리스트
    gadgets: List[dict] = []

    # 1) AST 기반 탐지
    # 파서가 있으면 AST 기반으로 분석
    if parser is not None:
        try:
            code_bytes = code.encode('utf-8')

            # _find_call_expressions 함수는 AST에서 모든 함수 호출 노드(call_node)와 시작 위치(pos)를 찾는다
            for call_node, pos in _find_call_expressions(parser, code):
                # 함수를 추출하고 SINK 함수가 아닌 경우 continue
                fn = _extract_function_name_from_call(call_node, code_bytes)
                if not fn or fn not in SINK_FUNCS:
                    continue

                # 호출한 전체 문자열
                sink_text = _node_text(call_node, code_bytes).strip()
                # 괄호 안에 있는 인자 부분
                arg_text = _extract_arg_text_from_call(call_node, code_bytes)
                # 인자에서 추출한 변수명 리스트
                idents = extract_identifiers_from_arg_text(arg_text)

                # sink가 등장한 코드의 라인 번호
                sink_li = bytepos_to_line_index(code, pos)
                # sink가 속한 함수 정의 AST 노드
                func_node = _find_enclosing_function(call_node)

                # 함수 범위 내에서 sink 인자와 관련된 선언/할당 라인 수집
                # defs_line은 해당 라인 번호, defs_texts는 코드 문자열이다
                defs_texts: List[str] = []
                defs_lines: List[int] = []

                # 함수 범위 내에서 sink 인자와 관련된 선언/할당 라인 수집한다
                if func_node is not None and idents:
                    defs = collect_defs_in_function(func_node, code, code_bytes, idents, sink_li)
                    if defs:
                        defs_lines = [li for li,_ in defs]
                        defs_texts = [txt for _,txt in defs]

                # sink 기준으로 위쪽으로 거슬러 올라가서 변수/입력 관련 문맥 라인들을 모은다
                ctx_raw, ctx_begin, ctx_end = backward_context_search(code, pos, idents)
                # defs + ctx 합치기 (중복 제거)
                merged_lines = []
                seen = set()

                # def_use 라인들을 먼저 넣는다
                for li, txt in zip(defs_lines, defs_texts):
                    # (라인 번호, 코드) 쌍을 중복 제거해서 저장한다
                    key = (li, txt)

                    if key not in seen:
                        seen.add(key); merged_lines.append((li, txt))

                # ctx_raw를 라인 단위로 다시 붙이기 위해 분리
                ctx_lines = ctx_raw.splitlines()
                ctx_start_li = ctx_begin - 1

                # backward context 결과도 라인 단위로 넣는다
                for off, ln in enumerate(ctx_lines):
                    li = ctx_start_li + off
                    key = (li, ln)

                    if key not in seen:
                        seen.add(key); merged_lines.append((li, ln))

                # 정렬 후 텍스트화
                # 라인 번호 순서대로 정렬한다
                # 가젯 최종 코드 조각을 완성한다
                merged_lines.sort(key=lambda x: x[0])
                gadget_raw = "\n".join(t for _,t in merged_lines).strip()

                # --normalize 옵션이 있으면 주석/공백을 정리한다
                out_txt = normalize_code_for_model(gadget_raw) if normalize else gadget_raw
                # --cwe-rules 있으면 sink 함수 이름 -> CWE 후보 리스트 매핑한다
                cwe = SINK_TO_CWE.get(fn, []) if cwe_rules else []

                # 가젯 하나를 dict으로 정리한다
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
        # 파서가 없거나 AST 탐색이 실패했을 경우를 대비해서 정규식으로 sink 탐색
        regex_sinks = regex_find_sinks(code)

        for fn, sink_line_text, pos in regex_sinks:
            # 이미 동일 sink 라인이 있는 경우 스킵(대략적 중복 제거)
            # AST 기반 탐지에서 이미 같은 sink를 처리했다면 중복 저장을 하지 않는다
            if any(g.get('sink_name') == fn and g.get('sink_line') == bytepos_to_line_index(code, pos) + 1 for g in gadgets):
                continue

            # sink 함수 호출의 인자를 추출하여 변수명 식별자를 뽑는다
            m = re.search(r'\((.*)\)', sink_line_text, flags=re.S)
            args = m.group(1) if m else ''
            idents = extract_identifiers_from_arg_text(args)

            # backward context 수집한다
            # normalize와 CWE 후보 매핑을 처리한다
            ctx_raw, ctx_begin, ctx_end = backward_context_search(code, pos, idents)
            out_txt = normalize_code_for_model(ctx_raw) if normalize else ctx_raw
            cwe = SINK_TO_CWE.get(fn, []) if cwe_rules else []

            # AST 없는 경우에도 최소한 sinnk 주변 문맥을 가젯 dict으로 생성한다
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
# 해당 경로 디렉토리 안에 있는 모든 파일을 전부 모아 반환하는 함수
def iter_code_files(indir: str, exts: Tuple[str,...]) -> List[Path]:
    out = []

    # 인자로 받은 경로를 재귀적으로 순회
    for root, _, files in os.walk(indir):
        for f in files:
            # 순회한 파일들 중 이름이 exts 중 하나로 끝나면 Path(root) / f 로 만든 후 결과 리스트에 반환
            if f.endswith(exts):
                out.append(Path(root) / f)

    # 최종적으로 조건에 맞는 경로를 반환
    return out


# 디렉토리 안에 있는 코드 파일 전체를 대상으로 코드 가젯 추출을 실행하는 함수이다
# indir는 분석할 코드 디렉토리 경로, parser는 Tree-sitter 파서 객체, normalize는 True이면 결과 가젯의 주석, 공백 제거 해서 모델 입력용으로 가공
# cwe_rules는 True이면 sink 함수 이름 기반으로 CWE 후보 리스트를 붙여준다, exts는 처리할 파일 확장자 튜플, 기본값 .cpp, .c, .cc, .h, .hpp
def extract_from_dir(indir: str, parser=None, normalize: bool=False, cwe_rules: bool=False, exts: Tuple[str,...]=DEFAULT_EXTS) -> List[dict]:
    # indir 경로를 Path 객체로 바꾼다
    pdir = Path(indir)

    # 경로가 존재하지 않으면 에러
    if not pdir.exists():
        raise FileNotFoundError(f"Input dir not found: {indir}")

    # iter_code_files 함수를 호출하여 indir 디렉토리 내부에 있는 모든 소스 코드 파일 경로를 리스트로 얻는다
    files = iter_code_files(indir, exts)
    out: List[dict] = []

    # 얻은 파일들을 모두 순회하며 extract_from_file 호출하여 해당 파일에서 가젯들을 추출한다
    for p in files:
        try:
            gs = extract_from_file(p, parser=parser, normalize=normalize, cwe_rules=cwe_rules)

            # 값이 있으면 결과 out 리스트에 가젯들을 합친다
            if gs:
                out.extend(gs)
        except Exception:
            print(f"[warn] error processing {p}", file=sys.stderr)

    # 최종적으로 프로그램들에서 추출한 모든 가젯들을 담은 리스트를 반환한다
    return out

# ----------------- CLI -----------------
# argparse 라이브러리를 사용해서 명령줄 인자(CLI arguments)를 정의하고 파싱하는 함수이다
def parse_args():
    # 새로운 ArgumentParser 객체 생성, description은 --help 옵션을 쳤을 때 보이는 간단한 설명 문구
    ap = argparse.ArgumentParser(description='C/C++ static gadget extractor (tree-sitter + intra-proc slice)')
    # code_dir는 분석할 C/C++ 소스 디렉토리 경로이다
    ap.add_argument('code_dir', nargs='?', help='Directory containing C/C++ sources')
    # out_jsonl은 추출 결과를 JSONL 형식으로 저장할 경로이다
    ap.add_argument('out_jsonl', nargs='?', help='Output JSONL path')

    # --use-prebuilt 플래그는 주면 True, 안 주면 False
    # tree_sitter_languages 패키지의 사전 빌드된 파서를 사용하겠다는 옵션이다
    ap.add_argument('--use-prebuilt', action='store_true', help="Use tree_sitter_languages prebuilt parser for 'c'")
    # --ts-lib 옵션, 사용자가 직접 빌드한 Tree-sitter 공유 라이브러리 파일 경로 저장
    ap.add_argument('--ts-lib', help='Path to tree-sitter shared library (.so/.dylib/.dll) built with C grammar')

    # --build-grammar는 grammar 디렉토리(tree-sitter-c) 경로들을 하나 이상 입력을 받는다
    ap.add_argument('--build-grammar', nargs='+', help='Build a tree-sitter lib from given grammar directories and exit')
    # --out-lib은 빌드된 .so 라이브러리의 저장 경로이다
    ap.add_argument('--out-lib', default='build/my-languages.so', help='Output path for built tree-sitter lib (used with --build-grammar)')

    # --normalize 플래그는 켜면 추출된 가젯에서 주석을 제거, 공백을 정리한다
    ap.add_argument('--normalize', action='store_true', help='Normalize gadget text for model input')
    # --cwe-rules 플래그는 켜면 sink_name을 기준으로 CWE 후보 목록을 결과 JSONL에 붙여준다
    ap.add_argument('--cwe-rules', action='store_true', help='Attach simple CWE candidate list from sink name')

    # --ext는 처리할 파일 확장자 목룍, 여러 개 입력 가능하다
    ap.add_argument('--ext', nargs='*', default=list(DEFAULT_EXTS), help='File extensions to include (e.g., --ext .c .h .cpp)')
    # --quiet는 출력 최소화 모드로 경고/정보 메세지를 보여준다
    ap.add_argument('--quiet', action='store_true')

    return ap.parse_args()


def main():
    # 사용자가 넘긴 옵션(경로, 플래그들)을 args에 담아온다
    args = parse_args()

    # 실행했을 때 --build-grammar가 주어지면 코드 추출은 하지 않고 지정한 grammer 디렉토리들로부터 Tree-sitter 공유 라이브러리 .so/.dylib/.dll를 빌드한다
    # 1) Grammar build mode
    if args.build_grammar:
        try:
            # 만약 CLI를 이렇게 실행했다면 python static_extractor.py --build-grammar ./tree-sitter-c --out-lib build/c-langs.so
            # args.build_grammar = ["./tree-sitter-c"]
            # args.out_lib = "build/c-langs.so"
            # 명령문 실행 후 build/c-langs.so 파일이 생성 (path)
            # path는 실제로 생성된 Tree-sitter 라이브러리 파일의 경로 문자열이다
            path = build_tree_sitter_lib(args.build_grammar, args.out_lib)
            print(f"[main] built tree-sitter lib: {path}")

            return
        except Exception as e:
            print('[main] build failed:', e, file=sys.stderr)
            sys.exit(2)

    # build 모드가 아니라면, 입력 디렉토리와 출력 경로가 필수적이다
    if not args.code_dir or not args.out_jsonl:
        print('Usage: python static_extractor.py <code_dir> <out_jsonl> [--use-prebuilt | --ts-lib PATH] [--normalize] [--cwe-rules]', file=sys.stderr)
        sys.exit(1)

    # 파서 초기화
    # 2) Parser init
    parser = None

    # --use-prebuilt를 쓴 경우에는 tree_sitter_languages 패키지의 사전 빌드된 C 파서를 시도한다
    if args.use_prebuilt:
        parser = init_parser(use_prebuilt=True)
        if parser is None and not args.quiet:
            print('[main] prebuilt parser not available. Fallback to regex.', file=sys.stderr)
    elif args.ts_lib: # --ts-lib가 제공되면 사용자가 지정한 공유 라이브러리 경로에서 파서를 로드 시도
        parser = init_parser(ts_lib=args.ts_lib)
        if parser is None and not args.quiet:
            print('[main] ts-lib provided but init failed. Fallback to regex.', file=sys.stderr)
    else: # 위 두 가지가 모두 아니라면, 환경변수 TREE_SITTER_LIB가 있으면 그 경로로 파서 로드 시도
        env_lib = os.environ.get('TREE_SITTER_LIB')

        if env_lib:
            parser = init_parser(ts_lib=env_lib)

            if parser is None and not args.quiet:
                print('[main] TREE_SITTER_LIB init failed. Fallback to regex.', file=sys.stderr)

        # 파서가 None이면 prebuilt 시도 한 번 더 한다
        if parser is None:
            # try prebuilt silently
            parser = init_parser(use_prebuilt=True)

    # 최종적으로 parser가 준비되지 않았다면 quiet가 아닌 경우 정규식만 사용한다고 알린다
    if parser is None and not args.quiet:
        print('[main] note: tree-sitter parser not initialized; regex fallback only', file=sys.stderr)

    # 어떤 디렉토리에서 추출을 시작하는지 콘솔에 알린다
    # 3) Extract
    print('[main] extracting from:', args.code_dir)
    gadgets = extract_from_dir(args.code_dir, parser=parser, normalize=args.normalize, cwe_rules=args.cwe_rules, exts=tuple(args.ext))

    # 4) Save
    write_jsonl(gadgets, args.out_jsonl)
    print(f"[main] saved {len(gadgets)} gadgets -> {args.out_jsonl}")


if __name__ == '__main__':
    main()
