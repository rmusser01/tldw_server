"""
Code-aware chunking strategy that segments source files into logical blocks
(imports, classes, functions) across multiple languages using heuristic parsing.

Notes:
- Supports brace-based languages (C/C++/Java/C#/Go/Rust/JS/TS/Swift/Kotlin) and
  indentation-based languages (Python, Ruby) with best-effort detection.
- Returns chunks as plain strings; metadata can be added by the caller.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass

from loguru import logger

from ..base import BaseChunkingStrategy, ChunkMetadata, ChunkResult


@dataclass
class _Header:
    kind: str
    name: str | None
    line_index: int


class CodeChunkingStrategy(BaseChunkingStrategy):
    """Heuristic, language-agnostic code chunker."""

    # Regex patterns for various languages
    PY_HEADER_RE = re.compile(r"^\s*(def|class)\s+([A-Za-z_][\w]*)[\(:]", re.UNICODE)
    # JS/TS
    JSTYPE_FUNC_RE = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_]\w*)\s*\(")
    JSTYPE_CLASS_RE = re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_][\w]*)\b")
    JSTYPE_EXPORT_DEFAULT_CLASS_RE = re.compile(r"^\s*export\s+default\s+class\s+([A-Za-z_][\w]*)\b")
    JSTYPE_CONST_ARROW_RE = re.compile(r"^\s*(?:export\s+)?const\s+([A-Za-z_]\w*)\s*=\s*\([^;{}]*\)\s*=>\s*\{\s*$")
    JSTYPE_CONST_FUNC_RE = re.compile(r"^\s*(?:export\s+)?const\s+([A-Za-z_]\w*)\s*=\s*function\s*\(")
    JSTYPE_EXPORT_DEFAULT_FUNC_NAMED_RE = re.compile(r"^\s*export\s+default\s+(?:async\s+)?function\s+([A-Za-z_]\w*)\s*\(")
    JSTYPE_EXPORT_DEFAULT_FUNC_ANON_RE = re.compile(r"^\s*export\s+default\s+(?:async\s+)?function\s*\(")
    JSTYPE_EXPORT_DEFAULT_ARROW_RE = re.compile(r"^\s*export\s+default\s*\([^;{}]*\)\s*=>\s*\{\s*$")
    TSTYPE_INTERFACE_RE = re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z_][\w]*)\b")
    TSTYPE_TYPE_RE = re.compile(r"^\s*(?:export\s+)?type\s+([A-Za-z_][\w]*)\s*=")
    JSTYPE_METHOD_RE = re.compile(r"^\s*(?:public|private|protected|static|async)?\s*([A-Za-z_][\w]*)\s*\([^;{}]*\)\s*\{\s*$")
    # C/C++/Java/C# (very heuristic)
    C_LIKE_FUNC_RE = re.compile(r"^\s*(?:template\s*<[^>]+>\s*)?(?:[\w:\*\&\[\]<>]+\s+)+([A-Za-z_][\w]*)\s*\([^;{]*\)\s*(?:const\s*)?\{\s*$")
    C_LIKE_CLASS_RE = re.compile(r"^\s*(?:public|private|protected|abstract|final|sealed|partial|static)?\s*class\s+([A-Za-z_][\w]*)\b")
    GO_FUNC_RE = re.compile(r"^\s*func\s+(?:\([^)]*\)\s+)?([A-Za-z_]\w*)\s*\(")
    RUST_FUNC_RE = re.compile(r"^\s*fn\s+([A-Za-z_]\w*)\s*\(")
    RUST_TYPE_RE = re.compile(r"^\s*(struct|enum|trait|impl)\s+([A-Za-z_][\w]*)\b")
    RUBY_HEADER_RE = re.compile(r"^\s*(def|class|module)\s+([A-Za-z_][\w]*)")
    SHELL_FUNC_RE = re.compile(r"^\s*([A-Za-z_][\w]*)\s*\(\)\s*\{\s*$")

    IMPORT_LINE_RE = re.compile(
        r"^\s*(?:import\b|from\s+\S+\s+import\b|#include\b|using\s+\w+;|package\b)"
    )

    def _is_brace_lang(self, language: str) -> bool:
        return language.lower() in {
            'c', 'cpp', 'csharp', 'java', 'go', 'rust', 'swift', 'kotlin', 'javascript', 'typescript', 'tsx', 'jsx',
        }

    def _find_headers(self, lines: list[str], language: str) -> list[_Header]:
        headers: list[_Header] = []
        lang = language.lower()
        for idx, line in enumerate(lines):
            m = None
            if lang == 'python':
                m = self.PY_HEADER_RE.match(line)
                if m:
                    headers.append(_Header(m.group(1), m.group(2), idx))
                    continue
            if lang in ('javascript', 'typescript', 'tsx', 'jsx'):
                m = self.JSTYPE_EXPORT_DEFAULT_CLASS_RE.match(line)
                if m:
                    headers.append(_Header('class', m.group(1), idx))
                    continue
                m = self.JSTYPE_CLASS_RE.match(line)
                if m:
                    headers.append(_Header('class', m.group(1), idx))
                    continue
                m = self.JSTYPE_FUNC_RE.match(line)
                if m:
                    headers.append(_Header('function', m.group(1), idx))
                    continue
                m = self.JSTYPE_EXPORT_DEFAULT_FUNC_NAMED_RE.match(line)
                if m:
                    headers.append(_Header('function', m.group(1), idx))
                    continue
                m = self.JSTYPE_EXPORT_DEFAULT_FUNC_ANON_RE.match(line)
                if m:
                    headers.append(_Header('function', 'default', idx))
                    continue
                m = self.JSTYPE_EXPORT_DEFAULT_ARROW_RE.match(line)
                if m:
                    headers.append(_Header('function', 'default', idx))
                    continue
                m = self.JSTYPE_CONST_ARROW_RE.match(line)
                if m:
                    headers.append(_Header('function', m.group(1), idx))
                    continue
                m = self.JSTYPE_CONST_FUNC_RE.match(line)
                if m:
                    headers.append(_Header('function', m.group(1), idx))
                    continue
                m = self.TSTYPE_INTERFACE_RE.match(line)
                if m:
                    headers.append(_Header('interface', m.group(1), idx))
                    continue
                m = self.TSTYPE_TYPE_RE.match(line)
                if m:
                    headers.append(_Header('type', m.group(1), idx))
                    continue
                # methods will be handled inside class blocks via braces
            if lang in ('c', 'cpp', 'csharp', 'java'):
                m = self.C_LIKE_CLASS_RE.match(line)
                if m:
                    headers.append(_Header('class', m.group(1), idx))
                    continue
                m = self.C_LIKE_FUNC_RE.match(line)
                if m:
                    headers.append(_Header('function', m.group(1), idx))
                    continue
            if lang == 'go':
                m = self.GO_FUNC_RE.match(line)
                if m:
                    headers.append(_Header('func', m.group(1), idx))
                    continue
            if lang == 'rust':
                m = self.RUST_TYPE_RE.match(line)
                if m:
                    headers.append(_Header(m.group(1), m.group(2), idx))
                    continue
                m = self.RUST_FUNC_RE.match(line)
                if m:
                    headers.append(_Header('fn', m.group(1), idx))
                    continue
            if lang == 'ruby':
                m = self.RUBY_HEADER_RE.match(line)
                if m:
                    headers.append(_Header(m.group(1), m.group(2), idx))
                    continue
            # Shell-like
            m = self.SHELL_FUNC_RE.match(line)
            if m:
                headers.append(_Header('function', m.group(1), idx))
                continue
        headers.sort(key=lambda h: h.line_index)
        return headers

    def _indent(self, s: str) -> int:
        return len(s) - len(s.lstrip(' '))

    def _block_end_indent(self, lines: list[str], start_idx: int) -> int:
        return self._indent(lines[start_idx])

    def _find_block_for_header_indent(self, lines: list[str], header_idx: int) -> int:
        # For indentation-based (Python/Ruby). End before next header at same or shallower indent
        base_indent = self._indent(lines[header_idx])
        i = header_idx + 1
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                i += 1
                continue
            if self._indent(line) <= base_indent and (self.PY_HEADER_RE.match(line) or self.RUBY_HEADER_RE.match(line)):
                return i
            i += 1
        return len(lines)

    def _find_block_for_header_braces(self, lines: list[str], header_idx: int) -> int:
        # For brace languages: find first '{' after header, then match braces
        i = header_idx
        # find start brace
        brace_count = 0
        found = False
        while i < len(lines):
            line = lines[i]
            if '{' in line:
                brace_count = line.count('{') - line.count('}')
                found = True
                if brace_count <= 0:
                    return i + 1
                i += 1
                break
            # In some languages, the brace may be on the next line after signature
            i += 1
        if not found:
            # Fallback: single-line or no brace; use until next header or blank line
            j = header_idx + 1
            while j < len(lines) and lines[j].strip():
                j += 1
            return j
        # Now find when brace_count returns to zero from this point
        while i < len(lines):
            brace_count += lines[i].count('{') - lines[i].count('}')
            if brace_count <= 0:
                return i + 1
            i += 1
        return len(lines)

    def _extract_import_block(self, lines: list[str]) -> tuple[int, int]:
        # From start of file to last consecutive import-like line (skipping comments and blanks)
        start = 0
        end = 0
        i = 0
        seen_non_import = False
        while i < len(lines):
            s = lines[i]
            if not s.strip() or s.strip().startswith(('#', '//', '/*')):
                # Keep leading comments as part of header block until imports end
                if not seen_non_import:
                    end = i + 1
                i += 1
                continue
            if self.IMPORT_LINE_RE.match(s):
                end = i + 1
                i += 1
                continue
            # First non-import code line
            seen_non_import = True
            break
        return (start, end)

    def _pack_blocks(self, blocks: list[tuple[int, int]], lines: list[str], max_chars: int, overlap_chars: int) -> list[str]:
        # Greedy pack blocks into chunks respecting max_chars; use simple character overlap
        chunks: list[str] = []
        buf = ''
        for (s, e) in blocks:
            text = '\n'.join(lines[s:e]).rstrip()
            if not text:
                continue
            if not buf:
                buf = text
            elif len(buf) + 2 + len(text) <= max_chars:
                buf = f"{buf}\n\n{text}"
            else:
                if buf:
                    chunks.append(buf)
                if len(text) > max_chars:
                    # Split oversized block into non-overlapping windows.
                    # Overlap is added later via prefixing tails.
                    t = text
                    while t:
                        part = t[:max_chars]
                        chunks.append(part)
                        t = '' if len(t) <= max_chars else t[max_chars:]
                    buf = ''
                else:
                    buf = text
        if buf:
            chunks.append(buf)
        # Add overlap between consecutive chunks if requested
        if overlap_chars > 0 and len(chunks) > 1:
            overlapped: list[str] = []
            prev = ''
            for i, ch in enumerate(chunks):
                if i == 0:
                    overlapped.append(ch)
                    prev = ch
                    continue
                tail = prev[-overlap_chars:] if prev else ''
                if tail:
                    overlapped.append(f"{tail}{ch}")
                else:
                    overlapped.append(ch)
                prev = ch
            return overlapped
        return chunks

    def chunk(self, text: str, max_size: int, overlap: int = 0, **options) -> list[str]:
        if not self.validate_parameters(text, max_size, overlap):
            return []
        language = str(options.get('language') or self.language or 'text')
        lines = text.splitlines()
        if not lines:
            return []

        try:
            # Identify headers per language
            headers = self._find_headers(lines, language)
            blocks: list[tuple[int, int]] = []
            if headers:
                # Only split out the leading import/comment prefix when we also
                # found later code headers. Otherwise the full file is the block.
                imp_s, imp_e = self._extract_import_block(lines)
                if imp_e > imp_s:
                    blocks.append((imp_s, imp_e))
                for _idx, h in enumerate(headers):
                    start = h.line_index
                    if self._is_brace_lang(language):
                        end = self._find_block_for_header_braces(lines, start)
                    else:
                        end = self._find_block_for_header_indent(lines, start)
                    # Ensure monotonic and non-overlapping
                    if blocks and start < blocks[-1][1]:
                        start = blocks[-1][1]
                    if end <= start:
                        continue
                    blocks.append((start, end))
            else:
                # Fallback: treat the whole file as one block
                blocks.append((0, len(lines)))

            chunks = self._pack_blocks(blocks, lines, max_chars=max_size, overlap_chars=overlap)
            logger.info(f"CodeChunkingStrategy produced {len(chunks)} chunks (language={language})")
            return chunks
        except Exception as e:
            logger.warning(f"Code chunking failed, falling back to whole text: {e}")
            return [text]

    def chunk_with_metadata(self, text: str, max_size: int, overlap: int = 0, **options) -> list[ChunkResult]:
        if not self.validate_parameters(text, max_size, overlap):
            return []
        language = str(options.get('language') or self.language or 'text')
        # Preserve line endings for accurate char offsets
        lines_ke = text.splitlines(keepends=True)
        if not lines_ke:
            return []
        # Build cumulative char positions for each line start
        line_starts = []
        pos = 0
        for ln in lines_ke:
            line_starts.append(pos)
            pos += len(ln)
        total_chars = pos

        # Helper to convert (start_line, end_line_exclusive) -> (start_char, end_char)
        def span_chars(s_line: int, e_line: int) -> tuple:
            # Clamp indices safely
            s_char = 0 if not 0 <= s_line < len(line_starts) else line_starts[s_line]
            e_char = line_starts[e_line] if 0 <= e_line < len(line_starts) else total_chars
            if e_char < s_char:
                e_char = s_char
            return s_char, e_char

        # Derive structured blocks (imports and headers)
        # Use non-keepends lines for regex detection
        lines = [ln.rstrip('\n').rstrip('\r') for ln in lines_ke]
        blocks: list[tuple[int, int, str, str | None]] = []  # (s_line, e_line_excl, block_type, name)
        headers = self._find_headers(lines, language)
        if headers:
            imp_s, imp_e = self._extract_import_block(lines)
            if imp_e > imp_s:
                blocks.append((imp_s, imp_e, 'imports', None))
            for h in headers:
                s = h.line_index
                if self._is_brace_lang(language):
                    e = self._find_block_for_header_braces(lines, s)
                else:
                    e = self._find_block_for_header_indent(lines, s)
                if blocks and s < blocks[-1][1]:
                    s = blocks[-1][1]
                if e <= s:
                    continue
                # Normalize header kinds
                kind_map = {
                    'def': 'function', 'func': 'function', 'function': 'function', 'fn': 'function',
                    'class': 'class', 'struct': 'struct', 'enum': 'enum', 'trait': 'trait', 'impl': 'impl', 'module': 'module',
                }
                btype = kind_map.get(h.kind, h.kind)
                blocks.append((s, e, btype, h.name))
        if not blocks:
            blocks.append((0, len(lines), 'module', None))

        # Greedy pack blocks into chunks
        results: list[ChunkResult] = []
        buf_start_char = None
        buf_end_char = None
        buf_blocks: list[tuple[int, int, str, str | None]] = []

        def flush_chunk():
            nonlocal buf_start_char, buf_end_char, buf_blocks
            if buf_start_char is None or buf_end_char is None:
                return
            end_char = buf_end_char
            # Expand end bound to avoid splitting graphemes in metadata
            try:
                end_char = self._expand_end_to_grapheme_boundary(text, end_char)
            except Exception:
                end_char = buf_end_char
            ch_text = text[buf_start_char:end_char]
            # Compute line span for this chunk
            # Find first line index where start_char >= line_start
            start_line_idx = 0
            for i, ls in enumerate(line_starts):
                if ls <= buf_start_char:
                    start_line_idx = i
                else:
                    break
            end_line_idx = start_line_idx
            for i, ls in enumerate(line_starts):
                if ls < end_char:
                    end_line_idx = i
                else:
                    break
            md = ChunkMetadata(
                index=len(results),
                start_char=buf_start_char,
                end_char=end_char,
                word_count=len(ch_text.split()),
                language=language,
                method='code',
                options={
                    'start_line': start_line_idx + 1,
                    'end_line': end_line_idx + 1,
                    'lines_in_chunk': (end_line_idx - start_line_idx + 1),
                    'blocks': [
                        {
                            'type': btype,
                            'name': name,
                            'start_line': s + 1,
                            'end_line': e,
                        }
                        for (s, e, btype, name) in buf_blocks
                    ],
                }
            )
            results.append(ChunkResult(text=ch_text, metadata=md))
            buf_start_char = None
            buf_end_char = None
            buf_blocks = []

        for (s_line, e_line, btype, name) in blocks:
            s_char, e_char = span_chars(s_line, e_line)
            blen = e_char - s_char
            if buf_start_char is None:
                if blen <= max_size:
                    buf_start_char = s_char
                    buf_end_char = e_char
                    buf_blocks = [(s_line, e_line, btype, name)]
                else:
                    # Split oversized block into windows; overlap is applied via prefixing tails.
                    start = s_char
                    while start < e_char:
                        end = min(e_char, start + max_size)
                        # Calculate line indices
                        try:
                            end_expanded = self._expand_end_to_grapheme_boundary(text, end)
                        except Exception:
                            end_expanded = end
                        ch_text = text[start:end_expanded]
                        start_line_idx = max(0, max([i for i, ls in enumerate(line_starts) if ls <= start], default=0))
                        end_line_idx = max(start_line_idx, max([i for i, ls in enumerate(line_starts) if ls < end_expanded], default=start_line_idx))
                        md = ChunkMetadata(
                            index=len(results),
                            start_char=start,
                            end_char=end_expanded,
                            word_count=len(ch_text.split()),
                            language=language,
                            method='code',
                            options={
                                'start_line': start_line_idx + 1,
                                'end_line': end_line_idx + 1,
                                'lines_in_chunk': (end_line_idx - start_line_idx + 1),
                                'blocks': [
                                    {'type': btype, 'name': name, 'start_line': s_line + 1, 'end_line': e_line}
                                ],
                                'partial_block': True,
                            }
                        )
                        results.append(ChunkResult(text=ch_text, metadata=md))
                        if end_expanded >= e_char:
                            break
                        # Split continues without intra-block overlap; overlap is applied via prefixing tails.
                        start = start + max_size
                    # Reset buffer
                    buf_start_char = None
                    buf_end_char = None
                    buf_blocks = []
            else:
                # Try to add current block into buffer
                new_len = (e_char - buf_start_char)
                if new_len <= max_size:
                    buf_end_char = e_char
                    buf_blocks.append((s_line, e_line, btype, name))
                else:
                    flush_chunk()
                    # Start new buffer with current block (or split if needed)
                    if (e_char - s_char) <= max_size:
                        buf_start_char = s_char
                        buf_end_char = e_char
                        buf_blocks = [(s_line, e_line, btype, name)]
                    else:
                        # Split oversized block as above
                        start = s_char
                        while start < e_char:
                            end = min(e_char, start + max_size)
                            try:
                                end_expanded = self._expand_end_to_grapheme_boundary(text, end)
                            except Exception:
                                end_expanded = end
                            ch_text = text[start:end_expanded]
                            start_line_idx = max(0, max([i for i, ls in enumerate(line_starts) if ls <= start], default=0))
                            end_line_idx = max(start_line_idx, max([i for i, ls in enumerate(line_starts) if ls < end_expanded], default=start_line_idx))
                            md = ChunkMetadata(
                                index=len(results),
                                start_char=start,
                                end_char=end_expanded,
                                word_count=len(ch_text.split()),
                                language=language,
                                method='code',
                                options={
                                    'start_line': start_line_idx + 1,
                                    'end_line': end_line_idx + 1,
                                    'lines_in_chunk': (end_line_idx - start_line_idx + 1),
                                    'blocks': [
                                        {'type': btype, 'name': name, 'start_line': s_line + 1, 'end_line': e_line}
                                    ],
                                    'partial_block': True,
                                }
                            )
                            results.append(ChunkResult(text=ch_text, metadata=md))
                            if end_expanded >= e_char:
                                break
                            # Split continues without intra-block overlap; overlap is applied via prefixing tails.
                            start = start + max_size
                        buf_start_char = None
                        buf_end_char = None
                        buf_blocks = []

        # Flush any remaining buffer
        flush_chunk()

        # Apply prefix-tail overlap so chunks can grow beyond max_size while preserving metadata offsets.
        if overlap > 0 and results:
            def _line_index_for_char(pos: int) -> int:
                if pos <= 0:
                    return 0
                if pos >= total_chars:
                    return max(0, len(line_starts) - 1)
                return max(0, bisect_right(line_starts, pos) - 1)

            prev = results[0]
            for cur in results[1:]:
                try:
                    prev_start = int(prev.metadata.start_char)
                    prev_end = int(prev.metadata.end_char)
                    cur_start = int(cur.metadata.start_char)
                    cur_end = int(cur.metadata.end_char)
                except Exception:
                    prev = cur
                    continue
                prev_len = max(0, prev_end - prev_start)
                tail_len = min(overlap, prev_len)
                if tail_len <= 0:
                    prev = cur
                    continue
                new_start = max(0, prev_end - tail_len)
                if new_start < cur_start and cur_end > new_start:
                    cur.metadata.start_char = new_start
                    cur.metadata.char_count = cur_end - new_start
                    cur.metadata.overlap_with_previous = tail_len
                    prev.metadata.overlap_with_next = tail_len
                    cur.text = text[new_start:cur_end]
                    cur.metadata.word_count = len(cur.text.split()) if cur.text else 0
                    if isinstance(cur.metadata.options, dict):
                        start_line_idx = _line_index_for_char(new_start)
                        end_line_idx = _line_index_for_char(cur_end - 1 if cur_end > 0 else 0)
                        cur.metadata.options['start_line'] = start_line_idx + 1
                        cur.metadata.options['end_line'] = end_line_idx + 1
                        cur.metadata.options['lines_in_chunk'] = (end_line_idx - start_line_idx + 1)
                prev = cur

        logger.info(f"CodeChunkingStrategy produced {len(results)} chunks with metadata (language={language})")
        return results
