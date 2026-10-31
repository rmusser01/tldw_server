"""
Proposition-based chunking strategy.

This strategy splits text into minimal meaning-bearing propositions using
lightweight heuristics (punctuation, clause markers, and coordination),
then packs propositions into chunks of configurable size with optional overlap.

Design goals:
- Zero new heavy dependencies (uses built-in/available libs)
- Deterministic, fast baseline with optional aggressiveness tuning
- Works across general English prose; degrades gracefully

Options (via **options):
- aggressiveness (int: 0,1,2): how aggressively to split (default: 1)
  0 = punctuation-only, 1 = + subordinate markers, 2 = + coordination (and/or/but) when verb-like on both sides
- min_proposition_length (int): minimum characters per proposition before merging (default: 15)
- keep_markers (bool): keep the split markers at the edges (default: True)

Notes:
- For higher quality, a future variant can use spaCy dependency parses;
  this baseline avoids requiring model downloads. If spaCy with en_core_web_* is
  available, we can upgrade later to leverage it behind a feature flag.
"""

import contextlib
import re
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.Utils.prompt_loader import load_prompt

from ..base import BaseChunkingStrategy, ChunkMetadata, ChunkResult


class PropositionChunkingStrategy(BaseChunkingStrategy):
    """Split text into propositions and pack into chunks by count.

    max_size: number of propositions per chunk
    overlap: number of propositions to overlap between consecutive chunks
    """

    # Common clause markers and subordinators
    _SUBORDINATE_MARKERS = [
        "because", "since", "although", "though", "unless", "until",
        "while", "whereas", "where", "when", "if", "that", "which",
        "who", "whom", "whose", "after", "before", "as", "so that",
        "in order to"
    ]

    # Coordinating conjunctions used for splitting at higher aggressiveness
    _COORD_CONJ = ["and", "or", "but"]

    # Very lightweight verb-like detection: auxiliaries + common morphological hints
    _AUXILIARIES = {
        "be", "am", "is", "are", "was", "were", "been", "being",
        "have", "has", "had", "do", "does", "did",
        "can", "could", "should", "would", "will", "shall", "may", "might", "must"
    }

    def __init__(self, language: str = 'en', llm_call_func: Optional[Any] = None, llm_config: Optional[dict[str, Any]] = None):
        super().__init__(language)
        self.llm_call_func = llm_call_func
        self.llm_config = (llm_config or {}).copy()

    def chunk(self, text: str, max_size: int, overlap: int = 0, **options) -> list[str]:
        if not self.validate_parameters(text, max_size, overlap):
            return []

        # Engine selection: 'heuristic' | 'spacy' | 'llm' | 'auto'
        engine = str(options.get("engine", options.get("proposition_engine", "heuristic"))).lower()
        aggressiveness = int(options.get("aggressiveness", options.get("proposition_aggressiveness", 1)))
        min_prop_len = int(options.get("min_proposition_length", 15))

        # Select engine behavior
        prompt_profile = str(options.get("proposition_prompt_profile", options.get("prompt_profile", "generic"))).lower()

        if engine in ("auto", "spacy"):
            propositions = self._propositions_via_spacy(text, aggressiveness, min_prop_len)
            if not propositions and engine == "auto":
                # Fallback to heuristic
                propositions = self._propositions_via_heuristics(text, aggressiveness, min_prop_len)
        elif engine == "llm":
            propositions = self._propositions_via_llm(text, min_prop_len, prompt_profile)
            if not propositions:
                logger.warning("LLM extraction unavailable/failed; falling back to heuristics")
                propositions = self._propositions_via_heuristics(text, aggressiveness, min_prop_len)
        else:
            # heuristic default
            propositions = self._propositions_via_heuristics(text, aggressiveness, min_prop_len)

        if not propositions:
            # Fallback: return original text as a single chunk
            return [text.strip()] if text.strip() else []

        # Step 3: pack propositions into chunks with overlap (by proposition count)
        step = max(1, max_size - overlap)
        chunks: list[str] = []
        for i in range(0, len(propositions), step):
            window = propositions[i:i + max_size]
            if not window:
                continue
            chunk_text = self._join_props(window)
            chunks.append(chunk_text)

        logger.debug(f"Created {len(chunks)} chunks from {len(propositions)} propositions")
        return chunks

    def chunk_with_metadata(self, text: str, max_size: int, overlap: int = 0, **options) -> list[ChunkResult]:
        """Chunk text and return metadata with reliable source offsets.

        For proposition chunking, offsets are derived from source spans using a
        normalization-aware mapping (whitespace collapsing and dash handling)
        to keep offsets stable even when chunk text is normalized.
        """
        if not self.validate_parameters(text, max_size, overlap):
            return []

        if overlap >= max_size:
            logger.warning(f"Overlap ({overlap}) >= max_size ({max_size}), setting to max_size - 1")
            overlap = max_size - 1

        # Extract options (mirror chunk())
        engine = str(options.get("engine", options.get("proposition_engine", "heuristic"))).lower()
        aggressiveness = int(options.get("aggressiveness", options.get("proposition_aggressiveness", 1)))
        min_prop_len = int(options.get("min_proposition_length", 15))

        # Build proposition spans from source text (heuristic path for reliable offsets)
        spans = self._proposition_spans_with_offsets(
            text=text,
            aggressiveness=aggressiveness,
            min_prop_len=min_prop_len,
        )

        if not spans:
            # Fallback: return original text as a single chunk when no propositions extracted
            if not text.strip():
                return []
            md = ChunkMetadata(
                index=0,
                start_char=0,
                end_char=len(text),
                word_count=len(text.split()),
                language=self.language,
                overlap_with_previous=0,
                overlap_with_next=0,
                method='propositions',
                options={
                    'engine': engine,
                    'aggressiveness': aggressiveness,
                    'min_proposition_length': min_prop_len,
                },
            )
            return [ChunkResult(text=text, metadata=md)]

        results: list[ChunkResult] = []
        step = max(1, max_size - overlap)
        total_props = len(spans)

        for i in range(0, total_props, step):
            window = spans[i:i + max_size]
            if not window:
                continue
            start_char = min(s for s, _e in window)
            end_char = max(e for _s, e in window)
            if end_char < start_char:
                end_char = start_char
            with contextlib.suppress(Exception):
                end_char = self._expand_end_to_grapheme_boundary(text, end_char, options=options)
            chunk_text = text[start_char:end_char]
            md = ChunkMetadata(
                index=len(results),
                start_char=start_char,
                end_char=end_char,
                word_count=len(chunk_text.split()) if chunk_text else 0,
                language=self.language,
                overlap_with_previous=overlap if i > 0 else 0,
                overlap_with_next=overlap if (i + step) < total_props else 0,
                method='propositions',
                options={
                    'engine': engine,
                    'aggressiveness': aggressiveness,
                    'min_proposition_length': min_prop_len,
                },
            )
            results.append(ChunkResult(text=chunk_text, metadata=md))

        return results

    # --- helpers ---
    def _propositions_via_heuristics(self, text: str, aggressiveness: int, min_prop_len: int) -> list[str]:
        # Step 1: sentence segmentation (reuse sentence strategy heuristics)
        sentences = self._split_sentences_fast(text)
        if not sentences:
            return []

        # Step 2: split each sentence into propositions
        propositions: list[str] = []
        for sent in sentences:
            props = self._split_sentence_into_propositions(sent, aggressiveness)
            props = self._merge_short_propositions(props, min_prop_len)
            for p in props:
                p_clean = self._normalize_space(p)
                if p_clean:
                    propositions.append(p_clean)
        return propositions

    def _propositions_via_spacy(self, text: str, aggressiveness: int, min_prop_len: int) -> list[str]:
        try:
            import spacy  # type: ignore
        except Exception:
            logger.info("spaCy not installed; cannot use 'spacy' engine")
            return []

        nlp = None
        try:
            # Prefer small English model if available for parser; else blank with sentencizer
            try:
                nlp = spacy.load("en_core_web_sm")
            except Exception:
                nlp = spacy.blank("en")
                if not nlp.has_pipe("sentencizer"):
                    nlp.add_pipe("sentencizer")
        except Exception as e:
            logger.warning("Failed to initialize spaCy pipeline; error_type={}", type(e).__name__)
            return []

        doc = nlp(text)
        propositions: list[str] = []

        for sent in getattr(doc, "sents", [doc]):
            # If no parser is available, fall back to heuristic splitting within the sentence
            if not nlp.has_pipe("parser") and not any(t.dep_ for t in sent):
                props = self._split_sentence_into_propositions(sent.text, aggressiveness)
                props = self._merge_short_propositions(props, min_prop_len)
                propositions.extend(props)
                continue

            # Identify clause boundary indices using dependency labels: 'mark', 'cc' + 'conj'
            boundaries = [0]
            tokens = list(sent)
            for i, tok in enumerate(tokens):
                dep = tok.dep_.lower() if tok.dep_ else ""
                text_low = tok.text.lower()
                if dep == "mark":
                    # subordinate clause marker starts a new proposition
                    if i > 0 and i not in boundaries:
                        boundaries.append(i)
                elif dep == "cc" and text_low in self._COORD_CONJ:
                    # check right sibling conj for clause-likeness
                    if i + 1 < len(tokens):
                        right = sent[i+1:]
                        left = sent[boundaries[-1]:i]
                        if self._looks_clause_like(left.text) and self._looks_clause_like(right.text):
                            boundaries.append(i+1)

            if boundaries[-1] != len(tokens):
                boundaries.append(len(tokens))

            # Slice by boundaries
            for bi in range(len(boundaries) - 1):
                start, end = boundaries[bi], boundaries[bi+1]
                seg = sent[start:end].text.strip()
                if seg:
                    propositions.append(seg)

        propositions = self._merge_short_propositions(propositions, min_prop_len)
        propositions = [self._normalize_space(p) for p in propositions if p and self._normalize_space(p)]
        return propositions

    def _propositions_via_llm(self, text: str, min_prop_len: int, prompt_profile: str = "generic") -> list[str]:
        if not self.llm_call_func:
            logger.info("No LLM function provided; cannot use 'llm' engine")
            return []

        # Break text into manageable windows (approx. by characters)
        windows = self._windows_by_chars(text, target=self.llm_config.get('window_chars', 1200))
        results: list[str] = []
        for win in windows:
            prompt = self._build_llm_prompt(win, profile=prompt_profile)
            try:
                extracted = self._call_llm(prompt)
                props = self._parse_llm_props(extracted)
                if props:
                    results.extend(props)
            except Exception as e:
                logger.warning(
                    "LLM proposition extraction failed on a window; error_type={}",
                    type(e).__name__,
                )
                continue

        if not results:
            return []
        # Normalize and merge shorts
        normed = [self._normalize_space(p) for p in results if p and self._normalize_space(p)]
        return self._merge_short_propositions(normed, min_prop_len)

    def _windows_by_chars(self, text: str, target: int = 1200) -> list[str]:
        # Split into windows near target size, preferring sentence boundaries
        sentences = self._split_sentences_fast(text)
        if not sentences:
            return [text]
        windows: list[str] = []
        cur: list[str] = []
        size = 0
        for s in sentences:
            if size + len(s) + 1 > target and cur:
                windows.append(" ".join(cur))
                cur = []
                size = 0
            cur.append(s)
            size += len(s) + 1
        if cur:
            windows.append(" ".join(cur))
        return windows

    def _build_llm_prompt(self, text: str, profile: str = "generic") -> str:
        text = text.strip()
        if profile == "claimify":
            override = load_prompt("chunking", "proposition_claimify")
            if override:
                return f"{override}\n\nText:\n{text}\n\nClaims (JSON array):"
            return (
                "You extract high-quality factual claims. "
                "Given the input, identify atomic, verifiable claims expressed explicitly. "
                "Guidelines: (1) One claim per item; (2) Avoid pronouns - restate entities; (3) No speculation; "
                "(4) Use present or past tense as appropriate; (5) Be precise, concise, and faithful to the text; "
                "(6) Do not add information; (7) Normalize units and dates when explicit.\n\n"
                "Return ONLY a JSON array of strings with each string a single claim.\n\n"
                f"Text:\n{text}\n\nClaims (JSON array):"
            )
        elif profile == "gemma_aps":
            override = load_prompt("chunking", "proposition_gemma_aps")
            if override:
                return f"{override}\n\nText:\n{text}\n\nAtomic propositions (JSON array):"
            return (
                "Perform Atomic Proposition Simplification (APS). "
                "Decompose the text into minimal propositions that preserve entailment and meaning. "
                "Each proposition must be independently meaningful and unambiguous. "
                "Avoid overlapping content and avoid coreference by resolving pronouns. "
                "No paraphrasing beyond necessary normalization; do not hallucinate.\n\n"
                "Return ONLY a JSON array of strings with each string one atomic proposition.\n\n"
                f"Text:\n{text}\n\nAtomic propositions (JSON array):"
            )
        else:
            override = load_prompt("chunking", "proposition_generic")
            if override:
                return f"{override}\n\nText:\n{text}\n\nOutput JSON array:"
            return (
                "Extract the minimal factual propositions from the text below. "
                "Return ONLY a JSON array of strings (no commentary). "
                "Each string should be a single, self-contained claim without pronouns if possible.\n\n"
                f"Text:\n{text}\n\nOutput JSON array:"
            )

    def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            config = self.llm_config.copy()
            snapshot_kwargs = {}
            if 'app_config' in config:
                snapshot_kwargs['app_config'] = config['app_config']
            if 'credentials_resolved' in config:
                snapshot_kwargs['credentials_resolved'] = config['credentials_resolved']
            if 'provider_credentials' in config:
                snapshot_kwargs['provider_credentials'] = config['provider_credentials']
            result = self.llm_call_func(
                config.get('api_name', 'openai'),
                prompt,
                None,
                config.get('api_key'),
                config.get('system_message', 'You extract atomic factual propositions.'),
                config.get('temp', 0.2),
                False,
                False,
                False,
                model_override=config.get('model_override'),
                **snapshot_kwargs,
            )
            if result and isinstance(result, tuple) and len(result) > 0:
                return result[0]
            elif isinstance(result, str):
                return result
        except Exception as e:
            logger.error("Error calling LLM for propositions; error_type={}", type(e).__name__)
        return None

    def _parse_llm_props(self, output: Optional[str]) -> list[str]:
        if not output:
            return []
        output = output.strip()
        # Attempt JSON parse first
        try:
            import json
            data = json.loads(output)
            if isinstance(data, list):
                return [str(x).strip() for x in data if isinstance(x, (str, int, float)) and str(x).strip()]
        except Exception as e:
            logger.debug(
                "Proposition chunking JSON parse failed; falling back to line parsing; error_type={}",
                type(e).__name__,
            )
        # Fallback: parse lines starting with hyphen/number
        lines = [l.strip("- ") for l in output.splitlines() if l.strip()]
        return [l for l in lines if len(l) > 0]
    def _split_sentences_fast(self, text: str) -> list[str]:
        """Lightweight sentence splitter. Avoid heavy dependencies.
        Falls back to regex-based splitting similar to SentenceChunkingStrategy.
        """
        # Normalize whitespace a bit
        t = self._normalize_space(text)
        if not t:
            return []

        # Split on typical sentence terminators followed by space/newline
        # Keep the delimiter with the previous sentence when possible
        parts = re.split(r"(?<=[.!?])[\s\n]+", t)
        sentences = [p.strip() for p in parts if p and p.strip()]
        return sentences

    def _split_sentence_into_propositions(self, sentence: str, aggressiveness: int) -> list[str]:
        s = sentence.strip()
        if not s:
            return []

        # 0) Pre-split by strong punctuation: semicolons, em/en dashes, parentheses separations, colons
        # Retain punctuation where useful by splitting on the boundary and later normalizing spaces
        prelim = self._split_on_punct(s)

        props: list[str] = []
        for seg in prelim:
            seg = seg.strip()
            if not seg:
                continue

            # 1) Subordinate clause markers (aggressiveness >= 1)
            seg_parts = self._split_on_subordinate_markers(seg) if aggressiveness >= 1 else [seg]

            for part in seg_parts:
                part = part.strip()
                if not part:
                    continue
                # 2) Coordination split (aggressiveness >= 2) when both sides look clause-like
                if aggressiveness >= 2:
                    props.extend(self._split_on_coordination(part))
                else:
                    props.append(part)

        return [p for p in (pp.strip() for pp in props) if p]

    def _split_on_punct(self, s: str) -> list[str]:
        """Split a sentence into chunks around strong punctuation and parentheticals.

        Handles semicolons, em/en dashes, and optional colon boundaries, while
        treating longer parenthetical spans as separate propositions.
        """
        # Split on semicolons and em/en dashes; keep colon splits cautiously
        # Use regex that keeps delimiters by splitting on boundary while later trimming
        s = s.replace("—", " - ").replace("\u2013", " - ")
        # Split on ; or standalone dashes
        parts = re.split(r"\s*[;]+\s*|\s+[---]\s+", s)
        # Further split around parentheses content as its own proposition when long
        final = []
        for p in parts:
            # Extract parenthetical segments
            # Index-based scan avoids regex backtracking on untrusted input.
            start = 0
            i = p.find("(")
            while i != -1:
                pre = p[start:i].strip()
                if pre:
                    final.append(pre)
                j = p.find(")", i + 1)
                if j == -1:
                    tail = p[i:].strip()
                    if tail:
                        final.append(tail)
                    start = len(p)
                    break
                inside = p[i + 1:j].strip()
                if len(inside) > 10:  # treat longer parentheses as a proposition
                    final.append(inside)
                else:
                    # keep small parenthetical attached
                    if final:
                        final[-1] = (final[-1] + " (" + inside + ")").strip()
                    else:
                        final.append("(" + inside + ")")
                start = j + 1
                i = p.find("(", start)
            if start < len(p):
                tail = p[start:].strip()
                if tail:
                    final.append(tail)
        # Optionally split on colon if it looks like clause boundary
        colon_split: list[str] = []
        for p in final:
            # Split on colon only when the right side looks clause-like (has a verb-like token)
            colon_parts = re.split(r"\s*:\s*", p)
            if len(colon_parts) == 2 and self._looks_clause_like(colon_parts[1]):
                colon_split.extend([colon_parts[0], colon_parts[1]])
            else:
                colon_split.append(p)
        return colon_split

    def _split_on_subordinate_markers(self, s: str) -> list[str]:
        # Build a regex that splits on commas + marker or space + marker when mid-sentence
        # Keep the marker at the beginning of the following proposition for context
        parts = [s]
        for marker in sorted(self._SUBORDINATE_MARKERS, key=len, reverse=True):
            new_parts: list[str] = []
            pattern = rf"[,\s]+({re.escape(marker)})\b"
            for seg in parts:
                # Split but keep the marker in the next segment
                last = 0
                for m in re.finditer(pattern, seg, flags=re.IGNORECASE):
                    cut = seg[last:m.start()].strip()
                    if cut:
                        new_parts.append(cut)
                    # start next with the marker
                    last = m.start(1)
                tail = seg[last:].strip()
                if tail:
                    new_parts.append(tail)
            parts = new_parts if new_parts else parts
        return parts

    def _split_on_coordination(self, s: str) -> list[str]:
        tokens = s.split()
        if len(tokens) < 5:
            return [s]

        indices = []
        for i, tok in enumerate(tokens):
            low = tok.lower().strip(",;:.-")
            if low in self._COORD_CONJ:
                left = " ".join(tokens[:i]).strip()
                right = " ".join(tokens[i+1:]).strip()
                if self._looks_clause_like(left) and self._looks_clause_like(right):
                    indices.append(i)

        if not indices:
            return [s]

        # Split at selected coordinates
        parts: list[str] = []
        start = 0
        for idx in indices:
            left = " ".join(tokens[start:idx]).strip()
            if left:
                parts.append(left)
            start = idx + 1
        tail = " ".join(tokens[start:]).strip()
        if tail:
            parts.append(tail)
        return parts or [s]

    def _looks_clause_like(self, text: str) -> bool:
        # Very light heuristic: contains an auxiliary or a word that looks like a finite/gerund verb
        words = [w.strip("\"'()[]{}.,;:!?-") for w in text.split()]
        if not words:
            return False
        for w in words:
            lw = w.lower()
            if lw in self._AUXILIARIES:
                return True
            if len(lw) > 3 and (lw.endswith("ed") or lw.endswith("ing")):
                return True
        return False

    def _merge_short_propositions(self, props: list[str], min_len: int) -> list[str]:
        if not props:
            return []
        merged: list[str] = []
        for p in props:
            p = p.strip()
            if not merged:
                merged.append(p)
                continue
            if len(p) < min_len:
                merged[-1] = self._join_props([merged[-1], p])
            else:
                merged.append(p)
        return merged

    def _normalize_space(self, s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    def _join_props(self, props: list[str]) -> str:
        # Join with a single space, ensuring basic punctuation spacing
        out = " ".join(p.strip() for p in props if p and p.strip())
        out = re.sub(r"\s+([,.;:!?])", r"\1", out)
        return out.strip()

    # ------------------ span helpers for metadata ------------------
    def _split_sentences_with_spans(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into sentences with source spans using a lightweight regex."""
        if not text:
            return []
        spans: list[tuple[str, int, int]] = []
        # Split on sentence terminators followed by whitespace
        pattern = re.compile(r"(?<=[.!?])\s+")
        start = 0
        for m in pattern.finditer(text):
            end = m.start()
            segment = text[start:end]
            if segment and segment.strip():
                ltrim = len(segment) - len(segment.lstrip())
                rtrim = len(segment) - len(segment.rstrip())
                s_start = start + ltrim
                s_end = end - rtrim if rtrim else end
                if s_end > s_start:
                    spans.append((text[s_start:s_end], s_start, s_end))
            start = m.end()
        if start < len(text):
            segment = text[start:]
            if segment and segment.strip():
                ltrim = len(segment) - len(segment.lstrip())
                rtrim = len(segment) - len(segment.rstrip())
                s_start = start + ltrim
                s_end = len(text) - rtrim if rtrim else len(text)
                if s_end > s_start:
                    spans.append((text[s_start:s_end], s_start, s_end))
        return spans

    def _normalize_for_mapping(self, text: str) -> tuple[str, list[int]]:
        """Normalize text to proposition-splitting form while tracking index mapping.

        - Collapses whitespace runs to a single space.
        - Replaces em/en dashes with " - " (matching _split_on_punct behavior).
        Returns (normalized_text, mapping), where mapping maps normalized
        character indices back to original indices.
        """
        norm_chars: list[str] = []
        mapping: list[int] = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch in ("—", "\u2013"):
                for rep in (" ", "-", " "):
                    norm_chars.append(rep)
                    mapping.append(i)
                i += 1
                continue
            if ch.isspace():
                j = i
                while j < n and text[j].isspace():
                    j += 1
                if not norm_chars or norm_chars[-1] != " ":
                    norm_chars.append(" ")
                    mapping.append(i)
                i = j
                continue
            norm_chars.append(ch)
            mapping.append(i)
            i += 1

        # Trim leading/trailing spaces to mirror proposition normalization
        while norm_chars and norm_chars[0] == " ":
            norm_chars.pop(0)
            mapping.pop(0)
        while norm_chars and norm_chars[-1] == " ":
            norm_chars.pop()
            mapping.pop()
        return "".join(norm_chars), mapping

    def _proposition_spans_with_offsets(
        self,
        text: str,
        aggressiveness: int,
        min_prop_len: int,
    ) -> list[tuple[int, int]]:
        """Return proposition spans as (start, end) offsets in the original text."""
        spans: list[tuple[int, int]] = []
        sentences = self._split_sentences_with_spans(text)
        if not sentences:
            return spans

        for sentence_text, sent_start, _sent_end in sentences:
            norm_sent, mapping = self._normalize_for_mapping(sentence_text)
            if not norm_sent or not mapping:
                continue
            props = self._split_sentence_into_propositions(norm_sent, aggressiveness)
            if not props:
                continue

            # Map proposition strings back to source indices using normalized mapping
            norm_cursor = 0
            local_spans: list[tuple[int, int, int]] = []  # (start, end, prop_len)
            for prop in props:
                prop_clean = prop.strip()
                if not prop_clean:
                    continue
                idx = norm_sent.find(prop_clean, norm_cursor)
                if idx == -1:
                    idx = norm_cursor
                end_idx = min(len(norm_sent), idx + len(prop_clean))
                if not mapping:
                    continue
                start_map_idx = min(max(0, idx), len(mapping) - 1)
                end_map_idx = min(max(0, end_idx - 1), len(mapping) - 1)
                orig_start = sent_start + mapping[start_map_idx]
                orig_end = sent_start + mapping[end_map_idx] + 1
                if orig_end < orig_start:
                    orig_end = orig_start
                if orig_end > orig_start:
                    local_spans.append((orig_start, orig_end, len(prop_clean)))
                norm_cursor = end_idx

            if not local_spans:
                continue

            # Merge short propositions per sentence (same semantics as chunk())
            merged: list[tuple[int, int, int]] = []
            for s, e, plen in local_spans:
                if not merged:
                    merged.append((s, e, plen))
                    continue
                if plen < min_prop_len:
                    prev_s, prev_e, prev_len = merged[-1]
                    merged[-1] = (prev_s, max(prev_e, e), prev_len + plen)
                else:
                    merged.append((s, e, plen))

            spans.extend((s, e) for s, e, _plen in merged)

        return spans
