# rolling_summarize.py
"""
Rolling summarization chunking strategy.

This strategy creates chunks by progressively summarizing content using an LLM,
building a rolling context that maintains continuity across chunk boundaries.
"""

from typing import Any, Callable, Optional

from loguru import logger

from tldw_Server_API.app.core.Utils.prompt_loader import load_prompt

from ..base import BaseChunkingStrategy, ChunkMetadata, ChunkResult
from ..exceptions import ProcessingError

LLM_USAGE_TRACKER_KEY = "_provider_usage_tracker"
LLM_USAGE_SUCCEEDED_KEY = "provider_succeeded"


class RollingSummarizeStrategy(BaseChunkingStrategy):
    """
    Implements rolling summarization chunking.

    This strategy:
    1. Splits text into initial segments
    2. Summarizes each segment with rolling context
    3. Maintains continuity between chunks through overlapping summaries
    """

    def __init__(self,
                 language: str = 'en',
                 llm_call_func: Optional[Callable] = None,
                 llm_config: Optional[dict[str, Any]] = None):
        """
        Initialize rolling summarize strategy.

        Args:
            language: Language code for text processing
            llm_call_func: Function to call LLM for summarization
            llm_config: Configuration for LLM calls
        """
        super().__init__(language)
        self.llm_call_func = llm_call_func
        self.llm_config = llm_config or {}

    def chunk(self,
              text: str,
              max_size: int,
              overlap: int = 0,
              **options) -> list[str]:
        """
        Chunk text using rolling summarization.

        Args:
            text: Text to chunk
            max_size: Maximum size of each chunk (in sentences for initial split)
            overlap: Number of sentences to overlap (used for context)
            **options: Additional options:
                - summarization_detail: Float 0.0-1.0, how detailed summaries should be
                - preserve_structure: Whether to preserve document structure
                - context_window: Number of previous summaries to include as context

        Returns:
            List of summarized chunks
        """
        if not self.validate_parameters(text, max_size, overlap):
            return []

        # Get options
        summarization_detail = options.get('summarization_detail', 0.5)
        preserve_structure = options.get('preserve_structure', True)
        context_window = options.get('context_window', 2)

        segments_with_spans = self._build_segments_with_spans(text, max_size, overlap)
        if not segments_with_spans:
            return []
        segments = [segment for segment, _start, _end, _count in segments_with_spans]

        # Process segments with rolling summarization
        summarized_chunks = []
        rolling_context = []

        for i, segment in enumerate(segments):
            # Build context from previous summaries
            context = ""
            if rolling_context:
                # Use last N summaries as context
                context_items = rolling_context[-context_window:]
                context = "Previous context:\n" + "\n".join(context_items) + "\n\n"

            # Create prompt for summarization
            prompt = self._create_summarization_prompt(
                segment,
                context,
                summarization_detail,
                preserve_structure,
                i == 0  # First segment
            )

            summary = self._call_llm(prompt)
            summarized_chunks.append(summary)
            rolling_context.append(self._create_context_summary(summary))

        return summarized_chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be enhanced with pysbd
        import re

        # Handle common abbreviations
        text = re.sub(r'\b(Mr|Mrs|Dr|Ms|Prof|Sr|Jr)\.\s*', r'\1<DOT> ', text)
        text = re.sub(r'\b(Inc|Ltd|Corp|Co)\.\s*', r'\1<DOT> ', text)
        text = re.sub(r'\b(i\.e|e\.g|etc|vs|viz)\.\s*', r'\1<DOT> ', text)

        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        # Restore dots
        sentences = [s.replace('<DOT>', '.') for s in sentences]

        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _split_into_sentences_with_spans(self, text: str) -> list[tuple[str, int, int]]:
        """Split sentences and return spans via rolling forward search."""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
        spans: list[tuple[str, int, int]] = []
        pos = 0
        n = len(text)
        for s in sentences:
            idx = text.find(s, pos)
            if idx == -1:
                idx = pos
            end = min(idx + len(s), n)
            spans.append((s, idx, end))
            pos = end
        return spans

    def _build_segments_with_spans(
        self,
        text: str,
        max_size: int,
        overlap: int,
    ) -> list[tuple[str, int, int, int]]:
        """Group sentence spans into segments and carry their source spans."""
        sentences_with_spans = self._split_into_sentences_with_spans(text)
        if not sentences_with_spans:
            return []
        segments: list[tuple[str, int, int, int]] = []
        current: list[tuple[str, int, int]] = []
        for sent, start, end in sentences_with_spans:
            current.append((sent, start, end))
            if len(current) >= max_size:
                seg_text = " ".join(s for s, _s0, _e0 in current)
                seg_start = current[0][1]
                seg_end = current[-1][2]
                segments.append((seg_text, seg_start, seg_end, len(current)))
                current = current[-overlap:] if overlap > 0 else []
        if current:
            seg_text = " ".join(s for s, _s0, _e0 in current)
            seg_start = current[0][1]
            seg_end = current[-1][2]
            segments.append((seg_text, seg_start, seg_end, len(current)))
        return segments

    def _create_summarization_prompt(self,
                                    segment: str,
                                    context: str,
                                    detail_level: float,
                                    preserve_structure: bool,
                                    is_first: bool) -> str:
        """Create prompt for LLM summarization.

        If a custom instruction is defined in Prompts/chunking (key: 'Rolling Summarization'),
        it will be used as the base instruction.
        """

        # Determine target length based on detail level
        if detail_level < 0.3:
            length_instruction = "very brief summary (1-2 sentences)"
        elif detail_level < 0.6:
            length_instruction = "concise summary (3-4 sentences)"
        elif detail_level < 0.8:
            length_instruction = "detailed summary (5-7 sentences)"
        else:
            length_instruction = "comprehensive summary (8-10 sentences)"

        base_instruction = load_prompt("chunking", "Rolling Summarization") or ""
        # Build prompt
        if is_first:
            prompt = f"""{base_instruction}\nPlease provide a {length_instruction} of the following text.
Focus on the main points and key information."""
        else:
            prompt = f"""{base_instruction}\nContinue summarizing the document. Provide a {length_instruction} of the following text.
Maintain continuity with the previous context."""

        if context:
            prompt += f"\n\n{context}"

        if preserve_structure:
            prompt += "\nPreserve any important structural elements (headings, lists, etc.) in the summary."

        prompt += f"\n\nText to summarize:\n{segment}\n\nSummary:"

        return prompt

    def _create_context_summary(self, summary: str) -> str:
        """Create a brief context summary for rolling context."""
        # Take first 150 characters or first sentence for context
        if len(summary) <= 150:
            return summary

        # Try to break at sentence boundary
        first_period = summary.find('. ', 0, 150)
        if first_period > 0:
            return summary[:first_period + 1]

        return summary[:150] + "..."

    def _call_llm(self, prompt: str) -> str:
        """Call LLM for summarization."""
        if not self.llm_call_func:
            raise ProcessingError(
                "Rolling summarization provider is unavailable.",
                stage="summarization",
                operation="provider_call",
            )

        provider_failed = False
        result: Any = None
        try:
            # Prepare config for LLM call
            config = self.llm_config.copy()
            snapshot_kwargs = {}
            if 'app_config' in config:
                snapshot_kwargs['app_config'] = config['app_config']
            if 'credentials_resolved' in config:
                snapshot_kwargs['credentials_resolved'] = config['credentials_resolved']
            if 'provider_credentials' in config:
                snapshot_kwargs['provider_credentials'] = config['provider_credentials']
            if config.get('model'):
                snapshot_kwargs['model_override'] = config['model']

            # Use the provided LLM function
            # The analyze function signature: analyze(api_name, input_data, custom_prompt_arg, api_key, system_message, temp, ...)
            result = self.llm_call_func(
                config.get('api_name', 'openai'),  # api_name
                prompt,  # input_data
                None,  # custom_prompt_arg (use None since prompt already contains instructions)
                config.get('api_key'),  # api_key
                config.get('system_message', "You are a helpful assistant that creates concise, accurate summaries."),  # system_message
                config.get('temp', 0.3),  # temp
                False,  # streaming
                False,  # recursive_summarization
                False,  # chunked_summarization
                None,  # chunk_options
                **snapshot_kwargs,
            )
        except Exception:
            logger.error("Rolling summarization provider call failed")
            provider_failed = True

        if provider_failed:
            raise ProcessingError(
                "Rolling summarization provider call failed.",
                stage="summarization",
                operation="provider_call",
            )

        summary = result[0] if isinstance(result, tuple) and result else result
        if (
            not isinstance(summary, str)
            or not summary.strip()
            or summary.lstrip().casefold().startswith("error:")
        ):
            logger.warning("Rolling summarization provider returned an invalid response")
            raise ProcessingError(
                "Rolling summarization provider returned an invalid response.",
                stage="summarization",
                operation="provider_response",
            )

        tracker = self.llm_config.get(LLM_USAGE_TRACKER_KEY)
        if isinstance(tracker, dict):
            tracker[LLM_USAGE_SUCCEEDED_KEY] = True
        return summary

    def chunk_with_metadata(self,
                            text: str,
                            max_size: int,
                            overlap: int = 0,
                            **options) -> list[ChunkResult]:
        """Chunk text and return metadata mapping summaries to source spans."""
        if not self.validate_parameters(text, max_size, overlap):
            return []

        summarization_detail = options.get('summarization_detail', 0.5)
        preserve_structure = options.get('preserve_structure', True)
        context_window = options.get('context_window', 2)

        segments_with_spans = self._build_segments_with_spans(text, max_size, overlap)
        if not segments_with_spans:
            return []

        results: list[ChunkResult] = []
        rolling_context: list[str] = []
        total = len(segments_with_spans)

        for i, (segment, seg_start, seg_end, sentence_count) in enumerate(segments_with_spans):
            context = ""
            if rolling_context:
                context_items = rolling_context[-context_window:]
                context = "Previous context:\n" + "\n".join(context_items) + "\n\n"

            prompt = self._create_summarization_prompt(
                segment,
                context,
                summarization_detail,
                preserve_structure,
                i == 0,
            )
            summary = self._call_llm(prompt)
            rolling_context.append(self._create_context_summary(summary))

            metadata = ChunkMetadata(
                index=i,
                start_char=int(seg_start),
                end_char=int(seg_end),
                word_count=len(summary.split()) if summary else 0,
                sentence_count=sentence_count,
                language=self.language,
                overlap_with_previous=overlap if i > 0 else 0,
                overlap_with_next=overlap if i < total - 1 else 0,
                method='rolling_summarize',
                options={
                    'summarization_detail': summarization_detail,
                    'preserve_structure': preserve_structure,
                    'context_window': context_window,
                    'source_span': (int(seg_start), int(seg_end)),
                },
            )
            results.append(ChunkResult(text=summary, metadata=metadata))

        return results
