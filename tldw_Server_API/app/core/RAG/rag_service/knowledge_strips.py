"""
Knowledge Strips for Self-Correcting RAG

This module partitions documents into semantic units (strips) and grades each
strip for relevance. Only relevant strips are passed to generation, improving
answer quality and reducing noise.

Part of the Self-Correcting RAG feature set (Stage 4).
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from loguru import logger

from tldw_Server_API.app.core.LLM_Calls.structured_output import (
    StructuredOutputOptions,
    parse_structured_output,
)

from .types import DataSource, Document


@dataclass
class KnowledgeStrip:
    """A semantic unit extracted from a document."""

    doc_id: str
    strip_id: str
    text: str
    start_offset: int
    end_offset: int
    relevance_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeStripsResult:
    """Result of processing documents into knowledge strips."""

    strips: list[KnowledgeStrip]
    documents: list[Document]  # Documents rebuilt from strips
    total_strips: int
    relevant_strips: int
    filtered_strips: int
    avg_relevance: float
    processing_time_ms: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _estimate_tokens(text: str) -> int:
    """Rough token count estimation (words * 1.3)."""
    if not text:
        return 0
    # Simple heuristic: approximately 1.3 tokens per word
    words = len(text.split())
    return int(words * 1.3)


def _split_into_strips(
    text: str,
    strip_size_tokens: int = 100,
) -> list[dict[str, Any]]:
    """
    Split text into semantic strips.

    Uses sentence boundaries to create strips of approximately strip_size_tokens.

    Args:
        text: The text to split
        strip_size_tokens: Target size for each strip in tokens

    Returns:
        List of strip dictionaries with text, start_offset, end_offset
    """
    if not text:
        return []

    import re

    # Split into sentences
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_pattern, text)

    strips: list[dict[str, Any]] = []
    current_strip: list[str] = []
    current_tokens = 0
    current_start = 0

    char_offset = 0

    for sentence in sentences:
        sentence_tokens = _estimate_tokens(sentence)

        # If adding this sentence would exceed the limit and we have content,
        # finalize the current strip
        if current_tokens + sentence_tokens > strip_size_tokens and current_strip:
            strip_text = ' '.join(current_strip)
            strips.append({
                "text": strip_text,
                "start_offset": current_start,
                "end_offset": char_offset,
            })
            current_strip = []
            current_tokens = 0
            current_start = char_offset

        # Add sentence to current strip
        current_strip.append(sentence.strip())
        current_tokens += sentence_tokens
        char_offset += len(sentence) + 1  # +1 for space

    # Don't forget the last strip
    if current_strip:
        strip_text = ' '.join(current_strip)
        strips.append({
            "text": strip_text,
            "start_offset": current_start,
            "end_offset": char_offset,
        })

    return strips


def _score_strip_relevance(
    query: str,
    strip_text: str,
) -> float:
    """
    Score strip relevance using simple keyword matching.

    This is a fast heuristic for when LLM grading is not available.

    Args:
        query: The search query
        strip_text: The strip text

    Returns:
        Relevance score between 0.0 and 1.0
    """
    import re

    # Extract query keywords
    query_words = {
        w.lower() for w in re.findall(r'\b\w+\b', query)
        if len(w) > 2
    }

    # Count keyword matches in strip
    strip_lower = strip_text.lower()
    matches = sum(1 for w in query_words if w in strip_lower)

    if not query_words:
        return 0.5  # Default for empty query

    # Normalize by query word count
    score = min(1.0, matches / len(query_words))

    return score


class KnowledgeStripsProcessor:
    """
    Processes documents into knowledge strips and grades each for relevance.

    Knowledge strips are fine-grained semantic units that are individually
    graded, allowing for more precise filtering than document-level grading.
    """

    def __init__(
        self,
        strip_size_tokens: int = 100,
        min_relevance_score: float = 0.3,
        analyze_fn: Optional[Callable] = None,
    ):
        """
        Initialize the knowledge strips processor.

        Args:
            strip_size_tokens: Target size for each strip in tokens
            min_relevance_score: Minimum relevance score to keep a strip
            analyze_fn: Optional LLM function for grading (uses heuristic if None)
        """
        self.strip_size_tokens = strip_size_tokens
        self.min_relevance_score = min_relevance_score
        self._analyze = analyze_fn

    async def process(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 20,
    ) -> KnowledgeStripsResult:
        """
        Process documents into knowledge strips and filter by relevance.

        Args:
            query: The search query
            documents: Documents to process
            top_k: Maximum number of strips to return

        Returns:
            KnowledgeStripsResult with filtered strips and rebuilt documents
        """
        start_time = time.time()

        if not documents:
            return KnowledgeStripsResult(
                strips=[],
                documents=[],
                total_strips=0,
                relevant_strips=0,
                filtered_strips=0,
                avg_relevance=0.0,
                processing_time_ms=0,
            )

        # Step 1: Extract strips from all documents
        all_strips: list[KnowledgeStrip] = []

        for doc in documents:
            doc_id = getattr(doc, "id", str(uuid.uuid4().hex[:8]))
            content = getattr(doc, "content", "")

            raw_strips = _split_into_strips(content, self.strip_size_tokens)

            for idx, raw_strip in enumerate(raw_strips):
                strip = KnowledgeStrip(
                    doc_id=doc_id,
                    strip_id=f"{doc_id}_strip_{idx}",
                    text=raw_strip["text"],
                    start_offset=raw_strip["start_offset"],
                    end_offset=raw_strip["end_offset"],
                    relevance_score=0.0,
                    metadata={
                        "doc_source": str(getattr(doc, "source", DataSource.MEDIA_DB)),
                        "doc_score": getattr(doc, "score", 0.0),
                        "strip_index": idx,
                    },
                )
                all_strips.append(strip)

        total_strips = len(all_strips)

        if not all_strips:
            return KnowledgeStripsResult(
                strips=[],
                documents=[],
                total_strips=0,
                relevant_strips=0,
                filtered_strips=0,
                avg_relevance=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000),
            )

        # Step 2: Score each strip for relevance
        if self._analyze:
            # Use LLM-based scoring
            all_strips = await self._score_strips_llm(query, all_strips)
        else:
            # Use heuristic scoring
            all_strips = self._score_strips_heuristic(query, all_strips)

        # Step 3: Filter by minimum relevance
        relevant_strips = [
            s for s in all_strips
            if s.relevance_score >= self.min_relevance_score
        ]

        # Sort by relevance and take top_k
        relevant_strips.sort(key=lambda s: s.relevance_score, reverse=True)
        filtered_strips = relevant_strips[:top_k]

        # Step 4: Rebuild documents from filtered strips
        rebuilt_docs = self._rebuild_documents(filtered_strips, documents)

        # Compute metrics
        avg_relevance = (
            sum(s.relevance_score for s in filtered_strips) / len(filtered_strips)
            if filtered_strips else 0.0
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        return KnowledgeStripsResult(
            strips=filtered_strips,
            documents=rebuilt_docs,
            total_strips=total_strips,
            relevant_strips=len(relevant_strips),
            filtered_strips=len(filtered_strips),
            avg_relevance=avg_relevance,
            processing_time_ms=processing_time_ms,
            metadata={
                "strip_size_tokens": self.strip_size_tokens,
                "min_relevance_score": self.min_relevance_score,
                "top_k": top_k,
            },
        )

    def _score_strips_heuristic(
        self,
        query: str,
        strips: list[KnowledgeStrip],
    ) -> list[KnowledgeStrip]:
        """Score strips using keyword matching heuristic."""
        for strip in strips:
            strip.relevance_score = _score_strip_relevance(query, strip.text)
        return strips

    async def _score_strips_llm(
        self,
        query: str,
        strips: list[KnowledgeStrip],
    ) -> list[KnowledgeStrip]:
        """
        Score strips using LLM-based grading.

        Uses batch processing for efficiency.
        """
        analyze_fn = self._analyze
        if analyze_fn is None:
            return self._score_strips_heuristic(query, strips)

        # Process in small batches
        batch_size = 5

        for i in range(0, len(strips), batch_size):
            batch = strips[i:i + batch_size]

            # Build batch prompt
            strips_text = "\n".join([
                f"[Strip {idx + 1}]: {s.text[:500]}"
                for idx, s in enumerate(batch)
            ])

            prompt = f"""Rate the relevance of each text strip to the query on a scale of 0.0 to 1.0.

Query: {query}

Strips:
{strips_text}

For each strip, provide a JSON object with:
- "strip_num": the strip number (1-{len(batch)})
- "relevance": score from 0.0 to 1.0

Respond with a JSON array of objects.
JSON:"""

            try:
                raw_response = await asyncio.to_thread(
                    analyze_fn,
                    "openai",
                    "",
                    prompt,
                    None,
                    "You are a relevance grader. Output valid JSON only.",
                    0.1,
                )

                # Parse response
                scores_payload = parse_structured_output(
                    str(raw_response),
                    options=StructuredOutputOptions(parse_mode="lenient", strip_think_tags=True),
                )
                score_items: list[Any] = []
                if isinstance(scores_payload, list):
                    score_items = scores_payload
                elif isinstance(scores_payload, dict):
                    wrapped_scores = scores_payload.get("scores")
                    if isinstance(wrapped_scores, list):
                        score_items = wrapped_scores
                for score_obj in score_items:
                    if not isinstance(score_obj, dict):
                        continue
                    strip_num = score_obj.get("strip_num", 0) - 1
                    relevance = float(score_obj.get("relevance", 0.0))
                    if 0 <= strip_num < len(batch):
                        batch[strip_num].relevance_score = min(1.0, max(0.0, relevance))

            except Exception:
                logger.warning("LLM strip scoring failed, falling back to heuristic")
                # Fall back to heuristic for this batch
                for strip in batch:
                    strip.relevance_score = _score_strip_relevance(query, strip.text)

        return strips

    def _rebuild_documents(
        self,
        strips: list[KnowledgeStrip],
        original_docs: list[Document],
    ) -> list[Document]:
        """
        Rebuild documents from filtered strips.

        Groups strips by original document and creates new Document objects
        containing only the relevant strips.
        """
        from collections import defaultdict

        # Group strips by doc_id
        doc_strips: dict[str, list[KnowledgeStrip]] = defaultdict(list)
        for strip in strips:
            doc_strips[strip.doc_id].append(strip)

        # Create a lookup for original documents
        doc_lookup = {getattr(d, "id", None): d for d in original_docs}

        rebuilt_docs = []

        for doc_id, doc_strip_list in doc_strips.items():
            # Sort strips by offset to maintain order
            doc_strip_list.sort(key=lambda s: s.start_offset)

            # Combine strip texts
            combined_content = "\n\n".join([s.text for s in doc_strip_list])

            # Get original document metadata
            original_doc = doc_lookup.get(doc_id)
            original_metadata = getattr(original_doc, "metadata", {}) if original_doc else {}
            original_source = getattr(original_doc, "source", DataSource.MEDIA_DB) if original_doc else DataSource.MEDIA_DB

            # Calculate combined score (average of strip scores)
            avg_score = sum(s.relevance_score for s in doc_strip_list) / len(doc_strip_list)

            # Create new document with filtered content
            rebuilt_doc = Document(
                id=f"{doc_id}_filtered",
                content=combined_content,
                source=original_source,
                score=avg_score,
                metadata={
                    **original_metadata,
                    "original_doc_id": doc_id,
                    "strip_count": len(doc_strip_list),
                    "strip_ids": [s.strip_id for s in doc_strip_list],
                    "filtered_from_knowledge_strips": True,
                },
            )
            rebuilt_docs.append(rebuilt_doc)

        # Sort by score
        rebuilt_docs.sort(key=lambda d: d.score, reverse=True)

        return rebuilt_docs


# Convenience function for pipeline integration
async def process_knowledge_strips(
    query: str,
    documents: list[Document],
    strip_size_tokens: int = 100,
    min_relevance: float = 0.3,
    max_strips: int = 20,
    use_llm_grading: bool = False,
) -> tuple[list[Document], dict[str, Any]]:
    """
    Convenience function to process documents into knowledge strips.

    Args:
        query: The search query
        documents: Documents to process
        strip_size_tokens: Target size for each strip
        min_relevance: Minimum relevance score to keep
        max_strips: Maximum strips to return
        use_llm_grading: Whether to use LLM for grading (default: heuristic)

    Returns:
        Tuple of (filtered_documents, metadata)
    """
    analyze_fn = None
    if use_llm_grading:
        try:
            import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

            def _analyze(*args, **kwargs):
                return sgl.analyze(*args, **kwargs)

            analyze_fn = _analyze
        except ImportError:
            logger.warning("LLM module not available for strip grading")

    processor = KnowledgeStripsProcessor(
        strip_size_tokens=strip_size_tokens,
        min_relevance_score=min_relevance,
        analyze_fn=analyze_fn,
    )

    result = await processor.process(query, documents, top_k=max_strips)

    metadata = {
        "knowledge_strips_enabled": True,
        "total_strips": result.total_strips,
        "relevant_strips": result.relevant_strips,
        "filtered_strips": result.filtered_strips,
        "avg_relevance": result.avg_relevance,
        "processing_time_ms": result.processing_time_ms,
        "strip_size_tokens": strip_size_tokens,
        "min_relevance": min_relevance,
    }

    return result.documents, metadata
