"""
Document processing strategies for the RAG service.

This module handles ranking, deduplication, and context building
optimized for a single-user TUI application.
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import time

from loguru import logger
import numpy as np

from .types import (
    ProcessingStrategy, Document, SearchResult, RAGContext,
    ProcessingError, DataSource
)
from .utils import (
    TokenCounter, deduplicate_documents, combine_scores,
    normalize_scores, format_sources
)

# Try to import FlashRank
try:
    from flashrank import Ranker
    FLASHRANK_AVAILABLE = True
except ImportError:
    logger.warning("FlashRank not available. Using fallback ranking.")
    FLASHRANK_AVAILABLE = False


class BaseProcessor(ProcessingStrategy):
    """
    Base document processor with common functionality.
    
    For single-user app, we can:
    - Keep ranker instance in memory
    - Use simpler deduplication strategies
    - Optimize for quality over throughput
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._token_counter = TokenCounter()
        
        # Initialize reranker if available and enabled
        self._ranker = None
        if self.config.get("enable_reranking", True) and FLASHRANK_AVAILABLE:
            try:
                self._ranker = Ranker()
                logger.info("FlashRank reranker initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize FlashRank: {e}")
    
    def process(
        self,
        search_results: List[SearchResult],
        query: str,
        max_context_length: int = 4096
    ) -> RAGContext:
        """Process search results into a context for generation."""
        raise NotImplementedError


class DefaultProcessor(BaseProcessor):
    """
    Default document processor with ranking and deduplication.
    
    Optimized for single-user quality over speed.
    """
    
    def process(
        self,
        search_results: List[SearchResult],
        query: str,
        max_context_length: int = 4096
    ) -> RAGContext:
        """
        Process search results into a context for generation.
        
        Steps:
        1. Combine documents from all search results
        2. Deduplicate similar documents
        3. Rerank if enabled
        4. Build context within token limit
        """
        try:
            start_time = time.time()
            
            # Step 1: Combine all documents
            all_documents = []
            source_counts = defaultdict(int)
            
            for result in search_results:
                all_documents.extend(result.documents)
                for doc in result.documents:
                    source_counts[doc.source.name] += 1
            
            logger.debug(f"Combined {len(all_documents)} documents from {len(search_results)} search results")
            logger.debug(f"Source distribution: {dict(source_counts)}")
            
            if not all_documents:
                return RAGContext(
                    documents=[],
                    combined_text="",
                    total_tokens=0,
                    metadata={"processing_time": time.time() - start_time}
                )
            
            # Step 2: Deduplicate
            dedup_start = time.time()
            unique_docs = self._deduplicate_documents(
                all_documents,
                threshold=self.config.get("deduplication_threshold", 0.85)
            )
            logger.debug(f"Deduplication: {len(all_documents)} -> {len(unique_docs)} docs in {time.time() - dedup_start:.2f}s")
            
            # Step 3: Rerank if enabled
            if self._ranker and len(unique_docs) > 1:
                rerank_start = time.time()
                ranked_docs = self._rerank_documents(unique_docs, query)
                logger.debug(f"Reranked {len(ranked_docs)} documents in {time.time() - rerank_start:.2f}s")
            else:
                # Simple scoring if no reranker
                ranked_docs = sorted(unique_docs, key=lambda d: d.score, reverse=True)
            
            # Step 4: Build context within token limit
            context_docs, combined_text, total_tokens = self._build_context(
                ranked_docs,
                query,
                max_context_length
            )
            
            # Create context
            context = RAGContext(
                documents=context_docs,
                combined_text=combined_text,
                total_tokens=total_tokens,
                metadata={
                    "processing_time": time.time() - start_time,
                    "total_documents": len(all_documents),
                    "unique_documents": len(unique_docs),
                    "context_documents": len(context_docs),
                    "source_distribution": dict(source_counts)
                }
            )
            
            logger.info(
                f"Processed context: {len(context_docs)} docs, "
                f"{total_tokens} tokens in {time.time() - start_time:.2f}s"
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise ProcessingError(f"Failed to process documents: {e}")
    
    def _deduplicate_documents(
        self,
        documents: List[Document],
        threshold: float = 0.85
    ) -> List[Document]:
        """
        Deduplicate documents based on content similarity.
        
        For single-user app, we can afford more sophisticated deduplication.
        """
        if len(documents) <= 1:
            return documents
        
        # Group by source for smarter deduplication
        by_source = defaultdict(list)
        for doc in documents:
            by_source[doc.source].append(doc)
        
        # Deduplicate within each source first
        deduped_by_source = {}
        for source, docs in by_source.items():
            deduped_by_source[source] = deduplicate_documents(
                docs,
                key_func=lambda d: d.content,
                similarity_threshold=threshold
            )
        
        # Then deduplicate across sources
        all_deduped = []
        for source_docs in deduped_by_source.values():
            all_deduped.extend(source_docs)
        
        # Final cross-source deduplication with lower threshold
        final_deduped = deduplicate_documents(
            all_deduped,
            key_func=lambda d: d.content,
            similarity_threshold=threshold * 0.9  # Slightly lower for cross-source
        )
        
        return final_deduped
    
    def _rerank_documents(
        self,
        documents: List[Document],
        query: str
    ) -> List[Document]:
        """
        Rerank documents using FlashRank.
        
        For single-user app, we can rerank all documents for best quality.
        """
        if not self._ranker or not documents:
            return documents
        
        try:
            # Prepare passages for FlashRank
            passages = [
                {"text": doc.content, "meta": {"doc_id": doc.id}}
                for doc in documents
            ]
            
            # Rerank
            reranked = self._ranker.rerank(query, passages)
            
            # Map scores back to documents
            doc_map = {doc.id: doc for doc in documents}
            ranked_docs = []
            
            for result in reranked:
                doc_id = result["meta"]["doc_id"]
                if doc_id in doc_map:
                    doc = doc_map[doc_id]
                    doc.score = result["score"]
                    ranked_docs.append(doc)
            
            # Add any documents that weren't reranked (shouldn't happen)
            ranked_ids = {doc.id for doc in ranked_docs}
            for doc in documents:
                if doc.id not in ranked_ids:
                    ranked_docs.append(doc)
            
            return ranked_docs
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Using original order.")
            return documents
    
    def _build_context(
        self,
        documents: List[Document],
        query: str,
        max_tokens: int
    ) -> Tuple[List[Document], str, int]:
        """
        Build context from documents within token limit.
        
        For single-user app, we can be more careful about context building.
        """
        context_docs = []
        context_parts = []
        total_tokens = 0
        
        # Reserve tokens for query and formatting
        query_tokens = self._token_counter.count(query)
        reserved_tokens = query_tokens + 50  # Extra for formatting
        available_tokens = max_tokens - reserved_tokens
        
        # Add documents until we hit the limit
        for doc in documents:
            doc_tokens = self._token_counter.count(doc.content)
            
            if total_tokens + doc_tokens <= available_tokens:
                context_docs.append(doc)
                
                # Format document for context
                source_info = self._format_document_source(doc)
                context_parts.append(f"[{source_info}]\n{doc.content}")
                
                total_tokens += doc_tokens
            else:
                # Try to fit partial document
                remaining_tokens = available_tokens - total_tokens
                if remaining_tokens > 100:  # Only if we have reasonable space
                    truncated_content = self._token_counter.truncate(
                        doc.content,
                        remaining_tokens - 20  # Leave room for source info
                    )
                    
                    doc_truncated = Document(
                        id=doc.id,
                        content=truncated_content + "...",
                        metadata={**doc.metadata, "truncated": True},
                        source=doc.source,
                        score=doc.score
                    )
                    
                    context_docs.append(doc_truncated)
                    source_info = self._format_document_source(doc_truncated)
                    context_parts.append(f"[{source_info}]\n{truncated_content}...")
                    
                    total_tokens = available_tokens
                break
        
        # Combine context
        combined_text = "\n\n".join(context_parts)
        
        return context_docs, combined_text, total_tokens + reserved_tokens
    
    def _format_document_source(self, doc: Document) -> str:
        """Format source information for a document."""
        source_parts = [doc.source.name]
        
        metadata = doc.metadata
        if "title" in metadata:
            source_parts.append(metadata["title"])
        elif "conversation_title" in metadata:
            source_parts.append(metadata["conversation_title"])
        
        if "timestamp" in metadata:
            source_parts.append(metadata["timestamp"])
        
        return " | ".join(source_parts)


class AdvancedProcessor(DefaultProcessor):
    """
    Advanced processor with additional features.
    
    Since this is a single-user app, we can enable all quality improvements.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Additional configuration
        self.combination_method = config.get("combination_method", "weighted")
        self.enable_snippets = config.get("enable_snippets", True)
        self.snippet_length = config.get("snippet_length", 200)
    
    def process(
        self,
        search_results: List[SearchResult],
        query: str,
        max_context_length: int = 4096
    ) -> RAGContext:
        """Enhanced processing with additional features."""
        # Get base processing
        context = super().process(search_results, query, max_context_length)
        
        # Add enhancements
        if self.enable_snippets:
            self._add_relevant_snippets(context, query)
        
        # Add source diversity score
        context.metadata["source_diversity"] = self._calculate_source_diversity(context.documents)
        
        return context
    
    def _add_relevant_snippets(self, context: RAGContext, query: str):
        """
        Extract and highlight most relevant snippets from documents.
        
        For single-user app, we can spend time finding the best snippets.
        """
        query_terms = set(query.lower().split())
        
        for doc in context.documents:
            # Find sentences with query terms
            sentences = doc.content.split(". ")
            relevant_sentences = []
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                overlap = sum(1 for term in query_terms if term in sentence_lower)
                if overlap > 0:
                    relevant_sentences.append((sentence, overlap))
            
            # Sort by relevance and take top snippets
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            
            if relevant_sentences:
                top_snippets = [s[0] for s in relevant_sentences[:3]]
                doc.metadata["snippets"] = top_snippets
    
    def _calculate_source_diversity(self, documents: List[Document]) -> float:
        """Calculate how diverse the sources are."""
        if not documents:
            return 0.0
        
        source_counts = defaultdict(int)
        for doc in documents:
            source_counts[doc.source] += 1
        
        # Simple entropy calculation
        total = len(documents)
        entropy = 0.0
        
        for count in source_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(DataSource))
        diversity = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return diversity


class StreamingProcessor(BaseProcessor):
    """
    Processor optimized for streaming responses.
    
    For single-user TUI, this allows showing results progressively.
    """
    
    async def process_streaming(
        self,
        search_results: List[SearchResult],
        query: str,
        max_context_length: int = 4096
    ):
        """
        Process documents in a streaming fashion.
        
        Yields partial contexts as documents are processed.
        """
        all_documents = []
        for result in search_results:
            all_documents.extend(result.documents)
        
        if not all_documents:
            yield RAGContext(documents=[], combined_text="", total_tokens=0, metadata={})
            return
        
        # Sort by relevance
        sorted_docs = sorted(all_documents, key=lambda d: d.score, reverse=True)
        
        # Stream context building
        context_docs = []
        context_parts = []
        total_tokens = 0
        
        query_tokens = self._token_counter.count(query)
        available_tokens = max_context_length - query_tokens - 50
        
        for i, doc in enumerate(sorted_docs):
            doc_tokens = self._token_counter.count(doc.content)
            
            if total_tokens + doc_tokens <= available_tokens:
                context_docs.append(doc)
                source_info = f"{doc.source.name} | {doc.metadata.get('title', 'Untitled')}"
                context_parts.append(f"[{source_info}]\n{doc.content}")
                total_tokens += doc_tokens
                
                # Yield intermediate context every few documents
                if i % 3 == 0 or i == len(sorted_docs) - 1:
                    yield RAGContext(
                        documents=context_docs.copy(),
                        combined_text="\n\n".join(context_parts),
                        total_tokens=total_tokens + query_tokens + 50,
                        metadata={"partial": i < len(sorted_docs) - 1}
                    )
            else:
                break
        
        # Final context
        yield RAGContext(
            documents=context_docs,
            combined_text="\n\n".join(context_parts),
            total_tokens=total_tokens + query_tokens + 50,
            metadata={"partial": False, "final": True}
        )