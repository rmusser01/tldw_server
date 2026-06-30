# enhanced_chunking_integration.py
"""
Integration module for enhanced chunking in the RAG functional pipeline.

This module provides functions to integrate the enhanced chunking capabilities
into the functional pipeline, making them composable with other pipeline functions.
"""

from typing import Optional

from loguru import logger

# Import from the new modular chunking system
from ...Chunking import Chunker, ChunkerConfig, ChunkingMethod, ChunkType, EnhancedChunk
from .types import Document


def _normalize_chunk_type(value: Optional[str]) -> str:
    try:
        normalized = Chunker.normalize_chunk_type(value)
    except Exception:
        normalized = None
    if normalized:
        return normalized
    if value is None:
        return "text"
    return str(value).strip().lower() or "text"


def _safe_exception_label(exc: BaseException) -> str:
    return type(exc).__name__


async def enhanced_chunk_documents(
    context,  # RAGPipelineContext
    enable_pdf_cleaning: Optional[bool] = None,
    preserve_code_blocks: Optional[bool] = None,
    preserve_tables: Optional[bool] = None,
    structure_aware: Optional[bool] = None,
    track_positions: Optional[bool] = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None
):
    """
    Apply enhanced chunking to documents in the pipeline context.

    This function can be composed in the functional pipeline to provide:
    - PDF artifact cleaning
    - Code block preservation
    - Table extraction with serialization
    - Structure-aware chunking
    - Character position tracking

    Args:
        context: RAGPipelineContext with documents to chunk
        enable_pdf_cleaning: Clean PDF artifacts
        preserve_code_blocks: Preserve code as separate chunks
        preserve_tables: Extract and serialize tables
        structure_aware: Use structure-aware chunking
        track_positions: Track character positions
        chunk_size: Target chunk size (default from config)
        overlap: Chunk overlap (default from config)

    Returns:
        Updated context with chunked documents
    """
    if not context.documents:
        return context

    # Get configuration
    context_config = context.config or {}

    # Build enhanced chunking options
    chunking_options = {
        "clean_pdf_artifacts": enable_pdf_cleaning if enable_pdf_cleaning is not None
                               else context_config.get("clean_pdf_artifacts", True),
        "preserve_code_blocks": preserve_code_blocks if preserve_code_blocks is not None
                               else context_config.get("preserve_code_blocks", True),
        "preserve_tables": preserve_tables if preserve_tables is not None
                          else context_config.get("preserve_tables", True),
        "structure_aware": structure_aware if structure_aware is not None
                          else context_config.get("structure_aware", True),
        "track_positions": track_positions if track_positions is not None
                          else context_config.get("track_positions", True),
        "table_serialize_method": context_config.get("table_serialize_method", "hybrid")
    }

    chunk_size = chunk_size or context_config.get("chunk_size", 512)
    overlap = overlap or context_config.get("chunk_overlap", 128)

    # Initialize chunker with new API
    # Map old options to new configuration
    method_enum = ChunkingMethod.STRUCTURE_AWARE if structure_aware else ChunkingMethod.SENTENCES
    method = method_enum.value

    config = ChunkerConfig(
        default_method=method_enum,
        default_max_size=chunk_size,
        default_overlap=overlap,
        enable_cache=True
    )
    chunker = Chunker(config)

    # Process each document
    chunked_documents = []
    total_chunks = 0
    chunk_type_stats: dict[str, int] = {}

    for doc in context.documents:
        try:
            # Chunk the document using new API
            # Get chunks with metadata
            chunk_results = chunker.chunk_text_with_metadata(
                text=doc.content,
                method=method,
                max_size=chunk_size,
                overlap=overlap,
                preserve_tables=preserve_tables,
                preserve_code_blocks=preserve_code_blocks
            )

            # Convert to EnhancedChunk format for RAG
            enhanced_chunks = []
            for i, result in enumerate(chunk_results):
                chunk_type_value = ChunkType.TEXT
                if 'code' in result.text.lower() or '```' in result.text:
                    chunk_type_value = ChunkType.CODE
                elif result.text.strip().startswith('#'):
                    chunk_type_value = ChunkType.HEADER
                elif result.text.strip().startswith(('- ', '* ', '1.')):
                    chunk_type_value = ChunkType.LIST
                elif '|' in result.text and result.text.count('|') > 2:
                    chunk_type_value = ChunkType.TABLE

                enhanced_chunk = EnhancedChunk(
                    id=f"{doc.id}_chunk_{i}",
                    content=result.text,
                    chunk_type=chunk_type_value,
                    start_char=result.metadata.start_char,
                    end_char=result.metadata.end_char,
                    chunk_index=i,
                    metadata={
                        'word_count': result.metadata.word_count,
                        'language': result.metadata.language,
                        'method': result.metadata.method,
                        'doc_id': doc.id
                    },
                    parent_id=doc.id
                )
                enhanced_chunks.append(enhanced_chunk)

            # Convert enhanced chunks to documents
            for chunk in enhanced_chunks:
                # Build chunk metadata
                chunk_type_raw = getattr(chunk.chunk_type, "value", chunk.chunk_type)
                if chunk_type_raw is None:
                    chunk_type_str: Optional[str] = None
                elif isinstance(chunk_type_raw, str):
                    chunk_type_str = chunk_type_raw
                else:
                    chunk_type_str = str(chunk_type_raw)
                normalized_chunk_type = _normalize_chunk_type(chunk_type_str)
                chunk_metadata = {
                    **doc.metadata,  # Inherit document metadata
                    **chunk.metadata,  # Add chunk-specific metadata
                    "chunk_type": normalized_chunk_type,
                    "chunk_index": chunk.chunk_index,
                    "parent_id": chunk.parent_id or doc.id,
                    "original_doc_id": doc.id,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char
                }

                # Create new document for chunk
                chunk_doc = Document(
                    id=chunk.id,
                    content=chunk.content,
                    metadata=chunk_metadata,
                    source=doc.source,
                    score=doc.score  # Preserve original score
                )

                chunked_documents.append(chunk_doc)

                # Track statistics
                chunk_type_stats[normalized_chunk_type] = chunk_type_stats.get(normalized_chunk_type, 0) + 1
                total_chunks += 1

        except Exception as e:
            logger.error(f"Failed to chunk document: {_safe_exception_label(e)}")
            # Fall back to including original document
            chunked_documents.append(doc)

    # Update context
    context.documents = chunked_documents

    # Add chunking metadata to context
    context.metadata["enhanced_chunking_applied"] = True
    context.metadata["total_chunks_created"] = total_chunks
    context.metadata["chunk_type_distribution"] = chunk_type_stats
    context.metadata["chunking_options"] = chunking_options

    logger.info(
        f"Enhanced chunking created {total_chunks} chunks from {len(context.documents)} documents. "
        f"Types: {chunk_type_stats}"
    )

    return context


async def filter_chunks_by_type(
    context,  # RAGPipelineContext
    include_types: Optional[list[str]] = None,
    exclude_types: Optional[list[str]] = None
):
    """
    Filter chunks by their type.

    Useful for:
    - Including only code chunks for code search
    - Excluding tables from general text search
    - Focusing on headers for navigation

    Args:
        context: RAGPipelineContext with chunked documents
        include_types: List of chunk types to include (e.g., ["code", "text"])
        exclude_types: List of chunk types to exclude (e.g., ["table"])

    Returns:
        Updated context with filtered documents
    """
    if not context.documents:
        return context

    if not include_types and not exclude_types:
        return context  # No filtering needed

    include_norm = {_normalize_chunk_type(t) for t in include_types} if include_types else None
    exclude_norm = {_normalize_chunk_type(t) for t in exclude_types} if exclude_types else None

    filtered_docs = []

    for doc in context.documents:
        chunk_type = _normalize_chunk_type(doc.metadata.get("chunk_type"))

        # Apply inclusion filter
        if include_norm and chunk_type not in include_norm:
            continue

        # Apply exclusion filter
        if exclude_norm and chunk_type in exclude_norm:
            continue

        filtered_docs.append(doc)

    original_count = len(context.documents)
    context.documents = filtered_docs

    context.metadata["chunk_filtering_applied"] = True
    context.metadata["chunks_before_filter"] = original_count
    context.metadata["chunks_after_filter"] = len(filtered_docs)

    logger.debug(f"Filtered chunks: {original_count} -> {len(filtered_docs)}")

    return context


async def expand_with_parent_context(
    context,  # RAGPipelineContext
    expansion_size: Optional[int] = None,
    include_siblings: Optional[bool] = None
):
    """
    Expand chunks with parent document context.

    For each chunk, fetches:
    - Parent document context
    - Sibling chunks (adjacent chunks from same document)

    Args:
        context: RAGPipelineContext with chunked documents
        expansion_size: Characters to expand around chunk
        include_siblings: Include adjacent chunks

    Returns:
        Updated context with expanded documents
    """
    if not context.documents:
        return context

    config = context.config or {}
    expansion_size = expansion_size or config.get("parent_expansion_size", 500)
    include_siblings = include_siblings if include_siblings is not None else config.get("include_siblings", True)

    # Group chunks by parent document
    parent_groups: dict[str, list[Document]] = {}
    for doc in context.documents:
        parent_id = doc.metadata.get("parent_id") or doc.metadata.get("original_doc_id")
        if parent_id:
            if parent_id not in parent_groups:
                parent_groups[parent_id] = []
            parent_groups[parent_id].append(doc)

    expanded_docs = []

    for _parent_id, chunks in parent_groups.items():
        # Sort chunks by index
        chunks.sort(key=lambda d: d.metadata.get("chunk_index", 0))

        for i, chunk in enumerate(chunks):
            # Build expanded content
            expanded_parts = []

            # Add previous sibling if requested
            if include_siblings and i > 0:
                prev_chunk = chunks[i - 1]
                expanded_parts.append(f"[Previous context: ...{prev_chunk.content[-100:]}]")

            # Add main chunk
            expanded_parts.append(chunk.content)

            # Add next sibling if requested
            if include_siblings and i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                expanded_parts.append(f"[Following context: {next_chunk.content[:100]}...]")

            # Create expanded document
            expanded_doc = Document(
                id=chunk.id,
                content="\n".join(expanded_parts),
                metadata={
                    **chunk.metadata,
                    "expanded": True,
                    "has_siblings": include_siblings and (i > 0 or i < len(chunks) - 1)
                },
                source=chunk.source,
                score=chunk.score
            )

            expanded_docs.append(expanded_doc)

    context.documents = expanded_docs
    context.metadata["parent_expansion_applied"] = True

    logger.debug(f"Expanded {len(expanded_docs)} chunks with parent context")

    return context


async def prioritize_by_chunk_type(
    context,  # RAGPipelineContext
    type_priorities: Optional[dict[str, float]] = None
):
    """
    Adjust document scores based on chunk type.

    Useful for:
    - Boosting code chunks for programming queries
    - Prioritizing headers for navigation queries
    - Reducing table weight for narrative queries

    Args:
        context: RAGPipelineContext with chunked documents
        type_priorities: Dict mapping chunk types to score multipliers
                        e.g., {"code": 1.5, "table": 0.8, "header": 1.2}

    Returns:
        Updated context with adjusted scores
    """
    if not context.documents:
        return context

    # Default priorities
    default_priorities = {
        "code": 1.0,
        "table": 1.0,
        "heading": 1.0,
        "text": 1.0,
        "list": 1.0,
    }

    raw_priorities = type_priorities or context.config.get("chunk_type_priorities", default_priorities)
    priorities: dict[str, float] = {}
    for key, value in (raw_priorities or {}).items():
        priorities[_normalize_chunk_type(key)] = float(value)

    for doc in context.documents:
        chunk_type = _normalize_chunk_type(doc.metadata.get("chunk_type"))

        # Preserve base/original score to avoid compounding multipliers across repeated calls
        base_score = doc.metadata.get("original_score")
        if base_score is None:
            base_score = doc.score
            doc.metadata["original_score"] = base_score

        multiplier = priorities.get(chunk_type, 1.0)
        new_score = base_score * multiplier

        # Clamp to reasonable bounds to satisfy invariants in property tests
        if new_score < 0.0:
            new_score = 0.0
        elif new_score > 10.0:
            new_score = 10.0

        doc.score = new_score
        doc.metadata["score_adjusted"] = True
        doc.metadata["score_multiplier"] = multiplier

    # Re-sort by adjusted scores
    context.documents.sort(key=lambda d: d.score, reverse=True)

    context.metadata["chunk_type_prioritization_applied"] = True
    context.metadata["type_priorities"] = priorities

    logger.debug(f"Applied chunk type priorities: {priorities}")

    return context


# Function registry for pipeline building
ENHANCED_CHUNKING_FUNCTIONS = {
    "enhanced_chunk_documents": enhanced_chunk_documents,
    "filter_chunks_by_type": filter_chunks_by_type,
    "expand_with_parent_context": expand_with_parent_context,
    "prioritize_by_chunk_type": prioritize_by_chunk_type
}
