# document_processing_integration.py
"""
Document processing integration for the RAG service.

This module integrates with the unified v2 chunking module to provide enhanced
document processing capabilities specifically optimized for RAG retrieval.
"""

import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

# Import from new modular chunking system
try:
    from ...Chunking import (
        ChunkingError,
        ChunkType,
        EnhancedChunk,
        InvalidInputError,
        improved_chunking_process,
    )
    CHUNK_LIB_AVAILABLE = True
except ImportError as e:
    CHUNK_LIB_AVAILABLE = False
    logger.warning(f"Chunking module not available: {e}. Document processing will be limited.")

from .types import Document


@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    clean_artifacts: bool = True
    fix_encoding: bool = True
    detect_structure: bool = True
    enrich_metadata: bool = True
    optimize_boundaries: bool = True
    merge_small_chunks: bool = True
    min_chunk_size: int = 100
    max_chunk_size: int = 1000
    target_chunk_size: int = 500
    preserve_formatting: bool = True


class DocumentProcessor:
    """Enhanced document processor for RAG pipeline."""

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize document processor.

        Args:
            config: Processing configuration
        """
        self.config = config or ProcessingConfig()
        self.artifact_patterns = self._compile_artifact_patterns()

    def _compile_artifact_patterns(self) -> list[re.Pattern]:
        """Compile regex patterns for artifact detection."""
        patterns = [
            # PDF artifacts
            re.compile(r'\x00+'),  # Null bytes
            re.compile(r'[\x01-\x08\x0B\x0C\x0E-\x1F]'),  # Control characters
            re.compile(r'(?:^\d+\s*$\n?)', re.MULTILINE),  # Page numbers
            re.compile(r'(?:^Page \d+ of \d+$\n?)', re.MULTILINE),  # Page indicators

            # Encoding artifacts
            re.compile(r'â€™|â€"|â€œ|â€�|â€˜'),  # Common UTF-8 decode errors
            re.compile(r'Ã¢|Ã©|Ã¨|Ã |Ã§'),  # Latin-1 artifacts

            # Formatting artifacts
            re.compile(r'\s{5,}'),  # Excessive whitespace
            re.compile(r'\n{4,}'),  # Excessive newlines
            re.compile(r'(?:^-{3,}$\n?)', re.MULTILINE),  # Separator lines
        ]
        return patterns

    async def process_document(
        self,
        content: str,
        source: str = "unknown",
        metadata: Optional[dict[str, Any]] = None
    ) -> list[EnhancedChunk]:
        """
        Process document with enhanced chunking for RAG.

        Args:
            content: Document content
            source: Document source
            metadata: Additional metadata

        Returns:
            List of enhanced chunks
        """
        if not CHUNK_LIB_AVAILABLE:
            logger.error("Chunking module not available")
            return []

        # Pre-process content
        if self.config.clean_artifacts:
            content = self._clean_artifacts(content)

        if self.config.fix_encoding:
            content = self._fix_encoding(content)

        # Detect document structure
        structure_info = {}
        if self.config.detect_structure:
            structure_info = self._detect_structure(content)

        # Use improved_chunking_process from new module
        chunk_options = {
            "method": "semantic" if len(content) > 5000 else "sentences",
            "max_size": self.config.target_chunk_size,
            "overlap": 50
        }

        try:
            # Call backward-compatible chunking function
            chunk_results = improved_chunking_process(
                content,
                chunk_options
            )

            # Convert to EnhancedChunk objects
            enhanced_chunks = []
            for i, chunk_data in enumerate(chunk_results):
                chunk_text = chunk_data.get("text", "")
                chunk_metadata = chunk_data.get("metadata", {})

                # Determine chunk type
                chunk_type = self._determine_chunk_type(chunk_text, structure_info)

                # Calculate positions
                start_char = chunk_metadata.get("start_index", i * self.config.target_chunk_size)
                end_char = chunk_metadata.get("end_index", start_char + len(chunk_text))

                # Create enhanced chunk
                enhanced_chunk = EnhancedChunk(
                    id=self._generate_chunk_id(source, i),
                    content=chunk_text,
                    chunk_type=chunk_type,
                    start_char=start_char,
                    end_char=end_char,
                    chunk_index=i,
                    metadata={
                        **chunk_metadata,
                        **structure_info.get(f"chunk_{i}", {}),
                        "source": source,
                        "processing_config": self.config.__dict__
                    },
                    parent_id=metadata.get("parent_id") if metadata else None
                )

                enhanced_chunks.append(enhanced_chunk)

            # Post-process chunks
            if self.config.optimize_boundaries:
                enhanced_chunks = self._optimize_boundaries(enhanced_chunks)

            if self.config.merge_small_chunks:
                enhanced_chunks = self._merge_small_chunks(enhanced_chunks)

            if self.config.enrich_metadata:
                enhanced_chunks = self._enrich_metadata(enhanced_chunks, metadata)

            logger.info(f"Processed document into {len(enhanced_chunks)} enhanced chunks")
            return enhanced_chunks

        except (ChunkingError, InvalidInputError) as e:
            logger.error(f"Chunking error during document processing: {type(e).__name__}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during document processing: {type(e).__name__}")
            return []

    def _clean_artifacts(self, content: str) -> str:
        """Remove common artifacts from content."""
        cleaned = content

        for pattern in self.artifact_patterns:
            cleaned = pattern.sub(' ', cleaned)

        # Normalize whitespace
        cleaned = ' '.join(cleaned.split())

        # Remove zero-width characters
        cleaned = ''.join(char for char in cleaned if unicodedata.category(char) != 'Cf')

        return cleaned

    def _fix_encoding(self, content: str) -> str:
        """Fix common encoding issues."""
        # Common replacements
        replacements = {
            'â€™': "'",
            'â€"': "-",
            'â€œ': '"',
            'â€�': '"',
            'â€˜': "'",
            'Ã©': 'é',
            'Ã¨': 'è',
            'Ã ': 'à',
            'Ã§': 'ç',
            'Ã¢': 'â',
        }

        fixed = content
        for old, new in replacements.items():
            fixed = fixed.replace(old, new)

        # Try to fix mojibake
        try:
            # Attempt to decode as UTF-8 and re-encode
            fixed = fixed.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')
        except Exception:
            logger.debug("Mojibake fix failed; returning replacements-only content")

        return fixed

    def _detect_structure(self, content: str) -> dict[str, Any]:
        """Detect document structure elements."""
        structure = {
            "has_headers": False,
            "has_lists": False,
            "has_tables": False,
            "has_code": False,
            "sections": []
        }

        # Detect headers (Markdown style)
        header_pattern = re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE)
        headers = header_pattern.findall(content)
        if headers:
            structure["has_headers"] = True
            structure["headers"] = headers

        # Detect lists
        list_pattern = re.compile(r'^[\*\-\+•]\s+', re.MULTILINE)
        numbered_list_pattern = re.compile(r'^\d+\.\s+', re.MULTILINE)
        if list_pattern.search(content) or numbered_list_pattern.search(content):
            structure["has_lists"] = True

        # Detect tables (simple pipe tables)
        table_pattern = re.compile(r'\|.*\|.*\|', re.MULTILINE)
        if table_pattern.search(content):
            structure["has_tables"] = True

        # Detect code blocks
        code_pattern = re.compile(r'```[\s\S]*?```|`[^`]+`')
        if code_pattern.search(content):
            structure["has_code"] = True

        return structure

    def _determine_chunk_type(
        self,
        chunk_text: str,
        structure_info: dict[str, Any]
    ) -> ChunkType:
        """Determine the type of a chunk based on its content."""
        # Check for specific patterns
        if re.match(r'^#{1,6}\s+', chunk_text):
            return ChunkType.HEADER
        elif re.match(r'^[\*\-\+•]\s+', chunk_text) or re.match(r'^\d+\.\s+', chunk_text):
            return ChunkType.LIST
        elif '|' in chunk_text and chunk_text.count('|') > 2:
            return ChunkType.TABLE
        elif '```' in chunk_text or re.search(r'(def |class |function |import |from )', chunk_text):
            return ChunkType.CODE
        else:
            return ChunkType.PARAGRAPH

    def _optimize_boundaries(self, chunks: list[EnhancedChunk]) -> list[EnhancedChunk]:
        """Optimize chunk boundaries to avoid breaking sentences."""
        optimized = []

        for i, chunk in enumerate(chunks):
            content = chunk.content

            # Check if chunk ends mid-sentence
            if not content.endswith(('.', '!', '?', '\n')):
                # Try to find the last sentence boundary
                last_period = content.rfind('.')
                last_exclaim = content.rfind('!')
                last_question = content.rfind('?')

                last_boundary = max(last_period, last_exclaim, last_question)

                if last_boundary > len(content) * 0.7:  # If boundary is in last 30%
                    # Move content after boundary to next chunk
                    if i < len(chunks) - 1:
                        next_content = content[last_boundary + 1:].strip()
                        if next_content:
                            chunks[i + 1].content = next_content + ' ' + chunks[i + 1].content
                            chunk.content = content[:last_boundary + 1]
                            chunk.end_char = chunk.start_char + last_boundary + 1

            optimized.append(chunk)

        return optimized

    def _merge_small_chunks(self, chunks: list[EnhancedChunk]) -> list[EnhancedChunk]:
        """Merge chunks that are too small."""
        if not chunks:
            return chunks

        merged = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            current_size = len(current.content)
            next_size = len(next_chunk.content)

            # Merge if current is too small and combined wouldn't be too large
            if (current_size < self.config.min_chunk_size and
                current_size + next_size < self.config.max_chunk_size):

                # Merge chunks
                current.content += '\n\n' + next_chunk.content
                current.end_char = next_chunk.end_char

                # Merge metadata
                if 'merged_chunks' not in current.metadata:
                    current.metadata['merged_chunks'] = []
                current.metadata['merged_chunks'].append(next_chunk.id)

                # Update chunk type if needed
                if current.chunk_type == ChunkType.TEXT and next_chunk.chunk_type != ChunkType.TEXT:
                    current.chunk_type = next_chunk.chunk_type
            else:
                merged.append(current)
                current = next_chunk

        merged.append(current)

        # Re-index chunks
        for i, chunk in enumerate(merged):
            chunk.chunk_index = i

        return merged

    def _enrich_metadata(
        self,
        chunks: list[EnhancedChunk],
        document_metadata: Optional[dict[str, Any]] = None
    ) -> list[EnhancedChunk]:
        """Enrich chunks with additional metadata for RAG."""
        doc_metadata = document_metadata or {}

        for i, chunk in enumerate(chunks):
            # Add positional metadata
            chunk.metadata['position'] = {
                'index': i,
                'total': len(chunks),
                'relative_position': i / len(chunks) if chunks else 0,
                'is_first': i == 0,
                'is_last': i == len(chunks) - 1
            }

            # Add document metadata
            chunk.metadata['document'] = doc_metadata

            # Add content statistics
            chunk.metadata['stats'] = {
                'char_count': len(chunk.content),
                'word_count': len(chunk.content.split()),
                'sentence_count': chunk.content.count('.') + chunk.content.count('!') + chunk.content.count('?'),
                'has_numbers': bool(re.search(r'\d', chunk.content)),
                'has_urls': bool(re.search(r'https?://', chunk.content))
            }

            # Add chunk relationships
            if i > 0:
                chunk.metadata['prev_chunk_id'] = chunks[i - 1].id
            if i < len(chunks) - 1:
                chunk.metadata['next_chunk_id'] = chunks[i + 1].id

        return chunks

    def _generate_chunk_id(self, source: str, index: int) -> str:
        """Generate unique chunk ID."""
        data = f"{source}_{index}_{hash(source)}"
        return hashlib.md5(data.encode(), usedforsecurity=False).hexdigest()[:12]


class DocumentValidator:
    """Validates document chunks for quality."""

    @staticmethod
    def validate_chunks(chunks: list[EnhancedChunk]) -> tuple[bool, list[str]]:
        """
        Validate chunks for quality issues.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        if not chunks:
            issues.append("No chunks provided")
            return False, issues

        # Check for empty chunks
        empty_chunks = [c.id for c in chunks if not c.content.strip()]
        if empty_chunks:
            issues.append(f"Empty chunks found: {empty_chunks}")

        # Check for duplicate content
        content_hashes = defaultdict(list)
        for chunk in chunks:
            content_hash = hashlib.md5(chunk.content.encode(), usedforsecurity=False).hexdigest()
            content_hashes[content_hash].append(chunk.id)

        duplicates = {h: ids for h, ids in content_hashes.items() if len(ids) > 1}
        if duplicates:
            issues.append(f"Duplicate content in chunks: {list(duplicates.values())}")

        # Check for position gaps
        positions = sorted([(c.start_char, c.end_char) for c in chunks])
        for i in range(1, len(positions)):
            prev_end = positions[i - 1][1]
            curr_start = positions[i][0]
            if curr_start > prev_end + 100:  # Allow small gaps
                issues.append(f"Large gap in positions: {prev_end} to {curr_start}")

        # Check chunk size distribution
        sizes = [len(c.content) for c in chunks]
        if sizes:
            avg_size = sum(sizes) / len(sizes)
            outliers = [c.id for c in chunks if abs(len(c.content) - avg_size) > avg_size * 2]
            if outliers:
                issues.append(f"Chunk size outliers: {outliers}")

        return len(issues) == 0, issues


# Pipeline integration functions

async def preprocess_for_rag(context: Any, **kwargs) -> Any:
    """Preprocess documents for RAG pipeline."""
    config_dict = context.config.get("document_processing", {})

    # Create processor
    processor_config = ProcessingConfig(
        clean_artifacts=config_dict.get("clean_artifacts", True),
        fix_encoding=config_dict.get("fix_encoding", True),
        detect_structure=config_dict.get("detect_structure", True),
        enrich_metadata=config_dict.get("enrich_metadata", True),
        optimize_boundaries=config_dict.get("optimize_boundaries", True),
        merge_small_chunks=config_dict.get("merge_small_chunks", True),
        min_chunk_size=config_dict.get("min_chunk_size", 100),
        max_chunk_size=config_dict.get("max_chunk_size", 1000),
        target_chunk_size=config_dict.get("target_chunk_size", 500)
    )

    processor = DocumentProcessor(processor_config)

    # Process documents if they need chunking
    if hasattr(context, "raw_content") and context.raw_content:
        chunks = await processor.process_document(
            context.raw_content,
            source=context.metadata.get("source", "unknown"),
            metadata=context.metadata
        )

        # Convert chunks to documents
        documents = []
        for chunk in chunks:
            doc = Document(
                id=chunk.id,
                content=chunk.content,
                metadata={
                    **chunk.metadata,
                    "chunk_type": chunk.chunk_type.value,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "chunk_index": chunk.chunk_index
                }
            )
            documents.append(doc)

        context.documents = documents
        context.metadata["preprocessing"] = {
            "chunks_created": len(chunks),
            "processing_config": processor_config.__dict__
        }

        logger.info(f"Preprocessed content into {len(chunks)} chunks")

    return context


async def enrich_document_chunks(context: Any, **kwargs) -> Any:
    """Enrich document chunks with additional metadata."""
    if not hasattr(context, "documents") or not context.documents:
        return context

    # Add enrichment metadata
    for i, doc in enumerate(context.documents):
        # Add sequential metadata
        doc.metadata["sequence"] = {
            "position": i,
            "total": len(context.documents),
            "is_first": i == 0,
            "is_last": i == len(context.documents) - 1
        }

        # Add query relevance hints
        if hasattr(context, "query"):
            query_terms = set(context.query.lower().split())
            content_terms = set(doc.content.lower().split())
            overlap = query_terms & content_terms

            doc.metadata["query_relevance"] = {
                "term_overlap": len(overlap),
                "term_overlap_ratio": len(overlap) / len(query_terms) if query_terms else 0
            }

    context.metadata["enrichment_complete"] = True
    return context


async def validate_document_chunks(context: Any, **kwargs) -> Any:
    """Validate document chunks for quality."""
    if not hasattr(context, "documents") or not context.documents:
        return context

    # Convert documents to chunks for validation
    chunks = []
    for doc in context.documents:
        if "chunk_type" in doc.metadata:
            chunk = EnhancedChunk(
                id=doc.id,
                content=doc.content,
                chunk_type=ChunkType(doc.metadata.get("chunk_type", "text")),
                start_char=doc.metadata.get("start_char", 0),
                end_char=doc.metadata.get("end_char", len(doc.content)),
                chunk_index=doc.metadata.get("chunk_index", 0),
                metadata=doc.metadata
            )
            chunks.append(chunk)

    if chunks:
        is_valid, issues = DocumentValidator.validate_chunks(chunks)

        context.metadata["validation"] = {
            "is_valid": is_valid,
            "issues": issues,
            "chunks_validated": len(chunks)
        }

        if not is_valid:
            logger.warning(f"Document validation issues: {issues}")

    return context
