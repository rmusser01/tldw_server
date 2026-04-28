# citations.py
"""
Enhanced citation generation system for the RAG service.

This module provides:
1. Academic citation formatting (MLA, APA, Chicago, Harvard)
2. Chunk-level citations for answer verification
3. Dual citation output for complete traceability
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.Metrics import get_metrics_registry

from .types import Document, EvidenceChain, EvidenceNode


class CitationStyle(Enum):
    """Supported academic citation styles."""
    MLA = "mla"
    APA = "apa"
    CHICAGO = "chicago"
    HARVARD = "harvard"
    IEEE = "ieee"
    CHUNK = "chunk"  # Special format for chunk citations


@dataclass
class ChunkCitation:
    """Citation for a specific chunk used in answer generation."""
    chunk_id: str
    source_document_id: str
    source_document_title: str
    location: str  # "Chapter 3, Page 45" or "Paragraph 5"
    text_snippet: str
    confidence: float
    usage_context: str  # How this chunk was used in the answer

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "source_document_id": self.source_document_id,
            "source_document_title": self.source_document_title,
            "location": self.location,
            "text_snippet": self.text_snippet[:200] + "..." if len(self.text_snippet) > 200 else self.text_snippet,
            "confidence": self.confidence,
            "usage_context": self.usage_context
        }


@dataclass
class DualCitationResult:
    """Combined result containing both academic and chunk citations."""
    academic_citations: list[str]  # Formatted academic citations
    chunk_citations: list[ChunkCitation]  # Chunk-level citations
    inline_markers: dict[str, str]  # Mapping of inline markers to chunks
    citation_map: dict[str, list[str]]  # Document ID to chunk IDs mapping

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "academic_citations": self.academic_citations,
            "chunk_citations": [c.to_dict() for c in self.chunk_citations],
            "inline_markers": self.inline_markers,
            "citation_map": self.citation_map
        }


class AcademicCitationFormatter:
    """Formats citations according to academic standards."""

    def format_citation(
        self,
        metadata: dict[str, Any],
        style: CitationStyle
    ) -> str:
        """
        Format a citation according to the specified style.

        Args:
            metadata: Document metadata containing author, title, date, etc.
            style: Citation style to use

        Returns:
            Formatted citation string
        """
        if style == CitationStyle.MLA:
            return self._format_mla(metadata)
        elif style == CitationStyle.APA:
            return self._format_apa(metadata)
        elif style == CitationStyle.CHICAGO:
            return self._format_chicago(metadata)
        elif style == CitationStyle.HARVARD:
            return self._format_harvard(metadata)
        elif style == CitationStyle.IEEE:
            return self._format_ieee(metadata)
        else:
            return self._format_generic(metadata)

    def _format_mla(self, meta: dict[str, Any]) -> str:
        """
        Format MLA citation.
        Format: Author. "Title." Publication, Date, Pages.
        """
        parts = []

        # Author(s)
        author = self._format_author(meta.get("author"), style="mla")
        if author:
            parts.append(author)

        # Title
        title = meta.get("title", "Untitled")
        if meta.get("is_article"):
            parts.append(f'"{title}"')
        else:
            parts.append(f"*{title}*")

        # Publication/Container
        if meta.get("publication"):
            parts.append(meta["publication"])
        elif meta.get("website"):
            parts.append(meta["website"])

        # Volume/Issue
        if meta.get("volume"):
            vol_issue = f"vol. {meta['volume']}"
            if meta.get("issue"):
                vol_issue += f", no. {meta['issue']}"
            parts.append(vol_issue)

        # Date
        date = self._format_date(meta.get("date"), style="mla")
        if date:
            parts.append(date)

        # Pages
        if meta.get("pages"):
            parts.append(f"pp. {meta['pages']}")
        elif meta.get("page"):
            parts.append(f"p. {meta['page']}")

        # URL/DOI
        if meta.get("doi"):
            parts.append(f"doi:{meta['doi']}")
        elif meta.get("url"):
            parts.append(meta["url"])

        return ". ".join(parts) + "."

    def _format_apa(self, meta: dict[str, Any]) -> str:
        """
        Format APA citation.
        Format: Author. (Date). Title. Publication.
        """
        parts = []

        # Author(s)
        author = self._format_author(meta.get("author"), style="apa")
        if author:
            parts.append(author)

        # Date
        date = self._format_date(meta.get("date"), style="apa")
        parts.append(f"({date})")

        # Title
        title = meta.get("title", "Untitled")
        if meta.get("is_article"):
            parts.append(title)
        else:
            parts.append(f"*{title}*")

        # Publication
        if meta.get("publication"):
            pub = meta["publication"]
            if meta.get("volume"):
                pub += f", {meta['volume']}"
                if meta.get("issue"):
                    pub += f"({meta['issue']})"
            if meta.get("pages"):
                pub += f", {meta['pages']}"
            parts.append(pub)

        # DOI/URL
        if meta.get("doi"):
            parts.append(f"https://doi.org/{meta['doi']}")
        elif meta.get("url"):
            parts.append(f"Retrieved from {meta['url']}")

        return ". ".join(parts) + "."

    def _format_chicago(self, meta: dict[str, Any]) -> str:
        """
        Format Chicago citation (Notes-Bibliography style).
        """
        parts = []

        # Author(s)
        author = self._format_author(meta.get("author"), style="chicago")
        if author:
            parts.append(author)

        # Title
        title = meta.get("title", "Untitled")
        if meta.get("is_article"):
            parts.append(f'"{title}"')
        else:
            parts.append(f"*{title}*")

        # Publication details
        if meta.get("publication"):
            parts.append(meta["publication"])

        # Volume/Issue
        if meta.get("volume"):
            vol_str = str(meta["volume"])
            if meta.get("issue"):
                vol_str += f", no. {meta['issue']}"
            parts.append(vol_str)

        # Date
        date = self._format_date(meta.get("date"), style="chicago")
        if date:
            parts.append(f"({date})")

        # Pages
        if meta.get("pages"):
            parts.append(meta["pages"])

        # DOI/URL
        if meta.get("doi"):
            parts.append(f"https://doi.org/{meta['doi']}")
        elif meta.get("url"):
            parts.append(meta["url"])

        return ". ".join(parts) + "."

    def _format_harvard(self, meta: dict[str, Any]) -> str:
        """
        Format Harvard citation.
        """
        parts = []

        # Author(s) and date
        author = self._format_author(meta.get("author"), style="harvard")
        date = self._format_date(meta.get("date"), style="harvard")

        if author:
            parts.append(f"{author} {date}")
        else:
            parts.append(date)

        # Title
        title = meta.get("title", "Untitled")
        if meta.get("is_article"):
            parts.append(f"'{title}'")
        else:
            parts.append(f"*{title}*")

        # Publication
        if meta.get("publication"):
            pub = meta["publication"]
            if meta.get("volume"):
                pub += f", vol. {meta['volume']}"
                if meta.get("issue"):
                    pub += f"({meta['issue']})"
            if meta.get("pages"):
                pub += f", pp. {meta['pages']}"
            parts.append(pub)

        # Available at
        if meta.get("url"):
            parts.append(f"Available at: {meta['url']}")
            parts.append(f"[Accessed {datetime.now().strftime('%d %B %Y')}]")

        return ". ".join(parts) + "."

    def _format_ieee(self, meta: dict[str, Any], number: int = 1) -> str:
        """
        Format IEEE citation.
        """
        parts = []

        # [Number] Author(s)
        parts.append(f"[{number}]")

        author = self._format_author(meta.get("author"), style="ieee")
        if author:
            parts.append(author)

        # "Title"
        title = meta.get("title", "Untitled")
        parts.append(f'"{title}"')

        # Publication
        if meta.get("publication"):
            parts.append(f"in {meta['publication']}")

        # Volume/Issue/Pages
        if meta.get("volume"):
            vol_str = f"vol. {meta['volume']}"
            if meta.get("issue"):
                vol_str += f", no. {meta['issue']}"
            if meta.get("pages"):
                vol_str += f", pp. {meta['pages']}"
            parts.append(vol_str)

        # Date
        date = self._format_date(meta.get("date"), style="ieee")
        if date:
            parts.append(date)

        return " ".join(parts) + "."

    def _format_generic(self, meta: dict[str, Any]) -> str:
        """Fallback generic citation format."""
        parts = []

        if meta.get("author"):
            parts.append(str(meta["author"]))

        if meta.get("title"):
            parts.append(f'"{meta["title"]}"')

        if meta.get("publication"):
            parts.append(meta["publication"])

        if meta.get("date"):
            parts.append(str(meta["date"]))

        if meta.get("url"):
            parts.append(meta["url"])

        return ". ".join(parts) if parts else "Unknown source"

    def _format_author(self, author: Any, style: str) -> Optional[str]:
        """Format author name(s) according to style."""
        if not author:
            return None

        authors = author if isinstance(author, list) else [str(author)]

        if not authors:
            return None

        if style in ["mla", "chicago"]:
            # Last, First for first author
            if len(authors) == 1:
                return self._reverse_name(authors[0])
            elif len(authors) == 2:
                return f"{self._reverse_name(authors[0])}, and {authors[1]}"
            else:
                return f"{self._reverse_name(authors[0])}, et al."

        elif style == "apa":
            # Last, F. M.
            formatted = []
            for author in authors[:3]:  # APA shows up to 3 authors
                formatted.append(self._initials_format(author))

            if len(authors) > 3:
                formatted.append("et al.")

            return ", ".join(formatted)

        elif style == "harvard":
            # Last, F
            if len(authors) == 1:
                return self._surname_initial(authors[0])
            elif len(authors) == 2:
                return f"{self._surname_initial(authors[0])} and {self._surname_initial(authors[1])}"
            else:
                return f"{self._surname_initial(authors[0])} et al."

        elif style == "ieee":
            # F. Last
            formatted = []
            for author in authors[:3]:
                formatted.append(self._ieee_format(author))

            if len(authors) > 3:
                formatted.append("et al.")

            return ", ".join(formatted)

        return ", ".join(authors)

    def _reverse_name(self, name: str) -> str:
        """Convert 'First Last' to 'Last, First'."""
        parts = name.strip().split()
        if len(parts) >= 2:
            return f"{parts[-1]}, {' '.join(parts[:-1])}"
        return name

    def _initials_format(self, name: str) -> str:
        """Convert 'First Middle Last' to 'Last, F. M.'."""
        parts = name.strip().split()
        if len(parts) >= 2:
            last = parts[-1]
            initials = ". ".join([p[0].upper() for p in parts[:-1]]) + "."
            return f"{last}, {initials}"
        return name

    def _surname_initial(self, name: str) -> str:
        """Convert 'First Last' to 'Last, F'."""
        parts = name.strip().split()
        if len(parts) >= 2:
            return f"{parts[-1]}, {parts[0][0].upper()}"
        return name

    def _ieee_format(self, name: str) -> str:
        """Convert 'First Middle Last' to 'F. M. Last'."""
        parts = name.strip().split()
        if len(parts) >= 2:
            initials = ". ".join([p[0].upper() for p in parts[:-1]]) + "."
            return f"{initials} {parts[-1]}"
        return name

    def _format_date(self, date: Any, style: str) -> str:
        """Format date according to style."""
        if not date:
            return "n.d." if style == "apa" else ""

        # Try to parse date
        if isinstance(date, datetime):
            dt = date
        else:
            try:
                # Try common date formats
                dt = datetime.fromisoformat(str(date))
            except Exception:
                # Fallback to string representation with a warning metric
                logger.debug("Citation date parse failed; returning raw")
                try:
                    get_metrics_registry().increment(
                        "app_warning_events_total",
                        labels={"component": "rag", "event": "citation_date_parse_failed"},
                    )
                except Exception:
                    logger.debug("metrics increment failed for rag citation_date_parse_failed")
                return str(date)

        if style == "mla":
            return dt.strftime("%d %b. %Y")  # 15 Jan. 2023
        elif style == "apa":
            return dt.strftime("%Y")  # 2023
        elif style == "chicago":
            return dt.strftime("%B %Y")  # January 2023
        elif style == "harvard":
            return f"({dt.strftime('%Y')})"  # (2023)
        elif style == "ieee":
            return dt.strftime("%b. %Y")  # Jan. 2023

        return str(date)


class CitationGenerator:
    """Generate both academic and chunk-level citations for documents."""

    def __init__(
        self,
        max_citation_length: int = 200,
        context_window: int = 50,
        enable_fuzzy: bool = True,
        fuzzy_threshold: float = 0.8,
        enable_semantic: bool = True,
        semantic_threshold: float = 0.7
    ):
        """
        Initialize citation generator.

        Args:
            max_citation_length: Maximum length of citation text
            context_window: Characters of context around citation
            enable_fuzzy: Enable fuzzy matching
            fuzzy_threshold: Minimum similarity for fuzzy matches
            enable_semantic: Enable semantic matching
            semantic_threshold: Minimum similarity for semantic matches
        """
        self.max_citation_length = max_citation_length
        self.context_window = context_window
        self.enable_fuzzy = enable_fuzzy
        self.fuzzy_threshold = fuzzy_threshold
        self.enable_semantic = enable_semantic
        self.semantic_threshold = semantic_threshold

        self.formatter = AcademicCitationFormatter()

        # Compile regex patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for performance."""
        self.sentence_pattern = re.compile(r'[.!?]\s+')
        self.word_boundary = re.compile(r'\b')
        self.whitespace_pattern = re.compile(r'\s+')

    async def generate_citations(
        self,
        documents: list[Document],
        query: str = "",
        style: CitationStyle = CitationStyle.MLA,
        include_chunks: bool = True,
        max_citations: int = 10
    ) -> DualCitationResult:
        """
        Generate both academic and chunk-level citations.

        Args:
            documents: Documents to generate citations from
            query: The search query (for relevance matching)
            style: Academic citation style
            include_chunks: Whether to include chunk-level citations
            max_citations: Maximum number of citations to return

        Returns:
            DualCitationResult with both types of citations
        """
        # Group documents by source document
        source_groups = self._group_by_source(documents)

        # Generate academic citations for unique source documents
        academic_citations = []
        citation_map = {}

        for source_id, doc_list in source_groups.items():
            # Get metadata from first document in group
            first_doc = doc_list[0]
            source_meta = first_doc.get_source_info()

            # Generate academic citation
            if source_meta:
                academic_cite = self.formatter.format_citation(source_meta, style)
                academic_citations.append(academic_cite)

                # Map source to chunks
                citation_map[source_id] = [d.id for d in doc_list]

        # Generate chunk citations if requested
        chunk_citations = []
        inline_markers = {}

        if include_chunks:
            for i, doc in enumerate(documents[:max_citations]):
                chunk_cite = self._generate_chunk_citation(doc, query, i + 1)
                chunk_citations.append(chunk_cite)

                # Create inline marker
                marker = f"[{i + 1}]"
                inline_markers[marker] = doc.id

        return DualCitationResult(
            academic_citations=academic_citations,
            chunk_citations=chunk_citations,
            inline_markers=inline_markers,
            citation_map=citation_map
        )

    def _group_by_source(self, documents: list[Document]) -> dict[str, list[Document]]:
        """Group documents by their source document ID."""
        groups = defaultdict(list)

        for doc in documents:
            # Use source_document_id if available, otherwise use document id
            source_id = doc.source_document_id or doc.id
            groups[source_id].append(doc)

        return dict(groups)

    def _generate_chunk_citation(
        self,
        document: Document,
        query: str,
        number: int
    ) -> ChunkCitation:
        """Generate a chunk-level citation for verification."""
        # Extract relevant snippet
        snippet = self._extract_relevant_snippet(document.content, query)

        # Get location information
        location = document.get_location_string()

        # Determine usage context
        usage_context = self._determine_usage_context(document, query)

        return ChunkCitation(
            chunk_id=document.id,
            source_document_id=document.source_document_id or document.id,
            source_document_title=document.metadata.get("title", "Untitled"),
            location=location,
            text_snippet=snippet,
            confidence=document.score,
            usage_context=usage_context
        )

    def _extract_relevant_snippet(
        self,
        content: str,
        query: str,
        max_length: int = 200
    ) -> str:
        """Extract the most relevant snippet from content."""
        if not query:
            # Return beginning of content
            return content[:max_length] + "..." if len(content) > max_length else content

        # Find best matching section
        query_lower = query.lower()
        content_lower = content.lower()

        # Look for exact match first
        pos = content_lower.find(query_lower)
        if pos != -1:
            start = max(0, pos - 50)
            end = min(len(content), pos + len(query) + 50)
            snippet = content[start:end]

            if start > 0:
                snippet = "..." + snippet
            if end < len(content):
                snippet = snippet + "..."

            return snippet

        # Look for keyword matches
        keywords = query_lower.split()
        best_pos = -1
        best_score = 0

        for i in range(0, len(content) - 100, 50):
            window = content_lower[i:i + 200]
            score = sum(1 for kw in keywords if kw in window)

            if score > best_score:
                best_score = score
                best_pos = i

        if best_pos >= 0:
            snippet = content[best_pos:best_pos + max_length]
            if best_pos > 0:
                snippet = "..." + snippet
            if best_pos + max_length < len(content):
                snippet = snippet + "..."
            return snippet

        # Fallback to beginning
        return content[:max_length] + "..." if len(content) > max_length else content

    def _determine_usage_context(
        self,
        document: Document,
        query: str
    ) -> str:
        """Determine how this chunk relates to the query."""
        if document.score > 0.9:
            return "Direct answer to query"
        elif document.score > 0.7:
            return "Highly relevant context"
        elif document.score > 0.5:
            return "Supporting information"
        else:
            return "Background context"

    def format_inline_citations(
        self,
        text: str,
        citations: DualCitationResult
    ) -> str:
        """
        Add inline citation markers to generated text.

        Args:
            text: The generated text
            citations: Citation results

        Returns:
            Text with inline citation markers
        """
        # This would require more sophisticated NLP to determine
        # where to place citations in the generated text
        # For now, append citations at the end

        if citations.inline_markers:
            markers = " ".join(citations.inline_markers.keys())
            return f"{text} {markers}"

        return text

    def format_bibliography(
        self,
        citations: DualCitationResult,
        style: CitationStyle = CitationStyle.MLA
    ) -> str:
        """
        Format a bibliography section.

        Args:
            citations: Citation results
            style: Citation style

        Returns:
            Formatted bibliography
        """
        lines = ["## References\n"]

        for i, cite in enumerate(citations.academic_citations, 1):
            lines.append(f"{i}. {cite}")

        if citations.chunk_citations:
            lines.append("\n## Source Chunks\n")
            for chunk in citations.chunk_citations:
                lines.append(f"- {chunk.source_document_title}, {chunk.location}")
                lines.append(f"  Confidence: {chunk.confidence:.2f}")
                lines.append(f"  Usage: {chunk.usage_context}")
                lines.append("")

        return "\n".join(lines)

    def link_citations_to_chain(
        self,
        citations: DualCitationResult,
        chain: EvidenceChain,
    ) -> DualCitationResult:
        """
        Link citations to an evidence chain for multi-hop transparency.

        Args:
            citations: The citation results
            chain: Evidence chain to link

        Returns:
            Updated DualCitationResult with chain information
        """
        if not chain or not chain.nodes:
            return citations

        # Build a mapping of document IDs to chain nodes
        doc_to_nodes: dict[str, list[EvidenceNode]] = {}
        for node in chain.nodes:
            if node.document_id not in doc_to_nodes:
                doc_to_nodes[node.document_id] = []
            doc_to_nodes[node.document_id].append(node)

        # Update chunk citations with chain info
        for chunk_cite in citations.chunk_citations:
            doc_id = chunk_cite.source_document_id
            if doc_id in doc_to_nodes:
                nodes = doc_to_nodes[doc_id]
                # Add chain metadata to the citation
                chunk_cite.usage_context = self._enhance_usage_context(
                    chunk_cite.usage_context,
                    nodes,
                    chain,
                )

        return citations

    def _enhance_usage_context(
        self,
        current_context: str,
        nodes: list,
        chain: EvidenceChain,
    ) -> str:
        """Enhance usage context with chain information."""
        facts = [n.fact[:50] + "..." if len(n.fact) > 50 else n.fact for n in nodes[:2]]
        claims_supported = set()
        for n in nodes:
            claims_supported.update(n.supports)

        chain_info = []
        if chain.hop_count > 1:
            chain_info.append(f"Multi-hop evidence ({chain.hop_count} sources)")
        if claims_supported:
            chain_info.append(f"Supports {len(claims_supported)} claim(s)")
        if facts:
            chain_info.append(f"Key facts: {'; '.join(facts)}")

        if chain_info:
            return f"{current_context}. {'. '.join(chain_info)}"
        return current_context

    async def generate_citations_with_chains(
        self,
        documents: list[Document],
        query: str = "",
        generated_answer: Optional[str] = None,
        style: CitationStyle = CitationStyle.MLA,
        include_chunks: bool = True,
        max_citations: int = 10,
    ) -> tuple[DualCitationResult, Optional[Any]]:
        """
        Generate citations with evidence chain building.

        Args:
            documents: Documents to generate citations from
            query: The search query
            generated_answer: The generated answer for claim extraction
            style: Academic citation style
            include_chunks: Whether to include chunk-level citations
            max_citations: Maximum number of citations

        Returns:
            Tuple of (DualCitationResult, ChainBuildResult or None)
        """
        # Generate base citations
        citations = await self.generate_citations(
            documents=documents,
            query=query,
            style=style,
            include_chunks=include_chunks,
            max_citations=max_citations,
        )

        # Try to build evidence chains
        chain_result = None
        try:
            from .evidence_chains import EvidenceChainBuilder

            builder = EvidenceChainBuilder()
            chain_result = await builder.build_chains(
                query=query,
                documents=documents,
                generated_answer=generated_answer,
            )

            # Link citations to chains
            if chain_result and chain_result.chains:
                for chain in chain_result.chains:
                    citations = self.link_citations_to_chain(citations, chain)

        except ImportError:
            logger.debug("Evidence chains module not available")
        except Exception:
            logger.warning("Evidence chain building failed")

        return citations, chain_result


# Pipeline integration function
async def generate_citations(context: Any, **kwargs) -> Any:
    """Generate citations for pipeline context."""
    config = context.config.get("citations", {})

    # Check if citations are enabled
    if not config.get("enabled", True):
        return context

    # Create citation generator
    generator = CitationGenerator(
        max_citation_length=config.get("max_length", 200),
        context_window=config.get("context_window", 50),
        enable_fuzzy=config.get("enable_fuzzy", True),
        fuzzy_threshold=config.get("fuzzy_threshold", 0.8),
        enable_semantic=config.get("enable_semantic", True),
        semantic_threshold=config.get("semantic_threshold", 0.7)
    )

    # Determine citation style
    style_str = config.get("style", "mla").lower()
    style_map = {
        "mla": CitationStyle.MLA,
        "apa": CitationStyle.APA,
        "chicago": CitationStyle.CHICAGO,
        "harvard": CitationStyle.HARVARD,
        "ieee": CitationStyle.IEEE
    }
    style = style_map.get(style_str, CitationStyle.MLA)

    # Generate citations
    result = await generator.generate_citations(
        documents=context.documents,
        query=context.query,
        style=style,
        include_chunks=config.get("include_chunks", True),
        max_citations=config.get("max_citations", 10)
    )

    # Add to context
    context.citations = result
    context.metadata["citations"] = {
        "count": len(result.academic_citations),
        "chunks": len(result.chunk_citations),
        "style": style.value
    }

    logger.info(f"Generated {len(result.academic_citations)} academic citations and {len(result.chunk_citations)} chunk citations")

    return context
