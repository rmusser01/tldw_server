"""Citation and bibliography adapters.

This module includes adapters for citation operations:
- citations: Generate citations
- bibliography_generate: Generate bibliography
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.content._config import (
    BibliographyGenerateConfig,
    CitationsConfig,
)


@registry.register(
    "citations",
    category="content",
    description="Generate citations",
    parallelizable=True,
    tags=["content", "citations"],
    config_model=CitationsConfig,
)
async def run_citations_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Generate academic citations from documents.

    Config:
      - documents: Optional[List[Dict]] - from last.documents or explicit
      - style: str = "apa" - citation style
        Options: "mla", "apa", "chicago", "harvard", "ieee"
      - include_inline: bool = True - include inline markers
      - max_citations: int = 10 - maximum number of citations
    Output:
      - {"citations": [str], "chunk_citations": [dict], "inline_markers": dict,
         "citation_map": dict, "style": str}
    """
    # Cancellation check
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    # Get documents
    documents_raw = config.get("documents")
    documents: list[dict[str, Any]] = []

    if documents_raw:
        # Template if it's a string reference
        if isinstance(documents_raw, str):
            rendered = apply_template_to_string(documents_raw, context)
            try:
                documents = json.loads(rendered) if rendered else []
            except Exception:
                documents = []
        elif isinstance(documents_raw, list):
            documents = documents_raw
    else:
        # Try to get from last.documents
        try:
            last = context.get("prev") or context.get("last") or {}
            if isinstance(last, dict):
                docs = last.get("documents") or last.get("results") or []
                if isinstance(docs, list):
                    documents = docs
        except Exception as documents_context_error:
            logger.debug("Citations adapter failed to read documents from context fallback", exc_info=documents_context_error)

    if not documents:
        return {
            "error": "missing_documents",
            "citations": [],
            "chunk_citations": [],
            "inline_markers": {},
            "citation_map": {},
        }

    style = str(config.get("style") or "apa").strip().lower()
    valid_styles = {"mla", "apa", "chicago", "harvard", "ieee"}
    if style not in valid_styles:
        style = "apa"

    include_inline = config.get("include_inline")
    include_inline = True if include_inline is None else bool(include_inline)

    max_citations = int(config.get("max_citations") or 10)
    max_citations = max(1, min(max_citations, 50))

    # Get query for relevance matching (optional)
    query = ""
    query_t = config.get("query")
    if query_t:
        query = apply_template_to_string(str(query_t), context) or str(query_t)
    else:
        try:
            last = context.get("prev") or context.get("last") or {}
            if isinstance(last, dict):
                query = str(last.get("query") or "")
        except Exception as query_context_error:
            logger.debug("Citations adapter failed to read query from context fallback", exc_info=query_context_error)

    # Test mode simulation
    if is_test_mode():
        # Simulate citations
        simulated_citations = []
        simulated_chunks = []
        inline_markers = {}
        citation_map = {}

        for i, doc in enumerate(documents[:max_citations]):
            # Build simulated citation
            author = doc.get("metadata", {}).get("author") or doc.get("author") or "Unknown Author"
            title = doc.get("metadata", {}).get("title") or doc.get("title") or f"Document {i+1}"
            date = doc.get("metadata", {}).get("date") or "n.d."

            if style == "apa":
                citation = f"{author}. ({date}). {title}."
            elif style == "mla":
                citation = f'{author}. "{title}." {date}.'
            elif style == "chicago":
                citation = f'{author}. "{title}." ({date}).'
            elif style == "harvard":
                citation = f"{author} ({date}) '{title}'."
            else:  # ieee
                citation = f'[{i+1}] {author}, "{title}", {date}.'

            simulated_citations.append(citation)

            # Build chunk citation
            chunk_cite = {
                "chunk_id": doc.get("id") or f"chunk_{i}",
                "source_document_id": doc.get("source_id") or f"doc_{i}",
                "source_document_title": title,
                "location": f"Section {i+1}",
                "text_snippet": (doc.get("content") or doc.get("text") or "")[:100] + "...",
                "confidence": float(doc.get("score") or 0.8),
                "usage_context": "Relevant context",
            }
            simulated_chunks.append(chunk_cite)

            # Inline marker
            marker = f"[{i+1}]"
            inline_markers[marker] = doc.get("id") or f"chunk_{i}"

            # Citation map
            source_id = doc.get("source_id") or f"source_{i}"
            if source_id not in citation_map:
                citation_map[source_id] = []
            citation_map[source_id].append(doc.get("id") or f"chunk_{i}")

        return {
            "citations": simulated_citations,
            "chunk_citations": simulated_chunks,
            "inline_markers": inline_markers,
            "citation_map": citation_map,
            "style": style,
            "count": len(simulated_citations),
            "simulated": True,
        }

    try:
        from tldw_Server_API.app.core.RAG.rag_service.citations import (
            CitationGenerator,
            CitationStyle,
        )
        from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document

        # Map style string to enum
        style_enum_map = {
            "mla": CitationStyle.MLA,
            "apa": CitationStyle.APA,
            "chicago": CitationStyle.CHICAGO,
            "harvard": CitationStyle.HARVARD,
            "ieee": CitationStyle.IEEE,
        }
        style_enum = style_enum_map.get(style, CitationStyle.APA)

        # Convert input documents to Document objects
        doc_objects: list[Document] = []
        for i, doc in enumerate(documents[:max_citations]):
            content = doc.get("content") or doc.get("text") or str(doc)
            metadata = doc.get("metadata") or {}

            # Ensure metadata has citation-relevant fields
            if "title" not in metadata and "title" in doc:
                metadata["title"] = doc["title"]
            if "author" not in metadata and "author" in doc:
                metadata["author"] = doc["author"]
            if "date" not in metadata and "date" in doc:
                metadata["date"] = doc["date"]

            doc_obj = Document(
                id=doc.get("id") or f"doc_{i}",
                content=content,
                metadata=metadata,
                source=DataSource.WEB_CONTENT,
                score=float(doc.get("score") or 0.5),
                source_document_id=doc.get("source_id") or doc.get("source_document_id"),
            )
            doc_objects.append(doc_obj)

        # Create generator
        generator = CitationGenerator()

        # Generate citations
        result = await generator.generate_citations(
            documents=doc_objects,
            query=query,
            style=style_enum,
            include_chunks=include_inline,
            max_citations=max_citations,
        )

        # Convert chunk citations to dicts
        chunk_citations_out = []
        for cc in result.chunk_citations:
            chunk_citations_out.append(cc.to_dict() if hasattr(cc, 'to_dict') else {
                "chunk_id": cc.chunk_id,
                "source_document_id": cc.source_document_id,
                "source_document_title": cc.source_document_title,
                "location": cc.location,
                "text_snippet": cc.text_snippet,
                "confidence": cc.confidence,
                "usage_context": cc.usage_context,
            })

        return {
            "citations": result.academic_citations,
            "chunk_citations": chunk_citations_out,
            "inline_markers": result.inline_markers,
            "citation_map": result.citation_map,
            "style": style,
            "count": len(result.academic_citations),
        }

    except Exception:
        logger.exception("Citations adapter error")
        return {"error": "citations_error"}


@registry.register(
    "bibliography_generate",
    category="content",
    description="Generate bibliography",
    parallelizable=True,
    tags=["content", "citations"],
    config_model=BibliographyGenerateConfig,
)
async def run_bibliography_generate_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Generate formatted bibliography/citations from sources.

    Config:
      - sources: list[dict] - Source documents with metadata
      - format: str - Citation format: "apa", "mla", "chicago", "harvard", "bibtex" (default: "apa")
      - sort_by: str - Sort order: "author", "date", "title" (default: "author")
    Output:
      - bibliography: str - Formatted bibliography
      - citations: list[dict] - Individual citations with keys
      - format: str
      - count: int
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    sources = config.get("sources")
    if not sources:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            sources = prev.get("documents") or prev.get("sources") or prev.get("results") or []

    if not isinstance(sources, list) or not sources:
        return {"error": "missing_sources", "bibliography": "", "citations": [], "count": 0}

    citation_format = str(config.get("format", "apa")).lower()
    sort_by = str(config.get("sort_by", "author")).lower()

    # Extract citation metadata from sources
    citations = []
    for i, source in enumerate(sources):
        if not isinstance(source, dict):
            continue

        citation = {
            "key": source.get("id") or source.get("key") or f"source_{i+1}",
            "title": source.get("title") or source.get("name") or "Untitled",
            "author": source.get("author") or source.get("authors") or "Unknown",
            "date": source.get("date") or source.get("published") or source.get("year") or "",
            "url": source.get("url") or source.get("link") or "",
            "type": source.get("type") or "document",
            "publisher": source.get("publisher") or source.get("source") or "",
            "pages": source.get("pages") or "",
        }
        citations.append(citation)

    # Sort citations
    if sort_by == "date":
        citations.sort(key=lambda x: str(x.get("date", "")), reverse=True)
    elif sort_by == "title":
        citations.sort(key=lambda x: str(x.get("title", "")).lower())
    else:  # author
        citations.sort(key=lambda x: str(x.get("author", "")).lower())

    # Format citations
    formatted_entries = []
    for cit in citations:
        author = cit["author"]
        if isinstance(author, list):
            author = ", ".join(author)

        title = cit["title"]
        date = cit["date"]
        url = cit["url"]
        publisher = cit["publisher"]

        if citation_format == "apa":
            entry = f"{author} ({date}). {title}."
            if publisher:
                entry += f" {publisher}."
            if url:
                entry += f" Retrieved from {url}"
        elif citation_format == "mla" or citation_format == "chicago":
            entry = f'{author}. "{title}."'
            if publisher:
                entry += f" {publisher},"
            if date:
                entry += f" {date}."
            if url:
                entry += f" {url}."
        elif citation_format == "harvard":
            entry = f"{author} ({date}) {title}."
            if publisher:
                entry += f" {publisher}."
            if url:
                entry += f" Available at: {url}"
        elif citation_format == "bibtex":
            key = cit["key"].replace(" ", "_")
            entry = f"@misc{{{key},\n"
            entry += f"  author = {{{author}}},\n"
            entry += f"  title = {{{title}}},\n"
            if date:
                entry += f"  year = {{{date}}},\n"
            if url:
                entry += f"  url = {{{url}}},\n"
            entry = entry.rstrip(",\n") + "\n}"
        else:
            entry = f"{author}. {title}. {date}."

        cit["formatted"] = entry
        formatted_entries.append(entry)

    bibliography = "\n\n".join(formatted_entries)

    return {
        "bibliography": bibliography,
        "citations": citations,
        "format": citation_format,
        "count": len(citations),
    }
