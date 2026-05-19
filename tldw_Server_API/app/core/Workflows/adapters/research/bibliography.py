"""Research bibliography and citation adapters.

This module includes adapters for citation and bibliography operations:
- doi_resolve: Resolve DOI to metadata
- reference_parse: Parse citation strings
- bibtex_generate: Generate BibTeX entries
- literature_review: Generate literature review summaries
"""

from __future__ import annotations

import json
import os
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.Workflows.adapters._common import _extract_openai_content
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.research._config import (
    BibtexGenerateConfig,
    DOIResolveConfig,
    LiteratureReviewConfig,
    ReferenceParseConfig,
)


@registry.register(
    "doi_resolve",
    category="research",
    description="Resolve DOI",
    parallelizable=True,
    tags=["research", "citations"],
    config_model=DOIResolveConfig,
)
async def run_doi_resolve_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Resolve DOI to metadata.

    Config:
      - doi: str - DOI to resolve (e.g., "10.1000/xyz123")
    Output:
      - metadata: dict - Paper metadata
      - resolved: bool
    """
    import httpx

    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    doi = config.get("doi") or ""
    if isinstance(doi, str):
        doi = _tmpl(doi, context) or doi

    if not doi:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            doi = prev.get("doi") or ""

    if not doi:
        return {"metadata": {}, "error": "missing_doi", "resolved": False}

    # Clean DOI
    doi = doi.strip()
    if doi.startswith("https://doi.org/"):
        doi = doi[16:]
    elif doi.startswith("http://doi.org/"):
        doi = doi[15:]
    elif doi.startswith("doi:"):
        doi = doi[4:]

    # TEST_MODE: return simulated result without network call
    if is_test_mode():
        return {
            "metadata": {
                "doi": doi,
                "title": f"Simulated Paper for DOI {doi}",
                "authors": ["Test Author", "Another Author"],
                "journal": "Simulated Journal",
                "year": 2023,
                "volume": "1",
                "issue": "1",
                "pages": "1-10",
                "publisher": "Simulated Publisher",
                "url": f"https://doi.org/{doi}",
            },
            "resolved": True,
            "simulated": True,
        }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://doi.org/{doi}",
                headers={"Accept": "application/vnd.citationstyles.csl+json"},
                follow_redirects=True,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            metadata = {
                "doi": doi,
                "title": data.get("title", ""),
                "authors": [
                    f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in data.get("author", [])
                ],
                "container_title": data.get("container-title", ""),
                "publisher": data.get("publisher", ""),
                "issued": data.get("issued", {}).get("date-parts", [[]])[0],
                "type": data.get("type", ""),
                "abstract": data.get("abstract", ""),
                "url": data.get("URL", ""),
            }

            return {"metadata": metadata, "resolved": True}

    except Exception:
        logger.error("DOI resolve error")
        return {"metadata": {}, "error": "DOI resolution failed", "resolved": False}


@registry.register(
    "reference_parse",
    category="research",
    description="Parse references",
    parallelizable=True,
    tags=["research", "citations"],
    config_model=ReferenceParseConfig,
)
async def run_reference_parse_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Parse citation string to structured data.

    Config:
      - citation: str - Citation string to parse
      - provider: str - LLM provider (for parsing)
      - model: str - Model to use
    Output:
      - parsed: dict - Structured citation data
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    citation = config.get("citation") or ""
    if isinstance(citation, str):
        citation = _tmpl(citation, context) or citation

    if not citation:
        return {"parsed": {}, "error": "missing_citation"}

    try:
        from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

        prompt = f"""Parse this citation into structured JSON with these fields:
- authors: list of author names
- title: paper/article title
- journal: journal or publication name
- year: publication year
- volume: volume number
- issue: issue number
- pages: page range
- doi: DOI if present
- url: URL if present

Citation: {citation}

Return JSON only."""

        messages = [{"role": "user", "content": prompt}]
        response = await perform_chat_api_call_async(
            messages=messages,
            api_provider=config.get("provider"),
            model=config.get("model"),
            system_message="Parse citations into structured JSON.",
            max_tokens=500,
            temperature=0.1,
        )

        result_text = _extract_openai_content(response) or ""
        try:
            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(result_text[start:end])
                return {"parsed": parsed}
        except json.JSONDecodeError:
            pass

        return {"parsed": {}, "raw_text": result_text}

    except Exception:
        logger.error("Reference parse error")
        return {"parsed": {}, "error": "Reference parsing failed"}


@registry.register(
    "bibtex_generate",
    category="research",
    description="Generate BibTeX",
    parallelizable=True,
    tags=["research", "citations"],
    config_model=BibtexGenerateConfig,
)
async def run_bibtex_generate_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Generate BibTeX entry from metadata.

    Config:
      - metadata: dict - Paper metadata (title, authors, year, etc.)
      - entry_type: str - BibTeX entry type (article, book, inproceedings)
      - cite_key: str - Citation key (auto-generated if not provided)
    Output:
      - bibtex: str
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    metadata = config.get("metadata") or {}

    if not metadata:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            metadata = prev.get("metadata") or prev.get("parsed") or {}

    if not metadata:
        return {"bibtex": "", "error": "missing_metadata"}

    entry_type = config.get("entry_type", "article")
    cite_key = config.get("cite_key")

    # Auto-generate cite key
    if not cite_key:
        authors = metadata.get("authors", [])
        first_author = authors[0].split()[-1] if authors else "unknown"
        year = metadata.get("year", "")
        cite_key = f"{first_author.lower()}{year}"

    # Build BibTeX
    lines = [f"@{entry_type}{{{cite_key},"]

    field_map = {
        "title": "title",
        "authors": "author",
        "journal": "journal",
        "year": "year",
        "volume": "volume",
        "number": "number",
        "pages": "pages",
        "doi": "doi",
        "url": "url",
        "publisher": "publisher",
        "booktitle": "booktitle",
        "abstract": "abstract",
    }

    for meta_key, bib_key in field_map.items():
        value = metadata.get(meta_key)
        if value:
            if isinstance(value, list):
                value = " and ".join(value)
            lines.append(f"  {bib_key} = {{{value}}},")

    lines.append("}")
    bibtex = "\n".join(lines)

    return {"bibtex": bibtex, "text": bibtex, "cite_key": cite_key}


@registry.register(
    "literature_review",
    category="research",
    description="Generate literature review",
    parallelizable=True,
    tags=["research", "academic"],
    config_model=LiteratureReviewConfig,
)
async def run_literature_review_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Generate literature review summary from search results.

    Config:
      - papers: list[dict] - Papers to summarize
      - topic: str - Review topic
      - style: str - "brief", "detailed", "comparative" (default: "brief")
      - provider: str - LLM provider
      - model: str - Model to use
    Output:
      - review: str
      - paper_count: int
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    papers = config.get("papers") or []

    if not papers:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            papers = prev.get("papers") or []

    if not papers:
        return {"review": "", "error": "missing_papers"}

    topic = config.get("topic", "")
    if isinstance(topic, str):
        topic = _tmpl(topic, context) or topic

    style = config.get("style", "brief")

    # Format papers for prompt
    papers_text = ""
    for i, paper in enumerate(papers[:15]):
        title = paper.get("title", "")
        authors = paper.get("authors", [])
        if isinstance(authors, list):
            authors = ", ".join(authors[:3])
        year = paper.get("year", "")
        abstract = paper.get("abstract", paper.get("summary", ""))[:500]
        papers_text += f"\n{i + 1}. {title} ({year})\nAuthors: {authors}\nAbstract: {abstract}\n"

    style_instructions = {
        "brief": "Write a concise 2-3 paragraph overview.",
        "detailed": "Write a comprehensive review with sections for themes, gaps, and future directions.",
        "comparative": "Compare and contrast the different approaches and findings.",
    }

    try:
        from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

        prompt = f"""Generate a literature review {f'on the topic of "{topic}"' if topic else ''} based on these papers:

{papers_text}

{style_instructions.get(style, style_instructions['brief'])}"""

        messages = [{"role": "user", "content": prompt}]
        response = await perform_chat_api_call_async(
            messages=messages,
            api_provider=config.get("provider"),
            model=config.get("model"),
            system_message="Write academic literature reviews.",
            max_tokens=2000,
            temperature=0.5,
        )

        review = _extract_openai_content(response) or ""
        return {"review": review, "text": review, "paper_count": len(papers), "style": style}

    except Exception:
        logger.error("Literature review error")
        return {"review": "", "error": "Literature review generation failed"}
