"""Research search adapters.

This module includes adapters for academic search operations:
- arxiv_search: Search arXiv
- arxiv_download: Download from arXiv
- pubmed_search: Search PubMed
- semantic_scholar_search: Search Semantic Scholar
- google_scholar_search: Search Google Scholar
- patent_search: Search patents
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.testing import is_test_mode as _is_test_mode
from tldw_Server_API.app.core.Workflows.adapters._common import _resolve_artifacts_dir
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.research._config import (
    ArxivDownloadConfig,
    ArxivSearchConfig,
    GoogleScholarSearchConfig,
    PatentSearchConfig,
    PubmedSearchConfig,
    SemanticScholarSearchConfig,
)


@registry.register(
    "arxiv_search",
    category="research",
    description="Search arXiv",
    parallelizable=True,
    tags=["research", "academic"],
    config_model=ArxivSearchConfig,
)
async def run_arxiv_search_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Search arXiv for papers.

    Config:
      - query: str - Search query
      - max_results: int - Maximum results (default: 10)
      - sort_by: str - "relevance", "lastUpdatedDate", "submittedDate"
      - sort_order: str - "ascending", "descending"
    Output:
      - papers: list[dict] - Paper metadata
      - total_results: int
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    query = config.get("query") or ""
    if isinstance(query, str):
        query = _tmpl(query, context) or query

    if not query:
        return {"papers": [], "error": "missing_query"}

    # TEST_MODE: return simulated results without network call
    if _is_test_mode():
        return {
            "papers": [
                {
                    "arxiv_id": "2301.00001",
                    "title": f"Simulated Paper on {query}",
                    "authors": ["Test Author", "Another Author"],
                    "summary": f"This is a simulated paper about {query}.",
                    "published": "2023-01-01T00:00:00",
                    "updated": "2023-01-02T00:00:00",
                    "pdf_url": "https://arxiv.org/pdf/2301.00001.pdf",
                    "categories": ["cs.AI", "cs.LG"],
                    "doi": None,
                }
            ],
            "total_results": 1,
            "query": query,
            "simulated": True,
        }

    max_results = int(config.get("max_results", 10))
    sort_by = config.get("sort_by", "relevance")
    sort_order = config.get("sort_order", "descending")

    try:
        import arxiv

        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
        }
        order_map = {
            "ascending": arxiv.SortOrder.Ascending,
            "descending": arxiv.SortOrder.Descending,
        }

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_map.get(sort_by, arxiv.SortCriterion.Relevance),
            sort_order=order_map.get(sort_order, arxiv.SortOrder.Descending),
        )

        papers = []
        for result in search.results():
            papers.append({
                "arxiv_id": result.entry_id.split("/")[-1],
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "summary": result.summary,
                "published": result.published.isoformat() if result.published else None,
                "updated": result.updated.isoformat() if result.updated else None,
                "pdf_url": result.pdf_url,
                "categories": result.categories,
                "doi": result.doi,
            })

        return {"papers": papers, "total_results": len(papers), "query": query}

    except ImportError:
        return {"papers": [], "error": "arxiv_library_not_installed"}
    except Exception:
        logger.error("arXiv search error")
        return {"papers": [], "error": "arXiv search failed"}


@registry.register(
    "arxiv_download",
    category="research",
    description="Download from arXiv",
    parallelizable=False,
    tags=["research", "academic"],
    config_model=ArxivDownloadConfig,
)
async def run_arxiv_download_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Download paper PDF from arXiv.

    Config:
      - arxiv_id: str - arXiv paper ID (e.g., "2301.00001")
      - pdf_url: str - Direct PDF URL (alternative)
    Output:
      - pdf_path: str
      - downloaded: bool
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    arxiv_id = config.get("arxiv_id") or ""
    pdf_url = config.get("pdf_url") or ""

    if isinstance(arxiv_id, str):
        arxiv_id = _tmpl(arxiv_id, context) or arxiv_id

    if not arxiv_id and not pdf_url:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            arxiv_id = prev.get("arxiv_id") or ""
            pdf_url = prev.get("pdf_url") or ""

    if not arxiv_id and not pdf_url:
        return {"error": "missing_arxiv_id_or_pdf_url", "downloaded": False}

    # TEST_MODE: return simulated result without actual download
    if _is_test_mode():
        return {
            "pdf_path": str(Path(tempfile.gettempdir()) / f"simulated_{arxiv_id or 'paper'}.pdf"),
            "downloaded": True,
            "arxiv_id": arxiv_id,
            "simulated": True,
        }

    step_run_id = str(context.get("step_run_id") or f"arxiv_download_{int(time.time() * 1000)}")
    art_dir = _resolve_artifacts_dir(step_run_id)
    art_dir.mkdir(parents=True, exist_ok=True)

    try:
        if arxiv_id:
            import arxiv
            paper = next(arxiv.Search(id_list=[arxiv_id]).results())
            filename = f"{arxiv_id.replace('/', '_')}.pdf"
            output_path = str(art_dir / filename)
            paper.download_pdf(dirpath=str(art_dir), filename=filename)
        else:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(pdf_url, follow_redirects=True, timeout=60)
                response.raise_for_status()
                filename = pdf_url.split("/")[-1] or "paper.pdf"
                output_path = str(art_dir / filename)
                Path(output_path).write_bytes(response.content)

        if callable(context.get("add_artifact")):
            context["add_artifact"](
                type="pdf",
                uri=f"file://{output_path}",
                mime_type="application/pdf",
            )

        return {"pdf_path": output_path, "downloaded": True, "arxiv_id": arxiv_id}

    except ImportError:
        return {"error": "arxiv_library_not_installed", "downloaded": False}
    except Exception:
        logger.error("arXiv download error")
        return {"error": "arXiv download failed", "downloaded": False}


@registry.register(
    "pubmed_search",
    category="research",
    description="Search PubMed",
    parallelizable=True,
    tags=["research", "academic"],
    config_model=PubmedSearchConfig,
)
async def run_pubmed_search_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Search PubMed for biomedical papers.

    Config:
      - query: str - Search query
      - max_results: int - Maximum results (default: 10)
    Output:
      - papers: list[dict]
      - total_results: int
    """
    import httpx

    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    query = config.get("query") or ""
    if isinstance(query, str):
        query = _tmpl(query, context) or query

    if not query:
        return {"papers": [], "error": "missing_query"}

    # TEST_MODE: return simulated results without network call
    if _is_test_mode():
        return {
            "papers": [
                {
                    "pmid": "12345678",
                    "title": f"Simulated PubMed Paper on {query}",
                    "authors": ["Test Author", "Medical Researcher"],
                    "source": "Test Journal",
                    "pubdate": "2023 Jan",
                    "doi": "10.1000/simulated",
                }
            ],
            "total_results": 1,
            "query": query,
            "simulated": True,
        }

    max_results = int(config.get("max_results", 10))

    try:
        # Use NCBI E-utilities
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

        async with httpx.AsyncClient() as client:
            # Search for IDs
            search_url = f"{base_url}/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
            }
            search_response = await client.get(search_url, params=search_params, timeout=30)
            search_data = search_response.json()

            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return {"papers": [], "total_results": 0, "query": query}

            # Fetch details
            fetch_url = f"{base_url}/esummary.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "json",
            }
            fetch_response = await client.get(fetch_url, params=fetch_params, timeout=30)
            fetch_data = fetch_response.json()

            papers = []
            result_data = fetch_data.get("result", {})
            for pmid in id_list:
                if pmid in result_data:
                    paper = result_data[pmid]
                    papers.append({
                        "pmid": pmid,
                        "title": paper.get("title", ""),
                        "authors": [a.get("name", "") for a in paper.get("authors", [])],
                        "source": paper.get("source", ""),
                        "pubdate": paper.get("pubdate", ""),
                        "doi": paper.get("elocationid", ""),
                    })

            return {"papers": papers, "total_results": len(papers), "query": query}

    except Exception:
        logger.error("PubMed search error")
        return {"papers": [], "error": "PubMed search failed"}


@registry.register(
    "semantic_scholar_search",
    category="research",
    description="Search Semantic Scholar",
    parallelizable=True,
    tags=["research", "academic"],
    config_model=SemanticScholarSearchConfig,
)
async def run_semantic_scholar_search_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Search Semantic Scholar for papers.

    Config:
      - query: str - Search query
      - max_results: int - Maximum results (default: 10)
      - fields: list[str] - Fields to return
    Output:
      - papers: list[dict]
      - total_results: int
    """
    import httpx

    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    query = config.get("query") or ""
    if isinstance(query, str):
        query = _tmpl(query, context) or query

    if not query:
        return {"papers": [], "error": "missing_query"}

    # TEST_MODE: return simulated results without network call
    if _is_test_mode():
        return {
            "papers": [
                {
                    "paper_id": "abc123",
                    "title": f"Simulated Semantic Scholar Paper on {query}",
                    "authors": ["Test Author", "AI Researcher"],
                    "abstract": f"This is a simulated abstract about {query}.",
                    "year": 2023,
                    "citation_count": 42,
                    "url": "https://www.semanticscholar.org/paper/abc123",
                }
            ],
            "total_results": 1,
            "query": query,
            "simulated": True,
        }

    max_results = int(config.get("max_results", 10))
    fields = config.get("fields") or ["title", "authors", "abstract", "year", "citationCount", "url"]

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": query,
                    "limit": max_results,
                    "fields": ",".join(fields),
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            papers = []
            for paper in data.get("data", []):
                papers.append({
                    "paper_id": paper.get("paperId"),
                    "title": paper.get("title"),
                    "authors": [a.get("name", "") for a in paper.get("authors", [])],
                    "abstract": paper.get("abstract"),
                    "year": paper.get("year"),
                    "citation_count": paper.get("citationCount"),
                    "url": paper.get("url"),
                })

            return {"papers": papers, "total_results": data.get("total", len(papers)), "query": query}

    except Exception:
        logger.error("Semantic Scholar search error")
        return {"papers": [], "error": "Semantic Scholar search failed"}


@registry.register(
    "google_scholar_search",
    category="research",
    description="Search Google Scholar",
    parallelizable=True,
    tags=["research", "academic"],
    config_model=GoogleScholarSearchConfig,
)
async def run_google_scholar_search_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Search Google Scholar for papers.

    Config:
      - query: str - Search query
      - max_results: int - Maximum results (default: 10)
    Output:
      - papers: list[dict]
      - total_results: int
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    query = config.get("query") or ""
    if isinstance(query, str):
        query = _tmpl(query, context) or query

    if not query:
        return {"papers": [], "error": "missing_query"}

    # TEST_MODE: return simulated results without network call (Google Scholar is rate-limited)
    if _is_test_mode():
        return {
            "papers": [
                {
                    "title": f"Simulated Google Scholar Paper on {query}",
                    "authors": ["Test Author", "Scholar Researcher"],
                    "abstract": f"This is a simulated abstract about {query}.",
                    "year": "2023",
                    "venue": "Simulated Conference",
                    "citation_count": 100,
                    "url": "https://scholar.google.com/simulated",
                }
            ],
            "total_results": 1,
            "query": query,
            "simulated": True,
        }

    max_results = int(config.get("max_results", 10))

    try:
        from scholarly import scholarly

        search_query = scholarly.search_pubs(query)
        papers = []

        for i, result in enumerate(search_query):
            if i >= max_results:
                break
            papers.append({
                "title": result.get("bib", {}).get("title", ""),
                "authors": result.get("bib", {}).get("author", []),
                "abstract": result.get("bib", {}).get("abstract", ""),
                "year": result.get("bib", {}).get("pub_year", ""),
                "venue": result.get("bib", {}).get("venue", ""),
                "citation_count": result.get("num_citations", 0),
                "url": result.get("pub_url", ""),
            })

        return {"papers": papers, "total_results": len(papers), "query": query}

    except ImportError:
        return {"papers": [], "error": "scholarly_library_not_installed"}
    except Exception:
        logger.error("Google Scholar search error")
        return {"papers": [], "error": "Google Scholar search failed"}


@registry.register(
    "patent_search",
    category="research",
    description="Search patents",
    parallelizable=True,
    tags=["research", "patents"],
    config_model=PatentSearchConfig,
)
async def run_patent_search_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Search patent databases.

    Config:
      - query: str - Search query
      - max_results: int - Maximum results (default: 10)
      - database: str - "google_patents" (default)
    Output:
      - patents: list[dict]
      - total_results: int
    """
    import httpx

    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    query = config.get("query") or ""
    if isinstance(query, str):
        query = _tmpl(query, context) or query

    if not query:
        return {"patents": [], "error": "missing_query"}

    # TEST_MODE: return simulated results without network call
    if _is_test_mode():
        return {
            "patents": [
                {
                    "patent_id": "US-12345678-A1",
                    "title": f"Simulated Patent on {query}",
                    "assignee": "Test Corporation",
                    "inventors": ["Test Inventor"],
                    "filing_date": "2023-01-15",
                    "publication_date": "2023-07-15",
                    "abstract": f"This is a simulated patent about {query}.",
                    "url": "https://patents.google.com/patent/US12345678A1",
                }
            ],
            "total_results": 1,
            "query": query,
            "simulated": True,
        }

    max_results = int(config.get("max_results", 10))

    # Use Google Patents search via web scraping or API
    try:
        import urllib.parse

        async with httpx.AsyncClient() as client:
            encoded_query = urllib.parse.quote(query)
            url = f"https://patents.google.com/xhr/query?url=q%3D{encoded_query}&num={max_results}"

            response = await client.get(url, timeout=30, headers={"Accept": "application/json"})

            if response.status_code == 200:
                try:
                    data = response.json()
                    patents = []
                    for result in data.get("results", {}).get("cluster", [])[:max_results]:
                        patent = result.get("result", {}).get("patent", {})
                        patents.append({
                            "patent_id": patent.get("publication_number", ""),
                            "title": patent.get("title", ""),
                            "abstract": patent.get("abstract", ""),
                            "assignee": patent.get("assignee", ""),
                            "filing_date": patent.get("filing_date", ""),
                            "publication_date": patent.get("publication_date", ""),
                        })
                    return {"patents": patents, "total_results": len(patents), "query": query}
                except json.JSONDecodeError:
                    pass

            return {"patents": [], "total_results": 0, "query": query, "note": "google_patents_api_unavailable"}

    except Exception:
        logger.error("Patent search error")
        return {"patents": [], "error": "Patent search failed"}
