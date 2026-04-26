"""
Semantic_Scholar.py

Adapter for the Semantic Scholar API using the centralized HTTP client.
"""

from typing import Any, Optional

from tldw_Server_API.app.core.http_client import fetch_json

FIELDS_OF_STUDY_CHOICES = [
    "Computer Science", "Medicine", "Chemistry", "Biology", "Materials Science",
    "Physics", "Geology", "Psychology", "Art", "History", "Geography",
    "Sociology", "Business", "Political Science", "Economics", "Philosophy",
    "Mathematics", "Engineering", "Environmental Science",
    "Agricultural and Food Sciences", "Education", "Law", "Linguistics"
]

PUBLICATION_TYPE_CHOICES = [
    "Review", "JournalArticle", "CaseReport", "ClinicalTrial", "Conference",
    "Dataset", "Editorial", "LettersAndComments", "MetaAnalysis", "News",
    "Study", "Book", "BookSection"
]

SEMANTIC_SCHOLAR_API_BASE_URL = "https://api.semanticscholar.org/graph/v1"
DEFAULT_SEARCH_FIELDS = (
    "paperId,title,abstract,year,citationCount,authors,venue,openAccessPdf,url,"
    "publicationTypes,publicationDate,externalIds"
)


def search_papers_semantic_scholar(
    query: str,
    offset: int = 0,
    limit: int = 10,
    fields_of_study: Optional[list[str]] = None,
    publication_types: Optional[list[str]] = None,
    year_range: Optional[str] = None,
    venue: Optional[list[str]] = None,
    min_citations: Optional[int] = None,
    open_access_only: bool = False,
    fields_to_return: str = DEFAULT_SEARCH_FIELDS,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Search for papers using the Semantic Scholar API with filters."""
    if not query or not query.strip():
        return {"total": 0, "offset": offset, "next": offset, "data": []}, None

    try:
        url = f"{SEMANTIC_SCHOLAR_API_BASE_URL}/paper/search"
        params: dict[str, Any] = {
            "query": query,
            "offset": offset,
            "limit": limit,
            "fields": fields_to_return,
        }
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        if publication_types:
            params["publicationTypes"] = ",".join(publication_types)
        if venue:
            params["venue"] = ",".join(venue)
        if min_citations is not None and min_citations >= 0:
            params["minCitationCount"] = str(min_citations)
        if open_access_only:
            # Already included openAccessPdf in fields; caller can filter client-side
            pass
        if year_range:
            try:
                yr = year_range.strip()
                if "-" in yr:
                    start_year, end_year = yr.split("-")
                    if start_year.isdigit() and end_year.isdigit() and len(start_year) == 4 and len(end_year) == 4:
                        params["year"] = yr
                elif yr.isdigit() and len(yr) == 4:
                    params["year"] = yr
            except Exception as year_parse_error:
                _ = year_parse_error  # keep request without year filter

        data = fetch_json(method="GET", url=url, params=params, timeout=15)
        # Optional pacing if needed; retries/backoff handled centrally
        # time.sleep(0.2)
        return data, None
    except Exception:
        return None, "Semantic Scholar search failed."


def get_paper_details_semantic_scholar(
    paper_id: str,
    fields_to_return: str = DEFAULT_SEARCH_FIELDS,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Get detailed information about a specific paper."""
    if not paper_id or not paper_id.strip():
        return None, "Paper ID cannot be empty."
    try:
        url = f"{SEMANTIC_SCHOLAR_API_BASE_URL}/paper/{paper_id}"
        params = {"fields": fields_to_return}
        data = fetch_json(method="GET", url=url, params=params, timeout=10)
        # time.sleep(0.2)
        return data, None
    except Exception:
        return None, "Semantic Scholar paper details lookup failed."


def format_paper_info(paper: dict[str, Any]) -> str:
    """Format paper information for display."""
    authors = ", ".join([author["name"] for author in paper.get("authors", [])])
    year = f"Year: {paper.get('year', 'N/A')}"
    venue = f"Venue: {paper.get('venue', 'N/A')}"
    citations = f"Citations: {paper.get('citationCount', 0)}"
    pub_types = f"Types: {', '.join(paper.get('publicationTypes', ['N/A']))}"

    pdf_link = ""
    if paper.get("openAccessPdf"):
        pdf_link = f"\nPDF: {paper['openAccessPdf']['url']}"

    s2_link = f"\nSemantic Scholar: {paper.get('url', '')}"

    formatted = f"""# {paper.get('title', 'No Title')}

Authors: {authors}
{year} | {venue} | {citations}
{pub_types}

Abstract:
{paper.get('abstract', 'No abstract available')}

Links:{pdf_link}{s2_link}
"""
    return formatted
