# Semantic_Scholar.py
# Description: This file contains the functions to interact with the Semantic Scholar API
#
# Imports
from typing import List, Dict, Any

import requests
#
####################################################################################################
#
# Functions

# Constants
FIELDS_OF_STUDY = [
    "Computer Science", "Medicine", "Chemistry", "Biology", "Materials Science",
    "Physics", "Geology", "Psychology", "Art", "History", "Geography",
    "Sociology", "Business", "Political Science", "Economics", "Philosophy",
    "Mathematics", "Engineering", "Environmental Science",
    "Agricultural and Food Sciences", "Education", "Law", "Linguistics"
]

PUBLICATION_TYPES = [
    "Review", "JournalArticle", "CaseReport", "ClinicalTrial", "Conference",
    "Dataset", "Editorial", "LettersAndComments", "MetaAnalysis", "News",
    "Study", "Book", "BookSection"
]


def search_papers(
        query: str,
        page: int,
        fields_of_study: List[str],
        publication_types: List[str],
        year_range: str,
        venue: str,
        min_citations: int,
        open_access_only: bool,
        limit: int = 10
) -> Dict[str, Any]:
    """Search for papers using the Semantic Scholar API with all available filters"""
    if not query.strip():
        return {"total": 0, "offset": 0, "next": 0, "data": []}

    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "offset": page * limit,
            "limit": limit,
            "fields": "title,abstract,year,citationCount,authors,venue,openAccessPdf,url,publicationTypes,publicationDate"
        }

        # Add optional filters
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        if publication_types:
            params["publicationTypes"] = ",".join(publication_types)
        if venue:
            params["venue"] = venue
        if min_citations:
            params["minCitationCount"] = str(min_citations)
        if open_access_only:
            params["openAccessPdf"] = ""
        if year_range:
            try:
                if "-" in year_range:
                    start_year, end_year = year_range.split("-")
                    params["year"] = f"{start_year.strip()}-{end_year.strip()}"
                else:
                    params["year"] = year_range.strip()
            except ValueError:
                pass

        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API Error: {str(e)}", "total": 0, "offset": 0, "data": []}


def get_paper_details(paper_id):
    """Get detailed information about a specific paper"""
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
        params = {
            "fields": "title,abstract,year,citationCount,authors,venue,openAccessPdf,url,references,citations"
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API Error: {str(e)}"}


def format_paper_info(paper: Dict[str, Any]) -> str:
    """Format paper information for display"""
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


def search_and_display(
        query: str,
        page: int,
        fields_of_study: List[str],
        publication_types: List[str],
        year_range: str,
        venue: str,
        min_citations: int,
        open_access_only: bool
) -> tuple[str, int, int, str]:
    """Search for papers and return formatted results with pagination info"""
    result = search_papers(
        query, page, fields_of_study, publication_types,
        year_range, venue, min_citations, open_access_only
    )

    if "error" in result:
        return result["error"], 0, 0, "0"

    if not result["data"]:
        return "No results found.", 0, 0, "0"

    papers = result["data"]
    total_results = int(result.get("total", "0"))
    max_pages = (total_results + 9) // 10  # Ceiling division

    results = []
    for paper in papers:
        results.append(format_paper_info(paper))

    formatted_results = "\n\n---\n\n".join(results)

    # Add pagination information
    pagination_info = f"\n\n---\n\nShowing results {result['offset'] + 1}-{result['offset'] + len(papers)} of {total_results}"

    return formatted_results + pagination_info, page, max_pages - 1, str(total_results)

#
# End of Semantic_Scholar.py
####################################################################################################
