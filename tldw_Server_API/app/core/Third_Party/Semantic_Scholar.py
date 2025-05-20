# Semantic_Scholar.py
# Description: This file contains the functions to interact with the Semantic Scholar API
#
# Imports
from typing import List, Dict, Any, Optional, Tuple  # Added Optional, Tuple
import time  # For potential delays if Semantic Scholar API has strict rate limits
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry  # Correct import for Retry
#
####################################################################################################
#
# Functions

# Constants (keep these, they are useful for API documentation or default values)
FIELDS_OF_STUDY_CHOICES = [  # Renamed for clarity if used in API docs
    "Computer Science", "Medicine", "Chemistry", "Biology", "Materials Science",
    "Physics", "Geology", "Psychology", "Art", "History", "Geography",
    "Sociology", "Business", "Political Science", "Economics", "Philosophy",
    "Mathematics", "Engineering", "Environmental Science",
    "Agricultural and Food Sciences", "Education", "Law", "Linguistics"
]

PUBLICATION_TYPE_CHOICES = [  # Renamed for clarity
    "Review", "JournalArticle", "CaseReport", "ClinicalTrial", "Conference",
    "Dataset", "Editorial", "LettersAndComments", "MetaAnalysis", "News",
    "Study", "Book", "BookSection"
]

SEMANTIC_SCHOLAR_API_BASE_URL = "https://api.semanticscholar.org/graph/v1"
DEFAULT_SEARCH_FIELDS = "paperId,title,abstract,year,citationCount,authors,venue,openAccessPdf,url,publicationTypes,publicationDate,externalIds"  # Added paperId and externalIds


def search_papers_semantic_scholar(  # Renamed for clarity
        query: str,
        offset: int = 0,
        limit: int = 10,
        fields_of_study: Optional[List[str]] = None,
        publication_types: Optional[List[str]] = None,
        year_range: Optional[str] = None,  # e.g., "2019-2021" or "2020"
        venue: Optional[List[str]] = None,  # API takes a list of venues
        min_citations: Optional[int] = None,
        open_access_only: bool = False,
        fields_to_return: str = DEFAULT_SEARCH_FIELDS
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:  # Return (data, error_message)
    """
    Search for papers using the Semantic Scholar API with available filters.
    Returns the JSON response from the API or an error message.
    """
    if not query or not query.strip():
        # Return a structure similar to a successful empty search, and no error message
        return {"total": 0, "offset": offset, "next": offset, "data": []}, None

    try:
        url = f"{SEMANTIC_SCHOLAR_API_BASE_URL}/paper/search"
        params = {
            "query": query,
            "offset": offset,
            "limit": limit,
            "fields": fields_to_return
        }

        # Add optional filters
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)  # API expects comma-separated string
        if publication_types:
            params["publicationTypes"] = ",".join(publication_types)  # API expects comma-separated string
        if venue:
            params["venue"] = ",".join(venue)  # API expects comma-separated string for venues
        if min_citations is not None and min_citations >= 0:  # API expects non-negative
            params["minCitationCount"] = str(min_citations)
        if open_access_only:  # If true, the API expects the parameter to be present (value doesn't matter for the flag)
            # The Semantic Scholar API docs are a bit ambiguous on boolean flags.
            # For "openAccessPdf", if you want *only* open access, you specify it.
            # If you want papers *that have* an openAccessPdf, it's usually included in fields.
            # The "search" endpoint doesn't have a boolean "openAccessOnly" flag directly.
            # Instead, you request "openAccessPdf" in fields and filter client-side,
            # OR if the API supported it as a filter you'd pass it.
            # Let's assume for now it means we filter results that *have* an openAccessPdf.
            # This parameter is tricky for search. `get_paper_details` is where `openAccessPdf` field shines.
            # For search, it's better to request `openAccessPdf` in `fields` and filter later if strictly "only".
            # The query parameter `openAccessPdf` (empty value) is not standard for boolean filtering in S2 API.
            # We will rely on requesting the field `openAccessPdf` and the client can see if it's present.
            pass  # The field `openAccessPdf` is already in DEFAULT_SEARCH_FIELDS

        if year_range:
            try:
                # Validate year format "YYYY" or "YYYY-YYYY"
                if "-" in year_range:
                    start_year, end_year = year_range.split("-")
                    if start_year.strip().isdigit() and end_year.strip().isdigit() and len(
                            start_year.strip()) == 4 and len(end_year.strip()) == 4:
                        params["year"] = f"{start_year.strip()}-{end_year.strip()}"
                    else:
                        # Invalid range format, could log a warning or ignore
                        print(f"Warning: Invalid year range format: {year_range}. Ignoring.")
                elif year_range.strip().isdigit() and len(year_range.strip()) == 4:
                    params["year"] = year_range.strip()
                else:
                    # Invalid single year format
                    print(f"Warning: Invalid year format: {year_range}. Ignoring.")
            except ValueError:
                print(f"Warning: Could not parse year range: {year_range}. Ignoring.")
                pass  # Ignore if parsing fails

        # Configure retry strategy for requests
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
            allowed_methods=["HEAD", "GET", "OPTIONS"],  # Methods to retry
            backoff_factor=1  # sleep for {backoff factor} * (2 ** ({number of total retries} - 1))
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http_session = requests.Session()
        http_session.mount("https://", adapter)
        http_session.mount("http://", adapter)

        response = http_session.get(url, params=params, timeout=15)  # Added timeout
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)

        # Optional: small delay if making many calls, though retry handles 429
        # time.sleep(0.2)

        return response.json(), None
    except requests.exceptions.Timeout:
        error_msg = f"Request to Semantic Scholar API timed out. URL: {url} Params: {params}"
        # print(f"**Error:** {error_msg}") # Logging handled by API endpoint
        return None, error_msg
    except requests.exceptions.HTTPError as e:
        error_msg = f"Semantic Scholar API HTTP Error: {e.response.status_code} - {e.response.text}. URL: {e.request.url}"
        # print(f"**Error:** {error_msg}")
        return None, error_msg
    except requests.exceptions.RequestException as e:
        error_msg = f"Semantic Scholar API Request Error: {e}. URL: {e.request.url if e.request else url}"
        # print(f"**Error:** {error_msg}")
        return None, error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred during Semantic Scholar search: {e}"
        # print(f"**Error:** {error_msg}")
        return None, error_msg


def get_paper_details_semantic_scholar(paper_id: str, fields_to_return: str = DEFAULT_SEARCH_FIELDS) -> Tuple[
    Optional[Dict[str, Any]], Optional[str]]:
    """Get detailed information about a specific paper."""
    if not paper_id or not paper_id.strip():
        return None, "Paper ID cannot be empty."
    try:
        url = f"{SEMANTIC_SCHOLAR_API_BASE_URL}/paper/{paper_id}"
        params = {"fields": fields_to_return}

        retry_strategy = Retry(total=3, status_forcelist=[429, 500, 502, 503, 504], backoff_factor=1)
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http_session = requests.Session()
        http_session.mount("https://", adapter)

        response = http_session.get(url, params=params, timeout=10)
        response.raise_for_status()
        # time.sleep(0.2)
        return response.json(), None
    except requests.exceptions.Timeout:
        error_msg = f"Request for paper details (ID: {paper_id}) timed out."
        return None, error_msg
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP Error fetching paper details (ID: {paper_id}): {e.response.status_code} - {e.response.text}"
        return None, error_msg
    except requests.exceptions.RequestException as e:
        error_msg = f"Request Error fetching paper details (ID: {paper_id}): {e}"
        return None, error_msg
    except Exception as e:
        error_msg = f"Unexpected error fetching paper details (ID: {paper_id}): {e}"
        return None, error_msg


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


# End of Semantic_Scholar.py
#######################################################################################################################
