# Arxiv.py
# Description: This file contains the functions for searching and ingesting arXiv papers.
import time
from typing import Any, Optional  # Added for type hinting
from urllib.parse import quote_plus, urlencode

from bs4 import BeautifulSoup, FeatureNotFound

from tldw_Server_API.app.core.http_client import fetch

_ARXIV_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeError,
    ValueError,
)

#
# Local Imports (ensure path is correct if this file is moved/used elsewhere)
# from App_Function_Libraries.DB.DB_Manager import add_media_with_keywords
# For a search-only endpoint, add_media_with_keywords might not be directly used by the search function itself.
#
#####################################################################################################
#
# Functions:

# Default number of results per page if not specified by the caller
ARXIV_DEFAULT_PAGE_SIZE = 10


def fetch_arxiv_pdf_url(paper_id: str) -> Optional[str]:
    base_url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
    try:
        r = fetch(method="GET", url=base_url, timeout=10)
        if r.status_code >= 400:
            return None
        time.sleep(1)  # Keep a small delay, 2s might be too long for an API response time
        soup = BeautifulSoup(r.content, 'xml')  # Use response.content for bytes
        pdf_link_tag = soup.find('link', attrs={'title': 'pdf', 'rel': 'related', 'type': 'application/pdf'})
        if pdf_link_tag and pdf_link_tag.has_attr('href'):
            return pdf_link_tag['href']
        return None
    except _ARXIV_NONCRITICAL_EXCEPTIONS:
        print("**Error fetching arXiv PDF URL.**")
        return None


def search_arxiv_custom_api(query: Optional[str], author: Optional[str], year: Optional[str], start_index: int,
                            page_size: int) -> tuple[Optional[list[dict[str, Any]]], int, Optional[str]]:
    """
    Searches arXiv using the custom built URL and parses the feed.
    Returns a list of papers, total results found by the API for this query, and an error message if any.
    """
    query_url = build_query_url(query, author, year, start_index, page_size)

    try:
        r = fetch(method="GET", url=query_url, timeout=10)
        if r.status_code >= 400:
            return None, 0, f"arXiv API request failed: HTTP {r.status_code}"

        # Brief delay after successful request
        time.sleep(0.5)  # Reduced delay

        parsed_entries = parse_arxiv_feed(r.content)  # Pass response.content (bytes)

        soup = BeautifulSoup(r.content, 'xml')
        total_results_tag = soup.find('opensearch:totalResults')
        total_results = int(total_results_tag.text) if total_results_tag and total_results_tag.text.isdigit() else 0

        return parsed_entries, total_results, None
    except TimeoutError:
        error_msg = "arXiv API request timed out."
        print(f"**Error:** {error_msg}")
        return None, 0, error_msg
    except _ARXIV_NONCRITICAL_EXCEPTIONS:
        error_msg = "arXiv API request failed."
        print(f"**Error:** {error_msg}")
        return None, 0, error_msg


def fetch_arxiv_xml(paper_id: str) -> Optional[str]:
    base_url = "http://export.arxiv.org/api/query?id_list="
    try:
        r = fetch(method="GET", url=base_url + paper_id, timeout=10)
        if r.status_code >= 400:
            return None
        time.sleep(1)  # Keep delay
        return r.text
    except _ARXIV_NONCRITICAL_EXCEPTIONS:
        print("**Error fetching arXiv XML.**")
        return None


def parse_arxiv_feed(xml_content: bytes) -> list[dict[str, Any]]:
    try:
        soup = BeautifulSoup(xml_content, 'lxml-xml')
    except FeatureNotFound:
        print("Warning: Failed to use 'lxml-xml' parser. Falling back to Python's built-in 'xml' parser.")
        print("For potentially better performance and XML feature support, consider installing lxml: pip install lxml")
        soup = BeautifulSoup(xml_content, 'xml') # Fallback

    entries = []

    for entry_tag in soup.find_all('entry'):
        # Title
        title_tag = entry_tag.find('title')
        title = title_tag.text.strip() if title_tag and title_tag.text else None

        # Paper ID
        paper_id_xml_tag = entry_tag.find('id')
        paper_id = None
        if paper_id_xml_tag and paper_id_xml_tag.text:
            id_text = paper_id_xml_tag.text.strip()
            if '/abs/' in id_text:
                paper_id = id_text.split('/abs/')[-1]

        # Authors
        authors_list = []
        for author_tag in entry_tag.find_all('author'):
            name_tag = author_tag.find('name')
            if name_tag and name_tag.text:
                authors_list.append(name_tag.text.strip())
        authors_str = ', '.join(authors_list) if authors_list else None

        # Published Date
        published_tag = entry_tag.find('published')
        published_date = None
        if published_tag and published_tag.text:
            published_date = published_tag.text.strip().split('T')[0]

        # Abstract (summary)
        summary_tag = entry_tag.find('summary')
        abstract = summary_tag.text.strip() if summary_tag and summary_tag.text else None

        # Fetch PDF link
        pdf_url: Optional[str] = None
        pdf_link_tag = entry_tag.find('link', attrs={'title': 'pdf', 'type': 'application/pdf'})
        if pdf_link_tag and pdf_link_tag.has_attr('href'):
            pdf_url = pdf_link_tag['href']
        else:
            generic_pdf_links = entry_tag.find_all('link', rel='related', type='application/pdf')
            if generic_pdf_links:
                for link_tag in generic_pdf_links:
                    if link_tag.get('title') == 'pdf' and link_tag.has_attr('href'):
                        pdf_url = link_tag['href']
                        break
                if pdf_url is None and generic_pdf_links[0].has_attr('href'):  # Check if first link has href
                    pdf_url = generic_pdf_links[0]['href']


        entries.append({
            'id': paper_id,
            'title': title,
            'authors': authors_str,
            'published_date': published_date,
            'abstract': abstract,
            'pdf_url': pdf_url
        })
    return entries


def build_query_url(query: Optional[str], author: Optional[str], year: Optional[str], start: int,
                    max_results: int = ARXIV_DEFAULT_PAGE_SIZE) -> str:
    base_url = "http://export.arxiv.org/api/query?"  # HTTP, not HTTPS for export.arxiv.org
    search_terms = []

    if query:
        search_terms.append(f"all:{query}")
    if author:
        search_terms.append(f'au:"{author}"')
    if year:
        year_str = str(year)  # Ensure it's a string
        search_terms.append(f'submittedDate:[{year_str}01010000 TO {year_str}12312359]')

    search_query_value = "+AND+".join(search_terms) if search_terms else "all:*"

    # Construct URL with parameters
    # Note: requests will handle URL encoding of parameters if passed as a dict.
    # Here, we are manually constructing, so ensure correctness or let requests do it.
    # For simplicity, direct string construction (be cautious with special chars in query/author if not handled by requests).
    params = {
        "search_query": search_query_value,
        "start": max(0, int(start)),
        "max_results": max(1, int(max_results)),
        "sortBy": "relevance",  # Default sort
        "sortOrder": "descending"
    }
    query_string = urlencode(params, quote_via=quote_plus)
    return f"{base_url}{query_string}"


def convert_xml_to_markdown(xml_content: str) -> tuple[str, Optional[str], list[str], list[str]]:
    soup = BeautifulSoup(xml_content, 'xml')
    entry = soup.find('entry')
    if not entry:
        return "Error: No entry found in XML.", None, [], []

    title_tag = entry.find('title')
    title = title_tag.text.strip() if title_tag and title_tag.text else None
    authors = [author.find('name').text.strip() for author in entry.find_all('author') if author.find('name')]
    abstract_tag = entry.find('summary')
    abstract = abstract_tag.text.strip() if abstract_tag and abstract_tag.text else None
    published_tag = entry.find('published')
    published = published_tag.text.strip() if published_tag and published_tag.text else None
    categories = [category['term'] for category in entry.find_all('category') if category.has_attr('term')]

    title_for_md = title or "Untitled"
    authors_for_md = ', '.join(authors) if authors else "Unknown authors"
    published_for_md = published or "Unknown publication date"
    abstract_for_md = abstract or "No abstract available."

    markdown = f"# {title_for_md}\n\n"
    markdown += f"**Authors:** {authors_for_md}\n\n"
    markdown += f"**Published Date:** {published_for_md}\n\n"
    markdown += f"**Abstract:**\n\n{abstract_for_md}\n\n"
    if categories:
        markdown += f"**Categories:** {', '.join(categories)}\n\n"

    # Add PDF link if available
    pdf_link_tag = entry.find('link', title='pdf')
    if pdf_link_tag and pdf_link_tag.has_attr('href'):
        markdown += f"**PDF Link:** {pdf_link_tag['href']}\n\n"

    return markdown, title, authors, categories


def get_arxiv_by_id(paper_id: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Fetch a single arXiv entry by its arXiv ID using export API and normalize to ArxivPaper shape.

    Returns (item_dict, error_message). item_dict keys match ArxivPaper schema: id, title, authors,
    published_date, abstract, pdf_url.
    """
    try:
        if not paper_id or not str(paper_id).strip():
            return None, "Paper ID cannot be empty"
        xml_text = fetch_arxiv_xml(paper_id)
        if not xml_text:
            return None, None  # treat as not found
        try:
            parsed = parse_arxiv_feed(xml_text.encode("utf-8"))
        except _ARXIV_NONCRITICAL_EXCEPTIONS:
            return None, "Failed to parse arXiv XML."
        if not parsed:
            return None, None
        # parse_arxiv_feed already returns dict with required keys
        return parsed[0], None
    except _ARXIV_NONCRITICAL_EXCEPTIONS:
        return None, "Failed to fetch arXiv paper."

#
# End of Arxiv.py
#######################################################################################################################
