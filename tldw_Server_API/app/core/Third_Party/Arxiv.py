# Arxiv.py
# Description: This file contains the functions for searching and ingesting arXiv papers.
import time
import arxiv  # Keep this if search_arxiv is used, or for reference
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Optional, List, Dict, Any  # Added for type hinting

from requests.adapters import HTTPAdapter
from urllib3 import Retry

from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase

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
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=1
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http_session = requests.Session()  # Renamed for clarity
    http_session.mount("https://", adapter)
    http_session.mount("http://", adapter)

    try:
        response = http_session.get(base_url)
        response.raise_for_status()
        time.sleep(1)  # Keep a small delay, 2s might be too long for an API response time
        soup = BeautifulSoup(response.content, 'xml')  # Use response.content for bytes
        pdf_link_tag = soup.find('link', attrs={'title': 'pdf', 'rel': 'related', 'type': 'application/pdf'})
        if pdf_link_tag and pdf_link_tag.has_attr('href'):
            return pdf_link_tag['href']
        return None
    except requests.exceptions.RequestException as e:
        print(f"**Error fetching PDF URL for {paper_id}:** {e}")
        return None
    except Exception as e:
        print(f"**Unexpected error fetching PDF URL for {paper_id}:** {e}")
        return None


def search_arxiv_custom_api(query: Optional[str], author: Optional[str], year: Optional[str], start_index: int,
                            page_size: int) -> tuple[Optional[List[Dict[str, Any]]], int, Optional[str]]:
    """
    Searches arXiv using the custom built URL and parses the feed.
    Returns a list of papers, total results found by the API for this query, and an error message if any.
    """
    query_url = build_query_url(query, author, year, start_index, page_size)

    retry_strategy = Retry(total=3, status_forcelist=[429, 500, 502, 503, 504], backoff_factor=1)
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http_session = requests.Session()
    http_session.mount("http://", adapter)  # ArXiv API is HTTP
    http_session.mount("https://", adapter)

    try:
        response = http_session.get(query_url, timeout=10)  # Added timeout
        response.raise_for_status()

        # Brief delay after successful request
        time.sleep(0.5)  # Reduced delay

        parsed_entries = parse_arxiv_feed(response.content)  # Pass response.content (bytes)

        soup = BeautifulSoup(response.content, 'xml')
        total_results_tag = soup.find('opensearch:totalResults')
        total_results = int(total_results_tag.text) if total_results_tag and total_results_tag.text.isdigit() else 0

        return parsed_entries, total_results, None
    except requests.exceptions.Timeout:
        error_msg = "Request to arXiv API timed out."
        print(f"**Error:** {error_msg}")
        return None, 0, error_msg
    except requests.exceptions.RequestException as e:
        error_msg = f"arXiv API request failed: {e}"
        print(f"**Error:** {error_msg}")
        return None, 0, error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred during arXiv search: {e}"
        print(f"**Error:** {error_msg}")
        return None, 0, error_msg


def fetch_arxiv_xml(paper_id: str) -> Optional[str]:
    base_url = "http://export.arxiv.org/api/query?id_list="
    try:
        response = requests.get(base_url + paper_id)
        response.raise_for_status()
        time.sleep(1)  # Keep delay
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"**Error fetching XML for {paper_id}:** {e}")
        return None


def parse_arxiv_feed(xml_content: bytes) -> List[Dict[str, Any]]:
    try:
        soup = BeautifulSoup(xml_content, 'lxml-xml')
    except Exception as e: # Broad exception, can be narrowed to bs4.FeatureNotFound if desired
        print(f"Warning: Failed to use 'lxml-xml' parser ({e}). Falling back to Python's built-in 'xml' parser.")
        print("For potentially better performance and XML feature support, consider installing lxml: pip install lxml")
        soup = BeautifulSoup(xml_content, 'xml') # Fallback

    entries = []

    for entry_tag in soup.find_all('entry'):
        # Title
        title_tag = entry_tag.find('title')
        title = title_tag.text.strip() if title_tag and title_tag.text else "N/A"

        # Paper ID
        paper_id_xml_tag = entry_tag.find('id')
        paper_id = "N/A"
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
        authors_str = ', '.join(authors_list) if authors_list else "N/A"

        # Published Date
        published_tag = entry_tag.find('published')
        published_date = "N/A"
        if published_tag and published_tag.text:
            published_date = published_tag.text.strip().split('T')[0]

        # Abstract (summary)
        summary_tag = entry_tag.find('summary')
        abstract = summary_tag.text.strip() if summary_tag and summary_tag.text else "N/A"

        # Fetch PDF link
        pdf_url = "N/A"
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
                if pdf_url == "N/A" and generic_pdf_links[0].has_attr('href'): # Check if first link has href
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
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": "relevance",  # Default sort
        "sortOrder": "descending"
    }
    # This is safer if requests library is used to make the actual call
    # query_string = requests.compat.urlencode(params)
    # url = f"{base_url}{query_string}"
    # Manual construction:
    url = f"{base_url}search_query={search_query_value}&start={start}&max_results={max_results}&sortBy=relevance&sortOrder=descending"
    return url


def convert_xml_to_markdown(xml_content: str) -> tuple[str, str, List[str], List[str]]:
    soup = BeautifulSoup(xml_content, 'xml')
    entry = soup.find('entry')
    if not entry:
        return "Error: No entry found in XML.", "N/A", [], []

    title = entry.find('title').text.strip() if entry.find('title') else "N/A"
    authors = [author.find('name').text.strip() for author in entry.find_all('author') if author.find('name')]
    abstract = entry.find('summary').text.strip() if entry.find('summary') else "N/A"
    published = entry.find('published').text.strip() if entry.find('published') else "N/A"
    categories = [category['term'] for category in entry.find_all('category') if category.has_attr('term')]

    markdown = f"# {title}\n\n"
    markdown += f"**Authors:** {', '.join(authors)}\n\n"
    markdown += f"**Published Date:** {published}\n\n"
    markdown += f"**Abstract:**\n\n{abstract}\n\n"
    if categories:
        markdown += f"**Categories:** {', '.join(categories)}\n\n"

    # Add PDF link if available
    pdf_link_tag = entry.find('link', title='pdf')
    if pdf_link_tag and pdf_link_tag.has_attr('href'):
        markdown += f"**PDF Link:** {pdf_link_tag['href']}\n\n"

    return markdown, title, authors, categories

#
# End of Arxiv.py
#######################################################################################################################
