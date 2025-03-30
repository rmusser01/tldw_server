# Arxiv.py
# Description: This file contains the functions for searching and ingesting arXiv papers.
import time

import arxiv
import requests
from bs4 import BeautifulSoup
from datetime import datetime

from requests.adapters import HTTPAdapter
from urllib3 import Retry

#
# Local Imports
from PoC_Version.App_Function_Libraries.DB.DB_Manager import add_media_with_keywords
#
#####################################################################################################
#
# Functions:

# Number of results per page
ARXIV_PAGE_SIZE = 10


def fetch_arxiv_pdf_url(paper_id):
    base_url = f"http://export.arxiv.org/api/query?id_list={paper_id}"

    # Configure retry strategy
    retry_strategy = Retry(
        total=3,  # Maximum number of retries
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
        backoff_factor=1  # Exponential backoff factor
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)

    try:
        response = http.get(base_url)
        response.raise_for_status()
        # Delay between requests to avoid rate limiting
        time.sleep(2)
        soup = BeautifulSoup(response.text, 'xml')
        pdf_link = soup.find('link', title='pdf')['href']
        return pdf_link
    except requests.exceptions.RequestException as e:
        print(f"**Error:** {e}")
        return None


def search_arxiv(query):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=10,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = []
    for result in client.results(search):
        results.append([
            result.title,
            result.entry_id.split('/')[-1],  # Extract the ID from the entry_id
            ', '.join(author.name for author in result.authors),
            result.summary
        ])

    return results


def fetch_arxiv_xml(paper_id):
    base_url = "http://export.arxiv.org/api/query?id_list="
    response = requests.get(base_url + paper_id)
    response.raise_for_status()
    return response.text


def parse_arxiv_feed(xml_content):
    soup = BeautifulSoup(xml_content, 'xml')
    entries = []
    for entry in soup.find_all('entry'):
        title = entry.title.text.strip()
        paper_id = entry.id.text.strip().split('/abs/')[-1]
        authors = ', '.join(author.find('name').text.strip() for author in entry.find_all('author'))
        published = entry.published.text.strip().split('T')[0]
        abstract = entry.summary.text.strip()
        entries.append({
            'id': paper_id,
            'title': title,
            'authors': authors,
            'published': published,
            'abstract': abstract
        })
    return entries


def build_query_url(query, author, year, start):
    # HTTP? FIXME....
    base_url = "http://export.arxiv.org/api/query?"
    search_params = []

    # Build search query
    if query:
        search_params.append(f"all:{query}")
    if author:
        search_params.append(f'au:"{author}"')
    if year:
        search_params.append(f'submittedDate:[{year}01010000 TO {year}12312359]')

    search_query = "+AND+".join(search_params) if search_params else "all:*"

    url = f"{base_url}search_query={search_query}&start={start}&max_results={ARXIV_PAGE_SIZE}"
    return url

def convert_xml_to_markdown(xml_content):
    soup = BeautifulSoup(xml_content, 'xml')

    # Extract title, authors, abstract, and other relevant information from the specific paper entry
    entry = soup.find('entry')
    title = entry.find('title').text.strip()
    authors = [author.find('name').text.strip() for author in entry.find_all('author')]
    abstract = entry.find('summary').text.strip()
    published = entry.find('published').text.strip()

    categories = [category['term'] for category in entry.find_all('category')]

    # Constructing a markdown representation for the paper
    markdown = f"# {title}\n\n"
    markdown += f"**Authors:** {', '.join(authors)}\n\n"
    markdown += f"**Published Date:** {published}\n\n"
    markdown += f"**Abstract:**\n\n{abstract}\n\n"
    markdown += f"**Categories:** {', '.join(categories)}\n\n"

    return markdown, title, authors, categories


def process_and_ingest_arxiv_paper(paper_id, additional_keywords):
    try:
        xml_content = fetch_arxiv_xml(paper_id)
        markdown, title, authors, categories = convert_xml_to_markdown(xml_content)

        keywords = f"arxiv,{','.join(categories)}"
        if additional_keywords:
            keywords += f",{additional_keywords}"

        add_media_with_keywords(
            url=f"https://arxiv.org/abs/{paper_id}",
            title=title,
            media_type='document',
            content=markdown,
            keywords=keywords,
            prompt='No prompt for arXiv papers',
            summary='arXiv paper ingested from XML',
            transcription_model='None',
            author=', '.join(authors),
            ingestion_date=datetime.now().strftime('%Y-%m-%d')
        )

        return f"arXiv paper '{title}' ingested successfully."
    except Exception as e:
        return f"Error processing arXiv paper: {str(e)}"

#
# End of Arxiv.py
####################################################################################################
