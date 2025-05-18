# Article_Extractor_Lib.py
#########################################
# Article Extraction Library
# This library is used to handle scraping and extraction of articles from web pages.
#
####################
# Function List
#
# 1. get_page_title(url)
# 2. get_article_text(url)
# 3. get_article_title(article_url_arg)
#
####################
#
# Import necessary libraries
import hashlib
from datetime import datetime
import json
import os
import tempfile
from typing import Any, Dict, List, Union, Optional, Tuple
#
# 3rd-Party Imports
import asyncio
from urllib.parse import urljoin, urlparse
from xml.dom import minidom
import xml.etree.ElementTree as xET
#
# External Libraries
from bs4 import BeautifulSoup
import pandas as pd
from playwright.async_api import async_playwright
import requests
import trafilatura
from tqdm import tqdm
#
# Import Local
from PoC_Version.App_Function_Libraries.DB.DB_Manager import ingest_article_to_db
from PoC_Version.App_Function_Libraries.Metrics.metrics_logger import log_histogram, log_counter
from PoC_Version.App_Function_Libraries.Summarization.Summarization_General_Lib import summarize
from PoC_Version.App_Function_Libraries.Utils.Utils import logging

#######################################################################################################################
# Function Definitions
#

web_scraping_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

#################################################################
#
# Scraping-related functions:

def get_page_title(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('title')
        title = title_tag.string.strip() if title_tag else "Untitled"
        log_counter("page_title_extracted", labels={"success": "true"})
        return title
    except requests.RequestException as e:
        logging.error(f"Error fetching page title: {e}")
        log_counter("page_title_extracted", labels={"success": "false"})
        return "Untitled"


async def scrape_article(url: str, custom_cookies: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    logging.info(f"Scraping article from URL: {url}")
    async def fetch_html(url: str) -> str:
        logging.info(f"Fetching HTML from {url}")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                context = await browser.new_context(
                    user_agent=web_scraping_user_agent,
                #viewport = {"width": 1280, "height": 720},
                #java_script_enabled = True
                )
                if custom_cookies:
                    await context.add_cookies(custom_cookies)
                page = await context.new_page()
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=10000)  # 10-second timeout
                    await page.wait_for_load_state("networkidle", timeout=10000)  # 10-second timeout
                    content = await page.content()
                    logging.info(f"HTML fetched successfully from {url}")
                    log_counter("html_fetched", labels={"url": url})
                    return content
                except Exception as e:
                    logging.error(f"Error fetching HTML for {url}: {e}")
                    return ""
            finally:
                await browser.close()

    def extract_article_data(html: str, url: str) -> dict:
        logging.info(f"Extracting article data from HTML for {url}")
        # FIXME - Add option for extracting comments/tables/images
        downloaded = trafilatura.extract(html, include_comments=False, include_tables=False, include_images=False)
        metadata = trafilatura.extract_metadata(html)

        result = {
            'title': 'N/A',
            'author': 'N/A',
            'content': '',
            'date': 'N/A',
            'url': url,
            'extraction_successful': False
        }

        if downloaded:
            logging.info(f"Content extracted successfully from {url}")
            log_counter("article_extracted", labels={"success": "true", "url": url})
            # Add metadata to content
            result['content'] = ContentMetadataHandler.format_content_with_metadata(
                url=url,
                content=downloaded,
                pipeline="Trafilatura",
                additional_metadata={
                    "extracted_date": metadata.date if metadata and metadata.date else 'N/A',
                    "author": metadata.author if metadata and metadata.author else 'N/A'
                }
            )
            result['extraction_successful'] = True

        if metadata:
            result.update({
                'title': metadata.title if metadata.title else 'N/A',
                'author': metadata.author if metadata.author else 'N/A',
                'date': metadata.date if metadata.date else 'N/A'
            })
        else:
            log_counter("article_extracted", labels={"success": "false", "url": url})
            logging.warning("Metadata extraction failed.")

        if not downloaded:
            logging.warning("Content extraction failed.")

        return result

    def convert_html_to_markdown(html: str) -> str:
        logging.info("Converting HTML to Markdown")
        soup = BeautifulSoup(html, 'html.parser')
        for para in soup.find_all('p'):
            # Add a newline at the end of each paragraph for markdown separation
            para.append('\n')
        # Use .get_text() with separator to keep paragraph separation
        return soup.get_text(separator='\n\n')

    html = await fetch_html(url)
    article_data = extract_article_data(html, url)
    if article_data['extraction_successful']:
        article_data['content'] = convert_html_to_markdown(article_data['content'])
        logging.info(f"Article content length: {len(article_data['content'])}")
        log_histogram("article_content_length", len(article_data['content']), labels={"url": url})
    return article_data


# FIXME - Add keyword integration/tagging
async def scrape_and_summarize_multiple(
    urls: str,
    custom_prompt_arg: Optional[str],
    api_name: str,
    api_key: Optional[str],
    keywords: str,
    custom_article_titles: Optional[str],
    system_message: Optional[str] = None,
    summarize_checkbox: bool = False,
    custom_cookies: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.7
) -> List[Dict[str, Any]]:
    urls_list = [url.strip() for url in urls.split('\n') if url.strip()]
    custom_titles = custom_article_titles.split('\n') if custom_article_titles else []

    results = []
    errors = []

    # Create a tqdm progress bar
    progress_bar = tqdm(total=len(urls_list), desc="Scraping and Summarizing")

    # Loop over each URL to scrape and optionally summarize
    for i, url in enumerate(urls_list):
        custom_title = custom_titles[i] if i < len(custom_titles) else None
        try:
            # Scrape the article
            article = await scrape_article(url, custom_cookies=custom_cookies)
            if article and article['extraction_successful']:
                log_counter("article_scraped", labels={"success": "true", "url": url})
                if custom_title:
                    article['title'] = custom_title

                # If summarization is requested
                if summarize_checkbox:
                    content = article.get('content', '')
                    if content:
                        # Prepare prompts
                        system_message_final = system_message or \
                                               "Act as a professional summarizer and summarize this article."
                        article_custom_prompt = custom_prompt_arg or \
                                                "Act as a professional summarizer and summarize this article."

                        # Summarize the content using the summarize function
                        summary = summarize(
                            input_data=content,
                            custom_prompt_arg=article_custom_prompt,
                            api_name=api_name,
                            api_key=api_key,
                            temp=temperature,
                            system_message=system_message_final
                        )
                        article['summary'] = summary
                        log_counter("article_summarized", labels={"success": "true", "url": url})
                        logging.info(f"Summary generated for URL {url}")
                    else:
                        article['summary'] = "No content available to summarize."
                        logging.warning(f"No content to summarize for URL {url}")
                else:
                    article['summary'] = None

                results.append(article)
            else:
                error_message = f"Extraction unsuccessful for URL {url}"
                errors.append(error_message)
                logging.error(error_message)
                log_counter("article_scraped", labels={"success": "false", "url": url})
        except Exception as e:
            log_counter("article_processing_error", labels={"url": url})
            error_message = f"Error processing URL {i + 1} ({url}): {str(e)}"
            errors.append(error_message)
            logging.error(error_message, exc_info=True)
        finally:
            # Update the progress bar
            progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    if errors:
        logging.error("\n".join(errors))

    if not results:
        logging.error("No articles were successfully scraped and summarized/analyzed.")
        return []

    log_histogram("articles_processed", len(results))
    return results


def scrape_and_no_summarize_then_ingest(url, keywords, custom_article_title):
    try:
        # Step 1: Scrape the article
        article_data = asyncio.run(scrape_article(url))
        print(f"Scraped Article Data: {article_data}")  # Debugging statement
        if not article_data:
            log_counter("article_scrape_failed", labels={"url": url})
            return "Failed to scrape the article."

        # Use the custom title if provided, otherwise use the scraped title
        title = custom_article_title.strip() if custom_article_title else article_data.get('title', 'Untitled')
        author = article_data.get('author', 'Unknown')
        content = article_data.get('content', '')
        ingestion_date = datetime.now().strftime('%Y-%m-%d')

        print(f"Title: {title}, Author: {author}, Content Length: {len(content)}")  # Debugging statement

        # Step 2: Ingest the article into the database
        ingestion_result = ingest_article_to_db(url, title, author, content, keywords, ingestion_date, None, None)
        log_counter("article_ingested", labels={"success": str(ingestion_result).lower(), "url": url})

        # When displaying content, we might want to strip metadata
        display_content = ContentMetadataHandler.strip_metadata(content)
        return f"Title: {title}\nAuthor: {author}\nIngestion Result: {ingestion_result}\n\nArticle Contents: {display_content}"
    except Exception as e:
        log_counter("article_processing_error", labels={"url": url})
        logging.error(f"Error processing URL {url}: {str(e)}")
        return f"Failed to process URL {url}: {str(e)}"


def scrape_from_filtered_sitemap(sitemap_file: str, filter_function) -> list:
    """
    Scrape articles from a sitemap file, applying an additional filter function.

    :param sitemap_file: Path to the sitemap file
    :param filter_function: A function that takes a URL and returns True if it should be scraped
    :return: List of scraped articles
    """
    try:
        tree = xET.parse(sitemap_file)
        root = tree.getroot()

        articles = []
        for url in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
            if filter_function(url.text):
                article_data = scrape_article(url.text)
                if article_data:
                    articles.append(article_data)

        return articles
    except xET.ParseError as e:
        logging.error(f"Error parsing sitemap: {e}")
        return []


def is_content_page(url: str) -> bool:
    """
    Determine if a URL is likely to be a content page.
    This is a basic implementation and may need to be adjusted based on the specific website structure.

    :param url: The URL to check
    :return: True if the URL is likely a content page, False otherwise
    """
    # Add more specific checks here based on the website's structure
    # Exclude common non-content pages
    exclude_patterns = [
        '/tag/', '/category/', '/author/', '/search/', '/page/',
        'wp-content', 'wp-includes', 'wp-json', 'wp-admin',
        'login', 'register', 'cart', 'checkout', 'account',
        '.jpg', '.png', '.gif', '.pdf', '.zip'
    ]
    return not any(pattern in url.lower() for pattern in exclude_patterns)

def scrape_and_convert_with_filter(source: str, output_file: str, filter_function=is_content_page, level: int = None):
    """
    Scrape articles from a sitemap or by URL level, apply filtering, and convert to a single markdown file.

    :param source: URL of the sitemap, base URL for level-based scraping, or path to a local sitemap file
    :param output_file: Path to save the output markdown file
    :param filter_function: Function to filter URLs (default is is_content_page)
    :param level: URL level for scraping (None if using sitemap)
    """
    if level is not None:
        # Scraping by URL level
        articles = scrape_by_url_level(source, level)
        articles = [article for article in articles if filter_function(article['url'])]
    elif source.startswith('http'):
        # Scraping from online sitemap
        articles = scrape_from_sitemap(source)
        articles = [article for article in articles if filter_function(article['url'])]
    else:
        # Scraping from local sitemap file
        articles = scrape_from_filtered_sitemap(source, filter_function)

    articles = [article for article in articles if filter_function(article['url'])]
    markdown_content = convert_to_markdown(articles)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    logging.info(f"Scraped and filtered content saved to {output_file}")


async def scrape_entire_site(base_url: str) -> List[Dict]:
    """
    Scrape the entire site by generating a temporary sitemap and extracting content from each page.

    :param base_url: The base URL of the site to scrape
    :return: A list of dictionaries containing scraped article data
    """
    # Step 1: Collect internal links from the site
    links = collect_internal_links(base_url)
    log_histogram("internal_links_collected", len(links), labels={"base_url": base_url})
    logging.info(f"Collected {len(links)} internal links.")

    # Step 2: Generate the temporary sitemap
    temp_sitemap_path = generate_temp_sitemap_from_links(links)

    # Step 3: Scrape each URL in the sitemap
    scraped_articles = []
    try:
        async def scrape_and_log(link):
            logging.info(f"Scraping {link} ...")
            article_data = await scrape_article(link)

            if article_data:
                logging.info(f"Title: {article_data['title']}")
                logging.info(f"Author: {article_data['author']}")
                logging.info(f"Date: {article_data['date']}")
                logging.info(f"Content: {article_data['content'][:500]}...")

                return article_data
            return None

        # Use asyncio.gather to scrape multiple articles concurrently
        scraped_articles = await asyncio.gather(*[scrape_and_log(link) for link in links])
        # Remove any None values (failed scrapes)
        scraped_articles = [article for article in scraped_articles if article is not None]
        log_histogram("articles_scraped", len(scraped_articles), labels={"base_url": base_url})

    finally:
        # Clean up the temporary sitemap file
        os.unlink(temp_sitemap_path)
        logging.info("Temporary sitemap file deleted")

    return scraped_articles


def scrape_by_url_level(base_url: str, level: int) -> list:
    """Scrape articles from URLs up to a certain level under the base URL."""

    def get_url_level(url: str) -> int:
        return len(urlparse(url).path.strip('/').split('/'))

    links = collect_internal_links(base_url)
    filtered_links = [link for link in links if get_url_level(link) <= level]

    return [article for link in filtered_links if (article := scrape_article(link))]


def scrape_from_sitemap(sitemap_url: str) -> list:
    """Scrape articles from a sitemap URL."""
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        root = xET.fromstring(response.content)

        return [article for url in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if (article := scrape_article(url.text))]
    except requests.RequestException as e:
        logging.error(f"Error fetching sitemap: {e}")
        return []

#
# End of Scraping Functions
#######################################################
#
# Sitemap/Crawling-related Functions


def collect_internal_links(base_url: str) -> set:
    visited = set()
    to_visit = {base_url}

    while to_visit:
        current_url = to_visit.pop()
        if current_url in visited:
            continue

        try:
            response = requests.get(current_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Collect internal links
            for link in soup.find_all('a', href=True):
                full_url = urljoin(base_url, link['href'])
                # Only process links within the same domain
                if urlparse(full_url).netloc == urlparse(base_url).netloc:
                    if full_url not in visited:
                        to_visit.add(full_url)

            visited.add(current_url)
        except requests.RequestException as e:
            logging.error(f"Error visiting {current_url}: {e}")
            continue

    return visited


def generate_temp_sitemap_from_links(links: set) -> str:
    """
    Generate a temporary sitemap file from collected links and return its path.

    :param links: A set of URLs to include in the sitemap
    :return: Path to the temporary sitemap file
    """
    # Create the root element
    urlset = xET.Element("urlset")
    urlset.set("xmlns", "http://www.sitemaps.org/schemas/sitemap/0.9")

    # Add each link to the sitemap
    for link in links:
        url = xET.SubElement(urlset, "url")
        loc = xET.SubElement(url, "loc")
        loc.text = link
        lastmod = xET.SubElement(url, "lastmod")
        lastmod.text = datetime.now().strftime("%Y-%m-%d")
        changefreq = xET.SubElement(url, "changefreq")
        changefreq.text = "daily"
        priority = xET.SubElement(url, "priority")
        priority.text = "0.5"

    # Create the tree and get it as a string
    xml_string = xET.tostring(urlset, 'utf-8')

    # Pretty print the XML
    pretty_xml = minidom.parseString(xml_string).toprettyxml(indent="  ")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as temp_file:
        temp_file.write(pretty_xml)
        temp_file_path = temp_file.name

    logging.info(f"Temporary sitemap created at: {temp_file_path}")
    return temp_file_path


def generate_sitemap_for_url(url: str) -> List[Dict[str, str]]:
    """
    Generate a sitemap for the given URL using the create_filtered_sitemap function.

    Args:
        url (str): The base URL to generate the sitemap for

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing 'url' and 'title' keys
    """
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".xml", delete=False) as temp_file:
        create_filtered_sitemap(url, temp_file.name, is_content_page)
        temp_file.seek(0)
        tree = xET.parse(temp_file.name)
        root = tree.getroot()

        sitemap = []
        for url_elem in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
            loc = url_elem.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc").text
            sitemap.append({"url": loc, "title": loc.split("/")[-1] or url})  # Use the last part of the URL as a title

    return sitemap

def create_filtered_sitemap(base_url: str, output_file: str, filter_function):
    """
    Create a sitemap from internal links and filter them based on a custom function.

    :param base_url: The base URL of the website
    :param output_file: The file to save the sitemap to
    :param filter_function: A function that takes a URL and returns True if it should be included
    """
    links = collect_internal_links(base_url)
    filtered_links = set(filter(filter_function, links))

    root = xET.Element("urlset")
    root.set("xmlns", "http://www.sitemaps.org/schemas/sitemap/0.9")

    for link in filtered_links:
        url = xET.SubElement(root, "url")
        loc = xET.SubElement(url, "loc")
        loc.text = link

    tree = xET.ElementTree(root)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Filtered sitemap saved to {output_file}")


#
# End of Crawling Functions
#################################################################
#
# Utility Functions

def convert_to_markdown(articles: list) -> str:
    """Convert a list of article data into a single markdown document."""
    markdown = ""
    for article in articles:
        markdown += f"# {article['title']}\n\n"
        markdown += f"Author: {article['author']}\n"
        markdown += f"Date: {article['date']}\n\n"
        markdown += f"{article['content']}\n\n"
        markdown += "---\n\n"  # Separator between articles
    return markdown

def compute_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def load_hashes(filename: str) -> Dict[str, str]:
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        return {}

def save_hashes(hashes: Dict[str, str], filename: str):
    with open(filename, 'w') as f:
        json.dump(hashes, f)

def has_page_changed(url: str, new_hash: str, stored_hashes: Dict[str, str]) -> bool:
    old_hash = stored_hashes.get(url)
    return old_hash != new_hash


#
#
###################################################
#
# Bookmark Parsing Functions

def parse_chromium_bookmarks(json_data: dict) -> Dict[str, Union[str, List[str]]]:
    """
    Parse Chromium-based browser bookmarks from JSON data.

    :param json_data: The JSON data from the bookmarks file
    :return: A dictionary with bookmark names as keys and URLs as values or lists of URLs if duplicates exist
    """
    bookmarks = {}

    def recurse_bookmarks(nodes):
        for node in nodes:
            if node.get('type') == 'url':
                name = node.get('name')
                url = node.get('url')
                if name and url:
                    if name in bookmarks:
                        if isinstance(bookmarks[name], list):
                            bookmarks[name].append(url)
                        else:
                            bookmarks[name] = [bookmarks[name], url]
                    else:
                        bookmarks[name] = url
            elif node.get('type') == 'folder' and 'children' in node:
                recurse_bookmarks(node['children'])

    # Chromium bookmarks have a 'roots' key
    if 'roots' in json_data:
        for root in json_data['roots'].values():
            if 'children' in root:
                recurse_bookmarks(root['children'])
    else:
        recurse_bookmarks(json_data.get('children', []))

    return bookmarks


def parse_firefox_bookmarks(html_content: str) -> Dict[str, Union[str, List[str]]]:
    """
    Parse Firefox bookmarks from HTML content.

    :param html_content: The HTML content from the bookmarks file
    :return: A dictionary with bookmark names as keys and URLs as values or lists of URLs if duplicates exist
    """
    bookmarks = {}
    soup = BeautifulSoup(html_content, 'html.parser')

    # Firefox stores bookmarks within <a> tags inside <dt>
    for a in soup.find_all('a'):
        name = a.get_text()
        url = a.get('href')
        if name and url:
            if name in bookmarks:
                if isinstance(bookmarks[name], list):
                    bookmarks[name].append(url)
                else:
                    bookmarks[name] = [bookmarks[name], url]
            else:
                bookmarks[name] = url

    return bookmarks


def load_bookmarks(file_path: str) -> Dict[str, Union[str, List[str]]]:
    """
    Load bookmarks from a file (JSON for Chrome/Edge or HTML for Firefox).

    :param file_path: Path to the bookmarks file
    :return: A dictionary with bookmark names as keys and URLs as values or lists of URLs if duplicates exist
    :raises ValueError: If the file format is unsupported or parsing fails
    """
    if not os.path.isfile(file_path):
        logging.error(f"File '{file_path}' does not exist.")
        raise FileNotFoundError(f"File '{file_path}' does not exist.")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.json' or ext == '':
        # Attempt to parse as JSON (Chrome/Edge)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            return parse_chromium_bookmarks(json_data)
        except json.JSONDecodeError:
            logging.error("Failed to parse JSON. Ensure the file is a valid Chromium bookmarks JSON file.")
            raise ValueError("Invalid JSON format for Chromium bookmarks.")
    elif ext in ['.html', '.htm']:
        # Parse as HTML (Firefox)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return parse_firefox_bookmarks(html_content)
        except Exception as e:
            logging.error(f"Failed to parse HTML bookmarks: {e}")
            raise ValueError(f"Failed to parse HTML bookmarks: {e}")
    else:
        logging.error("Unsupported file format. Please provide a JSON (Chrome/Edge) or HTML (Firefox) bookmarks file.")
        raise ValueError("Unsupported file format for bookmarks.")


def collect_bookmarks(file_path: str) -> Dict[str, Union[str, List[str]]]:
    """
    Collect bookmarks from the provided bookmarks file and return a dictionary.

    :param file_path: Path to the bookmarks file
    :return: Dictionary with bookmark names as keys and URLs as values or lists of URLs if duplicates exist
    """
    try:
        bookmarks = load_bookmarks(file_path)
        logging.info(f"Successfully loaded {len(bookmarks)} bookmarks from '{file_path}'.")
        return bookmarks
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Error loading bookmarks: {e}")
        return {}


def parse_csv_urls(file_path: str) -> Dict[str, Union[str, List[str]]]:
    """
    Parse URLs from a CSV file. The CSV should have at minimum a 'url' column,
    and optionally a 'title' or 'name' column.

    :param file_path: Path to the CSV file
    :return: Dictionary with titles/names as keys and URLs as values
    """
    try:
        # Read CSV file
        df = pd.read_csv(file_path)

        # Check if required columns exist
        if 'url' not in df.columns:
            raise ValueError("CSV must contain a 'url' column")

        # Initialize result dictionary
        urls_dict = {}

        # Determine which column to use as key
        key_column = next((col for col in ['title', 'name'] if col in df.columns), None)

        for idx in range(len(df)):
            url = df.iloc[idx]['url'].strip()

            # Use title/name if available, otherwise use URL as key
            if key_column:
                key = df.iloc[idx][key_column].strip()
            else:
                key = f"Article {idx + 1}"

            # Handle duplicate keys
            if key in urls_dict:
                if isinstance(urls_dict[key], list):
                    urls_dict[key].append(url)
                else:
                    urls_dict[key] = [urls_dict[key], url]
            else:
                urls_dict[key] = url

        return urls_dict

    except pd.errors.EmptyDataError:
        logging.error("The CSV file is empty")
        return {}
    except Exception as e:
        logging.error(f"Error parsing CSV file: {str(e)}")
        return {}


def collect_urls_from_file(file_path: str) -> Dict[str, Union[str, List[str]]]:
    """
    Unified function to collect URLs from either bookmarks or CSV files.

    :param file_path: Path to the file (bookmarks or CSV)
    :return: Dictionary with names as keys and URLs as values
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.csv':
        return parse_csv_urls(file_path)
    else:
        return collect_bookmarks(file_path)

# Usage:
# from Article_Extractor_Lib import collect_bookmarks
#
# # Path to your bookmarks file
# # For Chrome or Edge (JSON format)
# chromium_bookmarks_path = "/path/to/Bookmarks"
#
# # For Firefox (HTML format)
# firefox_bookmarks_path = "/path/to/bookmarks.html"
#
# # Collect bookmarks from Chromium-based browser
# chromium_bookmarks = collect_bookmarks(chromium_bookmarks_path)
# print("Chromium Bookmarks:")
# for name, url in chromium_bookmarks.items():
#     print(f"{name}: {url}")
#
# # Collect bookmarks from Firefox
# firefox_bookmarks = collect_bookmarks(firefox_bookmarks_path)
# print("\nFirefox Bookmarks:")
# for name, url in firefox_bookmarks.items():
#     print(f"{name}: {url}")

#
# End of Bookmarking Parsing Functions
#####################################################################


#####################################################################
#
# Article Scraping Metadata Functions

class ContentMetadataHandler:
    """Handles the addition and parsing of metadata for scraped content."""

    METADATA_START = "[METADATA]"
    METADATA_END = "[/METADATA]"

    @staticmethod
    def format_content_with_metadata(
            url: str,
            content: str,
            pipeline: str = "Trafilatura",
            additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format content with metadata header.

        Args:
            url: The source URL
            content: The scraped content
            pipeline: The scraping pipeline used
            additional_metadata: Optional dictionary of additional metadata to include

        Returns:
            Formatted content with metadata header
        """
        metadata = {
            "url": url,
            "ingestion_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "content_hash": hashlib.sha256(content.encode('utf-8')).hexdigest(),
            "scraping_pipeline": pipeline
        }

        # Add any additional metadata
        if additional_metadata:
            metadata.update(additional_metadata)

        formatted_content = f"""{ContentMetadataHandler.METADATA_START}
        {json.dumps(metadata, indent=2)}
        {ContentMetadataHandler.METADATA_END}
        
        {content}"""

        return formatted_content

    @staticmethod
    def extract_metadata(content: str) -> Tuple[Dict[str, Any], str]:
        """
        Extract metadata and content separately.

        Args:
            content: The full content including metadata

        Returns:
            Tuple of (metadata dict, clean content)
        """
        try:
            metadata_start = content.index(ContentMetadataHandler.METADATA_START) + len(
                ContentMetadataHandler.METADATA_START)
            metadata_end = content.index(ContentMetadataHandler.METADATA_END)
            metadata_json = content[metadata_start:metadata_end].strip()
            metadata = json.loads(metadata_json)
            clean_content = content[metadata_end + len(ContentMetadataHandler.METADATA_END):].strip()
            return metadata, clean_content
        except (ValueError, json.JSONDecodeError) as e:
            return {}, content

    @staticmethod
    def has_metadata(content: str) -> bool:
        """
        Check if content contains metadata.

        Args:
            content: The content to check

        Returns:
            bool: True if metadata is present
        """
        return (ContentMetadataHandler.METADATA_START in content and
                ContentMetadataHandler.METADATA_END in content)

    @staticmethod
    def strip_metadata(content: str) -> str:
        """
        Remove metadata from content if present.

        Args:
            content: The content to strip metadata from

        Returns:
            Content without metadata
        """
        try:
            metadata_end = content.index(ContentMetadataHandler.METADATA_END)
            return content[metadata_end + len(ContentMetadataHandler.METADATA_END):].strip()
        except ValueError:
            return content

    @staticmethod
    def get_content_hash(content: str) -> str:
        """
        Get hash of content without metadata.

        Args:
            content: The content to hash

        Returns:
            SHA-256 hash of the clean content
        """
        clean_content = ContentMetadataHandler.strip_metadata(content)
        return hashlib.sha256(clean_content.encode('utf-8')).hexdigest()

    @staticmethod
    def content_changed(old_content: str, new_content: str) -> bool:
        """
        Check if content has changed by comparing hashes.

        Args:
            old_content: Previous version of content
            new_content: New version of content

        Returns:
            bool: True if content has changed
        """
        old_hash = ContentMetadataHandler.get_content_hash(old_content)
        new_hash = ContentMetadataHandler.get_content_hash(new_content)
        return old_hash != new_hash

#
# End of Article_Extractor_Lib.py
#######################################################################################################################
