# html_to_markdown/dom_utils.py

from bs4 import BeautifulSoup, Tag
from typing import Optional
import logging

from conversion_options import ConversionOptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_main_content(soup: BeautifulSoup, options: ConversionOptions) -> Tag:
    logger.debug("Entering find_main_content function")

    main_element = soup.find('main')
    if main_element:
        logger.debug("Existing <main> element found")
        return main_element

    logger.debug("No <main> element found. Detecting main content.")
    if not soup.body:
        logger.debug("No body element found, returning the entire document")
        return soup

    return detect_main_content(soup.body, options)

def wrap_main_content(main_content: Tag, soup: BeautifulSoup):
    if main_content.name.lower() != 'main':
        logger.debug("Wrapping main content in <main> element")
        main_element = soup.new_tag('main')
        main_content.wrap(main_element)
        main_element['id'] = 'detected-main-content'
        logger.debug("Main content wrapped successfully")
    else:
        logger.debug("Main content already wrapped")

def detect_main_content(element: Tag, options: ConversionOptions) -> Tag:
    candidates = []
    min_score = 20
    logger.debug(f"Collecting candidates with minimum score: {min_score}")
    collect_candidates(element, candidates, min_score, options)

    logger.debug(f"Total candidates found: {len(candidates)}")

    if not candidates:
        logger.debug("No suitable candidates found, returning root element")
        return element

    # Sort candidates by score descending
    candidates.sort(key=lambda x: calculate_score(x, options), reverse=True)
    logger.debug("Candidates sorted by score")

    best_candidate = candidates[0]
    for candidate in candidates[1:]:
        if not any(other.contains(candidate) for other in candidates):
            if calculate_score(candidate, options) > calculate_score(best_candidate, options):
                best_candidate = candidate
                logger.debug(f"New best independent candidate found: {element_to_string(best_candidate)}")

    logger.debug(f"Final main content candidate: {element_to_string(best_candidate)}")
    return best_candidate

def element_to_string(element: Optional[Tag]) -> str:
    if not element:
        return 'No element'
    classes = '.'.join(element.get('class', []))
    return f"{element.name}#{element.get('id', 'no-id')}.{classes}"

def collect_candidates(element: Tag, candidates: list, min_score: int, options: ConversionOptions):
    score = calculate_score(element, options)
    if score >= min_score:
        candidates.append(element)
        logger.debug(f"Candidate found: {element_to_string(element)}, score: {score}")

    for child in element.find_all(recursive=False):
        collect_candidates(child, candidates, min_score, options)

def calculate_score(element: Tag, options: ConversionOptions) -> int:
    score = 0
    score_log = []

    # High impact attributes
    high_impact_attributes = ['article', 'content', 'main-container', 'main', 'main-content']
    for attr in high_impact_attributes:
        if 'class' in element.attrs and attr in element['class']:
            score += 10
            score_log.append(f"High impact attribute found: {attr}, score increased by 10")
        if 'id' in element.attrs and attr in element['id']:
            score += 10
            score_log.append(f"High impact ID found: {attr}, score increased by 10")

    # High impact tags
    high_impact_tags = ['article', 'main', 'section']
    if element.name.lower() in high_impact_tags:
        score += 5
        score_log.append(f"High impact tag found: {element.name}, score increased by 5")

    # Paragraph count
    paragraph_count = len(element.find_all('p'))
    paragraph_score = min(paragraph_count, 5)
    if paragraph_score > 0:
        score += paragraph_score
        score_log.append(f"Paragraph count: {paragraph_count}, score increased by {paragraph_score}")

    # Text content length
    text_content_length = len(element.get_text(strip=True))
    if text_content_length > 200:
        text_score = min(text_content_length // 200, 5)
        score += text_score
        score_log.append(f"Text content length: {text_content_length}, score increased by {text_score}")

    # Link density
    link_density = calculate_link_density(element)
    if link_density < 0.3:
        score += 5
        score_log.append(f"Link density: {link_density:.2f}, score increased by 5")

    # Data attributes
    if element.has_attr('data-main') or element.has_attr('data-content'):
        score += 10
        score_log.append("Data attribute for main content found, score increased by 10")

    # Role attribute
    if element.get('role') and 'main' in element.get('role'):
        score += 10
        score_log.append("Role attribute indicating main content found, score increased by 10")

    if options.debug and score_log:
        logger.debug(f"Scoring for {element_to_string(element)}:")
        for log in score_log:
            logger.debug(f"  {log}")
        logger.debug(f"  Final score: {score}")

    return score

def calculate_link_density(element: Tag) -> float:
    links = element.find_all('a')
    link_length = sum(len(link.get_text(strip=True)) for link in links)
    text_length = len(element.get_text(strip=True)) or 1  # Avoid division by zero
    return link_length / text_length
