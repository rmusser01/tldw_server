from bs4 import BeautifulSoup

from tldw_Server_API.app.core.Utils.Utils import logging


def convert_html_to_markdown(html: str) -> str:
    """Convert raw HTML to Markdown-friendly plain text."""
    logging.info("Converting HTML to Markdown")
    soup = BeautifulSoup(html, "html.parser")
    for para in soup.find_all("p"):
        para.append("\n")
    return soup.get_text(separator="\n\n")
