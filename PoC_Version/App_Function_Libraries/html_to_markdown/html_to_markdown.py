# html_to_markdown/html_to_markdown.py

from bs4 import BeautifulSoup
from typing import Optional

from conversion_options import ConversionOptions
from dom_utils import find_main_content, wrap_main_content
from html_to_markdown_ast import html_to_markdown_ast
from markdown_ast_to_string import markdown_ast_to_string
from url_utils import refify_urls

import logging

def convert_html_to_markdown(html: str, options: Optional[ConversionOptions] = None) -> str:
    if options is None:
        options = ConversionOptions()

    if options.debug:
        logger.setLevel(logging.DEBUG)

    soup = BeautifulSoup(html, 'html.parser')

    if options.extract_main_content:
        main_content = find_main_content(soup, options)
        if options.include_meta_data and soup.head and not main_content.find('head'):
            # Reattach head for metadata extraction
            new_html = f"<html>{soup.head}{main_content}</html>"
            soup = BeautifulSoup(new_html, 'html.parser')
            main_content = soup.html
    else:
        if options.include_meta_data and soup.head:
            main_content = soup
        else:
            main_content = soup.body if soup.body else soup

    markdown_ast = html_to_markdown_ast(main_content, options)

    if options.refify_urls:
        options.url_map = refify_urls(markdown_ast, options.url_map)

    markdown_string = markdown_ast_to_string(markdown_ast, options)

    return markdown_string
