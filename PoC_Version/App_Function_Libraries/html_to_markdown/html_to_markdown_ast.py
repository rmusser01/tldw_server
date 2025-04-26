# html_to_markdown/html_to_markdown_ast.py

from bs4 import BeautifulSoup, Tag, NavigableString
from typing import List, Optional, Union

from s_types import (
    SemanticMarkdownAST, TextNode, BoldNode, ItalicNode, StrikethroughNode,
    HeadingNode, LinkNode, ImageNode, VideoNode, ListNode, ListItemNode,
    TableNode, TableRowNode, TableCellNode, CodeNode, BlockquoteNode,
    SemanticHtmlNode, CustomNode, MetaDataNode
)
from conversion_options import ConversionOptions
import logging

def escape_markdown_characters(text: str, is_inline_code: bool = False) -> str:
    if is_inline_code or not text.strip():
        return text
    # Replace special characters
    replacements = {
        '\\': '\\\\',
        '`': '\\`',
        '*': '\\*',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
        '[': '\\[',
        ']': '\\]',
        '(': '\\(',
        ')': '\\)',
        '#': '\\#',
        '+': '\\+',
        '-': '\\-',
        '.': '\\.',
        '!': '\\!',
        '|': '\\|',
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text

def html_to_markdown_ast(element: Tag, options: Optional[ConversionOptions] = None, indent_level: int = 0) -> List[SemanticMarkdownAST]:
    if options is None:
        options = ConversionOptions()

    result: List[SemanticMarkdownAST] = []

    for child in element.children:
        if isinstance(child, NavigableString):
            text_content = escape_markdown_characters(child.strip())
            if text_content:
                logger.debug(f"Text Node: '{text_content}'")
                result.append(TextNode(content=child.strip()))
        elif isinstance(child, Tag):
            # Check for overridden element processing
            if options.override_element_processing:
                overridden = options.override_element_processing(child, options, indent_level)
                if overridden:
                    logger.debug(f"Element Processing Overridden: '{child.name}'")
                    result.extend(overridden)
                    continue

            tag_name = child.name.lower()

            if tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                level = int(tag_name[1])
                content = escape_markdown_characters(child.get_text(strip=True))
                if content:
                    logger.debug(f"Heading {level}: '{content}'")
                    result.append(HeadingNode(level=level, content=content))
            elif tag_name == 'p':
                logger.debug("Paragraph")
                result.extend(html_to_markdown_ast(child, options, indent_level))
                # Add a new line after the paragraph
                result.append(TextNode(content='\n\n'))
            elif tag_name == 'a':
                href = child.get('href', '#')
                if href.startswith("data:image"):
                    # Skip data URLs for images
                    result.append(LinkNode(href='-', content=html_to_markdown_ast(child, options, indent_level)))
                else:
                    href = href
                    if options.website_domain and href.startswith(options.website_domain):
                        href = href[len(options.website_domain):]
                    # Check if all children are text
                    if all(isinstance(c, NavigableString) for c in child.children):
                        content = [TextNode(content=child.get_text(strip=True))]
                        result.append(LinkNode(href=href, content=content))
                    else:
                        content = html_to_markdown_ast(child, options, indent_level)
                        result.append(LinkNode(href=href, content=content))
            elif tag_name == 'img':
                src = child.get('src', '')
                alt = child.get('alt', '')
                if src.startswith("data:image"):
                    src = '-'
                else:
                    if options.website_domain and src.startswith(options.website_domain):
                        src = src[len(options.website_domain):]
                logger.debug(f"Image: src='{src}', alt='{alt}'")
                result.append(ImageNode(src=src, alt=alt))
            elif tag_name == 'video':
                src = child.get('src', '')
                poster = child.get('poster', '')
                controls = child.has_attr('controls')
                logger.debug(f"Video: src='{src}', poster='{poster}', controls='{controls}'")
                result.append(VideoNode(src=src, poster=poster, controls=controls))
            elif tag_name in ['ul', 'ol']:
                logger.debug(f"{'Unordered' if tag_name == 'ul' else 'Ordered'} List")
                ordered = tag_name == 'ol'
                items = []
                for li in child.find_all('li', recursive=False):
                    item_content = html_to_markdown_ast(li, options, indent_level + 1)
                    items.append(ListItemNode(content=item_content))
                result.append(ListNode(ordered=ordered, items=items))
            elif tag_name == 'br':
                logger.debug("Line Break")
                result.append(TextNode(content='\n'))
            elif tag_name == 'table':
                logger.debug("Table")
                table_node = TableNode()
                rows = child.find_all('tr')
                for row in rows:
                    table_row = TableRowNode()
                    cells = row.find_all(['th', 'td'])
                    for cell in cells:
                        colspan = int(cell.get('colspan', 1))
                        rowspan = int(cell.get('rowspan', 1))
                        cell_content = cell.get_text(strip=True)
                        table_row.cells.append(TableCellNode(content=cell_content, colspan=colspan if colspan >1 else None,
                                                            rowspan=rowspan if rowspan >1 else None))
                    table_node.rows.append(table_row)
                result.append(table_node)
            elif tag_name == 'head' and options.include_meta_data:
                meta_node = MetaDataNode(content={
                    'standard': {},
                    'openGraph': {},
                    'twitter': {},
                    'jsonLd': []
                })
                title = child.find('title')
                if title:
                    meta_node.content['standard']['title'] = title.get_text(strip=True)
                meta_tags = child.find_all('meta')
                non_semantic_tags = ["viewport", "referrer", "Content-Security-Policy"]
                for meta in meta_tags:
                    name = meta.get('name')
                    prop = meta.get('property')
                    content = meta.get('content', '')
                    if prop and prop.startswith('og:') and content:
                        if options.include_meta_data == 'extended':
                            meta_node.content['openGraph'][prop[3:]] = content
                    elif name and name.startswith('twitter:') and content:
                        if options.include_meta_data == 'extended':
                            meta_node.content['twitter'][name[8:]] = content
                    elif name and name not in non_semantic_tags and content:
                        meta_node.content['standard'][name] = content
                # Extract JSON-LD data
                if options.include_meta_data == 'extended':
                    json_ld_scripts = child.find_all('script', type='application/ld+json')
                    for script in json_ld_scripts:
                        try:
                            import json
                            parsed_data = json.loads(script.string)
                            meta_node.content['jsonLd'].append(parsed_data)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON-LD: {e}")
                result.append(meta_node)
            elif tag_name in ['strong', 'b']:
                content = html_to_markdown_ast(child, options, indent_level + 1)
                result.append(BoldNode(content=content if content else ""))
            elif tag_name in ['em', 'i']:
                content = html_to_markdown_ast(child, options, indent_level + 1)
                result.append(ItalicNode(content=content if content else ""))
            elif tag_name in ['s', 'strike']:
                content = html_to_markdown_ast(child, options, indent_level + 1)
                result.append(StrikethroughNode(content=content if content else ""))
            elif tag_name == 'code':
                is_code_block = child.parent.name == 'pre'
                content = child.get_text(strip=True)
                language = ""
                if not is_code_block:
                    classes = child.get('class', [])
                    for cls in classes:
                        if cls.startswith("language-"):
                            language = cls.replace("language-", "")
                            break
                result.append(CodeNode(content=content, language=language, inline=not is_code_block))
            elif tag_name == 'blockquote':
                content = html_to_markdown_ast(child, options, indent_level +1)
                result.append(BlockquoteNode(content=content))
            elif tag_name in [
                'article', 'aside', 'details', 'figcaption', 'figure', 'footer',
                'header', 'main', 'mark', 'nav', 'section', 'summary', 'time'
            ]:
                logger.debug(f"Semantic HTML Element: '{tag_name}'")
                content = html_to_markdown_ast(child, options, indent_level +1)
                result.append(SemanticHtmlNode(htmlType=tag_name, content=content))
            else:
                # Handle unhandled elements
                if options.process_unhandled_element:
                    processed = options.process_unhandled_element(child, options, indent_level)
                    if processed:
                        logger.debug(f"Processing Unhandled Element: '{tag_name}'")
                        result.extend(processed)
                        continue
                # Generic HTML elements
                logger.debug(f"Generic HTMLElement: '{tag_name}'")
                result.extend(html_to_markdown_ast(child, options, indent_level +1))
    return result
