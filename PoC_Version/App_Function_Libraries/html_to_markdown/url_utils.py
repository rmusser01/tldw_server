# html_to_markdown/url_utils.py

from typing import Dict

media_suffixes = [
    "jpeg", "jpg", "png", "gif", "bmp", "tiff", "tif", "svg",
    "webp", "ico", "avi", "mov", "mp4", "mkv", "flv", "wmv",
    "webm", "mpeg", "mpg", "mp3", "wav", "aac", "ogg", "flac",
    "m4a", "pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx",
    "txt", "css", "js", "xml", "json", "html", "htm"
]

def add_ref_prefix(prefix: str, prefixes_to_refs: Dict[str, str]) -> str:
    if prefix not in prefixes_to_refs:
        prefixes_to_refs[prefix] = f'ref{len(prefixes_to_refs)}'
    return prefixes_to_refs[prefix]

def process_url(url: str, prefixes_to_refs: Dict[str, str]) -> str:
    if not url.startswith('http'):
        return url
    else:
        parts = url.split('/')
        media_suffix = parts[-1].split('.')[-1].lower()
        if media_suffix in media_suffixes:
            prefix = '/'.join(parts[:-1])
            ref_prefix = add_ref_prefix(prefix, prefixes_to_refs)
            return f"{ref_prefix}://{parts[-1]}"
        else:
            if len(parts) > 4:
                return add_ref_prefix(url, prefixes_to_refs)
            else:
                return url

def refify_urls(markdown_elements: list, prefixes_to_refs: Dict[str, str] = {}) -> Dict[str, str]:
    for element in markdown_elements:
        if isinstance(element, dict):
            node_type = element.get('type')
            if node_type == 'link':
                original_href = element.get('href', '')
                element['href'] = process_url(original_href, prefixes_to_refs)
                refify_urls(element.get('content', []), prefixes_to_refs)
            elif node_type in ['image', 'video']:
                original_src = element.get('src', '')
                element['src'] = process_url(original_src, prefixes_to_refs)
            elif node_type == 'list':
                for item in element.get('items', []):
                    refify_urls(item.get('content', []), prefixes_to_refs)
            elif node_type == 'table':
                for row in element.get('rows', []):
                    for cell in row.get('cells', []):
                        if isinstance(cell.get('content'), list):
                            refify_urls(cell['content'], prefixes_to_refs)
            elif node_type in ['blockquote', 'semanticHtml']:
                refify_urls(element.get('content', []), prefixes_to_refs)
    return prefixes_to_refs
