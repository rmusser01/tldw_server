# html_to_markdown/conversion_options.py

from typing import Callable, Optional, Union, Dict, Any, List
from dataclasses import dataclass, field

from s_types import SemanticMarkdownAST, CustomNode

@dataclass
class ConversionOptions:
    website_domain: Optional[str] = None
    extract_main_content: bool = False
    refify_urls: bool = False
    url_map: Dict[str, str] = field(default_factory=dict)
    debug: bool = False
    override_dom_parser: Optional[Callable[[str], Any]] = None  # Placeholder for DOMParser override
    enable_table_column_tracking: bool = False
    override_element_processing: Optional[Callable[[Any, 'ConversionOptions', int], Optional[List[SemanticMarkdownAST]]]] = None
    process_unhandled_element: Optional[Callable[[Any, 'ConversionOptions', int], Optional[List[SemanticMarkdownAST]]]] = None
    override_node_renderer: Optional[Callable[[SemanticMarkdownAST, 'ConversionOptions', int], Optional[str]]] = None
    render_custom_node: Optional[Callable[[CustomNode, 'ConversionOptions', int], Optional[str]]] = None
    include_meta_data: Union[str, bool] = False  # 'basic', 'extended', or False
