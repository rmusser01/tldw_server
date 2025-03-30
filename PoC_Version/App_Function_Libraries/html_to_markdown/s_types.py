# html_to_markdown/types.py

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any

@dataclass
class TextNode:
    type: str = "text"
    content: str = ""

@dataclass
class BoldNode:
    type: str = "bold"
    content: Union[str, List['SemanticMarkdownAST']] = ""

@dataclass
class ItalicNode:
    type: str = "italic"
    content: Union[str, List['SemanticMarkdownAST']] = ""

@dataclass
class StrikethroughNode:
    type: str = "strikethrough"
    content: Union[str, List['SemanticMarkdownAST']] = ""

@dataclass
class HeadingNode:
    type: str = "heading"
    level: int = 1
    content: str = ""

@dataclass
class LinkNode:
    type: str = "link"
    href: str = ""
    content: List['SemanticMarkdownAST'] = field(default_factory=list)

@dataclass
class ImageNode:
    type: str = "image"
    src: str = ""
    alt: Optional[str] = ""

@dataclass
class VideoNode:
    type: str = "video"
    src: str = ""
    poster: Optional[str] = ""
    controls: bool = False

@dataclass
class ListItemNode:
    type: str = "listItem"
    content: List['SemanticMarkdownAST'] = field(default_factory=list)

@dataclass
class ListNode:
    type: str = "list"
    ordered: bool = False
    items: List[ListItemNode] = field(default_factory=list)

@dataclass
class TableCellNode:
    type: str = "tableCell"
    content: Union[str, List['SemanticMarkdownAST']] = ""
    colId: Optional[str] = None
    colspan: Optional[int] = None
    rowspan: Optional[int] = None

@dataclass
class TableRowNode:
    type: str = "tableRow"
    cells: List[TableCellNode] = field(default_factory=list)

@dataclass
class TableNode:
    type: str = "table"
    rows: List[TableRowNode] = field(default_factory=list)
    colIds: Optional[List[str]] = None

@dataclass
class CodeNode:
    type: str = "code"
    language: Optional[str] = ""
    content: str = ""
    inline: bool = False

@dataclass
class BlockquoteNode:
    type: str = "blockquote"
    content: List['SemanticMarkdownAST'] = field(default_factory=list)

@dataclass
class CustomNode:
    type: str = "custom"
    content: Any = None

@dataclass
class SemanticHtmlNode:
    type: str = "semanticHtml"
    htmlType: str = ""
    content: List['SemanticMarkdownAST'] = field(default_factory=list)

@dataclass
class MetaDataNode:
    type: str = "meta"
    content: Dict[str, Any] = field(default_factory=dict)

# Union of all node types
SemanticMarkdownAST = Union[
    TextNode,
    BoldNode,
    ItalicNode,
    StrikethroughNode,
    HeadingNode,
    LinkNode,
    ImageNode,
    VideoNode,
    ListNode,
    TableNode,
    CodeNode,
    BlockquoteNode,
    SemanticHtmlNode,
    CustomNode,
    MetaDataNode
]
