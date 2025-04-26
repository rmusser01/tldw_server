# html_to_markdown/markdown_ast_to_string.py
import json
from ast_utils import find_in_ast
from typing import List, Optional, Union
from s_types import (
    SemanticMarkdownAST, TextNode, BoldNode, ItalicNode, StrikethroughNode,
    HeadingNode, LinkNode, ImageNode, VideoNode, ListNode, ListItemNode,
    TableNode, TableRowNode, TableCellNode, CodeNode, BlockquoteNode,
    SemanticHtmlNode, CustomNode, MetaDataNode
)
from conversion_options import ConversionOptions
import logging

def markdown_ast_to_string(nodes: List[SemanticMarkdownAST], options: Optional[ConversionOptions] = None, indent_level: int = 0) -> str:
    if options is None:
        options = ConversionOptions()

    markdown_string = ""
    markdown_string += markdown_meta_ast_to_string(nodes, options, indent_level)
    markdown_string += markdown_content_ast_to_string(nodes, options, indent_level)
    return markdown_string

def markdown_meta_ast_to_string(nodes: List[SemanticMarkdownAST], options: ConversionOptions, indent_level: int) -> str:
    markdown_string = ""
    if options.include_meta_data:
        markdown_string += "---\n"
        node = find_in_ast(nodes, lambda x: isinstance(x, MetaDataNode))
        if node and isinstance(node, MetaDataNode):
            standard = node.content.get('standard', {})
            for key, value in standard.items():
                markdown_string += f'{key}: "{value}"\n'
            if options.include_meta_data == 'extended':
                open_graph = node.content.get('openGraph', {})
                twitter = node.content.get('twitter', {})
                json_ld = node.content.get('jsonLd', [])

                if open_graph:
                    markdown_string += "openGraph:\n"
                    for key, value in open_graph.items():
                        markdown_string += f"  {key}: \"{value}\"\n"

                if twitter:
                    markdown_string += "twitter:\n"
                    for key, value in twitter.items():
                        markdown_string += f"  {key}: \"{value}\"\n"

                if json_ld:
                    markdown_string += "schema:\n"
                    for item in json_ld:
                        jld_type = item.get('@type', '(unknown type)')
                        markdown_string += f"  {jld_type}:\n"
                        for key, value in item.items():
                            if key in ['@context', '@type']:
                                continue
                            markdown_string += f"    {key}: {json.dumps(value)}\n"
        markdown_string += "---\n\n"
    return markdown_string

def markdown_content_ast_to_string(nodes: List[SemanticMarkdownAST], options: ConversionOptions, indent_level: int) -> str:
    markdown_string = ""
    for node in nodes:
        # Skip meta nodes as they are already handled
        if isinstance(node, MetaDataNode):
            continue

        # Override node renderer if provided
        if options.override_node_renderer:
            override = options.override_node_renderer(node, options, indent_level)
            if override:
                markdown_string += override
                continue

        if isinstance(node, TextNode):
            markdown_string += f"{node.content}"
        elif isinstance(node, BoldNode):
            content = ast_to_markdown(node.content, options, indent_level)
            markdown_string += f"**{content}**"
        elif isinstance(node, ItalicNode):
            content = ast_to_markdown(node.content, options, indent_level)
            markdown_string += f"*{content}*"
        elif isinstance(node, StrikethroughNode):
            content = ast_to_markdown(node.content, options, indent_level)
            markdown_string += f"~~{content}~~"
        elif isinstance(node, HeadingNode):
            markdown_string += f"\n{'#' * node.level} {node.content}\n\n"
        elif isinstance(node, LinkNode):
            content = ast_to_markdown(node.content, options, indent_level)
            if all(isinstance(c, TextNode) for c in node.content):
                markdown_string += f"[{content}]({node.href})"
            else:
                # Use HTML <a> tag for links with rich content
                markdown_string += f"<a href=\"{node.href}\">{content}</a>"
        elif isinstance(node, ImageNode):
            alt = node.alt or ""
            src = node.src or ""
            if alt.strip() or src.strip():
                markdown_string += f"![{alt}]({src})"
        elif isinstance(node, VideoNode):
            markdown_string += f"\n![Video]({node.src})\n"
            if node.poster:
                markdown_string += f"![Poster]({node.poster})\n"
            if node.controls:
                markdown_string += f"Controls: {node.controls}\n"
            markdown_string += "\n"
        elif isinstance(node, ListNode):
            for idx, item in enumerate(node.items):
                prefix = f"{idx + 1}." if node.ordered else "-"
                content = ast_to_markdown(item.content, options, indent_level +1).strip()
                markdown_string += f"{'  ' * indent_level}{prefix} {content}\n"
            markdown_string += "\n"
        elif isinstance(node, TableNode):
            if not node.rows:
                continue
            max_columns = max(
                sum(cell.colspan or 1 for cell in row.cells) for row in node.rows
            )
            for row_idx, row in enumerate(node.rows):
                for cell in row.cells:
                    content = cell.content if isinstance(cell.content, str) else ast_to_markdown(cell.content, options, indent_level +1).strip()
                    markdown_string += f"| {content} "
                # Fill remaining columns
                remaining = max_columns - sum(cell.colspan or 1 for cell in row.cells)
                for _ in range(remaining):
                    markdown_string += "|  "
                markdown_string += "|\n"
                if row_idx == 0:
                    # Add header separator
                    markdown_string += "|" + "|".join([' --- ' for _ in range(max_columns)]) + "|\n"
            markdown_string += "\n"
        elif isinstance(node, CodeNode):
            if node.inline:
                markdown_string += f"`{node.content}`"
            else:
                language = node.language or ""
                markdown_string += f"\n```{language}\n{node.content}\n```\n\n"
        elif isinstance(node, BlockquoteNode):
            content = ast_to_markdown(node.content, options, indent_level).strip()
            markdown_string += f"> {content}\n\n"
        elif isinstance(node, SemanticHtmlNode):
            if node.htmlType in ["summary", "time", "aside", "nav", "figcaption", "main", "mark", "header", "footer", "details", "figure"]:
                markdown_string += f"\n<-{node.htmlType}->\n{ast_to_markdown(node.content, options, indent_level)}\n\n</-{node.htmlType}->\n\n"
            elif node.htmlType == "article":
                markdown_string += f"\n\n{ast_to_markdown(node.content, options, indent_level)}\n\n"
            elif node.htmlType == "section":
                markdown_string += "---\n\n"
                markdown_string += f"{ast_to_markdown(node.content, options, indent_level)}\n\n---\n\n"
        elif isinstance(node, CustomNode):
            if options.render_custom_node:
                custom_render = options.render_custom_node(node, options, indent_level)
                if custom_render:
                    markdown_string += custom_render
        # Add more node types as needed
    return markdown_string

def ast_to_markdown(content: Union[str, List[SemanticMarkdownAST]], options: ConversionOptions, indent_level: int) -> str:
    if isinstance(content, str):
        return content
    else:
        return markdown_content_ast_to_string(content, options, indent_level)

