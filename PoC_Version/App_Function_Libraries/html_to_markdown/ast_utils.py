# html_to_markdown/ast_utils.py

from typing import Callable, Optional, List, Union
from s_types import SemanticMarkdownAST

def find_in_ast(ast: Union[SemanticMarkdownAST, List[SemanticMarkdownAST]], predicate: Callable[[SemanticMarkdownAST], bool]) -> Optional[SemanticMarkdownAST]:
    if isinstance(ast, list):
        for node in ast:
            result = find_in_ast(node, predicate)
            if result:
                return result
    else:
        if predicate(ast):
            return ast
        # Recursively search based on node type
        if hasattr(ast, 'content'):
            content = ast.content
            if isinstance(content, list):
                result = find_in_ast(content, predicate)
                if result:
                    return result
            elif isinstance(content, SemanticMarkdownAST):
                result = find_in_ast(content, predicate)
                if result:
                    return result
        if hasattr(ast, 'items'):
            for item in ast.items:
                result = find_in_ast(item, predicate)
                if result:
                    return result
        if hasattr(ast, 'rows'):
            for row in ast.rows:
                result = find_in_ast(row, predicate)
                if result:
                    return result
    return None

def find_all_in_ast(ast: Union[SemanticMarkdownAST, List[SemanticMarkdownAST]], predicate: Callable[[SemanticMarkdownAST], bool]) -> List[SemanticMarkdownAST]:
    results = []
    if isinstance(ast, list):
        for node in ast:
            results.extend(find_all_in_ast(node, predicate))
    else:
        if predicate(ast):
            results.append(ast)
        # Recursively search based on node type
        if hasattr(ast, 'content'):
            content = ast.content
            if isinstance(content, list):
                results.extend(find_all_in_ast(content, predicate))
            elif isinstance(content, SemanticMarkdownAST):
                results.extend(find_all_in_ast(content, predicate))
        if hasattr(ast, 'items'):
            for item in ast.items:
                results.extend(find_all_in_ast(item, predicate))
        if hasattr(ast, 'rows'):
            for row in ast.rows:
                results.extend(find_all_in_ast(row, predicate))
    return results
