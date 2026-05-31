"""
RAG (Retrieval-Augmented Generation) Service

This package provides a functional pipeline implementation of RAG functionality
for the tldw_server application. It uses composable functions that can be
combined into custom pipelines for different use cases.

Main components:
- functional_pipeline.py: Core pipeline functions and presets
- config.py: Configuration management
- types.py: Type definitions
- database_retrievers.py: Database retrieval strategies
- query_expansion.py: Query expansion strategies
- advanced_reranking.py: Document reranking
- Various feature modules for caching, monitoring, etc.
"""

from importlib import import_module
from typing import Any

# Expose commonly patched submodules for tests
from . import (
    advanced_reranking,  # noqa: F401
    chromadb_optimizer,  # noqa: F401
    semantic_cache,  # noqa: F401
)
from .config import RAGConfig
from .types import DataSource, Document, SearchResult

_LAZY_EXPORTS = {
    "ResolvedRAGRequest": (".request_resolution", "ResolvedRAGRequest"),
    "resolve_rag_request": (".request_resolution", "resolve_rag_request"),
    "RAGResult": (".result_model", "RAGResult"),
    "rag_result_from_unified_search_result": (".response_mapping", "rag_result_from_unified_search_result"),
    "rag_result_to_response": (".response_mapping", "rag_result_to_response"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    'RAGConfig', 'DataSource', 'Document', 'SearchResult',
    'semantic_cache', 'chromadb_optimizer', 'advanced_reranking',
    'ResolvedRAGRequest', 'resolve_rag_request',
    'RAGResult', 'rag_result_from_unified_search_result', 'rag_result_to_response',
]
