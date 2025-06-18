"""
RAG (Retrieval-Augmented Generation) Service

This package provides a modular, extensible implementation of RAG functionality
for the tldw_chatbook application. It replaces the monolithic Unified_RAG_v2.py
with a clean, testable architecture.

Main components:
- app.py: Main RAGApplication class that orchestrates the service
- config.py: Configuration management with TOML integration
- retrieval.py: Document retrieval strategies for different data sources
- processing.py: Document processing, ranking, and deduplication
- generation.py: Response generation with LLM integration
- utils.py: Utility functions and helpers
"""

from .app import RAGApplication
from .config import RAGConfig

__all__ = ['RAGApplication', 'RAGConfig']