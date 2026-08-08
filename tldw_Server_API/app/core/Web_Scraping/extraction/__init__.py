"""Canonical facade for HTML extraction foundations."""

from .caches import clear_extraction_caches, get_extraction_cache_stats
from .dependencies import ExtractionDependencies, build_default_dependencies
from .pipeline import (
    DEFAULT_EXTRACTION_STRATEGY_ORDER,
    extract_article_data_from_html,
    extract_article_with_pipeline,
)
from .strategies import (
    extract_cluster_entities,
    extract_jsonld_entities,
    extract_llm_entities,
    extract_regex_entities,
    generate_regex_pattern_from_llm,
    generate_schema_rules_from_llm,
)

__all__ = [
    "ExtractionDependencies",
    "build_default_dependencies",
    "DEFAULT_EXTRACTION_STRATEGY_ORDER",
    "clear_extraction_caches",
    "get_extraction_cache_stats",
    "extract_cluster_entities",
    "extract_article_data_from_html",
    "extract_article_with_pipeline",
    "extract_jsonld_entities",
    "extract_llm_entities",
    "extract_regex_entities",
    "generate_regex_pattern_from_llm",
    "generate_schema_rules_from_llm",
]
