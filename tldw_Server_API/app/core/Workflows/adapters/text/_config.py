"""Pydantic config models for text adapters."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field

from tldw_Server_API.app.core.Workflows.adapters._base import BaseAdapterConfig


class HTMLToMarkdownConfig(BaseAdapterConfig):
    """Config for HTML to Markdown conversion adapter."""

    html: str = Field(..., description="HTML content to convert (templated)")
    strip_tags: list[str] | None = Field(None, description="HTML tags to strip")
    preserve_links: bool = Field(True, description="Preserve hyperlinks")
    preserve_images: bool = Field(True, description="Preserve image references")
    heading_style: Literal["atx", "setext"] = Field("atx", description="Heading style")


class MarkdownToHTMLConfig(BaseAdapterConfig):
    """Config for Markdown to HTML conversion adapter."""

    markdown: str = Field(..., description="Markdown content to convert (templated)")
    extensions: list[str] | None = Field(None, description="Markdown extensions to enable")
    safe_mode: bool = Field(True, description="Enable safe mode (sanitize HTML)")
    include_toc: bool = Field(False, description="Include table of contents")


class CSVToJSONConfig(BaseAdapterConfig):
    """Config for CSV to JSON conversion adapter."""

    csv: str = Field(..., description="CSV content to convert (templated)")
    delimiter: str = Field(",", description="CSV delimiter")
    has_header: bool = Field(True, description="First row is header")
    output_format: Literal["array", "object"] = Field("array", description="JSON output format")


class JSONToCSVConfig(BaseAdapterConfig):
    """Config for JSON to CSV conversion adapter."""

    json_data: Any = Field(..., description="JSON data to convert")
    include_header: bool = Field(True, description="Include header row")
    delimiter: str = Field(",", description="CSV delimiter")
    flatten_nested: bool = Field(True, description="Flatten nested objects")


class JSONTransformConfig(BaseAdapterConfig):
    """Config for JSON transformation adapter."""

    input: Any = Field(..., description="Input JSON data")
    jq_expression: str | None = Field(None, description="JQ expression for transformation")
    jsonpath: str | None = Field(None, description="JSONPath expression")
    template: str | None = Field(None, description="Jinja template for transformation")


class JSONValidateConfig(BaseAdapterConfig):
    """Config for JSON validation adapter."""

    model_config = ConfigDict(serialize_by_alias=True)

    data: Any = Field(..., description="JSON data to validate")
    schema_definition: dict[str, Any] = Field(..., alias="schema", description="JSON Schema for validation")
    strict: bool = Field(False, description="Strict validation mode")


class XMLTransformConfig(BaseAdapterConfig):
    """Config for XML transformation adapter."""

    xml: str = Field(..., description="XML content to transform (templated)")
    xslt: str | None = Field(None, description="XSLT stylesheet for transformation")
    xpath: str | None = Field(None, description="XPath expression for extraction")
    output_format: Literal["xml", "json", "text"] = Field("xml", description="Output format")


class TemplateRenderConfig(BaseAdapterConfig):
    """Config for Jinja template rendering adapter."""

    template: str = Field(..., description="Jinja2 template (templated)")
    variables: dict[str, Any] | None = Field(None, description="Template variables")
    strict: bool = Field(False, description="Strict undefined variable handling")
    autoescape: bool = Field(False, description="Enable HTML autoescaping")


class RegexExtractConfig(BaseAdapterConfig):
    """Config for regex extraction adapter."""

    text: str = Field(..., description="Text to extract from (templated)")
    pattern: str = Field(..., description="Regex pattern")
    flags: list[Literal["i", "m", "s", "x"]] | None = Field(
        None, description="Regex flags (i=ignorecase, m=multiline, s=dotall, x=verbose)"
    )
    group: int | None = Field(None, ge=0, description="Capture group to extract")
    all_matches: bool = Field(False, description="Return all matches")


class TextCleanConfig(BaseAdapterConfig):
    """Config for text cleaning adapter."""

    text: str = Field(..., description="Text to clean (templated)")
    operations: list[str] | None = Field(
        None,
        description=(
            "Ordered clean operations. Common values: strip_html, fix_encoding, strip_markdown, "
            "strip_reasoning_blocks, tts_normalize, normalize_whitespace, strip, lowercase, "
            "remove_urls, remove_emails, normalize_unicode"
        ),
    )
    remove_html: bool = Field(True, description="Remove HTML tags")
    remove_urls: bool = Field(False, description="Remove URLs")
    remove_emails: bool = Field(False, description="Remove email addresses")
    remove_numbers: bool = Field(False, description="Remove numbers")
    lowercase: bool = Field(False, description="Convert to lowercase")
    strip_whitespace: bool = Field(True, description="Strip extra whitespace")
    normalize_unicode: bool = Field(True, description="Normalize Unicode characters")


class KeywordExtractConfig(BaseAdapterConfig):
    """Config for keyword extraction adapter."""

    text: str = Field(..., description="Text to extract keywords from (templated)")
    method: Literal["tfidf", "rake", "yake", "keybert", "llm"] = Field(
        "yake", description="Extraction method"
    )
    max_keywords: int = Field(10, ge=1, le=100, description="Maximum keywords to extract")
    min_score: float | None = Field(None, ge=0, le=1, description="Minimum keyword score")
    ngram_range: tuple = Field((1, 3), description="N-gram range for extraction")
    provider: str | None = Field(None, description="LLM provider (for llm method)")
    model: str | None = Field(None, description="Model (for llm method)")


class SentimentAnalyzeConfig(BaseAdapterConfig):
    """Config for sentiment analysis adapter."""

    text: str = Field(..., description="Text to analyze (templated)")
    method: Literal["vader", "textblob", "transformers", "llm"] = Field(
        "vader", description="Analysis method"
    )
    granularity: Literal["document", "sentence", "aspect"] = Field(
        "document", description="Analysis granularity"
    )
    aspects: list[str] | None = Field(None, description="Aspects for aspect-based analysis")
    provider: str | None = Field(None, description="LLM provider (for llm method)")
    model: str | None = Field(None, description="Model (for llm method)")


class LanguageDetectConfig(BaseAdapterConfig):
    """Config for language detection adapter."""

    text: str = Field(..., description="Text to detect language of (templated)")
    method: Literal["langdetect", "fasttext", "transformers"] = Field(
        "langdetect", description="Detection method"
    )
    top_k: int = Field(1, ge=1, le=5, description="Number of top languages to return")
    min_confidence: float = Field(0.5, ge=0, le=1, description="Minimum confidence threshold")


class TopicModelConfig(BaseAdapterConfig):
    """Config for topic modeling adapter."""

    texts: list[str] = Field(..., description="Texts for topic modeling")
    method: Literal["lda", "nmf", "bertopic", "llm"] = Field(
        "lda", description="Topic modeling method"
    )
    num_topics: int = Field(5, ge=2, le=50, description="Number of topics to extract")
    words_per_topic: int = Field(10, ge=3, le=30, description="Words per topic")
    provider: str | None = Field(None, description="LLM provider (for llm method)")
    model: str | None = Field(None, description="Model (for llm method)")


class EntityExtractConfig(BaseAdapterConfig):
    """Config for named entity extraction adapter."""

    text: str = Field(..., description="Text to extract entities from (templated)")
    method: Literal["spacy", "transformers", "llm"] = Field(
        "spacy", description="Extraction method"
    )
    entity_types: list[str] | None = Field(
        None, description="Entity types to extract (PERSON, ORG, LOC, etc.)"
    )
    link_entities: bool = Field(False, description="Link entities to knowledge base")
    provider: str | None = Field(None, description="LLM provider (for llm method)")
    model: str | None = Field(None, description="Model (for llm method)")


class TokenCountConfig(BaseAdapterConfig):
    """Config for token counting adapter."""

    text: str = Field(..., description="Text to count tokens in (templated)")
    model: str = Field("gpt-4", description="Model to use for tokenization")
    encoding: str | None = Field(None, description="Specific encoding to use")
