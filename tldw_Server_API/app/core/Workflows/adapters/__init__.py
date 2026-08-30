"""Workflow adapters submodule.

This module provides a decorator-based registry system for workflow step adapters.
All adapters are registered during import and can be looked up by step type name.

Usage:
    from tldw_Server_API.app.core.Workflows.adapters import get_adapter, registry

    # Get an adapter by name
    adapter = get_adapter("llm")
    if adapter:
        result = await adapter(config, context)

    # Get all parallelizable adapter names
    parallel_adapters = get_parallelizable()

    # Get the full catalog
    catalog = registry.get_catalog()
"""

import asyncio  # Re-export for tests that patch adapters.asyncio
from typing import Any

# Re-export exceptions and internal module references for backward compatibility
from tldw_Server_API.app.core.exceptions import AdapterError
from tldw_Server_API.app.core.http_client import create_client as _wf_create_client

# Import all category modules to register their adapters
# Each module's import triggers the @registry.register decorators
from tldw_Server_API.app.core.Workflows.adapters._base import (
    AdapterContext,
    AdapterFunc,
    AdapterResult,
    BaseAdapterConfig,
)
from tldw_Server_API.app.core.Workflows.adapters._common import (
    AsyncFileWriter,
    _artifacts_base_dir,
    _async_file_writer,
    _extract_mcp_policy,
    # Backward compatible underscore-prefixed aliases
    _extract_openai_content,
    _extract_tool_scopes,
    _format_time_srt,
    _format_time_vtt,
    _is_subpath,
    _normalize_str_list,
    _resolve_artifact_filename,
    _resolve_artifacts_dir,
    _resolve_context_user_id,
    _resolve_workflow_file_path,
    _resolve_workflow_file_uri,
    _sanitize_path_component,
    _tool_matches_allowlist,
    _unsafe_file_access_allowed,
    _workflow_file_base_dir,
    artifacts_base_dir,
    extract_mcp_policy,
    extract_openai_content,
    extract_tool_scopes,
    format_time_srt,
    format_time_vtt,
    is_subpath,
    normalize_str_list,
    resolve_artifact_filename,
    resolve_artifacts_dir,
    resolve_context_user_id,
    resolve_workflow_file_path,
    resolve_workflow_file_uri,
    sanitize_path_component,
    tool_matches_allowlist,
    unsafe_file_access_allowed,
    workflow_file_base_dir,
)
from tldw_Server_API.app.core.Workflows.adapters._registry import (
    AdapterRegistry,
    AdapterSpec,
    get_adapter,
    get_parallelizable,
    registry,
)

# Re-export all adapter functions for backward compatibility
# This allows existing code to import like: from ...adapters import run_llm_adapter
# Audio adapters

# Content adapters

# Control adapters

# Evaluation adapters

# Integration adapters

# Knowledge adapters

# LLM adapters

# Media adapters

# RAG adapters

# Research adapters

# Text adapters

# Utility adapters

# Video adapters

# ---------------------------------------------------------------------------
# Lazy category loading
#
# Importing every category module at package import time ran each module's
# @registry.register decorators, which is what made the registry work -- but it
# also reached RAG.query_features (nltk -> scipy -> sklearn -> pandas),
# Kanban_DB (chromadb) and TTS (av). That cost roughly 1.4 s and ~1,800 modules
# in every process that imported workflows, including CLI runs and pytest.
#
# Categories are now imported on first registry access, or on first attribute
# access for a re-exported adapter function.
# ---------------------------------------------------------------------------

_ADAPTER_CATEGORIES: tuple[str, ...] = (
    "audio",
    "content",
    "control",
    "evaluation",
    "integration",
    "knowledge",
    "llm",
    "media",
    "rag",
    "research",
    "text",
    "utility",
    "video",
)

# Re-exported adapter function name -> category module that defines it.
_LAZY_ADAPTER_EXPORTS: dict[str, str] = {
    "run_acp_stage_adapter": "integration",
    "run_arxiv_download_adapter": "research",
    "run_arxiv_search_adapter": "research",
    "run_audio_concat_adapter": "audio",
    "run_audio_convert_adapter": "audio",
    "run_audio_diarize_adapter": "audio",
    "run_audio_extract_adapter": "audio",
    "run_audio_mix_adapter": "audio",
    "run_audio_normalize_adapter": "audio",
    "run_audio_trim_adapter": "audio",
    "run_batch_adapter": "control",
    "run_bibliography_generate_adapter": "content",
    "run_bibtex_generate_adapter": "research",
    "run_branch_adapter": "control",
    "run_cache_result_adapter": "control",
    "run_character_chat_adapter": "integration",
    "run_chatbooks_adapter": "integration",
    "run_checkpoint_adapter": "control",
    "run_chunking_adapter": "knowledge",
    "run_citations_adapter": "content",
    "run_claims_extract_adapter": "knowledge",
    "run_collections_adapter": "knowledge",
    "run_context_build_adapter": "utility",
    "run_context_window_check_adapter": "evaluation",
    "run_csv_to_json_adapter": "text",
    "run_deep_research_adapter": "research",
    "run_deep_research_load_bundle_adapter": "research",
    "run_deep_research_select_bundle_fields_adapter": "research",
    "run_deep_research_wait_adapter": "research",
    "run_delay_adapter": "control",
    "run_diagram_generate_adapter": "content",
    "run_diff_change_adapter": "utility",
    "run_document_diff_adapter": "utility",
    "run_document_merge_adapter": "utility",
    "run_document_table_extract_adapter": "media",
    "run_doi_resolve_adapter": "research",
    "run_email_send_adapter": "integration",
    "run_embed_adapter": "utility",
    "run_entity_extract_adapter": "text",
    "run_eval_readability_adapter": "evaluation",
    "run_evaluations_adapter": "evaluation",
    "run_flashcard_generate_adapter": "content",
    "run_github_create_issue_adapter": "integration",
    "run_glossary_extract_adapter": "content",
    "run_google_scholar_search_adapter": "research",
    "run_html_to_markdown_adapter": "text",
    "run_hyde_generate_adapter": "rag",
    "run_image_describe_adapter": "content",
    "run_image_gen_adapter": "content",
    "run_json_to_csv_adapter": "text",
    "run_json_transform_adapter": "text",
    "run_json_validate_adapter": "text",
    "run_kanban_adapter": "integration",
    "run_keyword_extract_adapter": "text",
    "run_language_detect_adapter": "text",
    "run_literature_review_adapter": "research",
    "run_llm_adapter": "llm",
    "run_llm_compare_adapter": "llm",
    "run_llm_critique_adapter": "llm",
    "run_llm_with_tools_adapter": "llm",
    "run_log_adapter": "control",
    "run_map_adapter": "control",
    "run_markdown_to_html_adapter": "text",
    "run_mcp_tool_adapter": "integration",
    "run_media_ingest_adapter": "media",
    "run_mindmap_generate_adapter": "content",
    "run_moderation_adapter": "llm",
    "run_newsletter_generate_adapter": "content",
    "run_notes_adapter": "knowledge",
    "run_notify_adapter": "integration",
    "run_ocr_adapter": "media",
    "run_outline_generate_adapter": "content",
    "run_parallel_adapter": "control",
    "run_patent_search_adapter": "research",
    "run_pdf_extract_adapter": "media",
    "run_podcast_rss_publish_adapter": "integration",
    "run_policy_check_adapter": "llm",
    "run_process_media_adapter": "media",
    "run_prompt_adapter": "control",
    "run_prompts_adapter": "knowledge",
    "run_pubmed_search_adapter": "research",
    "run_query_expand_adapter": "rag",
    "run_query_rewrite_adapter": "rag",
    "run_quiz_evaluate_adapter": "evaluation",
    "run_quiz_generate_adapter": "content",
    "run_rag_search_adapter": "rag",
    "run_reference_parse_adapter": "research",
    "run_regex_extract_adapter": "text",
    "run_report_generate_adapter": "content",
    "run_rerank_adapter": "content",
    "run_retry_adapter": "control",
    "run_rss_fetch_adapter": "rag",
    "run_s3_download_adapter": "integration",
    "run_s3_upload_adapter": "integration",
    "run_sandbox_exec_adapter": "utility",
    "run_schedule_workflow_adapter": "utility",
    "run_screenshot_capture_adapter": "utility",
    "run_search_aggregate_adapter": "rag",
    "run_semantic_cache_check_adapter": "rag",
    "run_semantic_scholar_search_adapter": "research",
    "run_sentiment_analyze_adapter": "text",
    "run_slides_generate_adapter": "content",
    "run_stt_transcribe_adapter": "audio",
    "run_subtitle_burn_adapter": "video",
    "run_subtitle_generate_adapter": "video",
    "run_subtitle_translate_adapter": "video",
    "run_summarize_adapter": "content",
    "run_template_render_adapter": "text",
    "run_text_clean_adapter": "text",
    "run_timing_start_adapter": "utility",
    "run_timing_stop_adapter": "utility",
    "run_token_count_adapter": "text",
    "run_topic_model_adapter": "text",
    "run_translate_adapter": "llm",
    "run_tts_adapter": "audio",
    "run_video_concat_adapter": "video",
    "run_video_convert_adapter": "video",
    "run_video_extract_frames_adapter": "video",
    "run_video_thumbnail_adapter": "video",
    "run_video_trim_adapter": "video",
    "run_voice_intent_adapter": "knowledge",
    "run_web_search_adapter": "rag",
    "run_webhook_adapter": "integration",
    "run_workflow_call_adapter": "control",
    "run_xml_transform_adapter": "text",
}


def _load_all_categories() -> None:
    """Import every adapter category so its @registry.register calls run."""
    from importlib import import_module

    for _cat in _ADAPTER_CATEGORIES:
        import_module(f"{__name__}.{_cat}")


registry.set_loader(_load_all_categories)


def __getattr__(name: str) -> Any:
    """Resolve category modules and re-exported adapters on first access.

    Args:
        name: Attribute requested from this package.

    Returns:
        The category module or adapter function bound to ``name``.

    Raises:
        AttributeError: If ``name`` is neither a category nor a known adapter.
    """
    from importlib import import_module

    if name in _ADAPTER_CATEGORIES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    category = _LAZY_ADAPTER_EXPORTS.get(name)
    if category is not None:
        value = getattr(import_module(f"{__name__}.{category}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the lazily resolvable names alongside the eager ones."""
    return sorted(set(globals()) | set(__all__))


__all__ = [
    # Compatibility exports
    "asyncio",
    "AdapterError",
    "_wf_create_client",
    "_artifacts_base_dir",
    "_async_file_writer",
    "_extract_mcp_policy",
    "_extract_openai_content",
    "_extract_tool_scopes",
    "_format_time_srt",
    "_format_time_vtt",
    "_is_subpath",
    "_normalize_str_list",
    "_resolve_artifact_filename",
    "_resolve_artifacts_dir",
    "_resolve_context_user_id",
    "_resolve_workflow_file_path",
    "_resolve_workflow_file_uri",
    "_sanitize_path_component",
    "_tool_matches_allowlist",
    "_unsafe_file_access_allowed",
    "_workflow_file_base_dir",
    # Registry
    "registry",
    "get_adapter",
    "get_parallelizable",
    "AdapterSpec",
    "AdapterRegistry",
    # Base types
    "AdapterContext",
    "BaseAdapterConfig",
    "AdapterFunc",
    "AdapterResult",
    # Common utilities
    "extract_openai_content",
    "sanitize_path_component",
    "is_subpath",
    "resolve_context_user_id",
    "artifacts_base_dir",
    "resolve_artifacts_dir",
    "resolve_artifact_filename",
    "unsafe_file_access_allowed",
    "workflow_file_base_dir",
    "resolve_workflow_file_path",
    "resolve_workflow_file_uri",
    "normalize_str_list",
    "extract_mcp_policy",
    "tool_matches_allowlist",
    "extract_tool_scopes",
    "format_time_srt",
    "format_time_vtt",
    "AsyncFileWriter",
    # Category modules
    "control",
    "llm",
    "audio",
    "video",
    "media",
    "rag",
    "knowledge",
    "content",
    "text",
    "integration",
    "evaluation",
    "research",
    "utility",
    # Audio adapter exports
    "run_tts_adapter",
    "run_stt_transcribe_adapter",
    "run_audio_normalize_adapter",
    "run_audio_concat_adapter",
    "run_audio_trim_adapter",
    "run_audio_convert_adapter",
    "run_audio_extract_adapter",
    "run_audio_mix_adapter",
    "run_audio_diarize_adapter",
    # Video adapter exports
    "run_video_thumbnail_adapter",
    "run_video_trim_adapter",
    "run_video_concat_adapter",
    "run_video_convert_adapter",
    "run_video_extract_frames_adapter",
    "run_subtitle_generate_adapter",
    "run_subtitle_translate_adapter",
    "run_subtitle_burn_adapter",
    # Media adapter exports
    "run_media_ingest_adapter",
    "run_process_media_adapter",
    "run_pdf_extract_adapter",
    "run_ocr_adapter",
    "run_document_table_extract_adapter",
    # RAG adapter exports
    "run_rag_search_adapter",
    "run_web_search_adapter",
    "run_rss_fetch_adapter",
    "run_query_rewrite_adapter",
    "run_query_expand_adapter",
    "run_hyde_generate_adapter",
    "run_semantic_cache_check_adapter",
    "run_search_aggregate_adapter",
    # Knowledge adapter exports
    "run_notes_adapter",
    "run_prompts_adapter",
    "run_collections_adapter",
    "run_chunking_adapter",
    "run_claims_extract_adapter",
    "run_voice_intent_adapter",
    # Content adapter exports
    "run_summarize_adapter",
    "run_citations_adapter",
    "run_bibliography_generate_adapter",
    "run_image_gen_adapter",
    "run_image_describe_adapter",
    "run_rerank_adapter",
    "run_flashcard_generate_adapter",
    "run_quiz_generate_adapter",
    "run_outline_generate_adapter",
    "run_glossary_extract_adapter",
    "run_mindmap_generate_adapter",
    "run_report_generate_adapter",
    "run_newsletter_generate_adapter",
    "run_slides_generate_adapter",
    "run_diagram_generate_adapter",
    # Text adapter exports
    "run_html_to_markdown_adapter",
    "run_markdown_to_html_adapter",
    "run_json_transform_adapter",
    "run_json_validate_adapter",
    "run_csv_to_json_adapter",
    "run_json_to_csv_adapter",
    "run_xml_transform_adapter",
    "run_template_render_adapter",
    "run_regex_extract_adapter",
    "run_text_clean_adapter",
    "run_keyword_extract_adapter",
    "run_sentiment_analyze_adapter",
    "run_language_detect_adapter",
    "run_topic_model_adapter",
    "run_entity_extract_adapter",
    "run_token_count_adapter",
    # Integration adapter exports
    "run_webhook_adapter",
    "run_notify_adapter",
    "run_mcp_tool_adapter",
    "run_acp_stage_adapter",
    "run_s3_upload_adapter",
    "run_s3_download_adapter",
    "run_podcast_rss_publish_adapter",
    "run_github_create_issue_adapter",
    "run_kanban_adapter",
    "run_chatbooks_adapter",
    "run_character_chat_adapter",
    "run_email_send_adapter",
    # Evaluation adapter exports
    "run_evaluations_adapter",
    "run_quiz_evaluate_adapter",
    "run_eval_readability_adapter",
    "run_context_window_check_adapter",
    # Research adapter exports
    "run_arxiv_search_adapter",
    "run_arxiv_download_adapter",
    "run_pubmed_search_adapter",
    "run_semantic_scholar_search_adapter",
    "run_google_scholar_search_adapter",
    "run_patent_search_adapter",
    "run_deep_research_adapter",
    "run_deep_research_wait_adapter",
    "run_deep_research_load_bundle_adapter",
    "run_deep_research_select_bundle_fields_adapter",
    "run_doi_resolve_adapter",
    "run_reference_parse_adapter",
    "run_bibtex_generate_adapter",
    "run_literature_review_adapter",
    # Utility adapter exports
    "run_diff_change_adapter",
    "run_document_diff_adapter",
    "run_document_merge_adapter",
    "run_context_build_adapter",
    "run_embed_adapter",
    "run_sandbox_exec_adapter",
    "run_screenshot_capture_adapter",
    "run_schedule_workflow_adapter",
    "run_timing_start_adapter",
    "run_timing_stop_adapter",
    # Control adapter exports
    "run_prompt_adapter",
    "run_delay_adapter",
    "run_log_adapter",
    "run_branch_adapter",
    "run_map_adapter",
    "run_parallel_adapter",
    "run_batch_adapter",
    "run_cache_result_adapter",
    "run_retry_adapter",
    "run_checkpoint_adapter",
    "run_workflow_call_adapter",
    # LLM adapter exports
    "run_llm_adapter",
    "run_llm_with_tools_adapter",
    "run_llm_compare_adapter",
    "run_llm_critique_adapter",
    "run_moderation_adapter",
    "run_policy_check_adapter",
    "run_translate_adapter",
]
