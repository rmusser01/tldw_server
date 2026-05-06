from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from tldw_Server_API.app.core.Chunking.auto_planner import (
    merge_profiles,
    plan_auto_chunking,
    profile_from_source,
    profile_from_text,
)
from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.Utils.Utils import logging

_UNSUPPORTED_MEDIA_CHUNKING_KEYS = (
    "tokenizer_name_or_path",
    "code_mode",
    "semantic_similarity_threshold",
    "json_chunkable_data_key",
    "summarization_detail",
    "llm_options_for_internal_steps",
    "enable_frontmatter_parsing",
    "frontmatter_sentinel_key",
)

_ALLOWED_MEDIA_CHUNKING_KEYS = (
    "method",
    "max_size",
    "overlap",
    "adaptive",
    "multi_level",
    "language",
    "custom_chapter_pattern",
    "enable_contextual_chunking",
    "contextual_llm_model",
    "context_window_size",
    "context_strategy",
    "context_token_budget",
    "proposition_engine",
    "proposition_aggressiveness",
    "proposition_min_proposition_length",
    "proposition_prompt_profile",
)

_CHUNKING_OPTIONS_COERCE_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)

_CHUNKING_OPTIONS_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)


def _get_raw_form_value(form_data: Any, key: str) -> Any:
    if isinstance(form_data, Mapping):
        return form_data.get(key)
    if hasattr(form_data, "model_dump"):
        try:
            dumped = form_data.model_dump()
        except _CHUNKING_OPTIONS_COERCE_EXCEPTIONS:
            dumped = {}
        if isinstance(dumped, Mapping):
            return dumped.get(key)
    return getattr(form_data, key, None)


def _find_unsupported_chunking_keys(form_data: Any) -> list[str]:
    unsupported: list[str] = []
    for key in _UNSUPPORTED_MEDIA_CHUNKING_KEYS:
        value = _get_raw_form_value(form_data, key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, bool) and value is False:
            continue
        unsupported.append(key)
    return unsupported


def _validate_media_chunking_options(form_data: Any) -> None:
    unsupported = _find_unsupported_chunking_keys(form_data)
    if not unsupported:
        return
    allowed = ", ".join(_ALLOWED_MEDIA_CHUNKING_KEYS)
    raise ValueError(
        "Unsupported chunking options for media processing: "
        f"{', '.join(unsupported)}. Supported options: {allowed}."
    )


def prepare_chunking_options_dict(form_data: Any) -> dict[str, Any] | None:
    """
    Prepare the dictionary of chunking options based on form data.

    This is extracted from the former endpoint-local chunking helper so it
    can be reused by core ingestion helpers and modular endpoints.
    """
    if not getattr(form_data, "perform_chunking", False):
        logging.info("Chunking disabled.")
        return None

    _validate_media_chunking_options(form_data)

    default_chunk_method = "sentences"
    media_type = str(getattr(form_data, "media_type", "") or "")
    if media_type == "ebook":
        default_chunk_method = "ebook_chapters"
        logging.info("Setting chunk method to 'ebook_chapters' for ebook type.")
    elif media_type in ["video", "audio"]:
        default_chunk_method = "sentences"

    final_chunk_method = getattr(form_data, "chunk_method", None) or default_chunk_method

    chunk_size_used = getattr(form_data, "chunk_size", None)
    chunk_overlap_used = getattr(form_data, "chunk_overlap", None)

    if media_type in ["document", "email"]:
        try:
            if chunk_size_used is None or int(chunk_size_used) == 500:
                chunk_size_used = 1000
        except _CHUNKING_OPTIONS_COERCE_EXCEPTIONS:
            chunk_size_used = 1000

    if media_type == "email":
        try:
            if chunk_overlap_used is None or int(chunk_overlap_used) == 200:
                chunk_overlap_used = 150
        except _CHUNKING_OPTIONS_COERCE_EXCEPTIONS:
            chunk_overlap_used = 150

    if media_type == "ebook":
        final_chunk_method = "ebook_chapters"

    inferred_enable_contextual = bool(
        getattr(form_data, "contextual_llm_model", None)
        or getattr(form_data, "context_window_size", None)
    )

    language: str | None
    if media_type in ["audio", "video"]:
        language = getattr(form_data, "chunk_language", None) or getattr(
            form_data, "transcription_language", None
        )
    else:
        language = getattr(form_data, "chunk_language", None)

    chunk_options: dict[str, Any] = {
        "method": final_chunk_method,
        "max_size": chunk_size_used,
        "overlap": chunk_overlap_used,
        "adaptive": getattr(form_data, "use_adaptive_chunking", False),
        "multi_level": getattr(form_data, "use_multi_level_chunking", False),
        "language": language,
        "custom_chapter_pattern": getattr(form_data, "custom_chapter_pattern", None),
        "enable_contextual_chunking": bool(
            getattr(form_data, "enable_contextual_chunking", False)
            or inferred_enable_contextual
        ),
        "contextual_llm_model": getattr(form_data, "contextual_llm_model", None),
        "context_window_size": getattr(form_data, "context_window_size", None),
        "context_strategy": getattr(form_data, "context_strategy", None),
        "context_token_budget": getattr(form_data, "context_token_budget", None),
    }

    proposition_engine = getattr(form_data, "proposition_engine", None)
    if proposition_engine:
        chunk_options["proposition_engine"] = proposition_engine
    proposition_aggressiveness = getattr(form_data, "proposition_aggressiveness", None)
    if proposition_aggressiveness is not None:
        chunk_options["proposition_aggressiveness"] = proposition_aggressiveness
    proposition_min_len = getattr(form_data, "proposition_min_proposition_length", None)
    if proposition_min_len is not None:
        chunk_options["proposition_min_proposition_length"] = proposition_min_len
    proposition_prompt_profile = getattr(form_data, "proposition_prompt_profile", None)
    if proposition_prompt_profile:
        chunk_options["proposition_prompt_profile"] = proposition_prompt_profile

    try:
        hier_flag = getattr(form_data, "hierarchical_chunking", None)
        hier_template = getattr(form_data, "hierarchical_template", None)
        if hier_flag is True or (hier_template and isinstance(hier_template, dict)):
            chunk_options["hierarchical"] = True
            if isinstance(hier_template, dict):
                chunk_options["hierarchical_template"] = hier_template
            chunk_options.setdefault("method", "sentences")
    except _CHUNKING_OPTIONS_COERCE_EXCEPTIONS:
        pass

    if final_chunk_method == "propositions":
        try:
            cfg = load_and_log_configs()
            cfg_dict = cfg if isinstance(cfg, dict) else {}
            c = cfg_dict.get("chunking_config", {}) if isinstance(cfg_dict, dict) else {}
            if "proposition_engine" in c and "proposition_engine" not in chunk_options:
                chunk_options["proposition_engine"] = c.get("proposition_engine")
            if (
                "proposition_prompt_profile" in c
                and "proposition_prompt_profile" not in chunk_options
            ):
                chunk_options["proposition_prompt_profile"] = c.get(
                    "proposition_prompt_profile"
                )
            if "proposition_aggressiveness" in c:
                try:
                    if "proposition_aggressiveness" not in chunk_options:
                        chunk_options["proposition_aggressiveness"] = int(
                            c.get("proposition_aggressiveness")
                        )
                except _CHUNKING_OPTIONS_COERCE_EXCEPTIONS:
                    pass
            if "proposition_min_proposition_length" in c:
                try:
                    if "proposition_min_proposition_length" not in chunk_options:
                        chunk_options["proposition_min_proposition_length"] = int(
                            c.get("proposition_min_proposition_length")
                        )
                except _CHUNKING_OPTIONS_COERCE_EXCEPTIONS:
                    pass
        except _CHUNKING_OPTIONS_NONCRITICAL_EXCEPTIONS as cfg_err:
            logging.debug(f"Proposition config defaults not loaded: {cfg_err}")

    logging.info("Chunking enabled with options: {}", chunk_options)
    return chunk_options


def resolve_chunking_options_and_plan(
    form_data: Any,
    *,
    media_type: str | None = None,
    source_name: str | None = None,
    extracted_text: str | None = None,
    template_name: str | None = None,
    template_status: str | None = None,
    template_error: str | None = None,
    llm_available: bool = False,
    semantic_available: bool = True,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """
    Resolve effective chunking options and optional Auto Chunking plan metadata.

    Legacy requests that omit ``chunking_mode`` and explicit Manual requests
    keep the existing ``prepare_chunking_options_dict`` behavior. Auto requests
    intentionally ignore stale manual fields and use the deterministic planner.
    """
    if not getattr(form_data, "perform_chunking", False):
        logging.info("Chunking disabled.")
        return None, None

    if getattr(form_data, "chunking_mode", None) != "auto":
        return prepare_chunking_options_dict(form_data), None

    effective_media_type = media_type or getattr(form_data, "media_type", None)
    source_profile = _profile_from_form_source(
        form_data,
        media_type=effective_media_type,
        source_name=source_name,
    )
    text_profile = profile_from_text(extracted_text)
    profile = merge_profiles(source_profile, text_profile)

    decision = plan_auto_chunking(
        perform_chunking=True,
        chunking_mode="auto",
        media_type=effective_media_type,
        goal=getattr(form_data, "auto_chunking_goal", "balanced"),
        profile=profile,
        template_name=template_name or getattr(form_data, "chunking_template_name", None),
        template_status=template_status,
        template_error=template_error,
        requested_llm=bool(getattr(form_data, "auto_chunking_use_llm", False)),
        llm_available=llm_available,
        semantic_available=semantic_available,
    )
    return decision.chunk_options, decision.chunking_plan


def attach_chunking_plan_to_result(
    result: dict[str, Any],
    chunking_plan: dict[str, Any] | None,
) -> None:
    """Attach Auto Chunking plan metadata to an endpoint result in-place."""
    if not chunking_plan:
        return
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        result["metadata"] = metadata
    metadata["chunking_plan"] = chunking_plan


def resolve_chunking_for_result(
    form_data: Any,
    result: dict[str, Any],
    *,
    media_type: str | None = None,
    default_chunk_options: dict[str, Any] | None = None,
    default_chunking_plan: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """
    Resolve final chunking options for a completed process result.

    Auto requests can use extracted content signals here, while legacy/manual
    paths keep the options prepared before processor dispatch.
    """
    if getattr(form_data, "chunking_mode", None) != "auto":
        return default_chunk_options, default_chunking_plan

    extracted_text = result.get("content")
    if not isinstance(extracted_text, str):
        extracted_text = None
    source_name = result.get("input_ref") or result.get("processing_source")
    if source_name is not None:
        source_name = str(source_name)

    chunk_options, chunking_plan = resolve_chunking_options_and_plan(
        form_data,
        media_type=media_type,
        source_name=source_name,
        extracted_text=extracted_text,
    )
    return (
        chunk_options if chunk_options is not None else default_chunk_options,
        chunking_plan if chunking_plan is not None else default_chunking_plan,
    )


def _profile_from_form_source(
    form_data: Any,
    *,
    media_type: str | None,
    source_name: str | None,
):
    source = source_name or _first_form_source_name(form_data)
    filename: str | None = None
    url: str | None = None
    if source:
        source_text = str(source)
        if "://" in source_text:
            url = source_text
        else:
            filename = source_text

    language = getattr(form_data, "chunk_language", None)
    if not language:
        language = getattr(form_data, "transcription_language", None)

    return profile_from_source(
        media_type=media_type,
        filename=filename,
        url=url,
        title=getattr(form_data, "title", None),
        language=language,
    )


def _first_form_source_name(form_data: Any) -> str | None:
    urls = getattr(form_data, "urls", None)
    if isinstance(urls, list) and urls:
        first = urls[0]
        if first:
            return str(first)
    for attr_name in ("original_filename", "filename", "source_name"):
        value = getattr(form_data, attr_name, None)
        if value:
            return str(value)
    return None


def apply_chunking_template_if_any(
    form_data: Any,
    db: Any,
    chunking_options_dict: dict[str, Any] | None,
    *,
    TemplateClassifier: Any | None = None,
    first_url: str | None = None,
    first_filename: str | None = None,
) -> dict[str, Any] | None:
    """
    Apply an explicit or auto-selected chunking template to the provided
    chunking options dictionary.

    This helper encapsulates the template application logic that was
    previously embedded in the `/media/add` orchestration so it can be
    reused by process-* endpoints without duplicating behaviour.
    """
    try:
        if not getattr(form_data, "perform_chunking", False):
            return chunking_options_dict

        opts = chunking_options_dict or {}

        # 1) Apply explicit template by name when provided.
        template_name = getattr(form_data, "chunking_template_name", None)
        if template_name:
            try:
                tpl = db.get_chunking_template(name=template_name)
            except _CHUNKING_OPTIONS_NONCRITICAL_EXCEPTIONS as db_err:
                logging.warning("Failed to load chunking template '%s': %s", template_name, db_err)
                return opts

            if tpl and tpl.get("template_json"):
                import json as _json

                raw_cfg = tpl["template_json"]
                try:
                    cfg = _json.loads(raw_cfg) if isinstance(raw_cfg, str) else raw_cfg
                except _CHUNKING_OPTIONS_COERCE_EXCEPTIONS:
                    cfg = {}
                cfg = cfg or {}
                hier_cfg = (cfg.get("chunking") or {}).get("config", {}) or {}
                hier_tpl = hier_cfg.get("hierarchical_template")
                if isinstance(hier_tpl, dict):
                    opts = opts or {}
                    tpl_method = (cfg.get("chunking") or {}).get("method") or "sentences"
                    # Respect explicit user chunk_method if set, but let the
                    # template override any default method chosen earlier.
                    if not getattr(form_data, "chunk_method", None):
                        opts["method"] = tpl_method

                    # Allow template to provide max_size/overlap so callers do
                    # not need to redundantly pass chunk_size/chunk_overlap.
                    tpl_max_size = hier_cfg.get("max_size")
                    tpl_overlap = hier_cfg.get("overlap")
                    if isinstance(tpl_max_size, int):
                        opts["max_size"] = tpl_max_size
                    if isinstance(tpl_overlap, int):
                        opts["overlap"] = tpl_overlap

                    opts["hierarchical"] = True
                    opts["hierarchical_template"] = hier_tpl
            return opts

        # 2) Respect explicit user hierarchical/method flags (already
        # encoded in chunking_options_dict by prepare_chunking_options_dict).

        # 3) Auto-match a template when requested and the user has not
        # explicitly requested hierarchical chunking.
        if (
            getattr(form_data, "auto_apply_template", False)
            and not getattr(form_data, "hierarchical_chunking", False)
            and TemplateClassifier is not None
        ):
            try:
                candidates = db.list_chunking_templates(
                    include_builtin=True,
                    include_custom=True,
                    tags=None,
                    user_id=None,
                    include_deleted=False,
                )
            except _CHUNKING_OPTIONS_NONCRITICAL_EXCEPTIONS as list_err:
                logging.warning("Failed to list chunking templates for auto-apply: {}", list_err)
                return opts

            best_cfg: dict[str, Any] | None = None
            best_key: tuple[float, int] | None = None

            for t in candidates:
                try:
                    import json as _json

                    cfg = _json.loads(t.get("template_json") or "{}")
                    if not isinstance(cfg, dict):
                        cfg = {}
                except _CHUNKING_OPTIONS_COERCE_EXCEPTIONS:
                    cfg = {}

                try:
                    score = TemplateClassifier.score(  # type: ignore[call-arg]
                        cfg,
                        media_type=getattr(form_data, "media_type", None),
                        title=getattr(form_data, "title", None),
                        url=first_url,
                        filename=first_filename,
                    )
                except _CHUNKING_OPTIONS_NONCRITICAL_EXCEPTIONS:
                    score = 0.0

                if score <= 0:
                    continue

                priority = ((cfg.get("classifier") or {}).get("priority") or 0)  # type: ignore[assignment]
                key = (float(score), int(priority))

                if best_cfg is None or best_key is None or key > best_key:
                    best_cfg, best_key = cfg, key

            if best_cfg:
                hier_cfg = (best_cfg.get("chunking") or {}).get("config") or {}
                tpl = hier_cfg.get("hierarchical_template")
                if isinstance(tpl, dict):
                    opts = opts or {}
                    tpl_method = (best_cfg.get("chunking") or {}).get("method", "sentences")
                    if not getattr(form_data, "chunk_method", None):
                        opts["method"] = tpl_method

                    tpl_max_size = hier_cfg.get("max_size")
                    tpl_overlap = hier_cfg.get("overlap")
                    if isinstance(tpl_max_size, int):
                        opts["max_size"] = tpl_max_size
                    if isinstance(tpl_overlap, int):
                        opts["overlap"] = tpl_overlap

                    opts["hierarchical"] = True
                    opts["hierarchical_template"] = tpl

        return opts
    except _CHUNKING_OPTIONS_NONCRITICAL_EXCEPTIONS as auto_err:  # Defensive: never break callers
        logging.warning("Auto-apply chunking template helper failed: {}", auto_err)
        return chunking_options_dict


def prepare_common_options(
    form_data: Any,
    chunk_options: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Prepare the dictionary of common processing options for ingestion.

    Extracted from the prior endpoint-local helper to share behavior
    between `/media/add` code paths.
    """
    return {
        "keywords": getattr(form_data, "keywords", []),
        "custom_prompt": getattr(form_data, "custom_prompt", None),
        "system_prompt": getattr(form_data, "system_prompt", None),
        "overwrite_existing": bool(getattr(form_data, "overwrite_existing", False)),
        "perform_analysis": bool(getattr(form_data, "perform_analysis", False)),
        "chunk_options": chunk_options,
        "api_name": getattr(form_data, "api_name", None),
        "api_provider": getattr(form_data, "api_provider", None),
        "model_name": getattr(form_data, "model_name", None),
        "store_in_db": True,
        "summarize_recursively": bool(
            getattr(form_data, "summarize_recursively", False)
        ),
        "author": getattr(form_data, "author", None),
    }
