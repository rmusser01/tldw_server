from __future__ import annotations

import json
from typing import Any

from fastapi import Form, status
from loguru import logger
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.API_Deps.form_coercion import (
    chunking_contract_kwargs,
)
from tldw_Server_API.app.api.v1.schemas.media_request_models import (
    TRANSCRIPTION_MODEL_ENUM,
    ProcessAudiosForm,
    ProcessDocumentsForm,
    ProcessEbooksForm,
    ProcessEmailsForm,
    ProcessPDFsForm,
    ProcessVideosForm,
)
from tldw_Server_API.app.core.exceptions import APIValidationError
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter import (
    resolve_default_transcription_model,
)

try:
    HTTP_422_UNPROCESSABLE = status.HTTP_422_UNPROCESSABLE_CONTENT
except AttributeError:  # Starlette < 0.27
    HTTP_422_UNPROCESSABLE = status.HTTP_422_UNPROCESSABLE_ENTITY


def _coerce_urls(urls: list[str] | None) -> list[str] | None:
    """
    Normalize urls form input into a list of strings.

    Allows clients to send:
      - a single string
      - a JSON-encoded list string
      - a list of strings
    """
    if urls is None:
        return None
    if isinstance(urls, list):
        if len(urls) == 1 and isinstance(urls[0], str):
            first = urls[0]
            stripped = first.strip()
            if stripped.startswith("[") or stripped.startswith('"'):
                try:
                    parsed = json.loads(first)
                    if isinstance(parsed, list):
                        return [str(u) for u in parsed]
                    return [str(parsed)]
                except Exception:
                    return [first]
        return [str(u) for u in urls]
    if isinstance(urls, str):
        stripped = urls.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [str(u) for u in parsed]
            return [str(parsed)]
        except Exception:
            return [urls]
    return [str(urls)]


def _resolve_transcription_model_or_default(
    transcription_model: str | None,
    *,
    fallback_whisper_model: str,
    context: str,
) -> str:
    """Return a validated transcription model or the configured default."""
    model = (transcription_model or "").strip()
    default_model = resolve_default_transcription_model(fallback_whisper_model)
    if model and model not in TRANSCRIPTION_MODEL_ENUM:
        logger.warning(
            "Invalid transcription_model '{}' for {}; defaulting to {}",
            model,
            context,
            default_model,
        )
        return default_model
    return model or default_model


def _raise_422(exc: ValidationError) -> None:
    serializable_errors = []
    for error in exc.errors():
        err = error.copy()
        ctx = err.get("ctx")
        loc = list(err.get("loc") or [])
        if loc and loc[0] != "body":
            loc = ["body"] + loc
        elif not loc:
            loc = ["body"]
        err["loc"] = loc
        if isinstance(ctx, dict):
            err["ctx"] = {k: (str(v) if isinstance(v, Exception) else v) for k, v in ctx.items()}
        serializable_errors.append(err)
    raise APIValidationError(
        detail=serializable_errors,
        status_code=HTTP_422_UNPROCESSABLE,
    ) from exc


async def get_process_documents_form(
    urls: list[str] | None = Form(None),
    title: str | None = Form(None),
    titles: str | None = Form(None),
    author: str | None = Form(None),
    keywords: str | None = Form(None),
    keywords_str: str | None = Form(None),
    custom_prompt: str | None = Form(None),
    system_prompt: str | None = Form(None),
    keep_original_file: bool = Form(False),
    perform_analysis: bool = Form(True),
    perform_chunking: bool = Form(True),
    summarize_recursively: bool = Form(False),
    chunk_method: str | None = Form(None),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(200),
    chunking_mode: str | None = Form(None),
    auto_chunking_goal: str = Form("balanced"),
    auto_chunking_use_llm: bool = Form(False),
    auto_apply_template: bool = Form(False),
    chunking_template_name: str | None = Form(None),
    hierarchical_chunking: bool = Form(False),
    hierarchical_template: str | dict[str, Any] | None = Form(None),
    api_provider: str | None = Form(None),
    model_name: str | None = Form(None),
    api_name: str | None = Form(None),
    use_cookies: bool = Form(False),
    cookies: str | None = Form(None),
) -> ProcessDocumentsForm:
    """
    Dependency that parses multipart/form-data into a ProcessDocumentsForm.

    Used by /media/process-documents (no DB persistence).
    """
    try:
        urls_norm = _coerce_urls(urls)
        keywords_value = keywords if keywords is not None else (keywords_str if keywords_str is not None else "")
        title_val = title or titles
        return ProcessDocumentsForm(
            urls=urls_norm,
            title=title_val,
            author=author,
            keywords=keywords_value,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            keep_original_file=keep_original_file,
            perform_analysis=perform_analysis,
            perform_chunking=perform_chunking,
            summarize_recursively=summarize_recursively,
            chunk_method=chunk_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **chunking_contract_kwargs(
                chunking_mode=chunking_mode,
                auto_chunking_goal=auto_chunking_goal,
                auto_chunking_use_llm=auto_chunking_use_llm,
                auto_apply_template=auto_apply_template,
                chunking_template_name=chunking_template_name,
                hierarchical_chunking=hierarchical_chunking,
                hierarchical_template=hierarchical_template,
            ),
            api_provider=api_provider,
            model_name=model_name,
            api_name=api_name,
            use_cookies=use_cookies,
            cookies=cookies,
        )
    except ValidationError as exc:
        _raise_422(exc)


async def get_process_videos_form(
    urls: list[str] | None = Form(None),
    title: str | None = Form(None),
    titles: str | None = Form(None),
    author: str | None = Form(None),
    keywords: str = Form(""),
    custom_prompt: str | None = Form(None),
    system_prompt: str | None = Form(None),
    perform_analysis: bool = Form(True),
    perform_chunking: bool = Form(True),
    summarize_recursively: bool = Form(False),
    transcription_model: str | None = Form(
        None,
        description="Transcription model for video audio tracks (defaults to config when omitted)",
        json_schema_extra={"enum": TRANSCRIPTION_MODEL_ENUM},
    ),
    transcription_language: str = Form("en"),
    diarize: bool = Form(False),
    timestamp_option: bool = Form(True),
    vad_use: bool = Form(False),
    perform_confabulation_check_of_analysis: bool = Form(False),
    start_time: str | None = Form(None),
    end_time: str | None = Form(None),
    api_provider: str | None = Form(None),
    model_name: str | None = Form(None),
    api_name: str | None = Form(None),
    use_cookies: bool = Form(False),
    cookies: str | None = Form(None),
    chunk_method: str | None = Form(None),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(200),
    chunking_mode: str | None = Form(None),
    auto_chunking_goal: str = Form("balanced"),
    auto_chunking_use_llm: bool = Form(False),
    auto_apply_template: bool = Form(False),
    chunking_template_name: str | None = Form(None),
    hierarchical_chunking: bool = Form(False),
    hierarchical_template: str | dict[str, Any] | None = Form(None),
) -> ProcessVideosForm:
    """
    Dependency that parses multipart/form-data into a ProcessVideosForm.

    Used by /media/process-videos (no DB persistence).
    """
    transcription_model = _resolve_transcription_model_or_default(
        transcription_model,
        fallback_whisper_model="deepdml/faster-distil-whisper-large-v3.5",
        context="process-videos",
    )
    try:
        urls_norm = _coerce_urls(urls)
        title_val = title or titles
        return ProcessVideosForm(
            urls=urls_norm,
            title=title_val,
            author=author,
            keywords=keywords,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            perform_analysis=perform_analysis,
            perform_chunking=perform_chunking,
            summarize_recursively=summarize_recursively,
            transcription_model=transcription_model,
            transcription_language=transcription_language,
            diarize=diarize,
            timestamp_option=timestamp_option,
            vad_use=vad_use,
            perform_confabulation_check_of_analysis=perform_confabulation_check_of_analysis,
            start_time=start_time,
            end_time=end_time,
            api_provider=api_provider,
            model_name=model_name,
            api_name=api_name,
            use_cookies=use_cookies,
            cookies=cookies,
            chunk_method=chunk_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **chunking_contract_kwargs(
                chunking_mode=chunking_mode,
                auto_chunking_goal=auto_chunking_goal,
                auto_chunking_use_llm=auto_chunking_use_llm,
                auto_apply_template=auto_apply_template,
                chunking_template_name=chunking_template_name,
                hierarchical_chunking=hierarchical_chunking,
                hierarchical_template=hierarchical_template,
            ),
        )
    except ValidationError as exc:
        _raise_422(exc)


async def get_process_audios_form(
    urls: list[str] | None = Form(None),
    title: str | None = Form(None),
    titles: str | None = Form(None),
    author: str | None = Form(None),
    keywords: str = Form(""),
    custom_prompt: str | None = Form(None),
    system_prompt: str | None = Form(None),
    perform_analysis: bool = Form(True),
    perform_chunking: bool = Form(True),
    summarize_recursively: bool = Form(False),
    transcription_model: str | None = Form(
        None,
        description="Transcription model for audio inputs (defaults to config when omitted)",
        json_schema_extra={"enum": TRANSCRIPTION_MODEL_ENUM},
    ),
    transcription_language: str = Form("en"),
    diarize: bool = Form(False),
    timestamp_option: bool = Form(True),
    vad_use: bool = Form(False),
    perform_confabulation_check_of_analysis: bool = Form(False),
    start_time: str | None = Form(None),
    end_time: str | None = Form(None),
    api_provider: str | None = Form(None),
    model_name: str | None = Form(None),
    api_name: str | None = Form(None),
    use_cookies: bool = Form(False),
    cookies: str | None = Form(None),
    chunk_method: str | None = Form(None),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(200),
    chunking_mode: str | None = Form(None),
    auto_chunking_goal: str = Form("balanced"),
    auto_chunking_use_llm: bool = Form(False),
    auto_apply_template: bool = Form(False),
    chunking_template_name: str | None = Form(None),
    hierarchical_chunking: bool = Form(False),
    hierarchical_template: str | dict[str, Any] | None = Form(None),
) -> ProcessAudiosForm:
    """
    Dependency that parses multipart/form-data into a ProcessAudiosForm.

    Used by /media/process-audios (no DB persistence).
    """
    transcription_model = _resolve_transcription_model_or_default(
        transcription_model,
        fallback_whisper_model="deepdml/faster-distil-whisper-large-v3.5",
        context="process-audios",
    )
    try:
        urls_norm = _coerce_urls(urls)
        title_val = title or titles
        return ProcessAudiosForm(
            urls=urls_norm,
            title=title_val,
            author=author,
            keywords=keywords,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            perform_analysis=perform_analysis,
            perform_chunking=perform_chunking,
            summarize_recursively=summarize_recursively,
            transcription_model=transcription_model,
            transcription_language=transcription_language,
            diarize=diarize,
            timestamp_option=timestamp_option,
            vad_use=vad_use,
            perform_confabulation_check_of_analysis=perform_confabulation_check_of_analysis,
            start_time=start_time,
            end_time=end_time,
            api_provider=api_provider,
            model_name=model_name,
            api_name=api_name,
            use_cookies=use_cookies,
            cookies=cookies,
            chunk_method=chunk_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **chunking_contract_kwargs(
                chunking_mode=chunking_mode,
                auto_chunking_goal=auto_chunking_goal,
                auto_chunking_use_llm=auto_chunking_use_llm,
                auto_apply_template=auto_apply_template,
                chunking_template_name=chunking_template_name,
                hierarchical_chunking=hierarchical_chunking,
                hierarchical_template=hierarchical_template,
            ),
        )
    except ValidationError as exc:
        _raise_422(exc)


async def get_process_pdfs_form(
    urls: list[str] | None = Form(None),
    title: str | None = Form(None),
    author: str | None = Form(None),
    keywords: str = Form(""),
    custom_prompt: str | None = Form(None),
    system_prompt: str | None = Form(None),
    perform_analysis: bool = Form(True),
    perform_chunking: bool = Form(True),
    summarize_recursively: bool = Form(False),
    chunk_method: str | None = Form(None),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(200),
    chunking_mode: str | None = Form(None),
    auto_chunking_goal: str = Form("balanced"),
    auto_chunking_use_llm: bool = Form(False),
    auto_apply_template: bool = Form(False),
    chunking_template_name: str | None = Form(None),
    hierarchical_chunking: bool = Form(False),
    hierarchical_template: str | dict[str, Any] | None = Form(None),
    pdf_parsing_engine: str = Form("pymupdf4llm"),
    enable_ocr: bool = Form(False),
    ocr_backend: str | None = Form(None),
    ocr_lang: str | None = Form("eng"),
    ocr_dpi: int = Form(300),
    ocr_mode: str | None = Form("fallback"),
    ocr_min_page_text_chars: int = Form(40),
    ocr_output_format: str | None = Form(None),
    ocr_prompt_preset: str | None = Form(None),
    use_adaptive_chunking: bool = Form(False),
    use_multi_level_chunking: bool = Form(False),
    chunk_language: str | None = Form(None),
    proposition_engine: str | None = Form(None),
    proposition_aggressiveness: int | None = Form(None),
    proposition_min_proposition_length: int | None = Form(None),
    proposition_prompt_profile: str | None = Form(None),
) -> ProcessPDFsForm:
    """
    Dependency that parses multipart/form-data into a ProcessPDFsForm.

    Used by /media/process-pdfs (no DB persistence).
    """
    try:
        urls_norm = _coerce_urls(urls)
        return ProcessPDFsForm(
            urls=urls_norm,
            title=title,
            author=author,
            keywords=keywords,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            perform_analysis=perform_analysis,
            perform_chunking=perform_chunking,
            summarize_recursively=summarize_recursively,
            chunk_method=chunk_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **chunking_contract_kwargs(
                chunking_mode=chunking_mode,
                auto_chunking_goal=auto_chunking_goal,
                auto_chunking_use_llm=auto_chunking_use_llm,
                auto_apply_template=auto_apply_template,
                chunking_template_name=chunking_template_name,
                hierarchical_chunking=hierarchical_chunking,
                hierarchical_template=hierarchical_template,
            ),
            pdf_parsing_engine=pdf_parsing_engine,
            enable_ocr=enable_ocr,
            ocr_backend=ocr_backend,
            ocr_lang=ocr_lang,
            ocr_dpi=ocr_dpi,
            ocr_mode=ocr_mode,
            ocr_min_page_text_chars=ocr_min_page_text_chars,
            ocr_output_format=ocr_output_format,
            ocr_prompt_preset=ocr_prompt_preset,
            use_adaptive_chunking=use_adaptive_chunking,
            use_multi_level_chunking=use_multi_level_chunking,
            chunk_language=chunk_language,
            proposition_engine=proposition_engine,
            proposition_aggressiveness=proposition_aggressiveness,
            proposition_min_proposition_length=proposition_min_proposition_length,
            proposition_prompt_profile=proposition_prompt_profile,
        )
    except ValidationError as exc:
        _raise_422(exc)


async def get_process_ebooks_form(
    urls: list[str] | None = Form(None),
    title: str | None = Form(None),
    author: str | None = Form(None),
    keywords: str | None = Form(None),
    keywords_str: str | None = Form(None),
    custom_prompt: str | None = Form(None),
    system_prompt: str | None = Form(None),
    perform_analysis: bool = Form(True),
    perform_chunking: bool = Form(True),
    summarize_recursively: bool = Form(False),
    chunk_method: str | None = Form(None),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(200),
    chunk_language: str | None = Form(None),
    custom_chapter_pattern: str | None = Form(None),
    chunking_mode: str | None = Form(None),
    auto_chunking_goal: str = Form("balanced"),
    auto_chunking_use_llm: bool = Form(False),
    auto_apply_template: bool = Form(False),
    chunking_template_name: str | None = Form(None),
    hierarchical_chunking: bool = Form(False),
    hierarchical_template: str | dict[str, Any] | None = Form(None),
    extraction_method: str = Form("filtered"),
    api_name: str | None = Form(None),
    api_key: str | None = Form(None),
) -> ProcessEbooksForm:
    """
    Dependency that parses multipart/form-data into a ProcessEbooksForm.

    Used by /media/process-ebooks (no DB persistence).
    """
    try:
        urls_norm = _coerce_urls(urls)
        keywords_value = keywords if keywords is not None else (keywords_str if keywords_str is not None else "")
        return ProcessEbooksForm(
            urls=urls_norm,
            title=title,
            author=author,
            keywords=keywords_value,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            perform_analysis=perform_analysis,
            perform_chunking=perform_chunking,
            summarize_recursively=summarize_recursively,
            chunk_method=chunk_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_language=chunk_language,
            custom_chapter_pattern=custom_chapter_pattern,
            **chunking_contract_kwargs(
                chunking_mode=chunking_mode,
                auto_chunking_goal=auto_chunking_goal,
                auto_chunking_use_llm=auto_chunking_use_llm,
                auto_apply_template=auto_apply_template,
                chunking_template_name=chunking_template_name,
                hierarchical_chunking=hierarchical_chunking,
                hierarchical_template=hierarchical_template,
            ),
            extraction_method=extraction_method,
            api_name=api_name,
            api_key=api_key,
        )
    except ValidationError as exc:
        _raise_422(exc)


async def get_process_emails_form(
    urls: list[str] | None = Form(None),
    title: str | None = Form(None),
    author: str | None = Form(None),
    keywords: str = Form(""),
    custom_prompt: str | None = Form(None),
    system_prompt: str | None = Form(None),
    overwrite_existing: bool = Form(False),
    perform_analysis: bool = Form(False),
    perform_claims_extraction: bool | None = Form(None),
    claims_extractor_mode: str | None = Form(None),
    claims_max_per_chunk: int | None = Form(None),
    perform_chunking: bool = Form(True),
    chunk_method: str | None = Form("sentences"),
    chunk_language: str | None = Form(None),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    custom_chapter_pattern: str | None = Form(None),
    use_adaptive_chunking: bool = Form(False),
    use_multi_level_chunking: bool = Form(False),
    chunking_mode: str | None = Form(None),
    auto_chunking_goal: str = Form("balanced"),
    auto_chunking_use_llm: bool = Form(False),
    auto_apply_template: bool = Form(False),
    chunking_template_name: str | None = Form(None),
    hierarchical_chunking: bool = Form(False),
    hierarchical_template: str | dict[str, Any] | None = Form(None),
    accept_archives: bool = Form(False),
    accept_mbox: bool = Form(False),
    accept_pst: bool = Form(False),
    ingest_attachments: bool = Form(False),
    max_depth: int = Form(2),
) -> ProcessEmailsForm:
    """
    Dependency that parses multipart/form-data into a ProcessEmailsForm.

    Used by /media/process-emails (no DB persistence).
    """
    try:
        urls_norm = _coerce_urls(urls)
        return ProcessEmailsForm(
            urls=urls_norm,
            title=title,
            author=author,
            keywords=keywords,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            overwrite_existing=overwrite_existing,
            perform_analysis=perform_analysis,
            perform_claims_extraction=perform_claims_extraction,
            claims_extractor_mode=claims_extractor_mode,
            claims_max_per_chunk=claims_max_per_chunk,
            perform_chunking=perform_chunking,
            chunk_method=chunk_method,
            chunk_language=chunk_language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            custom_chapter_pattern=custom_chapter_pattern,
            use_adaptive_chunking=use_adaptive_chunking,
            use_multi_level_chunking=use_multi_level_chunking,
            **chunking_contract_kwargs(
                chunking_mode=chunking_mode,
                auto_chunking_goal=auto_chunking_goal,
                auto_chunking_use_llm=auto_chunking_use_llm,
                auto_apply_template=auto_apply_template,
                chunking_template_name=chunking_template_name,
                hierarchical_chunking=hierarchical_chunking,
                hierarchical_template=hierarchical_template,
            ),
            accept_archives=accept_archives,
            accept_mbox=accept_mbox,
            accept_pst=accept_pst,
            ingest_attachments=ingest_attachments,
            max_depth=max_depth,
        )
    except ValidationError as exc:
        _raise_422(exc)


__all__ = [
    "get_process_documents_form",
    "get_process_videos_form",
    "get_process_audios_form",
    "get_process_pdfs_form",
    "get_process_ebooks_form",
    "get_process_emails_form",
]
