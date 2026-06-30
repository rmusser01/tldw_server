# Translation Endpoint
# Provides text translation using configured LLM providers
#
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from tldw_Server_API.app.api.v1.schemas.translate_schemas import (
    TranslateRequest,
    TranslateResponse,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User
from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze

router = APIRouter(tags=["Translation"])

# Translation prompt template
TRANSLATION_PROMPT = """Translate the following text to {target_language}.
Preserve the original formatting, meaning, and tone.
Only output the translation, no explanations, notes, or additional text.

Text to translate:
{text}"""

# System message for translation
TRANSLATION_SYSTEM_MESSAGE = """You are an expert translator. Your task is to provide accurate,
natural-sounding translations that preserve the original meaning, tone, and formatting.
Do not add explanations or notes - only provide the translation."""


def _get_default_provider() -> str:
    """Get the default LLM provider from config."""
    try:
        config = load_and_log_configs()
        # Check for a configured default provider
        if config:
            # Check common provider sections
            for provider in ["openai", "anthropic", "openrouter", "groq", "deepseek"]:
                section = config.get(provider, {})
                if section.get("api_key"):
                    return provider
        return "openai"  # Fallback default
    except Exception:
        return "openai"


@router.post(
    "/translate",
    response_model=TranslateResponse,
    status_code=status.HTTP_200_OK,
    summary="Translate text to target language",
    responses={
        200: {"description": "Translation successful"},
        400: {"description": "Invalid request (text too long, etc.)"},
        500: {"description": "Translation failed (LLM error)"},
        503: {"description": "Translation service unavailable"},
    },
)
async def translate_text(
    request: TranslateRequest,
    current_user: User = Depends(get_request_user),
) -> TranslateResponse:
    """
    Translate text to a target language using an LLM.

    ## Features

    - Supports translation to/from many languages
    - Uses configured LLM provider (OpenAI, Anthropic, etc.)
    - Preserves original formatting and tone
    - Auto-detects source language if not specified

    ## Supported Languages

    Any language supported by the underlying LLM model.
    Common options: English, Spanish, French, German, Chinese,
    Japanese, Korean, Portuguese, Russian, Arabic, and more.

    ## Rate Limits

    Subject to your LLM provider's rate limits and token quotas.
    """
    logger.debug(
        "Translation request: {} chars to {}, user={}",
        len(request.text),
        request.target_language,
        getattr(current_user, "id", "?"),
    )

    # Build the translation prompt
    prompt = TRANSLATION_PROMPT.format(
        target_language=request.target_language,
        text=request.text,
    )

    # Determine provider
    provider = request.provider or _get_default_provider()
    model = request.model  # None means use provider default

    try:
        # Use the existing analyze function for LLM call
        result = analyze(
            api_name=provider,
            input_data=prompt,
            custom_prompt_arg=None,
            api_key=None,  # Uses configured key
            system_message=TRANSLATION_SYSTEM_MESSAGE,
            temp=0.3,  # Low temperature for consistent translation
            streaming=False,
            model_override=model,
        )

        # Check for error response
        if isinstance(result, str) and result.startswith("Error:"):
            logger.error("Translation failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Translation failed",
            )

        # Clean up the result (remove any leading/trailing whitespace)
        translated_text = result.strip() if isinstance(result, str) else str(result).strip()

        logger.debug(
            "Translation successful: {} chars -> {} chars",
            len(request.text),
            len(translated_text),
        )

        return TranslateResponse(
            translated_text=translated_text,
            detected_source_language=request.source_language,
            target_language=request.target_language,
            model_used=model or f"{provider}_default",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected translation error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Translation failed",
        ) from e


__all__ = ["router"]
