"""Natural language processing adapters.

This module includes adapters for NLP operations:
- keyword_extract: Extract keywords
- sentiment_analyze: Analyze sentiment
- language_detect: Detect language
- topic_model: Topic modeling
- entity_extract: Extract named entities
- token_count: Count tokens
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Workflows.adapters._common import extract_openai_content
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.text._config import (
    EntityExtractConfig,
    KeywordExtractConfig,
    LanguageDetectConfig,
    SentimentAnalyzeConfig,
    TokenCountConfig,
    TopicModelConfig,
)


@registry.register(
    "keyword_extract",
    category="text",
    description="Extract keywords",
    parallelizable=True,
    tags=["text", "nlp"],
    config_model=KeywordExtractConfig,
)
async def run_keyword_extract_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Extract keywords from text content.

    Config:
      - text: str (templated) - Text to extract keywords from
      - method: Literal["tfidf", "rake", "yake", "llm"] = "rake"
      - max_keywords: int = 10 - Maximum keywords to return
      - provider: str (optional) - LLM provider for llm method
      - model: str (optional) - Model for llm method
    Output:
      - {"keywords": [str], "scores": [float], "count": int}
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    text = config.get("text") or ""
    if isinstance(text, str):
        text = _tmpl(text, context) or text

    if not text:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            text = prev.get("text") or prev.get("content") or ""

    if not text:
        return {"keywords": [], "scored_keywords": [], "error": "missing_text"}

    method = str(config.get("method", "llm")).lower()
    max_keywords = int(config.get("max_keywords", 10))

    if method == "rake":
        try:
            from rake_nltk import Rake
            r = Rake()
            r.extract_keywords_from_text(text)
            scored = r.get_ranked_phrases_with_scores()[:max_keywords]
            keywords = [kw for _, kw in scored]
            scored_keywords = [{"keyword": kw, "score": score} for score, kw in scored]
            return {"keywords": keywords, "scored_keywords": scored_keywords, "method": "rake"}
        except ImportError:
            method = "llm"  # Fallback

    if method == "yake":
        try:
            import yake
            kw_extractor = yake.KeywordExtractor(top=max_keywords)
            keywords_scored = kw_extractor.extract_keywords(text)
            keywords = [kw for kw, _ in keywords_scored]
            scored_keywords = [{"keyword": kw, "score": 1 - score} for kw, score in keywords_scored]
            return {"keywords": keywords, "scored_keywords": scored_keywords, "method": "yake"}
        except ImportError:
            method = "llm"  # Fallback

    # LLM method
    try:
        from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

        prompt = f"""Extract the {max_keywords} most important keywords from this text.
Return only the keywords, one per line, no numbering.

Text:
{text[:4000]}"""

        messages = [{"role": "user", "content": prompt}]
        response = await perform_chat_api_call_async(
            messages=messages,
            api_provider=config.get("provider"),
            model=config.get("model"),
            system_message="Extract keywords from text. Return only keywords, one per line.",
            max_tokens=300,
            temperature=0.3,
        )

        result_text = extract_openai_content(response) or ""
        keywords = [line.strip() for line in result_text.strip().split("\n") if line.strip()][:max_keywords]
        scored_keywords = [{"keyword": kw, "score": 1.0 - (i * 0.05)} for i, kw in enumerate(keywords)]

        return {"keywords": keywords, "scored_keywords": scored_keywords, "method": "llm"}

    except Exception:
        logger.exception("Keyword extract error")
        return {"keywords": [], "scored_keywords": [], "error": "keyword_extract_error"}


@registry.register(
    "sentiment_analyze",
    category="text",
    description="Analyze sentiment",
    parallelizable=True,
    tags=["text", "nlp"],
    config_model=SentimentAnalyzeConfig,
)
async def run_sentiment_analyze_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Analyze sentiment of text content.

    Config:
      - text: str (templated) - Text to analyze
      - method: Literal["vader", "textblob", "llm"] = "vader"
      - provider: str (optional) - LLM provider for llm method
      - model: str (optional) - Model for llm method
    Output:
      - {"sentiment": str, "score": float, "confidence": float}
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    text = config.get("text") or ""
    if isinstance(text, str):
        text = _tmpl(text, context) or text

    if not text:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            text = prev.get("text") or prev.get("content") or ""

    if not text:
        return {"sentiment": "neutral", "score": 0.0, "confidence": 0.0, "error": "missing_text"}

    try:
        from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

        prompt = f"""Analyze the sentiment of this text. Respond with JSON only:
{{"sentiment": "positive|negative|neutral", "score": <-1 to 1>, "confidence": <0 to 1>}}

Text: {text[:3000]}"""

        messages = [{"role": "user", "content": prompt}]
        response = await perform_chat_api_call_async(
            messages=messages,
            api_provider=config.get("provider"),
            model=config.get("model"),
            system_message="Analyze sentiment. Return JSON only.",
            max_tokens=100,
            temperature=0.1,
        )

        result_text = extract_openai_content(response) or ""
        try:
            result = json.loads(result_text)
            return {
                "sentiment": result.get("sentiment", "neutral"),
                "score": float(result.get("score", 0)),
                "confidence": float(result.get("confidence", 0.5)),
            }
        except json.JSONDecodeError:
            # Parse from text
            text_lower = result_text.lower()
            if "positive" in text_lower:
                return {"sentiment": "positive", "score": 0.7, "confidence": 0.6}
            elif "negative" in text_lower:
                return {"sentiment": "negative", "score": -0.7, "confidence": 0.6}
            return {"sentiment": "neutral", "score": 0.0, "confidence": 0.5}

    except Exception:
        logger.exception("Sentiment analyze error")
        return {"sentiment": "neutral", "score": 0.0, "confidence": 0.0, "error": "sentiment_analyze_error"}


@registry.register(
    "language_detect",
    category="text",
    description="Detect language",
    parallelizable=True,
    tags=["text", "nlp"],
    config_model=LanguageDetectConfig,
)
async def run_language_detect_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Detect the language of text content.

    Config:
      - text: str (templated) - Text to analyze
      - method: Literal["langdetect", "fasttext", "llm"] = "langdetect"
      - return_all: bool = False - Return all detected languages
    Output:
      - {"language": str, "code": str, "confidence": float}
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    text = config.get("text") or ""
    if isinstance(text, str):
        text = _tmpl(text, context) or text

    if not text:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            text = prev.get("text") or prev.get("content") or ""

    if not text:
        return {"language": "unknown", "language_name": "Unknown", "confidence": 0.0, "error": "missing_text"}

    try:
        from langdetect import detect, detect_langs
        lang = detect(text[:5000])
        probs = detect_langs(text[:5000])
        confidence = probs[0].prob if probs else 0.5

        lang_names = {
            "en": "English", "es": "Spanish", "fr": "French", "de": "German",
            "it": "Italian", "pt": "Portuguese", "ru": "Russian", "zh-cn": "Chinese (Simplified)",
            "zh-tw": "Chinese (Traditional)", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
            "hi": "Hindi", "nl": "Dutch", "pl": "Polish", "tr": "Turkish",
        }

        return {
            "language": lang,
            "language_name": lang_names.get(lang, lang.upper()),
            "confidence": confidence,
        }
    except ImportError:
        return {"language": "unknown", "language_name": "Unknown", "confidence": 0.0, "error": "langdetect_not_installed"}
    except Exception:
        logger.exception("Language detect error")
        return {"language": "unknown", "language_name": "Unknown", "confidence": 0.0, "error": "language_detect_error"}


@registry.register(
    "topic_model",
    category="text",
    description="Topic modeling",
    parallelizable=True,
    tags=["text", "nlp"],
    config_model=TopicModelConfig,
)
async def run_topic_model_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Extract topics from text content.

    Config:
      - text: str (templated) - Text to analyze
      - method: Literal["lda", "nmf", "llm"] = "llm"
      - num_topics: int = 5 - Number of topics to extract
      - provider: str (optional) - LLM provider for llm method
      - model: str (optional) - Model for llm method
    Output:
      - {"topics": [{"name": str, "keywords": [str]}], "count": int}
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    text = config.get("text") or ""
    if isinstance(text, str):
        text = _tmpl(text, context) or text

    if not text:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            text = prev.get("text") or prev.get("content") or ""

    if not text:
        return {"topics": [], "error": "missing_text"}

    num_topics = int(config.get("num_topics", 5))

    try:
        from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

        prompt = f"""Identify the {num_topics} main topics in this text.
For each topic, provide a label and 3-5 keywords.
Return as JSON array: [{{"label": "Topic Name", "keywords": ["kw1", "kw2"]}}]

Text:
{text[:5000]}"""

        messages = [{"role": "user", "content": prompt}]
        response = await perform_chat_api_call_async(
            messages=messages,
            api_provider=config.get("provider"),
            model=config.get("model"),
            system_message="Extract topics from text. Return JSON array only.",
            max_tokens=800,
            temperature=0.3,
        )

        result_text = extract_openai_content(response) or ""
        # Try to extract JSON from response
        try:
            # Find JSON array in response
            start = result_text.find("[")
            end = result_text.rfind("]") + 1
            if start >= 0 and end > start:
                topics = json.loads(result_text[start:end])
                return {"topics": topics[:num_topics]}
        except json.JSONDecodeError:
            pass

        # Fallback: parse as text
        topics = [{"label": line.strip(), "keywords": []} for line in result_text.split("\n") if line.strip()]
        return {"topics": topics[:num_topics]}

    except Exception:
        logger.exception("Topic model error")
        return {"topics": [], "error": "topic_model_error"}


@registry.register(
    "entity_extract",
    category="text",
    description="Extract named entities",
    parallelizable=True,
    tags=["text", "nlp"],
    config_model=EntityExtractConfig,
)
async def run_entity_extract_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Extract named entities from text content.

    Config:
      - text: str (templated) - Text to analyze
      - method: Literal["spacy", "flair", "llm"] = "spacy"
      - entity_types: list[str] (optional) - Entity types to extract
      - provider: str (optional) - LLM provider for llm method
      - model: str (optional) - Model for llm method
    Output:
      - {"entities": [{"text": str, "type": str, "start": int, "end": int}], "count": int}
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    text = config.get("text") or ""
    if isinstance(text, str):
        text = _tmpl(text, context) or text
    text = str(text).strip()

    if not text:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            text = prev.get("text") or prev.get("content") or prev.get("transcript") or ""
        text = str(text).strip()

    if not text:
        return {"error": "missing_text", "entities": {}, "total_count": 0}

    entity_types = config.get("entity_types", ["all"])
    if isinstance(entity_types, str):
        entity_types = [entity_types]
    if "all" in entity_types:
        entity_types = ["person", "place", "organization", "date", "event"]

    provider = config.get("provider")
    model = config.get("model")
    include_context = bool(config.get("include_context", False))

    types_str = ", ".join(entity_types)
    context_instruction = "Include a brief context snippet for each entity." if include_context else ""

    system_prompt = f"""Extract named entities from the text. Focus on: {types_str}.
{context_instruction}

Return a JSON object with entity types as keys and arrays of entities as values.
Each entity should have: "name", "type", and optionally "context".

Example:
{{"person": [{{"name": "John Smith", "type": "person"}}], "place": [{{"name": "New York", "type": "place"}}]}}"""

    user_prompt = f"Extract entities from:\n\n{text[:8000]}"

    try:
        from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

        messages = [{"role": "user", "content": user_prompt}]
        response = await perform_chat_api_call_async(
            messages=messages,
            api_provider=provider,
            model=model,
            system_message=system_prompt,
            max_tokens=2000,
            temperature=0.3,
        )

        response_text = extract_openai_content(response) or "{}"

        # Parse JSON from response
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            entities = json.loads(json_match.group()) if json_match else {}
        except json.JSONDecodeError:
            entities = {}

        # Count total entities
        total_count = sum(len(v) if isinstance(v, list) else 0 for v in entities.values())

        return {
            "entities": entities,
            "total_count": total_count,
            "text_length": len(text),
        }

    except Exception:
        logger.exception("Entity extract adapter error")
        return {"error": "entity_extract_error", "entities": {}, "total_count": 0}


@registry.register(
    "token_count",
    category="text",
    description="Count tokens",
    parallelizable=True,
    tags=["text", "utility"],
    config_model=TokenCountConfig,
)
async def run_token_count_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Count tokens in text using various tokenizers.

    Config:
      - text: str (templated) - Text to count tokens in
      - tokenizer: str = "cl100k_base" - Tokenizer to use
      - model: str (optional) - Model to use for tokenization
    Output:
      - {"tokens": int, "characters": int, "words": int}
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    text = config.get("text") or ""
    if isinstance(text, str):
        text = _tmpl(text, context) or text

    if not text:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            text = prev.get("text") or prev.get("content") or ""

    model = config.get("model", "gpt-4")
    char_count = len(text)
    word_count = len(text.split())

    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoding.encode(text))
        return {"token_count": token_count, "char_count": char_count, "word_count": word_count, "model": model}
    except ImportError:
        # Fallback: rough estimate
        token_count = int(char_count / 4)
        return {"token_count": token_count, "char_count": char_count, "word_count": word_count, "estimated": True}
