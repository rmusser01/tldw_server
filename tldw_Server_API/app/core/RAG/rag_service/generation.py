# generation.py
"""
Response generation strategies for the RAG service.

This module provides LLM integration for generating responses using retrieved context,
with support for multiple providers, streaming, and fallback strategies.
"""

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, Protocol, Union, cast

from loguru import logger

from tldw_Server_API.app.core.Utils.prompt_loader import load_prompt

from .types import Document

if TYPE_CHECKING:
    from .claims import ClaimsEngine as ClaimsEngineType
else:
    ClaimsEngineType = Any

ClaimsEngine: Optional[type[ClaimsEngineType]] = None
try:
    from . import claims as _claims_mod
    ClaimsEngine = cast(Optional[type[ClaimsEngineType]], getattr(_claims_mod, "ClaimsEngine", None))
except ImportError:
    ClaimsEngine = None


class GenerationStrategy(Protocol):
    """Protocol for response generation strategies."""

    async def generate(
        self,
        context: Any,  # RAGPipelineContext
        query: str,
        **kwargs
    ) -> "GenerationResult":
        """Generate a response using the context and query."""
        ...

    def generate_stream(
        self,
        context: Any,
        query: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        ...


@dataclass
class GenerationConfig:
    """Configuration for response generation."""
    provider: str = "openai"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1024
    streaming: bool = False
    fallback_enabled: bool = True
    prompt_template: str = "default"
    system_prompt: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 60
    retry_attempts: int = 3
    retry_delay: int = 2


@dataclass
class GenerationResult:
    """Result from response generation."""
    response: str
    tokens_used: int
    generation_time: float
    provider: str
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)


class PromptTemplates:
    """Collection of prompt templates for different use cases."""

    DEFAULT = """You are a helpful AI assistant. Use the following context to answer the user's question.
If the context doesn't contain relevant information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

    DETAILED = """You are an expert research assistant. Analyze the following context carefully and provide a comprehensive answer to the user's question.

Context Documents:
{context}

User Question: {question}

Instructions:
1. Provide a detailed answer based on the context
2. Cite specific information from the context when possible
3. If information is missing, clearly state what is not available
4. Structure your response with clear sections if appropriate

Answer:"""

    CONCISE = """Based on the context below, provide a brief, direct answer to the question.

Context: {context}

Question: {question}

Brief Answer:"""

    ACADEMIC = """You are an academic researcher. Use the provided sources to answer the question with scholarly precision.

Research Sources:
{context}

Research Question: {question}

Provide a well-referenced answer with clear attribution to sources:"""

    CONVERSATIONAL = """Hey! I've found some information that might help answer your question.

Here's what I found:
{context}

Your question: {question}

Let me explain:"""

    @staticmethod
    @lru_cache(maxsize=64)
    def _load_rag_prompt_cached(name: str) -> Optional[str]:
        """Load prompt snippets from rag.prompts.* with a small process cache."""
        try:
            prompt_text = load_prompt("rag", name)
        except Exception:  # noqa: BLE001 - prompt loading must remain best-effort
            logger.debug("Prompt loader failed for rag prompt '{}'", name)
            return None
        if isinstance(prompt_text, str) and prompt_text.strip():
            return prompt_text.strip()
        return None

    @classmethod
    async def warm_template_async(cls, name: str) -> None:
        """Preload a template off the event loop for async request paths."""
        if not isinstance(name, str):
            return
        normalized_name = name.strip()
        if not normalized_name:
            return
        try:
            await asyncio.to_thread(cls._load_rag_prompt_cached, normalized_name)
        except Exception as exc:  # noqa: BLE001 - warmup remains best-effort
            logger.debug(
                "Prompt warmup failed for rag prompt '{}': {}",
                normalized_name,
                exc,
            )

    @classmethod
    def get_template(cls, name: str) -> str:
        """Get a template by name."""
        external = cls._load_rag_prompt_cached(name)
        if external is not None:
            return external

        templates = {
            "default": cls.DEFAULT,
            "detailed": cls.DETAILED,
            "concise": cls.CONCISE,
            "academic": cls.ACADEMIC,
            "conversational": cls.CONVERSATIONAL
        }
        return templates.get(name, cls.DEFAULT)


def _extract_openai_text_content(response: Any) -> Optional[str]:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        choices = response.get("choices") or []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = [part.get("text", "") for part in content if isinstance(part, dict)]
                    if parts:
                        return "".join(parts)
            delta = choice.get("delta")
            if isinstance(delta, dict):
                delta_content = delta.get("content")
                if isinstance(delta_content, str):
                    return delta_content
                if isinstance(delta_content, list):
                    parts = [part.get("text", "") for part in delta_content if isinstance(part, dict)]
                    if parts:
                        return "".join(parts)
            text = choice.get("text")
            if isinstance(text, str):
                return text
        content = response.get("content") or response.get("text")
        if isinstance(content, str):
            return content
        return None
    return None


def _extract_openai_content(response: Any) -> str:
    text = _extract_openai_text_content(response)
    if text is not None:
        return text
    return str(response)


def _extract_stream_text(chunk: Any) -> Optional[str]:
    if isinstance(chunk, dict):
        return _extract_openai_text_content(chunk)
    if isinstance(chunk, (bytes, bytearray)):
        try:
            chunk = chunk.decode("utf-8", errors="ignore")
        except UnicodeDecodeError:
            return None
    if isinstance(chunk, str):
        stripped = chunk.strip()
        if not stripped:
            return None
        if stripped.lower().startswith("data:"):
            data = stripped[5:].strip()
            if data.lower() == "[done]":
                return None
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                return data or None
            if isinstance(payload, dict):
                return _extract_openai_text_content(payload)
            if isinstance(payload, str):
                return payload or None
            return None
        return stripped
    try:
        return str(chunk)
    except (TypeError, ValueError):
        return None


class BaseGenerator(ABC):
    """Base class for response generators."""

    def __init__(self, config: GenerationConfig):
        """Initialize generator with configuration."""
        self.config = config
        self.prompt_template = PromptTemplates.get_template(config.prompt_template)

    def format_context(self, documents: list[Document]) -> str:
        """Format documents into context string."""
        if not documents:
            return "No relevant context found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Format each document with metadata
            source = doc.metadata.get("source", "Unknown")
            title = doc.metadata.get("title", f"Document {i}")

            context_parts.append(f"[Source {i}: {title} ({source})]")
            context_parts.append(doc.content)
            context_parts.append("")  # Empty line between documents

        return "\n".join(context_parts)

    def build_prompt(self, context_text: str, query: str) -> str:
        """Build the final prompt from template."""
        return self.prompt_template.format(
            context=context_text,
            question=query
        )

    @abstractmethod
    async def generate(
        self,
        context: Any,
        query: str,
        **kwargs
    ) -> GenerationResult:
        """Generate a response."""
        pass

    async def generate_stream(
        self,
        context: Any,
        query: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        # Default implementation: yield complete response
        result = await self.generate(context, query, **kwargs)
        yield result.response


class LLMGenerator(BaseGenerator):
    """Generator using LLM API calls."""

    async def generate(
        self,
        context: Any,
        query: str,
        **kwargs
    ) -> GenerationResult:
        """Generate response using configured LLM provider."""
        start_time = time.time()

        try:
            # Extract documents from context
            documents = context.documents if hasattr(context, 'documents') else []

            # Format context
            context_text = self.format_context(documents)

            # Build prompt
            prompt = self.build_prompt(context_text, query)

            # Add system prompt if configured
            full_prompt = f"{self.config.system_prompt}\n\n{prompt}" if self.config.system_prompt else prompt

            # Call appropriate LLM provider
            response: Any = await self._call_llm(full_prompt, **kwargs)
            if asyncio.iscoroutine(response):
                response = await response

            # Extract text from response
            response_text = _extract_openai_content(response)
            if isinstance(response, dict):
                tokens_used = response.get("usage", {}).get("total_tokens", 0)
            else:
                tokens_used = len(response_text.split()) * 1.3  # Rough estimate

            generation_time = time.time() - start_time

            logger.info(
                f"Generated response using {self.config.provider}/{self.config.model} "
                f"in {generation_time:.2f}s ({tokens_used} tokens)"
            )

            return GenerationResult(
                response=response_text,
                tokens_used=int(tokens_used),
                generation_time=generation_time,
                provider=self.config.provider,
                model=self.config.model,
                metadata={
                    "prompt_length": len(full_prompt),
                    "context_documents": len(documents)
                }
            )

        except Exception:
            logger.error("Error generating response")

            # Try fallback if enabled
            if self.config.fallback_enabled:
                logger.info("Attempting fallback generation")
                fallback_gen = FallbackGenerator(self.config)
                return await fallback_gen.generate(context, query, **kwargs)

            raise

    async def _call_llm(self, prompt: str, **kwargs) -> Any:
        """Call the appropriate LLM provider via the chat service."""
        # Lazy import to avoid circular dependencies
        from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

        provider = (self.config.provider or "").lower()
        streaming = bool(kwargs.get("streaming", self.config.streaming))
        model = kwargs.get("model", self.config.model)
        api_key = kwargs.get("api_key", self.config.api_key)
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        call_kwargs: dict[str, Any] = {
            "api_provider": provider,
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": streaming,
        }
        if api_key:
            call_kwargs["api_key"] = api_key

        return await perform_chat_api_call_async(**call_kwargs)


class StreamingGenerator(LLMGenerator):
    """Generator with streaming response support."""

    async def generate_stream(
        self,
        context: Any,
        query: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        # Enable streaming in config
        original_streaming = self.config.streaming
        self.config.streaming = True

        try:
            # Extract documents from context
            documents = context.documents if hasattr(context, 'documents') else []

            # Format context
            context_text = self.format_context(documents)

            # Build prompt
            prompt = self.build_prompt(context_text, query)

            # Add system prompt if configured
            full_prompt = f"{self.config.system_prompt}\n\n{prompt}" if self.config.system_prompt else prompt

            # Call LLM with streaming
            response = await self._call_llm(full_prompt, **kwargs)

            # Handle streaming response
            if hasattr(response, '__aiter__'):
                # Async iterator
                async for chunk in response:
                    text = _extract_stream_text(chunk)
                    if text:
                        yield text
            elif hasattr(response, '__iter__'):
                # Sync iterator - convert to async
                for chunk in response:
                    text = _extract_stream_text(chunk)
                    if text:
                        yield text
                    await asyncio.sleep(0)  # Allow other tasks
            else:
                # Non-streaming response
                text = _extract_openai_content(response)

                # Simulate streaming by yielding in chunks
                chunk_size = 50  # characters
                for i in range(0, len(text), chunk_size):
                    yield text[i:i+chunk_size]
                    await asyncio.sleep(0.01)  # Small delay for streaming effect

        finally:
            # Restore original streaming setting
            self.config.streaming = original_streaming


class FallbackGenerator(BaseGenerator):
    """Fallback generator when LLM is unavailable."""

    async def generate(
        self,
        context: Any,
        query: str,
        **kwargs
    ) -> GenerationResult:
        """Generate a simple response without LLM."""
        start_time = time.time()

        # Extract documents from context
        documents = context.documents if hasattr(context, 'documents') else []

        if not documents:
            response = (
                f"I couldn't find any relevant information to answer your question: '{query}'. "
                "Please try rephrasing your question or providing more context."
            )
        else:
            # Build a simple response from the context
            response_parts = [
                f"Based on the available information, here's what I found regarding: '{query}'",
                "",
                "Relevant Information:"
            ]

            for i, doc in enumerate(documents[:3], 1):  # Limit to top 3 documents
                title = doc.metadata.get("title", f"Source {i}")
                content_preview = doc.content[:500] + "..." if len(doc.content) > 500 else doc.content

                response_parts.append(f"\n[{i}] From {title}:")
                response_parts.append(content_preview)

            response_parts.append(
                "\nNote: This is a simplified response. For a more detailed answer, "
                "please ensure the AI service is properly configured."
            )

            response = "\n".join(response_parts)

        generation_time = time.time() - start_time

        logger.info(f"Generated fallback response in {generation_time:.2f}s")

        return GenerationResult(
            response=response,
            tokens_used=len(response.split()),
            generation_time=generation_time,
            provider="fallback",
            model="none",
            metadata={
                "context_documents": len(documents),
                "fallback_reason": "LLM unavailable or error"
            }
        )


def create_generator(config: Union[GenerationConfig, dict[str, Any]]) -> GenerationStrategy:
    """Factory function to create appropriate generator."""
    if isinstance(config, dict):
        config = GenerationConfig(**config)

    if config.streaming:
        logger.debug(f"Creating StreamingGenerator with provider: {config.provider}")
        return StreamingGenerator(config)
    elif config.provider == "fallback":
        logger.debug("Creating FallbackGenerator")
        return FallbackGenerator(config)
    else:
        logger.debug(f"Creating LLMGenerator with provider: {config.provider}")
        return LLMGenerator(config)


def _sanitize_generation_config(config: dict[str, Any]) -> dict[str, Any]:
    """Drop non-GenerationConfig keys before instantiating a generator."""
    allowed_fields = set(GenerationConfig.__dataclass_fields__.keys())
    return {
        key: value
        for key, value in config.items()
        if key in allowed_fields
    }


# Pipeline integration functions

async def generate_response(context: Any, **kwargs) -> Any:
    """Generate response for pipeline context."""
    config_dict = context.config.get("generation", {})

    # Override with kwargs
    config_dict.update(kwargs)

    prompt_name = config_dict.get("prompt_template", "default")
    if isinstance(prompt_name, str):
        await PromptTemplates.warm_template_async(prompt_name)

    # Create generator
    generator = create_generator(_sanitize_generation_config(config_dict))

    # Generate response
    result = await generator.generate(context, context.query)

    # Add to context
    context.response = result.response
    context.metadata["generation"] = {
        "provider": result.provider,
        "model": result.model,
        "tokens_used": result.tokens_used,
        "generation_time": result.generation_time
    }

    return context


# Thin wrapper expected by unified_pipeline
class AnswerGenerator:
    """Minimal wrapper to generate answers used by unified_pipeline.

    Provides a simple interface: initialize with optional model/provider and
    call `generate(query=..., context=..., prompt_template=..., max_tokens=...)`.
    Returns a plain string or a dict with an `answer` key for backward compatibility.
    """

    def __init__(self, model: Optional[str] = None, provider: Optional[str] = None, system_prompt: Optional[str] = None):
        # Lazy-configure provider/model from env/config when not provided
        try:
            from tldw_Server_API.app.core.config import load_and_log_configs
            cfg = load_and_log_configs() or {}
        except Exception:  # noqa: BLE001 - config load best-effort
            cfg = {}
        self.provider = (provider or cfg.get("RAG_DEFAULT_LLM_PROVIDER") or "openai").strip()
        self.model = (model or cfg.get("RAG_DEFAULT_LLM_MODEL") or "gpt-4o-mini").strip()
        self.system_prompt = system_prompt or cfg.get("RAG_DEFAULT_SYSTEM_PROMPT")

    async def generate(
        self,
        *,
        query: str,
        context: str,
        prompt_template: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Union[str, dict[str, Any]]:
        # Build a minimal GenerationConfig and use LLMGenerator under the hood
        gcfg = GenerationConfig(
            provider=self.provider,
            model=self.model,
            max_tokens=int(max_tokens or 500),
            prompt_template=(prompt_template or "default"),
            system_prompt=self.system_prompt,
        )
        await PromptTemplates.warm_template_async(gcfg.prompt_template)
        gen = LLMGenerator(gcfg)

        # Create a tiny context holder compatible with BaseGenerator expectations
        class _Ctx:
            def __init__(self, documents: list[Document], query: str):
                self.documents = documents
                self.query = query

        # Convert raw context string into a single Document to preserve downstream formatting
        doc = Document(id="ctx", content=context or "", metadata={"source": "context", "title": "Context"})
        ctx = _Ctx([doc], query)
        res = await gen.generate(ctx, query)
        # Normalize to simple shape
        return {"answer": res.response, "provider": res.provider, "model": res.model, "tokens_used": res.tokens_used, "generation_time": res.generation_time}


async def generate_streaming_response(context: Any, **kwargs) -> Any:
    """Generate streaming response for pipeline context."""
    config_dict = context.config.get("generation", {})
    config_dict["streaming"] = True

    # Override with kwargs
    config_dict.update(kwargs)

    prompt_name = config_dict.get("prompt_template", "default")
    if isinstance(prompt_name, str):
        await PromptTemplates.warm_template_async(prompt_name)

    # Create generator
    generator = create_generator(_sanitize_generation_config(config_dict))

    # Store generator in context for streaming
    base_stream = generator.generate_stream(context, context.query)

    # Optional: streaming claims overlay with slight buffer
    enable_claims = bool(kwargs.get("enable_claims", False))
    claims_top_k = int(kwargs.get("claims_top_k", 3))
    claims_max = int(kwargs.get("claims_max", 10))
    try:
        claims_concurrency = int(kwargs.get("claims_concurrency", 8))
    except (TypeError, ValueError):
        claims_concurrency = 8

    if enable_claims and ClaimsEngine is not None:
        try:

            def _analyze(api_name: str, input_data: Any, custom_prompt_arg: Optional[str] = None,
                         api_key: Optional[str] = None, system_message: Optional[str] = None,
                         temp: Optional[float] = None, **k):
                # For streaming overlay, avoid heavy LLM calls; use heuristic path via empty analyze
                return "{\"claims\": []}"

            engine = ClaimsEngine(_analyze)

            async def _wrapped_stream():
                buffer = ""
                last_emit = 0
                last_emit_time = 0.0
                sentence_re = re.compile(r"(?<=[\.!?])\s+")
                async for chunk in base_stream:
                    buffer += chunk
                    # Yield original chunk immediately
                    yield chunk
                    # When buffer has at least two sentences, run lightweight claim extraction
                    parts = sentence_re.split(buffer)
                    if len(parts) >= 2 and len(buffer) - last_emit > 200:
                        # Debounce: limit overlay extraction rate
                        now = time.time()
                        if now - last_emit_time < 0.4:
                            continue
                        tail = " ".join(parts[-2:])
                        try:
                            claims_out = await engine.run(
                                answer=tail,
                                query=context.query,
                                documents=getattr(context, 'documents', []) or [],
                                claim_extractor="auto",
                                claim_verifier="hybrid",
                                claims_top_k=claims_top_k,
                                claims_conf_threshold=0.75,
                                claims_max=min(5, claims_max),
                                retrieve_fn=None,
                                claims_concurrency=claims_concurrency,
                            )
                            context.metadata["claims_overlay"] = claims_out
                            last_emit = len(buffer)
                            last_emit_time = now
                        except Exception:  # noqa: BLE001 - claims overlay best-effort
                            logger.debug("Claims overlay enrichment failed during streaming generation")
                # done
                return

            context.stream_generator = _wrapped_stream()
        except Exception:  # noqa: BLE001 - fallback to base stream
            context.stream_generator = base_stream
    else:
        context.stream_generator = base_stream
    context.metadata["streaming"] = True

    return context
