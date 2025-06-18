"""
Response generation strategies for the RAG service.

This module handles LLM integration for generating responses
optimized for a single-user TUI application.
"""

import asyncio
from typing import Dict, Any, Optional, AsyncIterator, List
from abc import abstractmethod
import time
import json

from loguru import logger

from .types import (
    GenerationStrategy, RAGContext, GenerationError
)
from .utils import TokenCounter


class BaseGenerator(GenerationStrategy):
    """
    Base generator with common functionality.
    
    For single-user app, we can:
    - Keep LLM clients initialized
    - Use user's preferred models
    - Optimize for response quality
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._token_counter = TokenCounter()
        
        # Default generation parameters
        self.default_temperature = self.config.get("default_temperature", 0.7)
        self.default_max_tokens = self.config.get("max_tokens", 1024)
        self.system_prompt_template = self.config.get(
            "system_prompt_template",
            """You are a helpful AI assistant. Use the following context to answer the user's question.
If the context doesn't contain relevant information, say so clearly.

Context:
{context}

Question: {question}
Answer:"""
        )
    
    @abstractmethod
    async def generate(
        self,
        context: RAGContext,
        query: str,
        **kwargs
    ) -> str:
        """Generate response using the context."""
        pass
    
    def _format_prompt(self, context: RAGContext, query: str) -> str:
        """Format the prompt with context and query."""
        return self.system_prompt_template.format(
            context=context.combined_text,
            question=query
        )


class LLMGenerator(BaseGenerator):
    """
    Generator using the existing LLM infrastructure.
    
    This integrates with the app's existing LLM call system.
    """
    
    def __init__(
        self,
        llm_handler: Any,  # The app's LLM handler
        config: Dict[str, Any] = None
    ):
        super().__init__(config)
        self.llm_handler = llm_handler
        self.model = config.get("default_model")
        self.provider = config.get("default_provider")
    
    async def generate(
        self,
        context: RAGContext,
        query: str,
        **kwargs
    ) -> str:
        """
        Generate response using the app's LLM infrastructure.
        
        For single-user app, we use their configured LLM settings.
        """
        try:
            # Override with kwargs if provided
            model = kwargs.get("model", self.model)
            provider = kwargs.get("provider", self.provider)
            temperature = kwargs.get("temperature", self.default_temperature)
            max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
            
            # Format the prompt
            prompt = self._format_prompt(context, query)
            
            # Count tokens for logging
            prompt_tokens = self._token_counter.count(prompt)
            logger.debug(f"Generation prompt: {prompt_tokens} tokens")
            
            # Call LLM through the app's infrastructure
            response = await self._call_llm(
                prompt=prompt,
                model=model,
                provider=provider,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Extract answer from response
            if isinstance(response, dict):
                answer = response.get("content", response.get("text", str(response)))
            else:
                answer = str(response)
            
            logger.info(f"Generated response: {len(answer)} characters")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise GenerationError(f"Failed to generate response: {e}")
    
    async def _call_llm(
        self,
        prompt: str,
        model: str,
        provider: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Any:
        """
        Call the LLM through the app's infrastructure.
        
        This should be adapted to match the actual LLM call interface.
        """
        # This is a placeholder - replace with actual LLM call
        # In the real implementation, this would use the app's
        # existing LLM infrastructure
        
        if hasattr(self.llm_handler, 'generate_response'):
            return await self.llm_handler.generate_response(
                prompt=prompt,
                model=model,
                provider=provider,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        else:
            # Fallback synchronous call
            return self.llm_handler(
                prompt=prompt,
                model=model,
                provider=provider,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )


class StreamingGenerator(BaseGenerator):
    """
    Generator with streaming support for TUI display.
    
    Perfect for showing responses as they're generated in the TUI.
    """
    
    def __init__(
        self,
        llm_handler: Any,
        config: Dict[str, Any] = None
    ):
        super().__init__(config)
        self.llm_handler = llm_handler
        self.stream_chunk_size = config.get("stream_chunk_size", 10)
    
    async def generate(
        self,
        context: RAGContext,
        query: str,
        **kwargs
    ) -> str:
        """Generate complete response (non-streaming)."""
        # Collect all chunks
        chunks = []
        async for chunk in self.generate_stream(context, query, **kwargs):
            chunks.append(chunk)
        return "".join(chunks)
    
    async def generate_stream(
        self,
        context: RAGContext,
        query: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate response in streaming fashion.
        
        Yields chunks of text as they're generated.
        """
        try:
            prompt = self._format_prompt(context, query)
            
            # Stream from LLM
            async for chunk in self._stream_llm(
                prompt=prompt,
                temperature=kwargs.get("temperature", self.default_temperature),
                max_tokens=kwargs.get("max_tokens", self.default_max_tokens),
                **kwargs
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            yield f"\n\nError: {str(e)}"
    
    async def _stream_llm(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream from LLM.
        
        This should be adapted to the actual streaming interface.
        """
        # Placeholder for actual streaming implementation
        # In real implementation, this would use the app's
        # streaming LLM interface
        
        if hasattr(self.llm_handler, 'stream_response'):
            async for chunk in self.llm_handler.stream_response(
                prompt=prompt,
                **kwargs
            ):
                yield chunk
        else:
            # Fallback: simulate streaming
            response = await self.generate(
                RAGContext(
                    documents=[],
                    combined_text=prompt,
                    total_tokens=0,
                    metadata={}
                ),
                "",
                **kwargs
            )
            
            # Yield in chunks
            for i in range(0, len(response), self.stream_chunk_size):
                yield response[i:i + self.stream_chunk_size]
                await asyncio.sleep(0.01)  # Small delay for UI


class MockGenerator(BaseGenerator):
    """
    Mock generator for testing without LLM calls.
    
    Useful for development and testing the RAG pipeline.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.mock_delay = config.get("mock_delay", 0.5)
    
    async def generate(
        self,
        context: RAGContext,
        query: str,
        **kwargs
    ) -> str:
        """Generate a mock response based on context."""
        # Simulate processing time
        await asyncio.sleep(self.mock_delay)
        
        # Generate mock response
        num_sources = len(context.documents)
        sources_summary = []
        
        for doc in context.documents[:3]:  # Show first 3 sources
            source_type = doc.source.name
            title = doc.metadata.get("title", "Untitled")
            sources_summary.append(f"- {source_type}: {title}")
        
        response = f"""Based on the {num_sources} sources found, here's a summary regarding "{query}":

The context contains information from:
{chr(10).join(sources_summary)}

The main points from the context are:
1. [Mock summary point 1 based on the query]
2. [Mock summary point 2 based on the query]
3. [Mock summary point 3 based on the query]

This is a mock response for testing. In production, this would be replaced with actual LLM-generated content based on the provided context.

Context preview: {context.combined_text[:200]}...
"""
        
        return response


class FallbackGenerator(BaseGenerator):
    """
    Fallback generator that provides responses without LLM.
    
    Useful when LLM is unavailable or for certain types of queries.
    """
    
    async def generate(
        self,
        context: RAGContext,
        query: str,
        **kwargs
    ) -> str:
        """Generate a structured response without LLM."""
        if not context.documents:
            return "I couldn't find any relevant information to answer your question."
        
        # Build a structured response
        response_parts = [
            f"Found {len(context.documents)} relevant sources for: {query}",
            "",
            "Here are the key sources:"
        ]
        
        # Add source summaries
        for i, doc in enumerate(context.documents[:5], 1):
            source_type = doc.source.name.replace("_", " ").title()
            title = doc.metadata.get("title", doc.metadata.get("conversation_title", "Untitled"))
            
            # Extract first few sentences
            sentences = doc.content.split(". ")[:2]
            preview = ". ".join(sentences)
            if len(preview) > 200:
                preview = preview[:197] + "..."
            
            response_parts.extend([
                "",
                f"{i}. {source_type}: {title}",
                f"   {preview}"
            ])
        
        if len(context.documents) > 5:
            response_parts.append(f"\n...and {len(context.documents) - 5} more sources")
        
        response_parts.extend([
            "",
            "Note: This is a structured summary without LLM processing. "
            "For a more natural response, please ensure your LLM is configured correctly."
        ])
        
        return "\n".join(response_parts)


class AdaptiveGenerator(BaseGenerator):
    """
    Adaptive generator that chooses the best generation strategy.
    
    For single-user app, this can adapt based on:
    - Query type
    - Context size
    - User preferences
    """
    
    def __init__(
        self,
        generators: Dict[str, GenerationStrategy],
        config: Dict[str, Any] = None
    ):
        super().__init__(config)
        self.generators = generators
        self.default_generator = config.get("default_generator", "llm")
    
    async def generate(
        self,
        context: RAGContext,
        query: str,
        **kwargs
    ) -> str:
        """
        Generate using the most appropriate strategy.
        
        Chooses generator based on context and query characteristics.
        """
        # Determine best generator
        generator_name = self._select_generator(context, query, **kwargs)
        
        if generator_name not in self.generators:
            logger.warning(f"Generator '{generator_name}' not found, using default")
            generator_name = self.default_generator
        
        if generator_name not in self.generators:
            raise GenerationError("No valid generator available")
        
        generator = self.generators[generator_name]
        logger.debug(f"Using generator: {generator_name}")
        
        # Generate response
        return await generator.generate(context, query, **kwargs)
    
    def _select_generator(
        self,
        context: RAGContext,
        query: str,
        **kwargs
    ) -> str:
        """Select the best generator based on context."""
        # Override with explicit choice
        if "generator" in kwargs:
            return kwargs["generator"]
        
        # Simple heuristics for single-user app
        
        # No context -> fallback
        if not context.documents:
            return "fallback"
        
        # Very short context -> might not need LLM
        if context.total_tokens < 100:
            return "fallback"
        
        # Streaming requested
        if kwargs.get("stream", False):
            return "streaming"
        
        # Default to LLM
        return "llm"