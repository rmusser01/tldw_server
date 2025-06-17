# exceptions.py
# Description: Exception hierarchy for RAG (Retrieval-Augmented Generation) operations
#
"""
RAG Exception Hierarchy
========================

This module defines a comprehensive exception hierarchy for RAG operations,
providing clear categorization of different error types and consistent
error handling patterns across the RAG system.

Exception Categories:
- RAGError: Base exception for all RAG-related errors
- RAGSearchError: Search and retrieval operation failures
- RAGConfigurationError: Configuration and setup issues
- RAGDatabaseError: Database connection and query failures
- RAGEmbeddingError: Embedding generation and storage failures
- RAGGenerationError: LLM response generation failures
"""

from typing import Optional, Any, Dict, List


class RAGError(Exception):
    """
    Base exception for all RAG-related errors.
    
    Provides common functionality for all RAG exceptions including
    error context, operation tracking, and structured error information.
    
    Attributes:
        operation: The RAG operation that failed (e.g., "vector_search", "generate_answer")
        context: Additional context about the error (IDs, parameters, etc.)
        original_error: The original exception that caused this error (if any)
    """
    
    def __init__(
        self, 
        message: str, 
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.operation = operation
        self.context = context or {}
        self.original_error = original_error
    
    def __str__(self) -> str:
        base_message = super().__str__()
        parts = [base_message]
        
        if self.operation:
            parts.append(f"Operation: {self.operation}")
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")
            
        if self.original_error:
            parts.append(f"Caused by: {type(self.original_error).__name__}: {self.original_error}")
        
        return " | ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": type(self).__name__,
            "message": str(super().__str__()),  # Get base message without formatting
            "operation": self.operation,
            "context": self.context,
            "original_error": str(self.original_error) if self.original_error else None
        }


class RAGSearchError(RAGError):
    """
    Exception for search and retrieval operation failures.
    
    This covers errors in:
    - Vector search operations
    - Full-text search failures
    - Embedding retrieval issues
    - Search result processing errors
    """
    
    def __init__(
        self,
        message: str,
        search_type: Optional[str] = None,
        query: Optional[str] = None,
        database_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})  # Use pop to avoid duplicate
        if search_type:
            context['search_type'] = search_type
        if query:
            context['query'] = query[:100] + "..." if len(query) > 100 else query  # Truncate long queries
        if database_type:
            context['database_type'] = database_type
        
        super().__init__(message, operation="search", context=context, **kwargs)


class RAGConfigurationError(RAGError):
    """
    Exception for configuration and setup issues.
    
    This covers errors in:
    - Missing or invalid configuration values
    - API key validation failures
    - Model configuration issues
    - Environment setup problems
    """
    
    def __init__(
        self,
        message: str,
        config_section: Optional[str] = None,
        config_key: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})  # Use pop to avoid duplicate
        if config_section:
            context['config_section'] = config_section
        if config_key:
            context['config_key'] = config_key
        
        super().__init__(message, operation="configuration", context=context, **kwargs)


class RAGDatabaseError(RAGError):
    """
    Exception for database connection and query failures.
    
    This covers errors in:
    - Database connection issues
    - SQL query execution failures
    - Transaction failures
    - Database schema issues
    """
    
    def __init__(
        self,
        message: str,
        database_name: Optional[str] = None,
        operation_type: Optional[str] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[Any] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})  # Use pop to avoid duplicate
        if database_name:
            context['database_name'] = database_name
        if operation_type:
            context['operation_type'] = operation_type
        if entity_type:
            context['entity_type'] = entity_type
        if entity_id is not None:
            context['entity_id'] = str(entity_id)
        
        super().__init__(message, operation="database", context=context, **kwargs)


class RAGEmbeddingError(RAGError):
    """
    Exception for embedding generation and storage failures.
    
    This covers errors in:
    - Embedding API calls
    - Vector storage operations
    - ChromaDB operations
    - Embedding model issues
    """
    
    def __init__(
        self,
        message: str,
        embedding_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})  # Use pop to avoid duplicate
        if embedding_provider:
            context['embedding_provider'] = embedding_provider
        if model_name:
            context['model_name'] = model_name
        if collection_name:
            context['collection_name'] = collection_name
        
        super().__init__(message, operation="embedding", context=context, **kwargs)


class RAGGenerationError(RAGError):
    """
    Exception for LLM response generation failures.
    
    This covers errors in:
    - LLM API calls
    - Response parsing issues
    - Token limit exceeded
    - Model availability issues
    """
    
    def __init__(
        self,
        message: str,
        api_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        context_length: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})  # Use pop to avoid duplicate
        if api_provider:
            context['api_provider'] = api_provider
        if model_name:
            context['model_name'] = model_name
        if context_length is not None:
            context['context_length'] = context_length
        
        super().__init__(message, operation="generation", context=context, **kwargs)


class RAGValidationError(RAGError):
    """
    Exception for input validation failures.
    
    This covers errors in:
    - Invalid query parameters
    - Malformed input data
    - Missing required fields
    - Type validation failures
    """
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        validation_rule: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})  # Use pop to avoid duplicate
        if field_name:
            context['field_name'] = field_name
        if field_value is not None:
            context['field_value'] = str(field_value)[:200]  # Truncate long values
        if validation_rule:
            context['validation_rule'] = validation_rule
        
        super().__init__(message, operation="validation", context=context, **kwargs)


class RAGTimeoutError(RAGError):
    """
    Exception for operation timeout failures.
    
    This covers errors in:
    - API request timeouts
    - Database query timeouts
    - Embedding generation timeouts
    - Search operation timeouts
    """
    
    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        operation_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})  # Use pop to avoid duplicate
        if timeout_seconds is not None:
            context['timeout_seconds'] = timeout_seconds
        if operation_type:
            context['operation_type'] = operation_type
        
        super().__init__(message, operation="timeout", context=context, **kwargs)


# Utility functions for exception handling

def wrap_database_error(
    original_error: Exception,
    operation: str,
    **context_kwargs
) -> RAGDatabaseError:
    """
    Convert a generic database error to a RAGDatabaseError with context.
    
    Args:
        original_error: The original exception
        operation: The database operation that failed
        **context_kwargs: Additional context for the error
    
    Returns:
        RAGDatabaseError with proper context
    """
    return RAGDatabaseError(
        message=f"Database {operation} failed: {str(original_error)}",
        operation_type=operation,
        context=context_kwargs,
        original_error=original_error
    )


def wrap_search_error(
    original_error: Exception,
    search_type: str,
    query: str,
    **context_kwargs
) -> RAGSearchError:
    """
    Convert a generic search error to a RAGSearchError with context.
    
    Args:
        original_error: The original exception
        search_type: The type of search that failed
        query: The search query
        **context_kwargs: Additional context for the error
    
    Returns:
        RAGSearchError with proper context
    """
    return RAGSearchError(
        message=f"{search_type} search failed: {str(original_error)}",
        search_type=search_type,
        query=query,
        context=context_kwargs,
        original_error=original_error
    )


def handle_rag_error(
    error: Exception,
    operation: str,
    fallback_result: Any = None,
    log_function: Optional[callable] = None
) -> Any:
    """
    Standard error handling pattern for RAG operations.
    
    Args:
        error: The exception that occurred
        operation: The operation that failed
        fallback_result: Value to return on error (default: None)
        log_function: Optional logging function to call
    
    Returns:
        fallback_result
    """
    if isinstance(error, RAGError):
        # Already a RAG error, re-raise
        if log_function:
            log_function(f"RAG error in {operation}: {error}")
        raise error
    else:
        # Convert to appropriate RAG error
        rag_error = RAGError(
            message=f"{operation} failed: {str(error)}",
            operation=operation,
            original_error=error
        )
        if log_function:
            log_function(f"Converted error in {operation}: {rag_error}")
        raise rag_error