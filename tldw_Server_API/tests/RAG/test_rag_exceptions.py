# test_rag_exceptions.py
# Description: Comprehensive tests for the RAG exception hierarchy
#
# Imports
import pytest
import json
from unittest.mock import MagicMock
#
# Local Imports
from tldw_Server_API.app.core.RAG.exceptions import (
    RAGError,
    RAGSearchError,
    RAGConfigurationError,
    RAGDatabaseError,
    RAGEmbeddingError,
    RAGGenerationError,
    RAGValidationError,
    RAGTimeoutError,
    wrap_database_error,
    wrap_search_error,
    handle_rag_error
)
#
#######################################################################################################################
#
# Test Classes

class TestRAGErrorBase:
    """Test the base RAGError class functionality."""
    
    def test_basic_error_creation(self):
        """Test basic RAGError creation and attributes."""
        error = RAGError("Test error message")
        assert str(error) == "Test error message"
        assert error.operation is None
        assert error.context == {}
        assert error.original_error is None
    
    def test_error_with_operation(self):
        """Test RAGError with operation context."""
        error = RAGError("Test error", operation="test_operation")
        assert "Operation: test_operation" in str(error)
        assert error.operation == "test_operation"
    
    def test_error_with_context(self):
        """Test RAGError with context dictionary."""
        context = {"param1": "value1", "param2": 42}
        error = RAGError("Test error", context=context)
        assert "Context: param1=value1, param2=42" in str(error)
        assert error.context == context
    
    def test_error_with_original_error(self):
        """Test RAGError with original error chaining."""
        original = ValueError("Original error")
        error = RAGError("Wrapped error", original_error=original)
        assert "Caused by: ValueError: Original error" in str(error)
        assert error.original_error == original
    
    def test_error_with_all_attributes(self):
        """Test RAGError with all attributes set."""
        original = KeyError("Missing key")
        context = {"operation_id": "123", "retry_count": 3}
        error = RAGError(
            "Complex error",
            operation="complex_operation",
            context=context,
            original_error=original
        )
        
        error_str = str(error)
        assert "Complex error" in error_str
        assert "Operation: complex_operation" in error_str
        assert "Context: operation_id=123, retry_count=3" in error_str
        assert "Caused by: KeyError:" in error_str and "Missing key" in error_str
    
    def test_to_dict_serialization(self):
        """Test error serialization to dictionary."""
        original = RuntimeError("Runtime issue")
        context = {"key": "value"}
        error = RAGError(
            "Serializable error",
            operation="serialize_test",
            context=context,
            original_error=original
        )
        
        error_dict = error.to_dict()
        assert error_dict["error_type"] == "RAGError"
        assert error_dict["message"] == "Serializable error"
        assert error_dict["operation"] == "serialize_test"
        assert error_dict["context"] == context
        assert "Runtime issue" in error_dict["original_error"]


class TestSpecializedExceptions:
    """Test specialized RAG exception types."""
    
    def test_search_error_creation(self):
        """Test RAGSearchError with search-specific attributes."""
        error = RAGSearchError(
            "Search failed",
            search_type="vector_search",
            query="test query",
            database_type="ChromaDB"
        )
        
        assert error.context["search_type"] == "vector_search"
        assert error.context["query"] == "test query"
        assert error.context["database_type"] == "ChromaDB"
        assert error.operation == "search"
    
    def test_search_error_long_query_truncation(self):
        """Test that long queries are truncated in SearchError."""
        long_query = "a" * 200  # 200 character query
        error = RAGSearchError("Search failed", query=long_query)
        
        # Should be truncated to 100 chars + "..."
        assert len(error.context["query"]) == 103
        assert error.context["query"].endswith("...")
    
    def test_configuration_error_creation(self):
        """Test RAGConfigurationError with config-specific attributes."""
        error = RAGConfigurationError(
            "Config missing",
            config_section="api_settings",
            config_key="api_key"
        )
        
        assert error.context["config_section"] == "api_settings"
        assert error.context["config_key"] == "api_key"
        assert error.operation == "configuration"
    
    def test_database_error_creation(self):
        """Test RAGDatabaseError with database-specific attributes."""
        error = RAGDatabaseError(
            "DB connection failed",
            database_name="PostgreSQL",
            operation_type="select",
            entity_type="character_cards",
            entity_id=123
        )
        
        assert error.context["database_name"] == "PostgreSQL"
        assert error.context["operation_type"] == "select"
        assert error.context["entity_type"] == "character_cards"
        assert error.context["entity_id"] == "123"  # Should be string
        assert error.operation == "database"
    
    def test_embedding_error_creation(self):
        """Test RAGEmbeddingError with embedding-specific attributes."""
        error = RAGEmbeddingError(
            "Embedding failed",
            embedding_provider="openai",
            model_name="text-embedding-ada-002",
            collection_name="test_collection"
        )
        
        assert error.context["embedding_provider"] == "openai"
        assert error.context["model_name"] == "text-embedding-ada-002"
        assert error.context["collection_name"] == "test_collection"
        assert error.operation == "embedding"
    
    def test_generation_error_creation(self):
        """Test RAGGenerationError with generation-specific attributes."""
        error = RAGGenerationError(
            "Generation failed",
            api_provider="anthropic",
            model_name="claude-3",
            context_length=5000
        )
        
        assert error.context["api_provider"] == "anthropic"
        assert error.context["model_name"] == "claude-3"
        assert error.context["context_length"] == 5000
        assert error.operation == "generation"
    
    def test_validation_error_creation(self):
        """Test RAGValidationError with validation-specific attributes."""
        error = RAGValidationError(
            "Invalid input",
            field_name="query",
            field_value="",
            validation_rule="not_empty"
        )
        
        assert error.context["field_name"] == "query"
        assert error.context["field_value"] == ""
        assert error.context["validation_rule"] == "not_empty"
        assert error.operation == "validation"
    
    def test_validation_error_long_value_truncation(self):
        """Test that long field values are truncated in ValidationError."""
        long_value = "x" * 300  # 300 character value
        error = RAGValidationError("Invalid", field_value=long_value)
        
        # Should be truncated to 200 chars
        assert len(error.context["field_value"]) == 200
    
    def test_timeout_error_creation(self):
        """Test RAGTimeoutError with timeout-specific attributes."""
        error = RAGTimeoutError(
            "Operation timed out",
            timeout_seconds=30.5,
            operation_type="api_call"
        )
        
        assert error.context["timeout_seconds"] == 30.5
        assert error.context["operation_type"] == "api_call"
        assert error.operation == "timeout"


class TestExceptionUtilities:
    """Test utility functions for exception handling."""
    
    def test_wrap_database_error(self):
        """Test database error wrapping utility."""
        original_error = ConnectionError("Database unreachable")
        wrapped = wrap_database_error(
            original_error,
            operation="insert",
            table_name="character_cards",
            record_id=456
        )
        
        assert isinstance(wrapped, RAGDatabaseError)
        assert "Database insert failed" in str(wrapped)
        assert wrapped.context["operation_type"] == "insert"
        assert wrapped.context["table_name"] == "character_cards"
        assert wrapped.context["record_id"] == 456
        assert wrapped.original_error == original_error
    
    def test_wrap_search_error(self):
        """Test search error wrapping utility."""
        original_error = TimeoutError("Search timed out")
        wrapped = wrap_search_error(
            original_error,
            search_type="full_text",
            query="test search",
            collection="documents"
        )
        
        assert isinstance(wrapped, RAGSearchError)
        assert "full_text search failed" in str(wrapped)
        assert wrapped.context["search_type"] == "full_text"
        assert wrapped.context["query"] == "test search"
        assert wrapped.context["collection"] == "documents"
        assert wrapped.original_error == original_error
    
    def test_handle_rag_error_reraises_rag_errors(self):
        """Test that handle_rag_error re-raises existing RAG errors."""
        original_rag_error = RAGSearchError("Original search error")
        
        with pytest.raises(RAGSearchError) as exc_info:
            handle_rag_error(original_rag_error, "test_operation")
        
        assert exc_info.value == original_rag_error
    
    def test_handle_rag_error_wraps_non_rag_errors(self):
        """Test that handle_rag_error wraps non-RAG errors."""
        original_error = ValueError("Some value error")
        
        with pytest.raises(RAGError) as exc_info:
            handle_rag_error(original_error, "test_operation")
        
        wrapped_error = exc_info.value
        assert isinstance(wrapped_error, RAGError)
        assert "test_operation failed" in str(wrapped_error)
        assert wrapped_error.operation == "test_operation"
        assert wrapped_error.original_error == original_error
    
    def test_handle_rag_error_with_logging(self):
        """Test handle_rag_error with logging function."""
        logged_messages = []
        
        def mock_logger(message):
            logged_messages.append(message)
        
        original_error = RuntimeError("Runtime issue")
        
        with pytest.raises(RAGError):
            handle_rag_error(original_error, "test_op", log_function=mock_logger)
        
        assert len(logged_messages) == 1
        assert "Converted error in test_op" in logged_messages[0]
    
    def test_handle_rag_error_with_fallback_result(self):
        """Test handle_rag_error fallback result (not implemented in current version)."""
        # Note: Current implementation always raises, but test documents expected behavior
        original_error = ValueError("Test error")
        
        with pytest.raises(RAGError):
            handle_rag_error(original_error, "test_operation", fallback_result="fallback")


class TestExceptionHierarchy:
    """Test that exception hierarchy works correctly."""
    
    def test_all_exceptions_inherit_from_rag_error(self):
        """Test that all specialized exceptions inherit from RAGError."""
        exceptions_to_test = [
            RAGSearchError("test"),
            RAGConfigurationError("test"),
            RAGDatabaseError("test"),
            RAGEmbeddingError("test"),
            RAGGenerationError("test"),
            RAGValidationError("test"),
            RAGTimeoutError("test")
        ]
        
        for exc in exceptions_to_test:
            assert isinstance(exc, RAGError)
    
    def test_exception_catching_with_base_class(self):
        """Test that all RAG exceptions can be caught with base RAGError."""
        exceptions = [
            RAGSearchError("search error"),
            RAGConfigurationError("config error"),
            RAGDatabaseError("db error"),
            RAGEmbeddingError("embedding error"),
            RAGGenerationError("generation error"),
            RAGValidationError("validation error"),
            RAGTimeoutError("timeout error")
        ]
        
        for exc in exceptions:
            try:
                raise exc
            except RAGError as caught:
                assert caught == exc
            else:
                pytest.fail(f"Exception {type(exc)} was not caught by RAGError")
    
    def test_specific_exception_catching(self):
        """Test that specific exceptions can be caught individually."""
        search_error = RAGSearchError("search failed")
        
        # Should be caught by specific type
        try:
            raise search_error
        except RAGSearchError as caught:
            assert caught == search_error
        except RAGError:
            pytest.fail("Should have been caught by RAGSearchError, not base RAGError")


class TestExceptionContextHandling:
    """Test context handling in exceptions."""
    
    def test_context_merging_in_subclasses(self):
        """Test that subclass-specific context merges with base context."""
        base_context = {"operation_id": "123", "user_id": "user456"}
        error = RAGSearchError(
            "Search failed",
            search_type="vector",
            query="test",
            context=base_context
        )
        
        # Should have both base context and search-specific context
        assert error.context["operation_id"] == "123"
        assert error.context["user_id"] == "user456"
        assert error.context["search_type"] == "vector"
        assert error.context["query"] == "test"
    
    def test_context_override_behavior(self):
        """Test context override behavior when keys conflict."""
        base_context = {"search_type": "original_type"}
        error = RAGSearchError(
            "Search failed",
            search_type="new_type",  # This should override
            context=base_context
        )
        
        # The parameter should override the context value
        assert error.context["search_type"] == "new_type"
    
    def test_none_context_handling(self):
        """Test handling of None values in context."""
        error = RAGEmbeddingError(
            "Embedding failed",
            embedding_provider=None,
            model_name="test-model",
            collection_name=None
        )
        
        # None values should not be added to context
        assert "embedding_provider" not in error.context
        assert "collection_name" not in error.context
        assert error.context["model_name"] == "test-model"


class TestExceptionSerialization:
    """Test exception serialization and deserialization."""
    
    def test_complex_exception_serialization(self):
        """Test serialization of complex exception with all attributes."""
        original = ConnectionError("Connection failed")
        error = RAGDatabaseError(
            "Complex database error",
            database_name="test_db",
            operation_type="query",
            entity_type="users",
            entity_id="user123",
            context={"retry_count": 3, "timeout": 30},
            original_error=original
        )
        
        serialized = error.to_dict()
        
        # Check all expected fields are present
        assert serialized["error_type"] == "RAGDatabaseError"
        assert serialized["message"] == "Complex database error"
        assert serialized["operation"] == "database"
        assert serialized["context"]["database_name"] == "test_db"
        assert serialized["context"]["operation_type"] == "query"
        assert serialized["context"]["entity_type"] == "users"
        assert serialized["context"]["entity_id"] == "user123"
        assert serialized["context"]["retry_count"] == 3
        assert serialized["context"]["timeout"] == 30
        assert "Connection failed" in serialized["original_error"]
    
    def test_json_serialization_compatibility(self):
        """Test that serialized exceptions can be JSON serialized."""
        error = RAGGenerationError(
            "Generation failed",
            api_provider="openai",
            model_name="gpt-4",
            context_length=4000
        )
        
        error_dict = error.to_dict()
        
        # Should be JSON serializable
        try:
            json_str = json.dumps(error_dict)
            reconstructed = json.loads(json_str)
            assert reconstructed == error_dict
        except (TypeError, ValueError) as e:
            pytest.fail(f"Exception serialization is not JSON compatible: {e}")


class TestRealWorldScenarios:
    """Test real-world error scenarios."""
    
    def test_nested_error_chain(self):
        """Test handling of nested error chains."""
        # Simulate a chain of errors
        root_cause = ConnectionError("Network unreachable")
        db_error = wrap_database_error(root_cause, "connect", database_name="primary")
        
        # This would happen in higher-level code
        final_error = RAGSearchError(
            "Search failed due to database issues",
            search_type="vector_search",
            query="user query",
            original_error=db_error
        )
        
        # Should maintain the full error chain
        assert isinstance(final_error.original_error, RAGDatabaseError)
        assert isinstance(final_error.original_error.original_error, ConnectionError)
        assert "Network unreachable" in str(final_error)
    
    def test_error_context_for_debugging(self):
        """Test that errors provide sufficient context for debugging."""
        error = RAGGenerationError(
            "LLM generation failed",
            api_provider="anthropic",
            model_name="claude-3",
            context_length=8000,
            context={
                "request_id": "req_123",
                "user_id": "user_456",
                "prompt_length": 500,
                "max_tokens": 1000,
                "temperature": 0.7
            }
        )
        
        error_dict = error.to_dict()
        
        # Should have enough context for debugging
        context = error_dict["context"]
        assert "api_provider" in context
        assert "model_name" in context
        assert "context_length" in context
        assert "request_id" in context
        assert "user_id" in context
        assert "prompt_length" in context
        assert "max_tokens" in context
        assert "temperature" in context
    
    def test_graceful_degradation_pattern(self):
        """Test error handling pattern for graceful degradation."""
        def mock_operation_that_might_fail():
            raise RAGEmbeddingError("Embedding service unavailable")
        
        # Pattern: Try operation, catch specific error, provide fallback
        try:
            result = mock_operation_that_might_fail()
        except RAGEmbeddingError as e:
            # Log the error and provide fallback
            assert "Embedding service unavailable" in str(e)
            result = "fallback_result"
        
        assert result == "fallback_result"