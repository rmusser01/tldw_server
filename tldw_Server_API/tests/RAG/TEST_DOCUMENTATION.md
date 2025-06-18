# RAG Integration Test Documentation

## Overview

This document describes the comprehensive test suite for the new modular RAG (Retrieval-Augmented Generation) service integration in the tldw_server application.

## Test Structure

### 1. Unit Tests

#### `test_rag_endpoints_unit.py`
Tests individual endpoint functions in isolation with mocked dependencies.

**Coverage:**
- `get_rag_service_for_user()` - User-specific service creation and caching
- `perform_search()` - Search endpoint logic
- `run_retrieval_agent()` - Agent endpoint logic

**Key Test Cases:**
- Service creation for new users
- Service caching per user
- Configuration application from settings
- Search with various parameters (filters, date ranges, databases)
- RAG generation with streaming and non-streaming modes
- Conversation history handling
- Error handling and validation

#### `test_rag_service_unit.py`
Tests the RAGService integration class methods.

**Coverage:**
- Service initialization with different configurations
- Retriever setup (Media DB, ChaChaNotes DB, Vector DB)
- Processor and generator configuration
- Search and generation methods
- Utility methods (embeddings, stats, cache)

### 2. Integration Tests

#### `test_rag_endpoints_integration.py`
Tests the full integration of endpoints with real database connections.

**Coverage:**
- End-to-end search across multiple databases
- RAG generation with actual retrieval
- Streaming response handling
- Multi-user isolation
- Database error scenarios

**Test Fixtures:**
- Sample media items with diverse content
- Notes and character cards
- Conversation histories
- Test users with different permissions

### 3. Property-Based Tests

#### `test_rag_endpoints_property.py`
Uses Hypothesis to generate test cases and verify system properties.

**Strategies:**
- Valid search queries (words, phrases, questions, keywords)
- Search filters with various combinations
- Date ranges with proper ordering
- Conversation histories with valid structure

**Properties Tested:**
- Search request validation invariants
- Conversation structure (alternating user/assistant)
- Pagination bounds
- Cache size limits
- State machine properties for RAG interactions

### 4. Test Fixtures (`conftest.py`)

Provides shared fixtures and utilities for all tests:

**User Fixtures:**
- `test_user` - Single test user
- `test_users` - Multiple users for multi-tenancy tests

**Database Fixtures:**
- `mock_media_db` - Media database with search capabilities
- `mock_chacha_db` - ChaChaNotes database with notes/characters
- `temp_db_dir` - Temporary directory for test databases

**RAG Fixtures:**
- `mock_rag_config` - Test-friendly RAG configuration
- `mock_rag_service` - Async mock of RAG service
- `mock_llm_handler` - LLM handler for generation tests

**Data Generators:**
- `sample_media_items` - Generate test media documents
- `sample_notes` - Generate test notes
- `sample_conversations` - Generate test chat histories
- `create_test_embeddings` - Create deterministic embeddings

## Running the Tests

### Run All RAG Tests
```bash
python -m pytest tests/RAG/ -v
```

### Run Specific Test Types
```bash
# Unit tests only
python -m pytest tests/RAG/test_rag_*_unit.py -v

# Integration tests only
python -m pytest tests/RAG/test_rag_*_integration.py -v

# Property-based tests only
python -m pytest tests/RAG/test_rag_*_property.py -v
```

### Run with Coverage
```bash
python -m pytest tests/RAG/ --cov=app.api.v1.endpoints.rag --cov=app.core.RAG.rag_service --cov-report=html
```

### Run Performance Tests
```bash
python -m pytest tests/RAG/ -v -m performance --durations=10
```

## Test Scenarios

### Multi-User Scenarios
1. **User Isolation**: Each user gets separate RAG service instance
2. **Database Isolation**: User-specific database paths
3. **Cache Isolation**: Per-user result caching
4. **Concurrent Access**: Multiple users accessing simultaneously

### Search Scenarios
1. **Basic Search**: Simple keyword search
2. **Advanced Search**: With filters, date ranges, specific databases
3. **Hybrid Search**: Combining keyword and semantic search
4. **Empty Results**: Handling no matches gracefully
5. **Large Result Sets**: Pagination and limits

### Generation Scenarios
1. **Simple Q&A**: Basic question answering
2. **Research Mode**: Different source selection
3. **Streaming**: Server-sent events for progressive responses
4. **Context Handling**: Managing conversation history
5. **Citation Generation**: Proper source attribution

### Error Scenarios
1. **Database Connection Failures**
2. **Invalid Parameters**
3. **Service Initialization Errors**
4. **LLM API Failures**
5. **Timeout Handling**

## Performance Considerations

### Caching Strategy
- Services cached per user (memory consideration)
- Search results cacheable (configurable TTL)
- Embeddings cached persistently

### Resource Management
- Database connection pooling per user
- Async operations for non-blocking behavior
- Streaming for large responses

### Scaling Considerations
- Service cache size limits
- Database connection limits
- Memory usage per user

## Continuous Integration

### Pre-commit Checks
```bash
# Run fast unit tests
python -m pytest tests/RAG/test_*_unit.py -v --maxfail=1

# Check test coverage
python -m pytest tests/RAG/ --cov=app.api.v1.endpoints.rag --cov-fail-under=80
```

### CI Pipeline
1. Unit tests (fast, isolated)
2. Integration tests (with test databases)
3. Property-based tests (thorough edge cases)
4. Performance benchmarks
5. Coverage report generation

## Debugging Tests

### Enable Detailed Logging
```python
# In test file
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Use Test Fixtures Interactively
```python
# In Python REPL
from tests.RAG.conftest import *
service = mock_rag_service()
result = await service.search("test")
```

### Inspect Test Database
```python
# After test run with --no-cleanup
import sqlite3
conn = sqlite3.connect("test_db_path/media.sqlite")
# Inspect tables and data
```

## Best Practices

1. **Test Independence**: Each test should be independent
2. **Fixture Reuse**: Use conftest fixtures for common setup
3. **Async Testing**: Use `pytest.mark.asyncio` for async tests
4. **Mock External Services**: Don't call real APIs in tests
5. **Clear Test Names**: Describe what is being tested
6. **Arrange-Act-Assert**: Follow AAA pattern
7. **Property Tests**: Think about invariants, not just examples

## Future Improvements

1. **Load Testing**: Add performance benchmarks under load
2. **Fault Injection**: Test resilience to failures
3. **Security Testing**: Auth/authz edge cases
4. **Compatibility Testing**: Test with different database versions
5. **End-to-End Tests**: Full user journey tests

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PYTHONPATH includes project root
2. **Async Warnings**: Use `pytest-asyncio` plugin
3. **Database Lock**: Clean up test databases between runs
4. **Memory Issues**: Limit property test examples
5. **Flaky Tests**: Add retries for network-dependent tests

### Debug Commands
```bash
# Run single test with output
python -m pytest tests/RAG/test_file.py::TestClass::test_method -v -s

# Run with debugger
python -m pytest tests/RAG/test_file.py --pdb

# Show test durations
python -m pytest tests/RAG/ --durations=10
```