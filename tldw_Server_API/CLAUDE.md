# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

tldw_server (Too Long; Didn't Watch Server) is a FastAPI-based media processing and summarization server that handles audio/video transcription, document processing, and AI-powered chat capabilities with RAG support.

## Development Commands

### Running the Server
```bash
# Start the development server with hot reload
python -m uvicorn tldw_Server_API.app.main:app --reload

# Access the API documentation at http://127.0.0.1:8000/docs
```

### Running Tests
```bash
# Install test dependencies (if not already installed)
pip install pytest httpx

# Run all tests
python -m pytest -v

# Run specific test file
python -m pytest tests/Chat/test_chat_endpoint.py -v

# Run tests in a specific directory
python -m pytest tests/Media_Ingestion_Modification/ -v
```

### Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt
```

## Architecture & Code Organization

### Core Architecture Pattern
The codebase follows a layered architecture with clear separation of concerns:

1. **API Layer** (`/app/api/v1/`) - FastAPI endpoints and request/response schemas
2. **Core Business Logic** (`/app/core/`) - Feature-specific libraries
3. **Services** (`/app/services/`) - Background processing and long-running tasks
4. **Database Layer** (`/app/core/DB_Management/`) - Database abstraction and management

### Adding New Features

When implementing new functionality:
1. Write core logic in `/app/core/<feature_name>/`
2. Create API endpoint in `/app/api/v1/endpoints/<endpoint_name>.py`
3. Define request/response schemas in `/app/api/v1/schemas/`
4. Register the route in `main.py`
5. Write tests in `/tests/<feature_name>/`

### Key Architectural Components

- **Media Processing Pipeline**: Handles multiple formats (audio, video, PDF, EPUB, etc.) through `/app/core/Ingestion_Media_Processing/`
- **LLM Integration**: Supports multiple providers (OpenAI, Anthropic, local models) via `/app/core/LLM_Calls/`
- **Vector Storage**: ChromaDB integration for embeddings and RAG in `/app/core/Embeddings/`
- **Authentication**: JWT-based auth system in `/app/core/AuthNZ/`
- **Database**: Supports SQLite (default), Elasticsearch, and ChromaDB

### Important Implementation Notes

- API provider keys and validation are defined in `/app/api/v1/schemas/chat_request_schemas.py`
- Use Loguru for logging throughout the application
- Rate limiting is implemented via slowapi
- All media processing should handle errors gracefully and provide meaningful error messages
- Database operations should use the abstraction layer in `/app/core/DB_Management/`

### Testing Strategy

- Unit tests for individual components in `/app/core/`
- Integration tests for API endpoints
- Use pytest fixtures for database and dependency injection
- Mock external services (LLMs, transcription services) in tests
- Test files should mirror the source code structure

### Common Development Patterns

- Use Pydantic models for all API request/response validation
- Implement background tasks using FastAPI's background tasks or the services layer
- Handle file uploads through the ingestion pipeline, not directly in endpoints
- Use dependency injection for database connections and service instances
- Follow the existing error handling patterns with proper HTTP status codes