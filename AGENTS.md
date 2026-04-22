# AGENTS.md - tldw_server Project Guide

This file provides concise guidance to coding agents working with the tldw_server codebase.

## Project Overview

**tldw_server** (Too Long; Didn't Watch Server) is an ambitious open-source project building a comprehensive research assistant and media analysis platform. It's designed to help users ingest, transcribe, analyze, and interact with various media formats through both a powerful API and a web interface.

### Project Vision
The long-term goal is to create something akin to "The Young Lady's Illustrated Primer" from Neal Stephenson's "The Diamond Age" - a personal knowledge assistant that helps users learn and research at their own pace. While acknowledging the inherent difficulties in replicating such a device, this project serves as a practical step toward that vision.

### Current Status (v0.1.0)
The project is a FastAPI-first backend with a Next.js WebUI, mature AuthNZ (single-user API key and multi-user JWT modes), unified RAG and Evaluation modules, OpenAI-compatible Chat and Audio APIs (including real-time streaming transcription), and a production-grade MCP Unified module. The previous Gradio UI is deprecated.

## Repository Structure

```
<repo_root>/
├── tldw_Server_API/              # Main API server implementation
│   ├── app/
│   │   ├── api/v1/
│   │   │   ├── endpoints/        # All REST endpoints (media, chat, audio, rag, evals, etc.)
│   │   │   ├── schemas/          # Pydantic models
│   │   │   └── API_Deps/         # Shared dependencies (auth, DB, rate limits)
│   │   ├── core/                 # Core business logic (AuthNZ, RAG, LLM, DB, TTS, MCP, etc.)
│   │   ├── services/             # Background services
│   │   └── main.py               # FastAPI entry point
│   ├── Config_Files/             # config.txt, example YAMLs, migration helpers
│   ├── Databases/                # Default DBs (runtime data; some are gitignored)
│   └── tests/                    # Pytest suite
├── Dockerfiles/                  # Docker images and compose files
├── Docs/                         # Documentation (API, Development, RAG, AuthNZ, TTS, etc.)
├── Helper_Scripts/               # Utilities (installers, prompt tools, doc generators)
├── mock_openai_server/           # Mock server for OpenAI-compatible API tests
├── apps/tldw-frontend/                # Next.js WebUI (primary web client)
├── Databases/                    # DBs (AuthNZ defaults here; Media DB is per-user under user_databases/)
├── models/                       # Optional model assets (if used)
├── pyproject.toml                # Project configuration
├── README.md                     # Project README
└── Project_Guidelines.md         # Development philosophy
```

## Core Features

### Implemented Features
1. **Media Ingestion & Processing**
   - Supports: Video, Audio, PDF, EPUB, DOCX, HTML, Markdown, XML, MediaWiki dumps
   - yt-dlp for video/audio downloads from 1000+ sites
   - Automatic metadata extraction and storage

2. **Audio STT, TTS & Analysis**
   - Transcription: faster_whisper, NVIDIA NeMo (Parakeet, Canary), Qwen2Audio
   - Real-time streaming transcription over WebSocket
   - Text-to-speech (OpenAI-compatible TTS + local Kokoro ONNX)
   - Chunked processing for long-form content; optional diarization

3. **Search & Retrieval (RAG)**
   - Full-text search using SQLite FTS5
   - Vector embeddings with ChromaDB
   - BM25 + vector search + re-ranking pipeline
   - Contextual retrieval for improved accuracy

4. **Chat & Interaction**
   - OpenAI-compatible Chat API (`/chat/completions`)
   - 16+ LLM providers (commercial & local)
   - Character cards (SillyTavern-compatible) + character chat sessions
   - Chat history management and search

5. **Knowledge Management**
   - Note-taking system (notebook-style)
   - Prompt library with import/export
   - Tagging and categorization
   - Soft delete with recovery options

6. **API Providers Supported**
   - **Commercial**: OpenAI, Anthropic, Cohere, DeepSeek, Google, Groq, HuggingFace, Mistral, OpenRouter, Qwen, Moonshot, Z.AI
   - **Local**: Llama.cpp, Kobold.cpp, Oobabooga, TabbyAPI, vLLM, Ollama, Aphrodite, Custom OpenAI-compatible

7. **MCP Unified**
   - Production-ready Model Context Protocol implementation with JWT/RBAC
   - Status, metrics, tool execution endpoints; WebSocket support

8. **Prompt Studio & Chatbooks**
   - Prompt Studio endpoints for projects, prompts, tests, and optimization
   - Chatbooks for export/import and background job handling

### Work-in-Progress Features
- Browser extension for web content capture
- Selected writing assistance tools
- Additional research providers (beyond current arXiv/web scraping)

## Technical Architecture

### Database Design
- **SQLite Databases (default)**:
  - Content (Media DB v2): per-user `Databases/user_databases/<user_id>/Media_DB_v2.db` (root-level path deprecated)
  - AuthNZ users: `Databases/users.db` (SQLite by default; PostgreSQL supported)
  - Evaluations: `Databases/evaluations.db`
  - Notes/Chats: per-user `Databases/user_databases/<user_id>/ChaChaNotes.db`
  - Implements soft deletes, versioning, and sync logging

- **Vector Storage**:
  - ChromaDB for embeddings (configurable providers/models)

### API Design
- RESTful API following OpenAPI 3.0
- Consistent endpoint naming: `/api/v1/{resource}/{action}`
- Pydantic models for request/response validation
- Comprehensive error handling with meaningful messages

### Key Technologies
- **Backend**: FastAPI, SQLite/PostgreSQL, ChromaDB
- **ML/AI**: faster_whisper, NeMo (Parakeet/Canary), Qwen2Audio, sentence-transformers
- **Audio/Video**: ffmpeg, yt-dlp
- **Document Processing**: pymupdf, docling, ebooklib, pandoc
- **Testing**: pytest, httpx
- **Logging**: loguru

### Key Architectural Components

- **Media Processing**
  - `/app/core/Ingestion_Media_Processing/` for ingestion, chunking, conversion

- **LLM Integration**
  - `/app/core/LLM_Calls/` unified interface with streaming responses

- **Embeddings**
  - `/app/core/Embeddings/` with ChromaDB integration and batching

- **Authentication (AuthNZ)**
  - `/app/core/AuthNZ/` single-user (X-API-KEY) and multi-user (JWT) modes

- **RAG Service**
  - `/app/core/RAG/` unified RAG pipeline (hybrid FTS5 + vector + rerank)

- **Audio & TTS**
  - `/app/api/v1/endpoints/audio.py` and `/app/core/TTS/` for STT/TTS and streaming

- **MCP Unified**
  - `/app/core/MCP_unified/` production-ready MCP server + endpoints

## Scheduler vs Jobs Decision Guide

Use **Jobs** when:
- The work is user-facing or needs admin controls like pause/resume/drain, retries, quotas, or RLS.
- You need stable API endpoints for status/summary and cross-domain queues.
- You want worker processes using the Jobs `WorkerSDK`.

Use **Scheduler** when:
- The work is internal orchestration with task dependencies, idempotency keys, and handler registration.
- You want tasks registered via the `@task` decorator and executed by the core Scheduler worker pool.
- The feature fits the existing Workflows or Watchlists model.

Recurring schedules:
- Use APScheduler services to enqueue into whichever backend you chose.
- Workflows uses APScheduler → Scheduler; Reading Digest uses APScheduler → Jobs.

Default choice:
- Use **Jobs** for new user-visible features or anything that needs admin/ops visibility.
- Use **Scheduler** for internal system orchestration where dependency handling is central.

## Development Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use type hints for function parameters and returns
- Implement comprehensive docstrings for modules, classes, and functions
- Prefer async/await for I/O operations

### Adding New Features
1. **Design First**: Create a design document in `/Docs/Design/`
2. **Core Implementation**: Add business logic to `/app/core/{feature}/`
3. **API Endpoint**: Create endpoint in `/app/api/v1/endpoints/`
4. **Schemas**: Define Pydantic models in `/app/api/v1/schemas/`
5. **Tests**: Write comprehensive tests in `/tests/{feature}/`
6. **Documentation**: Update relevant documentation

### Common Development Patterns
- **Pydantic Models**: Use for all API request/response validation
- **Dependency Injection**: For database connections and service instances
- **Background Tasks**: Use FastAPI's background tasks or services layer
- **Streaming**: Support streaming responses where applicable
- **Error Responses**: Follow consistent HTTP status codes and error formats
- **Async/Await**: Use async patterns for I/O operations

### Important Implementation Notes
- **Logging**: Use Loguru throughout (`from loguru import logger`)
- **Error Handling**: Graceful errors with meaningful messages
- **Rate Limiting**: Present across modules (embeddings: slowapi; chat/evals: module rate limiters)
- **Database Operations**: Use `/app/core/DB_Management/` abstractions (no raw SQL outside)
- **File Handling**: Route uploads through ingestion pipeline
- **Secrets**: Prefer `.env` for API keys; config.txt still supported; never log secrets

### Testing Requirements
- All new functionality must include unit, integration, and property-based tests where applicable
- Write unit tests for all new functions
- Include integration tests for API endpoints
- Use pytest fixtures for common test data
- Mock external services (LLMs, APIs)
- Aim for >80% code coverage

### Testing Strategy
- **Test Structure**: Tests mirror source code structure
- **Test Types**:
  - Unit tests for individual components
  - Integration tests for API endpoints
  - Property-based tests for complex logic
  - (Property-based tests use frameworks like Hypothesis to validate invariants across many generated inputs; use for algorithms, parsers, and stateful systems.)
- **Fixtures**: Use pytest fixtures for database and dependency injection
- **Mocking**: Mock external services (LLMs, transcription services)
- **Test Markers**: `unit`, `integration`, `external_api`, `local_llm_service`

### Error Handling
- Use custom exceptions in `/app/core/exceptions.py`
- Return appropriate HTTP status codes
- Provide meaningful error messages
- Log errors with context using loguru

### Security Best Practices
- Validate all user input
- Use parameterized queries for database operations
- Never log sensitive information (API keys, passwords)
- Implement rate limiting for API endpoints
- Validate file uploads (type, size, content)
- Configure CORS in `main.py` for production deployments

## Configuration

### Configuration Files
- `tldw_Server_API/Config_Files/config.txt`: Main configuration (provider settings)
- `.env`: AuthNZ and sensitive keys (migrate helpers in `Config_Files/`)
- `mediawiki_import_config.yaml`: MediaWiki import settings
- Environment variables override file settings
- Database location configurable via `DATABASE_URL` (AuthNZ) and config helpers

### Required Setup
1. **Activate venv first**: `source .venv/bin/activate` (do this before running `python`, `pip`, or `pytest` commands)
2. **Dependencies**: `pip install -e .` (add extras as needed, e.g., `.[dev]`, `.[multiplayer]`)
3. **FFmpeg**: Required for audio/video processing
4. **Auth Setup**: `cp tldw_Server_API/Config_Files/.env.example tldw_Server_API/Config_Files/.env && python -m tldw_Server_API.app.core.AuthNZ.initialize`
5. **Provider Keys**: Add to `.env` or `Config_Files/config.txt`
6. **Optional**: CUDA for accelerated STT

## Common Tasks

### Starting the Server
```bash
python -m uvicorn tldw_Server_API.app.main:app --reload
# API docs:   http://127.0.0.1:8000/docs
# Quickstart: http://127.0.0.1:8000/api/v1/config/quickstart
# Setup UI:   http://127.0.0.1:8000/setup (if required)
```

### Running Tests
```bash
# Activate project virtual environment first
source .venv/bin/activate

# All tests (from repo root)
python -m pytest -v

# With coverage
python -m pytest --cov=tldw_Server_API --cov-report=term-missing

# Run tests with markers
python -m pytest -m "unit" -v
python -m pytest -m "integration" -v
```

### AuthNZ PostgreSQL Fixture
- `tldw_Server_API/tests/AuthNZ/conftest.py` provisions a per-test Postgres database via the `isolated_test_environment` fixture.
- Tests auto-start a local Dockerized Postgres unless `TLDW_TEST_NO_DOCKER=1`.
- Provide `TEST_DATABASE_URL` (or related `TEST_DB_*` vars) to reuse an existing cluster.
- Skip Postgres-dependent tests only when the fixture reports Postgres unavailable; never roll your own database setup.

### Database Operations
```python
# Use the MediaDatabase class for all operations
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase

db = MediaDatabase(db_path="path/to/media.db", client_id="api_client")
# Always use context managers for transactions
```

### Adding a New LLM Provider
1. Add provider configuration to `Config_Files/config.txt` (or `.env`)
2. Implement provider in `/app/core/LLM_Calls/`
3. Register in chat schemas and provider manager
4. Add tests; update docs and examples

## Performance & Deployment

### Performance Optimization
- **Database**: Use indexes, FTS5 for search
- **Chunking**: Process large files in chunks
- **Caching**: Implement caching for expensive operations
- **Connection Pooling**: For database connections
- **Async Operations**: For I/O-bound tasks

### Deployment Considerations
- **Docker**: See `Dockerfiles/`
- **Environment**: Linux, macOS, Windows supported
- **Auth Modes**: Single-user (X-API-KEY) and multi-user (JWT)
- **Backup**: Built-in DB backup/exports (Chatbooks)
- **CORS**: Configured in `main.py`; adjust for production

## Debugging Tips

### Common Issues
1. **Import Errors**: Check PYTHONPATH and virtual environment
2. **Database Locks**: Ensure proper connection management
3. **Transcription Failures**: Verify ffmpeg installation and CUDA setup
4. **API Key Errors**: Check config.txt formatting

### Logging
- Loguru with color-coded output
- Startup displays auth mode and URLs; single-user prints API key
- Check logs for stack traces and rate-limit messages

## Project Philosophy

The project follows these core principles (from Project_Guidelines.md):
1. Keep the project actively developed with clear progress
2. Be respectful to all contributors and users
3. Remain open to criticism and new ideas
4. Balance expertise with newcomer perspectives
5. Be kind and answer questions
6. Acknowledge contributions
7. Compensate significant contributions when possible

## Important Notes

### Licensing
- GNU General Public License v2.0 (see README)

### Privacy & Security
- Designed for local/self-hosted deployment
- No telemetry or data collection
- Users own and control their data

### Contributing
- Follow the existing code patterns
- Write tests for new features
- Update documentation
- Be respectful in discussions

## Quick Reference

### Key Endpoints
- `POST /api/v1/media/process`         - Ingest and process media
- `GET  /api/v1/media/search`          - Search ingested content
- `POST /api/v1/chat/completions`      - OpenAI-compatible chat
- `POST /api/v1/embeddings`            - OpenAI-compatible embeddings
- `POST /api/v1/rag/search`            - Unified RAG search
- `POST /api/v1/research/websearch`    - Web search (multi-provider) with optional aggregation
- `POST /api/v1/evaluations/...`       - Unified evaluation API (geval, rag, batch, metrics)
- `GET  /api/v1/llm/providers`         - List configured LLM providers
- `WS   /api/v1/audio/stream/transcribe` - Real-time audio transcription
- `POST /api/v1/audio/transcriptions`  - File-based transcription (OpenAI compatible)
- `POST /api/v1/audio/speech`          - TTS (streaming and non-streaming)
- `GET  /api/v1/audio/voices/catalog`  - TTS voice catalog across providers
- `GET  /api/v1/mcp/status`            - MCP server status
- `POST /api/v1/chatbooks/export`      - Export content to chatbook
- `POST /api/v1/chatbooks/import`      - Import chatbook

### Environment Variables
- `AUTH_MODE`            : `single_user` or `multi_user`
- `SINGLE_USER_API_KEY`  : API key for single-user mode
- `DATABASE_URL`         : AuthNZ DB URL (e.g., `sqlite:///./Databases/users.db`)
- `OPENAI_API_KEY`       : OpenAI API key (or in config.txt)
- `ANTHROPIC_API_KEY`    : Anthropic API key (or in config.txt)
- Provider-specific vars : As needed by your configured providers

### Useful Commands
```bash
# Activate project virtual environment first
source .venv/bin/activate

# Run specific test markers
python -m pytest -m "unit" -v
python -m pytest -m "integration" -v

# Check coverage
python -m pytest --cov=tldw_Server_API --cov-report=term-missing

# Optional formatting/type-checking
black tldw_Server_API/             # if black is installed
mypy tldw_Server_API/              # if mypy is installed
```

---

This guide is maintained to help coding agents understand the project structure, conventions, and best practices. When in doubt, look at existing code patterns, `main.py`, and tests for guidance.

# Development Guidelines

## Philosophy

### Core Beliefs

- **Incremental progress over big bangs** - Small changes that compile and pass tests
- **Learning from existing code** - Study and plan before implementing
- **Pragmatic over dogmatic** - Adapt to project reality
- **Clear intent over clever code** - Be boring and obvious

### Simplicity Means

- Single responsibility per function/class
- Avoid premature abstractions
- No clever tricks - choose the boring solution
- If you need to explain it, it's too complex

## Process

### 1. Planning & Staging

Break complex work into 3-5 stages. Document in a uniquely named plan file for the specific task (avoid generic names like `IMPLEMENTATION.md` or `IMPLEMENTATION_PLAN.md`), for example `IMPLEMENTATION_PLAN_<short_task_slug>.md` (e.g., `IMPLEMENTATION_PLAN_feedback_system.md`, `IMPLEMENTATION_PLAN_auth_refactor.md`):

```markdown
## Stage N: [Name]
**Goal**: [Specific deliverable]
**Success Criteria**: [Testable outcomes]
**Tests**: [Specific test cases]
**Status**: [Not Started|In Progress|Complete]
```
- Update status as you progress
- Remove only your own plan file when all stages are done; never delete other agents' plans

### 2. Implementation Flow

1. **Understand** - Study existing patterns in codebase
2. **Test** - Write test first (red)
3. **Implement** - Minimal code to pass (green)
4. **Refactor** - Clean up with tests passing
5. **Commit** - With clear message linking to plan

### 3. When Stuck (After 3 Attempts)

**CRITICAL**: Maximum 3 attempts per issue, then STOP.

1. **Document what failed**:
   - What you tried
   - Specific error messages
   - Why you think it failed

2. **Research alternatives**:
   - Find 2-3 similar implementations
   - Note different approaches used

3. **Question fundamentals**:
   - Is this the right abstraction level?
   - Can this be split into smaller problems?
   - Is there a simpler approach entirely?

4. **Try different angle**:
   - Different library/framework feature?
   - Different architectural pattern?
   - Remove abstraction instead of adding?

## Technical Standards

### Architecture Principles

- **Composition over inheritance** - Use dependency injection
- **Interfaces over singletons** - Enable testing and flexibility
- **Explicit over implicit** - Clear data flow and dependencies
- **Test-driven when possible** - Never disable tests, fix them

### Code Quality

- **Every commit must**:
  - Compile successfully
  - Pass all existing tests
  - Include tests for new functionality
  - Follow project formatting/linting

- **Before committing**:
  - Run formatters/linters
  - Self-review changes
  - Ensure commit message explains "why"

### Error Handling

- Fail fast with descriptive messages
- Include context for debugging
- Handle errors at appropriate level
- Never silently swallow exceptions

## Decision Framework

When multiple valid approaches exist, choose based on:

1. **Testability** - Can I easily test this?
2. **Readability** - Will someone understand this in 6 months?
3. **Consistency** - Does this match project patterns?
4. **Simplicity** - Is this the simplest solution that works?
5. **Reversibility** - How hard to change later?

## Project Integration

### Learning the Codebase

- Find 3 similar features/components
- Identify common patterns and conventions
- Use same libraries/utilities when possible
- Follow existing test patterns

### Tooling

- Use project's existing build system
- Use project's test framework
- Use project's formatter/linter settings
- Don't introduce new tools without strong justification

## Quality Gates

### Definition of Done

- [ ] Tests written and passing
- [ ] Code follows project conventions
- [ ] No linter/formatter warnings
- [ ] No new security findings introduced in touched code
- [ ] Commit messages are clear
- [ ] Implementation matches plan
- [ ] AI-generated PRs include a human-written `Change summary` that explains both what changed and why those implementation choices were made
- [ ] No TODOs without issue numbers

### AI-Generated PR Merge Gate

- For PRs materially authored by AI, merge is blocked unless the human requester writes a `Change summary`.
- That summary must explain both what changed and why those specific implementation choices were made.
- A diff recap or pasted AI text without clear human ownership does not satisfy this gate.
- If the human requester cannot explain the rationale in their own words, the PR is not merge-ready.
- Canonical policy: `Docs/superpowers/AI_GENERATED_PR_CHANGE_SUMMARY_POLICY_2026_04_17.md`

### Security Validation

- Run Bandit on the touched scope before considering work complete.
- Use the project virtual environment when running Bandit.
- Fix new findings in changed code before finishing; do not defer silently.
- Recommended command pattern:
  `source .venv/bin/activate && python -m bandit -r <touched_paths> -f json -o /tmp/bandit_<task>.json`

### Test Guidelines

- Test behavior, not implementation
- One assertion per test when possible
- Clear test names describing scenario
- Use existing test utilities/helpers
- Tests should be deterministic

## Important Reminders

**NEVER**:
- Use `--no-verify` to bypass commit hooks
- Disable tests instead of fixing them
- Commit code that doesn't compile
- Make assumptions - verify with existing code

**ALWAYS**:
- Commit working code incrementally
- Update plan documentation as you go
- Learn from existing implementations
- Stop after 3 failed attempts and reassess
