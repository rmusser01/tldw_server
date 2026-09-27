"""
Pytest configuration for Evaluations module tests.

This file provides shared fixtures and configuration for all test types:
- Unit tests (with minimal mocking for external services only)
- Integration tests (no mocking, real components)
- Property tests (no mocking, generated test data)
"""

import asyncio
import tempfile
import sqlite3
import sys
from pathlib import Path
from typing import Generator, AsyncGenerator, Dict, Any
from datetime import datetime
import pytest
import os
import pytest_asyncio
from aiohttp import web


# Import application components
from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager
from tldw_Server_API.app.core.Evaluations.rag_evaluator import RAGEvaluator
from tldw_Server_API.app.core.Evaluations.response_quality_evaluator import ResponseQualityEvaluator
from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import UnifiedEvaluationService
from tldw_Server_API.app.core.Evaluations.connection_pool import ConnectionPool
from tldw_Server_API.app.core.Evaluations.circuit_breaker import CircuitBreaker


# ============================================================================
# Test Markers
# ============================================================================

def pytest_configure(config):

    """Register custom markers."""
    # Ensure TEST_MODE is enabled for the Evaluations test suite to bypass
    # global API rate limiting paths that are unrelated to unit correctness.
    os.environ.setdefault("TEST_MODE", "true")
    # Set deterministic, low rate limits for integration tests
    os.environ.setdefault("TEST_EVALUATIONS_RATE_LIMIT", "2")
    os.environ.setdefault("TEST_EVALUATIONS_RATE_WINDOW_MINUTES", "1")
    config.addinivalue_line("markers", "unit: Unit tests with minimal mocking")
    config.addinivalue_line("markers", "integration: Integration tests with real components")
    config.addinivalue_line("markers", "property: Property-based tests with generated data")
    config.addinivalue_line("markers", "slow: Tests that take >1 second")
    config.addinivalue_line("markers", "requires_llm: Tests requiring LLM API access")
    config.addinivalue_line("markers", "requires_embeddings: Tests requiring embedding API access")


@pytest.fixture(scope="session")
def _evaluations_authnz_db_url(tmp_path_factory) -> str:
    return f"sqlite:///{tmp_path_factory.mktemp('evals_authnz') / 'users.db'}"


@pytest.fixture(autouse=True)
def _isolate_authnz_daily_ledger(monkeypatch, _evaluations_authnz_db_url):
    """Keep the suite out of the checkout's Databases/users.db (TASK-13368).

    The root conftest defaults DATABASE_URL to the checkout's users.db, so the
    evaluations limiter's ResourceDailyLedger spent real daily quota on every local
    run. Point AuthNZ at a session-scoped temp DB, and rebuild the cached settings,
    pool and ledger only when something (another suite, or a test resetting the
    pool) left them pointing elsewhere.
    """
    from tldw_Server_API.app.core.AuthNZ import database as authnz_db
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    from tldw_Server_API.app.core.Evaluations import user_rate_limiter

    url = _evaluations_authnz_db_url
    monkeypatch.setenv("DATABASE_URL", url)
    pool = authnz_db._db_pool
    pool_url = getattr(getattr(pool, "settings", None), "DATABASE_URL", None)
    if pool is not None and pool_url != url:
        asyncio.run(authnz_db.reset_db_pool())
        user_rate_limiter._evals_daily_ledger = None
    elif get_settings().DATABASE_URL != url:
        reset_settings()
        user_rate_limiter._evals_daily_ledger = None
    yield


# ============================================================================
# Database Fixtures with Full Schema
# ============================================================================

# Removed apply_full_evaluation_schema function - no longer needed
# Tests now use EvaluationsDatabase class directly for proper schema initialization


@pytest.fixture(scope="function")
def temp_db_path() -> Generator[Path, None, None]:
    """Create a temporary database file with full schema for testing."""
    with tempfile.NamedTemporaryFile(suffix="_test_eval.db", delete=False) as f:
        db_path = Path(f.name)

    # Use the actual EvaluationsDatabase to initialize the schema
    from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase

    # This will create the database with the exact same schema as production
    eval_db = EvaluationsDatabase(str(db_path))

    # Also ensure evaluation_manager tables exist
    from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager
    # The manager will add any additional tables it needs
    manager = EvaluationManager()
    # Override the db path for the manager
    manager.db_path = str(db_path)
    manager._init_database()

    yield db_path

    # Cleanup - ensure all connections are closed before deleting
    try:
        # Force close any remaining connections
        import gc
        gc.collect()  # Force garbage collection of any remaining connections

        if db_path.exists():
            db_path.unlink()
    except Exception as e:
        # Log but don't fail if cleanup has issues
        import logging
        logging.warning(f"Failed to cleanup test database: {e}")


@pytest.fixture(scope="function")
def in_memory_db() -> Generator[sqlite3.Connection, None, None]:
    """Create an in-memory SQLite database with full schema for fast testing."""
    # For in-memory testing, we need to use a temp file approach
    # since EvaluationsDatabase expects a file path
    with tempfile.NamedTemporaryFile(suffix="_test_mem.db", delete=False) as f:
        db_path = Path(f.name)

    try:
        # Initialize using the production database class
        from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase
        eval_db = EvaluationsDatabase(str(db_path))

        # Now open a connection to use in tests
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        yield conn
        conn.close()
    finally:
        # Clean up temp file
        if db_path.exists():
            db_path.unlink()


# ============================================================================
# Mock OpenAI Server Fixture
# ============================================================================

@pytest.fixture(scope="session")
def mock_openai_server():
    """Start a mock OpenAI server for testing."""
    import subprocess
    import time
    import requests
    from pathlib import Path

    # Path to mock server
    mock_server_path = Path(__file__).parent.parent.parent.parent / "mock_openai_server"

    # Start the mock server in background. Avoid PIPEs (to prevent blocking if
    # not consumed) and run in a new session to simplify cleanup.
    process = subprocess.Popen(
        [sys.executable, "run_server.py"],
        cwd=str(mock_server_path),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Wait for server to start
    max_retries = 30
    # Configurable for slower CI/test hosts; defaults to a short local timeout.
    try:
        health_timeout_s = float(os.getenv("MOCK_OPENAI_HEALTH_TIMEOUT_SECONDS", "5"))
    except ValueError:
        health_timeout_s = 5.0
    for _ in range(max_retries):
        try:
            response = requests.get("http://localhost:8080/health", timeout=health_timeout_s)
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            _ = None
        time.sleep(0.5)
    else:
        process.terminate()
        raise RuntimeError("Mock OpenAI server failed to start")

    yield "http://localhost:8080"

    # Cleanup
    try:
        process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # Force kill if graceful termination takes too long
        process.kill()
        try:
            process.wait(timeout=3)
        except Exception:
            _ = None


@pytest.fixture(scope="function")
def mock_openai_config(mock_openai_server, monkeypatch):
    """Configure tests to use mock OpenAI server."""
    # Set environment variables to use mock server
    monkeypatch.setenv("OPENAI_API_BASE", mock_openai_server)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Patch the OpenAI base URL in the config
    import tldw_Server_API.app.core.config as config
    if hasattr(config, 'loaded_config_data'):
        if 'openai_api' in config.loaded_config_data:
            config.loaded_config_data['openai_api']['base_url'] = mock_openai_server
            config.loaded_config_data['openai_api']['api_key'] = "test-key"

    return mock_openai_server


# ============================================================================
# Rate Limiting Mock Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def mock_rate_limiter(monkeypatch):
    """Mock rate limiter to prevent 429 errors in tests.

    Use this fixture explicitly in tests that need to bypass rate limiting.
    """
    from unittest.mock import Mock, AsyncMock

    # Create a stub rate limiter that always allows calls
    mock_limiter = Mock()
    mock_limiter.check_rate_limit = AsyncMock(return_value=(True, {"remaining": 100, "reset_at": None}))
    mock_limiter.update_usage = AsyncMock()
    mock_limiter.get_usage_summary = AsyncMock(return_value={
        "tier": "test",
        "limits": {"evaluations_per_minute": 100},
        "usage": {"evaluations": 0},
        "reset_at": None
    })

    # Mock the get_rate_limiter_dep to return our mock
    async def mock_get_rate_limiter_dep():
        return mock_limiter

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.auth_deps.get_rate_limiter_dep",
        mock_get_rate_limiter_dep
    )

    # Also mock the actual check_evaluation_rate_limit function
    async def mock_check_evaluation_rate_limit(request, rate_limiter=None):
        # Always allow - no rate limiting in tests
        return None

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.evaluations_unified.check_evaluation_rate_limit",
        mock_check_evaluation_rate_limit
    )

    # Legacy evals and evals_openai endpoints removed; no additional patching required

    return mock_limiter


# ============================================================================
# Webhook Receiver Fixtures (real local HTTP servers)
# ============================================================================

@pytest_asyncio.fixture(scope="function")
async def webhook_receiver_server(monkeypatch):
    """Start a real local HTTP server to receive webhooks.

    Returns a dict with 'url' and 'received' list of captured requests.
    """
    # Ensure webhook delivery path awaits completions during tests
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "false")
    monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWED_PORTS", "*")
    monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWLIST", "127.0.0.1,localhost")
    monkeypatch.delenv("EGRESS_DENYLIST", raising=False)
    monkeypatch.delenv("WORKFLOWS_EGRESS_DENYLIST", raising=False)
    app = web.Application()
    received = []

    async def handle(request: web.Request):
        try:
            payload = await request.json()
        except Exception:
            payload = None
        received.append({
            "path": request.path,
            "headers": dict(request.headers),
            "json": payload,
            "body": await request.text()
        })
        return web.json_response({"ok": True})

    app.add_routes([web.post('/webhook', handle)])

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    # In restricted sandboxes, binding to localhost may be disallowed.
    # Skip tests gracefully in that case to avoid hard failures.
    try:
        await site.start()
    except (PermissionError, OSError) as e:
        import pytest
        await runner.cleanup()
        pytest.skip(f"Local socket binding not permitted in sandbox; skipping webhook tests ({e})")

    # Discover the bound port
    sockets = getattr(site, '_server').sockets  # type: ignore[attr-defined]
    port = sockets[0].getsockname()[1]
    url = f"http://127.0.0.1:{port}/webhook"

    # In some sandboxes, outbound connections to localhost are blocked.
    # Perform a quick connectivity check and skip if not reachable.
    try:
        import aiohttp
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1)) as session:
            try:
                async with session.post(url, json={"probe": True}) as _resp:
                    pass
            except Exception as e:
                import pytest
                await runner.cleanup()
                pytest.skip(f"Local HTTP connections blocked in sandbox; skipping webhook tests ({e})")
        # Clear the probe request from captured events
        received.clear()
    except Exception:
        # If aiohttp is not available or other import issues, proceed; tests may still pass.
        _ = None

    try:
        yield {"url": url, "received": received}
    finally:
        await runner.cleanup()


@pytest_asyncio.fixture(scope="function")
async def flaky_webhook_receiver_server(monkeypatch):
    """Local webhook receiver that fails the first two attempts (500), then succeeds.

    Useful for testing retry logic without mocks.
    """
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "false")
    monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWED_PORTS", "*")
    monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWLIST", "127.0.0.1,localhost")
    monkeypatch.delenv("EGRESS_DENYLIST", raising=False)
    monkeypatch.delenv("WORKFLOWS_EGRESS_DENYLIST", raising=False)
    app = web.Application()
    received = []
    call_count = {"n": 0}

    async def handle(request: web.Request):
        call_count["n"] += 1
        try:
            payload = await request.json()
        except Exception:
            payload = None
        received.append({
            "attempt": call_count["n"],
            "path": request.path,
            "headers": dict(request.headers),
            "json": payload,
            "body": await request.text()
        })
        # Fail first two attempts
        if call_count["n"] < 3:
            return web.Response(status=500, text="temporary failure")
        return web.json_response({"ok": True})

    app.add_routes([web.post('/webhook', handle)])

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    try:
        await site.start()
    except (PermissionError, OSError) as e:
        import pytest
        await runner.cleanup()
        pytest.skip(f"Local socket binding not permitted in sandbox; skipping webhook tests ({e})")

    sockets = getattr(site, '_server').sockets  # type: ignore[attr-defined]
    port = sockets[0].getsockname()[1]
    url = f"http://127.0.0.1:{port}/webhook"

    # Connectivity check for sandboxed environments
    try:
        import aiohttp
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1)) as session:
            try:
                async with session.post(url, json={"probe": True}) as _resp:
                    pass
            except Exception as e:
                import pytest
                await runner.cleanup()
                pytest.skip(f"Local HTTP connections blocked in sandbox; skipping webhook tests ({e})")
        received.clear()
    except Exception:
        _ = None

    try:
        yield {"url": url, "received": received}
    finally:
        await runner.cleanup()


# ============================================================================
# Mock Fixtures for External Services
# ============================================================================

@pytest.fixture(scope="function")
def mock_llm_analyze(monkeypatch):
    """Mock the analyze function to avoid real OpenAI API calls in unit tests.

    Use this fixture explicitly in unit tests that need to mock LLM calls.
    """

    # Create a mock analyze function that returns realistic scores
    # The actual signature: analyze(api_name, input_data, custom_prompt_arg, api_key="", system_message="", temp=0.1)
    def mock_analyze(api_name, input_data, custom_prompt_arg="", api_key="", system_message="", temp=0.1, **kwargs):
        """Mock analyze function that returns scores based on content."""
        # Check both input_data and custom_prompt_arg for keywords
        combined_text = f"{input_data} {custom_prompt_arg}".lower()

        # Return different scores based on the prompt content for variety
        if "relevance" in combined_text:
            return "4.3"
        elif "faithfulness" in combined_text:
            return "4.7"
        elif "similarity" in combined_text:
            return "0.85"
        elif "coherence" in combined_text:
            return "4.4"
        elif "quality" in combined_text:
            return "4.5"
        else:
            return "4.0"  # Default score

    # Patch the analyze function in the rag_evaluator module
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.rag_evaluator.analyze",
        mock_analyze
    )

    # Also patch it in response_quality_evaluator if it exists
    try:
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Evaluations.response_quality_evaluator.analyze",
            mock_analyze
        )
    except:
        _ = None  # Module might not import it

    return mock_analyze


@pytest.fixture(scope="function")
def local_embeddings_config():
    """Configure tests to use local embeddings model.

    Uses sentence-transformers with a small, fast model that doesn't require API keys.
    """
    return {
        "provider": "sentence_transformers",
        "model": "all-MiniLM-L6-v2",
        "api_key": None  # No API key needed for local models
    }


@pytest.fixture(scope="function")
def mock_embeddings(monkeypatch):
    """Mock embedding creation for unit tests that don't need real embeddings.

    Use this fixture explicitly in unit tests. For integration tests,
    use local_embeddings_config instead.
    """
    import numpy as np

    def mock_create_embedding(text, provider="openai", model="text-embedding-3-small", api_key=None):

        """Generate deterministic fake embeddings based on text hash."""
        # Generate a deterministic embedding based on the input text
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()  # nosec B324
        seed = int(text_hash[:8], 16)
        np.random.seed(seed)
        # Return 384 dimensions for all-MiniLM-L6-v2 compatibility
        return np.random.randn(384).tolist()

    # Patch the create_embedding function
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.rag_evaluator.create_embedding",
        mock_create_embedding
    )

    return mock_create_embedding

# ============================================================================
# Component Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def setup_auth_db(tmp_path):
    """Setup auth database for tests that need authentication."""
    # Import here to avoid circular imports
    from tldw_Server_API.app.core.AuthNZ.migrations import migration_001_create_users_table

    # Create a temporary auth database
    auth_db_path = tmp_path / "test_auth.db"

    # Initialize using the migration function
    conn = sqlite3.connect(str(auth_db_path))

    # Use the actual migration which creates proper SQLite schema
    migration_001_create_users_table(conn)

    # Note: The migration creates the users table with INTEGER for is_active
    # which is the correct SQLite type (not BOOLEAN)

    conn.commit()
    conn.close()

    yield auth_db_path

@pytest.fixture(scope="function")
def evaluation_manager(temp_db_path, monkeypatch) -> EvaluationManager:
    """Create an EvaluationManager instance with test database."""
    # Import here to avoid circular imports
    from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase

    # Initialize the database using the actual function from the module
    eval_db = EvaluationsDatabase(str(temp_db_path))
    # Note: Database is initialized in __init__, no need to call again

    # Ensure the manager uses our temp database, not production
    # Mock the config to return our temp path
    def mock_get_db_path(self, explicit_path=None):
        if explicit_path is not None:
            try:
                return Path(explicit_path)
            except Exception:
                return temp_db_path
        return temp_db_path

    # Now create the manager with mocked path
    monkeypatch.setattr(EvaluationManager, "_get_db_path", mock_get_db_path)
    manager = EvaluationManager()

    return manager


@pytest.fixture(scope="function")
def rag_evaluator() -> RAGEvaluator:
    """Create a RAGEvaluator instance for testing."""
    # Use fallback mode (no embeddings) for unit tests
    evaluator = RAGEvaluator(
        embedding_provider=None,
        embedding_model=None
    )
    return evaluator


@pytest.fixture(scope="function")
def rag_evaluator_with_embeddings(local_embeddings_config) -> RAGEvaluator:
    """Create a RAGEvaluator instance with local embeddings for integration testing."""
    # Use local embeddings that don't require API keys
    evaluator = RAGEvaluator(
        embedding_provider=local_embeddings_config["provider"],
        embedding_model=local_embeddings_config["model"]
    )
    return evaluator


@pytest.fixture(scope="function")
def quality_evaluator() -> ResponseQualityEvaluator:
    """Create a ResponseQualityEvaluator instance for testing."""
    return ResponseQualityEvaluator()


import pytest_asyncio
@pytest_asyncio.fixture(scope="function")
async def unified_service(temp_db_path) -> AsyncGenerator[UnifiedEvaluationService, None]:
    """Create a UnifiedEvaluationService instance for testing."""
    service = UnifiedEvaluationService(
        db_path=str(temp_db_path),
        enable_webhooks=False,  # Disable webhooks for most tests
        enable_caching=True
    )
    await service.initialize()
    yield service
    await service.shutdown()


@pytest_asyncio.fixture(scope="function")
async def unified_service_with_webhooks(temp_db_path) -> AsyncGenerator[UnifiedEvaluationService, None]:
    """Create a UnifiedEvaluationService instance with webhooks enabled."""
    service = UnifiedEvaluationService(
        db_path=str(temp_db_path),
        enable_webhooks=True,
        enable_caching=True
    )
    await service.initialize()
    yield service
    await service.shutdown()


@pytest.fixture(scope="function")
def override_unified_service(temp_db_path, monkeypatch):
    """Force API endpoints to use an isolated evaluations database."""
    from tldw_Server_API.app.core.Evaluations import unified_evaluation_service as service_module
    from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_unified as router_module

    try:
        from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths as _DP
        _test_user_id = _DP.get_single_user_id()
    except Exception:
        _test_user_id = 1

    user_db_base = temp_db_path.parent / "user_eval_dbs"
    user_db_base.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("USER_DB_BASE_DIR", str(user_db_base))
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("EVALUATIONS_TEST_DB_PATH", str(temp_db_path))

    from tldw_Server_API.app.core.Evaluations import webhook_manager as webhook_module
    from tldw_Server_API.app.core.Evaluations.db_adapter import (
        DatabaseAdapterFactory,
        DatabaseConfig,
        DatabaseType,
    )

    # Preserve current adapter details so we can restore after the test
    original_config = None
    existing_adapter = getattr(webhook_module.webhook_manager, "db_adapter", None)
    if existing_adapter is not None:
        original_config = getattr(existing_adapter, "config", None)

    test_adapter = DatabaseAdapterFactory.create(
        DatabaseConfig(
            db_type=DatabaseType.SQLITE,
            connection_string=str(temp_db_path)
        )
    )
    webhook_module.webhook_manager.set_adapter(test_adapter)

    service = UnifiedEvaluationService(db_path=str(temp_db_path))
    service_module._service_instance = service
    try:
        cache = getattr(service_module, "_service_instances_by_user")
        cache.clear()
        cache[_test_user_id] = service
    except Exception:
        try:
            from collections import OrderedDict  # type: ignore
            service_module._service_instances_by_user = OrderedDict(((_test_user_id, service),))
        except Exception:
            service_module._service_instances_by_user = {_test_user_id: service}  # type: ignore[assignment]
    router_module._evaluation_service = service

    yield service

    router_module._evaluation_service = None
    try:
        cache = getattr(service_module, "_service_instances_by_user")
        cache.pop(_test_user_id, None)
    except Exception:
        _ = None
    service_module._service_instance = None

    # Restore the original webhook adapter to avoid leaking test state
    if original_config is not None:
        restore_config = DatabaseConfig(
            db_type=original_config.db_type,
            connection_string=original_config.connection_string,
            pool_size=original_config.pool_size,
            max_overflow=original_config.max_overflow,
            echo=original_config.echo,
            options=dict(original_config.options) if original_config.options else {}
        )
        restore_adapter = DatabaseAdapterFactory.create(restore_config)
        webhook_module.webhook_manager.set_adapter(restore_adapter)


@pytest.fixture(scope="function")
def connection_pool(temp_db_path) -> ConnectionPool:
    """Create a ConnectionPool instance for testing."""
    pool = ConnectionPool(
        db_path=str(temp_db_path),
        min_connections=1,
        max_connections=5
    )
    pool.initialize()
    yield pool
    pool.close()


@pytest.fixture(scope="function")
def circuit_breaker() -> CircuitBreaker:
    """Create a CircuitBreaker instance for testing."""
    return CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=5,
        expected_exception=Exception
    )


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_evaluation_data() -> Dict[str, Any]:
    """Provide sample evaluation data for testing."""
    return {
        "name": "test_evaluation",
        "eval_type": "model_graded",
        "description": "Test evaluation for unit tests",
        "eval_spec": {
            "evaluator_model": "gpt-4",
            "metrics": ["accuracy", "relevance", "coherence"],
            "threshold": 0.7,
            "temperature": 0.0
        },
        "dataset": [
            {
                "input": {"question": "What is the capital of France?"},
                "expected": {"answer": "Paris"},
                "context": "France is a country in Europe."
            },
            {
                "input": {"question": "What is 2+2?"},
                "expected": {"answer": "4"},
                "context": "Basic arithmetic."
            }
        ],
        "metadata": {
            "author": "test_suite",
            "version": "1.0.0",
            "tags": ["test", "sample"]
        }
    }


@pytest.fixture
def sample_rag_query() -> Dict[str, Any]:
    """Provide sample RAG query data."""
    return {
        "query": "What are the key features of the evaluation system?",
        "context": [
            "The evaluation system supports multiple metrics.",
            "It can evaluate both RAG and standard responses.",
            "The system includes webhook notifications."
        ],
        "response": "The evaluation system has three key features: multi-metric support, RAG/standard evaluation, and webhooks.",
        "ground_truth": "Key features include metric support, response evaluation, and notifications.",
        "metadata": {
            "source": "test_suite",
            "timestamp": datetime.utcnow().isoformat()
        }
    }


@pytest.fixture
def sample_response_quality_data() -> Dict[str, Any]:
    """Provide sample response quality evaluation data."""
    return {
        "response": "The capital of France is Paris. Paris is known for the Eiffel Tower and is located on the Seine River.",
        "reference": "Paris is the capital and largest city of France.",
        "criteria": {
            "coherence": 0.8,
            "relevance": 0.9,
            "fluency": 0.85,
            "factuality": 0.95
        }
    }


@pytest.fixture
def sample_webhook_config() -> Dict[str, Any]:
    """Provide sample webhook configuration."""
    return {
        "url": "https://example.com/webhook",
        "events": ["evaluation.completed", "evaluation.failed"],
        "headers": {
            "Authorization": "Bearer test_token",
            "Content-Type": "application/json"
        },
        "retry_config": {
            "max_retries": 3,
            "retry_delay": 1,
            "backoff_factor": 2
        }
    }


@pytest.fixture
def sample_batch_evaluation_data() -> Dict[str, Any]:
    """Provide sample batch evaluation data."""
    return {
        "evaluations": [
            {
                "name": "batch_eval_1",
                "eval_type": "g_eval",
                "data": {
                    "summary": "Paris is the capital of France.",
                    "source_text": "France is a country in Europe. Its capital is Paris."
                }
            },
            {
                "name": "batch_eval_2",
                "eval_type": "rag",
                "data": {
                    "query": "What is 2+2?",
                    "response": "4",
                    "context": ["Basic math"],
                    "ground_truth": "4"
                }
            }
        ],
        "batch_config": {
            "parallel": True,
            "max_workers": 2,
            "timeout": 30
        }
    }


# ============================================================================
# API Testing Fixtures
# ============================================================================

@pytest.fixture
def api_client(override_unified_service):
    """Create a test client for API testing."""
    from fastapi.testclient import TestClient
    from tldw_Server_API.app.main import app

    with TestClient(app) as client:
        yield client


@pytest_asyncio.fixture
async def async_api_client(override_unified_service):
    """Create an async test client for API testing."""
    from httpx import AsyncClient, ASGITransport
    from tldw_Server_API.app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
def auth_headers() -> Dict[str, str]:
    """Provide authentication headers for API testing."""
    # For single-user mode, use X-API-KEY header
    # Get the actual API key from settings
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    settings = get_settings()

    return {
        "X-API-KEY": settings.SINGLE_USER_API_KEY,
        "Content-Type": "application/json"
    }


# ============================================================================
# Async Helpers
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Performance Testing Fixtures
# ============================================================================

@pytest.fixture
def performance_timer():
    """Provide a timer for performance testing."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):

            self.start_time = time.perf_counter()

        def stop(self):

            self.end_time = time.perf_counter()
            return self.elapsed

        @property
        def elapsed(self):
            if self.start_time is None:
                return 0
            if self.end_time is None:
                return time.perf_counter() - self.start_time
            return self.end_time - self.start_time

    return Timer()


# ============================================================================
# Cleanup Helpers
# ============================================================================

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    # Note: The webhook_manager and user_rate_limiter classes store their state
    # in the database, not in memory attributes. They don't have _webhooks,
    # _delivery_stats, _user_requests, or _user_tiers attributes.
    # If cleanup is needed, it should be done via database operations or
    # by creating new instances with clean test databases.

    # Set connection pool limits to prevent file descriptor exhaustion
    import sqlite3
    # Limit the number of connections that can be cached
    sqlite3.connect(':memory:').execute('PRAGMA max_page_count = 1000')

    yield

    # Cleanup after test - force garbage collection to close connections
    import gc
    gc.collect()


# ============================================================================
# Mock Data Generators for Property Testing
# ============================================================================

@pytest.fixture
def evaluation_data_generator():
    """Generate random evaluation data for property testing."""
    import random
    import string

    def generate():

        return {
            "name": ''.join(random.choices(string.ascii_letters, k=10)),
            "eval_type": random.choice(["model_graded", "g_eval", "rag", "response_quality"]),
            "eval_spec": {
                "evaluator_model": random.choice(["gpt-4", "gpt-3.5-turbo", "claude-2"]),
                "metrics": random.sample(["accuracy", "relevance", "coherence", "fluency"], k=2),
                "threshold": random.uniform(0.5, 0.9)
            },
            "dataset": [
                {
                    "input": {"text": ''.join(random.choices(string.ascii_letters + ' ', k=50))},
                    "expected": {"score": random.uniform(0, 1)}
                }
                for _ in range(random.randint(1, 5))
            ]
        }

    return generate


@pytest.fixture
def mock_rag_evaluator(monkeypatch):
    """Mock RAGEvaluator for unit tests."""

    # Mock the LLM analyze function for RAG tests
    def mock_analyze(*args, **kwargs):
             # Return scores based on what's being evaluated
        prompt = args[0] if args else kwargs.get('prompt', '')
        if 'faithfulness' in prompt.lower():
            return "0.85"
        elif 'relevance' in prompt.lower():
            return "0.9"
        elif 'similarity' in prompt.lower():
            return "0.88"
        else:
            return "0.8"

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib.analyze",
        mock_analyze
    )

    # Mock embedding similarity if needed
    async def mock_compute_similarity(*args, **kwargs):
        return 0.85

    return mock_analyze


@pytest.fixture(autouse=True)
def auto_mock_llm_for_unit_tests(request, monkeypatch):
    """Automatically mock LLM calls for unit tests in test_rag_evaluator."""
    if "test_rag_evaluator" in request.node.module.__name__:
        # Mock the LLM analyze function with correct signature
        def mock_analyze(api_name, input_data, custom_prompt_arg="", api_key="", system_message="", temp=0.1, **kwargs):
                     # Check both input_data and custom_prompt_arg for keywords
            combined_text = f"{input_data} {custom_prompt_arg}".lower()
            if 'faithfulness' in combined_text:
                return "4.7"  # Use same value as mock_llm_analyze
            elif 'relevance' in combined_text:
                return "4.3"
            elif 'similarity' in combined_text:
                return "0.88"
            else:
                return "4.0"

        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib.analyze",
            mock_analyze
        )


@pytest.fixture(autouse=True)
def mock_llm_for_requires_llm(request, monkeypatch):
    """Automatically mock LLM calls for tests marked with requires_llm."""
    if "requires_llm" in [m.name for m in request.node.iter_markers()]:
        # Mock the run_geval function to return proper structured data
        def mock_run_geval(*args, **kwargs):
            return {
                "metrics": {
                    "coherence": 4.5,
                    "consistency": 4.2,
                    "fluency": 4.8,
                    "relevance": 4.3
                },
                "average_score": 4.45,
                "assessment": "The summary is well-written and captures the main points.",
                "explanations": {
                    "coherence": "Good structure",
                    "consistency": "Factually accurate",
                    "fluency": "Well-written",
                    "relevance": "Covers main points"
                }
            }

        # Mock the RAG evaluator
        async def mock_rag_evaluate(*args, **kwargs):
            return {
                "metrics": {
                    "answer_relevance": 0.85,
                    "context_relevance": 0.90,
                    "answer_faithfulness": 0.87,
                    "answer_similarity": 0.88
                },
                "overall_score": 0.88,
                "retrieval_quality": 0.90,
                "generation_quality": 0.86,
                "suggestions": [
                    "Consider improving context retrieval",
                    "Response could be more detailed"
                ]
            }

        # Mock the response quality evaluator
        async def mock_response_quality_evaluate(*args, **kwargs):
            return {
                "metrics": {
                    "relevance": 0.9,
                    "completeness": 0.85,
                    "accuracy": 0.88,
                    "clarity": 0.92
                },
                "overall_score": 0.89,
                "format_compliance": True,
                "issues": [],
                "suggestions": ["Consider adding more detail"]
            }

        # Mock the LLM calls in Summarization_General_Lib
        def mock_analyze(*args, **kwargs):
            return "4"  # Return a simple score string

        # Apply mocks
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Evaluations.ms_g_eval.run_geval",
            mock_run_geval
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Evaluations.rag_evaluator.RAGEvaluator.evaluate",
            mock_rag_evaluate
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Evaluations.response_quality_evaluator.ResponseQualityEvaluator.evaluate",
            mock_response_quality_evaluate
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib.analyze",
            mock_analyze
        )
