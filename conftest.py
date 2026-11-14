"""
Top-level pytest configuration for the repository.

Registers shared test plugins to satisfy pytest's top-level requirement.
"""

import os

import pytest

# Provide a deterministic single-user test key before loading plugins.
os.environ.setdefault("SINGLE_USER_TEST_API_KEY", "test-api-key-12345")
os.environ.setdefault("SINGLE_USER_API_KEY", os.environ["SINGLE_USER_TEST_API_KEY"])
os.environ.setdefault("AUTH_MODE", "single_user")
# TASK-13950.19: replaces the TEST_MODE skip in the auth ingress fallback limiter.
# The suite raises the per-bucket rate instead (bucket capacity still applies); a test of the brute-force floor sets its own.
os.environ.setdefault("AUTH_DEPS_FALLBACK_RATE_LIMIT", "1000000")
os.environ.setdefault("AUTH_DEPS_AUTH_FALLBACK_RATE_LIMIT", "1000000")
os.environ.pop("PROFILE", None)
# Single-threaded numeric libraries keep joblib/loky from leaking semaphores under pytest.
import tldw_Server_API.tests._thread_env  # noqa: E402,F401

pytest_plugins = (
    "tldw_Server_API.tests.helpers.pgvector",
    "tldw_Server_API.tests._plugins.e2e_fixtures",
    "tldw_Server_API.tests._plugins.e2e_state_fixtures",
    "tldw_Server_API.tests._plugins.media_fixtures",
    "tldw_Server_API.tests._plugins.postgres",
)


@pytest.fixture(autouse=True, scope="session")
def _audit_inline_io():
    """Audit services flush inline and start no background loops in tests (TASK-13950.19).

    Many tests patch asyncio.sleep, which would spin the flush loop. A test of the
    background path passes UnifiedAuditService(..., inline_io=False).
    """
    from tldw_Server_API.app.core.Audit.unified_audit_service import UnifiedAuditService

    UnifiedAuditService.inline_io_default = True
    yield
    UnifiedAuditService.inline_io_default = False
