"""
Pytest configuration for e2e tests.

Defines custom markers and test configuration for end-to-end testing
that simulates real user interactions with the application.
"""

import pytest
import os
import asyncio
import uuid
from typing import Dict, Any, Optional

# Disable rate limiting for all e2e tests
@pytest.fixture(autouse=True, scope="session")
def disable_rate_limiting():
    """Disable rate limiting for all tests in e2e suite"""
    os.environ["TESTING"] = "true"  # For embeddings endpoint
    os.environ["TEST_MODE"] = "true"  # For regular limiter
    os.environ.setdefault("ENABLE_REGISTRATION", "true")
    os.environ.setdefault("REQUIRE_REGISTRATION_CODE", "false")
    yield
    # Clean up after tests
    if "TESTING" in os.environ:
        del os.environ["TESTING"]
    if "TEST_MODE" in os.environ:
        del os.environ["TEST_MODE"]

# Custom markers for conditional test execution
pytest_markers = [
    "single_user: marks tests that only run in single-user mode",
    "multi_user: marks tests that only run in multi-user mode",
    "requires_llm: marks tests that require a configured LLM provider",
    "requires_transcription: marks tests that require transcription services",
    "requires_embeddings: marks tests that require embedding services",
    "slow: marks tests that take a long time to run (>30s)",
    "critical: marks tests that are critical for basic functionality",
    "benchmark: marks benchmark-related tests",
    "media_processing: marks media processing tests",
    "auth: marks authentication-related tests",
    "timeout: marks tests with explicit timeout limits (requires pytest-timeout plugin)",
]

def pytest_configure(config):
    """Register custom markers."""
    for marker in pytest_markers:
        config.addinivalue_line("markers", marker)
    # Additional marker for explicit rate-limit verification tests
    config.addinivalue_line("markers", "rate_limits: Tests that verify rate limiting behavior")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on environment and configuration."""

    # Check environment to determine which tests to skip
    auth_mode = os.getenv("E2E_AUTH_MODE", "auto")  # auto, single_user, multi_user
    skip_slow = config.getoption("--skip-slow", default=False)
    run_slow = config.getoption("--run-slow", default=False)
    run_critical_only = config.getoption("--critical-only", default=False)

    # Get auth mode from API if auto
    if auth_mode == "auto":
        auth_mode = _detect_auth_mode()

    for item in items:
        # Skip tests based on auth mode
        if auth_mode == "single_user":
            if "multi_user" in item.keywords:
                item.add_marker(pytest.mark.skip(
                    reason="Multi-user test skipped in single-user mode"
                ))
        elif auth_mode == "multi_user":
            if "single_user" in item.keywords:
                item.add_marker(pytest.mark.skip(
                    reason="Single-user test skipped in multi-user mode"
                ))

        # Skip slow tests if requested
        if skip_slow and not run_slow and "slow" in item.keywords:
            item.add_marker(pytest.mark.skip(
                reason="Slow test skipped (use --run-slow to include)"
            ))

        # Run only critical tests if requested
        if run_critical_only and "critical" not in item.keywords:
            item.add_marker(pytest.mark.skip(
                reason="Non-critical test skipped (--critical-only mode)"
            ))


def _detect_auth_mode() -> str:
    """Detect auth mode from running API server."""
    if _env_truthy("E2E_INPROCESS"):
        return os.getenv("AUTH_MODE", "single_user").strip().lower().replace("-", "_")

    try:
        import httpx
        base_url = os.getenv("E2E_TEST_BASE_URL", "http://localhost:8000")
        with httpx.Client(timeout=5) as client:
            response = client.get(f"{base_url}/api/v1/health")
            if response.status_code == 200:
                return response.json().get("auth_mode", "multi_user")
    except:
        _ = None
    return "multi_user"  # Default


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="Skip slow-running tests"
    )
    parser.addoption(
        "--critical-only",
        action="store_true",
        default=False,
        help="Run only critical tests"
    )
    parser.addoption(
        "--auth-mode",
        action="store",
        default="auto",
        choices=["auto", "single_user", "multi_user"],
        help="Force specific auth mode for testing"
    )
    # Alias to include slow tests (pairs with --skip-slow default behavior)
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Include tests marked as slow"
    )


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "y", "on"}

# Attach the shared test results dict to config so sessionfinish can print a summary
@pytest.fixture(scope="session", autouse=True)
def _attach_results_to_config(test_results, request):
    request.config._test_results = test_results  # type: ignore[attr-defined]
    yield


@pytest.fixture(scope="session", autouse=True)
def _ensure_inprocess_single_user_profile(disable_rate_limiting):
    """Bootstrap single-user AuthNZ profile for in-process e2e runs."""
    if not _env_truthy("E2E_INPROCESS"):
        yield
        return

    from tldw_Server_API.tests.e2e.fixtures import ensure_server_running

    try:
        health = ensure_server_running()
    except Exception as exc:
        pytest.fail(f"❌ Failed to initialize in-process API for single-user bootstrap: {exc}")

    auth_mode = str(health.get("auth_mode") or os.getenv("AUTH_MODE", "")).lower()
    if auth_mode not in {"single_user", "single-user", "singleuser"}:
        yield
        return

    from tldw_Server_API.app.core.AuthNZ.initialize import (
        bootstrap_single_user_profile,
        ensure_single_user_rbac_seed_if_needed,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    from tldw_Server_API.app.core.DB_Management.Users_DB import get_users_db

    async def _bootstrap_single_user():
        await ensure_single_user_rbac_seed_if_needed()
        await bootstrap_single_user_profile()
        settings = get_settings()
        users_db = await get_users_db()
        user = await users_db.get_user_by_id(int(settings.SINGLE_USER_FIXED_ID))
        if not user:
            raise RuntimeError(
                f"single-user bootstrap did not create user id={settings.SINGLE_USER_FIXED_ID}"
            )

    try:
        asyncio.run(_bootstrap_single_user())
    except Exception as exc:
        pytest.fail(f"❌ Failed single-user bootstrap for in-process e2e tests: {exc}")

    yield


@pytest.fixture(scope="session", autouse=True)
def _ensure_inprocess_admin_bearer(disable_rate_limiting, _ensure_inprocess_single_user_profile):
    if not _env_truthy("E2E_INPROCESS"):
        yield
        return
    if os.getenv("E2E_ADMIN_BEARER"):
        yield
        return

    from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
    from tldw_Server_API.app.core.AuthNZ.username_utils import (
        InvalidUsernameError,
        normalize_admin_username,
    )
    from tldw_Server_API.app.core.DB_Management.Users_DB import (
        DuplicateUserError,
        ensure_user_directories,
        get_users_db,
    )
    from tldw_Server_API.tests.e2e.fixtures import APIClient

    admin_username_raw = os.getenv("E2E_ADMIN_USERNAME", "e2e_admin")
    try:
        admin_username = normalize_admin_username(admin_username_raw)
    except InvalidUsernameError:
        admin_username = "e2e_admin"

    admin_email = os.getenv("E2E_ADMIN_EMAIL", f"{admin_username}@example.com")
    admin_password = os.getenv("E2E_ADMIN_PASSWORD", "Tlp9!ZxVq8@M")

    api_client = APIClient()
    try:
        try:
            liveness = api_client.client.get("/health")
            liveness.raise_for_status()
            if liveness.json() != {"status": "ok"}:
                raise RuntimeError("public liveness response did not match the minimal contract")
        except Exception as exc:
            pytest.fail(f"❌ Failed to initialize in-process API for admin token minting: {exc}")
        auth_mode = str(os.getenv("AUTH_MODE", "")).lower()
        if auth_mode not in {"multi_user", "multi-user", "multiuser"}:
            yield
            return

        async def _ensure_admin_user():
            users_db = await get_users_db()
            password_hash = PasswordService().hash_password(admin_password)
            existing = await users_db.get_user_by_username(admin_username)
            if existing:
                admin_user = await users_db.update_user(
                    int(existing["id"]),
                    password_hash=password_hash,
                    role="admin",
                    is_superuser=True,
                    is_active=True,
                    is_verified=True,
                )
            else:
                admin_user = None
                for email_candidate in (
                    admin_email,
                    f"{admin_username}+{uuid.uuid4().hex[:6]}@example.com",
                ):
                    try:
                        admin_user = await users_db.create_user(
                            username=admin_username,
                            email=email_candidate,
                            password_hash=password_hash,
                            role="admin",
                            is_active=True,
                            is_verified=True,
                            is_superuser=True,
                        )
                        break
                    except DuplicateUserError:
                        continue

                if admin_user is None:
                    raise RuntimeError("Unable to create in-process admin user for E2E tests.")

            admin_user_id = int(admin_user["id"])
            users_repo = await AuthnzUsersRepo.from_pool()
            await users_repo.assign_role_if_missing(user_id=admin_user_id, role_name="admin")
            if not await users_repo.has_role_assignment(user_id=admin_user_id, role_name="admin"):
                raise RuntimeError("Canonical admin role membership is unavailable for E2E tests.")
            await ensure_user_directories(admin_user_id)
            return admin_user

        asyncio.run(_ensure_admin_user())
        api_client.login(admin_username, admin_password)
        if not api_client.token:
            pytest.fail("❌ Failed to mint E2E_ADMIN_BEARER: login did not return access_token")
        os.environ["E2E_ADMIN_BEARER"] = api_client.token
        yield
    finally:
        api_client.close()

# For tests that validate rate limiting, temporarily disable the global TEST_MODE
# env that turns off rate limits. Controlled via @pytest.mark.rate_limits
@pytest.fixture(autouse=True)
def _rate_limit_env_toggle(request):
    if request.node.get_closest_marker("rate_limits"):
        original_test_mode = os.environ.get("TEST_MODE")
        original_testing = os.environ.get("TESTING")
        # Remove flags that disable rate limiting
        os.environ.pop("TEST_MODE", None)
        os.environ.pop("TESTING", None)
        try:
            yield
        finally:
            if original_test_mode is None:
                os.environ.pop("TEST_MODE", None)
            else:
                os.environ["TEST_MODE"] = original_test_mode
            if original_testing is None:
                os.environ.pop("TESTING", None)
            else:
                os.environ["TESTING"] = original_testing
    else:
        yield


@pytest.fixture(autouse=True)
def _cleanup_open_httpx_clients():
    yield
    try:
        from tldw_Server_API.tests.e2e.fixtures import APIClient
        APIClient.close_open_clients()
    except Exception:
        _ = None
