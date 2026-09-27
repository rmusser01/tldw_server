"""Local pytest configuration for Audio tests."""

import copy
import os

import pytest

# Audio test modules assert against /api/v1/audio REST and WS routes.
# Keep global MINIMAL_TEST_APP behavior, but opt this suite into mounting audio routers.
os.environ.setdefault("MINIMAL_TEST_INCLUDE_AUDIO", "1")


@pytest.fixture(autouse=True)
def _isolate_provider_override_cache():
    """Keep background refresh failures from leaking across Audio tests."""
    from tldw_Server_API.app.core.AuthNZ import (
        llm_provider_overrides as overrides_module,
    )

    with overrides_module._OVERRIDE_LOCK:
        original = copy.deepcopy(overrides_module._OVERRIDE_CACHE)
        original_healthy = overrides_module._OVERRIDE_CACHE_HEALTHY
        original_ttl_enabled = (
            not overrides_module._OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS
        )
    overrides_module.set_llm_provider_overrides_cache_for_tests({})
    try:
        yield
    finally:
        overrides_module.set_llm_provider_overrides_cache_for_tests(
            original,
            healthy=original_healthy,
            ttl_enabled=original_ttl_enabled,
        )
