"""
Top-level pytest configuration for the repository.

Registers shared test plugins to satisfy pytest's top-level requirement.
"""

import os

# Provide a deterministic single-user test key before loading plugins.
os.environ.setdefault("SINGLE_USER_TEST_API_KEY", "test-api-key-12345")
os.environ.setdefault("SINGLE_USER_API_KEY", os.environ["SINGLE_USER_TEST_API_KEY"])
os.environ.setdefault("AUTH_MODE", "single_user")
os.environ.pop("PROFILE", None)

pytest_plugins = (
    "pytest_mock",
    "tldw_Server_API.tests.helpers.pgvector",
    "tldw_Server_API.tests._plugins.e2e_fixtures",
    "tldw_Server_API.tests._plugins.e2e_state_fixtures",
    "tldw_Server_API.tests._plugins.media_fixtures",
    "tldw_Server_API.tests._plugins.postgres",
)
