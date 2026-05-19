import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig, BackendType
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import ensure_prompt_studio_rls


pytestmark = pytest.mark.integration


def test_apply_rls_policies_smoke(pg_database_config: DatabaseConfig):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    applied = ensure_prompt_studio_rls(backend)
    if applied is not True:
        pytest.fail("expected PostgreSQL RLS installation to succeed in the integration fixture")
