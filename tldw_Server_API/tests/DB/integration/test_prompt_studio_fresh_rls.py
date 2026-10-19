"""Fresh PostgreSQL startup installs policies without a prior Prompt Studio visit."""

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import (
    ensure_chacha_rls,
    ensure_prompt_studio_rls,
)
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
from tldw_Server_API.app.services.startup_pg_rls import run_pg_rls_auto_ensure

pytestmark = pytest.mark.integration


def test_fresh_prompt_studio_bootstrap_preserves_forced_tenant_isolation(pg_restricted_backend):
    """Real startup ordering, idempotence, data preservation and foreign denial."""
    backend = pg_restricted_backend
    assert not backend.table_exists("prompt_studio_projects")  # nosec B101 - pytest fresh-state assertion.
    assert run_pg_rls_auto_ensure(  # nosec B101 - pytest actual startup result.
        backend,
        ensure_prompt_studio_rls=ensure_prompt_studio_rls,
        ensure_chacha_rls=ensure_chacha_rls,
        logger_obj=logger,
    ) == (True, True)
    with backend.transaction() as conn:
        policies = backend.execute(
            "SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE oid = 'prompt_studio_projects'::regclass",
            connection=conn,
        ).first
        assert policies == {"relrowsecurity": True, "relforcerowsecurity": True}  # nosec B101 - pytest real RLS flags.
    alice = PromptStudioDatabase(":memory:", "a01-alice", backend=backend)
    bob = PromptStudioDatabase(":memory:", "a01-bob", backend=backend)
    try:
        project = alice.create_project("First owned project")
        alice.close_connection()
        assert bob.get_project(project["id"]) is None  # nosec B101 - pytest foreign-row denial.
        bob.close_connection()
        assert ensure_prompt_studio_rls(backend) is True  # nosec B101 - pytest repeated migration/policy install.
        assert alice.get_project(project["id"])["name"] == "First owned project"  # nosec B101 - pytest owned-data preservation.
    finally:
        alice.close()
        bob.close()
