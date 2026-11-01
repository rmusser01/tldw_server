"""Jobs scopes restore tokens, including nonempty nested/error contexts."""

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager


def context_values():
    return (
        JobManager._RLS_IS_ADMIN.get(),
        JobManager._RLS_DOMAIN_ALLOWLIST.get(),
        JobManager._RLS_OWNER_USER_ID.get(),
    )


def test_nested_scope_restores_nonempty_prior_values():
    JobManager.set_rls_context(is_admin=True, domain_allowlist="prior", owner_user_id="prior-owner")
    try:
        with JobManager.scoped_rls_context(is_admin=False, domain_allowlist="claims", owner_user_id="1"):
            assert context_values() == (False, "claims", "1")
            with JobManager.scoped_rls_context(is_admin=False, domain_allowlist="notes", owner_user_id="2"):
                assert context_values() == (False, "notes", "2")
            assert context_values() == (False, "claims", "1")
        assert context_values() == (True, "prior", "prior-owner")
    finally:
        JobManager.clear_rls_context()


def test_exception_restores_all_previous_values():
    JobManager.set_rls_context(is_admin=False, domain_allowlist="previous", owner_user_id="owner")
    try:
        with pytest.raises(RuntimeError, match="test exit"):
            with JobManager.scoped_rls_context(is_admin=True, domain_allowlist=None, owner_user_id=None):
                raise RuntimeError("test exit")
        assert context_values() == (False, "previous", "owner")
    finally:
        JobManager.clear_rls_context()
