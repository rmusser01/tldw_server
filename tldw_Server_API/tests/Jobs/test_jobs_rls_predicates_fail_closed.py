"""The Jobs RLS predicates must deny on missing tenant context, not allow.

These policies previously read

    (NULLIF(current_setting('app.owner_user_id', true), '') IS NULL
     OR owner_user_id = ...)

so an unset GUC matched every row of every user. Any query issued without a
prior set_rls_context saw the whole jobs table. These tests pin the inverted
behaviour: unset owner means no rows.

Pure string assertions on the SQL builder, so they run without a database.
"""

import pytest

from tldw_Server_API.app.core.Jobs.pg_migrations import build_jobs_rls_predicates

pytestmark = pytest.mark.unit


def test_owner_predicate_has_no_is_null_escape_hatch():
    """The regression. `<expr> IS NULL OR ...` matches everything when unset."""
    _admin, _domain, owner = build_jobs_rls_predicates()

    assert "IS NULL OR" not in owner


def test_owner_predicate_compares_the_column_to_the_tenant_guc():
    _admin, _domain, owner = build_jobs_rls_predicates()

    assert "owner_user_id = " in owner
    assert "current_setting('app.owner_user_id', true)" in owner
    # NULLIF keeps an empty GUC equivalent to an absent one, and both must
    # compare as NULL so the row is filtered rather than returned.
    assert "NULLIF(" in owner


def test_admin_claim_is_still_an_explicit_bypass():
    """Workers legitimately span owners; they do it through the admin claim."""
    admin, _domain, _owner = build_jobs_rls_predicates()

    assert "current_setting('app.is_admin', true)" in admin
    assert "= 'true'" in admin


def test_domain_stays_permissive_when_unset():
    """Domain is a worker capability filter, not the tenant boundary.

    A worker that has not narrowed its domains polls all of them, so this one
    keeps its IS NULL arm on purpose. The tenant line is held by owner.
    """
    _admin, domain, _owner = build_jobs_rls_predicates()

    assert "IS NULL OR" in domain
    assert "domain = ANY(" in domain
