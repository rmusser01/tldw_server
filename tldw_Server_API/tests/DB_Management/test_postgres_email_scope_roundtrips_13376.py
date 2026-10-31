"""Scope installation retains all fields in one database round trip."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.postgresql_backend import PostgreSQLBackend
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context


@pytest.mark.parametrize('user_id,org_ids,team_ids,is_admin,expected', [
    (42, [7, 8], [9], True, ('42', '42', '7,8', '9', '1')),
    (None, [], [], False, ('', '', '', '', '0')),
])
def test_scope_fields_are_set_together(monkeypatch, user_id, org_ids, team_ids, is_admin, expected):
    monkeypatch.delenv('TLDW_CONTENT_PG_ROLE_SWITCH', raising=False)
    calls = []
    @contextmanager
    def cursor():
        yield SimpleNamespace(execute=lambda sql, params=(): calls.append((sql, params)))
    pool = object.__new__(PostgreSQLBackend)
    with scoped_context(user_id=user_id, org_ids=org_ids, team_ids=team_ids, is_admin=is_admin):
        pool._apply_scope_settings(SimpleNamespace(cursor=cursor))
    scope_calls = [(sql, params) for sql, params in calls if 'set_config' in sql]
    assert len(scope_calls) == 1
    sql, params = scope_calls[0]
    assert params == expected
    assert all(key in sql for key in ('app.current_user_id', 'app.user_id', 'app.org_ids', 'app.team_ids', 'app.is_admin'))
    assert calls[0][0] == 'RESET SESSION AUTHORIZATION'
