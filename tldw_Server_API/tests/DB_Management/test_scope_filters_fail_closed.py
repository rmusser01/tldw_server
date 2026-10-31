"""Evaluations_DB must not run an unscoped query when a scope was intended.

_append_user_filter returned the query unchanged whenever the caller's user id
produced no filter variants, which a blank or whitespace id does. The caller
believed it was scoping; the predicate silently vanished. On SQLite the
per-user database file hides that. On PostgreSQL every account shares the
table, and delete_evaluation runs through the same helper.
"""

import pytest

pytestmark = pytest.mark.unit


class TestEvaluationsUserFilter:
    """Evaluations_DB._append_user_filter dropped its predicate on a blank id."""

    @staticmethod
    def _filter(db, user_id):
        return db._append_user_filter("SELECT 1 WHERE 1=1", [], "created_by", user_id)

    @pytest.fixture()
    def db(self, tmp_path):
        from tldw_Server_API.app.core.DB_Management.Evaluations_DB import (
            EvaluationsDatabase,
        )

        return EvaluationsDatabase(str(tmp_path / "evals.db"))

    def test_blank_user_id_raises_instead_of_running_unscoped(self, db):
        """The regression. A blank scope silently read and deleted every row."""
        with pytest.raises(ValueError, match="no filter variants"):
            self._filter(db, "")

    def test_whitespace_user_id_raises(self, db):
        with pytest.raises(ValueError, match="no filter variants"):
            self._filter(db, "   ")

    def test_none_remains_an_explicit_opt_out(self, db):
        """Workers and admin paths legitimately span accounts; that stays."""
        query, params = self._filter(db, None)

        assert "created_by" not in query
        assert params == []

    def test_a_real_user_id_is_applied(self, db):
        query, params = self._filter(db, "7")

        assert "created_by" in query
        assert "7" in params


class TestClaimsPostgresOwnerScope:
    """On PostgreSQL the normal claims path had no owner predicate at all.

    _resolve_media_db only ever set owner_user_id when an admin passed an
    explicit ?user_id=. On SQLite that is fine: get_media_db_for_user hands back
    the caller's own database file, so the file is the boundary. On PostgreSQL
    every account shares one set of tables and the query went out unscoped.
    """

    @staticmethod
    def _resolve(backend, user_id=None, is_admin=False):
        from types import SimpleNamespace

        from tldw_Server_API.app.core.Claims_Extraction import claims_service

        db = SimpleNamespace(backend_type=backend)
        user = SimpleNamespace(id=7, is_superuser=is_admin)
        with claims_service._resolve_media_db(
            db=db,
            current_user=user,
            user_id=user_id,
            admin_required=True,
            owner_filter=True,
        ) as (resolved_db, owner):
            return resolved_db, owner

    def test_postgres_normal_path_scopes_to_the_caller(self):
        """The regression: this yielded owner_user_id=None on a shared table."""
        from tldw_Server_API.app.core.DB_Management.backends.base import BackendType

        _db, owner = self._resolve(BackendType.POSTGRESQL)

        assert owner == 7

    def test_sqlite_normal_path_still_relies_on_the_per_user_file(self):
        """No owner predicate needed; the database file is the boundary."""
        from tldw_Server_API.app.core.DB_Management.backends.base import BackendType

        _db, owner = self._resolve(BackendType.SQLITE)

        assert owner is None
