from __future__ import annotations

import pytest


def test_get_db_path_rejects_path_like_user_ids_outside_tests(monkeypatch, tmp_path) -> None:
    from tldw_Server_API.app.api.v1.endpoints.watchlist_alert_rules import _get_db_path

    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("TLDW_USER_DB_DIR", str(tmp_path / "user_dbs"))
    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.db_path_utils._is_test_context",
        lambda: False,
    )

    with pytest.raises(ValueError, match="Invalid user_id"):
        _get_db_path("../escape")
