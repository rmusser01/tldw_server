from __future__ import annotations

import contextlib


def test_claims_rebuild_db_session_uses_managed_media_database(monkeypatch, tmp_path) -> None:
    from tldw_Server_API.app.services import startup_claims_rebuild

    captured: dict[str, object] = {}
    db_path = str(tmp_path / "media-17.db")

    @contextlib.contextmanager
    def _fake_managed_media_database(
        client_id: str,
        *,
        db_path: str,
        initialize: bool = True,
    ):
        captured["client_id"] = client_id
        captured["db_path"] = db_path
        captured["initialize"] = initialize
        yield "db-sentinel"

    monkeypatch.setattr(
        startup_claims_rebuild,
        "_get_user_media_db_path",
        lambda user_id: str(tmp_path / f"media-{user_id}.db"),
    )
    monkeypatch.setattr(
        startup_claims_rebuild,
        "_managed_media_database",
        _fake_managed_media_database,
    )

    settings = {
        "SINGLE_USER_FIXED_ID": "17",
        "SERVER_CLIENT_ID": "startup-client",
    }

    with startup_claims_rebuild._claims_rebuild_db_session(settings) as (user_id, db_path, db):
        assert user_id == 17
        assert db_path == str(tmp_path / "media-17.db")
        assert db == "db-sentinel"

    assert captured == {
        "client_id": "startup-client",
        "db_path": db_path,
        "initialize": False,
    }


def test_startup_claims_rebuild_media_id_helper_delegates_to_claims_service(monkeypatch) -> None:
    from tldw_Server_API.app.services import startup_claims_rebuild
    from tldw_Server_API.app.core.Claims_Extraction import claims_service

    captured: dict[str, object] = {}

    def _fake_list_claims_rebuild_media_ids(
        db,
        *,
        policy,
        stale_days,
        compare_media_last_modified,
        limit,
    ):
        captured["db"] = db
        captured["policy"] = policy
        captured["stale_days"] = stale_days
        captured["compare_media_last_modified"] = compare_media_last_modified
        captured["limit"] = limit
        return [101, 202]

    monkeypatch.setattr(
        claims_service,
        "list_claims_rebuild_media_ids",
        _fake_list_claims_rebuild_media_ids,
    )

    db = object()
    result = startup_claims_rebuild._list_claims_rebuild_media_ids(
        db,
        policy="stale",
        stale_days=7,
        compare_media_last_modified=False,
        limit=25,
    )

    assert result == [101, 202]
    assert captured == {
        "db": db,
        "policy": "stale",
        "stale_days": 7,
        "compare_media_last_modified": False,
        "limit": 25,
    }
