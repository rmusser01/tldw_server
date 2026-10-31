"""Local startup projection authorizes historical references without rewriting storage."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.Chat.assistant_startup import AssistantStartup, encode_assistant_startup
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, InputError
from tldw_Server_API.app.core.Workspaces import assistant_defaults

pytestmark = pytest.mark.integration

UNKNOWN = {"schema_version": 1, "source": "unknown", "workspace_id": None, "workspace_version": None}


@pytest.fixture
def projection_db(tmp_path: Path) -> Iterator[CharactersRAGDB]:
    """Keep Workspace authorization on the real per-user SQLite boundary."""
    db = CharactersRAGDB(tmp_path / "projection.db", client_id="owner")
    db.upsert_workspace("private-origin", "Origin")
    try:
        yield db
    finally:
        db.close_all_connections()


@pytest.mark.parametrize("source", ["workspace_default", "system_fallback"])
@pytest.mark.parametrize("visibility", ["visible", "archived", "deleted", "staged", "missing"])
def test_projection_checks_origin_visibility_without_rewriting_storage(
    projection_db: CharactersRAGDB, source: str, visibility: str,
) -> None:
    """Readable current scope cannot authorize a hidden historical Workspace."""
    db = projection_db
    origin = AssistantStartup(source=source, workspace_id="private-origin", workspace_version=1)
    cid = db.add_conversation({"id": "moved", "title": "Now global"}, assistant_startup=origin)
    raw = db.get_conversation_by_id(cid)["assistant_startup_json"]
    if visibility == "archived":
        db.update_workspace("private-origin", {"archived": True}, 1)
    elif visibility == "deleted":
        db.delete_workspace("private-origin", expected_version=1)
    elif visibility == "staged":
        with db.transaction() as conn:
            conn.execute("UPDATE workspaces SET system_operation_state = 'staged' WHERE id = ?", ("private-origin",))
    elif visibility == "missing":
        with db.transaction() as conn:
            conn.execute("DELETE FROM workspaces WHERE id = ?", ("private-origin",))

    projected = assistant_defaults.project_assistant_startup(db, raw=raw, user_id="owner")
    expected = {"schema_version": 1, "source": source, "workspace_id": "private-origin", "workspace_version": 1}
    assert projected.model_dump() == (expected if visibility in {"visible", "archived"} else UNKNOWN)
    assert db.get_conversation_by_id(cid)["assistant_startup_json"] == raw


@pytest.mark.parametrize("raw", [None, "not-json", '{"schema_version":1,"source":"workspace_default"}', "x" * 1025])
def test_projection_decodes_legacy_or_corrupt_values_before_any_lookup(
    projection_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, raw: object,
) -> None:
    """Corrupt reference-shaped data cannot reach a Workspace authorization query."""
    def unexpected_lookup(*args: Any, **kwargs: Any) -> None:
        """Make an unnecessary authorization lookup observable."""
        pytest.fail("Invalid startup data must not trigger Workspace lookup")

    monkeypatch.setattr(projection_db, "get_workspace", unexpected_lookup)
    assert assistant_defaults.project_assistant_startup(projection_db, raw=raw, user_id="owner").model_dump() == UNKNOWN


@pytest.mark.parametrize("source", ["unknown", "explicit", "explicit_none", "fork", "system_fallback"])
def test_nonreference_sources_do_not_need_workspace_access(
    projection_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, source: str,
) -> None:
    """A non-reference value remains useful even when Workspace storage is unavailable."""
    def unavailable_lookup(*args: Any, **kwargs: Any) -> None:
        """Fail if a value without references is coupled to the Workspace database."""
        raise CharactersRAGDBError("Workspace storage unavailable")

    raw = encode_assistant_startup(AssistantStartup(source=source))
    monkeypatch.setattr(projection_db, "get_workspace", unavailable_lookup)
    assert assistant_defaults.project_assistant_startup(projection_db, raw=raw, user_id="owner").model_dump() == {
        "schema_version": 1, "source": source, "workspace_id": None, "workspace_version": None,
    }


def test_projection_does_not_mask_database_failures(
    projection_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed authorization read is not successful redaction or proof of absence."""
    def unavailable_lookup(*args: Any, **kwargs: Any) -> None:
        """Inject a storage failure at the existing read boundary."""
        raise CharactersRAGDBError("Workspace storage unavailable")

    raw = encode_assistant_startup(AssistantStartup(source="workspace_default", workspace_id="private-origin", workspace_version=1))
    monkeypatch.setattr(projection_db, "get_workspace", unavailable_lookup)
    with pytest.raises(CharactersRAGDBError):
        assistant_defaults.project_assistant_startup(projection_db, raw=raw, user_id="owner")


def test_projection_cache_is_explicitly_scoped_to_one_request(
    projection_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated origins are deduplicated while the next request observes access loss."""
    db = projection_db
    raw = encode_assistant_startup(AssistantStartup(source="workspace_default", workspace_id="private-origin", workspace_version=1))
    real_lookup = db.get_workspace
    calls: list[str] = []

    def counted_lookup(workspace_id: str, **kwargs: Any) -> dict[str, Any] | None:
        """Count real reads without replacing authorization behavior."""
        calls.append(workspace_id)
        return real_lookup(workspace_id, **kwargs)

    monkeypatch.setattr(db, "get_workspace", counted_lookup)
    first_request: dict[str, bool] = {}
    for _ in range(3):
        result = assistant_defaults.project_assistant_startup(db, raw=raw, user_id="owner", workspace_visibility_cache=first_request)
        assert result.source == "workspace_default"
    assert calls == ["private-origin"]
    db.delete_workspace("private-origin", expected_version=1)
    calls.clear()
    assert assistant_defaults.project_assistant_startup(db, raw=raw, user_id="owner", workspace_visibility_cache={}).model_dump() == UNKNOWN
    assert calls == ["private-origin"]


def test_projection_cannot_reuse_visibility_for_a_different_owner(projection_db: CharactersRAGDB) -> None:
    """A caller-supplied user cannot authorize cached references from this scoped DB."""
    raw = encode_assistant_startup(AssistantStartup(source="workspace_default", workspace_id="private-origin", workspace_version=1))
    with pytest.raises(InputError):
        assistant_defaults.project_assistant_startup(
            projection_db, raw=raw, user_id="other-user", workspace_visibility_cache={"private-origin": True},
        )


def test_sync_outgoing_payloads_omit_local_startup(projection_db: CharactersRAGDB) -> None:
    """Actual trigger logs and the v2 payload builder cannot leak historical origin."""
    from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import _conversation_sync_payload

    db = projection_db
    cid = db.add_conversation(
        {"title": "Sync boundary"},
        assistant_startup=AssistantStartup(source="workspace_default", workspace_id="private-origin", workspace_version=1),
    )
    row = db.get_conversation_by_id(cid)
    raw = row["assistant_startup_json"]
    v2_payload = _conversation_sync_payload(row)
    assert v2_payload["title"] == "Sync boundary"
    db.update_conversation(cid, {"title": "Renamed"}, row["version"])
    db.soft_delete_conversation(cid, db.get_conversation_by_id(cid)["version"])
    deleted = db.get_conversation_by_id(cid, include_deleted=True)
    db.restore_conversation(cid, deleted["version"])
    records = db.execute_query(
        "SELECT operation, payload FROM sync_log WHERE entity = ? AND entity_id = ? ORDER BY change_id",
        ("conversations", cid),
    ).fetchall()
    assert {record["operation"] for record in records} == {"create", "update", "delete"}
    for serialized in [json.dumps(v2_payload), *(record["payload"] for record in records)]:
        for forbidden in ("assistant_startup", "assistant_startup_json", "private-origin"):
            assert forbidden not in serialized
    assert db.get_conversation_by_id(cid)["assistant_startup_json"] == raw
