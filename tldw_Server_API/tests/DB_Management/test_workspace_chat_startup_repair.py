"""Current-schema repair cannot change accepted Workspace startup authority."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.tests.DB_Management.test_conversation_assistant_startup import db_factory as db_factory
from tldw_Server_API.tests.DB_Management.test_workspace_assistant_creation_atomic import creation_db as creation_db
from tldw_Server_API.tests.DB_Management.test_workspace_chat_startup_acceptance import _start
from tldw_Server_API.tests.DB_Management.test_workspace_chat_startup_lifecycle import _receipt

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("selection", ["inherit", "none"])
@pytest.mark.parametrize("missing", ["kind", "id", "both"])
def test_current_schema_character_repair_preserves_accepted_startup(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB],
    selection: str, missing: str,
) -> None:
    """Reopening repairs historical Character omissions without touching strict chats."""
    payload = {
        "scope_type": "workspace", "workspace_id": "ws", "workspace_assistant_selection": selection,
    }
    if selection == "inherit":
        payload["workspace_assistant_default_version"] = 2
    first = _start(creation_db, payload=payload)
    receipt = _receipt(creation_db)
    character = creation_db.add_character_card({"name": "Historical Character"})
    legacy = creation_db.add_conversation({"title": "Legacy", "character_id": character})
    with creation_db.transaction() as conn:
        conn.execute(
            "UPDATE conversations SET assistant_kind = ?, assistant_id = ? WHERE id = ?",
            (None if missing in ("kind", "both") else "character",
             None if missing in ("id", "both") else str(character), legacy),
        )
    creation_db.close_all_connections()
    reopened = db_factory()
    with reopened.transaction():
        repaired = reopened.get_conversation_by_id(legacy)
        accepted = reopened.get_conversation_by_id(first.conversation["id"])
    assert (repaired["assistant_kind"], repaired["assistant_id"]) == ("character", str(character))
    assert _receipt(reopened) == receipt
    assert accepted["assistant_startup_json"] == first.conversation["assistant_startup_json"]
    assert accepted["assistant_id"] == first.conversation["assistant_id"]
    assert _start(reopened, payload=payload).replayed
