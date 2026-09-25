"""Live PostgreSQL coverage for Persona visual-pack lifecycle lookups."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import (
    DatabaseBackendFactory,
)
from tldw_Server_API.app.core.DB_Management.chacha.persona_state_store import (
    PersonaStateStore,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Persona.visual_service import (
    PersonaVisualService,
    PersonaVisualServiceError,
)


@pytest.mark.unit
def test_pack_by_user_lookup_binds_postgres_false_values() -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []

    def execute_query(query: str, params: tuple[object, ...]) -> SimpleNamespace:
        calls.append((query, params))
        return SimpleNamespace(fetchone=lambda: {"id": "pack-1"})

    db = SimpleNamespace(
        backend_type=BackendType.POSTGRESQL,
        execute_query=execute_query,
        _persona_visual_pack_row_to_dict=lambda row: row,
    )

    result = PersonaStateStore(db).get_persona_visual_pack_for_user(
        pack_id="pack-1", user_id="user-1"
    )

    assert result is not None
    assert result["id"] == "pack-1"
    assert "p.deleted = ?" in calls[0][0]
    assert "pp.deleted = ?" in calls[0][0]
    assert calls[0][1] == ("pack-1", "user-1", False, False)


@pytest.mark.integration
def test_postgres_pack_by_user_lookup_allows_review_to_reach_validation(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(
        db_path=":memory:",
        client_id="persona-visual-postgres-lookup",
        backend=backend,
    )
    try:
        persona_id = db.create_persona_profile(
            {"user_id": "user-1", "name": "PostgreSQL Visual Persona"}
        )
        pack = db.create_persona_visual_pack(
            persona_id=persona_id,
            user_id="user-1",
            title="PostgreSQL Visual Pack",
        )

        assert db.get_persona_visual_pack_for_user(
            pack_id=str(pack["id"]), user_id="user-1"
        ) == pack
        with pytest.raises(PersonaVisualServiceError) as exc_info:
            PersonaVisualService(db).review_pack(
                pack_id=str(pack["id"]),
                user_id="user-1",
                reviewer_user_id="user-1",
                expected_version=int(pack["version"]),
            )

        assert exc_info.value.code == "invalid_renderer_contract"
    finally:
        db.close_connection()
        backend.get_pool().close_all()
