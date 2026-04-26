from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_rag_pipeline


@pytest.mark.asyncio
async def test_cleanup_ephemeral_collections_sanitizes_item_failure(monkeypatch):
    deleted_collections: list[str] = []
    marked_deleted: list[str] = []

    class _FakeDB:
        def list_expired_ephemeral_collections(self):
            return ["cleanup_success", "leaky_collection"]

        def mark_ephemeral_deleted(self, name: str):
            marked_deleted.append(name)

    class _FakeService:
        db = _FakeDB()

    class _FakeAdapter:
        async def initialize(self):
            return None

        async def delete_collection(self, name: str):
            if name == "leaky_collection":
                raise RuntimeError("vector backend exploded at /private/db/chroma")
            deleted_collections.append(name)

    monkeypatch.setattr(
        evaluations_rag_pipeline,
        "get_unified_evaluation_service_for_user",
        lambda _user_id: _FakeService(),
    )
    monkeypatch.setattr(
        evaluations_rag_pipeline,
        "create_from_settings_for_user",
        lambda _settings, _user_id: _FakeAdapter(),
    )

    response = await evaluations_rag_pipeline.cleanup_ephemeral_collections(
        current_user=SimpleNamespace(id=7, id_str="tenant-7"),
    )

    assert response.expired_count == 2
    assert response.deleted_count == 1
    assert response.errors == ["Collection cleanup failed"]
    assert deleted_collections == ["cleanup_success"]
    assert marked_deleted == ["cleanup_success"]
    assert "vector backend exploded" not in str(response)
    assert "/private/db/chroma" not in str(response)
    assert "leaky_collection" not in str(response.errors)
