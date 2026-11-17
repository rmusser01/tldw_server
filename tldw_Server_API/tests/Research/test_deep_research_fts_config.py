"""Deep Research can use actual full-text retrieval without vector/model startup."""

import pytest

from tldw_Server_API.app.core.Research.providers import config as configuration
from tldw_Server_API.app.core.Research.providers import local

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        ("false", False),
        ("yes", True),
        ("no", False),
        ("1", True),
        ("0", False),
        ("on", True),
        ("off", False),
    ],
)
def test_operator_local_vector_setting_is_resolved(monkeypatch, value, expected):
    monkeypatch.setattr(
        configuration,
        "get_config_value",
        lambda section, key, default=None: value if key == "deep_research_local_use_vector" else default,
    )
    resolved = configuration.resolve_provider_config({"local": {"use_vector": not expected}})
    assert resolved["local"]["use_vector"] is expected  # nosec B101 -- Regression checks the configured retrieval/model boundary.


def test_operator_local_vector_setting_rejects_invalid_value(monkeypatch):
    monkeypatch.setattr(
        configuration,
        "get_config_value",
        lambda section, key, default=None: "invalid" if key == "deep_research_local_use_vector" else default,
    )
    with pytest.raises(ValueError, match="deep_research_local_use_vector"):
        configuration.resolve_provider_config(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_local_provider_shares_vector_policy_with_constructor_and_query(monkeypatch, tmp_path, enabled):
    captured = {}
    monkeypatch.setattr(local, "is_test_mode", lambda: False)
    monkeypatch.setattr(local.DatabasePaths, "get_media_db_path", lambda user: tmp_path / "media.db")

    class Retriever:
        def __init__(self, db_paths, user_id, **kwargs):
            captured["constructor"] = kwargs["media_config"]

        async def retrieve(self, query, *, config, **kwargs):
            captured["query"] = config
            return []

        def close(self):
            captured["closed"] = True

    await local.LocalResearchProvider(retriever_cls=Retriever).search(
        focus_area="Rowan",
        query="Observatory",
        owner_user_id="7",
        config={"sources": ["media_db"], "top_k": 2, "use_vector": enabled},
    )
    assert captured["constructor"] is captured["query"]  # nosec B101 -- Regression checks the configured retrieval/model boundary.
    assert captured["query"].use_vector is enabled  # nosec B101 -- Regression checks the configured retrieval/model boundary.
    assert captured["query"].use_fts is True  # nosec B101 -- Regression checks the configured retrieval/model boundary.
    assert captured["closed"] is True  # nosec B101 -- Regression checks the configured retrieval/model boundary.


@pytest.mark.asyncio
async def test_media_constructor_and_real_fts_never_enter_vector_path(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database, get_media_repository
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
    from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
        MediaDBRetriever,
        MultiDatabaseRetriever,
        RetrievalConfig,
    )
    from tldw_Server_API.app.core.RAG.rag_service.types import DataSource

    def forbidden_initialization(*args, **kwargs):
        raise AssertionError("FTS constructor attempted vector initialization")

    async def forbidden_query(*args, **kwargs):
        raise AssertionError("FTS query attempted vector retrieval")

    monkeypatch.setattr(MediaDBRetriever, "_initialize_vector_store", forbidden_initialization)
    monkeypatch.setattr(MediaDBRetriever, "_retrieve_vector", forbidden_query)
    path = tmp_path / "media.db"
    with scoped_context(user_id=1, is_admin=False):
        media = create_media_database(client_id="1", db_path=path)
        retriever = None
        try:
            get_media_repository(media).add_media_with_keywords(
                url="research://rowan",
                title="Rowan Observatory",
                media_type="document",
                content="Rowan Observatory telescope is Selene.",
                keywords=[],
                overwrite=False,
            )
            settings = RetrievalConfig(use_fts=True, use_vector=False, max_results=2)
            retriever = MultiDatabaseRetriever(
                {"media_db": str(path)},
                user_id="1",
                media_db=media,
                media_config=settings,
            )
            documents = await retriever.retrieve("Rowan", sources=[DataSource.MEDIA_DB], config=settings)
            assert len(documents) == 1  # nosec B101 -- Regression checks the configured retrieval/model boundary.
            assert "Selene" in documents[0].content  # nosec B101 -- Regression checks the configured retrieval/model boundary.
        finally:
            if retriever is not None:
                retriever.close()
            media.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["false", 0, None])
async def test_invalid_local_vector_value_cannot_construct_retriever(monkeypatch, tmp_path, value):
    monkeypatch.setattr(local, "is_test_mode", lambda: False)
    monkeypatch.setattr(local.DatabasePaths, "get_media_db_path", lambda user: tmp_path / "media.db")

    class ForbiddenRetriever:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Invalid vector policy reached construction")

    with pytest.raises(ValueError, match="use_vector"):
        await local.LocalResearchProvider(retriever_cls=ForbiddenRetriever).search(
            focus_area="Rowan",
            query="Observatory",
            owner_user_id="7",
            config={"sources": ["media_db"], "use_vector": value},
        )
