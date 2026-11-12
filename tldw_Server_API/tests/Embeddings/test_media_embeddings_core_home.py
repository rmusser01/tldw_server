import pytest

pytestmark = pytest.mark.unit
MOVED = ("generate_embeddings_for_media", "get_media_content", "chunk_media_content", "_resolve_model_provider",
         "_allow_zero_embeddings_for_media", "_user_embedding_config", "_embeddings_jobs_backend")


def test_workers_and_the_endpoint_share_the_core_helpers() -> None:
    from tldw_Server_API.app.core.Embeddings import media_embeddings as core
    from tldw_Server_API.app.core.Embeddings.services import jobs_worker

    assert jobs_worker.generate_embeddings_for_media is core.generate_embeddings_for_media
    assert jobs_worker.FALLBACK_EMBEDDING_MODEL == core.FALLBACK_EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"


def test_the_endpoint_no_longer_defines_the_helpers() -> None:
    from tldw_Server_API.app.api.v1.endpoints import media_embeddings as endpoint

    assert [name for name in MOVED if hasattr(endpoint, name)] == []  # a stale endpoint patch raises instead of missing
