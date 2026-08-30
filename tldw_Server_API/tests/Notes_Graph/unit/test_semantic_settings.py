from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Notes_Graph.semantic_settings import SemanticIndexSettings

pytestmark = pytest.mark.unit


def test_semantic_settings_exposes_bounded_operator_controls() -> None:
    settings = SemanticIndexSettings(
        indexing_enabled=False,
        max_active_notes=12,
        max_stored_note_bytes=4_096,
        max_canonical_field_code_points=2_048,
        max_chunk_code_points=256,
        max_chunks_per_note=4,
        max_chunks_per_run=16,
        max_provider_input_bytes=2_048,
        max_provider_batch_inputs=4,
        max_provider_batch_bytes=8_192,
        max_provider_bytes_per_run=32_768,
        max_provider_requests_per_run=8,
        max_query_neighbors=6,
        max_query_vectors_per_call=3,
        max_cleanup_vectors_per_run=20,
        max_retries=2,
        retry_backoff_seconds=1,
        retry_max_backoff_seconds=4,
        pgvector_hnsw_max_scan_tuples=12,
        pgvector_allowed_dimensions=frozenset({384, 768}),
    )

    assert settings.indexing_enabled is False
    assert settings.max_provider_batch_bytes == 8_192
    assert settings.max_chunks_per_run == 16
    assert settings.max_query_vectors_per_call == 3
    assert settings.pgvector_hnsw_max_scan_tuples == 12
    assert settings.pgvector_allowed_dimensions == frozenset({384, 768})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_active_notes": 0},
        {"max_stored_note_bytes": 0},
        {"max_canonical_field_code_points": 0},
        {"max_chunk_code_points": 0},
        {"max_chunks_per_note": -1},
        {"max_chunks_per_run": 1_000_001},
        {"max_provider_input_bytes": 0},
        {"max_provider_batch_inputs": 0},
        {"max_provider_batch_bytes": 0},
        {"max_provider_bytes_per_run": 0},
        {"max_provider_requests_per_run": 0},
        {"max_query_neighbors": 0},
        {"max_query_vectors_per_call": 0},
        {"max_query_vectors_per_call": True},
        {"max_query_vectors_per_call": 257},
        {"max_cleanup_vectors_per_run": 0},
        {"max_retries": -1},
        {"retry_backoff_seconds": 0},
        {"retry_max_backoff_seconds": 0},
        {"retry_backoff_seconds": 8, "retry_max_backoff_seconds": 4},
        {"pgvector_hnsw_max_scan_tuples": 0},
        {"pgvector_hnsw_max_scan_tuples": True},
        {"pgvector_hnsw_max_scan_tuples": 100_001},
        {"pgvector_allowed_dimensions": frozenset()},
        {"pgvector_allowed_dimensions": frozenset({0})},
    ],
)
def test_semantic_settings_rejects_invalid_or_unbounded_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        SemanticIndexSettings(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_provider_input_bytes": 1_024, "max_provider_batch_bytes": 1_023},
        {"max_provider_batch_bytes": 2_048, "max_provider_bytes_per_run": 2_047},
        {"max_chunks_per_note": 5, "max_chunks_per_run": 4},
        {"max_provider_batch_inputs": 5, "max_chunks_per_run": 4},
        {"max_chunk_code_points": 257, "max_canonical_field_code_points": 256},
        {"max_query_vectors_per_call": 5, "max_chunks_per_note": 4},
        {"max_query_neighbors": 6, "pgvector_hnsw_max_scan_tuples": 5},
    ],
)
def test_semantic_settings_rejects_contradictory_cross_limits(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        SemanticIndexSettings(**kwargs)
