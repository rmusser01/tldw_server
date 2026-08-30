"""Bounded operator controls for Notes semantic indexing."""

from __future__ import annotations

from dataclasses import dataclass

_HARD_MAXIMUMS = {
    "max_active_notes": 100_000,
    "max_stored_note_bytes": 10_000_000,
    "max_canonical_field_code_points": 2_000_000,
    "max_chunk_code_points": 8_192,
    "max_chunks_per_note": 2_000,
    "max_chunks_per_run": 100_000,
    "max_provider_input_bytes": 1_000_000,
    "max_provider_batch_inputs": 2_048,
    "max_provider_batch_bytes": 16_000_000,
    "max_provider_bytes_per_run": 1_000_000_000,
    "max_provider_requests_per_run": 10_000,
    "max_query_neighbors": 100,
    "max_query_vectors_per_call": 256,
    "max_query_candidates_per_call": 204_800,
    "query_candidate_oversampling_factor": 8,
    "max_cleanup_vectors_per_run": 100_000,
    "max_retries": 10,
    "retry_backoff_seconds": 3_600,
    "retry_max_backoff_seconds": 86_400,
    "pgvector_hnsw_max_scan_tuples": 100_000,
    "pgvector_dimension": 2_000,
}


@dataclass(frozen=True, slots=True)
class SemanticIndexSettings:
    """Validated, bounded semantic-index operator policy."""

    indexing_enabled: bool = True
    max_active_notes: int = 10_000
    max_stored_note_bytes: int = 1_000_000
    max_canonical_field_code_points: int = 250_000
    max_chunk_code_points: int = 480
    max_chunks_per_note: int = 200
    max_chunks_per_run: int = 10_000
    max_provider_input_bytes: int = 16_384
    max_provider_batch_inputs: int = 128
    max_provider_batch_bytes: int = 1_048_576
    max_provider_bytes_per_run: int = 67_108_864
    max_provider_requests_per_run: int = 1_000
    max_query_neighbors: int = 50
    max_query_vectors_per_call: int = 16
    max_query_candidates_per_call: int = 1_600
    query_candidate_oversampling_factor: int = 2
    max_cleanup_vectors_per_run: int = 10_000
    max_retries: int = 3
    retry_backoff_seconds: int = 1
    retry_max_backoff_seconds: int = 60
    pgvector_hnsw_max_scan_tuples: int = 10_000
    pgvector_allowed_dimensions: frozenset[int] = frozenset({384, 768, 1_024, 1_536})

    def __post_init__(self) -> None:
        if type(self.indexing_enabled) is not bool:
            raise TypeError("indexing_enabled must be a boolean")
        for field_name, hard_maximum in _HARD_MAXIMUMS.items():
            if field_name == "pgvector_dimension":
                continue
            value = getattr(self, field_name)
            if type(value) is not int:
                raise TypeError(f"{field_name} must be an integer")
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            if value > hard_maximum:
                raise ValueError(f"{field_name} exceeds its hard maximum")
        if self.retry_backoff_seconds > self.retry_max_backoff_seconds:
            raise ValueError("retry_backoff_seconds cannot exceed retry_max_backoff_seconds")
        for smaller, larger in (
            ("max_chunk_code_points", "max_canonical_field_code_points"),
            ("max_chunks_per_note", "max_chunks_per_run"),
            ("max_provider_batch_inputs", "max_chunks_per_run"),
            ("max_provider_input_bytes", "max_provider_batch_bytes"),
            ("max_provider_batch_bytes", "max_provider_bytes_per_run"),
            ("max_query_vectors_per_call", "max_chunks_per_note"),
            ("max_query_neighbors", "pgvector_hnsw_max_scan_tuples"),
        ):
            if getattr(self, smaller) > getattr(self, larger):
                raise ValueError(f"{smaller} cannot exceed {larger}")
        candidates_per_query = (
            self.max_query_neighbors * self.query_candidate_oversampling_factor
        )
        if candidates_per_query > self.pgvector_hnsw_max_scan_tuples:
            raise ValueError(
                "maximum candidates per query cannot exceed pgvector_hnsw_max_scan_tuples"
            )
        if (
            self.max_query_vectors_per_call * candidates_per_query
            > self.max_query_candidates_per_call
        ):
            raise ValueError(
                "maximum query candidate product cannot exceed max_query_candidates_per_call"
            )
        if not isinstance(self.pgvector_allowed_dimensions, frozenset):
            raise TypeError("pgvector_allowed_dimensions must be a frozenset")
        if not self.pgvector_allowed_dimensions:
            raise ValueError("pgvector_allowed_dimensions cannot be empty")
        for dimension in self.pgvector_allowed_dimensions:
            if type(dimension) is not int:
                raise TypeError("pgvector dimensions must be integers")
            if dimension <= 0 or dimension > _HARD_MAXIMUMS["pgvector_dimension"]:
                raise ValueError("pgvector dimensions must be bounded positive integers")


DEFAULT_SEMANTIC_INDEX_SETTINGS = SemanticIndexSettings()


__all__ = ["DEFAULT_SEMANTIC_INDEX_SETTINGS", "SemanticIndexSettings"]
