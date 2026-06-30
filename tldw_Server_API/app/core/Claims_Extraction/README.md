# Claims_Extraction

Claims_Extraction extracts, parses, stores, clusters, reviews, monitors, and
verifies factual claims from generated answers and ingested content. It also
supports RAG streaming overlays so clients can see claim support/refutation
signals and evidence while an answer is being produced.

## Start Here

- Engine: `claims_engine.py`.
- Service/API persistence helpers: `claims_service.py`, `claims_utils.py`,
  `ingestion_claims.py`, and `claims_rebuild_service.py`.
- Parsing and validation: `output_parser.py`, `prompt_validation.py`,
  `span_alignment.py`, and `runtime_config.py`.
- Review/monitoring/clustering: `review_assignment.py`, `monitoring.py`,
  `claims_clustering.py`, `claims_embeddings.py`, and `claims_notifications.py`.
- API endpoint and schemas: `app/api/v1/endpoints/claims.py` and
  `app/api/v1/schemas/claims_schemas.py`.
- Tests: `tests/Claims/`.

## Responsibilities

- Extract claims with LLM-backed, heuristic, or configured extractor paths.
- Parse model output defensively, including fenced JSON and strict-parse modes.
- Verify claims against retrieved evidence, numeric/date heuristics, optional
  NLI/LLM judge paths, and citation offsets.
- Persist claims into Media DB claim tables and maintain FTS/rebuild health.
- Support clustering, review assignment, dashboard metrics, alert digests, and
  webhook/notification delivery.

## Module Map

- `claims_engine.py` defines extractors, verifiers, and orchestration.
- `claims_service.py` provides API-facing service operations.
- `ingestion_claims.py` extracts claims during ingestion workflows.
- `claims_rebuild_service.py` rebuilds claim indexes and health state.
- `claims_embeddings.py` and `claims_clustering.py` support semantic grouping.
- `monitoring.py`, `claims_notifications.py`, and `verification_report.py`
  support reporting and alerting.
- `budget_guard.py` limits expensive claim work.

## How It Connects

- RAG streaming emits claim overlay events from `rag_unified.py`.
- Media DB owns persisted claims, FTS, clusters, and review records.
- Watchlists and monitoring can deliver claim alert notifications.
- Embeddings/Chroma support semantic claim clustering and refresh flows.

## Extension Points

- Add extractors through the extractor registry/catalog and cover strict parsing
  plus fallback behavior in tests.
- Add verifiers by keeping evidence selection, label semantics, and confidence
  outputs explicit.
- Add dashboards or alerts through service/query helpers rather than embedding
  direct SQL in endpoint handlers.

## Testing

- Engine and parsing: `tests/Claims/test_claims_engine_modes.py`,
  `tests/Claims/test_claims_output_parser.py`, and
  `tests/Claims/test_claims_prompt_validation.py`.
- API/service/persistence: `tests/Claims/test_claims_endpoints_api.py`,
  `tests/Claims/test_claims_service_backend_selection.py`, and
  `tests/Claims/test_claims_utils_persistence.py`.
- Rebuild/monitoring/review: `tests/Claims/test_claims_rebuild_service_failure.py`,
  `tests/Claims/test_claims_monitoring_api.py`, and
  `tests/Claims/test_claims_review_api.py`.
- Clustering/embeddings: `tests/Claims/test_claims_clustering_embeddings.py`
  and `tests/Claims/test_claim_embeddings_chroma.py`.

## Gotchas

- Citation offsets are best-effort and sensitive to text normalization. Keep span
  alignment tests close to parser changes.
- Strict-parse mode should fail loudly for malformed model output; fallback mode
  should remain useful without hiding parser defects in tests.
