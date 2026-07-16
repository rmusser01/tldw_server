# Claims Extraction - Deep Dive

This doc explains how claims (concise, verifiable factual statements) are extracted and optionally verified in two contexts:
- Ingestion-time: Extract a small number of claims per chunk and store them for search/RAG workflows.
- Answer-time: Extract and verify claims from a generated answer using retrieval + NLI/LLM judging.

References:
- Atomic Proposition Simplification (APS): https://arxiv.org/pdf/2408.03325
- AttributionBench (faithfulness): https://osu-nlp-group.github.io/AttributionBench/
- Evaluation perspectives: https://arxiv.org/pdf/2408.07852

## Ingestion-Time Claims

Path: `tldw_Server_API.app.core.Claims_Extraction.ingestion_claims`

Flow (behind `ENABLE_INGESTION_CLAIMS`):
- Hooked inside the embeddings pipeline (`ChromaDB_Library`). After chunking a document, the system optionally extracts claims (`heuristic` by default) and stores them in the `Claims` SQL table.
- Optional embedding: when `CLAIMS_EMBED=True`, claim texts are embedded into a separate Chroma collection per user.
- API clients can now control ingestion-time claims extraction per request:
  - `perform_claims_extraction` - tri-state toggle. `true/false` overrides the server config; `null`/omitted falls back to `ENABLE_INGESTION_CLAIMS`.
  - `claims_extractor_mode` - optional per-request override (e.g., `heuristic`, `ner`, provider id). When absent, `CLAIM_EXTRACTOR_MODE` is used.
  - `claims_max_per_chunk` - optional per-request override (1-12). Defaults to `CLAIMS_MAX_PER_CHUNK` when omitted.
  These fields are exposed on `/media/add` as well as the `/process-*` helper endpoints so WebUI and API consumers can surface them directly in their UX.

Extractor modes:
- `heuristic` (default): fast sentence-based heuristic extraction.
- `llm` or a provider name (e.g., `openai`, `anthropic`): use the unified chat API to prompt an LLM for claim extraction. Prompts and provider/model can be configured via settings.
- `auto`: currently behaves like `llm` in this module; falls back to heuristics on failures.

Config (env or `[Claims]` in `Config_Files/config.txt`, see `tldw_Server_API.app.core.config.settings`):
- `ENABLE_INGESTION_CLAIMS`: bool
- `CLAIM_EXTRACTOR_MODE`: `heuristic|llm|auto|<provider>`
- `CLAIMS_MAX_PER_CHUNK`: int (default 3)
- `CLAIMS_EMBED`: bool (optional)
- `CLAIMS_EMBED_MODEL_ID`: str (optional)
- `CLAIMS_LLM_PROVIDER`, `CLAIMS_LLM_MODEL`, `CLAIMS_LLM_TEMPERATURE`: used for the LLM path
- `CLAIMS_JSON_PARSE_MODE`: `lenient|strict` (default `lenient`)
- `CLAIMS_ALIGNMENT_MODE`: `off|exact|fuzzy` (default `fuzzy`)
- `CLAIMS_ALIGNMENT_THRESHOLD`: float (default `0.75`)
- `CLAIMS_MONITORING_ENABLED`: enable claims telemetry counters
- `CLAIMS_ADAPTIVE_THROTTLE_ENABLED` plus thresholds:
  - `CLAIMS_ADAPTIVE_THROTTLE_LATENCY_MS`
  - `CLAIMS_ADAPTIVE_THROTTLE_ERROR_RATE`
  - `CLAIMS_ADAPTIVE_THROTTLE_BUDGET_RATIO`

Storage schema: see the package-native MediaDatabase claims tables (table `Claims` + `claims_fts`)
- FTS: `claims_fts` is content-backed and maintained via SQLite triggers (insert/update/delete).
- Helpers: `upsert_claims`, `get_claims_by_media`, `soft_delete_claims_for_media`, `rebuild_claims_fts`.

APIs and service:
- `GET /api/v1/claims/{media_id}` - list stored claims
- `POST /api/v1/claims/{media_id}/rebuild` - enqueue rebuild for one item
- `POST /api/v1/claims/rebuild/all` - enqueue rebuild for many items (policies: `missing|all|stale`)
- `POST /api/v1/claims/rebuild_fts` - rebuild FTS
- Background worker: `ClaimsRebuildService` (chunks content, extracts claims, stores/replaces; see `core/Claims_Extraction/claims_rebuild_service.py`).

## Answer-Time Claims (Extraction + Verification)

Path: `tldw_Server_API.app.core.Claims_Extraction.claims_engine`

High-level entry: `ClaimsEngine(analyze_fn).run(...)` returns:
```
{
  "claims": [
    {"id": "c1", "text": "...", "span": [s,e]|null, "label": "supported|refuted|nei", "confidence": 0.0-1.0,
     "evidence": [{"doc_id": "...", "snippet": "...", "score": 0.0}], "citations": [], "rationale": "..."}
  ],
  "summary": {"supported": n, "refuted": n, "nei": n, "precision": x, "coverage": y, "claim_faithfulness": z}
}
```

Extractor options (argument `claim_extractor`):
- `aps`: Use PropositionChunkingStrategy with the `gemma_aps` profile to produce APS-style atomic propositions (`max_size=1` ensures one proposition per chunk). Requires an LLM via `analyze_fn`.
- `ner`: NER-assisted sentence selection using spaCy (model from `CLAIMS_LOCAL_NER_MODEL`, default `en_core_web_sm`). Sentences with named entities are returned as claims. Falls back to LLM if NER unavailable.
- otherwise: LLM-based extractor (JSON of claims) with fallback to heuristics. Uses prompt_loader keys `ingestion/claims_extractor_system` and `ingestion/claims_extractor_prompt` when provided, with safe defaults if not.

Verifier: `HybridClaimVerifier`
- Retrieval: uses provided `retrieve_fn` to fetch context per claim, otherwise uses the base `documents` (top_k, default 5). Evidence snippets are collected (up to 3).
- Numeric/date boosting: documents containing numbers/dates present in the claim receive a small score bonus.
- Strategy (`claim_verifier`):
  - `hybrid` (default): NLI then fallback to LLM judge on low-confidence or unavailability.
  - `nli`: NLI only; returns `nei` when NLI is unavailable or below threshold (no LLM fallback).
  - `llm`: judge only; skips NLI entirely.
- NLI: local pipeline (default `roberta-large-mnli` or `RAG_NLI_MODEL[_PATH]`).
- LLM judge: prompts `analyze_fn` with a strict-JSON judge instruction to produce `{label, confidence, rationale}`.

Notes:
- `claim_extractor="auto"` in `ClaimsEngine.run` currently uses the LLM path by default with heuristic fallback. The ingestion-time module’s `auto` may behave differently (see above).
- `claims_concurrency` bounds parallel verifications (default 8; range 1-32).
- `nli_model` can be passed to override the default NLI model.

## Prompt Customization

Prompts can be overridden via files in `tldw_Server_API/Config_Files/Prompts/` and are loaded with `prompt_loader`:
- Ingestion-time LLM extractor: module=`ingestion`, keys=`claims_extractor_system`, `claims_extractor_prompt`.
- Proposition strategy (APS/claimify/generic) prompts: module=`chunking`, keys=`proposition_gemma_aps`, `proposition_claimify`, `proposition_generic`.

## Integration with RAG

- The RAG module re-exports the engine types via `core/RAG/rag_service/claims.py` for compatibility.
- Stored claims can be indexed (FTS/vectors) and retrieved to support fact-seeking queries or faithfulness checks.

## Observability and Structured Output

- Claims extraction/judging calls attempt `response_format` automatically:
  - `json_schema` when provider capabilities support it,
  - `json_object` otherwise,
  - and fallback to plain output when unsupported.
- Output parsing is centralized (`output_parser.py`) and records telemetry outcomes.
- Key counters:
  - `claims_response_format_selected_total`
  - `claims_output_parse_events_total`
  - `claims_fallback_total`
  - `claims_provider_budget_exhausted_total`
  - `claims_provider_throttled_total`
- Dashboards and alert thresholds are maintained in:
  - `Docs/Monitoring/claims_grafana_dashboard.json`
  - `Docs/Monitoring/claims_alerts_prometheus.yaml`
  - `Docs/Operations/Claims_Alerts_Runbook.md`
