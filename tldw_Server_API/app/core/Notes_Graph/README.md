# Notes Graph

Notes_Graph builds bounded authoritative graph views over notes using manual note links, wikilinks, backlinks, tag relationships, source relationships, and recency constraints. It also provides a small in-memory cache, a Cytoscape formatter, and an optional reviewable suggestion pipeline. Generated suggestions remain provisional until an explicit accepted decision creates a canonical link or tag relationship.

## Start Here

- `graph_service.py` builds note graph responses with node, edge, radius, tag, source, and time filters.
- `wikilink_parser.py` extracts supported note id wikilinks from note content.
- `projection_service.py` maintains persistent owner-scoped wikilink projections and rebuild state.
- `graph_cache.py` provides a TTL cache for graph responses.
- `formatters.py` converts graph responses to Cytoscape-compatible JSON.
- `suggestion_api.py`, `suggestion_jobs.py`, and `suggestion_service.py` own nested suggestion admission, Jobs integration, generation, publication, and decisions.
- `suggestion_retrieval.py`, `suggestion_content.py`, and `suggestion_generation.py` enforce bounded lexical retrieval, evidence, prompt, and output contracts.
- `suggestion_maintenance.py` reconciles Jobs receipts, cancellation, publication, acceptance leases, and retention.
- Related API surface: `tldw_Server_API/app/api/v1/endpoints/notes_graph.py`.
- Suggestion API surface: `tldw_Server_API/app/api/v1/endpoints/notes_graph_suggestions.py`.
- Related schemas: `tldw_Server_API/app/api/v1/schemas/notes_graph.py`.
- Suggestion API reference: `Docs/API/Notes_Graph_Suggestions.md`.
- Related tests: `tldw_Server_API/tests/Notes_Graph/`, including `evaluation/` for privacy and deterministic quality gates.

## Responsibilities

- Read explicit manual links from canonical `notes.link` product/Sync state.
- Parse `[[id:<UUID>]]` wikilinks from note text into deterministic local projections; backlinks are the reverse view of the same projection.
- Build live-only note graph nodes and edges for manual links, wikilinks, backlinks, tag membership, and source membership.
- Enforce graph caps for node count, edge count, and per-node degree.
- Support radius-limited graph expansion, neighbor lookups, and revision-bound keyset orphan pages.
- Cache graph responses only by canonical dataset, graph revision, parser version, and normalized request.
- Format graph responses for Cytoscape consumers.
- Retrieve at most 30 owner-scoped lexical candidates and expose at most five related-note and five tag suggestions per run.
- Keep staged and pending suggestions outside authoritative graph responses; only accepted decisions use existing Sync-aware coordinators.
- Bind one provider attempt to capability disclosure, set Jobs `max_retries=0`, and publish only after an exact owner-scoped terminal Job receipt is verified.

## Module Map

- `graph_service.py`: graph expansion, pruning, filtering, edge construction, and metrics hooks.
- `wikilink_parser.py`: supported wikilink extraction.
- `projection_service.py`: bounded dirty-note processing, parser-version rebuilds, and exact projection repair.
- `graph_cache.py`: thread-safe TTL cache with max-key eviction.
- `formatters.py`: Cytoscape response conversion.
- `suggestion_capabilities.py`: provider/model, data-boundary, outbound-category, limit, and ETag disclosure.
- `suggestion_api.py`: idempotent nested API orchestration and opaque cursor handling.
- `suggestion_jobs.py`: `graph-suggestions` admission, cancellation, and safe Jobs payload/result contracts.
- `suggestion_service.py`: one-attempt worker execution and receipt-gated publication.
- `suggestion_decisions.py`: fenced accept/reject/reset operations through canonical link and keyword coordinators.
- `suggestion_maintenance.py`: bounded reconciliation and cleanup.
- `__init__.py`: package marker.

## How It Connects

- `notes_graph.py` exposes graph routes under the notes API surface: `/notes/graph`, `/notes/graph/orphans`, `/notes/{note_id}/neighbors`, `POST /notes/{note_id}/links`, and list/detail/PATCH/DELETE/restore operations under `/notes/links`.
- The endpoint uses ChaChaNotes DB dependencies, AuthNZ permissions, token-scope guards, and rate limiting.
- `dataset_id` is optional. Active Sync resolves omission to the one active default-personal Notes dataset and rejects any other supplied dataset; inactive omission preserves the legacy product path.
- Manual links come from owner-scoped canonical link rows. Derived links come from persisted projection rows, not read-time parsing. Tag and source nodes remain compatible projections.
- Environment variables such as `NOTES_GRAPH_ENABLED`, `NOTES_GRAPH_MAX_NODES`, `NOTES_GRAPH_MAX_EDGES`, `NOTES_GRAPH_MAX_DEGREE`, and cache settings tune runtime behavior.
- Suggestion routes are nested below `/notes/{note_id}/graph/suggestions`. They require `notes.graph.read`, `notes.graph.suggest`, and token scope `notes`; acceptance additionally checks the canonical link or keyword mutation permission required by the suggestion kind.
- Provider disclosure is authoritative. A boundary of `unknown` is treated as external, and generation requires the disclosed ETag in `If-Match` plus a bounded `Idempotency-Key`.

## Extension Points

- Add an edge type in `graph_service.py`, schemas, and endpoint parsing together.
- Change wikilink syntax in `wikilink_parser.py` and update parser tests.
- Add response formats in `formatters.py` and route handling in `notes_graph.py`.
- Enable or tune caching by injecting `GraphCache` where the service is constructed.
- Adjust graph caps in `graph_service.py` and verify pruning behavior.

## Testing

- Unit tests for the parser, cache, and graph service live under `tldw_Server_API/tests/Notes_Graph/unit/`.
- Endpoint integration coverage for `/graph` and `/neighbors` lives in `tldw_Server_API/tests/Notes_Graph/integration/test_graph_endpoint.py`.
- Suggestion API, worker, persistence, privacy, and quality coverage lives under `tldw_Server_API/tests/Notes_Graph/` and `tldw_Server_API/tests/Services/test_notes_graph_suggestions_workers.py`.

## Gotchas

- The parser intentionally supports `[[id:<UUID>]]` links, not arbitrary title links.
- Manual-only graph reads remain available while a derived projection rebuild is pending; derived-edge and orphan reads return retryable 503 until the projection is current.
- Trashing a note hides its incident manual and derived edges without deleting canonical link history; restoring the note makes those edges visible again when both endpoints are live.
- Graph cursors are revision-bound pagination hints, never authorization tokens. Authorization and current revision are resolved before cache or cursor use.
- Suggestion cursors are also opaque bounded hints. They are bound to owner, dataset, note, and filters and never grant authority.
- Jobs payloads/results, run rows, operation receipts, events, logs, and metric labels must not contain note text, evidence excerpts, prompts, provider responses, rationales, proposed tags, candidate IDs, credentials, or raw provider errors.
- Operation receipts retain bounded replay state for 90 days unless hard note/user deletion cascades it. Publication recovery requires Jobs terminal receipts for at least 30 days.
- Radius 2 requests apply stricter built-in caps than caller-supplied maximums.
- The graph feature can be disabled with `NOTES_GRAPH_ENABLED`.
