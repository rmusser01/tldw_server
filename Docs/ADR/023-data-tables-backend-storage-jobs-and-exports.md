# ADR-023: Data Tables Backend Storage, Jobs, And Exports

**Status:** Accepted
**Date:** 2026-06-07
**Backfilled from:** `Docs/Design/Data_Tables_Backend.md`
**Decision owner:** TASK-2272 confirmation and TASK-2273 backfill scope
**Related task:** TASK-2273
**Related spec/plan:** `Docs/ADR/inventory/2026-06-07-data-tables-confirmation-audit.md`

## Decision

Data Tables persist backend table state in per-user Media DB helpers, use Jobs-backed generation and regeneration through the `data_tables` worker, store bounded source snapshots for reproducible regeneration, expose table APIs by UUID while retaining numeric job IDs, and keep exports server-side through direct rendering or File Artifacts delegation.

## Context

Data Tables bridge user prompts, media/chat/RAG sources, LLM-generated structured output, durable table storage, job lifecycle controls, and downloadable exports. The backend needs one clear ownership model so table state, generation status, source snapshots, and export behavior do not drift across independent storage tables or competing worker abstractions.

TASK-2272 confirmed the current Data Tables backend behavior that bounds this ADR:

- Data Tables endpoints receive the per-user Media DB through `get_media_db_for_user`, while the worker resolves user-specific Media DB paths for sidecar execution.
- Media DB owns table metadata, selected source rows, source snapshots, column definitions, generated rows, user-visible table status, soft-delete/version metadata, and table UUIDs.
- Data Tables owner scoping is implemented through the existing owner/client filter used by Media DB helpers; there is not a dedicated `owner_user_id` column on every Data Tables row.
- Generate and regenerate routes create or reuse Media DB table/source state and enqueue core Jobs records in the `data_tables` domain with `job_type="data_table_generate"`.
- The Data Tables worker owns source resolution, bounded prompt construction, LLM adapter invocation, structured JSON parsing, column/row normalization, persistence, cancellation checks, and failure/status mirroring.
- RAG query sources store retrieval params plus bounded chunk snapshots so regeneration can use stored source state instead of re-running retrieval when a snapshot exists.
- Export routes either render content directly through `DataTableAdapter` or delegate generated-file metadata/export handling to File Artifacts using `file_type="data_table"`.
- Table routes and table response models use UUIDs externally and resolve to internal numeric IDs server-side. Job status/cancel routes and response fields still expose numeric `job_id` values, with optional job UUIDs.

This ADR is intentionally bounded. It does not claim all Data Tables operations are asynchronous, does not prove complete ownership validation for every current or future source adapter, does not treat snapshots as a full provenance ledger, does not decide File Artifacts internals, and does not cover frontend table editing.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Store Data Tables in a new standalone database | Would bypass existing per-user Media DB path/backend ownership, owner/client filtering, soft-delete/version conventions, and content-backend migration path. |
| Add a dedicated Data Tables export table | Current export behavior either directly renders server-side content or delegates generated-file metadata/export handling to File Artifacts, avoiding another export-status store. |
| Generate tables synchronously inside API handlers | Generation uses external source resolution and LLM calls, needs progress/cancellation/failure visibility, and already fits the user/admin-visible Jobs lifecycle model. |
| Let regeneration re-run RAG retrieval by default | Re-running retrieval would make regeneration sensitive to index drift and source changes. Stored snapshots give bounded reproducibility for the source state used by table generation. |
| Expose internal numeric table IDs in public table routes | Current table APIs use UUIDs externally and resolve numeric IDs server-side, matching the broader Media DB UUID pattern and avoiding leaking internal row IDs as table identity. |
| Claim every Data Tables-related ID must be a UUID | Current Jobs status and cancellation APIs still use numeric `job_id` values, so this ADR preserves that caveat instead of rewriting the Jobs API contract. |

## Consequences

Data Tables schema and persistence changes should keep Media DB as the owner of table metadata, source rows, source snapshots, column definitions, generated rows, table status, and table UUID identity unless a later ADR supersedes this one.

Generation and regeneration should continue to enter the durable worker path through Jobs records in the `data_tables` domain. The Data Tables worker should report progress, cancellation, completion, and failures through Jobs while mirroring user-visible table status into Media DB for table reads.

Source resolution changes must preserve the stored snapshot contract. New source types should decide what source state is stored for regeneration and should update the worker, endpoint/schema validation, and Data Tables tests together.

Export changes should remain server-side. Direct downloads can render from table content, while generated-file metadata and asynchronous export behavior should continue to use File Artifacts instead of introducing a competing Data Tables export store.

Public table APIs should remain UUID-first. Numeric table IDs stay internal to Media DB helper calls, while numeric Jobs identifiers remain an explicit compatibility caveat for job status and cancellation routes.

Wait-for-completion responses, direct download exports, source ownership depth, snapshot size limits, File Artifacts internals, and frontend editing workflows remain separate concerns rather than accepted claims in this ADR.

## Follow-up

- Use this ADR as the covering record for INV-025.
- Keep `Docs/ADR/inventory/2026-06-07-data-tables-confirmation-audit.md` as the evidence record and caveat boundary for this backfill.
- Create separate ADRs if File Artifacts export internals, source authorization guarantees, frontend editing ownership, or a future non-Media DB Data Tables storage backend becomes a durable architecture decision.
