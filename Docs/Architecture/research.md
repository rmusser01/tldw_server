# Research jobs engine

This document describes the phased deep-research engine: sessions and runs, phase jobs executed on the Jobs substrate with fencing-token leases (the semantics the chatbook's engine mirrors), checkpoints and human review, the broker's lanes, the academic discovery catalog, and the artifact store.

## Authoritative files

All paths under `tldw_Server_API/` unless noted.

| Concern | File | Key symbols |
| --- | --- | --- |
| Phase jobs | `app/core/Research/jobs.py` | `enqueue_research_phase_job`, `handle_research_phase_job`, per-phase handlers, halt/cancel helpers |
| Service | `app/core/Research/service.py` | `ResearchService` — session CRUD, `approve_checkpoint`, pause/resume/cancel, bundling; executable/checkpoint phase sets; terminal statuses `completed|failed|cancelled` |
| Worker | `app/core/Research/jobs_worker.py` | `run_research_jobs_worker` — WorkerSDK loop with env-driven lease config |
| Artifacts | `app/core/Research/artifact_store.py` | `ResearchArtifactStore` — versioned, checksummed, atomic writes (json/jsonl/text) |
| Broker | `app/core/Research/broker.py` | `ResearchBroker` — local/academic/web lane selection, dedupe, evidence normalization |
| Discovery | `app/core/Research/discovery/{catalog,service}.py` | `ResearchSourceCatalog` (`research-discovery-v1`, max 8 selected sources), provider router, OA resolver candidates |
| Support | `Research/{planner,synthesizer,exporter,streaming,checkpoint_service,chat_handoff,limits,models}.py` | plan drafting, synthesis, packaging, reconnect-safe SSE snapshots, checkpoint patching, limits |
| Jobs substrate | `app/core/Jobs/manager.py`, `worker_sdk.py` | acquisition with fencing tokens, lease renewal, expiry recovery (see [jobs-background.md](./jobs-background.md)) |
| Routes | `app/api/v1/endpoints/research_runs.py`, `research_discovery.py`, `research.py` | `/research/runs*` (+ events SSE), `/research/sources` + discovery search, deprecated academic routes and the live `POST /research/websearch` |
| Websearch | `app/core/Web_Scraping/WebSearch_APIs.py` | production `generate_and_search` / `analyze_and_aggregate` (the legacy twin under `core/WebSearch/` is non-production) |

## Job lifecycle with lease (dataflow)

1. `POST /research/runs` creates a per-user session (ResearchSessionsDB) and enqueues the first phase: a job with domain `research`, queue `default`, type `research_phase`, priority 5, and idempotency key `research:{session}:{phase}:{checkpoint}:{policy_version}`.
2. The worker acquires through the Jobs manager: expired processing jobs are first atomically requeued or failed per the retry budget, then the claim sets the holder, lease expiry, and increments a **fencing token** — the claim/expiry/requeue-with-attempt-budget semantics the chatbook's ADR-070 mirrors.
3. The phase handler dispatches: `drafting_plan` → plan artifact (+ optional `awaiting_plan_review` pause); `collecting` → broker lanes → source registry/evidence artifacts (+ `awaiting_source_review`); `synthesizing` → claims/summaries (+ `awaiting_outline_review`); `packaging` → final bundle. Halt-for-control checks run before and inside each phase.
4. Phase transitions record progress and persist run events for reconnect-safe SSE streaming.
5. Checkpoint approval validates the expected phase, applies typed patches (plan/sources/outline), and enqueues the next phase job; blocked control states refuse approval.
6. Completion acks with worker+lease identity; leases renew through the renewal APIs; adaptive leases exist but only activate under their own enable flag.

## Discovery catalog

`default_source_catalog()`: openalex (priority 10), semantic_scholar, crossref, arxiv (ingestable), pubmed, zenodo, figshare, osf — each with adapter, site hosts, full-text resolvability, ingestability, and trust notes. Selection enforces the max-sources cap; the discovery service fans out via the provider router, merges OpenAccess resolver candidates, sanitizes warnings, and persists request/response snapshots.

## Config keys (exact)

`RESEARCH_JOBS_WORKER_ID`, `RESEARCH_JOBS_LEASE_SECONDS` (fallback `JOBS_LEASE_SECONDS`, default 60), `RESEARCH_JOBS_RENEW_JITTER_SECONDS` (5), `RESEARCH_JOBS_RENEW_THRESHOLD_SECONDS` (10), `RESEARCH_SESSIONS_DB_PATH` / `RESEARCH_OUTPUTS_DIR` (single-user overrides); Jobs keys in [jobs-background.md](./jobs-background.md).

## Boundaries

- Research phases never bypass Jobs; the legacy academic routes use Third_Party helpers outside the phase pipeline.
- Artifacts are written only through the artifact store into per-user outputs dirs with DB manifest rows; the read allowlist is enforced by the service.
- Research shares the `default` queue — queue-level pause/drain affects it.

## Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| Unsupported phase/job_type | Job failure |
| Session deleted mid-flight | Typed key error ends the job |
| Cancel before/during a phase | Session halted; active job cancelled best-effort |
| Expired lease | Atomic requeue if retry budget remains, else terminal fail (double-attempt guarded) |
| Checkpoint approval in the wrong phase | Rejected by state validation |

## Verified gotchas

1. `app/core/WebSearch/README.md` line anchors for `generate_and_search`/`analyze_and_aggregate` are stale (actual: :533/:695 in `Web_Scraping/WebSearch_APIs.py`).
2. `core/WebSearch/Web_Search.py` is a legacy twin of the production module — production delegates to `Web_Scraping` only.
3. The research worker always passes lease 60, so adaptive leases never engage for research.
4. The module README under `core/Research/` is fresh and accurate — prefer it over older docs.

## Governing decisions

ADR-003 (Jobs vs Scheduler), ADR-007 (research workspace canonical first slice), ADR-051 (retry backoff), ADR-058 (jobs completion row identity). The chatbook counterpart is `research-workspace.md` in the tldw_chatbook repo's architecture reference.

## Related docs

- [jobs-background.md](./jobs-background.md) — the substrate
- [rag.md](./rag.md) — the deep-search pipeline phases call
