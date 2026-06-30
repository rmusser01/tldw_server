# Phase 4 Parking Lot Triage

**Date:** 2026-04-25

**Status:** Triage complete; Phase 4 remains deferred until Phase 2/3 closeout is stable.

## Purpose

Keep Phase 4 visible without letting it preempt Phase 2 and Phase 3 closeout. This is a planning artifact only. It records first-pass owners, prerequisites, and static hotspot signals for each Phase 4 item.

## Global Phase 4 Gate

Do not start Phase 4 implementation until:

- Phase 2 closeout PRs are merged or explicitly accepted as stable.
- PR `#1125` is merged or explicitly accepted as stable.
- Phase 3 helper contracts are accepted.
- The `skills` pilot has either completed or been intentionally paused with documented blockers.

## Recommended Phase 4 Order

1. Phase 4.5 API versioning strategy alignment.
2. Phase 4.6 OpenAPI contract testing.
3. Phase 4.1 coverage ratchet baseline.
4. Phase 4.2 deployment docs refresh.
5. Phase 4.3 DB file decomposition.
6. Phase 4.4 endpoint file decomposition.

Rationale:

- API versioning and OpenAPI contract tests directly depend on Phase 3 envelope and pagination contracts.
- Coverage ratchet needs stable route and helper boundaries.
- Deployment docs can proceed once the current onboarding/CI gates stop moving.
- Decomposition is large and should wait until contract churn slows.

## Phase 4.1 Coverage Ratchet To 25%

Current signals:

- `pyproject.toml` includes `pytest-cov`.
- `apps/tldw-frontend/package.json` includes a frontend coverage script.
- No fresh backend coverage baseline was run in this triage pass.

Required planning before implementation:

- Define backend-only baseline command.
- Define frontend-only baseline command.
- Decide whether the ratchet applies to global repo coverage or touched-scope coverage.
- Decide whether slow/external suites are excluded from the baseline.
- Record current coverage before setting any threshold.

Recommended first artifact:

- `Docs/superpowers/plans/2026-04-25-phase4-1-coverage-ratchet-baseline-plan.md` (created)

Measurement handoff packet:

- `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-coverage-ratchet-measurement-packet.md` (created)

Do not do:

- Add a 25% gate before measuring the current baseline.
- Mix coverage ratchet changes with endpoint or DB refactors.

## Phase 4.2 Deployment Docs

Current signals:

- Deployment docs exist under `Docs/Deployment/`.
- Getting Started profiles exist for local, Docker single-user, Docker multi-user + Postgres, CPU audio setup, GPU audio setup, and HA guide work.
- Published docs mirrors exist under `Docs/Published/`.
- The tracker already has onboarding and docs gates in active CI.

Required planning before implementation:

- Inventory canonical docs versus published mirrors.
- Decide whether edits target source docs only or also `Docs/Published`.
- Map deployment modes:
  - local single-user
  - Docker single-user + WebUI
  - Docker multi-user + Postgres
  - production/horizontal scaling
  - offline/air-gapped
  - sidecar workers
- Identify doc owner review needs before edits.

Recommended first artifact:

- `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-deployment-docs-inventory.md` (created)

Draft refresh plan:

- `Docs/superpowers/plans/2026-04-25-phase4-2-deployment-docs-refresh-plan.md` (created)

Do not do:

- Rewrite deployment docs while onboarding CI is actively failing on unrelated PRs.
- Update published mirrors without confirming the docs publishing flow.

## Phase 4.3 Decompose Remaining Large DB Files

Static line-count snapshot from `wc -l tldw_Server_API/app/core/DB_Management/*.py` on 2026-04-25.

Top DB hotspots:

| File | Lines | First-pass risk |
| --- | ---: | --- |
| `ChaChaNotes_DB.py` | 32696 | Very high. Broad user-data surface; decompose only with extensive focused tests. |
| `Kanban_DB.py` | 8626 | High. Workflow/state behavior and soft-delete semantics. |
| `PromptStudioDatabase.py` | 7181 | High. Prompt Studio schema and jobs coupling. |
| `Collections_DB.py` | 5377 | High. Active local dirty work exists in this workspace; avoid touching here casually. |
| `Guardian_DB.py` | 3297 | Medium-high. Safety/moderation data; security-sensitive. |
| `Watchlists_DB.py` | 3178 | Medium-high. Scheduler/jobs integration. |
| `Evaluations_DB.py` | 2943 | Medium. Eval schemas and batch behavior. |
| `Prompts_DB.py` | 2881 | Medium. Prompt library compatibility. |
| `Workflows_DB.py` | 2871 | Medium-high. Workflow execution and scheduler dependencies. |
| `ManuscriptDB.py` | 2471 | Medium. Writing/manuscript domain coupling. |

Recommended first target:

- Do not start with `ChaChaNotes_DB.py`.
- Start with a medium-sized, well-tested file where module boundaries are obvious.
- Candidate: `Prompts_DB.py` or `Evaluations_DB.py`, after test inventory.

Required planning before implementation:

- Per-file test inventory.
- Public method call graph.
- Transaction boundary inventory.
- Migration/schema boundary inventory.
- Rollback plan for each file.

Recommended first artifact:

- `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-db-hotspot-inventory.md` (created)

Draft first-target plan:

- `Docs/superpowers/plans/2026-04-25-phase4-3-prompts-db-decomposition-plan.md` (created)

## Phase 4.4 Decompose Large Endpoint Files

Static line-count snapshot from `wc -l tldw_Server_API/app/api/v1/endpoints/*.py` on 2026-04-25.

Top endpoint hotspots:

| File | Lines | First-pass risk |
| --- | ---: | --- |
| `persona.py` | 7395 | Very high. Large user-facing surface and likely frontend coupling. |
| `character_chat_sessions.py` | 6787 | Very high. Chat/session behavior and provider coupling. |
| `chat.py` | 6269 | Very high. Provider-compatible and streaming behavior; avoid early. |
| `watchlists.py` | 5889 | High. Scheduler/jobs/watchlist coupling. |
| `paper_search.py` | 5201 | High. Many provider integrations and external failure modes. |
| `embeddings_v5_production_enhanced.py` | 4749 | High. ML/provider coupling and health behavior. |
| `workflows.py` | 4651 | Very high. Workflow execution, adapters, tokens, and scheduler coupling. |
| `notes.py` | 4287 | Medium-high. Core user-data surface. |
| `mcp_hub_management.py` | 3921 | High. MCP management and auth behavior. |
| `auth.py` | 3760 | Very high. AuthNZ behavior; avoid until Phase 3.4 settles. |

Recommended first target:

- Do not start with `chat.py`, `auth.py`, `persona.py`, `character_chat_sessions.py`, or `workflows.py`.
- Consider a bounded medium-sized endpoint family with clear route grouping and good focused tests.
- Candidate after inventory: `slides.py`, `data_tables.py`, or `storage.py`, because these are already Phase 3 follow-up candidates.

Required planning before implementation:

- Route grouping by prefix and dependency set.
- Response-shape and pagination compatibility check.
- Auth dependency map.
- Frontend caller map.
- OpenAPI diff check.

Recommended first artifact:

- `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-endpoint-hotspot-inventory.md` (created)

Draft first-target plan:

- `Docs/superpowers/plans/2026-04-25-phase4-4-storage-endpoint-decomposition-plan.md` (created)

## Phase 4.5 API Versioning

Current signals:

- `Docs/API/api-versioning-strategy.md` already exists.
- Phase 3.1 and 3.2 are defining response envelope and pagination compatibility contracts.

Required planning before implementation:

- Reconcile `Docs/API/api-versioning-strategy.md` with the Phase 3 envelope opt-in strategy.
- Decide whether `v1` legacy defaults remain forever or get a deprecation window.
- Decide whether standard envelope/pagination defaults belong in a future `v2`.
- Decide how OpenAI-compatible/provider-compatible endpoints are versioned or exempted.

Recommended first artifact:

- `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-api-versioning-phase3-alignment.md` (created)

Draft decision packet:

- `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-api-versioning-policy-decision-packet.md` (created)

Do not do:

- Make the response envelope default in `v1` before versioning policy is accepted.

## Phase 4.6 OpenAPI Contract Testing

Current signals:

- `apps/packages/ui/package.json` includes `verify:openapi`.
- Phase 3 response and pagination schemas will affect OpenAPI output.

Required planning before implementation:

- Decide canonical OpenAPI artifact source.
- Decide whether generated clients must tolerate legacy and envelope response variants.
- Add contract tests after Phase 3 helper schema names are stable.
- Ensure provider-compatible routes are excluded or explicitly covered.

Recommended first artifact:

- `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-openapi-contract-testing-plan.md` (created)

Do not do:

- Ratchet OpenAPI checks before Phase 3 generic schema output is readable.
- Force envelope schemas onto streaming/file routes.

## Parking Lot Summary

Phase 4 is visible but still blocked by Phase 2/3 stabilization. The safest next non-runtime Phase 4 work is maintainer review of the readiness gate, deployment docs inventory, API versioning decision packet, and first DB/endpoint decomposition targets. Runtime implementation should wait for a clean accepted base, per-file or route-family plans, and focused tests.

Readiness gate:

- `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-phase4-readiness-gate.md` (created)

Consolidated Phase 3/4 remaining-work handoff:

- `Docs/superpowers/reviews/phase3-pilots/2026-04-25-phase3-phase4-remaining-work-handoff.md` (created)
