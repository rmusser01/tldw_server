# Architecture Decision Records

Architecture Decision Records (ADRs) capture durable architecture decisions for `tldw_server`: what was decided, why, what alternatives were considered, and what tradeoffs were accepted.

Module docs, design specs, and plans describe how things work. ADRs explain why important architecture rules exist.

## Workflow

1. Search existing ADRs before creating a new one.
2. Create a Backlog.md task or use the task already associated with the work.
3. Use `000-template.md`.
4. Use the next sequential number.
5. Record one decision per ADR.
6. Write ADRs at decision time whenever possible.
7. If backfilling, keep `Status: Accepted` for still-governing decisions and set `Backfilled from:` to the source path.
8. Do not rewrite accepted ADR rationale. To change a decision, create a new ADR and mark the old one `Superseded by ADR-{N}`.

## Status Rules

- `Proposed`: drafted for review but not yet accepted.
- `Accepted`: current governing decision.
- `Superseded by ADR-{N}`: no longer governing because a newer ADR replaced it.
- Backfill is metadata, not status. Backfilled still-governing decisions use `Status: Accepted` plus `Backfilled from: <source>`.

## ADR Required When

An ADR is required when a decision creates or changes a durable rule for module boundaries, public API shape, persistence, security, worker ownership, provider integration, WebUI/extension conventions, major dependencies, or repository workflow gates.

Small bug fixes, local implementation details, product copy, temporary experiments, and test-only changes usually do not need ADRs unless they create durable policy.

## Index

| ADR | Status | Decision |
| --- | --- | --- |
| [ADR-001](001-adr-workflow-and-governance.md) | Accepted | Adopt `Docs/ADR/` as the canonical ADR workflow. |
| [ADR-002](002-backlog-md-task-tracking.md) | Accepted | Require Backlog.md tasks for repo-changing work. |
| [ADR-003](003-jobs-vs-scheduler-default.md) | Accepted | Use Jobs by default for new user-visible work and Scheduler for internal dependency orchestration. |
| [ADR-004](004-ai-generated-pr-change-summary-gate.md) | Accepted | Require human-written change summaries for materially AI-authored PRs. |
| [ADR-005](005-bandit-touched-scope-security-gate.md) | Superseded by ADR-006 | Run Bandit on touched Python/code scope before completion. |
| [ADR-006](006-bandit-report-path-portability.md) | Accepted | Keep the Bandit touched-scope gate but require portable report output paths. |
| [ADR-007](007-research-workspace-canonical-first-slice-shell.md) | Accepted | Use `ResearchWorkspace` as the canonical first-slice workspace shell while preserving specialized routes. |
| [ADR-008](008-workspace-split-key-persistence-and-indexeddb-offload.md) | Accepted | Use split localStorage workspace persistence with optional IndexedDB offload for heavy payloads. |
| [ADR-009](009-quick-chat-docs-assistant-modes.md) | Accepted | Keep Quick Chat split into `Chat`, `Docs Q&A`, and `Browse Guides` modes. |
| [ADR-010](010-sandbox-vz-runtime-ownership.md) | Accepted | Keep `vz_linux` as a repo-owned sandbox runtime path instead of requiring Apple `container`. |
| [ADR-011](011-audio-api-semantics.md) | Accepted | Use centralized Audio API auth, model-first TTS routing, structured streaming errors, and non-streaming-only download links. |
| [ADR-012](012-evaluations-resource-id-prefixes.md) | Accepted | Use OpenAI-style type-prefixed IDs for primary Evaluations API resources. |
| [ADR-013](013-evaluations-deletion-lifecycle.md) | Accepted | Use soft deletes for evaluation definitions and hard deletes for datasets. |
| [ADR-014](014-evaluations-openai-compatible-schemas.md) | Accepted | Use separate request and response schemas with OpenAI-compatible response conventions. |
| [ADR-015](015-evaluations-existing-evaluator-integration.md) | Accepted | Wrap existing evaluator modules instead of rewriting evaluator logic inside the API runner or endpoints. |
| [ADR-016](016-acp-session-and-orchestration-persistence.md) | Accepted | Persist ACP shared operational state in `Databases/acp_sessions.db` and user orchestration state in per-user `orchestration.db` files. |
| [ADR-017](017-scoped-org-team-rbac-core-semantics.md) | Accepted | Use feature-flagged scoped Org/Team RBAC overlays with `require_active` default scope and denylist-filtered grants. |
| [ADR-018](018-resource-governance-endpoint-policy-and-route-map.md) | Accepted | Use claim-first new-endpoint authorization with explicit Resource Governor applicability decisions, route-map ownership, DB/file route-map merge semantics, and request-ingress missing-policy denial. |
| [ADR-019](019-security-request-edge-middleware.md) | Accepted | Own request-edge Security middleware in the Security module with path-scoped setup/docs/API behavior, always-installed request IDs and drain gate, and production-default security headers. |
| [ADR-020](020-db-management-per-user-paths-and-content-backend.md) | Accepted | Use `DatabasePaths` for per-user database paths, SQLite as the default content mode, and explicit PostgreSQL content mode with startup validation. |
| [ADR-021](021-services-lifecycle-startup-and-shutdown.md) | Accepted | Use focused Services lifespan orchestration with lifecycle worker session ownership, cooperative stop-event workers, staged shutdown, and bounded timeout/cancel fallback. |
| [ADR-022](022-embeddings-api-and-media-pipeline.md) | Accepted | Use OpenAI-compatible Embeddings API safeguards, bounded provider routing, endpoint reliability controls, and Jobs-root/Redis-stage media pipeline ownership. |
| [ADR-023](023-data-tables-backend-storage-jobs-and-exports.md) | Accepted | Use Media DB-owned Data Tables state, Jobs-backed generation/regeneration, bounded source snapshots, UUID table APIs, and server-side exports. |
| [ADR-024](024-deepseek-ocr-local-transformers-backend.md) | Accepted | Support DeepSeek OCR as a local Transformers-only backend with explicit dependency gates, temporary output by default, and security caveats for `trust_remote_code=True`. |
| [ADR-025](025-llm-provider-adapter-routing-and-overrides.md) | Accepted | Use the LLM provider adapter registry boundary with OpenAI-compatible normalization, trusted allowlisted `base_url` overrides, and local endpoint URL override rejection. |
| [ADR-026](026-security-outbound-egress-and-ssrf-policy.md) | Accepted | Require outbound integrations to use Security egress helpers for untrusted URLs, with central scheme, host, port, allow/deny, profile, tenant, DNS, and private-address checks. |
| [ADR-027](027-security-aes-gcm-json-envelope-helpers.md) | Accepted | Use Security AES-GCM JSON envelope helpers as the shared primitive for configured encrypted structured-metadata persistence and key-rotation paths. |
