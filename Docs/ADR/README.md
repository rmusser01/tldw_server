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
