---
id: TASK-13138
title: Implement first-class Notes graph workspace and reviewable AI suggestions
status: In Progress
created_date: 2026-08-27 03:40
labels:
- notes
- knowledge-graph
- webui
- browser-extension
- llm
- jobs
priority: High
references:
- TASK-13134
- TASK-13135
- TASK-13136
- TASK-13137
documentation:
- Docs/superpowers/specs/2026-08-26-notes-second-brain-graph-suggestions-design.md
- Docs/superpowers/plans/2026-08-26-notes-second-brain-graph-suggestions.md
updated_date: 2026-08-27 06:31
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add Graph as a first-class Notes view mode shared by the WebUI and browser extension, and add an on-demand, source-grounded suggestion workflow for the selected note. Suggestions use a whole-library Notes FTS shortlist plus one bounded configured LLM invocation to propose related-note links and tags. Suggestions remain provisional until explicitly accepted or rejected; acceptance uses the existing manual-link and tag mutation paths. This slice does not add embeddings, semantic edge types, background organization, library-wide themes, or saved layouts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Notes exposes Graph as a first-class view mode with a focused graph canvas, search, focus control, edge-type filters, layout/fit controls, and a responsive inspector rather than a routine modal.
- [ ] #2 The graph initially focuses on the selected or most-recent note, supports bounded interactive expansion, and offers all-notes mode only below the configured cap.
- [ ] #3 The inspector provides Details and Suggestions views, grounded evidence for both notes, Strong match/Possible match bands, accept/reject controls, provisional dashed edges, and tag suggestion chips.
- [ ] #4 Canvas and Relationships views provide equivalent access to graph relationships; keyboard, focus, non-color state, narrow-screen, overflow, and long-content behavior are covered.
- [ ] #5 Suggestion generation is exposed only beneath the existing Notes graph namespace and runs through Jobs with owner/dataset scoping, idempotent admission, max_retries=0, cancellation, bounded inputs/outputs, and no note content or credentials in Job payloads or logs.
- [ ] #6 FTS searches the active owner-scoped Notes library and excludes only the selected note, trash, and directly linked note pairs; shared tag/source membership does not exclude candidates.
- [ ] #7 One configured LLM invocation receives only a bounded allowlisted shortlist and tag catalog, treats notes as untrusted data, uses a strict output schema, and cannot introduce unknown note IDs, tools, provider settings, or unbounded fields.
- [ ] #8 Suggestion runs and provisional suggestions are durable, paginated, retention bounded, and keyed by content fingerprints; evidence is stored as fingerprint-bound canonical-text offsets and reconstructed on read rather than copied into suggestion records.
- [ ] #9 Relationship suggestions accept as ordinary undirected manual links with weight 1.0 and no model-selected semantic label/properties; tag suggestions use existing tag normalization and cap newly invented tags.
- [ ] #10 Accept/reject operations are idempotent and race safe. Acceptance uses compare-and-swap plus the existing mutation path and a bounded reconciliation lease; unchanged-version rejection suppresses the same pair/tag across model or prompt versions.
- [ ] #11 Accepting one tag does not stale sibling suggestions: title/body content fingerprints are independent of tag membership, and existing-tag acceptance resolves as an idempotent success.
- [ ] #12 Request-time validation uses HTTP conflict/validation/rate-limit/readiness responses; failures after 202 are represented by durable run status, stable error codes, and sanitized user guidance.
- [ ] #13 Generation validates the top-level response strictly, drops individually invalid or duplicate items, atomically persists the validated set, and records only aggregate validation counts; no invalid suggestion is exposed.
- [ ] #14 Current projection freshness is verified or the run reports degraded/unavailable discovery rather than claiming a complete current-library search.
- [ ] #15 A successful current-version run supersedes older pending suggestions while preserving current-version rejections; stale/obsolete records follow bounded retention and note/user deletion cascades.
- [ ] #16 Backend unit/integration/property tests, frontend component/contract/accessibility tests, Playwright desktop/mobile visual checks, and an offline suggestion-quality evaluation corpus cover the approved design.
- [ ] #17 Relevant Notes and API documentation is updated, touched code passes targeted tests and lint/type checks, and Bandit reports no new findings.
- [ ] #18 A nested capability preflight discloses and ETag-binds the effective provider/model, endpoint-origin revision, local/remote/unknown data boundary, outbound data categories, permissions, and limits; the worker revalidates the same revision immediately before provider invocation.
- [ ] #19 Publication verifies an immutable owner-scoped terminal Job receipt across active/archive Jobs and revalidates source/target freshness before a fenced ChaChaNotes activation transaction; abandoned active run states reconcile fail closed.
- [ ] #20 Suggestion preprocessing enforces explicit pre-transfer UTF-8 byte limits for selected and candidate notes, reports oversized selected notes with actionable 422 guidance, and never silently truncates analyzed content.
- [ ] #21 Suggestion reconciliation runs independently of provider-worker readiness, and acceptance uses a renewable lease plus an in-transaction coordinator guard that fences both canonical mutation and suggestion finalization.
- [ ] #22 Static reset routing, loaded-graph search scope, paginated Relationships accessibility behavior, and the documented 90-day idempotency replay horizon are covered by contract tests.
- [ ] #23 All mutating suggestion routes use durable operation receipts with request fingerprints and bounded replay envelopes; reset is revision-guarded so replay cannot delete later rejections.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute the approved 12-task test-first plan in Docs/superpowers/plans/2026-08-26-notes-second-brain-graph-suggestions.md. It stages schema/store work, bounded retrieval and provider contracts, Jobs publication, guarded Sync decisions, nested API/RBAC, the shared Graph workspace, accessibility, E2E, documentation, and security verification.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Deferred work is tracked separately: TASK-13134 embeddings and semantic edges; TASK-13135 automatic background organization; TASK-13136 library-wide recurring themes; TASK-13137 saved graph views/layouts. This task remains review-first and on-demand.

Design approved in chat and written to Docs/superpowers/specs/2026-08-26-notes-second-brain-graph-suggestions-design.md. Self-review completed for incomplete markers, internal consistency, scope, asynchronous state transitions, privacy, cancellation, retention, and RBAC. The design-only change has no Bandit-applicable code scope.
Written-spec hardening pass incorporated provider capability preflight, active/archive publication receipts, Job-row owner authority, bounded note analysis, static route ordering, renewable acceptance fencing, provider-independent maintenance, loaded-graph search and accessible relationship pagination, and an explicit idempotency replay horizon.
Analysis limits were refined from post-load code-point counts to backend-portable pre-transfer UTF-8 byte predicates so oversized Notes cannot consume application memory before enforcement.
First independent spec review found five state-machine blockers. The spec now binds provider configuration/data categories through worker invocation, revalidates freshness at activation, reconciles every active run state, fences canonical mutations inside coordinator transactions, and persists 90-day idempotency receipts for every mutating route.
Second spec review found an interrupted enqueue/cancellation continuation ambiguity. Operation receipts now distinguish terminal replay from narrowly scoped in-progress continuation, maintenance resends the same idempotent cancellation command, and provider adapters must disable internal retries. Third review approved the written spec with no blocking issues; its advisory replay-wording clarification was also incorporated.
Verification before amendment commit: third independent review status Approved with no issues; advisory wording incorporated. `git diff --check` passed. The Backlog acceptance-criteria assertion reported sequential criteria 1-23. Scope is two documentation/tracking files. Runtime tests and Bandit are not applicable to this docs-only design amendment.

Implementation plan completed at Docs/superpowers/plans/2026-08-26-notes-second-brain-graph-suggestions.md. Three independent plan-review iterations were completed. Review findings addressed queue-specific 30-day Jobs receipt retention, graph-reader All-notes metadata, reload run discovery, non-vacuous observability coverage, complete Bandit scope, keyword lifecycle/RBAC tests, exact offline quality thresholds, and a baseline-aware Ruff gate. Final reviewer status: Approved with no issues.
Plan verification: 12 tasks and 82 explicit checkbox steps; all modification/deletion paths exist or are created by an earlier task; no TODO/TBD/placeholder markers; all Python commands use the repository virtual environment; `git diff --check` passed before staging. Runtime tests and Bandit are not applicable to this documentation-only planning change.

Implementation started with the subagent-driven-development workflow after confirming the isolated worktree and rebasing on current origin/dev.
Task 1 started: implementing ChaChaNotes schema v64 and the typed graph-suggestion store skeleton from the approved plan. Scope is SQLite/PostgreSQL migration parity, typed records/store initialization, focused migration tests, regression coverage, Bandit, and a task report.
Task 1 implementation completed locally: schema v64 adds SQLite/PostgreSQL Notes graph suggestion tables, indexes, forced PostgreSQL RLS, typed records, and an owner-bound store skeleton. Verification: required v61-v64 migration regression suite passed (25 tests); Ruff passed on touched Python files; Bandit scanned the production scope with 0 findings. Final report: .superpowers/sdd/2026-08-26-notes-second-brain-graph-suggestions/task-1-report.md. Parent task remains In Progress for later tasks.
Final Task 1 candidate verification: exact required v61-v64 migration regression suite passed 26 tests (including live PostgreSQL v63-to-v64 upgrade); Ruff passed; `git diff --check` passed; Bandit found 0 findings. Parent task remains In Progress for later tasks.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
