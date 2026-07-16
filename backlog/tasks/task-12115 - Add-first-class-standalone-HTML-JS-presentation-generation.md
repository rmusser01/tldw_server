---
id: TASK-12115
title: Add first-class standalone HTML-JS presentation generation
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-07-16 01:50
labels:
- slides
- presentation-studio
- backend
- frontend
- security
dependencies: []
documentation:
- Docs/superpowers/specs/2026-07-15-standalone-html-presentations-design.md
- Docs/superpowers/plans/2026-07-15-standalone-html-presentations-implementation-plan.md
priority: high
modified_files:
- Docs/superpowers/plans/2026-07-15-standalone-html-presentations-implementation-plan.md
- backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement a hardened standalone HTML+JavaScript presentation mode shared across existing Slides source types, with a form-first Presentation Studio flow, strict content-kind invariants, bounded LLM output, explicit-save editing, a text-only safe outline, attachment-only file handoff, compatibility guards, tests, documentation, and a firm no-execution boundary across every tldw surface.
<!-- SECTION:DESCRIPTION:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An approved design spec and implementation plan document the architecture, no-execution security boundary, compatibility behavior, and deferred scope.
- [ ] #2 The Slides backend supports structured_slides and standalone_html as explicit, validated content kinds without permitting split-brain records.
- [ ] #3 Standalone HTML generation uses one shared mode-aware service across supported source kinds, submission-time immutable source snapshots, and one administrator-configured concrete allowlisted provider/model/adapter/endpoint target.
- [ ] #4 Presentation Studio exposes a form-first HTML+JavaScript generation flow and a dedicated code, text-only safe-outline, save, conflict, recovery, and attachment-download experience.
- [ ] #5 Generated HTML/JavaScript is never rendered or executed by a tldw server, WebUI, extension, worker, MCP path, or renderer; source is never served as text/html.
- [ ] #6 Legacy presentations and clients remain structured by default, schema-v2 and version migrations are covered, and capabilities fail closed without blocking existing HTML read/edit/export.
- [ ] #7 Focused backend, frontend, security, integration, and E2E tests pass, and Bandit reports no new findings in touched Python.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-15-standalone-html-presentations-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-07-15 requester-approved V1 direction: standalone HTML+JavaScript is generated, stored, edited, versioned, and downloaded as opaque source. Presentation Studio exposes the form first and shows only a trusted text safe outline. Every tldw execution or fidelity-render path remains prohibited in V1.

2026-07-15 final design hardening: the shared backend binds each accepted generation to an owner-scoped public generation UUID, an immutable internal Jobs UUID, domain-separated HMAC receipts, an immutable source snapshot, and one server-selected concrete allowlisted provider/model/adapter/endpoint target. The design also specifies retrieval-only owner-local RAG, bounded source adapters and provider reads, killable validation and outline workers, raw octet-stream save and draft attachment paths, explicit content-kind negotiation for compatibility, crash recovery, and an emergency standalone-HTML egress kill.

2026-07-15 fresh re-review: backend, security, and product reviewers all returned APPROVED with no remaining P0-P3 or blocking design findings. Four embedded JSON contracts parse, Markdown fences are balanced, the related link resolves, heading hierarchy is valid, required hardened contracts are present, and stale superseded contracts are absent. This revision changes only documentation and Backlog metadata, so Python tests, frontend builds, and Bandit are not applicable. Implementation has not started; the next step is the task-specific implementation plan.
<!-- SECTION:NOTES:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-15 implementation plan approved: the five-stage, 17-task TDD plan locks the closed provider adapter catalog, external-secret/shared-Jobs-store key and reconciliation metadata, fenced receipt/worker recovery, guarded per-request MCP discovery and execution, and the inert form/editor/download boundary. A fresh independent plan review returned APPROVED after correcting the HTML slides=[] persistence invariant, shared Jobs coordination, guarded Uvicorn/WebSocket pins, exact outer-fence parsing, dependency smoke gates, per-commit Backlog staging, and mechanically complete Bandit scope. Backend Slides baseline: 100 passed with 5 warnings. The isolated frontend worktree had no installed workspace dependencies, so no frontend product test ran or failed; Task 13 begins with a frozen clean install and pre-change regression gate. Implementation code has not started.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
