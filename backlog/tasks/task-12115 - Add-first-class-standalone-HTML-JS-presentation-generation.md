---
id: TASK-12115
title: Add first-class standalone HTML-JS presentation generation
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-15 23:52'
labels:
  - slides
  - presentation-studio
  - backend
  - frontend
  - security
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-07-15-standalone-html-presentations-design.md
priority: high
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
- [ ] #3 Standalone HTML generation uses one shared mode-aware service across supported source kinds, submission-time immutable source snapshots, and one administrator-configured allowlisted default provider/model pair.
- [ ] #4 Presentation Studio exposes a form-first HTML+JavaScript generation flow and a dedicated code, text-only safe-outline, save, conflict, recovery, and attachment-download experience.
- [ ] #5 Generated HTML/JavaScript is never rendered or executed by a tldw server, WebUI, extension, worker, MCP path, or renderer; source is never served as text/html.
- [ ] #6 Legacy presentations and clients remain structured by default, schema-v2 and version migrations are covered, and capabilities fail closed without blocking existing HTML read/edit/export.
- [ ] #7 Focused backend, frontend, security, integration, and E2E tests pass, and Bandit reports no new findings in touched Python.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-07-15 requester-approved V1 design: standalone HTML+JavaScript is generated, stored, edited, versioned, and downloaded as opaque source; Presentation Studio shows only a trusted text safe outline, and every tldw execution/fidelity-render path is prohibited. The shared backend snapshots sources in the per-user Slides database, passes only receipt references/digests through Jobs, uses queue=default with numeric public IDs/internal UUID correlation, and resolves one server-selected allowlisted default model. The spec includes schema-v2 migration, terminal idempotency receipts, immutable bounded provenance, exact MIME/no-store protections, explicit save/conflict/recovery, a minimal paginated index, and extension metadata-only handoff. Independent product, backend, and security/consistency reviews approve the written design. Implementation remains gated on requester review and a separate implementation plan.

2026-07-15 design verification: targeted git diff check passed; Markdown fences are balanced; capability JSON parses and contains no execution-preview capability; stale Validate-and-run/runtime contracts are absent; the related PRD link resolves; and the task parses through the official Backlog CLI. This revision is documentation/task metadata only, so Python tests, frontend builds, and Bandit are not applicable at this stage. No implementation code has started.
<!-- SECTION:NOTES:END -->

<!-- SECTION:FINAL_SUMMARY:END -->
