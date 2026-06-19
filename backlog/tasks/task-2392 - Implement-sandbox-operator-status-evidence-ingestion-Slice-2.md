---
id: TASK-2392
title: Implement sandbox operator-status evidence ingestion Slice 2
status: In Progress
labels:
- sandbox
- operator-ux
- vz_linux
- implementation
modified_files:
- Docs/superpowers/plans/2026-06-19-sandbox-operator-evidence-ingestion-implementation-plan.md
- backlog/tasks/task-2392 - Implement-sandbox-operator-status-evidence-ingestion-Slice-2.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Slice 2 design for env-configured, read-only host-gated VZ smoke evidence bundle ingestion into the consolidated sandbox operator-status endpoint. Scope includes parser, projection/service integration, portable tests, and docs/task updates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Add a bounded server-side parser for env-configured host smoke evidence directories without Markdown scraping or request-supplied paths.
- [ ] #2 Project parsed evidence into the existing operator-status evidence section with advisory status/action behavior from the spec.
- [ ] #3 Keep evidence ingestion read-only and fail closed for unsafe paths, symlinks, oversized/malformed JSON, unsupported schema, and parser operational errors.
- [ ] #4 Add focused portable parser/projection/service tests covering success, invalid, stale, skipped, failed, and privacy-boundary cases.
- [ ] #5 Validate with focused pytest, git diff --check, and Bandit on touched server files.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created implementation plan for Slice 2 evidence ingestion.
- Reviewed the plan against the spec and existing code boundaries.
- Corrected the initial plan to require descriptor-safe direct-child reads from the first parser commit, avoiding an unsafe intermediate direct-read parser.
- Added portability guidance for symlink-dependent parser tests.
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
