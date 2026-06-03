---
id: TASK-512
title: Address ADR workflow PR 2230 review comments
status: In Progress
labels:
- docs
- process
- adr
- review-followup
modified_files:
- AGENTS.md
- Docs/ADR/005-bandit-touched-scope-security-gate.md
- Docs/ADR/006-bandit-report-path-portability.md
- Docs/ADR/README.md
- backlog/tasks/task-508 - Implement-ADR-workflow-adoption-Stage-1.md
- backlog/tasks/task-512 - Address-ADR-workflow-PR-2230-review-comments.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a follow-up PR for unresolved review comments on PR #2230. Update TASK-508 traceability metadata, supersede ADR-005 with portable Bandit report path guidance, and document/preserve canonical Backlog task filenames rather than renaming them to non-Backlog paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 TASK-508 modified_files frontmatter includes the primary Stage 1 deliverables and Backlog task files changed by PR #2230.
- [ ] #2 ADR-005 is superseded by accepted portable Bandit report path guidance, and current AGENTS.md guidance avoids a hardcoded /tmp report path.
- [ ] #3 Backlog task filename review comments are addressed without breaking the repository's canonical Backlog task-file naming convention.
- [ ] #4 Verification commands pass, including git diff --check and targeted text checks.
- [ ] #5 A new draft or ready PR is opened and linked from this Backlog task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Scope: follow-up PR for unresolved PR #2230 review comments only. Valid fixes: add Stage 1 deliverables to TASK-508 modified_files, supersede ADR-005 with portable Bandit report path guidance, and update current AGENTS.md guidance. Filename-renaming comments will be addressed by preserving actual Backlog task paths and documenting why nonstandard paths are not used.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR #2230 filename review comments suggested lowercase kebab-case Backlog task paths, but this repository's Backlog.md task files use the canonical `task-<id> - <Title>.md` format. This follow-up preserves actual task file paths and avoids listing non-existent renamed paths in `modified_files`.
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
