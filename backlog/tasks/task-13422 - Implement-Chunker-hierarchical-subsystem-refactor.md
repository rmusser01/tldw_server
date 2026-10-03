---
id: TASK-13422
title: Implement Chunker hierarchical subsystem refactor
status: In Progress
created_date: 2026-10-03 01:30
dependencies:
- TASK-13421
labels:
- chunking
- refactor
- implementation
priority: high
references:
- TASK-13421
- backlog/tasks/task-13215 - Implement-Chunker-hierarchical-subsystem-refactor.md
documentation:
- Docs/superpowers/specs/2026-08-23-chunker-hierarchical-subsystem-refactor-design.md
- Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md
modified_files:
- tldw_Server_API/app/core/Chunking/chunker.py
- tldw_Server_API/app/core/Chunking/hierarchical/
- tldw_Server_API/app/core/Chunking/process_text/models.py
- tldw_Server_API/app/core/Chunking/process_text/dispatch.py
- tldw_Server_API/tests/Chunking/
- Docs/superpowers/specs/2026-08-23-chunker-hierarchical-subsystem-refactor-design.md
- Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md
updated_date: 2026-10-03 01:31
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the user-approved compatibility-first hierarchical subsystem refactor. This replacement task resolves the active TASK-13215 collision with unrelated upstream Writing work after October 2 baseline reconciliation; approved scope and completed implementation stages are unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The implementation branch is reconciled with current origin/dev and focused baseline characterization is rerun before production edits
- [x] #2 Frozen characterization tests cover option handling, leaf call traces, spans, malformed trees, flatten aliasing, logging, signatures, and import boundaries
- [ ] #3 The hierarchical package is extracted with the approved component interfaces and dependency direction while public hierarchy behavior remains compatible
- [x] #4 The approved private span and header-title helpers are removed and process_text imports the shared span function directly
- [ ] #5 Focused and complete Chunking tests, compileall, Ruff, scoped Black, Bandit, and git diff --check pass with results recorded
- [ ] #6 The PR remains non-merge-ready until the human requester supplies the required Change summary explaining what changed and why
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Continue Tasks7-8 in the approved plan after completed Tasks1-6 and specification/quality approvals: extract flatten traversal and complete public delegation test-first; enforce boundaries and resolve only hierarchy-local typing issues without behavior changes; run focused/full Chunking and static/security gates; final independent reviews; push and create draft PR against dev with human-written Change summary merge blocker. No behavior corrections pre-approved.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-02 tracking reconciliation: supersedes ambiguous TASK-13215, retaining complete historical evidence at the explicit old record path. New design authority is TASK-13421. Pinned origin/dev86e287fee7bfa1a1588639232e35db3666851ded; isolated branch codex/chunker-hierarchical-refactor-design at3e34078eef78111633a210cda4a476baa25abbeb. Tasks1-6 complete with separate specification and quality approvals. Task5 regression caused by erroneous controller grouping-passthrough instruction was validated and reverted in b8d015c582; exact seven-key envelope remains. Task6 required155pass/65contracts, all static gates/Bandit0. Full pre-grouping suite669pass/1PyThaiNLPskip/1747warnings/47.45s; initial tokenizer cache failure resolved by priming cl100k_base in one network-enabled existing test, with no code/skip changes. Two informational hierarchy-local mypy errors in spans.py122/127 reserved for final cleanup. Task7 worker interrupted before code edits when ID collisions discovered; one start note remains in the historical task, unrelated upstream records unchanged. Resume same Task7 worker after tracking reconciliation. Draft PR creation authorized; keep task In Progress until human-written Change summary gate satisfied.
The previously completed acceptance criteria 1,2,4 are retained from the old implementation record; remaining package extraction/full final verification/human-summary gates remain open. Forward-looking spec and plan references now use unique TASK-13421/TASK-13422 IDs. Old records and historical IDs remain provenance, with no further mutation by ambiguous ID.
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
