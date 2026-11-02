---
id: TASK-15413
title: Design Chunker shared execution subsystem refactor
status: In Progress
labels:
- chunking
- refactor
- design
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design the requester-selected approach B: an internal execution subsystem for chunk_text, chunk_text_with_metadata and chunk_text_generator, with thin public Chunker wrappers and compatibility-first path-specific behavior. This follows merged PR #3095; separate bookkeeping PR #3397 and its branch remain untouched. This task is design-only: implementation and an implementation plan require later explicit design approval.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Record approaches A and B, requester selection B, goals, non-goals and a pinned current-dev baseline with fresh Chunking verification
- [x] #2 Specify internal ownership, one-way dependencies, facade and override compatibility, and exact eager/metadata/generator behavior including laziness, exception regions, caching and lock boundaries
- [x] #3 Define focused characterization, behavioral/property/concurrency tests and staged verification without introducing production changes or an implementation plan
- [x] #4 Record ADR assessment, complete a fresh independent builder check and resolve material design gaps
- [ ] #5 Present the reviewed design for requester approval; do not begin implementation or mark approval complete without their response
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Requester chose B on 2026-11-02 after considering shared setup helpers (A) versus a full internal execution subsystem (B). Use codex/chunker-execution-refactor-design in existing isolated worktree /Users/appledev/Documents/GitHub/tldw_server/.worktrees/chunker-hierarchical-refactor-design; dirty main checkout and bookkeeping branch codex/chunker-hierarchical-refactor-design remain untouched. Pinned dev 3ab1ce5d3a49d5a7fd130e8939d55d7639e5a373. Production Chunking code is unchanged since merge 6bf581da08; intervening lifecycle test updates a shared-provider fixture to fetch_authorized_secret_for_user and asserts scope/user/provider. Fresh full Chunking baseline: 796 passed, one existing PyThaiNLP skip, 2000 warnings, 42.02s (/tmp/chunker_execution_design_baseline.txt). Scanned origin/dev backlog and all 38 open PR file lists, paginating oversized PRs; maximum task ID 15412, no duplicate execution-design path. Planned spec Docs/superpowers/specs/2026-11-02-chunker-shared-execution-refactor-design.md. Prior hierarchy spec Docs/superpowers/specs/2026-08-23-chunker-hierarchical-subsystem-refactor-design.md and process_text spec Docs/superpowers/specs/2026-06-24-chunker-process-text-refactor-design.md establish existing local composition/delegation conventions. ADR required: no for this local implementation decomposition with unchanged public/security/concurrency semantics; no new durable architecture policy is proposed. ADR-002/059/084 govern tracking and IDs, ADR-006 governs portable Bandit output. Reassess ADR need if a new durable rule is introduced. No product/test edits, new dependencies or runtime hardening are authorized by design selection alone.
2026-11-02 full draft spec written at Docs/superpowers/specs/2026-11-02-chunker-shared-execution-refactor-design.md. Three path-specific service entries retain text-only result caching, metadata identity/alignment and generator first-advance/iterator/lock semantics; common preparation is shared only where equivalent. Existing context helper seams and mutable facilities stay on Chunker; module-level observability uses live forwarding rather than importing the facade from the new package. Exact exception-region and path-specific option differences, public signatures, override hooks, correction gate, test/CI/security requirements and ADR assessment are explicit. A fresh read-only independent builder check is running; reviewer must return only material unanswered planning questions. Requester design approval remains pending. No implementation plan, product/test changes, dependency installation, PR creation or merge is authorized by this draft. Bookkeeping PR #3397 remains open/draft on its original branch and is untouched.
2026-11-02 design-only validation: staged scope is exactly the new specification and TASK-15413; no production, test, configuration or instruction files changed. Official canonical task check, duplicate-ID check and staged diff check pass. Task-format ratchet: four passed, 20 warnings, 2.76s (/tmp/chunker_execution_design_tracking.txt). Product baseline remains 796 passed, one established skip, 2000 warnings, 42.02s. Bandit: not applicable to this Markdown-only edit; no Python code was added or changed. Builder check remains pending; requester approval is still unchecked and no implementation plan has started.
2026-11-02 fresh independent builder check completed: reviewer read the draft and relevant source/tests, returned nothing important left, and reported no material unanswered planning questions. Reviewer was read-only, did not duplicate the baseline tests or dispatch descendants, and is closed. Final editorial clarification distinguishes prohibited subsystem facade imports from the existing root package eager facade export; package bootstrap redesign remains out of scope. No new requester decision is inferred from the builder check. AC1-4 complete; AC5, requester approval and overall task completion remain pending. No implementation plan, source/test edits, PR publication or merge has started. Before commit, rerun canonical/task-format/ID/scope checks and commit only this task and the new specification.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Drafted the requester-selected approach B for a shared internal execution subsystem with three path-specific entries and thin Chunker delegates. The full design records the pinned dev baseline, alternatives, ownership and live context/observability hooks, option/cache/metadata/generator/locking/error compatibility, deterministic characterization and staged verification, correction boundaries and ADR assessment. Fresh full Chunking baseline: 796 passed, one existing PyThaiNLP skip, 2000 warnings, 42.02s. Task-format four passed; canonical and task-ID checks pass. A fresh independent builder check returned nothing important left. Edits are Markdown-only; Bandit is not applicable because no code changed. Design task remains In Progress pending requester approval. Implementation plan and product/test changes are not authorized until that approval. Main checkout, original bookkeeping branch/PR and historical task records are untouched.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
