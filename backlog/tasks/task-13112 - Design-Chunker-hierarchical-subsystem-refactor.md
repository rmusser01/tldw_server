---
id: TASK-13112
title: Design Chunker hierarchical subsystem refactor
status: In Progress
created_date: 2026-08-24 05:37
dependencies:
- TASK-9937
labels:
- chunking
- refactor
- design
priority: High
references:
- Docs/superpowers/specs/2026-06-24-chunker-process-text-refactor-design.md
- Docs/superpowers/plans/2026-06-24-chunker-process-text-refactor.md
- https://github.com/rmusser01/tldw_server/pull/2517
- TASK-13113
documentation:
- Docs/superpowers/specs/2026-08-23-chunker-hierarchical-subsystem-refactor-design.md
modified_files:
- Docs/superpowers/specs/2026-08-23-chunker-hierarchical-subsystem-refactor-design.md
- backlog/tasks/task-13112 - Design-Chunker-hierarchical-subsystem-refactor.md
- backlog/tasks/task-13113 - Implement-Chunker-hierarchical-subsystem-refactor.md
updated_date: 2026-08-24 07:15
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a behavior-preserving design for extracting Chunker's full hierarchical chunking subsystem into focused internal modules. Preserve public tree/flat methods and output contracts, remove the approved private span/title helper seams, allow only reproduced corrections, and base the work on current origin/dev after the merged process_text refactor. Scope is design documentation only; implementation planning follows user review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The spec defines focused hierarchy package ownership, dependency direction, and narrow context contracts
- [x] #2 The spec preserves public hierarchy signatures, tree/flat dictionary contracts, composition seams, and module-level fallback behavior
- [x] #3 The spec defines the direct shared span API and approved removal of private span/title helper seams
- [x] #4 The spec defines reproduced-defect correction gates, characterization coverage, staged extraction, and verification
- [ ] #5 The committed spec is self-reviewed and presented to the user before implementation planning
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reconcile current origin/dev hierarchy behavior and focused tests. 2. Write the approved design spec. 3. Self-review for placeholders, contradictions, scope, and ambiguity. 4. Commit the design and request user review before writing an implementation plan.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved design decisions: extract the full hierarchy subsystem through a stateless context-backed service; split spans, leaves, tree building, grouping, and flatten traversal into focused modules; keep public tree/flatten/flat signatures and public flat composition; preserve the package-level flatten fallback; remove the private paragraph-span and header-title seams; route multi-level process_text directly through hierarchical.spans; allow only reproduced, separately tested corrections. Baseline pinned to refreshed origin/dev 4958cfed65d3c6e9baa43ea47e2b155fed204e13. Focused baseline verification: 91 passed, 1 skipped, 0 failures. Spec self-review found no placeholders, contradictions, unresolved scope decisions, or ambiguous compatibility requirements. Bandit is not applicable because this task changes documentation and Backlog metadata only. Acceptance criterion 5 remains open until the committed spec is presented for user review.
Committed-spec review validated five ambiguities to amend before implementation planning: exact metadata fallback triggers and leaf call multiplicity; sanitize_output truthiness, failure default, and removal order; shallow flatten metadata/ancestry aliasing; regex helper fallback, logging, signature, and structural import characterization; and explicit reconciliation if origin/dev advances beyond the pinned baseline. User requested all five be addressed.
Amended the spec to address all five validated review findings. The revision now pins sanitize_output preparation/removal, exception-triggered metadata fallback and bounded existing retry call traces, shallow metadata and ancestry aliasing, regex helper/direct-search fallback plus logging/signature/AST test requirements, and the advanced-origin reconciliation procedure. Self-review removed an initial overstatement that would have prohibited the current outer fallback's one bounded second plain attempt. Placeholder scan is clean, Markdown fences are balanced, and git diff --check passes. During review origin/dev advanced by 46 commits to 2ebf14c145f7ce7e4e8ee9c2c6dce3da25006780; the intervening range has no changes under tldw_Server_API/app/core/Chunking or tldw_Server_API/tests/Chunking. The implementation stage must still rebase, update the recorded baseline, and rerun focused characterization before production edits. Bandit remains not applicable because only documentation and Backlog metadata changed.
Second committed-spec review identified six delivery and design gaps. Addressed them by creating dependent implementation task TASK-13113; adding the human-written Change summary merge gate; defining LeafChunkingContext, HierarchyTextViews, builder/leaf/flatten callable boundaries, and mutation ownership; limiting logging compatibility to level/message/exception behavior while permitting source-provenance drift; replacing unspecified malformed-tree coverage with a concrete minimum matrix; and aligning verification with the repository baseline using Ruff, scoped Black, informational hierarchy-local mypy, compileall, Bandit, hooks, and diff checks.
Post-amendment self-review found no placeholders, contradictory ownership, unresolved malformed-tree selection, or unbalanced Markdown fences; git diff --check passed. Tooling calibration used the shared project virtualenv: Ruff passed for chunker.py and process_text; Black would reformat chunker.py plus process_text/metadata.py, options.py, and preparation.py; mypy reported five existing process_text errors across metadata.py, dispatch.py, and pipeline.py. The spec therefore makes Ruff a touched-production gate, Black a new-file gate, and mypy informational for the new package without forcing unrelated baseline cleanup. Bandit remains inapplicable to this documentation-only amendment.
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
