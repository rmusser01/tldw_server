---
id: TASK-588
title: Refresh core module README developer docs
status: Done
assignee: []
created_date: '2026-06-01 07:18'
updated_date: '2026-06-01 17:18'
labels: []
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-06-01-core-module-readme-refresh-design.md
  - >-
    Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source-informed documentation pass for all 88 top-level tldw_Server_API/app/core modules. Phase 1 creates or refreshes concise contributor-oriented README.md files from actual source, endpoint, schema, configuration, and test context. Phase 2 follows with deeper architecture-guide review for all 88 modules as candidates, prioritized by risk and complexity without padding simple modules unnecessarily.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every top-level app/core module has a README.md.
- [x] #2 README content is source-informed and avoids placeholder scaffolding.
- [x] #3 Existing strong READMEs are preserved or tightened rather than rewritten wholesale.
- [x] #4 Verification records README coverage and markdown/link sanity checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan: Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-06-01: Design approved by user: source-informed orientation pass for all 88 top-level core modules first, followed by a deeper architecture-guide pass.

2026-06-01: Pre-plan review found and corrected spec risks: Phase 2 now aligns with the user's 'approach 3 after approach 2' direction by treating all 88 modules as deep-guide candidates; scope now explicitly covers all immediate non-cache app/core directories; implementation inventory and local verification fallback are documented.

2026-06-01: Implementation-plan inventory corrected the top-level README baseline to 48 existing and 40 missing. The earlier 49/39 count included a nested README and has been corrected in the design spec.

2026-06-01: Implementation plan written at Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-implementation-plan.md. Worktree: .worktrees/core-module-readmes on branch codex/core-module-readmes.

Implementation inventory created at Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-inventory.md. Initial red checks: 40 top-level core modules missing README.md; Writing README contains scaffold placeholder text.

Verification: Task 5 refreshed existing core READMEs and updated the inventory. Coverage check passed with no missing top-level app/core README files. Placeholder scan returned no matches. Local Markdown link sanity check passed with 'core README markdown sanity checks passed'. git diff --check and git diff --cached --check passed. Optional codespell scan skipped because codespell is not installed. Bandit skipped because this task changed Markdown documentation only; no Python or runtime source files were modified.

2026-06-01: Spec compliance follow-up corrected stale endpoint/test evidence in affected READMEs and inventory (evaluations/media/audio endpoint packages; moderation, Guardian family-wizard, and WebSub test paths). Re-ran targeted replacement path existence checks, missing README coverage, placeholder scan, local Markdown sanity, and diff checks. Bandit remains skipped because only Markdown documentation changed.

2026-06-01: Second spec re-review follow-up corrected remaining stale README path evidence for LLM_Calls, Evaluations, DB_Management, and TTS. Re-ran targeted existence checks, targeted stale-path inspection, missing README coverage, placeholder scan, local Markdown sanity, and diff checks before commit. Bandit remains skipped because only Markdown documentation changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed source-informed README orientation pass for all 88 top-level app/core modules. Added missing README files in earlier batches, refreshed scaffolded/thin existing READMEs, preserved strong long-form guides, fixed Markdown sanity issues, and recorded verification results plus docs-only Bandit skip.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
