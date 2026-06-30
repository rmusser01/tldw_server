---
id: TASK-34
title: Address PR 1256 merge conflict
status: Done
assignee:
  - codex
created_date: '2026-05-04 03:58'
updated_date: '2026-05-04 04:07'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1256'
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #1256 onto latest dev and resolve any merge conflicts while preserving the narrow outputs_templates/outputs lazy router import tranche.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #1256 branch rebased onto latest origin/dev
- [x] #2 Conflict resolution preserves output router lazy registration and existing metadata
- [x] #3 Focused router group contract test, full touched router contracts, Bandit touched source scan, and diff check are run before force-push
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Fetch latest origin/dev and inspect the delta causing PR #1256 to be dirty. 2. Rebase codex/phase2-2-content-outputs-router-conditionals-j onto origin/dev. 3. Resolve conflicts by preserving the already-merged dev changes while retaining only the outputs_templates/outputs ImportedRouterSpec conversion and its focused test. 4. Run focused output router laziness test, full router group contract test, Bandit on content.py, and git diff --check. 5. Update TASK-34, force-push with lease, and verify PR #1256 no longer reports DIRTY.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rebased PR #1256 from dev 49f46dbc39 to 1193f296bf. The only content.py conflict was outputs_templates: latest dev already included outputs_templates in the utility_spec tuple, so the resolution kept dev's utility tuple and retained this PR's lazy outputs router registration lower in the file.

Verification after rebase: focused output_router_attr_lookup passed; full router_groups_contract passed 51 tests; main_router_contract passed 6 tests; openapi_contracts passed 69 tests; Bandit content.py JSON reported 0 results and 0 errors; git diff --check was clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved PR #1256's merge conflict by rebasing onto latest origin/dev and preserving both sides of the overlapping router-group work. The resolved content router keeps outputs_templates in dev's utility/content ImportedRouterSpec tuple while retaining this PR's lazy outputs router registration. Post-rebase focused and contract verification passed, and Bandit/diff hygiene remained clean.
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
