---
id: TASK-81
title: 'Address PR #1317 CodeGraph context ranking review comments'
status: Done
assignee: []
created_date: '2026-05-05 17:59'
updated_date: '2026-05-05 18:04'
labels:
  - codegraph
  - mcp
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1317'
  - 'https://github.com/rmusser01/tldw_server/issues/1314'
documentation:
  - >-
    Docs/superpowers/plans/2026-05-05-native-codegraph-context-ranking-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve unresolved review comments on PR #1317 for the native CodeGraph context-ranking slice. Verify each external review finding against the current branch before changing code, keep the fix narrowly scoped, and preserve the public codegraph.context response shape and existing bounds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Relationship ranking boosts count only candidate-to-candidate relationships or otherwise avoid boosting nodes for edges to non-candidate neighbors.
- [x] #2 Candidate term collection remains bounded while ensuring later task terms are not dropped solely because the candidate budget is small.
- [x] #3 Filename/path token scoring handles common file stems with extensions instead of giving only substring-level relevance.
- [x] #4 Ranking tie-breakers are clear and reachable, or unreachable fields are removed if original-order stability is intended.
- [x] #5 Focused CodeGraph/MCP tests, Ruff on touched scope, Bandit on touched production scope, and git diff --check pass or blockers are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified PR #1317 review feedback against the branch. Fixed valid findings by filtering relationship ranking boosts to candidate-to-candidate edges, removing unreachable sort-key fields while preserving original-order tie stability, boosting filename stem matches such as pkg/app.py for token app, and replacing max_search_results-based term slicing with a fixed bounded search-term cap so small candidate budgets still rotate through later task terms.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved all current PR #1317 review comments. Relationship boosts now ignore edges to non-candidate neighbors, candidate term search stays bounded without dropping later terms solely because max_search_results is small, filename stem path matches receive the stronger relevance score, and ranking tie-breakers now reflect the intended original-order fallback without unreachable fields. Added regressions for external hub relationships, filename stem scoring, and small-search-budget context ranking. Verification passed locally: focused CodeGraph/MCP suite 165 passed and 5 warnings; Ruff touched scope passed; Bandit touched production scope reported 0 results; git diff --check passed. Known blockers: none.
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
