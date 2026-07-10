---
id: TASK-12946
title: Rebase PR 2706 and address review feedback
status: Done
labels:
- security
- codeql
- frontend
- code-review
priority: High
references:
- https://github.com/rmusser01/tldw_server/pull/2706
documentation: []
modified_files:
- apps/packages/ui/src/components/Option/Watchlists/ItemsTab/items-utils.ts
- apps/packages/ui/src/components/Option/Watchlists/ItemsTab/__tests__/items-utils.test.ts
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.advanced-details.test.tsx
- backlog/tasks/task-12946 - Rebase-PR-2706-and-address-review-feedback.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase the CodeQL remediation PR #2706 onto the latest origin/dev, evaluate and address every actionable PR review comment, verify the focused frontend behavior and checks, reply in the original review threads, and update the PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #2706 is rebased onto the latest origin/dev without dropping remediation changes.
- [x] #2 All technically valid inline and summary review findings are addressed with focused regression coverage.
- [x] #3 Focused tests, frontend typecheck, diff checks, and applicable security validation pass.
- [x] #4 Review threads receive precise replies and the updated branch is pushed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Review inventory on 2026-07-10 found two inline threads and one summary-only issue. Root-cause verification confirmed that the existing single-match extraction rejected valid whitespace and stopped after an unsafe candidate, suppressing later safe previews. The fix in 69de8756a9 uses global candidate scans, preserves HTML-before-Markdown precedence, and keeps every returned value behind safeImageUrl. The query fix in 8b8a33e4e7 uses selectedGroupId != null; backend group endpoints still constrain IDs to >=1, so the zero case is defensive frontend type-contract coverage rather than a server behavior change.

`git fetch origin dev` followed by `git rebase origin/dev` reported the branch already up to date, so no history rewrite or force-push was required. RED evidence: ItemsTab added cases failed 5/53 for the expected whitespace/early-return reasons; SourcesTab failed 1/8 with groups omitted. GREEN/final evidence: focused review suites passed 61/61; the complete affected CodeQL regression set passed 139/139 across 10 files; `NODE_OPTIONS=--max-old-space-size=8192 bun run typecheck` passed; `git diff --check origin/dev...HEAD` passed; origin/dev is an ancestor of HEAD; independent final review found no actionable issues. Bandit is not applicable because no Python source changed.

Both inline review threads received commit-specific replies and are resolved. The Qodo summary-only group finding received a PR-level response. Residual, non-blocking observations from final review: the lightweight regex retains pre-existing malformed-input worst-case scaling and is not a full HTML parser; safeImageUrl still enforces the security boundary. JavaScript CodeQL cannot be claimed on dev unless GitHub emits a JavaScript analysis.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2706 onto the latest dev (already current), addressed all actionable review feedback, and updated the PR. Image extraction now accepts valid src-assignment whitespace and scans past unsafe HTML/Markdown candidates to the first safe URL without changing HTML precedence. Group query construction now follows explicit number|null semantics. Focused tests passed 61/61, the full affected CodeQL set passed 139/139, frontend typecheck and diff checks passed, final review found no issues, and all review threads were answered/resolved.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests and verification recorded
- [x] #3 Bandit run for touched Python code or explicit not-applicable note
- [x] #4 Final summary and PR link recorded
- [x] #5 Known skips or blockers documented
<!-- DOD:END -->
