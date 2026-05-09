---
id: TASK-45.32
title: Allow RecoveryCallout adapters in product-state guard
status: Done
assignee:
  - Codex
created_date: '2026-05-09 21:11'
updated_date: '2026-05-09 21:35'
labels:
  - design-system
  - ui
  - product-state
  - guard
dependencies: []
references:
  - apps/packages/ui/scripts/design-system-product-state-rules.mjs
  - apps/packages/ui/src/design-system/__tests__/product-state-guard.test.ts
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
  - apps/packages/ui/src/components/Sidepanel/Chat/ConnectionBanner.tsx
parent_task_id: TASK-45
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The product-state guard does not flag local recovery-banner adapters that return the shared RecoveryCallout or StatePanel primitive.
- [x] #2 The guard still flags local recovery-banner components that render bespoke recovery UI.
- [x] #3 The Sidepanel ConnectionBanner local-recovery-banner baseline exception is removed because it already renders the shared RecoveryCallout.
- [x] #4 Focused guard tests and design-system verification cover the new recovery adapter allowance and stale-baseline cleanup.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing guard tests for recovery-banner adapters: allow a component returning RecoveryCallout/StatePanel while continuing to flag bespoke recovery banners. 2. Update canonical usage collection and local recovery-banner finding logic to recognize returned RecoveryCallout/StatePanel owners. 3. Remove the Sidepanel ConnectionBanner baseline entry that becomes stale under the updated rule. 4. Run focused product-state guard tests, design-system verifier, git diff --check, and a touched-scope type/filter check; document Bandit skip for UI JS/TS/JSON-only changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented recovery adapter recognition in the product-state guard by collecting returned JSX-tree owners for canonical RecoveryCallout and StatePanel imports. Added focused tests that prove canonical adapters are allowed while bespoke recovery banners remain findings. Removed the now-stale Sidepanel ConnectionBanner local-recovery-banner baseline entry.

Verification: RED run of bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot failed the two new canonical adapter tests with local-recovery-banner findings. GREEN/final run passed 51/51 tests. bun run verify:design-system-state passed with Baseline exceptions: 510 and local-recovery-banner: 3. node --check apps/packages/ui/scripts/design-system-product-state-rules.mjs passed. git diff --check passed. bunx tsc --noEmit --pretty false exited 2 with 236 lines of pre-existing unrelated UI type errors; touched-file filter for design-system-product-state-rules, product-state-guard, design-system-product-state-baseline, task-45.32, RecoveryCallout, StatePanel, and ConnectionBanner returned no diagnostics. Bandit skipped because touched implementation/test/config files are UI JS/TS/JSON/Markdown only, with no Python execution surface. No standalone docs update was needed; the executable guard tests and baseline update document the rule behavior.

PR #1451 review follow-up: Qodo identified that the canonical recovery adapter tests only covered direct primitive returns, while the production motivating ConnectionBanner returns a wrapper element containing RecoveryCallout. Reopening the task to update tests so returned-JSX-tree scanning is covered explicitly before pushing a review-fix commit.

PR #1451 review fix implemented: updated the canonical recovery adapter tests so both RecoveryCallout and StatePanel are nested inside returned wrapper elements. This directly covers collectReturnedJsxTreeTagOwners behavior and matches the real Sidepanel ConnectionBanner shape.

Review-fix verification: bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed 51/51. bun run verify:design-system-state passed with Baseline exceptions: 510 and local-recovery-banner: 3. node --check apps/packages/ui/scripts/design-system-product-state-rules.mjs passed. git diff --check passed. bunx tsc --noEmit --pretty false still exits 2 on 236 lines of existing unrelated UI type errors, with no touched-file filter hits.

PR #1451 second review follow-up: CodeRabbit identified the recovery exemption is still too broad for mixed bespoke recovery UI plus nested canonical primitives. Reopening task to add a mixed recovery regression test and tighten recovery owner collection so only direct canonical returns or one-wrapper canonical adapter returns are exempted.

Second PR #1451 review fix implemented: added a mixed bespoke recovery markup regression test that fails under the old tree-wide recovery exemption, then replaced the recovery owner collection with a boundary helper. Recovery adapters are now exempt only when the returned JSX is the canonical primitive itself or a single returned wrapper whose only substantive child is RecoveryCallout/StatePanel. Mixed bespoke markup plus nested canonical recovery UI remains flagged as local-recovery-banner.

Second review-fix verification: RED run of bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot failed the mixed recovery test because findings were empty. GREEN/final run passed 52/52. bun run verify:design-system-state passed with Baseline exceptions: 510 and local-recovery-banner: 3. node --check apps/packages/ui/scripts/design-system-product-state-rules.mjs passed. git diff --check passed. bunx tsc --noEmit --pretty false still exits 2 on 236 lines of unrelated existing UI type errors, with no touched-file filter hits.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Change summary: taught the product-state guard to recognize canonical RecoveryCallout and StatePanel recovery adapters, removed the stale ConnectionBanner baseline entry, updated the adapter tests so canonical primitives are nested inside returned wrapper elements, and tightened the recovery exemption after review so mixed bespoke recovery markup plus a nested canonical primitive is still flagged.

Why: the production motivating adapter wraps RecoveryCallout in layout markup, but a tree-wide scan was too permissive. The final recovery rule allows direct canonical returns or one-wrapper canonical adapters while preserving detection for components that still render custom recovery UI alongside the canonical primitive.

Verification: focused guard suite passed 52/52 after a red/green mixed-recovery regression, design-system verifier passed with 510 baseline exceptions and 3 remaining local-recovery-banner entries, JS syntax check passed, git diff whitespace check passed, and full UI type-check still reports only unrelated existing diagnostics with no touched-file hits. Bandit remains not applicable for this UI-only JS/TS/JSON/Markdown change.
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
