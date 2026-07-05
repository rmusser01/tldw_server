---
id: TASK-12817
title: Rebase PR 2541 and address Writing annotation review feedback
status: Done
labels:
- pr
- review
- writing
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/2541
modified_files:
- Docs/superpowers/plans/2026-06-29-pr2541-review-rebase.md
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-editor-adapter.test.ts
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-tiptap-utils.test.ts
- apps/packages/ui/src/components/Option/WritingPlayground/writing-editor-adapter.ts
- apps/packages/ui/src/components/Option/WritingPlayground/writing-tiptap-utils.ts
- backlog/tasks/task-12019 - UAT-Writing-Playground-manuscript-annotations.md
- backlog/tasks/task-12054 - Rebase-PR-2541-and-address-Writing-annotation-review-feedback.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #2541 onto latest origin/dev, evaluate all PR review comments, address valid Writing Playground annotation UAT feedback, verify the touched frontend scope, and push the updated PR branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR branch is rebased onto the latest `origin/dev`.
- [x] #2 All PR review comments and review threads are evaluated and addressed or documented as non-actionable.
- [x] #3 Focused Writing Playground annotation tests pass after review fixes.
- [x] #4 Formatting/lint/type/security checks for the touched scope pass or are documented with rationale.
- [x] #5 Updated branch is pushed to PR #2541 and review threads are resolved where applicable.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Plan: `Docs/superpowers/plans/2026-06-29-pr2541-review-rebase.md`
- Rebased PR #2541 onto `origin/dev` at `ccd07118855e153780011823155cef6cafa59030`.
- Addressed actionable review threads:
  - Preserved empty TipTap paragraphs by joining serialized blocks with a consistent `\n\n` separator.
  - Simplified top-level block separator mapping now that every previous block contributes two plain-text separator characters.
  - Added nested block-container serialization and selection-offset mapping for lists, list items, and blockquotes.
  - Converted single plain-text newlines to TipTap `hardBreak` nodes while retaining `\n\n` paragraph boundaries.
  - Narrowed the prior UAT task summary to WebUI evidence and explicit extension-harness blocked status.
- Verification:
  - `./node_modules/.bin/vitest run src/components/Option/WritingPlayground/__tests__/writing-tiptap-utils.test.ts src/components/Option/WritingPlayground/__tests__/writing-editor-adapter.test.ts --maxWorkers=1 --no-file-parallelism` passed: 17 tests.
  - `./node_modules/.bin/vitest run src/components/Option/WritingPlayground/__tests__/WritingTipTapEditor.external-sync.test.tsx src/components/Option/WritingPlayground/__tests__/WritingTipTapEditor.ssr-options.test.tsx src/components/Option/WritingPlayground/__tests__/useActiveManuscriptScene.test.tsx src/components/Option/WritingPlayground/__tests__/writing-editor-adapter.test.ts src/components/Option/WritingPlayground/__tests__/writing-tiptap-utils.test.ts --maxWorkers=1 --no-file-parallelism` passed: 30 tests before and after the final rebase onto `ccd07118855e153780011823155cef6cafa59030`.
  - `git diff --check` passed.
  - `./node_modules/.bin/tsc --noEmit --pretty false` hit Node's default heap limit. Retried as `NODE_OPTIONS=--max-old-space-size=8192 ./node_modules/.bin/tsc --noEmit --pretty false`; after temporarily correcting the local `antd` node_modules symlink for the check, it failed only on existing package-wide diagnostics outside the touched Writing Playground files.
  - Bandit skipped: touched implementation scope is TypeScript/Markdown, not Python.
- Pushed the rebased branch to `codex/writing-annotations-uat-polish`.
- Resolved all five actionable inline review threads and posted a PR summary comment: https://github.com/rmusser01/tldw_server/pull/2541#issuecomment-4828919427
- `gh pr view` confirmed PR #2541 remains open against `dev`.
- `gh pr checks` showed newly triggered GitHub checks still pending at handoff.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2541 onto latest `dev`, fixed the actionable Writing Playground annotation review feedback, pushed the updated branch, resolved the review threads, and posted the verification summary on the PR. Focused and broader Writing Playground Vitest slices passed; package-wide `tsc` remains blocked by existing unrelated UI diagnostics after the local dependency symlink issue is accounted for.
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
