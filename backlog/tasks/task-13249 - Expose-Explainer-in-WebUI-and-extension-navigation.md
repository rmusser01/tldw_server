---
id: TASK-13249
title: Expose Explainer in WebUI and extension navigation
status: In Progress
assignee: []
created_date: '2026-09-13 18:22'
updated_date: '2026-09-13 19:02'
labels:
  - frontend
  - explainer
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2953'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The merged Explainer workspace is reachable directly but missing from the WebUI Pages launcher and shared route metadata. Add it to Research and page search using existing visibility and personalization behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pages launcher search and Research category offer a link to /explainer in current and legacy views.
- [x] #2 Explainer is available in default and researcher shortcuts and shared command palette metadata, while saved custom selections remain respected.
- [x] #3 Focused navigation and settings checks pass with verification recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Add focused launcher regression coverage; register Explainer in shortcut IDs, Research, English locale, and route metadata; verify settings compatibility, navigation suites, lint, and security applicability; self-review and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
User approved the bounded navigation follow-up. Backlog MCP resource and task search requests did not respond; using CLI fallback. Worktree: .worktrees/codex-explainer-navigation.

User explicitly requested WebUI and extension parity. Verified WXT resolves shared UI routes; the active shared route-registry.tsx is missing /explainer even though an older route file under tldw-frontend/extension contains it. Adding the active registration and a runtime lazy-route test. Initial 96 focused tests pass in both WebUI and extension Vitest configurations.

Implementation complete: shared Research launcher entry and persisted shortcut ID; default/Researcher presets; shared English label/description; command palette metadata; active extension route registration; WebUI and extension page inventories. Existing saved selections remain opt-in for Explainer via Show all features, and the previous Sources migration is preserved. Regression tests demonstrated missing launcher results and the extension Page not found state before the fix. WebUI and extension configurations each pass 106 focused tests across 11 suites. ESLint reports zero errors and six pre-existing no-explicit-any warnings on unchanged settings lines. Bandit is not applicable: touched code is TypeScript/TSX/JSON only. Next.js compiled /explainer and returned HTTP 200; live browser inspection stopped at the backend readiness guard because the local API was not ready. No live installed-extension E2E run or full production build performed.

Extension compile passed: bun run compile (tsc --noEmit -p tsconfig.compile.json). Self-review found no remaining issues in the changed scope. New route test formatted using the shared UI style; git diff --check passed. Follow-up runtime limitation: browser preview at http://127.0.0.1:8091/explainer requires a ready local backend; no backend settings changed.

Published draft PR #2953 against dev: https://github.com/rmusser01/tldw_server/pull/2953. Fetched origin/dev before publishing; the branch was current with no rebase needed. Reverified 106 tests across 11 suites under each WebUI and extension configuration, extension compile, and diff checks. Draft remains pending the human-written Change summary required by repository merge policy; browser verification limitations are documented in the PR.

User requested latest-dev rebase, resolution of PR comments and issues after Qodo review, and merge. Initial review readback shows no Qodo/inline review yet and a draft PR. User provided Change summary: Fixes navigation for explainer page. Requested the missing human-authored implementation rationale required before merge; review and verification continue independently.

Latest origin/dev rebase returned already up to date. Qodo completed its review of 228195c0e0 with zero bugs, zero rule violations, and zero requirement gaps: https://github.com/rmusser01/tldw_server/pull/2953#issuecomment-5655313750. GraphQL reviewThreads inventory is empty. No code review fixes were required. Reported required checks are passing; GitHub still reports BLOCKED while frontend unit shard 8/8 and remaining smoke jobs finish. Human implementation rationale remains pending before merge.

Frontend shard 8/8 failed: all 22 assertions also failed on exact dev, but one SkillsManager message differed and triggered full-context replay. The trusted replay refused the newly added option-explainer.route.test.tsx filename because it is absent from base. Consolidating the same real-router regression into the existing ExplainerWorkspace suite with shared fixtures; no test or gate will be disabled and no workflow policy changed.

Deeper investigation found the context replay also requires identical test identities, so test consolidation is not a sufficient root fix and will be reverted. CI Skills live-region test asserts Loading skills before async scope resolution starts the query; when it fails, clearAllMocks leaves its unconsumed mockImplementationOnce for the next empty-state test. The different Ant Design css-var IDs in that subsequent failure produce the ratchet mismatch. Fix only the live-region test synchronization, preserving all assertions and production behavior.

Verified the final CI fix: replaced only the premature Loading skills assertion with waitFor; original Explainer route test layout restored. All 83 SkillsManager tests pass (116.98s); 106 navigation tests pass in both WebUI and extension configs; extension compile passes; git diff --check passes. Shared UI ESLint uses the repository-pinned frontend binary: zero errors, 18 pre-existing no-explicit-any warnings on unchanged Skills test lines. Bandit remains not applicable to TS/TSX/JSON. Latest origin/dev remains c70387f496. No application behavior or CI gate policy was changed by this follow-up; awaiting new-head CI/Qodo and human rationale before merge.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Explainer is discoverable from Research and search in the shared Pages launcher for WebUI and extension, appears in default/Researcher shortcuts and shared page search, and resolves through the active extension options router. Both page inventories now include it. Saved custom selections are respected, and Sources migration compatibility is preserved. Verified 106 focused tests in each client configuration, extension TypeScript compile, zero lint errors, formatting, and diff checks. Live browser verification remains limited by local backend readiness; no production build or installed-extension E2E claimed.
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
