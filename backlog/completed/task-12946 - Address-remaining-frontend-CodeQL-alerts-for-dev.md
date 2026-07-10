---
id: TASK-12946
title: Address remaining frontend CodeQL alerts for dev
status: Done
labels:
- security
- codeql
- frontend
priority: High
references:
- https://github.com/rmusser01/tldw_server/security/code-scanning
- https://github.com/rmusser01/tldw_server/pull/2696
- https://github.com/rmusser01/tldw_server/pull/2706
documentation:
- Docs/superpowers/specs/2026-07-10-remaining-frontend-codeql-alerts-design.md
modified_files:
- Docs/superpowers/specs/2026-07-10-remaining-frontend-codeql-alerts-design.md
- apps/packages/ui/src/components/Common/CharacterSelect.tsx
- apps/packages/ui/src/components/Common/Playground/DocumentGeneratorDrawer.tsx
- apps/packages/ui/src/components/Option/Watchlists/ItemsTab/__tests__/items-utils.domless.test.ts
- apps/packages/ui/src/components/Option/Watchlists/ItemsTab/__tests__/items-utils.test.ts
- apps/packages/ui/src/components/Option/Watchlists/ItemsTab/items-utils.ts
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.advanced-details.test.tsx
- apps/packages/ui/src/components/Quiz/tabs/ManageTab.tsx
- apps/packages/ui/src/components/Quiz/tabs/__tests__/ManageTab.bulk-duplicate.test.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/CharacterSelect.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/__tests__/CharacterSelect.persona-avatar.test.tsx
- apps/packages/ui/src/services/timeline/api.ts
- apps/packages/ui/src/services/watchlists.ts
- apps/packages/ui/src/types/__tests__/assistant-selection.test.ts
- apps/packages/ui/src/types/assistant-selection.ts
- apps/packages/ui/src/utils/__tests__/assistant-overlay.test.ts
- apps/packages/ui/src/utils/__tests__/codeql-source-contracts.test.ts
- apps/packages/ui/src/utils/__tests__/image-utils.test.ts
- apps/packages/ui/src/utils/__tests__/provider-registry-tts.test.ts
- apps/packages/ui/src/utils/assistant-overlay.ts
- apps/packages/ui/src/utils/image-utils.ts
- apps/packages/ui/src/utils/provider-registry.ts
- backlog/tasks/task-12946 - Address-remaining-frontend-CodeQL-alerts-for-dev.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the 12 JavaScript/TypeScript CodeQL alerts #2251-#2262 that remain unpatched on origin/dev after merged PR #2696 remediated the other 149 current alerts. Apply minimal root-cause fixes, add focused regression coverage, verify frontend typechecking and CodeQL-relevant behavior, and open a PR against dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All remaining CodeQL alert classes #2251-#2262 are addressed in source on a branch based on origin/dev.
- [x] #2 Focused regression tests cover unsafe HTML/URLs, OPML-free group filtering, provider inference, and logging behavior.
- [x] #3 Frontend typechecking and targeted tests pass; skipped checks are documented.
- [x] #4 A pull request is opened against dev with alert mapping and verification results.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Pre-implementation review selected four root-cause changes: use the existing server-side `groups` source query instead of OPML/client filtering; centralize an image-specific URL validator with analyzer-visible safe prefixes and embedded-image fallback; sanitize article HTML before the non-DOM text scanner; and preserve the printable quiz shell with DOMPurify `WHOLE_DOCUMENT` plus a trusted doctype. The task-specific implementation plan was executed with required red-green and two-stage review gates, then removed after all stages completed per repository instructions.

Implementation completed through commit a288abad0e. The fixes centralize safe raster/URL handling across alerted and sibling image sinks; remove untrusted HTML/XML parser paths while using the existing server-side groups query; escape/validate printable quiz fields and sanitize the whole document immediately before document.write; convert tainted console format strings to constant-first calls (including sibling timeline sinks); and rename the provider predicate member from match to matches without changing rule order or behavior.

Fresh verification on 2026-07-10: 10 focused Vitest files passed (132/132 tests); the CodeQL source-contract suite also passed from the required apps/tldw-frontend CI working directory (3/3); `NODE_OPTIONS=--max-old-space-size=8192 bun run typecheck` passed; `git diff --check origin/dev...HEAD` passed; final independent whole-branch review reported no remaining findings and Ready: Yes. Bandit is not applicable because no Python source changed.

Known verification limitation: alerts 2251-2262 remain open in the live default-branch inventory until an analyzed branch contains the fixes. The checked-in advanced CodeQL workflow analyzes only Python with SARIF upload disabled, and GitHub currently reports no CodeQL analyses for dev, so this dev-targeted PR cannot itself prove JavaScript alert closure. Residual pre-existing risks kept out of scope: combined group/type filtering retains the existing 1,000-item client-filter ceiling, and a print failure after opening the popup can leave a blank window.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved frontend CodeQL alerts 2251-2262 in PR #2706 against dev. The branch adds one shared image-source trust boundary, removes the reported HTML/XML parser paths, sanitizes the printable quiz document at its write sink, hardens tainted console format calls, and makes provider predicates analyzer-safe without changing inference behavior. Fresh verification passed 132/132 focused tests, 3/3 source contracts from the CI working directory, full frontend typechecking, and diff checks; independent final review found no remaining issues. Bandit was not applicable because no Python changed. JavaScript alert closure must be confirmed after the fixes reach a JavaScript-analyzed branch because dev currently has no CodeQL analysis.
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
