# Task 9 Implementer Report

## Implementation Summary

- Replaced the Task 8 loaded placeholder with the approved recipient product surface: a compact identity/access header, stable Sources and Chat panes on desktop, semantic Sources/Chat tabs on mobile, and a responsive source preview drawer/sheet.
- Bound every source, history, preview, scope, submission, error, retry, and rate-limit state to `useSharedResearchWorkspace`. The UI preserves Task 8 `all` versus persistent cross-page `include` semantics and blocks all-scope submission above 500 queryable sources until the recipient chooses a 1-500 source subset.
- Added server-backed source search, state filtering, pagination, readiness reasons, select-all/clear controls, keyboard source activation, and chunk-focused citation previews.
- Reused `fetchChatModels`, `useModelSelector`, `ChatModelSelectorDropdown`, and `resolveStartupSelectedModel`. The exact server `generation_default` is seeded before startup preferences and injected as one bounded display option when absent from the generic catalog.
- Rendered preview text, model output, and citation quotes through the existing sanitized `Markdown` component. Origin links use only projected server values and `rel="noopener noreferrer"`.
- Added direct in-pane unavailable-provider, source-conflict, removed-source, retrieval, context-budget, and rate-limit copy while preserving draft and source selection.
- Added the required owner-facing revocation/provider paragraph to the existing Share dialog.
- Added and mirrored the English `sharedWorkspace` locale namespace. The mirror gate also exposed and corrected 17 pre-existing extension-locale aliases without changing source English copy.
- Kept the route gate isolated from the recipient component internals and retained a semantic loading heading.

## Pattern Research

1. Model selection: the local Research Workspace Chat pane's `fetchChatModels`, `useModelSelector`, `ChatModelSelectorDropdown`, and `resolveStartupSelectedModel` stack.
2. Sanitized Markdown: `@/components/Common/Markdown`, with external images and Mermaid disabled for recipient/server content.
3. Responsive interaction: existing Research Workspace `useMobile`, Ant Design Drawer, and semantic tab patterns.

## TDD Evidence

Exact RED command from the brief:

```text
cd apps/packages/ui && bunx vitest run src/components/Option/ResearchWorkspace/__tests__/SharedResearchWorkspace.test.tsx src/components/Option/ResearchWorkspace/__tests__/SharedResearchWorkspace.accessibility.test.tsx src/components/Option/ResearchWorkspace/__tests__/SharedResearchWorkspace.responsive.test.tsx --maxWorkers=1 --no-file-parallelism
```

Initial RED result:

```text
3 failed files; 12 failed tests.
The placeholder lacked the dedicated panes, responsive tabs, preview, model selector, chat, scope controls, and accessibility behavior.
```

Final GREEN result for the same three files:

```text
3 passed files; 18 passed tests.
```

Coverage includes identity/access authority, no recipient mutation controls, all/include scope, zero/over-500 gating, server source queries and pagination, exact default-model injection, UUID renewal, history deduplication, citation evidence, removed sources, error/draft/selection recovery, rate timing, heading focus, semantic tabs/labels/live regions, keyboard activation/focus return, stable controls, and bounded mobile/desktop tracks.

A supplemental keyboard RED produced 1 failed/2 passed before mobile ArrowLeft/ArrowRight tab activation and focus movement were added. Its GREEN rerun passed 3/3. That cycle also corrected the deferred chat fixture to echo the submitted UUID, so the success live-region assertion now exercises a controller-correlated stored response rather than a rejected mismatch.

## Verification

Final changed-scope matrix:

```text
8 passed files; 80 passed tests.
```

This matrix included the three Task 9 UI files, route gate, Share dialog, Task 8 reducer, canonical shared-workspace service, and locale mirror.

Locale mirror:

```text
2 passed tests.
```

Mandated complete focused command:

```text
64 files: 60 passed, 4 failed; 816 passed, 20 failed; 1 unhandled error.
```

Seven route-gate failures and the unhandled import error were caused by the route fixture importing the newly substantial model stack. The route fixture was narrowed to its boundary, the loading heading remained semantic, and its standalone rerun passed 17/17. The remaining 13 failures reproduce standalone in three untouched baseline files: `SourceViewControls.test.tsx` (7), `ResearchWorkspace.stage12.source-list-view-state.test.tsx` (5), and `SourcesPane.stage2.test.tsx` (1). They concern Ant modal/dropdown leave timing and an existing incomplete source-view fixture. No implementation file used by those failures was changed.

Focused ESLint over every touched TS/TSX file exited 0, with only the shared Next.js missing-pages notice for this library package. `git diff --check` passed.

Package TypeScript exhausted Node's 4 GB heap before diagnostics. A temporary changed-root config reached only existing test matcher and imported package environment/global diagnostics (`chrome`, `browser`, `import.meta.env`, optional OCR); no changed Task 9 production component had a diagnostic. The temporary config was removed.

Bandit is not applicable because Task 9 changes no Python production code.

## Files Changed

- `.superpowers/sdd/2026-08-21-recipient-shared-research-workspace-data-plane/progress.md`
- `.superpowers/sdd/2026-08-21-recipient-shared-research-workspace-data-plane/task-9-implementer-report.md`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/SharedResearchWorkspace/index.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/SharedResearchWorkspace/SharedWorkspaceHeader.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/SharedResearchWorkspace/SharedWorkspaceSourcesPane.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/SharedResearchWorkspace/SharedWorkspaceChatPane.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/SharedResearchWorkspace/SharedWorkspacePreview.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/ShareDialog.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/SharedResearchWorkspace.test.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/SharedResearchWorkspace.accessibility.test.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/SharedResearchWorkspace.responsive.test.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/shared-research-workspace-test-utils.ts`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspaceRouteGate.test.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ShareDialog.test.tsx`
- `apps/packages/ui/src/components/Option/Playground/__tests__/playground-locale-mirror.test.ts`
- `apps/packages/ui/src/assets/locale/en/playground.json`
- `apps/packages/ui/src/public/_locales/en/playground.json`
- `backlog/tasks/task-12020.40 - Bind-recipient-shared-workspace-sources-and-chat-to-the-canonical-share.md`

## Scope And Concerns

- No trust/shared/migration/status/onboarding banner, nested/decorative card, mutation toolbar, Studio, General Chat, notes, MCP, ACP, sandbox, artifacts, local workspace store/API request, redirect, alias, or `/research` coupling was added.
- Server `allowed_actions` remains the sole action authority. The recipient surface exposes read, preview, and grounded chat only.
- Extension capture/destination contracts are unchanged. The two unrelated untracked watchlist templates remain untouched and unstaged.
- The directory-wide focused suite remains non-green on the 13 reproducible baseline source-view failures described above. Task 11 still owns live backend/provider/browser UAT.

## Fix Round 1/5 - Recipient isolation, action authority, recovery, and responsive evidence

Reviewed head: `9846518465a0d3db2d1d306bf600513bbe4f2a1f`.

### Blocking findings addressed

- Replaced the generic product Markdown dependency with `SharedWorkspaceSafeMarkdown`, whose only imports are React, ReactMarkdown, and GFM. It skips raw HTML, drops every image, renders fenced code as inert `pre > code`, permits only absolute HTTP/HTTPS/mailto links, and has no route to CodeBlock, artifacts, Zustand, Mermaid, actions, iframe/sandbox, or storage settings. A structural import guard covers both the renderer and its two callers.
- Made `allowed_actions.inspect_sources` the sole source-inspection authority at the rendered controls, callback boundary, and Task 8 controller boundary. Search, state filtering, pagination, selection, and preview fail closed and render the server reason. Header capability text comes only from `allowed_actions`; `access_level` is retained only as the policy-ceiling tooltip tier. Model readiness can block submission without rewriting the server action decision.
- Replaced invented UI error codes with canonical `no_provider_configured`, `generation_failed`, `shared_chat_context_too_large`, `retrieval_unavailable`, and preview `shared_workspace_not_found`, using `TldwApiError` fixtures.
- Preserved the first existing message's viewport offset when older history is prepended. History failures now render an alert and retry action, and upward pagination no longer invokes newest-message scrolling.
- Added `previewStarted`, `previewLoading`, and `previewTarget` to the Task 8 reducer/controller. Starting a newer preview immediately clears old evidence; aborted or reordered older responses cannot render under the newer target.
- Strengthened mobile tabpanel, keyboard, focus-return, dynamic submit-label, dynamic preview-label, and drawer behavior tests. Native buttons now use native Enter/Space activation, and both mobile tabpanels remain mounted with the inactive one hidden.

### TDD evidence

Initial blocking RED:

```text
cd apps/packages/ui && bunx vitest run src/components/Option/ResearchWorkspace/__tests__/SharedWorkspaceSafeMarkdown.test.tsx src/components/Option/ResearchWorkspace/__tests__/SharedResearchWorkspace.test.tsx src/components/Option/ResearchWorkspace/__tests__/shared-research-workspace-reducer.test.ts --maxWorkers=1 --no-file-parallelism
```

Result: 3 failed files; 8 failed and 36 passed. The safe renderer was absent, source inspection was not action-gated, history lacked anchor/retry behavior, recovery fixtures used invented codes, and preview start did not clear stale evidence. The same command passed 3 files/46 tests after the first implementation pass.

Semantic/dynamic-label RED used the exact Task 9 UI command from the brief and produced 2 failed files; 4 failed and 19 passed. Stable mobile tabpanels, dynamic submit labeling, and dynamic preview labeling were absent. One intermediate rerun exposed the Ant Drawer title as its accessible name; the final exact command passed 3 files/23 tests:

```text
cd apps/packages/ui && bunx vitest run src/components/Option/ResearchWorkspace/__tests__/SharedResearchWorkspace.test.tsx src/components/Option/ResearchWorkspace/__tests__/SharedResearchWorkspace.accessibility.test.tsx src/components/Option/ResearchWorkspace/__tests__/SharedResearchWorkspace.responsive.test.tsx --maxWorkers=1 --no-file-parallelism
```

Controller authority RED:

```text
cd apps/packages/ui && bunx vitest run src/components/Option/ResearchWorkspace/__tests__/shared-research-workspace-reducer.test.ts --maxWorkers=1 --no-file-parallelism -t "makes every inspect operation"
```

Result: 1 failed/29 skipped before the controller guard; GREEN: 1 passed/29 skipped. Final Task 8 controller plus safe-renderer command passed 2 files/32 tests.

Final self-review authority RED:

```text
cd apps/packages/ui && bunx vitest run src/components/Option/ResearchWorkspace/__tests__/shared-research-workspace-reducer.test.ts --maxWorkers=1 --no-file-parallelism -t "without rewriting server actions"
```

Result: 1 failed/29 skipped because an inconsistent generation default rewrote `ask_grounded_questions` to denied. After keeping model readiness separate from authorization, GREEN was 1 passed/29 skipped. Final combined Task 9/controller/safe-renderer verification passed 5 files/55 tests.

Covering test files:

- `SharedWorkspaceSafeMarkdown.test.tsx`
- `SharedResearchWorkspace.test.tsx`
- `SharedResearchWorkspace.accessibility.test.tsx`
- `SharedResearchWorkspace.responsive.test.tsx`
- `shared-research-workspace-reducer.test.ts`

### Regression and static verification

Changed-scope command covering Task 9 UI, Task 8 controller, route gate, Share dialog, shared-workspace domain service, and locale mirror passed 9 files/90 tests.

The mandated complete focused command was run verbatim:

```text
cd apps/packages/ui && bunx vitest run src/components/Option/ResearchWorkspace/__tests__ src/services/tldw/domains/__tests__/shared-workspaces.test.ts src/components/Option/Playground/__tests__/playground-locale-mirror.test.ts --maxWorkers=1 --no-file-parallelism
```

Result: 65 files; 61 passed/4 failed; 832 passed/14 failed. Thirteen failures are the already documented untouched baseline: 7 `SourceViewControls`, 5 Stage 12 saved-view state, and 1 incomplete `SourcesPane.stage2` fixture. The additional `StudioPane.stage2` test timed out only under full-suite load and passed standalone in 530 ms (1 passed/31 skipped).

Locale mirror passed 1 file/2 tests. Focused ESLint over every changed TS/TSX file exited 0 with only the package's existing Next missing-pages notice. `node --check apps/tldw-frontend/scripts/shared-workspace-task9-cdp-check.mjs` exited 0. Package TypeScript completed with the repository's existing diagnostics; a second filtered run produced no diagnostic matching `SharedWorkspace`, `SharedResearchWorkspace`, or `shared-research-workspace`. The production static forbidden-dependency/invented-code grep returned no matches, while canonical error-code grep found only the required mappings. Bandit is not applicable to frontend-only production. `git diff --check` passed.

### CDP-only browser geometry

Server command:

```text
cd apps/tldw-frontend && NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run dev -- -p 18109
```

Deterministic raw-CDP command, with no computer-control tool:

```text
cd apps/tldw-frontend && TLDW_WEB_URL=http://127.0.0.1:18109 node scripts/shared-workspace-task9-cdp-check.mjs
```

Final measured result:

- Mobile 390x844: document/body/shell/source horizontal overflow all 0; shell 390x753 at top 91; Sources panel 390x253 at top 195; inactive Chat panel measured 0x0; no control escaped viewport bounds. The loading preview sheet measured top 0, bottom 844, width 390, height 844 with body overflow 0; its accessible name was `Loading source preview`, then changed to the loaded preview name.
- Desktop 1440x900: document/body/shell/source/chat horizontal overflow all 0; shell x=48, width 1392, height 849; Sources x=48..549.109, width 501.109, height 788; Chat x=549.109..1440, width 890.891, height 788; no control escaped bounds. Submit measured 44x44 before and during submission; its name changed from `Ask shared workspace` to `Asking shared workspace` and it was disabled while pending.

Browser-check diagnostics were bounded to three attempts: the first found no Playwright-managed browser and switched to installed Chrome; the second timed out because deterministic fixtures lacked CORS preflight; the third exposed a checker assertion that treated a mounted hidden tabpanel as visible. CORS was stubbed and visibility was changed to measured geometry; the final command passed. The dev server was stopped cleanly.

### Scope and concerns

- The two English locale formats remain value-equivalent under the canonical mirror test.
- The two unrelated untracked watchlist templates remain untouched and unstaged.
- The 13 untouched local-workspace baseline failures remain outside Task 9. Task 11 still owns real-backend/provider UAT; this round deliberately adds only deterministic focused CDP behavior evidence.
