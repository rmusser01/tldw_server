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
