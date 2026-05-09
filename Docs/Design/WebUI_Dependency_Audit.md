# WebUI Dependency Audit

Date: 2026-05-07
Status: Draft audit for issue #1346

## References

- GitHub issue: https://github.com/rmusser01/tldw_server/issues/1346
- Design spec: ../superpowers/specs/2026-05-07-webui-dependency-trimming-design.md
- Parent design task: TASK-100
- Audit task: TASK-104
- Lockfile follow-up tasks: TASK-134, TASK-141
- Audit refresh task: TASK-144

## Scope

This audit covers direct package declarations and usage signals for:

- `apps/tldw-frontend/package.json`
- `apps/packages/ui/package.json`
- `apps/bun.lock`
- `apps/extension/package.json` as an impact-check surface for shared UI candidates

This audit does not remove packages or rewrite runtime code.

## Methodology

1. Read direct dependency declarations from the WebUI, shared UI, and extension manifests.
2. Scan source, test, script, and config files for import/config usage.
3. Classify direct dependencies with the approved decision values.
4. Rank follow-up work into quick cleanup, replacement, deferred design, and keep groups.
5. Record verification commands and known skips.

## Decision Legend

| Decision | Meaning |
| --- | --- |
| `keep` | Current dependency is justified. |
| `remove-now` | Candidate for a narrow package-removal PR. |
| `replace-later` | Replacement is plausible but needs its own PR. |
| `defer-design` | Needs a separate design before replacement. |
| `investigate-lockfile` | Needs lockfile or ownership confirmation before action. |
| `removed` | Direct declaration was removed by a completed follow-up; row retained for audit history. |

## Dependency Inventory

| Package | Declared locations | Import count | Representative sites | Consumer surface | Category | Decision | Risk | Expected impact | Follow-up slice |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `@ant-design/cssinjs` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 3 | apps/packages/ui/src/entries/shared/AppShell.tsx, apps/tldw-frontend/__tests__/components/app-providers-import.test.tsx, apps/tldw-frontend/components/AppProviders.tsx | shared UI, web tests, web app | frontend/runtime | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `@ant-design/icons` | `web:dependencies`, `shared-ui:peerDependencies` | 28 | apps/packages/ui/src/components/Common/confirm-danger.tsx, apps/packages/ui/src/components/Notes/NotesEditorPane.tsx, apps/packages/ui/src/components/Notes/NotesSidebar.tsx, apps/packages/ui/src/components/Option/Admin/BillingDashboardPage.tsx | shared UI | icons | `defer-design` | Medium; icon consolidation touches many visible components and needs visual review. | Potentially meaningful bundle reduction only after an icon-system design. | Icon-stack consolidation design. |
| `@axe-core/playwright` | `web:devDependencies` | 2 | apps/tldw-frontend/e2e/smoke/composer-a11y.spec.ts, apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts | web tests | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `@dnd-kit/abstract` | `web:dependencies` | 0 | none found | web app declaration only | drag/drop | `investigate-lockfile` | Low/medium; source usage is absent or indirect, but dependency graph or scripts need confirmation. | Small if removable; avoid manifest churn until confirmed. | Lockfile/manifest investigation slice. |
| `@dnd-kit/collision` | `shared-ui:peerDependencies`, `extension:dependencies` | 5 | apps/packages/ui/src/components/Option/AudiobookStudio/ChapterEditor/SortableChapterItem.tsx, apps/packages/ui/src/components/Option/DataTables/EditableDataTable.tsx, apps/packages/ui/src/components/Option/DataTables/TablePreview.tsx, apps/packages/ui/src/components/Option/KanbanPlayground/BoardView.tsx | shared UI | drag/drop | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `@dnd-kit/dom` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 0 | none found | web app, shared UI, extension impact declaration only | drag/drop | `investigate-lockfile` | Low/medium; source usage is absent or indirect, but dependency graph or scripts need confirmation. | Small if removable; avoid manifest churn until confirmed. | Lockfile/manifest investigation slice. |
| `@dnd-kit/helpers` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 1 | apps/packages/ui/src/components/Option/KanbanPlayground/BoardView.tsx | shared UI | drag/drop | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `@dnd-kit/react` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 10 | apps/packages/ui/src/components/Option/AudiobookStudio/ChapterEditor/ChapterList.tsx, apps/packages/ui/src/components/Option/AudiobookStudio/ChapterEditor/SortableChapterItem.tsx, apps/packages/ui/src/components/Option/DataTables/EditableDataTable.tsx, apps/packages/ui/src/components/Option/DataTables/TablePreview.tsx | shared UI | drag/drop | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `@eslint/eslintrc` | none after TASK-141 | 0 | none found; flat config uses `@eslint/js` and concrete plugins instead | removed WebUI dev declaration | tooling/dev | `removed` | Low; direct declaration had no source/config/script evidence. The package remains in `apps/bun.lock` only as an ESLint transitive dependency. | One direct dev declaration removed; no standalone lockfile package removal because ESLint still owns it transitively. | TASK-141 complete |
| `@eslint/js` | `web:devDependencies` | 1 | apps/tldw-frontend/eslint.config.mjs | web config/script | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `@heroicons/react` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 2 | apps/packages/ui/src/components/Option/Settings/TldwConnectionSettings.tsx, apps/packages/ui/src/components/Option/Settings/__tests__/tldw-review-comments.test.tsx | shared UI, shared UI tests | icons | `defer-design` | Medium; icon consolidation touches many visible components and needs visual review. | Potentially meaningful bundle reduction only after an icon-system design. | Icon-stack consolidation design. |
| `@hookform/resolvers` | none after PR #1365 | 0 | none found | removed WebUI declaration | frontend/runtime | `removed` | Low; no import/config/package-script evidence in scanned roots before removal. | One direct runtime declaration removed; no active manifest declaration remains. | PR #1365 complete |
| `@monaco-editor/react` | `web:dependencies`, `shared-ui:peerDependencies` | 3 | apps/packages/ui/src/components/Option/Watchlists/TemplatesTab/TemplateCodeEditor.tsx, apps/tldw-frontend/components/ui/JsonEditor.tsx | shared UI, web app | editor/terminal | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `@mozilla/readability` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 2 | apps/packages/ui/src/parser/default.ts, apps/packages/ui/src/parser/reader.ts | shared UI | parser/conversion | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `@next/eslint-plugin-next` | `web:devDependencies` | 1 | apps/tldw-frontend/eslint.config.mjs | web config/script | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `@plasmohq/storage` | `shared-ui:peerDependencies`, `extension:dependencies` | 228 | apps/packages/ui/src/components/Agent/WorkspaceSelector.tsx, apps/packages/ui/src/components/Common/AssistantSelect.tsx, apps/packages/ui/src/components/Common/CharacterSelect.tsx, apps/packages/ui/src/components/Common/ChatGreetingPicker.tsx | shared UI, shared UI tests, web tests, web app | state/data | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `@playwright/test` | `web:devDependencies`, `extension:devDependencies` | 302 | apps/extension/playwright.config.ts, apps/extension/scripts/review-ux.js, apps/extension/scripts/sidepanel-ux-live.js, apps/extension/scripts/spotcheck.js | extension tests/config, extension, web tests, web config/script, package scripts | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `@radix-ui/react-dialog` | `web:dependencies` | 1 | apps/tldw-frontend/components/content-review/ReattachSourceModal.tsx | web app | frontend/runtime | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `@sentry/nextjs` | `web:dependencies` | 5 | apps/tldw-frontend/components/ErrorBoundary.tsx, apps/tldw-frontend/next.config.mjs, apps/tldw-frontend/sentry.client.config.ts, apps/tldw-frontend/sentry.edge.config.ts | web app, web config/script | frontend/runtime | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `@tailwindcss/forms` | `web:devDependencies`, `extension:dependencies` | 1 | apps/tldw-frontend/tailwind.config.js | styling config/build | styling/build | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `@tailwindcss/typography` | `web:devDependencies`, `extension:dependencies` | 1 | apps/tldw-frontend/tailwind.config.js | styling config/build | styling/build | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `@tanstack/react-query` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 486 | apps/packages/ui/src/components/Common/CharacterSelect.tsx, apps/packages/ui/src/components/Common/ChatSidebar/FolderChatList.tsx, apps/packages/ui/src/components/Common/ChatSidebar/LocalChatList.tsx, apps/packages/ui/src/components/Common/ChatSidebar/ServerChatList.tsx | shared UI, shared UI tests, web tests, web app | state/data | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `@tanstack/react-virtual` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 15 | apps/packages/ui/src/components/Review/MediaReviewReadingPane.tsx, apps/packages/ui/src/components/Review/__tests__/MediaReviewPage.stage1.selectionLimit.test.tsx, apps/packages/ui/src/components/Review/__tests__/MediaReviewPage.stage3.search-filter-sort.test.tsx, apps/packages/ui/src/components/Review/__tests__/MediaReviewPage.stage4.card-density.test.tsx | shared UI, shared UI tests | state/data | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `@testing-library/jest-dom` | `web:devDependencies`, `shared-ui:devDependencies` | 2 | apps/packages/ui/vitest.setup.ts, apps/tldw-frontend/vitest.setup.ts | shared UI tests, web app | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `@testing-library/react` | `web:devDependencies`, `shared-ui:devDependencies` | 861 | apps/packages/ui/src/components/Chat/composer/__tests__/BriefField.test.tsx, apps/packages/ui/src/components/Chat/composer/__tests__/ChatComposer.test.tsx, apps/packages/ui/src/components/Chat/composer/__tests__/ComposerStyleSettings.test.tsx, apps/packages/ui/src/components/Chat/composer/__tests__/FacetRow.test.tsx | shared UI tests, web tests, web app | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `@testing-library/user-event` | `web:devDependencies`, `shared-ui:devDependencies` | 160 | apps/packages/ui/src/components/Common/Playground/__tests__/Message.error-recovery.integration.test.tsx, apps/packages/ui/src/components/Common/Playground/__tests__/MessageSource.integration.test.tsx, apps/packages/ui/src/components/Common/QuickIngest/__tests__/FileDropZone.acceptance.test.tsx, apps/packages/ui/src/components/Common/QuickIngest/__tests__/IngestWizardContext.test.tsx | shared UI tests, web tests | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `@tiptap/core` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 3 | apps/packages/ui/src/components/Option/WritingPlayground/extensions/AIAnnotationExtension.ts, apps/packages/ui/src/components/Option/WritingPlayground/extensions/CitationExtension.ts, apps/packages/ui/src/components/Option/WritingPlayground/extensions/SceneBreakExtension.ts | shared UI | editor/terminal | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `@tiptap/extension-character-count` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 1 | apps/packages/ui/src/components/Option/WritingPlayground/WritingTipTapEditor.tsx | shared UI | editor/terminal | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `@tiptap/extension-placeholder` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 1 | apps/packages/ui/src/components/Option/WritingPlayground/WritingTipTapEditor.tsx | shared UI | editor/terminal | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `@tiptap/pm` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 0 | none found | web app, shared UI, extension impact declaration only | editor/terminal | `investigate-lockfile` | Medium; no import/config/package-script evidence, but package sits in editor behavior. Confirm direct-vs-transitive ownership and Tiptap peer/runtime coverage before removal. | Potential install/bundle reduction if direct declaration proves unused. | Lockfile/editor-domain investigation slice. |
| `@tiptap/react` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 9 | apps/packages/ui/src/components/Option/WritingPlayground/WritingTipTapEditor.tsx, apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingTipTapEditor.external-sync.test.tsx, apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-session-payload-utils.test.ts, apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-tiptap-utils.test.ts | shared UI, shared UI tests | editor/terminal | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `@tiptap/starter-kit` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 1 | apps/packages/ui/src/components/Option/WritingPlayground/WritingTipTapEditor.tsx | shared UI | editor/terminal | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `@tldw/ui` | `web:dependencies`, `extension:dependencies` | 48 | apps/extension/entrypoints/background.ts, apps/extension/entrypoints/copilot-popup.content.tsx, apps/extension/entrypoints/hf-pull.content.ts, apps/extension/entrypoints/options/main.tsx | extension, extension tests/config, web app | frontend/runtime | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `@types/cytoscape` | `web:devDependencies`, `extension:devDependencies` | 0 | none found | tooling/dev declaration only | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `@types/d3-dsv` | `web:devDependencies`, `extension:devDependencies` | 0 | none found | tooling/dev declaration only | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `@types/html-to-text` | `web:devDependencies`, `extension:devDependencies` | 0 | none found | tooling/dev declaration only | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `@types/katex` | `web:devDependencies` | 0 | none found | tooling/dev declaration only | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `@types/node` | `web:devDependencies`, `extension:devDependencies` | 0 | none found | tooling/dev declaration only | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `@types/pubsub-js` | none after PR #1357 | 0 | none found | removed WebUI and extension dev declarations | tooling/dev | `removed` | Low; no import/config/package-script evidence in scanned roots before removal. | Two direct dev declarations removed; no active manifest declaration remains. | PR #1357 complete |
| `@types/react` | `web:devDependencies`, `shared-ui:devDependencies`, `extension:devDependencies` | 0 | none found | tooling/dev declaration only | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `@types/react-dom` | `web:devDependencies`, `shared-ui:devDependencies`, `extension:devDependencies` | 0 | none found | tooling/dev declaration only | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `@types/react-syntax-highlighter` | none after TASK-134 | 0 | none found | removed WebUI dev declaration | tooling/dev | `removed` | Low; type package had no source/config/script evidence and belonged to the removed `react-syntax-highlighter` declaration. | One direct dev declaration and its lockfile record removed. | TASK-134 complete |
| `@types/turndown` | `web:devDependencies`, `extension:devDependencies` | 0 | none found | tooling/dev declaration only | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `@typescript-eslint/eslint-plugin` | `web:devDependencies` | 1 | apps/tldw-frontend/eslint.config.mjs | web config/script | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `@typescript-eslint/parser` | `web:devDependencies` | 1 | apps/tldw-frontend/eslint.config.mjs | web config/script | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `@vitejs/plugin-react` | `web:devDependencies`, `extension:dependencies` | 2 | apps/tldw-frontend/vitest.config.ts, apps/tldw-frontend/vitest.extension.config.ts | web config/script | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `@vitest/coverage-v8` | `web:devDependencies` | 1 | apps/tldw-frontend/package.json test:coverage script | package scripts | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `@xterm/addon-fit` | `web:dependencies`, `shared-ui:peerDependencies` | 2 | apps/packages/ui/src/components/Option/ACPPlayground/ACPWorkspacePanel.tsx, apps/tldw-frontend/__tests__/extension/entry-shell-performance.test.ts | shared UI, web tests | editor/terminal | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `@xyflow/react` | `shared-ui:peerDependencies`, `extension:dependencies` | 7 | apps/packages/ui/src/components/WorkflowEditor/WorkflowCanvas.tsx, apps/packages/ui/src/components/WorkflowEditor/connection-validation.ts, apps/packages/ui/src/components/WorkflowEditor/nodes/WorkflowNode.tsx, apps/packages/ui/src/store/workflow-editor.ts | shared UI | graph/rendering | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `ajv` | `web:dependencies` | 1 | apps/tldw-frontend/lib/ajv.ts | web app | schema validation | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `antd` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 1135 | apps/packages/ui/src/components/Agent/AgentErrorBoundary.tsx, apps/packages/ui/src/components/Agent/ApprovalBanner.tsx, apps/packages/ui/src/components/Agent/DiffViewer.tsx, apps/packages/ui/src/components/Agent/ErrorBoundaryTestTrigger.tsx | shared UI, shared UI tests, shared UI config, web tests, web app | frontend/runtime | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `autoprefixer` | `web:devDependencies`, `extension:devDependencies` | 2 | apps/extension/postcss.config.js, apps/tldw-frontend/postcss.config.mjs | styling config/build | styling/build | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `axe-core` | `shared-ui:devDependencies` | 10 | apps/packages/ui/src/components/Common/NotesDock/__tests__/NotesDockPanel.stage2.accessibility-regression.test.tsx, apps/packages/ui/src/components/Common/Playground/__tests__/Playground.accessibility-regression.test.tsx, apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage22.accessibility-regression.test.tsx, apps/packages/ui/src/components/Option/Dictionaries/__tests__/Manager.accessibilityStage3.test.tsx | shared UI tests | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `axios` | none after PR #1375 | 0 | no active package imports after fetch-helper migration | removed WebUI, shared UI, and extension declarations | api/transport | `removed` | High before migration; transport wrapper behavior was preserved in the dedicated fetch-backed helper PR. | Direct runtime declarations removed; `axios` remains in `apps/bun.lock` only through optional/transitive ownership outside active WebUI imports. | PR #1375 complete |
| `buffer` | none after PR #1359 | 0 | none found | removed WebUI and extension declarations | polyfill/shim | `removed` | Low; no package import/config/package-script evidence in scanned roots before removal. | Two direct runtime declarations removed; `buffer` remains in `apps/bun.lock` only through unrelated transitive packages. | PR #1359 complete |
| `cheerio` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 5 | apps/packages/ui/src/parser/amazon.ts, apps/packages/ui/src/parser/default.ts, apps/packages/ui/src/parser/google-sheets.ts, apps/packages/ui/src/parser/twitter.ts | shared UI | parser/conversion | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `clsx` | none after PR #1368 | 0 | no active package imports after local helper compatibility slice | removed WebUI declaration | classnames | `removed` | Medium before migration; compatibility tests covered the local class-name helper shape. | One direct runtime declaration removed; `clsx` remains in `apps/bun.lock` only through UI-library transitives. | PR #1368 complete |
| `cross-env` | `web:devDependencies`, `extension:devDependencies` | 18 | apps/tldw-frontend/package.json scripts, apps/extension/package.json scripts | package scripts | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `cytoscape` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 6 | apps/packages/ui/src/components/Notes/NotesGraphModal.tsx, apps/packages/ui/src/components/Notes/__tests__/NotesGraphModal.stage2.graph-view.test.tsx, apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage21.accessibility-modal-focus.test.tsx, apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage22.accessibility-regression.test.tsx | shared UI, shared UI tests | graph/rendering | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `cytoscape-dagre` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 6 | apps/packages/ui/src/components/Notes/NotesGraphModal.tsx, apps/packages/ui/src/components/Notes/__tests__/NotesGraphModal.stage2.graph-view.test.tsx, apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage21.accessibility-modal-focus.test.tsx, apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage22.accessibility-regression.test.tsx | shared UI, shared UI tests | graph/rendering | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `d3-dsv` | `web:dependencies`, `extension:dependencies` | 0 | none found | web app, extension impact declaration only | parser/conversion | `investigate-lockfile` | Medium; no import/config/package-script evidence, but package sits in parser/conversion behavior. Confirm direct-vs-transitive ownership and CSV/DSV coverage before removal. | Potential install/bundle reduction if direct declaration proves unused. | Lockfile/parser-domain investigation slice. |
| `dayjs` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 7 | apps/packages/ui/src/components/Media/FilterPanel.tsx, apps/packages/ui/src/components/Option/Collections/ReadingList/ReadingItemsList.tsx, apps/packages/ui/src/components/Option/DataTables/EditableCell.tsx, apps/packages/ui/src/components/Option/Items/ItemsWorkspace.tsx | shared UI | frontend/runtime | `defer-design` | Medium; remaining shared UI imports are Ant Design DatePicker/DateRangePicker value surfaces that currently exchange `Dayjs` values and types. | No immediate dependency reduction until shared UI date-picker value contracts are redesigned or isolated. | Date-picker contract design before manifest removal. |
| `dexie` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 6 | apps/packages/ui/src/db/dexie/chat.ts, apps/packages/ui/src/db/dexie/schema.ts, apps/packages/ui/src/hooks/document-workspace/__tests__/offlineQueue.test.ts, apps/packages/ui/src/hooks/document-workspace/offlineQueue.ts | shared UI, shared UI tests, web tests, web app | state/data | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `dexie-react-hooks` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 1 | apps/packages/ui/src/components/Sidepanel/Chat/TtsClipsDrawer.tsx | shared UI | state/data | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `dompurify` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 11 | apps/packages/ui/src/components/Common/CodeBlock.tsx, apps/packages/ui/src/components/Notes/NotesStudioDiagramCard.tsx, apps/packages/ui/src/components/Notes/export-utils.ts, apps/packages/ui/src/components/Option/Collections/ReadingList/ReadingItemDetail.tsx | shared UI | security/sanitization | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `epubjs` | `web:dependencies`, `shared-ui:peerDependencies` | 8 | apps/packages/ui/src/components/DocumentWorkspace/DocumentViewer/EpubViewer/EpubSearch.tsx, apps/packages/ui/src/components/DocumentWorkspace/DocumentViewer/EpubViewer/index.tsx, apps/packages/ui/src/hooks/document-workspace/useEpubOutline.ts, apps/packages/ui/src/hooks/document-workspace/useEpubReader.ts | shared UI | document/rendering | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `eslint` | `web:devDependencies` | 2 | apps/tldw-frontend/package.json lint scripts, apps/tldw-frontend/eslint.config.mjs | package scripts | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `eslint-config-next` | none after TASK-141 | 0 | none found; flat config imports `@next/eslint-plugin-next` directly | removed WebUI dev declaration | tooling/dev | `removed` | Low; package scripts call `eslint .`, not Next's legacy ESLint config. | One direct dev declaration removed; `eslint-config-next` and its import/a11y resolver tree dropped from the lockfile. | TASK-141 complete |
| `eslint-config-prettier` | none after TASK-141 | 0 | none found; no flat config import found | removed WebUI dev declaration | tooling/dev | `removed` | Low; the active flat config does not extend Prettier's ESLint config and no package script invokes its CLI. | One direct dev declaration removed; package record dropped from the lockfile. | TASK-141 complete |
| `eslint-plugin-react` | `web:devDependencies` | 1 | apps/tldw-frontend/eslint.config.mjs | web config/script | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `eslint-plugin-react-hooks` | `web:devDependencies` | 1 | apps/tldw-frontend/eslint.config.mjs | web config/script | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `exceljs` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 2 | apps/extension/tests/unit/data-table-export.test.ts, apps/packages/ui/src/utils/data-table-export.ts | extension tests/config, shared UI | document/rendering | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `fake-indexeddb` | none after TASK-141 | 0 | none found; no test setup import found | removed WebUI dev declaration | tooling/dev | `removed` | Low; tests rely on current jsdom/browser shims and had no direct fake IndexedDB package usage. | One direct dev declaration removed; package record dropped from the lockfile. | TASK-141 complete |
| `globals` | `web:devDependencies` | 1 | apps/tldw-frontend/eslint.config.mjs | web config/script | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `gpt-tokenizer` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 2 | apps/packages/ui/src/components/Option/Repo2Txt/formatter/TokenizerWorker.ts, apps/packages/ui/src/components/Option/Repo2Txt/workers/tokenizer.worker.ts | shared UI | parser/conversion | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `html-to-text` | `web:dependencies`, `extension:dependencies` | 0 | none found | web app, extension impact declaration only | parser/conversion | `investigate-lockfile` | Medium; no import/config/package-script evidence, but package sits in parser/conversion behavior. Confirm direct-vs-transitive ownership and HTML text-conversion coverage before removal. | Potential install/bundle reduction if direct declaration proves unused. | Lockfile/parser-domain investigation slice. |
| `html2canvas` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 2 | apps/packages/ui/src/components/Layouts/MoreOptions.tsx, apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/ArtifactModalContent.tsx | shared UI | document/rendering | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `i18next` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 61 | apps/packages/ui/src/components/Agent/AgentErrorBoundary.tsx, apps/packages/ui/src/components/Agent/ToolCallLog.tsx, apps/packages/ui/src/components/Common/ChatQueuePanel.tsx, apps/packages/ui/src/components/Common/ChatSidebar/ServerChatRow.tsx | shared UI, shared UI tests, web app | i18n | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `i18next-browser-languagedetector` | `web:dependencies`, `extension:dependencies` | 0 | none found | web app, extension impact declaration only | i18n | `investigate-lockfile` | Low/medium; source usage is absent or indirect, but dependency graph or scripts need confirmation. | Small if removable; avoid manifest churn until confirmed. | Lockfile/manifest investigation slice. |
| `i18next-icu` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 1 | apps/packages/ui/src/i18n/icu-format.ts | shared UI | i18n | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `jsdom` | `web:devDependencies`, `shared-ui:devDependencies` | 0 | none found | tooling/dev declaration only | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `jszip` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 4 | apps/packages/ui/src/components/Option/Repo2Txt/providers/LocalProvider.ts, apps/packages/ui/src/store/__tests__/workspace-bundle.test.ts, apps/packages/ui/src/store/workspace-bundle.ts | shared UI, shared UI tests | document/rendering | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `katex` | `web:dependencies`, `shared-ui:peerDependencies` | 2 | apps/packages/ui/src/components/Common/Markdown.tsx, apps/packages/ui/src/utils/marked/katex.tsx | shared UI | markdown/rendering | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `lucide-react` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 568 | apps/packages/ui/src/components/Agent/AgentErrorBoundary.tsx, apps/packages/ui/src/components/Agent/ApprovalBanner.tsx, apps/packages/ui/src/components/Agent/DiffViewer.tsx, apps/packages/ui/src/components/Agent/ErrorBoundaryTestTrigger.tsx | shared UI, shared UI tests, shared UI config, web tests, web app | icons | `defer-design` | Medium; icon consolidation touches many visible components and needs visual review. | Potentially meaningful bundle reduction only after an icon-system design. | Icon-stack consolidation design. |
| `marked` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 4 | apps/packages/ui/src/components/Notes/export-utils.ts, apps/packages/ui/src/components/Option/Watchlists/TemplatesTab/TemplatePreviewPane.tsx, apps/packages/ui/src/utils/chat-rich-text.ts, apps/packages/ui/src/utils/clipboard.ts | shared UI | markdown/rendering | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `mermaid` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 1 | apps/packages/ui/src/components/Common/Mermaid.tsx | shared UI | markdown/rendering | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `next` | `web:dependencies` | 158 | apps/tldw-frontend/next.config.mjs, apps/tldw-frontend/components/layout/Header.tsx, apps/tldw-frontend/components/landing/LandingLayout.tsx, apps/tldw-frontend/hooks/useAuth.tsx | web tests, web app, package scripts | frontend/runtime | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `pa-tesseract.js` | `web:dependencies`, `extension:dependencies` | 1 | apps/packages/ui/src/utils/ocr.ts | shared UI | document/rendering | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `pdfjs-dist` | `web:dependencies`, `shared-ui:peerDependencies` | 2 | apps/tldw-frontend/scripts/copy-pdf-worker.mjs, apps/packages/ui/src/components/DocumentWorkspace/DocumentViewer/PdfViewer/PdfDocument.tsx | web config/script, shared UI runtime worker reference | document/rendering | `keep` | Low; script resolves pdfjs-dist directly and shared PDF viewer uses react-pdf's pdfjs object to set pdfjs-dist worker URLs. | No immediate reduction; keep current PDF worker behavior. | none |
| `playwright` | `web:devDependencies`, `extension:devDependencies` | 46 | apps/tldw-frontend/e2e/interactive-review.ts, apps/tldw-frontend/scripts/ux-audit-cdp.ts | web tests, web config/script, package scripts | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `postcss` | `web:devDependencies`, `extension:devDependencies` | 2 | apps/tldw-frontend/postcss.config.mjs, apps/extension/postcss.config.js (PostCSS config files; plugin keys counted separately) | styling config/build | styling/build | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `postcss-import` | `web:devDependencies` | 1 | apps/tldw-frontend/postcss.config.mjs | styling config/build | styling/build | `keep` | Low; configured directly as a PostCSS plugin key. | No immediate reduction; keep current CSS processing behavior. | none |
| `prettier` | `web:devDependencies`, `extension:devDependencies` | 2 | apps/tldw-frontend/package.json | package scripts | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `prism-react-renderer` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 7 | apps/packages/ui/src/components/Common/CodeBlock.tsx, apps/packages/ui/src/components/Common/Markdown.tsx, apps/packages/ui/src/components/Option/Evaluations/components/JsonEditor.tsx, apps/packages/ui/src/components/Option/Settings/system-settings.tsx | shared UI | markdown/rendering | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `property-information` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 1 | apps/packages/ui/src/components/Common/Markdown.tsx | shared UI | markdown/rendering | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `pubsub-js` | none after PR #1357 | 0 | none found | removed WebUI and extension declarations | eventing | `removed` | Low; no import/config/package-script evidence in scanned roots before removal. | Two direct runtime declarations removed; no active manifest declaration remains. | PR #1357 complete |
| `puppeteer` | `web:devDependencies` | 2 | apps/extension/scripts/cdp-examine-extension-workflows.js, apps/tldw-frontend/scripts/cdp-examine-workflows.ts | extension tests/config, web config/script | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `react` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 2045 | apps/extension/tests/unit/queued-row-keydown.test.ts, apps/packages/ui/src/components/Agent/AgentErrorBoundary.tsx, apps/packages/ui/src/components/Agent/ApprovalBanner.tsx, apps/packages/ui/src/components/Agent/DiffViewer.tsx | extension tests/config, shared UI, shared UI tests, shared UI config, web tests, web app | frontend/runtime | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `react-dom` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 16 | apps/packages/ui/src/components/Common/CommandPalette.tsx, apps/packages/ui/src/components/Common/KeyboardShortcutsModal.tsx, apps/packages/ui/src/components/Common/NotesDock/NotesDockPanel.tsx, apps/packages/ui/src/components/Common/PageHelpModal.tsx | shared UI, shared UI tests, web tests, web app | frontend/runtime | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `react-hook-form` | none after PR #1365 | 0 | none found | removed WebUI declaration | frontend/runtime | `removed` | Low; no import/config/package-script evidence in scanned roots before removal. | One direct runtime declaration removed; no active manifest declaration remains. | PR #1365 complete |
| `react-i18next` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 1186 | apps/packages/ui/src/components/Agent/ApprovalBanner.tsx, apps/packages/ui/src/components/Agent/DiffViewer.tsx, apps/packages/ui/src/components/Agent/SessionHistoryPanel.tsx, apps/packages/ui/src/components/Agent/SessionRestoreDialog.tsx | shared UI, shared UI tests, shared UI config, web tests, web app | i18n | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `react-icons` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 2 | apps/packages/ui/src/components/Sidepanel/Chat/TemporaryChatBadge.tsx, apps/packages/ui/src/components/Sidepanel/Chat/form.tsx | shared UI | icons | `defer-design` | Medium; icon consolidation touches many visible components and needs visual review. | Potentially meaningful bundle reduction only after an icon-system design. | Icon-stack consolidation design. |
| `react-joyride` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 3 | apps/packages/ui/src/components/Common/TutorialRunner.tsx, apps/packages/ui/src/components/Common/__tests__/TutorialRunner.retry.test.tsx, apps/packages/ui/src/tutorials/registry.ts | shared UI, shared UI tests | frontend/runtime | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `react-markdown` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 6 | apps/packages/ui/src/components/Common/Markdown.tsx, apps/packages/ui/src/components/Flashcards/components/FlashcardMarkdownSnippet.tsx, apps/packages/ui/src/components/Knowledge/QASearchTab/GeneratedAnswerCard.tsx, apps/packages/ui/src/components/Option/ACPPlayground/ACPChatPanel.tsx | shared UI | markdown/rendering | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `react-pdf` | `web:dependencies`, `shared-ui:peerDependencies` | 5 | apps/packages/ui/src/components/DocumentWorkspace/DocumentViewer/PdfViewer/PdfDocument.tsx, apps/packages/ui/src/components/DocumentWorkspace/DocumentViewer/PdfViewer/PdfPage.tsx, apps/packages/ui/src/components/DocumentWorkspace/LeftSidebar/PagesTab.tsx, apps/packages/ui/src/hooks/document-workspace/usePdfSearch.ts | shared UI | document/rendering | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `react-router-dom` | `shared-ui:peerDependencies`, `extension:dependencies` | 383 | apps/packages/ui/src/components/Common/ChatSidebar.tsx, apps/packages/ui/src/components/Common/ChatSidebar/__tests__/ChatSidebar.lazy-history.test.tsx, apps/packages/ui/src/components/Common/CommandPalette.tsx, apps/packages/ui/src/components/Common/CommandPaletteHost.tsx | shared UI, shared UI tests, web tests, web app | frontend/runtime | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `react-syntax-highlighter` | none after TASK-134 | 0 | none found | removed WebUI declaration | markdown/rendering | `removed` | Medium; markdown code-block rendering is covered by active `rehype-highlight` and `prism-react-renderer` declarations, with no direct package evidence for this library. | One direct runtime declaration removed; `react-syntax-highlighter`, `lowlight`, `refractor`, and old highlight/prism records dropped from the lockfile. | TASK-134 complete |
| `react-toastify` | none after TASK-134 | 0 | none found | removed WebUI, shared UI, and extension declarations | frontend/runtime | `removed` | Low/medium; no source/config/script evidence across WebUI, shared UI, or extension surfaces. | Three direct declarations removed; both React Toastify lockfile records dropped. | TASK-134 complete |
| `rehype-highlight` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 1 | apps/packages/ui/src/components/Option/ACPPlayground/ACPChatPanel.tsx | shared UI | markdown/rendering | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `rehype-katex` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 1 | apps/packages/ui/src/components/Common/Markdown.tsx | shared UI | markdown/rendering | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `rehype-mathjax` | none after TASK-134 | 0 | none found | removed WebUI and extension declarations | markdown/rendering | `removed` | Medium; active markdown math rendering uses `remark-math` plus `rehype-katex`, and MathJax had no direct package evidence. | Two direct declarations removed; MathJax/jsdom-related lockfile records dropped. | TASK-134 complete |
| `remark-gfm` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 6 | apps/packages/ui/src/components/Common/Markdown.tsx, apps/packages/ui/src/components/Flashcards/components/FlashcardMarkdownSnippet.tsx, apps/packages/ui/src/components/Knowledge/QASearchTab/GeneratedAnswerCard.tsx, apps/packages/ui/src/components/Option/ACPPlayground/ACPChatPanel.tsx | shared UI | markdown/rendering | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `remark-math` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 1 | apps/packages/ui/src/components/Common/Markdown.tsx | shared UI | markdown/rendering | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `stream-browserify` | none after PR #1359 | 0 | none found | removed WebUI and extension declarations | polyfill/shim | `removed` | Low; no package import/config/package-script evidence in scanned roots before removal. | Two direct runtime declarations removed; no active manifest declaration remains. | PR #1359 complete |
| `tailwind-merge` | `web:dependencies` | 1 | apps/tldw-frontend/lib/utils.ts | web app | frontend/runtime | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `tailwindcss` | `web:dependencies`, `extension:devDependencies` | 2 | apps/extension/postcss.config.js, apps/tldw-frontend/postcss.config.mjs | styling config/build | styling/build | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `turndown` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 2 | apps/packages/ui/src/parser/amazon.ts, apps/packages/ui/src/parser/default.ts | shared UI | parser/conversion | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `typescript` | `web:devDependencies`, `extension:devDependencies` | 3 | apps/extension/scripts/verify-openapi-client-paths.mjs | extension tests/config, package scripts | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `unist-util-visit` | none after TASK-134 | 0 | none found | removed WebUI and extension declarations | markdown/rendering | `removed` | Medium; no direct package evidence, but markdown packages still own it transitively where needed. | Two direct declarations removed; package remains only as markdown transitive ownership. | TASK-134 complete |
| `vite` | `web:devDependencies`, `extension:devDependencies` | 0 | none found; likely framework/toolchain dependency for Vitest/WXT rather than runtime package evidence | tooling/dev declaration only | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `vitest` | `web:devDependencies`, `shared-ui:devDependencies` | 1498 | apps/extension/tests/e2e/setup/build-extension.test.ts, apps/extension/tests/e2e/utils/extension-build.test.ts, apps/extension/tests/e2e/utils/extension-paths.test.ts, apps/extension/tests/e2e/utils/extension.launch.test.ts | extension tests/config, shared UI tests, web tests, web config/script, web app, package scripts | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `wxt` | `shared-ui:peerDependencies`, `extension:devDependencies` | 58 | apps/extension/scripts/wxt-prepare.mjs, apps/extension/wxt.config.ts, apps/packages/ui/src/components/Common/CharacterSelect.tsx, apps/packages/ui/src/components/Common/ChatSidebar/ServerChatList.tsx | extension tests/config, shared UI, shared UI tests, web tests, web app, package scripts | tooling/dev | `keep` | Low; dev/test/build tool rather than runtime surface. | No runtime bundle impact; keep unless a toolchain cleanup proves it unused. | none |
| `xterm` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 7 | apps/packages/ui/src/ambient.d.ts, apps/packages/ui/src/components/Option/ACPPlayground/ACPWorkspacePanel.tsx, apps/tldw-frontend/__tests__/extension/entry-shell-performance.test.ts | shared UI, web tests | editor/terminal | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |
| `zod` | none after TASK-134 | 0 | none found | removed WebUI declaration | schema validation | `removed` | Medium; no WebUI runtime schema usage was found, while tooling packages still own their transitive Zod needs. | One direct runtime declaration removed; `zod` remains only through tooling transitives. | TASK-134 complete |
| `zustand` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | 83 | apps/packages/ui/src/components/Common/ChatSidebar/ServerChatList.tsx, apps/packages/ui/src/components/Common/Settings/ActorPopout.tsx, apps/packages/ui/src/components/Common/Settings/CurrentChatModelSettings.tsx, apps/packages/ui/src/components/Folders/FolderPicker.tsx | shared UI, shared UI tests, web tests, web app | state/data | `keep` | Low; import/config/package-script evidence in current WebUI or shared UI paths. | No immediate reduction; keep current behavior. | none |

## Ranked Follow-Up Queue

### Completed Follow-Ups

- TASK-134 removed unused direct declarations for `react-syntax-highlighter`,
  `@types/react-syntax-highlighter`, `react-toastify`, `rehype-mathjax`,
  `unist-util-visit`, and `zod` after confirming no current WebUI, shared UI,
  or extension source/config/script references. `unist-util-visit` and `zod`
  remain in `apps/bun.lock` only as transitive dependencies owned by markdown
  and tooling packages. Measured against `origin/dev`, the three scanned
  manifests went from 270 to 260 direct declaration entries, with all 10
  candidate declaration entries removed.
- TASK-134 retained `@dnd-kit/abstract`, `@dnd-kit/dom`, and `@tiptap/pm` in
  direct manifests. The DnD packages are still owned by the active
  `@dnd-kit/react`/helpers graph, and Tiptap packages declare `@tiptap/pm` as
  a peer/runtime dependency.
- TASK-141 removed unused direct tooling declarations for `@eslint/eslintrc`,
  `eslint-config-next`, `eslint-config-prettier`, and `fake-indexeddb`.
  Measured against `origin/dev`, the three scanned manifests went from 260 to
  256 direct declaration entries, with all 4 candidate declaration entries
  removed. `@eslint/eslintrc` remains only as an ESLint transitive dependency.
- TASK-144 reconciled the audit with already-merged issue #1346 cleanup PRs:
  PR #1357 removed `pubsub-js` and `@types/pubsub-js`; PR #1359 removed
  direct `buffer` and `stream-browserify` declarations; PR #1365 removed
  `@hookform/resolvers` and `react-hook-form`; PR #1368 replaced the direct
  `clsx` helper usage; PR #1375 replaced active `axios` imports with fetch
  helpers; PR #1385 and PR #1390 are reflected by TASK-134/TASK-141 rows.
  Current manifests retain only `dayjs` among the original quick/replacement
  package names checked in this refresh.
- TASK-147 removed `dayjs` duration usage from the display-only
  `humanizeMilliseconds` utility by replacing it with local millisecond
  arithmetic. The shared UI `dayjs` import count dropped from 21 to 19 while
  leaving the package declared for remaining relative-time and Ant Design
  `Dayjs` value-contract surfaces.
- TASK-149 removed `dayjs` relative-time usage from the display-only
  WorldBooks last-modified formatter by replacing it with a local helper that
  preserves representative relative labels and UTC absolute formatting. The
  shared UI `dayjs` import count dropped from 19 to 17 while leaving the
  package declared for remaining Flashcards/Models display formatting and Ant
  Design `Dayjs` value-contract surfaces.
- TASK-153 removed `dayjs` time formatting from the display-only Models
  last-refreshed label by replacing `dayjs(...).format("HH:mm")` with a small
  native Date helper. The shared UI `dayjs` import count
  dropped from 17 to 15 while leaving the package declared for remaining
  Flashcards display formatting and Ant Design `Dayjs` value-contract surfaces.
- TASK-158 removed `dayjs` scheduling metadata formatting from the display-only
  FlashcardEditDrawer due/last-reviewed labels by replacing absolute and
  relative labels with native Date helpers. The shared UI `dayjs` import count
  dropped from 15 to 11 while leaving the package declared for remaining
  Flashcards Review/Manage display formatting and Ant Design `Dayjs`
  value-contract surfaces.
- TASK-164 removed `dayjs` relative/absolute display formatting from
  Flashcards ReviewTab and ManageTab by extending the shared native Date
  helpers used by FlashcardEditDrawer. The shared UI `dayjs` import count
  dropped from 11 to 7 while leaving the package declared for Ant Design
  `Dayjs` value-contract surfaces.

### Quick Cleanup Candidates

No immediate low-risk quick-cleanup package remains from the issue #1346 queue
after PRs #1357, #1359, #1365, #1368, #1375, #1385, and #1390. Remaining
zero-evidence rows are either tooling/type declarations, active transitive
ownership checks, or complex-domain packages that should stay on the
`investigate-lockfile` path before any manifest edit.

### Replacement Candidates

1. `dayjs`: remaining shared UI imports are Ant Design date-control value
   surfaces. Do not attempt a direct dependency removal until those surfaces
   that pass or type `Dayjs` values are redesigned or isolated.

### Deferred Design Candidates

- Icon-stack consolidation: `lucide-react`, `@heroicons/react`, `@ant-design/icons`, and `react-icons` are active visible UI dependencies and should be handled with a visual/design pass.
- Date/time consolidation: current shared UI uses `Dayjs` values with Ant
  Design date controls in media, reading list, items, data table, and kanban
  surfaces. Treat this as a compatibility/design slice, not a quick manifest
  cleanup.
- PDF, ePub, document rendering, rich text editor, Mermaid, KaTeX, markdown, parser, graph/layout, OCR, tokenizer, schema, Monaco, Tiptap, and archive packages with active evidence are kept or deferred rather than replaced with hand-rolled browser code. Remaining zero-evidence complex declarations should keep using the `investigate-lockfile` path before any manifest edit.
- DnD package declarations with no direct import evidence, such as `@dnd-kit/abstract` and `@dnd-kit/dom`, are retained after TASK-134 because the current lockfile still routes active DnD packages through the DnD abstract/dom graph.

### Explicit Keeps

- Core app/runtime: `next`, `react`, `react-dom`, `antd`, `@ant-design/cssinjs`, `@tldw/ui`, `@tanstack/react-query`, `react-router-dom`, `zustand`, `dexie`, and `@plasmohq/storage` have active import/config evidence.
- Tooling/dev dependencies such as Playwright, Vitest, Testing Library, ESLint, TypeScript, PostCSS, Tailwind, and Prettier are classified as `tooling/dev` or `styling/build`; zero source-import counts are not treated as runtime removability by themselves.
- Active or security-sensitive `dompurify`, `ajv`, PDF/ePub/document packages, rich text editor packages, Mermaid/KaTeX/markdown rendering, graph/layout, OCR/tokenizer/schema, and parser/conversion packages are not quick-cleanup targets without a separate design or lockfile investigation.

## Verification

- 2026-05-07 declaration inventory: ran the Task 2 Node manifest reader from
  `Docs/superpowers/plans/2026-05-07-webui-dependency-audit-implementation-plan.md`
  against `apps/tldw-frontend/package.json`, `apps/packages/ui/package.json`,
  and `apps/extension/package.json`, reading `dependencies`, `devDependencies`,
  `peerDependencies`, and `optionalDependencies`; wrote temporary JSON to
  `/tmp/tldw-webui-dependency-declarations.json`. Observed 138 unique package
  declarations across all three manifests. The table includes 125 packages
  declared by `web` and/or `shared-ui`; 13 extension-only packages are excluded
  because the extension is only an impact-check surface for this audit slice.
- 2026-05-07 corrected usage scan: regenerated
  `/tmp/tldw-webui-dependency-usage.json` from `apps/tldw-frontend`,
  `apps/packages/ui`, and `apps/extension` using only `.ts`, `.tsx`,
  `.js`, `.jsx`, `.mjs`, `.cjs`, and `.css` files. The scan excluded
  `node_modules`, `.next`, `.output`, `build`, `dist`, `coverage`,
  `test-results`, `apps/bun.lock`, the three package manifests, and
  `apps/packages/ui/src/public/pdf.worker.min.mjs`. It scanned 4,605 files,
  then reviewed the three package manifests only for explicit package-script
  evidence.
- Correction note: the first Task 3 pass included a broad text/config-key signal
  that produced false positives for generic package names, such as local
  `next` object keys in non-config source files. This pass replaces that with
  import/config-aware package evidence only.
- Corrected evidence counted package specifiers from `import ... from`,
  side-effect imports, `export ... from`, dynamic `import("pkg")`,
  `require("pkg")`, `require.resolve("pkg")`, `vi.mock`/`jest.mock`,
  CSS `@import`, PostCSS config files and plugin keys, selected tool config literals, and
  explicit package-script commands. It does not count generic substring matches
  or local identifiers that happen to share a package name.
- Corrected quick-candidate inspection: `pubsub-js`, `buffer`, and
  `stream-browserify` had 0 package import/config/package-script matches.
  `clsx` had 1 match at `apps/tldw-frontend/lib/utils.ts`. `axios` had 3
  matches at `apps/tldw-frontend/lib/api.ts`,
  `apps/tldw-frontend/types/common.ts`, and
  `apps/packages/ui/src/services/elevenlabs.ts`.
- Corrected generic-name check: `next` now counts only WebUI Next imports,
  mocks, dynamic imports, and package scripts; no shared UI local-variable hits
  are counted as package evidence.
- Data-quality correction: `postcss-import` is counted from the direct
  PostCSS plugin key in `apps/tldw-frontend/postcss.config.mjs`. `pdfjs-dist`
  counts both direct worker-package resolution in
  `apps/tldw-frontend/scripts/copy-pdf-worker.mjs` and the shared PDF viewer's
  runtime worker/version reference in
  `apps/packages/ui/src/components/DocumentWorkspace/DocumentViewer/PdfViewer/PdfDocument.tsx`.
- 2026-05-07 Bandit: skipped for the initial audit slice because changes were
  documentation and Backlog task metadata only; no Python or runtime code was
  modified.
- 2026-05-08 TASK-134 lockfile follow-up: confirmed no exact source, config,
  script, or manifest references remained for `react-syntax-highlighter`,
  `@types/react-syntax-highlighter`, `react-toastify`, `rehype-mathjax`,
  `unist-util-visit`, or `zod` after direct manifest removal. Regenerated
  `apps/bun.lock` with `bun install`; the lockfile removed the direct
  `react-toastify`, `react-syntax-highlighter`, and `rehype-mathjax` trees.
  `unist-util-visit` and `zod` remain only through markdown/tooling transitives.
- 2026-05-08 TASK-134 impact deltas, measured against `origin/dev`: direct
  declaration entries across `apps/tldw-frontend/package.json`,
  `apps/extension/package.json`, and `apps/packages/ui/package.json` changed
  from 270 to 260 (-10). The removed candidate declaration entries changed
  from 10 to 0 (-10) across 6 unique package names. `apps/bun.lock` changed
  from 536,939 bytes to 518,386 bytes (-18,553), from 4,641 lines to 4,473
  lines (-168), and from 2,156 package records to 2,077 package records (-79).
- 2026-05-08 TASK-134 retained-package check: `@dnd-kit/abstract` and
  `@dnd-kit/dom` remain in the lockfile through active `@dnd-kit/collision`,
  `@dnd-kit/helpers`, and `@dnd-kit/react` dependencies. `@tiptap/pm` remains
  declared because current Tiptap packages list it as a peer/runtime dependency.
- 2026-05-08 TASK-134 verification: `bun install --frozen-lockfile` from
  `apps`, `bun run compile` from `apps/extension`,
  `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run compile` from
  `apps/tldw-frontend`, `bun run lint` from `apps/tldw-frontend`,
  `bunx vitest run --changed=origin/dev` from `apps/tldw-frontend`, and
  `git diff --check` all exited 0. The Vitest changed-file probe reported no
  matching test files for this manifest-only change.
- Bandit: skipped for TASK-134 because the slice changed TypeScript package
  manifests, `apps/bun.lock`, documentation, and Backlog metadata only; no
  Python files were modified.
- 2026-05-08 TASK-141 tooling follow-up: confirmed no exact source, config,
  script, or manifest references remained for `@eslint/eslintrc`,
  `eslint-config-next`, `eslint-config-prettier`, or `fake-indexeddb` after
  direct manifest removal. Regenerated `apps/bun.lock` with `bun install`;
  `eslint-config-next`, `eslint-config-prettier`, and `fake-indexeddb` dropped
  out completely, while `@eslint/eslintrc` remains through ESLint's transitive
  dependency graph.
- 2026-05-08 TASK-141 impact deltas, measured against `origin/dev`: direct
  declaration entries across `apps/tldw-frontend/package.json`,
  `apps/extension/package.json`, and `apps/packages/ui/package.json` changed
  from 260 to 256 (-4). The removed tooling candidate declaration entries
  changed from 4 to 0 (-4) across 4 unique package names. `apps/bun.lock`
  changed from 518,386 bytes to 501,563 bytes (-16,823), from 4,473 lines to
  4,347 lines (-126), and from 2,077 package records to 2,016 package records
  (-61).
- 2026-05-08 TASK-144 audit refresh: confirmed PRs #1357, #1359, #1365,
  #1368, #1375, #1385, and #1390 were merged into `origin/dev` before this
  doc-only refresh. Current manifest check across
  `apps/tldw-frontend/package.json`, `apps/packages/ui/package.json`, and
  `apps/extension/package.json` found none of `pubsub-js`, `@types/pubsub-js`,
  `buffer`, `stream-browserify`, `@hookform/resolvers`, `react-hook-form`,
  `axios`, or `clsx` as direct declarations. The only original quick/replacement
  name still directly declared is `dayjs`, with active shared UI imports and
  Ant Design `Dayjs` value contracts.
- 2026-05-08 TASK-144 active-code scan: exact package-import scan for
  `pubsub-js`, `buffer`, `stream-browserify`, `@hookform/resolvers`,
  `react-hook-form`, `axios`, `clsx`, and `dayjs` found only `dayjs` imports in
  shared UI source/tests. `axios`, `buffer`, and `clsx` remain in `apps/bun.lock`
  only through transitive or optional-peer ownership, not direct manifest
  declarations or active package imports.
- 2026-05-09 TASK-147 active-code scan: exact shared UI package-import scan
  found 19 remaining `dayjs` import lines after removing both imports from
  `apps/packages/ui/src/utils/humanize-milliseconds.ts`. Remaining imports are
  concentrated in Flashcards relative-time displays, WorldBooks relative-time
  displays, and Ant Design `Dayjs` value/type surfaces in media, reading list,
  items, data table, and kanban code.
- 2026-05-09 TASK-147 verification: `bunx vitest run
  src/utils/__tests__/humanize-milliseconds.test.ts` from `apps/packages/ui`
  passed after first failing on the dependency guard before implementation.
- 2026-05-09 TASK-149 active-code scan: exact shared UI package-import scan
  found 17 remaining `dayjs` import lines after removing both imports from
  `apps/packages/ui/src/components/Option/WorldBooks/worldBookListUtils.ts`.
  Remaining imports are concentrated in Flashcards relative-time displays,
  Models last-refresh formatting, and Ant Design `Dayjs` value/type surfaces
  in media, reading list, items, data table, and kanban code.
- 2026-05-09 TASK-149 verification: `bunx vitest run
  src/components/Option/WorldBooks/__tests__/worldBookListUtils.test.ts` from
  `apps/packages/ui` passed after first failing on the dependency guard before
  implementation.
- 2026-05-09 TASK-153 active-code scan: exact shared UI package-import scan
  found 15 remaining `dayjs` import lines after removing both imports from
  `apps/packages/ui/src/components/Option/Models/index.tsx`. Remaining imports
  are concentrated in Flashcards relative-time/date displays and Ant Design
  `Dayjs` value/type surfaces in media, reading list, items, data table, and
  kanban code.
- 2026-05-09 TASK-153 verification: `bunx vitest run
  src/components/Option/Models/__tests__/modelsDisplayUtils.test.ts` from
  `apps/packages/ui` passed after first failing on the missing native helper.
  The PR review follow-up removed the filesystem-based source guard from the
  Vitest unit test; dependency regression coverage for this slice is recorded
  through the exact Models-tree package-import scan instead.
- 2026-05-09 TASK-158 active-code scan: exact shared UI package-import scan
  found 11 remaining `dayjs` import lines after removing both runtime imports
  from `apps/packages/ui/src/components/Flashcards/components/FlashcardEditDrawer.tsx`
  and both test imports from the scheduling metadata test. Remaining
  Flashcards imports are in ReviewTab and ManageTab display formatting; the
  other remaining imports are Ant Design `Dayjs` value/type surfaces in media,
  reading list, items, data table, and kanban code.
- 2026-05-09 TASK-158 verification: `bunx vitest run
  src/components/Flashcards/utils/__tests__/date-display.test.ts
  src/components/Flashcards/components/__tests__/FlashcardEditDrawer.scheduling-metadata.test.tsx`
  from `apps/packages/ui` passed after first failing on the missing native
  date-display helper.
- 2026-05-09 TASK-164 active-code scan: exact shared UI package-import scan
  found 7 remaining `dayjs` import lines after removing both runtime imports
  from `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx` and
  `apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx`. Remaining
  imports are Ant Design `Dayjs` value/type surfaces in media, reading list,
  items, data table, and kanban code.
- 2026-05-09 TASK-164 verification: `bunx vitest run
  src/components/Flashcards/utils/__tests__/date-display.test.ts
  src/components/Flashcards/tabs/__tests__/ManageTab.scheduling-metadata.test.tsx
  src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx` from
  `apps/packages/ui` passed after first failing on the missing native
  date-display helpers. `bun run lint` from `apps/tldw-frontend` passed with
  the existing warning baseline, and `git diff --check` passed. The shared UI
  TypeScript check still fails on existing repo-wide test/service baseline
  errors outside this slice.
- Bandit: skipped for TASK-144 because the slice changed documentation and
  Backlog metadata only; no Python files were modified.
- Bandit: skipped for TASK-147 because the slice changed TypeScript,
  documentation, and Backlog metadata only; no Python files were modified.
- Bandit: skipped for TASK-149 because the slice changed TypeScript,
  documentation, and Backlog metadata only; no Python files were modified.
- Bandit: skipped for TASK-153 because the slice changed TypeScript,
  documentation, and Backlog metadata only; no Python files were modified.
- Bandit: skipped for TASK-158 because the slice changed TypeScript,
  documentation, and Backlog metadata only; no Python files were modified.
- Bandit: skipped for TASK-164 because the slice changed TypeScript,
  documentation, and Backlog metadata only; no Python files were modified.

## Known Skips And Blockers

- The usage JSON is a source/config/package-script signal, not a bundler trace. It does not prove transitive dependency ownership or tree-shaken bundle impact.
- The extension is included as an impact-check surface because it consumes `@tldw/ui`; extension-only packages remain outside the primary table unless they overlap WebUI/shared UI declarations.
- Decisions marked `investigate-lockfile` should not be removed until a follow-up confirms direct-vs-transitive ownership in `apps/bun.lock` and validates install/build behavior.
- The audit now treats `dayjs` as the next compatibility/design target rather
  than a quick removal. Native `Intl` helpers can reduce simple formatting
  usage, but direct dependency removal is blocked while shared UI date-picker
  flows pass or type `Dayjs` values.
