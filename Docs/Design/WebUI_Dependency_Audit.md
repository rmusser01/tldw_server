# WebUI Dependency Audit

Date: 2026-05-07
Status: Draft audit for issue #1346

## References

- GitHub issue: https://github.com/rmusser01/tldw_server/issues/1346
- Design spec: ../superpowers/specs/2026-05-07-webui-dependency-trimming-design.md
- Parent design task: TASK-100
- Backlog task: TASK-101

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

## Dependency Inventory

| Package | Declared locations | Import count | Representative sites | Consumer surface | Category | Decision | Risk | Expected impact | Follow-up slice |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `@ant-design/cssinjs` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@ant-design/icons` | `web:dependencies`, `shared-ui:peerDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@axe-core/playwright` | `web:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@dnd-kit/abstract` | `web:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@dnd-kit/collision` | `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@dnd-kit/dom` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@dnd-kit/helpers` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@dnd-kit/react` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@eslint/eslintrc` | `web:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@eslint/js` | `web:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@heroicons/react` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@hookform/resolvers` | `web:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@monaco-editor/react` | `web:dependencies`, `shared-ui:peerDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@mozilla/readability` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@next/eslint-plugin-next` | `web:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@plasmohq/storage` | `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@playwright/test` | `web:devDependencies`, `extension:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@radix-ui/react-dialog` | `web:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@sentry/nextjs` | `web:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@tailwindcss/forms` | `web:devDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@tailwindcss/typography` | `web:devDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@tanstack/react-query` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@tanstack/react-virtual` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@testing-library/jest-dom` | `web:devDependencies`, `shared-ui:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@testing-library/react` | `web:devDependencies`, `shared-ui:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@testing-library/user-event` | `web:devDependencies`, `shared-ui:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@tiptap/core` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@tiptap/extension-character-count` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@tiptap/extension-placeholder` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@tiptap/pm` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@tiptap/react` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@tiptap/starter-kit` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@tldw/ui` | `web:dependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@types/cytoscape` | `web:devDependencies`, `extension:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@types/d3-dsv` | `web:devDependencies`, `extension:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@types/html-to-text` | `web:devDependencies`, `extension:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@types/katex` | `web:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@types/node` | `web:devDependencies`, `extension:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@types/pubsub-js` | `web:devDependencies`, `extension:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@types/react` | `web:devDependencies`, `shared-ui:devDependencies`, `extension:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@types/react-dom` | `web:devDependencies`, `shared-ui:devDependencies`, `extension:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@types/react-syntax-highlighter` | `web:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@types/turndown` | `web:devDependencies`, `extension:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@typescript-eslint/eslint-plugin` | `web:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@typescript-eslint/parser` | `web:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@vitejs/plugin-react` | `web:devDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@vitest/coverage-v8` | `web:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@xterm/addon-fit` | `web:dependencies`, `shared-ui:peerDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `@xyflow/react` | `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `ajv` | `web:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `antd` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `autoprefixer` | `web:devDependencies`, `extension:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `axe-core` | `shared-ui:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `axios` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `buffer` | `web:dependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `cheerio` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `clsx` | `web:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `cross-env` | `web:devDependencies`, `extension:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `cytoscape` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `cytoscape-dagre` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `d3-dsv` | `web:dependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `dayjs` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `dexie` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `dexie-react-hooks` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `dompurify` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `epubjs` | `web:dependencies`, `shared-ui:peerDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `eslint` | `web:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `eslint-config-next` | `web:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `eslint-config-prettier` | `web:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `eslint-plugin-react` | `web:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `eslint-plugin-react-hooks` | `web:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `exceljs` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `fake-indexeddb` | `web:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `globals` | `web:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `gpt-tokenizer` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `html-to-text` | `web:dependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `html2canvas` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `i18next` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `i18next-browser-languagedetector` | `web:dependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `i18next-icu` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `jsdom` | `web:devDependencies`, `shared-ui:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `jszip` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `katex` | `web:dependencies`, `shared-ui:peerDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `lucide-react` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `marked` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `mermaid` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `next` | `web:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `pa-tesseract.js` | `web:dependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `pdfjs-dist` | `web:dependencies`, `shared-ui:peerDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `playwright` | `web:devDependencies`, `extension:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `postcss` | `web:devDependencies`, `extension:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `postcss-import` | `web:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `prettier` | `web:devDependencies`, `extension:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `prism-react-renderer` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `property-information` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `pubsub-js` | `web:dependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `puppeteer` | `web:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `react` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `react-dom` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `react-hook-form` | `web:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `react-i18next` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `react-icons` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `react-joyride` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `react-markdown` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `react-pdf` | `web:dependencies`, `shared-ui:peerDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `react-router-dom` | `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `react-syntax-highlighter` | `web:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `react-toastify` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `rehype-highlight` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `rehype-katex` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `rehype-mathjax` | `web:dependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `remark-gfm` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `remark-math` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `stream-browserify` | `web:dependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `tailwind-merge` | `web:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `tailwindcss` | `web:dependencies`, `extension:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `turndown` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `typescript` | `web:devDependencies`, `extension:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `unist-util-visit` | `web:dependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `vite` | `web:devDependencies`, `extension:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `vitest` | `web:devDependencies`, `shared-ui:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `wxt` | `shared-ui:peerDependencies`, `extension:devDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `xterm` | `web:dependencies`, `shared-ui:peerDependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `zod` | `web:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `zustand` | `web:dependencies`, `shared-ui:peerDependencies`, `extension:dependencies` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Ranked Follow-Up Queue

### Quick Cleanup Candidates

### Replacement Candidates

### Deferred Design Candidates

### Explicit Keeps

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

## Known Skips And Blockers
