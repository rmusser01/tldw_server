---
id: TASK-15622
title: Stop chat markdown remounting on every re-render
status: Done
labels:
- webui
- performance
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Markdown.tsx passed react-markdown an inline components object, so every render produced new tag component functions and React unmounted and remounted the whole rendered message (paragraphs, lists, code blocks, KaTeX, Mermaid). Markdown was not memoized and react-markdown 10 re-parses on every render. On an idle /chat every health or research poll re-rendered each chat row, adding ~680 DOM nodes per 30 s while connected and ~2570 per 30 s while the backend was unreachable. Mermaid 11 also leaves its #d<id> error element in <body> on each failed render. Fix: module-level tag components that read per-instance values from React context, Markdown wrapped in React.memo, and mermaid suppressErrorRendering so it removes its temp element on failure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Re-rendering Markdown with identical props keeps the rendered <p> DOM node
- [x] #2 Appending text to the last paragraph keeps the first <p>, list and code block DOM nodes
- [x] #3 Appending text after a closed mermaid fence keeps the Mermaid container DOM node
- [x] #4 A failing Mermaid render leaves no #d<id> element under document.body and still shows the source fallback
- [x] #5 Existing Markdown and Mermaid tests pass
- [x] #6 Code blocks re-colour when the page theme changes, with or without a parent re-render, including an OS colour-scheme change with no theme class
- [x] #7 Pro-mode table auto-open re-opens with the rows streamed after the header
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: not required. This is a render-performance bug fix inside two existing components (Markdown.tsx, Mermaid.tsx) with no new or changed durable architecture rule; Docs/ADR/README.md has no ADR governing markdown rendering or component memoization.
Root cause verified: Markdown.tsx passed react-markdown an inline components object; hast-util-to-jsx-runtime 2.3.6 uses each component as the element type, so new functions per render remounted the whole message. react-markdown 10.1.0 Markdown() calls createProcessor + parse on every render (no memoization); the old comment claiming otherwise was corrected. Fix: tag components moved to a module-level MARKDOWN_COMPONENTS map reading per-instance values (closedMermaidFenceSources, searchQuery, codeBlockVariant, codeTheme, headingAnchorIds, allowExternalImages, mermaid/artifact flags, index refs) from MarkdownRenderContext; Markdown exported as React.memo(MarkdownContent). Call sites checked: all pass strings/booleans except ContentRenderer headingAnchorIds (fresh useMemo array, never mutated). Mermaid: suppressErrorRendering: true makes mermaid 11.12.2 call removeTempElements() before throwing on both parse and draw errors (dist/mermaid.core.mjs ~1029-1055); the MermaidDiagramBlock <pre> source fallback still shows.
Tests: new Markdown.stable-components.test.tsx (identical-props node identity, streaming first p/ul/code block identity, mermaid container identity with renderMock called once, heading anchors + search highlight regression) and Mermaid.error-cleanup.test.tsx (real mermaid; no dmermaid-* element left in body, source fallback shown). Red before fix, green after. Neighbours: 16 Markdown/Mermaid/TableBlock/wikilinks files 88 passed; 17 Playground/QuickChat/Documentation/Review/StudioPane files 469 passed; 12 MediaReviewPage files 217 passed. eslint (lint:ui config) clean.
Docs: none needed (internal render behaviour, no user-facing or API change). Bandit: not applicable, only TypeScript/TSX files touched. Known skips: no full vitest suite or full tsc (per run rules); a narrowed tsc over Markdown.tsx and the new tests reported no errors in those files. No live browser re-measurement of DOM node growth was run.
PR: https://github.com/rmusser01/tldw_server/pull/3573
Review round 1 (PR #3573, verdict with fixes). (1) Code block colours froze after a theme switch: resolveTheme('auto') reads <html> class and matchMedia at render, and the memo stopped parent re-renders from reaching Markdown. Fix: Markdown subscribes with useSyncExternalStore (MutationObserver on <html> class + prefers-color-scheme change listener; snapshot is resolveTheme('auto'), one of two module constants) and passes autoCodeTheme through the render context; github/compact Highlight use it, CodeBlock re-renders with MarkdownPre and resolves its own. Chose this over useDarkModeStore because the store defaults to 'dark' and only tracks the page when a useDarkMode() caller is mounted, and tldw-frontend _document sets the class without it. (2) TableBlock pro-mode auto-open effect lacked children in deps, so with stable mounting it opened once with rows [] and never again; added children (the id-keyed autoOpenStateMap guard keeps base behaviour: 3 opens for header, +1 row, +2 rows). (3) MarkdownRenderContext now defaults to null and tag components throw a clear error outside <Markdown>. (4) New tests: theme switch with and without parent re-render for default/compact/github, OS scheme change, table rows, heading anchors and block indexes across a streamed re-render, changed searchQuery/codeBlockVariant/allowExternalImages reaching the stable components. The test file's react-i18next mock now returns a stable t (a new t per render had re-run the TableBlock effect and hidden the bug).
Review round 1 greps. Render-time reads in the Markdown subtree: only resolveTheme('auto') (Markdown compact/github, CodeBlock) was frozen by the memo; fixed. Others are safe: CodeBlock/TableBlock window.__*State maps are read into refs once and synced by effects; Mermaid's theme already uses a MutationObserver + matchMedia effect; artifacts store matchMedia runs inside openArtifact, not render; MermaidDiagramBlock/MermaidPreviewDialog/CodeCopyButton/ManagedMarkdownImage read DOM/navigator only in handlers or effects. Effects that relied on remount: only TableBlock auto-open (fixed). TableBlock artifactOriginId has [children], tableLabel runs every render, useHorizontalOverflow watches content with Mutation/ResizeObserver; CodeBlock preview/key/auto-open effects list value; Mermaid [code, theme]; MermaidDiagramBlock resets on source change during render; ManagedMarkdownImage [assetUuid]; MermaidPreviewDialog [generatedSvg, open, source]. Verification: 21 Common files (Markdown*/Mermaid*/TableBlock*/CodeBlock*/ArtifactOriginAnchors/render-purity) + Notes wikilinks 118 passed; 17 Playground/QuickChat/Documentation/Review/StudioPane files 469 passed; Playground.search.integration + ArtifactsPanel.footer + 12 MediaReviewPage files 254 passed. eslint: no new warnings (TableBlock's 3 warnings pre-exist on dev).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Markdown no longer remounts its rendered message on re-render: tag components are module-level and read per-instance values from React context, and the Markdown export is React.memo so unchanged messages skip react-markdown's per-render re-parse. Because memo stops incidental re-renders, Markdown now subscribes to the page theme so code blocks re-colour on a theme switch, and TableBlock's pro-mode auto-open re-runs on content changes. Mermaid initializes with suppressErrorRendering so failed renders no longer leave #d<id> nodes in <body>; the source fallback still shows. New node-identity, theme, table, anchor/index and real-mermaid cleanup tests; all neighbouring consumer tests pass.
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
