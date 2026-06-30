---
id: TASK-2311
title: Close out Mermaid chat card build and browser QA
status: Done
references:
- docs/superpowers/specs/2026-06-04-chat-mermaid-diagrams-design.md
- docs/superpowers/plans/2026-06-04-chat-mermaid-diagrams-implementation-plan.md
- https://github.com/rmusser01/tldw_server/pull/2298
modified_files:
- apps/packages/ui/src/components/Common/Mermaid.tsx
- apps/packages/ui/src/components/Common/MermaidDiagramBlock.tsx
- apps/packages/ui/src/components/Common/MermaidPreviewDialog.tsx
- apps/packages/ui/src/components/Common/Markdown.tsx
- apps/packages/ui/src/components/Common/__tests__/MermaidPreviewDialog.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify the merged Mermaid chat/card experience end to end against current origin/dev. Scope covers dependency/build resolution, focused Mermaid tests, and browser QA for inline render, preview controls, disabled setting fallback, user-message unchanged behavior, invalid Mermaid fallback, and Graphviz/DOT code fallback. Patch only issues found during QA.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Focused Mermaid and Markdown tests pass for inline assistant rendering, preview controls, disabled fallback, invalid syntax fallback, Graphviz/DOT code fallback, and user-message unchanged behavior.
- [x] Frontend compile succeeds with the Mermaid implementation present.
- [x] Static review finds no source issue requiring a follow-up patch.
- [x] Browser QA attempt is recorded with the environment blocker and no workaround was used for the Browser file URL policy.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
No app source patch was needed. Static review confirmed Mermaid rendering remains assistant-facing only, requires closed `mermaid` fences, is disabled while streaming, respects the chat setting fallback, sanitizes rendered SVG output, falls back to source on render errors, and leaves non-Mermaid diagram languages such as Graphviz/DOT as code blocks.

Verification performed:

- `bunx vitest run src/components/Common/__tests__/Mermaid.test.tsx src/components/Common/__tests__/MermaidDiagramBlock.test.tsx src/components/Common/__tests__/MermaidPreviewDialog.test.tsx src/components/Common/__tests__/Markdown.mermaid.test.tsx src/components/Common/__tests__/Markdown.github-code-blocks.test.tsx src/components/Common/__tests__/CodeBlock.artifacts.test.tsx src/components/Common/Playground/__tests__/Message.mermaid-rendering.test.tsx src/components/Common/QuickChatHelper/__tests__/QuickChatMessage.mermaid.test.tsx src/components/Sidepanel/Chat/__tests__/ArtifactsPanel.mermaid.test.tsx src/components/Option/Settings/__tests__/ChatSettings.test.tsx` passed with 10 files and 79 tests.
- `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run compile` passed in `apps/tldw-frontend`.
- `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run build` was attempted and blocked by sandbox process/port restrictions in Turbopack while writing `/api/hello`; Webpack compile is the usable local build signal.
- Browser QA reached the Next debug route, but full render verification was blocked because the Browser page context exposed `localStorage` and `sessionStorage` as missing, the route remained behind readiness/first-run gates, and Browser policy blocked the temporary `file://` harness.
- `bunx tsc --noEmit` in `apps/packages/ui` was attempted and failed only in unrelated KnowledgeQA fixture type drift around `sourceStatus` and `sourceHealth`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed Mermaid chat/card QA closeout. Focused Mermaid/UI tests passed: `bunx vitest run src/components/Common/__tests__/Mermaid.test.tsx src/components/Common/__tests__/MermaidDiagramBlock.test.tsx src/components/Common/__tests__/MermaidPreviewDialog.test.tsx src/components/Common/__tests__/Markdown.mermaid.test.tsx src/components/Common/__tests__/Markdown.github-code-blocks.test.tsx src/components/Common/__tests__/CodeBlock.artifacts.test.tsx src/components/Common/Playground/__tests__/Message.mermaid-rendering.test.tsx src/components/Common/QuickChatHelper/__tests__/QuickChatMessage.mermaid.test.tsx src/components/Sidepanel/Chat/__tests__/ArtifactsPanel.mermaid.test.tsx src/components/Option/Settings/__tests__/ChatSettings.test.tsx` with 10 files / 79 tests passing. `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run compile` passed for the frontend. Turbopack `bun run build` reached sandbox process/port restrictions while writing `/api/hello`, so Webpack compile is the reliable local build signal. Browser QA reached the debug route, but full in-app render verification was blocked by the Browser environment: page JavaScript reported missing `localStorage`/`sessionStorage`, chat debug route remained behind readiness/first-run gates, and Browser policy blocked the temporary file harness. Bandit was skipped because this closeout changed no Python source code. Known unrelated `tsc --noEmit` failures remain in KnowledgeQA fixtures (`sourceStatus`/`sourceHealth` shape drift), outside this task.
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
