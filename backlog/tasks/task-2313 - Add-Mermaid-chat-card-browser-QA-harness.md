---
id: TASK-2313
title: Add Mermaid chat-card browser QA harness
status: Done
labels:
- frontend
- chat
- mermaid
- qa
references:
- Docs/superpowers/specs/2026-06-04-chat-mermaid-diagrams-design.md
- backlog/tasks/task-2311 - Close-out-Mermaid-chat-card-build-and-browser-QA.md
modified_files:
- Docs/superpowers/specs/2026-06-07-mermaid-chat-card-browser-qa-harness-design.md
- Docs/superpowers/plans/2026-06-07-mermaid-chat-card-browser-qa-harness-implementation-plan.md
- apps/packages/ui/src/components/Common/Markdown.tsx
- apps/packages/ui/src/components/Common/__tests__/Markdown.mermaid.test.tsx
- apps/packages/ui/src/routes/route-metadata.ts
- apps/tldw-frontend/e2e/smoke/mermaid-chat-cards.spec.ts
- apps/tldw-frontend/pages/__debug__/mermaid-chat-cards.tsx
- apps/tldw-frontend/pages/_app.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a stable browser QA harness for Mermaid chat/card rendering so assistant-facing Markdown diagrams can be verified without real chat backend state, readiness gates, first-run onboarding, or temporary file harnesses.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Spec review follow-up applied before implementation planning:

- Route governance now classifies the debug route in `route-metadata.ts` only and keeps it out of `page-inventory.ts`.
- Playwright coverage must assert `server-readiness-recovery` and `first-run-gate-overlay` are absent.
- Invalid Mermaid assertions should use the existing `Unable to render Mermaid diagram.` fallback text plus raw source.
- Harness fixture sections should provide stable wrapper `data-testid` selectors instead of requiring new component-level test ids in `MermaidDiagramBlock`.
- Implementation plan created at `Docs/superpowers/plans/2026-06-07-mermaid-chat-card-browser-qa-harness-implementation-plan.md`. The plan preserves TDD red/green verification while avoiding a failing red-test-only commit, in line with the repo working-commit policy.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Mermaid chat-card browser QA harness and completed PR follow-up for rmusser01/tldw_server#2306. The branch is rebased onto current `origin/dev`.

Initial implementation: added `/__debug__/mermaid-chat-cards` with assistant, user, disabled, invalid Mermaid, Graphviz, and artifact-style fixtures; added Playwright smoke coverage for offline auth seeding, readiness/setup gate absence, assistant/artifact Mermaid controls, and source-only fallback paths; added `/__debug__` gate bypassing in `_app.tsx`; registered `/__debug__/mermaid-chat-cards` as `internal_qa_debug` with `smoke: "exclude"`; fixed shared Markdown Mermaid fence matching so React dev/Strict render replay does not fall back to `CodeBlock` while preserving global code-block indices.

PR review follow-up: Gemini flagged that `INDENTED_CODE_LINE = /^(?: {4}|\t)/` counted standalone whitespace-only lines as indented code blocks. Verified the concern with a failing regression test: standalone whitespace before a Mermaid fence shifted `blockIndex` from 0 to 1. Fixed the regex to require non-whitespace content when starting an indented code block, while leaving blank/whitespace lines inside active indented blocks handled by `findIndentedCodeBlockEnd`.

Qodo flagged potential Mermaid `blockIndex` drift and suggested passing the render-callback `blockIndex` directly. Evaluated and did not apply that part because the render-callback index is mutable during React dev/Strict render replay; using it would make Mermaid artifact IDs depend on callback replay order. The scanner-derived matched fence index remains the stable artifact index, and the new Gemini regression specifically covers the drift case Qodo described by preventing whitespace-only lines from diverging scanner counts from ReactMarkdown code-block counts.

Verification: `bunx vitest run src/components/Common/__tests__/Markdown.mermaid.test.tsx --testNamePattern "standalone whitespace-only"` failed before the regex fix and passed after it. Focused suite passed: `bunx vitest run src/components/Common/__tests__/Markdown.mermaid.test.tsx src/components/Common/__tests__/MermaidDiagramBlock.test.tsx src/components/Common/__tests__/MermaidPreviewDialog.test.tsx src/components/Common/Playground/__tests__/Message.mermaid-rendering.test.tsx src/components/Common/QuickChatHelper/__tests__/QuickChatMessage.mermaid.test.tsx src/components/Sidepanel/Chat/__tests__/ArtifactsPanel.mermaid.test.tsx` with 63 tests passing. `npx playwright test e2e/smoke/mermaid-chat-cards.spec.ts --reporter=line` passed. `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run compile` passed with token sync OK. Bandit follow-up via the main project venv reported zero findings and expected TypeScript/TSX parser errors. CodeRabbit pre-merge checks still include non-blocking warnings for docstring coverage and the human-written `Change summary` placeholder; the latter must be filled by the human requester before merge per repo policy.
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
