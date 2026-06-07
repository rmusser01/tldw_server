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
Implemented a stable `/__debug__/mermaid-chat-cards` WebUI harness with assistant, user, disabled, invalid Mermaid, Graphviz, and artifact-style fixtures. Added Playwright smoke coverage that seeds offline auth, asserts readiness/first-run blockers are absent, verifies assistant/artifact Mermaid controls render, and verifies user/disabled/non-Mermaid paths remain source-only. Added `/__debug__` gate bypassing in `_app.tsx` and route metadata for `/__debug__/mermaid-chat-cards` as `internal_qa_debug` with `smoke: "exclude"`, without adding it to page inventory. The harness exposed a shared Markdown replay bug under React dev/Strict rendering, so `Markdown` now matches Mermaid fences without render-time cursor mutation while preserving global code-block indices, with a StrictMode regression test.

Verification: RED observed first with `npx playwright test e2e/smoke/mermaid-chat-cards.spec.ts --reporter=line` failing on missing harness. Green checks: focused Mermaid unit suite `bunx vitest run src/components/Common/__tests__/Markdown.mermaid.test.tsx src/components/Common/__tests__/MermaidDiagramBlock.test.tsx src/components/Common/__tests__/MermaidPreviewDialog.test.tsx src/components/Common/Playground/__tests__/Message.mermaid-rendering.test.tsx src/components/Common/QuickChatHelper/__tests__/QuickChatMessage.mermaid.test.tsx src/components/Sidepanel/Chat/__tests__/ArtifactsPanel.mermaid.test.tsx` passed 62 tests; `npx playwright test e2e/smoke/mermaid-chat-cards.spec.ts --reporter=line` passed; `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run compile` passed with token sync OK; Browser-plugin manual QA confirmed harness visible, gate blockers absent, assistant/artifact actions present, and fallback sections source-only as expected. Route governance check `bunx vitest run src/routes/__tests__/route-governance.metadata-coverage.test.ts` still fails on existing metadata/inventory baseline gaps unrelated to this new debug route; the smoke-excluded invariant passed. Bandit was run from the main project venv against touched TS/TSX files and reported zero findings; it also reported syntax-parse errors because Bandit cannot parse TypeScript/TSX.
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
