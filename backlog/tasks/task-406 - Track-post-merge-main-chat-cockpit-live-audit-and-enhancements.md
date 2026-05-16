---
id: TASK-406
title: Track post-merge main chat cockpit live audit and enhancements
status: In Progress
labels:
- chat
- webui
- ux
- audit
priority: Medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fresh post-merge tracker for the main WebUI /chat cockpit only. Use live browser/server evidence from origin/dev-derived work to identify enhancement follow-ups after the collapsible sidechannel slice, without drifting into extension sidepanel/sidebar or unrelated pages.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] Inspect the post-merge main `/chat` cockpit from this branch or a fresh `origin/dev` branch with real-server/browser evidence.
- [ ] Identify enhancement follow-ups for main `/chat` only, covering first-use comprehension, power-user flow, IA, rail controls, composition flow, accessibility, and responsive behavior.
- [ ] Separate quick wins from larger cockpit redesign or interaction-model opportunities.
- [ ] Record screenshots or browser notes where they materially support the findings.
- [ ] Do not include extension sidepanel/sidebar or unrelated WebUI pages.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started after the collapsible sidechannel slice. The sidechannel implementation is tracked separately in TASK-405.

Initial real-server evidence pass:

- Branch UI: `codex/chat-cockpit-sidechannels`, served from the worktree on `http://127.0.0.1:18002/chat`.
- Backend: real local tldw server on `http://127.0.0.1:8000` with single-user API key configuration. No mocked API routes were used.
- Evidence screenshots from the focused real-server proof:
  - `apps/tldw-frontend/test-results/workflows-chat-cockpit.rea-eebf0-kpit-focus-controls-working-chromium/chat-cockpit-desktop-initial.png`
  - `apps/tldw-frontend/test-results/workflows-chat-cockpit.rea-eebf0-kpit-focus-controls-working-chromium/chat-cockpit-desktop-conversation.png`

Scoped findings for main `/chat` only:

| Priority | Area | Evidence | Follow-up |
| --- | --- | --- | --- |
| P1 | Empty assistant response recovery | The real-server conversation proof can show an assistant turn card with provider/model metadata but no visible response text while the page returns to `Ready`. | Add an explicit empty-response state in the message timeline and runtime panel, with a recoverable explanation and regenerate/retry path. |
| P1 | Collapsed-rail context continuity | Collapsible sidechannels free the chat canvas, but a fully collapsed cockpit leaves the composer/status strip as the only persistent summary. | Add a compact, always-visible active composition summary near the composer when one or both rails are collapsed: model, assistant/persona, prompt, context count, MCP/tool state. |
| P2 | First-paint density | Initial cockpit view shows left rail, center mode chooser, right rail, global nav shortcuts, composer controls, and bottom status all at once. | Tighten first-use hierarchy by reducing duplicate labels and making the center mode chooser yield visual priority to the composer once the user starts typing or has an active conversation. |
| P2 | Duplicate prompt/composition language | The left rail repeats `No prompt selected` in both the composition summary and the dedicated prompt card. | Consolidate prompt state copy so the rail reads as one clear prompt-management flow instead of two separate empty states. |
| P2 | Rail control discoverability | New sidechannel collapse buttons are keyboard-accessible, but visually they are icon-only and rely on icon comprehension. | Add design-system tooltip/title treatment for rail-local collapse/restore controls and align icon meaning with existing header visibility controls. |
| P2 | Power-user controls | Advanced composer controls expose dense icon clusters for modes, MCP, prompt, attachments, tuning, and utilities. | Keep the density, but improve scan grouping and hover/focus names so advanced controls remain fast without becoming recall-heavy. |
| P3 | Bottom status duplication | Bottom status repeats readiness/mode/session/save/model data already visible in the header, rails, and composer. | Review which status chips must remain always visible and which should collapse into a single health/session affordance. |

Quick wins:

- Add empty assistant response messaging and retry affordance. Implemented in this branch: blank assistant turns now render explicit message-card recovery actions and a runtime sidechannel warning with regenerate available for both `role: "assistant"` and real-server `isBot: true` message shapes.
- Add tooltips to sidechannel collapse/restore controls.
- Show compact composition summary while rails are collapsed.
- De-duplicate prompt empty-state copy in the Context rail.

Larger design opportunities:

- Define explicit cockpit density modes for first-use, active conversation, and power-user workbench states.
- Rework center empty-state mode cards so they do not compete with the composer after the user has a clear chat intent.
- Audit mobile cockpit parity after desktop rail collapse lands, especially tab/restore behavior and focus order.

Verification for implemented empty-response quick win:

- `bunx vitest run src/components/Common/Playground/__tests__/Message.error-recovery.integration.test.tsx src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx --config vitest.config.ts` passed with 32 tests.
- `TLDW_WEB_AUTOSTART=false TLDW_WEB_URL=http://127.0.0.1:18002 NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY bunx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium --reporter=line --grep "uses the running server"` passed against the real local server and branch WebUI.
- Updated screenshot: `apps/tldw-frontend/test-results/workflows-chat-cockpit.rea-eebf0-kpit-focus-controls-working-chromium/chat-cockpit-desktop-conversation.png` shows the message-card recovery and runtime sidechannel warning.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
