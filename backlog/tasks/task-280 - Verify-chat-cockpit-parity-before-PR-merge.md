---
id: TASK-280
title: Verify /chat cockpit parity before PR merge
status: Done
assignee: []
created_date: '2026-05-12 01:23'
updated_date: '2026-05-12 02:56'
labels:
  - webui
  - chat
  - frontend
  - pr-review
dependencies:
  - TASK-275
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1582'
  - Docs/superpowers/plans/2026-05-11-chat-cockpit-focus-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Validate and, if needed, patch PR #1582 so the redesigned /chat cockpit preserves the existing chat-page workflows and the new cockpit sidepanel controls work end to end. Scope stays limited to /chat WebUI/extension shared Playground surfaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Existing /chat composer controls remain reachable in cockpit and focus layouts: model selector/settings, character selector/settings, Search & Context, web search, MCP/tools, attachments, send controls, artifacts, shortcuts, and thread search where applicable.
- [x] #2 New cockpit sidepanel controls open or invoke their intended existing /chat workflows rather than only rendering static summaries.
- [x] #3 Focused tests cover parity interactions and sidepanel end-to-end behavior without broad unrelated page coverage.
- [x] #4 Browser or equivalent rendered verification records desktop cockpit/focus and mobile cockpit/focus behavior after fixes.
- [x] #5 PR #1582 is updated with the validated changes and known baseline blockers remain clearly separated.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Patched degraded readiness layout so degraded non-chat subsystems show a warning while /chat still receives a full-height application viewport.

Kept legacy composer controls immediately below the textarea and before transient notices so model, MCP, Search & Context, prompt, character, attachments, tools, send, and advanced controls remain reachable in cockpit/focus layouts.

Hardened in-flight model metadata fetch failures so transient backend warmup errors resolve through cached/empty fallback instead of surfacing a Next dev runtime overlay over /chat.

Real-server correction pass: added apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts. This Playwright spec intentionally uses no page.route, route.fulfill, or __tldw_test_bypass hooks. It seeds only the real server URL/API key and exercises /chat through the browser against http://127.0.0.1:8000.

Real server evidence collected 2026-05-12: /api/v1/health returned HTTP 206 with JSON status degraded; degraded check was chacha_notes while database and metrics were healthy. /api/v1/llm/providers returned HTTP 200 with total_configured 26, configured_count 20, default_provider openai. /api/v1/llm/models/metadata returned HTTP 200 with total 852.

Real-server Playwright verification passed after formatting: TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY TLDW_SERVER_URL=http://127.0.0.1:8000 NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 TLDW_WEB_URL=http://localhost:18014 TLDW_WEB_CMD='bun run dev -- -H 127.0.0.1 -p 18014' bunx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium --reporter=line. Result: 2 passed in 10.4s. Desktop covered degraded warning pass-through, cockpit controls, Search & Context, MCP unavailable disabled state, advanced controls, model/prompt/character selectors, tools menu, Current Chat Model Settings, Scene Director, focus mode hide/restore. Mobile covered default focus composer, opening cockpit rails, Context/Runtime summaries, model/character sidepanel actions, and returning to focus.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verified PR #1582 with a no-stub Playwright spec against the real running server at http://127.0.0.1:8000. The live server reported degraded health via HTTP 206 because chacha_notes was degraded while database and metrics were healthy; /chat remained usable and showed the degraded warning instead of blocking. Real provider/model endpoints returned configured data: /llm/providers HTTP 200 with total_configured 26 and default_provider openai, and /llm/models/metadata HTTP 200 with total 852. The new real-server Playwright spec passed 2/2 after formatting and covers desktop cockpit/focus controls plus mobile focus/cockpit rail behavior without page.route, route.fulfill, or __tldw_test_bypass.
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
