---
id: TASK-280
title: Verify /chat cockpit parity before PR merge
status: Done
assignee: []
created_date: '2026-05-12 01:23'
updated_date: '2026-05-12 02:21'
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

Rendered browser verification on http://127.0.0.1:18012/chat with backend http://127.0.0.1:8000 confirmed: no runtime overlay, cockpit Search & Context close/open works, MCP tools opens, Advanced controls toggles, model selector opens, Prompt selector opens, character selector opens, More tools opens, sidepanel model settings opens the model/character workflow, and sidepanel character settings opens the character workflow.

Rendered focus verification confirmed Enter focus chat hides both cockpit side panels while preserving composer textarea, Send, MCP tools, Search & Context, and Current Chat Model Settings; Show cockpit panels restores both side panels without runtime overlay.

Focused frontend suite passed: 17 files, 89 tests. Backend model filter pytest passed: 6 tests. Bandit on llm_providers.py wrote /tmp/bandit_chat_cockpit_parity.json with 0 errors and 0 results. git diff --check passed.

Known baseline/environment notes: focused tests still log mocked 'tldw server not configured' 400s in server chat settings paths while passing; the live browser environment can show the existing no-provider empty-state copy if no real LLM API key is configured.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verified and patched PR #1582 for /chat merge readiness. Existing chat composer controls remain reachable in cockpit and focus layouts; new cockpit sidepanel actions invoke the existing Search & Context, model settings, and character workflows; degraded health now permits /chat with a warning instead of collapsing the app viewport; transient model metadata failures fall back without masking /chat behind a dev runtime overlay. Validation: focused frontend suite 17 files / 89 tests passed, backend llm model filter tests 6 passed, Bandit on llm_providers.py found 0 results, git diff --check passed, and rendered browser verification covered desktop cockpit controls plus focus-mode panel dismissal/restoration. Mobile behavior is covered by the existing focused responsive/mobile component tests in the same suite.
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
