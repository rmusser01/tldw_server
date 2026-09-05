---
id: TASK-13174
title: Fix Persona setup voice-default version handoff
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 14:54'
updated_date: '2026-09-05 15:58'
labels: []
dependencies: []
references:
  - Docs/Reviews/MIGU_BUDDY_UAT_2026_09_05.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live Migu UAT on dev 2742468a19: creating a persona then saving voice defaults causes HTTP 409 during setup advancement (database version 2, expected 1). Reloading and reselecting repeats it (4 versus 3), blocking ordinary first-run setup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A newly created persona can save voice defaults and advance to starter commands without reloading.
- [x] #2 Concurrent genuine edits still produce a recoverable conflict without overwriting newer data.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace panel save and setup checkpoint version ownership.
2. Add failing regressions for returned-version handoff, persona switches, and concurrent conflicts.
3. Propagate the saved profile through the callback and verify targeted frontend tests.
ADR required: no
ADR path: N/A
Reason: Routine bugfix preserving the existing optimistic version contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation: AssistantDefaultsPanel now passes the saved profile with voice defaults; setup adopts that response and explicitly uses its returned version for the checkpoint PATCH. Late panel saves are discarded on persona change/unmount; setup responses/errors/saving state are fenced to the selected persona. No backend optimistic-lock behavior changed.
Files: AssistantDefaultsPanel.tsx, usePersonaSetupOrchestrator.ts, sidepanel-persona.tsx, their three targeted regression files.
ADR required: no; routine repair of the existing optimistic version contract.
Red evidence: version-enforcing setup flow failed both normal and competing-edit cases at voice-to-commands; panel tests failed discarded response version and previous-persona form contamination; checkpoint tests failed old response clearing new saving state.
Green evidence: bun x vitest run ../packages/ui/src/components/PersonaGarden/__tests__/AssistantDefaultsPanel.test.tsx ../packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx ../packages/ui/src/routes/hooks/__tests__/usePersonaSetupOrchestrator.test.tsx from apps/tldw-frontend: 94 passed, 3 files (2026-09-05). Logs: /private/tmp/task-13174-final-tests.log.
Static checks: targeted ESLint 0 errors, 88 existing warnings; git diff --check clean. New hook regression formatted to surrounding shared-UI conventions. Bandit not applicable: TypeScript/TSX-only changes.
Limitations: live newly-created Migu retest delegated to coordinating agent; task remains In Progress until that UAT. Tests use a version-enforcing HTTP double, not a real backend. Existing Node localStorage experimental warning and jsdom navigation warning remain. Existing first defaults PATCH remains unversioned; this task repairs the following setup checkpoint version contract.

Real Chromium UAT created Migu UAT Repaired (e0a442a5-3861-4529-a332-a5391626f51f), saved assistant defaults, advanced to Starter commands with Voice defaults completed, then continued to Safety without a 409. Before/after setup snapshots retained in coordinated UAT report.

Review follow-up: reproduce an A-to-B-to-A selection round trip while the first A checkpoint is pending, then fence checkpoint completion/error/finally by selection generation. Preserve existing returned-version handoff and targeted validation scope.

Review follow-up resolved: an A-to-B-to-A selection round trip previously allowed a pending old-A checkpoint to apply version 3 or publish an old conflict onto the reselected A. Two new regressions failed with those exact stale mutations. Each checkpoint now captures a selection-generation token; selection effect cleanup increments it on switch/unmount, and success, error, and finally all require matching persona ID and generation.
Follow-up verification: 96/96 tests passed across the same three targeted frontend files (/private/tmp/task-13174-roundtrip-green.log); red evidence /private/tmp/task-13174-roundtrip-red.log. Targeted orchestrator/test ESLint: 0 errors, 27 existing warnings. git diff --check clean. No additional runtime or browser changes.

Coordinated final validation: 265 focused frontend tests, 54 backend tests, production Bandit0 findings, scoped frontend ESLint0 errors (warnings documented), unchanged Python lint baseline, real browser evidence and limitations recorded in Docs/Reviews/MIGU_BUDDY_UAT_2026_09_05.md. Repository-wide typechecking remains limited by80 diagnostics across6 unchanged unrelated files; no full suite run.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Setup uses the returned profile version, preserves real optimistic conflicts, and ignores stale completions including A-to-B-to-A selection. Fresh persona and final clean-browser saves advance correctly.96 focused setup tests pass within the265-test final frontend run.
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
