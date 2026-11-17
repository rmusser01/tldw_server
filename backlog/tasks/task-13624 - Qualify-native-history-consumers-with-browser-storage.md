---
id: TASK-13624
title: Qualify native history consumers with browser storage
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #3229 shared UI CI executes real Plasmo storage, while three native history consumer suites only succeed under the WebUI storage alias. Install the existing browser API fixture before source imports and qualify unchanged ownership, recovery and settlement assertions in both environments.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All three native history consumer suites pass under the shared UI configuration with real Plasmo storage
- [x] #2 Original guard, recovery and retirement assertions remain intact; both client types and bounded review pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new ADR; test-environment dependency restoration only. Reproduction: two representative fresh-native admission tests pass under WebUI config and fail under shared UI config with recovery-storage-unavailable. Plan: Docs/Plans/IMPLEMENTATION_PLAN_chat_native_history_storage_13624.md. PR https://github.com/rmusser01/tldw_server/pull/3229.
Independent bounded review found native Plasmo get/set are instance fields, so prototype fault spies were WebUI-only, and nested global cleanup retired the Chrome API. Stopped the first full run (exit130), migrated controls through actual native storage I/O with the WebUI instance path retained, restored the declared API after nested cleanup, and added physical held-read/write controls. No production storage or admission guards changed.
Final four-file native and WebUI runs each pass836/836: actions331, normal268, compact230, physical SDK7. Bounded review found enumeration/legacy-deserialization regression in the first adapter; no-key/null reads now delegate to captured native get and keyed values reuse safeStorageSerde. Enumeration coexistence and held physical read/write controls pass. WebUI persistent certification remains independent of Chrome API retirement; native certification still refuses retirement. Both client type checks and final review/hooks pending.
Final native and WebUI four-file runs each pass836/836 after the last review fixes. Both normal client type checks exit0. Independent bounded review clear; review document Docs/Reviews/CHAT_NATIVE_HISTORY_STORAGE_13624_2026_10_12.md. Bandit N/A TS test-only. Normal exact-file hooks next.
Normal exact-file pre-commit hooks all passed. Native and WebUI each836 tests, both client types, and bounded review clear. Installed-extension, remaining whole-PR CI/UX/H2 and final review/merge remain open.
Completed owned plan retired to /private/tmp/IMPLEMENTATION_PLAN_chat_native_history_storage_13624_completed.md; durable evidence remains in the review document.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Declared the actual native browser storage dependency in three history consumer suites and moved fault controls to the captured SDK I/O while preserving the WebUI backend. Restored runtime after nested cleanup and kept enumeration/legacy values intact. Each runner passes836 tests, both client types and normal hooks pass; bounded review clear. No production guard changes.
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
