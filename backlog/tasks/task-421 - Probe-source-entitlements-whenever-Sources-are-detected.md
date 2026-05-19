---
id: TASK-421
title: Probe source entitlements whenever Sources are detected
status: Done
labels:
- review
- sources
- webui
- capabilities
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the PR review follow-up requiring `/api/v1/ingestion-sources/capabilities` to be attempted whenever generic ingestion source support is detected, with failures ignored safely.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Source entitlement probing is attempted whenever generic ingestion source support is detected.
- [x] #2 Entitlement endpoint failures remain non-fatal and preserve generic Sources support.
- [x] #3 Local-directory create and Notes Sync folder controls block only on a known false entitlement, not unknown entitlement state.
- [x] #4 Focused regression tests and whitespace checks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added nullable entitlement state to `ServerCapabilities.canCreateLocalDirectoryIngestionSource`, defaulting detected Sources support to unknown until the authenticated entitlement endpoint returns a boolean. Removed the OpenAPI-route gate around `/api/v1/ingestion-sources/capabilities`, kept the existing fail-soft catch behavior, updated SourceForm and Notes Sync folder gating to block only on `false`, and bumped the persisted capabilities cache key to V5 so stale V4 `false` values from the old skipped-probe behavior are not reused.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the review follow-up for local-directory entitlement discovery. The WebUI now probes `/api/v1/ingestion-sources/capabilities` whenever Sources support is detected, treats probe failures as unknown instead of disabled, and lets backend create/browse enforcement remain authoritative.

Verification:
- Red tests were observed for the older-authoritative-spec probe, unknown-entitlement SourceForm/Notes gating, and stale V4 capability cache behavior before implementation.
- `bunx vitest run src/services/__tests__/server-capabilities.test.ts src/components/Option/Sources/__tests__/SourceForm.test.tsx src/components/Notes/__tests__/NotesListPanel.sources-sync.test.tsx` -> 3 files, 57 tests passed.
- `git diff --check` -> exit 0.
- `bunx tsc --noEmit -p tsconfig.json` still fails on existing repo-wide baseline type errors in unrelated tests/components; no reported errors were in the capability or source/notes files changed for this task.
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
