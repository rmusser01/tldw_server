---
id: TASK-2361
title: Fix VZ helper bundle paths with spaces and live smoke socket reuse
status: Done
labels:
- sandbox
- vz_linux
- macos
- bugfix
priority: High
modified_files:
- tools/macos-vz-helper/Sources/Templates/BundleTemplateResolver.swift
- tools/macos-vz-helper/Tests/TemplateResolverTests.swift
- tools/vz-linux-image/scripts/run-host-e2e-smoke.sh
- tools/vz-linux-image/tests/test_host_e2e_smoke_script.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the macOS VZ helper bundle resolver so filesystem paths containing spaces are not percent-encoded, and harden the host E2E smoke script so it refuses an already-live helper socket instead of unlinking it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Bundle validation and real vz_linux execution work with a canonical bundle under ~/Library/Application Support/...
- [x] #2 Host E2E smoke removes stale AF_UNIX sockets but refuses live sockets before starting its own helper
- [x] #3 Focused Swift and shell/Python tests cover both regressions
- [x] #4 Verification and Bandit results are recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented after plan review. Verification: Swift RED test initially failed with vz_linux_bundle_manifest_missing, then passed after replacing URL.path() filesystem uses with URL.path; live socket RED test initially failed because run-host-e2e-smoke.sh unlinked/proceeded, then passed after adding a connection probe before stale socket removal. Full Swift helper tests passed (92 tests). Host smoke script tests passed (16 tests). Real host-gated vz_linux smoke against /Users/macbook-dev/Library/Application Support/tldw/sandbox-images/source-bundles/debian-bookworm-arm64/bundle passed (3 passed, 11 deselected). Bandit on touched Python test file still reports pre-existing low-severity test harness findings (B101/B404/B603); new B108/B101 lines are annotated/removed from actionable findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed macOS VZ bundle handling for filesystem paths containing spaces by using non-percent-encoded URL.path values when resolving bundle artifacts. Hardened run-host-e2e-smoke.sh so it probes an existing AF_UNIX socket and refuses live helper sockets instead of unlinking them, while preserving stale closed-socket cleanup. Verification: Swift path regression failed before the fix with vz_linux_bundle_manifest_missing and passed after; live socket regression failed before the guard and passed after; full Swift helper tests passed (92 tests); host smoke script tests passed (16 tests); real host-gated vz_linux smoke passed against /Users/macbook-dev/Library/Application Support/tldw/sandbox-images/source-bundles/debian-bookworm-arm64/bundle (3 passed, 11 deselected). Bandit on the touched Python test file still reports pre-existing low-severity test harness patterns (B101/B404/B603); new B108/B101 lines are annotated.
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
