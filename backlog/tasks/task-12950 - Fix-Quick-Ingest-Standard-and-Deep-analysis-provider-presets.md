---
id: TASK-12950
title: Fix Quick Ingest Standard and Deep analysis provider presets
status: In Progress
labels:
- bug
- frontend
- quick-ingest
documentation:
- Docs/superpowers/specs/2026-07-12-quick-ingest-preset-provider-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Target dev with a shared WebUI/browser-extension fix so the active Quick Ingest wizard hydrates saved preset configuration, lets users select an analysis provider in the configure step, and processes Standard/Deep without a misleading provider error when configured. Preserve the early missing-provider guard and add focused regression coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 New Quick Ingest sessions use the saved preset configuration for the selected/default preset.
- [ ] #2 The active wizard configure step exposes an analysis provider control when analysis is enabled.
- [ ] #3 Standard and Deep can proceed when a provider is configured; missing-provider flows stay on a recoverable wizard step without a render loop.
- [ ] #4 Focused unit/integration tests cover WebUI/extension shared behavior and pass.
- [ ] #5 A pull request targets dev.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
