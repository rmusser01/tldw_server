---
id: TASK-15483
title: Pin CI runner image to ubuntu-24.04 until Playwright supports 26.04
status: In Progress
labels:
- ci
- infra
- hotfix
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On 2026-11-08 GitHub rolled ubuntu-latest to Ubuntu 26.04. The repo's pinned Playwright cannot install chromium on ubuntu26.04-x64 ('Playwright does not support chromium on ubuntu26.04-x64'), so every playwright-based required check fails on every open PR at the browser-install step (observed on PRs #3372/#3435: E2E UX Regression, UX Smoke Gate, Onboarding E2E Gate, E2E Critical, Extension Research Workspace Parity, frontend-ux-gates, research-workspace-parity, watchlists e2e, e2e-required). Minimal unblock: pin runs-on: ubuntu-24.04 in the jobs that install playwright browsers (7 workflows identified). Long-term fix (separate task): bump Playwright + pinned browser versions for 26.04 support, then unpin.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All playwright-based required checks install browsers and run on the pinned image
- [ ] #2 No other workflow behavior changes (diff is runs-on pins only)
- [ ] #3 Follow-up task filed for the Playwright bump + unpin
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Pinned runs-on: ubuntu-24.04 across all ubuntu-latest jobs in the 7 playwright workflows (ux-regression-nightly, frontend-ux-gates, frontend-required, ui-research-workspace-parity, ui-research-workspace-nightly, ui-watchlists-extension-e2e, frontend-e2e-tiers) — 20 job pins total, runs-on lines only.
Contract ratchet caught over-pinning: queue-tick/changes (merge-queue + change-detection machinery, no playwright) in frontend-required.yml and frontend-e2e-tiers.yml were collateral pins; reverted to ubuntu-latest. Playwright-running jobs stay pinned. Net: 18 job pins.
Scope extension (2026-11-09): the 26.04 image also drops ffmpeg from PATH, breaking the backend Full Suite shards (auth-integration-b-z, integrations: 'Preflight [ffmpeg_available]: fail - ffmpeg not found') — the 'CI' workflow's 16 ubuntu-latest jobs pinned to ubuntu-24.04 as well; workflow-contract ratchet verified green locally (95 passed) with the full pin.
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
