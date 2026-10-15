---
id: TASK-13698
title: Use the explicitly installed Playwright browser in UX CI
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR3229 final onboarding job fails twice before tests in dependency installation; attempt2 identifies Puppeteer postinstall chrome-headless-shell download ECONNRESET. UX workflows invoke Playwright and explicitly install its Chromium, not the separate Puppeteer CDP tool. Skip only redundant Puppeteer postinstall browser download during these two UX dependency installs, retaining explicit Playwright Chromium and all browser checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both UX dependency installs skip only Puppeteer browser download
- [x] #2 Explicit Playwright Chromium installation and onboarding/smoke test commands remain required
- [x] #3 Connected workflow contracts, actionlint, hooks, security comparison and independent review pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required:no new architecture decision; existing explicit browser provisioning and license-first workflow policies govern. Stages: classify actual failure and caller reach; add failing contract then minimal scoped environment; qualify contracts, review, exact-head GitHub execution. Actual Puppeteer24.36 getConfiguration reads PUPPETEER_SKIP_DOWNLOAD. Only scripts/cdp-examine-workflows.ts uses Puppeteer and neither UX gate invokes it. No dependency/version/lockfile changes; no retry loop, ignore-scripts or test deadline changes. Original logs private chat-pr3229-6b1334-onboarding.log and onboarding-attempt2.log.
FailingworkflowcontractREDthen127connectedcontractsPASS. ActualPuppeteer24.36postinstallwithscopedflagprintsSkippingdownloadingbrowsersandexit0. ExplicitPlaywrightChromiuminstallationandalltestcommandsrequiredunchanged; noCDPexaminerusesingates. Black/Ruff/actionlint/hooksPASS; touchedBandit50baseline→56currentordinaryassertions,0errors. Bothindependentreviewersclear. Exactnewheadonboardingexecutionremainsrequiredatlanding.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Skip only unused Puppeteer browser postinstall download in two Playwright UX dependency installs. Retain explicit required Playwright Chromium provision and all actual browser gates. Contracts127pass, actual postinstall verified, lint/security/hooks/review clear; published-head browser execution remains merge gate.
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
