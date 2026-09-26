---
id: TASK-13377
title: Run real PST adapter tests and publish email PR against dev
status: In Progress
assignee: []
created_date: '2026-09-26 21:50'
updated_date: '2026-09-26 22:01'
labels: []
dependencies: []
documentation:
  - Docs/Plans/IMPLEMENTATION_PLAN_email_real_pst_pr_13377.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User requested executing the two previously skipped pypff/PST tests, fixing verified failures if necessary, updating evidence and creating a PR against dev. Use only public synthetic fixture data and isolated optional parser; preserve measured benchmark source/results and personal mail exclusion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both optional installed-parser and valid-PST endpoint tests run without skips and verified results are retained
- [ ] #2 Any verified PST regression is fixed with focused tests and lint/security validation
- [ ] #3 A reviewed pull request targets dev and links measured email evidence; human Change summary merge gate remains explicit
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1: inspect parser/test fixture and current dev diff. Stage 2: run native parser and valid synthetic PST tests; investigate and fix any failures. Stage 3: reconcile evidence, verify and commit. Stage 4: push branch, create and attach PR against dev, record URL.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Native libpff-python20231205 installed only into /tmp/tldw-real-pst-13376-python (project venv unchanged). Pinned public Apache Tika testPST_variousBodyTypes.pst revisione3c6b6b18537100a7016b55cb8d29fa06cf4233a has four fabricated body-format emails,271360bytes,sha25624c5e6bbb8bf26a817c977283e40e7b69d2661fec0845abbe177f97efcb05fb0. Existing two enabled-parser endpoint baseline:1passed1failed (native API has no recipient methods; datetime dropped by string-only helper). Eight metadata regressions RED before fix; eight+two native endpoint tests GREEN10passed. Preserve native datetime and use selected transport-header metadata fallback, without copying MIME encoding/Content-Type. Independent review has no blocking findings; Ruff and Bandit0 for changed parser. Broader88-case run:87passed1failure from missing-parser test assuming absent optional dependency; test now explicitly monkeypatches pypff unavailable and full24endpoint/metadata rerun is underway. Current dev merge preview has13 conflicts; reviewed hunk-resolution guidance preserves latest dev fallback/cancellation/context/UUID-probe changes and email tenant/transaction fixes.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
