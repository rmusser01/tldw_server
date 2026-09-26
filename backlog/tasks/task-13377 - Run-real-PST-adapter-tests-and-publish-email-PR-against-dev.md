---
id: TASK-13377
title: Run real PST adapter tests and publish email PR against dev
status: Done
assignee: []
created_date: '2026-09-26 21:50'
updated_date: '2026-09-26 22:31'
labels: []
dependencies: []
documentation:
  - Docs/Operations/Email_Real_PST_Validation_2026-09-26.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User requested executing the two previously skipped pypff/PST tests, fixing verified failures if necessary, updating evidence and creating a PR against dev. Use only public synthetic fixture data and isolated optional parser; preserve measured benchmark source/results and personal mail exclusion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both optional installed-parser and valid-PST endpoint tests run without skips and verified results are retained
- [x] #2 Any verified PST regression is fixed with focused tests and lint/security validation
- [x] #3 A reviewed pull request targets dev and links measured email evidence; human Change summary merge gate remains explicit
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1: inspect parser/test fixture and current dev diff. Stage 2: run native parser and valid synthetic PST tests; investigate and fix any failures. Stage 3: reconcile evidence, verify and commit. Stage 4: push branch, create and attach PR against dev, record URL.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Native libpff-python20231205 installed only into /tmp/tldw-real-pst-13376-python (project venv unchanged). Pinned public Apache Tika testPST_variousBodyTypes.pst revisione3c6b6b18537100a7016b55cb8d29fa06cf4233a has four fabricated body-format emails,271360bytes,sha25624c5e6bbb8bf26a817c977283e40e7b69d2661fec0845abbe177f97efcb05fb0. Existing two enabled-parser endpoint baseline:1passed1failed (native API has no recipient methods; datetime dropped by string-only helper). Eight metadata regressions RED before fix; eight+two native endpoint tests GREEN10passed. Preserve native datetime and use selected transport-header metadata fallback, without copying MIME encoding/Content-Type. Independent review has no blocking findings; Ruff and Bandit0 for changed parser. Broader88-case run:87passed1failure from missing-parser test assuming absent optional dependency; test now explicitly monkeypatches pypff unavailable and full24endpoint/metadata rerun is underway. Current dev merge preview has13 conflicts; reviewed hunk-resolution guidance preserves latest dev fallback/cancellation/context/UUID-probe changes and email tenant/transaction fixes.

Dev f5fa1f3a41855aa02871d8b76d0ec0cebbaf9e07 merged with 13 resolved conflicts. Postmerge core110 and real Postgres54 passed. Combined154 run initially141passed13fixture setup errors; shared fixture registration fixed with explicit alias. Subsequent24auth cases23passed1failure exposed cached canonical-auth restoration overwriting validated selected org; synthetic trace confirmed quota selected secondorg but identity wrote firstorg. Six cached-boundary regressions RED2failed5passed then GREEN with canonical authority tests and real authenticated integration:63passed. Trusted request-local marker tied to deep-copied original principal preserves only validated org/team selectors; changed claims/raw state remain ignored. Independent review has no blockers. Final guarded native suite10passed (both realPST cases no skips; outbound/model0). Actual dev diff90Python files compiled,47production/probe +4test Ruff clean. Bandit47 scope8 inherited access/api_key/service label findings exact-matched to dev baseline;0new findings/errors. Final201case combined regression run underway; original benchmark source certificates preserved.

Final combined merged-source regression suite completed:201passed,0failures/errors/skips,10warnings,228.07s. Evidence retained in Docs/Operations/evidence/email_core_closeout_13376/real_pst_13377.json; reproduction and scope in Docs/Operations/Email_Real_PST_Validation_2026-09-26.md. Stage3 complete; stage4 publishing PR underway.

Published https://github.com/rmusser01/tldw_server/pull/3023 against dev, draft/OPEN/MERGEABLE confirmed. All four plan stages complete; task-specific plan retired. Final closeout contains docs/evidence/tracking only. Real OST/live Gmail/staging lag remain unverified optional scope.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Both installed-parser and strict valid-PST endpoint cases execute without skips using pinned public Apache Tika synthetic data and libpff20231205. Fixed native datetime/transport metadata loss, deterministic missing-parser testing, shared fixture registration and dev cached-auth reset of validated org selection. Current dev integrated and independently reviewed. Final201combined/110core/54officialPostgres regressions passed; guarded10native cases0outbound/model.90Python files compile;47production/probe +4tests Ruff clean; Bandit8inherited label findings exactly match dev and0new/errors. Evidence/report source hashes retained; older benchmark source certificates unchanged. Draft PR3023 targetsdev, attached and mergeable; human-written Change summary required before merge, separate release approval unrecorded. Temporary parser/fixture/diagnostic resources cleaned; sharedvenv/Postgres preserved. Completed task-specific plan retired; active worktree retained for PR review.
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
