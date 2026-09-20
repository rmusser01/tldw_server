---
id: TASK-13263
title: Prepare the 0.1.43 release with all changes since v0.1.42
status: In Progress
assignee: []
created_date: '2026-09-20 19:56'
updated_date: '2026-09-20 21:56'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare a reviewed release candidate based on v0.1.42 and frozen dev d72b1d2850ea947b6d12cac19f6b95867b68a580. Preserve 0.1.42 release fixes, reconcile released main into dev, inventory every new commit and merged PR, update release metadata and protected source records, and open a draft release PR. Track outstanding 0.1.42 publication verification in TASK-13013.3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The candidate includes v0.1.42 and frozen dev as ancestors, with reviewed conflict resolutions.
- [x] #2 Changelog and release notes cover all post-0.1.42 changes, with an exhaustive commit inventory.
- [x] #3 Version metadata, documentation and protected-source records are consistent and verified.
- [x] #4 A draft PR and release plan record checks, publication state and remaining human decisions.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Release plan: Docs/superpowers/plans/2026-09-20-release-0.1.43-plan.md. Frozen delta: 616 commits and 19 first-parent merges after v0.1.42. Five conflicts resolved retaining transport security and dev persistence. Independent static review found no concrete merge regression. Notes regression exposed obsolete authority mock, removed; 26 Notes tests pass and 107 other merge regressions pass. Historical license manifest check updated to pin immutable published bytes.

Strict docs build passes. Sync draft PR2971 opened. Existing plugin isolation test failed before fixing four AuthNZ registrations, then 6 passed; 2559 affected tests collect without error. Prompt/editor regressions:1125 passed. Bandit has no new findings; two pre-existing B105 synthetic-secret fixture findings exactly match HEAD. Source manifest and candidate records will be regenerated after frontend type repairs.

Typing reassessment: intersection return type left unknown-array filter overloads; Omit on the transport index signature erased variables; compiler inferred normalization already preserves both concrete arrays, so use inferred return type rather than layering annotations. Reviewer identified future-development manifest coupling; ordinary CI checks record consistency, while the release plan explicitly runs TLDW_VERIFY_RELEASE_SOURCE=1 for checkout equality.

Final protected source ef64db48f520971bd593e6caec87e368b0354470,7321files,manifest a54e754a3e9edb29ad91ef74c8000244bda720d50b8f7588ce6f213827111eeb. Full WebUI typecheck passes. Metadata/workflow92pass, explicit release source12pass, strict MkDocs and wheel/sdist/backend-boundary checks pass. Legal dates remain proposed; release PR, CI and human Change summary pending.

Draft release PR2972 contains all616commits/19integrationPRs, consistent metadata and protected-source record. Publication evidence JSON records all three0.1.42signed image attestations against25608249ed. PyPI404 remains; test-only recoveryPR2973 is prepared, CI restarted after automatic approval rejected merge/publication. Requester rejected extra human-summary gate for recovery. Next candidate legal dates and publication remain unapproved; main/dev syncPR2971 remains draft.

Fresh candidate CodeQL review includes alert2693: buffered character-chat SSE fallback returns raw exception text, unlike existing safe lazy-stream handlers. Investigating and repairing under this release task with error-injection regression; five frontend test-fixture alerts require evidence-backed disposition. No release publication performed.

Requester explicitly approved2973 human-summary waiver/merge/PyPI retry. Recovery merged cd2dbc792b8888555abac5c5c9eafa7a43d9b0e4; package inputs unchanged vs immutablev0.1.42,PyPI404 verified. Publication dispatched. Candidate currentCI exposed backend mypy targeting3.11 while runner NumPy stubs require3.12; frontend shard5 fails separately. Fixing under existing release task; freshQodo Notes boolean/endpoint-owner findings under13263.1.
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
