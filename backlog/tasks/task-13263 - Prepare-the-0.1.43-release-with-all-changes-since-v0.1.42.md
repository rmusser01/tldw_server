---
id: TASK-13263
title: Prepare the 0.1.43 release with all changes since v0.1.42
status: In Progress
assignee: []
created_date: '2026-09-20 19:56'
updated_date: '2026-09-20 23:11'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare a reviewed release candidate based on v0.1.42 and frozen dev d72b1d2850ea947b6d12cac19f6b95867b68a580. Preserve 0.1.42 release fixes, reconcile released main into dev, inventory every new commit and merged PR, update release metadata and protected source records, and open a draft release PR. Track outstanding 0.1.42 publication verification in TASK-13013.3.

Protected source: `cca220627a8f0f9f3124126b50add2e0e88ee8a7`.
Protected manifest SHA-256: `bfd14b9e3fb5efa6ecbe54c2e27267cd4958b51881aad9cd3e1b4ab100ecd4e1`.
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
Historical execution notes: earlier source digests, pending approvals and CI diagnoses below describe their checkpoint only. Use the current source authority in the description and the latest approved recovery state in the release plan.

Release plan: Docs/superpowers/plans/2026-09-20-release-0.1.43-plan.md. Frozen delta: 616 commits and 19 first-parent merges after v0.1.42. Five conflicts resolved retaining transport security and dev persistence. Independent static review found no concrete merge regression. Notes regression exposed obsolete authority mock, removed; 26 Notes tests pass and 107 other merge regressions pass. Historical license manifest check updated to pin immutable published bytes.

Strict docs build passes. Sync draft PR2971 opened. Existing plugin isolation test failed before fixing four AuthNZ registrations, then 6 passed; 2559 affected tests collect without error. Prompt/editor regressions:1125 passed. Bandit has no new findings; two pre-existing B105 synthetic-secret fixture findings exactly match HEAD. Source manifest and candidate records will be regenerated after frontend type repairs.

Typing reassessment: intersection return type left unknown-array filter overloads; Omit on the transport index signature erased variables; compiler inferred normalization already preserves both concrete arrays, so use inferred return type rather than layering annotations. Reviewer identified future-development manifest coupling; ordinary CI checks record consistency, while the release plan explicitly runs TLDW_VERIFY_RELEASE_SOURCE=1 for checkout equality.

Historical protected source ef64db48f520971bd593e6caec87e368b0354470,7321files,manifest a54e754a3e9edb29ad91ef74c8000244bda720d50b8f7588ce6f213827111eeb. Full WebUI typecheck passes. Metadata/workflow92pass, explicit release source12pass, strict MkDocs and wheel/sdist/backend-boundary checks pass. Legal dates remain proposed; release PR, CI and human Change summary pending.

Draft release PR2972 contains all616commits/19integrationPRs, consistent metadata and protected-source record. Publication evidence JSON records all three0.1.42signed image attestations against25608249ed. PyPI404 remains; test-only recoveryPR2973 is prepared, CI restarted after automatic approval rejected merge/publication. Requester rejected extra human-summary gate for recovery. Next candidate legal dates and publication remain unapproved; main/dev syncPR2971 remains draft.

Fresh candidate CodeQL review includes alert2693: buffered character-chat SSE fallback returns raw exception text, unlike existing safe lazy-stream handlers. Investigating and repairing under this release task with error-injection regression; five frontend test-fixture alerts require evidence-backed disposition. No release publication performed.

Requester explicitly approved2973 human-summary waiver/merge/PyPI retry. Recovery merged cd2dbc792b8888555abac5c5c9eafa7a43d9b0e4; package inputs unchanged vs immutablev0.1.42,PyPI404 verified. Publication dispatched. Candidate currentCI exposed backend mypy targeting3.11 while runner NumPy stubs require3.12; frontend shard5 fails separately. Fixing under existing release task; freshQodo Notes boolean/endpoint-owner findings under13263.1.

Recovery2973 approved/merged cd2dbc792b; publication35540174556 testgate running; candidate mergedrecovery eb4ec4817d. Backend actualfailureOpenAPIdescriptiondrift fixed222a6833b1,exactCIhashreproducedandtypesgenerated,fulltscpass. Frontendfixture b0585d63de,82pass. Protectedsource222a6833b14bd847cf6783adc0fe4f829a1ba386,7321files,manifest419a2b8f8dd81f23da35360a6ec5771c2d396ae6fc6f6ba5678faa21edac1ba5. Explicitcheck12pass. AdvisorymypyNumPystubs mismatch isnotjobfailure.

Current protected source645e58c6a7 and manifest1d3af39b50da7797f8d6ec2aaabbedee22d5a6d7cf7ee6a76a388cc32c283b51 cover7322 files. Explicit protected-checkout gate13passed. All18 agentic fixes/dispositions locally verified; final push/thread replies/remote CI next. Recovery2973 merged under requester approval, PyPI35540174556 still testing. Sync2971ca39055070 typing repairs verified1125tests/fulltsc. Companion2763all8followups addressed4030d6d58d.

New CI follow-up after Qodo closure: c5453d4385 frontend shards4/8 and8/8 fail in media page tests. Logs identify missing useQueryClient exports in two React Query mocks after authority-cache cleanup and obsolete stale-selection expectation. Reproduce both complete files, repair fixture contract/security-state expectation only if confirmed, then focused media regressions/tsc/lint. Parent owns source commit and protected-manifest refresh.

c545 CI follow-up verified: media page mock regressions reproduce41failed/1pass, then both files plus real-query outage/hydration58pass/4files. Added stable useQueryClient/removeQueries fixture contracts; stale-deletion security test now asserts selection cleared before and after late404, preserving no-warning/no-refetch assertions. Full nonincremental WebUI tsc exits0. ScopedESLint baseline10/current10/new0; diffcheckclean; Bandit N/A TS-only. Two testfiles+this tracking only; no production edits/skip/timeout changes. Evidence /tmp/candidate-c545-frontend-ci-results.md. Parent owns review/sourcecommit/protectedmanifest/push.

Required-CI test-only repaircca220627a verified independently58tests, nonincrementaltsc, no new ESLint. Refreshed7322-file protected manifest and all current source authorities; explicit licensing13pass. Separate0.1.42 PyPI timeout/failure triage now in release plans; publication remains blocked, no gate weakened.
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
