---
id: TASK-13696
title: Bound database CI workload without weakening integration coverage
status: In Progress
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR3229 published4b CI db-privileges collected3827 cases, progressed75% without assertions failing, then hit existing55minute global limit in ordinary per-test PostgreSQL schema setup. Split DB_Management m-r and s-z from privilege/media/unit paths in all five existing matrices. Retain unique fresh databases, all tests and existing limits; prove exact coverage and disjoint ownership in every matrix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Regression rejects unsplit m-z workload across all five matrices
- [x] #2 Partition file union matches prior assignment exactly with no duplicate ownership and no changed test deadlines
- [ ] #3 Workflow contracts, shard coverage, hooks and independent review pass; final exact-head CI completes bounded shards
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required:no; this follows existing CI matrix partitioning and changes no durable runtime, persistence or security rule. Plan: stage1 capacity evidence and red partition contract; stage2 minimal five-matrix split and green contracts; stage3 exact-head CI and merge. CI log /private/tmp/chat-pr3229-db-privileges-4b.log; reviewerB independently confirms setup cost not leaked pools; reject schema-template caching because it adds migration semantics unnecessarily.
RED: expanded exact partition regression rejects original matrix with KeyError missing db-management-m-r. Minimal all-five-matrix split retains original220file union:134m-r,42s-z,44privilege/media/unit; zero duplicate assignments. First full contract run caught named owner set needing both added shards; corrected expected ownership, no weakened assertions. Final58workflow contracts PASS. Shard coverage848patterns5104test files0new uncovered44baseline. Ruff0. Bandit373baseline→374current: sole added finding is ordinary pytest B101 assertion, existing B105unchanged,0scanerrors,0production changes. Independent reviewerB CLEAR source. Existing55min job execution and300s test limits unchanged. Final actual bounded shard runtimes and exact-head CI pending publication.
Formatted the sole added assertion using Black style; remaining79historical formatting hunks equal baseline. Complete normal configured hooks PASS; no bypass. Detailed final evidence appended to Docs/Reviews/CHAT_FINAL_RECOVERY_SETUP_HANDOFF_13673_13680_2026_10_14.md. Exact bounded shard runtimes remain pending final-head CI.
Prepublication matrix expansion check found existing release defect: two OS values times187shards gives374jobs (preceding baseline185times2=370), above GitHub256matrix limit. Extend same capacity unit: split existing release OS axis into macOS and Windows jobs, share only unchanged shard/step/env/condition/needs via supported YAML anchors, preserve all374release cells, add RED matrix cardinality and OS parity regressions, extend every existing platform contract to sixth job. GitHub official workflow reuse docs confirm anchors/aliases; no merge keys or new dependencies. Final production types/build98hashes and23cockpit cases already PASS on bd4139; subsequent edits CI/docs-only.
Release repair qualified: exact374unique OS/shard/path cells retained; two187job matrices each below256,10parallel perOS preserves prior20combined,35minute limits unchanged. Added overflow/platform parity regressions RED then GREEN. All102 connected workflow/Admin-ratchet/license-first contracts PASS. Actionlint1.7.12 actualCIversion PASS. Expanded job inventories preserve immutable checkout/admission/cache restrictions; rawsource checkout reference counts unchanged throughaliases. Both independent working-diff reviewers CLEAR. Coverage guard0new uncovered. Final actual Linux bounded shard execution remains a published CI gate; release-only OS execution not claimed.
Final three-file Bandit comparison:689→696 ordinary pytest B101 assertions; B404/B105/B607/B603 counts unchanged and0scanerrors, no new security findings. Ruff0 acrossallthree touched contract files. Final hooks and exact committed review precede publication. Original55minute exhausted shard was unique; other slowest completed shards33.48min/30.10min, supporting a targeted database capacity split rather than unrelated fixture changes.
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
