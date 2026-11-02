---
id: TASK-15413
title: Triage post-merge CI failures for Moderation PR 3396
status: In Progress
labels:
- moderation
- ci
- investigation
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Read-only root-cause triage requested after Moderation compiler cleanup PR3396 merged with failed non-required UX/legacy full-suite checks. Inventory every failed job, reproduce representative causes, distinguish evidence from hypotheses and current-dev fixes, and record owner follow-ups without bundling unrelated runtime fixes into Moderation hardening.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Inventory UX failure and all 27 failed Python shards, with aggregate failure relationship and exact tested revision.
- [x] #2 Reproduce provider-inventory CI allowlist contamination and verify the existing GitHub fixture fix under inherited CI environment.
- [x] #3 Compare relevant source with PR base, identify matching current-dev UX repair, and keep unverified residual root causes explicit.
- [ ] #4 Commit a diagnostic report with commands, evidence, ownership routing and ADR assessment; leave production and test source unchanged.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Known ids checked across origin/dev and full paginated large open PR inventories: maximum15412; allocated15413. Read-only inspection began before task creation. Exact PR head a423dcda4d; merge61d02ed484; current dev3ab1ce5d3a. Parent owns Python triage; independent sidecars inspect UX and residual Python clusters. ADR check: ADR required:no, diagnostic records only; no durable architecture rule changed. No Bandit scope because no source/test changes are planned.
Diagnostic report written at Docs/superpowers/reviews/moderation-backend/2026-11-02-pr3396-post-merge-ci-triage.md. Inventory covers all27 Python shards/161 failed nodes plus UX and aggregate. Controlled provider-inventory reproduction:2fail under CI allowlist,2pass isolated, no source edits. Current-dev GitHub module24pass under CI env confirms existing21dee fixture fix. UX independent review finds matching landed15410/13001.2 repairs; this job message count remains unconfirmed without trace, so causal attribution is qualified. Residual contract clusters recorded without claiming all are pre-existing or fixed. Final source/scope/format verification and residual sidecar consolidation remain before commit.
Residual Python sidecar independently reconciled all161 failed nodes/27shards. Confirmed93 Slides cases split92endpoint-denial signatures and1cancellation timeout;3Streaming header mismatches; ACP/OCR/DB doubles and domain-specific assertions cataloged. None of inspected residual owning/test paths changed base-to-PR; not treated as proof all failures baseline. Existing Admin webhook fixture correction b2f24ab377 also on current dev, with 48-case CI-env validation recorded in TASK-13001.2 but not rerun here. Source scope check confirms no changes under app/tests/WebUI/workflows. Canonical task check, full duplicate-id check and whitespace checks passed; documentation-only Bandit not applicable. Awaiting report commit and final metadata ratchet for completion.
Self-review and independent domain classifications completed. Verified existing GitHub/Admin/UX correction commits are ancestors of current branch. Report preserves root-cause uncertainty for remaining161-case inventory and does not infer blanket baseline status from unchanged paths.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
