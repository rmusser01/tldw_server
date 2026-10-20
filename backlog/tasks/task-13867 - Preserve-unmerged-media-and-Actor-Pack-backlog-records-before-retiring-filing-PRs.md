---
id: TASK-13867
title: Preserve unmerged media and Actor Pack backlog records before retiring filing
  PRs
status: In Progress
labels:
- maintenance
- backlog
- documentation
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve the useful records from PR #2783 and PR #2810 under globally unused task IDs on dev, qualify dated media evidence, and close the original filing PRs only after the replacements land. Also record verified task closeouts for the same authorized batch (#3061, #3207, #3176), marking those tasks Done only after their actual normal merges. Runtime fixes and Actor Pack design remain separate future work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Three canonical replacement tasks preserve the original requirements, provenance, and historical evidence without colliding with unrelated records.
- [x] #2 Current media wording incorporates the existing model-based provider selection and all actionable filing review feedback.
- [ ] #3 Independent requesting-code-review and applicable documentation checks pass on the replacement head.
- [ ] #4 The replacement PR lands through normal required gates before the two original filing PRs close with replacement links.
- [ ] #5 Associated task records for #3061, #3207 and #3176 receive accurate completed status and merge/verification evidence only after those PRs have actually merged.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-20-preserve-stale-filing-records.md
ADR required: no. ADR-002 and ADR-059 govern task tracking; this filing cleanup makes no API or architecture decision. Future media compatibility and Actor Pack design decisions retain their own ADR assessments.
Allocation: fresh origin/dev and every open PR file inventory had maximum TASK-13866 on 2026-10-20; selected TASK-13867 through TASK-13870.
Source PRs: https://github.com/rmusser01/tldw_server/pull/2783 and https://github.com/rmusser01/tldw_server/pull/2810.
2026-10-20 verification: deterministic preservation checks pass for four canonical fresh records, all three preserved requests remain To Do, all nine Actor Pack criteria match the original verbatim, and unrelated automation/other TASK-13111 files remain byte-identical to dev. Python/Bandit/runtime tests are inapplicable to documentation-only scope; no live API or cross-repo content verification is claimed. Independent review and hosted merge are pending.
Independent requesting-code-review of 2b925562e585fe1703219dd4af5eac29fafbce09..efb4a544ad606a6bb7c4ae64a6a8b52d29793e6a completed with no Critical, Important, or Minor findings. Reviewer independently confirmed historical description preservation, all nine Actor Pack criteria, To Do status, canonical format and unrelated-record preservation. Applicable pre-commit and diff checks pass. Human Change summary, hosted exact-head/current-base gates, merge and source-PR retirement remain pending.
2026-10-20 batch closeout scope: requester also authorized driving #3061/#3207/#3176 to completion. This final documentation PR will carry their associated task-only closeouts after actual verified merges, keeping Backlog status current alongside the three preserved To Do requests. #3061 has merged normally as 454974e812f8faf8532c931f3f7c8e0bf8d9445c; its reviewed implementation remains unchanged and its separate local closeout is 942396fa39eb087cbb51ddd591bc9ba89095a844. Other closeouts remain pending. AC3 is reopened for fresh independent review of the combined final documentation scope; the earlier clean review applies to the original preservation-only head. No extra runtime or architecture work is added; ADR-002/059 continue to govern. Human summary/waiver is still pending, and neither source filing PR has closed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The three requested records are preserved locally as TASK-13868 (media provider/translation), TASK-13869 (undeclared media fields), and TASK-13870 (Actor Pack design). All remain To Do. Historical evidence and references are retained and qualified; current wording addresses the original filing findings. The initial preservation-only documentation checks and independent review passed. Fresh independent review and checks of the combined final documentation scope, including verified implementation-task closeouts, remain pending. This tracking task stays In Progress until the replacement PR lands, dev is verified, and #2783/#2810 close with links.
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
