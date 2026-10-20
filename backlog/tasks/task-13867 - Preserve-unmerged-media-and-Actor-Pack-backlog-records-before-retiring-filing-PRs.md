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
- [x] #5 Associated task records for #3061, #3207 and #3176 receive accurate completed status and merge/verification evidence only after those PRs have actually merged.
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
2026-10-20 combined documentation preparation: all four original preservation/tracking patches replay unchanged onto current dev e521605198dd311b94d2c63416bf07db3e5e7a0a. Collected the three task-only postmerge closeouts: #3061 merged 454974e812f8faf8532c931f3f7c8e0bf8d9445c from reviewed de8a8c0cb2e0529c10de85aa2f52f63ae6ee60f0; #3207 merged 2e6ceab9fb97c5712533be670504a81a24525489 from reviewed 1938258c37dee1ed776fa8fea02e80ed4a4a1e70; #3176 merged e521605198dd311b94d2c63416bf07db3e5e7a0a from reviewed 9ceeb6f3cdc9bdb5914983f048cf632b42da8f96. Each had all seven exact-head/current-base required statuses and settled independent requesting-code-review before its normal merge. Their associated implementation tasks are now Done with actual verification evidence. Existing historical plan suffixes and task evidence remain intact. The four #3176 report-only AuthNZ failures are public PostgreSQL mirror quota errors before checkout/tests; GitHub refused a single-job retry while the parent workflow runs. No whole-suite pass is claimed. All three preserved feature/design requests remain To Do, neither filing PR is closed, and the final combined review, human Change summary/waiver, fresh hosted gates and verified replacement merge remain pending. ADR-002/059 continue to govern; no runtime or accepted ADR bytes change.
2026-10-20 fresh combined-scope verification on current dev e521605198dd311b94d2c63416bf07db3e5e7a0a: ten changed files are documentation only, all seven changed task records are canonical, three preserved requests remain To Do, all nine original Actor Pack criteria match verbatim, unrelated colliding records are byte-preserved, all three implementation closeouts are Done with every AC/DoD checked after verified merges, historical VN/moderation plan suffixes are byte-identical, and no ADR bytes change. Fresh other-open-PR/dev inventory confirms TASK-13867 through TASK-13870 have no other published owner. Canonical-format ratchet: 4 passed (6 existing warnings); applicable pre-commit hooks and diff check passed. Bandit/runtime tests do not apply to this final documentation-only diff; original implementation verification is recorded in the respective closeouts. Final combined requesting-code-review, human summary/waiver, fresh hosted gates, merge and source retirement are still pending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The three requested records are preserved as TASK-13868 (media provider/translation), TASK-13869 (undeclared media fields), and TASK-13870 (Actor Pack design), all To Do. Requirements, provenance and historical evidence are retained and qualified. Verified #3061/#3207/#3176 task closeouts are collected, with those implementation tasks Done after actual merges. Fresh combined preservation/canonical checks, four format-ratchet tests and applicable hooks pass. Initial preservation-only review passed; fresh independent review of the combined current-base documentation diff remains pending, as do the human Change summary/waiver, current-head hosted gates, replacement merge and source-PR retirement. This tracking task stays In Progress until replacements are verified on dev and #2783/#2810 close with links.
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
