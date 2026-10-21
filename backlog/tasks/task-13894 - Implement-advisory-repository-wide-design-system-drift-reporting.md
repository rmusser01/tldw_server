---
id: TASK-13894
title: Implement advisory repository-wide design-system drift reporting
status: Done
labels:
- design-system
- webui
- extension
- governance
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the reviewed five-stage plan for repo-wide advisory token/color/radius/layout reporting; related design task TASK-45.44.16 and issue #1673.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Four UI roots are scanned with syntax-aware rules and scoped definitions
- [x] #2 Reports preserve findings/exceptions/provenance and distinguish incomplete scans
- [x] #3 CI is advisory and respects existing license-first contracts
- [x] #4 Focused tests, report sample review, lint and touched-scope Bandit are recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-20-design-system-drift-reporting.md. Spec: Docs/superpowers/specs/2026-10-20-design-system-drift-reporting-design.md. ADR check: proposed ADR-079 governs the lasting advisory UI convention; no new ADR. Existing source/dependency state will be verified before implementation.
Implementation progress: locked JS dependencies installed without lockfile changes. Token authority39 tests and rule engine37 tests pass. Report15 tests pass. CLI19 tests pass, including source/output protection, hard-linked output replacement, unavailable sparse source and hidden Git-index edits; actual RED verified for both review bugs and both hidden-index cases before fixes. Existing product-state guard54 tests remains passing; CI contract113 tests pass. Scoped new-test TypeScript compilation passes; pinned frontend ESLint returns0 with existing Next pages-directory advisory. Fullrepo first scan completed on3,644 source files before vendor/system-color refinements; final recount pending. Review findings are being addressed before publication. Bandit touched Python test reports233 existing low findings, no scan errors; baseline comparison pending. Ruling: default jsdom Vitest environment replaces plan node annotation because shared setup requires window. Ruling: exact upstream react-pdf.css copy excluded as vendor based on file license/header; malformed-vendor regression verified RED/GREEN. Atomic report replacement prevents hard-link truncation.
Final review: CLI/report scoped re-review found no remaining findings; independent probes verify atomic replacement preserves hard-linked originals, report permissions0600, temp cleanup, and genuine deletions stay complete/dirty. A separate final token/rule/CI review reproduced3 silent rule gaps: non-definition custom-property color overrides, quoted/spread CVA options, and literal RGB channels with variable alpha. Rules worker is adding regression tests and minimal fixes. Parent current pre-fix matrix169 focused tests and113 CI tests passes; final counts and fullscan will be rerun after these fixes.
Additional validation:169 focused tests pass (token44/rules37/report15/CLI19/existing guard54),113 CI contract tests pass, new-test scoped tsc passes, actionlint1.7.12 passes, all script node --check passes,7 nonempty surface/rule sample locations exactly match source, local document links pass. Bandit baseline238 findings vs current233 all-low findings, zero scan errors and zero findings on added lines; reports /tmp/bandit_design_system_drift_stage5_base.json and /tmp/bandit_design_system_drift_reporting.json. Current scan after narrow vendor exclusion:3,643 analyzed files,2,249 token/radius findings and1,976 layout candidates,986 coverage diagnostics,0 error diagnostics. These are advisory measurements, not enforced thresholds. Pending3 final rule-review fixes will require fresh counts/tests before PR publication.
Fresh post-fix matrix181 focused tests passes;113 CI contracts remain valid. Whole-repo report at fb15f10ed2 is complete/clean,3,643 analyzed files,2,249 token/radius plus1,976 layout candidates,992 coverage diagnostics,0 errors. Three original rule-review holes now reproduced correctly; scoped re-review found1 additional spread-order false positive where a later semantic CVA variants property overwrites an earlier literal spread. Worker is adding minimal overwrite-order regressions and fix. CLI/report re-review remains clean. Core stages committed independently; CI/docs publication pending final scoped rule re-review.
Final verification after all review corrections:189 focused tests pass (44 token,57 rule,15 report,19 CLI,54 existing guard);113 CI contracts pass. Scoped tsc, pinned ESLint, script syntax checks, actionlint and diff checks pass. Existing Node localStorage and Next pages-directory advisories remain. Original review findings and residual CVA overwrite order have passing RED/GREEN regressions; final CLI and rule scoped re-reviews report no findings. Required product-state guard/baseline, tokens, product components, lockfile and required workflows are unchanged. Draft PR/public-tracker publication is next; hosted Actions/full application/browser suites have not been run locally.
Published draft PR https://github.com/rmusser01/tldw_server/pull/3294 against dev. Issue #1673 now links the spec/plan/ADR and verified progress; epic #1655 guard row is In Progress with PR #3294. Clean source report at7b4cd63336:3,643 analyzed files,2,249 token/radius findings,1,976 layout candidates,992 coverage diagnostics,zero errors. Local implementation and scoped reviews complete; hosted Actions, human Change summary and merge remain public delivery gates. Full app/browser suites not run; existing localStorage/Next advisories and baseline Bandit findings explicitly documented. Plan retained because public delivery/review is not complete.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented repository-wide advisory design-system drift reporting in draft PR #3294: static surface-specific token authority, four syntax-aware rules, exact definition scopes, visible intentional/stale exceptions, deterministic JSON/summary, reliable Git provenance and safe atomic report output. Added independent nonrequired CI reporting and migration guidance without changing product UI, tokens, existing guards, dependencies or required gates. Verified189 focused tests and113 CI tests, scoped TypeScript/ESLint, Node syntax, actionlint, document/task checks and zero new Bandit findings on added lines. All identified review issues have regression coverage and clean scoped re-reviews. Issues #1673/#1655 updated; hosted CI and requester-authored Change summary remain pending before merge.
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
