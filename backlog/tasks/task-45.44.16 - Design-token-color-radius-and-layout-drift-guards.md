---
id: TASK-45.44.16
title: Design token, color, radius, and layout drift guards
status: Done
assignee: []
created_date: 2026-05-14 03:20
labels:
- design-system
- webui
- extension
- governance
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/issues/1673
- https://github.com/rmusser01/tldw_server/issues/1655
- https://github.com/rmusser01/tldw_server/pull/3294
- Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
- Docs/Design/tldw_web_design_system_contract.md
- Docs/ADR/098-advisory-ui-design-drift-reporting.md
parent_task_id: TASK-45.44
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Mirror the linked GitHub governance issue. Closure requires a durable guard, documented policy, CI path, component ownership decision, documentation artifact, or visual QA checklist as specified by the GitHub issue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The linked GitHub issue owns public status.
- [x] #2 Backlog notes record PR links and verification evidence.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-20: User approved repo-wide UI drift reporting with no blocking. Writing a design-only spec for a standalone verifier covering shared UI, WebUI, extension, and admin UI; existing and new findings stay visible. ADR check: ADR required: yes; proposed Docs/ADR/079-advisory-ui-design-drift-reporting.md because this establishes lasting UI conventions and nonrequired reporting policy. No runtime, verifier, or CI changes in this design slice.
Design artifact: Docs/superpowers/specs/2026-10-20-design-system-drift-reporting-design.md; proposed ADR-079 linked from ADR index and main design-system contract. Validation: 78 relative Markdown links resolve across four documents; no TODO/TBD/FIXME placeholders; backlog-py normalize --check passes; git diff --check passes. Bandit and runtime/browser tests skipped because this slice changes only Markdown. Written-spec approval and implementation planning remain pending; issue #1673 remains open.
Read-only independent review found no substantive contradictions or scope issues in the spec and ADR. Parser feasibility and actual CI behavior remain implementation-plan verification work; neither is claimed complete.
2026-10-20: Subsequent owner-requested review found four valid spec gaps, now authorized for correction: declaration/export-level definition exemptions, named CSS colors, Tailwind @apply analysis, and source provenance for dirty/untracked local files. This supersedes the earlier no-findings review assessment. Scope remains spec and task documentation; ADR-079 advisory policy is unchanged.
All four review findings addressed in the design spec: mixed definition/product files remain scanned at declaration/export granularity; named CSS colors are covered in color-valued positions; @apply utilities use the same scanner; sourceCommit/dirtySource/untrackedSourceFileCount define local provenance and nonignored untracked inputs. Added corresponding acceptance test requirements. Documentation-only validation: diff whitespace and task canonical formatting pass; placeholder scan is clear. No executable verifier/CI edits; Bandit, runtime tests, and browser checks remain inapplicable to this revision.
2026-10-20: Owner requested continuation after all four spec-review fixes. Preparing a five-stage implementation plan from the corrected spec. Confirmed declared dependencies in @tldw/ui: TypeScript, PostCSS, Tailwind, and jsdom; this fresh worktree has no installed JS dependencies yet. Plan must preserve license-first CI ordering and keep reporting outside required gate aggregates. ADR check: existing proposed ADR-079 covers this plan; no additional architecture decision.
Implementation plan written: Docs/superpowers/plans/2026-10-20-design-system-drift-reporting.md. Five stages cover static token authority, syntax-aware rules, reports/exceptions, Git-aware CLI/provenance, and advisory CI/docs. Includes named colors, @apply, declaration-level exemptions, dirty/untracked source, safe output destinations, and current frozen ordinary-workflow inventory/license-first requirements. Self-review mapped all spec sections and checked interfaces. Three local document links resolve; five stage sections exist; placeholder and diff-whitespace checks pass. Spec status records owner-requested planning after review fixes. Planning-only change: dependency installation, executable tests, Bandit, browser checks, reporter implementation, and live CI validation have not run.
2026-10-20: Reviewed spec and implementation plan are committed on codex/design-system-drift-guard-spec-20261020. Implementation is tracked by TASK-13894: standalone four-root advisory reporting, tests and independent nonrequired CI workflow are implemented; current final scoped review is correcting CVA spread overwrite order before draft PR publication. Existing required product-state guard, baseline, tokens and product components are unchanged. Public issue #1673 remains open until delivery/review completion.
Draft implementation PR #3294: https://github.com/rmusser01/tldw_server/pull/3294. TASK-13894 is Done for the reviewable implementation/local-verification unit. Public issue #1673 and epic #1655 now contain artifacts, validation and PR progress; this governance task remains In Progress until public delivery/review/merge is complete. No blocking enforcement enabled. Local result189 focused tests,113 CI tests and clean committed-source scan; full details and skips in TASK-13894/PR.
2026-11-24: PR #3294 has been rebased onto dev 0a5a41294a and the review follow-up is recorded in TASK-13894. ADR-098 remains proposed and advisory; no new ADR decision or enforcement. The requester waived the Change summary for #3294 only. Keep the governance task In Progress until hosted validation and public delivery are complete.
Final 2026-11-24 rebase base: dev b4a9550f76. The PR's advisory scope and ADR-098 are unchanged; closure still requires hosted validation and delivery.
2026-11-27: Verified that PR #3294 merged on 2026-11-25 in 246cb92795 and is an ancestor of remote dev 2053ec5ad4. The repository issue and epic copies now link the delivered reporter and review path. GitHub #1673 remains the public status record.
ADR assessment: no new ADR is required for the delivery record. ADR-098 records the proposed advisory-only policy. Any required enforcement needs a separate owner decision and CI policy change.
Fresh local verification on clean 2053ec5ad4 used Node 24.6.0 and Bun 1.3.2. The advisory report completed with clean provenance, 3,863 analyzed files across all four roots, 4,220 advisory findings, 1,267 coverage diagnostics, and zero operational errors. All 232 focused reporter and product-state guard tests passed.
Hosted validation: verified the report generation, upload, and job-summary steps and downloaded the resulting JSON from https://github.com/rmusser01/tldw_server/actions/runs/46053191753/job/140694788637. The artifact reports complete and clean provenance at ed36f29369, a post-delivery PR head, with 3,863 analyzed files and zero operational errors. Five documentation links or anchors and git diff --check passed. Bandit is inapplicable because the closeout changes only Markdown.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #3294 delivered the advisory styling drift reporter and CI workflow. Verified a clean latest-dev scan and the published hosted artifact, and linked the reporter and review guidance from the repository issue and epic copies.
<!-- SECTION:FINAL_SUMMARY:END -->
