---
id: TASK-45.44.16
title: Design token, color, radius, and layout drift guards
status: In Progress
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
- Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
parent_task_id: TASK-45.44
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Mirror the linked GitHub governance issue. Closure requires a durable guard, documented policy, CI path, component ownership decision, documentation artifact, or visual QA checklist as specified by the GitHub issue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The linked GitHub issue owns public status.
- [ ] #2 Backlog notes record PR links and verification evidence.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
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
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
