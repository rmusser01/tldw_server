# Global Superpowers ADR Workflow Design

**Related task:** TASK-511
**Status:** Draft for owner review

## Problem

The repo-local ADR workflow is now validated enough to evaluate whether global Superpowers skills should include repository-agnostic ADR prompts.

The global skills should not become `tldw_server`-specific. They should only prompt for ADR or decision-record handling when a repository already has an ADR policy, has another decision-record convention, or the current work creates or changes a durable architecture rule.

## Evidence From Repo-Local Workflow

| Evidence | Outcome |
| --- | --- |
| `Docs/ADR/README.md` and `Docs/ADR/001-adr-workflow-and-governance.md` | Established `Docs/ADR/` as the canonical repo-local ADR home and required ADR assessment for substantial specs, plans, and PRs. |
| `Docs/ADR/inventory/2026-06-03-decision-inventory.md` | Proved that decision backfill needs classification, source review, owner defaults, and an ambiguity gate before accepted ADRs are created. |
| TASK-509 | Produced the decision inventory and recorded documentation-only verification plus Bandit skip. |
| TASK-510 | Required at least one owner-reviewed backfill child slice before global workflow evaluation. |
| TASK-514 | Completed the Workspace/WebUI pilot backfill slice and produced ADR-007, ADR-008, and ADR-009. |
| ADR-007, ADR-008, ADR-009 | Demonstrated that reviewed inventory rows can be converted into accepted ADRs with `Backfilled from:` metadata and source-doc links. |
| Owner-review correction | Vague review prompts were not useful. The workflow needs proposed concrete defaults that an owner can approve, reject, or override. |

## Candidate Skill Changes

| Skill | Current gap | Proposed trigger wording | Risk | Owner decision |
| --- | --- | --- | --- | --- |
| `superpowers:brainstorming` (`$CODEX_HOME/superpowers/skills/brainstorming/SKILL.md`) | The design workflow asks for architecture, components, data flow, error handling, and testing, but it does not check whether the design needs an ADR or decision record. | During design finalization, add an `ADR Assessment` when the repository has ADRs/decision records or when the design creates/changes a durable architecture rule. The assessment must state `Required: yes/no`, the reason, and the target existing or new ADR/decision record. If owner review is needed, propose a concrete default disposition rather than asking abstract questions. | Over-triggering ADRs for small tasks. | Recommended minimal global edit. Gate it to durable architecture decisions and repos with decision-record conventions. |
| `superpowers:writing-plans` (`$CODEX_HOME/superpowers/skills/writing-plans/SKILL.md`) | Plans can be produced without carrying forward governing ADRs or ADR creation work from the spec. | Add an `ADR Assessment` item to the plan header or file-structure section. The plan must list governing ADRs/decision records, say whether a new or superseding record is required, and include exact ADR paths/tasks when creation or updates are required. | Plan bloat and repo-specific path assumptions. | Recommended minimal global edit. Use generic "ADR/decision record" wording and let repo instructions define paths such as `Docs/ADR/`. |
| `superpowers:verification-before-completion` (`$CODEX_HOME/superpowers/skills/verification-before-completion/SKILL.md`) | Completion verification covers tests, lint, build, symptoms, and requirements, but not unresolved ADR assessments. | Before claiming completion, verify that any ADR assessment from the spec/plan is resolved: required ADR/decision record created or superseded, source/index links updated when repo policy requires them, or a no-ADR rationale recorded. | False completion blockers when a repo has no ADR convention. | Recommended minimal global edit. Trigger only when the current work/spec/plan included an ADR assessment or the repo instructions require one. |

## Recommendation

Do not edit global Superpowers skill files in this repo PR.

Create a separate global Superpowers skill update task after owner review of this spec. That task should use `superpowers:writing-skills`, patch only the three candidate skills above, and keep the wording repo-agnostic:

- Use `ADR/decision record`, not only `Docs/ADR/`.
- Trigger on durable architecture decisions or repositories that already define decision-record policy.
- Require concrete proposed dispositions for owner review.
- Avoid making ADRs mandatory for small bug fixes, local implementation details, tests, copy, temporary experiments, or repositories with no decision-record convention.

## Non-Goals

- Do not force ADRs on every repository or every task.
- Do not backfill historical decisions globally.
- Do not replace repo-local `AGENTS.md`, Backlog.md, or existing ADR policies.
- Do not edit `$CODEX_HOME/superpowers/**` from this repo PR.
- Do not encode `tldw_server` paths into global skills.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| ADR over-triggering creates process noise. | Trigger only for durable architecture rules, public API shape, persistence, security, worker ownership, provider integration, major dependency choices, or repo workflow gates. |
| Global skills leak `tldw_server`-specific paths. | Use repo-agnostic wording and require agents to follow local repo instructions for ADR location/template. |
| Owner review becomes vague or frustrating. | Require proposed concrete defaults: current governing, already covered, backfill candidate, superseded/stale, inventory-only, or no ADR needed. |
| Completion checks block work in repos without ADRs. | Verification triggers only when the repo policy or the current spec/plan includes an ADR assessment. |
| Global skill edits bypass review. | Keep this repo PR to a design/spec only; perform global edits in a separate work item using `superpowers:writing-skills`. |

## Rollout Plan

1. Owner reviews this spec.
2. Create a separate global Superpowers skill task outside this repo PR.
3. Use `superpowers:writing-skills` to patch `brainstorming`, `writing-plans`, and `verification-before-completion`.
4. Test the changed skills against this repo's ADR workflow and one repository with no ADR policy.
5. Keep or adjust the global wording based on whether the ADR assessment appears only when it should.

## Verification

- Confirm this spec identifies candidate global skill files and trigger wording.
- Confirm no global Superpowers skill files were modified in this repo PR.
- Bandit is skipped for this task because the work is documentation-only and touches no Python/code paths.
