# ADR Workflow Adoption Design

Date: 2026-06-02
Owner: Codex collaboration session
Backlog task: TASK-506
Status: Spec reviewed and approved, pending user review and implementation planning

## Summary

Adopt a repo-local Architecture Decision Record workflow for `tldw_server`.

The ADR system should make significant architecture decisions durable, discoverable, and reviewable without turning every design note into ceremony. The canonical ADR directory will be `Docs/ADR/`. The initial adoption should update repo workflow documentation and then backfill still-governing architectural decisions through staged audit slices.

This is repo-first. Global Superpowers skill changes are intentionally out of scope for the first implementation and should be tracked as a later follow-up after this repo workflow proves useful.

## Goals

1. Create a canonical ADR home at `Docs/ADR/`.
2. Add an ADR template and index that explain numbering, statuses, supersession, and backfill rules.
3. Update root `AGENTS.md` with a concise ADR policy.
4. Integrate ADR checks with existing Backlog.md and Superpowers workflows.
5. Require ADRs for significant durable architecture decisions.
6. Require every substantial spec, plan, or PR to explicitly answer whether an ADR is needed.
7. Define the staged path for backfilling an authoritative ADR set for still-governing architecture decisions already scattered across repository docs.
8. Keep old docs useful by linking to ADRs where practical rather than rewriting them wholesale.

## Non-Goals

1. Changing global Superpowers skill files in this first adoption.
2. Replacing design specs, implementation plans, Backlog.md tasks, PRs, or module documentation.
3. Creating ADRs for every product preference, temporary implementation detail, or stale idea.
4. Rewriting every historical design document during initial setup.
5. Treating backfilled ADRs as if they were written at original decision time.
6. Enforcing ADRs with a custom script in the first implementation unless the manual workflow proves insufficient.

## Implementation Boundary

This design covers the full ADR adoption program, but implementation should be split into reviewable slices.

The first implementation plan should cover Stage 1 only: create the ADR framework, update root `AGENTS.md`, create the required seed ADRs, and create/link follow-up Backlog tasks for the inventory and module-by-module backfill. Stages 2 and 3 are the authoritative migration target, but they should be planned and executed as later slices because they require broad audit and owner review.

## ADR Governance

The repo should use a hybrid ADR rule:

- ADRs are required for significant, durable architecture decisions.
- Every substantial Superpowers spec, implementation plan, or PR must include an explicit `ADR needed?` check.
- If the answer is no, the artifact records a brief reason.
- If the answer is yes, the ADR is written in the same reviewable unit of work.
- Accepted ADRs are immutable except for metadata needed to mark supersession.
- If a decision changes, a new ADR supersedes the old one.
- Backfilled ADRs are allowed, but backfill is metadata, not decision status. A backfilled still-governing ADR should normally use `Status: Accepted` plus `Backfilled from: ...`.

This keeps the relationship simple:

- Module docs, design docs, and `CONTEXT.md`-style documents describe how things work.
- ADRs explain why durable architecture rules exist.

## ADR Triggers

An ADR should be required when a decision creates or changes a durable rule for one or more of these areas:

- Module boundaries, ownership, or layering.
- Public API shape, compatibility, versioning, or envelope conventions.
- Persistence model, database ownership, migrations, tenancy, or sync semantics.
- Security model, AuthNZ behavior, sandbox isolation, secrets handling, or trust boundaries.
- Scheduler, Jobs, worker, queue, or lifecycle ownership defaults.
- Provider integration architecture or default provider selection rules.
- Cross-app WebUI/extension conventions.
- Long-term dependency choices and major framework/toolchain direction.
- Repository workflow gates that materially affect future contributors or agents.

An ADR is usually not needed for:

- Small bug fixes that preserve existing architecture.
- Local implementation details that do not set precedent.
- One-off product copy or UI preference choices.
- Temporary experiments that are not accepted as project direction.
- Test-only changes that do not define a broader testing policy.

## ADR Directory And Template

`Docs/ADR/` should contain:

- `000-template.md`
- `README.md` as the ADR index and workflow guide
- numbered ADR files like `001-backlog-md-task-tracking.md`

Template:

```markdown
# ADR-{N}: {Short title}

**Status:** Proposed | Accepted | Superseded by ADR-{N}
**Date:** YYYY-MM-DD
**Backfilled from:** {source path, or "not backfilled"}
**Decision owner:** {human/session/reviewer}
**Related task:** {Backlog task ID/link}
**Related spec/plan:** {paths}

## Decision

One sentence stating what was decided.

## Context

Why this decision was needed.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| {Alternative A} | {Reason} |

## Consequences

What this means going forward, including accepted tradeoffs.

## Follow-up

Optional implementation, audit, or documentation follow-up links.
```

Numbering should use the next sequential integer based on `Docs/ADR/*.md`, with `000-template.md` reserved. Backfilled ADRs still receive normal numbers, but their `Backfilled from:` source makes the timing clear.

## Workflow Integration

### Backlog.md

Any repo-changing ADR work needs a Backlog.md task before file edits begin, following the existing repo policy.

Tasks should record ADR-relevant decisions in notes, acceptance criteria, implementation plan links, or final summary. If a PR requires an ADR, the task should reference the target ADR path.

### Superpowers Specs

Substantial design specs should include an `ADR Assessment` section:

```markdown
## ADR Assessment

Required: yes | no
Reason: {short rationale}
Target ADR: Docs/ADR/NNN-title.md | none
```

If `Required: yes`, implementation planning should include a concrete task for creating or superseding the ADR.

### Implementation Plans

Substantial implementation plans should include an `ADR Assessment` section or explicitly reference the approved design spec's ADR assessment when the plan introduces no new durable decision. Plans should include an ADR task when the approved design requires one. The plan should also list any existing ADRs that govern the work.

### Review And Verification

Before completion, agents should verify that significant durable decisions have either:

- an ADR, or
- a documented `ADR not needed` rationale in the task, spec, plan, or PR notes.

### AGENTS.md

Root `AGENTS.md` should get a dedicated ADR section near the Backlog and planning guidance. It should define:

- canonical ADR directory
- ADR triggers
- `ADR needed?` requirement
- immutability and supersession rules
- backfill labeling rules
- relationship between ADRs, design docs, module docs, and Backlog tasks
- pointer to `Docs/ADR/README.md`

### Global Superpowers

Global Superpowers files should not be modified in this first implementation. A follow-up Backlog task should track whether to update the broader brainstorming, writing-plans, and verification skills after the repo-local workflow is tested.

## Broad Migration Plan

The authoritative backfill should run as staged audit slices rather than one uncontrolled sweep.

### Stage 1: Framework And Seed ADRs

Create the ADR framework and these required seed records for current governing decisions already present in root `AGENTS.md` or approved by this design:

- ADR workflow and governance.
- Backlog.md task tracking requirement.
- Jobs vs Scheduler default.
- AI-generated PR human change-summary gate.
- Bandit touched-scope security gate.

If implementation discovers that one of these seed decisions is already fully covered by an existing ADR, the plan should link the existing ADR instead of duplicating it and record the deviation in the Backlog task.

### Stage 2: Decision Inventory

Audit existing decision sources and produce a default inventory at `Docs/ADR/inventory/YYYY-MM-DD-decision-inventory.md` covering:

- `Docs/Design/**`
- `Docs/Plans/**`
- `Docs/superpowers/specs/**`
- `Docs/superpowers/plans/**`
- embedded ADRs such as `Docs/Evals/Evals-Plan-1.md`
- module docs containing terms like `Decision`, `Architecture Decisions`, `Resolved Decisions`, or `Superseded`

The inventory should classify candidates as:

- `current`
- `superseded`
- `stale`
- `duplicate`
- `needs-owner-review`

### Stage 3: Module-By-Module ADR Conversion

Convert current governing decisions into ADRs by module or domain. Old docs should receive short references such as `Covered by ADR-012` or `Superseded by ADR-018` where useful.

Do not rewrite old plans wholesale. Preserve historical docs as history while making the ADR set authoritative for current architecture.

### Stage 4: Optional Automation

Add a lightweight verification script or checklist later if the manual process proves insufficient. Initial adoption should stay procedural and readable.

## Error Handling And Scope Control

If a discovered decision is stale, ambiguous, or contradicted by implementation, do not convert it into an accepted ADR. Put it in the inventory as `needs-owner-review`.

If an old design doc contains many decisions, split only still-governing architectural decisions into ADRs. Product preferences, temporary implementation details, and rejected old plans stay in their original docs.

If two docs conflict, prefer current implementation plus maintainer confirmation over older prose.

If an ADR is requested during a PR but the decision is not actually architectural, record `ADR not needed` in the Backlog task, plan, or PR notes instead of creating noise.

If a decision changes, create a new ADR and mark the old one `Superseded by ADR-N`. Do not rewrite the old rationale.

## ADR Assessment

Required: yes

Reason: This workflow creates durable repository governance for how future architecture decisions are recorded, reviewed, superseded, and backfilled.

Target ADR: `Docs/ADR/NNN-adr-workflow-and-governance.md`, to be created during implementation after the ADR framework exists.

## Initial Implementation Acceptance Criteria

1. `Docs/ADR/000-template.md` exists.
2. `Docs/ADR/README.md` documents ADR numbering, status, supersession, backfill, and workflow rules.
3. Root `AGENTS.md` contains a dedicated ADR workflow section.
4. The workflow requires ADRs for significant durable architecture decisions.
5. The workflow requires substantial specs, plans, or PRs to make an explicit `ADR needed?` call.
6. The workflow preserves Backlog.md and Superpowers requirements rather than replacing them.
7. Required seed ADRs are created or existing ADR coverage is linked for ADR governance, Backlog.md tracking, Jobs vs Scheduler, AI-generated PR change summaries, and Bandit touched-scope validation.
8. A decision inventory/backfill follow-up path is created for broad migration.
9. A follow-up task captures potential global Superpowers skill updates after repo-local validation.
10. Verification includes `git diff --check`; Bandit is documented as not applicable for docs-only changes unless code is touched.

## Program Completion Criteria

The broader authoritative backfill program is complete when:

1. The decision inventory has covered the configured source sets.
2. Current governing architectural decisions have ADRs or explicit owner-reviewed exclusions.
3. Stale, superseded, duplicate, and ambiguous decisions are classified in the inventory rather than silently ignored.
4. High-value source docs link to covering or superseding ADRs where practical.

## Verification Plan

For this design document:

1. Run spec review loop.
2. Run `git diff --check`.
3. Commit the design spec and Backlog task update.
4. Ask the user to review the written spec before implementation planning.

For implementation:

1. Inspect `Docs/ADR/` files and root `AGENTS.md`.
2. Confirm seed ADRs and inventory follow-up tasks are linked from Backlog.md.
3. Confirm `ADR Assessment` appears in the implementation plan.
4. Run `git diff --check`.
5. Document Bandit as not applicable if only docs/process files are touched.
