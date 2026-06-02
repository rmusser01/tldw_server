# ADR-004: AI-Generated PR Change Summary Gate

**Status:** Accepted
**Date:** 2026-06-02
**Backfilled from:** `AGENTS.md`, `Docs/superpowers/AI_GENERATED_PR_CHANGE_SUMMARY_POLICY_2026_04_17.md`
**Decision owner:** User + prior project guidance
**Related task:** TASK-506, TASK-507, TASK-508
**Related spec/plan:** `Docs/superpowers/AI_GENERATED_PR_CHANGE_SUMMARY_POLICY_2026_04_17.md`

## Decision

Materially AI-authored PRs are not merge-ready until the human requester writes a `Change summary` explaining what changed and why those implementation choices were made.

## Context

The project allows AI-assisted development but needs human ownership of architectural and implementation rationale before merge.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Allow AI-generated summaries | A diff recap or AI-authored rationale does not prove human understanding or ownership. |
| Require no summary | Reviewers lose a concise human explanation of why the implementation is the right one. |
| Ban AI-authored PRs | Too restrictive for the project workflow. |

## Consequences

AI-generated PRs need a human-written summary. If the requester cannot explain the rationale in their own words, the PR is not merge-ready. Agents may prepare context, but the merge gate requires human ownership.

## Follow-up

None for Stage 1.
