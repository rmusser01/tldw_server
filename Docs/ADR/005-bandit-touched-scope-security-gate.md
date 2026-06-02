# ADR-005: Bandit Touched-Scope Security Gate

**Status:** Accepted
**Date:** 2026-06-02
**Backfilled from:** `AGENTS.md`
**Decision owner:** User + prior project guidance
**Related task:** TASK-506, TASK-507, TASK-508
**Related spec/plan:** `Docs/superpowers/specs/2026-06-02-adr-workflow-adoption-design.md`

## Decision

Run Bandit on touched Python/code scope before considering work complete; for docs-only changes, document why Bandit is not applicable.

## Context

The project handles authentication, media ingestion, sandboxing, providers, and local/self-hosted data. Security-sensitive Python changes need an explicit security scan gate that scales to the touched scope.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Full-repo Bandit every time | Can be expensive and noisy for narrow changes. |
| No routine Bandit gate | Security regressions in touched code could be missed. |
| Run only in CI | Local completion should catch new findings before review. |

## Consequences

Agents should activate the project virtual environment and run `python -m bandit -r <touched_paths> -f json -o /tmp/bandit_<task>.json` for touched Python/code paths. New findings in changed code should be fixed before finishing. Docs-only work records Bandit as not applicable.

## Follow-up

None for Stage 1.
