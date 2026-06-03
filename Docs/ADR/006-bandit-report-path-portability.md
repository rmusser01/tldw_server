# ADR-006: Bandit Report Path Portability

**Status:** Accepted
**Date:** 2026-06-03
**Backfilled from:** not backfilled
**Decision owner:** User + Codex collaboration session
**Related task:** TASK-512
**Related spec/plan:** PR #2230 review follow-up
**Supersedes:** ADR-005

## Decision

Bandit remains required for touched Python/code scope, but report output paths must be portable and must not hard-code `/tmp`.

## Context

ADR-005 established the touched-scope Bandit gate but used `/tmp/bandit_<task>.json` as the example report path. The project supports Windows, macOS, and Linux, so hard-coding a Unix temporary directory makes the guidance less portable and creates unnecessary friction for agents working outside Unix-like environments.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep `/tmp/bandit_<task>.json` | Not portable to all supported platforms. |
| Use platform-specific temporary environment variables | Adds shell-specific complexity to a simple project guidance command. |
| Omit the report output path | Loses the durable JSON artifact useful for recording verification evidence. |

## Consequences

Agents should activate the project virtual environment and run `python -m bandit -r <touched_paths> -f json -o bandit_<task>.json` or another explicitly chosen portable output path for touched Python/code paths. New findings in changed code should be fixed before finishing. Docs-only work records Bandit as not applicable. Bandit report artifacts should not be committed unless explicitly requested.

## Follow-up

None for this follow-up.
