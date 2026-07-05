# ADR-002: Backlog.md Task Tracking

**Status:** Accepted
**Date:** 2026-06-02
**Backfilled from:** `AGENTS.md`, `Docs/superpowers/specs/2026-05-03-backlog-md-task-tracking-design.md`
**Decision owner:** User + prior Codex collaboration session
**Related task:** TASK-506, TASK-507, TASK-508
**Related spec/plan:** `Docs/superpowers/specs/2026-05-03-backlog-md-task-tracking-design.md`

## Decision

Require an associated Backlog.md task before work changes repository files.

## Context

The repository needs a durable task and history layer that records why work exists, how it was planned, what files changed, what verification ran, and what was skipped or blocked.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Git commits only | Commits do not capture task state, verification history, blockers, or reviewable unit boundaries. |
| GitHub issues only | Not every local agent task maps cleanly to a remote issue, and local work needs MCP/CLI-first task tracking. |
| Manual markdown notes | Too easy to duplicate or bypass; Backlog.md provides a consistent task workflow. |

## Consequences

Repo-changing work must search for or create a Backlog task before edits begin. Read-only investigation can proceed without a task. Backlog tasks link to specs, plans, PRs, verification, and final summaries; they do not replace those artifacts.

## Follow-up

None for Stage 1.
