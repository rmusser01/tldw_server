# ADR-071: Conversation source incarnation

**Status:** Proposed
**Date:** 2026-10-12
**Backfilled from:** not backfilled
**Decision owner:** Human project owner; proposal prepared in the authorized chat remediation session
**Related task:** TASK-13605, prerequisite for TASK-13604
**Related spec/plan:** Docs/Plans/IMPLEMENTATION_PLAN_chat_source_incarnation_13605.md

## Decision

A source precondition binds a private server-issued conversation row incarnation, preserved during updates/restore and regenerated on actual insertion, in addition to owner, scope, reusable ID and mutable revisions.

## Context

An empty source hard-deleted and recreated with the same explicit ID and fixed clock reproduces its counters and content digests. Both independent reviewers and actual regression identify the gap. ADR-049/050 ownership/lifecycle requirements need a concrete original-record boundary. SQLite Sync creation logs are prunable; PostgreSQL does not install their equivalent trigger and repairs sequences. A latest remaining log or timestamp is not portable immutable identity.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Existing counters/content hashes | Reinitialize identically on same-ID insertion. |
| Creation timestamp | Same-clock regression reproduces it. |
| Sync journal | Missing PostgreSQL trigger, removable records and sequence repair prevent original-row authority. |

## Consequences

New registered migrations issue identity once for each existing row without rewriting user data. Ordinary and Sync insertion generate fresh server identity; updates/restore preserve it. Caller/copy/import payload cannot choose it. Missing identity refuses the private source precondition. This does not establish native manifest readiness or permit retention publication. Existing accepted ADR rationale is unchanged. Broader H1 selection and backup/import implications need separate qualified integration before public rollout.

## Follow-up

Complete migration/writer and TASK-13604 source-binding regressions; retain public H2/Sync admission gates until qualified.
