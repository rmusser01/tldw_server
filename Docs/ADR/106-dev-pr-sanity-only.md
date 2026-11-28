# ADR-106: Limit dev PR CI to sanity checks

**Status:** Proposed
**Date:** 2026-11-27
**Decision owner:** Repository maintainer
**Related task:** TASK-15693
**Related plan:** [CI required gates](https://github.com/rmusser01/tldw_server/blob/dev/Docs/Development/CI_REQUIRED_GATES.md#server-ci-tiers)

## Decision

Run lightweight, path-scoped checks on PRs targeting `dev`. Backend sanity covers Python syntax, the HTTP client patch guard, and the quickstart dry run. Frontend sanity covers lint and TypeScript checks. Keep backlog ID validation, trusted license admission, and static security checks.

Docker builds, coverage floors, browser tests, and broad test suites run on their existing `main`, publication, nightly, or manual routes. The four full-suite workflows still named in the `dev` ruleset keep their PR triggers and report skipped required jobs. The three active gates enforce license admission and change detection.

This updates the `dev` PR portion of [ADR-096](096-tiered-development-and-release-ci.md). Full release qualification follows ADR-096. Strict current-base enforcement and queueing follow [ADR-063](063-in-repo-merge-queue.md).

## Context

The earlier tiering reduced supplemental server CI while retaining the seven required workflows. A backlog-only PR still built five images, and frontend changes could start dozens of runner jobs. When `dev` advanced, the queue had to repeat required work on the rebased head.

## Alternatives considered

| Option | Reason |
| --- | --- |
| Skip image builds only for docs and backlog changes | Code PRs would still start image builds and heavier test suites. |
| Stop requiring an up-to-date branch on `dev` | It reduces reruns and allows changes to merge without fresh checks on the combined tree. |

## Consequences

Runner work on each fresh `dev` head is smaller. Full suites can find regressions later, during nightly runs or `main` qualification. GitHub requires checks on the current head when a rebase is necessary.
