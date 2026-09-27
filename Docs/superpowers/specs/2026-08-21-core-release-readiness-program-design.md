# Core Release Readiness Program Design

**Status:** Approved on 2026-08-21

**Tracking epic:** TASK-13013

## Purpose

Produce an immutable, verifiable public-core release candidate that a
downstream operator can deploy without inheriting undocumented CI, security,
dependency, configuration, migration, data-isolation, or capacity risk.

This program stops at the reusable release handoff. Private infrastructure,
proprietary overlays, customer onboarding, billing, commercial terms, support
operations, and launch decisions are downstream responsibilities and are not
tracked or referenced here.

## Audited Baseline

The task graph was created from remote `origin/dev` at
`2e0815c1e4577902a220044619822ab6b1cb395f`. The dirty local checkout was not
used as source evidence and is not modified by this program branch.

The 2026-08-21 audit found:

- the latest merged pull request had a failing `backend-required` result caused
  by checked-in versus runtime OpenAPI fingerprint drift;
- the live dev ruleset required only the frontend license-policy check and did
  not enforce all checks named by `Docs/Development/CI_REQUIRED_GATES.md`;
- the documented release process cuts releases from `main`, not directly from
  `dev`;
- the dev train was substantially ahead of the latest tagged release while the
  Unreleased changelog section did not describe that train;
- source version, GitHub release, and package-distribution versions were not in
  one coherent state;
- stale security pull requests still covered RAG query logging, MediaWiki user
  scoping, media permissions, weather egress, and admin impersonation;
- Bun applications were outside the complete dependency-update and SBOM path;
- production dependency and base-image resolution was not fully immutable;
- TASK-12116 still tracked disabled frontend safety gates; and
- TASK-12983 incorrectly owned a private customer deployment and depended on a
  missing TASK-12982 record.

## Ownership and Information Boundary

The public program owns only behavior and artifacts useful to every operator:

- required CI and merge enforcement;
- release version and lineage;
- reusable security and privacy primitives;
- production-safe reference configuration;
- dependency and software-supply-chain evidence;
- frontend safety gates;
- tenant-isolation and data-lifecycle primitives;
- capacity and soak-test tooling; and
- an immutable downstream release handoff.

The public task graph must not contain a private repository URL, infrastructure
address, credential reference, customer identity, commercial term,
proprietary patch, or protected reusable artifact.

## Task Decomposition

### CI and Release Lineage

- TASK-13013.1 restores every required gate to green on one exact dev head.
- TASK-13013.2 aligns the live ruleset with the documented required gates.
- TASK-13013.3 freezes and publishes a coherent release through `main`.

TASK-13013.3 cannot complete from a red, stale, or unreviewed head.

### Security and Production Defaults

- TASK-13013.4 resolves or supersedes stale security PRs 2610, 2614, 2622,
  2623, and 2625 against current dev.
- TASK-13013.5 hardens trusted proxy identity and login-lockout isolation.
- TASK-13013.6 ships a safe reference deployment and minimal public health
  surface.
- TASK-13013.7 closes dependency, SBOM, vulnerability, and immutable artifact
  provenance gaps.

### Product Safety and Operational Evidence

- Existing TASK-12116 completes frontend type, lint, persisted-state, and
  dependency-alignment safety gates.
- TASK-13013.8 proves reusable tenant isolation, export, deletion, and durable
  cleanup primitives.
- TASK-13013.9 provides an operator-runnable capacity and soak-test harness.
- TASK-13013.10 repairs active tracker identity and dependency integrity.

### Downstream Handoff

TASK-12983 is re-scoped from customer deployment to the public release handoff.
It depends on the verified release, security, deployment, dependency,
data-lifecycle, and frontend safety tasks. It records exact source and artifact
identity, SBOM, configuration schema, migration compatibility, and rollback
compatibility. It does not deploy or describe a private service.

## Release Candidate Exit Criteria

The public release candidate is ready for downstream acceptance only when:

1. every documented required gate passes on the exact candidate;
2. the live ruleset prevents red or stale required checks from merging;
3. version, changelog, tag, release, package, and artifact provenance agree;
4. no unresolved critical or high-severity release finding remains;
5. production defaults, proxy trust, health exposure, dependencies, tenant
   isolation, data lifecycle, and frontend safety meet their task contracts;
6. capacity evidence defines a reproducible reference envelope; and
7. TASK-12983 publishes the immutable, privacy-safe downstream handoff.

Passing this gate states that the reusable core candidate is suitable for
downstream evaluation. It does not approve any operator's production,
commercial, paid, or public launch.
