# TASK-13147 — Materialize Personal Context Sync domains

## Stage 1: Contract analysis and RED tests

**Goal:** Pin exact five-domain validation and service-bound materialization.

**Success criteria:** New focused tests fail because Personal Context adapters,
materializers, authenticated sync application, and factory registration are absent.

**Status:** Complete

## Stage 2: Whole-object adapters

**Goal:** Validate canonical schema-v1 payloads, HMAC tags, identities, sizes,
purge generations, and optimistic lineage without touching canonical storage.

**Success criteria:** Invalid or stale envelopes yield stable rejected/conflict
outcomes; literal replays remain idempotent; no raw body reaches results or logs.

**Status:** Complete

## Stage 3: Service-owned materialization

**Goal:** Apply accepted objects only through the authenticated
`PersonalContextService` and complete five-domain factory registration.

**Success criteria:** Canonical versions are exact, authorization precedes body
handling, Task-1 availability becomes true only for a complete usable transport.

**Status:** Complete

## Stage 4: Verification and review

**Goal:** Complete targeted regressions, static/security gates, task evidence,
and independent review.

**Success criteria:** Scoped Sync and Personalization tests plus Ruff, compilation,
Bandit, diff hygiene, and review pass; TASK-13147 is complete.

**Status:** Complete

## ADR check

ADR required: no (existing)

ADR path: `backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md`

Reason: ADR-002 already governs canonical whole-object transport, server mutation
authority, key custody, integrity, and purge fencing.
