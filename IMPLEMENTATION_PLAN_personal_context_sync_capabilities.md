# TASK-13146 — Advertise Personal Context Sync capabilities

## Stage 1: Contract analysis and RED tests

**Goal:** Pin the server Sync v2 domain, operation, capability, authorization,
key-availability, and quota contract.

**Success criteria:** Focused model, service, and endpoint tests fail only
because Personal Context capability advertisement is absent.

**Status:** Complete

## Stage 2: Typed protocol advertisement

**Goal:** Add the five approved domains and exact typed Personal Context
capability object to Sync v2.

**Success criteria:** Capabilities advertise schema version 1, HMAC-SHA-256,
wrapped bootstrap, cleanup acknowledgments, purge generations, and approved
quotas with stable serialization.

**Status:** Complete

## Stage 3: Availability gates

**Goal:** Derive readiness from `server_trusted_v1`, Shared Core compatibility,
and valid server profile key custody.

**Success criteria:** Missing key configuration fails closed with a stable
blocker while existing Sync domains remain unaffected.

**Status:** Complete

## Stage 4: Verification and review

**Goal:** Complete targeted regressions, static/security gates, documentation,
and independent review.

**Success criteria:** Focused Sync tests, Ruff/format, compilation, Bandit,
diff hygiene, and independent review pass; TASK-13146 is complete.

**Status:** Complete

## ADR check

ADR required: no (existing)

ADR path: `backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md`

Reason: ADR-002 already governs server authority, the Personal Context Sync
contract, integrity, cleanup acknowledgments, and purge generations.
