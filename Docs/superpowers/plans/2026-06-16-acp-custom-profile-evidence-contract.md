# ACP Custom Profile Evidence Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Define the minimum evidence contract for certifying concrete custom ACP profiles while keeping the seeded `custom` profile template-only.

**Architecture:** Add a machine-readable `evidence_requirements` block to custom-template certification manifests, then align registry wording and docs to the same contract. Do not change runtime launch semantics or certify any new profile without live evidence.

**Tech Stack:** Python smoke manifest helper and pytest coverage; Markdown docs; Go runner wording/tests where the setup guidance mirrors Python.

---

## Stage 1: Manifest Contract
**Goal:** Expose the custom concrete-profile evidence contract in the agent-profile manifest.
**Success Criteria:** `--agent-profile custom --format json` includes `evidence_requirements`; the seeded template still emits no runnable commands.
**Tests:** `tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py`.
**Status:** Complete

- [x] Write failing pytest assertions for custom `evidence_requirements`.
- [x] Implement the manifest field and markdown rendering.
- [x] Run the focused helper tests.

## Stage 2: Registry And Runner Wording
**Goal:** Make setup/status language distinguish the seeded `custom` template from a named certifiable custom profile.
**Success Criteria:** Python and Go blocked-status text tells operators to create a distinct named profile and never implies generic support.
**Tests:** Python registry tests and Go runner tests.
**Status:** Complete

- [x] Update failing tests for the revised custom-template wording.
- [x] Update Python registry, seeded YAML notes, and Go runner wording.
- [x] Run focused Python and Go tests.

## Stage 3: Documentation And Tracker Closeout
**Goal:** Document the contract in the compatibility matrix and checklist, then update local Backlog and GitHub trackers.
**Success Criteria:** Matrix and checklist list required custom-profile evidence fields, redaction policy, and live-result requirements; issue #2052 can close after merge while #1563 remains open.
**Tests:** Markdown inspection plus `git diff --check`.
**Status:** Complete

- [x] Update ACP compatibility and certification docs.
- [x] Run final verification, including Bandit on touched Python scope.
- [x] Push PR and update #2052/#1563 appropriately.
