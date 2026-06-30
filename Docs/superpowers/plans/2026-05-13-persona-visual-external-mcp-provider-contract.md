# Persona Visual External MCP Provider Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Define the external MCP-compatible Persona Visual pack-provider contract without adding runtime renderer activation, provider execution, or shared-library behavior.

**Architecture:** Keep the current server as the trust boundary. External providers may describe proposed packs, generated candidates, manifest patches, or portable archives, but every durable result must enter the existing Persona Visual review flow before commit and must remain inactive until explicit user activation.

**Tech Stack:** Markdown product/design documentation, existing Persona Visual import-preview and renderer capability vocabulary, Backlog task tracking, docs-only verification.

---

## Stage 1: Contract Design Document
**Goal**: Add a durable design document for external MCP-compatible Persona Visual pack providers.
**Success Criteria**: The document defines provider discovery, provider result envelopes, allowed outputs, blocked diagnostics, safety rules, and the import-preview/review handoff.
**Tests**: `git diff --check` and targeted text scans for forbidden runtime/activation claims.
**Status**: Complete

### Task 1: Write the Provider Contract
**Files:**
- Create: `Docs/Design/2026-05-13-persona-visual-external-mcp-provider-contract.md`

- [x] Define provider-facing goals and non-goals.
- [x] Define provider discovery and pack-offer response shapes.
- [x] Add valid, blocked, and review-handoff examples.
- [x] State that providers cannot activate packs, submit runtime code, write assets directly, or bypass import preview.

## Stage 2: Product And Code Documentation Alignment
**Goal**: Update existing Persona Visual documentation so external MCP providers are no longer only an undefined future item.
**Success Criteria**: PRD and code docs point to the new contract and preserve the reference-backed, user-owned, review-first model.
**Tests**: Targeted `rg` checks for reference-backed/no-snapshot language and provider boundaries.
**Status**: Complete

### Task 2: Update Existing Docs
**Files:**
- Modify: `Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md`
- Modify: `Docs/Code_Documentation/Persona_Visual_Packs.md`

- [x] Add the external provider contract to the Phase 3 implementation snapshot and follow-up list.
- [x] Add provider-specific product requirements and MCP boundary notes.
- [x] Add code documentation for provider outputs and review-first import-preview handoff.
- [x] Ensure docs do not imply Live2D, runtime adapter, marketplace, or silent activation support.

## Stage 3: Tracking And Verification
**Goal**: Close the docs-only slice cleanly with task notes and deterministic verification.
**Success Criteria**: Backlog task records verification and acceptance criteria; docs-only skip decisions are explicit.
**Tests**: `git diff --check`, targeted `rg` scans, and no Bandit run because no Python code is touched.
**Status**: Complete

### Task 3: Verify And Record Results
**Files:**
- Modify: `backlog/tasks/task-335 - Define-external-MCP-Persona-Visual-pack-provider-contract.md`

- [x] Run `git diff --check`.
- [x] Run targeted `rg` checks for provider boundaries, no automatic activation, no runtime plugin behavior, and no snapshot reintroduction.
- [x] Update TASK-335 acceptance criteria and final summary.
