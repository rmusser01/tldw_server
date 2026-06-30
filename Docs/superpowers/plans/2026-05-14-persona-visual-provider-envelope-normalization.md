# Persona Visual Provider Envelope Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a contract-backed server helper that normalizes untrusted external Persona Visual provider result envelopes into bounded review metadata and fail-closed diagnostics.

**Architecture:** Keep the slice as a pure Python intake-boundary helper under `tldw_Server_API/app/core/Persona/visual_portability/`. The helper accepts raw provider result dictionaries, never retrieves MCP resources, never writes assets, never queues jobs, and returns a deterministic normalized envelope with `commit_eligible: false` plus machine-readable blockers when the provider output violates the contract.

**Tech Stack:** Python 3.11, pytest, existing Persona Visual docs/contracts, no database or external MCP runtime dependencies.

---

### Task 1: Add Failing Provider Envelope Tests

**Files:**
- Create: `tldw_Server_API/tests/Persona/test_persona_visual_provider_envelope.py`
- Reference: `Docs/Design/2026-05-13-persona-visual-external-mcp-provider-contract.md`

- [x] **Step 1: Write tests for valid and blocked provider envelopes**

Add tests for:
- a valid `portable_archive` envelope using `application/vnd.tldw.persona.visual-pack+zip`;
- a blocked `live2d` diagnostic envelope with structured blocker/warning objects.

- [x] **Step 2: Write tests for fail-closed contract violations**

Add tests for:
- `activation_allowed: true`;
- unknown `result_type`;
- stale archive MIME `application/vnd.tldw.persona-visual-pack`;
- string diagnostics instead of `{code, message}` objects;
- provenance/metadata containing secrets, absolute paths, or host-local identifiers.

- [x] **Step 3: Run tests to verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_provider_envelope.py -q
```

Expected: fails because the provider envelope module does not exist yet.

### Task 2: Implement Pure Envelope Normalization

**Files:**
- Create: `tldw_Server_API/app/core/Persona/visual_portability/provider_envelope.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_visual_provider_envelope.py`

- [x] **Step 1: Add a module docstring and constants**

Define:
- `PROVIDER_CONTRACT_VERSION = 1`
- allowed result types: `portable_archive`, `generated_candidate`, `manifest_patch`, `draft_pack_request`
- canonical archive media type: `application/vnd.tldw.persona.visual-pack+zip`
- compatible archive media type: `application/zip`

- [x] **Step 2: Implement `normalize_provider_result_envelope(raw: Mapping[str, Any]) -> dict[str, Any]`**

Return a dictionary with:
- `contract_version`
- `result_type`
- `review_required`
- `activation_allowed`
- `import_preview_required`
- bounded `provider`, `pack`, `diagnostics`, `provenance`, and `payload`
- `commit_eligible`
- `blockers`
- `warnings`

- [x] **Step 3: Enforce fail-closed safety**

Add blockers for:
- non-object envelope;
- unsupported contract version;
- unknown result type;
- missing or false `review_required`;
- any true `activation_allowed`;
- missing/false `import_preview_required` for `portable_archive`;
- stale or unsupported archive MIME type;
- malformed diagnostics;
- unsafe metadata/provenance/payload strings that look like secrets, tokens, absolute paths, path traversal, file URLs, or host-local identifiers.

- [x] **Step 4: Keep output bounded and deterministic**

Normalize strings with trimming and max lengths, copy only known high-level sections, and avoid echoing raw unsafe text into blocker messages.

- [x] **Step 5: Run tests to verify GREEN**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_provider_envelope.py -q
```

Expected: all provider envelope tests pass.

### Task 3: Update Tracking Notes And Verify

**Files:**
- Modify: `backlog/tasks/task-340 - Normalize-Persona-Visual-external-provider-result-envelopes.md`

- [x] **Step 1: Update TASK-340 implementation notes**

Record that this is a non-persistent intake-boundary foundation only, with no provider execution, no MCP resource retrieval, no import-preview enqueueing, no asset writes, no runtime activation, and no Persona Garden UI.

- [x] **Step 2: Run focused validation**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_provider_envelope.py -q
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile tldw_Server_API/app/core/Persona/visual_portability/provider_envelope.py tldw_Server_API/tests/Persona/test_persona_visual_provider_envelope.py
git diff --check
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Persona/visual_portability/provider_envelope.py -f json -o /tmp/bandit_persona_visual_provider_envelope.json
```

Expected: tests pass, compile passes, diff check passes, Bandit reports no findings.

- [ ] **Step 3: Commit and open draft PR**

Commit the plan, task, helper, and tests. Open a draft PR against `dev` linked to #1689. Keep the PR body explicit that the change is review-only and does not execute providers or activate visual packs.
