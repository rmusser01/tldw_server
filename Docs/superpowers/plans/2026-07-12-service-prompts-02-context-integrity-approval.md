# Service Prompt Context Integrity Approval Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the signed-manifest mutation, anti-rollback, and request-time verification infrastructure that plan 4 uses to review and approve exact per-user service-prompt revisions without creating a second trust mechanism.

**Architecture:** Represent trust with two owner-scoped `db_prompt` assets: an immutable revision asset keyed by definition and revision UUID, plus a stable definition-state asset that binds either the active revision identity/digest or a no-override reset state. It deliberately excludes the general optimistic-concurrency generation so saving a pending draft cannot invalidate the prior active override. Add a locked, compare-and-swap manifest store over the operator-configured manifest path, active signer, verification key ring, and anti-rollback anchor provider. Approval revalidates the live pending revision and trusted baseline, atomically adds the revision entry and replaces the state entry in one signed manifest, then compare-and-swaps per-user state; retry safely completes either side of this two-store protocol. Rejection records an immutable state event but does not change trusted runtime state.

**Tech Stack:** Existing Context Integrity canonicalization/manifest modules, HMAC-SHA256 with externally supplied keys, OS keyring anti-rollback anchoring, atomic filesystem replacement/locking, pytest/Hypothesis.

---

Before every commit below, satisfy the umbrella plan's mandatory per-commit gate.

## Task 1: Inventory service-prompt revisions as context assets

**Files:**

- Modify: `tldw_Server_API/app/core/Context_Integrity/models.py`
- Modify: `tldw_Server_API/app/core/Context_Integrity/inventory.py`
- Create: `tldw_Server_API/app/core/Context_Integrity/service_prompt_assets.py`
- Test: `tldw_Server_API/tests/Context_Integrity/unit/test_service_prompt_assets.py`

- [ ] Create the implementation Backlog task and link this plan before edits.
- [ ] Write failing tests for stable asset identity, Unicode/LF canonicalization, owner/tenant separation, definition/revision/contract binding, editable-part ordering, baseline digest binding, and omission of prompt text from descriptors.
- [ ] Represent the source through the existing `db_prompt` asset type; do not add a parallel approval-only source enum.
- [ ] Define immutable revision IDs as `db_prompt:service_prompt_revision:<tenant>:<user>:<definition>:<revision_uuid>`; their digest binds editable parts, contract version, registry schema digest, locked assembly digest, and trusted server-default digest.
- [ ] Define stable state IDs as `db_prompt:service_prompt_state:<tenant>:<user>:<definition>`; their digest binds state kind (`active_override` or `no_override`), active revision UUID/digest when present, and the stored reset-baseline digest when no override is active. It does not bind pending pointer, acknowledgement, catalog generation, or the general state generation.
- [ ] Add an adapter over immutable `ServicePromptRevisionAssetInput` data supplied by a caller; it has no repository import and never scans arbitrary user databases. Plan 4 maps its repository row into this input.
- [ ] Run `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Context_Integrity/unit/test_service_prompt_assets.py`.
- [ ] Commit: `feat: inventory pending service prompts for integrity review (<task-id>)`.

## Task 2: Add a mutable signed-manifest store with anti-rollback CAS

**Files:**

- Modify: `tldw_Server_API/app/core/Context_Integrity/manifest.py`
- Create: `tldw_Server_API/app/core/Context_Integrity/manifest_store.py`
- Create: `tldw_Server_API/app/core/Context_Integrity/manifest_lock.py`
- Create: `tldw_Server_API/app/core/Context_Integrity/anchor_store.py`
- Test: `tldw_Server_API/tests/Context_Integrity/unit/test_manifest_store.py`
- Test: `tldw_Server_API/tests/Context_Integrity/unit/test_manifest_lock.py`
- Test: `tldw_Server_API/tests/Context_Integrity/unit/test_anchor_store.py`
- Test: `tldw_Server_API/tests/Context_Integrity/unit/test_manifest.py`

- [ ] Write failing tests for load, atomic multi-entry insert/replace, monotonic sequence, expected-sequence conflict, duplicate assets, invalid signatures, rollback, concurrent writers/processes, lock timeout, interrupted writes, permission errors, and symlink/path substitution.
- [ ] Add verification-key-ring support keyed by `key_id`; signing uses only the configured active key while verification accepts retained prior keys. Keep the existing single-signer API as a compatibility wrapper.
- [ ] Define `ContextIntegrityManifestStore` around injected manifest path, active signer, verifier key ring, and `AntiRollbackAnchorProvider`. Implement the first provider as `KeyringAntiRollbackAnchorProvider` using the existing `keyring` dependency, service name `tldw.context-integrity`, and an account derived from the canonical manifest-path digest.
- [ ] Implement a small stdlib-only lock-file helper using `fcntl.flock` on POSIX and `msvcrt.locking` on Windows with a bounded timeout and fixed lock path beside the manifest; add no locking dependency.
- [ ] Under that inter-process lock: open the configured manifest without following a substituted symlink, verify signature and external anchor, compare expected sequence/digest, apply one validated batch of asset insert/replace/removal operations, sign sequence+1, fsync a same-directory temporary file, atomically replace, advance the external anchor, and reload/verify.
- [ ] If the anchor update fails after manifest replacement, return an indeterminate result and keep context use blocked until reconciliation verifies which signed version the external anchor accepts.
- [ ] Reject plaintext/file keyring backends and keyring read-after-write mismatches. Never synthesize a local key/anchor or silently downgrade. Missing signer, writable manifest, verifier key, or secure anti-rollback provider makes approval unavailable.
- [ ] Rerun `python -m pytest -q tldw_Server_API/tests/Context_Integrity/unit/test_manifest.py tldw_Server_API/tests/Context_Integrity/unit/test_manifest_lock.py tldw_Server_API/tests/Context_Integrity/unit/test_anchor_store.py tldw_Server_API/tests/Context_Integrity/unit/test_manifest_store.py`.
- [ ] Commit: `feat: advance context integrity manifests safely (<task-id>)`.

## Task 3: Wire active and retained verification keys

**Files:**

- Modify: `tldw_Server_API/app/services/startup_context_integrity.py`
- Create: `tldw_Server_API/app/api/v1/API_Deps/context_integrity_deps.py`
- Test: `tldw_Server_API/tests/Services/test_startup_context_integrity.py`
- Test: `tldw_Server_API/tests/Context_Integrity/unit/test_context_integrity_deps.py`
- Modify: `tldw_Server_API/Config_Files/config.txt`

- [ ] Write failing tests for current single-key compatibility, active key selection, old-key verification, unknown key rejection, malformed key-ring config, missing anchor, and secret-free logs.
- [ ] Keep `CONTEXT_INTEGRITY_HMAC_SECRET` and `CONTEXT_INTEGRITY_HMAC_KEY_ID` as the active-key inputs. Add `CONTEXT_INTEGRITY_HMAC_VERIFY_KEYS_JSON` with exact schema `{"prior-key-id":"prior-hmac-secret"}`: a JSON object of 0–16 unique nonblank string key IDs to nonblank string secrets; reject arrays, nested values, duplicate-after-normalization IDs, the active key ID, malformed JSON, and unknown fields. Values remain environment-only and are never returned/logged.
- [ ] Require operators to retain prior HMAC verification secrets for at least the greater of 30 days or the configured maximum Jobs retention; report a capability warning when configured keys cannot verify still-retained artifacts.
- [ ] Put the verified manifest store/key ring/anchor state on `app.state` and expose narrow dependencies. Provide an atomically replaceable latest-`VerifiedManifest` snapshot so request-time DB asset verification observes approved manifest advances without restart. Endpoint modules never read secrets or paths directly.
- [ ] Preserve existing read-only boot verification behavior when mutation support is unconfigured; service-prompt approval reports unavailable rather than weakening boot checks.
- [ ] Rerun focused startup/dependency tests and commit: `feat: wire rotatable context integrity trust keys (<task-id>)`.

## Task 4: Security verification

- [ ] Run `python -m pytest -q tldw_Server_API/tests/Context_Integrity tldw_Server_API/tests/Services/test_startup_context_integrity.py`.
- [ ] Run `python -m bandit -r tldw_Server_API/app/core/Context_Integrity tldw_Server_API/app/services/startup_context_integrity.py tldw_Server_API/app/api/v1/API_Deps/context_integrity_deps.py -f json -o /tmp/bandit_service_prompt_integrity_foundation.json` and review the JSON.
- [ ] Run `git diff --check` and inspect logs/tests for prompt text, secrets, signatures, or MAC values.
- [ ] Update the Backlog task with key-rotation/anchor assumptions, evidence, and final summary.
- [ ] Commit: `test: verify service prompt manifest infrastructure (<task-id>)`.
