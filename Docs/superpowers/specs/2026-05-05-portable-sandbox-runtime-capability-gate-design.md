# Portable Sandbox Runtime Capability Gate Design

**Date:** 2026-05-05
**Status:** Approved working design for TASK-72
**Scope:** Sandbox runtime discovery, metadata contracts, docs alignment, and portable tests

## Goal

Add a portable gate that prevents sandbox runtime capability drift as runtimes evolve. The gate must run on normal developer and CI hosts without Docker, Lima, Firecracker, Apple Virtualization.framework, or `sandbox-exec` availability.

## Design

The gate uses the existing runtime seams rather than adding a new runtime abstraction:

- `RuntimeType` remains the source of truth for runtime names.
- `runtime_capabilities.py` remains the source of truth for implementation state, isolation metadata, network policy metadata, session semantics, and normalized preflight reasons.
- `SandboxService.feature_discovery()` remains the API-facing projection contract.
- `Docs/Sandbox/sandbox-runtime-capability-inventory.md` remains the operator-facing inventory.

The test should inject synthetic `RuntimePreflightResult` objects for every runtime through the service seam, then validate that each discovery row can be parsed by the public schema and contains the required capability fields. This avoids host probing while still exercising the API projection.

## Contract Rules

For every `RuntimeType`:

- implementation state must be non-empty and in the supported vocabulary
- isolation, network policy, and session contract metadata must exist
- discovery output must include raw reasons and normalized reasons
- discovery output must validate against `SandboxRuntimesResponse`
- the runtime must be documented in the capability inventory

For run status taxonomy:

- current runtime-specific policy failure messages must normalize to `policy_failed`
- current runtime-specific unavailable messages that are actually emitted must normalize to `runtime_unavailable`

## Non-Goals

- Do not require real runtime binaries or VM hosts.
- Do not change runtime execution behavior.
- Do not generalize VZ repair ownership beyond `vz_linux`.
- Do not make host-local runtimes appear VM-grade.

## Risks And Mitigations

- Risk: a broad gate becomes brittle and blocks unrelated work.
  Mitigation: assert stable contract fields and emitted reason vocabularies only, not host availability.
- Risk: docs parsing becomes too strict.
  Mitigation: check runtime row presence and named gate documentation, not exact table formatting beyond runtime names.
- Risk: status taxonomy invents aliases for messages no runner emits.
  Mitigation: cover emitted policy/unavailable messages from current service/runner paths only.

## Verification

Expected focused verification:

- `python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py -q`
- `python -m pytest tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py -q`
- `python -m py_compile` for touched Python files
- `python -m ruff check` for touched Python files
- Bandit on touched production Python only, or document skip for test/docs-only changes
- `git diff --check`
