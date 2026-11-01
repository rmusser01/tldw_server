---
id: TASK-15411
title: Enforce sandbox admission, egress, cancellation, and collection contracts
status: Done
labels:
- sandbox
- security
- lifecycle
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the six verified enforcement and lifecycle findings before expanding Sandbox. Enforce VM-only untrusted admission; fail closed on unimplemented Lima egress; attach an owned Docker allowlist chain first; confirm termination before terminal status; bound collection and use effective per-run resource limits; gate real Firecracker on host containment. Preserve existing provider interfaces and defer speculative provider, snapshot, and warm-pool expansion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Admission and discovery agree on VM trust and implemented networking/containment.
- [x] #2 Docker allowlist attachment and controlled negative integration assertions fail closed.
- [x] #3 Cancellation retains active capacity until termination is confirmed and backend identity is durable.
- [x] #4 Output collection and Docker resource profiles enforce limits before unbounded consumption.
- [x] #5 Regression tests, touched-scope lint and Bandit, documentation, and review completed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR check: ADR required: no. Docs/ADR/010-sandbox-vz-runtime-ownership.md governs VZ ownership and separate network semantics. These changes repair existing documented contracts without selecting a new provider or persistence model.

The four-stage temporary plan Docs/superpowers/plans/2026-11-01-sandbox-enforcement-lifecycle.md was completed and removed per AGENTS.md.

Implemented findings A-F:
- Untrusted admission requires VM-grade isolation and eligibility; shared preflight discovery removes unsupported trust levels.
- Lima strict network denial/allowlisting is unsupported regardless of OS/readiness flags. Real Firecracker is blocked by firecracker_host_containment_not_implemented until containment is implemented, rather than accepting a binary/wrapper as proof.
- Docker uses owned IPv4 policy chains attached first to INPUT/FORWARD, read-back verification, exact cleanup, serialized refresh, and fail-closed IPv6/dedicated-network checks. Controlled endpoint integration tests have positive reachability controls and integer negative-exit assertions. Allowlisting authorizes IP destinations, not HTTP/TLS identities.
- Cancellation atomically fences queued dispatch or confirms backend termination. Persisted backend/host identifiers support same-host Docker/VZ recovery. Unconfirmed cleanup retains active state/capacity, even after dispatch lease expiry; killed-state fences reject late results. Host-local process groups require their owning worker and never signal a reused PID from durable metadata alone.
- Docker drains output concurrently in bounded chunks and bounds daemon logs; Lima bounds guest log storage and incremental reads. Docker PID/workspace/FD caps apply the stricter trust/admin profile; unquotable writable host bind mounts are rejected.

Verification:
- Whole portable Sandbox suite: TEST_MODE=1 MINIMAL_TEST_APP=1 ROUTES_ENABLE=sandbox python -m pytest -q --confcutdir=tldw_Server_API/tests --randomly-dont-reorganize tldw_Server_API/tests/sandbox -> 982 passed, 35 skipped, 34 warnings (238.77s). Preserve the parent tests/conftest.py lifecycle fixtures; narrowing confcutdir to sandbox causes shutdown_in_progress 503s between API tests.
- Latest process/cancellation/Docker confirmation checks: 47 passed. Earlier lifecycle lane 48 passed; VZ lane 64 passed. New regressions were observed failing before fixes.
- Adjacent Docs/ACP/MCP/startup-warning lane: 18 passed; final public docs contract rerun: 9 passed.
- Independent reviewers found process-group survivors, in-memory alias/fencing, VZ ambiguous exec/startup identity, Docker termination failures, and premature completion events; all actionable blockers fixed and rechecked (final three-path reviewer rerun: 3 passed).
- Touched Python Ruff and compilation passed; git diff --check and configured pre-commit hooks passed.
- Touched production Bandit: 89 existing LOW findings versus HEAD baseline 112; no increased per-file/rule counts, no errors/MEDIUM/HIGH. Reports /tmp/tldw15411-bandit-final.json and /tmp/tldw15411-bandit-baseline.json.

Limits: Prepared-host Docker/iptables, Lima, Firecracker, and VZ validation not available here; portable mocks do not prove hostile-runtime isolation. PostgreSQL fixture was attempted and reported unavailable (3 skips). Full-suite verification used deterministic file order; randomized store-module reload interference was not repaired. Provider migration, generic reconciliation, pause/resume, environment layering, and warm-pool expansion remain separate work.
PR preparation: isolated the sandbox commit on codex/sandbox-enforcement-pr-dev from origin/dev (6bf581da08), excluding unrelated inherited RPG/C4/chat commits. Syncing the published Sandbox API and Firecracker host-checklist mirrors using the repository docs generator before PR creation.
Clean dev-based PR branch verification (origin/dev 6bf581da08; isolated cherry-pick e3e7ed308f): 996 passed, 24 skipped, 34 warnings in 141.68s using the parent tests fixtures and deterministic file order. Isolated PostgreSQL ownership/capacity fixture tests passed here; DSN-gated PostgreSQL integrations still skipped. Production source matches the previously reviewed sandbox commit. Ruff/compile/hooks passed; Bandit remains 89 LOW, zero errors/MEDIUM/HIGH. Published API/checklist mirrors regenerated with Helper_Scripts/refresh_docs_published.sh; only the two related mirrors changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed Sandbox findings A-F with fail-closed trust/network/containment admission, verified owned Docker egress chains, confirmed cancellation and durable backend ownership, retained capacity on uncertainty, and bounded output/resource collection. Updated capability docs and regression coverage. Whole portable suite: 982 passed, 35 skipped; lint, compilation, pre-commit and review passed with no new Bandit findings. Real Lima strict networking and real Firecracker remain deliberately unavailable until concrete enforcement exists; prepared-host checks and broader provider expansion are deferred.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
