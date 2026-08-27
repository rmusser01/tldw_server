# Scheduled Tasks Phase 4D.0F Execution Feasibility Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for each implementation unit, superpowers:systematic-debugging for unexpected probe failures, and superpowers:verification-before-completion before every commit. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement TASK-13129 as an API-first, fail-closed feasibility gate that produces reproducible evidence for the approved seven security/operational domains, publishes an ADR, and prevents Agent automation execution from being advertised or dispatched before both deployment certification and the later execution stack are ready.

**Architecture:** Add a small pure certification domain under Scheduled Tasks, a separate evidence harness that characterizes existing Sandbox/ACP/MCP behavior without granting authority, and an additive versioned capability projection on the existing `/api/v1/scheduled-tasks/capabilities` endpoint. Use one shared fail-closed readiness decision at capability projection, Run Now admission, scheduler arming, and worker admission. The first run is expected to reject certification honestly: statically eligible runtimes remain `draft_only` while required cross-system proof is missing, and statically ineligible runtimes are `unsupported`. Even a certified test fixture only clears the feasibility prerequisite; this slice has no Phase 4D execution stack and therefore never advertises or dispatches Agent automation. Later dependency work may satisfy the same evidence contract; it must not weaken the evaluator.

**Tech Stack:** Python 3.10+, dataclasses and Pydantic v2, FastAPI, existing Sandbox runtime metadata/preflights, ACP session and sandbox adapters, MCP credential broker, pytest, Loguru, JSON/Markdown evidence artifacts, Bandit.

**Spec:** `Docs/superpowers/specs/2026-08-24-scheduled-tasks-phase4d-agent-task-execution-design.md`

**Backlog task:** `TASK-13129`

**Pre-created dependency tasks:** `TASK-13130` (isolation/hostile proof), `TASK-13131` (secure transcripts), `TASK-13132` (dispatch/evidence), `TASK-13133` (identity/credentials/mediation)

## Global Constraints

- This task proves or rejects feasibility. It does not implement Phase 4D revisions, grants, secure payloads, scheduled-mode transcripts, new adapter dispatch, execution, approvals, migration activation, or frontend controls. It does add admission gates that stop the existing generic automation feed and Run Now route from queuing `agent_task` work prematurely.
- No production resolver or artifact ingestion path can turn mocked, unit-only, self-asserted, unsigned, stale, wrong-subject, or partial evidence into `certified`; pure rule tests may construct an internal authoritative receipt only to exercise the closed outcome table.
- A deployment class is the canonical tuple of host OS family, host architecture, AuthNZ mode, sandbox runtime, adapter ID/version, server build SHA, and an isolation-profile fingerprint computed from runtime/image, mount, egress, credential-broker, tenant-boundary, and policy versions. The API exposes only its stable digest, not raw host details.
- The API resolver obtains server build identity only from strict `TLDW_BUILD_SHA` configuration. It never invokes Git or a subprocess on a request. A missing or malformed build SHA is normalized to an unverified identity and cannot certify.
- Certification applies only to the exact tuple and evidence validity window. A runtime, image, adapter, network, auth mode, policy, signer, or build change produces a different class or stale evidence.
- `certified` requires every required domain to be `passed`, `server_verified`, subject-bound, unexpired, and covered by an authoritative bundle receipt from the server trust boundary. Missing, partial, unverified, stale, or receipt-less evidence produces `draft_only` for a statically eligible runtime.
- A statically ineligible runtime or a demonstrated isolation/bypass boundary failure produces `unsupported`. A missing feature is not mislabeled as a proven boundary compromise.
- Current expected baseline is conservative: `docker`, `firecracker`, `lima`, and `vz_linux` can be evaluated as candidates because runtime metadata marks them untrusted-eligible, but they remain `draft_only` until all seven requirements pass. `vz_macos`, `seatbelt`, and `worktree` are `unsupported` because current runtime metadata does not make them untrusted-eligible and/or does not strictly enforce deny-all networking.
- The evidence harness must refuse hostile execution until the selected runtime has a server-verified attestation path. A refusal is recorded as missing proof, never converted into a pass.
- Evidence artifacts contain reason codes, bounded summaries, hashes, versions, timestamps, and command results. They never contain prompts, transcript content, credentials, environment values, raw tool arguments or argument values, local absolute paths, hostnames, user-controlled names, or raw stdout/stderr. Command manifests expose only a fixed invocation template and parameter names.
- Existing ordinary ACP transcripts remain explicitly unsuitable: `record_prompt()` stores raw prompt content and normal detail/fork/bootstrap paths can return it. This task records that dependency; it does not repurpose ordinary ACP storage for scheduled prompts.
- Existing Sandbox idempotency does not satisfy adapter recovery while `ACPSandboxRunnerManager.create_session()` passes `idem_key=None` and its control record lacks a dispatch token. This task records that dependency; it does not claim generic Sandbox idempotency is sufficient.
- Existing MCP managed credential brokering is useful prior art but does not satisfy scheduled execution identity, delegation/grant binding, per-action live revocation, or removal of ambient `session_env`. This task records the partial capability without certifying it.
- The capability endpoint remains the server authority. The WebUI and extension are untouched and must later consume the API outcome rather than infer support from deployment mode or local configuration.
- Preview/create-definition discovery may remain available for `draft_only`. `unsupported` keeps the family visible but unavailable with a stable reason and recovery action; direct Agent preview-create, definition-create, and duplicate admission return the same typed refusal while existing definitions remain listable, pausable, archivable, and inspectable.
- Certification is necessary but not sufficient for execution. In this slice, `execute` and `run_now` for the `agent_task` family are always `disabled`: non-certified deployments report the certification blocker, while a synthetic certified fixture reports `agent_execution_stack_unimplemented`. No state maps to `available` until the later 4D.1B execution stack adds an independently tested readiness input.
- Manual Run Now, scheduler arming/fire, and worker admission apply the same conjunction. They reject or skip `agent_task` before adapter execution in this slice, including a synthetically certified deployment, while recurring-question execution remains unchanged.
- Preserve standalone Agent Tasks. Its ordinary interactive ACP lifecycle and UX are a separate job/persona and are not gated or renamed by this work.
- ADR-040 is occupied by synchronized moodboards on the implementation baseline. ADR-041 is reserved for this feasibility decision; recheck `Docs/ADR/README.md` immediately before creating the ADR and advance again if current `dev` claims it first.

## Evidence Domains And Current Gaps

| Requirement ID | Passing proof required | Current reusable evidence | Current blocking gap |
| --- | --- | --- | --- |
| `isolation_attestation` | Server verifies signed tenant/workspace/runtime/image/mount/egress/credential/signer/expiry bindings against a live trust root | Runtime isolation/network metadata, policy hashes, image digests, VZ host evidence reader | No scheduled-execution attestation verifier, trust-root/signer revocation check, or exact subject binding |
| `hostile_boundary` | Attested runtime blocks host file access, uncontrolled network, subprocess bypass, direct MCP/tool access, inherited secrets, and ambient credentials | Docker hardening/egress tests and host-gated VZ/Lima tests | No scheduled-agent hostile suite under the exact isolation profile; launch must remain blocked before attestation |
| `scheduled_transcript_non_disclosure` | Prompt sentinel absent from ordinary ACP detail, fork, export, bootstrap, search, logs, errors, and audit | ACP redacted support views and retention tests | No scheduled transcript mode; ordinary ACP persists raw prompts and copies messages when forking |
| `adapter_dispatch_recovery` | Idempotent adapter session creation and exact lookup by stable dispatch token after process loss | Sandbox session/run idempotency and durable ACP sandbox control records | ACP creates session/run with no idempotency key and persists no dispatch token |
| `monotonic_execution_evidence` | Durable ordered terminal, timeout, pre-action approval, effect, and cancellation events tied to dispatch token | ACP cancel API, Sandbox run status, permission decisions, audit helpers | Cancellation is a notification/action without the required per-attempt monotonic evidence journal or race ordering |
| `brokered_credentials_and_mediation` | Scheduled subject/grant binding, no ambient credentials, per-action credential issue/revocation, and credible pre-action mediation | MCP managed external credential broker and ACP governance hooks | Broker is MCP-policy scoped, ACP accepts/merges session env, and no Scheduled Tasks grant/attestation/action-token binding exists |
| `operational_fail_closed` | Install, upgrade, health, evidence freshness, and fail-closed behavior for the exact deployment class | Runtime preflights, ACP health, Sandbox operator status, VZ evidence ingestion | No unified scheduled-execution certification state, upgrade invalidation, or capability gate |

## Delivery Stages

| Stage | Goal | Success Criteria | Tests | Status |
| --- | --- | --- | --- | --- |
| 1 | Freeze the certification vocabulary and fail-closed evaluator | All outcome and freshness rules are deterministic and mock evidence cannot certify | Pure unit tests | Complete |
| 2 | Produce reproducible current-state evidence | Seven requirement records and a sanitized manifest are generated for an exact deployment class | Helper and characterization tests | In Progress |
| 3 | Publish and enforce the API-first readiness gate | Versioned capability metadata, typed Run Now refusal, no Agent scheduler enqueue, and worker defense-in-depth agree | Schema/service/API/OpenAPI/feed/consumer tests | Not Started |
| 4 | Publish the decision and current baseline | ADR, operator guide, JSON/Markdown evidence, and dependency tasks agree | Artifact validator and docs checks | Not Started |
| 5 | Complete regression and security gates | Focused cross-module tests, compile, lint, Bandit, diff review, and Backlog evidence pass | Verification matrix | Not Started |

## File Map

**Create**

- `tldw_Server_API/app/core/Scheduled_Tasks/execution_certification.py` - deployment-class identity, evidence records, reason codes, outcome evaluator, and current fail-closed resolver.
- `tldw_Server_API/tests/Notifications/test_scheduled_task_execution_certification.py` - pure outcome, subject-binding, freshness, and runtime-eligibility tests in the existing Scheduled Tasks test area.
- `Helper_Scripts/Testing-related/scheduled_agent_execution_certification.py` - sanitized manifest/evidence generator and explicit host-gated refusal/run controls.
- `tldw_Server_API/tests/Helper_Scripts/test_scheduled_agent_execution_certification.py` - helper CLI, evidence schema, sentinel exclusion, and no-false-certification tests.
- `Docs/ADR/041-scheduled-agent-execution-feasibility.md` - accepted/rejected deployment-class decision and consequences.
- `Docs/Development/Scheduled_Agent_Execution_Certification.md` - operator commands, trust/evidence contract, outcome interpretation, and rerun rules.
- `Docs/Evidence/Scheduled_Agent_Execution/2026-08-24-phase4d0f-baseline.json` - machine-readable current repository result.
- `Docs/Evidence/Scheduled_Agent_Execution/2026-08-24-phase4d0f-baseline.md` - bounded human-readable result and gaps.

**Modify**

- `tldw_Server_API/app/api/v1/schemas/scheduled_tasks_automation_schemas.py` - additive typed certification capability.
- `Docs/ADR/README.md` - register ADR-041 in the canonical ADR index.
- `tldw_Server_API/app/services/scheduled_task_automation_service.py` - inject/resolve certification and gate Agent capability projection and Run Now admission.
- `tldw_Server_API/app/services/scheduled_task_automation_scheduler.py` - prevent Agent definitions from arming or enqueueing before the complete execution stack is ready.
- `tldw_Server_API/app/core/Scheduled_Tasks/agent_task_jobs.py` - worker-side defense in depth for already-queued Agent Jobs.
- `tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py` - stable typed HTTP refusal mapping for unavailable Agent execution.
- `tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py` - versioned API contract, Run Now refusal, and action-gating regressions.
- `tldw_Server_API/tests/Notifications/test_automation_definition_feed.py` - recurring-question preservation and Agent arming/fire refusal.
- `tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py` - already-queued Agent Job refusal before executor dispatch.
- `backlog/tasks/task-13129 - Implement-Scheduled-Tasks-Phase-4D.0F-execution-feasibility-gate.md` - progress, evidence, dependency tasks, and final decision.
- `backlog/tasks/task-13130 - Add-scheduled-execution-isolation-attestation-and-hostile-runtime-proof.md` - attach ADR/evidence and the exact isolation requirement result.
- `backlog/tasks/task-13131 - Add-ACP-scheduled-mode-secure-transcripts-and-leakage-gates.md` - attach ADR/evidence and the exact transcript requirement result.
- `backlog/tasks/task-13132 - Add-ACP-dispatch-recovery-and-monotonic-execution-evidence.md` - attach ADR/evidence and the exact recovery/evidence requirement results.
- `backlog/tasks/task-13133 - Add-scheduled-execution-identity-credentials-and-pre-action-mediation.md` - attach ADR/evidence and the exact credential/mediation requirement result.

**Read-only evidence sources**

- `tldw_Server_API/app/core/Sandbox/runtime_capabilities.py`
- `tldw_Server_API/app/core/Sandbox/operator_evidence.py`
- `tldw_Server_API/app/core/Sandbox/operator_status.py`
- `tldw_Server_API/app/core/Agent_Client_Protocol/sandbox_bridge.py`
- `tldw_Server_API/app/core/Agent_Client_Protocol/sandbox_runner_client.py`
- `tldw_Server_API/app/core/DB_Management/ACP_Sessions_DB.py`
- `tldw_Server_API/app/services/admin_acp_sessions_service.py`
- `tldw_Server_API/app/services/mcp_credential_broker_service.py`
- `Helper_Scripts/Testing-related/acp_certification_smoke.py`

### Task 0: Rebase, Reserve The ADR, And Record Baseline Evidence Sources

**Files:**
- Modify: `backlog/tasks/task-13129 - Implement-Scheduled-Tasks-Phase-4D.0F-execution-feasibility-gate.md`

**Interfaces:**
- Consumes: completed TASK-13127, current `origin/dev`, the approved Phase 4D spec, and existing Sandbox/ACP/MCP evidence contracts.
- Produces: one clean implementation worktree and an explicit record of every evidence source and known limitation.

- [x] **Step 1: Start only after TASK-13127 is merged**

```bash
git fetch origin dev
git worktree add .worktrees/scheduled-tasks-phase4d-feasibility -b codex/scheduled-tasks-phase4d-feasibility origin/dev
cd .worktrees/scheduled-tasks-phase4d-feasibility
git status --short --branch
git log -1 --format='%H %s'
backlog task TASK-13127 --plain
```

Expected: clean worktree on current dev and TASK-13127 Done. If TASK-13127 is not merged, stop; do not combine the prerequisite and security gate.

- [x] **Step 2: Reserve the next free ADR after finding ADR-040 occupied**

```bash
ls Docs/ADR
rg -n '^# ADR-04[01]:' Docs/ADR
```

Observed: ADR-040 is occupied by synchronized moodboards. ADR-041 is the next free number and is reserved in this plan and TASK-13129 before any ADR file is created.

- [x] **Step 3: Mark TASK-13129 In Progress and record the evidence inventory**

Use Backlog.md MCP to record the dev SHA and the exact reusable sources from the evidence table. Record these baseline facts: ordinary ACP stores raw prompts; ACP sandbox creation has no dispatch token/idempotency key; current cancellation lacks the required event journal; MCP credential brokering is partial; no deployment class is currently certified.

- [x] **Step 4: Run the focused pre-change baseline**

```bash
source "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv/bin/activate"
python -m pytest -q --tb=short \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_sandbox_bridge.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py \
  tldw_Server_API/tests/sandbox/test_runtime_capabilities_policy.py \
  tldw_Server_API/tests/sandbox/test_operator_evidence.py \
  tldw_Server_API/tests/MCP_Hub/test_mcp_slot_status.py
```

Observed on baseline SHA `2306c1939f3b460f9c62da8ae83a1aa47c02ee0d`: 171 passed, 0 failed, 0 skipped, and 19 warnings in 78.28 seconds. These tests establish reusable primitives; they do not count as Phase 4D.0F certification.

### Task 1: Define Deployment Identity And Fail-Closed Outcome Rules

**Files:**
- Create: `tldw_Server_API/app/core/Scheduled_Tasks/execution_certification.py`
- Test: `tldw_Server_API/tests/Notifications/test_scheduled_task_execution_certification.py`

**Interfaces:**
- Consumes: a normalized `DeploymentClass`, seven `RequirementEvidence` records, runtime isolation/network metadata, and an injected UTC clock.
- Produces: immutable `ExecutionCertification` with outcome, opaque deployment-class ID, optional evidence ID, validity timestamps, and bounded stable reason codes.

- [x] **Step 1: Write failing type and identity tests**

Cover the exact public vocabulary:

```python
CertificationOutcome = Literal["certified", "draft_only", "unsupported"]
EvidenceState = Literal["passed", "failed", "missing", "stale"]
EvidenceVerification = Literal[
    "server_verified", "host_gated_unverified", "repository_characterization",
    "mock", "self_asserted",
]
```

Define these seven closed requirement IDs in the order shown in the evidence table. Test that `DeploymentClass.canonical_payload()` sorts fields and that `deployment_class_id` is `sha256:` plus 64 lowercase hex characters. Changing any identity field must change the digest; insertion order must not.

- [x] **Step 2: Write failing evaluator tests**

Test all rules independently:

- seven fresh, exact-subject, `passed` records accompanied by an authoritative verification receipt produce `certified` in the pure rule test;
- one missing, stale, wrong-subject, wrong-build, self-asserted, host-gated-unverified, mock, or expired record produces `draft_only` for an eligible runtime;
- raw JSON, CLI values, and records that merely label themselves `server_verified` cannot construct an authoritative receipt or produce `certified` through the production resolver;
- unknown requirement IDs and duplicate requirement IDs are rejected, not ignored;
- no `evidence_id`, missing validity, or malformed timestamp can certify;
- a runtime with `untrusted_eligible=False` or non-strict `deny_all` produces `unsupported` before evidence is considered;
- a record marked `safety_boundary_breached=True` produces `unsupported` and a stable boundary reason;
- ordinary missing dependencies, including the absent scheduled transcript mode, produce `draft_only`, not a false claim that the runtime itself was breached;
- reason codes are sorted, deduplicated, bounded, and contain no free text.

- [x] **Step 3: Run RED**

```bash
source "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv/bin/activate"
python -m pytest -q --tb=short \
  tldw_Server_API/tests/Notifications/test_scheduled_task_execution_certification.py
```

Expected: FAIL during collection because `execution_certification.py` does not exist.

- [x] **Step 4: Implement the minimal pure domain**

Use frozen dataclasses and pure helpers. `evaluate_execution_certification(subject, evidence, verification_receipt, *, now)` must never read environment, files, databases, or network. Validate all requirement IDs, subject digests, timestamps, verification levels, evidence IDs, and the already-verified receipt's bundle digest before applying the outcome rules. The receipt type is an internal trust-boundary input, is never accepted from the API or evidence JSON, and has no production constructor in this task. Unit tests may construct it only through a test helper to prove the outcome table; that does not create a production certification path.

Add `resolve_current_agent_execution_certification()` as a separate fail-closed projection. It may read the configured ACP sandbox runtime, platform/AuthNZ identity, and existing static Sandbox metadata, but it must supply only repository-characterization evidence for the known gaps and no authoritative receipt. Therefore it cannot return `certified` in this task. It returns `unsupported` for current ineligible runtimes and `draft_only` with bounded codes for every missing evidence domain plus any build/profile identity problem for eligible runtimes.

Add a separate pure `agent_execution_dispatch_readiness(certification, *, execution_stack_ready)` decision. It returns ready only when certification is `certified` **and** the later stack input is true. The production call sites in this task always pass a source-defined false constant; no environment variable, config file, API parameter, or evidence artifact can change it. Tests prove a certified fixture remains blocked with `agent_execution_stack_unimplemented`.

- [x] **Step 5: Run GREEN and commit the domain unit**

```bash
source "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv/bin/activate"
python -m pytest -q --tb=short \
  tldw_Server_API/tests/Notifications/test_scheduled_task_execution_certification.py
git add \
  tldw_Server_API/app/core/Scheduled_Tasks/execution_certification.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_execution_certification.py \
  'backlog/tasks/task-13129 - Implement-Scheduled-Tasks-Phase-4D.0F-execution-feasibility-gate.md'
git diff --cached --check
git commit -m "feat(scheduled-tasks): add execution certification gate"
```

Expected: pure tests pass and the first commit cannot activate execution.

### Task 2: Build The Reproducible Evidence Harness

**Files:**
- Create: `Helper_Scripts/Testing-related/scheduled_agent_execution_certification.py`
- Test: `tldw_Server_API/tests/Helper_Scripts/test_scheduled_agent_execution_certification.py`

**Interfaces:**
- Consumes: explicit `--host-os`, `--host-arch`, `--runtime`, `--auth-mode`, `--adapter-id`, `--adapter-version`, `--source-commit`, `--server-build-sha`, `--image-digest`, `--mount-policy-hash`, `--egress-policy-hash`, `--credential-policy-hash`, `--tenant-boundary-policy-hash`, `--mediation-policy-hash`, `--isolation-profile-version`, output format/destination, and optional host-gated run flag.
- Produces: schema `scheduled-agent-execution-certification.v1`, one exact deployment-class digest, seven sanitized requirement records, the derived outcome, and a command manifest.

- [ ] **Step 1: Write failing manifest and sanitization tests**

The default invocation emits a manifest and repository characterization without launching a sandbox. Test exact top-level keys:

```text
schema_version
evidence_id
deployment_class_id
source_commit
created_at
valid_until
outcome
reason_codes
requirements
commands
```

Each requirement contains only `requirement_id`, `state`, `verification`, `subject_id`, `observed_at`, `valid_until`, `reason_codes`, `evidence_sha256`, and `safety_boundary_breached`. Each command contains `id`, `description`, `invocation_template`, `parameter_names`, `safe_to_run_by_default`, and `required_environment_names`. `invocation_template` contains the repository-relative helper name and fixed option names only; it never contains option values, a URL, an environment value, or a local path.

Seed tests with prompt, credential, path, hostname, tool-argument, and environment sentinels and assert none appears anywhere in JSON or Markdown output. Test that absolute paths and all argument values are removed before serialization. Recompute `evidence_id` from canonical sanitized content and reject a mismatched digest during artifact validation.

- [ ] **Step 2: Define all seven evidence commands**

Use these stable command IDs:

1. `isolation_attestation`
2. `hostile_boundary`
3. `scheduled_transcript_non_disclosure`
4. `adapter_dispatch_recovery`
5. `monotonic_execution_evidence`
6. `brokered_credentials_and_mediation`
7. `operational_fail_closed`

The repository-characterization implementations must inspect behavior through imports, temporary databases, typed runtime metadata, and public method signatures. Do not parse source files with regex. Record the exact current gaps from the evidence table.

For transcript characterization, write a random sentinel to a temporary ordinary ACP session through `record_prompt()`, confirm the ordinary store can retrieve it and fork copies it, and record only `sha256(sentinel)` plus `scheduled_transcript_mode_unimplemented`; never serialize the sentinel.

For adapter recovery, verify generic Sandbox session/run idempotency exists but ACP sandbox `create_session()` has no dispatch-token parameter and its durable control record has no dispatch token. For monotonic evidence, verify current cancel/terminal primitives exist but no per-attempt ordered journal contract is available. For credentials, verify the managed external broker exists while current ACP session env remains a possible ambient channel. Partial primitives remain `missing`, not `passed`.

- [ ] **Step 3: Make hostile execution opt-in and fail closed**

`--run-hostile` requires `--evidence-dir`, a local server URL, an API key environment-variable name, and a pre-existing server-verified attestation reference for the exact deployment class. Missing any prerequisite exits non-zero before launch and records `hostile_probe_blocked_by_missing_attestation` in the in-memory result only; it must not write a misleading pass artifact.

The generated hostile test vector must enumerate attempts against a controlled host-file sentinel, a controlled denied network listener, public egress, subprocess launch, direct MCP/tool access, inherited environment sentinel, and ambient credential sentinel. The harness records pass only when every attempt is denied and the server verifies the exact attestation. In this task's baseline, the hostile command is expected to refuse because the attestation dependency is absent.

- [ ] **Step 4: Run RED, implement, then run GREEN**

```bash
source "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv/bin/activate"
python -m pytest -q --tb=short \
  tldw_Server_API/tests/Helper_Scripts/test_scheduled_agent_execution_certification.py
```

Expected RED: collection fails because the helper is absent. Expected GREEN after implementation: all helper tests pass; eligible current profiles emit `draft_only`; ineligible profiles emit `unsupported`; no fixture can emit `certified` without seven server-verified records.

- [ ] **Step 5: Verify the CLI behavior**

```bash
source "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv/bin/activate"
python Helper_Scripts/Testing-related/scheduled_agent_execution_certification.py \
  --host-os darwin \
  --host-arch arm64 \
  --runtime docker \
  --auth-mode single_user \
  --adapter-id acp \
  --adapter-version 1 \
  --source-commit "$(git rev-parse HEAD)" \
  --image-digest unverified \
  --mount-policy-hash unverified \
  --egress-policy-hash unverified \
  --credential-policy-hash unverified \
  --tenant-boundary-policy-hash unverified \
  --mediation-policy-hash unverified \
  --isolation-profile-version phase4d0f-baseline \
  --server-build-sha "$(git rev-parse HEAD)" \
  --format json
python Helper_Scripts/Testing-related/scheduled_agent_execution_certification.py \
  --host-os darwin \
  --host-arch arm64 \
  --runtime worktree \
  --auth-mode single_user \
  --adapter-id acp \
  --adapter-version 1 \
  --source-commit "$(git rev-parse HEAD)" \
  --image-digest unverified \
  --mount-policy-hash unverified \
  --egress-policy-hash unverified \
  --credential-policy-hash unverified \
  --tenant-boundary-policy-hash unverified \
  --mediation-policy-hash unverified \
  --isolation-profile-version phase4d0f-baseline \
  --server-build-sha "$(git rev-parse HEAD)" \
  --format markdown
```

Expected: Docker reports `draft_only`; worktree reports `unsupported`; neither output contains an absolute local path, raw prompt, credential value, or raw environment value.

- [ ] **Step 6: Commit the evidence harness**

```bash
git add \
  Helper_Scripts/Testing-related/scheduled_agent_execution_certification.py \
  tldw_Server_API/tests/Helper_Scripts/test_scheduled_agent_execution_certification.py \
  'backlog/tasks/task-13129 - Implement-Scheduled-Tasks-Phase-4D.0F-execution-feasibility-gate.md'
git diff --cached --check
git commit -m "test(scheduled-tasks): add execution feasibility evidence harness"
```

Expected: one commit containing the reusable sanitized harness and its tests, without generated evidence or a capability claim.

### Task 3: Add And Enforce The API-First Readiness Gate

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/scheduled_tasks_automation_schemas.py`
- Modify: `tldw_Server_API/app/services/scheduled_task_automation_service.py`
- Modify: `tldw_Server_API/app/services/scheduled_task_automation_scheduler.py`
- Modify: `tldw_Server_API/app/core/Scheduled_Tasks/agent_task_jobs.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py`
- Modify: `tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py`
- Modify: `tldw_Server_API/tests/Notifications/test_automation_definition_feed.py`
- Modify: `tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py`

**Interfaces:**
- Consumes: `ExecutionCertification` from an injectable resolver and an independently injected `execution_stack_ready` input that is false in this task.
- Produces: additive versioned `execution_certification` metadata, honest execution-action status, a typed manual refusal, no Agent scheduler enqueue, and worker-side refusal of already-queued Agent Jobs.

- [ ] **Step 1: Write failing schema and service tests**

Add `ScheduledTaskExecutionCertificationCapability` with exactly:

```python
schema_version: Literal["scheduled_task_execution_certification.v1"]
outcome: Literal["certified", "draft_only", "unsupported"]
deployment_class_id: str
evidence_id: str | None
evidence_source: Literal[
    "server_verified", "repository_characterization", "none",
]
observed_at: datetime | None
expires_at: datetime | None
reason_codes: list[str]
recovery_action: str | None
```

Add `execution_certification: ScheduledTaskExecutionCertificationCapability | None = None` to `ScheduledTaskAutomationCapability` and bump its additive `schema_version` default from `2026-06-09` to `2026-08-24`. Add optional `evidence_source`, `recovery_action`, `observed_at`, and `expires_at` fields to `ScheduledTaskActionCapability`; populate them for Agent `execute`/`run_now` and leave existing families/actions backward-compatible with `None`.

Test the default current resolver: the `agent_task` item includes a non-certified outcome, `execute.status == "disabled"`, `run_now.status == "disabled"`, and both reasons use a bounded stable code with recovery/freshness metadata. Preview/create/update/pause/archive/duplicate remain available for `draft_only` according to their existing contract. For `unsupported`, the Agent family remains visible with `family_availability="unavailable"`, a stable reason/recovery action, and no enabled creation or execution action. The recurring-question item has `execution_certification is None` and unchanged actions.

Inject a complete synthetic `certified` domain object directly into `ScheduledTaskAutomationService` and assert `execute` and `run_now` remain disabled with `agent_execution_stack_unimplemented`. This proves certification clears only the feasibility prerequisite and cannot falsely advertise an execution stack that this task does not implement.

Assert the generated OpenAPI document exposes `execution_certification` on `ScheduledTaskAutomationCapability`, references `ScheduledTaskExecutionCertificationCapability`, and retains the existing `/api/v1/scheduled-tasks/capabilities` response route without adding an execution route.

Add API tests that Run Now on an `agent_task` returns HTTP 409 with code `scheduled_task_agent_execution_unavailable`, `details.reason` set to the stable readiness reason, and no Job or audit event. For an injected `unsupported` outcome, Agent preview-create, definition-create using a previously valid preview, and duplicate return HTTP 409 with `scheduled_task_agent_automation_unsupported` and create no new resource; existing list/detail/pause/archive remain usable. The existing Recurring Question Run Now and creation tests remain successful and preserve their Job payload/idempotency behavior.

Add scheduler tests that an `agent_task` definition is not armed during load/reconcile/rescan, an already-armed race is refused again in `_fire()`, and no Job is created. Recurring-question definitions continue to arm and enqueue normally. Add a consumer test that an already-queued `agent_task` Job creates no adapter call even when a test executor is registered; it completes with a typed skipped result and a valid blocked run/audit record when the definition exists.

- [ ] **Step 2: Run RED**

```bash
source "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv/bin/activate"
python -m pytest -q --tb=short \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py \
  tldw_Server_API/tests/Notifications/test_automation_definition_feed.py \
  tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py \
  -k 'capabilities or agent_execution'
```

Expected: FAIL because the response lacks `execution_certification`, currently advertises Agent `run_now` as available, Run Now enqueues an Agent Job, and the scheduler has no Agent readiness gate.

- [ ] **Step 3: Implement the projection and gate**

Add optional certification and execution-stack readiness resolvers to `ScheduledTaskAutomationService.__init__`; default them to `resolve_current_agent_execution_certification` and a source-defined function that returns false. The certification resolver must be side-effect-free and must not perform host probes on an API request. The production stack-readiness function must not read environment or configuration; only a later reviewed code slice may replace it. Reuse the same pure readiness helper in the service, scheduler, and consumer instead of duplicating outcome logic.

`resolve_current_agent_execution_certification()` reads `TLDW_BUILD_SHA` as the only server build identity source. Accept only a lowercase or uppercase 40- or 64-character hexadecimal digest and normalize it to lowercase. Missing or malformed input adds `server_build_identity_unverified` and prevents certification; never run Git, inspect the repository, or derive identity from a mutable version string inside the API process.

Map `draft_only` and `unsupported` to disabled execution actions with stable reasons `execution_certification_draft_only` or `execution_certification_unsupported`. Map `certified` plus the current false stack input to disabled with `agent_execution_stack_unimplemented`. Run Now raises the same typed reason before idempotency or Job creation. For `unsupported`, preview-create, definition-create, and duplicate raise `agent_automation_unsupported` before persistence; retain ordinary management of existing definitions. The scheduler refuses Agent arming and rechecks at fire time. The consumer rechecks after the TASK-13127 owner-scoped definition preflight and records a skipped blocked run without calling an executor. Do not change recurring-question scheduling/execution or draft-only definition mutation.

- [ ] **Step 4: Run GREEN and API regressions**

```bash
source "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv/bin/activate"
python -m pytest -q --tb=short \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py \
  tldw_Server_API/tests/Notifications/test_scheduled_tasks_control_plane.py \
  tldw_Server_API/tests/Notifications/test_automation_definition_feed.py \
  tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py
```

Expected: PASS. Capability discovery remains additive, owner-independent, and protected by `TASKS_READ`; direct API, scheduler, and stale-Job paths all fail closed for Agent execution while Recurring Questions are unchanged.

- [ ] **Step 5: Commit the API gate**

```bash
git add \
  tldw_Server_API/app/api/v1/schemas/scheduled_tasks_automation_schemas.py \
  tldw_Server_API/app/services/scheduled_task_automation_service.py \
  tldw_Server_API/app/services/scheduled_task_automation_scheduler.py \
  tldw_Server_API/app/core/Scheduled_Tasks/agent_task_jobs.py \
  tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py \
  tldw_Server_API/tests/Notifications/test_automation_definition_feed.py \
  tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py \
  'backlog/tasks/task-13129 - Implement-Scheduled-Tasks-Phase-4D.0F-execution-feasibility-gate.md'
git diff --cached --check
git commit -m "feat(scheduled-tasks): expose agent execution feasibility"
```

Expected: one API-first commit with no frontend, Watchlists, standalone Agent Tasks, or Agent adapter implementation; its only runtime behavior is fail-closed admission.

### Task 4: Generate The Baseline, ADR, Operator Guide, And Dependencies

**Files:**
- Create: `Docs/ADR/041-scheduled-agent-execution-feasibility.md`
- Modify: `Docs/ADR/README.md`
- Create: `Docs/Development/Scheduled_Agent_Execution_Certification.md`
- Create: `Docs/Evidence/Scheduled_Agent_Execution/2026-08-24-phase4d0f-baseline.json`
- Create: `Docs/Evidence/Scheduled_Agent_Execution/2026-08-24-phase4d0f-baseline.md`
- Modify: `backlog/tasks/task-13129 - Implement-Scheduled-Tasks-Phase-4D.0F-execution-feasibility-gate.md`

**Interfaces:**
- Consumes: the helper at the two preceding commits and current runtime metadata.
- Produces: a reproducible reviewed decision that no deployment class is certified yet, plus bounded follow-on dependency tasks.

- [ ] **Step 1: Generate the exact current baseline artifacts**

Run the helper for the implementation host's observed OS/architecture/AuthNZ/runtime/adapter tuple and use the current implementation commit SHA as both `source_commit` and the baseline build identity. The values below are examples for the current worktree host and must be replaced if observation differs; the helper must refuse a claimed host OS/architecture that disagrees with its local observation unless `--repository-characterization-only` is explicit. The JSON artifact is evidence for that one exact, unverified deployment class. The Markdown artifact renders the same record and adds a clearly labeled repository-static eligibility appendix for all runtime values; neither the appendix nor a repository-characterization baseline is deployment certification evidence.

```bash
source "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv/bin/activate"
python Helper_Scripts/Testing-related/scheduled_agent_execution_certification.py \
  --host-os darwin \
  --host-arch arm64 \
  --runtime docker \
  --auth-mode single_user \
  --adapter-id acp \
  --adapter-version 1 \
  --source-commit "$(git rev-parse HEAD)" \
  --server-build-sha "$(git rev-parse HEAD)" \
  --image-digest unverified \
  --mount-policy-hash unverified \
  --egress-policy-hash unverified \
  --credential-policy-hash unverified \
  --tenant-boundary-policy-hash unverified \
  --mediation-policy-hash unverified \
  --isolation-profile-version phase4d0f-baseline \
  --output-json Docs/Evidence/Scheduled_Agent_Execution/2026-08-24-phase4d0f-baseline.json \
  --output-markdown Docs/Evidence/Scheduled_Agent_Execution/2026-08-24-phase4d0f-baseline.md
```

Expected: both files are written atomically after schema validation; the exact deployment record is `draft_only`; unverified image/policy fields are stable reason codes rather than passing evidence.

The static appendix applies these expected default outcomes when all non-runtime identity fields are explicit but the approved cross-system evidence remains missing:

| Runtime metadata | Default outcome | Primary reason |
| --- | --- | --- |
| `docker` | `draft_only` | Required scheduled attestation/transcript/dispatch/evidence/credential proof is incomplete |
| `firecracker` | `draft_only` | Statically eligible but host-gated/scaffold proof is incomplete |
| `lima` | `draft_only` | Statically eligible but host-gated proof and required cross-system contracts are incomplete |
| `vz_linux` | `draft_only` | Statically eligible but existing host evidence is not the complete scheduled-execution proof |
| `vz_macos` | `unsupported` | Not currently untrusted-eligible and deny-all is not strictly supported |
| `seatbelt` | `unsupported` | Host-local boundary and no strict deny-all network support |
| `worktree` | `unsupported` | Host-local boundary and no strict deny-all network support |

The generator must validate that the JSON record and corresponding Markdown deployment-class section have identical outcome/reason codes before writing. It must also validate the static appendix directly from `runtime_capabilities.py`. Any unexpected `certified` result is a release-blocking failure.

- [ ] **Step 2: Write ADR-041**

Use the repository ADR template and add ADR-041 to `Docs/ADR/README.md` with its final status and one-sentence decision. The decision is:

> No current deployment class is certified for Scheduled Tasks Agent automation execution. Statically eligible isolation runtimes remain draft-only until all seven server-verified evidence domains pass for an exact deployment class; host-local and current non-eligible runtimes are unsupported. Existing ordinary ACP, Sandbox, and MCP primitives are retained as dependencies but are not treated as proof.

Alternatives rejected must include: trusting Docker/container isolation alone; reusing ordinary ACP transcripts; treating Sandbox idempotency as adapter dispatch recovery; treating generic MCP credentials/governance as scheduled grants; and hiding Agent automation entirely instead of retaining an explicit draft-only API state.

- [ ] **Step 3: Write the operator guide**

Document the exact CLI invocations, evidence schema, seven requirement definitions, trust levels, freshness/subject invalidation, safe refusal behavior, artifact redaction, API projection, and rerun procedure. State that changing evidence JSON manually cannot certify a deployment. Document that host-gated raw artifacts remain outside the repository and only sanitized digests/results are committed.

- [ ] **Step 4: Attach confirmed evidence to the pre-created dependency tasks**

Use Backlog.md MCP to add ADR-041 and both baseline artifacts to these existing tasks:

1. `TASK-13130`: `isolation_attestation` and `hostile_boundary`.
2. `TASK-13131`: `scheduled_transcript_non_disclosure`.
3. `TASK-13132`: `adapter_dispatch_recovery` and `monotonic_execution_evidence`.
4. `TASK-13133`: `brokered_credentials_and_mediation`.

Record `operational_fail_closed` as a cross-cutting exit criterion on TASK-13129 and all four dependency tasks rather than creating a fifth implementation silo. Do not begin any dependency task in this branch.

- [ ] **Step 5: Validate documentation and artifact consistency**

```bash
source "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv/bin/activate"
python Helper_Scripts/Testing-related/scheduled_agent_execution_certification.py \
  --validate-artifacts \
  Docs/Evidence/Scheduled_Agent_Execution/2026-08-24-phase4d0f-baseline.json \
  Docs/Evidence/Scheduled_Agent_Execution/2026-08-24-phase4d0f-baseline.md
! rg -n '[T]ODO|[T]BD|[F]IXME|certified.*true' \
  Docs/ADR/041-scheduled-agent-execution-feasibility.md \
  Docs/ADR/README.md \
  Docs/Development/Scheduled_Agent_Execution_Certification.md \
  Docs/Evidence/Scheduled_Agent_Execution/2026-08-24-phase4d0f-baseline.*
! rg -n 'raw_prompt|api_key|access_token' \
  Docs/Evidence/Scheduled_Agent_Execution/2026-08-24-phase4d0f-baseline.*
```

Expected: artifact validation passes; both negated scans exit 0 because no match exists; no document claims a certified current deployment. Review operator-guide prose separately because it intentionally names prohibited data categories without including values.

- [ ] **Step 6: Commit the evidence decision**

```bash
git add \
  Docs/ADR/041-scheduled-agent-execution-feasibility.md \
  Docs/ADR/README.md \
  Docs/Development/Scheduled_Agent_Execution_Certification.md \
  Docs/Evidence/Scheduled_Agent_Execution/2026-08-24-phase4d0f-baseline.json \
  Docs/Evidence/Scheduled_Agent_Execution/2026-08-24-phase4d0f-baseline.md \
  'backlog/tasks/task-13129 - Implement-Scheduled-Tasks-Phase-4D.0F-execution-feasibility-gate.md' \
  'backlog/tasks/task-13130 - Add-scheduled-execution-isolation-attestation-and-hostile-runtime-proof.md' \
  'backlog/tasks/task-13131 - Add-ACP-scheduled-mode-secure-transcripts-and-leakage-gates.md' \
  'backlog/tasks/task-13132 - Add-ACP-dispatch-recovery-and-monotonic-execution-evidence.md' \
  'backlog/tasks/task-13133 - Add-scheduled-execution-identity-credentials-and-pre-action-mediation.md'
git diff --cached --check
git commit -m "docs(scheduled-tasks): publish phase 4d feasibility decision"
```

Expected: only the ADR and index, guide, validated evidence, TASK-13129, and the four exact dependency records are staged.

### Task 5: Cross-Module Verification, Security Review, And Completion

**Files:**
- Verify all touched files.
- Modify: `backlog/tasks/task-13129 - Implement-Scheduled-Tasks-Phase-4D.0F-execution-feasibility-gate.md`

**Interfaces:**
- Consumes: all four implementation units and generated evidence.
- Produces: a reviewed, test-backed gate with no execution capability enabled.

- [ ] **Step 1: Run focused and adjacent tests**

```bash
source "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv/bin/activate"
python -m pytest -q --tb=short \
  tldw_Server_API/tests/Notifications/test_scheduled_task_execution_certification.py \
  tldw_Server_API/tests/Helper_Scripts/test_scheduled_agent_execution_certification.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py \
  tldw_Server_API/tests/Notifications/test_scheduled_tasks_control_plane.py \
  tldw_Server_API/tests/Notifications/test_automation_definition_feed.py \
  tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_sandbox_bridge.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py \
  tldw_Server_API/tests/sandbox/test_runtime_capabilities_policy.py \
  tldw_Server_API/tests/sandbox/test_operator_evidence.py \
  tldw_Server_API/tests/MCP_Hub/test_mcp_slot_status.py
```

Expected: PASS except explicitly recorded host-gated skips. A skipped host probe cannot count as passing evidence.

- [ ] **Step 2: Run syntax, lint, and Bandit gates**

```bash
source "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv/bin/activate"
python -m compileall -q \
  tldw_Server_API/app/core/Scheduled_Tasks/execution_certification.py \
  Helper_Scripts/Testing-related/scheduled_agent_execution_certification.py
python -m ruff check \
  tldw_Server_API/app/core/Scheduled_Tasks/execution_certification.py \
  tldw_Server_API/app/api/v1/schemas/scheduled_tasks_automation_schemas.py \
  tldw_Server_API/app/services/scheduled_task_automation_service.py \
  tldw_Server_API/app/services/scheduled_task_automation_scheduler.py \
  tldw_Server_API/app/core/Scheduled_Tasks/agent_task_jobs.py \
  tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py \
  Helper_Scripts/Testing-related/scheduled_agent_execution_certification.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_execution_certification.py \
  tldw_Server_API/tests/Helper_Scripts/test_scheduled_agent_execution_certification.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py \
  tldw_Server_API/tests/Notifications/test_automation_definition_feed.py \
  tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py
python -m bandit -r \
  tldw_Server_API/app/core/Scheduled_Tasks/execution_certification.py \
  tldw_Server_API/app/services/scheduled_task_automation_service.py \
  tldw_Server_API/app/services/scheduled_task_automation_scheduler.py \
  tldw_Server_API/app/core/Scheduled_Tasks/agent_task_jobs.py \
  tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py \
  Helper_Scripts/Testing-related/scheduled_agent_execution_certification.py \
  -f json -o /tmp/bandit_task_13129.json
```

Expected: compile and lint exit 0; Bandit reports no new findings in touched Python.

- [ ] **Step 3: Perform the security self-review**

Verify all of the following:

- no external file, environment flag, API input, raw evidence record, or mocked record can switch current production resolution to `certified`;
- wrong subject/build/runtime/network/auth/adapter evidence fails closed;
- stale evidence cannot remain certified;
- hostile probes cannot launch before attestation verification;
- evidence output excludes raw content and local paths;
- a synthetic `certified` fixture still cannot advertise or dispatch Agent execution while `execution_stack_ready` is false;
- Agent automation execute/run-now advertising, Run Now admission, scheduler arming/fire, and worker admission are mutually consistent and disabled;
- recurring questions, Watchlists, and standalone Agent Tasks are unchanged;
- there is no frontend inference or execution implementation in the diff.

- [ ] **Step 4: Finalize TASK-13129 and amend the evidence commit**

Use Backlog.md MCP to record exact test counts, host-gated skips, artifact evidence IDs, Bandit result, API/admission behavior, ADR decision, and TASK-13130 through TASK-13133. Check acceptance criteria and definition-of-done only after verification. Mark Done when the gate and documentation are complete even though the outcome is `draft_only`/`unsupported`; the task's objective is an honest feasibility decision, not forced certification.

```bash
git add 'backlog/tasks/task-13129 - Implement-Scheduled-Tasks-Phase-4D.0F-execution-feasibility-gate.md'
git diff --cached --check
git commit --amend --no-edit
```

Expected: the final Backlog evidence is included in the existing evidence/ADR commit, retaining four implementation commits.

- [ ] **Step 5: Review the complete branch diff after finalization**

```bash
git diff --check origin/dev...HEAD
git diff --stat origin/dev...HEAD
git diff --name-status origin/dev...HEAD
git log --oneline origin/dev..HEAD
```

Expected: only the complete file map, TASK-13129, and the four pre-created dependency task records appear; the worktree is clean. There are four clear commits: domain/evaluator, evidence harness, capability/admission gate, and evidence/ADR.

## Completion Review And Follow-On Gate

- Passing TASK-13129 does not authorize Agent automation execution. It establishes the machine-readable gate and admission boundary, then records why current execution is unavailable. Certification removes one prerequisite only; 4D.1B must independently set execution-stack readiness after its complete vertical slice passes.
- 4D.0E schema-epoch work may proceed after this decision because it is execution-independent. Revision-dependent executable work may proceed only for a deployment class whose later evidence changes this evaluator's result to `certified` through reviewed server-verified proof.
- Draft-only work must retain disabled `execute`/`run_now`, no canonical migration activation, and no implied background execution.
- The next implementation plan after TASK-13129 should be 4D.0E unless the user explicitly prioritizes one of the four certification dependencies first.

## Approved-Spec Coverage Matrix

| Approved requirement | Plan coverage | Later implementation boundary |
| --- | --- | --- |
| TASK-13127 missing-definition crash | Separate Phase 4D.0 prerequisite plan Tasks 1-4 | None after TASK-13127 passes |
| Server-verified isolation attestation | Feasibility Tasks 1, 2, and 4 | TASK-13130 implements the missing verifier/trust-root path |
| Hostile agent boundary attempts | Feasibility Task 2 manifest/refusal and Task 4 evidence | TASK-13130 implements and runs attested hostile probes |
| Scheduled transcript sentinel exclusion | Feasibility Task 2 ordinary-ACP characterization and Task 4 evidence | TASK-13131 implements scheduled-mode protected storage and all leakage gates |
| Idempotent adapter session and dispatch lookup | Feasibility Task 2 recovery characterization | TASK-13132 implements dispatch-token idempotency and exact lookup |
| Terminal, timeout, approval, effect, and cancellation ordering | Feasibility Task 2 evidence characterization | TASK-13132 implements the monotonic adapter journal |
| Brokered identity/credentials and pre-action mediation | Feasibility Task 2 credential characterization | TASK-13133 implements grants, issuance, revocation, and mediation |
| Installation, upgrade, health, and fail-closed operation | Feasibility Tasks 1-5 and capability API gate | Cross-cutting exit criterion on TASK-13130 through TASK-13133 |
| Explicit `certified`/`draft_only`/`unsupported` outcomes | Feasibility Tasks 1, 3, and 4 | Future evidence may promote only through the same evaluator |
| API-first product contract | Feasibility Task 3 versions the existing capabilities API and enforces the same decision at Run Now, scheduler, and worker admission | Full execution APIs remain Phase 4D.1A/4D.1B |
| No migration activation without certification | Global constraints, Task 3 gate, ADR, completion review | 4D.M1 remains dry-run; 4D.M2 remains blocked |
| Preserve Watchlists and standalone Agent Tasks | Global constraints and Task 5 regression review | Explicit non-regression remains required in every later slice |
