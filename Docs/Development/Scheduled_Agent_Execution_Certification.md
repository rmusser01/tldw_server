# Scheduled Agent Execution Certification

This guide defines the operator and API contract for Scheduled Tasks Agent execution feasibility. It does not enable Agent execution.

## Current decision

ADR-041 records that no current deployment class is certified.

- Candidate runtimes with untrusted isolation and strict deny-all metadata are `draft_only` while required proof is incomplete.
- Host-local or otherwise ineligible runtimes are `unsupported`.
- Agent execution requires both `certified` evidence and the separately delivered execution stack.
- The production execution-stack readiness function remains source-defined as false.

The reviewed baseline is:

- JSON: `Docs/Evidence/Scheduled_Agent_Execution/2026-08-24-phase4d0f-baseline.json`
- Markdown: `Docs/Evidence/Scheduled_Agent_Execution/2026-08-24-phase4d0f-baseline.md`
- Evidence ID: `sha256:1df8024b73472ea0a02a323fbad0d2f864d8b5f604611cb01bf49478f60a5874`
- Deployment class: `sha256:76a1074c303c74cd6db3f6823f391133e44437a0da019f99f5b02b95b2cb3337`
- Outcome: `draft_only`

This baseline is repository characterization. It is not an attestation or execution grant.

## Outcome contract

| Outcome | Meaning | Agent definition behavior | Execution behavior |
| --- | --- | --- | --- |
| `certified` | Every domain passed with fresh, exact-subject, server-verified evidence covered by an authoritative bundle receipt | Available, subject to ordinary permissions | Still disabled until the independent execution stack is ready |
| `draft_only` | The runtime is a static candidate, but proof is missing, partial, stale, unverified, or receipt-less | Preview, create, edit, duplicate, pause, resume, archive, and inspect remain available | Run Now, scheduled dispatch, and worker execution are blocked |
| `unsupported` | The runtime is ineligible or a verified safety boundary failed | Family remains discoverable; new preview, definition, update-preview, and duplicate operations are unavailable; existing resources remain manageable | Blocked at API, scheduler, and worker |

## Deployment-class identity

Certification is bound to one canonical digest derived from:

- host OS family and architecture;
- AuthNZ mode;
- sandbox runtime;
- adapter ID and version;
- server build SHA;
- runtime image identity;
- mount, egress, credential, tenant-boundary, and mediation policy identities;
- isolation-profile version.

The capabilities API exposes only the deployment-class digest. It does not expose the raw host or policy tuple.

`TLDW_BUILD_SHA` is the only production source for server build identity. It must contain a 40- or 64-character hexadecimal digest. Missing or malformed values become `unverified` and prevent certification. The API process never invokes Git to infer this value.

## Required evidence

| Requirement | Passing proof |
| --- | --- |
| `isolation_attestation` | The server verifies a signed, fresh binding for tenant, workspace, runtime, image, mounts, egress, credentials, signer, and exact deployment class against a live trust root |
| `hostile_boundary` | Attested probes deny controlled host-file access, denied and public network access, subprocess bypass, direct MCP/tool access, inherited environment data, and ambient credentials |
| `scheduled_transcript_non_disclosure` | Prompt sentinels are absent from ordinary ACP detail, events, artifacts, fork, export, bootstrap, search, logs, errors, audit, and backup surfaces |
| `adapter_dispatch_recovery` | Adapter session creation is idempotent by stable dispatch token and process-loss recovery finds the exact attempt |
| `monotonic_execution_evidence` | Terminal, timeout, pre-action approval, effect, and cancellation evidence is durable and ordered per attempt |
| `brokered_credentials_and_mediation` | Scheduled identity and grants bind exact actions; credentials are issued per action with live revocation and no ambient channel |
| `operational_fail_closed` | Installation, upgrade invalidation, health, evidence freshness, outages, and policy changes fail closed for the exact deployment |

All seven requirements must pass. Partial primitives remain `missing`; they are not promoted to passing evidence.

## Trust levels

| Verification | Authority |
| --- | --- |
| `server_verified` | Eligible only when subject-bound, fresh, complete, and covered by the internal authoritative receipt |
| `host_gated_unverified` | Diagnostic evidence only |
| `repository_characterization` | Documents current code and metadata; cannot certify |
| `self_asserted` | Cannot certify |
| `mock` | Test-only; cannot certify |

The authoritative receipt type is internal to the server trust boundary. The API, helper JSON, and operators cannot construct it.

## Evidence artifacts

The schema is `scheduled-agent-execution-certification.v1`. A JSON manifest contains:

- `schema_version`, `evidence_id`, and `deployment_class_id`;
- source commit and validity timestamps;
- outcome and bounded reason codes;
- exactly seven sanitized requirement records;
- exactly seven value-free command templates.

Requirement records contain state, verification level, subject digest, timestamps, reason codes, evidence digest, and the safety-boundary flag. They do not contain probe output.

Committed artifacts must not contain prompt or transcript content, credentials, environment values, raw action arguments, local paths, hostnames, user-controlled names, or raw stdout/stderr. Host-gated raw artifacts remain outside the repository under operator-controlled retention. Only sanitized digests and bounded results may be committed.

Editing the JSON cannot certify a deployment. Canonical identity validation rejects changed content with a stale `evidence_id`, and a recomputed manifest still has no authoritative server receipt. Markdown must exactly render the paired JSON, including the typed runtime eligibility appendix.

## Generate a characterization baseline

Run from the repository root with the project virtual environment active. Replace every identity with the exact deployment value. Use `unverified` only when the value is genuinely unavailable; doing so prevents certification.

```bash
source .venv/bin/activate
BUILD_SHA="$(git rev-parse HEAD)"
python Helper_Scripts/Testing-related/scheduled_agent_execution_certification.py \
  --host-os darwin \
  --host-arch arm64 \
  --runtime docker \
  --auth-mode single_user \
  --adapter-id acp \
  --adapter-version 1 \
  --source-commit "$BUILD_SHA" \
  --server-build-sha "$BUILD_SHA" \
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

Without `--repository-characterization-only`, the helper rejects a claimed host OS or architecture that differs from the observed host. That override permits a non-host repository characterization; it does not create host evidence.

## Validate a pair

```bash
source .venv/bin/activate
python Helper_Scripts/Testing-related/scheduled_agent_execution_certification.py \
  --validate-artifacts \
  Docs/Evidence/Scheduled_Agent_Execution/2026-08-24-phase4d0f-baseline.json \
  Docs/Evidence/Scheduled_Agent_Execution/2026-08-24-phase4d0f-baseline.md
```

Validation checks the exact schema, canonical evidence identity, allowed reason vocabulary, seven-domain order, subject binding, non-certified current outcome, exact Markdown rendering, and runtime appendix derived from typed runtime metadata.

## Hostile-probe admission

`--run-hostile` is opt-in. It requires:

- a real, non-symlinked evidence directory;
- a loopback server URL;
- the name of an environment variable containing the API credential;
- a server-verified attestation reference for the exact deployment class.

TASK-13129 intentionally has no server-side attestation verifier. Even with all inputs present, admission exits nonzero with `hostile_probe_blocked_server_attestation_verifier_unimplemented`. It does not launch a subprocess, call the server, or write a passing artifact.

## API projection

`GET /api/v1/scheduled-tasks/capabilities` returns `execution_certification` only for the `agent_task` family:

```json
{
  "schema_version": "scheduled_task_execution_certification.v1",
  "outcome": "draft_only",
  "deployment_class_id": "sha256:<opaque>",
  "evidence_id": null,
  "evidence_source": "repository_characterization",
  "observed_at": "<timestamp>",
  "expires_at": "<timestamp>",
  "reason_codes": ["isolation_attestation_missing"],
  "recovery_action": "Complete server-verified Scheduled Agent execution certification for this deployment class."
}
```

Agent `execute` and `run_now` actions repeat bounded evidence source, freshness, reason, and recovery metadata. Clients must consume this server response. They must not infer capability from browser environment, deployment mode, visible controls, or local configuration.

Manual Run Now returns HTTP 409 with `scheduled_task_agent_execution_unavailable`. Unsupported creation and duplication return HTTP 409 with `scheduled_task_agent_automation_unsupported`. The error envelope includes the same stable readiness reason and recovery action shown by capability discovery.

## Freshness and rerun rules

Repository characterization uses a 24-hour validity window. Authoritative evidence must define and enforce its own reviewed validity window.

Rerun and review evidence after any change to:

- server build or adapter version;
- host OS or architecture;
- AuthNZ mode;
- sandbox runtime or image;
- mount, egress, credential, tenant, or mediation policy;
- trust root, signer, or revocation state;
- transcript storage, dispatch recovery, journal, credential, or governance implementation.

A subject change creates a new deployment-class digest. Expired, wrong-subject, partial, failed, unsigned, or receipt-less evidence cannot retain `certified`.

## Release gate

An execution-capable release requires all of the following:

1. Every exact deployment class has seven fresh `server_verified` passing records.
2. The server verifier issues a matching authoritative bundle receipt.
3. Hostile probes pass under the attested profile without leaking raw evidence.
4. TASK-13130 through TASK-13133 satisfy their domain and cross-cutting operational criteria.
5. The later Phase 4D execution slice independently changes stack readiness through reviewed source code and passes API, scheduler, worker, recovery, and security tests.

Until then, `draft_only` and `unsupported` are successful honest outcomes of the feasibility gate, and Agent execution remains unavailable.
