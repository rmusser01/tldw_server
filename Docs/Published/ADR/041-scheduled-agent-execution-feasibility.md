# ADR-041: Scheduled Agent execution feasibility

**Status:** Accepted
**Date:** 2026-08-26
**Backfilled from:** not backfilled
**Decision owner:** tldw_server maintainers
**Related task:** TASK-13129
**Related spec/plan:** `Docs/superpowers/specs/2026-08-24-scheduled-tasks-phase4d-agent-task-execution-design.md`, `Docs/superpowers/plans/2026-08-24-scheduled-tasks-phase4d-feasibility-gate-implementation-plan.md`

## Decision

No current deployment class is certified for Scheduled Tasks Agent automation execution. Statically eligible isolation runtimes remain `draft_only` until all seven server-verified evidence domains pass for an exact deployment class; host-local and current non-eligible runtimes are `unsupported`. Existing ordinary ACP, Sandbox, and MCP primitives remain dependencies, but they are not proof.

Certification is necessary but not sufficient. The existing capabilities API, manual Run Now admission, scheduler arming and firing, and worker admission all require both a `certified` deployment result and a separately implemented execution stack. The production stack-readiness function is source-defined as false in TASK-13129.

## Context

Scheduled Agent execution combines delayed authority, untrusted model output, credentials, tools, durable recovery, and cross-process cancellation. Static runtime metadata or successful interactive-agent behavior cannot demonstrate those properties. A credible decision must bind evidence to one opaque deployment class containing host, architecture, AuthNZ mode, runtime, adapter version, server build, and isolation-policy identities.

The current repository has reusable pieces, but each required proof remains incomplete:

- ordinary ACP records and forks can retain prompt content;
- generic Sandbox idempotency is not an ACP dispatch-token recovery contract;
- cancellation and terminal state lack one ordered per-attempt evidence journal;
- MCP credential brokering and governance are not Scheduled Tasks grants or per-action mediation;
- no authoritative verifier signs and validates the exact seven-domain evidence bundle.

The baseline at `Docs/Evidence/Scheduled_Agent_Execution/2026-08-24-phase4d0f-baseline.json` therefore records `draft_only` for the observed Docker candidate class. Its corresponding Markdown artifact includes repository-static eligibility for all supported runtime values. Neither artifact grants execution authority.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Trust Docker or container isolation alone | Container selection does not prove exact mounts, deny-all egress, tenant binding, credential handling, mediation, recovery, or operational fail-closed behavior. |
| Reuse ordinary ACP transcripts for scheduled prompts | Ordinary storage and fork behavior can retain prompt content and do not provide the required scheduled-mode non-disclosure boundary. |
| Treat Sandbox idempotency as ACP dispatch recovery | Generic session/run idempotency is not bound to a stable adapter dispatch token and cannot recover the exact ACP attempt after process loss. |
| Treat generic MCP credentials and governance as scheduled grants | Existing policy is not bound to the Scheduled Tasks subject, definition revision, credential version, exact action arguments, or live per-action revocation. |
| Hide Agent automation until execution is complete | Hiding the family removes API discoverability and prevents clients from distinguishing draft-only work from a deployment that is structurally unsupported. |
| Allow an operator flag or edited evidence file to enable execution | Mutable local inputs would bypass the server trust boundary and make capability claims unverifiable. |

## Consequences

- `/api/v1/scheduled-tasks/capabilities` exposes a sanitized, versioned `execution_certification` object for `agent_task`.
- `execute` and `run_now` remain disabled for every production state in this slice.
- `draft_only` permits Agent definition drafting and ordinary management but not execution.
- `unsupported` keeps the family visible while refusing preview creation, definition creation, and duplication; existing definitions remain inspectable, pausable, resumable, and archivable.
- Manual Run Now returns a typed conflict before idempotency, Job creation, or audit creation.
- The scheduler refuses Agent arming, firing, and rearming. The worker independently records already-queued Agent Jobs as blocked skips before executor lookup.
- Recurring Questions, Watchlists, and standalone interactive Agent Tasks retain their existing contracts.
- Evidence expires and becomes invalid when its subject, build, runtime, image, policy, adapter, AuthNZ mode, signer, or validity window changes.
- The accepted tradeoff is visible draft functionality without background execution until dependency work supplies authoritative proof.

## Follow-up

- `TASK-13130`: isolation attestation and hostile runtime proof.
- `TASK-13131`: scheduled-mode secure ACP transcripts and leakage gates.
- `TASK-13132`: ACP dispatch recovery and monotonic execution evidence.
- `TASK-13133`: scheduled identity, credentials, revocation, and pre-action mediation.
- `operational_fail_closed` remains a cross-cutting exit criterion for all four tasks.
- Phase 4D.1B may change the source-defined execution-stack readiness only after its complete vertical slice passes independently.
