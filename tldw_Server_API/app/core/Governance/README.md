# Governance

Governance resolves policy rules and records policy gaps for ACP and MCP execution paths. It turns candidate rules into an effective action, classifies knowledge and validation requests by category and scope, stores unresolved governance questions in SQLite, and emits rollout/audit metrics for governance checks.

## Start Here

- `types.py` defines candidate and effective action dataclasses plus the governance action literals.
- `resolver.py` contains deterministic action precedence for competing rules.
- `service.py` is the policy-query and validation facade used by agent-facing call paths.
- `store.py` persists governance rules and deduplicated open gaps.
- `metrics.py` builds audit traces and rollout-mode metrics.
- Related MCP surface: `tldw_Server_API/app/core/MCP_unified/modules/implementations/governance_module.py`.
- Related ACP surface: `tldw_Server_API/app/core/Agent_Client_Protocol/runner_client.py`.
- Related tests: `tldw_Server_API/tests/Governance/`, `tldw_Server_API/app/core/MCP_unified/tests/test_governance_module.py`, and `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_governance_coordinator.py`.

## Responsibilities

- Resolve candidate governance rules into one effective `allow`, `warn`, `require_approval`, or `deny` action.
- Prefer stricter actions, more specific scopes, higher priority, and newer updates when rules conflict.
- Classify governance requests into categories such as security, privacy, dependencies, compliance, or general.
- Validate proposed changes and return warnings, approval requirements, or denials.
- Record unresolved governance gaps with a normalized fingerprint so repeated questions reuse the same open gap.
- Build rollout-mode traces for off, shadow, and enforce operation.

## Module Map

- `types.py`: shared dataclasses and action literals.
- `resolver.py`: action precedence and effective-action selection.
- `service.py`: policy lookup, category inference, fallback actions, and validation results.
- `store.py`: async SQLite rule and gap storage helpers.
- `metrics.py`: rollout-mode resolution, audit trace construction, and metrics recording.
- `__init__.py`: package marker.

## How It Connects

- MCP Unified exposes governance operations through `governance_module.py`.
- ACP uses `ACPGovernanceCoordinator` in `runner_client.py` to check execution-related governance decisions.
- The governance store owns its local SQLite schema for `governance_rules` and `governance_gaps`.
- Metrics are recorded through the MCP Unified metrics collector when available.
- Design and operations context live in `Docs/Plans/2026-02-24-unified-governance-plane-implementation.md` and `Docs/MCP/Unified/Governance_Operations.md`.

## Architecture Notes

### Core Flow

- ACP and MCP callers build a governance request, pass it to `GovernanceService`, and receive an effective action plus warnings, approval requirements, or gap metadata.
- `service.py` classifies the category and scope, loads candidate rules, handles empty candidate sets, and calls `resolver.py` only when there is at least one candidate.
- `resolver.py` applies deterministic precedence so stricter, more specific, higher-priority, and newer rules win.
- `metrics.py` records rollout traces for off, shadow, and enforce modes without changing the resolver contract.

### State And Data

- `store.py` owns the SQLite schema for `governance_rules` and `governance_gaps`.
- Gap fingerprints include category plus optional org, team, persona, and workspace scope so repeated unresolved questions reuse the same open gap.
- ACP and MCP keep their own execution context; Governance only returns policy decisions and trace data.

### Security And Operations

- Enforce rollout mode can block a denied action; shadow mode records the decision without making every trace a runtime block.
- Empty rule sets should be handled by service fallback or gap creation before calling `resolve_effective_action`.
- Audit traces should describe decisions and scope without embedding sensitive request bodies.

### Extension Checklist

- Rule conflict change: update `resolver.py` and Governance resolver tests first.
- New category or scope: update `service.py`, store query behavior, gap dedupe tests, and ACP/MCP integration tests.
- Rollout or metrics change: update `metrics.py` and trace/rollout tests.

## Extension Points

- Add a new rule source by implementing the loader methods expected by `GovernanceService`.
- Change conflict behavior in `resolver.py` and update resolver tests first.
- Extend category inference in `service.py` when new policy families are added.
- Add new gap lifecycle states in `store.py` only after checking dedupe behavior and schema tests.
- Extend audit or rollout behavior in `metrics.py` with coverage for trace payloads.

## Testing

- Resolver, service, store schema, gap dedupe, and rollout behavior are covered in `tldw_Server_API/tests/Governance/`.
- MCP integration is covered by `tldw_Server_API/app/core/MCP_unified/tests/test_governance_module.py`.
- ACP coordinator behavior is covered by `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_governance_coordinator.py`.

## Gotchas

- `resolve_effective_action` requires at least one candidate; callers must handle empty rule sets before invoking it.
- Gap dedupe is scoped by category and optional org, team, persona, or workspace ids, so changing the fingerprint shape can reopen old gaps.
- Shadow rollout mode records decisions without necessarily enforcing them; do not treat every trace as a blocking decision.
