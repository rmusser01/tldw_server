# ACP Downstream-Agent Compatibility Matrix

This document is the updateable contract for downstream agents launched through
the Agent Client Protocol (ACP) runner. It defines support states, verification
levels, required evidence, and caveat language for release notes, setup guides,
Agent Registry status, and operator troubleshooting.

The matrix is intentionally documentation-first. Updating support status should
not require code changes unless a surface needs to display a newly documented
field.

## Scope

This matrix covers downstream ACP-compatible agents launched by `tldw-agent`
through stdio, including Codex, Claude Code, OpenCode, custom ACP agents, and
the in-repo stub/smoke profile. It does not define a public agent marketplace,
installer, or guarantee for every third-party implementation.

Compatibility evidence must distinguish:

- protocol incompatibility in the downstream agent;
- missing local binary, provider key, workspace allowlist, or host runtime;
- unverified optional capabilities such as artifacts, MCP injection, sandbox
  behavior, or reviewer loops.

## Support States

| State | Meaning | Release language |
| --- | --- | --- |
| `supported` | Required checks for the stated verification level passed on a current branch and host profile. | May be listed as supported for the verified host/profile. |
| `supported_with_caveats` | Core protocol checks passed, but one or more optional capabilities or host runtimes have explicit caveats. | May be listed as supported with the named caveats. |
| `experimental` | Basic behavior works in limited testing, but coverage is incomplete or host-dependent. | May be offered for early adopters; avoid production-support claims. |
| `documented_unverified` | Configuration is documented, but no current run evidence exists. | May appear in setup docs as a candidate only. |
| `unsupported` | The agent does not speak the required ACP stdio contract or fails a required check for reasons not attributable to local prerequisites. | Do not present as usable through ACP until retested. |

Use `documented_unverified` rather than `unsupported` when the only blocker is
missing local setup such as absent credentials, missing binary, or unavailable
host runtime.

## Verification Levels

| Level | Required evidence | Typical use |
| --- | --- | --- |
| `documented_only` | Configuration snippet and prerequisite list are documented. No current run evidence is required. | Candidate agents and custom agent templates. |
| `stub_smoke_tested` | In-repo stub or fixture proves the ACP server/runner path can create a session, prompt, stream/update, cancel/close, and record diagnostics. | Baseline protocol health in CI. |
| `live_e2e_tested` | A real downstream agent binary ran through the server/runner path on a named host profile with provider credentials. | Release notes that claim a specific agent works locally. |
| `sandbox_tested` | Live or stub agent ran inside the configured sandbox backend and passed workspace/root/runtime checks. | Claims about Docker, Lima, or VZ isolation. |
| `production_supported` | Live E2E and relevant sandbox or workspace checks passed, caveats are documented, and the setup surface can explain failures. | Enterprise/operator support claims. |

Verification levels are cumulative only where relevant. For example, an agent can
be `live_e2e_tested` on the host runner while sandbox behavior remains
unverified.

## Capability Checks

Each matrix row should use these stable check IDs. Mark each check as `pass`,
`fail`, `skip`, or `n/a`, and explain every `fail` or `skip` in the caveats
column.

| Check ID | Capability | Minimum evidence |
| --- | --- | --- |
| `init` | Runner starts downstream process and receives `initialize` capabilities. | Health/setup output or session creation log. |
| `session_new` | ACP session starts with expected agent type and cwd/env profile. | `POST /api/v1/acp/sessions/new` result or orchestration run record. |
| `prompt` | Prompt request returns a completion or structured terminal state. | `session/prompt`, async task result, or WebSocket transcript. |
| `structured_completion` | Terminal status, reason code, and error/retry classification are machine-readable. | Session events or orchestration task run detail. |
| `artifacts` | Agent-produced artifacts appear through ACP artifact drill-through when applicable. | `/artifacts` result or explicit `n/a`. |
| `diagnostics` | Failures expose normalized diagnostics without raw server-log inspection. | `/diagnostics` result or task failure context. |
| `cancel_close` | Cancel and close/teardown paths finish without orphaned runner state. | `/cancel`, `/close`, teardown, or reconciliation evidence. |
| `review_loop` | Reviewer decisions or manual review state attach to durable task/run history when applicable. | Agent Tasks or orchestration task detail evidence. |
| `workspace_env` | Workspace cwd, allowed roots, and per-session env are applied or fail closed. | Workspace health/session request evidence. |
| `mcp_injection` | Workspace MCP servers are injected or explicitly unsupported. | Session request, setup status, or caveat. |
| `sandbox` | Configured sandbox backend starts and enforces workspace/runtime policy. | Sandbox runner test or live sandbox run evidence. |
| `redacted_support_view` | Detail/events/artifacts can be viewed through redacted support mode. | `?redacted=true` endpoint evidence. |

## Matrix Schema

Store the compatibility matrix as Markdown so operators can update it in a docs
PR without code changes. A row must include:

| Field | Required | Description |
| --- | --- | --- |
| Agent | Yes | Human-readable downstream agent or profile name. |
| Profile key | Yes | Stable key used in docs/setup examples, such as `codex`, `claude_code`, `opencode`, `custom_acp`, or `stub`. |
| Transport/runner mode | Yes | `host_stdio`, `sandbox_stdio`, or `stub_fixture`. |
| Host/runtime | Yes | OS/runtime profile such as `macos-host`, `linux-host`, `docker`, `lima`, or `vz`. |
| Version | Yes | Agent binary version, commit, or `unknown` with caveat. |
| Support state | Yes | One of the support states above. |
| Verification level | Yes | Highest verified level from the verification-level table. |
| Capability checks | Yes | Compact status list using the stable check IDs. |
| Evidence | Yes | Link or text naming command, branch/commit, date, and result. |
| Caveats | Yes | Empty only when no caveats remain for the claimed state. |
| Follow-up | No | Issue/PR link for missing capability, bug, or future retest. |

## Current Matrix

| Agent | Profile key | Mode | Host/runtime | Version | Support state | Verification level | Capability checks | Evidence | Caveats | Follow-up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| In-repo ACP stub profile | `stub` | `stub_fixture` | CI/local host | repo commit | `supported_with_caveats` | `stub_smoke_tested` | `init=pass`, `session_new=pass`, `prompt=pass`, `structured_completion=pass`, `diagnostics=pass`, `cancel_close=pass`, `redacted_support_view=pass`, `artifacts=limited`, `review_loop=n/a`, `workspace_env=limited`, `mcp_injection=n/a`, `sandbox=n/a` | Backend ACP pytest and mocked browser setup/run/diagnose evidence in `ACP_Production_Readiness.md`. | Proves server/runner protocol behavior, not live third-party agent compatibility. | Add per-release evidence when this row is updated. |
| Codex CLI | `codex` | `host_stdio` | host dependent | unknown | `documented_unverified` | `documented_only` | `init=skip`, `session_new=skip`, `prompt=skip`, `structured_completion=skip`, `artifacts=skip`, `diagnostics=skip`, `cancel_close=skip`, `review_loop=skip`, `workspace_env=skip`, `mcp_injection=skip`, `sandbox=skip`, `redacted_support_view=skip` | Configuration example exists in `Agent_Client_Protocol.md`. | Requires installed ACP-compatible Codex command, provider credentials, and live stdio verification before release claims. | Create per-agent verification issue when a test host is available. |
| Claude Code | `claude_code` | `host_stdio` | host dependent | unknown | `documented_unverified` | `documented_only` | `init=skip`, `session_new=skip`, `prompt=skip`, `structured_completion=skip`, `artifacts=skip`, `diagnostics=skip`, `cancel_close=skip`, `review_loop=skip`, `workspace_env=skip`, `mcp_injection=skip`, `sandbox=skip`, `redacted_support_view=skip` | Configuration example exists in `Agent_Client_Protocol.md`. | Requires installed ACP-compatible Claude command, provider credentials, and live stdio verification before release claims. | Create per-agent verification issue when a test host is available. |
| OpenCode | `opencode` | `host_stdio` | host dependent | unknown | `documented_unverified` | `documented_only` | `init=skip`, `session_new=skip`, `prompt=skip`, `structured_completion=skip`, `artifacts=skip`, `diagnostics=skip`, `cancel_close=skip`, `review_loop=skip`, `workspace_env=skip`, `mcp_injection=skip`, `sandbox=skip`, `redacted_support_view=skip` | Candidate profile tracked by #1539. | Requires an ACP stdio-compatible OpenCode entrypoint and live verification. | Create per-agent verification issue when a test host is available. |
| Custom ACP-compatible agent | `custom_acp` | `host_stdio` or `sandbox_stdio` | host dependent | operator supplied | `documented_unverified` | `documented_only` | `init=skip`, `session_new=skip`, `prompt=skip`, `structured_completion=skip`, `artifacts=skip`, `diagnostics=skip`, `cancel_close=skip`, `review_loop=skip`, `workspace_env=skip`, `mcp_injection=skip`, `sandbox=skip`, `redacted_support_view=skip` | Custom command template exists in `Agent_Client_Protocol.md`. | Operators must provide binary, args, env, workspace policy, and evidence. | Split into a dedicated issue when a named implementation is evaluated. |

## Evidence Record Template

Use one record per agent/profile/host combination. Add the record to the issue or
PR that updates the matrix.

```text
Agent:
Profile key:
Support state:
Verification level:
Commit/branch:
Host/runtime:
Agent binary/version:
Config profile:
Commands:
Capability results:
Caveats:
Follow-up issue:
```

## Minimum Certification Checklists

### Documented Only

- Document command, args, required environment, and workspace prerequisites.
- State whether the agent is expected to speak ACP stdio directly or through a
  wrapper.
- Mark support state `documented_unverified` unless prior evidence exists.

### Stub Or Smoke Tested

- Run focused ACP backend tests covering session lifecycle, WebSocket/update
  flow, diagnostics, cancel/close, and redacted views.
- Run the Go runner verification script when runner code or profile behavior is
  part of the claim.
- Record commit, command, and result in the evidence record.

### Live E2E Tested

- Verify the downstream binary is installed and record its version.
- Run a real session through tldw_server using the configured runner profile.
- Exercise `init`, `session_new`, `prompt`, `structured_completion`,
  `diagnostics`, and `cancel_close`.
- Exercise `artifacts`, `review_loop`, `workspace_env`, and `mcp_injection` when
  the support claim mentions those capabilities.
- Query redacted support views for sessions that include transcript or artifact
  data.

### Sandbox Tested

- Record sandbox backend (`docker`, `lima`, `vz`, or other configured runtime).
- Prove missing runtime/configuration fails closed with stable setup guidance.
- Run or explicitly skip workspace bind/mount, cwd, env, and network behavior.
- Do not upgrade a host-only live E2E result to sandbox-tested without this
  evidence.

### Production Supported

- Live E2E checks are current for the release branch.
- Required sandbox/workspace checks for the release claim are current.
- Setup and Agent Registry surfaces can explain missing prerequisites.
- Known caveats are linked to follow-up issues and release notes use the same
  caveat language.

## Caveat Taxonomy

Use these labels so Agent Registry/setup/docs/admin reporting can reuse the same
language later:

| Caveat | Meaning |
| --- | --- |
| `protocol_incompatibility` | Agent does not satisfy ACP stdio contract or required JSON-RPC behavior. |
| `binary_missing` | Expected command is not installed or not on PATH. |
| `credentials_missing` | Provider/API key or account login is unavailable. |
| `workspace_config_missing` | Workspace allowlist, cwd, or root policy is not configured. |
| `host_runtime_missing` | Docker/Lima/VZ or other required host runtime is unavailable. |
| `sandbox_unverified` | Host mode works or is documented, but sandbox mode has no current evidence. |
| `mcp_injection_unverified` | Workspace MCP server injection has no current evidence for this agent. |
| `artifact_capability_unverified` | Agent output/artifact behavior has no current evidence. |
| `review_loop_unverified` | Reviewer-agent or manual review loop behavior has no current evidence. |
| `redacted_view_unverified` | Support-safe redacted views were not checked for this agent run. |

## Status Surface Plan

The compatibility matrix should feed these surfaces when implementation work
begins:

- Agent Registry: show support state, verification level, last evidence date,
  and top caveats next to each configured agent.
- `/api/v1/acp/setup-guide`: include configured-but-unverified vs verified
  status and prerequisite-specific actions.
- `/api/v1/acp/health` and `/api/v1/acp/agents/health`: keep runtime failures
  separate from protocol incompatibilities.
- ACP operator docs and release notes: use only the support-state and caveat
  language from this document.
- Future admin reporting (#1537): aggregate by support state, verification
  level, and caveat taxonomy rather than inventing a second failure vocabulary.

Until a UI/API contract lands, this Markdown file is the source of truth for
compatibility claims and can be updated independently in documentation PRs.
