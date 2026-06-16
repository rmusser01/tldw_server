# ACP Downstream-Agent Compatibility Matrix

This document is the updateable contract for downstream agents launched through
the Agent Client Protocol (ACP) runner. It defines support states, verification
levels, required evidence, and caveat language for release notes, setup guides,
Agent Registry status, and operator troubleshooting.

The matrix is intentionally documentation-first. Updating support status should
not require code changes unless a surface needs to display a newly documented
field.

For the reproducible certification workflow and command manifest, see
`ACP_Certification_Checklist.md`.

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

## Entrypoint Strategy Terminology

Agent Registry rows use these stable strategy names:

| Strategy | Meaning |
| --- | --- |
| `native_acp` | The downstream agent binary itself exposes an ACP-compatible stdio entrypoint. |
| `external_acp_adapter` | A separate adapter binary exposes ACP stdio and controls the downstream agent CLI. The adapter command is the ACP entrypoint, while the agent command remains the display/downstream binary. |
| `documented_candidate` | Setup information exists, but no concrete ACP stdio entrypoint is configured for passive readiness or certification. |
| `custom_template` | Seeded template profile. It is not itself certifiable; concrete custom support requires a distinct named profile plus command, args, env redaction policy, workspace policy, and evidence. |

Legacy `adapter_acp` input may be imported as `external_acp_adapter` for
compatibility, but release-facing docs and seeded registry rows should use only
the canonical `external_acp_adapter` label.

## Matrix Schema

Store the compatibility matrix as Markdown so operators can update it in a docs
PR without code changes. A row must include:

| Field | Required | Description |
| --- | --- | --- |
| Agent | Yes | Human-readable downstream agent or profile name. |
| Profile key | Yes | Stable key used in docs/setup examples, such as `codex`, `claude_code`, `opencode`, `my_acp_agent`, or `stub`. The seeded `custom` key is reserved for the template row and is not a concrete certification target. |
| Transport/runner mode | Yes | `host_stdio`, `sandbox_stdio`, `external_acp_adapter`, or `stub_fixture`. |
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
| Codex CLI via Codex ACP adapter | `codex` | `external_acp_adapter` | macOS host; other hosts unverified | `codex-cli 0.128.0`; `codex-acp 0.15.0` | `supported_with_caveats` | `live_e2e_tested` | `init=pass`, `session_new=pass`, `prompt=pass`, `structured_completion=pass`, `artifacts=n/a`, `diagnostics=limited`, `cancel_close=pass`, `review_loop=n/a`, `workspace_env=limited`, `mcp_injection=skip`, `sandbox=skip`, `redacted_support_view=pass` | `PATH=/private/tmp/tldw-codex-acp-0.15.0/node_modules/.bin:$PATH TLDW_E2E_SERVER_URL=127.0.0.1:18003 TLDW_E2E_API_KEY=... ACP_AGENT_PROFILE=codex python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile live-e2e --run` on branch `codex/acp-codex-orchestration-progress`, macOS host, tldw-agent runner `0.1.0`, June 2, 2026: exit 0 after backend health/setup-guide, `sessions/new`, `sessions/prompt`, redacted detail/events/artifacts, diagnostics, cancel, close, and `tools/tldw-agent/scripts/verify-local-build.sh`; result summary: `stop_reason=end_turn`, `events_total=2`, `artifacts_total=0`, `diagnostics_total=0`, session `019e89c4-bfc8-7b30-a9e0-38917be6d117`. | Supported for the verified macOS host runner profile with delegated Codex adapter credentials and pinned `codex-acp` `0.15.0`. Passive readiness can still block on `adapter_missing`, `agent_binary_missing`, `adapter_auth_missing`, `adapter_auth_failed`, or `agent_auth_failed`. Artifact-producing workflows, non-empty MCP server injection, sandbox behavior, and reviewer-loop behavior remain unverified; run `python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile workspace-live-e2e --run` with `ACP_AGENT_PROFILE=codex` and a live Research Workspace ID before upgrading those checks. Diagnostics endpoint was reachable in the passing host run, but no failure diagnostic payload was produced. | #1564 |
| Claude Code via Claude Agent ACP adapter | `claude_code` | `external_acp_adapter` | macOS host; other hosts unverified | `2.1.142 (Claude Code)`; `@agentclientprotocol/claude-agent-acp` `0.40.0` candidate | `documented_unverified` | `documented_only` | `init=skip`, `session_new=skip`, `prompt=skip`, `structured_completion=skip`, `artifacts=skip`, `diagnostics=skip`, `cancel_close=skip`, `review_loop=skip`, `workspace_env=skip`, `mcp_injection=skip`, `sandbox=skip`, `redacted_support_view=skip` | June 3, 2026 local probe: `claude --version` returned `2.1.142 (Claude Code)`; `claude --help` and `claude mcp --help` exposed MCP management/server commands but no native ACP command; `npm view @agentclientprotocol/claude-agent-acp version repository bin --json` identified `0.40.0` with binary `claude-agent-acp`; `command -v claude-agent-acp` failed locally. | Claude Code has no native ACP stdio entrypoint identified in the installed CLI. The certifiable path is the pinned external `@agentclientprotocol/claude-agent-acp` adapter, but it is not installed or live-E2E certified in this environment. Keep release claims at `documented_unverified` until the adapter is installed, auth/runtime state is configured, and initialize/session/prompt live E2E passes. | #2244 |
| Aider via unverified Aider ACP adapter candidate | `aider` | `external_acp_adapter` | macOS host; other hosts unverified | `aider 0.86.2`; `aider-acp` candidate unverified | `documented_unverified` | `documented_only` | `init=skip`, `session_new=skip`, `prompt=skip`, `structured_completion=skip`, `artifacts=skip`, `diagnostics=skip`, `cancel_close=skip`, `review_loop=skip`, `workspace_env=skip`, `mcp_injection=skip`, `sandbox=skip`, `redacted_support_view=skip` | Aider was installed and configured in `~/.aider.conf.yml` to use local llama.cpp at `http://127.0.0.1:9099/v1` with model `gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf`; `aider --message "Reply with exactly AIDER_LOCAL_OK." --no-git --no-auto-commits --no-pretty --no-stream --map-tokens 0 --yes-always` returned `AIDER_LOCAL_OK` on branch `codex/acp-opencode-aider-llamacpp-certification` commit `53c018269`, macOS host, May 23, 2026. June 16, 2026 decision check: installed Aider `aider 0.86.2` still exposes no native ACP stdio server mode, but third-party `jorgejhms/aider-acp` is documented as an external adapter candidate. `python Helper_Scripts/Testing-related/acp_certification_smoke.py --agent-profile aider --format json` now records `external_acp_adapter` with `acp_command=aider-acp`; local passive readiness blocks on `adapter_missing` until that adapter is installed. | Direct Aider prompting is not ACP certification. The adapter candidate is not installed, pinned, audited, or live-E2E certified here. Keep release claims at `documented_unverified` until `aider-acp` or another maintained adapter passes initialize/session/prompt through the backend live-E2E path with provider/runtime caveats recorded. | #2050 |
| Goose | `goose` | `host_stdio` | macOS host; other hosts unverified | `1.35.0` on May 23, 2026 backend live E2E | `supported_with_caveats` | `live_e2e_tested` | `init=pass`, `session_new=pass`, `prompt=pass`, `structured_completion=pass`, `artifacts=n/a`, `diagnostics=limited`, `cancel_close=pass`, `review_loop=n/a`, `workspace_env=limited`, `mcp_injection=skip`, `sandbox=skip`, `redacted_support_view=pass` | `python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile live-e2e --run` with `ACP_AGENT_PROFILE=goose` on branch `codex/acp-goose-backend-live-e2e` commit `f9ff03f88`, macOS host, tldw-agent runner `0.1.0`, May 23, 2026: exit 0 after backend health/setup-guide, `sessions/new`, `sessions/prompt`, redacted detail/events/artifacts, diagnostics, cancel, close, and `tools/tldw-agent/scripts/verify-local-build.sh`; result summary: `stop_reason=end_turn`, `events_total=2`, `artifacts_total=0`, `diagnostics_total=0`. Earlier direct host-stdio evidence: `python Helper_Scripts/Testing-related/acp_certification_smoke.py --agent-profile goose --run` on `codex/acp-goose-hermes-certification` commit `8c4a76c15`, May 23, 2026. | Supported for the verified macOS host runner profile with configured Goose provider state. Artifact-producing workflows, non-empty MCP server injection, sandbox behavior, and reviewer-loop behavior remain unverified; diagnostics endpoint was reachable but no failure diagnostic payload was produced in the passing run. | #1563 |
| Hermes | `hermes` | `host_stdio` | macOS host; other hosts unverified | `Hermes Agent v0.13.0 (2026.5.7)` on May 23, 2026 backend live E2E | `supported_with_caveats` | `live_e2e_tested` | `init=pass`, `session_new=pass`, `prompt=pass`, `structured_completion=pass`, `artifacts=n/a`, `diagnostics=limited`, `cancel_close=pass`, `review_loop=n/a`, `workspace_env=limited`, `mcp_injection=skip`, `sandbox=skip`, `redacted_support_view=pass` | `python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile live-e2e --run` with `ACP_AGENT_PROFILE=hermes` on branch `codex/acp-hermes-live-e2e-certification` commit `5e6672f8f`, macOS host, tldw-agent runner `0.1.0`, May 23, 2026: exit 0 after backend health/setup-guide, `sessions/new`, `sessions/prompt`, redacted detail/events/artifacts, diagnostics, cancel, close, and `tools/tldw-agent/scripts/verify-local-build.sh`; result summary: `stop_reason=end_turn`, `events_total=2`, `artifacts_total=0`, `diagnostics_total=0`. Earlier direct host-stdio evidence: `python Helper_Scripts/Testing-related/acp_certification_smoke.py --agent-profile hermes --run` on `codex/acp-goose-hermes-certification` commit `8c4a76c15`, May 23, 2026. | Supported for the verified macOS host runner profile with configured Hermes provider state. Artifact-producing workflows, non-empty MCP server injection, sandbox behavior, and reviewer-loop behavior remain unverified; diagnostics endpoint was reachable but no failure diagnostic payload was produced in the passing run. | #1563 |
| Continue CLI | `continue_dev` | `host_stdio` | macOS host; other hosts unverified | `@continuedev/cli 1.5.46` package; no ACP entrypoint identified | `documented_unverified` | `documented_only` | `init=skip`, `session_new=skip`, `prompt=skip`, `structured_completion=skip`, `artifacts=skip`, `diagnostics=skip`, `cancel_close=skip`, `review_loop=skip`, `workspace_env=skip`, `mcp_injection=skip`, `sandbox=skip`, `redacted_support_view=skip` | Earlier local blocker: `continue` resolved to a zsh shell builtin rather than an installed ACP-compatible CLI. June 16, 2026 decision check: `npm view @continuedev/cli version bin repository --json` reported latest `1.5.46` with binary `cn`; `npx -y @continuedev/cli --help` exposed interactive/headless `-p` and review modes but no ACP stdio server command; `npx -y @continuedev/cli --version` returned `1.5.46`. `python Helper_Scripts/Testing-related/acp_certification_smoke.py --agent-profile continue_dev --format json` remains documented-only with `entrypoint_strategy_missing`. | Continue direct/headless prompting is not ACP certification. The registry uses display command `cn` for the current package, but no native ACP entrypoint or maintained adapter has been identified. Keep release claims at `documented_unverified` until a concrete ACP stdio command or adapter is found and live-certified. | #2051 |
| OpenCode | `opencode` | `host_stdio` | macOS host; other hosts unverified | `1.15.7` on May 23, 2026 backend live E2E | `supported_with_caveats` | `live_e2e_tested` | `init=pass`, `session_new=pass`, `prompt=pass`, `structured_completion=pass`, `artifacts=n/a`, `diagnostics=limited`, `cancel_close=pass`, `review_loop=n/a`, `workspace_env=limited`, `mcp_injection=skip`, `sandbox=skip`, `redacted_support_view=pass` | OpenCode was configured in `~/.config/opencode/opencode.json` to use local llama.cpp at `http://127.0.0.1:9099/v1` with model `gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf`; `opencode run --pure --format json --model llama.cpp/gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf "Reply with exactly ACP_LOCAL_OK."` returned `ACP_LOCAL_OK.`. `python Helper_Scripts/Testing-related/acp_certification_smoke.py --agent-profile opencode --run` exited 0. `python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile live-e2e --run` with `ACP_AGENT_PROFILE=opencode` on branch `codex/acp-opencode-aider-llamacpp-certification` commit `53c018269`, macOS host, tldw-agent runner `0.1.0`, May 23, 2026: exit 0 after backend health/setup-guide, `sessions/new`, `sessions/prompt`, redacted detail/events/artifacts, diagnostics, cancel, close, and `tools/tldw-agent/scripts/verify-local-build.sh`; result summary: `stop_reason=end_turn`, `events_total=2`, `artifacts_total=0`, `diagnostics_total=0`, session `ses_1a80f9407ffejmMKs2NWMrv52z`. | Supported for the verified macOS host runner profile with configured OpenCode local llama.cpp provider state. Artifact-producing workflows, non-empty MCP server injection, sandbox behavior, and reviewer-loop behavior remain unverified; diagnostics endpoint was reachable but no failure diagnostic payload was produced in the passing run. | #1563 |
| Custom ACP template | `custom` | `template` | n/a | n/a | `documented_unverified` | `documented_only` | `init=skip`, `session_new=skip`, `prompt=skip`, `structured_completion=skip`, `artifacts=skip`, `diagnostics=skip`, `cancel_close=skip`, `review_loop=skip`, `workspace_env=skip`, `mcp_injection=skip`, `sandbox=skip`, `redacted_support_view=skip` | Blocker evidence and requirements in `ACP_OSS_Custom_Certification_2026_05_11.md`; `python Helper_Scripts/Testing-related/acp_certification_smoke.py --agent-profile custom --format json` emits the concrete-profile evidence contract and no runnable commands. | The seeded `custom` profile is template-only. Operators must create a distinct named profile with an ACP-compatible binary, args, env redaction policy, workspace policy, host/runtime, version output, provider assumptions, and live evidence before making support claims. Do not imply generic support for arbitrary commands. | #2052 |

## Concrete Custom Profile Evidence Contract

The seeded `custom` registry profile is a template and must stay
`documented_unverified` / `documented_only`. A certifiable custom profile is a
separate named profile, for example `my_acp_agent`, with its own evidence row.
Do not upgrade the seeded `custom` row or use it as a generic support claim.

The smoke helper exposes the required contract:

```bash
python Helper_Scripts/Testing-related/acp_certification_smoke.py --agent-profile custom --format json
```

Minimum metadata for a concrete custom profile:

- `profile_key` and `profile_name`, distinct from the seeded `custom` template.
- `entrypoint_strategy`, `acp_command`, and `acp_args`.
- `env_var_names` only; never record secret values.
- `workspace_policy`, including cwd, allowed roots, and sandbox expectation.
- `host_runtime`, provider assumptions, agent version output, repo commit, and
  runner version.
- Live results for `initialize`, `session_new`, and `session_prompt` at minimum.

Redaction requirements:

- Record credential and environment variable names only, not values.
- Use redacted support views for transcript, event, artifact, and diagnostics
  evidence.
- Treat missing redaction evidence as a blocker for public support claims.

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
Runner version:
Config profile:
Env var names:
Workspace policy:
Provider assumptions:
Commands:
Capability results:
Caveats:
Follow-up issue:
```

## Minimum Certification Checklists

Use `Helper_Scripts/Testing-related/acp_certification_smoke.py` to emit the
current command manifest for `stub-smoke` and `live-e2e` evidence. The helper
reuses the existing backend ACP suites, mocked browser flow, and Go runner
verification rather than defining a parallel test harness. The detailed operator
checklist lives in `ACP_Certification_Checklist.md`.

### Documented Only

- Document command, args, required environment, and workspace prerequisites.
- State whether the agent is expected to speak ACP stdio directly or through a
  wrapper.
- Mark support state `documented_unverified` unless prior evidence exists.
- For seeded `custom`, document only the template and evidence contract; do not
  treat the template as a certifiable concrete profile.

### Stub Or Smoke Tested

- Run focused ACP backend tests covering session lifecycle, WebSocket/update
  flow, diagnostics, cancel/close, and redacted views.
- Run the Go runner verification script when runner code or profile behavior is
  part of the claim.
- Record commit, command, and result in the evidence record.

### Live E2E Tested

- Verify the downstream binary is installed and record its version.
- For concrete custom profiles, use a distinct profile key rather than `custom`.
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
| `custom_template` | The seeded `custom` profile is a template and cannot be live-certified without a distinct named concrete profile. |
| `entrypoint_strategy_missing` | Registry row has no verified ACP stdio entrypoint strategy or command. |
| `binary_missing` | Expected command is not installed or not on PATH. |
| `credentials_missing` | Provider/API key or account login is unavailable. |
| `adapter_required` | Agent needs a separate ACP adapter before live ACP certification can run. |
| `adapter_missing` | Adapter-backed strategy is configured but the adapter command is unavailable. |
| `live_certification_required` | A documented ACP path exists, but a live certification run has not passed. |
| `agent_binary_missing` | External adapter is configured, but the downstream agent CLI it controls is missing or not on PATH. |
| `adapter_auth_missing` | External adapter is installed but its required credentials or login state are unavailable. |
| `adapter_auth_failed` | External adapter credentials or login state were present but failed during an auth check or bounded probe. |
| `agent_auth_failed` | Downstream agent credentials or account state failed after the adapter launched the agent. |
| `acp_initialize_failed` | ACP stdio command started but failed the bounded initialize probe. |
| `shell_builtin_collision` | Configured ACP command resolves to a shell builtin or alias-like value instead of an executable. |
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
