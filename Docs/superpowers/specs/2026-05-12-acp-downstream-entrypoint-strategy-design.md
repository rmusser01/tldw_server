# ACP Downstream Entrypoint Strategy Design

Date: 2026-05-12
Status: Proposed
Tracking: TASK-286, GitHub issues #1563 and #1564

## Summary

ACP downstream certification currently treats an agent registry `command` as if
it were the ACP stdio entrypoint. The May 11, 2026 blocker evidence showed that
this is not reliable: `codex` and `claude` are user-facing CLIs, `codex
mcp-server` speaks MCP rather than ACP, several OSS candidates were not
installed, and custom profiles have no concrete command to test.

This design introduces an explicit downstream entrypoint strategy model. Each
registry profile can state whether ACP is reached through a native ACP command,
an adapter-backed ACP command, a documented candidate, or a custom template.
The backend can classify the profile, the certification helper can emit
profile-specific probe manifests, and setup/status surfaces can explain the
actual blocker without implying live support.

The goal is not to install agents or implement adapters in the first PR. The
goal is to make the product architecture honest enough that live certification
can proceed agent by agent without repeating the same ambiguity.

## Current Evidence

The current compatibility matrix and certification notes preserve conservative
support claims:

- Codex CLI remains `documented_unverified` / `documented_only`. The configured
  `codex` command did not answer ACP `initialize`, and `codex mcp-server`
  answered MCP JSON-RPC rather than ACP.
- Claude Code remains `documented_unverified` / `documented_only`. The
  configured `claude` command exited before ACP `initialize` because local
  auth state was missing, and no ACP-specific stdio entrypoint was identified.
- Aider, Goose, Continue, and OpenCode remain `documented_unverified` /
  `documented_only`. The local host did not have runnable downstream binaries
  for those rows; `continue` resolved to a shell builtin rather than a CLI.
- Custom profiles remain templates. They need a named command, args,
  environment, workspace policy, host/runtime, version, and evidence before any
  support claim.

The latest public docs change the next step. OpenCode and Goose document native
ACP commands, while Codex and Claude Code appear to require adapters. That means
the architecture needs to represent both native and adapter-backed ACP
entrypoints.

## Goals

1. Represent ACP entrypoint strategy explicitly in registry metadata.
2. Classify each profile into a probeable or blocked state before certification
   runs.
3. Generate profile-specific probe manifests that test the ACP command, not the
   user-facing CLI by accident.
4. Keep setup/API/UI surfaces aligned with the compatibility matrix.
5. Prevent blocker-documentation PRs from closing live-certification issues when
   live evidence failed or never ran.
6. Provide a staged path to certify native ACP agents first and adapter-backed
   agents next.

## Non-Goals

- Do not install OpenCode, Goose, Codex adapters, Claude adapters, Aider, or
  Continue as part of the architecture PR.
- Do not implement a Codex or Claude adapter in the first PR.
- Do not upgrade any profile to `supported`, `supported_with_caveats`, or
  `live_e2e_tested` without real `initialize`, `session/new`, and prompt-path
  evidence.
- Do not claim generic support for arbitrary custom commands.
- Do not replace the compatibility matrix as the source of release support
  claims.

## Entrypoint Strategies

Each registry profile gets one explicit `entrypoint_strategy`:

| Strategy | Meaning | Example |
| --- | --- | --- |
| `native_acp` | The downstream agent exposes an ACP stdio command directly. | `opencode acp`, `goose acp` |
| `adapter_acp` | ACP is provided by a separate adapter binary or wrapper around the agent SDK/CLI. | `codex-acp`, Claude Code SDK adapter |
| `documented_candidate` | Setup exists, but no ACP entrypoint has been identified or verified. | Aider, Continue until evidence changes |
| `custom_template` | Operator must provide a concrete ACP-compatible command and evidence. | Custom profile |

The strategy is product metadata, not a support claim. A profile can be
`native_acp` and still remain `documented_unverified` until a live run passes.

## Registry Metadata

The existing registry fields remain useful for display and installation:
`type`, `name`, `description`, `command`, `args`, `requires_api_key`,
`install_instructions`, `docs_url`, `support_state`, `verification_level`,
`compatibility_notes`, and `compatibility_docs_url`.

Add fields that describe the actual ACP launch surface:

```yaml
entrypoint_strategy: native_acp | adapter_acp | documented_candidate | custom_template
acp_command: opencode
acp_args:
  - acp
adapter_source: null
adapter_docs_url: null
certification_blocker: binary_missing
```

Field intent:

- `command` remains the familiar user-facing or install target command.
- `args` remains the familiar user-facing default args, if any.
- `acp_command` is the command that should answer ACP JSON-RPC over stdio.
- `acp_args` are the args for the ACP entrypoint.
- `adapter_source` names the adapter project or package when strategy is
  `adapter_acp`.
- `adapter_docs_url` links to the adapter docs or repository.
- `certification_blocker` carries the current top blocker using the caveat
  taxonomy from `ACP_Compatibility_Matrix.md` plus the entrypoint-specific
  labels defined below. It should be omitted when no current blocker is known.

The backend should default missing strategy fields conservatively, but
certifiable strategies must be explicit:

- Missing strategy means `documented_candidate`.
- Missing `acp_command` for `native_acp` or `adapter_acp` is an entrypoint
  strategy error.
- `custom_template` can have empty `acp_command`, but it must never be
  classified as ready to probe.
- The first implementation should not infer `acp_command` from `command`.
  Inference would hide the key distinction this design is adding: user-facing
  CLI command and ACP stdio entrypoint are different concepts. A future
  migration can add inference only for legacy compatibility, but seeded
  registry rows should be explicit.

Credential and workspace blockers do not need new registry metadata in the
first implementation. The classifier should derive credential blockers from
existing `requires_api_key` and runtime environment state, and workspace
blockers from existing ACP workspace configuration.

Strategy metadata applies to both YAML-backed and dynamically registered agent
entries. Stage 1 must thread these fields through `AgentRegistryEntry`, YAML
loading, API registration/update schemas, DB save/load, `_UPDATABLE_FIELDS`,
and response serialization. Existing API-backed rows that lack the new fields
should default to `documented_candidate` with an empty ACP command for backward
compatibility.

Do not reuse the existing `protocol`, `tool_execution_mode`,
`mcp_transport`, or MCP orchestration fields as ACP certification metadata.
Those fields describe how the workspace harness or MCP adapter talks to an
agent. `entrypoint_strategy`, `acp_command`, and `acp_args` describe the
downstream command that speaks Agent Client Protocol over stdio. An MCP stdio
transport such as `codex mcp-server` is not automatically an ACP entrypoint.

## Initial Target Classification

| Profile | Initial strategy | Intended ACP entrypoint | Current support state |
| --- | --- | --- | --- |
| OpenCode | `native_acp` | `opencode acp` | `supported_with_caveats` after May 23, 2026 macOS host backend live E2E; sandbox, MCP injection, artifact workflows, and reviewer loops remain unverified |
| Goose | `native_acp` | `goose acp` | `supported_with_caveats` after May 23, 2026 macOS host backend live E2E; sandbox, MCP injection, artifact workflows, and reviewer loops remain unverified |
| Hermes | `native_acp` | `hermes acp --accept-hooks` | `supported_with_caveats` after May 23, 2026 macOS host backend live E2E; sandbox, MCP injection, artifact workflows, and reviewer loops remain unverified |
| Codex CLI | `documented_candidate` | none seeded until an exact adapter command is selected | `documented_unverified` |
| Claude Code | `documented_candidate` | none seeded until an exact adapter command is selected | `documented_unverified` |
| Aider | `documented_candidate` | none known in repo evidence | `documented_unverified`; direct local llama.cpp prompting works, but no ACP-compatible stdio server entrypoint is available |
| Continue | `documented_candidate` | none known in repo evidence | `documented_unverified` |
| Custom | `custom_template` | operator supplied | `documented_unverified` |

These classifications should be revisited when upstream docs or local evidence
change. They should not upgrade support state on their own. Codex and Claude
should not be seeded as `adapter_acp` until the project chooses concrete
`acp_command` values such as a specific installed adapter binary. Once selected,
the adapter command can be represented as `adapter_acp` and certified like any
other ACP entrypoint.

## Classifier

Add a backend helper in the ACP registry layer that returns a normalized
classification for a registry entry. It should be small, deterministic, and
easy to test.

Inputs:

- Registry entry fields.
- Registry source, so YAML and API-backed entries are classified consistently.
- PATH command discovery result.
- Required API key presence, without exposing secret values.
- Optional custom profile fields.

Output shape:

```text
profile_key:
entrypoint_strategy:
probe_state: ready_to_probe | blocked | custom_template | documented_only
acp_command:
acp_args:
primary_blocker:
blockers:
status_message:
docs_url:
```

Probe states:

- `ready_to_probe`: strategy and ACP command are configured and the command is
  resolvable.
- `blocked`: the profile has a known ACP strategy but cannot be probed because
  a command, adapter, credential, or workspace prerequisite is missing.
- `custom_template`: the row is a template and requires a named custom profile.
- `documented_only`: the row is a candidate with no known ACP entrypoint.

The classifier does not run `initialize`; it only decides whether a probe can
be attempted and why not.

## Probe Manifest

Extend `Helper_Scripts/Testing-related/acp_certification_smoke.py` so it can
emit a profile-specific manifest. The profile manifest should reuse existing
live-e2e concepts and add entrypoint probes:

1. Resolve the ACP command from registry strategy metadata.
2. Emit version/discovery commands.
3. Emit a bounded ACP `initialize` probe for the selected `acp_command` and
   `acp_args`.
4. Emit `session/new` and `session/prompt` checks only when `initialize` passes
   and required runtime env is present.
5. Refuse to claim support when the profile is `documented_candidate`,
   `custom_template`, missing its adapter, or missing required live env.

The manifest should be useful without running live E2E. In dry-run mode it
should explain:

- what command would be executed;
- what blocker prevents execution;
- what evidence would be required to upgrade status.

## Evidence And Status Rules

Support state rules:

- `documented_unverified` means setup is documented but live support is not
  claimed.
- `live_e2e_tested` requires at least ACP `initialize`, `session/new`, and one
  prompt path success or a clearly classified auth-required response after the
  ACP session exists.
- `adapter_acp` is not weaker than `native_acp`; either can become supported if
  evidence passes.
- `custom_template` never becomes supported generically. Only a named custom
  profile can be certified.

Issue closeout rules:

- A PR that documents blockers does not close #1563 or #1564.
- A live-certification issue closes only when the named agent/profile has
  successful live evidence, or when the issue is explicitly replaced by narrower
  per-agent issues.
- If an issue acceptance criterion says "evidence or explicit blocker
  documented", it should be interpreted as completing the documentation slice,
  not completing live certification.

Failure taxonomy:

Reuse the existing caveat taxonomy and add only the missing entrypoint-specific
labels:

- `entrypoint_strategy_missing`
- `adapter_required`
- `adapter_missing`
- `acp_initialize_failed`
- `shell_builtin_collision`

## Setup/API/UI Surface Behavior

Setup-guide, health, agents, and Agent Registry surfaces should expose the same
model:

- Strategy: native, adapter, documented candidate, or custom template.
- Probe state: ready, blocked, documented-only, or template.
- Top blocker and next action.
- Link to compatibility docs and evidence.

Examples:

- OpenCode installed with `opencode acp` resolvable: "Ready to probe native ACP
  entrypoint."
- Goose not installed: "Native ACP command documented, but `goose` is missing."
- Codex without adapter: "Codex requires an ACP adapter; `codex` is not the ACP
  entrypoint."
- Custom profile: "Provide a named ACP command, args, env, workspace policy,
  and evidence bundle."

The UI should not display these rows as "supported" until the matrix support
state says so.

## Testing

Unit tests:

- Registry parsing accepts all strategy values.
- Missing strategy defaults to `documented_candidate`.
- `native_acp` and `adapter_acp` require `acp_command`.
- `custom_template` stays non-probeable without a concrete profile.
- Classifier distinguishes missing command, adapter required, adapter missing,
  shell builtin collision, and ready-to-probe states.
- Dynamic registration create/update/reload preserves strategy metadata through
  the API request schemas, registry persistence layer, and DB-backed reload.
- Existing API-backed registry rows without strategy columns remain loadable and
  classify as `documented_candidate`.

Helper tests:

- Profile-specific manifest includes `acp_command` and `acp_args`.
- Native profile manifest renders `opencode acp` / `goose acp` style commands.
- Adapter profile manifest renders adapter metadata and refuses when adapter is
  absent.
- Custom template manifest refuses and lists required custom evidence.
- Live-run mode refuses without `TLDW_E2E_SERVER_URL`, `TLDW_E2E_API_KEY`, and
  `ACP_AGENT_PROFILE`.

Integration/status tests:

- `/api/v1/acp/agents` and `/api/v1/acp/setup-guide` include strategy and
  blocker metadata.
- Dynamically registered agents expose the same strategy and blocker metadata
  as YAML-backed agents.
- Existing ACP health/status tests still pass.
- Compatibility matrix and `agents.yaml` stay aligned for support state,
  verification level, and strategy.

Docs verification:

- Matrix rows explain whether the profile is native, adapter-backed,
  documented-only, or custom.
- Release notes avoid live support claims until evidence exists.

## Staged Implementation Plan

Stage 1: Registry strategy schema and classifier

- Add strategy fields to registry entries, API schemas, dynamic registration
  persistence, and response serialization.
- Seed initial strategy metadata.
- Add classifier and focused tests.
- Do not run live agent commands.

Stage 2: Certification manifest profiles

- Extend the smoke helper with profile-specific dry-run manifests.
- Add bounded `initialize` probe command shape.
- Add tests for native, adapter, documented candidate, and custom profiles.

Stage 3: Setup/status surface alignment

- Add strategy and blocker metadata to setup-guide and agent list responses.
- Update frontend Agent Registry/setup copy if needed.
- Keep the compatibility matrix as the support claim source of truth.

Out of scope for this spec's first implementation plan:

- installing or configuring live downstream agents;
- certifying OpenCode, Goose, Codex, Claude, Aider, Continue, or a custom
  profile;
- closing #1563 or #1564;
- implementing Codex or Claude adapters.

Follow-on work after Stages 1-3:

- Native ACP certification: install/configure one native ACP target first,
  likely OpenCode or Goose, run the profile manifest, record evidence, and only
  then update the relevant matrix row.
- Adapter-backed certification: evaluate Codex and Claude adapters after the
  framework can represent them. Treat each adapter as its own certifiable
  command with version, source, and evidence.
- Issue decomposition: decide whether #1563 and #1564 should remain aggregate
  live-certification issues or split into narrower per-agent issues once
  profile manifests can produce per-agent evidence bundles.

## Open Questions

1. Should adapter-backed profiles include package-manager hints such as npm,
   pipx, or cargo package names in a first version?
2. Should #1563 and #1564 be split into per-agent issues after Stage 1, or only
   after profile manifests can produce per-agent evidence bundles?

## References

- ACP initialization protocol: https://agentclientprotocol.com/protocol/initialization
- OpenCode ACP docs: https://open-code.ai/en/docs/acp
- Goose ACP client docs: https://goose-docs.ai/docs/guides/acp-clients/
- Claude Code via ACP adapter example: https://zed.dev/blog/claude-code-via-acp
- Codex ACP adapter example: https://github.com/cola-io/codex-acp
- Current compatibility matrix: `Docs/Development/ACP_Compatibility_Matrix.md`
- Current certification checklist:
  `Docs/Development/ACP_Certification_Checklist.md`
