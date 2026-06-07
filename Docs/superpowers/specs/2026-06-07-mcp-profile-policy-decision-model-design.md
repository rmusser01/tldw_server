# MCP Profile Policy Decision Model And Shell Alias Hardening Design

Date: 2026-06-07
Status: Draft specification for TASK-2306
Branch: codex/mcp-profile-policy-decision-design

## Summary

Add a cross-cutting MCP/profile policy model built around explicit
`deny`, `ask`, and `allow` decisions. The decision model should become the
shared vocabulary for profile grants, tool catalog visibility, path grants,
external MCP server grants, pre-tool hooks, shell/run aliases, sandbox
assertions, approval flows, audit events, and effective-permission debugging.

This is a follow-up design slice, not an amendment to the completed safe file
tools implementation. The safe file tools branch added action-aware path grants
and guarded `fs.read`, `fs.patch`, and `fs.write`. This spec generalizes that
work into a broader policy layer and hardens adjacent surfaces such as
`run`, `bash`, `shell`, hooks, and external MCP wildcard grants.

## Source Feedback Incorporated

This spec incorporates senior review feedback that the MCP/profile design
should:

- add explicit permission outcomes: `deny`, `ask`, and `allow`;
- define named permission modes such as read-only, ask-by-default, accept-edits,
  locked, and sandboxed-auto;
- strengthen path-rule semantics with gitignore-style patterns, Windows
  normalization, and symlink target checks;
- treat shell aliasing as a parser problem, including independent checks for
  compound commands;
- handle shell wrappers conservatively;
- prefer dedicated structured tools over shell escape hatches;
- make hooks part of enforcement rather than only workflow automation;
- separate permission decisions from defense-in-depth sandboxing;
- support external MCP server/tool wildcard grants; and
- add effective-permission explanations for operators and debuggers.

## Related Existing Designs

- `Docs/superpowers/specs/2026-06-07-mcp-fs-patch-write-safe-edit-tools-design.md`
- `Docs/superpowers/specs/2026-06-07-mcp-tool-call-hooks-design.md`
- `Docs/superpowers/specs/2026-06-03-mcp-default-profile-tooling-presets-design.md`
- `Docs/superpowers/specs/2026-06-01-mcp-stdio-process-policy-design.md`
- `Docs/superpowers/specs/2026-06-06-mcp-tool-use-eval-reporting-design.md`

## Goals

- Define one permission-decision contract usable across package and host code.
- Preserve compatibility with existing `allowed_tools`, `denied_tools`, path
  grants, profile presets, and approval policy fields.
- Preserve compatibility with existing Claude-style command patterns such as
  `Bash(git *)` without treating them as raw shell authorization.
- Make deny precedence explicit and testable.
- Distinguish tool visibility from tool execution.
- Allow model-visible "ask" tools without silently granting execution.
- Make path grants expressive enough for realistic workspace policies while
  compiling to a flat effective policy for runtime enforcement.
- Harden the existing governed `run`, `bash`, and `shell` aliases against
  command-chain bypasses.
- Treat hooks as policy participants that can tighten but not loosen configured
  policy.
- Model sandbox state as a separate enforcement layer, not as a replacement for
  permissions.
- Provide an admin/debug explanation surface for tool, path, domain, external
  MCP, and shell-command decisions.

## Non-Goals

- No raw host shell execution.
- No broad bypass, superuser, or "YOLO" profile mode.
- No implementation in this spec-only branch.
- No UI implementation, though API/CLI surfaces are specified.
- No complete shell language implementation. The governed CLI parser remains a
  bounded virtual command language.
- No attempt to make string allowlists for arbitrary Bash safe. Dedicated tools
  remain the preferred surface for files, URLs, git, packages, tests, browser
  inspection, and deployment actions.
- No tldw_server-specific imports in the standalone `mcp_unified` policy core.
  Host adapters can expose FastAPI routes and persistence later.

## Local Design Review Adjustments

A local review pass found and incorporated four important clarifications:

- Rules need an explicit subject type so `fs.write`, `mcp__server__tool`, and
  `Bash(git *)` are not matched by the same ambiguous wildcard machinery.
- Existing `Bash(...)`/shell-style policy strings should compile into command
  rules for the governed virtual CLI only. They must never authorize raw host
  shell execution.
- Session-level overrides should only tighten the effective profile policy.
  They may convert `allow` to `ask` or `deny`, and `ask` to `deny`, but never
  the reverse.
- Ask-scoped tools should be visible by default only when the active profile
  intentionally exposes them. Large external catalogs should still use
  progressive disclosure rather than dumping all askable tools into context.

A second review pass added implementation-critical clarifications:

- `ask` must not be represented only as `canExecute=false`, because current
  virtual CLI catalog filtering treats `canExecute=false` as hidden. Askable
  tools need explicit approval metadata while remaining callable into the
  approval path.
- Sandbox/process assertions can downgrade `allow` to `ask` or `deny`, so they
  must run before final approval routing.
- Hooks need explicit denied-call semantics: blocking hooks do not run to
  loosen already-denied calls; managed observer hooks may receive redacted deny
  events for audit.
- Gitignore-style path policies compile to a flat matcher IR, not to enumerated
  filesystem paths.
- Command policy should use argv-token matchers rather than shell-string
  matchers.

## Core Decision Contract

The shared decision result should have exactly three outcomes:

```json
{
  "outcome": "deny",
  "reason_code": "tool_explicitly_denied",
  "subject": {
    "type": "tool",
    "normalized": "fs.write"
  },
  "matched_rules": [],
  "requires_approval": false,
  "visibility": "hidden",
  "call_state": "blocked",
  "explainable": true
}
```

Outcomes:

- `deny`: execution is blocked. Denied tools are hidden from normal model tool
  context unless an admin/debug caller explicitly asks for hidden entries.
- `ask`: execution requires an approval flow. The tool may remain visible to
  the model, but catalog metadata must indicate that execution is gated.
- `allow`: the policy layer has no objection. This is not sufficient if a
  sandbox, credential grant, path rule, hook, or runtime preflight later denies
  the call.

Catalog and call-state fields are derived from the outcome plus profile
visibility rules:

- `visibility`: one of `hidden`, `direct`, `deferred`, or `debug_only`.
- `call_state`: one of `blocked`, `approval_required`, or `callable`.
- `requires_approval`: `true` only when `call_state="approval_required"`.

Compatibility rule for existing `tools/list` payloads:

- `deny` should normally omit the tool from model-facing catalogs. If a debug
  surface includes it, `canExecute=false` is appropriate.
- `ask` should not rely on `canExecute=false` when the tool must remain visible.
  Use `canExecute=true` plus `requiresApproval=true` and
  `decision.outcome="ask"` so existing call paths can route into approval.
- `allow` uses `canExecute=true` and `requiresApproval=false`.

This avoids hiding askable tools from adapters such as the current virtual CLI,
which filters out tools where `canExecute` is false.

Precedence is always:

```text
deny > ask > allow
```

This applies across all merged inputs. If one matching rule says `deny`, no
other rule or hook can convert the decision to `ask` or `allow`. If no deny
matches but any applicable rule or hook returns `ask`, the final result is
`ask`. `allow` only wins when all applicable checks allow or abstain.

## Permission Modes

Profiles and sessions should be able to select a high-level mode that maps to
default decisions. The mode supplies defaults only; explicit denies always win.

### `plan/read-only`

- Read, search, inspect, list, describe, and planning tools may be `allow`.
- Mutating, process, browser-control, credentialed, network-write, memory-write,
  deploy, and destructive tools default to `deny`.
- Good default for architects, product owners, documentation planning, and code
  reviewers when no edits are intended.

### `default/ask`

- Read and low-risk inspect tools may be `allow`.
- File edits, test execution, browser interaction, package operations, external
  MCP calls with credentials, and memory writes default to `ask`.
- Destructive operations default to `deny` unless explicitly configured.
- Recommended default for most interactive agentic sessions.

### `accept-edits`

- Bounded edit tools such as `fs.patch` can be `allow` when profile grants and
  path grants allow the exact target.
- Whole-file writes, creates, package commands, process execution, and
  destructive actions remain `ask` unless explicitly allowed.
- Good fit for developer modes once a workspace root and path grants are bound.

### `locked/dontAsk`

- Runtime approval prompts are disabled.
- Any decision that would be `ask` becomes `deny`.
- Useful for unattended automation, CI, server-side ACP sessions, and hosted
  deployments where a model cannot ask a human interactively.

### `sandboxed-auto`

- Allows bounded operations only when both policy grants and sandbox assertions
  pass.
- If the runtime cannot prove sandbox isolation for the tool class, the decision
  degrades to `ask` or `deny` according to profile settings.
- This is the closest safe equivalent to "auto" mode and should not be treated
  as a bypass.

## Evaluation Pipeline

Tool-call preparation should produce an ordered policy trace:

1. Normalize subject identifiers: tool name, external MCP server/tool name,
   path candidates, domains, command invocations, and credential slots.
2. Resolve the effective profile and session policy.
3. Validate schema and derive path/command/external candidates needed for
   policy evaluation.
4. Evaluate server-wide disabled-tool settings and explicit deny rules.
5. If the call is already denied, skip blocking hooks and route only redacted
   observer/audit hooks that are configured for deny events.
6. Evaluate allow/ask rules by subject type.
7. Evaluate path, domain, credential, external MCP, and process constraints.
8. Run blocking pre-tool hooks that are allowed for this deployment.
9. If a hook rewrites input, re-run schema validation, candidate extraction,
   explicit deny checks, and scoped policy checks.
10. Check sandbox/process assertions for tools that spawn or delegate work.
11. Merge all decisions, applying `deny > ask > allow`.
12. Evaluate runtime approval if the final call state is `approval_required`.
13. Produce a final decision plus redacted explanation metadata.

Rules should be deterministic and stable. The policy trace should record enough
detail to explain a decision without storing raw file content, raw diffs,
secrets, full environment values, or absolute host paths unless the caller is an
authorized local admin and the endpoint is explicitly in debug mode.

## Compatibility With Current Policy Fields

Existing fields remain valid:

- `allowed_tools`
- `denied_tools`
- `capabilities`
- `approval_policy`
- `path_grants`
- `path_allowlist_prefixes`
- `external_server_grants`
- `credential_grants`

Compatibility mapping:

- `denied_tools` compiles to tool rules with `outcome="deny"`.
- `allowed_tools` compiles to tool rules with `outcome="allow"` unless a
  profile mode or approval policy upgrades the operation to `ask`.
- Existing command-pattern entries such as `Bash(git *)` compile to command
  rules for `run`/`bash`/`shell` aliases. They do not grant a raw Bash tool and
  do not match across command separators such as `&&`, `||`, `;`, or `|`.
- Existing approval policies compile to `ask` rules or mode-level defaults.
- Existing path grants with `effect="deny"` compile to `deny`; grants with
  `effect="allow"` compile to `allow`.
- New `outcome` fields supersede legacy `effect` fields when both are present,
  but conflicting values should fail validation rather than silently choosing
  one.
- If no decision model fields are present, existing behavior remains the
  compatibility fallback.

New policy documents may add a structured rule list:

```json
{
  "permission_mode": "default/ask",
  "tool_rules": [
    {"pattern": "fs.read", "outcome": "allow"},
    {"pattern": "fs.patch", "outcome": "ask"},
    {"pattern": "fs.write", "outcome": "deny"}
  ],
  "command_rules": [
    {"argv": ["git", "status"], "outcome": "allow"},
    {"argv": ["git", "*"], "outcome": "ask"},
    {"argv": ["rm", "*"], "outcome": "deny"}
  ],
  "mcp_rules": [
    {"pattern": "mcp__github", "outcome": "ask"},
    {"pattern": "mcp__github__*", "outcome": "ask"},
    {"pattern": "mcp__github__delete_repo", "outcome": "deny"}
  ]
}
```

## Tool Visibility Semantics

`tools/list` and progressive-disclosure surfaces should distinguish visibility
from execution:

- `deny`: hidden from normal model catalog and tool search.
- `ask`: visible only when directly granted or explicitly promoted, with
  `requiresApproval=true`, `decision.outcome="ask"`, and a call path that
  routes into approval.
- `allow`: visible and executable if all other constraints pass.

Default visibility rule:

- direct profile tools that resolve to `ask` may remain visible with approval
  metadata;
- external, plugin, or large category tools that resolve to `ask` should stay
  behind profile-scoped search/category disclosure unless explicitly promoted;
  and
- denied tools stay hidden outside admin/debug views.

Existing clients that only understand `canExecute` need a compatibility rule:
`canExecute=false` means hidden/blocked, not "askable". Askable direct tools
should keep `canExecute=true` and add approval metadata. Newer clients should
prefer `decision.outcome` and `call_state` over inferring behavior from
`canExecute` alone.

Admin/debug surfaces may include hidden denied tools when requested, but only
with explanation metadata and without sensitive tool arguments.

This distinction matters because a model should not waste context on tools it
can never call, but it should understand when a useful scoped tool exists and
requires approval.

## Path Rule Semantics

The executable runtime should continue consuming a flat effective grant list or
matcher list, but the authored policy layer should support richer patterns that
compile into a flat matcher IR. The compiler must not expand `**` patterns into
filesystem enumerations.

Recommended authored shape:

```json
{
  "path_policy_version": 2,
  "path_grants": [
    {"pattern": "/documents/**", "actions": ["read", "edit", "write"], "outcome": "allow"},
    {"pattern": "/documents/private/**", "actions": ["edit", "write"], "outcome": "deny"},
    {"pattern": "/downloads/**", "actions": ["read"], "outcome": "allow"}
  ]
}
```

Pattern rules:

- Patterns are workspace-relative by default.
- Leading `/` anchors to the workspace root.
- `**` matches across path segments; `*` matches within one segment.
- Bare directory patterns should be normalized to their subtree form only when
  explicitly documented by the compiler.
- Parent traversal, drive-qualified paths, UNC roots, and absolute host paths
  are rejected in authored grants.
- Windows separators are normalized to `/` for policy matching.
- Windows drive and UNC inputs are denied unless they are translated by a
  trusted workspace-root resolver into a workspace-relative path.
- Matching is segment-aware and must not let `docs` match `docs-private`.

Compiled matcher IR should preserve:

- normalized pattern;
- action set;
- outcome;
- source profile/policy id;
- anchor mode;
- case-folding mode for the target platform;
- original authored pattern for explanations; and
- validation warnings, if any.

Symlink rules:

- For reads and writes, evaluate both the symlink path as requested and the
  resolved target path.
- Allow only if both paths are allowed for the requested action.
- Deny if either path matches a deny rule.
- Deny if the target cannot be resolved safely.
- Deny if the resolved target leaves the trusted workspace/root set.
- Normal explain payloads may include only redacted or workspace-relative
  symlink target summaries, never absolute host target paths.
- Mutating tools should continue rejecting symlink targets unless a future
  design adds a tightly scoped symlink-follow mode.

## External MCP Server And Tool Wildcards

External MCP tools should have normalized names that are safe to match:

- `mcp__server` means the server as a subject, for lifecycle/discovery grants.
- `mcp__server__*` means every exposed tool for that server.
- `mcp__server__tool` means one concrete external tool.

Canonicalization:

- server ids and tool ids are lowercased for matching;
- spaces and non-identifier separators normalize to `_`;
- repeated separators collapse to one `_`;
- literal `__` in upstream ids is escaped before joining server/tool segments;
- collisions fail closed and require an explicit admin alias; and
- the original upstream display name is preserved only for UI/explain output.

Rules:

- Exact tool denies win over server or wildcard allows.
- Server-level grants are not credential grants.
- Tool grants are not install/start grants.
- Installation, runtime start, credential use, and tool execution are separate
  subjects that can each be `deny`, `ask`, or `allow`.
- Wildcards should be simple and explicit. Avoid regex unless the pattern is
  prefixed with `re:` and accepted by an admin-only validation path.

## Session Overrides

Session policy may narrow a profile but should not widen it. This lets an ACP
session, temporary workspace, or frontend mode reduce risk without mutating the
profile preset.

Allowed session changes:

- convert `allow` to `ask`;
- convert `allow` or `ask` to `deny`;
- reduce visible categories;
- reduce path/domain/MCP server scope;
- require approval for a normally automatic tool; and
- require stronger sandbox assertions.

Rejected session changes:

- convert `deny` to `ask` or `allow`;
- convert `ask` to `allow`;
- add paths, domains, tools, credential slots, or MCP servers not present in
  the profile's effective policy; and
- disable managed hooks or sandbox requirements.

## Hooks As Enforcement

Pre-tool hooks participate in policy after schema validation and candidate
extraction, before approval and execution.

Hook decisions:

- `deny`: blocks execution and wins over `ask` or `allow`.
- `ask`: requires approval unless the session is `locked/dontAsk`, in which
  case it becomes `deny`.
- `allow`: means the hook has no objection. It does not authorize the call by
  itself.

Hooks may tighten policy but cannot bypass:

- explicit deny rules;
- profile grants;
- path grants;
- credential grants;
- external MCP grants;
- sandbox/process policy; or
- server-wide disabled-tool settings.

A blocking hook failure should be configurable as fail-closed by default for
managed hooks and fail-open only for explicitly non-critical observer hooks.

Denied-call behavior:

- Blocking hooks do not run to reconsider calls already denied by explicit
  policy, server-wide disablement, or path/credential/external grants.
- Managed observer hooks may receive redacted deny events for audit and
  metrics, but their output cannot change the final decision.
- Hook rewrites create a new candidate set and restart scoped evaluation; the
  original and rewritten candidates should both be represented in explain
  metadata using redacted summaries.

## Shell Alias And Virtual CLI Hardening

The existing `run` module exposes `run`, `bash`, and `shell` as governed virtual
CLI entry points. They are not raw host shells. This design keeps that stance
and hardens the alias surface.

Rules:

- Parse command text into a command chain before authorization.
- Split compound commands and evaluate every subcommand independently.
- `safe-cmd && dangerous-cmd` must not pass because `safe-cmd` is allowed.
- Pipelines require every governed command and pure transform in the pipeline to
  be visible and allowed.
- Command aliases inherit the strictest decision of their expanded commands.
- Preflight should prepare every governed step before execution starts when a
  chain contains any governed step.
- Unknown commands return a bounded error rather than falling through to a host
  shell.

Legacy `Bash(...)` pattern rules:

- `Bash(git *)` compiles to command argv pattern `["git", "*"]` for governed
  `run`/`bash`/`shell` aliases.
- The pattern applies to one parsed command invocation, not the whole raw
  command string.
- `Bash(git *)` may match `git status` but not
  `git status && rm -rf build`, because the `rm` invocation is evaluated
  independently.
- `Bash(*)` should be rejected in hosted/default profiles and allowed only in
  tightly sandboxed local development modes with an explicit admin override.

Wrapper handling:

- Conservative unwrapping may support wrappers such as `timeout`, `time`, and
  `nice` only when their option grammar is explicitly modeled.
- Wrappers that can hide arbitrary execution should not be unwrapped by default:
  `docker exec`, `podman exec`, `npx`, `bunx`, `uvx`, `devbox run`,
  `nix develop --command`, `ssh`, `sudo`, language REPL runners, and package
  manager script runners.
- If a wrapper is not modeled, treat the wrapper itself as the requested
  command and deny it unless a dedicated structured tool exists.

Dedicated tools should be preferred for:

- file reads/writes/patches;
- URL fetch/search;
- git status/diff/log;
- package installs and scripts;
- test execution;
- browser/CDP inspection;
- CI/CD operations; and
- deployment actions.

Shell-style string matching is too fragile for these behaviors.

Command matcher grammar:

- Match parsed argv tokens, not raw command strings.
- `*` matches one argv token, not arbitrary text.
- `**` is not supported for commands in the first slice.
- Quoted tokens are compared after parsing and normalization.
- Environment assignment syntax, command substitution, shell functions,
  redirection, heredocs, glob expansion, variable expansion, and subshells are
  rejected unless a future parser explicitly models them.

## Canonical Risk Classes

Permission modes should use canonical risk classes from tool metadata rather
than free-form labels.

| Risk class | Typical tools | Default posture |
| --- | --- | --- |
| `read` | `fs.read`, search/list/describe tools | allow in most modes |
| `bounded_edit` | `fs.patch` on existing files with preimage checks | ask by default, allow in `accept-edits` with path grants |
| `whole_write` | `fs.write`, file create/replace | ask or deny unless explicitly granted |
| `destructive` | delete, purge, reset, revoke, deploy rollback | deny unless explicitly granted |
| `process` | test runners, package commands, external process launch | ask unless sandboxed and explicitly granted |
| `network_read` | web fetch/search, remote metadata reads | ask or allow based on provider configuration |
| `network_write` | issue creation, API mutation, CI/deploy actions | ask or deny unless explicitly granted |
| `credentialed` | calls using stored credential grants | ask unless the grant and profile explicitly allow |
| `browser_control` | CDP navigation, clicking, form entry | ask unless scoped to inspection/read-only |
| `memory_write` | Graphiti/memory mutation tools | ask or deny unless role explicitly owns memory writes |
| `admin` | profile, grant, server, secret, snapshot mutation | deny outside admin profiles |

Tools may carry multiple risk classes. The most restrictive resulting decision
wins.

## Sandbox Semantics

Permissions and sandboxing solve different problems:

- Permissions decide whether a model may request a tool/action.
- Sandboxing constrains what spawned or delegated processes can actually touch.

For process-like tools, final `allow` should require both:

- policy allow for the requested tool/action; and
- a sandbox/process assertion compatible with the requested risk class.

Example assertions:

```json
{
  "sandbox": {
    "filesystem": "workspace-write",
    "network": "restricted",
    "process": "no-shell",
    "cwd": "workspace-root",
    "allowed_roots": ["."]
  }
}
```

If the runtime cannot prove the necessary sandbox state, `sandboxed-auto`
degrades to `ask` or `deny`.

## Effective-Permission Explanation

Add an admin/debug surface that can answer:

- Why is this tool visible or hidden?
- Why is this tool call allowed, denied, or asking?
- Why is this path/domain/external MCP tool allowed or denied?
- Which profile, grant, hook, sandbox assertion, or deny rule decided it?

Suggested CLI/API:

```text
mcp policy explain --profile backend-engineer --tool fs.patch --path src/app.py --action edit
mcp policy explain --profile qa-engineer --tool bash --command "git status && rm -rf build"
mcp policy explain --profile devops --mcp-tool mcp__github__create_issue
```

Response shape:

```json
{
  "final_outcome": "deny",
  "reason_code": "command_step_denied",
  "profile_id": "qa-engineer",
  "permission_mode": "default/ask",
  "subject": {
    "type": "command",
    "normalized": "rm",
    "step_index": 1
  },
  "matches": [
    {
      "source": "profile.denied_tools",
      "rule_type": "tool",
      "pattern": "rm",
      "outcome": "deny"
    }
  ],
  "hook_results": [],
  "sandbox": {
    "required": "no-host-shell",
    "status": "satisfied"
  }
}
```

Redaction requirements:

- no raw file content;
- no raw diffs;
- no read receipts;
- no credential values;
- no full environment values;
- no absolute host paths in normal responses;
- bounded command previews; and
- explicit `redacted=true` markers where fields are omitted.

## Observability And Evaluation

Tool-use reporting should record safe policy metadata for every tool family:

- final outcome;
- reason code;
- permission mode;
- normalized subject type;
- matched rule source/type;
- whether a hook participated;
- whether approval was required;
- sandbox assertion status;
- redaction status; and
- elapsed time.

This should apply to all tools, not just filesystem or git tools. The metadata
should support evaluations of whether models select the right tool, understand
approval boundaries, recover from denies, and avoid shell escape hatches when a
dedicated tool exists.

## Implementation Slices

### Slice 1: Decision Core And Explain Contract

- Add package-level decision dataclasses/Pydantic models.
- Include `outcome`, `visibility`, `call_state`, `requires_approval`,
  `matched_rules`, and redaction metadata in the core response schema.
- Compile existing profile fields into decision rules.
- Add `deny > ask > allow` merge tests.
- Add explain response schema and a package-local simulation API.
- Keep runtime behavior compatibility unless the new explain path is invoked.

### Slice 2: Catalog Visibility And Permission Modes

- Add `permission_mode` handling.
- Update tool listing/search/category surfaces to hide denied tools and mark
  ask-scoped tools.
- Preserve visible askable tools with `canExecute=true`,
  `requiresApproval=true`, and `decision.outcome="ask"` so existing adapters do
  not hide them accidentally.
- Add regression coverage for `locked/dontAsk` converting ask to deny.

### Slice 3: Path Pattern Compiler

- Add authored gitignore-style path patterns that compile to flat matcher IR.
- Preserve current flat grants as executable runtime input.
- Add Windows normalization and symlink double-evaluation tests.
- Add tests for unresolved symlinks, out-of-workspace symlink targets, and
  redacted explain payloads.

### Slice 4: External MCP Wildcard Grants

- Normalize `mcp__server`, `mcp__server__*`, and `mcp__server__tool` matching.
- Keep server install/start, credential use, and tool execution as separate
  subjects.
- Add deny-over-wildcard-allow tests.
- Add canonicalization and collision tests for server/tool ids containing
  spaces, case differences, separators, and literal `__`.

### Slice 5: Shell Alias Hardening

- Extend parser/preflight tests for command chains and wrappers.
- Replace command-string matching with argv-token policy rules.
- Add conservative wrapper modeling for `timeout`, `time`, and `nice` only if
  their grammars are small enough to validate.
- Deny or recommend dedicated tools for high-risk wrappers and package runners.
- Add regression tests for `Bash(git *)` matching `git status` but not
  `git status && rm -rf build`.

### Slice 6: Hooks Enforcement Integration

- Make hook outputs use the shared decision model.
- Ensure hook `allow` cannot bypass explicit denies.
- Include hook outcomes in explain and tool-use reporting.
- Add denied-call observer hook behavior and fail-closed managed hook tests.

## Open Questions

1. Which direct profile tools should be promoted as visible askable tools versus
   deferred behind category/search disclosure?
2. Should the first explain API live in the standalone package CLI only, or also
   expose a FastAPI admin route in the same PR?
3. Where should tightening-only session overrides be stored and audited:
   request metadata, session records, profile assignments, or ACP session
   state?
4. How much shell wrapper support is worth shipping before dedicated git/test
   and package-manager tools make most wrappers unnecessary?

## Risks And Mitigations

- **Policy model churn:** keep old fields supported and compile them into the
  new model rather than replacing them in one step.
- **Overexposing ask tools:** make catalog visibility configurable and default
  conservative for hosted deployments.
- **Shell parser false confidence:** keep the language small, reject unknown
  syntax, and prefer dedicated tools over Bash-like matching.
- **Path pattern ambiguity:** document anchors and segment matching precisely,
  and add property-based path normalization tests.
- **Hook bypass bugs:** enforce precedence in one shared merge function and test
  every hook/profile/path combination against it.
- **Sandbox drift:** record sandbox assertions separately so an allow decision
  cannot be mistaken for an OS-level containment guarantee.

## Definition Of Ready For Implementation

- The owner chooses the first implementation slice.
- The spec review confirms no cross-slice dependency is missing.
- Backlog tasks are split so each PR can be reviewed independently.
- Tests are defined before each slice starts.
