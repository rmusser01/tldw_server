# MCP Default Profile Tooling Presets Design

Date: 2026-06-03
Status: Spec review approved; draft PR placeholder pending active-work reconciliation
Backlog: TASK-2232, TASK-2233, TASK-2234

## Summary

The standalone MCP gateway and tldw_server MCP/ACP workspace flows should ship
with role-oriented default profile presets that describe both immediate tool
authority and setup-dependent recommended tools. The presets should use a
hybrid model: stable abstract capabilities plus a vendor-neutral catalog of
recommended concrete tool and MCP-server bindings.

The default posture is conservative. Read, search, inspect, and discovery tools
can be enabled per role. Mutating work, credentialed calls, browser
interaction, test execution, deployment changes, memory writes, and any process
execution require explicit approval and workspace binding. Arbitrary shell,
SSH, debugger control, REPL kernels, CI mutation, and deployment mutation are
not enabled by default; they appear as recommended inactive capabilities where a
role needs them.

The design also adds progressive disclosure as a first-class profile contract.
Small, high-frequency tools are exposed directly. Large native, external, or
plugin catalogs are hidden behind profile-scoped category/search/describe/call
bridge tools so frontends and agents do not receive an unbounded tool list.

## Placeholder PR Scope

This spec is ready to serve as a draft PR anchor, but it is intentionally not
the final implementation contract yet. Active MCP/ACP gateway work may still
change runtime names, admin surfaces, permission-routing details, or external
server lifecycle APIs. The draft PR should keep this design visible for review
while those dependencies settle.

Before implementation planning begins, the owner should do a reconciliation
pass and update this spec if any of these inputs changed:

- Profile preset model fields or validation rules.
- Gateway tool catalog, discovery, or progressive-disclosure APIs.
- External server registry, install/update, runtime lifecycle, or credential
  grant surfaces.
- ACP workspace binding, approval callback, or permission-routing contracts.
- Browser inspection, safe test runner, git inspection, or LSP native tool
  implementation choices.
- Supported risk classes, approval policy semantics, or audit-event schema.
- Concrete external MCP install targets that graduate from category
  placeholders to exact package targets.

## Context

The `mcp_unified` package already has:

- `MCPProfile` and `ProfilePolicy` models.
- Built-in role-like profile presets in `mcp_unified/profiles/presets.py`.
- Profile resolution and profile-aware runtime enforcement.
- Profile, assignment, external-server, credential-grant, snapshot, and audit
  stores.
- External MCP server registry/lifecycle/admin surfaces.
- Package-local documentation and CLI commands for listing and duplicating
  presets.

The current presets mostly define capability strings and broad risk classes.
They do not yet describe the concrete tool families, external binding
categories, progressive disclosure behavior, or default-included native tools
needed for serious agentic workspace operation.

The tldw_server host already has a useful native foundation:

- Filesystem module with `fs.list`, `fs.read_text`, `fs.write_text`.
- CodeGraph module with status, indexing, symbol search, graph traversal, and
  context tools.
- Knowledge, media, notes, chats, prompts, and characters search/get tools.
- Kanban/workflow tools that can act as a native issue/story surface.
- External federation through `external.servers.list`,
  `external.tools.refresh`, and `ext.<server>.<tool>`.
- Gateway admin surfaces for external servers, credential grants, snapshots,
  and runtime lifecycle.

The next design step is to decide what tool substrate profiles draw from, what
is enabled by default for each role, what is recommended but inactive, and how
frontends expose those choices safely.

## Prior-Art Review

The reviewed agent harnesses point to the same conclusion: effective coding
agents need a workspace operating surface, not only a handful of generic MCP
tools.

- `SafeRL-Lab/cheetahclaws` exposes tool modules for browser, diagnostics,
  files/filesystem, interaction, notebook, research, security, shell, and web.
  Its repository also has memory, checkpoint, jobs, monitor, multi-agent,
  plugin, research, skill, task, bridge, and MCP areas.
- `can1357/oh-my-pi` has source areas for edit, eval, exec, DAP/debug, LSP,
  MCP, memories, memory backends, modes, plan mode, secrets, session, SSH,
  task, tool discovery, web, and workspace-tree behavior.
- `tanbiralam/claude-code` exposes a broad tool directory including Bash,
  PowerShell, file read/write/edit, glob, grep, LSP, MCP resource access,
  browser, web fetch/search, task/todo/team tools, plan/worktree tools, REPL,
  cron/remote triggers, monitoring, notifications, skills, and workflow tools.

These references support a full agent-harness catalog, but they do not imply
that all tools should be enabled for all roles. The catalog should be broad;
profiles should be narrow.

The progressive-disclosure references add a second important lesson:

- Hermes Tool Search keeps core tools direct and defers large MCP/plugin tool
  catalogs behind a search flow when the visible tool count crosses a threshold.
- Matthew Kruczek's progressive-disclosure MCP article compares two-stage
  tool use, category/Strata-style disclosure, semantic search, tree browsing,
  and skills. For tldw_server, the best fit is hybrid category navigation plus
  grant-aware BM25 search.

Sources:

- https://github.com/SafeRL-Lab/cheetahclaws
- https://github.com/can1357/oh-my-pi
- https://github.com/tanbiralam/claude-code
- https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-search
- https://matthewkruczek.ai/blog/progressive-disclosure-mcp-servers.html

## Goals

- Define default role profile presets for:
  - Product Owner
  - Architect
  - Merge Conflict Resolver
  - Documentation Writer
  - Project Researcher
  - Code Reviewer
  - DevOps Engineer
  - Backend Engineer
  - Frontend Engineer
  - QA Engineer
  - Software Development Engineer in Test
- Keep defaults vendor-neutral while surfacing recommended concrete bindings.
- Split immediately enabled tools from recommended inactive tools/servers.
- Add progressive-disclosure metadata for category browsing and profile-scoped
  tool search.
- Identify native/default-included tool additions needed for tldw_server MCP
  and ACP workspaces.
- Preserve the existing `MCPProfile` model shape where possible by placing
  richer setup metadata under `profile.metadata["tooling"]`.
- Maintain conservative approvals and auditability for all risky behavior.

## Non-Goals

- Do not implement the tool modules in this spec.
- Do not choose a single vendor baseline for issues, repo hosting, CI/CD,
  browser automation, cloud, or docs publishing.
- Do not grant arbitrary shell/process access through default profiles.
- Do not make recommended inactive tools callable.
- Do not embed credential secret values in profiles, grants, snapshots, or
  recommendations.
- Do not redesign ACP itself; ACP sessions should receive the same
  profile-filtered catalog and route execution through ACP permission flows
  where applicable.

## Design Principles

1. **Broad catalog, narrow profiles.** tldw_server can know about many tools,
   but each role sees only the enabled and recommended subset allowed by policy.
2. **Native where mature, external where vendor-specific.** Existing nearby
   tldw_server modules should become native tools when practical. Jira, Linear,
   GitHub, GitLab, cloud providers, CI systems, and specialized browsers should
   be external bindings.
3. **Recommended does not mean authorized.** Recommended tools and servers are
   setup guidance until an operator installs, binds, grants, and enables them.
4. **Execution is constrained by abstraction.** Tests, git, browser, CI, and
   deploy tools must not tunnel into arbitrary shell.
5. **External network is a risk even when read-only.** Web search/fetch is
   useful for several roles, but packaged defaults should expose it as
   recommended unavailable until a native configured provider or external
   binding supplies explicit provenance and grants.
6. **Progressive disclosure is policy-scoped.** Tool search and category
   browsing only expose tools visible to the active profile/session.
7. **Workspace binding is required for scoped writes and test execution.**
8. **Audit covers denies, approval requests, approval decisions, and execution.**

## Decision Status

The following decisions are settled enough for the placeholder PR:

- Use hybrid profile metadata: abstract capabilities for policy plus
  vendor-neutral concrete binding recommendations for setup guidance.
- Keep packaged defaults conservative. Writes, browser interaction, process
  execution, credential use, deployment mutation, memory writes, and git
  mutation require explicit enablement and approval.
- Treat recommended tools and servers as non-authoritative display/setup
  metadata until an operator installs, grants, and enables them.
- Use progressive disclosure through profile-scoped category, search, describe,
  and bridge-call tools.
- Prefer native/default-included tools for mature tldw_server workspace
  primitives and external bindings for vendor-specific systems.
- Keep web search/fetch recommended unavailable in packaged defaults until a
  configured provider or external binding supplies grants and provenance.
- Represent issue/story output through native Kanban/cards and markdown first,
  with Jira/Linear/GitHub/GitLab as optional external bindings.
- Use CDP as the first browser-inspection path.
- Rank tool-search results without semantic search in the first version:
  profile grants filter first, then installation status, then category filters,
  then BM25 matching.
- Manage profile recommendation catalog metadata separately from executable
  policy so operators can patch recommendations without granting authority.
- Treat `ChromeDevTools/chrome-devtools-mcp` as the first known exact external
  MCP install target for the browser/CDP category.

The following decisions remain provisional and must be rechecked before code
implementation:

- Exact public schemas and names for discovery, category, and bridge-call
  tools.
- Safe test runner command-definition source and workspace trust boundary.
- Exact external MCP install targets outside the browser/CDP category.

## Default Tool Substrate

The default catalog should be a full agent-harness catalog with explicit
activation state and maturity labels. The initial native/default candidates are
listed below.

| Tool family | Default posture | Notes |
| --- | --- | --- |
| Filesystem read/list/write | Enabled by role | Existing `fs.list`, `fs.read_text`, `fs.write_text`; add edit/search helpers. |
| Code search/intelligence | Enabled by role | Existing CodeGraph; add LSP tools later. |
| Git inspect | Enabled for engineering/review roles | Status, diff, log, blame, branches, conflicts. |
| Git mutate | Recommended inactive | Add, commit, merge, rebase, conflict write require enablement and approval. |
| Native Kanban/tasks | Enabled for Product Owner | Out-of-box issue/story surface. |
| Markdown/doc write | Enabled by PO/docs/engineering roles | Workspace-scoped and approval-gated. |
| Browser inspect | Enabled for QA/Frontend where CDP is available | Start with CDP for screenshots, DOM snapshot, console/network read. |
| Browser interact | Approval-gated | Navigation, click, type, select, and state mutation. |
| Safe test runner | Enabled for Backend/Frontend/QA/SDET | Project-configured commands only, approval required. |
| Shell/process | Not enabled by default | Recommended inactive for engineering/devops only. |
| Web search/fetch | Recommended unavailable unless configured | No vendor baseline. A configured native/provider binding can enable read-only access with `external_network` provenance and grants. |
| Issue tracker integrations | Recommended external | Jira, Linear, GitHub Issues, GitLab Issues, etc. |
| CI/CD/deploy/cloud | Recommended external | DevOps inspect can be enabled when bound; mutations require approval. |
| Memory recall | Enabled read-only | All roles can read memory/context. |
| Memory write | Recommended inactive | Memory Keeper or approved reflection only. |
| Checkpoint/rewind | Enabled for mutating engineering roles | Session safety surface. |
| Subagents/tasks/plans | Enabled by orchestrating roles | Bounded by profile and approval. |
| Debug/DAP/REPL/SSH | Recommended inactive | High risk; explicit enablement required. |

## Progressive Disclosure Contract

Profiles should expose tools in layers.

1. Core direct tools stay visible for most profiles:
   `tool_categories.list`, `tool_search`, `tool_describe`, `profile.tools.list`,
   file read/list, ask-user, todo, and memory recall.
2. Role-enabled direct tools remain visible while the direct tool count stays
   below the profile threshold.
3. Large catalogs move behind bridge tools:
   - `tool_search(query, category?, limit?)`
   - `tool_describe(tool_id)`
   - `tool_call(tool_id, arguments)`
4. Frontends browse category groups such as Files, Code Search, Git, Browser,
   Issues, CI/CD, Memory, Tests, Docs, External Servers, and DevOps.
5. Discovery is profile-scoped. A Product Owner cannot discover SSH, debugger,
   deploy, or shell tools because they exist globally.
6. Search and describe are read-only. They must not execute actions, consume
   credentials, call external APIs for side effects, or bypass approval.
7. `tool_call` is a public MCP bridge tool, but only for profiles that have
   deferred categories. It is exposed directly with the other discovery tools
   when `progressive_disclosure.deferred_categories` is non-empty; otherwise it
   may be omitted from the direct catalog.
8. `tool_call` has a small fixed schema with `additionalProperties: false`. It
   accepts only a `tool_id` string returned by `tool_search` or `tool_describe`
   plus an `arguments` object:

   ```json
   {
     "tool_id": "ext.issue_tracker.jira.issue.create",
     "arguments": {"title": "Example"}
   }
   ```

   `tool_id` and `arguments` are required. The bridge never trusts the caller's
   category label. It resolves `tool_id` to the underlying tool name,
   capability set, server binding, activation status, and risk metadata before
   dispatch.
9. `tool_call` must return `tool_not_enabled` for recommended inactive tools and
   `tool_not_found` for tools outside the active profile's discovery scope.
   Approvals, path scope, credential grants, external-server grants, and audit
   run against the resolved underlying tool, not against the generic bridge
   name.

### Tool Search Ranking

The first implementation should not use semantic search for tool discovery.
Ranking must be deterministic and policy-first:

1. Filter to tools visible under the active profile grants and workspace
   assignment.
2. Partition by installation status so installed/enabled tools rank ahead of
   recommended-but-unavailable tools.
3. Apply category filters when supplied.
4. Run BM25 text matching over tool id, display name, category, description,
   capability labels, and unavailable reason metadata.
5. Return stable tie-breaks by category priority and tool id.

This order keeps authorization and setup state ahead of text relevance. BM25
improves findability inside the already-allowed result set; it must not expose
tools outside the active profile's discovery scope.

## Profile Metadata Schema

The existing `MCPProfile` and `ProfilePolicy` fields should remain the
enforcement source of truth. Rich tooling setup metadata should live under
`profile.metadata["tooling"]`.

Example:

```json
{
  "enabled_tools": ["fs.list", "fs.read_text", "codegraph.search"],
  "enabled_capabilities": ["filesystem.read", "codegraph.read"],
  "recommended_tools": [
    {
      "id": "browser.inspect",
      "category": "browser",
      "status": "native_candidate",
      "activation": "requires_browser_runtime"
    }
  ],
  "recommended_servers": [
    {
      "category": "issue_tracker",
      "required": false,
      "binding_options": [
        {
          "id": "jira",
          "kind": "external_mcp",
          "install_target": null,
          "credential_slots": ["jira_api_token"]
        },
        {
          "id": "linear",
          "kind": "external_mcp",
          "install_target": null,
          "credential_slots": ["linear_api_key"]
        }
      ]
    }
  ],
  "progressive_disclosure": {
    "direct_categories": ["files", "tool_discovery", "memory"],
    "deferred_categories": ["issue_tracker", "ci_cd", "browser"],
    "max_direct_tools": 24
  }
}
```

Field meanings:

- `policy_document.allowed_tools`: immediately callable native tools.
- `policy_document.capabilities`: enforceable capability grants.
- `policy_document.risk_classes`: risk labels used by policy and approval.
- `metadata.tooling.enabled_tools`: frontend-friendly mirror of direct tools.
- `metadata.tooling.recommended_tools`: setup-dependent native candidates or
  externally supplied tools.
- `metadata.tooling.recommended_servers`: vendor-neutral binding categories
  with concrete options.
- `external_server_grants`: actual executable external-server grants after a
  binding is installed.
- `credential_grants`: broker metadata only; never secret values.
- `approval_policy`: required approvals for writes, execution, credentials,
  browser mutation, deployment mutation, and similar risks.

Recommended tools and servers must not grant authority. They are display and
setup metadata until converted into real profile policy/grants by an operator.

## Role Preset Matrix

| Profile | Enabled by default | Recommended inactive | External binding categories |
| --- | --- | --- | --- |
| Product Owner | File read/search/write markdown, native Kanban/cards, docs search, memory recall, tool discovery | Browser inspect, web fetch/search, issue tracker create/update | `issue_tracker`, `web_search`, `docs_search`, `browser` |
| Architect | File read/search, CodeGraph read/context, docs search, git inspect, memory recall, tool discovery, subtask planning | LSP full graph, diagram generation, web research | `repo_host`, `docs_search`, `diagram`, `web_search` |
| Merge Conflict Resolver | File read, git status/diff/conflict inspect, CodeGraph read, scoped file patch/edit, checkpoint | Git add/commit/rebase/merge, test runner | `repo_host`, `git_provider`, `test_runner` |
| Documentation Writer | File read/search/write markdown, docs search, memory recall, tool discovery | Browser inspect, web fetch/search, diagram generation, docs publishing | `docs_search`, `web_search`, `browser`, `diagram`, `cms_docs_publisher` |
| Project Researcher | File read/search, CodeGraph read/context, knowledge/media/notes/prompts/chats search, docs search, memory recall | Browser inspect, web fetch/search, deep research/citation tools | `web_search`, `docs_search`, `browser`, `citation_manager` |
| Code Reviewer | File read/search, CodeGraph read/context, git diff/log/blame inspect, test result read, memory recall, tool discovery | PR review/comment tools, LSP diagnostics, CI read | `repo_host`, `pr_review`, `ci_cd`, `lsp` |
| DevOps Engineer | Config file read, git inspect, native logs/service/runtime status, memory recall, tool discovery | CI/CD read, deploy/restart/rollback, cloud resource mutation, SSH, shell | `ci_cd`, `cloud_provider`, `ssh`, `secrets_manager`, `observability` |
| Backend Engineer | File read/search/write scoped, CodeGraph read/index, LSP diagnostics, git inspect, safe test runner, checkpoint, memory recall | Git mutate, shell, debugger, DB migration runner | `repo_host`, `test_runner`, `debugger`, `database`, `api_docs` |
| Frontend Engineer | File read/search/write scoped, browser inspect/screenshots/console read, LSP diagnostics, git inspect, safe test runner, checkpoint | Browser interact, visual regression, shell, design asset tools | `browser`, `visual_regression`, `design_assets`, `test_runner` |
| QA Engineer | Browser inspect/screenshots/console/network read, app state read, logs read, safe test runner, file read/search, memory recall | Browser interact, bug filing, trace/video capture | `browser`, `issue_tracker`, `observability`, `test_runner` |
| SDET | File read/search/write test files, safe test runner, CodeGraph read, LSP diagnostics, git inspect, checkpoint, memory recall | CI mutation, browser interact, shell, debugger | `test_runner`, `browser`, `ci_cd`, `debugger`, `repo_host` |

## Approval And Risk Defaults

Default presets should use conservative approval rules.

- Markdown writes, native Kanban/card creation, file edits, and test execution
  require approval.
- Browser inspect is read-only; browser interaction requires approval.
- Shell, SSH, debugger, deploy, CI mutation, REPL, and arbitrary process
  execution are not enabled by default.
- Web search/fetch is recommended unavailable in packaged defaults unless a
  native configured provider or external MCP binding exists. Once enabled, it is
  read-only, records `external_network` provenance, and requires the relevant
  external-server and credential grant metadata when the provider needs secrets.
- Memory recall is enabled; memory writes are not enabled for these role
  presets by default.
- Git mutate is recommended inactive unless explicitly enabled by an operator.
- Credentialed external calls require credential grant metadata and approval
  policy evaluation.
- Workspace binding is required for all scoped writes and test execution.

Suggested risk classes:

- `mutating`
- `external_network`
- `process_execution`
- `credential_use`
- `browser_mutation`
- `git_mutation`
- `deployment_mutation`
- `memory_mutation`
- `test_execution`

### Risk-Class Compatibility

The implementation plan must update preset safety validation before introducing
new risk classes. `validate_preset_safety()` currently recognizes only a narrow
high-risk set. These proposed classes must be treated as known reviewed risks,
not as accidental unknowns, and each must map to approval and provenance rules.

| Risk class | Default enablement | Required approval/provenance behavior | Validator change |
| --- | --- | --- | --- |
| `mutating` | Allowed by scoped-write roles | Approval required for writes/mutations. | Existing behavior remains valid. |
| `external_network` | Not enabled in packaged defaults unless configured | Requires provenance explaining provider/binding and external-server grant; credential grant when secrets are needed. | Existing behavior remains valid, but web-search presets must not claim enabled access without grants. |
| `process_execution` | Not enabled by default | Approval and high-risk provenance required. | Existing behavior remains valid. |
| `credential_use` | Not enabled by default | High-risk provenance and credential-grant metadata required; never stores secret values. | Existing behavior remains valid. |
| `browser_mutation` | Recommended inactive | Approval required for navigation, click, type, select, or page-state mutation. Browser inspect stays read-only. | Add as known high-risk class requiring approval/provenance. |
| `git_mutation` | Recommended inactive | Approval required for add, commit, merge, rebase, conflict writes, or ref changes. | Add as known high-risk class requiring approval/provenance. |
| `deployment_mutation` | Recommended inactive | Approval and provenance required for deploy, restart, rollback, cloud mutation, or CI mutation. | Add as known high-risk class requiring approval/provenance. |
| `memory_mutation` | Recommended inactive | Approval required for long-term memory writes/reflections outside Memory Keeper flows. | Add as known high-risk class requiring approval/provenance. |
| `test_execution` | Enabled only through safe runner roles | Approval required and audited as resolved configured command identity. | Add as known execution-adjacent class requiring approval/provenance but not arbitrary shell authority. |

Unknown risk classes should continue to fail safety validation or require
explicit spec review. New profile presets must not introduce new risk strings
without extending this compatibility table and the validator tests.

### Safe Test Runner Contract

The safe test runner is a constrained process abstraction, not a shell escape.
It may be enabled for Backend Engineer, Frontend Engineer, QA Engineer, and
SDET only when the workspace has immutable test command definitions from a
trusted project configuration source.

Required constraints:

- Agents call `tests.run_configured` with a stable command id, not a free-form
  shell command string.
- Command definitions are workspace-bound and resolved server-side from trusted
  config, not from request arguments.
- Allowed arguments must be declared per command id. Callers may only provide
  typed fields from that definition, such as test target, marker, file path, or
  reporter.
- `cwd`, environment, executable, and base argv are fixed by the command
  definition or selected from an allowlist. Callers cannot provide arbitrary
  env vars, shell fragments, redirections, pipes, command separators, or
  executable paths.
- Execution always requires approval, records `test_execution` provenance, and
  audits the resolved command id, target, workspace, approval decision, and
  result summary.
- Test-result reads through `tests.results.read` are read-only and do not imply
  authority to rerun tests.

## Native Tool Backlog

The design implies these native/default-included additions for tldw_server MCP
and ACP workspaces.

| Category | Tools |
| --- | --- |
| File operations | `fs.edit`, `fs.patch`, `fs.glob`, `fs.grep`, `fs.stat` |
| Git inspect | `git.status`, `git.diff`, `git.log`, `git.blame`, `git.branches`, `git.conflicts.list` |
| Git scoped mutate | `git.apply_patch`, `git.conflicts.resolve_file`, `git.add`, `git.commit` as inactive/approval-gated |
| Tool discovery | `tool_search`, `tool_describe`, `tool_call`, `tool_categories.list`, `profile.tools.list` |
| Safe test runner | `tests.list_commands`, `tests.run_configured`, `tests.results.read` |
| Browser inspect | `browser.snapshot`, `browser.screenshot`, `browser.console`, `browser.network`, `browser.page_state`; CDP first |
| Browser interact | `browser.navigate`, `browser.click`, `browser.type`, `browser.select`, all approval-gated |
| LSP/code intel | `lsp.diagnostics`, `lsp.symbols`, `lsp.references`, `lsp.definition`, `lsp.code_actions` |
| Review helpers | `review.findings.create`, `review.findings.list`, `review.summary.write` |
| Checkpoint/session | `checkpoint.create`, `checkpoint.list`, `checkpoint.diff`, `checkpoint.restore` gated |
| Memory | `memory.recall`, `memory.search`, `memory.write_reflection` gated |
| Ask/user coordination | `ask_user`, `todo.write`, `plan.enter`, `plan.update` |
| Issue abstraction | Native Kanban now; external `issue.create`, `issue.update`, `issue.search` through bindings |
| CI/CD inspect | `ci.runs.list`, `ci.logs.read`, `ci.artifacts.list` through external bindings first |
| DevOps inspect | `logs.search`, `service.status`, `env.inspect` through native or external bindings |
| Debug/REPL/SSH | Catalog-only initially; inactive until explicitly enabled |

These are not all same-priority. The first implementation slices should favor
the tools closest to existing modules:

1. File search/edit helpers and profile-scoped tool discovery.
2. Git inspect and conflict-read tools.
3. Safe test runner abstraction.
4. Browser inspect read tools.
5. LSP/code-intelligence read tools.
6. External binding catalog and setup status surfaces.

## External Server Categories

Presets should remain vendor-neutral. Recommended external bindings should be
grouped by category with concrete options where known.

| Category | Example binding options |
| --- | --- |
| `issue_tracker` | Jira, Linear, GitHub Issues, GitLab Issues |
| `repo_host` | GitHub, GitLab, Bitbucket, Forgejo/Gitea |
| `pr_review` | GitHub PRs, GitLab merge requests, Bitbucket PRs |
| `ci_cd` | GitHub Actions, GitLab CI, Buildkite, Jenkins, CircleCI |
| `browser` | Chrome DevTools/CDP MCP, Playwright MCP |
| `web_search` | Brave Search, Tavily, Exa, Kagi, SearxNG |
| `docs_search` | Context7-style docs search, vendor docs MCP, local docs index |
| `diagram` | Mermaid renderer, draw.io/Excalidraw integration |
| `citation_manager` | Zotero, reference manager, scholarly search |
| `cloud_provider` | AWS, Azure, GCP, Kubernetes |
| `observability` | Datadog, Grafana, Sentry, OpenTelemetry/Prometheus |
| `secrets_manager` | 1Password, Vault, cloud secret managers |
| `debugger` | DAP server adapters |
| `database` | Postgres, SQLite admin, migration tool adapters |
| `visual_regression` | Playwright traces, Percy, Chromatic |
| `design_assets` | Figma, image generation, asset catalog |
| `cms_docs_publisher` | Git-backed docs, Confluence, Notion, ReadMe |

Each option should carry:

- `id`
- `category`
- `kind` such as `external_mcp`, `native_candidate`, or `host_adapter`
- `install_target` when known
- `credential_slots`
- `required_scopes`
- `risk_classes`
- `maturity` such as `exact_target`, `category_placeholder`, or
  `documented_candidate`
- `setup_url` or package docs when known

Initial exact target:

| Category | Binding option | Maturity | Setup |
| --- | --- | --- | --- |
| `browser` | `ChromeDevTools/chrome-devtools-mcp` | `exact_target` | https://github.com/ChromeDevTools/chrome-devtools-mcp |

## Runtime Flow

Catalog flow:

1. Frontend or ACP session starts with `profile_id` and workspace binding.
2. Gateway resolves profile and assignment.
3. Gateway builds effective policy:
   - enabled tools/capabilities
   - path scopes
   - risk classes
   - approval policy
   - credential/external-server grants
   - progressive disclosure settings
4. Frontend requests direct tools, categories, and unavailable recommended
   bindings.
5. Gateway returns directly exposed schemas, deferred categories, missing
   recommended servers, and unavailable reason codes.

Execution flow:

1. Agent calls a direct tool or public `tool_call(tool_id, args)` bridge.
2. Gateway resolves the real tool, activation status, required capability, and
   risk metadata. `tool_call` can only resolve ids previously visible to this
   profile through search/describe.
3. Policy checks profile allow/deny, workspace/path scope, external-server
   grant, credential grant, risk class, and approval policy.
4. If approval is required, return structured `approval_required` with risk,
   target, and human-readable reason.
5. On approval, execute through native module, ACP host, or external MCP
   runtime.
6. Append audit event with profile, workspace, tool, risk, approval status, and
   target.

Safety invariants:

- Recommended tools do not grant authority.
- Search/describe never execute or consume credentials.
- Shell/process cannot be reached through test runner, git, browser, CI, or
  deploy abstractions. Each abstraction resolves a server-owned configured
  operation identity before policy checks.
- Credential grants never contain secret values.
- Workspace binding is required for scoped writes and tests.
- Audit runs for denial, approval request, approval grant/deny, and execution.

## ACP Workspace Implications

ACP-hosted agents should not receive a different conceptual tool set. They
should receive the same profile-filtered catalog as native MCP clients, but
execution may route through ACP/editor permission flows for filesystem,
terminal, or browser actions.

An ACP session should bind:

- workspace root
- path scopes
- profile id
- allowed categories
- approval callback channel
- audit stream
- checkpoint/session boundary

ACP permission prompts should preserve the same risk classification and
underlying tool identity used by MCP runtime policy. A frontend can display
profile recommendations and unavailable categories the same way for direct MCP
clients and ACP-routed agents.

## Testing Requirements

The implementation plan should include these test groups:

- Preset schema tests: every role preset has stable ids, display metadata,
  enabled tools, recommended tools/servers, risk classes, and progressive
  disclosure config.
- Safety tests: no profile enables shell, SSH, debugger, deploy, CI mutation,
  memory write, REPL, arbitrary process execution, or browser interaction by
  default.
- Role coverage tests: each requested profile has enough enabled/default tools
  for its purpose.
- Recommendation tests: setup-dependent tools are visible as recommended but
  not callable until explicitly enabled.
- Web-search tests: packaged defaults list web search/fetch as recommended
  unavailable unless a configured provider/binding and required grants exist.
- Progressive disclosure tests: direct tool list stays below threshold;
  deferred tools are found through `tool_search`; discovery is profile-scoped.
- Tool-search ranking tests: profile grants filter results before ranking,
  installed/enabled tools sort ahead of unavailable recommendations, category
  filters apply before BM25 scoring, and the first implementation does not
  require semantic-search infrastructure.
- Runtime policy tests: `tool_call` dispatch checks the real underlying tool,
  path scope, risk, approval, external grants, credential grants, and audit.
- Bridge-schema tests: `tool_call` requires `tool_id` and `arguments`, rejects
  unknown fields, rejects ids outside profile-scoped discovery, and returns
  `tool_not_enabled` for recommended inactive tools.
- Risk-validator tests: every proposed risk class is known, maps to approval
  and provenance requirements, and unknown future risk classes still fail.
- Safe-runner tests: `tests.run_configured` accepts only configured command
  ids, rejects free-form commands, constrains args/env/cwd, requires approval,
  and audits the resolved command identity.
- ACP tests: workspace binding and approval callbacks are enforced consistently
  for ACP-routed calls.
- Snapshot/import tests: profile recommendations and binding metadata survive
  export/import without embedding secrets.
- Docs tests: package-local user guide documents mode presets, categories, and
  setup-dependent bindings.

## Resolved Implementation Decisions

- First browser inspection path: CDP.
- Tool-search ranking: do not use semantic search initially. Filter first by
  profile grants and workspace assignment, rank installed/enabled tools ahead
  of unavailable recommendations, apply category filters, then use BM25 over
  the filtered catalog.
- Recommendation catalog mutability: operators can patch recommendation
  metadata separately from executable policy. Patchable recommendations still
  do not grant runtime authority.
- Initial exact external MCP target: `ChromeDevTools/chrome-devtools-mcp` for
  the browser/CDP category.

## Recommended Next Plan

Before writing the implementation plan, reconcile this placeholder spec against
the active MCP/ACP work listed in "Placeholder PR Scope". If the active work
changes a settled assumption above, update the spec and keep the PR in draft
until the mismatch is resolved.

The next implementation plan should split this work into small reviewable
slices:

1. Add schema tests and `metadata.tooling` fixtures for role presets.
2. Add native file search/edit helpers and profile-scoped tool discovery.
3. Add vendor-neutral external binding catalog metadata.
4. Add progressive disclosure runtime responses.
5. Add git inspect/conflict-read tools.
6. Add safe test runner abstraction.
7. Add browser inspect read tools.
8. Update package-local user guide and admin docs.
