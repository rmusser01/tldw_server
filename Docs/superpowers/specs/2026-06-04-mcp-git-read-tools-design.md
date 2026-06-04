# MCP Git Read Tools Design

## Status

Approved for specification. This spec covers the first read-only Git inspection slice for MCP Unified. It intentionally avoids Git mutations, caller-selected repository paths, and raw shell execution.

## Context

The MCP default profile work now includes profile-scoped tool discovery, native filesystem helpers, a governed `run`/`bash`/`shell` facade, and optional CDP browser inspection. The next missing native tool family is Git inspection for code review, merge-conflict triage, architecture review, and engineering workflows.

The first Git slice should be easy for front-ends and models to discover as explicit tools. It should not rely on free-form shell commands as the primary interface. It also needs to leave a clean path for standalone MCP server evaluations: operators should be able to compare how well different models use tools, how well each tool description/prompt performs, and which profile grants lead to better or worse outcomes.

## Goals

- Add a new read-only `GitModule` for active workspace Git inspection.
- Resolve the Git repository from the active workspace root only.
- Expose structured, profile-grantable tools for status, diff, log, blame, branch, and conflict inspection.
- Execute Git through a fixed argv allowlist, never through `shell=True`.
- Bound runtime, output size, and result counts for every command.
- Return stable response shapes and reason codes that are useful to front-ends, agents, tests, and evaluations.
- Capture tool metadata and result metadata needed for future metrics, traces, and standalone MCP tool-use evaluations.

## Non-Goals

- No Git mutations: no checkout, add, commit, merge, rebase, stash, reset, clean, push, pull, or conflict marker edits.
- No caller-provided `repo_path` in the first slice. The default behavior is active workspace repo root only.
- No arbitrary Git argv passthrough.
- No shell facade changes in this slice. A future governed `run` mapping can route `git status` or similar commands to native tools after the native API lands.
- No persistent evaluation store in this slice. The design only adds the metadata and event shape needed for a later eval surface.

## Tool Surface

The module exposes six first-slice tools:

| Tool | Purpose | Arguments |
| --- | --- | --- |
| `git.status` | Summarize current branch and working tree state. | `include_ignored?: bool`, `limit?: int` |
| `git.diff` | Return bounded diff text. | `scope?: "unstaged" | "staged" | "head"`, `path?: string`, `context_lines?: int`, `max_bytes?: int` |
| `git.log` | Return bounded commit metadata. | `limit?: int`, `path?: string` |
| `git.blame` | Return bounded blame metadata for one file. | `path: string`, `start_line?: int`, `end_line?: int`, `limit?: int` |
| `git.branches` | List local branches and current branch. | `limit?: int` |
| `git.conflicts.list` | List conflicted paths and Git conflict status codes. | `limit?: int` |

All schemas set `additionalProperties: false`. All tools are read-only and use metadata like:

```json
{
  "category": "git",
  "readOnlyHint": true,
  "uses_filesystem": true,
  "uses_processes": true,
  "path_boundable": true,
  "path_argument_hints": ["path"],
  "capabilities": ["git.read", "workspace.read"]
}
```

Tools with no `path` argument still include `path_boundable: true` because the module is bound to the active workspace Git root.

## Repository Resolution

`GitModule` should reuse `McpHubWorkspaceRootResolver` to get the active workspace root, matching `FilesystemModule`. It should then discover the Git root by running:

```text
git -C <workspace_root> rev-parse --show-toplevel
```

The discovered root must resolve inside the active workspace root. If it does not, the module fails closed with `reason_code: "repo_outside_workspace"`.

The module must not accept a caller-supplied repository path. That keeps profile grants simple and avoids accidental cross-repository access in multi-workspace deployments. Later work can add explicit multi-repo support behind a separate design and policy gate.

## Execution Model

Git execution should use an injected async runner abstraction, for example `GitCommandRunner`, with a production implementation based on `asyncio.create_subprocess_exec`.

Requirements:

- Invoke only `git` with a fixed allowlist of subcommands and flags.
- Always pass `-C <repo_root>` as argv, not through shell interpolation.
- Never use `shell=True`.
- Set a timeout for every process.
- Capture stdout and stderr separately.
- Decode as UTF-8 with replacement for invalid bytes.
- Apply per-tool stdout byte caps and structured truncation metadata.
- Return command failures as structured MCP tool results where possible instead of leaking raw tracebacks.
- Allow tests to inject a fake runner and a fake workspace root resolver.

`git` availability failures should return `reason_code: "git_not_available"`. A workspace that is not inside a Git repository should return `reason_code: "not_git_repository"`.

## Path Handling

Path-bearing tools accept workspace-relative paths only. Absolute paths are rejected unless existing project conventions already allow absolute workspace-contained paths for MCP filesystem tools and the implementation intentionally mirrors that behavior. In either case, the normalized target must stay under the active workspace root and the Git root.

Path validation should happen before invoking Git. A rejected path returns `reason_code: "path_outside_workspace"` or `reason_code: "path_outside_repository"` depending on where containment fails.

The module should use portable `/` separators in returned paths.

## Response Shapes

Every successful response should include:

- `repository_root`: workspace-relative path to the Git root.
- `truncated`: boolean.
- `limits`: effective limits applied.
- `git`: compact command metadata such as subcommand, exit code, and duration in milliseconds.

Tool-specific payloads:

- `git.status`: `branch`, `upstream`, `ahead`, `behind`, `entries`, and counts grouped by `staged`, `unstaged`, `untracked`, `conflicted`, and `ignored`.
- `git.diff`: `scope`, optional `path`, `text`, `bytes`, and `truncated`.
- `git.log`: `commits` with `hash`, `short_hash`, `author_name`, `author_email`, `author_date`, and `subject`.
- `git.blame`: `path`, `start_line`, `end_line`, `lines` with commit hash, author, timestamp, line number, and line text.
- `git.branches`: `current`, `branches`, and optional `truncated`.
- `git.conflicts.list`: `conflicts` with `path`, `xy_status`, and `conflict_type` where derivable.

Error results should use stable reason codes:

- `git_not_available`
- `not_git_repository`
- `repo_outside_workspace`
- `path_outside_workspace`
- `path_outside_repository`
- `invalid_git_output`
- `git_command_failed`
- `git_command_timeout`
- `output_truncated`

## Profile Grants

Add the Git read tools to existing preset metadata where the workflow naturally benefits:

- Code Reviewer
- Merge Conflict Resolver
- Architect
- Backend Engineer
- Frontend Engineer
- DevOps Engineer
- QA Engineer
- Software Development Engineer in Test

Product Owner and Documentation Writer should not receive Git read tools by default in this slice. They can still rely on filesystem and documentation tools. If later product workflows need changelog or history context, that can be a separate profile recommendation update.

The built-in `_GIT_READ_TOOLS` constant already names the likely first Git tool family. The implementation should align the actual tool list with that constant and profile metadata.

## Observability And Evaluation Contract

This slice should make Git tool use easy to evaluate later without adding a full eval product now.

Each tool definition should include evaluation-oriented metadata under a stable metadata key, for example:

```json
{
  "eval": {
    "tool_prompt_id": "mcp.git.status.v1",
    "tool_prompt_version": "2026.06.04",
    "task_families": ["code_review", "merge_conflict_triage", "repository_research"],
    "expected_result_kind": "structured_git_state",
    "success_signals": ["used_bounded_path", "selected_correct_scope", "avoided_mutation"]
  }
}
```

The exact key names can be adjusted to match existing MCP metadata conventions, but the implementation should preserve these concepts:

- A stable prompt/description identifier for each tool.
- A version for the tool prompt/description text.
- Task families where the tool is expected to help.
- Machine-readable success signals for later eval labeling.
- Risk and capability labels for profile comparison.

Each execution result should include non-sensitive evaluation metadata:

- requested tool name
- effective tool prompt id/version
- selected Git subcommand family
- profile or mode id when available from request context
- whether a path filter was used
- output truncation status
- reason code on failure
- latency bucket or duration

Do not include raw diff text, file contents, full local absolute paths, secrets, or author emails in metrics labels. Rich traces may include outputs only when an explicit eval capture mode is enabled by the host and redaction policy allows it.

The later standalone MCP eval surface can use this contract to compare:

- model success rate by profile and tool grant set
- tool prompt variants for the same tool
- misuse rate, such as calling `git.diff` when `git.status` was sufficient
- truncation and retry behavior
- whether models avoid mutation requests when only read tools are granted
- whether a profile overexposes tools and causes lower precision

This also supports A/B prompt tests for tool descriptions without changing executable policy. Tool prompt metadata should be patchable separately from executable policy, consistent with the profile recommendation catalog direction.

## Security Considerations

- No raw shell execution.
- No arbitrary Git arguments.
- No mutating Git subcommands.
- No environment variable passthrough except a minimal safe environment if needed to make Git deterministic.
- Disable pagers with environment or flags such as `GIT_PAGER=cat` / `--no-pager`.
- Avoid network-capable Git operations.
- Avoid config writes and hooks.
- Bound process runtime and output.
- Do not log raw diffs or absolute local paths.
- Treat stderr as potentially sensitive and return bounded, sanitized text.

## Testing Strategy

Use TDD for implementation. Focused tests should cover:

- Tool schema metadata and `additionalProperties: false`.
- Active workspace repo discovery.
- Non-Git workspace failure.
- Missing Git binary through injected runner.
- Git root outside workspace failure.
- Path containment and path normalization.
- `git.status` porcelain parsing for staged, unstaged, untracked, ignored, and conflicted files.
- `git.diff` scopes for unstaged, staged, and head.
- Diff byte truncation.
- `git.log` limit handling and path filtering.
- `git.blame` line range validation and bounded results.
- `git.branches` current branch parsing.
- `git.conflicts.list` conflict status parsing.
- Profile preset grants.
- Metrics/eval metadata presence without sensitive labels.

Verification should include:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py -q
python -m bandit -r tldw_Server_API/app/core/MCP_unified/modules/implementations/git_module.py mcp_unified/profiles/presets.py -f json -o /tmp/bandit_mcp_git_read_tools.json
git diff --check
```

If the implementation changes command-runtime aliases, include the relevant command runtime tests as well.

## Rollout

Default-on behavior should be considered carefully. Filesystem helpers are already default-on, but Git invokes a process. For the first implementation, prefer explicit registration by environment flag or module configuration unless the project decides that read-only Git inspection belongs in the same default workspace tool baseline.

Recommended first rollout:

- Register `GitModule` when `MCP_ENABLE_GIT_MODULE=true`.
- Add profile metadata grants so the tools are discoverable when the module is enabled.
- Document how standalone MCP users enable it.

A later PR can make the module default-on after the security and platform behavior is proven across macOS, Linux, and Windows.

## Future Work

- Governed `run` command aliases for common read-only Git commands.
- Multi-repository workspace support with explicit repo selection policy.
- Safe test-runner MCP tools.
- LSP/code-intelligence read tools.
- Git mutation tools behind explicit approval and profile gates.
- Standalone MCP eval endpoints for trace export, prompt variant comparison, and tool-use benchmark runs.
