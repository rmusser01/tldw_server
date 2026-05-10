# Backlog.md Python Compatibility Clone Design

Date: 2026-05-10
Owner: Codex collaboration session
Status: Approved design direction, pending spec review and user review
Tracking task: TASK-244

## Summary

Build a Python compatibility clone of upstream Backlog.md that preserves existing
Markdown storage, CLI behavior, MCP resources/tools, and browser workflow while
removing Node/Bun from the required runtime.

The approved direction is a staged compatibility clone with an oracle harness.
The current upstream Backlog.md binary remains available during development only
as a behavior oracle for golden fixtures. The Python implementation becomes the
runtime implementation once parity gates pass.

The design center is:

- Markdown compatibility is the primary contract.
- Upstream behavior is the temporary oracle.
- Python owns the long-term runtime.
- Existing agent workflows in this repository must keep working.

## Context

This repository already uses Backlog.md as required task tracking for repo file
changes. Root `AGENTS.md` requires a Backlog.md task before repository edits,
uses MCP-first task operations, and allows the CLI as a fallback. The project
stores tasks under `backlog/` with repo-local config at `backlog/config.yml`.

The installed local Backlog.md CLI is version `1.44.0` and is launched from
`/Users/macbook-dev/.bun/bin/backlog`. That entrypoint is a Node script that
resolves and spawns a platform binary package. The installed macOS package is a
native `darwin-arm64` executable distributed through npm optional dependencies.

Upstream Backlog.md documents a Markdown-native project board with CLI, MCP,
browser UI, configuration, search, decisions, docs, milestones, cleanup,
overview, and completion support. Its published package metadata shows a
Bun/TypeScript source entry and npm optional dependencies for platform-specific
binaries.

## Goals

1. Preserve existing Backlog.md repositories without conversion.
2. Preserve command names, common flags, output shapes, and exit behavior where
   current agent workflows depend on them.
3. Preserve MCP resources/tools used by Codex and other agents.
4. Preserve browser workflow without requiring Node/Bun at runtime.
5. Reduce runtime supply-chain surface by making the core CLI/MCP path small and
   Python-native.
6. Improve maintainability through a shared domain library used by CLI, MCP,
   browser, search, and TUI surfaces.
7. Improve startup behavior by avoiding a Node shim and secondary binary
   resolution on the hot path.
8. Use upstream Backlog.md as a test oracle during development, not as a runtime
   dependency.

## Non-Goals

1. Designing a Pythonic replacement with incompatible commands or file formats.
2. Migrating existing task files to a new schema.
3. Changing this repository's Backlog.md task-tracking policy.
4. Enabling Backlog.md auto-commit or hook bypass by default.
5. Replacing GitHub issues, pull requests, superpowers specs/plans, or normal git
   history.
6. Reimplementing every interactive and browser feature in the first milestone.
7. Depending on a JavaScript build chain for normal Python package development.

## Approved Approach

Use a staged compatibility clone with an oracle harness.

During development, the existing upstream `backlog` binary is allowed only in
tests that generate or compare golden fixtures. The Python port must not call
Node, Bun, or the upstream binary at runtime.

Rejected alternatives:

- A repo-local Python wrapper around the upstream binary. This would keep
  Node/Bun and npm platform packages in the runtime path, which defeats the main
  goals.
- A Python-native reimagining with approximate compatibility. This would be
  easier to build but would break existing agent instructions and Backlog.md
  repositories.
- A big-bang full rewrite. This maximizes risk and delays the first useful
  parity checkpoint.

## Architecture

The Python port should have three layers.

### Canonical Domain Library

The domain library owns config discovery, ID generation, Markdown/frontmatter
parsing, task/doc/decision models, dependency validation, status transitions,
search indexing, and file writes. Every interface calls this library.

### Compatibility Surfaces

The CLI, MCP server, shell completions, browser API/routes, and TUI views are
surface adapters over the domain library. They should preserve upstream command
names and common flags wherever possible:

- `backlog init`
- `backlog task` / `backlog tasks`
- `backlog search`
- `backlog draft`
- `backlog milestone` / `backlog milestones`
- `backlog board`
- `backlog doc`
- `backlog decision`
- `backlog agents`
- `backlog config`
- `backlog sequence`
- `backlog cleanup`
- `backlog browser`
- `backlog overview`
- `backlog completion`
- `backlog mcp`

### Oracle And Golden Harness

The oracle harness runs upstream Backlog.md against temporary repositories and
captures before/after files, stdout, stderr, exit codes, and MCP responses.
Those fixtures become the compatibility spec. Golden tests should cover normal
paths and important errors before the Python implementation claims parity.

## Components

### `backlog_py.core`

Owns task, document, decision, milestone, and config domain models. It validates
IDs, statuses, dependencies, parent/subtask relationships, acceptance criteria,
Definition of Done, and section ownership.

### `backlog_py.storage`

Owns project discovery, config loading, backlog directory resolution, filename
sanitization, atomic writes, archive/completed moves, lock handling, and
git-aware branch scans.

### `backlog_py.markdown`

Owns frontmatter parsing, Markdown section preservation, checklist parsing, and
round-trip rendering. This is the most important boundary because existing
Markdown files are the compatibility contract.

The parser must preserve:

- unknown frontmatter keys
- date string formats
- section markers such as `<!-- SECTION:NOTES:BEGIN -->`
- checklist ordering
- unowned Markdown sections
- task/doc/decision body text outside the edited field

It should normalize only fields explicitly owned by the requested operation.

### `backlog_py.search`

Owns indexing and fuzzy search across tasks, docs, decisions, and milestones.
It should support upstream-compatible filters for status, priority, assignee,
labels, milestone, parent, and related fields where upstream supports them.

`rapidfuzz` is acceptable only if supply-chain review approves it. Otherwise the
first milestone should use a small deterministic Python search implementation
and defer advanced ranking.

### `backlog_py.cli`

Owns the command tree. Prefer Click over Typer for precise command and option
compatibility, predictable exit-code handling, and mature shell completion
control.

The first CLI milestone should prioritize non-interactive commands and `--plain`
output because those are the safest agent fallback surface.

### `backlog_py.mcp`

Owns a stdio MCP server that exposes the same workflow resources and tools used
by agents. The MCP layer should expose typed operations only. It must not expose
generic shell execution.

### `backlog_py.browser`

Owns the local browser workflow. To avoid Node/Bun at runtime, implement the
browser as a Python web app using FastAPI or Starlette with server-rendered
templates, packaged static assets, and optional HTMX-style interactions.

The browser can lag the CLI/MCP parity milestone, but the final compatibility
clone must preserve `backlog browser` behavior enough for users to manage tasks
locally.

### `backlog_py.tui`

Owns terminal board and overview rendering. Start with plain table/text output.
Interactive TUI behavior should be deferred until the core parser, storage,
CLI, and MCP surfaces are stable.

### `tests/oracle`

Owns upstream comparison fixtures. It should support:

- generating fixtures when upstream Backlog.md is available
- running Python-only tests when upstream is unavailable
- freezing fixtures so CI does not require Node/Bun
- comparing output, files, and errors without overfitting to terminal color or
  incidental whitespace

## Data Flow

All surfaces should follow the same operation path:

1. Discover the project root and backlog directory from `cwd`, config files, and
   explicit command or MCP inputs.
2. Load config using upstream-compatible precedence: explicit operation inputs,
   project config, then built-ins.
3. Read task, doc, decision, and milestone Markdown through the round-trip
   parser.
4. Apply a domain operation: create, edit, archive, complete, search, list,
   board render, config update, MCP tool call, or browser action.
5. Validate cross-object rules before writing.
6. Write changes atomically.
7. Apply optional git-aware behavior only when config and command semantics
   require it.
8. Return interface-specific output: CLI text, MCP content/tool results, browser
   HTML/JSON, or TUI render.

For this repository, the Python implementation must respect:

- `auto_commit: false`
- `bypass_git_hooks: false`
- `remote_operations: false`
- existing Definition of Done defaults in `backlog/config.yml`
- existing root `AGENTS.md` instruction to use MCP first and CLI fallback second

## Compatibility Rules

### File Format

Existing `backlog/` directories must work without migration. Running read-only
commands against existing task files must produce no file changes.

Write operations must preserve unrelated formatting and unknown metadata. A
command that edits notes should not rewrite description, acceptance criteria, or
frontmatter fields it does not own.

### CLI

Command names, aliases, core options, `--plain` output, and common errors should
match upstream wherever golden fixtures cover them. Interactive behavior may be
deferred, but unsupported interactive paths must fail clearly rather than
silently doing the wrong thing.

### MCP

MCP resources and tools should keep stable names, schemas, and error shapes.
Agent workflows should be able to use the same task creation, task execution,
task finalization, document, milestone, and Definition of Done operations.

### Browser

The browser surface should remain local-first and project-bound. It should not
introduce a separate database or require a build step for normal use.

## Error Handling

The implementation should fail closed for mutating operations.

Reject:

- absolute paths where upstream does not explicitly permit them
- `..` traversal outside the backlog root
- malformed task IDs
- duplicate IDs
- unknown statuses
- invalid checklist indexes
- nonexistent dependencies
- circular dependency graphs
- conflicting parent/subtask relationships
- malformed frontmatter in fields about to be mutated
- writes that resolve outside the discovered backlog root

Parser ambiguity should produce structured errors. It should not silently
rewrite task files.

CLI errors should preserve upstream exit codes and message shapes where golden
fixtures cover them. MCP errors should be typed and predictable for agents.
Browser errors should not partially write files.

## Security Posture

The core runtime dependency set should be small and auditable. Optional features
should be extras, such as:

- `backlog-py[search]`
- `backlog-py[browser]`
- `backlog-py[tui]`

Core CLI and MCP operations should avoid shell execution. Command execution
surfaces such as editor launch, browser open, git operations, and completion
installation should exist only behind explicit commands or config.

Never enable `autoCommit` or `bypassGitHooks` by default. If a project sets
`bypass_git_hooks: true`, the Python implementation should make that behavior
visible because it maps to bypassing normal git hooks.

The MCP server should expose data operations only. It must not provide a generic
command runner.

## Testing Strategy

The test strategy is the migration mechanism.

### Upstream Inventory

Create a command and behavior inventory from upstream docs and observed local
usage. Mark each behavior as one of:

- `golden-required`
- `interactive-deferred`
- `browser-deferred`
- `not-supported-with-explicit-reason`

### Golden Fixtures

For each golden-required behavior, capture:

- initial repository files
- command invocation or MCP request
- stdout
- stderr
- exit code
- final repository files
- expected structured MCP result when relevant

Fixtures should include both successful operations and errors.

### Round-Trip Tests

Run the parser over existing `backlog/` task files and assert read-only commands
do not change files. Mutating tests should operate only on temporary copies.

### Security Tests

Cover traversal, duplicate IDs, invalid parent/dependency graphs, malformed
Markdown/frontmatter, invalid checklist indexes, lock contention, disabled
remote operations, and blocked generic shell paths.

### Repository Validation

Before cutover, run the Python clone read-only against this repository:

- `backlog task list --plain`
- `backlog task TASK-1 --plain`
- `backlog search "Backlog.md" --plain`
- `backlog board`
- `backlog config list`
- MCP workflow resource reads
- MCP task search/view operations

Write tests should use copied temporary fixtures, not the live repository.

## Migration Plan

### Milestone 1: Read-Only Core Parity

Implement project discovery, config loading, Markdown parsing, models, read-only
task/doc/decision/milestone listing, viewing, search, board rendering, and MCP
workflow resources.

Success criteria:

- existing repos load without conversion
- this repository's backlog files round-trip without diffs
- read-only CLI and MCP paths pass golden tests

### Milestone 2: Safe Mutations

Implement task creation, task edit, acceptance criteria checks, Definition of
Done checks, notes, final summaries, archive/complete moves, milestones, docs,
and decisions.

Success criteria:

- mutating operations pass golden tests on temporary repos
- atomic write and traversal tests pass
- existing unowned Markdown is preserved

### Milestone 3: Agent Cutover Candidate

Implement enough MCP tools/resources and CLI fallback behavior for Codex and
other agents to use the Python implementation by default.

Success criteria:

- this repository can use Python `backlog` for MCP-first and CLI-fallback task
  workflow
- upstream binary remains available only as `backlog-upstream` during the
  comparison window
- no runtime Node/Bun dependency is needed for agent workflows

### Milestone 4: Browser And Interactive Surface

Implement `backlog browser`, richer board/overview rendering, completion
installation, and deferred interactive flows.

Success criteria:

- browser task management works locally without a Node/Bun build chain
- packaged static assets are included as Python package data
- browser actions call the same domain operations as CLI/MCP

### Milestone 5: Packaging And Decommission

Package the Python implementation for normal install. Keep upstream comparison
fixtures but remove Node/Bun from required runtime and normal CI.

Success criteria:

- `backlog` resolves to the Python implementation in target environments
- upstream binary is no longer required except for explicit fixture refresh
- install, completions, MCP startup, and browser startup are documented

## Cutover Rules

Do not cut over this repository until:

1. Read-only commands produce no diffs against existing `backlog/` files.
2. Golden tests pass for all agent-critical CLI and MCP operations.
3. Mutating operations pass on temporary copies of this repository's backlog.
4. MCP schemas and workflow resource names match the current agent workflow.
5. Security tests pass.
6. Browser parity gaps are either closed or explicitly documented as deferred
   non-agent blockers.
7. Rollback is simple: `backlog-upstream` remains available during the
   comparison window.

## Open Questions

1. Should the browser use FastAPI/Starlette templates or a smaller stdlib HTTP
   server plus packaged static files?
2. How exact must non-plain colored terminal output be?
3. Should advanced fuzzy search depend on `rapidfuzz`, or should the first
   implementation use a dependency-free ranking algorithm?
4. Should package distribution target PyPI only, or also standalone binaries via
   PyInstaller or similar tooling?
5. Which upstream browser behaviors are required before calling the project a
   full compatibility clone?

## Acceptance Criteria

1. Design documents the approved staged compatibility-clone approach and
   trade-offs.
2. Design covers architecture, components, data flow, error handling, security,
   testing, and migration gates.
3. Design explicitly preserves Markdown file-format compatibility, CLI/MCP
   compatibility, and existing repo Backlog.md workflow requirements.
4. Design records upstream Node/Bun packaging facts used in the review.
5. Spec review issues are resolved or documented before user review.

## References

- Upstream Backlog.md repository: `https://github.com/MrLesk/Backlog.md`
- Upstream package metadata:
  `https://raw.githubusercontent.com/MrLesk/Backlog.md/main/package.json`
- Local adoption design:
  `Docs/superpowers/specs/2026-05-03-backlog-md-task-tracking-design.md`
- Local adoption plan:
  `Docs/superpowers/plans/2026-05-03-backlog-md-task-tracking-implementation-plan.md`
- Tracking task:
  `backlog/tasks/task-244 - Design-Backlog.md-Python-compatibility-clone-migration.md`
