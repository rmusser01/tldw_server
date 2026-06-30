# Interactive And Automation Deferrals

This document records CLI/TUI, automation, and git behavior that is intentionally
outside the first Backlog.md Python agent cutover candidate. These items are not
ignored; they are deferred because local, deterministic, reviewable file
operations are the first compatibility target.

## Deferral Matrix

| Capability | Classification | Agent cutover impact | Decision and reason |
| --- | --- | --- | --- |
| Colored output exactness | Interactive polish | Not required for agent cutover | Agents consume plain output. Exact ANSI colors can be added later without changing storage semantics. |
| Interactive board | Interactive TUI | Intentionally deferred | The cutover requires deterministic `board` output, not keyboard-driven task movement or terminal UI state. |
| Overview TUI | Interactive TUI | Intentionally deferred | A human dashboard can follow after the core inventory and mutation paths remain stable. |
| Editor launch | Interactive TUI | Intentionally deferred | Launching `$EDITOR` is environment-dependent and not needed for non-interactive agent workflows. |
| Shell completions | Shell integration | Not required for agent cutover | Completion install is convenience tooling and should not affect runtime compatibility. |
| onStatusChange | Automation hook | Intentionally deferred | Hook execution can run arbitrary commands, so it remains disabled until a dedicated safety design and tests exist. |
| auto-commit | Git automation | Intentionally deferred | Automatic commits hide mutation boundaries from reviewers and are outside the first local-file compatibility gate. |
| hook bypass | Git safety bypass | Rejected for first cutover | Bypassing hooks conflicts with repo safety policy and must not be implemented as part of agent cutover. |
| Remote operations | Git/network behavior | Intentionally deferred | Remote git behavior introduces network and credential effects that are unnecessary for local Backlog.md compatibility. |

## Required Before Enabling Deferred Behavior

Any future implementation of these features must provide:

- A dedicated Backlog task and implementation plan.
- Tests proving the feature is opt-in and does not run during normal CLI, MCP,
  or test execution.
- Clear documentation of environment variables, subprocess behavior, and failure
  handling.
- A security review for any behavior that launches editors, runs hooks, bypasses
  hooks, performs auto-commit, or touches remotes.

## Current Runtime Policy

The Python clone keeps these features out of the first cutover path:

- Plain output is the compatibility contract for agents.
- `onStatusChange` remains disabled by default.
- auto-commit and remote operations are deferred.
- hook bypass is rejected for first cutover.
- Shell completion installation and rich interactive UI behavior are later
  human-operator conveniences, not agent blockers.
