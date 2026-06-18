---
id: TASK-2283
title: Add Claude-style governed Bash and PowerShell runtime tools
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-18 04:17'
labels:
  - mcp
  - command-runtime
  - security
  - tools
  - agentic-execution
dependencies: []
references:
  - 'https://code.claude.com/docs/en/tools-reference'
  - Docs/superpowers/specs/2026-03-28-mcp-virtual-cli-run-command-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement Claude-style governed shell runtime parity for Bash and PowerShell where appropriate. Cover command pattern permissions, per-command timeouts, output caps with full-output artifacts, cwd carry-over constrained to workspace/additional dirs, env-file support, shell selection, PowerShell platform behavior, hook integration, telemetry redaction, and safe denial of unsupported shell features.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Bash/shell/PowerShell-facing tool names are governed aliases over the virtual CLI and never execute a raw host shell.
- [ ] #2 Unsupported raw-shell features such as redirection, command substitution, environment expansion, environment assignment prefixes, and background execution fail closed before any backend MCP tool call.
- [ ] #3 Compound command chains continue to be parsed and governed per subcommand with existing preflight behavior.
- [ ] #4 Focused run-command/module tests, py_compile, Bandit, and git diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
First slice: harden the existing virtual CLI shell facade rather than adding raw shell execution. Add tests first for PowerShell alias exposure/execution and fail-closed unsupported shell syntax. Implement a small token-aware unsupported-feature detector in the run command path, extend the alias list to PowerShell/pwsh, document the behavior, run focused tests and security checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
First slice implemented: expanded governed run aliases to include powershell and pwsh, added token-aware fail-closed detection for unsupported raw shell features before backend MCP preparation, and documented that the aliases remain virtual CLI facades rather than raw host shells. Verification: focused command runtime pytest 91 passed; Ruff passed for touched Python files; py_compile passed for touched Python files; Bandit report /tmp/bandit_mcp_shell_facade_2283.json had results=0 errors=0; git diff --check passed.

Second slice implemented: added optional timeoutSeconds / timeout_seconds support for run, bash, shell, powershell, and pwsh. The timeout wraps the governed command chain including preflight and nested MCP execution, validates positive finite numeric values, rejects conflicting snake/camel timeout aliases, and returns exit code 124 on timeout. Remaining broader TASK-2283 areas include richer output artifact, cwd carry-over, env-file, shell selection, session, and telemetry parity if desired in later slices. Verification: focused command runtime pytest 97 passed; Ruff passed for touched Python files; py_compile passed for touched Python files; Bandit report /tmp/bandit_mcp_shell_timeout_2283.json had results=0 errors=0; git diff --check passed.

Third slice implemented: added explicit cwd / workingDirectory support for one governed command chain. The cwd is normalized as a workspace-relative path, rejects absolute paths, Windows drive roots, home-relative paths, and traversal, rewrites relative file/search-base arguments before nested MCP fs calls, and salts nested idempotency keys so the same parent key cannot collide across cwd scopes. Remaining broader TASK-2283 areas include output artifact, env-file, shell selection, session, and telemetry parity if desired in later slices. Verification: focused command runtime pytest 103 passed; Ruff passed for touched Python files; py_compile passed for touched Python files; Bandit report /tmp/bandit_mcp_shell_cwd_2283.json had results=0 errors=0; git diff --check passed.

Fourth slice implemented: added opt-in retainOutputArtifacts / retain_output_artifacts support for oversized governed run output. Default behavior still deletes spill files after rendering, while retained output keeps the private spill file and reports a redacted mcp-run-output:// handle instead of an absolute filesystem path. Remaining broader TASK-2283 areas include env-file, shell selection, session, and telemetry parity if desired in later slices. Verification: focused command runtime pytest 107 passed; Ruff passed for touched Python files; py_compile passed for touched Python files; Bandit report /tmp/bandit_mcp_shell_artifacts_2283.json had results=0 errors=0; git diff --check passed.

Fifth slice implemented: added sandboxSessionId / sandbox_session_id support for governed sandbox command steps. When set, sandbox steps call sandbox.run with session_id instead of the default base_image, validation rejects empty/non-string/conflicting aliases, and nested idempotency keys are salted by sandbox session scope. Remaining broader TASK-2283 areas include env-file, shell selection, and telemetry parity if desired in later slices. Verification: focused command runtime pytest 112 passed; Ruff passed for touched Python files; py_compile passed for touched Python files; Bandit report /tmp/bandit_mcp_shell_sandbox_session_2283.json had results=0 errors=0; git diff --check passed.

PR review hardening implemented: addressed Qodo, Gemini, and CodeRabbit comments by adding docstrings to newly added helpers/tests, replacing the sleep-based timeout test with deterministic cancellation, preserving whitespace in cwd-rewritten file path tokens, hashing idempotency scope with structured JSON serialization, and using per-invocation spill directories so timeout cleanup removes spills even when execution is cancelled before a result is returned. Verification: focused command runtime pytest 115 passed; Ruff passed for touched Python files; py_compile passed for touched Python files; Bandit report /tmp/bandit_mcp_shell_pr2384_review.json had results=0 errors=0; git diff --check passed.

Sixth slice implemented: added envFile / env_file support for governed sandbox command steps. Env files are validated as workspace-relative paths, resolved under the active workspace root with symlink target containment, bounded to 65536 bytes, parsed as simple UTF-8 .env KEY=value entries without expansion, forwarded only to sandbox.run, and salted into nested idempotency scope by path/content digest without exposing secret values. Non-sandbox command chains fail closed instead of silently loading env files. Verification: focused run command pytest 69 passed; Ruff passed for touched Python files; py_compile passed for touched Python files; Bandit report /tmp/bandit_mcp_run_env_file.json had results=0 errors=0 skipped=0.

Sixth-slice hardening added: sandbox.run env values are redacted from tool hook contexts while preserving the real prepared/executed sandbox arguments and argument hash behavior. Verification rerun after hardening: run-command plus protocol hook pytest 78 passed; Ruff passed for touched Python files; py_compile passed for touched Python files; Bandit report /tmp/bandit_mcp_run_env_file.json had results=0 errors=0 skipped=0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Governed shell facade slices complete so far: bash, shell, powershell, and pwsh share the virtual CLI runtime; unsupported raw-shell syntax fails before nested MCP tool calls are prepared; optional governed-chain timeouts return exit code 124 when exceeded; per-call cwd scopes rewrite relative workspace paths without bypassing nested MCP policy while preserving literal path token whitespace; oversized output can be retained through redacted artifact handles in private invocation spill directories; and sandbox commands can target an existing sandbox session. The broader TASK-2283 remains open for future richer shell-runtime parity work.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
