---
id: TASK-2296
title: Add Claude-style tool permission rule parser and evaluator
status: Done
labels:
- mcp
- policy
- permissions
- profiles
- agentic-execution
references:
- https://code.claude.com/docs/en/tools-reference
- Docs/superpowers/specs/2026-06-07-mcp-tool-call-hooks-design.md
- Docs/superpowers/specs/2026-06-07-mcp-fs-patch-write-safe-edit-tools-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement Claude-style ToolName(specifier) permission-rule parsing for MCP profiles and agentic execution. Cover command patterns for Bash/Monitor/PowerShell, path patterns for Read/Grep/Glob/LSP/Edit/Write/NotebookEdit, domain rules for WebFetch, skill name matching, agent type matching, MCP external tool names, deny/ask/allow precedence, hooks integration, and migration from existing profile grants.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Claude-style `ToolName(specifier)` strings and structured `permission_rules` entries compile into package-owned policy rule primitives.
- [x] #2 Command rules use argv-token semantics, reject broad or wildcard executables, and do not authorize raw host shell execution.
- [x] #3 Path, domain, skill, agent, and external MCP wildcard subjects have bounded matching helpers with `deny > ask > allow` precedence.
- [x] #4 Existing exact tool policy behavior remains compatible; runtime `evaluate_profile_tool_decision()` does not accidentally treat non-tool permission rules as direct tool grants.
- [x] #5 Parser/evaluator exports, docs, and focused tests cover valid examples, malformed rules, redacted matched-rule metadata, and deferred runtime integration boundaries.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
First slice: add a package-level Claude-style ToolName(specifier) permission rule parser/evaluator that compiles to existing PolicyDecisionRule primitives, with tests and docs. Defer runtime shell/WebFetch/LSP integration to later tasks.

Detailed plan: `Docs/superpowers/plans/2026-06-10-mcp-tool-permission-rule-parser-implementation-plan.md`
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the first package-level parser/evaluator slice in `mcp_unified.profiles.permission_rules`. The parser accepts Claude-style `ToolName(specifier)` strings and structured `permission_rules` documents for exact tools, governed command aliases, path subjects, domain subjects, external MCP wildcard names, skills, and agents. Compiled rules use the existing `PolicyDecisionRule`, `PolicyDecision`, `PolicyMatchedRule`, and `merge_policy_decisions()` models so follow-up hooks, runtime checks, and policy explain surfaces can consume the same decision metadata.

Command rules intentionally parse argv tokens instead of raw shell strings. Broad command wildcards and shell control syntax are rejected, and the grammar does not grant raw host shell execution. `compile_profile_policy_rules()` now includes `permission_rules`, while `evaluate_profile_tool_decision()` remains exact-tool compatible and does not treat path/domain/command rules as direct tool grants.

Verification:
- Red test phase: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_permission_rules.py -q` failed with `ModuleNotFoundError: No module named 'mcp_unified.profiles.permission_rules'`.
- Focused tests: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_permission_rules.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py -q` passed with 65 tests.
- Ruff: `python -m ruff check mcp_unified/profiles/permission_rules.py mcp_unified/profiles/decisions.py mcp_unified/profiles/__init__.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_permission_rules.py` passed.
- Compile smoke: `python -m compileall -q mcp_unified/profiles/permission_rules.py mcp_unified/profiles/decisions.py mcp_unified/profiles/__init__.py` passed.
- Import smoke: package-level parser/evaluator import and sample path decision passed.
- Bandit: `python -m bandit -r mcp_unified/profiles/permission_rules.py mcp_unified/profiles/decisions.py mcp_unified/profiles/__init__.py -f json -o /tmp/bandit_mcp_tool_permission_rules.json` passed with no findings after removing a wildcard-comparison false positive trigger.
- Whitespace: `git diff --check` passed.

PR review follow-up after rebasing on latest `origin/dev`: verified and fixed all still-valid Qodo/Gemini findings. Domain normalization now uses URL host parsing and normalizes bracketed IPv6 literals without ports/brackets; MCP rule detection and precompiled MCP matching are case-insensitive; command argv validation allows empty string arguments after a fixed executable; and path matching is segment-aware so `*` does not cross `/` while `**` remains the cross-segment wildcard. Added regression tests for each issue and documented the clarified semantics.

Review-fix verification:
- Rebase: `git rebase origin/dev` reported the branch was up to date.
- Red regression run: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_permission_rules.py -q` failed on IPv6 normalization, mixed-case MCP detection, empty argv token validation, and segment-aware path matching before the fixes.
- Focused tests: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_permission_rules.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py -q` passed with 70 tests.
- Ruff: `python -m ruff check mcp_unified/profiles/permission_rules.py mcp_unified/profiles/decisions.py mcp_unified/profiles/__init__.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_permission_rules.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py` passed.
- Compile smoke: `python -m compileall -q mcp_unified/profiles/permission_rules.py mcp_unified/profiles/decisions.py mcp_unified/profiles/__init__.py` passed.
- Bandit: `python -m bandit -r mcp_unified/profiles/permission_rules.py mcp_unified/profiles/decisions.py mcp_unified/profiles/__init__.py -f json -o /tmp/bandit_mcp_tool_permission_rules_review.json` passed.
- Whitespace: `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Claude-style MCP profile permission-rule parsing and bounded evaluation for tool, command, path, domain, external MCP, skill, and agent subjects. The implementation preserves existing exact tool behavior, documents the new grammar, and defers runtime integrations for governed shell execution, WebFetch/WebSearch, LSP diagnostics, hooks, and admin simulation to follow-up tasks.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
