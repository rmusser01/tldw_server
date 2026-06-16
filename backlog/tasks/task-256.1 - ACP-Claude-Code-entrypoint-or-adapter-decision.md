---
id: TASK-256.1
title: ACP Claude Code entrypoint or adapter decision
status: Done
labels:
- ACP
- compatibility
- certification
priority: medium
parent_task_id: TASK-256
references:
- https://github.com/rmusser01/tldw_server/issues/2244
- https://github.com/rmusser01/tldw_server/issues/1564
- https://github.com/rmusser01/tldw_server/pull/2248
documentation:
- Docs/Development/ACP_Certification_Checklist.md
- Docs/Development/ACP_Compatibility_Matrix.md
- Docs/Development/Agent_Client_Protocol.md
- tldw_Server_API/Config_Files/agents.yaml
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track the narrowed Claude Code ACP follow-up from GitHub issue #2244. Recheck Claude Code CLI capabilities, decide whether the project can use a native ACP entrypoint, external ACP adapter, or must keep Claude Code documented-unverified, and update compatibility/setup surfaces only as far as evidence supports.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Claude Code entrypoint strategy is classified as external ACP adapter required, using pinned `@agentclientprotocol/claude-agent-acp@0.40.0` as the documented candidate.
- [x] #2 Compatibility matrix, registry YAML, runner bundled config, and user/development setup surfaces reflect `external_acp_adapter` / `documented_unverified` / `documented_only`.
- [x] #3 Claude Code support remains `documented_unverified` with explicit `adapter_missing` local blocker evidence; no live support claim was added.
- [x] #4 Parent issue #1564 was updated via issue comment and not closed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-06-03: Local Claude Code probe found `claude --version` = `2.1.142 (Claude Code)`, CLI help/MCP help but no native ACP command. Upstream npm metadata identifies `@agentclientprotocol/claude-agent-acp` `0.40.0` with binary `claude-agent-acp`; local `command -v claude-agent-acp` fails, so readiness correctly reports `adapter_missing`. Updated registry/config/docs to classify Claude Code as an `external_acp_adapter` candidate with `documented_unverified` / `documented_only` support.
2026-06-03: Opened draft PR #2248 and commented on GitHub issues #2244 and #1564 with the decision and blocker evidence. Both issues remain open pending PR merge/reconciliation.
2026-06-15: Rebased PR #2248 onto latest `origin/dev`, resolved the compatibility-matrix conflict by preserving the newer Codex workspace-live caveat, and addressed Gemini's review thread by changing the Claude adapter install instruction in `agents.yaml` to `npm install -g @agentclientprotocol/claude-agent-acp@0.40.0`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Classified Claude Code ACP as an external adapter path, not native host stdio: use pinned @agentclientprotocol/claude-agent-acp@0.40.0 and keep the profile documented_unverified until adapter installation, auth/runtime setup, and live initialize/session/prompt E2E pass. Added focused Python and Go regression coverage, synchronized docs/setup surfaces, rebased PR #2248 onto current dev, and addressed the open install-instruction review thread.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched Python test scope; B101 assert warnings are test-file baseline and rerun with B101 skipped passed
- [x] #5 Final summary added
- [x] #6 Known blocker documented: local `claude-agent-acp` adapter is missing, so no live Claude ACP support claim is made
<!-- DOD:END -->
