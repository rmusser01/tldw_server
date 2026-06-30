---
id: TASK-594
title: Certify Codex ACP adapter live path
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 15:54'
labels:
  - ACP
  - Codex
  - agents
  - certification
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-06-01-acp-codex-orchestration-design.md
  - Docs/superpowers/plans/2026-06-01-codex-acp-adapter-implementation-plan.md
priority: high
ordinal: 594
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Validate the live Codex ACP adapter path after the first-slice implementation. Scope includes host prerequisite discovery for Codex CLI and codex-acp, running or explicitly blocking the existing ACP certification smoke harness for the Codex external_acp_adapter profile, recording bounded evidence for initialize/session/prompt/permission/cancel/close/failure behavior where reachable, and fixing any small harness/status issues that prevent honest diagnostics. Do not claim live support unless the adapter path actually passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Codex and codex-acp host prerequisite state is recorded with exact commands/versions or stable blocker codes.
- [x] #2 Existing ACP certification helper can build a Codex external_acp_adapter manifest and either run the live probe or refuse with actionable prerequisites.
- [x] #3 Live initialize/session/prompt/cancel/close/failure evidence is recorded where reachable, or blockers are documented without overclaiming support.
- [x] #4 Any code changes are covered by focused tests and Bandit/diff hygiene when applicable.
- [x] #5 Compatibility/status docs or Backlog closeout clearly distinguish live-certified, blocked, and deferred app-server work.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Local prerequisite discovery: codex resolved to /opt/homebrew/bin/codex and codex --version reported codex-cli 0.128.0. codex-acp was not on PATH, so live Codex ACP certification is blocked by adapter_missing.

Fixed two certification-helper issues found during the blocked run: profile manifests now preserve registry support/verification state plus pinned adapter metadata, and blocked live-agent manifests refuse with exit 2 instead of returning a false green.

Verification: focused red-green regressions passed, full helper suite passed with 39 tests, Codex manifest JSON validated, actual Codex profile run refused with exit 2 on adapter_missing, git diff --check passed, and Bandit on acp_certification_smoke.py passed with no findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the first live-certification attempt for Codex ACP without overclaiming support. The host has Codex CLI 0.128.0 but does not have `codex-acp` on PATH, so live certification is blocked at `adapter_missing`. Fixed the certification helper so blocked profile runs refuse with exit 2 instead of returning a false success, and so Codex profile manifests preserve the registry's experimental support state plus pinned `zed-industries/codex-acp` 0.15.0 adapter metadata. Verification: red-green focused regressions, full helper suite `39 passed`, manifest JSON validation passed, actual Codex profile run refused with exit 2 on `adapter_missing`, `git diff --check` passed, and Bandit on the touched helper passed with no findings.
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
