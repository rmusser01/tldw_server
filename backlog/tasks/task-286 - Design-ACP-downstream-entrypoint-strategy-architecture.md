---
id: TASK-286
title: Design ACP downstream entrypoint strategy architecture
status: In Progress
assignee: []
created_date: '2026-05-12 03:35'
labels:
  - ACP
  - design
  - compatibility
  - certification
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1563'
  - 'https://github.com/rmusser01/tldw_server/issues/1564'
documentation:
  - Docs/Development/ACP_Compatibility_Matrix.md
  - Docs/Development/ACP_Certification_Checklist.md
  - Docs/Development/Agent_Client_Protocol.md
  - tldw_Server_API/Config_Files/agents.yaml
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a design spec for an architecture-first ACP downstream entrypoint strategy. The spec should distinguish native ACP commands, adapter-backed ACP commands, documented candidates, and custom templates; define how registry metadata, classifier output, probe manifests, certification evidence, setup surfaces, and issue closeout rules should work; and preserve the rule that unsuccessful live certification keeps #1563/#1564 open.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design defines entrypoint strategy categories for native ACP, adapter ACP, documented candidates, and custom templates.
- [x] #2 Design explains registry metadata fields, classifier responsibilities, probe manifest generation, evidence recording, and setup/API/UI status consumption.
- [x] #3 Design includes status and error-handling rules that prevent blocker-documentation PRs from closing unsuccessful live-certification issues.
- [x] #4 Design identifies initial target classifications for OpenCode, Goose, Codex, Claude Code, Aider, Continue, and custom profiles.
- [x] #5 Design includes testing expectations and staged implementation boundaries without installing agents or implementing adapters in the design task.
- [x] #6 Design spec is reviewed and committed on an isolated branch.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
- Wrote design spec at `Docs/superpowers/specs/2026-05-12-acp-downstream-entrypoint-strategy-design.md`.
- Spec review pass 1 found two blockers: `acp_command` required-vs-inferred was unresolved, and live certification/adapters were in the first implementation scope.
- Spec review pass 2 found one blocker: Codex/Claude were marked `adapter_acp` without concrete `acp_command` values.
- Spec review pass 3 approved the revised spec with no blocking issues.
- Post-approval design critique found two implementation risks and patched the spec: YAML-only strategy fields would miss API/DB-backed dynamic agents, and existing MCP `protocol`/transport fields could be confused with ACP entrypoint metadata.
- Verification: `git diff --check` passed.
- Bandit: skipped because this task changed only Markdown/Backlog design artifacts and no Python code.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and reviewed the ACP downstream entrypoint strategy design. The spec defines native ACP, adapter ACP, documented candidate, and custom template strategies; requires explicit `acp_command` for certifiable native/adapter rows; scopes the first implementation to registry strategy metadata, classifier, profile manifests, and setup/status alignment; and keeps live certification, agent installation, adapter implementation, and issue closeout as follow-on work. The spec review loop approved the final revision.
<!-- SECTION:FINAL_SUMMARY:END -->
