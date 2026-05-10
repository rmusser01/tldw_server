---
id: TASK-223.2
title: 'PR 2: MCP Hub setup polish and diagnostics'
status: To Do
assignee: []
created_date: '2026-05-10 06:13'
labels:
  - mcp
  - webui
  - ux
  - diagnostics
dependencies:
  - TASK-223.1
parent_task_id: TASK-223
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the second PR-sized remediation slice from the MCP Hub walkthrough. This phase should make setup states easier to understand after the live-discovery and chat blocker fixes land.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No-auth local stdio servers render a neutral or healthy no-credentials-required state instead of missing-auth warnings.
- [ ] #2 Legacy Secret Fallback appears only when the selected managed server actually uses the transitional server-level secret flow.
- [ ] #3 Tool Catalog empty and stale states offer clear Add server and Refresh discovery actions with setup, runtime, and permissions distinctions.
- [ ] #4 MCP Hub or shared diagnostics expose effective deployment mode, API origin, and health endpoint enough to diagnose quickstart versus advanced split-brain configuration.
- [ ] #5 Setup isolation expectations for local walkthrough or E2E runs are documented or verified where practical.
- [ ] #6 Focused UI tests and a toy MCP E2E smoke cover the polished setup path.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
