---
id: TASK-230
title: Document ACP downstream-agent release caveat
status: Done
assignee: []
created_date: '2026-05-10 06:46'
updated_date: '2026-05-10 06:47'
labels:
  - acp
  - release-signoff
  - docs
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1504'
  - 'https://github.com/rmusser01/tldw_server/issues/1500'
documentation:
  - Docs/Development/ACP_Production_Readiness.md
  - Docs/Development/Agent_Client_Protocol.md
  - CHANGELOG.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Issue #1504 live signoff found the Go runner build/test gate green and the backend ACP runner healthy when ACP_RUNNER_CWD is pointed at tools/tldw-agent, but this host does not currently have provider API-key env vars or an installed downstream ACP stdio agent suitable for a real create/prompt/cancel verification. Make the release-facing caveat explicit so ACP release notes do not claim live downstream-agent support beyond protocol/runner validation until a real downstream ACP agent is configured and verified.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CHANGELOG.md or the active release-note surface explicitly says ACP downstream-agent support is limited to protocol/runner validation unless a real downstream ACP stdio agent plus provider credentials are configured and verified.
- [x] #2 The caveat references the operator prerequisites without exposing secrets or implying Codex/Claude CLI support that was not live-verified.
- [x] #3 Issue #1504 evidence can cite the doc change plus the runner and backend health checks.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Docs-only caveat added to CHANGELOG.md and Docs/Development/ACP_Production_Readiness.md after live #1504 validation found runner/backend health green but no real downstream ACP stdio agent/provider credentials available. Verification: git diff --check passed; rg confirmed release caveat text. Bandit skipped because only docs/backlog files changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added explicit ACP downstream-agent release caveat. The release surface now limits downstream-agent support to protocol/runner validation unless a real downstream ACP stdio agent and required provider credentials are configured and verified. Recorded docs-only verification and non-code Bandit skip.
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
