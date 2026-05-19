---
id: TASK-244
title: Design Backlog.md Python compatibility clone migration
status: Done
assignee:
  - codex
created_date: '2026-05-10 20:27'
updated_date: '2026-05-10 21:01'
labels: []
dependencies: []
references:
  - 'https://github.com/MrLesk/Backlog.md'
  - 'https://raw.githubusercontent.com/MrLesk/Backlog.md/main/package.json'
documentation:
  - >-
    Docs/superpowers/specs/2026-05-10-backlog-md-python-compatibility-clone-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a design artifact reviewing a full Python compatibility clone of upstream Backlog.md. The design must preserve existing repo task-tracking workflows while assessing how to remove Node/Bun runtime dependence and reduce maintainability, supply-chain, and startup risks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design documents the approved staged compatibility-clone approach and trade-offs
- [x] #2 Design covers architecture, components, data flow, error handling, security, testing, and migration gates
- [x] #3 Design explicitly preserves Markdown file-format compatibility, CLI/MCP compatibility, and existing repo Backlog.md workflow requirements
- [x] #4 Design records upstream Node/Bun packaging facts used in the review
- [x] #5 Spec review issues are resolved or documented before user review
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Capture the approved migration review as a design spec under Docs/superpowers/specs/2026-05-10-backlog-md-python-compatibility-clone-design.md.
2. Ground the design in this repo's existing Backlog.md adoption policy and upstream Backlog.md CLI/MCP/browser/package facts.
3. Cover the approved staged compatibility-clone approach across architecture, components, data flow, security/error handling, testing, and migration gates.
4. Run a spec review pass, revise the document if issues are found, then record verification and ask the user to review the written spec before implementation planning.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Wrote design spec at Docs/superpowers/specs/2026-05-10-backlog-md-python-compatibility-clone-design.md. Initial spec review subagent approved it with no blocking issues. Advisory recommendations were to resolve open questions during early planning, start the implementation plan with upstream command/MCP inventory, and define agent-critical CLI/MCP operations in the first milestone checklist.

Verification for initial spec: rg marker scan over the design spec and TASK-244 passed. git diff --check over the design spec and TASK-244 passed. Bandit was skipped because this task changed only Markdown design/backlog files and no Python code.

User reviewed the committed spec and asked for a critique-first pass before continuing. Reopened TASK-244 to harden the design based on current upstream docs and local review findings. Addressed planning risks around upstream config/default drift, BACKLOG_CWD and --cwd MCP behavior, --no-git initialization, modified-file search, datetime/date-only compatibility, onStatusChange callbacks, browser UI/service-mode complexity, pinned oracle fixture snapshots, and measurable agent-critical CLI/MCP parity.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and refined the Backlog.md Python compatibility clone migration design spec. The spec captures the approved staged compatibility-clone approach with an oracle/golden harness, preserves existing Markdown/CLI/MCP/browser workflow expectations, documents upstream Node/Bun packaging facts, and defines architecture, components, data flow, security/error handling, testing strategy, migration milestones, and cutover gates. After user-requested critique, the design now explicitly covers upstream drift, config discovery, BACKLOG_CWD/--cwd, --no-git, modified-file search, datetime compatibility, onStatusChange security, browser parity cost, fixture pinning, and agent-critical MCP/CLI measurability. Verification: spec review approved the initial document with no blocking issues, marker scans passed, git diff --check passed, and Bandit was skipped because the task is docs/backlog only.
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
