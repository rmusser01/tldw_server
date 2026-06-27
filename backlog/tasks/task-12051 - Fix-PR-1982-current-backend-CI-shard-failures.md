---
id: TASK-12051
title: Fix PR 1982 current backend CI shard failures
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-27 15:27'
labels:
  - ci
  - pr-1982
  - tests
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1982'
  - 'https://github.com/rmusser01/tldw_server/actions/runs/28282225659'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track the current PR #1982 CI failures after the full matrix appeared on head 93fb333a09. Known groups include workflow contract drift for the watchlists extension job, tokenizer metadata test monkeypatch drift, provider readiness tests affected by CI egress env, audio artifact invalid path handling on Windows, distributed lock residual file cleanup on Windows, workflow scheduler stats, and new llm-adapters/orchestrator/chat endpoint shard failures that need log triage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

2026-06-27 PR #1982 CI follow-up:
- Current live run checked before push: 28282225659 still shows 25 failed, 737 passed, 9 canceled, 4 skipped checks.
- Local focused regression set covering the failed shards passed: 23 passed, 8 warnings.
- Workflow YAML parse passed for ui-watchlists-extension-e2e.yml and ci.yml.
- git diff --check passed.
- Bandit on tldw_Server_API/app/core/Workflows/engine.py passed with 0 findings (/tmp/bandit_pr1982_workflows_engine.json).
- Remaining action: commit and push fixes so GitHub re-runs the failed matrix against the patched branch.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented and locally verified the current PR #1982 CI shard fixes for run 28282225659. The patch covers the watchlists extension Chromium install contract, tokenizer metadata test isolation, egress-env leakage in readiness/model metadata tests, AuthNZ schema readiness for adapter endpoint tests, Windows path/lock/artifact range assumptions, deterministic circuit-breaker recovery tests, and workflow scheduler active-count cleanup. Pre-push verification: focused pytest set passed with 23 passed and 8 warnings; workflow YAML parse passed; git diff --check passed; Bandit on app/core/Workflows/engine.py reported 0 findings. Known pending item: GitHub Actions must rerun after the push; the 25 live failures observed before this push were from the previous head commit/run, not from the patched commit.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
