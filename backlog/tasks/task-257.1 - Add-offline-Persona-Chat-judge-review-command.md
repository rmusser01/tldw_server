---
id: TASK-257.1
title: Add offline Persona Chat judge review command
status: Done
assignee:
  - codex
created_date: '2026-05-12 01:12'
updated_date: '2026-05-12 01:22'
labels:
  - persona-chat
  - evaluations
  - stage-2
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1579'
  - 'https://github.com/rmusser01/tldw_server/issues/1566'
  - 'https://github.com/rmusser01/tldw_server/issues/1543'
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
documentation:
  - Docs/Reviews/PERSONA_CHAT_JUDGE_EVAL_CONTRACT_2026_05_11.md
  - Docs/Reviews/PERSONA_CHAT_JUDGE_EVALUATION_CONTRACT_2026_05_11.md
parent_task_id: TASK-257
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a no-provider review command for the Persona Chat judge harness. The command should load candidate judge outputs keyed by PC-JUDGE case id, compare them to the checked-in V1 fixture using the existing offline harness, and emit a bounded JSON report without model calls, DB persistence, Jobs, WebUI, or runtime chat gating.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Command runs without configured commercial providers and reuses the existing offline harness.
- [x] #2 Candidate input JSON must be an object keyed by PC-JUDGE case id; malformed JSON, non-object roots, and missing files fail cleanly.
- [x] #3 Command emits bounded JSON to stdout and can write the same report to an explicit output file path.
- [x] #4 Docs show usage and state the offline-only boundary and non-goals.
- [x] #5 Focused pytest, Bandit on touched Python, and diff hygiene are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing CliRunner coverage for a new tldw-evals persona-chat-judge review command, including success, explicit output file, missing file, malformed JSON, and non-object candidate roots. 2. Add a focused offline CLI module that loads the default V1 fixture and candidate JSON, validates object roots, calls build_persona_chat_judge_report(), and emits bounded sorted JSON to stdout and optional --output. 3. Register the command group on the unified evals CLI without importing providers or adding DB/Jobs/API/WebUI/runtime gating paths. 4. Document command usage and offline-only boundaries in the judge contract doc. 5. Run focused pytest, Bandit on touched Python, placeholder scan, and git diff --check; record evidence before finalizing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented TDD red/green for the offline review command. Red run failed with No such command persona-chat-judge. Green/closeout verification passed: python -m pytest tldw_Server_API/tests/Evaluations/unit/test_persona_chat_judge_review_command.py tldw_Server_API/tests/Evaluations/test_persona_chat_judge_harness.py tldw_Server_API/tests/Evaluations/test_persona_chat_judge_contract.py -q (23 passed, 5 warnings); python -m bandit -r tldw_Server_API/app/core/Evaluations/cli/persona_chat_judge_cli.py tldw_Server_API/tests/Evaluations/unit/test_persona_chat_judge_review_command.py -f json -o /tmp/bandit_persona_chat_judge_review_command.json (0 issues); placeholder scan returned no matches; git diff --check passed.

Draft PR opened: https://github.com/rmusser01/tldw_server/pull/1583. PR is draft because this AI-authored change still needs the required human-written Change summary before merge readiness.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added an offline `tldw-evals persona-chat-judge review` command that loads candidate outputs keyed by PC-JUDGE case id, validates JSON object roots, reuses the existing harness for bounded comparison reports, prints sorted JSON to stdout, and can write the same report to an explicit file. The contract doc now shows the workflow and reiterates the offline-only boundary: no model provider calls, database persistence, Jobs, API endpoint, WebUI state, runtime chat gating, or response mutation.

Known skips/blockers: none.

PR opened: https://github.com/rmusser01/tldw_server/pull/1583
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
