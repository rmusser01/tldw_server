---
id: TASK-257.2
title: Add Persona Chat judge calibration policy
status: Done
assignee: []
created_date: '2026-05-12 02:06'
updated_date: '2026-05-12 02:13'
labels:
  - persona-chat
  - evaluations
  - stage-2
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1586'
  - 'https://github.com/rmusser01/tldw_server/pull/1588'
  - 'https://github.com/rmusser01/tldw_server/issues/1566'
  - 'https://github.com/rmusser01/tldw_server/issues/1543'
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
documentation:
  - Docs/Reviews/PERSONA_CHAT_JUDGE_EVAL_CONTRACT_2026_05_11.md
  - Docs/Reviews/PERSONA_CHAT_JUDGE_EVALUATION_CONTRACT_2026_05_11.md
  - Docs/Reviews/PERSONA_CHAT_TRACE_ERROR_TAXONOMY_2026_05_10.md
  - Docs/Reviews/PERSONA_CHAT_QUALITY_EVAL_FOLLOWUP_2026_05_10.md
parent_task_id: TASK-257
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1586 as the next optional Persona Chat Stage 2 judge slice under #1566. Add a narrow calibration-policy layer over the existing V1 contract fixture, no-provider harness, and offline review command so maintainers can classify offline judge reports as advisory/review-only without adding live provider execution, DB persistence, Jobs, API/WebUI state, runtime chat gating, or response mutation. Policy output must stay trace-safe by exposing only bounded identifiers and reason keys.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A tested policy helper classifies clean, invalid, missing, extra, and low-agreement Persona Chat judge reports with stable status and reason fields.
- [x] #2 Policy output includes only bounded case/source ids and mismatch or reason keys, not raw prompts, assistant responses, exemplar text, memories, paths, secrets, or database content.
- [x] #3 Current synthetic fixture reports remain advisory and not production-calibrated because sample counts are below the production threshold.
- [x] #4 Documentation explains threshold policy, trace-safe link behavior, V1 review-only boundaries, failure modes, and residual risks.
- [x] #5 Focused pytest, Bandit for touched Python, placeholder scan, and git diff hygiene are recorded before closeout.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan file: Docs/superpowers/plans/2026-05-12-persona-chat-judge-calibration-policy.md

1. Add policy-helper tests first for advisory clean reports, blocked invalid/missing/extra/low-agreement reports, trace-safe serialization, and dict input compatibility.
2. Implement the minimal policy helper over the existing Persona Chat judge harness report without provider execution, persistence, Jobs, API/WebUI state, or runtime gating.
3. Update Persona Chat judge docs and Stage 2 follow-up docs with policy semantics, thresholds, trace-safe output, failure modes, and residual risks.
4. Run focused pytest, Bandit on touched Python, placeholder scan, and git diff hygiene before final Backlog closeout.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
GitHub issue: https://github.com/rmusser01/tldw_server/issues/1586

Draft PR: https://github.com/rmusser01/tldw_server/pull/1588

Implemented `tldw_Server_API/app/core/Evaluations/persona_chat_judge_policy.py` as a review-only policy layer over the existing Persona Chat judge harness report. The helper accepts the harness dataclass or bounded report dict, classifies reports as `advisory` or `blocked`, always keeps `runtime_gating_allowed` false, marks the current synthetic fixture as `sample_too_small`, and emits only case ids, source case ids, and reason/mismatch keys.

Added `tldw_Server_API/tests/Evaluations/test_persona_chat_judge_policy.py` with TDD coverage for advisory clean reports, invalid/missing/extra candidates, low agreement, dict report input, stable JSON serialization, malformed report blocking, and raw prompt/assistant text exclusion.

Updated Persona Chat judge review docs with policy status values, thresholds, trace-safe output semantics, V1 non-goals, failure modes, and residual risks.

Verification:

- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Evaluations/test_persona_chat_judge_policy.py tldw_Server_API/tests/Evaluations/test_persona_chat_judge.py tldw_Server_API/tests/Evaluations/test_persona_chat_judge_harness.py tldw_Server_API/tests/Evaluations/test_persona_chat_judge_contract.py tldw_Server_API/tests/Evaluations/unit/test_persona_chat_judge_review_command.py -q` passed: 43 passed, 5 warnings.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Evaluations/persona_chat_judge_policy.py tldw_Server_API/tests/Evaluations/test_persona_chat_judge_policy.py -s B101 -f json -o /tmp/bandit_persona_chat_judge_policy.json` passed: 0 findings.
- Placeholder scan over touched docs, plan, and Backlog task passed with no matches.
- `git diff --check` passed.

Known failure modes and residual risks:

- The current packaged fixture is intentionally too small for production calibration; policy results remain advisory until a reviewed held-out set meets threshold.
- The policy classifies already-produced reports only. It does not parse model output, execute providers, persist reports, or prove future adapter behavior.
- Trace-safe output depends on future callers continuing to pass bounded harness reports rather than raw candidate payloads into user-facing surfaces.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a review-only Persona Chat judge calibration policy over the existing offline harness and review command. The policy fails closed for malformed, missing, extra, invalid, or low-agreement reports; keeps clean synthetic fixture reports advisory until sample thresholds are met; and emits only bounded trace-safe ids and reason keys. Runtime Persona Chat behavior, provider execution, persistence, Jobs, API/WebUI state, and response mutation remain unchanged.
<!-- SECTION:FINAL_SUMMARY:END -->
