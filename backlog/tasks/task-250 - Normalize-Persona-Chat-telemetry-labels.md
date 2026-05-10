---
id: TASK-250
title: Normalize Persona Chat telemetry labels
status: Done
assignee:
  - Codex
created_date: '2026-05-10 23:08'
updated_date: '2026-05-10 23:49'
labels:
  - persona
  - chat
  - stage-2
  - telemetry
  - tests
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1557'
  - 'https://github.com/rmusser01/tldw_server/issues/1543'
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
documentation:
  - Docs/Reviews/PERSONA_CHAT_TRACE_ERROR_TAXONOMY_2026_05_10.md
  - Docs/superpowers/plans/2026-05-10-persona-chat-telemetry-labels.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1557 as the next Stage 2 Persona Chat quality slice. Normalize ordinary persona-backed chat telemetry/debug labels so persona chat records carry assistant_kind=persona and stable persona identity where telemetry/debug surfaces already exist, while preserving character-chat behavior and avoiding provider payload changes. Keep scope out of Persona Live renderer, visual packs, VN/CYOA, native companion, and LLM-as-judge work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Persona-backed chat telemetry/debug records distinguish assistant_kind=persona from character and generic chat records.
- [x] #2 Persona identity labels are stable and redaction-safe in existing telemetry/debug surfaces.
- [x] #3 Character chat telemetry/debug behavior remains covered or unaffected.
- [x] #4 PC-TEL-001 is linked to deterministic regression coverage or an explicitly documented surface gap.
- [x] #5 Focused backend/frontend tests, Bandit where applicable, and git diff hygiene are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan: Docs/superpowers/plans/2026-05-10-persona-chat-telemetry-labels.md

Scope: normalize ordinary persona-backed chat telemetry/debug labels for PC-TEL-001 without changing Persona Live, visual packs, VN/CYOA, or LLM-judge behavior.

Planned steps:
1. Add failing PC-TEL-001 regression coverage in persona telemetry metrics summary tests.
2. Extend chat telemetry hook labels and alert-window key with assistant_kind/assistant_id while preserving character_id.
3. Add metrics summary grouping by assistant_kind and assistant_id.
4. Run focused pytest, Bandit on touched backend files, and git diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created implementation plan for the telemetry-label normalization slice before production code changes.

Implemented persona telemetry label normalization. Touched files:
- tldw_Server_API/app/api/v1/endpoints/chat.py
- tldw_Server_API/app/core/Evaluations/persona_telemetry_metrics.py
- tldw_Server_API/app/core/Metrics/metrics_manager.py
- tldw_Server_API/tests/Evaluations/test_persona_telemetry_metrics_summary.py
- tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py
- Docs/superpowers/plans/2026-05-10-persona-chat-telemetry-labels.md

Verification:
- RED: focused PC-TEL-001 tests failed before implementation with missing samples_by_assistant_kind / missing persona telemetry labels.
- GREEN: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Evaluations/test_persona_telemetry_metrics_summary.py tldw_Server_API/tests/Persona/test_persona_chat_quality_fixtures.py tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py::test_persona_backed_chat_records_telemetry_with_persona_identity_labels -q -> 12 passed.
- Adjacent existing integration: test_persona_backed_chat_appends_persona_exemplar_guidance_in_runtime_path -> 1 passed.
- Full chat integration file attempted but timed out in TestClient app-lifecycle setup after earlier tests passed; narrower deterministic checks are recorded above.
- Bandit touched backend files -> 0 findings in /tmp/bandit_persona_chat_telemetry_labels.json.
- git diff --check -> clean.

Review-fix pass for PR #1558 addressed 5 actionable review threads: preserved legacy character metric labels, restored character IOO reader/writer key compatibility, sanitized and bounded persona assistant_id labels, capped persona IOO alert-window storage with LRU eviction, used persona_assistant_id for persona-backed chat telemetry, and switched metrics sample counts to cumulative histogram series when available.

Review-fix verification:
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/unit/test_persona_telemetry_hooks.py tldw_Server_API/tests/Evaluations/test_persona_telemetry_metrics_summary.py tldw_Server_API/tests/Persona/test_persona_chat_quality_fixtures.py tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py::test_persona_backed_chat_records_telemetry_with_persona_identity_labels tldw_Server_API/tests/Metrics/test_metrics_logger_registry_bridge.py -q -> 24 passed, 5 warnings.
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/chat.py tldw_Server_API/app/core/Evaluations/persona_telemetry_metrics.py tldw_Server_API/app/core/Metrics/metrics_manager.py -f json -o /tmp/bandit_persona_chat_telemetry_labels_review.json -> 0 findings.
- python -m py_compile touched production modules -> exit 0.
- git diff --check -> exit 0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Normalized Persona Chat telemetry so persona-backed chat metrics now carry assistant_kind and assistant_id labels while preserving character_id compatibility. Added summary grouping by assistant identity through MetricsRegistry, regression coverage linked to PC-CASE-019 / PC-TEL-001, and recorded focused verification plus Bandit and diff hygiene.
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
