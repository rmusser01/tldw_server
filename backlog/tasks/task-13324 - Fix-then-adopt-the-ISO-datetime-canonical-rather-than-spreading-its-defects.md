---
id: TASK-13324
title: Fix then adopt the ISO datetime canonical rather than spreading its defects
status: To Do
assignee: []
created_date: '2026-09-22 04:56'
labels:
  - duplication
  - correctness
  - migration
dependencies: []
references:
  - 'tldw_Server_API/app/api/v1/utils/datetime_utils.py:21'
  - 'tldw_Server_API/app/core/AuthNZ/repos/datetime_utils.py:6'
  - 'tldw_Server_API/app/services/workflows_webhook_dlq_service.py:45'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measured: 239 datetime.utcnow() sites in app/ (tz-naive, deprecated on the Python 3.12 CI runs), 39 private _utc_now/_now_iso definitions, and a canonical api/v1/utils/datetime_utils.py with 4 importers.

THE CANONICAL IS UNADOPTABLE, NOT MERELY UNADOPTED. coerce_datetime (21-47): returns naive datetimes unchanged despite promising tz-aware; tries strptime BEFORE fromisoformat so awareness depends on input format; returns now() for unparseable input. It also module-level-imports api/v1/schemas/chat_dictionary_schemas.TimedEffects - a shared datetime util coupled to one feature schema, which is why it has one importer.

AuthNZ repeats the shape one layer down: repos/datetime_utils.py:_strip_tzinfo exists with 3 importers and implements the LOSSY one of three semantics, against 11 copies and 40 inline replace(tzinfo=None) calls. Adopting it would push wrong semantics onto session-expiry and secret-rotation timestamps.

Anchor case: workflows_webhook_dlq_service.py:45 (tz-naive) vs meetings_webhook_dlq_service.py:40 (tz-aware, second-truncated) - same helper name, two sibling files, differing on both awareness and precision.

SEQUENCING: fix or replace the canonical FIRST (split parse_timed_effects out; new api/v1/utils/iso_datetime.py with utc_now_iso and parse_iso_utc), THEN migrate. Not the reverse.

Source: synthesis F23
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 parse_timed_effects split out of datetime_utils
- [ ] #2 A corrected parser exists whose awareness does not depend on input format and which never substitutes now()
- [ ] #3 AuthNZ _strip_tzinfo fixed to the correct semantics before its 11 copies adopt it
- [ ] #4 Migration is staged, not a single sweep
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
