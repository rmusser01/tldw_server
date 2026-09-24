---
id: TASK-13324
title: Fix then adopt the ISO datetime canonical rather than spreading its defects
status: In Progress
assignee: []
created_date: '2026-09-22 04:56'
updated_date: '2026-09-24 00:02'
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
- [x] #1 parse_timed_effects split out of datetime_utils
- [x] #2 A corrected parser exists whose awareness does not depend on input format and which never substitutes now()
- [ ] #3 AuthNZ _strip_tzinfo fixed to the correct semantics before its 11 copies adopt it
- [x] #4 Migration is staged, not a single sweep
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Commit 5921509761. Destination is core/Utils/iso_datetime.py, not api/v1/utils/iso_datetime.py as the description suggested: services and core modules are the adopters, and core -> api/v1 imports are the inversion the DB_Management review forbids. parse_iso_utc: datetime or ISO string (T or space, fractional, Z, offsets) -> aware UTC; naive taken as UTC; None/blank/garbage/non-str -> None (never now()). utc_now_iso: datetime.now(timezone.utc).isoformat(). AC1: parse_timed_effects moved into schemas/chat_dictionary_schemas.py beside TimedEffects; api/v1/utils/datetime_utils.py deleted (only importer was chat_dictionaries). chat_dictionaries keeps a local coerce_datetime that uses parse_iso_utc and makes the required-field now() fallback explicit + logged; _coerce_optional_datetime = parse_iso_utc. AC4 stage 1 adopters (bound under their old names so monkeypatching still works): _coerce_datetime in telegram_support, mcp_credential_broker_service, integrations_control_plane_service (identical semantics); _utc_now/_utc_now_iso/_utcnow_iso in Setup/{readiness_store,audio_readiness_store,readiness_service,audio_pack_service}, Workspaces/source_preview, Watchlists/briefing_fulfillment, Research_Workspace/output_jobs (byte-identical bodies). Dead utcnow _now_iso removed from workflows_webhook_dlq_service. Not migrated on purpose: second-truncated, Z/millisecond and isoformat(sep=' ') variants (different wire formats, which the DB_Management stage-3 review assigns to core/DB_Management/timestamps.py), and chat.py's _coerce_datetime. Red-before/green-after: _entry_dict_to_response on SQLite-format created_at returned a naive datetime on 4a84d02b55 (fails) and aware UTC now; tests/Utils/test_iso_datetime.py 18 pass. Suite diff: see TASK-13322 notes (same runs), zero new failures. Bandit -ll: clean. FINDING (not fixed, out of scope): workflows_webhook_dlq_service.py:379 writes next_attempt_at as naive 'YYYY-MM-DDTHH:MM:SS.ffffff' but Workflows_DB.list_webhook_dlq_due compares it lexically against SQLite datetime('now') 'YYYY-MM-DD HH:MM:SS', so a retry due later the same day always sorts after now ('T' > ' ') and waits until the date rolls over. OPEN: AC3 (AuthNZ repos/datetime_utils._strip_tzinfo should convert to UTC before dropping tzinfo: dt.astimezone(timezone.utc).replace(tzinfo=None)) not done: core/AuthNZ/repos/ is owned by another agent in this batch, so it is left to that owner.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
