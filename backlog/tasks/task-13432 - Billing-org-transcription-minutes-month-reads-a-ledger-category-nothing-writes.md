---
id: TASK-13432
title: >-
  Billing: org transcription-minutes month reads a ledger category nothing
  writes
status: To Do
assignee: []
created_date: '2026-10-03 02:25'
labels:
  - billing
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the spec 2 review (Docs/Design/2026-10-02-usage-quota-posture-design.md, Known defects). BillingEnforcer._get_transcription_minutes_month (core/Billing/enforcement.py ~625-651) sums ResourceDailyLedger rows for entity_scope=org, category=minutes. Nothing writes org/minutes: transcription records org/cost_units (core/Billing/cost_units.py ~122-129, audio_transcriptions.py ~1162-1182) and user/minutes (core/Usage/audio_quota.py). So the hosted TRANSCRIPTION_MINUTES_MONTH limit only sees the enforcer's 60 s in-memory usage delta and never the month's real total.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The org monthly transcription-minutes usage reflects every transcription by org members in the calendar month
- [ ] #2 A test proves a hosted plan limit on transcription_minutes_month denies once the month's real usage exceeds it
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
