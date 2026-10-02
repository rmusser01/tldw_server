---
id: TASK-13415
title: >-
  CI setup-ffmpeg apt-get install has no timeout, so stalled mirrors kill jobs
  (3 runs on 2026-10-01)
status: To Do
assignee: []
created_date: '2026-10-01 22:14'
labels:
  - ci
  - infra
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
.github/actions/setup-ffmpeg/action.yml passes Acquire::Retries and http/https timeouts to apt-get update, but its apt-get install calls (ffmpeg, portaudio19-dev, python3-all-dev) have none. On 2026-10-01 the step hung until the job timeout three times: #3065 core-security shard (60 min, run 36887984839), #3063 backend-required (30 min, run 36918098987) and sync-pc-conflicts shard (60 min, run 36918099133). Each one cancelled a required or full-suite job and forced a rerun.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 apt-get install in setup-ffmpeg gets bounded retries and network timeouts, and the step has its own wall-clock bound (e.g. timeout-minutes or a timeout wrapper with one retry) well below the job timeout
- [ ] #2 A stalled mirror fails or retries the step quickly with a clear message instead of consuming the whole job timeout
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
