---
id: TASK-13415
title: >-
  CI setup-ffmpeg apt-get install has no timeout, so stalled mirrors kill jobs
  (3 runs on 2026-10-01)
status: Done
assignee: []
created_date: '2026-10-01 22:14'
updated_date: '2026-10-02 01:16'
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
- [x] #1 apt-get install in setup-ffmpeg gets bounded retries and network timeouts, and the step has its own wall-clock bound (e.g. timeout-minutes or a timeout wrapper with one retry) well below the job timeout
- [x] #2 A stalled mirror fails or retries the step quickly with a clear message instead of consuming the whole job timeout
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Log evidence (#3063 backend-required job 110569371819): the update went past the azure mirror (Ign after 20s), fetched archive.ubuntu.com backports and security InRelease, then stalled from 20:40:25 until the 21:09:43 cancel. The data trickled, so Acquire::https::Timeout=20 never fired. Fix in .github/actions/setup-ffmpeg/action.yml: apt_bounded() runs each apt-get update/install under timeout --kill-after=15s 300, 3 attempts, dpkg --configure -a between attempts, and ::error:: after the last one. install now gets the same Acquire options, plus Acquire::Languages=none. Verified locally with a timeout(1) shim and a stub apt-get: a hung attempt is killed at the bound and the retry succeeds (rc 0); a persistent failure stops after 3 attempts with the error (rc 1). The YAML parses and the script passes bash -n. A container test against real apt couldn't run because the local Docker daemon was unresponsive. Bandit and docs not applicable (CI YAML only).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
setup-ffmpeg now bounds each apt-get attempt at 300s with 3 retries, so a stalled mirror fails or retries in minutes with a clear error instead of consuming the 30-60 minute job timeout.
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
