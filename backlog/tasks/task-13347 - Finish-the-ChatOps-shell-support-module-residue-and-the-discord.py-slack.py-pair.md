---
id: TASK-13347
title: >-
  Finish the ChatOps shell: support-module residue and the discord.py/slack.py
  pair
status: To Do
assignee: []
created_date: '2026-09-22 23:52'
labels:
  - duplication
  - api
  - integrations
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stages 2c and 3 of the extraction started in TASK-13326 (ADR-050 carries the staging table).

Stage 2c -- discord_support.py and slack_support.py are still 78.2% similar with a 123-line identical run after the policy schema, normaliser, evaluator, error envelope and metric labels were shared. What remains: the _env_int-based accessors, the OAuth config getters (_oauth_client_id / _oauth_redirect_uri / _oauth_auth_url / _oauth_token_url / _oauth_state_ttl_seconds), and _public_installation_record.

Stage 3 -- discord.py and slack.py at 61.3%. This is where the two genuinely per-protocol pieces live and must stay injected: the request signature algorithm (Ed25519 for Discord, HMAC-SHA256 v0= for Slack) and the command parser. Everything else -- the receipt/dedupe wiring, ingress rate limiting, and the job-submission path -- is shared. This is also the pair where the IDOR fix on GET /{discord|slack}/jobs/{job_id} had to be written four times.

Both are owner-only paths (app/api/v1/**).

Source: TASK-13326, synthesis F25.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 support pair residue extracted: env accessors, OAuth config getters, installation-record shape
- [ ] #2 discord.py/slack.py share the receipt, dedupe, rate-limit and job-submission path
- [ ] #3 signature algorithm and command parser remain injected, not shared
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
