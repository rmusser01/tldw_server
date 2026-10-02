---
id: TASK-13347
title: >-
  Finish the ChatOps shell: support-module residue and the discord.py/slack.py
  pair
status: Done
assignee: []
created_date: '2026-09-22 23:52'
updated_date: '2026-09-23 18:06'
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
- [x] #1 support pair residue extracted: env accessors, OAuth config getters, installation-record shape
- [x] #2 discord.py/slack.py share the receipt, dedupe, rate-limit and job-submission path
- [x] #3 signature algorithm and command parser remain injected, not shared
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-09-23: 3cbee3a79a stage 2c (_chatops/settings.py; PolicyStore/resolve_actor_id); e3c31fc40b stage 3 (_chatops/ingress.py). Signature verification and command parsing remain per protocol. ADR-050 table updated.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stages 2c and 3 done. Support pair 78.2% -> 74.4% (largest identical run 123 -> 100: remaining run is HTTP/metric/crypto plumbing tests patch per module, kept local deliberately). Endpoint pair 61.3% -> 56.2% (remaining longest run is per-router OAuth/admin route declarations). Shared: env/OAuth settings, installation record, policy store, actor mapping, ingress responses, job submission, status-command scoping, job-status route, org resolution. Per protocol: signature algorithm, command parser, tenant fields. Tests: 732 passed; 14 failures identical on HEAD (router_groups_contract, auth_dependency_contract). Finding filed separately: the job-status HTTP route is unauthenticated.
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
