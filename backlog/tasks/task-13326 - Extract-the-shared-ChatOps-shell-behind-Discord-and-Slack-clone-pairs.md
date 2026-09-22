---
id: TASK-13326
title: Extract the shared ChatOps shell behind Discord and Slack clone pairs
status: To Do
assignee: []
created_date: '2026-09-22 04:56'
labels:
  - duplication
  - api
  - integrations
dependencies: []
references:
  - tldw_Server_API/app/api/v1/endpoints/discord_oauth_admin.py
  - 'tldw_Server_API/app/api/v1/endpoints/discord_support.py:241'
  - 'tldw_Server_API/app/api/v1/endpoints/slack_support.py:241'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measured by difflib after s/discord/slack/; s/guild/team/:
- discord_oauth_admin.py vs slack_oauth_admin.py: 371/407 = 91.2% (NOT in the original review seed; found during verification)
- discord_support.py vs slack_support.py: 533/677 = 78.6%, with _error_response and _metric_labels byte-identical at the SAME line numbers (:241, :248)
- discord.py vs slack.py: 368/600 = 61.3%
Plus four test clone pairs, one at 100%.

The policy contract has ALREADY drifted into two vocabularies: team_quota_per_minute vs workspace_quota_per_minute; status_scope {team,team_and_user} vs {workspace,workspace_and_user}; default_response_mode gains a thread value on one side only.

Amplification is visible in the uncommitted working tree: the IDOR fix on GET /{discord|slack}/jobs/{job_id} had to be written FOUR times with byte-identical comment text.

Destination: app/api/v1/endpoints/_chatops/ owning the transport-agnostic shell (policy schema + normaliser + quota enforcement, installation record, OAuth state machine, receipt/dedupe wiring, metric-label and error-envelope helpers). The two protocol pieces stay injected: signature algorithm (Ed25519 vs HMAC-SHA256 v0=) and command parser.

Stage 1 is the 91.2% *_oauth_admin.py pair - smallest, highest identity, tested on both sides. Owner-only.

Source: synthesis F25
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design doc and ADR recorded
- [ ] #2 One policy vocabulary, with any intentional divergence documented
- [ ] #3 oauth_admin pair extracted first
- [ ] #4 Test clones collapsed to parametrised suites
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
