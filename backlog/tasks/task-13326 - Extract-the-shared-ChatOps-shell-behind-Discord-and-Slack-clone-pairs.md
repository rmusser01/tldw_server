---
id: TASK-13326
title: Extract the shared ChatOps shell behind Discord and Slack clone pairs
status: In Progress
assignee: []
created_date: '2026-09-22 04:56'
updated_date: '2026-09-22 23:42'
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
- [x] #1 Design doc and ADR recorded
- [ ] #2 One policy vocabulary, with any intentional divergence documented
- [x] #3 oauth_admin pair extracted first
- [ ] #4 Test clones collapsed to parametrised suites
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
STAGE 1 DONE in 360696f8be. Stages 2-4 remain open -- this task is deliberately staged, see ADR-050.

AC1 (done): Docs/ADR/050-chatops-shared-shell.md records the destination, the staging table, what is genuinely per-protocol, and what is a contract to be described rather than unified.

AC3 (done): the 90.9% *_oauth_admin.py pair is extracted. Re-measured independently before starting: 370/407 matched lines = 90.9% (the review said 371/407 = 91.2%; the small delta is normalisation, the finding stands). The flow now lives once in endpoints/_chatops/oauth_admin.py; each provider keeps a ChatOpsOAuthProvider descriptor plus thin wrappers.

Five genuine protocol differences found and carried by the descriptor:
1. Authorize query -- Discord sends response_type=code and an optional permissions; Slack sends neither.
2. Token form -- Discord sends grant_type=authorization_code; Slack does not.
3. Success test -- Slack's token response carries an ok flag that must be true.
4. Installation record -- Discord keeps refresh_token; Slack keeps enterprise_id, bot_user_id and authed_user_id (the last lifted from a nested object).
5. Missing workspace id -- Discord answers 400 (its callback accepts guild_id on the query, so absence is a bad request); Slack answers 502 (it derives the id solely from the token response, so absence is an upstream fault).

Two differences are CONTRACTS and are described, not unified: the public key names guild_id/guild_name vs team_id/team_name (breaking clients), and the policy metric scope label "guild" vs "workspace" (breaking dashboards). Both are descriptor fields so one implementation emits each provider's own vocabulary.

Design choice worth recording: the wrappers keep the existing public function names and keyword arguments, so discord.py and slack.py are untouched and the blast radius is three files. The consequence is that the two wrapper modules remain ~89% similar to each other -- but that residue is pure parameter forwarding with no behaviour, and the logic behind it is single-sourced. Collapsing the wrappers too would mean renaming call sites in discord.py/slack.py, which are the stage 3 files; doing it now would have merged two stages and widened the risk on owner-only API surface.

Verification beyond the suite: both installation records were driven through the new path and their key sets compared against the originals field by field -- exact match on both sides. Discord + Slack + Integrations suites: 68 passed, 0 failed (baseline 68 passed). Bandit clean.

Bandit note: the descriptor field was first named token_entity_key, then token_entity_field; B106 reads any kwarg name containing "token" as a credential and flagged the literals "guild"/"team". Renamed to response_entity_field, which is also the more accurate name -- a rename beat a nosec suppression.

STILL OPEN:
- AC2 (one policy vocabulary): the drift -- team_quota_per_minute vs workspace_quota_per_minute, status_scope {team,team_and_user} vs {workspace,workspace_and_user}, default_response_mode gaining a thread value on one side only -- lives in discord.py/slack.py and belongs to stage 3. It is a live divergence until then.
- AC4 (test clones): the four clone pairs, one at 100%, are stage 4.
- Stage 2: the *_support.py pair at 78.6%, where _error_response and _metric_labels are byte-identical at the same line numbers (:241, :248).
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
