---
id: TASK-13326
title: Extract the shared ChatOps shell behind Discord and Slack clone pairs
status: Done
assignee: []
created_date: '2026-09-22 04:56'
updated_date: '2026-09-22 23:52'
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
- [x] #2 One policy vocabulary, with any intentional divergence documented
- [x] #3 oauth_admin pair extracted first
- [x] #4 Test clones collapsed to parametrised suites
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
STAGES 1, 2a, 2b and 4 DONE. Stage 2c and stage 3 remain. ADR-050 carries the staging table.

Commits: 360696f8be (stage 1), c901fd83b7 (2a), ff1200b9fb (2b), e895450c0a (stage 4).

AC1 DONE -- Docs/ADR/050-chatops-shared-shell.md.

AC3 DONE -- the 90.9% *_oauth_admin.py pair. Flow extracted to endpoints/_chatops/oauth_admin.py behind a ChatOpsOAuthProvider descriptor. Five genuine protocol differences carried by the descriptor (authorize query, token form, Slack's ok flag, installation-record fields, and 400-vs-502 for a missing workspace id); two contract differences described rather than unified (public key names, policy metric scope label). Wrappers keep the existing function names and keyword arguments so discord.py and slack.py are untouched. Verified beyond the suite: both installation records driven through the new path, key sets match the originals exactly.

AC2 DONE -- one policy schema and normaliser in _chatops/policy.py, parameterised by ChatOpsPolicySpec. The drift is real and is public API: guild_quota_per_minute vs workspace_quota_per_minute, status_scope {guild,guild_and_user} vs {workspace,workspace_and_user}, and default_response_mode gaining "thread" on Slack. Renaming any of them would break existing callers and stored policies, so each provider's spelling is declared beside the reason. The "thread" mode is a genuine Slack capability, not drift, and is recorded as such. Verified by differential test against the pre-refactor modules loaded from git: 28 payloads x 2 providers x (with and without an explicit base) = 112 comparisons, 0 mismatches.

Stage 2b (beyond the stated ACs) -- _evaluate_*_policy, _*_policy_error_response and _*_action_route were identical character for character once vocabulary is normalised away, 103 lines, and include the quota enforcement the shell is meant to own. Now shared behind ChatOpsPolicyRuntime. Differential-tested: 8 policy scenarios per provider, the quota-exhaustion path driven twice to force a 429, rendered error body/status/Retry-After for both branches, all six action routes -- 0 mismatches. Emitted metrics captured separately (the differential harness does not see them) and match name for name and label for label.

AC4 DONE -- measured across every test file naming either provider, not just the two obvious directories. Two pairs found: test_{discord,slack}_endpoint_sanitizers.py at 100%, collapsed into one parametrised test_chatops_endpoint_sanitizers.py; and the oauth_lifecycle pair, whose _FakeOAuthStateRepo and _FakeUserSecretRepo were 90 lines BYTE-identical with no normalisation needed, moved to tests/_chatops_helpers/fake_oauth_repos.py. The lifecycle tests themselves stay per-provider because they assert genuinely different OAuth payloads and endpoints. Lifecycle pair 84.9% -> 78.1%; sanitizer pair gone.

MEASURED RESULTS:
  oauth_admin pair   407+407 lines, 90.9% similar -> logic single-sourced (wrappers remain ~89% similar to each other, but that residue is pure parameter forwarding with no behaviour)
  support pair       677+678, 81.9%, largest identical run 187 lines
                  -> 564+565, 78.2%, largest identical run 123 lines
  test clones        100% pair eliminated; 84.9% pair -> 78.1%

STILL OPEN:
- Stage 2c: the support pair's remaining 123-line identical run -- env accessors, OAuth config getters, and the installation-record shape.
- Stage 3: discord.py / slack.py at 61.3%, which is where the signature algorithm (Ed25519 vs HMAC-SHA256 v0=) and the command parser live. Those two stay per-protocol by design; the rest does not.

Verification throughout: Discord + Slack + Integrations suites 68 passed, 0 failed, at every commit (baseline 68 passed). Bandit clean on every touched file. One bandit note: the descriptor field was named token_entity_field, which B106 flagged because it reads any kwarg name containing "token" as a credential; renamed to response_entity_field, which is also more accurate -- a rename beat a suppression.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
All four acceptance criteria met across four commits: the 90.9% oauth_admin pair extracted behind a provider descriptor, one policy schema and normaliser with the public vocabulary divergence declared where it is used, the identical policy evaluator shared, and both test clone pairs collapsed. Every extraction was differential-tested against the pre-refactor modules loaded from git rather than trusted to the suite. Stages 2c and 3 (the support-module residue and the discord.py/slack.py pair) are tracked separately; ADR-050 carries the staging table.
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
