# ADR-050: One ChatOps shell behind Discord and Slack

**Status:** Accepted
**Date:** 2026-09-22
**Backfilled from:** not backfilled
**Decision owner:** repository owner (decided 2026-09-22 during core-module review remediation)
**Related task:** TASK-13326
**Related spec/plan:** `Docs/superpowers/reviews/2026-09-21-core-module-duplication-synthesis.md` (F25)

## Decision

The transport-agnostic ChatOps shell lives in
`tldw_Server_API/app/api/v1/endpoints/_chatops/`. Discord and Slack keep one module
each, holding a `ChatOpsOAuthProvider` descriptor and thin wrappers; the flow itself
is written once.

Only two things stay genuinely per-protocol: the request **signature algorithm**
(Ed25519 for Discord, HMAC-SHA256 `v0=` for Slack) and the **command parser**.

Extraction is staged, smallest and highest-identity first:

| Stage | Pair | Identity at start | State |
| --- | --- | --- | --- |
| 1 | `*_oauth_admin.py` | 90.9% | **done** — flow extracted to `_chatops/oauth_admin.py` |
| 2a | `*_support.py` policy schema + normaliser + envelope | 81.9% (pair) | **done** — `_chatops/policy.py` |
| 2b | `*_support.py` policy store + evaluator + routes | — | **done** — pair now 78.2%, largest identical run 187 → 123 lines |
| 2c | `*_support.py` env accessors, OAuth config getters, installation-record shape; per-tenant policy store and actor mapping | — | **done** — `_chatops/settings.py`, `PolicyStore`/`resolve_actor_id` in `_chatops/policy.py`; pair 78.2% → 74.4%, largest identical run 123 → 100 lines. The remaining run is HTTP/metric/crypto plumbing that tests patch per module (`_http_afetch`, `log_counter`, `loads_envelope`), kept local on purpose |
| 3 | `discord.py` / `slack.py` | 61.3% | **done** — `_chatops/ingress.py`: rate-limit and duplicate responses, job submission, the `status` command's tenant/owner scoping, the job-status route and workspace org resolution; the endpoints' private copies of `_metric_labels` and the policy-error response are gone. Signature verification and command parsing stay per protocol. Pair 61.3% → 56.2%; the longest remaining run (95 lines) is the per-router OAuth/admin route declarations |
| 4 | the test clone pairs | one at 100% | **done** — 100% pair collapsed to one parametrised suite; the lifecycle pair's byte-identical fakes moved to `tests/_chatops_helpers/` |

Stage 2a settles the policy vocabulary question (AC2 of TASK-13326): one schema and
one normaliser, with each provider's public spelling declared beside the reason it
cannot simply be renamed.

## Context

Measured by `difflib` after normalising `discord`/`slack` and `guild`/`team`:

- `discord_oauth_admin.py` vs `slack_oauth_admin.py` — **370/407 = 90.9%**
- `discord_support.py` vs `slack_support.py` — 533/677 = 78.6%, with `_error_response`
  and `_metric_labels` byte-identical **at the same line numbers** (`:241`, `:248`)
- `discord.py` vs `slack.py` — 368/600 = 61.3%

The cost is not hypothetical. The IDOR fix on `GET /{discord|slack}/jobs/{job_id}` had
to be written **four times** with byte-identical comment text. And the policy contract
has already drifted into two vocabularies for the same concept:
`team_quota_per_minute` against `workspace_quota_per_minute`; `status_scope` of
`{team, team_and_user}` against `{workspace, workspace_and_user}`;
`default_response_mode` gained a `thread` value on one side only.

## What is genuinely different, and stays described rather than unified

Stage 1 found five real differences, all carried by the descriptor:

1. **Authorize query.** Discord sends `response_type=code` and an optional
   `permissions`; Slack sends neither.
2. **Token form.** Discord sends `grant_type=authorization_code`; Slack does not.
3. **Success test.** Slack's token response carries an `ok` flag that must be true.
4. **Installation record.** Discord keeps `refresh_token`; Slack keeps `enterprise_id`,
   `bot_user_id` and `authed_user_id` (the last lifted from a nested object).
5. **Missing-workspace status.** Discord answers **400** — its callback accepts
   `guild_id` on the query, so absence is a bad request. Slack derives the workspace
   solely from the token response, so absence is an upstream fault and it answers
   **502**.

Two further differences are **contracts, not drift to repair**:

- Public key names `guild_id`/`guild_name` against `team_id`/`team_name`. Changing
  either breaks clients.
- The policy metric scope label: `guild` against `workspace`. Changing either breaks
  existing dashboards and alerts.

Both are expressed as descriptor fields so the shell can emit each provider's own
vocabulary from one implementation.

## Consequences

- A fix to the OAuth install flow is written once. That is the whole point: the
  four-times IDOR fix is what this prevents next time.
- `discord_oauth_admin.py` and `slack_oauth_admin.py` keep their existing public
  function names and keyword arguments, so `discord.py` and `slack.py` are untouched
  by stage 1 and the blast radius is three files.
- 814 lines across the pair become 417 of wrappers plus 526 shared. The line count
  barely moves; the number of places a bug must be fixed halves, and for the test
  clones it quarters.
- The descriptor is additive: a third ChatOps provider is a descriptor and wrappers,
  not a third copy.
- The policy vocabulary drift (`team_*` vs `workspace_*`) is **not** addressed here —
  it lives in `discord.py`/`slack.py` and belongs to stage 3. It remains a live
  divergence until then.

## Addendum (2026-09-23): job-status route authorization

`GET /api/v1/{discord,slack}/jobs/{job_id}` required no authentication; its only guard
was that the job belonged to that integration (the "IDOR fix" above), which stops
cross-integration reads, not unauthenticated ones. It now requires a logged-in web user
and returns a job only to its owner, or to an active member of an org that installed
the job's guild/workspace unless that tenant's policy restricts status to the job owner
(`*_and_user`), matching the in-platform `status` command. Single-user mode sees every
job of the integration. Everything else is 404. TASK-13364.

