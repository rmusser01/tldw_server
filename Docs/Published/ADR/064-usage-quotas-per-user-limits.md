# ADR-064: Usage quotas are per-user limits, off by default

**Status:** Accepted
**Date:** 2026-10-06
**Backfilled from:** not backfilled
**Decision owner:** Repository owner (@rmusser01)
**Related task:** TASK-13434
**Related spec/plan:** `Docs/Design/2026-10-02-usage-quota-posture-design.md`

## Decision

Usage quotas are per-user UserProfiles `limits.*` values, off by default (`USAGE_QUOTAS_ENABLED`), resolved per user from the user's own value, then the most generous team value, then the most generous org value; none is set by default.

## Context

A stock install enforced usage budgets nobody configured: 30 audio minutes a day from the "free" audio tier, 25 MB transcription uploads, 5 GB of storage per user, 10 chatbook exports a day, a billing "free plan" for every multi-user org, and more. Each quota had its own knob, several had no off switch, and none could be set per team or org. Self-hosters were limited by tier defaults they never chose.

The owner's commercial offering needs the opposite: no limit unless one is assigned, and an assignable value per user and per group. In the owner's words, the limits are only there for the commercial offering and can all be blank by default.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep the tiers (audio, evaluations, chatbooks) and add an off switch | Tiers apply a numeric default to everyone and cannot be assigned per team or org. Each module kept its own knob and its own store. |
| A shared group pool: a team or org quota is one budget its members draw down | The owner's decision is that a group value is each member's allowance. A pool also needs cross-member accounting at every site. The two pools that already exist (admin-created team/org storage pools, hosted org plans) are kept as they are. |
| The billing plan only | Billing plans need a billing repository, which the open-source server does not wire, and they are per org, not per user or team. |

## Consequences

- **One resolver.** Every enforcement site and every reporting view reads the same value. Order: the user's own value, otherwise the most generous value among the teams that set the key, otherwise the most generous among the orgs. Joining a group never lowers an allowance.
- **0 blocks.** A value of 0 means none allowed; no value anywhere means unlimited, and reporting shows `null`.
- **Counters always run.** The switch gates the check, never the write, so usage is recorded whether quotas are on or off and turning the switch on counts correctly.
- **Platform admin only.** `limits.*` values are editable only by platform admins, so a customer's org admin cannot lift members above the operator's value. Team and org values are written through `PUT`/`DELETE /api/v1/admin/{orgs|teams}/{id}/profile/overrides/{key}`.
- **Billing plans are separate and hosted-only.** They run only when a billing repository is wired in and the switch is on.
- **Deprecated, still present.** The audio tier admin API and `AUDIO_TIER_LIMITS_JSON` / `[Audio-Quota] {tier}_*` no longer affect any limit (the settings log a warning). `DEFAULT_STORAGE_QUOTA_MB` is deprecated and sets no one's quota.
- **The column stays, unread.** `users.storage_quota_mb` is kept (`NOT NULL DEFAULT 5120`) but is no longer read for enforcement. An upgrade migration copies non-default values into `limits.storage_quota_mb`; values equal to 5120 or to the configured default cannot be told apart from "never set" and are dropped.
- **Unchanged guardrails.** Per-file size caps, per-minute request rates and synchronous concurrency are not usage quotas and are not touched.
- **Operator guide.** See `Docs/Operations/Usage_Quotas.md`.

## Follow-up

- Audit events on the storage quota admin endpoints that do not emit one yet: TASK-13510, under TASK-13434.
