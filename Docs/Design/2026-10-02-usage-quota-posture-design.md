# Usage quotas: off by default, set per user, team or org

- **Date:** 2026-10-02
- **Status:** Draft, awaiting owner review
- **Backlog:** implementation tasks are filed after this spec is approved.
- **Scope:** Spec 2 of 2. Spec 1, the ingress safety net (`Docs/Design/2026-09-29-rg-ingress-safety-net-design.md`, ADR-056), is implemented. Spec 1 handles request rates; this spec handles usage budgets.

## Owner decisions (2026-10-02)

1. **One master switch, off by default.** In the owner's words: "each quota should be easily configurable per-user/group and not a blanket everyone."
2. **A quota set on a team or org is each member's allowance.** It is not a shared pool.
3. **Quota values live as UserProfiles `limits.*` keys,** resolved by one resolver.
4. **Every quota is blank (unlimited) by default, concurrency caps included.** In the owner's words: "the limits are only there for my commercial offering. I'm fine with them all being blank by default." No numeric quota default ships. This supersedes an earlier answer that treated concurrency caps as raised server protection.

## Problem

A stock install enforces usage budgets that nobody configured:

| Quota | Default today | Who it hits | Off switch today |
|---|---|---|---|
| Audio transcription minutes | 30/day ("free" audio tier, every user) | single-user and multi-user | none |
| Audio transcription upload size | 25 MB (same tier) | both | none |
| Billing "free plan" | 300k LLM tokens/month (402 on chat), 50 RAG queries/day, 60 transcription minutes/month, 100 API calls/day (counts every media ingest, embeddings and prompts call), 1 GB org storage, 1 concurrent job | every multi-user org | `LIMIT_ENFORCEMENT_ENABLED=false` |
| Per-user storage | 5 GB, written into every user row; media ingest checks it with no single-user exemption | both | none; the admin API rejects values below 100 MB |
| Chatbooks | 10 exports and 10 imports per day, 2 concurrent | both | `CHATBOOKS_DISABLE_QUOTAS` (partial) |
| Media ingest | 2 concurrent jobs, 2 GiB/day | both | only `RG_ENABLED=false` |
| Workflows | 1000 runs/day | both | `WORKFLOWS_DISABLE_QUOTAS`, or `RG_ENABLED=false` |
| Evaluations | 500/day and 500k tokens/day (RG policy); $1/day and $10/month cost caps (always on) | both | partial |
| Audio concurrency | 2 streams, 2 jobs | both | only `RG_ENABLED=false` |

Each quota has its own knob, spelled its own way. Several have no off switch at all, and none can be set per team or org. Three also have defects:
- **Audio `/limits` reports numbers nothing enforces.** It shows per-tier concurrency (1/3/10), while every user is actually capped at 2.
- **Billing plans can't be assigned in OSS.** `SubscriptionService` is built without a billing repo, so every org gets the free plan's numbers.
- **Chatbooks has dead quota code.** `check_storage_quota` is never called.

These already default to off and are not touched: the `JOBS_QUOTA_*` limits, the embeddings and ingest per-tenant limits, chat token daily caps, and the Prompt Studio limits.

## Goals

1. A stock install, single-user or multi-user, never returns a usage-quota 402, 413 or 429.
2. With the switch on, a quota applies only to the users, teams and orgs an admin assigned it to. Every quota is set the same way.
3. The hosted product turns everything on with one setting and keeps its billing-plan hook.

## Non-goals

- **Request rates.** Those are spec 1. Per-minute rates in module limiters, such as evaluations per minute, are unchanged.
- **Fixed per-request guardrails.** Per-file upload size caps (`[Media-Processing] max_*_file_size_mb`, the chatbooks 100 MB file cap) and character-chat count caps stay as they are.
- **Pooled group budgets.** The two that exist stay: admin-created team/org storage pools, and the hosted product's org plans.
- **New admin UI.** The existing profile editor and the profile bulk API set `limits.*` keys.
- **Carrying tier assignments forward.** Audio, evaluations and chatbooks tiers are not migrated.

## Design

### 1. Master switch

Add `usage_quotas_enabled()` to `core/config.py`, next to `rg_enabled()`. It resolves in this order:

1. env `USAGE_QUOTAS_ENABLED`;
2. env `LIMIT_ENFORCEMENT_ENABLED`, if explicitly set. This is the legacy spelling; it logs a one-time deprecation warning, and it keeps hosted deploys that set it working;
3. config.txt `[Usage-Quotas] enabled`;
4. otherwise **false**.

The switch is independent of `rg_enabled()`. Every quota check below runs only when the switch is on. `Billing.enforcement.enforcement_enabled()` becomes a thin alias for it.

### 2. One resolver

`core/Usage/quota_resolver.py`:

```python
async def user_quota(user_id: int, key: str) -> float | None:
    """The user's effective limits.<name> value, or None for unlimited."""
```

- **Returns None (unlimited)** when the switch is off, or when no value is set for the user, any of their teams, or any of their orgs.
- **Precedence:** the user's own override wins; otherwise the **most generous** value among the user's teams; otherwise the most generous among their orgs. Joining a group never lowers an allowance.
- **Zero means none allowed.** An explicit 0 blocks. Deleting the override returns the user to unlimited.
- **Storage:** it reads the UserProfiles override repos directly (user, team and org tables), so it doesn't build the full effective profile.
- **Caching:** results are cached per user for 60 s. Writing a user override clears that user's entry; writing a team or org override clears the whole cache.
- **Lookup errors fail open.** An error returns None and logs a warning; a DB hiccup must not block users. (The hosted billing path keeps its own `BILLING_ENFORCEMENT_FAILURE_MODE`.)
- **The profile view uses the same rule.** UserProfiles' effective-config builder uses the most-generous rule for `limits.*` keys, so the profile shows the value that is enforced. Today the builder picks the lowest group id.

### 3. Catalog keys

Every key goes in `Config_Files/user_profile_catalog.yaml` with:
- `default: null`;
- `minimum: 0`;
- `editable_by: [org_admin, platform_admin]`;
- type integer, except the cost keys, which are numbers.

Every key is a plain layered override. The special write paths go away:
- `limits.storage_quota_mb` stops writing `users.storage_quota_mb`;
- `limits.evaluations_*` stops writing the evaluations tier store.

| Key | Status | Counter (existing unless noted) |
|---|---|---|
| `limits.audio_daily_minutes` | exists | `ResourceDailyLedger` audio minutes |
| `limits.transcription_minutes_per_month` | new | sum of the same ledger over the calendar month |
| `limits.audio_concurrent_jobs` | exists | active audio job leases |
| `limits.audio_concurrent_streams` | new | active stream leases |
| `limits.llm_tokens_per_month` | new | `llm_usage_log`, summed per user over the calendar month |
| `limits.rag_queries_per_day` | new | `ResourceDailyLedger` RAG queries, per user. If today's rows are keyed per org only, also record them per user. |
| `limits.storage_quota_mb` | exists | `users.storage_used_mb` |
| `limits.media_ingest_mb_per_day` | new | `ResourceDailyLedger` ingestion bytes |
| `limits.media_concurrent_jobs` | new | media ingest job leases |
| `limits.workflows_runs_per_day` | new | the workflows run ledger |
| `limits.evaluations_per_day` | exists | evaluations daily counter |
| `limits.evaluation_tokens_per_day` | new | evaluations token counter |
| `limits.evaluation_cost_per_day_usd`, `limits.evaluation_cost_per_month_usd` | new | evaluations cost tracking |
| `limits.chatbooks_exports_per_day`, `limits.chatbooks_imports_per_day` | new | export and import job tables |
| `limits.chatbooks_concurrent_jobs` | new | active chatbook jobs |

`limits.evaluations_per_minute` and the `limits.prompt_studio_*` keys are unchanged.

### 4. Enforcement

At each site: `limit = await user_quota(user_id, key)`. If it is None, skip the check. Otherwise compare the existing counter plus this request's units against the limit, and deny with the site's **existing** status code and message (402, 413 or 429, as today). Built-in defaults and tier lookups are deleted from every site.

| Key(s) | Site |
|---|---|
| audio minutes | `core/Usage/audio_quota.py` check/consume (the `TIER_LIMITS` lookup goes); HTTP 402 in `audio_transcriptions.py`; WebSocket close in `audio_streaming.py` |
| transcription minutes/month | the billing re-check in `audio_transcriptions.py` and `audio_streaming.py` |
| audio concurrency | `audio_quota.can_start_job` / `can_start_stream` |
| LLM tokens/month | the `LimitEnforcer` site in `chat.py` |
| RAG queries/day | `rag_unified.py`, `text2sql.py`, `RAG/rag_service/transport.py` |
| storage | `storage_quota_service.check_quota`, called from media ingest (`persistence.py`) and `file_artifacts_service.py` |
| media bytes/day, media concurrency | `Ingestion_Media_Processing/persistence.py` |
| workflows runs/day | `workflows.py::_enforce_workflows_daily_cap` |
| evaluations | `Evaluations/user_rate_limiter.py`, for the daily caps and the cost caps |
| chatbooks | `Chatbooks/chatbook_service.py::_check_chatbook_job_admission` |

**Admin-created team/org storage pools** (`guard_storage_quota`) keep their shared-pool semantics but also run only when the switch is on. `STORAGE_QUOTA_ENFORCEMENT` stays as an extra off switch.

**Daily and monthly budgets don't depend on the rate limiter.** They read their counters directly, not through the RG governor.

**Concurrency keys still count through the existing RG leases.** The site passes the user's limit in place of the policy's `max_concurrent`, so a concurrency quota also needs `RG_ENABLED`. This is stated in the docs; the hosted product runs with both switches on.

### 5. Billing hook

- **The OSS free plan becomes all-unlimited.** `Billing/plan_limits.py` `DEFAULT_LIMITS[FREE]` sets every limit to -1; the feature flags are unchanged. With no billing repo, which is every OSS install, org-level billing checks therefore always pass, and the per-user `limits.*` checks do the work.
- **A hosted deploy keeps its plans.** A deploy that wires a `billing_repo` into `SubscriptionService` keeps its plan enforcement as today, under the master switch.
- **These dependencies stay** and stop limiting anything in OSS: the `API_CALLS_DAY` and `STORAGE_MB` dependencies on media, embeddings and prompts endpoints, and the org `CONCURRENT_JOBS` check.

### 6. Removals

- **Audio tiers.**
  - `TIER_LIMITS` no longer feeds enforcement. `AUDIO_TIER_LIMITS_JSON` and `[Audio-Quota] {tier}_*` are deprecated, with a warning when set.
  - The tier admin API (`PUT/GET /api/v1/audio/jobs/admin/tiers/{user_id}`) stays, is documented as deprecated, and no longer affects limits.
  - The 25 MB transcription upload cap is replaced by `[Media-Processing] max_audio_file_size_mb`.
- **The RG policy YAML loses its quota blocks:**
  - `media.default` `ingestion_bytes` and `jobs`;
  - `audio.default` `streams`, `jobs` and the dead `minutes`;
  - `workflows.default` `workflows_runs`;
  - the `evaluations` and `tokens` daily caps in `evals.*`.
  - Their request-rate entries stay.
- **Evaluations tiers.** The FREE/BASIC/PREMIUM/ENTERPRISE/CUSTOM tables stop driving the daily and cost caps.
- **Chatbooks tiers.** `QuotaManager`'s tier tables and the dead `check_storage_quota` go. `CHATBOOKS_DISABLE_QUOTAS` and `WORKFLOWS_DISABLE_QUOTAS` are still honored as extra off switches.
- **Per-user storage default.** The `DEFAULT_STORAGE_QUOTA_MB` setting is deprecated and ignored, and registration stops writing a quota.

### 7. Upgrading existing installs

- **Self-hosters have nothing to do.** Quotas they never configured stop applying.
- **Hosted deploys set `USAGE_QUOTAS_ENABLED=true`.** `LIMIT_ENFORCEMENT_ENABLED=true` keeps working, with a warning.
- **Per-user storage.** An AuthNZ migration (SQLite and Postgres) copies every `users.storage_quota_mb` value **other than 5120** into a user-scope `limits.storage_quota_mb` override, and enforcement stops reading the column. 5120 can't be told apart from the old default, so it is dropped; an admin who chose exactly 5 GB sets it again.
  - `PUT /admin/quotas/user/{id}` writes the override. Its 100 MB minimum drops to 0.
- **Tiers.** Audio, evaluations and chatbooks tier assignments are not carried over. The release notes say so.

### 8. Reporting

Every endpoint that reports a quota reports the resolver's value, and says "unlimited" when it is None:
- `/api/v1/audio/limits` and `/api/v1/audio/jobs/limits`;
- the chatbooks usage summary;
- the evaluations limits view;
- the user profile `quotas` section.

## Testing

1. **The switch:** off by default; precedence of `USAGE_QUOTAS_ENABLED`, then legacy `LIMIT_ENFORCEMENT_ENABLED`, then config.txt; the warning on the legacy spelling.
2. **The resolver:**
   - a user override beats teams, and teams beat orgs;
   - the most generous of two teams wins;
   - None when unset or when the switch is off;
   - 0 blocks;
   - the cache clears on override writes;
   - a lookup error fails open.
3. **A stock install admits past every old default,** with the switch off and with it on but nothing assigned. In single-user and multi-user, one test per site:
   - 31 audio minutes, and a 26 MB transcription upload;
   - 51 RAG queries, and 300,001 LLM tokens in a month;
   - a 5 GB + 1 MB upload;
   - 11 chatbook exports;
   - 3 concurrent media jobs, and 2 GiB + 1 of ingest in a day;
   - 1001 workflow runs;
   - 501 evaluations.
4. **Assigned values deny one unit over,** each set at user, team and org level, with the site's existing status code.
5. **The hosted path still works:** a fake `billing_repo` still enforces its plan limits.
6. **The storage migration** keeps non-5120 values as overrides and drops 5120, on SQLite and on Postgres (via the AuthNZ Postgres fixture).
7. **Reporting endpoints** say "unlimited" by default.
8. **Tests that pinned the old defaults** are rewritten to this contract. Before the switch flips, grep every symbol whose default changes and run every hit: the billing free-tier tests, the audio quota unit tests, the evaluations tier tests, and the storage 5120 fixtures.

## Delivery

Three PRs against `dev`:

- **PR A (relief).** Adds `usage_quotas_enabled()` and gates every site in §4 on it. Self-hosters are relieved at once. With the switch on, behavior is unchanged from today, so hosted deploys that set `LIMIT_ENFORCEMENT_ENABLED=true` keep their limits.
- **PR B (per-user values).** The resolver, the catalog keys, the most-generous rule, every site reading the resolver, the free plan made unlimited, and the removals in §6.
- **PR C (upgrade and docs).** The storage migration, the reporting endpoints, ADR-058 and the docs.
  - **Env and config:** `Env_Vars.md` and config.txt `[Usage-Quotas]`.
  - **Rewritten:** the billing section of `Organization_Administration.md`, which documents a `BILLING_ENABLED` flag and `/billing/*` routes that don't exist.
  - **Updated:** the chatbooks code guide and the storage API doc's endpoint citations.
  - **New:** a quotas page in Operations.
  - The `Docs/Published` mirror is refreshed for each mirrored doc.
