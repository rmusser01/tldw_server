# Usage quotas: off by default, set per user, team or org

- **Date:** 2026-10-02
- **Status:** Draft, revised after code review; awaiting owner review
- **Backlog:** implementation tasks are filed after this spec is approved. Pre-existing bugs found during review are filed now (see [Known defects](#known-defects-outside-this-spec)).
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
| Audio transcription minutes | 30/day (the "free" audio tier applies to every user) | single-user and multi-user | none |
| Audio transcription upload size | 25 MB (same tier) | both | none |
| Queued audio jobs | 1 per user (the tier's `concurrent_jobs`, counted by the audio Jobs worker) | both | none |
| Billing "free plan" | 300k LLM tokens/month (402 on chat), 50 RAG queries/day, 60 transcription minutes/month, 100 API calls/day, 1 GB org storage | every multi-user org | `LIMIT_ENFORCEMENT_ENABLED=false` |
| Per-user storage | 5 GB, written into every user row and checked by media ingest, downloads and generated files, with no single-user exemption | both | none; admin APIs reject values below 100 MB |
| Chatbooks | 10 exports and 10 imports per day, 2 concurrent jobs | both | `CHATBOOKS_DISABLE_QUOTAS` (partial) |
| Media ingest | 2 concurrent requests, 2 GiB/day | both | only `RG_ENABLED=false` |
| Workflows | 1000 runs/day | both | `WORKFLOWS_DISABLE_QUOTAS`, or `RG_ENABLED=false` |
| Evaluations | `evals.free` RG policy: 100/day and 100k tokens/day | both | only `RG_ENABLED=false` |
| Audio streams and synchronous jobs | 2 each (RG leases) | both | only `RG_ENABLED=false` |

Notes on the billing row:
- With the switch on, a multi-user account with no active org gets **403** on every endpoint that carries a billing dependency.
- `API_CALLS_DAY` gates chat, transcription, media, embeddings and prompts. It counts from `usage_daily`, which is written only when `USAGE_LOG_ENABLED`, so on stock installs it reads 0.
- Billing `CONCURRENT_JOBS` and `TEAM_MEMBERS` are defined but never enforced.

Each quota has its own knob, spelled its own way, and several have no off switch at all. None can be set per team or org. There are defects too:
- **Dead plan assignment.** `SubscriptionService` is built without a billing repo, so every org gets the free plan's numbers.
- **Inert cost caps.** The evaluations cost caps can never fire: every caller passes `estimated_cost=0.0`, and the monthly cap is never compared.
- **Double-counted evaluations.** They land twice in the ledger (once from the RG reserve, once from a shadow write), and the ledger's `user/tokens` category mixes chat and evaluation tokens.
- **Stale "used" figures.** The audio limits endpoints report "used minutes" from a table nothing writes any more.
- **A broken endpoint.** `PUT/GET /api/v1/admin/storage-quotas/users/{user_id}` passes the user id as an org id.

These already default to off and are not touched: the `JOBS_QUOTA_*` limits, the embeddings and ingest per-tenant limits, chat token daily caps, and the Prompt Studio limits.

## Goals

1. A stock install, single-user or multi-user, never returns a usage-quota 402, 403, 413 or 429.
2. With the switch on, a quota applies only to the users, teams and orgs a platform admin assigned it to. Every quota is set the same way.
3. The hosted product keeps its billing-plan enforcement unchanged, behind one setting.

## Non-goals

- **Request rates.** Those are spec 1. Per-minute rates in module limiters, such as evaluations per minute, are unchanged.
- **Fixed per-request guardrails.** Per-file upload size caps (`[Media-Processing] max_*_file_size_mb`, the chatbooks 100 MB file cap) and character-chat count caps stay as they are.
- **Pooled group budgets.** The two that exist stay: admin-created team/org storage pools, and the hosted product's org plans.
- **New admin UI.** The profile editor and the new override routes are the interface.
- **Carrying tier assignments forward.** Audio, evaluations and chatbooks tiers are not migrated.
- **Deferred, each filed as a follow-up task:**
  - per-user concurrency on synchronous paths (media ingest, audio streams, synchronous transcription), which needs an RG per-request limit;
  - evaluations cost caps, which would be new enforcement;
  - token gating at LLM entry points other than `/chat/completions`.

## Design

### 1. Master switch

Add `usage_quotas_enabled()` to `core/config.py`, next to `rg_enabled()`. It resolves in this order:

1. env `USAGE_QUOTAS_ENABLED`;
2. env `LIMIT_ENFORCEMENT_ENABLED`, if explicitly set. This is the legacy spelling and logs a one-time deprecation warning;
3. config.txt `[Usage-Quotas] enabled`;
4. otherwise **false**.

Rules:
- **Independent of the rate limiter.** The switch does not depend on `rg_enabled()`. `Billing.enforcement.enforcement_enabled()` becomes a thin alias for it.
- **Gate the check, never the record.** The switch gates every quota *check*. Usage counters are always written, so turning the switch on mid-day or mid-month counts correctly.
- **Commercial-mode warning.** If a billing repo is wired (commercial mode) while the switch is off, startup logs a WARNING saying plan limits are not enforced. Nothing in this repo sets `LIMIT_ENFORCEMENT_ENABLED`, so a deploy that relied on its old `true` default would otherwise lose enforcement silently.

### 2. One resolver

`core/Usage/quota_resolver.py`:

```python
async def user_quota(user_id: int, key: str) -> float | None:
    """The user's effective limits.<name> value, or None for unlimited."""
```

- **Returns None (unlimited)** when the switch is off, or when no value is set for the user, any team they belong to, or any org they belong to.
- **Precedence.**
  - The user's own value wins.
  - Otherwise the **most generous** value among the user's teams **that set the key**.
  - Otherwise the most generous among the orgs that set it.
  - 0 is the least generous value and means none allowed. Joining a group never lowers an allowance.
- **Storage:** it reads the UserProfiles override repos directly (user, team and org tables), not the full effective profile.
- **Caching.**
  - Results are cached per user for 60 s.
  - In the writing process, a user write clears that user's entry, and a team or org write clears the whole cache.
  - Other workers see a change within 60 s.
- **Lookup errors fail open.** An error returns None and logs a warning.
- **Counters cost nothing when unlimited.** A site reads its counter only when the resolver returns a value.
- **The profile view uses the same rule.** UserProfiles' effective-config builder uses the same precedence for `limits.*` keys, so the profile shows the value that is enforced. Today the builder picks the lowest group id.

### 3. Setting values

- **Every key is a plain layered override.** `update_service.py` today accepts only five whitelisted `limits.*` keys, through special write paths. A generic branch replaces them: any catalog `limits.*` key is written to the user override table.
  - `limits.storage_quota_mb` stops writing `users.storage_quota_mb`.
  - `limits.evaluations_per_day` stops writing the evaluations tier store.
- **`null` deletes.** A null value deletes the override at every scope; today it is rejected for everything except `preferences.*`. Deleting returns that scope to "not set".
- **New routes for teams and orgs.** `PUT` and `DELETE` on `/api/v1/admin/orgs/{org_id}/profile/overrides/{key}` and `/api/v1/admin/teams/{team_id}/profile/overrides/{key}`.
  - They write `OrgProfileOverridesRepo` / `TeamProfileOverridesRepo`, which have no writers today.
  - Each validates the value against the catalog, emits an audit event, and clears the resolver cache.
  - **Platform admin only.** `limits.*` catalog entries change to `editable_by: [platform_admin]`, so a customer's org admin can't lift their own members above the operator's value.
- **The bulk profile API is unchanged.** It writes per-user overrides for the users it selects, so it is not a group setting; the docs say so.
- **`limits.evaluations_per_minute` is deprecated.** It is enforced nowhere: writing it flips the user to the CUSTOM evaluations tier, which *disables* their per-minute and daily enforcement. Writing it no longer touches the tier store.

### 4. Keys and enforcement

Every key in `Config_Files/user_profile_catalog.yaml` has:
- `default: null`;
- `minimum: 0`;
- `editable_by: [platform_admin]`;
- type integer.

Months are calendar months in UTC.

| Key | Counter (scope, category, units) | Checked at |
|---|---|---|
| `limits.audio_daily_minutes` (exists) | ledger `user/minutes`, in **seconds** | `core/Usage/audio_quota.py` check/consume. This covers HTTP transcription, WebSocket streaming, the realtime handler and the audio workers. |
| `limits.transcription_minutes_per_month` | the same ledger rows, summed with `peek_range` over the month | `audio_quota.py`, next to the daily check |
| `limits.audio_concurrent_jobs` (exists) | the audio Jobs worker's count of `processing` rows per owner | `services/audio_jobs_worker.py` |
| `limits.llm_tokens_per_month` | `llm_usage_log` summed per `user_id` over the month. Add a `(user_id, ts)` index. Pass a naive UTC datetime, because `ts` is `TIMESTAMP` on Postgres. | the `/chat/completions` token check in `chat.py` |
| `limits.rag_queries_per_day` | ledger `user/rag_queries`. **New** unconditional per-user write in `RAG/rag_service/transport.py`, plus a new count in `text2sql.py`. | `rag_unified.py`, `text2sql.py`, `transport.py` (this also covers the MCP RAG module) |
| `limits.storage_quota_mb` (exists) | `users.storage_used_mb` | see §5 |
| `limits.media_ingest_mb_per_day` | ledger `user/ingestion_bytes`, in **bytes**. The write moves out of the RG-only block in `persistence.py` so it always runs. It counts uploaded files. | `Ingestion_Media_Processing/persistence.py` |
| `limits.workflows_runs_per_day` | ledger `workflows_runs`, keyed on `user:{id}` | `workflows.py::_enforce_workflows_daily_cap` **and** the scheduler (`core/Scheduler/handlers/workflows.py`), which records but never checks today |
| `limits.evaluations_per_day` (exists) | evaluations DB `daily_usage.total_evaluations`. Not the ledger, which double-counts. | `Evaluations/user_rate_limiter.py` |
| `limits.evaluation_tokens_per_day` | evaluations DB `daily_usage.total_tokens` | `user_rate_limiter.py` |
| `limits.chatbooks_exports_per_day`, `limits.chatbooks_imports_per_day` | `export_jobs` / `import_jobs` rows since 00:00 UTC | `chatbook_service.py::_check_chatbook_job_admission` |
| `limits.chatbooks_concurrent_jobs` | active chatbook job rows | the same admission check. It runs synchronously inside a DB transaction, so the async callers (`create_chatbook`, `import_chatbook`) resolve the limits first and pass them in. |

**How a site checks.** At each site, `limit = await user_quota(user_id, key)`. If it is None, skip the check. Otherwise deny when the counter plus this request's units exceeds the limit. Denials use the site's **existing** status code and message.

**What sites stop sending.** Sites stop sending quota categories to the RG governor: `jobs`, `streams`, `ingestion_bytes`, `workflows_runs`, and the evaluations daily `evaluations`/`tokens`. RG policies kept in the DB policy store, or in an operator's copied YAML, would otherwise keep enforcing them. The stock YAML blocks are deleted for hygiene; request-rate entries stay.

**What becomes unlimited.** With those categories gone, synchronous concurrency (media ingest requests, audio streams, synchronous transcription) has no per-user cap.

**Admin storage pools.** Admin-created team/org storage pools (`guard_storage_quota`) keep their shared-pool semantics but also run only when the switch is on. `STORAGE_QUOTA_ENFORCEMENT` stays as an extra off switch.

### 5. Storage cut-over

- **Enforcement** reads `limits.storage_quota_mb` through the resolver at every site:
  - `storage_quota_service.check_quota`, from media ingest (`persistence.py`, including the original-file store);
  - audio and video downloads (`Audio_Files._enforce_download_quota`, `Video_DL_Ingestion_Lib.py`);
  - file artifacts;
  - `check_combined_quota` for generated files.
  - The 300 s `quota_cache` keeps usage only; the quota comes from the resolver.
- **Every writer writes the user override** instead of the column:
  - admin user create and update, which is what the admin UI's "edit user" sends;
  - registration and registration codes, including the privileged override;
  - `storage_quota_service.set_user_quota`;
  - `PUT /api/v1/storage/admin/quotas/user/{id}`;
  - the profile update path.
  - The 100 MB minimum drops to 0. Registration writes nothing by default.
- **The broken endpoint is fixed.** `/api/v1/admin/storage-quotas/users/{user_id}` stops treating the user id as an org id and uses the same user-override path.
- **Every reader that shows a quota uses the resolver:**
  - `GET /users/storage`;
  - the profile `quotas` section;
  - admin user listings and system stats;
  - the data-ops export;
  - the storage breakdown views.
  - Response fields become `Optional[int]`, where null means unlimited. The frontend types (`TldwApiClient.ts`) become `number | null`; the UI consumers already tolerate null.
- **The column itself is unchanged.** `users.storage_quota_mb` keeps `NOT NULL DEFAULT 5120`, because the profile write guard pins that default. It is simply no longer read for enforcement.

### 6. Billing hook

- **No billing repo, no billing checks.** When no billing repo is wired into `SubscriptionService`, which is every OSS install, every billing check returns early:
  - `require_within_limit` and `LimitEnforcer`;
  - `get_billing_org_id` and `add_billing_headers`;
  - the chat and audio rechecks;
  - the RAG transport check.
- **Effects in OSS.** Org resolution and usage aggregation never run; orgless users never get 403; the per-user `limits.*` checks do the work. `DEFAULT_LIMITS` and the fallbacks are unchanged.
- **With a repo wired (hosted).** Enforcement runs as today under the master switch, with the same merge base and the same plans.
- **Nothing in this repository wires a billing repository**; a hosted deployment must add that wiring, after which the switch alone controls enforcement.

### 7. Removals

- **Audio tiers.**
  - `TIER_LIMITS` no longer feeds enforcement or reporting. `AUDIO_TIER_LIMITS_JSON` and `[Audio-Quota] {tier}_*` are deprecated, with a warning when set.
  - The tier admin API (`PUT/GET /api/v1/audio/jobs/admin/tiers/{user_id}`) stays, is documented as deprecated, and no longer affects limits.
  - The 25 MB transcription upload cap and its hard-coded fallbacks (`audio_transcriptions.py`, `audio_streaming.py`, `audio_jobs_worker.py`) are replaced by `[Media-Processing] max_audio_file_size_mb`, already available as `Audio_Files.MAX_FILE_SIZE`.
- **Evaluations tiers.** They stop driving daily caps: `RateLimitConfig.for_tier`, `Config_Files/evaluations_config.yaml`, the `user_rate_limits` defaults, and RG `evals.{tier}` selection for daily categories.
  - Existing CUSTOM-tier rows are ignored for daily caps.
  - **They still bypass RG's per-minute rate, as today.** Fixing that is outside this spec.
- **Chatbooks tiers.** `QuotaManager`'s tier tables and the dead `check_storage_quota` go. The 100 MB file-size cap is extracted into a constant first. `CHATBOOKS_DISABLE_QUOTAS` and `WORKFLOWS_DISABLE_QUOTAS` stay as extra off switches.
- **Per-user storage default.** `DEFAULT_STORAGE_QUOTA_MB` is deprecated and ignored.

### 8. Upgrading existing installs

- **Self-hosters have nothing to do.** Quotas they never configured stop applying.
- **A future hosted deploy turns quotas on with `USAGE_QUOTAS_ENABLED=true`.** `LIMIT_ENFORCEMENT_ENABLED=true` also works, with a warning. No hosted deployments exist today, and the commercial-mode warning in §1 catches a missed setting.
- **Storage migration.** An AuthNZ migration (a numbered SQLite migration in `migrations.py` and a Postgres entry in `pg_migrations_extra.py`) runs in this order:
  1. It ensures the override table exists.
  2. It copies every `users.storage_quota_mb` value into a user-scope `limits.storage_quota_mb` override, **except** values equal to 5120 or to the `DEFAULT_STORAGE_QUOTA_MB` configured at migration time. Those can't be told apart from "never set", so they are dropped.
- **Existing `limits.evaluations_per_day` user overrides** were never enforced. Once the switch is on, they become enforced daily caps. The release notes say so.
- **Tiers.** Audio, evaluations and chatbooks tier assignments are not carried over.

### 9. Reporting

- **Endpoints.** These report the resolver's value, and "unlimited" when it is None:
  - `GET /api/v1/audio/stream/limits`;
  - `GET /api/v1/audio/jobs/admin/owner/{id}/processing`;
  - the profile `quotas` section;
  - the chatbooks usage summary;
  - the evaluations limits view.
- **"Used" figures** come from the counters in §4. Audio's come from the ledger, not the dead `audio_usage_daily` table.

## Known defects outside this spec

These were found during review. They are pre-existing bugs on the hosted billing path and are filed as separate backlog tasks:
- **TASK-13432: transcription minutes are never counted at org level.** `_get_transcription_minutes_month` reads ledger `org/minutes`, which nothing writes (transcription writes `org/cost_units` and `user/minutes`). The monthly limit only sees the 60 s in-memory delta.
- **TASK-13433: LLM token usage probably reads 0 on Postgres.** `_get_llm_tokens_month` passes a tz-aware datetime to a `TIMESTAMP` column. asyncpg raises `DataError`, which is swallowed as noncritical.

## Testing

1. **The switch:**
   - off by default;
   - precedence: `USAGE_QUOTAS_ENABLED`, then `LIMIT_ENFORCEMENT_ENABLED`, then config.txt;
   - a warning on the legacy spelling;
   - the commercial-mode warning when a billing repo is wired and the switch is off.
2. **The resolver:**
   - user beats team, and team beats org;
   - the most generous of the teams that set the key wins;
   - a team without the key does not unlimit anyone;
   - 0 blocks;
   - None when unset or when the switch is off;
   - the cache clears on writes;
   - a lookup error fails open.
3. **Override routes:**
   - org and team PUT/DELETE;
   - catalog validation;
   - platform-admin only, and an org admin is refused;
   - null deletes at all three scopes;
   - an audit event per write.
4. **A stock install admits past every old default,** with the switch off and with it on but nothing assigned. In single-user and multi-user, one test per site:
   - 31 audio minutes, a 26 MB transcription upload, and 2 queued audio jobs;
   - 51 RAG queries, and 300,001 LLM tokens in a month;
   - a 5 GB + 1 MB upload;
   - 11 chatbook exports and 3 concurrent chatbook jobs;
   - 3 concurrent media ingests, and 2 GiB + 1 of ingest in a day;
   - 1001 workflow runs, both API-started and scheduled;
   - 101 evaluations, spaced past the per-minute rate;
   - an orgless multi-user account gets no 403.
5. **Assigned values deny one unit over,** each set at user, team and org level, with the site's existing status code.
6. **Counters run with the switch off.** Ledger rows for audio, RAG, media bytes and workflows are still written.
7. **The hosted path still works:** with a fake `billing_repo` wired, its plan limits are enforced exactly as before.
8. **The storage migration** keeps non-default values and drops 5120 and the configured default, on SQLite and on Postgres (via the AuthNZ Postgres fixture).
9. **Storage writers and readers.** Admin user edit, registration codes and both admin quota endpoints write the override. Readers return null for unlimited.
10. **Reporting endpoints** say "unlimited" by default and show used minutes from the ledger.
11. **Tests that pinned the old defaults** are rewritten to this contract. About 60–80 files truly pin them; 144 reference the affected symbols. Before each default flips, grep every affected symbol and run every hit.

## Delivery

Four PRs against `dev`:

- **PR A (relief).** Adds `usage_quotas_enabled()` and the commercial-mode warning, and gates every quota *check* (never a counter write) on the switch. It also adds the no-billing-repo short-circuit.
  - Self-hosters are relieved at once.
  - With the switch on and a billing repository wired, billing behavior is unchanged; evaluations daily caps return in PR B.
- **PR B (per-user values).**
  - The resolver and the precedence rule.
  - The generic `limits.*` write path with null-as-delete, and the team/org override routes.
  - Every non-storage site reading the resolver.
  - The counter changes: the RAG per-user write, text2sql counting, the media-bytes hoist, and the scheduler workflow check.
  - The removals in §7, and sites no longer sending quota categories to RG.
- **PR C (storage).** The cut-over in §5 and the migration in §8.
- **PR D (reporting and docs).** The reporting endpoints in §9 and ADR-058.
  - **Env and config:** `Env_Vars.md` and config.txt `[Usage-Quotas]`.
  - **Rewritten:** the billing section of `Organization_Administration.md`, which documents a `BILLING_ENABLED` flag and `/billing/*` routes that don't exist.
  - **Updated:** the chatbooks code guide, and the storage API doc's endpoint citations.
  - **New:** a quotas page in Operations.
  - The `Docs/Published` mirror is refreshed for each mirrored doc.
