# Governance and billing (deep dive)

This document goes deep on the two policy cores [authnz.md](./authnz.md) summarizes: the Governance rule plane (candidate resolution with stricter-wins precedence, rollout modes, gap tracking, the MCP/ACP consumption seams) and Billing enforcement (limit categories, plan tiers, overage policy, the three-system quota posture, the Stripe compatibility surface).

## Governance core

### Authoritative files

All paths under `tldw_Server_API/` unless noted.

| Concern | File | Key symbols |
| --- | --- | --- |
| Types/resolver | `app/core/Governance/{types,resolver}.py` | `GovernanceAction` (allow/warn/require_approval/deny); `resolve_effective_action` — precedence: action severity first, then scope level (workspace > persona > team > org > global), then priority, then recency; the full ordering is kept as provenance |
| Service | `service.py` | `validate_change` — category resolution (explicit > metadata > keyword patterns > general), candidate loading, fallback modes (default `warn_only`) |
| Store | `store.py` | aiosqlite `governance_rules`/`governance_gaps`; active rules in the effective window with NULL-or-matching scope columns; invalid stored actions **coerced to deny**; sha256-fingerprinted gap dedupe under a unique open-gap index |
| Metrics/rollout | `metrics.py` | `GovernanceRolloutMode` (off/shadow/enforce); audit traces; checks recorded through the MCP Unified metrics collector |
| MCP surface | `MCP_unified/modules/implementations/governance_module.py` + `tool_execution/security.py` | the three tools (query_knowledge / validate_change / resolve_gap) with caller-scope verification; the tool-execution **preflight** with `governance_bypass` metadata and `governance.*` self-exemption |
| ACP consumer | `Agent_Client_Protocol/runner_client.py` | `ACPGovernanceCoordinator` — same store/service construction, fail-open (see the posture note) |

### Rule lookup (dataflow)

1. Category resolves (with the source reported).
2. Candidates load: active rules in window, category-matching or `general`, scope NULL-or-matching. Loader contract violations ⇒ **deny**; store unavailable ⇒ the fallback action with `backend_unavailable`.
3. **An empty candidate set is `allow`** — "no rules" is permissive by design.
4. Non-empty ⇒ stricter-wins resolution; matched rule ids returned as provenance.
5. **Rollout is applied by callers, not the service**: MCP preflight and the ACP coordinator block only on deny-under-enforce; `off` short-circuits to allow with an `unknown` metric; shadow records without blocking; ACP treats a malformed rollout field as enforce (fail-safe).
6. `resolve_gap` upserts a deduplicated open gap (fingerprinted question + category + coalesced scope); races return the existing row.

### Boundaries

- Governance has **no REST endpoint and no first-party admin writer** — rules enter the SQLite DB out-of-band; the only API surfaces are the three MCP tools and the internal checks.
- The `Moderation` core (with its governance import/export) is a separate, richer system — do not conflate.
- Traces deliberately avoid embedding request bodies; ACP prompt summaries sent to governance are truncated to 1200 chars.

### Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| Empty candidates to the resolver | `ValueError` — the service shields callers |
| Unknown action string / invalid stored action | Contract error → deny; DB row coerced to deny |
| Store unavailable | Fallback action (`warn` default) |
| MCP preflight internal error | **Fail open** (logged + metric) |
| Deny under enforce (MCP) | `GovernanceDeniedError`, detail surfaced additively in `error.data.governance` |

### Config

`GOVERNANCE_ROLLOUT_MODE` env > `[Governance] rollout_mode` > default **off** — the shipped config has no `[Governance]` section, so stock installs run off. DB path: the MCP config attr else `Databases/governance.db`.

### Gotchas

1. Stricter-wins sorts by severity **first** — a global deny beats a workspace allow; specificity only breaks ties within the same action.
2. `general` rules match every category.
3. Shadow-mode traces are not blocks; changing the gap-scope shape reopens old gaps.
4. Governance checks emit metrics through the MCP collector even on the ACP path.

## Billing core

### Authoritative files

| Concern | File | Key symbols |
| --- | --- | --- |
| Enforcement | `app/core/Billing/enforcement.py` | `LimitCategory` (api_calls_day, llm_tokens_month, storage_mb, team_members, transcription_minutes_month, rag_queries_day, concurrent_jobs); `EnforcementAction` (allow/warn/soft_block/hard_block); the `BillingEnforcer` singleton with per-org 60 s caches |
| Plan tiers | `plan_limits.py` | `PlanTier` FREE-only; `get_plan_limits` ignores the plan name; soft limit at 80 %; grace period 3 days |
| Overage | `overage_config.py` | `hard_block|degraded|notify_only` (default notify-only) + notify/grace percentages |
| OSS posture | `runtime_flags.py`, `subscription_service.py` | `is_billing_enabled()` hardwired false; Stripe paths raise unless clients are injected; the public checkout/portal API is gone (pinned by test) |
| Deps | `app/api/v1/API_Deps/billing_deps.py` | `require_within_limit`, `LimitEnforcer` (post-execution accounting + the RG cost-units mirror), billing headers |
| Per-user quotas | `app/core/Usage/quota_resolver.py`, `UserProfiles/limits_precedence.py` | ADR-064 system: user > most-generous team > most-generous org; `0` blocks; unset = unlimited; off by default |

### Enforced request (dataflow — hosted mode only)

1. The gate: enforcement is active only when quotas are enabled **and** a billing repository is wired — OSS wires none, so checks return unlimited immediately.
2. Org resolution (query > header > principal's active org > first membership) with claim-scope verification.
3. `check_limit`: limits and usage from per-org caches; usage sources per category (API calls, LLM tokens with **single-pass primary-org attribution** so rows never double-count, team members + storage sums, embeddings-domain concurrent jobs, ledger minutes/queries).
4. Decision: invalid/`-1` limits are unlimited; over-limit → SOFT_BLOCK (→ **402**); ≥80 % → WARN; overage policy may escalate to HARD_BLOCK (→ **429**) — both carry `Retry-After` and the upgrade URL; responses carry usage headers.
5. Post-execution accounting updates the in-process cache and mirrors usage into the Resource Governor's cost-units ledger; accounting failures never fail the request.

### The three-system posture

Billing plan limits (per-org, hosted-only), per-user `limits.*` quotas (ADR-064, off by default, platform-admin-only writes, fail-open), and the Resource Governor (ADR-056, rate/concurrency governance) are **distinct systems**; billing touches RG only via the cost-units mirror. Deprecated knobs that no longer enforce: `AUDIO_TIER_LIMITS_JSON`, the `[Audio-Quota]` tier family, `DEFAULT_STORAGE_QUOTA_MB`, and the unread `users.storage_quota_mb` column.

### Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| Data-source error, failure mode `open` (default) | Permissive fallback limits |
| Same, `closed` | Zero limits + max usage ⇒ blocks |
| Plan limit exceeded / overage hard-block | 402 / 429 |
| Exit-accounting failure | Swallowed (debug) |
| Stripe path in OSS | Typed runtime error |
| Org resolution failure / no org in multi-user | 503 / 403 |

### Config (exact)

`USAGE_QUOTAS_ENABLED` (master; legacy `LIMIT_ENFORCEMENT_ENABLED` is its deprecated alias), `BILLING_ENFORCEMENT_FAILURE_MODE` (open/closed), `BILLING_CACHE_TTL_SECONDS` (60), `BILLING_OVERAGE_MODE/_GRACE_PCT/_NOTIFY_PCT`, `[Usage-Quotas] enabled`, per-user `limits.*` via admin endpoints.

### Gotchas

1. OSS never enforces plan limits even with rules present — the hosted repo is part of the gate.
2. The Billing README's switch name is stale (`LIMIT_ENFORCEMENT_ENABLED` is now the alias).
3. Storage usage reads `users.storage_used_mb` — the quota column is unread.
4. `concurrent_jobs` counts only embeddings-domain Jobs.
5. Usage deltas mutate the in-process cache — multi-worker drift up to the TTL.

## Governing decisions and docs

ADR-064 (`064-usage-quotas-per-user-limits.md`), ADR-056 + ADR-018 (RG boundary), ADR-052 (adjacent admin predicate). Current: `Docs/Operations/Usage_Quotas.md` (Oct), the Governance module README, `Docs/MCP/Unified/Governance_Operations.md` (current-on-content); the Billing README has the one stale switch name. No `Code_Documentation` guides exist for either core — these docs are the reference.

## Related docs

- [authnz.md](./authnz.md) — the one-paragraph summaries these deep dives expand
- [mcp-unified.md](./mcp-unified.md), [agent-orchestration-acp.md](./agent-orchestration-acp.md) — the consumption seams
