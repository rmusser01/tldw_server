# Usage

The Usage module records and normalizes usage events for quotas, billing, audio
minutes, LLM token accounting, pricing estimates, and usage middleware. It is
the lightweight accounting layer that feeds Billing, Resource Governance,
observability, and user-facing usage summaries.

## Start Here

- Usage tracking: `usage_tracker.py`.
- Audio quota helpers: `audio_quota.py`.
- LLM usage normalization: `llm_usage_normalizer.py`.
- Pricing catalog: `pricing_catalog.py`.
- Related docs: `Docs/Published/User_Guides/Server/Usage_Module.md`.
- Tests: `tests/Usage/`.

## Responsibilities

- Normalize usage payloads from provider-specific response shapes.
- Track per-user and per-resource usage in middleware and quota helpers.
- Provide pricing lookup and override behavior for cost estimates.
- Enforce audio daily-minute quotas through the shared ResourceDailyLedger,
  with Resource Governance heartbeat integration for concurrent audio work.

## Module Map

- `usage_tracker.py` normalizes and persists LLM usage into the AuthNZ usage
  tables, emits low-cardinality operational metrics, and exposes aggregation
  helpers used by middleware and reports.
- `audio_quota.py` implements audio limits. Daily minutes use
  ResourceDailyLedger as the canonical store; stream/job concurrency is routed
  through Resource Governance when enabled.
- `llm_usage_normalizer.py` converts provider response usage into a common
  shape.
- `pricing_catalog.py` loads model/provider pricing defaults and overrides.

## How It Connects

- Billing and virtual-key enforcement read durable usage logs when deciding
  token and USD budget limits.
- Audio flows call `consume_daily_minutes` to enforce and record daily-minute
  usage in one store-backed operation.
- Metrics receive provider/model/operation counters; user-level breakdowns stay
  in durable logs to avoid high-cardinality Prometheus labels.

## Extension Points

- Add resource-specific counters in `usage_tracker.py` only after defining the
  consumer and aggregation semantics.
- Add provider pricing in `pricing_catalog.py` with tests for override and path
  loading behavior. Use non-placeholder zero rates only for documented free
  models; placeholder rates fall back to conservative billable estimates.
- Keep new audio quota behavior separated by concern. `audio_quota.py` still
  owns daily-minute, Resource Governance concurrency, and legacy compatibility
  paths; larger additions should first extract minute ledger or concurrency
  helpers behind the existing public functions.

## Testing

- Tracker and middleware behavior: `tests/Usage/test_usage_tracker_sqlite.py`,
  `tests/Usage/test_usage_middleware.py`, and `tests/Usage/test_usage_aggregator.py`.
- Audio quota behavior: `tests/Usage/test_audio_quota_ttl_cache.py`,
  `tests/Usage/test_audio_rg_minutes_and_heartbeat.py`,
  `tests/Audio/test_audio_quota_rg_and_ledger.py`, and
  `tests/Usage/test_usage_review_fixes.py`.
- Pricing and normalization: `tests/Usage/test_pricing_catalog.py`,
  `tests/Usage/test_pricing_catalog_overrides.py`, and
  `tests/Usage/test_llm_usage_normalizer.py`.

## Gotchas

- Avoid high-cardinality metric labels. Provider/model/operation dimensions are
  acceptable for operational counters; per-user usage belongs in AuthNZ usage
  logs and query APIs.
- Provider usage fields vary widely; normalize defensively and preserve unknowns
  only when a downstream consumer needs them.
- Do not fall back to legacy `audio_usage_daily` for new daily-minute
  enforcement. It is only backfilled into ResourceDailyLedger for upgrade
  compatibility.
