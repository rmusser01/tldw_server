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
- Support audio-minute quota TTL caches and Resource Governance heartbeat
  integration.

## Module Map

- `usage_tracker.py` provides the SQLite-backed usage tracker and aggregation
  helpers used by middleware and reports.
- `audio_quota.py` implements audio-minute quota cache checks and updates.
- `llm_usage_normalizer.py` converts provider response usage into a common
  shape.
- `pricing_catalog.py` loads model/provider pricing defaults and overrides.

## How It Connects

- Billing enforcement reads usage context when deciding plan limits.
- User profile and audio flows use audio quota helpers for per-user limits.
- Logging and Metrics consume usage events for operational visibility.

## Extension Points

- Add resource-specific counters in `usage_tracker.py` only after defining the
  consumer and aggregation semantics.
- Add provider pricing in `pricing_catalog.py` with tests for override and path
  loading behavior.

## Testing

- Tracker and middleware behavior: `tests/Usage/test_usage_tracker_sqlite.py`,
  `tests/Usage/test_usage_middleware.py`, and `tests/Usage/test_usage_aggregator.py`.
- Audio quota behavior: `tests/Usage/test_audio_quota_ttl_cache.py` and
  `tests/Usage/test_audio_rg_minutes_and_heartbeat.py`.
- Pricing and normalization: `tests/Usage/test_pricing_catalog.py`,
  `tests/Usage/test_pricing_catalog_overrides.py`, and
  `tests/Usage/test_llm_usage_normalizer.py`.

## Gotchas

- Avoid high-cardinality metric labels for user/provider/model combinations.
- Provider usage fields vary widely; normalize defensively and preserve unknowns
  only when a downstream consumer needs them.
