# Billing

The Billing module owns OSS/self-host plan limits, overage settings, billing
audit records, and runtime limit enforcement. Historical Stripe checkout,
portal, and webhook compatibility remains in the service layer for non-public
or injected-client deployments, but the public OSS runtime keeps payment
billing disabled.

## Start Here

- Limit enforcement: `enforcement.py`, `plan_limits.py`, and
  `overage_config.py`.
- OSS billing runtime flag: `runtime_flags.py` returns disabled for payment
  billing.
- Historical subscription and injected-client Stripe compatibility:
  `subscription_service.py`.
- Audit helpers: `billing_audit.py`.
- API dependency helpers: `app/api/v1/API_Deps/billing_deps.py`.
- API endpoint and schemas: `app/api/v1/endpoints/billing.py` and
  `app/api/v1/schemas/billing_schemas.py`.
- Tests: `tests/Billing/`.

## Responsibilities

- Decide whether OSS limit enforcement is enabled and whether payment billing
  is disabled at runtime.
- Resolve free/self-host plan limits and overage behavior for
  organizations/users.
- Enforce storage, media, audio, and provider/resource limits through shared
  request contexts.
- Preserve non-public/injected-client checkout, portal, and webhook
  compatibility paths without exposing an active public payment runtime.
- Record billing audit events without leaking secrets or raw provider payloads.

## Module Map

- `enforcement.py` defines limit categories, contexts, and enforcement behavior.
- `plan_limits.py` maps plans to concrete limits and fallback tiers.
- `overage_config.py` parses overage settings and failure modes.
- `subscription_service.py` coordinates subscription, checkout, portal, cancel,
  resume, and webhook compatibility, but `_get_stripe_client()` raises in OSS
  unless a client is injected and `runtime_flags.is_billing_enabled()` still
  reports payment billing disabled.
- `billing_audit.py` records security-relevant billing events.
- `runtime_flags.py` centralizes the OSS payment-billing flag; `enforcement.py`
  reads `LIMIT_ENFORCEMENT_ENABLED` for quota enforcement.

## How It Connects

- Media ingestion, file storage, and other endpoints use Billing enforcement
  helpers before creating expensive or quota-bound work.
- AuthNZ repositories provide organization, user, subscription, and role data.
- Usage and Resource Governance provide usage counters and cost-unit context.
- Payment-provider compatibility is isolated from normal OSS limit enforcement;
  public builds should operate on the free/self-host tier without active
  checkout or portal flows.

## Architecture Notes

### Core Flow

- Endpoint dependencies build a `LimitEnforcementContext` with user/org/team
  identity, requested resource category, and usage quantity.
- `enforcement.py` resolves the applicable plan through AuthNZ billing/quota
  repositories, loads category limits from `plan_limits.py`, applies overage
  behavior from `overage_config.py`, and returns allow/deny decisions before the
  caller creates storage, media, audio, or provider work.
- `subscription_service.py` remains a compatibility boundary for injected
  payment clients; public OSS payment capability is still controlled by
  `runtime_flags.py` and must not be inferred from compatibility methods.

### State And Operations

- Limit state comes from AuthNZ billing/quota repositories and Usage/Resource
  Governance counters, not from this package alone.
- `LIMIT_ENFORCEMENT_ENABLED` controls quota enforcement; payment billing
  remains disabled through the OSS runtime flag.
- Billing audit events should capture decision context and provider event ids
  without logging secrets, signed webhook bodies, or raw provider payloads.

### Extension Checklist

- New resource limit: update `LimitCategory`, `plan_limits.py`,
  `billing_schemas.py`, endpoint/dependency wiring, and
  `tests/Billing/test_billing_enforcement.py`.
- New overage mode: update `overage_config.py`,
  `tests/Billing/test_overage_config.py`, and integration coverage for allow,
  deny, and failure-mode behavior.
- New provider compatibility event: update `subscription_service.py`,
  `billing_audit.py`, webhook sanitization tests, and idempotency checks.

## Extension Points

- Add a new enforced resource by extending `LimitCategory`, plan limits, endpoint
  dependency wiring, and tests together.
- Add webhook events in `subscription_service.py` only for injected-client or
  non-public deployments, with idempotency and audit coverage.
- Keep redirect URL validation strict; checkout/portal compatibility paths must
  honor the configured host allowlist and HTTPS policy.

## Testing

- Enforcement behavior: `tests/Billing/test_billing_enforcement.py`,
  `tests/Billing/test_limit_enforcer_context.py`, and
  `tests/Billing/test_overage_enforcement_integration.py`.
- Subscription and compatibility webhook behavior:
  `tests/Billing/test_subscription_service.py`,
  `tests/Billing/test_subscription_webhook_updates.py`, and
  `tests/Billing/test_subscription_service_updates.py`.
- Endpoint/schema/dependency coverage: `tests/Billing/test_billing_schemas.py`,
  `tests/Billing/test_billing_deps_helpers.py`, and
  `tests/Billing/test_billing_endpoint_sanitization.py`.

## Gotchas

- `BILLING_ENFORCEMENT_FAILURE_MODE=closed` changes allow-on-error behavior into
  deny-on-error behavior; tests should cover both modes for new resources.
- Historical Stripe cancel/resume compatibility paths fail closed when remote
  state cannot be reconciled.
- Never log injected payment-provider secrets or raw signed webhook bodies.
