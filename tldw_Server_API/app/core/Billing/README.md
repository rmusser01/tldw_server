# Billing

The Billing module owns subscription state, plan limits, overage settings,
Stripe-backed checkout/portal/webhook flows, billing audit records, and runtime
limit enforcement. It is an operational boundary between AuthNZ organization
state, Usage accounting, Resource Governance, Storage quotas, and Stripe.

## Start Here

- Limit enforcement: `enforcement.py`, `plan_limits.py`, `overage_config.py`,
  and `runtime_flags.py`.
- Stripe/subscription lifecycle: `subscription_service.py`.
- Audit helpers: `billing_audit.py`.
- API dependency helpers: `app/api/v1/API_Deps/billing_deps.py`.
- API endpoint and schemas: `app/api/v1/endpoints/billing.py` and
  `app/api/v1/schemas/billing_schemas.py`.
- Tests: `tests/Billing/`.

## Responsibilities

- Decide whether billing and limit enforcement are enabled at runtime.
- Resolve plan limits and overage behavior for organizations/users.
- Enforce storage, media, audio, and provider/resource limits through shared
  request contexts.
- Create Stripe checkout/portal sessions and reconcile Stripe webhooks.
- Record billing audit events without leaking secrets or Stripe payload details.

## Module Map

- `enforcement.py` defines limit categories, contexts, and enforcement behavior.
- `plan_limits.py` maps plans to concrete limits and fallback tiers.
- `overage_config.py` parses overage settings and failure modes.
- `subscription_service.py` coordinates subscription, checkout, portal, cancel,
  resume, and webhook updates.
- `billing_audit.py` records security-relevant billing events.
- `runtime_flags.py` centralizes feature flags such as `BILLING_ENABLED` and
  `LIMIT_ENFORCEMENT_ENABLED`.

## How It Connects

- Media ingestion, file storage, and other endpoints use Billing enforcement
  helpers before creating expensive or quota-bound work.
- AuthNZ repositories provide organization, user, subscription, and role data.
- Usage and Resource Governance provide usage counters and cost-unit context.
- Stripe integration is optional and guarded by runtime flags and secrets.

## Extension Points

- Add a new enforced resource by extending `LimitCategory`, plan limits, endpoint
  dependency wiring, and tests together.
- Add webhook events in `subscription_service.py` only with idempotency and audit
  coverage.
- Keep redirect URL validation strict; checkout/portal flows must honor the
  configured host allowlist and HTTPS policy.

## Testing

- Enforcement behavior: `tests/Billing/test_billing_enforcement.py`,
  `tests/Billing/test_limit_enforcer_context.py`, and
  `tests/Billing/test_overage_enforcement_integration.py`.
- Subscription and webhook flows: `tests/Billing/test_subscription_service.py`,
  `tests/Billing/test_subscription_webhook_updates.py`, and
  `tests/Billing/test_subscription_service_updates.py`.
- Endpoint/schema/dependency coverage: `tests/Billing/test_billing_schemas.py`,
  `tests/Billing/test_billing_deps_helpers.py`, and
  `tests/Billing/test_billing_endpoint_sanitization.py`.

## Gotchas

- `BILLING_ENFORCEMENT_FAILURE_MODE=closed` changes allow-on-error behavior into
  deny-on-error behavior; tests should cover both modes for new resources.
- Stripe cancel/resume paths fail closed when remote state cannot be reconciled.
- Never log `STRIPE_API_KEY`, `STRIPE_WEBHOOK_SECRET`, or raw signed webhook
  bodies.
