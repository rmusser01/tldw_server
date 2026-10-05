# M5.1 UX Smoke Warning Hard-Gate Allowlist Policy

- Status: Active
- Owner: WebUI + QA + Platform
- Date: February 13, 2026

Related Files:
- `apps/tldw-frontend/e2e/smoke/smoke.setup.ts`
- `apps/tldw-frontend/e2e/smoke/all-pages.spec.ts`
- `.github/workflows/frontend-ux-gates.yml`

## 1) Purpose

Define a strict, auditable policy for console/request warning noise in the WebUI smoke suite so CI can hard-fail on unexpected regressions while tolerating known, time-boxed exceptions.

## 2) Hard-Gate Contract

The smoke suite now classifies warning/error diagnostics into:

1. `allowlisted` (known, route-scoped, time-boxed)
2. `unexpected` (hard-fail)

Hard gate is enabled via:

- `TLDW_SMOKE_HARD_GATE=1`

Enforcement behavior in `all-pages.spec.ts`:

- Any `unexpectedConsoleErrors` => test failure
- Any `unexpectedRequestFailures` => test failure
- `allowlisted*` diagnostics are printed with allowlist rule IDs for visibility and trend review

## 3) Rule Requirements

Every allowlist rule must include:

- Stable `id`
- `scope` (`console` or `request`)
- Narrow `pattern`
- Route scope (`routes`) whenever feasible
- `rationale`
- `owner`
- `expiresOn` date

Rule hygiene constraints:

1. Prefer route-scoped rules over global rules.
2. Prefer specific signatures over generic catch-all patterns.
3. Expiry is mandatory; expired rules must be removed or renewed with fresh evidence.
4. Forced-error fixture noise must be isolated to fixture-target routes only.

## 4) Triaged Warning Classes (Baseline)

Baseline run used for triage:

- Command: `TLDW_SMOKE_HARD_GATE=1 bun run e2e:smoke`
- Result: `165 passed`
- Date: February 13, 2026

The February baseline above is historical. TASK13406 retired the final four general exceptions on October 1, 2026:

- Kanban's archive drawer uses the supported `size` prop and retains its 400px width.
- The minimal smoke profile supplies an explicit empty moderation review list; live-tier UAT continues to use the backend.
- Deliberate route-boundary tests require the exact route-specific forced-error log and handle its React component report only inside that test. The same emissions on ordinary routes fail the hard gate.

The general allowlist is empty. Ownership, expiry and unexpected-error guards remain active for any future exception. Synthetic metadata tests use their own rules, independently of whether production exceptions exist.

## 5) CI Policy

CI gate coverage:

1. Onboarding gate (`e2e:onboarding`) with evidence artifact upload
2. Broad UX smoke gate (`e2e:smoke`) with hard gate enabled

Both gates are required quality signals for M5 UX governance in PR validation.

## 6) Governance and Review Cadence

1. Weekly review of allowlisted warning counts from smoke logs.
2. Remove stale rules as route/components are remediated.
3. Escalate recurring high-volume warning classes to product backlog if they remain beyond expiry.
4. No new allowlist rule without linked evidence (test output and route context).

## 7) Add/Change Checklist for Future Rules

- [ ] Evidence captured from failing smoke run
- [ ] Rule added with route scope and expiry
- [ ] Rationale includes why warning is non-blocking today
- [ ] Owner assigned
- [ ] Follow-up remediation issue linked in PR or roadmap

Current-dev integration (2026-10-02, TASK13406): the deliberate Wayfinding document 404 added by TASK13414 is handled only in its own recovery fixture. That test requires an actual navigation response of 404 and removes only the exact response URL and browser 404 console text from its diagnostics. Other console errors, requests and page errors remain intact. The general classifier rejects that document error, including on the missing-route path; the general allowlist remains empty. Upstream source repairs for both Writing drawers are retained.
