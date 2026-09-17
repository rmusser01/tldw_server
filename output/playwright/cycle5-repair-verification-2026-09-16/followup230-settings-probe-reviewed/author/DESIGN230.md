# TASK13260.171 / UAT230 — bounded Settings OpenAPI guard

Parent-approved design: share the existing quickstart same-origin URL predicate with the mounted Settings billing probe. Export the existing pure predicate from TldwApiClient and add it to the effect precondition; retain the explicit Settings server URL, abort controller, five-second timeout, cancelled-result guard and four advertised GET routes. No singleton discovery request, proxy/config change or new abstraction.

Three inspected patterns: shared getOpenAPISpec already skips this exact URL; Settings existing capability effect owns timeout/cancellation and fails closed; mounted Settings cookie/auth and actual-form fixtures preserve real storage/account transitions. The shared client connection-sync test already proves its quickstart skip and direct-backend discovery.

## Stage 1 — causal mounted regression
Goal: actual authenticated Settings on the quickstart page origin must not issue unsupported OpenAPI or billing requests. Preserve direct backend discovery, genuine404 absence and lifecycle controls. Status: Complete — causal2FAIL/8controls retained.

## Stage 2 — minimal production guard
Goal: export the existing predicate and use it before creating the request. Keep full-module test doubles faithful by passing through this real function; do not replace it with a mock true/false result. Status: Complete — frozen minimal guard;87tests and static comparison pass; independent review/native acceptance pending.

## Stage 3 — verification/review handoff
Goal: focused and adjacent tests, scoped ESLint, compiler attribution, Bandit parser limits, source/patch/evidence hashes and independent review. Native acceptance remains parent-owned. Status: In Progress —87PASS/6files0skip; static checks complete; independent review/native remain pending.
