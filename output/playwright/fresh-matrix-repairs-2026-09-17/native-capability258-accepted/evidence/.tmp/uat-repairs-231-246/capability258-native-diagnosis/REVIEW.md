# UAT258 native acceptance; separate UAT264 Sources failure

**UAT258: CLEAR for the bounded native probe gate. Sources listing remains broken by a separate, pre-existing redirect/authentication-loss path.**

The audit verifies 18 checks and 53 hashed inputs. Runtime source is `edfd06ec40a173f2e38ec65af715abb29f3aa002`; both modes’ source manifests, original profile/initialization/holder hashes and owned API/Next process bindings match. Process receipts cover the relevant captures. The independent UAT258 source/test review remains clear.

## Bounded UAT258 acceptance

- **Fresh PG-multi:** `2026-09-18T00:59:57.411Z` through `01:00:01.653Z`.
- **Fresh PG-single:** `01:01:04.763Z` through `01:01:06.706Z`.

Each uses a new normal browser context with no injected state and its own observer of **all `/api/` requests**. Each displays the credential gate, records zero ingestion-capability requests and zero error/pageerror messages during the captured interval.

**Authenticated PG-multi Alice:** identity 200 observations identify user 2 immediately before the Sources visit. The bound backend access log records `GET /api/v1/ingestion-sources/capabilities` returning 200 at local`2026-09-17 18:05:37.553` = `2026-09-18T01:05:37.553Z`. This proves that the repaired guard allows a real authenticated protected probe. It is not a frontend-only mock or inferred success from a lack of errors.

This acceptance covers those two anonymous cases and the authenticated Alice positive case. It does not certify Sources listing/creation, PG-single authenticated capability behavior, every auth mode, or the full 48 matrix.

## Correction to the initial zero-dispatch inference

The standard `observe-native.js` matcher includes `ingestion` only when followed by slash, query or end. It **excludes `ingestion-sources`**. The authenticated and original Sources helpers return this incomplete `page.__matrixEvents` list, so their zero recorded source requests never established zero dispatch.

Those original captures remain unchanged and their limitation is explicit. The fresh-context captures use a different comprehensive observer and remain valid. The initial singleton/preflight hypothesis was withdrawn: a plain missing-config preflight does not produce the 401 classification shown by this UI, and the corrected capture shows an authenticated initiating request.

## Confirmed separate failure

Corrected passive capture begins `2026-09-18T01:12:20.108Z` and records only boolean header presence, never token values:

| Step | Request/response | Evidence |
| --- | --- | --- |
|1|GET `http://127.0.0.1:18783/api/v1/ingestion-sources`|Authorization present|
|2|Same-origin response|307|
|3|GET `http://127.0.0.1:18703/api/v1/ingestion-sources/`|Authorization absent; API key/cookie absent|
|4|Backend response|401|

The same sequence appears four times across request/query retries. The earlier owned backend log independently records the same slashless 307/trailing-slash 401 sequence during the original Sources visit and Retry. Authentication/identity calls still return 200; waiting and retrying do not correct the route.

### Source boundary

- `useIngestionSourcesQuery` calls the real singleton returned by `useTldwApiClient`; `collections.ts:918` issues the list request to the **slashless** path through `this.request`.
- `TldwApiClient.ts:1914` runs its normal readiness check before `bgRequest`. The corrected native request has Authorization, so this is not a missing-client-credential rejection.
- `ingestion_sources.py:420` registers the list at `@router.get("/")`. The slashless URL therefore redirects before reaching that handler.
- `next.config.mjs:147` proxies generic API paths to the internal API origin, preserving their path shape. It special-cases the Media collection’s slash, but not Sources. The observed redirect changes origin from 18783 to 18703; the redirected request has lost Authorization.
- `SourcesWorkspacePage.tsx:34` maps the list error through `buildCapabilityState`; `capability-state.ts:119` classifies 401 as auth-required. That produces “Sign in before using sources.” This is the list-query recovery panel, not a missing-setup gate.

The client domain file, backend route, Next rewrite and error classifier are byte-identical between the earlier278 runtime and edfd. The defect’s source predates UAT258. This comparison is not a claim that an earlier native Sources run was performed. UAT258 changes probe eligibility; it does not cause this collection redirect.

### Focused repair/test boundary for the next author

Align the Sources collection request with its canonical backend route so browser requests do not need this cross-origin redirect. Keep existing authorization checks. Do not forward credentials to arbitrary redirect origins or weaken missing-auth behavior to hide the problem.

Existing `tldw-api-client.ingestion-sources.test.ts` tests transport calls using mocks and currently expect the slashless collection path. Add a causal same-origin-proxy/real-redirect control, or equivalent focused integration boundary, so an authenticated list proves no cross-origin hop and returns owned data/empty state. Preserve unauthenticated 401, permission checks and source ownership. The collection create method also uses the slashless base path and deserves a scoped control; its failure was not exercised natively here. Capability/detail routes must retain their own canonical paths. Reuse the existing hook/page tests for loading/error/retry behavior rather than adding a new auth policy.

## Evidence limits

Raw private backend logs are hash-only inputs; the audit projects only endpoint, method, timestamp, line and status. Logs and process-receipt hashes represent the observed point in time. Browser source captures remain local, and no raw provider reasoning or credential value is copied into this report/audit.

No reviewer browser action, model call, DB action, runtime change, source/test edit, Git operation or tracker mutation occurred. Native observations were produced by root. This report accepts the bounded UAT258 probe requirement and diagnoses UAT264; it does **not** label Sources working.
