# Authenticated Email Upload Validation — 2026-09-25

Tracking: TASK-13256. Base commit: `54ecc7e773`.

Synthetic EML uploads now run through the production `/api/v1/media/add` router
with real AuthNZ API keys, RBAC, expected-user checks, organization storage quota
records, billing dependencies, and per-user SQLite Media databases. The test
fixture does not override those upload dependencies. It intercepts socket/DNS,
HTTP-client and model/background-job calls, including attempts caught by the app.
The Gmail connector remains disabled.

The route rejects missing or invalid credentials (401), a `read` API key (403),
a viewer without `media.create` (403), an expected-user mismatch (412), and an
exhausted organization storage quota (413). Rejections leave the email count
unchanged. A `write` key uploads successfully; the resulting message is found by
the owner and hidden from another user's database. An organization-scoped upload
is searchable in that same organization's tenant. Selecting an organization with
`X-TLDW-Org-Id` applies the same validated organization to billing, storage quota,
upload persistence, and email search/detail. A nonmember organization is rejected
with 403, as is another organization outside an org-scoped API key. A JWT's
validated active organization is the default when no organization is selected.
Reuploading the same EML into two organizations produces distinct Media
and email records; each organization's search sees only its own record. At a soft
storage limit, the success response carries `X-Storage-Warning` and the billing
limit header.

This validation found and fixed these production issues:

- Uploads wrote normalized email rows under raw client ID `1`, while authenticated
  search used tenant `user:1` or `org:1`. The resolved request tenant now crosses
  into the persistence worker and is used for the primary message and children.
- The legacy Media identity used the raw client ID while the normalized graph used
  the organization tenant. The same EML in two organizations could therefore reuse
  the first Media row and leave the second upload unsearchable. Media identity and
  worker scope now use the selected organization, while owner lookup remains tied
  to the owning user. The request selects one validated organization for quota,
  billing, persistence, search and detail.
- Explicit organization selection checked user membership but did not enforce an
  org-scoped API key's narrower claims. The billing resolver now checks the
  credential's organization scope, prefers a validated active organization over
  the first membership, and drops a prior organization's team scope when switching.
- The route checked `media.create` but let a `read` API key upload. It now requires
  the API-key `write` scope too; JWT and explicit admin behavior follows the
  existing scope guard.
- Explicit JSON responses discarded storage warning and billing limit headers.
  The route now forwards those headers from the injected response.

The storage quota service already denied backend errors by default, but its HTTP
guard allowed setup/check exceptions and reported repository outages as exhausted
quota (413). Both now honor `STORAGE_QUOTA_FAIL_OPEN`: the default returns 503
with a fixed, sanitized message; setting the variable to `1` permits the
documented fail-open behavior. The parser-only legacy fixture
uses a synthetic quota stub because that fixture has no AuthNZ user and does not
claim to test quotas. The authenticated fixture uses the real service and resets
its singleton between temporary databases.

The combined offline upload/auth/search/identity/billing/quota run passed **124
tests** with 11 environment/dependency warnings and **zero outbound attempts**.
Log: `/tmp/email_upload_combined_final3_13256.log`. A follow-up two-organization
team-scope assertion passed separately with zero outbound attempts. Repository
outage, credential-scope, active-JWT and two-organization regressions reproduced
their failures before the fixes. All
messages, keys, users and organizations were synthetic.

The existing email-child endpoint case and affected persistence unit modules
passed **27 tests** with zero outbound attempts
(`/tmp/email_upload_persistence_final_13256.log`). Ruff passes the touched
Python modules except `persistence.py`, whose 24 existing diagnostics are down
from 25 at the base revision; no new diagnostics were introduced there. Bandit
reports zero findings on touched production code and tests (pytest assertions
excluded for test files). `git diff --check` passes.

This is local ASGI and SQLite evidence. It does not exercise application lifespan,
deployed middleware, PostgreSQL/RLS, JWT login, configured billing-plan blocks,
request bodies without Content-Length, native PST/OST parsing, or the 1M-message
performance gate. The storage guard's preflight size is based on Content-Length;
this test used normal multipart requests with that header. Live Gmail/OAuth is
optional and was not used. No personal mail or LLM was accessed.
