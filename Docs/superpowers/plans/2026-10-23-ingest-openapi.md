# Web-content ingest quota boundary and connectors webhook operationIds (TASK-13889, TASK-13890)

> **Review folded in (2026-10-22).** Review: `review_plan_fu2_ingest_openapi.md` (1 Blocker, 2 Important, 3 Minor). Previous version: `plan_fu2_ingest_openapi.r1.md`.
> - **B1:** Task 2 now updates `Helper_Scripts/ci/route_auth_baseline.txt` (the split route is listed as two public routes) and verifies with `tests/lint/test_route_auth_ratchet.py`. RED measured: 1 failed with the fix and the old baseline. GREEN measured: 13 passed.
> - **I1:** Task 1 also updates `Docs/Design/2026-10-09-llm-usage-accounting-design.md:222` (no Docs/Published mirror: one copy under Docs).
> - **I2:** Decision 1 states the default-on behaviour change and the first-party callers finding.
> - **M1:** a Review Focus note says `individual` and `sitemap` are now marked prechecked.
> - M2, M3: no change.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** (1) An over-quota user's `/media/ingest-web-content` request with `perform_analysis` on gets the 402 before any page is fetched, whatever the scrape method and whether or not `api_name` is set. (2) The connectors webhook's GET and POST have distinct, stable OpenAPI operationIds.

**Architecture:**
- 13889: the endpoint already calls `enforce_llm_quota(current_user.id)` before any fetch, but only when a provider is named or the method is a crawl. Drop that condition: `if request.perform_analysis:`. This removes the last case (individual and sitemap with no `api_name`) where the quota was checked only at the LLM extraction call, whose refusal discards the scraped pages. `media_analysis_requested` and the `ScrapeMethod` import then have no use in that file and are removed.
- 13890: replace `@router.api_route(..., methods=["GET","POST"])` with stacked `@router.get(**kw)` / `@router.post(**kw)` on the same function, sharing one kwargs dict. FastAPI builds the operationId from `list(route.methods)[0]`; a two-method route puts both methods under one id, and which method's suffix it carries depends on set order (measured: `_get` in one process, `_post` in another). One route per method gives `..._get` and `..._post`, deterministically. The POST id stays what it was in the schema where it was `_post`.

**Tech Stack:** FastAPI, pytest, `Helper_Scripts/export_openapi_schema.py`, backlog-py.

**Spec:** `backlog/tasks/task-13889 - Web-content-ingest-discards-scraped-pages-when-its-LLM-extraction-is-refused.md`, `backlog/tasks/task-13890 - Connectors-webhook-GET-and-POST-share-one-OpenAPI-operationId.md`. Both were filed from the TASK-13864 work (PR #3292 and its plan review, `Docs/superpowers/plans/2026-10-20-keep-artifact-on-llm-quota-refusal.md`); 13889's description was corrected after the #3292 clean review (I1).

**Measured on:** `origin/dev` at `e06eeec68c` (2026-10-22), in an isolated worktree. Python is `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`; tests run with `TLDW_TEST_NO_DOCKER=1 ... -m pytest -p no:cacheprovider -q`.

## Global Constraints

- Test first: each RED below was measured failing on `e06eeec68c` for the stated reason.
- Quotas are off by default, so `enforce_llm_quota` is a no-op on a stock install. The 13889 behaviour change reaches only deployments that turned a limit on.
- `perform_analysis: false` and `credential_free: true` requests stay unchecked. `IngestWebContentRequest` already rejects `credential_free` with `perform_analysis` (schemas/media_request_models.py:947-963), so the preview path is unaffected.
- No new dependency, no new helper. Reuse `enforce_llm_quota`, `refuse_llm_work`, `assert_llm_quota_402` (tests/helpers/llm_quota.py).
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `--no-verify`.
- No file touched is listed in `Helper_Scripts/ci/scope_predicate_baseline.txt` (grep: `connectors`, `ingest_web_content`, `web_scraping_service` have 0 hits), so no baseline regeneration. Re-check at execution with `grep -c`.
- `Docs/Operations/Usage_Quotas.md` has no `Docs/Published` mirror (`find Docs -name Usage_Quotas.md` finds one file), so no `refresh_docs_published.sh`.

## Review Focus

- **Behaviour change for default-on analysis.** `perform_analysis` defaults to true, so an over-quota user's plain `{"urls":[...]}` request now gets 402 even when a static extractor (jsonld, schema, regex) would have succeeded without an LLM. The alternative (Decision 1) keeps the pages. Tests cover the 402 and the `perform_analysis: false` control, not "static extraction would have succeeded".
- **Prechecked marking.** `enforce_llm_quota` marks the request as prechecked (the §4a admission), so `individual` and `sitemap` requests now skip the call-site check, as the crawl methods already do. An admitted request is not refused part-way.
- **Sitemap path.** It runs `scrape_from_sitemap` in a thread; what the thread does with a call-site refusal was never measured. After this change the boundary check precedes it, so that path is unreachable for an over-quota user at request time. A user who goes over quota mid-request (between the check and an extraction call) still hits the call-site refusal; that race is out of scope and listed as a follow-up only if the owner wants it measured.
- **Fixture change.** The `context` fixture's sitemap stub did not record fetches, so "was a page fetched" could not be asserted for sitemap. The stub now appends the sitemap URL. Check that no other test in the file reads `context.fetches` after a sitemap request (the full file run in Task 1 covers this: 80 tests passed with the stub change).
- **operationId change reaches clients.** Only the default schema is generated into `apps/packages/api-types`, and the connectors router is non-stable, so the committed fingerprint and `paths.d.ts` should not move. Task 3 proves it by running the drift check rather than assuming. If a deployment enables non-stable routes and generates a client, the GET id changes from the shared id to `..._get`; no repo frontend code uses either (Task 3 step 1).
- **Decorator stacking order.** `@router.get` over `@router.post` registers POST first, then GET. Route matching is by method, so order is not observable; the test asserts both methods exist and the handler still serves both (the existing `test_connectors_webhooks.py` tests cover the validation challenge on GET).

## PR split

One PR, branch `fix/web-ingest-quota-boundary-and-webhook-opid`, base `dev`. The two changes are small, touch disjoint files, and neither blocks the other, but each is under 20 lines of code; two PRs would cost two serial merge-queue passes for no gain. Two commits, one per task.

## File map

- Modify `tldw_Server_API/app/api/v1/endpoints/media/ingest_web_content.py` (boundary check, imports).
- Modify `tldw_Server_API/tests/Media/test_web_summary_service_prompt.py` (two params, sitemap stub).
- Modify `Docs/Operations/Usage_Quotas.md` (line 93, entry-point list).
- Modify `tldw_Server_API/app/api/v1/endpoints/connectors.py` (webhook route).
- Modify `tldw_Server_API/tests/External_Sources/test_connectors_webhooks.py` (new test).
- Modify `Docs/Design/2026-10-09-llm-usage-accounting-design.md` (line 222, the web-content row).
- Modify `Helper_Scripts/ci/route_auth_baseline.txt` (webhook line 165 becomes a GET and a POST line).
- Possibly modify `apps/packages/api-types/openapi.fingerprint.json`, `apps/packages/api-types/paths.d.ts` (expected: no diff).
- Backlog: `backlog/tasks/task-13889*`, `task-13890*` via backlog-py.

## Task 0: Backlog and branch

- [ ] Branch: `git fetch origin dev && git switch -c fix/web-ingest-quota-boundary-and-webhook-opid origin/dev`.
- [ ] Move both tasks to In Progress (explicit ids, `PYTHONPATH=tools/backlog-py/src <python> -m backlog_py --cwd . task edit 13889 -s "In Progress"`; check the exact flag with `task edit --help`). Stage with `git add -A backlog/tasks`.

## Task 1: TASK-13889, check the quota at the boundary whenever analysis is on

**Files:** `tldw_Server_API/tests/Media/test_web_summary_service_prompt.py`, `tldw_Server_API/app/api/v1/endpoints/media/ingest_web_content.py`

**Interfaces:** no change. `ingest_web_content` keeps its signature. The check is `await enforce_llm_quota(current_user.id)`, which raises the 402 `HTTPException` (`limit_exceeded`, category `llm_tokens_month`, `Retry-After`).

### RED

- [ ] **Step 1: add the sitemap fetch recorder.** In the `context` fixture (around line 113), replace

```python
    monkeypatch.setattr(
        service, "scrape_from_sitemap", lambda *args, **kwargs: [article(), article("https://example.com/b")]
    )
```

with

```python
    def sitemap(url: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Replace the legacy sitemap fetch; it runs in a worker thread."""
        state.fetches.append(url)
        return [article(), article("https://example.com/b")]

    monkeypatch.setattr(service, "scrape_from_sitemap", sitemap)
```

- [ ] **Step 2: add two params to `WEB_SUMMARY_REQUESTS`**, immediately before the `process-web-scraping` param (after the `ingest-web-content-recursive-default-provider` param):

```python
    # No provider named: LLM extraction can still run, and its refusal would discard the scraped pages (TASK-13889).
    pytest.param(
        "/api/v1/media/ingest-web-content",
        {"urls": ["https://example.com/a"], "scrape_method": "individual", "perform_chunking": False},
        "perform_analysis",
        id="ingest-web-content-individual-no-provider",
    ),
    pytest.param(
        "/api/v1/media/ingest-web-content",
        {"urls": ["https://example.com/a"], "scrape_method": "sitemap", "perform_chunking": False},
        "perform_analysis",
        id="ingest-web-content-sitemap-no-provider",
    ),
```

The existing parametrized tests `test_summaries_for_an_over_quota_user_are_refused_before_any_page_is_fetched` (402, no fetch, no model call) and `test_scraping_without_summaries_is_not_refused` (control: `perform_analysis: false` returns 200 and fetches the page) now cover both.

- [ ] **Step 3: run, expect RED.**

```
TLDW_TEST_NO_DOCKER=1 <python> -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Media/test_web_summary_service_prompt.py
```

Expected on `e06eeec68c` (measured): `2 failed, 80 passed`. The failures are `...refused_before_any_page_is_fetched[ingest-web-content-individual-no-provider]` and `[ingest-web-content-sitemap-no-provider]`; the response is 200 with `analysis_status: "skipped"` and "Choose an analysis provider before running ingest analysis." (no 402, the page was fetched). Reason: `media_analysis_requested(request)` is false without `api_name`, and these two methods are not crawls. (Without the Step 1 stub change, the sitemap control also fails, `[] == ['https://example.com/a']`, which is why the stub records fetches.)

### GREEN

- [ ] **Step 4: edit the endpoint.** In `ingest_web_content.py` replace

```python
    # The crawl methods summarize with a default provider when none is named, so analysis alone means LLM work.
    crawl_summaries = request.scrape_method in {ScrapeMethod.URL_LEVEL, ScrapeMethod.RECURSIVE}
    if request.perform_analysis and (media_analysis_requested(request) or crawl_summaries):
        # Usage quotas (LLM usage accounting spec §4a), before any page is fetched.
```

with

```python
    # Analysis alone means LLM work: the crawl methods summarize with a default provider, and the others may
    # extract with an LLM even when no provider is named (TASK-13889).
    if request.perform_analysis:
        # Usage quotas (LLM usage accounting spec §4a), before any page is fetched.
```

Delete the now-unused `from ...media_processing_deps import media_analysis_requested` line and the `ScrapeMethod,` entry of the `media_request_models` import (leave `IngestWebContentRequest`). Check `grep -n "ScrapeMethod\|media_analysis_requested" <file>` shows no remaining use.

- [ ] **Step 5: run, expect GREEN.**

```
TLDW_TEST_NO_DOCKER=1 <python> -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Media/test_web_summary_service_prompt.py tldw_Server_API/tests/Media/test_ingest_web_content_endpoint_sanitization.py
```

Expected (measured): `106 passed` (82 + 24).

- [ ] **Step 6: neighbourhood.** `git grep -ln "ingest-web-content\|ingest_web_content" -- tldw_Server_API/tests` and run every hit that is not the two files above, with a baseline on unchanged dev for any failure before blaming this change (quotas are off in those tests, so none should change).

### Docs (AC 2)

- [ ] **Step 7: edit `Docs/Operations/Usage_Quotas.md` line 93.** Replace the fragment

```
`/media/ingest-web-content` when analysis is on and a provider is named, and its `url_level` and `recursive_scraping` methods whenever analysis is on (they summarize with a default provider when none is named);
```

with

```
`/media/ingest-web-content` whenever analysis is on, with or without a provider and for every scrape method (without a provider it makes no summary call, but it may still extract with an LLM, and the crawl methods summarize with a default provider);
```

`git grep -n "provider is named" -- Docs/Operations Docs/Design` to confirm no other sentence repeats the old condition for this endpoint (the `/media/process-*` and `/media/add` clause on the same line keeps "a provider is named"; leave it). Run `<python> -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Docs` if it exists and is quick, else the repo's docs check for edited files.

- [ ] **Step 7b: design doc.** `Docs/Design/2026-10-09-llm-usage-accounting-design.md:222` is one row for `/media/ingest-web-content`, `/media/process-web-scraping` and `POST /watchlists/outputs` and pins the old condition. Split it into two rows: (a) `/media/ingest-web-content` | "When `perform_analysis` is on, for every scrape method, with or without a provider. Without one it makes no summary call, but it may extract with an LLM, and checked only at the call a refusal discarded the pages already fetched." (b) `/media/process-web-scraping` and `POST /watchlists/outputs` | the existing text minus the ingest-web-content wording (`summarize_checkbox`, or `summarize`; checked only at the call, a refusal discarded the pages or items already fetched). `find Docs -name 2026-10-09-llm-usage-accounting-design.md` finds one copy, so no mirror refresh. `git grep -n "analysis with a provider named" -- Docs` afterwards shows no stale copy.
- [ ] **Step 8: commit.** `git add -A tldw_Server_API Docs && git commit` with message `fix(quotas): web content ingest checks the LLM quota whenever analysis is on (TASK-13889)` and the trailer.

## Task 2: TASK-13890, one operationId per connectors webhook method

**Files:** `tldw_Server_API/tests/External_Sources/test_connectors_webhooks.py`, `tldw_Server_API/app/api/v1/endpoints/connectors.py`

**Interfaces:** the route `/api/v1/connectors/providers/{provider}/webhook` keeps GET and POST, response model and responses doc. Only the registration changes.

### RED

- [ ] **Step 1: add the test** to `test_connectors_webhooks.py` (uses a bare app so it does not depend on what other tests mounted into the shared `app`):

```python
@pytest.mark.unit
def test_webhook_get_and_post_have_distinct_operation_ids() -> None:
    """A two-method route puts one operationId on both methods, which breaks client generation (TASK-13890)."""
    from fastapi import FastAPI

    from tldw_Server_API.app.api.v1.endpoints import connectors as connectors_router

    app = FastAPI()
    app.include_router(connectors_router.router, prefix="/api/v1")
    item = app.openapi()["paths"]["/api/v1/connectors/providers/{provider}/webhook"]
    assert set(item) == {"get", "post"}
    assert item["get"]["operationId"] != item["post"]["operationId"]
    assert item["get"]["operationId"].endswith("_get")
    assert item["post"]["operationId"].endswith("_post")
```

- [ ] **Step 2: run, expect RED.**

```
TLDW_TEST_NO_DOCKER=1 <python> -m pytest -p no:cacheprovider -q tldw_Server_API/tests/External_Sources/test_connectors_webhooks.py -k operation_ids
```

Expected on `e06eeec68c`: 1 failed, both methods carry the same id. Measured with an equivalent probe: both were `provider_webhook_callback_api_v1_connectors_providers__provider__webhook_get` in that process (the cross-process repro in AC 2 had `_post`; the suffix follows set order, which is why the `endswith` assertions only hold after the fix).

- [ ] **Step 3: record the AC 2 baseline on dev:**

```
TLDW_TEST_NO_DOCKER=1 <python> -m pytest -p no:cacheprovider -q "tldw_Server_API/tests/Watchlists/test_watchlists_api.py::test_output_summaries_return_a_quota_refusal_as_402" "tldw_Server_API/tests/Services/test_openapi_contracts.py::test_openapi_operation_ids_are_present_and_unique"
```

Expected on `e06eeec68c` (measured): `1 failed, 1 passed`, "Duplicate OpenAPI operationIds: provider_webhook_callback_api_v1_connectors_providers__provider__webhook_post: POST ... and GET ...".

### GREEN

- [ ] **Step 4: edit `connectors.py`.** Replace the decorator block above `async def provider_webhook_callback(` (currently `@router.api_route("/providers/{provider}/webhook", methods=["GET", "POST"], response_model=..., responses={...})`) with

```python
_WEBHOOK_ROUTE = {
    "path": "/providers/{provider}/webhook",
    "response_model": ConnectorWebhookCallbackResponse,
    "responses": {
        200: {
            "description": "Connector webhook callback result or plaintext validation challenge.",
            "content": {"text/plain": {}},
        },
    },
}


# One route per method: a two-method api_route gives both the same operationId (TASK-13890).
@router.get(**_WEBHOOK_ROUTE)
@router.post(**_WEBHOOK_ROUTE)
```

The function body is unchanged.

- [ ] **Step 5: run, expect GREEN.** Step 2's command: `1 passed` (measured via probe: `get -> ..._webhook_get`, `post -> ..._webhook_post`). Then the Step 3 command: `2 passed` (measured). Then `tldw_Server_API/tests/External_Sources/test_connectors_webhooks.py` whole file: `8 passed` before the new test, `9 passed` after. After Step 5b, `tldw_Server_API/tests/lint/test_route_auth_ratchet.py` whole file: `13 passed` (re-review measured). If `test_force_enabled_routers_are_mounted_with_explicit_config_file` fails once, rerun it and baseline it on unchanged dev before blaming the baseline edit (the re-review saw one transient failure). Also run `test_connectors_endpoints_api.py` and `test_connectors_endpoint_sanitization.py` and compare against a dev baseline for any failure.

- [ ] **Step 5b: route-auth baseline (B1).** The ratchet lists public routes by method set, so the split makes it fail.
  - RED (measured on e06eeec68c plus the fix): `TLDW_TEST_NO_DOCKER=1 <python> -m pytest -p no:cacheprovider -q tldw_Server_API/tests/lint/test_route_auth_ratchet.py` gives 1 failed, "Route authentication ratchet failed" (the new `GET` and `POST` routes are not in the baseline).
  - Fix: `PYTHONPATH=. <python> Helper_Scripts/ci/route_auth_ratchet.py --write-baseline` regenerates the file, **but it also deletes the `# TASK-13399` comments** (measured: a diff of 13 deletions). Do not commit that output. Instead `git checkout -- Helper_Scripts/ci/route_auth_baseline.txt` and hand-edit it (it is not a backlog file): delete the `GET,POST /api/v1/connectors/providers/{provider}/webhook` line and the `# TASK-13399: ... Signed webhook: provider-specific shared secret ...` comment above it (line 165), then insert that same comment followed by `GET /api/v1/connectors/providers/{provider}/webhook` after `GET /api/v1/config/tokenizer`, and the comment followed by `POST /api/v1/connectors/providers/{provider}/webhook` after `POST /api/v1/config/validate-provider` (sorted positions, as the regenerated file shows). Diff measured: 4 insertions, 2 deletions.
  - GREEN (measured with the hand edit): the same test, 13 passed.
- [ ] **Step 6: commit.** Add `Helper_Scripts/ci/route_auth_baseline.txt`, and say in the message why a public route is listed again: the one two-method route became two single-method routes, so the baseline's `GET,POST` entry became a `GET` and a `POST` entry; the route's auth is unchanged. Subject: `fix(openapi): connectors webhook GET and POST get distinct operationIds (TASK-13890)` with the trailer.

## Task 3: Schema artifacts and frontend check (TASK-13890)

- [ ] **Step 1: frontend usage.** `git grep -n "provider_webhook_callback" -- apps` and `git grep -n "providers/.*webhook" -- apps/tldw-frontend apps/packages`. Measured on `e06eeec68c`: no hit for `provider_webhook_callback` anywhere under `apps/`; `apps/packages/api-types/paths.d.ts` has 0 `connectors` entries (the router is not in the default schema). `hooks/useConnectorBackend.ts` and `scripts/smoke.mjs` mention `connectors/providers` paths for other endpoints, none use the webhook path or an operationId. Record the commands' output in the task notes.
- [ ] **Step 2: drift check** (measured passing on dev with the fix applied; must also pass at execution):

```
PYTHONPATH=. <python> Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts
```

Expected: `OpenAPI fingerprint matches the checked-in snapshot.` and `committed path union matches the canonical contract.`
- [ ] **Step 3: regenerate and diff.** Run the exporter's regenerate mode (`<python> Helper_Scripts/export_openapi_schema.py --help` for the exact flags that write `openapi.fingerprint.json` and `paths.d.ts`; the `apps/packages/api-types` package may also have an npm script that wraps it) and `git diff --stat apps/packages/api-types`. Expected: empty diff, because the default schema excludes connectors. If a diff appears, commit it in this PR and say why in the task notes (it would mean the router is in the default schema, contrary to the task text). Re-run Step 2.
- [ ] **Step 4:** `<python> -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Services/test_openapi_contracts.py` alone (measured on the 13864 PR: 74 passed; re-measure at execution and record the count).

## Backlog close-out

For each task, with backlog-py (explicit ids, `--append-notes`, never `--notes`; stage `git add -A backlog/tasks`):

- **TASK-13889:**
  - Notes: RED `2 failed, 80 passed` (e06eeec68c, test_web_summary_service_prompt.py); GREEN `106 passed` (with the sanitization file); the Usage_Quotas.md edit; ADR assessment; Decision 1 outcome.
  - ACs: #1 checked (402 before any page, per the two new params). #2 checked (Usage_Quotas.md line 93 says: 402 before any fetch whenever analysis is on). AC 1's alternative ("or keeps its pages without LLM extraction") was not taken; no rewording needed.
  - DoD: #1-#3 checked; #4 Bandit: run `<python> -m bandit -q tldw_Server_API/app/api/v1/endpoints/media/ingest_web_content.py` and record; #5 final summary; #6 note that the mid-request quota race is not covered.
  - Status Done.
- **TASK-13890:**
  - Notes: RED `1 failed` new test, and `1 failed, 1 passed` ordering pair on dev; GREEN `1 passed`, ordering pair `2 passed`, webhooks file all passed; drift check output; frontend grep result; fingerprint/paths.d.ts diff empty (or the diff).
  - ACs #1 and #2 checked. DoD #1-#6 checked (Bandit on connectors.py recorded; docs: none needed, say so).
  - Status Done.
- Add a PR link to both notes after opening.

## ADR assessment

Not required. No durable architecture rule changes: 13889 widens an existing quota check to the condition the quota design already states (spec §4a: refuse before LLM work); 13890 is a registration detail. Governing decisions: the LLM usage accounting design (`Docs/Design/2026-10-09-llm-usage-accounting-design.md` or its current path; link it in the notes) and ADR-059 for the backlog tooling. Re-check `Docs/ADR/README.md` for a quota ADR at execution.

## Follow-ups (file at execution as TASK-<N>)

- None required. Optional, only if the owner picks Decision 1 option B: a task to give `allow_llm_extraction=False` to an over-quota user's individual and sitemap scrapes (needs a non-raising quota probe).
- Optional: grep other `api_route(methods=[...two or more...])` declarations in `tldw_Server_API/app/api/v1/endpoints` for the same shared-operationId defect (`git grep -n "api_route" -- tldw_Server_API/app/api/v1/endpoints`); file one task per router that is mounted into the default schema. Not measured here.

## Verification before opening the PR

- Task 1 step 5 and Task 2 step 5 commands pass; Task 3 drift check passes.
- `ruff check` (or the repo's configured linter) on the four touched Python files; no trailing whitespace in the plan or diffs.
- Open the PR against `dev` with the Change summary section (ADR-004: the waiver in memory does not name these tasks, so ask the owner for the Change summary before merge). Run `/superpowers:requesting-code-review` in a clean subagent before merge; check `gh variable get MERGE_QUEUE` before arming.

## Decisions for the owner

1. **What an over-quota user gets from `/media/ingest-web-content` with default `perform_analysis`, individual or sitemap, no `api_name`.**
   - **Recommended: A, 402 before any fetch** (this plan). A changes behaviour under a default-on flag: `perform_analysis` defaults to true, so an over-quota user's bare `{"urls":[...]}` request gets 402 where it got 200, even when static extraction would have worked without an LLM. The review found no first-party caller that sends a bare request: `extractPublicArticle` sends `perform_analysis: false`, and Quick Ingest has a toggle. Only direct API clients on a deployment with a limit switched on see it. Cost: a plain `{"urls":[...]}` request is refused even when static extraction would have worked without an LLM; the user must send `perform_analysis: false` to ingest. It matches `process-web-scraping`, the crawl methods and `/media/add`, and is four lines.
   - B, keep the pages: pass `allow_llm_extraction=False` when the user is over quota, so jsonld/schema/regex extraction still works. Cost: needs a non-raising quota probe in `web_scraping_service`, threads through both methods, and the user gets 200 with silently weaker extraction. AC 1 allows it. More code and a new "quota-aware" branch for little value.
