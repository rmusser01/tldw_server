# UAT132 — media search FTS failure drops query filtering

## Confirmed outcome

The native harness used the correct request field. `POST /api/v1/media/search` takes `SearchRequest.query` (`media_request_models.py:102`), defaults fields to title/content, rejects unknown keys, and `media/listing.py:1115–1158` forwards it as `search_query`. The frontend `domains/media.ts:452` uses the same body contract. No harness field correction is justified.

`/private/tmp/cycle5-multi-native-isolation-api.json` records four POST200 requests: each account's own marker and the other account's marker all return that caller's sole own row. Alice gets only Aurora; Bob gets only Bob's private source. These are valid non-disclosure controls but not successful query filtering. No foreign row was exposed in the retained responses.

## Root cause and isolated proof

`.../core/DB_Management/media_db/repositories/media_search_repository.py:320–393` passes unquoted hyphenated strings into SQLite FTS5. While FTS is active, it does not prepare title/content LIKE predicates. At count fallback lines511–548 and result fallback lines580–620, the error handler removes the MATCH condition/parameter/join but does not add equivalent title/content filtering. It retains scope/deleted/trash constraints, so a malformed FTS query can become “all visible media.”

Private script `/private/tmp/cycle5-media-search-query-probe.py` runs the **actual MediaSearchRepository** and real scope context against isolated in-memory SQLite FTS5 with two synthetic owners/rows. It clears inherited environment, blocks outbound networking and uses no production DB or API. Output `/private/tmp/cycle5-media-search-query-probe.log`:

- **2 RED**: Alice query `CYCLE5-BOB-PRIVATE-MEDIA` returns Alice's unrelated row; Bob query `AURORA-CYCLE5-MULTI-23` returns Bob's unrelated row. Real SQLite raises `no such column: BOB` / `no such column: CYCLE5` before fallback.
- **6 GREEN controls**: simple unmatched words return zero; quoted foreign markers return zero; each own marker returns its own row. The own hyphenated positives also hit the bad fallback, so those two positives alone do not establish correct relevance.
- All eight results remain owner-scoped.

Command: `source .venv/bin/activate && PYTHONDONTWRITEBYTECODE=1 python /private/tmp/cycle5-media-search-query-probe.py` (exit1 from the two expected RED cases). Initial private assertion referenced an unselected owner column; corrected to the returned client_id before final run. This fixture correction is not a product finding.

## Minimal repair / required tests

Bounded owner: `media_search_repository.py` plus existing repository/API tests. On recognized FTS failure, preserve the original sanitized text-field query as parameterized LIKE filtering in both count and result paths, retaining scope/media-type/keyword/date/trash/deleted constraints and consistent total/pagination. Reuse one local fallback construction where practical. Do not indiscriminately quote every query and break existing supported FTS operators; do not return an unconstrained list while reporting search success.

Permanent tests: actual SQLite+repository→API for quoted/unquoted hyphenated matching/nonmatching markers; multiple own rows and another owner to distinguish filtering from isolation; count-error and result-only-error fallback; missing FTS table; title-only/content-only/author fields; valid FTS operators, pagination totals, trash/deleted/media-ID/team/org controls. Preserve Postgres SQL/tsquery behavior; real Postgres acceptance remains unverified by this SQLite-only probe. Independent review and native repeat after freeze are still required.

No repository, browser, live API/runtime or task edits. This is separate from Prompt-collection auth UAT129 and retains the observed account-isolation pass. Hashes: `/private/tmp/cycle5-controller-additional-diagnosis-hashes.txt`.
