# UAT208 — truthful optional visual metadata support

TASK13260.146. Parent approved option A in `.tmp/uat208-diagnosis-20260917/DESIGN208.md`. Four production files and two new regression files are frozen; independent review and native authenticated Character load/reload remain pending. No metadata PostgreSQL port or frontend hook/panel behavior change.

## Behavior and reason

The ordinary Character/Persona resolver previously depended on an eagerly constructed SQLite-only visual metadata repository. PostgreSQL therefore returned500 before reaching actor validation or a normal optional fallback.

The route now branches only for the explicitly selected PostgreSQL backend, validates the actor using the same existing lookups, and returns a typed no-asset result with `fallback_reason=metadata_backend_unsupported`. It preserves actor ID normalization, requested expression and role metadata. No pack/version/asset/storage/URL is invented. Existing frontend success caching and absent-asset handling consume this response without a new request or cache mechanism.

Explicit PostgreSQL pack/version overrides and metadata authoring/read operations through `_service` return501 with `detail=visual_identity_metadata_backend_unsupported`. They do not pretend to resolve or create a pack, even if override fallback was requested. SQLite's service/repository, existing legacy/pack resolution and strict override behavior remain on the existing path. The repository's direct SQLite-only guard remains unchanged.

Capabilities still report the same format/size limits and now additionally report `metadata_supported` and nullable `metadata_unavailable_reason` from the selected database. The matching frontend fields are optional for older-server compatibility. The pure image constraints builder remains format-only. Existing authoring UI does not consume this flag; its requests receive the explicit unsupported error. A dedicated disabled authoring UI and a PostgreSQL implementation remain outside this repair.

The service actor validator was extracted into a narrow module helper and the original method delegates to it. `source-review.json` proves the validation body is AST-identical after mapping `self.db`/`self.owner_user_id` to arguments. Existing actor lookup ownership/deletion rules remain authoritative. Authentication, rate-limit, malformed query and genuine lookup failures are not turned into success. No SQL or migration changed.

## Exact scope

Production:

- `tldw_Server_API/app/api/v1/endpoints/visual_identities.py`
- `tldw_Server_API/app/core/Visual_Identities/service.py`
- `tldw_Server_API/app/api/v1/schemas/visual_identity_schemas.py`
- `apps/packages/ui/src/types/visual-identities.ts`

New tests:

- `tldw_Server_API/tests/Visual_Identities/test_visual_identity_backend_support.py`
- `apps/packages/ui/src/hooks/__tests__/useVisualIdentityResolver.backend-support.test.tsx`

`owned-manifest.json`, `owned.patch` and `review-snapshot/` bind all six final paths. Baseline copies were retained before implementation. Existing tests were not edited. Root was notified after the final production edit; no runtime restart or browser action was performed by this agent.

## Causal proof and controls

The permanent backend suite mounts the actual API router and real `CharactersRAGDB` on SQLite and official `pg_database_config` fixtures. It creates real owned characters/personas. Only selected DB/principal, rate-limit and unused Jobs-manager dependency boundaries are substituted. It does not claim real authentication/RBAC policy, model inference or native application coverage.

- Permanent pre-change RED: **20 failed /24 passed /0 skipped**,57.09s (`permanent-red.redacted.log`). Missing support fields and the eager PG500 account for the failures.
- Initial post-change run: **43 passed /1 failed**. The foreign-character fixture incorrectly closed the shared backend pool before its route read. It now releases only its own checkout; production remained unchanged. Preserve `initial-green-fixture-failure.redacted.log` rather than counting it as an application defect.
- Final corrected-fixture replay against the three exact original Python modules with a nonmutating hash-checking import loader: **20 failed /24 passed /0 skipped**,45.00s (`final-baseline-red.redacted.log`). This confirms the final test bytes still detect the original defect; the fixture-only adjustment does not weaken owner assertions.
- Final current suite: **44 passed /0 skipped**,45.28s (`final-green.redacted.log`). Both actual PostgreSQL and SQLite ran.
- Adjacent actual API/service/repository/capabilities/VN bridge/Character expression metadata: **107 passed /0 skipped**,71.12s (`adjacent-green.redacted.log`).
- New actual-hook compatibility controls and existing hook suite: **13 passed /2 files /0 skipped** (`hook-final-green.log`); they passed before production changes too, as expected for an unchanged hook.

Backend cases cover owned characters/personas, alias normalization/role metadata, missing/deleted/foreign actors, PostgreSQL owner scoping versus SQLite device IDs, malformed queries, strict overrides, authoring, capabilities, unchanged direct repository rejection, principal401/rate429 propagation, and an actual failed database query that must remain500. The database-failure control records that the real lookup was reached so the old eager failure cannot falsely pass it.

Hook cases use the real hook and cache with only the remote client response seam replaced. They prove successful unavailable results coalesce/cache through remount, expose no asset/error, explicit refresh can recover a newly available asset, and real401/500 failures remain visible and are not cached as success. There is no error polling timer in this hook; different expression keys or an explicit reload may still make a successful unavailable request. No permanent cross-account support cache was added.

## Reproduction commands

From repository root, use the official required-PG runner (it owns disposable fixture databases; Docker autostart disabled):

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat208-independent-final node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/Visual_Identities/test_visual_identity_backend_support.py tldw_Server_API/tests/Visual_Identities/test_visual_identities_api.py tldw_Server_API/tests/Visual_Identities/test_visual_identity_service.py tldw_Server_API/tests/Visual_Identities/test_visual_identity_db.py tldw_Server_API/tests/Visual_Identities/test_visual_identity_capabilities.py tldw_Server_API/tests/Visual_Identities/test_visual_identity_vn_bridge.py tldw_Server_API/tests/Character_Chat/test_visual_identity_expression_metadata.py -q --tb=short
```

Expected combined count151. The author ran44 and107 in separate exact commands retained in the packet. For causal replay on final tests (expected20 failures/24 controls):

```sh
source .venv/bin/activate
PYTHONPATH="$PWD/.tmp/uat208-repair-20260917:$PWD" TLDW_UAT_EVIDENCE_LABEL=uat208-independent-baseline-red node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs -p baseline_replay tldw_Server_API/tests/Visual_Identities/test_visual_identity_backend_support.py -q --tb=short
```

From `apps/tldw-frontend`:

```sh
bunx vitest run ../packages/ui/src/hooks/__tests__/useVisualIdentityResolver.backend-support.test.tsx ../packages/ui/src/hooks/__tests__/useVisualIdentityResolver.test.tsx
bunx tsc --noEmit --incremental false
```

## Static/security validation

- Ruff:2 existing I001 import-order findings before and after, same files; zero added. New Python regression file clean. No broad import cleanup mixed in.
- Scoped ESLint:0 errors/0 warnings for the TypeScript type and new hook tests.
- Full TypeScript compiler:90 baseline/90 current, output byte-identical; no owned errors. This is not a clean whole-project compiler claim.
- Bandit production Python:0 findings/0 Python parse errors. The included TypeScript type has1 unsupported parse error. Test scan excluding normal B101 assertions:0 findings, new TSX test has1 unsupported parse error. Bandit supplies no TypeScript security assurance.
- Python AST parse, exact extracted-validator equivalence, owned patch whitespace and frozen source hashes pass.

Manual security review: no metadata read/write or guessed asset path is performed in the unsupported branch; real owner/deletion lookups run first, and actual dependency/DB errors are preserved. The repair does not weaken direct repository guards or change authentication, rate limits, storage, SQL, remote job execution, browser/session credentials or persisted data policy.

## Handoff limits

Native PostgreSQL Character load/retry/reload, continued core-chat behavior, and inspection of optional no-asset wire responses remain parent-owned after independent review. This test unit neither claims PostgreSQL visual-authoring support nor native acceptance. Root's original native evidence is separate. No task/tracker/git/staging/browser/native runtime or native-data mutation occurred here; the official fixtures created only disposable test databases. Only redacted test logs and command metadata were copied; private raw runner logs remain outside the packet.
