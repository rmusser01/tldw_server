# UAT264 Sources collection redirect authentication repair report

## Change

`collectionsMethods.listIngestionSources` and `createIngestionSource` now call the backend's registered collection endpoint, `/api/v1/ingestion-sources/`. The backend exposes its protected collection `GET` and `POST` routes at that slash-terminated path. Per-source paths and request behavior are unchanged.

The maintained client test now asserts the canonical list and create paths. Its causal boundary control rejects the former slashless route as the observed cross-origin redirect path, verifies authenticated client setup for each operation, and verifies neither request opts out of authentication. Existing detail, item, update, sync, archive, and reattach assertions remain controls for the unchanged per-source routes.

## Evidence

- Red: focused client suite exited 1 with the three intended stale-route failures (two exact path assertions and the authenticated redirect boundary); four controls passed.
- Green: the focused suite exited 0: 7 tests passed.
- Adjacent request-scope and auth regressions exited 0: 67 tests passed across the ingestion client, request-scope, and quickstart-auth suites. The quickstart fixture emitted its expected mocked cleanup and missing-key messages.
- ESLint on the two changed files exited 0 with no findings. `git diff --check` exited 0.
- The full TypeScript CLI emitted the repository's 90 existing diagnostics. Its shell status was not retained because the command crossed the tool's 30-second boundary. The retained TypeScript API comparison completed with `baselineCount: 90`, `currentCount: 90`, `added: 0`, and `removed: 0`.
- Bandit exited 0 and reported zero findings, while reporting parser errors for both `.ts` files. That is Bandit's Python AST limitation; it is not a TypeScript security assessment.

Command strings, statuses, retained log names, and hashes are in `.tmp/uat-repairs-231-246/sources264/command-receipts.json`.

## Source hashes

- `apps/packages/ui/src/services/tldw/domains/collections.ts`: `99d63fccf990319e5f5293db0d5c04fca5a6f837ebfae9ab5e750f3e58708ea0`
- `apps/packages/ui/src/services/__tests__/tldw-api-client.ingestion-sources.test.ts`: `459d4f15a5477661d8f6c4dd935dd9a1ae6fa63b2b808289d3404dd1a2db2175`
