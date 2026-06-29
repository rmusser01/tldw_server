# Local Single User

- Run id: `pre-main-uat-20260629054510`
- Task id: `TASK-12064`
- Backlog task: `TASK-12066`
- Status: Pass after fix

## Runtime

- API: `http://127.0.0.1:8000`, detached process listening on loopback.
- WebUI: `http://127.0.0.1:8080`, detached Next dev process listening on loopback.
- Runtime config: `GET /api/_tldw-webui/runtime-config` returned HTTP 200 with `runtimeAuth.available=true`, `authMode=single-user`, and an API key present. The key was not logged.
- Backend proxy: `GET /api/v1/setup/readiness/status` through the WebUI origin returned HTTP 200.
- Browser: fresh tab at `http://127.0.0.1:8080/` rendered the first-time setup screen with setup readiness lanes and setup path choices. Console contained only React DevTools, HMR, and Fast Refresh development messages.

## Finding Fixed

- UAT exposed a quickstart bootstrap bug in `apps/tldw-frontend/pages/api/_tldw-webui/runtime-config.ts`.
- The route rejected any forwarded metadata before exposing runtime auth. Next dev sends loopback forwarding metadata on local requests, so the WebUI stayed unauthenticated and routed to `/settings/tldw` with repeated 401s from readiness and notification requests.
- Fixed by allowing only loopback-only `Forwarded` and `x-forwarded-for` values, ignoring `x-forwarded-host` for the exposure decision, and continuing to reject external/empty forwarded client IP values and `x-real-ip`.
- Added regression coverage in `apps/tldw-frontend/__tests__/pages/api/runtime-config.test.ts`.

## Verification

- Red test observed before implementation:
  - `bunx vitest run __tests__/pages/api/runtime-config.test.ts`
  - Failed on loopback forwarded metadata being rejected.
- Final focused tests:
  - `bunx vitest run __tests__/pages/api/runtime-config.test.ts __tests__/extension/runtime-bootstrap.test.ts`
  - Result: 2 files passed, 67 tests passed.
- Final local probes:
  - `GET http://127.0.0.1:8080/api/_tldw-webui/runtime-config` returned HTTP 200 with runtime auth available.
  - `GET http://127.0.0.1:8080/api/v1/setup/readiness/status` returned HTTP 200 through the WebUI proxy.
- Final browser check:
  - Page URL: `http://127.0.0.1:8080/`
  - Visible state: first-time setup, setup readiness, and setup path buttons.
  - Console: no auth/readiness errors after the fix.

## Notes

- The temporary UAT launcher must parse `export NAME=value` lines in the run-scoped `uat.env`; an earlier local restart missed the API key because the parser treated `export UAT_API_KEY` as the variable name.
- `bun install --frozen-lockfile` in `apps/` repaired a stale local `apps/packages/ui/node_modules/antd` symlink needed for the WebUI dev server to resolve shared UI imports. This dependency-state repair is left unstaged and is not part of the product fix.
- Bandit is not applicable for this TypeScript-only fix.
