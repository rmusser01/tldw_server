# UAT129 / TASK13260.69 — Prompt collections uses the wrong auth mode

## Confirmed root cause

The Prompt endpoint's local auth-mode helper gives the legacy `settings['SINGLE_USER_MODE']` flag priority over canonical AuthNZ `AUTH_MODE`. Core config derives the legacy flag from `APP_MODE`, whose absent/default value is `single`. Thus a valid deployment with **AUTH_MODE=multi_user and APP_MODE unset** incorrectly enters single-user API-key validation for Prompt requests.

The supplied JWT is compared to the single-user key, rejected, and never reaches unified AuthNZ. This matches the native `401 Invalid authentication token` on Prompt collections while other canonical multi-user routes operate.

Source paths below are relative to `/Users/macbook-dev/Documents/GitHub/tldw_server2`:

- `tldw_Server_API/app/core/config.py:944–946`: APP_MODE defaults to single and derives the legacy boolean independently of AUTH_MODE.
- `tldw_Server_API/app/api/v1/endpoints/prompts.py:371–377`: early return for legacy true prevents canonical settings from being read.
- Same file `:421–432`: the incorrectly selected single-user branch compares the credential against the single-user key and emits the native error.
- Same file `:476–490`: the intended multi-user branch delegates to `get_request_user`; it is never reached in the failing case.
- Same file `:1732–1744`: real collections route uses `verify_prompts_user`, which calls that helper.
- `tldw_Server_API/app/core/AuthNZ/settings.py:1620–1641`: existing canonical mode helpers explicitly use AUTH_MODE and prohibit advisory mode/profile hints from relaxing authentication.

## Retained native evidence

- `/private/tmp/cycle5-multi-native-prompt-collections-reload.txt`: actual reload GET `/api/v1/prompts/collections` returns401 at 2026-09-16T08:29:03.945Z, body `Invalid authentication token`.
- `/private/tmp/cycle5-controller-multi-auth-mode-observation.json`: safe field-only capture at08:36:56.880Z confirms `AUTH_MODE: multi_user`, `APP_MODE: unset`; no credentials/provider values.
- `/private/tmp/cycle5-multi-native-prompt-collections-error.txt` contains an empty response-capture result; it does not independently prove another request or successful response.

Root reports same-Alice Note/Chat/QA200 and a Prompt “Synced #1” badge after reload. Those are useful contrasting observations; a persisted sync badge alone does not prove a fresh successful request through this same legacy Prompt dependency. No request credentials were inspected or logged in this diagnosis, and no inference about why any separate sync operation succeeded is needed for the reproduced cause.

## Isolated actual-module/route regression

Artifacts:

- `/private/tmp/cycle5-uat129-auth-precedence-probe.py`
- `/private/tmp/cycle5-uat129-auth-precedence-probe.log`
- `/private/tmp/cycle5-uat129-source-hashes.txt`

Command from repository root:

```sh
source .venv/bin/activate && PYTHONDONTWRITEBYTECODE=1 python /private/tmp/cycle5-uat129-auth-precedence-probe.py
```

The fresh process clears inherited environment values, uses only synthetic credentials, empty private config/env files and private data directories, and rejects outbound socket connections. It imports the **actual** Prompt module and canonical settings, mounts the **actual** collections route/dependency under FastAPI, and uses in-process HTTPX ASGI transport. The unified user resolver is a controlled authentication boundary and the DB dependency returns an empty fake collection DB. No application startup/runtime, real DB, live API, browser or inference is invoked.

Observed actual configuration before any test override:

```json
{"canonical_mode":"multi_user","APP_MODE_present":false,"legacy_single_user_mode":true,"prompt_helper_single_user_mode":true}
```

**1 regression failed, 9 controls passed.**

| Case | Expected | Actual | Unified auth calls |
| --- | ---: | ---: | ---: |
| Canonical multi, legacy true, controlled valid JWT | 200 | **401** | **0** |
| Same canonical mode/JWT, only legacy flag false | 200 | 200 | 1 |
| Invalid JWT | 401 | 401 | 1 |
| No credentials | 401 | 401 | 0 |
| Multi-user API key | 200 | 200 | 1 |
| Legacy Token: Bearer compatibility | 200 | 200 | 1 |
| Authenticated non-admin at admin dependency | 403 | 403 | 1 |
| PROMPTS_REQUIRE_ADMIN with ordinary user | 403 | 403 | 1 |
| Canonical single, legacy false, exact single-user key | 200 | 200 | 0 |
| Canonical single, wrong key | 401 | 401 | 0 |

Multi-user negative/compatibility controls use the aligned legacy-false state so they actually exercise delegation. Single-user controls supply canonical single mode in process. This verifies routing, header forwarding and claim checks; it does **not** independently validate JWT signatures, expiry/revocation, API-key DB lookup or per-user database isolation because those downstream boundaries are controlled here.

## Smallest post-freeze repair

Expected production ownership: **`prompts.py` auth-mode helper only**, plus focused auth/route regressions and task69. Do not change global APP_MODE defaults or the live deployment environment to conceal the mismatch.

Make the canonical AuthNZ setting determine single-user behavior. The existing canonical helper or direct `get_auth_settings().AUTH_MODE` comparison suffices; the early legacy-true shortcut must go. Also do not let a canonical settings read/validation failure fall back into privileged single-user behavior through the same legacy boolean. Preserve failure as an error/denial, using existing error conventions. No new auth abstraction is necessary.

Keep the actual key comparison, unified JWT/API-key resolver, credential header rules, per-user DB dependency, claim-first admin check and PROMPTS_REQUIRE_ADMIN semantics unchanged. The correction should route valid multi-user credentials to existing validators, not grant access itself or synthesize admin claims from a legacy flag. Do not auto-retry or replay writes as part of this backend mode correction.

## Permanent regression/verification plan

1. Real router/dependency regression with canonical multi_user, APP_MODE absent/legacy true and valid authenticated user: collections200 and unified resolver used. Keep legacy-false control. Assert the actual canonical setting, not a mocked `_is_single_user_auth_mode` result.
2. Canonical multi-user must win over either legacy boolean. Cover valid bearer, expired/revoked/invalid bearer, malformed/missing credentials, valid/invalid API keys and legacy Token header using existing AuthNZ fixtures for real credential validation where appropriate. Do not change precedence for requests containing multiple credential headers in this bounded repair.
3. Canonical single-user must win over legacy false, accept only its configured key through the supported headers, and reject wrong/missing keys. Preserve missing-key misconfiguration handling.
4. Canonical settings failure with legacy true must not authorize a synthetic single-user/admin identity. Explicitly test fail-closed behavior.
5. Ordinary authenticated user remains rejected by admin-only routes; claim-bearing admins remain accepted. PROMPTS_REQUIRE_ADMIN is enforced when enabled. Retain `tests/AuthNZ_Unit/test_prompts_admin_claims.py`; those existing tests mock `_resolve_prompts_auth_user` and therefore do not cover mode selection.
6. Endpoint integration should use the actual per-user DB dependency for an Alice/Bob isolation control. The diagnostic's fake DB must not be mistaken for tenant-isolation acceptance. Preserve structured/recipe persistence guards in related Prompt tests.
7. Run scoped Prompt/AuthNZ tests, Ruff/baseline comparison and **Bandit through the project venv on changed Python scope**. Independent review then native same-user collections reload200, unauthenticated/foreign negative controls, and genuine single-user key behavior after freeze. No provider inference is required.

## Freeze and limits

No repository/source/test files were edited; no live processes or browser state changed. The temporary script intentionally exits1 because the desired multi-user regression remains RED. Nine passing controls do not mean the repair is implemented. Import emitted existing Pydantic field-shadowing and FastAPI422 deprecation warnings; no clean-warning claim.

Five source/probe SHA256 records were retained and verified unchanged; Prompt source hash is `29878d25faad0cdeca6feafcd2f2d429b9fd306ebcab975e303a44d04bee3328`. Full paths and hashes are in `/private/tmp/cycle5-uat129-source-hashes.txt`.
