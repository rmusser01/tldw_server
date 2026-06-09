## Stage 1: Backend Filter And Sort Contract
**Goal**: Extend the Skills list API/service/registry query with a safe, whitelisted filter and sort contract.
**Success Criteria**: `GET /api/v1/skills` accepts `context`, `user_invocable`, `has_tools`, `model`, `sort`, and `order`; filters and sort apply before pagination; totals reflect the filtered set.
**Tests**: Focused Skills service/API tests, with the target rows outside the first unfiltered page.
**Status**: Complete

## Stage 2: Frontend Client Query Parameters
**Goal**: Expose the backend contract through typed WebUI client params without component-level URL construction.
**Success Criteria**: `listSkills` accepts typed camelCase params, serializes supported filter/sort params, trims string filters, and omits blank values.
**Tests**: API client boundary test for mixed search/filter/sort serialization.
**Status**: Complete

## Stage 3: Skills Manager Filter And Sort State
**Goal**: Let power users filter and sort the Skills table through controlled query state.
**Success Criteria**: The table query key includes filters and sort; changing filters resets to page 1; table sort changes request server-backed ordering.
**Tests**: Focused Skills manager Vitest coverage for filter requests, sorted table requests, pagination reset, and rendered filtered results.
**Status**: Complete

## Stage 4: Verification And Closeout
**Goal**: Prove the backend, client, and UI agree on the filter/sort contract and record the result in Backlog.
**Success Criteria**: Focused pytest/Vitest pass, Bandit passes on touched backend scope, `git diff --check` passes, and `TASK-530.4` records verification and known skips.
**Tests**: Focused pytest, focused Vitest, Bandit, diff check.
**Status**: Complete

## Verification Results

- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Skills/unit/test_skill_registry_queries.py tldw_Server_API/tests/Skills/unit/test_skills_service.py tldw_Server_API/tests/Skills/integration/test_skills_api.py -q` - 100 passed, 6 warnings.
- `bunx vitest run src/services/__tests__/tldw-api-client.boundary-slices.test.ts src/components/Option/Skills/__tests__/Manager.test.tsx` - 24 passed.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/skills.py tldw_Server_API/app/api/v1/schemas/skills_schemas.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/Skills/skills_service.py -f json -o /tmp/bandit_skills_filters_sorting_TASK_530_4.json` - exit 0; remaining warnings are baseline `nosec` comments in the large DB module.
- `git diff --check` - passed.
- `bunx tsc --noEmit -p tsconfig.json` - attempted as an extra guard; default heap run OOMed, 8GB retry reached existing package-level type failures in Notes tests, background response result handling, and voice-cloning ArrayBuffer typing, so this was not used as the completion gate.
