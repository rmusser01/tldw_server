## Stage 1: Verify Review Finding
**Goal**: Confirm PR #1104 sharing endpoint lazy repo factories await the async AuthNZ DB pool before constructing repositories.
**Success Criteria**: Focused tests fail when `_get_repo()` or `_get_prototype_repo()` stores the coroutine returned by `get_db_pool()`.
**Tests**: Focused pytest for sharing endpoint factory helpers.
**Status**: Complete

## Stage 2: Implement Awaitable Factory Resolution
**Goal**: Make sharing endpoint repo/service factories awaitable while preserving existing sync test monkeypatches.
**Success Criteria**: Shared and prototype repositories receive the resolved DB pool object, and endpoint call sites resolve patched sync or async factories consistently.
**Tests**: Focused sharing endpoint tests and prototype link exchange tests.
**Status**: Complete

## Stage 3: Verify and Publish
**Goal**: Run focused backend tests, Bandit on touched backend code, diff checks, then push/reply to the relevant PR thread.
**Success Criteria**: Local verification passes and the addressed PR #1104 review thread has a reply.
**Tests**: Focused pytest, Bandit, `git diff --check`.
**Status**: In Progress

### Verification
- Red: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sharing/test_sharing_endpoints.py::test_lazy_shared_repo_awaits_db_pool tldw_Server_API/tests/Sharing/test_sharing_endpoints.py::test_lazy_prototype_repo_awaits_db_pool -q` failed because both repos stored the coroutine from `get_db_pool()`.
- Green: the same focused command passed with 2 tests.
- Regression: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sharing/test_sharing_endpoints.py tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_link_exchange.py -q` passed with 50 tests.
- Bandit: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/sharing.py -f json -o /tmp/bandit_pr1104_async_sharing_repo.json` reported no results and no errors.
- Diff: `git diff --check` passed.
