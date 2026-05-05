## Stage 1: Verify Review Findings
**Goal**: Confirm the PR #1104 `revoked_at` authorization and Loguru lint comments still apply on the current branch.
**Success Criteria**: A focused test fails for a `revoked_at`-only actor/session, and Ruff can target the logger format issue.
**Tests**: Focused pytest for prototype workspace service authorization helpers; targeted Ruff PLE1205 check on `tldw_Server_API/app/main.py`.
**Status**: Complete

## Stage 2: Implement Narrow Fix
**Goal**: Treat `revoked_at` as revoked anywhere branch actor/session activity is checked, and convert the minimal-router Loguru call to supported formatting.
**Success Criteria**: Revoked collaborators cannot create/reuse branch sessions or save snapshots when only `revoked_at` is present; Ruff PLE1205 no longer flags the main.py line.
**Tests**: Focused pytest and Ruff target pass.
**Status**: Complete

## Stage 3: Verify and Publish
**Goal**: Run focused backend tests, Bandit on touched backend code, diff checks, then push/reply to the relevant PR threads.
**Success Criteria**: Local verification passes and the addressed PR #1104 review threads have replies.
**Tests**: Focused pytest, Ruff PLE1205, Bandit, `git diff --check`.
**Status**: In Progress

### Verification
- Red: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_service_authorization.py -q` failed with all three `revoked_at`-only cases not raising.
- Green: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_service_authorization.py -q` passed with 3 tests.
- Regression: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/PrototypeWorkspaces -q` passed with 59 tests.
- Ruff: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check --select PLE1205 tldw_Server_API/app/main.py` dropped from 27 to 26 existing findings after removing the reviewed prototype-workspaces warning; remaining findings are unrelated pre-existing logger calls.
- Bandit: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Prototype_Workspaces/service.py tldw_Server_API/app/main.py -f json -o /tmp/bandit_pr1104_revoked_at_slice.json` reported no results and no errors.
- Diff: `git diff --check` passed.
