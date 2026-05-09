## Stage 1: Verify Review Finding
**Goal**: Confirm PR #1104 blocks designated promoters at the promotion-review API layer.
**Success Criteria**: A focused endpoint test fails when a user listed in `designated_promoter_ids` reviews a promotion request.
**Tests**: Focused pytest for prototype promotion review endpoint.
**Status**: Complete

## Stage 2: Implement Service-Aligned Permission Check
**Goal**: Use the prototype service promoter policy for promotion-review authorization instead of owner-only endpoint logic.
**Success Criteria**: Workspace owners and designated promoters can review promotion requests; other users remain forbidden.
**Tests**: Focused prototype endpoint test.
**Status**: Complete

## Stage 3: Verify and Publish
**Goal**: Run focused backend tests, Bandit on touched backend code, diff checks, then push/reply to the relevant PR thread.
**Success Criteria**: Local verification passes and the addressed PR #1104 review thread has a reply.
**Tests**: Focused pytest, Bandit, `git diff --check`.
**Status**: In Progress

### Verification
- Red: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py::TestPrototypeWorkspaceEndpoints::test_designated_promoter_can_review_promotion_request -q` failed with HTTP 403 from the owner-only endpoint guard.
- Green: the same focused command passed with 1 test.
- Regression: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py -q` passed with 10 tests.
- Bandit: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py -f json -o /tmp/bandit_pr1104_designated_promoter_review.json` reported no results and no errors.
- Diff: `git diff --check` passed.
