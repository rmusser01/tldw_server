## Stage 1: Verify Review Finding
**Goal**: Confirm PR #1104 releases a token claim after a collaborator has already been provisioned.
**Success Criteria**: A focused prototype link exchange test fails when post-provision audit failure decrements a single-use token.
**Tests**: Focused pytest for prototype public link exchange claim accounting.
**Status**: Complete

## Stage 2: Implement Provisioning-Aware Release
**Goal**: Release claimed token uses only before provisioning succeeds; retain the consumed claim once a shared actor/session exists.
**Success Criteria**: Pre-provision failures still release the claim, while post-provision failures do not allow minting another actor through a retry.
**Tests**: Focused prototype link exchange tests.
**Status**: Complete

## Stage 3: Verify and Publish
**Goal**: Run focused backend tests, Bandit on touched backend code, diff checks, then push/reply to the relevant PR threads.
**Success Criteria**: Local verification passes and the addressed PR #1104 review threads have replies.
**Tests**: Focused pytest, Bandit, `git diff --check`.
**Status**: In Progress

### Verification
- Red: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_link_exchange.py::test_public_prototype_exchange_retains_claim_on_post_exchange_failure -q` failed because `use_count` was released back to 0 after provisioning.
- Green: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_link_exchange.py::test_public_prototype_exchange_releases_claim_on_unexpected_error tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_link_exchange.py::test_public_prototype_exchange_retains_claim_on_post_exchange_failure -q` passed with 2 tests.
- Regression: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_link_exchange.py -q` passed with 20 tests.
- Bandit: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/sharing.py -f json -o /tmp/bandit_pr1104_token_claim_release.json` reported no results and no errors.
- Diff: `git diff --check` passed.
