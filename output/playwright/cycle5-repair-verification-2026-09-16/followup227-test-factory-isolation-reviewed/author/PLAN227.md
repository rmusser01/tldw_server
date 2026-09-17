# UAT227 — TASK13260.167

## Stage1: Causal scope
Status: Complete
Retain prior mixed-suite38FAIL/111PASS caused by unscoped endpoint FakeAPI assignment. Add direct helper isolation controls for construction/OpenAPI, normal and exceptional client exits, and nested test clients. No frozen225/226 edits.

## Stage2: Minimal helper repair
Status: Complete
Use an ordinary async FastAPI lifespan with pytest.MonkeyPatch.context to install fake only for the active TestClient lifetime. Fourteen local transport call sites and one imported route-order call all enter client contexts; the remaining local call only reads OpenAPI. No signature/caller rewrite or product changes.

## Stage3: Both orders and review
Status: In Progress
Run ordered combined PostgreSQL-required/SQLite cases in both module orders, no skips, scoped Ruff/Bandit/compile, exact hash/patch freeze for root/Sidebar. No runtime/git/task edits.

Author verification complete; Stage3 remains In Progress for independent review.
