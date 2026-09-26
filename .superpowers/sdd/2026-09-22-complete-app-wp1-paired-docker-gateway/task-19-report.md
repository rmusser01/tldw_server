# Task 19: Actual installer readiness integration correction

Status: DONE for the scoped implementation and bounded proof; parent review and
fresh exact-artifact qualification remain pending. Associated task: TASK-13343.
Base was clean `47ec278c14dca875158aa7258d460289999c4ea6`.
Implementation HEAD/commit: `9e4f8fc620742870c47dafcfa34e89afa047820f`,
`fix(release): probe authenticated operator readiness (TASK-13343)`.
This report is committed separately with the same task association.

## Change and scope

The installer no longer asks a bridge-connected gateway to reach the backend's
loopback-only `/internal/ready`. After fresh bootstrap and cookie-only profile
success, the maintained probe requests authenticated `/api/v1/health/ready` and
requires HTTP 200 plus a JSON object whose `status` equals `ready`. Operator
details are consumed privately and discarded. The existing finally block, exact
cookie/CSRF DELETE, replay refusal, per-request/body/whole-probe bounds, and
separate ten-second cleanup window are unchanged.

Changed production/test files:

- `tldw_Server_API/scripts/app_bundle_readiness.py`
- `tldw_Server_API/tests/Release/test_app_bundle_readiness.py`

The real-socket fixture now returns 404 for `/internal/ready` and guards operator
readiness with the exact fresh session. Coverage includes absent/foreign cookies,
operator 401/403, 503/not_ready, HTTP 200 with not_ready, malformed/non-object/
missing-status JSON, oversized/truncated/stalled readiness bodies, and exact
session revocation/replay refusal after every post-bootstrap failure. Existing
bootstrap/auth/revocation/body/header-deadline regressions still run.

No backend guard, gateway, helper, dependency, image policy, build layer, public
liveness, release gate, or acceptance change was made. No subagents were used.

## TDD evidence

Initial sandbox execution could not bind the real HTTP fixture (`PermissionError`)
and was not treated as behavioral RED. With local socket access, before the
production edit:

```sh
source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Release/test_app_bundle_readiness.py
```

Exit 1: **15 failed, 22 passed, 4 warnings in 11.82s**. The success case raised
ReadinessError after the initial `/internal/ready` 404; the ten operator failure
cases and four existing post-bootstrap failures observed no revocation because
the old probe stopped before creating a session. This reproduced the actual
integration defect rather than accepting a permissive internal-route fixture.

After the minimal production change, initial readiness/control GREEN was 52
passed. After the fixture lint correction, final covering verification was:

```sh
source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Release/test_app_bundle_readiness.py tldw_Server_API/tests/Release/test_app_bundle_control.py --basetemp=/private/tmp/task19-pytest > /private/tmp/task19-tests.log 2>&1
```

Exit 0: **52 passed, 4 warnings in 12.19s**. No failures or errors. The test runner
suppresses warning details in its configured summary; no warning suppression or
dependency change was added. Control tests required no fixture edits.

## Format, lint, security and self-review

```sh
source .venv/bin/activate && python -m black --check tldw_Server_API/scripts/app_bundle_readiness.py tldw_Server_API/tests/Release/test_app_bundle_readiness.py && python -m ruff check tldw_Server_API/scripts/app_bundle_readiness.py tldw_Server_API/tests/Release/test_app_bundle_readiness.py && python -m bandit -r tldw_Server_API/scripts/app_bundle_readiness.py -f json -o /private/tmp/bandit_task19_readiness.json
git diff --check
```

Both commands exit 0. Black leaves both files unchanged; Ruff reports all checks
passed; production Bandit has zero findings/errors (existing private-container
listener B104 exclusion retained). One fixture SIM114 finding during iteration
was corrected before final verification. Self-review checked the full scoped diff,
ordering, strict readiness result, cookie-only auth, unchanged cleanup and limits;
no residual findings. Commit ran normal hooks, without bypass flags. Git emitted
existing garbage-collection/unreachable-object warnings; no repository pruning or
maintenance was attempted.

## Bounded actual cached proof

```sh
source .venv/bin/activate && PYTHONPATH=$PWD python /private/tmp/task19-cached-proof/run.py > /private/tmp/task19-cached-proof/driver.log 2>&1
```

Exit 0, one bounded proof attempt. Reused the prior verified diagnostic bundle,
instance and exact cached bd0de933 images with Compose `--no-build --pull never`.
No registry, build, push, public release or full candidate run. The cached control
image's source label and digest were checked. Fresh actual app/WebUI/gateway and
network inspection passed production `validate_runtime`; verified instance and
signed bundle identity were rechecked. The changed module was mounted read-only
into the cached control image on the verified private network. Its maintained
`probe_gateway` function ran directly; observation wrapped requests only to record
status/path booleans, without replacing request or readiness logic.

Observed full maintained probe success: WebUI page/asset 200, anonymous profile
401, fresh bootstrap 200, cookie-only profile 200, cookie-only operator readiness
200 with accepted JSON status ready, exact-session DELETE 200, revoked profile
401. An additional request with that same revoked cookie to operator readiness
returned 401. The prior provided Task13 actual proof already established bridge
internal404 and anonymous operator401; this run did not duplicate that workflow.

All eleven allowlisted proof/ownership booleans are true in private
`/private/tmp/task19-cached-proof/summary.json`. The captured control container ID
was removed; the verified diagnostic project containers, network and volumes were
removed and absence checked. Both unrelated PostgreSQL container IDs/running
states were checked before/after and remained unchanged. No unknown resources were
cleaned or pruned. Raw inspection, private configuration, cookies, Env, operator
response details and Docker output were never printed; private scripts/logs are
0600 under a 0700 directory. Cookies remained in the disposable control process.

Controlled actual operator not_ready injection was not attempted: no existing
external readiness toggle was identified that preserves auth/session cleanup
without extending the scoped proof or changing source guards. Real HTTP fixture
503/not_ready and strict-payload refusal prove failure cleanup; no claim is made
that an actual cached not_ready scenario was exercised.

## Remaining ownership and limitations

Stages 1 and 2 of the brief are complete. Stage 3 is parent-owned: scoped review,
fresh clean local and both native Linux architecture bundles, 13 lifecycle /
38 browser checks, signatures and independent artifact verification remain
pending. Cached proof qualifies only this integration correction, not a freshly
built final artifact. No G2/G4 acceptance is claimed. Windows runtime remains
unqualified and G12 false. Backlog, tracked design/plan updates and acceptance are
owned by the parent.
