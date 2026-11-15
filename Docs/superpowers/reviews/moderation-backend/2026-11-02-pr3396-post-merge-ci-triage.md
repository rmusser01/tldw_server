# PR 3396 Post-Merge CI Triage

Task: TASK-15416 (renumbered from TASK-15413 after an open-PR ID collision)
Date: 2026-11-02
Scope: diagnostics and follow-up routing only; no runtime, test, or CI edits.

## Revisions And Conclusion

- PR: https://github.com/rmusser01/tldw_server/pull/3396
- Base: `54b321dcec8cdbb6358581f42c1548696260570b`.
- Tested head: `a423dcda4d08b9b5cdf2ac0104821ca8479bf9a5`.
- Merge: `61d02ed484a3c80f5423676b959cf14e7b572a62`.
- Current dev inspected: `3ab1ce5d3a`.

The failures are real, but merge does not imply the legacy full suite or UX
Smoke passed. The required-check contract in `Docs/Development/CI_REQUIRED_GATES.md`
does not include these legacy checks. Their red results are not evidence of a
required-gate bypass.

The collected Python summaries contain 161 failed test nodes in 27 shards;
none has a Moderation-named test path or node. The PR changed only five unused
service delegates, related moderation tests, and documentation/tracking. The
involved frontend, workflows, egress, ingestion, and provider-inventory paths
have no base-to-head diff. This rules out direct edits to those paths, not all
possible indirect regressions. Do not label every failure pre-existing solely
from unchanged files or test names.

Two provider-inventory failures have a confirmed environment cause. The GitHub
fixture failure family and the matching UX test-precondition repair already
have corrections on current dev. Other contract failures remain routed for
independent CI maintenance; this investigation does not claim they are fixed.
No collected evidence currently justifies a further Moderation runtime patch.

## Confirmed Provider-Inventory Cause

The full-suite workflow exports:

```text
WORKFLOWS_EGRESS_ALLOWLIST=93.184.216.34,does-not-resolve.invalid,example.com
```

The two `False` cases of
`test_byok_private_gateway_is_refused_exactly_when_generation_refuses` use a
test-owned localhost gateway and explicitly disable private/port blocking.
Their fixture does not own this inherited allowlist input. Discovery refuses
the endpoint while base-URL validation allows it, so the intended parity
assertion fails. The CI log reports `assert True is False`.

Controlled reproduction on current dev, using the project .venv and the same
two node IDs:

| Process environment | Result |
| --- | --- |
| CI allowlist above | 2 failed, 16 warnings, exit 1 |
| Empty `WORKFLOWS_EGRESS_ALLOWLIST` and `EGRESS_ALLOWLIST` | 2 passed, 16 warnings, exit 0 |

No source changes occurred between runs. Only test-owned localhost HTTP servers
were used. This confirms the environment dependency; it does not authorize
broadening production egress policy or removing security assertions.

Follow-up: give this fixture explicit ownership of its policy inputs, following
the existing GitHub adapter fixture correction, and retain all private-address,
port, provenance, capability-policy, and request-hit assertions.

Evidence:

- https://github.com/rmusser01/tldw_server/actions/runs/41333175108/job/124936805980
- `/tmp/moderation-3396-egress-repro-ci-env.log`
- `/tmp/moderation-3396-egress-repro-isolated-env.log`

## GitHub Adapter: Existing Fix Verified

The `gap-verified-4` shard reports 11 failures in
`tests/Ingestion_Sources/test_git_repository_adapter.py`, dominated by
`GitHubAPIError: GitHub API request failed: Host not in allowlist` and consequent
status/message assertion mismatches. The same inherited workflow allowlist
omits `api.github.com`.

Current dev includes commit `21dee5349e`, which gives that test module an
autouse fixture that clears inherited allow/deny lists and explicitly allows
`api.github.com` before individual tests install deny overrides. It preserves
the redirect, metadata-address, status, and size-cap checks.

The complete module passed on current dev with the CI allowlist inherited:
24 passed, 60 warnings, exit 0. Do not duplicate that already-landed fix.

Evidence:

- https://github.com/rmusser01/tldw_server/actions/runs/41333175108/job/124936787931
- `git show 21dee5349e -- tldw_Server_API/tests/Ingestion_Sources/test_git_repository_adapter.py`
- `/tmp/moderation-3396-github-fixture-current-dev.log`

## UX Smoke: Matching Repair Already Landed

The failing case is `captures streaming stop and the selected-history
regeneration gate through the real cockpit`, in
`apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`.
All three attempts fail to find the exact `Review conversation history` button
within 20 seconds. The pending selection-response error follows that missing
button; regeneration assertions were not reached.

At the tested head, the second turn is sent only in the Stop-recovery branch.
If Stop leaves a saved assistant answer, the scenario can have only one ordinary
exchange. Production intentionally hides history review in that state, with
explicit presentation tests covering one versus two exchanges. This is a
high-confidence explanation, but this job's actual message count has not been
independently confirmed from its trace/snapshot.

TASK-15410 tracks this same precondition defect on other baseline-equivalent
runs. Its correction makes the existing follow-up unconditional, without
changing production UI or weakening assertions/retries/timeouts. Commits
`f4d75bda1f` and `805de2df3d` are present in current dev; TASK-13001.2 records the
later persisted-two-question guard. TASK-15410 also records a successful UX CI
run on the corrected test. Its remaining administrative status is not modified
by this work.

Next verification, if needed: inspect this job's saved error context/trace,
then run only the single Chromium case on current dev in an isolated real-server
harness. No new full browser run was performed here.

Evidence:

- https://github.com/rmusser01/tldw_server/actions/runs/41333175056/job/124930602471
- `/tmp/moderation-3396-ux-failed.log`, failure block at lines 2233-2378.
- TASK-15410; TASK-13001.2; merged PRs #3319 and #2821.

## Complete Failed-Shard Inventory

All rows are from run `41333175108`, attempt 1, on the tested head above.
Counts are failed test nodes, not inferred root-cause counts.

| Shard | Job ID | Failed |
| --- | --- | ---: |
| gap-verified-4 | 124936787931 | 11 |
| gap-verified-5 | 124936787962 | 93 |
| admin-watchlists-webhooks | 124936791940 | 5 |
| admin-webhooks-canonical | 124936792224 | 1 |
| chacha-core-stores | 124936794288 | 1 |
| db-management-a-c | 124936794714 | 5 |
| db-management-d-l | 124936794775 | 1 |
| media-audio | 124936801022 | 1 |
| media-ingestion-modification | 124936802131 | 2 |
| chat-character-unit-persona | 124936803278 | 4 |
| chat-new-unit-a-l | 124936804616 | 1 |
| llm-adapters-chat-endpoint | 124936805011 | 1 |
| chatbooks-streaming | 124936805483 | 3 |
| llm-adapters-chat-errors-core | 124936805509 | 1 |
| llm-adapters-chat-errors-extra | 124936805595 | 1 |
| llm-adapters-orchestrator-core | 124936805819 | 1 |
| llm-adapters-orchestrator-extra | 124936805887 | 1 |
| llm-calls-core | 124936805980 | 2 |
| product-flashcards | 124936807693 | 2 |
| product-watchlists-pipeline | 124936808806 | 1 |
| product-workflows-adapters-core | 124936808890 | 5 |
| product-prompt-studio | 124936810128 | 1 |
| platform-mcp-core | 124936810491 | 2 |
| platform-services-startup | 124936810719 | 1 |
| platform-services-core | 124936811014 | 1 |
| integrations | 124936811927 | 3 |
| platform-acp | 124936812171 | 10 |
| Total | 27 shards | 161 |

The separate aggregate job `125036342726` fails because the shard result is
`failure`; it is a consequence, not a 28th independent Python test failure.

## Residual Work And Limits

- Slides accounts for 93 failures in `gap-verified-5`: 92 endpoint-denial
  signatures and one cancellation `TimeoutError`. Endpoint-not-allowed results
  mask downstream response/timeout/size expectations; the cancellation case
  also logs a denial while waiting for stream startup. Inherited egress inputs
  are a candidate common cause, not yet a controlled reproduction of every case.
- Streaming has three stale-looking `Cache-Control` expectations:
  `no-cache` versus `no-cache, no-transform`. Confirm the intended contract
  before changing either production or tests.
- Database/character fixtures report missing attributes such as
  `_verify_dictionary_schema_postgres` and `_owner_user_id`.
- ACP fakes report new keyword/signature mismatches and missing
  `_unregister_permission_session`. These require owner-specific contract
  review, not generic retries or timeouts.
- Admin webhook host-policy isolation also has an existing current-dev fix,
  `b2f24ab377`, under TASK-13001.2. That task records the complete 48-case shard
  passing under the CI allowlist; that separate run was not repeated here.
- Remaining policy-error distinctions, authentication/status expectations,
  startup-provider, MCP installation/discovery, OCR, inventory, claims, and
  Watchlists failures are inventoried but not all reproduced here.

Route remaining cross-domain work through the existing CI-stability workstream
(TASK-2397) and the owning modules, using the exact job/node evidence. This is
a routing recommendation, not a claim those owners accepted or fixed it.
Do not disable tests, relax security policy globally, increase timeouts, or
fold these fixes into the Moderation regex hardening PR.

## Verification And ADR Assessment

Read-only comparisons confirmed no relevant base-to-head frontend, workflow,
egress, ingestion-test, or provider-inventory changes. Reproduction used the
existing project environment; no dependencies were installed, CI reruns
dispatched, production endpoints called, or runtime/test source edited.

Bandit: not applicable to this documentation-only change. ADR required: no;
this report changes no durable architecture or policy. Backlog records are
maintained with `backlog-py` under ADR-059.

The next Moderation deliverable is a separately approved design for bounded
regex detection and redaction, including an explicit decision on what callers
receive when evaluation cannot safely complete. No such behavior decision or
implementation is implied by this CI triage report.
