# Killable Moderation Release Evidence

Plan identity: 2026-11-02. Actual execution: **2026-11-03**.
TASK-15417 remains In Progress. This is Task5 release evidence, not independent
task/whole-branch approval, a Windows guarantee, or permission to publish.

## Revision And Scope

Worktree: `/Users/appledev/Documents/GitHub/tldw_server/.worktrees/moderation-policy-types-retirement`.
Branch: `codex/moderation-ci-triage-regex-design`.
Verified starting HEAD: `020a416f634bb0e8af26e195eaf267283bc7f1c6`.
Cumulative touched-scope inventory base: `c5561bf4d3`.
Tasks1-4 were independently approved before this work. No runtime algorithm,
deadline, safety ceiling, wire version or caller contract was changed in Task5.

The release fix is one setuptools package-data entry for the existing Moderation
README. New installed-wheel tests exercise the actual worker/controller graph
outside the checkout, flag preservation, forbidden imports, safe diagnostics and
missing interpreter/worker fail-closed behavior. A compatibility characterization
documents why neutral writes require compatible rollback readers. Browser smoke
adds decision-history/undo and safe operational detail assertions to the existing
neutral incident flow. The module README now describes exact approved defaults,
platform limits, process-count multiplication, degraded reporting and rollout.

## Environment And Boundaries

- Native macOS arm64, venv Python3.14.3, pytest9.0.2/pytest-asyncio1.3.0.
  Every host Python command activates
  `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate` first.
- Existing build1.4.0/setuptools81.0.0/wheel0.47.0; no host dependency changes.
  Wheel builds use tracked current package sources in disposable directories;
  no local untracked config, database or secret files are copied.
- Bun and installed local Vitest/Playwright replace the plan's pnpm examples,
  per the recorded environment ruling. No dependency/lockfile changes.
- Offline cached `tldw-server:prod`, entrypoint overridden, Python3.12.13.
  Current Python-only snapshot: `/tmp/moderation-15417-linux-root`; all3542 copied
  repository Python files byte-checked against starting HEAD. Additional source
  supports real incident/migration tests; worker launch remains the minimal graph.
  Worker SHA256: `38c3c2ff7f88ce509ed3e35bfc13ed8dec75ad4b49fa348f00510adc76d3eca8`.
- Docker: network none, read-only root, cap-drop ALL, no-new-privileges, nonroot
  65534:65534, 1 CPU, 1 GiB memory, 64 PIDs, disposable 96 MiB noexec/nosuid /tmp.
  Only source/wheel bind mounts are read-only. No production app startup.
- Linux installs the already prepared pure pytest9.0.2/asyncio1.3.0/timeout2.4.0
  wheels offline into disposable `/tmp/pytest-target`, including conditional
  typing-extensions4.15.0. Plugin autoload disabled; asyncio and timeout explicit.
  This is not the host plugin environment or the full Python3.12 application suite.
- No Windows runtime. Windows source compilation is **not** an OS pass.
  macOS has CPU/no-core backstops, not a reliable RLIMIT_AS guarantee. Linux
  requires verified 512 MiB address-space/3-second CPU/core0 limits before decode.

## TDD And Fault Evidence

All packaging commands: `python -m pytest tldw_Server_API/tests/unit/test_moderation_execution_packaging.py -q`.

| Run / log under /tmp | Actual result |
| --- | --- |
| `moderation-task5-packaging-initial.log` | exit1; 3 failed, 3 passed, 24 warnings, 17.03s; test incorrectly referenced nonexistent `matched` field; not a runtime RED |
| `moderation-task5-packaging-red.log` | exit1; 1 failed, 6 passed, 26 warnings, 15.28s; installed README genuinely absent |
| `moderation-task5-packaging-green.log` | exit0; 7 passed, 26 warnings, 14.11s; one package-data entry fixes the missing artifact |
| `moderation-task5-packaging-final.log` | exit0; 7 passed, 26 warnings, 14.97s; resolved absolute git inventory executable |
| `moderation-task5-packaging-compatibility.log` | exit0; 8 passed, 28 warnings, 13.00s; added compatible-reader restart/undo characterization |

The runtime fault/parity tests were characterization of already implemented
behavior, not invented failures or unavailable-OS RED. Missing interpreter
returns `worker_unavailable`; missing sibling worker blocks with `worker_crashed`.
Actual child import guards reject `regex`, FastAPI, SQLite, Loguru, Pydantic and
application/provider/config/DB initialization. Worker stderr is empty, normal
`re` flags are preserved, `re.DEBUG` cannot contaminate the protocol, and the
controller factory checks payload absence from worker argv/environment.
Installed targets are real no-deps/no-index pip installations, not zip extraction
or editable imports. These benign tests do not execute hostile expressions.

Wheel fixture's actual commands (sys.executable is the activated venv Python):

```sh
python -m build --wheel --no-isolation --outdir <temporary-dist>
python -m pip install --no-index --no-deps --target <temporary-installed> <built-wheel>
python -I -S -c <installed-worker-or-controller-probe>
```

For visible build diagnostics, the tested tracked-source staging directory was
also built with the same backend into `/tmp/moderation-task5-release-dist`;
`/tmp/moderation-task5-wheel-build.log` retains full output.
Build exit0, no build Warning/error diagnostics; artifact
`tldw_server-0.1.46-py3-none-any.whl`, SHA256
`857da9d86bbf0501e09c03df9b4ea64a089806b2b5ee6a3630a27208a9500cca`.
Archive hashes can vary across builds because wheel timestamps are not fixed.

## Backend Manifest

Fresh precommit full focused run: **2048 passed, 2 skipped, 5883 warnings,
261.25s, exit0**, `/tmp/moderation-task5-full-focused.log`. After the rollback
characterization was added, the same full manifest at committed revision
`b39236218a2c908772332cf98982ebad0a5f9fca` passed **2049 tests, 2 preexisting
skips, 5885 warnings, 221.14s, exit0**; output
`/tmp/moderation-task5-postcommit-focused.log`. The subsequent evidence commit
changes only this document and task notes, not tested source/UI/package metadata.
Repeated runs are not unique totals.

Exact focused command, from the worktree after venv activation:

```sh
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_*.py \
  tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py \
  tldw_Server_API/tests/Chat_NEW/integration/test_moderation_categories.py \
  tldw_Server_API/tests/Chat/unit/test_streaming_utils.py \
  tldw_Server_API/tests/Chat/unit/test_streaming_moderation_execution_failure.py \
  tldw_Server_API/tests/Chat/unit/test_streaming_generation_settlement.py \
  tldw_Server_API/tests/Chat/unit/test_streaming_structured_events.py \
  tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py \
  tldw_Server_API/tests/Audio/test_stt_policy_payloads.py \
  tldw_Server_API/tests/Audio/test_audio_transcription_retention_and_redaction.py \
  tldw_Server_API/tests/Audio/test_realtime_session.py \
  tldw_Server_API/tests/Audio/test_ws_vad_turn_detection.py \
  tldw_Server_API/tests/Audio/test_ws_transcribe_partial_persistence.py \
  tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py \
  tldw_Server_API/tests/Guardian \
  tldw_Server_API/tests/Workflows/adapters/test_workflow_adapters_user_scope.py \
  tldw_Server_API/tests/Workflows/adapters/test_llm_adapters.py \
  tldw_Server_API/tests/Workflows/adapters/test_control_adapters.py \
  tldw_Server_API/tests/Workflows/adapters/test_control_fanout_cancellation.py \
  tldw_Server_API/tests/Workflows/adapters/test_adapters_llm_quota_refusal.py \
  tldw_Server_API/tests/Workflows/test_engine*.py \
  tldw_Server_API/tests/Workflows/test_workflow_attempt_failures.py \
  tldw_Server_API/tests/Workflows/test_workflows_map_substeps_unit.py \
  tldw_Server_API/tests/Workflows/test_workflows_map_substeps_integration.py \
  tldw_Server_API/tests/Services/test_lifecycle_worker_catalog.py \
  tldw_Server_API/tests/Services/test_lifecycle_worker_specs.py \
  tldw_Server_API/tests/Services/test_lifecycle_worker_engine.py \
  tldw_Server_API/tests/Services/test_lifecycle_worker_session.py \
  tldw_Server_API/tests/Services/test_lifecycle_workers.py \
  tldw_Server_API/tests/Services/test_startup_worker_groups.py \
  tldw_Server_API/tests/Services/test_startup_worker_bootstrap.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_moderation_permissions_claims.py \
  -q > /tmp/moderation-task5-full-focused.log 2>&1
```

The unchanged skips are `test_streaming_utils.py::test_heartbeat_integration`
(existing complex async coordination) and `test_ws_vad_turn_detection.py::test_silero_turn_detector_real_vad_end_to_end`
(optional Silero unavailable). Host warnings include existing Starlette/httpx
deprecation, unknown pytest `plugins` option, Pydantic config/schema warnings,
and Python3.14 asyncio event-loop-policy deprecation. No warning suppression was
added. Earlier Task4 baseline reproduces pending async-generator teardown logs;
the Task5 focused log has no pending-task diagnostic. Whole-tree collection was
not repeated: known missing `httpx2` in Workspaces blocks approximately85k items.
Focused green does not claim whole-project green.

## Containment And Linux

Native benign prerequisite command, BEFORE measurements/hostile focused probes:

```sh
python -m pytest tldw_Server_API/tests/unit/test_moderation_process_harness.py -k 'outer_harness_timeout or probe_hard_exit' -q
```

Exit0, 2 passed/13 deselected/16 warnings, 2.55s;
`/tmp/moderation-task5-preflight.log`. A separate pytest temp-directory removal
warning was retained. Independent native process-tree cleanup, not pytest's
thread timeout, owns hostility. Linux controller first proves timeout, hard-exit,
invoker-loss and retained cleanup ownership, then runs eight actual runner probes:
fork during spawn, pathological expression, heartbeat, cancellation during spawn,
fork, nonzero child, invalid resource posture, blocked pipe. All exit0; Linux
posture reports address_space/cpu/core true. No Windows claim.

```sh
docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --cpus 1 --memory 1g --pids-limit 64 --user 65534:65534 --workdir /tmp --tmpfs /tmp:rw,noexec,nosuid,size=96m --mount type=bind,source=/tmp/moderation-15417-linux-root,target=/probe,readonly --entrypoint python tldw-server:prod -I -S /probe/tldw_Server_API/tests/helpers/moderation_posix_validation.py
docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --cpus 1 --memory 1g --pids-limit 64 --user 65534:65534 --workdir /tmp --tmpfs /tmp:rw,noexec,nosuid,size=96m --mount type=bind,source=/tmp/moderation-15417-linux-root,target=/probe,readonly --mount type=bind,source=/tmp/moderation-15417-py312-test-wheels,target=/wheels,readonly --entrypoint /bin/sh tldw-server:prod /probe/run-release-tests.sh
```

Logs: `/tmp/moderation-task5-linux-preflight.log` and
`/tmp/moderation-task5-linux-pytest.log`, both exit0. Offline script installs the
three exact pytest wheels with `--no-index --find-links=/wheels --target=/tmp/pytest-target`,
sets `PYTHONPATH=/tmp/pytest-target:/probe`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`,
`TESTING=1`, a disposable review DB, then runs:

```sh
python -m pytest -p pytest_asyncio.plugin -p pytest_timeout /probe/release-tests --confcutdir=/probe/release-tests -q -o asyncio_mode=auto -o asyncio_default_fixture_loop_scope=function -o cache_dir=/tmp/pytest-cache
```

Result: **276 selected passing nodes, 15 warnings, 13.52s**, not a claim of276
unique tests or extra disjoint coverage. The selection contains copied protocol,
worker, incidents, migration, review store/audit suites; their source/unit store
imports also appear in warnings, so imported/duplicate collection must not be
counted as new unique evidence. It exercises actual SQLite migration/reporting,
undo/history and reserved neutral response phases, including required Linux
limit-install failure denial. No missing-dependency failure occurred. Warnings
are unregistered `unit` markers in the deliberately minimal pytest bootstrap;
repository-wide fixtures/settings were not loaded or represented as validated.

## UI Evidence

From `apps/packages/ui`, existing installed local runner:

```sh
/Users/appledev/.bun/bin/bun run test --run src/components/Option/ModerationReview/__tests__ src/services/__tests__/moderation.service.contract.test.ts
```

Exit0, 7 files/37 tests, 4.18s; `/tmp/moderation-task5-ui-contracts.log`.
The initial component-only command passed6 files/32 tests in4.01s; not additive.
Existing Node `--localstorage-file` warnings retained.

From `apps/tldw-frontend`:

```sh
NODE_OPTIONS=--max-old-space-size=8192 /Users/appledev/.bun/bin/bun run typecheck
/Users/appledev/.bun/bin/bun x --no-install eslint e2e/workflows/tier-5-specialized/moderation-review.spec.ts
PATH=/Users/appledev/.bun/bin:$PATH NEXT_PUBLIC_API_URL=http://127.0.0.1:18086 TLDW_WEB_URL=http://localhost:18085 TLDW_WEB_CMD='bun run dev -- -H 127.0.0.1 -p 18085' /Users/appledev/.bun/bin/bun x --no-install playwright test e2e/workflows/tier-5-specialized/moderation-review.spec.ts --project=tier-5 --workers=1
```

Typecheck explicitly exited0, not inferred from quiet output. The package script
also sets the8GiB heap; initial plain `bun run typecheck` likewise exited0.
ESLint exit0 with the existing frontend/shared UI config. Browser:3 passed,
40.2s, exit0, `/tmp/moderation-task5-playwright-loopback.log`. Initial default
bind attempt failed EPERM and ran no browser tests; its log remains
`/tmp/moderation-task5-playwright.log`. Approved retry bound only127.0.0.1 on a
verified-free18085. Playwright owns/stops its server; no user backend/data.
No listener remains. Mocked browser evidence is not real backend persistence.

One neutral row retains identity through single approve, undo/history, bulk
dismiss and refresh. Reason, operation, correlation ID, safe excerpt and neutral
labels remain visible; no Matches field; no viewport overflow at390/1024/1280/1920.
Fresh `/tmp/task3-incident-{desktop,mobile,tablet,wide}.png` screenshots retain
the existing test filenames; desktop/mobile were visually inspected.

## Actual Measurements

Native Python3.14, approved bounds unchanged, 16 real stdlib patterns, default
evaluator scan/count settings. Full `ModerationService.evaluate_text` facade
performs decision AND separate redaction child batches, not eager decision-only
or worker-only timing. Every sample asserts action redact, redacted output and
no residual matched `secret`. Long inputs:800000 ASCII/400000 Unicode characters;
policy text is benign research text with one redactable token. Unicode uses
accented Latin and Japanese, not empty-policy no-ops. Five short/three long
samples per warm case, one cold sample. Timings are empirical, not an SLA.

Disposable helper: `/tmp/moderation-task5-measure.py`; invoked by
`run_isolated_probe([sys.executable, helper, mode])` with the proved10-second
independent controller and2-second tree cleanup. Modes: latency, stream,
saturation; logs `/tmp/moderation-task5-measurements2.{jsonl,stderr}`.

| Case | First run median / max ms | Quieter recheck median / max ms |
| --- | --- | --- |
| Cold short ASCII,22 chars | 2063.809 /2063.809 | 599.106 /599.106 |
| Warm short ASCII,22 chars | 323.963 /491.875 | 80.210 /100.732 |
| Warm short Unicode,15 chars | 159.480 /174.948 | 103.689 /105.010 |
| Long ASCII,800000 chars | 359.375 /373.617 | 275.921 /277.246 |
| Long Unicode,400000 chars | 364.201 /364.514 | 253.244 /260.570 |

First run overlapped other release checks. The2063.809ms cold WALL sample
returned successfully, exceeds2s wall time, and remains a release concern;
that sample did not instrument scope entry, so its exact scoped duration cannot
be claimed. Source inspection shows lazy `execution_incidents` import before
`execution_scope` entry. A separate fresh-process diagnostic measured519.450ms
wall =440.637ms pre-scope import/setup +78.787ms shared scoped decision/redaction
(evaluate40.541ms, redact38.201ms), with the same correlation/deadline and exact
`public research [REDACTED]` result. Log:
`/tmp/moderation-task5-cold-diagnostic.jsonl`. The quieter uninstrumented recheck
is `/tmp/moderation-task5-cold-recheck.jsonl`. These do not erase the first sample
or prove its split. No actual scoped legitimate deadline failure was observed,
no limit was tuned,
and no Task4 runtime fix or design amendment was implemented or is needed from
the observed79ms scoped diagnostic. The first wall-time sample is retained for
clarity, not presented as a request to widen approved bounds.

### Deadline Boundary Ruling

Task5 fix round1/3 records the parent's interpretation of the binding service
operation wording (focused spec lines125-129/151-154 and service boundary654):
the two-second deadline begins at execution-scope entry, before admission and
serialization, and is shared across child startup and decision/redaction batches.
It is not an end-to-end HTTP/public-call wall-clock guarantee including first
trusted module initialization, failure reporting or kernel cleanup. API lifecycle
preparation loads/prepares the incident reporter before traffic. The operator
README table and adjacent guidance now name this boundary explicitly.

The 2063.809 ms cold sample remains unpartitioned; the later 78.787 ms scoped
diagnostic is a separate call and cannot explain the earlier sample. Neither
justifies widening the approved deadline or limits. This is documentation of the
parent ruling, not a runtime/spec/ADR/cap change. Cost if the interpretation is
wrong: stricter direct-embedding latency requirements need separate review; the
parent's final whole-branch review will judge that interpretation.

Actual `StreamingResponseHandler.safe_stream_generator`, ten24-character chunks,
three samples: unchecked median0.337ms, checked median832.436ms, approximately
832.099ms total added overhead. Checked min/max811.561/1198.773ms. Each transform
uses the real async wrapper and service redaction; terminal `[DONE]` is asserted.
This measures per-chunk moderation overhead, not a whole-stream2s deadline.

Four real owned benign sleeping children saturate the runner: fifth admission
refused in0.119ms; cancellation, kill/reap and closed ownership for all four in
6.516ms; next real-worker request succeeds. This controlled waiting workload is
admission/cancellation evidence, not a regex throughput claim. All three modes
exit0 with empty stderr. No inline fallback, timeout widening or cap reduction.

The FIRST measurement invocation failed a helper assertion: it expected redacted
text to start with `[REDACTED]`, although short samples put `secret` mid-string.
`/tmp/moderation-task5-measurements.{jsonl,stderr}` retains exit1/traceback. Corrected
helper asserts containment of the replacement and absence of `secret`; production
code was untouched. It is not hidden as a runtime deadline failure or counted as
a passing benchmark.

## Static, Security, CI And Metadata

New Task5 files were staged before cumulative inventory:68 Python paths
(34 production/34 test). Exact commands, from worktree with venv active:

```sh
git diff --name-only --diff-filter=ACMR c5561bf4d3 -- tldw_Server_API | rg '\.py$' | xargs python -m py_compile
git diff --name-only --diff-filter=ACMR c5561bf4d3 -- tldw_Server_API | rg '\.py$' | xargs python -m ruff check
git diff --name-only --diff-filter=ACMR c5561bf4d3 -- tldw_Server_API/app | rg '\.py$' | xargs python -m bandit -r -f json -o /tmp/bandit_moderation_killable_re.json
git diff --name-only --diff-filter=ACMR c5561bf4d3 -- tldw_Server_API/tests | rg '\.py$' | xargs python -m bandit -r -s B101 -f json -o /tmp/bandit_moderation_killable_re_tests.json
PYTHONPATH=tools/backlog-py/src python -m backlog_py task normalize --check
PYTHONPATH=tools/backlog-py/src python -m backlog_py task normalize --check 'backlog/tasks/task-15417 - Implement-killable-stdlib-re-execution-for-Moderation.md'
python Helper_Scripts/ci/check_backlog_task_ids.py .
PYTHONPATH=tools/backlog-py/src python -m pytest tldw_Server_API/tests/CI/test_backlog_task_format_ratchet.py -q
python Helper_Scripts/ci/check_shard_coverage.py
python Helper_Scripts/checks/guard_no_nonempty_legacy_complete.py
python Helper_Scripts/checks/guard_http_client_patching.py
git diff --check
git diff --cached --check
```

- Compile exit0; Python3.14 emits the Task5-inherited, whole-branch-added `async_execution.py:102`
  SyntaxWarning (`return` in a `finally` block, the closed-loop offload-delivery
  branch). At that checkpoint there was no suppression or runtime edit; final-review M2
  subsequently removes the return and verifies closed-loop retirement. This was not
  preexisting BASE debt. Raw Ruff exit1/68 findings on cumulative68 paths; Task4's14
  findings used its narrower40-path scope. Exact path/code/message multiset
  comparison against both starting HEAD and c5561bf4d3 finds0 new findings
  (HEAD68/base89/current68). No blanket ignore or unrelated cleanup.
  `/tmp/moderation-task5-{inventory,ruff,ruff-comparison}.log`.
- Production Bandit exit0:0 findings/0 scan errors. Tests exit1:32 findings,
  0 scan errors after resolving the new partial-path B607 by resolving git.
  B4049/B60319 are subprocess heuristics: fixed executable vectors, no shell,
  trusted temporary/source paths and independently owned hostile probes.
  Task5 adds B4041/B6035 for git inventory, offline build/install and two benign
  installed probes. No input-controlled executable/code or shell interpolation.
  B1051/B1062 are preexisting dummy/token_type='access' fixtures. B6081 is the
  preexisting migration test's constant three-table tuple, no user-controlled SQL.
  Only test B101 excluded; no new suppression. Initial tests scan33 findings
  included the now-resolved B607. JSON reports were inspected individually.
- Shard coverage exit0:869 shards/5360 test files/4 ignored/44 baseline/
  new_uncovered0. Existing globs cover the new suite; no redundant CI workflow.
- Whole-tree normalize exit1 lists ten unrelated legacy task records;
  `/tmp/moderation-task5-backlog-normalize.log`. Scoped TASK-15417 normalization,
  task IDs, both repo guards, and both diff whitespace checks exit0. No unrelated
  records rewritten. Ratchet4 passed/20 warnings/1.28s, exit0.

## Handoff

Parent owns fresh Task5 review, whole-branch review/fixes, plan/ledger updates,
global completion and eventual cleanup. No reviewers/subagents dispatched; no
plan/SDD deletion, push, rebase, merge, publish, lock changes or hook bypass.
The ignored `task-5-report.md` records scoped commits, final revision/commands and
fresh postcommit results. Windows remains explicitly unverified. Cold wall-time
scope expectations and residual baseline gates require parent review, not an
unapproved runtime rewrite or a claim that every release gate is pristine.

## Final Review Fix Wave

TASK-15417, FIX_BASE `24b6e815076217abdeccbc9948992c75d851b06c`.
F2/F3/M1/M2 were corrected in `bf56d370ef16eaf1b63c5707e3decb2c610efae3`:
cancelled unconfirmed cleanup uses the shared independent incident slot without
masking cancellation or releasing quarantine; unavailable pre-traffic incident
preparation degrades reporting but retains API shutdown ownership; installed
ASCII checks now distinguish Unicode behavior; closed-loop offload delivery no
longer returns from finally. The earlier warning was added by this branch, not
preexisting BASE debt. The prior evidence above is retained as history.

Requester-approved F1 amendment and plan were recorded in `735fa65971`:
`Docs/superpowers/specs/2026-11-03-moderation-audio-batch-amendment.md` and
`Docs/superpowers/plans/2026-11-03-moderation-audio-batch-amendment-plan.md`.
The new private `redact_many` variant handles at most4096 independent fields in
one fresh child, reconstructing policy once. It retains aggregate1,000,000
input/output characters, all existing byte/rule ceilings, independent anchors,
per-field evaluator limits/literal replacements, shared2s, four-worker admission,
and all-or-nothing mapping. Trusted single-field overrides remain dispatched.

The original warmed100-segment/5099-character failure (~50 execute attempts)
remains documented in final-review.md. A fresh actual payload test with11 built-in
PII rules now reports43.714ms, one actual returned Popen child and one execute
attempt, exact result/segment metadata preservation and zero retained ownership.
This is not a universal wall-time SLA or an explanation of the older cold sample.

- Final host focused manifest:917 passed/1873 warnings,67.81s, exit0;
  `/tmp/moderation-f1-final-green.log`. Covers protocol/worker/runner/reporting,
  service/overrides, installed-wheel flags/imports, Audio HTTP/WS/retention,
  streaming/realtime and lifecycle. Initial seven obsolete Mock injection
  failures and their stronger real-service override correction are retained in
  `/tmp/moderation-f1-focused.log` and the local final-fix report.
  Final self-review also added a RED/GREEN empty-field/dual-key regression to
  preserve Audio's no-op guards before trusted override dispatch.
- Fresh native containment15 passed BEFORE hostile probes. Current Linux3.12.13
  source snapshot independently proves containment, failed-cleanup ownership,
  one-launch independent-field batching, and existing runner probes. All required
  Linux resource posture flags are true. Final batch probe102 fields/one launch
  took118.466ms. `/tmp/moderation-f1-linux-preflight-final.log`.
- SHA256 bytecheck covers9096 Python source files and6 selected copied suites,
  zero mismatches after all Python changes;
  `/tmp/moderation-f1-linux-bytecheck-complete.log`. The initial racing copy
  check correctly failed and is not used as proof. Current selected Linux
  protocol/worker/incident/migration/review suite:329 selected nodes/15 warnings,
  10.91s, exit0, `/tmp/moderation-f1-linux-pytest-final.log`. Its consumed modules
  are unchanged by the last Audio-only no-op correction. This remains source-suite
  evidence, not an installed Linux wheel or full-application claim.
- Whole-wave24 touched Python paths compile with SyntaxWarning as error.
  Production Bandit0 findings/0errors. Raw Ruff5 findings, exact same-path
  FIX_BASE baseline5, zero new. Test Bandit with only B101 excluded reports the
  same19 non-assert findings as the pre-F1 scope: one dummy token_type fixture,
  four subprocess imports, fourteen fixed-argv/no-shell subprocess heuristics.
  `/tmp/moderation-final-wave-static-comparison.log` and the corresponding
  `/tmp/bandit_moderation_final_wave_{prod,tests}.json`; no new suppression.

Full RED/GREEN commands, failure provenance, actual interfaces, final commit and
platform limits are in the original local `final-fix-report.md`. Parent owns one
scoped independent re-review of all findings from FIX_BASE, plan statuses and
global completion. Windows, whole-application collection, current-dev integration
and eventual required CI/human Change summary gates remain unverified or pending.
