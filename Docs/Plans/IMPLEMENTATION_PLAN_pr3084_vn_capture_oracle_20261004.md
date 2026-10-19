# PR3084 VN recipe capture oracle repair

Associated task: TASK-13260.278.18.83.45. Existing human instruction authorizes fixing demonstrated CI failures.

## Constraints
- Start from published b4831b5 on dev502da5bf; preserve all previous published changes and exclude local closures.
- NEW gap-verified-6 job111484892286/run37217761394 bound to pull_request/attempt1/b483/ci.yml failed17:12:00. Its log was read once in memory:1failed678passed21skipped4877warnings428.56s. Never reread.
- Preserve real endpoint, recipe, database/job behavior and202 assertion. No production/CI/source-environment change, larger timeout, weakened offload check, full heavy shard or native/UAT replay.
- At most3 distinct causal experiments for this separate issue. Ingestion remains2of3, old ingestion/native/UAT/build/model STOPs and pending API replacement approval intact.
- ADR required:no, test-only synchronization adds no durable architecture rule. ADR059 governs supported backlog-py task edits.

## Stage 1: Demonstrate the oracle defect
**Goal**: Separate request preparation latency from blocked-loop behavior.
**Success Criteria**: Unchanged request-start<1s oracle fails with the handler still offloaded and the loop available.
**Tests**: Experiment1 injects1.05s delay before ONLY the generation handler in its worker, via a restored in-memory FastAPI routing control. Real capture and all assertions remain unchanged. Outer600/per-case300 guards, existing proper Python3.12 stack; no log/proof file. First control missed FastAPI wrapper dispatch:1passed32warnings2.48s/4.251outer,controlcount0 and explicit harness natural1; not causal red. Distinct corrected attempt2 matches function=endpoint and delayed exactly one offloaded handler:1failed32warnings3.20s/4.800outer/natural1, elapsed1.137577083s. Two of3 attempts used; third reserved for inline-loop negative.
**Status**: Complete

## Stage 2: Verify causal synchronization
**Goal**: Check the loop can release recipe capture while capture is blocked.
**Success Criteria**: Loop callback releases the worker within the existing2s bound; a targeted inline endpoint control fails this check; delayed offloaded capture passes.
**Tests**: Slow recipe schedules release.set with loop.call_soon_threadsafe, then records release.wait(timeout=2). Preserve entered and202 assertions. Independent design review found early assertions could skip awaiting the original request task; await the real request directly and release in finally before any assertions, ensuring worker completion precedes fixture teardown. Experiment2 executes ONLY start_generation inline in the event loop, restoring routing afterwards. Same delayed-worker control now1passed32warnings3.34s/4.965outer/natural0/control1. Attempt3 strict FastAPI generation-wrapper inline control1failed32warnings4.05s/5.454outer/natural1 at release_observed[False]!=[True], after202 assertion/control1. All3 causal attempts used; no further experiments. Complete original generation file128passed102warnings112.80s pytest/114.679outer/natural0/no skips reported. All other module AST identical,3 assertions before/after; existing202 and entered assertions retained. Scoped Bandit baseline/current291 identical signatures0errors/no new; lint/compile/diffcheck0. Wholefileformatter1 inherited, touchedfunctionformatter0; no new assertion suppression. These controls demonstrate the request-start clock conflates dispatch latency with loop progress; they do not establish the exact hosted scheduling delay or full-shard/native/UAT acceptance.
**Status**: Complete

## Stage 3: Review and publish
**Goal**: Independently review and publish the minimal test correction.
**Success Criteria**: Actual-source and final staged review clear, original unrelated functions/assertions identical; task formatting/diff check pass; exact-source/dev/remote/human-summary guards and normal hooks/exact-lease publication pass and fresh corrected-head hosted gap-verified-6 outcome succeeds.
**Tests**: Preserve human summary and fresh Cubic suffix with exact readback. Fresh hosted qualification remains required; no merge/native/UAT acceptance. Independent actual-source three-path review CLEAR/no findings; source blob e14fe5ecc35512ac68d952fa0e4747d672597cd8. Fresh prepublication PRhead/remote b483 and dev502 unchanged, authored body/human summary exact. Final staged review/publication and hosted outcome remain required. Retain this plan until stages finish; remove only this own completed plan locally when appropriate, never publish tracking-only closures.
**Status**: In Progress
