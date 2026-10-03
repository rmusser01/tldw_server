# PR3084 Qodo follow-ups — TASK-13260.278.18.83.45

Original review scope: four actual Qodo threads on published50e4863d95. PR was open/ready with seven required contexts passed on50; those old greens do not accept later heads. Latest-dev integration and native whole-Prompt/first-import gates are tracked below. Retain the original human Change summary. The separate corrected frontend attempt remains STOP at attempts2/recovery1 and its new installation/build request is still pending.

## Stage 1: Verify actual feedback and source
**Goal**: Determine which findings can be corrected without weakening the retention regression.
**Success Criteria**: Complete four-thread census, actual source inspected, original task criteria preserved, distinct source worktree.
**Tests**: Read-only review/source inspection and baseline metrics unit selection.
**Status**: Complete

Three omissions are present: five return annotations, four docstrings and the metrics test unit marker. Baseline targeted metrics collection combines -m unit and -k test_api_metrics_preserves_its_existing_call_counter, selecting zero/deselecting all three (natural exit5, collection only). The two lifetime parameterizations already carry unit markers and are deliberately excluded; no test assertion is run. Private classifier assertions intentionally populate and verify the real FastAPI cache-admission boundary before the weakref outcome, and must remain under the user's no-weaker-assertions constraint. Add rationale rather than remove them. Retained first-import red WIP and its plan are untouched. Initial managed checkout failed because the old Git database lacked the PR commit; fetching the exact public PR ref allowed one corrected managed checkout at the same commit.

## Stage 2: Correct annotations, documentation and category
**Goal**: Address the three verified omissions and explain the deliberate causal probe.
**Success Criteria**: Accurate annotations; normal/fallback route response_model remains explicitly None where annotations would otherwise infer models; four docstrings; unit and asyncio markers both retained; every handler body and test assertion preserved.
**Tests**: Source AST comparison, targeted metrics/category and positive control-plane checks; compile, Ruff and touched-scope Bandit.
**Status**: Complete

Root returns either a welcome dictionary or a RedirectResponse, so Any keeps it registerable by existing callers. Main explicitly preserves response_model=None for root and both JSON metrics registrations; typing must not change response serialization or schemas. Newly documented public root/readiness-alias operations may gain only intentional OpenAPI descriptions. No private first-import/GC/negative-control replay, cache clearing, logger ownership repair, new dependencies or native run.

## Stage 3: Review and publish the bounded correction
**Goal**: Independently review actual source and normally publish a qualified correction, preserving task/human gates.
**Success Criteria**: Review clear; exact patch/source attribution and normal hooks; correct remote/head freshness; all native/root criteria remain open. Any later latest-dev rebase must preserve the owned patch and receive independent actual-source review before publishing that rebased head.
**Tests**: Relevant checks and independent source review, followed by current-head hosted contexts; no acceptance from old greens.
**Status**: In Progress

Latest fetched dev4c4f adds182 paths from the common1c849 base with no direct main/control-plane/lifetime-test change. Local available Python3.11.13/FastAPI0.142.1/Pydantic2.11.7/Starlette1.2.1/pytest8.4.1 is below both requires-python>=3.12 and Pydantic>=2.13.5: any local execution is compatibility evidence only, not declared-stack/native qualification. Do not install dependencies or rerun stopped lifetime controls to fill that gap. Scope Bandit does not establish full-project security or native shutdown. No live UAT, browser recovery, install/upgrade/full43 or all15-bug acceptance; no proof bundles or deleted-evidence recovery.

Bounded verification:67 existing positive metrics/readiness/permission/redirect/HEAD/tag cases passed,2warnings,138.56s pytest/148.3s outer,natural0/no skips reported. Unit selection changed from0selected/3deselected (natural5) to1/3collected/2deselected/2.22s/natural0 with only the metrics case selected (collection only); no lifetime bodies executed. All three Python files compile. Ruff and formatting signatures match the baseline (main9 diagnostics/format1; module/test0); Bandit production0/test7 unchanged LOW B101/0errors. Normalized AST comparison preserves all handler bodies and seven assertions after only declared metadata/three None keywords are removed. Independent actual five-path source/tracking review CLEAR with no Critical/Important/Minor findings; no reviewer tests/runtime actions and no declared-stack/native/final merge acceptance.

Initial correction publication:3474 was normally published atop50 without folding182 incoming paths into that bounded unit. Latest-dev integration is the separate unit below; fresh hosted/native gates remain required before merge. A later rebase must preserve the owned patch and receive independent actual-source review before publication.

Preedit integration plan: start a distinct latest-dev unit after bounded3474 publication. Complete six-path owned range does not overlap182 incoming paths; dev4c4f is the target, subject to immediate live verification. Verify both original commits and full binary patch through rebase/autostash, relevant positive route/config compatibility checks, and independent actual-source review before exact-lease publication. Required CI/license admission remains queued; no old-head/native acceptance or stopped-control replay.

Integration result: conflict-free rebase8af066b185 on verified4c4f preserves both commits identically by range-diff and the entire six-path owned binary patch exactly. Complete head delta equals182 incoming paths; source/test/assertions unchanged; autostash restored only official tracking edits. All72changed/ownedPython compile. Integrated positive group67passed/2warnings/112.27s pytest/119.82s outer/natural0/no skips reported, compatibility-only below both runtime floors. Existing identical-source Bandit/Ruff/format results apply without reruns for counts. Independent actual-source/final tracking review CLEAR with no Critical, Important, or Minor findings. Normal final tracking commit and exact-lease publication remain pending; new-head contexts/native/first-import/root criteria remain open.

Final review: the independent reviewer confirmed accurate route types, response-model registrations, metrics identity, permissions and unchanged causal assertions. Review was read-only; no tests or runtime actions. Final task/plan recording adds two tracking paths to the earlier incoming-only tree delta. Source and tests remain unchanged; the 67-pass result remains compatibility evidence below both declared runtime floors. Original task criteria and status remain open. Frontend authorization is still pending.
