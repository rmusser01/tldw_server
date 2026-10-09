# Knowledge corrective PR3214 CI closure

TASK-13534. Scope: actual hosted backend failure and newly observed advisory diagnostics on9a2f916262. Existing user authorization covers required CI repairs. Human Change summary remains required; no merge or gate bypass.

ADR assessment: no new durable decision. Existing ADR026/031/034/042/059/065/066 and the repository ratchet procedure govern; no helper, workflow, rule or exception expansion.

## Stage 1: Diagnose
**Goal**: reproduce the ratchet failure and compare type diagnostics against exact dev sources with identical local rules.
**Success Criteria**: unchanged guard AST/finding-count proof; explicit introduced and retained diagnostic inventory with environment qualifications.
**Tests**: canonical ratchet RED; mypy2.4.0 common20 inputs on base/current and full22 current inputs.
**Status**: Complete

## Stage 2: Repair
**Goal**: apply only demonstrated owning corrections through existing mechanisms.
**Success Criteria**: canonical baseline maintenance changes only one existing line position; any introduced type diagnostics are resolved without ignores or policy changes.
**Tests**: exact baseline diff; all tenant-isolation ratchets and affected owning tests for any source changes; touched-scope security and canonical hooks.
**Status**: Complete

## Stage 3: Historical workstream diagnostics
**Goal**: repair demonstrated diagnostics owned by the earlier Knowledge PRs rather than treating current-dev matches as proof of pre-workstream debt.
**Success Criteria**: exact historical-source and line ownership receipts; existing Notes facade/storage, Sync validators and HTTP lifecycle contracts reused; all82assigned diagnostic instances resolved or specifically adjudicated with evidence, without ignores/casts or gate relaxation.
**Tests**: current identical-rule type receipt; owning Notes/Sync/probe/HTTP tests; targeted RED/GREEN for any behavioral validation change; touched-scope security/lint/format/hooks; one scoped review of this newly discovered unit.
**Status**: In Progress

## Stage 4: Publish and verify
**Goal**: normal commit, actual latest-dev rebase and exact-head hosted verification.
**Success Criteria**: accurate audit/task13514 tracking, seven statuses and human-owned summary before merge; retain original failures and named follow-ups.
**Tests**: exact-head CI and actual comments; no stale-head acceptance or auto/admin workaround with queue unset.
**Status**: In Progress

CI scope continuation: consolidate6new HTML-prefix functions into the existing registered7-function security suite; preserve all cases/decorators/assertions. Actual unchanged contracts and owning tests255pass;21base/22current mypy504→501existing signatures with0introduced, not all-green. Fresh annotation owner1pass/bytecode-equivalent;51tenant tests and scopedRuff/Bandit/Black recorded. Original hosted/local failures and intermediate counts retained. Independent CI-unit review Approved at8822898f96 after root36entry hash verification. Actual fetch/rebase against dev97ea9cd5 and normal push/body update completed; fresh final-head CI, human-owned summary and normal merge remain pending. Existing501advisory diagnostics remain unresolved under TASK13534; no type-green or whole-task completion.

Historical extension after7775: exact unchanged-line/message plus AST-context mapping finds405pre-workstream matches.79suspect diagnostics last touch is in actualPR3205/3213/3214 exclusive commit sets;17others retain explicit provenance. Last-line membership is not causal proof; diagnose before fixes. Prior501baseline matching is limited to current dev, not proof of predating the workstream. Existing approved CI unit/client review stays approved; this is a new observed historical diagnostic unit, not a repeated broad/final UI review.

Historical repair candidate: all82 assigned instances removed; expanded645→522/0introduced and original22 projection501→391 are distinct nonadditive scopes. Current522 diagnostics remain red and individually retained (510exact pre-workstream/2moved historical signatures/10other-history). Final400connected before Notes-only changes and168finalNotes afterward pass with overlapping scope,9new RED/GREEN cases, scopedRuff0/seven-pathBandit0; four existing whole-file format failures and warnings remain qualified. Root29source/config+68receipt verification has0mismatches. Stage3 awaits its single scoped independent review; Stage4 awaits normal commit/publication, fresh dev/exactheadCI/human summary/merge.
