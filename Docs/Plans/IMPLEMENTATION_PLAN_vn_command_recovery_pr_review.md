# VN Command Recovery PR Review

Task: TASK-13385. PR: https://github.com/rmusser01/tldw_server/pull/3028.
Requester authorized protected rebases, scoped review fixes and a gated normal
merge. Preserve the human-written Change summary verbatim and unrelated work.

## Stage 1: Protected Rebase

**Goal**: Rebase the owned PR branch onto latest dev without changing prior patches.
**Success Criteria**: Clean tracked checkout and matching remote ownership before
rebase; conflict-free rebase and all prior patches unchanged in range-diff; explicit
expected-head lease on publication. Exclude local preview link and archive13379.
**Tests**: Range-diff, unrelated base-file equality, VN/fetch/shared-auth tests,
frontend typecheck, scoped lint, diff checks; unchanged VN Python Bandit baseline.
**Status**: Complete

During the final ownership check for the tested scoped verification-feedback
fix, dev advanced to `de7f453593dbb40f069a4666fd562fc5f3622817` through
independent PR #3039 (seven scheduled-automation message-store/settings/test/task
files, inspected read-only). The local fix was first preserved as
`4490d2dae00cede179a31eb5f5b1fed4e7b1a9b5`; tracked checkout was clean before
the conflict-free 28-commit rebase. The completed FINAL range-diff preserves all
28 patches unchanged at `470037b3520e086340fd19349ee742c29fa85300`.
The intermediate range while rebase was still running is not qualification.
Fresh 236 VN/fetch/shared-auth tests pass (50.66s, 11 files, one worker), as do
typecheck, both scoped lint commands, VN backend compilation, diff and base-file
equality checks. VN Python Bandit has zero findings/errors over 9068 lines; it
does not scan TypeScript. All 393 VN backend tests pass (504.28s, 13 warnings,
normal exit 0), using the activated main environment, existing CI overlay and
approved DB temporary root without changes or weakened limits.
The remote was owned `ba3f8d7e5909bd4a340658cb7cc18b96049a062a`; protected
publication as `7ee1cd4bf5b306ff417ca7889da3554018f678c1` used that full
expected-head lease after qualification and fresh live-dev/ownership checks.
The additional evidence commit changes only
TASK-13385 and this plan. New exact-base/head full reviews, CI and
verified normal merge remain pending. No independent PR #3039 files are edited.

Dev advanced to `bd2ae757d274e7eda3edb48eb682733c401efe56` through independently
merged PR #3008 (13 files). Its VN change requires an authenticated caller for
starter-matrix discovery; Start/Retry dispatch is unchanged. The auth-discovery
routes and OpenAPI fingerprint are inherited without edits. Clean tracked
checkout and owned remote `93428614b05ad22a7934bb68f0238649a50d3ef5` were
verified before the conflict-free rebase. The completed-rebase range-diff
preserves all 26 prior patches unchanged at local
`45a8dd9aec9c60205742aab0200e515312d43448`. Fresh 220 VN/fetch/shared-auth
tests pass (71.91s, 11 files, one worker), and all 393 VN backend tests pass
(735.97s, 13 warnings, normal exit 0). Main environment, existing temporary CI
overlay and approved DB temp root were used without changes. Typecheck, both
scoped lint commands, backend compilation and diff checks pass. VN Python Bandit
has zero findings/errors over 9068 lines; it does not scan TypeScript. Backend,
CI, unrelated tasks, fingerprint and shared TldwAuth match the new base.
The evidence commit changes only TASK-13385 and this plan. Publication uses an
explicit lease protecting the full original 934 remote head. Complete new-base/
head reviews, exact-head CI and verified normal merge remain pending. The skipped
Qodo 934 result and unanswered retry question concern the superseded base, not
new-head qualification. Stage 2/3 remain In Progress until verified merge.

Final merge audit found dev advanced to
`3c9d97c56b29abc4c0396274b9560859aee06959` through independently merged core
PR #3011 (611 files, including AuthNZ, DB, frontend and CI changes). Clean tracked
checkout and owned remote `5aeaa11ae02250d0d99e997b70f5d7b3727594be` were
verified before the conflict-free rebase. Final completed-rebase range-diff
preserves all 23 prior patches unchanged. Fresh 215 VN/fetch/shared-auth tests
pass (31.64s, one worker); all 393 VN backend tests pass (282.81s, 13 warnings,
normal exit 0), using the main environment, existing temporary CI overlay and
approved DB temp root. Typecheck, both scoped lint commands, backend compilation
and diff checks pass. Unchanged VN Python Bandit reports zero findings/errors
over 9064 lines; it does not scan TypeScript. Backend, workflows, unrelated tasks
and shared TldwAuth remain equal to the new base. No tests skipped, limits
weakened or environments changed. Publication must protect the full owned 5ae
remote head with an explicit lease. Its seven required gates and complete hosted
reviews passed before this base advance; CodeRabbit explicitly withdrew its
readable-command discard suggestion after the source/design disposition. Those
results are historical, not qualification of this new base/head. The additional
commit records only TASK-13385 and this plan; new complete reviews, exact-head CI
and verified normal merge remain pending. Stage 2/3 remain In Progress.

Dev advanced to `df1fcc7a52306f400b843c8b0ea0bc90d0396056` through independently
merged Persona PR #2817. Its 95-file integration includes shared UI, ChaChaNotes
SQLite V73/PostgreSQL V77 migrations, Persona schemas and fingerprint, published
VN documentation and existing CI entries. Clean tracked local and owned remote
`f5e96c0fb10c90aca68a74d59971a621eab2e376` were verified before the conflict-free
rebase. The final completed-rebase range-diff preserves all 21 prior patches
unchanged. Fresh 211 VN/fetch/shared-auth tests pass (47.83s, one worker),
typecheck, both scoped lint commands, compilation and diff checks pass. All 393 VN
backend tests pass against the inherited DB migrations (393.68s, 13 warnings,
normal exit 0), using the main environment and CI-aligned temporary overlay with
an approved temporary DB root. No test skips, timeouts or environments changed.
Unchanged VN Python Bandit has zero findings/errors over 9064 lines; it does not
scan TypeScript. Backend, workflows, inherited Persona UI, published docs,
OpenAPI fingerprint, unrelated tasks and shared TldwAuth match the new base.
Publication must protect the full original owned f5e96c0f remote head with an
explicit lease. Complete Qodo and CodeRabbit reviews of that prior head found no
actionable issues; they do not qualify this new base/head. The additional evidence
commit changes only TASK-13385 and this review plan. Fresh complete reviews
and all live dev gates remain pending, and Stage 2/3 remain In Progress.

Dev advanced again to `19215eb89ba658b13babe28fc2185d6821a0c462` through
PR #3031, changing only two unrelated Backlog records. Clean tracked checkout
and owned remote `97302526fb54b2c3b7a8006ddfb5b4dca6604054` were verified before
the conflict-free rebase. Final range-diff preserves all 17 prior patches.
Fresh 174 VN/fetch/shared-auth tests pass (33.51s, one worker), typecheck and
both scoped lint commands pass. Unchanged VN Python Bandit has zero findings
and errors over 9064 lines; it does not scan TypeScript. No base code, workflows,
unrelated tasks, shared environments or local preview artifacts are modified.
Publication must protect the full original 97302526 remote head with a lease.
Qodo completed that prior head with no active findings at 20:35:34 UTC; its
replacement license audit passed at 20:37:28 UTC. Those results do not qualify
the new base/head, which still needs complete hosted reviews and live CI.

Latest rebase onto `f4b69eabea7a1c72013cbb66275ce4d187e2dd69` inherits PR #3032's
unrelated license-audit verdict publication fix. Clean tracked local checkout and
owned remote `8e2df51c6a8cf1c2f7d25fd9f93d003f30d1cc9f` were verified before
rebasing the scoped epoch fix. Rebase was conflict-free; final range-diff after
completion confirms all 14 prior patches unchanged. Fresh typecheck, both scoped
lint commands and unchanged VN Python Bandit pass. The first full test run hit
six 5000ms timeouts, including existing remount/replay cases (139 passed,
126.87s); it reported no assertion failures. A bounded rerun of the identical
145-test suite with one worker passed (54.79s), without changing test timeouts,
skipping tests or modifying environments. Backend, base workflows, unrelated MCP
tasks and shared TldwAuth remain byte-identical to latest dev. Publication uses
an explicit lease protecting the full original owned remote head.

Published 581979a4a87543aedd73ac0f11667c2dc253e73b onto dev
35d6dd90d4c3b703a753efdbd926e30af4f9eac5 with an explicit lease protecting
5de2ed11671f593968a86aef24d3be0422488f75. All five prior patches unchanged;
97 tests, typecheck, scoped lint and diff checks pass. Bandit baseline has zero
findings/errors; it does not scan touched TypeScript.

Dev advanced to `a3d52f30b0d21b8528d426d16e06c4a013414807` through the
independently merged AuthNZ compatibility fix in PR #3030. The latest authorized
rebase from clean owned `98be158e87903b27afa4a98c2552bc08729e176d` is conflict-free;
all ten prior patches are unchanged in range-diff. Fresh 116 VN/fetch/shared-auth
tests pass (8.79s), frontend typecheck and scoped lint pass. The read-only AuthNZ
guard suite passed 123 tests (375.07s, exit 0) using isolated SQLGlot 30.20.0, including
the actual startup DDL and fail-closed controls. Slow session cleanup was sampled
in Python garbage collection, not a network wait. The process exited normally
before a bounded stop attempt reached it; no process was terminated. Direct
startup-DDL acceptance and off-id AUTOINCREMENT rejection checks also exited 0.
The unchanged VN Python Bandit
baseline has zero findings/errors over 9064 lines and does not scan TypeScript.
Publication uses an explicit lease on the full original owned head.

Dev subsequently advanced to `f4bcc9bd70b12e2aae71d1633bff0053946d0314`
through PR #3029's unrelated license cancellation, workflow-test and task updates.
Clean owned `d8b62eeea93129a7f0f92b29f6f3fac62b204077` was rebased without
conflicts; all eleven prior patches are unchanged in range-diff. No base backend,
workflow, unrelated task or shared auth-service files are modified. Publication
must protect that full expected remote head with an explicit lease.

## Stage 2: Current-Head Review

**Goal**: Address all actionable findings and obtain complete exact-head review.
**Success Criteria**: Explain Qodo's human-gate finding inline, distinguish design
approval from merge authorization, preserve requester summary; validate any new
scoped findings with regression tests before fixing. No backend scope expansion.
**Tests**: Exact base/head coverage in completed Qodo and CodeRabbit full reviews,
all review comments/threads checked, scoped regressions and relevant suites.
**Status**: In Progress

The first complete full reviews of published
`7ee1cd4bf5b306ff417ca7889da3554018f678c1` finished on 2026-09-28:
Qodo in Deep mode at 03:51:33 UTC (explicit update 5863015836), and
CodeRabbit at 03:54:59 UTC (trigger 5862971038, exact source/covered commit
with kind reviewed). Qodo finding 4118397636 asks to isolate successful URL
normalization, invalid URLs and unverified principal rejection. The original
29-test journal suite passed before the split (1.09s); this is test hygiene,
not a new runtime defect or invented red evidence. Separate normalization and
principal cases plus `it.each` invalid URLs now pass as 33 journal tests
(0.788s), with the complete 240-test VN/fetch/shared-auth suite passing in
53.78s across 11 files with one worker. CodeRabbit minor 4118400067 is corrected
through official task editing: the current final summary separates words and
counts into readable sentences and paragraphs, retaining historical notes.
Typecheck, both scoped lint commands, diff and runtime/base-file equality checks
pass. Fresh Python VN Bandit has zero findings/errors over 9068 lines and does
not scan TypeScript. Runtime, backend, design, auth, CI and shared environments
are unchanged; the fresh 393-test backend qualification above still applies.
CodeRabbit's remaining architecture inference is the already documented server
idempotency/rollback limitation, not an established authorization bypass or new
scoped code defect. This cleanup changes only tests, TASK-13385 and this plan.
Its published head still needs complete full reviews and all exact-head live
gates before verified normal merge. Stage 2/3 remain In Progress.

Qodo's first requested full `ba3f8d7e5909bd4a340658cb7cc18b96049a062a`
reassessment completed in Deep mode at 03:12:51 UTC (explicit update
5862605551), finding 4118238588. Ten rendered Start/Retry/Cancel and recovered
Start/Retry cases across focus/pageshow failed on missing unsent feedback
(6.94s test time, 9.69s total); six account/pack-context controls passed before
the fix. The existing matching-token completion helper now reports an unsent
verification only for the still-owned, selected-pack action. Verification
sequencing, authority epochs, current-capture checks and journal behavior are
unchanged; there is no automatic retry or POST. All 16 focused checks pass
(6.82s test time, 10.31s total), including unchanged journal bytes before an
explicit retry, exact original recovery body/key and acknowledgement-only
removal. Final 236 VN/fetch/shared-auth tests pass (67.46s, 11 files, one worker).
Typecheck, both scoped lint commands and diff checks pass. Unchanged VN Python
Bandit reports zero findings/errors over 9068 lines; it does not scan TypeScript.
The new-base 393 backend tests already passed (735.97s, 13 warnings, normal exit
0), and this frontend-only feedback fix changes no backend/auth/CI/environment.
CodeRabbit's full ba3 review completed at 03:17:04 UTC with exact source/covered
commit, kind reviewed, minimal merge risk and no actionable scoped finding.
Those complete reviews do not qualify the changed fix head. New complete
reassessments, exact-head gates and verified normal merge remain required.

Qodo's requested full 78bb52e2 reassessment completed at 23:53:20 UTC with no
active bugs/rules. CodeRabbit's full review completed at 23:57:42 UTC with exact
source/covered-commit coverage and one valid minor finding, 4117523366. A new
write exceeding 64 readable commands was classified as unreadable storage and
exposed whole-journal discard. Three storage validation cases and two rendered
Start/Retry capacity cases failed before the fix (5 failed, 0.687s test time,
3.03s total); the rendered failures showed the actual discard button. One shared
writer error change now reports an unsent write without marking saved records
unreadable. Corrupt reads, quota denial and warned-discard controls are unchanged.
All 26 focused checks pass (4.31s test time, 6.04s total), including exact-key
replay and removal of only the acknowledged entry. Final 220 VN/fetch/shared-auth
tests pass (50.89s, 11 files, one worker), typecheck and both scoped lint commands
pass, and diff checks are clean. Unchanged VN Python Bandit has zero
findings/errors over 9064 lines; it does not scan TypeScript. No backend, CI,
dependency or environment changes. The scoped fix was published as
`c2b242a916c0ce141555bc8351aab19e987af3c6`; finding 4117523366 was answered
and resolved. Complete reviews and exact-head CI for this tracking follow-up,
and verified normal merge, remain pending. Prior-head completion does not qualify
a changed tracking head.

Qodo completed the exact-head full-diff reassessment at issuecomment-5858115068
with no production defects and accepted the human-gate clarification. Its minor
tracking hygiene feedback was addressed in 804c0745c0. CodeRabbit completed the
full review of 804c0745c0 at 17:46:08 UTC with three scoped findings:
failed discard hides unreadable controls, a retired-plan link, and heading spacing.
The failed-discard workbench regression failed before the minimal hook fix because
the warned confirmation checkbox disappeared. With the fix, the journal and
generation lock survive removal denial, confirmation resets, and explicit retry
succeeds without generation or cancellation after storage access is restored.
The retired documentation link was removed through official Backlog mutation;
heading spacing is corrected here. All 98 VN/frontend fetch-client tests pass
(8.97s), typecheck and scoped ESLint pass, and unchanged Python VN Bandit has zero
findings/errors (not a TypeScript scan). Any changed head still requires complete
reassessment; no backend scope expansion is authorized.

Qodo reassessment of 70953ae10c (issuecomment-5858353005) identified an empty-string
journal being mistaken for absence. Storage and rendered workbench regressions
failed before the one-line null-only absence check: storage did not throw and
Start was enabled (2 failed, 10 passed, 2.15s). The initial UI wait timed out;
the corrected direct disabled-state assertion supplies genuine regression proof.
Malformed data stays intact until warned explicit discard; a missing key remains
valid. All 101 VN/frontend fetch-client tests pass (13.11s), typecheck, scoped
ESLint and diff checks pass. Complete new-head hosted review remains required.

Qodo's complete deep review of 5953de6951 reported zero bugs and the shared-auth
architecture finding in discussion 4116389900. The hook's profile-first policy is
moved into shared `services/tldw/verified-principal.ts`, using the existing caller
transport. No cached identity, new auth state, or change to `getCurrentUser()` is
introduced. Seven identity behavior checks passed before extraction; all 116
VN/frontend fetch-client/shared-auth checks pass after extraction (9.38s), as do
typecheck and scoped lint. The shared-file lint command required the installed
ESLint 9.39.2 binary with the existing frontend config and UI working directory;
the cached bunx 10.11.0/config-base setup failures are not source findings.
Bandit on unchanged Python VN baseline remains clear and does not scan TypeScript.
Complete exact-new-head hosted review and all required gates remain prerequisites.

CodeRabbit discussion 4116478050 flagged the final summary's outdated 97-test
checkpoint. Official Backlog mutation updates it to the latest 116 passing tests
and makes new rebased-head reviews explicitly pending. Historical test/review
evidence is retained, including Qodo's completed 98be158e87 reassessment.

Qodo's completed d8b62 deep reassessment found malformed truthy account status
passing verification (discussion 4116528569). Twelve rendered workbench cases
cover profile and legacy identity responses. Before the strict boolean guard,
eight malformed truthy cases enabled Start incorrectly; four false/missing
controls passed (2.65s). The guard now requires `is_active === true` before scope
creation. Valid fresh verification unlocks controls without automatic generation.
All 128 VN/fetch/shared-auth tests pass (14.56s), typecheck and both scoped lint
commands pass. Unchanged VN Python Bandit remains clear, not a TypeScript scan.

CodeRabbit's completed d8b62 full review recorded no actionable comments. Its
inferred crash-between-admission-and-idempotency-response limitation needs server
reconciliation outside this frontend slice. Valid ambiguous commands remain
locked with their original key; warned discard is only offered for unreadable
storage. This is not an exactly-once or definitive server-outcome guarantee.
Complete review and CI on the newly published base/head remain required.

The complete 97b7 reviews identified three additional scoped findings:
Qodo 4116594441 (invalid current server setting hidden by `isCurrent`), Qodo
4116594446 (replay CSRF rejection erases an earlier ambiguous command), and
CodeRabbit 4116602159 (generic HTTP 400 may follow batch creation).
Fourteen rendered regressions failed before the fixes (5.39s): two lacked a
validation warning and twelve lost their saved command. Current scope-validation
failure now reports a safe error and blocks generation without applying the
response or removing the journal. Stale authority guards run before validation.
Generic 400 no longer establishes initial rejection; all failed replays retain
the original request because they cannot prove its earlier outcome. No new
backend codes or admission behavior are introduced. Initial known 409 rejection
cleanup remains supported. Focused scope/fencing checks pass (5), admission and
known-rejection checks pass (4), and replay/rejection checks pass (15). The first
full run passed 142 tests (41.42s); with the stale-invalid-server control, final
verification passed all 143 tests (29.53s). Both scoped lint commands and
typecheck pass, and unchanged VN Python Bandit has zero findings/errors.

Qodo's complete 8e2df review found a further authority-fencing defect
(4116657605): invalid settings cleared scope but not its epoch, so a second
in-flight pack response became current after same-scope re-verification. Both
rendered concurrent-pack regressions failed before the fix (3.17s test time):
the late success and HTTP 422 each removed the second saved request. Advancing
the epoch in the current-capture validation catch keeps every earlier capture
stale without clearing commands. Five focused settings/fencing checks pass,
including old-account isolation. Both packs can explicitly replay their exact
original requests and clear only on acknowledgement, with no automatic POST.
All 145 VN/fetch/shared-auth tests pass (49.11s), typecheck and both scoped lint
commands pass. Fresh unchanged VN Python Bandit has zero findings/errors over
9064 lines; it is not a TypeScript scan. CodeRabbit's complete 8e2df full review
finished at 19:39:54 UTC with exact-head reviewed coverage and no actionable
comments; that prior-head assessment does not qualify this follow-up.

Qodo's complete 428cddc review identified corrupt saved scope strings being
mistaken for another authority (4116725510). Eight storage and eight rendered
workbench cases failed before the fix (16 failed, 3 controls passed): the journal
was removed and Start became enabled. Stored scope now uses the existing scope
validator and must equal its canonical form before authority comparison. Invalid
or noncanonical records remain byte-identical behind warned explicit discard;
the three genuine authority-mismatch controls still pass.

Initial green runs exposed a new test expectation error: unreadable recovery
prevents detail loading, so Retry is absent rather than disabled. Corrected that
assertion after reassessment; those fixture failures are not bug evidence.
A separate canonical-write regression then failed for repeated trailing slashes.
Normalizing all trailing slashes, consistent with shared networking, makes scope
creation stable so the new reader accepts newly written scopes. Final full
verification passes all 162 VN/fetch/shared-auth tests (50.93s, one worker),
typecheck and both scoped lint commands. Fresh unchanged VN Python Bandit has
zero findings/errors over 9064 lines, not a TypeScript scan. Existing Node/Next
advisories and shared environments are unchanged. CodeRabbit's complete 428cddc
full review finished at 20:03:45 UTC with exact-head reviewed coverage and no
actionable defect; it does not qualify this new scoped follow-up.

CodeRabbit's complete 40c6d full review finished at 20:17:41 UTC with explicit
exact-head reviewed coverage and no established merge blocker. Its cancellation
inference nevertheless yielded a concrete scoped regression: focus/pageshow
cleared an unresolved Cancel guard, enabling another Cancel or a new Start.
Eight rendered event/status/outcome cases failed before fixing (1.58s); two
authority-boundary controls passed. Initial tests used an incorrect accessible
name and timed out; only the corrected behavioral failures establish the bug.
The existing guard now stores token and command kind, preserving only Cancel
through non-account revalidation and clearing all on authority boundaries.
Matching-token completion releases the guard without applying stale results.

A controlled queued credential transition after verification exposed a narrower
Cancel-only pre-send fencing gap (1 failed Cancel, 1 passed Start, 0.296s).
Cancel now checks the current capture immediately before POST; Start/Retry
already use that check through remember. Supported same-tab credential mutation
paths synchronously invalidate captures, and transport header/fetch construction
has no intervening await. This is not evidence of cross-account backend execution
or a guarantee for unnotified or cross-tab credential changes. No durable Cancel
replay or server cancellation reconciliation is added. Fifteen focused checks
pass (3.18s); all 174 VN/fetch/shared-auth tests pass (32.69s), typecheck and both
scoped lint commands pass. Fresh unchanged VN Python Bandit remains clear over
9064 lines, not a TypeScript scan. Complete new-head reviews and CI remain gates.

CodeRabbit's prior 973025 review posted findings 4116871656 and 4116871660
during the protected rebase. An empty profile body throws before the hook can
display its safe missing-identity message. Two shared lookup and two rendered
workbench cases failed before optional profile access (4 failed, 0.187s test
time). No profile absence is converted into a legacy fallback.

Twelve rendered Start/Retry lifecycle cases failed before preserving ownership:
focus, pageshow and config revalidation enabled recovery while the original POST
was unresolved (12 failed, 12 controls passed, 8.81s test time). The first focused
filter matched only the empty-profile cases, not the quoted parameterized command
names; corrected the filter before claiming command regression evidence.
Same-account boundaries now preserve every in-flight command. Actual authority
boundaries still clear old ownership; matching tokens alone release a new guard.
The now-unnecessary command-kind wrapper was reduced to the original symbol map.
All 29 focused checks pass (11.27s test time), including four new old-account
Start/Retry completion controls and existing cancellation/profile controls.
Settled ambiguous requests and remounts remain explicitly recoverable with their
original key; no concurrent replay, automatic POST or backend guarantee is added.
Final full qualification passes all 194 VN/fetch/shared-auth tests (59.99s, one
worker), frontend typecheck and both scoped lint commands. Unchanged VN Python
Bandit has zero findings/errors over 9064 lines; it does not scan TypeScript.
Existing Node/Next advisories and environments remain unchanged.

Qodo's exact a642bfa reassessment completed at 20:56:06 UTC with zero bugs and
two scoped rule findings (4116918856 and 4116918857). Pre-send verification can
reread a different-key command for the same pack; `remember` blocked the new
request without explaining the conflict. Both rendered Start/Retry regressions
failed before the fix (2 failed, 0.422s test time). The hook now reports a safe
conflict message, leaves the original journal byte-identical, and sends no new
work. Explicit recovery still sends the saved body/key and clears only after
acknowledgement. Separately split quota, read and cleanup failures into three
isolated tests; the original combined test passed before splitting, so this is
test hygiene rather than new bug evidence. All five focused tests pass (0.873s
test time). Final full verification passes all 198 VN/fetch/shared-auth tests
(62.30s, one worker), typecheck and both scoped lint commands. Fresh unchanged
Python VN Bandit has zero findings/errors over 9064 lines, not a TypeScript scan.
Backend, workflows, unrelated tasks and shared auth-service files match dev
19215eb89ba658b13babe28fc2185d6821a0c462. Complete changed-head hosted reviews
and exact-head CI remain pending. The a642bfa CodeRabbit request was rate-limited;
no duplicate request or paid billing was enabled.

CodeRabbit completed the exact c806b624 full review at 21:17:39 UTC (trigger
5859845466); the sticky records that head as reviewed and raises one minor
command-error lifecycle issue (4116974418). Eight rendered Start/Retry cases
failed before the fix: focus/pageshow erased both ambiguous and known rejection
messages (8 failed, 5 controls passed, 3.57s test time). Error clearing now follows
selected pack/account context, not background verification revisions. Successful
detail reads still clear their own internal read errors, while command errors
remain visible. All 13 focused cases pass (4.24s test time), including pack/account
boundary clearing and recovered detail-error controls; original request replay
and acknowledgement cleanup remain intact. The first full suite passed 211 tests
(42.70s). A new test-call line break caused an ESLint style error, corrected without
changing behavior; final post-format qualification passes all 211 tests (51.63s,
one worker), typecheck and both scoped lint commands. Fresh unchanged VN Python
Bandit has zero findings/errors over 9064 lines, not a TypeScript scan.

The architecture note concerns existing service-exception claim release after
admission, not new backend code. Read-only inspection confirms service errors
release claims while response recording is outside that try/catch. The design
now explicitly distinguishes original-key preservation from duplicate-free
server execution if an admitted claim is released. Server reconciliation remains
outside this frontend slice; no backend/Jobs or unrelated issue changes are made.
Qodo's requested c806 reassessment completed at 21:09:42 UTC with no active
findings. Neither prior review qualifies the new scoped follow-up; complete
changed-head reviews and all live CI gates remain pending.

Qodo's explicit review of the rebased 82390fdae8 head found two verification
races (4117271887 and 4117271890). Four rendered workbench cases failed before
the fix: an older failed identity check replaced a newer successful check's
readiness, an older successful check sent a superseded Start, and both late
success and known-rejection responses erased a saved Start after a concurrent
identity HTTP 401 (4 failed, 126 skipped, 2.44s run). The hook now lets only the
latest applicable identity request publish verification state and advances the
authority epoch on an applicable failure without clearing the saved journal or
in-flight command ownership. The same four cases pass after the fix; both
late-response cases also verify that the selected pack's status stays unchanged.
All 215 VN/fetch/shared-auth tests pass on the final run (44.50s, 11 files, one
worker); typecheck,
frontend/shared scoped lint and diff checks pass. CodeRabbit's complete review
of 82390fdae8 finished at 22:27:29 UTC with exact reviewed coverage and no
additional actionable scoped finding. Both old-head reviews and CI will be
superseded by the fix commit; new-head complete reviews and live gates remain
required. No backend, Persona, Jobs or dependency-policy changes are included.

## Stage 3: Gated Merge

**Goal**: Merge normally only after current-head review and live dev gates pass.
**Success Criteria**: Fresh head/base/rules/summary check; backend-required,
security-required, coverage-required, frontend-required, e2e-required,
container-build-check and frontend-license-policy/trusted/dev all pass on exact
head. Full match-head normal merge, verified merge commit, task finalized and only
this completed plan removed. Preserve checkout and chat; stop own follow-up.
**Tests**: Live GitHub rules/checks, merge API verification and tracked diff check.
**Status**: In Progress

Exact 804c0745c0, 5953de6951 and 98be158e87 E2E failed before tests in unchanged
AuthNZ bootstrap. Isolated
30.19.0/30.20.0 SQLGlot comparison reproduces rejection of identical canonical SQL
because standalone AUTOINCREMENT rendering changed. PR #3030 independently
integrated the backend fix into dev with all its required gates passed. The
frontend PR is rebased onto that external integration; no separate-fix decision
is still needed. Shared environments and dependency policy are unchanged.
No merge attempted; all live current-head gates remain prerequisites.
