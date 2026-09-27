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
