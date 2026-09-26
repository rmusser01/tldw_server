# PR2967 integration verification

Tracking: TASK-13260.219. Target: `dev`. The user requested merge preparation,
then UAT resumption starting with TestBot/UAT-261. The exact-output criterion
remains unchanged; the next complete fresh-install matrix has not begun.

## Evidence storage

Generated `output/playwright/` artifacts are retained locally and excluded from
the PR. The cleanup removed 11,712 branch-added files from Git without deleting
their local copies or rewriting history. Historical tracker links refer to that
local archive. This summary and the running tracker retain reviewable outcomes.

## Completed checks

| Area | Result | Scope and limits |
| --- | --- | --- |
| Chat rejected Retry | 110 tests passed, zero skipped; independent review clear | Actual PostgreSQL HTTP 409, checkout release, lock availability and unchanged history; operation, image and provider-order controls. Full HTTP cancellation is not newly tested. |
| Frontend fixtures | 294 UI tests plus 7 character-selection tests passed, zero skipped | Real ownership/storage/query contracts retained. Local Node 26 differs from CI Node 20; hosted final-head CI remains required. |
| Further frontend readiness fixtures | 46 passed, four local timeouts on exact CI Node 20.20.2; subsequent hosted results qualified below | Study Pack 12/12 and Companion Home 17/17 pass locally; Admin 11/12 and Settings 6/9. Three Settings cases and an unchanged Admin password case exceeded their original five-second budgets during severe unrelated worktree CPU load. No timeout increase or blind local retry. |
| Notes remediation | All 32 tests passed, zero skipped | A further Stage4 fixture now supplies the verified owner required for saving; revision and attachment assertions are unchanged. |
| World Book consumers | 50 tests passed, zero skipped; bounded source review clear | Official PostgreSQL and SQLite; foreign/unassigned lore excluded from RAG, conversation materialization and Chatbook scope/name reads; original transaction and catalogue assertions retained. |
| World Book attachment UI | All 10 tests passed, zero skipped, unchanged time limits | Retry uses the rendered error/button and invalidates both original queries. Named-panel queries reduce unrelated DOM work; the final local retry case took 2504 ms against a 5000 ms limit. Hosted CI remains required. |
| World Book secondary operations | 111 focused tests passed, zero skipped | Real backend operations, rollback, literal LIKE search and PostgreSQL JSONB placeholder controls. Adjacent results are qualified below. |
| Onboarding | Both desktop and mobile tests passed, zero skipped | Fixture preserves the verified session owner while changing lifecycle state. |
| Approved extension replay | One passed, zero skipped/retried/flaky outcomes | Fresh build 29.1 seconds, scenario 6.927 seconds; privacy, edit/apply, focus and narrow layout assertions pass. Tested fixture cd8ed1452a50cb518be9fcdeb7126ca7e21711fc577f849aa252cd9797f92b27. Subsequent first-seed ordering refinement needs hosted CI. Axe has zero violations but one incomplete observation, now UAT281. |
| Settings radio repair | 24 tests passed on CI Node20.20.2, zero skipped | A causal regression reproduces AntD's test-mode native radio-name collision. Three explicit group names preserve independent selections. Auth-mode and timeout accessibility controls pass. Final test-only typing cleanup is source-reviewed; the separate Server URL assertion passes unchanged locally, with no claimed hydration cause. |
| Prompt-review semantics (UAT281) | All 10 component tests and final hosted extension gate pass; independent source review clear | Removing the new group role causes the named-content regression to fail. Hosted fa50b25ca4 passes14 Watchlists and22 Prompt Improvement scenarios, zero skipped/unexpected/flaky outcomes. Final Axe report has zero violations/incomplete observations; screenshot independently inspected. |
| OpenAPI | Canonical fingerprint check passed | Isolated Python 3.12.11, FastAPI 0.136.3, Pydantic 2.13.5; 2097 paths/3207 schemas. |
| Published docs | All 52 docs tests passed in a clean checkout; three macOS controls passed locally | Canonical mirrors have real Git revision dates; fixtures preserve `Site` and exclude `_site`. CI dependency versions remain unchanged; strict warnings are not suppressed. |
| Backend shard coverage | No newly uncovered tests | 805 shards, 4729 test files, 130 pre-existing baseline exclusions. |
| Ingestion smoke fixture | Multipart opt-out control passed | Real upload reaches the original embedding-progress loop; that test still skips because its worker is unavailable locally. No completed-job claim. |

New regression failures and unsuccessful intermediate attempts remain in the
local receipts. An earlier broad dependency-matched run had 252 passes, 10
ownerless legacy fixture failures and one existing Resource Governor skip.
The fixture failures and subsequent transaction regression are corrected in
the 50-case consumer/read result above. The adjacent snapshot/Chatbooks/RAG
run completed with 287 passes and four stale-fixture failures. All four focused
reruns now pass after independently reviewed fixture corrections; no assertions
were weakened. The previously passing 287 cases were not repeated.

Scoped Python compilation and new-regression Ruff checks pass. Bandit reports
zero findings/errors in the six World Book production files; the Chat repair
retains only the unchanged random-jitter B311. TypeScript is outside Bandit's
supported scope. These checks do not constitute whole-repository security
certification.

## Latest integration checkpoint

On8eca6c6bc4, all eight frontend unit shards pass. Shard2 passes31 frontend
and563 UI tests, including all10 Settings timeout/form cases and the original
save/reload scenario. The full extension gate also passes. The final
frontend-required job105657458181 subsequently fails its WebUI session-key
lifecycle assertion before reload on all three attempts: the helper returns
after local metadata exists, before the separate session credential write.
Independent review supported a storage-specific readiness correction. The
first corrected local run passes device and legacy cases but still fails session
readiness after15 seconds, without retries. Diagnosis continues; neither a
completed session save nor a production root cause is established yet.

The requester supplied the human-written Change summary, now published
verbatim in PR2967. The requested rebase replayed327 commits onto freshly
fetched dev59049e094e0845a4611ea725ae19b7c1754ea709. Original application/test
files match the pre-rebase candidate; the recovery ref preserves8eca6.
Historical plan details and local generated evidence are preserved. No
output/playwright files are tracked. Qodo review will run on the updated PR.

Qodo subsequently posted14 findings on the production-equivalent source. Their
individual status and retained verification results are in the
[review ledger](PR2967_QODO_REVIEW_2026_09_18.md). The alleged Flashcards syntax
failure is dismissed by Qodo after actual TypeScript parser verification. The
remaining review repairs are in progress. The session-key failure is reproduced
with native storage events and a controlled concurrent-client regression:
an initializer's stale missing read can schedule removal after a successful
save. Independent review requires serialization of the entire clear/write
mutation and under-lock cleanup, including the late removal window. The first
3/3 local browser candidate does not close that review requirement.

## Remaining merge gates

- The Settings readiness helper is confirmed in8eca6 hosted CI. At fa50b25ca4, shard2
  passes562 tests and fails only the first ordinary save/reload case's blank
  Server URL; the radio repair passes. The candidate now discovers the real
  textbox once, then waits for its unchanged exact value. It separates two
  readiness deadlines without changing product source or persistence checks.
  All10 focused tests pass on CI Node20.20.2. Instrumentation observes one
  field-arrival/value transition in an incomplete broader experiment; the
  earlier hosted timing cause remains unproven.
- Run final candidate CI after pushing the reviewed repairs, including the
  extension fixture's final first-seed ordering and named-content assertion.
  The one explicitly approved local extension replay is complete.
- At pushed1c4ff92226, Notes, World Books, onboarding docs, backend-required,
  coverage-required, e2e-required, security-required and seven frontend shards
  pass. Only the original extension fixture and two Settings assertions fail,
  plus the frontend aggregate. Conditional jobs do not imply broader coverage;
  historical local failures and every intermediate attempt remain retained.
- Verify the session-key fixture correction and final candidate CI; read and
  address every Qodo finding/comment after the PR is ready for review.
- Merge normally, verify the remote result, and resume UAT with UAT-261 first.

Existing PostgreSQL World Books without recorded ownership remain intact but
hidden until trusted maintenance assignment. See the
[upgrade guide](../Deployment/Database/world-book-owner-upgrade.md).


## Final local review acceptance, 2026-09-18

All 14 Qodo findings now have a verified fix or accepted disposition. The reviewed source is committed in b852f644a0 (complete credential mutation serialization), 3420b66228 (PostgreSQL upgrade, legacy Chatbook compatibility and DB/core boundaries), 9344f3ae95 (frontend transport, localization, timers and web route boundaries), and a55e86167f (durable offline Notes saves and owner-safe title recovery).

The final credential repair passes 113 focused and adjacent tests and all three real browser lifecycle cases with zero skips or retries. Backend verification passes 70 combined controls, then 35 focused controls after the final refinements, including four actual PostgreSQL cases with no skips. Frontend review verification passes 46 focused UI, 12 route-title and seven Login/navigation tests. Notes/title verification passes 18 focused UI and 18 existing web title tests; the final test-only lint correction also passes all three provenance cases. Independent reviews accept the resulting changes. Failed intermediate attempts remain retained.

The canonical OpenAPI fingerprint is 6dcb5357a7a6d13b636b6d1cae6a7275a432796b6a2105745fcde7e7dc89bc06. Only the two legacy flashcard field descriptions change; path and schema counts remain 2097 and 3207. Nine touched backend production files have zero Bandit findings.

Source repair is complete locally. Publication, individual Qodo replies and final required hosted CI remain merge gates. These results do not close UAT261 or certify the pending full fresh-install matrix. The requester-provided Change summary is already published, and the latest fetch still identifies dev as 59049e094e0845a4611ea725ae19b7c1754ea709.


Publication checkpoint: all reviewed source fixes are pushed at 38b686f0b2. All 14 Qodo threads are resolved, and the updated review reports zero open bugs, rule violations or cross-repo conflicts. Individual replies include fix commits and verification evidence. Final required hosted CI remains pending.


### Final CI billing-fixture correction

On 4752bbe5aa, frontend shard 4 reports three failures in the existing cookie/logout Billing discovery tests; 445 other shard cases pass. The same three failures reproduce locally (30 passing controls). The fixtures supply JSON bodies but their synthetic Response objects default to text/plain. The real shared request transport correctly uses Content-Type to choose JSON parsing; the actual retained API returns /openapi.json as application/json.

Seven fixture responses now supply that real Content-Type. No production code, assertion, timeout, cancellation or transport mock changes. All 33 cookie/logout cases and nine form lifecycle cases pass on Node 20.20.2. Independent source review confirms the correction preserves the positive/negative, same-origin, timeout and stale-response controls. Earlier failures remain retained. Final hosted CI must rerun on the updated commit. This is a test-fixture correction, not an additional native UAT product finding.


## Verified merge

Merged PR2967 normally into dev at `3cff7962721a60b768464221c1f7fe2a8b25e4d5` on 2026-09-18T21:06:54Z. All seven required gates pass; all eight frontend shards pass. Final hosted authentication lifecycles report 3 WebUI passes, 3 extension passes and 1 cookie pass. All 14 Qodo threads are resolved. The merge tree exactly equals tested head 35e46b7a14. Generated Playwright artifacts remain excluded. UAT261 and the next full fresh-install matrix are not certified by this merge.


## First post-merge UAT result

The original PostgreSQL single-user TestBot was checked first on the ordinary merged runtime, preserving its profile, configuration, official fixture holder, card and selected model. One browser completion submission returned and saved `BEEP BOOP` without the required final period. Normal reload preserved exactly one user and one assistant row. Independent evidence review confirms this bounded result, no Retry and no settings change. UAT261 remains open; no application cause or provider reliability claim is established. The requester explicitly authorized continuing the full fresh matrix with261 open. See the [current matrix](FRESH_INSTALL_UAT_MATRIX_2026_09_18.md). No additional production code changed in this checkpoint; Bandit is not applicable to these documentation and JavaScript-only UAT artifacts.
