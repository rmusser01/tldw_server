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
| Further frontend readiness fixtures | 46 passed, four local timeouts on exact CI Node 20.20.2 | Study Pack 12/12 and Companion Home 17/17 pass; Admin 11/12 and Settings 6/9. Three Settings cases and an unchanged Admin password case exceeded their original five-second budgets during severe unrelated worktree CPU load. The failures remain unresolved pending hosted CI; no local timeout increase or further retry. |
| Notes remediation | All 32 tests passed, zero skipped | A further Stage4 fixture now supplies the verified owner required for saving; revision and attachment assertions are unchanged. |
| World Book consumers | 50 tests passed, zero skipped; bounded source review clear | Official PostgreSQL and SQLite; foreign/unassigned lore excluded from RAG, conversation materialization and Chatbook scope/name reads; original transaction and catalogue assertions retained. |
| World Book attachment UI | All 10 tests passed, zero skipped, unchanged time limits | Retry uses the rendered error/button and invalidates both original queries. Named-panel queries reduce unrelated DOM work; the final local retry case took 2504 ms against a 5000 ms limit. Hosted CI remains required. |
| World Book secondary operations | 111 focused tests passed, zero skipped | Real backend operations, rollback, literal LIKE search and PostgreSQL JSONB placeholder controls. Adjacent results are qualified below. |
| Onboarding | Both desktop and mobile tests passed, zero skipped | Fixture preserves the verified session owner while changing lifecycle state. |
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

## Remaining merge gates

- Verify the reviewed extension Prompt Improvement fixture correction after
  explicit approval for one isolated, trace-enabled replay. Automatic approval
  review rejected another full reproduction under the stop-and-reassess rule;
  static reassessment is complete, but the correction remains untested and uncommitted.
- Run final candidate CI after pushing the reviewed repairs.
- Confirm the reviewed frontend readiness fixtures in hosted CI. The four
  local timing failures above are not passes, and the affected cases must
  complete under CI's unchanged time limits. Original failures and every
  intermediate attempt remain in the local archive.
- Obtain the requester's own human-written Change summary explaining what
  changed and why, as required by the repository merge policy.
- Merge normally, verify the remote result, and resume UAT with UAT-261 first.

Existing PostgreSQL World Books without recorded ownership remain intact but
hidden until trusted maintenance assignment. See the
[upgrade guide](../Deployment/Database/world-book-owner-upgrade.md).
