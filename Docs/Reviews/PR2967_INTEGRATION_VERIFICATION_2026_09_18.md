# PR2967 integration verification

Tracking: TASK13260.219. Target: `dev`. The user requested merge preparation,
then UAT resumption starting with TestBot/UAT261. The exact-output criterion
remains unchanged; the next complete fresh-install matrix has not begun.

## Evidence storage

Generated `output/playwright/` artifacts are retained locally and excluded from
the PR. The cleanup removed 11,712 branch-added files from Git without deleting
their local copies or rewriting history. Historical tracker links refer to that
local archive. This summary and the running tracker retain reviewable outcomes.

## Completed checks

| Area | Result | Scope and limits |
| --- | --- | --- |
| Chat rejected Retry | 110 tests passed, zero skipped; independent review clear | Actual PostgreSQL409, checkout release, lock availability and unchanged history; operation, image and provider-order controls. Full HTTP cancellation is not newly tested. |
| Frontend fixtures | 294 UI tests plus7 character-selection tests passed, zero skipped | Real ownership/storage/query contracts retained. Local Node26 differs from CI Node20; hosted final-head CI remains required. |
| World Book consumers | 50 tests passed, zero skipped; bounded source review clear | Official PostgreSQL and SQLite; foreign/unassigned lore excluded from RAG, conversation materialization and Chatbook scope/name reads; original transaction and catalogue assertions retained. |
| World Book secondary operations | 111 focused tests passed, zero skipped | Real backend operations, rollback, literal LIKE search and PostgreSQL JSONB placeholder controls. Broader adjacent verification remains in progress. |
| Onboarding | Desktop and mobile2 tests passed, zero skipped | Fixture preserves the verified session owner while changing lifecycle state. |
| OpenAPI | Canonical fingerprint check passed | Isolated Python3.12.11, FastAPI0.136.3, Pydantic2.13.5;2097 paths/3207 schemas. |
| Backend shard coverage | No newly uncovered tests | 805 shards,4729 test files,130 pre-existing baseline exclusions. |
| Ingestion smoke fixture | Multipart opt-out control passed | Real upload reaches the original embedding-progress loop; that test still skips because its worker is unavailable locally. No completed-job claim. |

New regression failures and unsuccessful intermediate attempts remain in the
local receipts. An earlier broad dependency-matched run had252 passes,10
ownerless legacy fixture failures and one existing Resource Governor skip.
The fixture failures and subsequent transaction regression are corrected in
the50-case consumer/read result above. The adjacent snapshot/Chatbooks/RAG
run completed with287 passes and four stale-fixture failures. All four focused
reruns now pass after independently reviewed fixture corrections; no assertions
were weakened. The previously passing287 cases were not repeated.

Scoped Python compilation and new-regression Ruff checks pass. Bandit reports
zero findings/errors in the six World Book production files; the Chat repair
retains only the unchanged random-jitter B311. TypeScript is outside Bandit's
supported scope. These checks do not constitute whole-repository security
certification.

## Remaining merge gates

- Resolve the extension Prompt Improvement timeout after reviewing all three
  failed attempts. Another full reproduction was rejected by automatic approval
  review under the stop-and-reassess rule; static reassessment is underway.
- Run final candidate CI after pushing the reviewed repairs.
- Obtain the requester's own human-written Change summary explaining what
  changed and why, as required by the repository merge policy.
- Merge normally, verify the remote result, and resume UAT with261 first.

Existing PostgreSQL World Books without recorded ownership remain intact but
hidden until trusted maintenance assignment. See the
[upgrade guide](../Deployment/Database/world-book-owner-upgrade.md).
