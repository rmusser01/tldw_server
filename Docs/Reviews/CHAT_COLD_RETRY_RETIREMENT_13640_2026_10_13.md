# Cold Retry retirement qualification — TASK-13640

The retirement harness now waits for the replacement account's durable record before taking its write baseline. It reuses the existing recovery-write helper, which permits only model-cache tombstones whose exact shape is asserted. Every other write remains counted. The test additionally checks the original Alice record, Bob record and account revision remain unchanged after the held Alice request rejects.

No production fence changed. The shared hook already rejects retired generations, unmount, aborted requests and account revision changes, and legacy routes use that hook. The original isolated4 cases and full271-case suite passed locally; key-only instrumentation found no extra write. Therefore this change does not identify the original CI sixth key or claim a production defect fixed. It makes the persistence baseline and permitted cache cleanup explicit while strengthening the chat invariants.

Final native and WebUI ownership/freshness suites each pass277tests. Independent bounded review is clear. Normal client compilers already passed at the immediate parent; this test-only delta changes no source compiler inputs. Bandit N/A TypeScript test-only scope. Normal exact-file hooks and next-head CI are recorded separately. The whole PR, final independent review, rebase and merge remain pending.

ADR required: no; existing account/revision ownership policy, harness qualification only. PR: https://github.com/rmusser01/tldw_server/pull/3229
