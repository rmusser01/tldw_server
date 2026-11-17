# Playground fixture contracts — TASK-13620

The cockpit and OpenUI fixtures now retain current settings exports, store revisions and controller methods. Each cockpit case owns its QueryClient and clears mutated model, actor and session stores. Memoized history context preserves identity across unchanged renders; an unstable mock previously restarted attachment restoration with a new empty history array until heap exhaustion.

OpenUI uses the real model store and catalog query lifecycle. Standalone SDK source cases declare a private fresh Scene locator without native H1 readiness. Actual capture helpers fence the original draft, session, command, account key, snapshot identity and authority signals. The Scene I/O fixture returns the captured target; new controls verify the receipt reaches real actions and retires with authority, and failed reads retain raw input/messages without retrieval or generation.

Settings-action fixtures retain ordinary/fork/pending routing, original rendered-source guards and acknowledged assistant publication. Four positive setter expectations now require the actual currentness option; all original behavioral and failure assertions remain. Independent review found invocation-time ownership acquisition could conceal a stale-target write. Two red controls reproduced that defect; rendered capture fixes different-chat and same-ID New retirement before any I/O.

Verification: the combined three-suite run passed145 cases; after the two retirement controls, the final cockpit suite passed31/31 while unchanged shell63 and OpenUI53 remain qualified (147 cases total). Model persistence/picker/synchronization regression30/30 passed. Client types and exact-file hooks are recorded in the task. Temporary diagnostics were removed. Some diagnostic runs crashed while identifying fixture contracts; no heap or timeout increase is committed.

ADR assessment: no new ADR; test contracts match existing shared ownership and query behavior. Bandit is not applicable to test/Markdown-only scope. This bounded review does not approve the whole PR or establish installed-extension, physical accessibility, PostgreSQL, public H2 or complete UX acceptance. Final whole-PR review, required statuses, rebase and merge remain open.

PR: https://github.com/rmusser01/tldw_server/pull/3229
