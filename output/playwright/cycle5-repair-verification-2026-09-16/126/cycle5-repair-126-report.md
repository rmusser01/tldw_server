# UAT126 controller implementation

Accepted tabs replace the canonical tab route after existing Scheduler confirmation; other query/hash context remains. Deck intent applies only when its incoming identity changes, preserving live deck choices on a tab echo. The actual storage regression proved a pending one-shot handoff was lost when a tab change aborted consume after storage removal. Consumption now stays keyed to token/account scope and cleans only the latest same-token route, preserving the newly accepted tab.

RED3initial plus one pending-handoff RED; GREEN90 tests/six suites. Actual MemoryRouter/HashRouter, private source, editor changes, reload, history replacement, dirty confirmation, external deck and existing scope/private transfer controls pass. Scoped ESLint0errors/0warnings. A failed initial pending-test act wait is retained as a harness error, not product evidence. Native acceptance and independent review pending.

Base ba5233a5a9. No runtime/browser/inference. Bandit not applicable (TypeScript only); integrated compiler comparison remains separate. Permanent source/test scope is frozen for review.
