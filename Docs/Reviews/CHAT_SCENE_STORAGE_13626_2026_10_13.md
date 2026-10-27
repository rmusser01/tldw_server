# Native scene and settings fixture dependencies — TASK-13626

Scene image preflight, Compare service prompts and settings ownership tests now declare the actual native browser storage API and reuse the qualified physical read controls. Shared fixture setup restores the declared API before each case, preserving explicit retirement within a case. Two controls prove one case can retire the runtime and the next restores its captured storage. The prior normal-history nested restoration is now handled centrally.

The Compare positive also lacked its local database transaction dependency. Spying userSettings.get alone could not make native Dexie enter a transaction without IndexedDB. It now uses the existing retaining memory schema and callback transaction, while actual profile verification, Scene copying and bind guards remain intact. Temporary primitive-only diagnostics confirmed the account/destination/preparation leases were current and were removed after diagnosis.

Native reproduction was69failed23passed; the storage repair reached100passed1failed before the DB dependency was fixed. Final seven-suite native and WebUI runs each pass930 tests, including all prior history and physical controls. Both client type checks pass. Independent bounded source review is clear. Normal exact-file hooks are recorded in the task.

ADR required: no; these are declared test dependencies, with no production storage, ownership, admission, API or persistence changes. Bandit N/A TypeScript test-only scope. Installed-extension qualification, other whole-PR CI/UX/H2 work, final independent review, rebase and merge remain open.

PR: https://github.com/rmusser01/tldw_server/pull/3229
