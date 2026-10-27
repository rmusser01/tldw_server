# Native storage area controls — TASK-13645

The existing test fault helper now reaches captured native local or sync APIs, retaining local as its default. Original API delegates are captured per area; a real Storage instance carries native call context. WebUI prototype delegation retains the actual caller instance. Three shared fixtures install the native API before source clients; toolbar faults explicitly target sync, owner verification targets local, and OpenUI handoff still uses real write/reread verification.

Two physical controls failed before the area repair. They now hold real sync reads/publication while sibling local access remains usable. The write control checks the actual prefixed physical key, unsettled acknowledgement before release, settled acknowledgement and exact serialized bytes afterward. Existing enumeration, legacy values, runtime retirement, watchers and refusal assertions remain.

The first complete native run passed933 and failed1, revealing the genuine Pro hydration issue tracked in TASK-13646. After that connected source repair, native and WebUI each pass986tests across10suites; the final affected native controls pass7 with41unselected. Both normal source compilers pass. Independent bounded review is clear. Bandit N/A TypeScript scope; exact hooks are recorded in both tasks.

ADR required:no; test boundary declaration, no production storage key/area or ownership policy change. The two tasks are committed together because their fixtures and source regression require each other to remain green. Whole-PR findings, final independent review, rebase and merge remain pending.

PR: https://github.com/rmusser01/tldw_server/pull/3229
