# Native history consumer storage — TASK-13624

Three native history consumer suites passed under the WebUI storage alias but failed under the shared UI runner, which loads actual Plasmo storage. They now install the existing declared browser storage fixture before source imports. Admission, recovery, retirement and original-row assertions are retained.

Fault controls also reach the actual captured browser storage client. The WebUI path retains its original instance backend. This is necessary because native Plasmo defines get/set on instances, while the WebUI shim uses prototype methods. The adapter delegates unkeyed enumeration unchanged and reuses the existing tolerant deserializer for legacy strings. Nested global cleanup restores the declared Chrome storage API for subsequent native cases. Physical held-read/write, enumeration coexistence and backend retirement controls qualify these boundaries.

Final native shared UI and WebUI runs each pass 836 tests: actions331, normal268, compact230 and physical storage7. Both normal client type checks pass. Independent bounded review is clear after addressing prototype-spy, cleanup, enumeration and legacy-value findings. Normal exact-file hooks are recorded in the task.

ADR required: no; this restores test-environment dependencies and changes no production storage, public API or admission behavior. Bandit is not applicable to TypeScript test-only scope. Remaining whole-PR frontend/UX/docs failures, H2 qualification, final whole-PR review, rebase and merge remain open. These browser-API fixtures do not constitute installed-extension qualification.

PR: https://github.com/rmusser01/tldw_server/pull/3229
