# Browser storage absence and SDK Scene ownership — TASK-13616

The browser storage shim returned requested missing keys as null. Native Plasmo consequently exposed a present null record instead of undefined absence, and the private Scene registry correctly refused it. Storage reads now omit only keys absent from the physical backend, while preserving stored JSON null, zero, false and empty strings. This follows the [browser StorageArea.get contract](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/storage/StorageArea/get). Enumeration also skips a key that disappears before its value is read.

Four SDK suites now declare a scoped Chrome storage API before source imports while retaining installed native Plasmo and real private Scene reads. Physical roundtrip, denied writes, API retirement and verified absence are tested. No production persistence guard or H1 authority is relaxed. The wider bootstrap fixture now uses the real Navigator with a restorable locks property; its former prototype clone broke the WebIDL userAgent receiver when Zod loaded.

## Evidence

- Initial four SDK suites:68failed because no captured storage API existed.
- API adapter controls passed, but three speculative ordering attempts still failed. Capability-only diagnostics then established a valid captured client and registry_record_invalid from missing-key null. Diagnostics and unnecessary hoisted setup were removed; no guard override remains.
- Browser missing/null/falsy controls before production repair:3failed7passed.
- Final actual native SDK/Scene and adapter controls:5suites72passed, no unhandled errors.
- Wider browser/bootstrap/Plasmo/watch/model-save callers:6suites112passed, including all37bootstrap auth/logout/ACK controls.
- Both normal Node26 client types:passed after final files.
- Locked Bun1.3.2/Next16.1.4 production/quickstart Turbopack build and token sync:passed. Shared589.7KB/heaviest868.4KB within unchanged600/900KB caps.
- Bounded independent review cleared the corrected API semantics; final Navigator fixture review is recorded in the task.

ADR assessment:no new ADR; this corrects existing browser API compatibility and retains the established fail-closed absence rule. Bandit not applicable:TypeScript/TSX and Markdown-only changes.

The scoped API is a test simulation backed by physical window storage, not installed-extension acceptance. Docker execution, physical AT and whole-PR/H2 acceptance remain open. Final requesting-code-review, rebase and merge remain pending. Completed unique plan retired to /private/tmp/IMPLEMENTATION_PLAN_sdk_scene_native_storage_13616_completed.md. PR:https://github.com/rmusser01/tldw_server/pull/3229.
