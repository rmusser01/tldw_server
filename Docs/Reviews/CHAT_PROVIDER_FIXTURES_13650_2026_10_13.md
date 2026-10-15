# Provider fixture contracts — TASK-13650

Original native fixtures reproduced5failures/26passes. No production behavior changed. The history harness now executes the actual displayMessages, inferenceIds and selected declarations with the actual formatter/diagnostic filter. Its declared capture supplies IDs, parent relationships, view and owner snapshot needed by those pure helpers. Existing native canonical-content and local image-event assertions remain; new native/local diagnostic-pair controls prove inference exclusion applies only to undispatched local diagnostics and preserve the complete source capture.

The pre-abort case still requires AbortError and now correctly requires zero fetch/runtime IO. Dispatched-abort and all owner/path admission controls remain. Model refresh declares asynchronous runtime transport; null and rejection both exercise the real warm-cache fallback and shared Query refetch. Success and repeated metadata failure retain their opposite notification assertions, with exact runtime request checked.

Final native/WebUI each35passes3suites0skipprocess0. Independent bounded review clear. Existing-config ESLint0findings/0errors across3files; whitespace clean. Bandit N/A TypeScript tests only. Normal exact-file hooks required. Projection fixture evidence is not H1 admission or native physical persistence evidence.

ADR assessment:no newADR; current inference, cancellation and discovery contracts retained. PR:https://github.com/rmusser01/tldw_server/pull/3229. Other remaining issues, whole-PR review, final rebase and merge pending.
