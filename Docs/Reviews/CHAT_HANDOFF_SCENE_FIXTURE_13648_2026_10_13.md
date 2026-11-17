# Handoff Scene fixture — TASK-13648

Original native suite reproduced6failures/71passes. Existing submit fixtures lacked the history controller, verified account and actual fresh draft required by real Scene capture. They now create an idle per-case controller with an irreversible captured AbortSignal, actual fresh address, model/Actor resets and account scope; queued dispatch gets the same controller. Real captureActorSceneSend is retained. Only Scene read IO is replaced, preserving its exact supplied target, actual defaults and target/lease refusal. Package storage mock retains original safeStorageSerde.

Two positive cases initially asserted before asynchronous Scene preparation settled. They now await actual dispatch and retain exact original payload assertions. Controls prove a held Scene read failure, retired account or new draft cannot dispatch or clear imported input/context; a captured receipt stays retired through a later fresh draft; a failed queued Scene read blocks the retained exact snapshot/source context.

Final native and WebUI suites each pass102tests across2suites,0skip,process0, including connected actual Scene target tests. Native3.15s,WebUI4.49s. ESLint0errors/0newfindings versus parent,6unchanged warnings; whitespace clean. Bandit N/A(TypeScript tests only). Independent bounded review clear; no source enforcement changed, no native physical Actor persistence claim. Normal exact-file commit hooks required.

ADR assessment:no newADR; current source ownership policy retained. PR:https://github.com/rmusser01/tldw_server/pull/3229. Whole-PR review, other remaining failures, final rebase/merge still pending.
