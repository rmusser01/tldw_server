# Shared composer selector equality — TASK-13621

The shared composer now uses the existing shallow comparator for its composite model-settings selection. Every selected primitive remains compared, while unchanged values retain a stable projection. Direct subscriptions, actual setting updates and captured-state/revision checks remain intact.

The actual-store first-mount regression exhausted memory. Bounded phase/state traces and a V8 profile ruled out setup and model writes; equality-only delegation completed the same case in250ms. The production correction removes the diagnostic adapter and uses shallow directly in the shared component. Independent source review found no actionable issue.

Verification: actual-store OpenUI/source53 cases, cockpit31 and shell63 are qualified with their repaired fixtures; model persistence/picker/synchronization30 regressions pass. Both normal client type configurations and hooks are recorded in the task. No production heap/timeout increase or admission relaxation was made.

ADR assessment: no new ADR; follow the existing composite selector convention used by the current settings panel. Bandit is not applicable to TS/Markdown-only scope. This is bounded regression evidence, not overall PR or complete UX acceptance. Final build and whole-PR checks remain required before merge.

PR: https://github.com/rmusser01/tldw_server/pull/3229
