# Chat model save recovery — TASK-13613

The Full and Compact model pickers observe acknowledged preference saves and show a failed-save warning with direct Retry. Retry retains the original selection revision; later selections, including same-value ABA, retire the warning and cannot grant a stale write. Awaiting editors still receive the original rejection.

Compact keeps its inline picker owner mounted when a valid model hides the trigger. Save recovery remains visible during catalog loading and discovery failure. ModelSelect uses the focused model hook instead of importing the full message controller.

Real WebUI persistence tests now live under the owning WebUI runner. Native shared-package aliases were not globally replaced. Affected ordinary-chat validation fixtures explicitly declare their roleplay context and focused model-hook dependency; storage failure injection targets the physical storage instance's prototype.

## Evidence

- Real preference failure controls: initial 2 failed / 18 passed.
- Compact conditional lifetime and catalog controls before repair: 4 failed / 2 passed.
- Owning WebUI storage, both pickers, actual Compact composer JSX, model validation and storage adapters: 8 suites / 97 tests passed.
- Unchanged native shared-package runner configuration: 3 suites / 52 tests passed.
- Normal Node26 WebUI tsconfig.json and extension tsconfig.compile.json: both passed after final fixture changes.
- Supported production/quickstart Webpack build: passed; token sync passed; shared app 596.4 KB gzip and heaviest route /chat 798.7 KB gzip, within unchanged 600/900 KB limits.
- Independent bounded source review found no remaining Critical, Important or Minor issue, including actual Compact JSX and fixture assertions.

## Limits and governing decisions

ADR assessment: no new ADR; this reuses the existing acknowledged preference operation and revision ownership contract. No storage protocol or durable architecture rule changed. Bandit is not applicable: only TypeScript/TSX, English locale JSON and tracking/review documentation changed.

Local Turbopack rejected the pre-existing dependency symlink pointing outside its filesystem root. Webpack also warned about copying two traced dependency files from the symlink layout, so standalone packaging is not qualified. The CI Turbopack bundle and full required gates remain open; Webpack results are not a claim of CI parity. No installed-extension or physical assistive-technology acceptance was performed. Whole-PR/H2 completion, final requesting-code-review, rebase and merge remain pending.

The uniquely named implementation plan completed all three bounded stages and was retired to /private/tmp/IMPLEMENTATION_PLAN_chat_model_save_recovery_13613_completed.md. PR: https://github.com/rmusser01/tldw_server/pull/3229.
