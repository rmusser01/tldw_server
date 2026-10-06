# Media follow-up fixes and latest-dev validation — 2026-10-06

All four findings from [the live validation](2026-10-05-media-live-validation.md) are fixed and verified. This closes TASK-13504–13507. Native screen-reader speech and human participant comprehension remain unverified; this report does not claim a full usability study.

## Source and implementation

Latest remote dev: `1fc353c3f67c93ba05102e7b0136ac4acac8f510`, refetched unchanged after final source verification. Tested source: `9a277cf162321753321ffa0448a51e73464d7150`, on `codex/media-live-ux-fixes-20261006`. The original Media implementation was already merged through PR 3194; this branch carries the subsequent validation evidence and bounded follow-up fixes.

ADR required: **no**. Existing detail/session/job contracts, account ownership checks, ingestion target selection, rich-content sanitizer and ICU localization remain the governing boundaries. [ADR-059](../ADR/059-backlog-py-task-editor-cutover.md) governs task editing. No new dependency, API, schema, persistence format or provider behavior was introduced.

| Finding | Final behavior and rationale | Verification |
|---|---|---|
| TASK-13504: generic reading labels | Shared detail fetch normalizes `source.title`/`source.type` and request identity. Reading, comparison and export retain actual identity even off the current list page. This repairs recognition and consistency using the existing DTO. | Actual API sources 3/4/5 show Example Domain, batch-a and batch-b; fresh saved IDs 9/10 show fix-batch-a/fix-batch-b. Nested/legacy DTO and off-page export regressions pass. |
| TASK-13505: raw ingestion envelope | Single/multiple reading, comparison, transcript segmentation and read-aloud use the body of a valid leading JSON envelope. Generated metadata-only sections are omitted; article character targets, statistics and reading progress align with the displayed body. Raw source remains available to editing, analysis, note/flashcard actions and export. Malformed envelopes and ordinary metadata examples remain intact. | Real stored source 3 still has the envelope and API count 48. The reader shows the article, 25 words, 156 characters and one paragraph. Desktop/mobile checks and raw-export/read-aloud/parser/navigation regressions pass. |
| TASK-13506: misleading resumed confirmation | Configure and Review consume the same actual retry targets as processing. Prior saved, completed-without-ID, skipped and duplicate outcomes stay distinct. Missing File handles on untargeted prior successes do not turn those successes Invalid. | Reloaded mixed import: Configure says one eligible item; Review says one item, with two Saved exclusions, one Skipped exclusion and one duplicate exclusion. Network observation proves one POST accepted one job, ID 16, for recoverable.pdf. Saved IDs 9/10 and their markers remain intact. |
| TASK-13507: singular action copy | Existing ICU translations and fallback names agree for one/multiple saved items in Results and Recent imports; Configure uses singular eligible-item copy. | Actual one-item Results: “Review this 1 saved item”; history: “Review 1 saved item.” The mixed import uses plural for two. Accessible names and real consumed translation namespaces pass one/three-item checks. |

The heuristic rationale follows the [NN/g mapping in the original report](2026-10-05-media-live-validation.md#remaining-findings-and-solutions): recognition rather than recall, consistency, minimalist presentation and visibility of system status. Severity remains historical workflow impact, not a claim about which commit introduced the issue.

## Real workflow evidence

All browser mutations used the throwaway isolated account on WebUI/API/Redis ports 18881/18882/18883. The API used actual ingestion workers with `MINIMAL_TEST_APP=1`, `TEST_MODE=false`, `TESTING=false`; no request interception or canned API responses. Existing test models and databases were reused from the prior validation. Ancillary routes omitted by the reduced router set and local preview restart/HMR messages do not establish a full production-server pass.

1. A new Markdown item containing `silver-panda-620` completed as media 8. The singular saved-set action opened exactly that selection.
2. A fresh five-input batch used two Markdown files, an invalid PDF, a web URL and its duplicate. It produced **2 saved / 1 failed / 2 excluded or skipped**: documents 9/10 saved; the existing web source skipped; its duplicate excluded. No skipped URL was labeled Saved.
3. Reload preserved outcomes and required original-name/size PDF reattachment. After reattachment, correction confirmed one target, submitted exactly one PDF job and retained prior saved outcomes. The intentionally invalid PDF failed again; this proves retry scope, not successful processing of corrupt input.
4. Reviewing the saved set opened only 9/10, with markers `copper-fox-621` and `blue-jay-622`. Existing real web/document set 3/4/5 retained actual titles and clean article text on desktop and 390×844 mobile; document-width overflow was zero.
5. The single-item mobile check used the Content tab and also verified the expanded chapter panel contains no ingestion envelope. Its expanded navigation occupies vertical space; the article remains in the content scroller. The final screenshot and receipt preserve that state rather than implying a participant-tested mobile hierarchy.

Final source edits initially left the running preview on stale reading controls. Restarting the isolated preview resolved this; the final receipt asserts clean sections and the corrected 25-word count. No application workaround was added for the test environment.

Durable [receipts.json](../../output/playwright/media-live-fixes-20261006/receipts.json) contains actual DTOs, sanitized accepted-job receipts, named UI snapshots, source IDs, fixture markers, mobile overflow, verification and cleanup. The mixed-import accepted receipt is explicitly the first file response; its complete outcome is established separately by the actual Results UI.

Screenshots:

- [Desktop saved-set identity](../../output/playwright/media-live-fixes-20261006/saved-set-desktop.png), [mobile saved-set reading](../../output/playwright/media-live-fixes-20261006/saved-set-mobile.png)
- [Single reader](../../output/playwright/media-live-fixes-20261006/single-reader.png), [single mobile controls](../../output/playwright/media-live-fixes-20261006/single-reader-mobile.png)
- [Singular saved result](../../output/playwright/media-live-fixes-20261006/single-saved-results.png), [singular history](../../output/playwright/media-live-fixes-20261006/single-import-history.png)
- [Mixed outcomes](../../output/playwright/media-live-fixes-20261006/mixed-saved-results.png), [resumed Configure](../../output/playwright/media-live-fixes-20261006/resumed-configure.png), [resumed Review](../../output/playwright/media-live-fixes-20261006/resumed-review.png)

## Verification and limits

- **17 files / 411 tests passed**, 54.93 seconds, with the repository CI timeout of 15000 ms. Includes session/authority fencing, nested identity, reading/keyboard scope, raw export, metadata/parser, read-aloud, navigation and actual ICU resources. Local Node 26 differs from CI Node 20; this is scoped local evidence.
- WebUI `bun run typecheck --incremental false`: passed separately from the build, which skips type validation by project configuration.
- WebUI `bun run lint`: exit 0, zero errors and the same 180 baseline warnings. No warning-free claim.
- WebUI `TLDW_INTERNAL_API_ORIGIN=http://127.0.0.1:18882 bun run compile:prod`: passed; token sync passed; shared app 589.6 KB gzip under the 600 KB budget.
- Chrome extension `bun run build:chrome:prod`: passed, 46.0 seconds, 49.26 MB. Shared components are tested and both clients compile; packaged capture gestures, all-browser extension testing and a complete all-pages E2E run were not repeated.
- Canonical Backlog normalization check and the task-format ratchet passed (4 tests).
- Independent source review found three material presentation/test-resource issues during implementation; regressions reproduced and corrected all three. The final reading-control delta also received a bounded independent review with no material findings.
- Required Bandit invocation covered 23 touched TypeScript files: zero findings, **23 parse errors** because Bandit cannot parse TypeScript. No Python application code changed. This is not a TypeScript security pass. Parser validation, unchanged sanitization, account ownership fences and saved-identity distinctions were reviewed and covered by affected tests.
- No full backend suite, new live axe pass, human novice/power-user study or verified VoiceOver speech pass is claimed. The earlier native AX/keyboard check and VoiceOver speech limitation retain their original evidence boundary in the prior report.

## Cleanup and review gate

The task browser and servers are stopped; ports 18881/18882/18883 have no listeners. Four task-only dependency symlinks, the standalone tracing copy and generated WebUI/extension build caches were removed. Test models remain in the temporary validation directory for reuse. The managed source worktree is retained for PR review.

Original tracked status matches its before snapshot, and every earlier untracked entry remains present. Exact full porcelain bytes differ because another workstream added 157 files under its own `.venv-uat-py312-20261006`; those files and the unrelated open browser were left untouched. No checkout/reset/clean/staging operation targeted the original workspace. Git's existing loose-object warning was left alone.

The completed task-specific implementation plan will be removed during PR finalization per repository guidance. No implementation item remains open in TASK-13504–13507. Merge still requires the requester’s own `Change summary` under [the repository policy](../superpowers/AI_GENERATED_PR_CHANGE_SUMMARY_POLICY_2026_04_17.md): “Every AI-generated pull request must include a human-written `Change summary`.” The PR is prepared as a draft; the human summary for earlier PR 3194 is not reused as ownership of this follow-up.
