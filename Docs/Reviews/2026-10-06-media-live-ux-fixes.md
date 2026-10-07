# Media follow-up fixes and latest-dev validation — 2026-10-06

All four findings from [the live validation](2026-10-05-media-live-validation.md) are fixed and verified. This closes TASK-13504–13507. Native screen-reader speech and human participant comprehension remain unverified; this report does not claim a full usability study.

Draft PR: [#3204](https://github.com/rmusser01/tldw_server/pull/3204).

## Source and implementation

Dev baseline at the completed source validation: `1fc353c3f67c93ba05102e7b0136ac4acac8f510`, refetched unchanged after that verification. Tested source: `9a277cf162321753321ffa0448a51e73464d7150`, on `codex/media-live-ux-fixes-20261006`. The original Media implementation was already merged through PR 3194; this branch carries the subsequent validation evidence and bounded follow-up fixes.

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
- CI exposed one missed integration expectation for the newly correct singular Configure label. The existing test failed before the one-line assertion correction; the full wizard integration file then passed **53/53 tests**, 12.06 seconds, with the same 15000 ms timeout. Application source did not change in this follow-up. The changed test passes the PR’s ESLint rules with zero errors; all 33 warning messages match the prior head. Bandit was invoked on the changed TypeScript test with zero findings and one unsupported-language parse error.
- [CI run 37492771649](https://github.com/rmusser01/tldw_server/actions/runs/37492771649), on head `8e1363e1cb69c08858e3760deeb2feaf02f0456d`, also reported `core-route-identity` expecting “First-time setup.” That failure reproduced on the exact dev baseline in both failed-file and shard-context replays; the final ratchet rejected differing test identities/order. It is not a Media regression. Required CI must be confirmed on the new head; no all-green CI claim is made.
- WebUI `bun run typecheck --incremental false`: passed separately from the build, which skips type validation by project configuration.
- WebUI `bun run lint`: exit 0, zero errors and the same 180 baseline warnings. No warning-free claim.
- WebUI `TLDW_INTERNAL_API_ORIGIN=http://127.0.0.1:18882 bun run compile:prod`: passed; token sync passed; shared app 589.6 KB gzip under the 600 KB budget.
- Chrome extension `bun run build:chrome:prod`: passed, 46.0 seconds, 49.26 MB. Shared components are tested and both clients compile; packaged capture gestures, all-browser extension testing and a complete all-pages E2E run were not repeated.
- Canonical Backlog normalization check and the task-format ratchet passed (4 tests).
- Independent source review found three material presentation/test-resource issues during implementation; regressions reproduced and corrected all three. The final reading-control delta also received a bounded independent review with no material findings.
- Required Bandit invocation covered 23 touched TypeScript files: zero findings, **23 parse errors** because Bandit cannot parse TypeScript. No Python application code changed. This is not a TypeScript security pass. Parser validation, unchanged sanitization, account ownership fences and saved-identity distinctions were reviewed and covered by affected tests.
- No full backend suite, new live axe pass, human novice/power-user study or verified VoiceOver speech pass is claimed. The earlier native AX/keyboard check and VoiceOver speech limitation retain their original evidence boundary in the prior report.

## Current-dev integration for PR 3204

The requester authorized the current-dev update and merge on 2026-10-06 PT. The merge queue variable was unset. Rebase onto dev `005802bdb070fd68e087c4db3f831c33bef07c39` completed without conflicts; tested rebased source was `5a8ea9b4b0bb367745f7a17647c48fed1dd1ee1c`. All 26 application files changed by this PR retain their pre-rebase contents. Existing dev changes supply the updated offline email fixture; no additional application fix was introduced for integration.

- **18 Media test files / 464 tests passed**, 56.18 seconds, with the 15000 ms CI timeout.
- **46 email integration tests passed**, 9.80 seconds, including the previously failed attachment-upload case and the current fixture’s guard/accounting regressions. The old head’s broader ingestion shard had 239 passes, one skip and one teardown error; the corrected fixture comes from dev.
- WebUI typecheck passed; lint had zero errors and 180 baseline warnings. Chrome production build passed in 48.4 seconds, 49.5 MB.
- WebUI production build, token sync and shared bundle budget passed: **595.6 KB gzip / 600 KB**. The initial local Node 26 build exhausted its 4 GB heap; the successful invocation set `NODE_OPTIONS=--max-old-space-size=8192`, matching the existing typecheck allowance. No source/config change was needed for that local environment limit.
- Current-base task-format ratchet: four tests passed. Required Bandit invocation on 24 changed TypeScript files: zero findings and 24 parse errors; this remains an unsupported-language limitation, not a TypeScript security pass.

Four temporary dependency links and task-generated build/tracing directories were removed before publishing. No browser or test servers were started for this integration pass. Actual browser/API/model evidence above remains tied to its original revision; it was not rerun on this new base. Final landing requires all seven protected checks on the pushed head and confirmation that dev has not advanced. The requester-owned Change summary remains verbatim on the PR.

## Cleanup and review gate

The task browser and servers are stopped; ports 18881/18882/18883 have no listeners. Four task-only dependency symlinks, the standalone tracing copy and generated WebUI/extension build caches were removed. Test models remain in the temporary validation directory for reuse. The managed source worktree is retained for PR review.

Original tracked status matches its before snapshot, and every earlier untracked entry remains present. Exact full porcelain bytes differ because another workstream added 157 files under its own `.venv-uat-py312-20261006`; those files and the unrelated open browser were left untouched. No checkout/reset/clean/staging operation targeted the original workspace. Git's existing loose-object warning was left alone.

The completed task-specific implementation plan is removed per repository guidance. No implementation item remains open in TASK-13504–13507. The requester supplied a new human-owned `Change summary` for PR 3204; it was saved verbatim and verified against the PR body, satisfying [the repository policy](../superpowers/AI_GENERATED_PR_CHANGE_SUMMARY_POLICY_2026_04_17.md). The requester subsequently authorized updating and merging this PR. It will be marked ready for review after the current-dev verification above; landing remains subject to the seven protected checks on the pushed head. The integration assertion follow-up temporarily reused three dependency links; those links are removed before committing. No new test services or browser were started.
