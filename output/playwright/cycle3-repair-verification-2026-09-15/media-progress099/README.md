# Media reading progress UAT099 — desktop pass, mobile open

TASK13260.40 under parent TASK13260. **Desktop reading-progress save and reload pass. Mobile acceptance remains unpassed: UAT101 / TASK13260.42 addresses clipped content and actions.** This is a compact evidence bundle, not full fresh UAT certification.

## Final native result

Root operated the existing single-user browser on UI18280/API18200. At1280×720, the repaired source body has clientHeight540 and scrollHeight835. Actual wheel160 moves its scrollTop0→160 while the document remains0. PUT returns200 with zoom_level100, percentage54.24 and cfi scroll:54.24 (2026-09-16T00:01:51.712Z).

Normal reload restores scrollTop160. Captured progress traffic contains only a GET200 returning percent_complete54.2/cfi scroll:54.24; there is no PUT during the recorded reload observation, which waited for restored position and then two seconds (00:06:49.217Z). The scrolled and reloaded PNGs were inspected and show the same source position and available controls. This does not prove indefinite absence of writes or console errors.

At390px, the mobile source panel measures739px wide; documentWidth remains390 because content is clipped. Actions starts at x431.64, Find at x704.19 and text-size M at x628.91, outside the viewport. The mobile PNG visibly confirms right-side content/control clipping. **This is a retained failure, not a mobile pass. AC2 remains open until TASK13260.42.** No source or browser correction occurred during retention.

## Original failures and correction

The original native report/geometry/wheel show an unbounded2361px source body: wheel560 moved the document while the source stayed0, and progress remained absent. The intermediate layout fix allowed an actual progress save, but normal reload returned54.2 then wrote0 without a user scroll. The exact intermediate GET, zero PUT payload and resulting zero response are retained. Large request histories and unrelated Flashcard checks are omitted from this compact selection; references in the historical report may point to private artifacts not copied here.

The five production files implement the existing constrained /media height chain, passive chapter highlighting without a seek, content-readiness gating, late restore cancellation after newer scroll, and pending progress snapshots through geometry/ref cleanup. The final seven-file manifest includes the two permanent regression test files. Both final manifests and independent pre/post hash checks are retained; all seven live hashes were also checked before this bundle was written. These files are the pre-.42 source state, not an assertion about later mobile changes.

## Automated review and static checks

Permanent actual-owner/hook RED logs are retained unchanged apart from declared trailing whitespace/final-newline normalization. Implementer and independent final runs both pass75tests/15suites; these are overlapping runs, not150 unique tests. Independent queued-frame controls add two cases, run with the nine existing hook cases (11/1). Their exact config and injected source bytes are retained. The review report records the initial temporary unstable-ref fixture error honestly; that harness-only log is omitted as unnecessary historical volume.

Independent review is CLEAR for the frozen scope. Root-scoped ESLint covers all seven paths:0errors,38unchanged warnings,0added. Comparison JSON and complete diagnostics without duplicated source bodies are retained. Original full ESLint input hashes are in eslint-diagnostics-compact.json. The root pages-directory advisory is unchanged. Combined compiler comparison at00:00:15 records90baseline/90current,0added/removed; this is not a clean typecheck. Scoped diff-check was clean. Bandit is inapplicable to the TypeScript-only change.

The permanent restoration regressions themselves were the original RED controls; there is no separate restoration replay config. The unrelated optional WebLayout auth-mock baseline is excluded and not counted as passing. No current source preimage is reconstructed or mislabeled as historical evidence.

## Provenance and integrity

INDEX.md is the exact inventory. retention-manifest.json maps copied inputs to original paths, source/retained hashes and any whitespace transformation. Original JSON, PNG and private probe/config source bytes are preserved; only Markdown/log/plain capture text may have trailing whitespace and final-newline normalization. SHA256SUMS covers every retained file except itself.

All three PNGs were visually inspected. All inputs and generated text were scanned privately against known isolated runtime credentials plus JWT/private-key patterns before writing; no matching value is exposed. Native capture text has no tool-level “### Error” blocks. The safe scan is a bounded artifact check, not a universal data-classification claim.

Retention changed only this task-associated evidence directory and official TASK13260.40 notes. No source, tests, browser, runtime, Git index/commit or global-document changes were made. Parent owns commit and the subsequent .42 repair. Task .40 remains In Progress; mobile and full-fresh acceptance remain open.
