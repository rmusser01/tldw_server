# Cycle4 targeted native coverage audit

Read-only audit of the latest cycle4 tracker, implementation plan and retained targeted evidence. No repository edits, browser actions, inference or runtime changes were performed. This is an artifact audit, not an independent native execution or full-UAT sign-off.

## Reviewed planning records

- `/Users/macbook-dev/Documents/GitHub/tldw_server2/IMPLEMENTATION_PLAN_uat_cycle_4.md`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/Docs/Design/2026-09-16-uat-cycle-4-repairs.md`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/Docs/Reviews/FRESH_INSTALL_SINGLE_MULTI_UAT_TRACKER_2026_09_14.md`

Reports were treated as observations. New timestamped native results were distinguished from retained historical failures and earlier pending-status paragraphs.

## Confirmed native claims

### UAT105 — saved source with analysis warning

The terminal job response is `completed` with nested `Warning`, `media_id: 2`, `error: null` and two identical Ollama analysis-failure warnings. The actual results snapshot and screenshot show one deduplicated warning, **0 succeeded / 1 saved with warnings / 0 failed**, and Open in Media. The destination snapshot is `/media?id=2`, with the Willow source intact and “No analysis yet.” This supports source-preserving warning acceptance, not successful analysis.

Evidence:

- `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-single2/uat-cycle4-targeted2-single-willow-terminal.json`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-single2/uat-cycle4-targeted2-single-willow-warning-result.txt`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-single2/uat-cycle4-targeted2-single-willow-warning.png`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-single2/uat-cycle4-targeted2-single-willow-media.txt`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/warning-ui105/cycle4-uat105-warning-ui-independent-review.md`

### UAT060 — Review Markdown

The Review screenshot and semantic snapshots show the saved Juniper heading and five bullet facts rendered as Markdown, including after normal reload/reselection. The earlier HMR-overlapping selection is correctly retained as inconclusive rather than a product failure or pass.

Evidence:

- `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-single2/uat-cycle4-targeted2-single-review-markdown.txt`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-single2/uat-cycle4-targeted2-single-review-markdown.png`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-single2/uat-cycle4-targeted2-single-review-markdown-reopened.txt`

### UAT109 — timeout and same-page recovery

The initial request-timeout attempt is correctly classified as a positive cited-answer control, not a timeout. The later streaming-idle failure shows local “Search timed out” / “Failed search” guidance. The same-page retrieval-only recovery completes with four sources, removes the assertive timeout copy and correctly suggests enabling generation. The audit verifies these saved states; console/no-overlay observations remain those recorded by the native runner rather than a new execution by this auditor.

Evidence:

- `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-single2/uat-cycle4-targeted2-single-qa-positive.txt`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-single2/uat-cycle4-targeted2-single-qa-idle-timeout.txt`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-single2/uat-cycle4-targeted2-single-qa-retrieval-recovery.txt`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-single2/uat-cycle4-targeted2-single-qa-timeout-requests.txt`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-single2/uat-cycle4-targeted2-single-running.md`

### UAT116 — generation requested but no answer returned

The saved actual request has `enable_generation: true`, `max_generation_tokens: 50`, and `search_mode: fts`. The result snapshot/screenshot retains four sources and shows “No generated answer,” Retry search and Review generation settings, rather than disabled-generation guidance. The settings-action capture opens the real settings panel and restores Balanced. This is truthful handling of negative output, not an answer-quality pass.

Evidence:

- `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-single2/uat-cycle4-targeted2-single-qa-short-request.json`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-single2/uat-cycle4-targeted2-single-qa-short-answer-config.txt`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-single2/uat-cycle4-targeted2-single-qa-short-answer-result.txt`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-single2/uat-cycle4-targeted2-single-qa-missing-answer.png`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-single2/uat-cycle4-targeted2-single-qa-settings-restored.txt`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-single2/uat-cycle4-targeted2-single-qa-final-requests.txt`

## Manifest verification

Read every entry from each listed `evidence-manifest.json`, resolved its file and independently recomputed SHA-256. **179 entries matched; zero missing files or hash mismatches.** This verifies listed entries, not completeness of every possible observation. Recorded credential-scan results were read; a new credential scan was not performed.

| Manifest | Entries | Missing/hash mismatches |
| --- | ---: | ---: |
| `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-single2/evidence-manifest.json` | 77 | 0 |
| `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-multi/evidence-manifest.json` | 57 | 0 |
| `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-retry108/evidence-manifest.json` | 22 | 0 |
| `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/warning-ui105/evidence-manifest.json` | 23 | 0 |

## Remaining targeted native boundaries

These are coverage gaps in already-approved work, not newly observed product failures. Root will execute them after the UAT108 freeze, before the next full fresh run.

1. **UAT102: Prompt synchronization failure feedback.** Native artifacts cover successful Prompt sync (POST201/Synced) and successful admin creation. They do not exercise the repaired Prompt-sync failure feedback path. Relevant existing evidence: `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-multi/prompt-visible-save.txt`, `prompt-after-save.txt`, `admin-created.txt`, and `admin-create-console.txt` in the same directory.
2. **UAT058: late Chat title publication after Settings navigation/reconnect.** Native artifacts verify ordinary Prompts and Characters titles. They do not exercise the late-completion/Settings boundary. Existing title controls: `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-multi/prompts-title.txt` and `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-multi/characters-cold.txt`.
3. **UAT117: newly completed reasoning-only response.** The native check restores an existing reasoning-only transcript, confirms missing-final-answer recovery presentation and retained reasoning, and explicitly performs no Retry/Continue generation. It does not yet exercise a new reasoning-only completion through the repaired completion boundary. Existing evidence: `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-multi/reasoning-chat-restored.txt`, `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-multi/reasoning-canonical-rows.txt`, `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-multi/reasoning-recovery.png`, and the explicit limits in `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-multi/FINAL_REPORT.md`.

The already-known **UAT108 failure** remains accurately recorded in `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-retry108/cycle4-uat108-native-multi-final-report.md`. The **UAT114 hidden-tab native-tool limitation** remains explicit in `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-repair-verification-2026-09-16/native-multi/native-harness-limitations.md`; visible-tab Prompt success does not establish hidden-stream release/catch-up.

After UAT108 settles, Stage5 still requires final combined checks, independent integration review and a frozen-source checkpoint before the fresh single/multi matrices. No additional product failure was found by this audit.
