# Final reviewed-source native Chat restoration checkpoint

Existing tasks TASK13260, TASK13260.33, TASK13260.34 and TASK13260.38. This is a bounded native evidence bundle, not full fresh UAT acceptance.

## Results and provenance

- **Single-user restoration passed** on the existing cycle3-single-20260915 browser, port 18280: original Aster Note 080ce690 → actual More actions / Open conversation; correct title, character 4, all 3 original messages and exact garden answer; two normal reloads; native readonly PageAssistDatabase owned mirror with canonical answerIDpa_ec2a-b676-39f-15e0; actual saved Robot mouse selection and Aster return. Captured independently by /root/provider_keys077 from 23:14 to 23:20 UTC. The retained single report gives exact IDs and evidence boundaries.
- **Multi-user saved Robot reload passed** in root's separate existing multi browser, port 18281, at 23:14:38UTC: correct Robot 5, two original messages, BEEP BOOP, loaded metadata and idle state. Root captured these checks; the retaining reviewer read the evidence without rerunning the browser.
- **Cedar title restored and reload passed.** Some intermediate layout/switch files show a temporary verified-rename title. Root restored the original title, Cycle3 Cedar Guide Chat (20260915_103705), through the actual title editor at 23:17:37UTC (PUT 200, version 5), then reloaded at 23:18:18UTC with correct original title, character 4, two messages and the original answer. The temporary title did not remain.
- **Desktop sidebar/composer checks passed within the recorded cases.** At 1280×720 the expanded sidebar has a 525 px scrollable middle and 138 px reserved footer; mouse Robot/Cedar navigation and keyboard Cedar activation succeed. 1024 px composer controls wrap within the viewport. Both modes' captures support actual saved row reachability; no universal responsive claim is made.
- **Mobile UAT098 FAILED at 390×844.** The Header title overlaps adjacent controls and extends beyond the viewport. Its measured x205.984375 + width260.765625 gives right466.75 on a 390 px viewport. The PNG and bounds remain explicit failure evidence. Header repair TASK13260.39 was underway during retention; this bundle does not certify that later change. A documentWidth equal to 390 alone does not prove control containment.

## Source and verification checkpoint

Native checks followed root's rebuild of the reviewed .33 source freeze at 2026-09-15 23:08:52.251UTC. The retained 20-file aggregate manifest was captured at 23:12:08.389UTC on head 95578e519dae5ea39874e05afb1f83828b8834b5. It identifies the frozen touched source/test set, not every file in the runtime. Single beginning/completion checks match those 20 files at 23:14:24 and23:20:40UTC. Root's compiler checkpoint at 23:13:28 reports 90 known baseline diagnostics and zero added/removed signatures; the raw compiler log is retained. These are historical checkpoint results, not a claim that typechecking has zero errors.

All native artifacts here predate TASK13260.39's Header source edit. No current-source revalidation or fresh browser work was performed during retention. Do not use this bundle to certify the later Header revision or the complete present workspace. No combined final scene-drawer native workflow is certified, and the full fresh single/multi UAT matrix remains pending.

## Integrity and privacy

All 7 retained PNGs were visually inspected; see SCREENSHOT_REVIEW.md. Original PNG and JSON bytes are unchanged. Text/log/Markdown files only have trailing whitespace and final newline normalized. Original and retained hashes, paths and groups are recorded in retention-manifest.json. INDEX.md is the exact retained-file list. SHA256SUMS indexes every file except itself.

Every input and generated text was scanned before writing against the known isolated single/multi runtime credential values and JWT/private-key patterns, with zero matches. Credential values and runtime-private files are not included. PNGs were also checked for embedded known credential bytes and visually inspected. Original browser outputs retain console counts where present; this bundle does not equate them with a clean console or classify unrelated errors.

Older pre-rebuild pauses, stale-ref automation output and unnecessary historical failures are omitted. Some copied reports refer to redundant screenshot-command logs that remain in private temporary storage; the substantive screenshots, action evidence and settled snapshots are retained here. Relative .playwright-cli links in raw CLI output identify original capture locations; inline snapshots/results and retained PNGs are the durable evidence.
