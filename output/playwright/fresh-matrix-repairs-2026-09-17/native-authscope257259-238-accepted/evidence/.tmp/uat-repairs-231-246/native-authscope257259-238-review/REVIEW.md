# Native UAT257 / UAT259 / UAT238 dependent-chain acceptance

**CLEAR within the submitted PostgreSQL single-user scope.** Independent retained-evidence audit: **38 checks passed, zero failures, 85 hashed inputs**. No new application defect was found in these repaired boundaries.

| Finding | Acceptance assessment |
| --- | --- |
| UAT257 / TASK13260.199 | Original source QA now returns a grounded, cited answer with unchanged request settings; citation inspection and canonical source reload succeed. AC3 is supported. Earlier reviewed actual-auth PG/SQLite regression evidence supports AC1/2. |
| UAT259 / TASK13260.201 | Normal Delete → empty library/cleared inspector → Trash → Restore → normal Media selection/reload succeeds with original source and analysis history preserved. AC3 is supported. Earlier reviewed restricted-PG/SQLite tests support AC1/2. |
| UAT238 / TASK13260.180 | The previously missing dependent QA, analysis/reanalysis, Multi-Item Review and sole-item Trash/Restore workflows are now exercised. Combined with the earlier independently accepted queued ingest, canonical content, Media-to-Chat and official fixture reviews, the bounded task ACs are supported. This does not certify broad provider quality or a full fresh-install matrix. |

## Exact QA control and cited-source inspection

On **2026-09-18 01:05:30.294 UTC**, the repaired runtime sends the original Rowan factual question to `/api/v1/rag/search/stream`. Its complete request body is JSON-identical to the original failed request at **2026-09-17 23:07:56.584**: same query, model/provider, source scope, thresholds and retrieval controls. It remains `media_db`, standard/hybrid/chunk, with web fallback disabled. The prior HTTP200 empty-context response remains hashed and retained locally.

The repaired response is observed at **01:07:31.057**, HTTP200, with five contexts all referring to Media1 and a completed answer. The answer names Dr. Mira Vale, Cedar Ridge and ORBIT-742 with citations `[1]` and `[3]`. Context1 is an exact substring of the public fixture and contains all three facts. The visible result reports a cited answer, five excerpts and two citations. No provider reasoning is copied into this report or audit.

The normal **View source 1** control opens the matching preview; **Open in Media** creates the observed second tab at `/media?id=1`. The first helper waited on an ambiguous heading in the parent QA tab and timed out. That failed helper is preserved; it is not counted as success. The actual new tab is then selected, and normal reload at **01:15:34–35** returns Media1 HTTP200 with its complete source and version history.

## Actual Delete, Trash and Restore

All following times are 2026-09-18 UTC:

- **01:17:23.409:** confirmed DELETE `/media/1` returns204. The subsequent catalogue returns zero items; the visible library shows Results0/0 and its initial-ingest state, with no selected source heading.
- **01:18:50.467:** Trash GET200 contains exactly original Media1 and the visible Restore control.
- **01:21:47.809:** normal Restore POST returns200 with the complete source and unchanged versions. The next Trash GET at **01:21:47.906** is empty; the settled UI shows **Trash is empty**. The earlier snapshot while restoring is not the acceptance proof.
- Normal Media navigation and selection recover Results1/1. Reload at **01:33:32–33**, observed by **01:33:33.920**, returns HTTP200.

Both pre-delete and post-restore canonical content equal the public fixture exactly after its single final newline is removed: **1,914 characters**, SHA-256 `a94b1e966d89b7b94e0cd69dafe9ab1c554dc81accf43e57957276b08294225c`. Media ID1 and all three version UUIDs, timestamps, prompts and analysis fields are unchanged. Original ingest UUID `d4a0d5e2-6e9a-4d7a-bc9e-f011af4149d0` is linked through the earlier reviewed queued-ingest record; the richer current response supplies Media ID and version identities rather than a new independent top-level media-UUID field.

## Earlier analysis workflow and honest limits

On source `6f6983b062`, the earlier default-prompt stream/fallback attempt returned **502 at23:22:52.989**. It remains a failed attempt. Explicit short-prompt analysis subsequently saved version2 at **23:25:17.411** with `LIVE_TIER_ANALYSIS_ONE`; Multi-Item Review visibly reads that saved result. Changed reanalysis saved version3 at **23:34:49.040** with **`"LIVE_TIER_ANALYSIS_TWO"` including quotes**. The quotes are a model-format deviation; this is not exact bare-token compliance.

The controlled failure test modifies only the outgoing provider/model to an unavailable configuration for two actual backend requests. Stream and fallback both return real **400** responses at **23:37:20.913/.973**; responses were not fabricated. No new version is requested or created, and normal reload preserves the source and versions3/2/1. The later snapshot does not establish the duration of the failure toast. Both canonical reloads after the auth-scope upgrade still contain the same analysis history.

Successful source persistence after the original RLS ingest repair did not establish successful default analysis. This bounded chain clears the source workflow blockers while preserving the original ingest failure, provider truncation warning, default analysis502, quoted output and controlled unavailable-model results.

## Source, fixture and process provenance

Preparation gates, completion records, source manifests and dependency-reuse proofs match each immutable binding. Original profile, initialization and official fixture-holder fingerprints are identical across `6f6983b0620aae1f0892c6b0d3ae3bebfc105e02` and **`edfd06ec40a173f2e38ec65af715abb29f3aa002`**. The latter contains the exact five independently reviewed auth-scope production/test file hashes. The independent implementation review executed **72 actual SQLite/restricted-PostgreSQL tests with zero skips**; adjacent virtual-key issue262 was subsequently reproduced against baseline and is separate.

Old API97807/Next91806 receipts cover the original QA/analysis/Trash-failure interval. Repaired API58057/Next58191 receipts cover the successful QA through canonical restore interval. The immutable **01:39:40.532 pre-stop snapshot** independently shows both repaired processes live after that interval. Their receipts now record normal exits at **01:41:07.317 / 01:41:06.833**, after all reviewed native evidence. These lifecycle updates are disclosed, not treated as source corruption. Hashes reflect this review's observed receipt bytes.

This is retained-evidence verification, not a new live-process inspection or DB probe. No fresh full48 matrix, clean dependency installation, PG-multi native QA/Trash, native foreign-owner deletion or broad model reliability is claimed. Ownership and cleanup controls remain explicitly reviewed fixture evidence. Task/tracker hashes record their mutable state at review time.

## Artifacts and verification

`audit.mjs` reads the original captures and emits only allowlisted facts, IDs, hashes and booleans to `audit.json`. Running it completed all38 checks. Raw captures, reasoning, private configuration and process details remain local and must be retained only as hash references if this safe packet is published. No new tests, inference, browser/API/DB/runtime actions, product edits, Git or task mutations were performed. Original review packets remain unchanged.
