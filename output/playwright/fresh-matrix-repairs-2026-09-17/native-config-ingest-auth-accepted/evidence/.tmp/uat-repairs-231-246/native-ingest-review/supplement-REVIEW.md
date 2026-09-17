# Native ingest supplement — UAT233 / UAT244 visible acceptance

**CLEAR: the new evidence closes both gaps identified in the original review: the visible saved-with-warning Results state and the literal Open in Media action.** The original review, parser and 34-check audit are byte-identical. This separate supplement passes **26/26 checks** and hashes **22 inputs**.

## Source and retained profile

The released `model232-upgrade-20260917` gate, preparation completion and PG-multi binding agree on source `6f6983b0620aae1f0892c6b0d3ae3bebfc105e02`. Retained backend/frontend launch receipts are bound to that source and the binding hash, starting at22:45:41.804/22:45:45.325 UTC before ingestion. Gate/completion/source-manifest/Python-reuse proof hashes match the binding. Original profile, initialization and holder fingerprints equal the preceding upgrade's fingerprints.

These are checks of retained receipts, not a new live-process inspection or rehash of every copied source file. The original profile remains `repairs231-250-targeted-20260917`.

## UAT233 — actual visible warning and saved source

`native-targeted/pg-multi/results-ingest-terminal.txt` records Alice **owner2 job6**, completed22:57:54, saved **Media3**, UUID **3936bff9-39ab-4fcd-8e18-d07a2900259b**. Both terminal HTTP200 readbacks (22:57:54.114 and22:57:54.620) contain nested `Warning`, null error and exactly one genuine provider-truncation warning.

The actual Results snapshot at22:58:10.865 shows:

- **Items saved with warnings → Saved with warnings (1)**.
- Exactly one truncation-warning paragraph for `linden-results-public-20260917.txt`.
- **0 succeeded, 1 saved with warnings, 0 skipped, 0 not submitted, 0 failed, 0 cancelled**.
- The source-specific **Open in Media** button.

This supplies the missing native UI portion of task13260.175 AC3. It does not convert unsuccessful analysis into clean success.

## UAT244 — literal action, correct route and complete source

`results-open-media.js`, reproduced in the action receipt, clicks the exact named button **Open linden-results-public-20260917.txt in Media**. The receipt at22:59:01.871 reaches **/media?id=3**, closes the wizard, and selects Linden. It initially reports loading; that first snapshot is not used as proof of loaded content.

`results-open-media-settled.txt` supplies the actual positive:

- GET **/api/v1/media/3 HTTP200** at22:59:01.935 and22:59:32.072.
- Both payloads equal the complete public Linden fixture after removal of its single final newline: **1,918 characters**. All six rendered content paragraphs also equal the complete source exactly.
- Settled route remains **/media?id=3**, **Showing linden-results-public-20260917**, catalogue **Results2/2**, with Linden and Alice's earlier Rowan. No Bob/Birch source or empty-library state appears.
- **No analysis yet**, canonical analysis null, chunking completed.

The catalogue already contains the new source before the Open action, while the Results dialog is visible. The helper and settled observation contain no manual reload or search. Combined with the original Bob empty→owned catalogue transition, this supplies the previously missing literal action branch of task13260.186 AC1 without substituting a direct URL load or wrong Chat navigation.

## Remaining scope and criteria

**No visible-warning or literal Open-in-Media gap remains for UAT233/244 in the submitted targeted PostgreSQL scope.** The original controlled tests and reciprocal-owner native evidence continue to support their other reviewed branches. The following existing limits are unchanged:

- Original SQLite Bob/admin and deliberate active-filter native permutations were not rerun. The event filter still does not establish an exact bare catalogue-list request timestamp.
- No new concurrent allocation or hard-quota rejection native claim; those remain reviewed controlled-test coverage.
- UAT238/task13260.180 AC3 source QA/reanalysis/sole-item Trash coverage remains outside this supplement. Root is handling those dependent scenarios separately.

No product/source/task/tracker/Git/browser/runtime changes or test/inference calls were made. Only `supplement-REVIEW.md`, `supplement-audit.mjs` and `supplement-audit.json` were written. No credentials or private source content are copied into this packet.

All short evidence names above are under `.tmp/uat-repairs-231-246/`. Exact input paths/hashes are in the supplement audit. Audit JSON SHA256: `189983c4964e5b24688441d8848398d19abce2463c236cb5faca407b46b3f4f2`.
