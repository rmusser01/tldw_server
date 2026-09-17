# Independent SQLite-single analysis and Trash audit

## Disposition

**Row 10's analysis/reanalysis, Multi-Item Review display and failure-preservation controls are supported. Row 11's permitted single-user delete, settled empty catalogue, dated Trash and exact restore are supported.** No new product defect is established by these 24 receipts. This is bounded acceptance of those paths, not a full matrix verdict.

Frozen revision `8f8774e6c868b304a96d95ab82e28389c129a78b` is the parent-declared running source. This evidence-only audit does not independently rehash the runtime. All times are UTC on 2026-09-17. Only the allowlisted `analysis-*.txt`/`trash-*.txt` receipts were used. Model reasoning is excluded from this report.

## Analysis sequence and persistence

The Generate dialog is filled and submitted through its real controls. Requests contain the unchanged 1,914-character Rowan source and the selected exact-token system prompt; successful calls use configured llama.cpp/Gemma.

| Attempt | Actual request / result | Saved result |
|---|---|---|
| ONE | Streaming request 13:44:32.434; 200; body read ends 13:45:03.322. Version POST 13:45:03.290 returns 201. | `LIVE_TIER_ANALYSIS_ONE`, version 2, UUID `a2376f20-9981-4480-bb59-4bfdf32cb9e6`. |
| TWO | Streaming request 13:47:35.394; 200; body read ends 13:47:55.561. Version POST 13:47:55.557 returns 201. | `LIVE_TIER_ANALYSIS_TWO`, version 3, UUID `5d28a2cc-d324-4c93-9679-81411448d4fb`. |
| THREE, one-shot fault | Original stream request 13:52:53.587 is continued with unavailable Ollama/model; actual backend 400 at 13:52:53.649. Automatic non-stream request uses original configured model and returns 200 at 13:53:08.717, exact THREE content, finish `stop`. | Successful fallback creates version 4 with 201 at 13:53:08.737: `LIVE_TIER_ANALYSIS_THREE`, UUID `43400fc8-88ee-4674-b96e-baa35722da89`. |
| FOUR, first both-failed control | Both stream and fallback requests are continued with unavailable model; actual 400 responses at 13:53:48.894 and 13:53:48.949. | No version POST in the retained failure window; subsequent readback and normal reload retain THREE/version 4. |
| FOUR, visible-error repeat | After restore, both requests again return real 400 at 13:56:39.145 and 13:56:39.202. The actual UI visibly reports `Failed to generate analysis`; Cancel then `page.reload()` runs. | Latest GET at 13:56:39.731 and reloaded UI retain THREE/version 4; FOUR is absent from saved analysis and all version records. |

The controlled failures are real backend `model_not_available` rejections. Route handlers change the outgoing request and use `route.continue`; no response is fulfilled synthetically. The ordinary request observer records the pre-interception configured model, while the fault receipts and real error bodies establish the effective unavailable model. This verifies rejection/fallback behavior, not an outage after upstream generation starts.

The first fault is **successful recovery**, not a failed-analysis-preservation control: its non-streaming fallback genuinely generates and persists THREE. The later two controls are the failed-operation preservation proof. The first late failure snapshot misses the transient error text; the subsequent explicitly awaited visible-error receipt closes that UI evidence gap. FOUR may appear in the submitted prompt, but never as the rendered or saved analysis result.

ONE is visibly rendered in the normal Media view. The Multi-Item Review route `/media-multi` performs a real search, opens Rowan, receives media 1 with content and latest analysis, and displays ONE at 13:47:05.608. This proves the same item is readable there; it does not claim a simultaneous multi-item comparison or batch operation. TWO is visibly rendered in the later Media snapshot and survives the reload before the THREE attempt.

Original version 1 remains `736f4de9-63de-4f39-99a7-611a8317b332` with no analysis. The later successful explicit analyses do not retrospectively certify the earlier truncated ingestion analysis. Exact-token prompts validate analysis execution/storage and changed-prompt behavior, not summary quality.

## Delete, empty state, dated Trash and restore

1. The allowed current-user control opens a confirmation stating that the item can be restored. Actual Delete sends `DELETE /media/1` at 13:55:10.206 and returns **204** at 13:55:10.221. The detail pane clears; its immediate receipt still has a transient 1/1/loading sidebar, which is not treated as the settled state.
2. The subsequent active catalogue response at **13:55:10.244** returns `items=[]`, total 0. `trash-open.txt` captures the settled Media **0/0 / Get started — ingest your first content** state at 13:55:27.546 immediately after clicking Trash and before the route transition completes. This supplies the sole-item empty-catalogue proof.
3. Trash GET returns exactly media ID **1**, the Rowan title and `deleted_at=2026-09-17T13:55:10.216Z`. The native Trash snapshot visibly shows **Deleted: Sep 17, 2026, 6:55 AM**, matching the host's UTC−07 display, with Restore available.
4. Actual Restore sends `POST /media/1/restore` at 13:56:09.560 and returns **200** at 13:56:09.581. The refreshed Trash response is empty/total 0, and UI says Trash is empty / Item restored. Normal navigation back to `/media?id=1` returns the restored item; catalogue responses at 13:56:09.984 onward contain exactly ID 1/total 1.
5. Pre-delete, restored and final post-failure readbacks have identical source fields, content object and all four version records. Latest analysis remains THREE/version 4 with the same version UUID. Source title/URL, 306 words and 1,914 characters are preserved. Source-text SHA256 is `a94b1e966d89b7b94e0cd69dafe9ab1c554dc81accf43e57957276b08294225c`; it also matches the prior source/reuse audit. The complete version-record array digest is `9805eb644fe757bcb06f6144f687e49a127de4f30e76039cd09e658607f6a1f3` (compact sorted-key JSON).

## Limits and harness observations

- Identity is directly established by media **ID 1**, unchanged source content/fields, and the exact four **version UUIDs**. These receipts do not emit the underlying media UUID `9023fb11-1882-49e6-b351-d7ee4df1bb32`; no separate post-restore media-UUID readback is claimed.
- Row 11 covers an authorized single-user operation. It does not test a forbidden role, reciprocal account isolation, permanent delete, scheduled purge, or multi-user Trash access.
- `analysis-media-entry.txt` preserves a harness timeout waiting for an absent Done button. The subsequent actual entry closes the ingest wizard and opens Media normally. It is not counted as a completed action or a product failure.
- Full reload is explicit before THREE, after the first both-failed control and after the visible-error repeat. Restore itself is followed by ordinary navigation/readback, then the last control's full reload.
- Expected failure responses and referenced console entries are retained; no clean-console claim. No browser, service, database, inference, test, source, tracker, task or git action was performed by this reviewer.

## Reviewed input hashes

Exact original bytes; no normalization. Inputs are under `.tmp/uat-next-matrix-20260916/native/sqlite-single/`.

| Input | Bytes | SHA256 |
|---|---:|---|
| `analysis-failure-both-result.txt` | 9242 | `dd8eff4148b56674524607d0e4d5f23528b9a05822b869942ada0b603909d868` |
| `analysis-failure-both-start.txt` | 5381 | `146288db28d1c1d4d970f726fd200b97f631b330ddde9081735374c114dda810` |
| `analysis-failure-reload-trash-entry.txt` | 21966 | `91a8550f4d154cc086b0c7218a0dc3cc436cb6e000668976421265f944eafa1b` |
| `analysis-failure-result.txt` | 38552 | `f80fc7e4f9a3846dca10ddf6e2b3ee4f412430bb161fba1f1907d3d3bd274a40` |
| `analysis-failure-start.txt` | 5833 | `b751240c6ac1ee1c378004229da83379d5df67b7b27b5b3313bf41f6cf68e4cc` |
| `analysis-media-entry-actual.txt` | 1744 | `a69334fc7d4e0e9deb2d5e29a65adaf39c812906f9fed7e81dc269b605042631` |
| `analysis-media-entry.txt` | 144 | `c1934449332680a1908e84d03be159cb60a4b12aba2515aecc975ccbdde2795d` |
| `analysis-multi-review-entry.txt` | 886 | `3c8fbaf7ed5799b737787e77866945ce387462711fd953330102fb87849de390` |
| `analysis-multi-review-one.txt` | 14699 | `3e36bbded0b1486fca3832f154625984a9f6fedb74262c853a7d71cc6e20d7fa` |
| `analysis-one-media-snapshot.txt` | 18286 | `617162efb624d5e63d033b93c1c49624693d2ca4d1fbbea2b7c59cb3f51cb889` |
| `analysis-one-result.txt` | 21930 | `88650bee01ed309bfa64ffe850d2d833b3dcd437083f556762dc478230620283` |
| `analysis-one-settled.txt` | 1345724 | `4e1af7e33661d562375acdbd64ea106ea31aa137741125935904d3277d74c34a` |
| `analysis-one-start.txt` | 4336 | `6870a6349f639c57517223175ab5b798ad4102aff2fee2ad5c4bcae7a59a8821` |
| `analysis-return-media.txt` | 634 | `3322a963bbfab83fece20317d986eadf6aa8e34739f9546dbc17944027b84b08` |
| `analysis-two-settled.txt` | 2226956 | `6ece99f09ebf1949caa6cb4ab515fcf76e9b79b893d95fc1052854b7db9bf4d7` |
| `analysis-two-snapshot.txt` | 17759 | `f964204da008122e2a38ce81f8d052d69e6f54fe503ab2c1d88770c7b6f216d2` |
| `analysis-two-start.txt` | 4649 | `ed79f180711b79cecaf70dfd785686170beffe9221234c8f0a2ce70171ff6b98` |
| `analysis-visible-failure-preserved.txt` | 38493 | `b3323d335546ac32a9d08cdb7ac71aebb9a06f4a2f62b095e2c1a8e645711738` |
| `trash-confirm-snapshot.txt` | 19185 | `dad23848afc93afb8ac6e9ef1a5a5619836e90fec6a761fda3bf106ff6469034` |
| `trash-dated-evidence.txt` | 6640 | `e55f1974bbd055baee34d16f3d667717e6ffefea35eb9281a7738eaeda54bef2` |
| `trash-deleted-empty.txt` | 26898 | `c1ccd510ae2ab9ef99336c54798845ee46b6184ceac6d78ab4c333640701ef2c` |
| `trash-list-snapshot.txt` | 4461 | `3d1158241d690b0d3108517a9b3022a99cb1c2d006a8dda5af71b936a26abb3a` |
| `trash-open.txt` | 872 | `a8e8da68b01ab29bda739bff3a931a4cea595ce6d1a8ee27c1d88ee63eca3978` |
| `trash-restored-evidence.txt` | 25326 | `d256797e62ef11072c37b58a252758f8a6f3907ca2835b7f6045684a00423ebd` |

Audit completed 2026-09-17T14:01:18.542625+00:00.
