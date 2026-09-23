# UAT engineering sweep: isolated integration

Task: TASK13260.278.6. Date: 2026-09-21. This is a running integration record, not a release or UAT pass.

## Source and ownership

- Fresh `git fetch origin dev` succeeded after approval review recovered. Integration base: `08e980a453d12155cccf10d0c5eb7fe32d05ac75`.
- Native-managed worktree: `/Users/macbook-dev/.codex/worktrees/uat-engineering-sweep-20260921/tldw_server2`; branch `codex/uat-engineering-sweep-20260921`. Worktree registration succeeded.
- Original shared checkout remains on `codex/post2970-uat-20260920`, HEAD `3c871df7173d7b87b64bbb421882c2088542cb7d`, original index unchanged.
- Refreshed private checkpoint: `.tmp/uat-engineering-sweep-20260921/checkpoint-integration/` in the original checkout.194 files copied;0hash mismatches;21 runtime/cache entries retained in place. Manifest SHA256 `ca4ec1e023152109bc88db9c326db1ebb1dd8b3862481398a10fea56cb6586cd`. This is source preservation, not a database backup.
- Seven attributable UAT commits plus owned dirty repairs/tracking were adopted using per-path common-base/current-dev/preserved-UAT three-way merges:213 paths,0transfer mismatches. Common ancestor `d72b1d2850ea947b6d12cac19f6b95867b68a580`. Source commits: `df42641000`, `df1245d7fd`, `c174a90113`, `2173084838`, `2896dac108`, `dfe36e91e7`, `3c871df717`.
- Distribution commits `bb437b3003` and `2406e666af`, unrelated Chatbook/TTS/roadmap work and all runtime/capture directories were excluded. Existing source plans/tasks remain with their recorded scope.
- Original and integrated sources diverge after adoption. New repairs and current tracker updates belong in the isolated checkout. Automatic approval review rejected a proposed tracker copy back to the shared checkout for concurrent-edit risk; no overwrite occurred. The backend test bundled with that rejected command did not run; its later standalone command was approved.

## Upstream overlaps

Thirteen files changed on both branches. Three textual conflicts were resolved:

| File | Resolution |
| --- | --- |
| Extension options-theme bootstrap test | Keep upstream Cheerio parsing and edge cases; run both options and sidepanel inputs. |
| UnifiedSetupWizard | Keep upstream anonymous-resume provider fallback and UAT connection-settings navigation. |
| ViewMediaPage | Same import differing only in quote style; retain both account-retirement and handoff guards. |

The other ten files merged mechanically. Independent review found one semantic gap: UAT393 loses saved image-detail metadata in Character preparation/provider paths. No other loss was found in this bounded overlap review. This is not the full43-family call-path review.

## Verification receipts

| Check | Result and scope |
| --- | --- |
| Locked dependencies | `bun install --frozen-lockfile` succeeded in clean worktree; upstream Zustand/Markdown and related upgrades retained. |
| Clean dev baseline |61passed across5 selected shared-UI files. |
| Adopted frontend regression |1393passed across62 files;0failed/0pending. |
| Theme conflict verification |22passed. First command used UI-only config and collected0tests; corrected command used the extension root. Zero collection was not counted as a pass. |
| Adopted backend regression before393 |187passed,0skips,274.29s; SQLite and required official PostgreSQL fixture. Existing venv reused with explicit isolated source paths; this is not clean-install evidence. |
| Initial type check |Both clean baseline and integration exceeded default4GB Node heap. Preserved failed logs;12GB reruns completed with diagnostics. |
| Matched type diagnostics |398baseline/421integrated before394. Absolute paths/truncated type displays require normalization; added diagnostics tracked in UAT394. |
| Matched frontend lint |10baseline/current errors;1524→1502warnings, with13 added explicit-any warnings in repair tests. UAT394 tracks corrections; total-count reduction does not waive additions. |
| Python lint |13baseline/current findings across13 touched source/test files;0added/removed. |
| Bandit |All8 touched Python production files scanned after393 source repair:0findings/0scan errors. |
| UAT393 causal check |26failed,8legacy controls passed across34 SQLite/PostgreSQL cases;242.94s. Four formatter callers, ordered detail, malformed options and actual metadata query failure. |
| UAT393 detail-only regression |221 passed, 0 skips, 345.69s; later placeholder finding required another causal repair. |
| UAT393 placeholder causal check |8 failed / 16 literal and edited controls passed; 152.73s. |
| Final backend regression |245 passed, 0 failures/errors/skips, 401.867s; all58 new UAT393 cases plus adopted surrounding suites. |
| Final frontend regression |1393 passed / 62 unique files, 0 failures/pending; includes all12 UAT394 files. The74 supplied path strings included12 duplicate relative/absolute aliases. |
| Final matched types/lint |398→353 type diagnostics, 0 added after root/truncation normalization;10 baseline/current lint errors,1524→1489 warnings,0 additions. Baseline errors remain. |
| Production builds |Chrome extension48.86MB; WebUI token-sync/budget pass,580.7KB shared-app gzip under600KB. Neither build is frozen or natively qualified. |
| Final independent review |UAT393 detail/placeholder paths and UAT394 diagnostic delta clear within reviewed scopes. |

Private logs and JSON/XML receipts remain outside the PR. The initial187-test JUnit filename was reused by the393red runner; the original stdout summary remains. The red XML was preserved separately and subsequent runs use distinct report names. The first private merge-preview script treated Git's positive conflict count as a fatal exit; the second stopped at an owned completed-plan deletion. Both partial previews remain; the third explicitly handled those conditions and produced the reviewed adoption manifest.

## Outstanding gates

UAT393 source verification/review and UAT394 frontend diagnostic repair are complete. UAT393 remains open for native acceptance. Existing UAT261/351/352/354/356/359–361/365/375/388–392 retain their original obligations. No new candidate is frozen. Exact executable variants/case accounting, all43-family source review, recovery qualification, four-cell UAT, installation and upgrades remain in the [approved plan](../../IMPLEMENTATION_PLAN_uat_engineering_sweep_20260921.md). Native evidence from earlier candidates cannot certify this changed source or dependency set.

## Workflow repair checkpoint after c2b7ab1fe8

UAT389–392 strengthen Content Review, source-grounded Chat, five-card study and release discovery/accounting. The review exposed minimal product repairs UAT395 (batch db_id extraction) and UAT396 (WebUI route), plus verified auth fixture correction UAT397 (no invented extension runtime ID). Combined126 tests/9files pass; frontend app TypeScript and final scoped lint are clean. The seven selected existing primary-save controls pass under their owning UI configuration after an initial wrong-config import failure. Full collection lists1525 registrations/671 workflow instances across9projects with0errors; no browser execution. Stable identity uses project/file/full describe-title path, validated against all13 Firefox cases in two selections.

Root reviewed the agent-written harness/product changes. Three delegated follow-up reviews were interrupted by account usage limits and are not recorded as completed. UAT395/396 and the strengthened journeys still require owned native execution. UAT392 retains the full manifest and verified production-runtime binding gate. Builds recorded above precede395/396 and must be rebuilt before using those changes natively. The original shared checkout is preserved.

2026-09-21 post395/396 production rebuild: WebUI and Chrome extension both exit0. WebUI token-sync passes, shared-app580.8KB gzip below600KB budget. Logs /tmp/uat-sweep-web-build-r2.log and /tmp/uat-sweep-extension-build-r2.log. These worktree compile checks precede owned archive builds; no native or release qualification is claimed.
