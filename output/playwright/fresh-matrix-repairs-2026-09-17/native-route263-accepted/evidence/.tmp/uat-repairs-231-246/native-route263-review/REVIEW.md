# UAT263 native Character Retry route acceptance

**CLEAR for TASK13260.205's owned child URL, persistence, feedback and reload boundary.** Independent audit: **26 checks passed, zero failures, 33 hashed inputs**. The earlier UAT261 provider failure remains separate and unchanged.

## Native result

On reviewed source **`2ff90d14ae2d8cc1686cb403b8c3a147cc53427c`**, normal navigation at **2026-09-18 01:43:52.637 UTC** loads original Character3 conversation `fc286396-bc65-4a6a-9e92-9034a3183294`. Its actual canonical GET200 has the original user question and 13,271-character reasoning-only assistant, whose final answer remains empty.

The visible **Retry same model** action at **01:44:32.133** creates child `0742e5b4-0ae7-4004-8691-1d0bc41c5f65`, HTTP201 at **01:44:32.219**, retaining the original parent and Character3. Actual requests copy two prefix records to that child. Completion200 and persistence200 also target the child; persistence at **01:44:34.668** reports saved assistant `pa_9a9f-515c-dd9-8ecc`.

The settled browser URL contains the child ID and shows the saved `BEEP BOOP.` answer. Actual feedback at **01:44:37.711** uses the same child conversation and new assistant ID, followed by200. Normal **page reload**, without navigating to a manually substituted child URL, returns child messages200 at **01:46:14.633** and retains the child URL, saved state and answer.

## Canonical history is three rows

| Canonical row | Identity and independently checked result |
| --- | --- |
| Copied user | `439fe389-cd47-4bd4-b3e1-876bbbe4c0b8`; exact original `Hello, who are you?` |
| Copied failed assistant | `2743aaa5-6fc6-4e5d-8778-a81904e83475`; exact original 13,271-character content, SHA-256 `8ade5470903fb6919060027a8d394a1112884979e349a56ac3a5d646bd1873b9`; final answer empty |
| New assistant | `pa_9a9f-515c-dd9-8ecc`; 13,212 characters, SHA-256 `eb2720e4d395baf6d1f0e307daf11ae505a959bc20e794a7a78bda09319dd240`; after removing think blocks, exact final **`BEEP BOOP.`** |

The old failed assistant remains visible as interrupted after reload. Acceptance comes from the newly identified canonical assistant and the correct route, not a generic UI search that could match an older answer. Raw reasoning remains local; the audit emits only hashes, lengths and final-answer checks.

## Prefix semantics and pre-submit limitation

The reviewed branch handler deliberately copies the accepted history prefix through `index + 1`, then invokes `onServerChatBranchAccepted` before setting active child state. Character Retry supplies the branch index through its existing regeneration flow. Playground accepts the child replacement only under the captured parent URL/store/history/restore-revision ownership checks. The six copied implementation/test files exactly match the independent source review and prepared source manifest.

The native child contains the exact two records submitted as that prefix, followed by the new assistant. However, **this is not a clean, fully settled two-row-parent experiment**. The initial canonical GET has two rows while Character preparation is ongoing. The later Retry-started UI already displays a prior successful local assistant variant with a5:42 PM timestamp. That state explains why a three-row child must be reported honestly; it does not prove why the local variant reappeared or that every fresh two-row Retry should have this prefix length.

The retained evidence does not distinguish intended local history hydration from a separate stale-content issue. No additional browser calls were made to classify it. This uncertainty does not defeat the observed UAT263 boundary: the accepted branch, URL, completion, persistence, feedback and subsequent canonical reload all agree. A fresh-profile matrix can exercise clean-history behavior separately.

## Provenance and reviewed controls

The original profile, initialization and official PostgreSQL fixture holder match their immutable binding fingerprints and the preceding upgrade. Gate, completion, source-manifest and dependency-reuse hashes match. Startup binds API77447/Next78635 to the reviewed source; their exact launch receipts cover original navigation through child reload. These are retained-receipt checks, not a new live runtime inspection.

The earlier independent source review approved **109 Character/coordinator/branch tests plus8 generic Retry tests**, including stale-parent reassertion, reload, cancellation, account and ownership controls. This review reuses that controlled evidence; it does not claim those cases were all exercised again natively. The old14-check stale-route diagnosis remains unchanged and hashed.

## Limits and verification

This accepts the targeted PG-single UAT263 route fix. It does not certify UAT261, general provider reliability, every local variant lifecycle, PG-multi native Retry, or the full fresh48 matrix. One successful answer does not erase the earlier reasoning-only response. Response observer timestamps are observations; no unrecorded first-byte or transport-timing claim is made.

Ran `node .tmp/uat-repairs-231-246/native-route263-review/audit.mjs`: **26 passed / zero failed**. No product/test/Git/Backlog/browser/model/runtime/DB changes or new inference were performed. Only this safe review packet was written. Exact input paths and hashes are in `audit.json`; omitted private/source/raw capture inputs require hash-only treatment if retained publicly.
