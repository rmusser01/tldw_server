# Independent targeted native audit — UAT221 / UAT224

**ACCEPT within the existing bounded criteria for TASK13260.159 and TASK13260.162.** Final native denied/authorized controls complement the independently reviewed automated lifetime and identity controls. No browser, runtime, native database, permission, product, task or git change was made by this audit.

## Source binding

`native221-224-final-source.json` records six final source/test hashes at 09:17:57.275Z (SHA256 34f19e36e2d7fafc4a71fc16174b24648aeae0b4d3f32d5acc912c9a46bbf61c). All six match current files and the final221/224 author manifests. This includes the corrected221 query-lifetime hook and224 service normalization. Parent reports integration commits a8daeb71fe /4e9d854f36. This audit verifies file receipts, not a browser bundle signature.

The native request timestamps are UTC. Requests use same-origin `http://127.0.0.1:18583/api/v1/...` against the parent's actual PostgreSQL profile. Native privileged-role qualification remains; this is frontend/native behavior acceptance, not a new raw-SQL RLS claim.

## UAT224 — focused identity and real relationship: PASS

- Last preceding identity receipt is administrator 1; later09:17:57 identity receipts remain administrator 1 before the subsequent account transition.
- `admin224-refresh.txt` retains the explicit Refresh graph click. At09:16:32.936 the graph request targets note 11c86b62-d045-4836-9e2b-f9b68b14650f; response 09:16:32.969 is200. Its original backend body still has that raw note UUID, tag:uat210-admin+tag, and the real connecting edge. There are 2 nodes/1 edge, active_note_count 1, no truncation and no more pages.
- `admin224-after-refresh.txt` shows the Connected tag and the selected “UAT210 admin graph 20260917” inspector, including Tags and Relationships. `admin224-canvas.png` was independently visually inspected: focused highlighted note with its label, tag node/edge, and matching populated inspector are visible.
- The earlier `admin224-relationships-open.txt` says no relationships are visible; `admin221-graph.png` was also inspected and shows an unlabeled/unselected note with the empty inspector. These are preserved as failed prior-state evidence and are not counted as final acceptance.
- AC 2/3 normalization, metadata/cursor and causal/static requirements are supported by final independent `REVIEW224.md`:75/4 tests plus8 private service/consumer controls passed, no skips; exact two hashes match. This native window adds the actual raw-response →visible-focus/relationship control.

## UAT221 — denied state after account transition: PASS

- Retained identity events change from administrator 1 at 09:17:57 to Alice 2 at 09:18:39 and remain Alice 2 in subsequent receipts. The parent reports using normal Settings login; this audit uses identity receipts and does not read session headers, login credentials or tokens.
- `alice221-final-notes-ready.txt` has Alice's 4-note catalogue. The graph request at 09:19:20.459 targets Alice's owned87c1d02b-0514-4400-bd7b-71be2dcb14f4 and returns 403 at 09:19:20.485:missing notes.graph.read. The settled graph region shows an alert explaining that graph access is unavailable for this account and suggests asking an administrator, plus Refresh graph. It contains no canvas, inspector, relationship content or false empty-graph claim.
- `alice221-final-refresh.txt` retains the explicit manual click. It produces exactly one further request at 09:20:50.445 and 403 at 09:20:50.472. The final capture at 09:20:51.644 spans 91.185 seconds from the first request, with exactly these two graph requests and no others. The manual click occurred 89.986 seconds after the initial request; this is not a claim of a full 90-second wait before clicking. The same denied guidance remains after the refresh.
- The authorized administrator positive control above covers ordinary permitted graph behavior. Final independent `REVIEW221.md` records 95/8 tests plus 19/1 unchanged private race controls passing with stable four-file hashes. It specifically covers late200/403 ordering, authority isolation, cursor cancellation, successful recovery, transient errors and offline behavior. Those scenarios were automated; no native race injection or permission grant is claimed.

## Limits

The administrator's optional graph-suggestion capability/list calls returned 503; the accepted graph 200, focus and real existing tag relationship are separate. This audit does not accept suggestion generation or claim a completely error-free console. The native window does not test paging, every layout, revoked-permission races, offline recovery or permission granting; the relevant221/224 automated controls remain the cited evidence for those contracts. Full fresh-matrix acceptance is not claimed.

## Compact evidence and preservation

`admin-events-window.json` retains 7 unmodified selected events (including preceding administrator identity and optional suggestion 503 responses), and `alice-events-window.json` retains 12 identity/graph events. Each records its cumulative original file hash and capture timestamp. `graph-region-excerpts.json` binds the four original snapshot hashes and exact graph-region excerpts. Large cumulative logs are not duplicated. The original failed screenshot and Relationships snapshot remain untouched and are hash-bound in reviewer-manifest.json.

`verification.json` records the checked counts/windows/source matches. Source manifests and both independent source-review reports are bound in the reviewer manifest. One initial extraction attempt used lexicographic fractional-second boundaries and failed its count assertion before writing evidence; the final extraction compares parsed UTC datetimes and all checks pass.
