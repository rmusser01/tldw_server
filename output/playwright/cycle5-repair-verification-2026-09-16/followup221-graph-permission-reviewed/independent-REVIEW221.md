# Independent final review: UAT221 / TASK13260.159

## Disposition

**CLEAR after correction.** The initial frozen candidate was blocked by a reproduced late-success cache race. The corrected source resolves the exact counterexample and adds meaningful request-lifetime controls. No further source change is requested.

Final independent verification: **95 passed /8 files /0 skipped /3.68s**. Exact original private race appended to the real final test file: **19 passed /1 file /0 skipped /3.07s** (18 permanent controls plus the unchanged independent case). All four final hashes remained stable through verification.

## Frozen scope

Author manifest SHA256: `8f4fc9b7579227ce7a509c9f46374e047b878c43fe5866b83bc43a0fbda657df`.

| File | SHA256 |
| --- | --- |
| NotesGraphWorkspace.tsx | `0a24a8a1de21f67511abdec257a9e2304fa2a86803b88054b4e2f341598cd573` |
| hooks/useNotesGraphWorkspace.tsx | `36b09e810ab8e0f00f2108408b822f1ec22415597c7becc34e49560d7bd01520` |
| assets/locale/en/option.json | `45dab4d26eef53c199a20fe525efdd0f5d4db2c28a13f7c365393ba36e645a60` |
| NotesGraphWorkspace.permission.test.tsx | `3f0c0a9c1ebd6d9d0f33266219170792c88812b35fa46138d2b52f8fd065f57a` |

All are under apps/packages/ui/src and match the author's review snapshots. The English resource differs from its baseline only by notesSearch.graphPermissionUnavailable. Root's service-level UAT224 normalization is separate and excluded from221 attribution. No backend, generic auth, other query domain, permission grant or suggestion lifecycle change belongs to221.

## Resolved independent finding

On the initial candidate, an earlier radius1 request could finish200 after radius2 received403. The one-time cache clear ran before the old success, so the old key regained private pages. The mounted denied view stayed hidden, but reopening radius1 immediately exposed its cached graph. The full actual-hook probe failed1 while all12 original permanent controls passed; initial source/snapshot hashes were stable. FINDING221.md, original private test/loader, RED logs and initial-snapshot/ preserve the evidence. The first candidate's89 passing tests did not cover this race and were not accepted as final.

The correction uses the installed TanStack query lifetime. A non-canceled403 cancels other queries under the exact authority prefix, excludes the reporting query, uses revert:false, waits for settlement, then empties paginated cache. Query keys identify the reporting request; the actual query AbortSignal prevents canceled late403 responses from wiping a later successful recovery or reinstating denial. Explicit cursor commands handle actual TanStack cancellation as null while unrelated503 still rejects. This retires stale cache writers without a new global authorization cache or request framework.

Installed query-core5.90.20 source confirms cancelQueries cancels retryers synchronously and settled retryers ignore late transport results; cancel aborts the associated query signal. The tests prove both an observer-switch case and a still-mounted concurrent observer, so the result is not attributed solely to automatic unmount cancellation. HTTP transport may continue; the correction governs whether its result can update query state, and does not claim physical network abort.

## Behavior and boundary review

The denied view renders short account-specific guidance with role=alert and an explicit Refresh graph button. Graph-derived content, search, inspector and suggestions are unavailable through graph=null and the render gate. Refresh disables while offline/fetching. It uses current authority/request eligibility and successful manual recovery clears only that authority's local denial; the request-count control proves one recovery request rather than a fetch loop. Other mounted authorities retain pending success and cache. A previous authority's late rejection cannot overwrite current denial.

Permanent tests exercise ordinary and cursor403, reconnection suppression, permission recovery, same-authority cache/reopen, two late-base variants, pending cursor settlement, late403 after recovery, other-authority continuation, unrelated cursor503, normal500/503, genuine loading and cached offline behavior. These are real Workspace/hook/service/QueryClient/i18n tests; only API transport and unavailable-jsdom Cytoscape canvas are doubled. Existing pagination, authority, layout, accessibility and Connections controls remain unchanged. The narrow numeric403 classification follows the graph service's existing typed status; raw server diagnostics are not displayed.

This is UI response handling after server denial. It does not replace server authorization, alter current permissions, or promise a durable authorization decision across future explicit openings. Scoped cache retirement removes prior paginated graph data; an explicit reopen may revalidate.

## Independent verification and limits

Ran the exact eight-file Vitest command in the author's report from apps/tldw-frontend; final-green.log records95/8. Ran the nonmutating private loader against the final permission test; late-success-final-green.log records19/1. No production or test file was edited by this reviewer.

- Scoped ESLint on both production TSX files and the new test:0 errors/0 warnings.
- Fresh full compiler:90 baseline/90 current diagnostics, identical after line-position normalization,0 added/removed. It includes final221 and separately reviewed224 bytes; no clean-build claim.
- Bandit through the project venv:0 findings/3 TSX parse errors. Python Bandit does not meaningfully analyze TypeScript; manual review covers error data flow, authority scoping and absence of permission/secret handling changes.
- Four final source hashes and snapshots unchanged; one locale key only.

Native screenshots, actual denied/manual-refresh/admin access and final browser behavior remain parent-owned. This review neither performs native actions nor claims full Graph/full-matrix acceptance. The separate171/181/213/214 native audit is in its own packet and leaves181's warmed-read ordering gap open.
