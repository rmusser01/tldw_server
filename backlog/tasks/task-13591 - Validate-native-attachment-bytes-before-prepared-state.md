---
id: TASK-13591
title: Validate native attachment bytes before prepared state
status: Done
labels:
- chat
- backend
- attachments
- validation
dependencies:
- TASK-13590
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue approved H2 Task 2.1 and 2.3 through a bounded private preparation adapter. Reuse the existing upload sink and shared image integrity validator; distinguish detector evidence from filename fallback, preserve truthful UTF-8 text representation and exact hash/size, and fence prepared state against stale owner/generation and changed effective limits. Public H2 remains disabled; adoption, retention routes, context manifest and full readers are later accepted units.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared validation records actual MIME detection provenance without changing existing callers or upload response shape.
- [x] #2 Actual uploaded image, document original and truthful UTF-8 text preparation verifies immutable descriptor and effective destination limits before guarded publication and prepared CAS; mismatches leave no prepared state.
- [x] #3 Held upload, changed configuration, cancellation and publication failure preserve original owner/generation and conservative candidate quota cleanup.
- [x] #4 Run causal red/green and affected upload, image, native I/O and fork regressions, scoped static/Bandit, two independent source reviews and normal exact-owned hooks before commit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. Accepted ADR-049/050 ownership and lifecycle, ADR-070 proposed quota authority and the approved H2 design section 8 govern this private adapter. No new representation, quota, upload parser, public route or runtime dependency. Latest fetched dev19d213 is included in HEAD3c605aff; collision scan saw remote maximum13585 across52openPRs and local13590. Root owns every runner and mutation; independent reviewers remain source-only.
Scope map: native_chat_assets private I/O adapter; Upload_Sink ValidationResult provenance; image_validation optional server-side byte cap preserving existing flashcard callers; genuine upload/native candidate tests and directly affected fixtures; qualification record, master notes and own three-stage plan. Reuse save_uploaded_files, TempDirManager, default FileValidator, current message image limit/document preflight and existing stable candidate guard/CAS. Native original requires content-detector evidence; truthful text requires strict UTF-8 and text MIME. No change to legacy generic document acceptance.
Actual causal controls reproduce missing preparation/provenance, cancelled worker staging loss, header-valid truncated JPEG accepted by verify-only parser, lowered real PDF policy ignored, oversized sink reads and false RTF original accepted by UTF-8 alone. Shared roots corrected: drain and fence cancelled workers, bounded decode, current policy, optional native sink byte bound and strict detector evidence for structured originals. Initial six fixture path assertions omitted chat_assets and are excluded diagnostics. No startup or PostgreSQL simulation counted as native proof.
Independent review additionally qualified the exact final-lock limit boundary, last-chunk cancellation and existing EPUB marker normalization. Real causal controls reproduced all three; final policy is checked after admission locks, cancellation check belongs immediately before preparation publication, and strict original validation retains the selected house media type. A misplaced checkpoint temporarily broke existing publish and was reproduced with its existing regression before exact correction. Final qualification runs the older publisher alongside preparation controls.
Final causal descriptor-wait cancellation RED2 and GREEN full preparation34/31 official PostgreSQL skips; final affected four suites138 passed/89 skipped/1 inherited YARA expected failure/16 warnings/84.13s. Matching house filename Ruff baseline0/current0, four production Bandit0/errors0 and supported6-file compilation pass. Both final independent source reviews bounded CLEAR. Review Docs/Reviews/CHAT_ASSET_VALIDATION_13591_2026_10_11.md; own three-stage plan linked. Earlier external-path Ruff14 and syntactic import-order assertion are superseded/excluded diagnostics, not passing evidence. Conservative cached lower allowance on upward reload remains documented; no public H2 or adoption/read/send acceptance. Contract and normal hooks pending.
Full canonical fingerprint/path check exit0; all11 exact-owned normal repository hooks exit0 without bypass. Final review and master/closure notes preserve broader open gates. Only owned completed plan retires privately; final metadata hooks and incremental local commit follow.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Prepared native uploads now validate actual bounded bytes, detector provenance, raster integrity and truthful text before durable original-owner/generation publication and final locked-policy CAS. Repeated cancellation drains staging and refuses subsequent byte helpers after admission waits. Causal RED/green and final138/89/1 inherited YARA expected failure, matching Ruff0/Bandit0/compile0, unchanged canonical API contract, two bounded CLEAR reviews and normal exact-owned hooks qualify this private preparation prerequisite. Public H2, adoption/reader/send/character and full usability acceptance remain open.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
