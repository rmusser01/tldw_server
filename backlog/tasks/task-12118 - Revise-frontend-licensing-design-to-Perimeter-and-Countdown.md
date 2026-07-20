---
id: TASK-12118
title: Revise frontend licensing design to Perimeter and Countdown
status: Done
documentation:
- Docs/superpowers/specs/2026-07-19-frontend-source-available-licensing-design.md
labels:
- licensing
- frontend
- design
priority: high
modified_files:
- Docs/superpowers/specs/2026-07-19-frontend-source-available-licensing-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the previously approved BSL-based frontend licensing design with the newly approved PolyForm Perimeter 1.0.1 plus release-specific PolyForm Countdown design. Document the conservative pre-counsel launch, protected-path scope, historical public-code cutoff, later community and dedicated-customer grants, contribution licensing, Apache OpenAPI boundary, artifact separation, and release/notice gates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The design spec selects unmodified PolyForm Perimeter 1.0.1 plus release-specific PolyForm Countdown grants that add AGPL-3.0-only as an option on each release's second anniversary.
- [x] #2 The design identifies protected frontend paths, preserves GPL-3.0-only for backend implementation, and applies Apache-2.0 to the canonical OpenAPI contract and published snapshots.
- [x] #3 The design records the public-history limitation around draft PR #2727 and defines a reproducible final pre-license cutoff process.
- [x] #4 The design separates the conservative pre-counsel launch from prospective counsel-reviewed community, customer, CLA, trademark, and commercial terms.
- [x] #5 The design covers artifact isolation, notices, CI/release gates, contribution intake, failure handling, non-goals, and accepted limitations.
- [x] #6 The written spec is self-reviewed for placeholders, contradictions, scope errors, and ambiguous timing, then committed with the task record.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Replaced the obsolete BSL design with the approved Perimeter 1.0.1 plus release-specific Countdown 1.0.0 model.
- Kept the conservative pre-counsel launch legally separate from the prospective custom Community Fork, Dedicated Customer, CLA, trademark, and commercial documents.
- Self-review clarified that AGPL becomes an additional option rather than terminating Perimeter, that Countdown must embed the full AGPL terms, that third-party material is not relicensed, and that free competition remains blocked until the Community Fork Grant is effective.
- Added exact historical-cutoff handling for public draft PR #2727, Apache OpenAPI boundaries, SDK exclusions, artifact/image separation, append-only release records, dependency inventory, and fail-closed release behavior.
- Verification: placeholder/obsolete-model scan found only the intentional rejected BSL alternative and the explicit prohibition on a broad `apps/**` scope; `git diff --cached --check` reported no whitespace errors.
- Bandit is not applicable because only Markdown records were changed.
- Commit `59b5bd56a2` records the revised approved design and task without staging unrelated user work.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Replaced the prior BSL proposal with the approved PolyForm Perimeter 1.0.1 plus release-specific PolyForm Countdown design. The spec now defines the protected frontend boundary, GPL backend and Apache OpenAPI boundaries, conservative pre-counsel cutoff, public-history limitations, prospective community/customer/contributor terms, artifact isolation, notices, verification gates, failure handling, and counsel-review checklist.
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
