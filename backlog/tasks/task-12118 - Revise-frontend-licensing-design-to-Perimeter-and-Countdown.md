---
id: TASK-12118
title: Revise frontend licensing design to Perimeter and Countdown
status: In Progress
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
- [ ] #1 The design spec selects unmodified PolyForm Perimeter 1.0.1 plus release-specific PolyForm Countdown grants that add AGPL-3.0-only as an option on each release's second anniversary.
- [ ] #2 The design identifies protected frontend paths, preserves GPL-3.0-only for backend implementation, and applies Apache-2.0 to the canonical OpenAPI contract and published snapshots.
- [ ] #3 The design records the public-history limitation around draft PR #2727 and defines a reproducible final pre-license cutoff process.
- [ ] #4 The design separates the conservative pre-counsel launch from prospective counsel-reviewed community, customer, CLA, trademark, and commercial terms.
- [ ] #5 The design covers artifact isolation, notices, CI/release gates, contribution intake, failure handling, non-goals, and accepted limitations.
- [ ] #6 The written spec is self-reviewed for placeholders, contradictions, scope errors, and ambiguous timing, then committed with the task record.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Replaced the obsolete BSL design with the approved Perimeter 1.0.1 plus release-specific Countdown 1.0.0 model.
- Kept the conservative pre-counsel launch legally separate from the prospective custom Community Fork, Dedicated Customer, CLA, trademark, and commercial documents.
- Self-review clarified that AGPL becomes an additional option rather than terminating Perimeter, that Countdown must embed the full AGPL terms, that third-party material is not relicensed, and that free competition remains blocked until the Community Fork Grant is effective.
- Added exact historical-cutoff handling for public draft PR #2727, Apache OpenAPI boundaries, SDK exclusions, artifact/image separation, append-only release records, dependency inventory, and fail-closed release behavior.
- Verification in progress: placeholder/obsolete-model scan and git diff whitespace checks are clean. Bandit is not applicable because only Markdown records are changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
