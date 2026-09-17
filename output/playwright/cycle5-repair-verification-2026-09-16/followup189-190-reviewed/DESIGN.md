# UAT189 / UAT190 — completed Cram presentation

Associated tasks: TASK13260.127 and TASK13260.128. Parent owns native capture, tasks and integration. The parent reviewed the causal RED and released the exact production scope after completing its native capture.

## Cause and bounded design

ReviewTab already distinguishes the successful raw queue (`hasCramPracticeCards`) from remaining unpracticed identities. After the final filtered card is rated, the raw queue remains nonempty and the active card becomes null. The empty description currently checks only whether a tag exists and therefore displays no-match guidance for this completed queue. Require an empty raw queue for that no-match branch; retain the existing completion branch for a nonempty exhausted queue. Do not alter fetching/success/error gates, identity progression, session ownership, scheduler or mutation behavior.

The Cram completion count currently uses a plural-only fallback and lacks an English resource entry. Add the existing ICU one/other expression to the English key and component fallback, preserving the same numeric count and display guard. Zero is verified through the real resource formatter; the component intentionally displays session statistics only for positive reviewed counts.

## Verification stages

1. **Complete:** preserved source baselines and added a separate actual ReviewTab completion regression file. Real AntD, i18next ICU, useFlashcardReviewRun and Cram query cover scheduled/practice final-card transitions, initial empty tag query, loading/failure/retry, reordered refetch and scope resets. Final causal RED:9 failures,4 passing controls.
2. **Complete:** after parent release, added only the condition and ICU text changes. Retained old20-test baseline PASS and subsequent three failures from old interpolation/grammar expectations, then updated the test formatter and two singular strings without weakening behavior assertions.
3. **Author verification complete:**64 tests/6 suites pass, lint0 errors/21 unchanged warnings, compiler90 unchanged diagnostics, no owned errors. Bandit TSX parsing limitation recorded. Source/tests frozen for independent review; parent native acceptance remains pending. No browser/runtime/provider actions occurred.
