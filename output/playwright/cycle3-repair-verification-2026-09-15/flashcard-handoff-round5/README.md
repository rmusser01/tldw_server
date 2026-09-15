# Targeted Flashcard transfer and selector verification

Existing isolated multi-user Bob browser/API; these are repair checks, not a fresh full UAT. Source transfer implementation aae6b72d05; selector fix f9b0dd2f53.

## Verified

- Actual Note More actions → Generate flashcards transferred the exact five-fact text, including whitespace, and its Note provenance into a clean /flashcards?tab=importExport URL.
- One real generation produced two drafts. Actual Save generated cards displayed success; filtered backend access lines corroborate one generation200 and two create200 responses. Independent owned GET200 contains both new cards with the original Note UUID.
- Native selector RED: Clear immediately reselected Biology. After removing allowClear and rebuilding the isolated frontend, Clear count is zero. Create new deck exposes its name field; selecting Biology restores the scheduler summary and hides the new-deck field.
- Existing generation/deck/decomposition regressions24/3 pass; independent source review is clear. Lint has0 errors and5 unchanged warnings. Full workspace compiler matches90 existing signatures; no clean typecheck claim.

## Limits and environment

The long-lived browser resource timing buffer does not show the generation/save requests; those are corroborated by the filtered server access log and independent owned read, not inferred from its unrelated resource entries. A response observer timed out while the actual generation completed; no duplicate generation was submitted.

Development HMR reset an earlier consumed draft; the source action was repeated during a coordinated stable window before generation. The stale compiled selector still contained allowClear after source editing; it was not counted as a successful check. Disk exhaustion then blocked staging and caused Next compiler HTTP500. Only unused original-run compiled bundles were removed; evidence/configuration/databases/browser state were preserved. Restart/rebuild returned Flashcards HTTP200 before the successful selector captures.

All seven cards in the retained independent read are in Biology deck1. This read does not certify the required mixed-deck/undecked Study run. Other transfer producers, cross-account native transfer checks, Study session accounting and both full fresh matrices remain pending. No new deck/card writes occurred during the selector-only check.

No raw credentials, headers or unfiltered network captures are retained here.

The retained regression log has trailing blank lines removed; test output is otherwise unchanged.
