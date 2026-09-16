# Targeted native verification: UAT124 and UAT125

Product: ab527eb3b4, branch codex/fresh-install-uat-fixes. Native checks completed 2026-09-16 07:49–07:56 UTC on the preserved isolated single-user API18402/UI18482. Existing dependencies and data were reused; this is not a new full fresh UAT run. Fresh origin/dev fetch at07:55 confirms59049e094e is contained with zero missing dev commits. The original behind-dev starting point remains disclosed in the main tracker.

## UAT124 — PASS for new failed turns

Actual New saved chat, text plus bluePNG attachment, Send, two Retry same model actions, then normal reload. Input: `Keep one error group after retry and reload. WILLOW VARIANT 0748.` Conversation b933cec5-259a-4caf-8545-2ce1441c0b39.

- User pa_6cb9-e289-ba6-c557 remains the only user before/after reload.
- Error IDs in order: pa_4348-ed2b-1cd-e55c, pa_a4ff-eb3c-a71-28c9, pa_865b-7a93-e0a-cf1e. Only the latest assistant is visible, labelled3of3, before/after reload.
- Exact PNG retained:386bytes, SHA256314f71d711b39db674e4679f96307ef75f4da2ab93c3c8947d6e5c3ec1d0f1f7.
- Actual canonical GET1407 returns HTTP200, messages[]/total0. The refusal occurs before inference; no successful generation is claimed.
- Reload screenshot visually inspected. Historical unparented records from the pre-fix run are preserved; no migration or guessing is claimed.

## UAT125 — PASS for the native failure trigger

Terminal Retry samples at159ms and128ms contain the explicit image-support alert/recovery actions and no Response complete announcement. The initial313ms sample also has no completion announcement but still contains transient generating text; it is not claimed as a terminal observation.

Separate actual New saved chat, Attach image, upload the samePNG, leave text empty, Send. New conversation aa726d1a-1293-453b-af54-7b0240cbcb76; user pa_9a55-2a7a-e53-ac66, assistant pa_01d2-8a63-de7-eaa4. In the same browser call, wait for the explicit error heading and visible Send button, then inspect article live regions/alerts. At688ms the only live-region text is the failure/recovery guidance; no Response complete. The screenshot was visually inspected. Actual canonical GET1597 HTTP200 has messages[]/total0. The image remains PNG with the same bytes/hash.

Successful completion announcements and active Retry behavior are covered by the permanent component tests, not a new native positive generation in this check. Final console query reports zero errors; this does not certify historical startup warnings. No confirmed vision provider is available, so successful vision/canonical-image recovery remains a coverage limitation.

## Verification context

Independent source reviews clear. Combined affected frontend verification:2604tests/97files pass. Full TypeScript matches90 pre-existing diagnostics with0added/removed; not a clean compiler result. Earlier backend456pass/2skips and Bandit8productionpaths0findings remain unchanged by these TypeScript-only repairs. Focused test counts overlap and are not additive.

Raw browser outputs, screenshots, identity hashes, canonical bodies and safe URL/status request inventories are retained alongside this report. No tokens or request headers are included. Independent evidence audit is recorded separately.
