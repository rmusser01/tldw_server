# Native combined scene ownership — TASK13260.33

Completed parent-run native controls on the isolated multi-user UI18281, 2026-09-15T23:46–23:51Z. This bundle retains10 nonempty original captures and the final .33/source and current .39 Header provenance. It records these narrow checks, not full fresh single/multi acceptance; that matrix remains pending.

## Observed results

| Control | Retained evidence | Result |
|---|---|---|
| Cedar initial saved chat and blank scene draft | before, drawer and identity-picker captures | Saved Cedar title, original question/answer and actual character picker options visible; Scene notes initially blank. |
| Current Cedar selection + synthetic scene Apply | same-identity-apply, applied-snapshot | At23:48:38.010Z, conversation812385d9-f19b-4bdf-82a0-1d850884a4b3, historypa_576f-cb09-174-81bf, character4 and2 original messages remain unchanged. |
| Reopen same scene | reopened-same and reopened-snapshot | At23:49:14.438Z, exact note “UAT ownership check: keep this Cedar scene on the same saved conversation.” remains in Scene notes. |
| Explicit Robot replacement + new scene Apply | replacement-picker and replacement-apply | At23:50:17.552Z, saved conversation/history IDs become null and messages0; reopened drawer names Cycle3 BEEP Robot and contains exact note “UAT ownership check: this scene belongs to the new Robot draft.” Server character metadata is null on the unsaved draft; Robot identity is evidenced by the actual reopened drawer. |
| Return to original Cedar and Clear scene Apply | original-preserved-and-cleared | At23:51:04.602Z, Cedar has its own prior note rather than Robot's; Clear scene makes notes blank and Apply dismisses while original saved ID, character4 and2 messages remain. |

The runner reports using the visible current-character/Robot picker buttons and entering both synthetic notes. Picker snapshots expose the real controls and returned Apply/reopen records corroborate resulting identity and content; the individual picker-click/fill command records are not separately retained here. The final Clear capture reads blank immediately before Apply and records identity/messages after Apply; no additional post-clear reopen/reload is claimed.

## Persistence and retained scratch state

Scene settings use the existing actor-settings local browser storage path, keyed by saved server conversation or draft context. They are not conversation API PUTs. The same-identity result's responses:[] is expected: its observer watched only POST/PUT/PATCH responses containing /chats/ during that Apply. It is not a complete network inventory and proves no general absence of background traffic. No inference or new saved conversation was dispatched by the runner; these files retain identity/message outcomes, not an independent server-wide mutation audit.

Both scenes remained draft notes with the Scene-enabled switch off/preview “No scene”; this verifies ownership/persistence, not generated scene-conditioned answers. Original Cedar notes were returned to blank through the normal Clear scene + Apply UI. The unsaved Robot scratch draft retains only its synthetic note; it was deliberately not deleted by this retention task. Existing source/chat messages were preserved.

## Source, console and title qualifications

Final .33 commit98a97eea7a matches all12 files in ownership-final-manifest.json, frozen23:08:52.251Z, and current filesystem hashes at packaging. Header .39 is separately bound to header098-production-freeze.json (hash99ead6…), not misrepresented as part of98a97eea7a. No product edits were made during packaging. See source-verification.json for exact hashes and observed packaging HEAD.

No tool-error result appears in the10 originals. The final capture references console lines10–15; their retained excerpt contains Fast Refresh rebuilding/done, React DevTools info and HMR connected, not an error. This is not a claim of zero console errors outside the excerpt. Snapshot accessibility alerts still carry a previous Robot route-announcement string; captured browser page title/header/state are Cedar. Those stale announcements are preserved and are not treated as active conversation identity.

## Provenance and safety

INDEX.json maps all13 retained original/excerpt artifacts to source paths, exact original and copied hashes, byte lengths, transforms and timestamps. The4 embedded JSON results are parsed into derived RESULTS.json; original result/command text is retained. JSON manifests preserve exact bytes. Text/log copies normalize trailing horizontal whitespace and final blank lines only. Originals remain unchanged. No PNG was produced by these actions, so no screenshot is invented or copied from another checkpoint.

SCAN.json reports known isolated-runtime secret matching plus JWT/private-key pattern scans, with no secret values retained. The private builder is /private/tmp/uat093-build-scene-native-evidence.mjs; private runtime manifests/credentials are never copied. SHA256SUMS indexes every payload file except itself. Native evidence collection was parent-owned; packaging made no browser/runtime/test/production/Git changes.
