# Fresh workflow UAT evidence — 2026-09-15 UTC

Authoritative results: [running tracker](../../../Docs/Reviews/FRESH_INSTALL_SINGLE_MULTI_UAT_TRACKER_2026_09_14.md). Backlog: TASK-13260. Product revision: `68863b90b7`.

Both modes used empty config/data/browser profiles, existing dependencies, isolated APIs18100/18101 and WebUIs18180/18181, and the real llama.cpp service on9099. No product changes or permission bypasses were made during this run. Full workflow attempts include failures, recovery controls and explicit downstream blocks; this evidence is not release sign-off.

- `multi/multi-uat-results.md` records multi-user steps and limitations.
- PNG files capture visible outcomes; YAML files capture browser accessibility snapshots. Screenshots were visually inspected.
- JSON files contain selected request/response bodies or structured corroborating results; they contain no raw authorization headers. Browser request indexes are page-local and can reset on navigation.
- `single/flashcards-claim-verification-422.json` was recovered from the browser's `[tldw:request] POST /flashcards/generate 422` warning after its original request index expired. It is the diagnostic JSON serialized in that warning, not a fresh API call. The invalid expired-index placeholder was discarded. No raw runtime log was copied.
- UAT-039's initial contradictory study-queue text was observed in a settled snapshot returned to the UAT session but was not saved separately. Later artifacts prove the available card/review, not that earlier text.
- The prompt's false unsaved confirmation was observed live; an empty attempted snapshot was discarded. The saved-editor snapshot and both mode reports preserve the surrounding sequence.
- Text files have trailing blank lines removed; content is otherwise retained. The manifest hashes these final evidence files.
- `evidence-manifest.json` records file sizes and SHA-256 hashes after artifact collection. The manifest excludes itself.

Private runtime manifests, browser authentication storage, password-entry snapshots and complete runtime logs are excluded. Text artifacts were checked against generated runtime credentials and token patterns; JSON artifacts were parsed before delivery. Synthetic note/source/account data and local model IDs are intentional test evidence.

The public Wikipedia fixture was attempted in both modes. Wikimedia's robot-policy response was incorrectly stored as successful article content (UAT-044); no workaround of that restriction was attempted. Separate synthetic public sources support the valid source-content controls.
