# Chat Header UAT098 — reviewed repair and targeted native verification

TASK13260.39, under parent TASK13260. The bounded Header overlap repair passes targeted native checks. **Separate UAT100 / TASK13260.41 remains open; full fresh UAT remains pending.**

## Native RED → GREEN

Original RED is retained once in the adjacent durable bundle: [390px Header bounds](../chat-restoration-native-final/uat097-reviewed-mobile-header-bounds.txt) and [390px screenshot](../chat-restoration-native-final/uat097-reviewed-390-layout.png). The saved title extended to x466.75 in a 390 px viewport and overlapped Shortcuts. Exact link targets and hashes are in native-red-references.json; both were checked against the private originals during retention.

Final native captures were recorded by root on the existing multi browser, port 18281; the retaining reviewer inspected the evidence without browser actions:

- At 390×844, saved title button is x16/y48/w220/h16. All named measured header controls fit the viewport, with overlaps[]. Cedar title/character 4/two original messages remain loaded and idle. The title has its own wrapped row rather than extending over neighboring controls.
- Actual title-editor snapshot and bounds show x16/y48/w181.5/h24 and the full original value, Cycle3 Cedar Guide Chat (20260915_103705). The bounds capture presses Escape; later snapshots return to the title button. No title rename was committed in this check.
- Actual Show shortcuts opens the dialog. Root used Escape to close it; later sidebar/ready captures show the normal header. Sidebar pointer control opens the actual Chats drawer. These actions preserve the full accessible title, editing and navigation behavior.
- At 1024×720 and 1280×720, the named measured header controls fit with overlaps[] and document width equal to viewport width. The three retained PNGs were inspected, and screenshots show readable, separate controls. This is the recorded viewport set, not a universal responsive claim.
- **Separate UAT100 remains:** actual mobile Robot row click resolves the correct saved Robot, two messages and BEEP BOOP, but dialogs remains1 and the drawer stays open. The failure capture and snapshot are retained honestly. Root later closes the drawer explicitly before desktop measurements. This is not reported as repaired by the Header classes.

## Automation and incidental-state qualifications

The 17 meaningful native files retained here contain no tool-level “### Error” block. Parent reported earlier stale-reference attempts and a click behind an already-open shortcuts modal. References were refreshed and Escape closed the modal before meaningful controls continued; those unretained attempts are automation context, not evidence of new product failures.

An incidental focus view appeared between checks. Root exited it through the UI; the after-focus-exit snapshot is retained. Its cause is unproven, and this bundle does not attribute it to development reload or dismiss it as conclusively harmless. There is no retained causal trace for that transition.

No full fresh single/multi matrix, UAT100 closure, all-header-mode geometry or indefinite clean-console claim is made. Earlier static reports correctly say native was pending at their checkpoints; this README records the later bounded native outcomes without altering those historical reports.

## Source and automated verification

Production freeze 2026-09-15T23:30:56.674Z: only apps/packages/ui/src/components/Layouts/ChatHeader.tsx, SHA256 99ead6a0de84203afbe797b24d8da293a87fd668e3453fd5a245d42939fb996f. The frozen current source copy ChatHeader.after.tsx matches that hash; original before-source and exact diff are retained. Five CSS-class edits bound/wrap the left group and badges, prevent brand/sidebar shrink, and constrain the title button to its existing wrapper. No handlers, full title text, props, auth, identity or network behavior changed.

The original implementer checksum list and all 11 inputs were validated before retention. The retained list is deliberately recomputed for the three normalized static test logs; original list/input hashes remain in retention-manifest.json. The retained list and all 11 retained inputs validate directly. Existing Header/notification/shortcuts 59/3 and title+SSR 16/2 passed after the change. Independent review is clear with 50/6 tests (shared Header/notification/character/share 34 and title+SSR 16). These runs overlap and must not be added as independent coverage. Scoped lint covers the actual file: 0 errors, 2 unchanged warnings, 0 added. The root pages-directory advisory is documented. No new class-only test or type-level code; no clean full-typecheck claim.

The historical owned-path manifest also hashes the task record at its earlier checkpoint. Official task notes change as evidence is recorded; that task hash is historical, not a requirement that the live task file stay frozen. Production hash remains the acceptance key.

## Integrity and retention

INDEX.md lists every retained file. retention-manifest.json records original paths, sizes and source/retained hashes; SHA256SUMS covers every file except itself. Five test logs had extra blank lines at EOF; only their trailing whitespace/final newline was normalized. The retained static checksum list was recomputed for its three affected log inputs. These six deliberate transformations and original/retained hashes are recorded in retention-manifest.json; all other 29 original artifacts remain byte-identical, including frozen sources and all native captures. Individual git diff --no-index --check checks report no remaining whitespace issues. Every input and generated text was checked against 14 known isolated runtime credential values and JWT/private-key patterns before writing. All 3 retained PNGs were inspected; no credentials are visible. See SCREENSHOT_REVIEW.md and capture-audit.json for exact limits.

No source/test/browser/runtime/global-document or commit change was made during packaging. Official TASK13260.39 notes record the targeted native pass and separate remaining drawer finding. Parent owns final source verification and commit.
