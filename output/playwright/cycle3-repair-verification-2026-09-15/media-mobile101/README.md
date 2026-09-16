# Media mobile width — UAT101 / TASK13260.42

**Targeted native PASS**, captured through 2026-09-16T00:19:40.253Z. Full fresh UAT remains parent-owned and is not completed by this bundle. Source frozen at 2026-09-16T00:15:52.146Z; exact three-file manifest retained. No source/test/browser/runtime changes were made during retention.

## Outcome

- At390×844, the library uses its existing toggle at x366/width24; collapsed reader width366 fits the viewport. Actions x251 and Find x286 fit. Actual Actions menu opens (see retained menu snapshot, not only the command). Find Aster returns1/1; input, previous, next and Close stay in view. Four PNGs were visually inspected.
- The full260-character synthetic Aster source is identical in mobile and desktop captures; audit records its hash.
- At1280×720, actual wheel moves the inner reader0→160 while documentTop stays0; owned PUT saves54.24%, cfi scroll:54.24, zoom100, HTTP200. Normal reload restores160 with two GET200s and no PUT in the captured1.5-second post-restoration observation. This is bounded evidence, not an indefinite no-write claim.

## Original failure and change

Original mobile739px clipping is retained once in [the .40 bundle](../media-progress099/uat099-final-native-mobile-bounds.txt), with [RED screenshot](../media-progress099/uat099-final-native-mobile.png) and [snapshot](../media-progress099/uat099-final-native-mobile-snapshot.txt). Do not duplicate those assets.

The three-file CSS-only change bounds nested flex panes and navigator, wraps viewer controls, and presents the existing mobile library above the reader with its toggle reachable. Desktop remains side-by-side; .40 progress hooks are unchanged. [Implementation report](uat101-repair-report.md) records diagnosis and exact test command; its pending-native wording is the historical implementation checkpoint, superseded by the final native result here. The independent report is also a pre-native review checkpoint.

## Verification

- Implementer85/16 and independent56/7 suites pass; overlapping counts must not be added.
- Scoped lint0errors/23unchanged warnings/0added. Compact comparison and independent scoped static verification retained; redundant source-bearing ESLint output excluded.
- Combined compiler90existing diagnostics,90current,0added/removed; this is unchanged baseline, not clean TypeScript.
- Independent review CLEAR; frozen hashes match. TSX class strings only: no Python Bandit scope.

## Evidence map

| Claim | Evidence |
|---|---|
| Mobile library and collapse | native-library-bounds, native-collapse, native-expanded-snapshot, native-library.png |
| Bounded reader and full source | native-reader-bounds, native-reader-snapshot, native-reader.png |
| Actual Actions popup | native-actions.txt, native-actions-menu-snapshot.yml |
| Find interaction | native-find-result.txt, native-find.png |
| Desktop owned save and reload | native-desktop-save.txt, native-desktop-reload.txt, native-desktop.png |
| Frozen source/review/tests/static checks | production-freeze, repair-report, independent-review, test logs and comparison JSONs |

All artifact basenames above have the uat101- prefix. retention-manifest.json records source/retained hashes and declared whitespace-only normalization. capture-audit.json records credential scan and JSON/PNG/source verification. SHA256SUMS covers every other bundle file. Raw source files, private manifests, authentication headers and unrelated captures are excluded. Browser tab summaries contain a previously existing chrome-error tab; this Media run stayed on the application URL and made no claim about that unrelated tab.
