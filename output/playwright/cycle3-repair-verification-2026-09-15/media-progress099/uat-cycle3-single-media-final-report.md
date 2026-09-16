# Targeted single-user Media follow-up — TASK13260.17 /13260.25

## Outcome

**Reading-progress acceptance fails at the actual scroll boundary (parent assigned UAT099/task13260.40).** The content container does not scroll, so no progress PUT is produced and valid zoom100 payload behavior cannot be certified live. **Flashcard→Media source-link acceptance was not exercised:** the existing unfiltered library has seven cards, all with Note provenance; none has a Media source action. No source/card creation or inference was used as a workaround.

Browser session `cycle3-single-20260915`, UI18280/API18200, tab0. Began with the existing Aster Chat, inspected its snapshot and used the visible Media shortcut. Ended online and paused at `http://127.0.0.1:18280/media?id=1`, selected `pasted-text`, original Medium text size restored. Other tabs were not operated. No source/test/global-document/Git/runtime/restart actions.

## Actual scroll and text-size controls

1. Selected existing synthetic Aster Media1 through the actual Media result row. GET3667 returned200 with the260-character source; GET3666 progress returned `{ "media_id": 1, "has_progress": false }`.
2. At1280×720, `content-scroll-container` has `clientHeight=2361`, `scrollHeight=2361`, `scrollTop=0`; its rect starts at y180 and extends2361pixels. The document has scrollable height2541. It is not an internally scrollable panel.
3. Changed the visible text-size button M→L. M/L is a separate display preference; the reading-progress hook uses fixed `zoom_level:100`. No claim that L should change that API field.
4. Moved the mouse over the actual content container and sent a560pixel wheel gesture. Document scrollTop became560, while the content container stayed0 with equal client/scroll heights. Waited1400ms, exceeding the hook's900ms debounce. Request inventory through3736 contains **no PUT `/media/1/progress`**. No request body or status is invented for a request that never happened; no422 occurred in this bounded check.
5. Normal reload kept `/media?id=1` selected and L pressed, but returned the document to0 with the same unbounded container geometry. Restored M through the actual button.
6. After the card-provenance check, returned directly to the existing source and captured fresh actual GET138/detail200 and GET143/progress200. Progress still reports `has_progress:false`. The final screenshot confirms Medium and exact source selected.

Evidence: `geometry-before.txt`, `wheel.txt`, `geometry-reloaded.txt`, `requests-after-wheel.txt`, `new-requests-after-wheel.txt`, `source-reloaded.txt`, `restore-medium.txt`, `progress-before.json`, `progress-returned.json`, `paused.txt`, and `wheel.png` / `paused.png`, all with prefix `uat-cycle3-single-media-final-`. Both PNGs visually inspected.

## Source preservation

The complete content object and stored versions are identical before/after. Content length260; SHA256 `239e2c002fe7304abae231f9a23816b65eb15e85d06b499249500ad80cd4ec3b` before and after. See `source-comparison.json`, `source-hash-before.json`, and the actual safe detail responses `source-response-before.txt` / `source-response-returned.json`. No source text, analysis, keyword or version mutation occurred.

Two response-body requests issued after navigating away referenced evicted cached IDs3944 and3901. The tool reported request-not-found; `source-response-after.txt` and `progress-after.txt` are empty and must be excluded from valid evidence. Fresh GET138/143 at the returned Media page supplied the final actual responses instead.

## Existing Flashcard provenance

Visited actual `/flashcards?tab=manage`; no search/deck filter, All selected,7Cards, one page with disabled next. All seven visible source links are `/notes?source_ref_id=...`; zero `/media` links. Saved card IDs/content were not altered and no card was rated. Evidence `card-library.txt` and `card-sources.txt`. Therefore .25 Media source-link behavior remains **not exercised**, not passed or failed.

## Limits and next action

Parent owns the separate UAT099 layout/listener correction. This follow-up does not override the earlier unit-tested zoom-unit repair, but it prevents claiming native end-to-end progress persistence. Browser remains paused on the existing source for a post-repair check. The source/hash and screenshots can be retained independently of the large request inventories, which contain earlier session history.
