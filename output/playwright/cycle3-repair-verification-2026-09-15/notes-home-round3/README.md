# Targeted Notes and Home repair verification

Existing isolated multi-user profile and normal Bob session; this is not a new fresh-install run. Reviewed frontend checkpoints: Notes af14ac1258, Home f45f7749c1, titles/guidance545a7a59d2. The frontend was restarted into a new isolated build directory to remove stale development errors; no runtime credentials/configuration/data were reset. Backend session patch982b03a940 was not yet running and is not certified here. Concurrent Flashcards/QA repairs remain excluded.

## Observed results

- Two synthetic notes created through the actual UI; both requests201. Five notes and five recent entries are available. No optional monitoring/alerts request followed the ordinary-user saves.
- At1280×720 with Views/Organize/Filters expanded, controls have749px scrollable content within299px, and the result list retains209px inner scroll space. Actual Tab traverses checkbox→pin→Open note, then Enter opens the fourth note. The keyboard-transition record intentionally still shows the old editor during its request; the settled desktop/reload records confirm the requested fourth note.
- Reopening after reload preserves the exact synthetic text and Saved status. At390×844, Browse notes selects the fifth note, the closed drawer finishes at right0, and document width remains390. Both retained PNGs were inspected.
- Home displays Reading Queue Setup required while personalization is unavailable, and Automation access restricted for tasks.read. Capability discovery and notification reads return200; no scheduled-task request is sent.
- Actual Media, Media Analysis and Knowledge routes receive their own document titles. Notes/Home also show correct titles. Flashcards and remaining title checks paused on a transient development build error resolving the newly added, uncommitted Review hook; no product acceptance claim is made for those routes.

## New finding and limits

UAT091P3: desktop header squeezes the title and stacks Saved/3m/ago across3lines. The before-fix PNG deliberately retains this failure; a narrow wrapping repair is underway under13260.18. All findings remain open until required targeted controls and the full fresh single/multi run pass.

An explicitly opened Notes Connections panel returned the intended graph.read403 on its neighbor endpoint. This was not an automatic monitoring request or a failed save. Earlier stale-build captures and probe-only strict-locator/URL-evaluation errors are not represented as product outcomes. No private request headers, tokens or runtime secrets are included.
