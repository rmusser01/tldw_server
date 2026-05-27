# Flashcards Study Guide

_Last updated: 2026-05-26_

This guide explains the Flashcards flow in `Study`, `Manage`, `Create & Import`, `Templates`, and `Scheduler`, including extension selected-text capture, scheduling basics, queue states, cloze syntax, and import/export formats.

## Card Images

Flashcard text fields now support inline images in `Front`, `Back`, `Extra`, and `Notes`.

How to add images:

- Use `Insert image` in the create drawer, edit drawer, or a document-mode row.
- The uploaded image is stored as a managed flashcard asset.
- The text field keeps a lightweight markdown reference instead of raw image bytes.
- Previews and read views resolve that reference through an authenticated image fetch.

Why the 8 KB field limit did not increase:

- The field cap still protects search indexing, list payload size, and editor performance.
- Inline image bytes would inflate `front`, `back`, and `notes` beyond what those text fields are meant to store.
- Managed asset references keep cards searchable and lightweight while letting images round-trip through export/import.

## Daily Study Workflow

1. Add cards through `Manage` manual creation, `Create & Import` import/generate/structured Q&A/image occlusion, or the extension selected-text handoff.
2. Open `Scheduler` when you want to tune a deck's spaced-repetition policy.
3. Open `Study` and review due cards.
4. Reveal the answer, then rate recall (`Again`, `Hard`, `Good`, `Easy`).
5. Use `Manage` when you want to inspect queue state on expanded cards or document rows while cleaning up a deck.
6. Repeat daily. The scheduler adjusts next due dates from your ratings.

## Extension Selected-Text Capture

Use this path when you find a useful passage while browsing and want to save one editable basic flashcard without leaving the sidepanel.

Workflow:

1. Select text on the source page.
2. Open the extension sidepanel and choose `Flashcards`.
3. Choose `Capture page selection`.
4. Select a deck, edit `Front` and `Back`, then choose `Save card`.
5. Use `Open full Flashcards` for generation, imports, templates, review, or broader deck management.

Important behavior:

- The sidepanel Flashcards route saves one basic card at a time and keeps the source page URL as provenance.
- Create the deck in full Flashcards first if the sidepanel shows no deck options.
- The browser context menu path still works: choose `tldw` -> `Save` -> `Save to Notes`, review the selection in the sidepanel dialog, then choose `Generate flashcards`.
- Use the context-menu path or full Web UI `/flashcards` -> `Create & Import` -> `Generate` when you want generated drafts instead of a single captured card.
- If the sidepanel cannot capture on the current site, use full Web UI `/flashcards` and paste text into `Create & Import`.

## Study Assistant Conflict Recovery

The flashcard study assistant and quiz remediation assistant now recover cleanly if the same thread changes in another tab or client.

What happens on a thread version conflict:

- The latest assistant thread is reloaded automatically.
- Your pending question or transcript is preserved instead of being dropped.
- The panel shows `Reload latest` and retry actions so you can continue without reopening the card or results view.

Retry labels:

- Text questions use `Retry my message`.
- Voice transcript fact-checks use `Retry transcript review`.

## Quiz Remediation Conversion State

Quiz-results remediation flashcards now use server-backed conversion records instead of browser-only session state.

What changed:

- `Already converted` status survives reloads, browser changes, and new devices.
- Each missed question keeps one active remediation conversion plus any superseded history.
- `Convert Again Anyway` creates a fresh active conversion and leaves the older one in history as `superseded`.
- If linked remediation cards were deleted later, the quiz results view marks that conversion as stale and lets you reconvert normally.

`Study Linked Cards` behavior:

- If all active remediation conversions for an attempt point to the same live deck, the handoff keeps that deck filter.
- If conversions span multiple decks, the handoff still opens Flashcards study for the quiz attempt, but without a deck filter.

## Scheduler Tab

Open `Scheduler` from the top-level Flashcards tabs to edit deck-level review policy.

What it includes:

- A searchable deck list with compact scheduler summaries.
- A scheduler-type selector for `SM-2+` or `FSRS` on each deck.
- A per-deck editor for step timing, interval growth, leech handling, and fuzz.
- Built-in presets:
  - `Default`: backend defaults for balanced daily review.
  - `Fast acquisition`: shorter early steps and shorter easy intervals.
  - `Conservative review`: slower acquisition and stronger long-term spacing.
  - `High retention`: FSRS preset tuned for shorter review intervals.
  - `Long horizon`: FSRS preset tuned for larger mature decks.
- `Copy settings` to clone another deck's scheduler into the current draft.
- `Reset to defaults` to restore the standard scheduler bundle.
- Active-deck counts for `Due review`, `New`, `Learning`, and total due cards.

Important behavior:

- New decks start with the default scheduler settings.
- New decks default to `SM-2+`, but you can switch them to `FSRS` before saving.
- Scheduler edits are deck-scoped; they do not affect other decks unless you copy them.
- Unsaved scheduler drafts are guarded when you switch decks or leave the `Scheduler` tab.
- If another client updates the same deck first, the tab shows `Reload latest` and `Reapply my draft` actions.
- Switching an existing deck to `FSRS` keeps review history intact and bootstraps FSRS state conservatively as cards are reviewed.

## Scheduler Settings During Deck Creation

You can now set scheduler policy at deck-creation time instead of creating the deck first and fixing it later.

Flashcards creation flows:

- `Manage` manual card creation
- `Create & Import` structured Q&A preview save
- `Create & Import` generated flashcards save
- `Create & Import` image occlusion save

How it works:

- Existing deck selectors now include `Create new deck`.
- Choosing that option opens a deck-name field plus the same scheduler preset/editor used by the scheduler UI.
- The inline editor also lets you choose the scheduler type for the new deck.
- Existing decks remain read-only in those flows and show a compact scheduler summary instead.
- New decks created there start with exactly the scheduler settings you selected.

Quiz remediation:

- `Results` → `Create Flashcards from Missed Questions` also supports `Create new deck`.
- The remediation modal lets you set scheduler settings for the new deck before conversion runs.
- If you choose an existing deck, the modal shows that deck's scheduler summary.

Later edits:

- If you need to change a deck after creation, open the `Scheduler` tab and edit that deck there.

## Queue States

Queue-state badges now appear on the active card in `Study`, on expanded cards in `Manage`, and on document-mode rows. The active card in `Study` also shows a small scheduler badge (`SM-2+` or `FSRS`) so you can see which policy is driving the interval previews.

Meanings:

- `New`: the card has not graduated into regular review yet.
- `Learning`: the card is moving through short learning steps.
- `Review`: the card is on the long-term review schedule.
- `Relearning`: the card lapsed and is moving through relearn steps.
- `Suspended`: the card is out of automatic rotation.
  - `Suspended (Leech)` means it hit the leech threshold.
  - `Suspended (Manual)` means it was suspended intentionally.

## Ratings and Scheduling Basics

The review buttons map to SM-2 quality values:

| Button | Quality | Meaning |
|---|---:|---|
| Again | 0 | You forgot the card; repeat soon. |
| Hard | 2 | You recalled with effort; keep interval short. |
| Good | 3 | Normal successful recall. |
| Easy | 5 | Very easy recall; increase interval more aggressively. |

Scheduling terms shown in UI:

- `Memory strength` (`ef`): Ease factor controlling interval growth.
- `Next review gap` (`interval_days`): Days until next scheduled review.
- `Recall runs` (`repetitions`): Successful recall count.
- `Relearns` (`lapses`): Times the card was forgotten after learning.

## Cloze Syntax

Use cloze cards when you want blanks inside sentence context.

- Required pattern: `{{c1::answer}}`
- Multiple deletions are allowed (`{{c2::...}}`, `{{c3::...}}`, etc.)
- At least one cloze deletion must appear in `Front` text for cloze cards

Example:

`The powerhouse of the cell is the {{c1::mitochondrion}}.`

## Manage Document Mode

Use `Manage` → `Doc` when you want to review and clean up a large filtered deck in one continuous scroll instead of page-by-page cards.

Document mode supports:

- Continuous loading of the filtered result set as you scroll
- Inline row editing for `Front`, `Back`, `Deck`, `Tags`, `Notes`, and `Template`
- Immediate per-row saves with row-local conflict recovery
- Inline undo after a successful row save

Behavior notes:

- Document mode only supports stable `Due date` and `Created` sorting.
- If a multi-tag query hits the scan cap, a truncation banner appears and `Select all` across results is disabled for that view.
- `Cmd/Ctrl+Enter` saves the active row.
- `Escape` cancels the active row edit and restores the last saved values.
- Use `Open drawer` on a row when you need the full edit surface, preview, or scheduling controls.

## Import and Export Formats

### Delimited (CSV/TSV)

Accepted header names include:

- `Deck`, `Front`, `Back`, `Tags`, `Notes`, `Extra`
- `Model_Type`, `Reverse`, `Is_Cloze`, `Deck_Description`

Notes:

- Without headers, default column order is `Deck, Front, Back, Tags, Notes`.
- `Tags` can be comma-separated or space-separated.
- If imports fail by row, use row-level errors in `Create & Import` and jump to inline format help.

### JSON / JSONL

Supported fields:

- `deck`/`deck_name`
- `front`/`question`
- `back`/`answer`
- `tags` (array or string)
- `notes`, `extra`, `model_type`, `reverse`, `is_cloze`

Accepted payload forms:

- JSON array of items
- JSON object containing `items`
- JSONL (`.jsonl` / `.ndjson`) with one object per line

### Structured Q And A Preview

Use `Create & Import` -> `Structured Q&A` when you already wrote your own questions and answers and only want the app to convert them into flashcards without LLM rewriting.

Accepted label pairs:

- `Q:` with `A:`
- `Question:` with `Answer:`

Preview rules:

- Each labeled block becomes an editable draft before anything is saved.
- Continuation lines stay attached to the current question or answer until the next labeled block.
- Blank lines are allowed.
- Saving selected drafts writes standard `basic` flashcards into the chosen deck.
- Unlabeled freeform notes are not inferred into cards in v1.

### Image Occlusion Authoring

Use `Create & Import` -> `Image Occlusion` when you want to turn one labeled screenshot, diagram, or slide into several image-backed cards.

Workflow:

- Upload one source image.
- Draw rectangular occlusion regions directly on the preview.
- Add an answer label for each region.
- Generate drafts. The app uploads:
  - one normalized source image
  - one masked prompt image per region
  - one highlighted answer image per region
- Review/edit the generated drafts, then save them in bulk.

Saved card shape:

- `Front` contains a prompt line plus the masked image.
- `Back` contains the label plus the highlighted answer image.
- `Notes` store a readable `[image-occlusion]` metadata block with source ref and normalized geometry.
- Cards are saved as standard `basic` flashcards with `source_ref_type = manual`.

Current limits:

- V1 supports one source image per generation run.
- Regions are rectangular only.
- Each region must have a label before drafts can be generated.
- The generation run is capped at 25 regions.
- Existing occlusion cards do not reopen in the authoring surface yet; edit the resulting flashcards directly instead.

### APKG Export

APKG export is available from `Create & Import` and preserves scheduling metadata for Anki import.

Image-backed cards:

- Managed flashcard image references are converted into packaged Anki media on export.
- APKG import rewrites packaged media back into managed flashcard asset references.
- `Notes` are included in `tldw` APKG round-trips even though that is not part of Anki's default basic/cloze field set.
- Large media-heavy APKG imports or exports can be rejected if they exceed the configured total media cap.

## Troubleshooting

- `Missing required field`: verify header mapping and ensure required values are present.
- `Invalid cloze`: ensure `Front` includes at least one `{{cN::...}}` deletion.
- `Line too long`: check delimiter selection and malformed line breaks.
- `Maximum import`: split into smaller batches and retry.
