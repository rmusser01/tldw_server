---
title: Flashcards (Experimental)
---

# Flashcards (Experimental)

The Flashcards page lets you create, review, and manage spaced-repetition cards backed by your tldw_server. It also supports CSV/TSV import and export, plus optional .apkg export where supported by the server.

Access it from the Web UI header by clicking the Layers icon, or navigate to `/flashcards`. In the extension sidepanel, the Flashcards entry opens a compact flashcards tool with actions for the full Flashcards workspace, selected-text card creation, native generated drafts, native template application, compact due-card review, and selected-text generation handoff.

## Prerequisites

- The extension must be connected to a running tldw_server instance (Options -> Settings -> tldw Server).
- You need to be signed in or configured with an API key if your server requires authentication.

## Tabs

- Study: Fetches the next due card, optionally by deck, reveals the answer, and lets you submit `Again`, `Hard`, `Good`, or `Easy`. The server schedules the next review via `/api/v1/flashcards/review`.
- Manage: Search/filter by deck/tag/due status, paginate, select cards, bulk move/delete/export, edit cards, or delete them with expected-version protection.
- Create & Import: Manually create cards, import delimited/JSON/APKG content, generate draft flashcards from text, run structured Q&A preview, author image occlusion cards, and export decks.
- Templates: Manage reusable flashcard templates and field presets.
- Scheduler: Edit deck-level scheduler policy, presets, queue visibility, and conflict recovery.

## Extension Selected-Text Flow

1. Select text on a web page.
2. Open the extension sidepanel and choose `Flashcards`.
3. Click `Capture page selection` to add the selection to the sidepanel draft queue.
4. Click `Generate draft cards` to create a small batch of editable draft cards in the sidepanel from the selected text.
5. Use `Apply template` on a queued draft when an existing template should reshape its fields before saving.
6. Choose a deck, edit each draft's `Front` and `Back` fields, then click `Save card` or `Save all`.
7. Click `Generate from selection` when you want the selected text to open in full Flashcards generation with the page URL/title attached.
8. Click `Review due card` to review the next due card for the selected deck, reveal the answer, and submit `Again`, `Hard`, `Good`, or `Easy`.
9. Use `Open full Flashcards` when you need imports, richer review tools, or broader deck management.

The sidepanel saves captured cards and generated draft cards from the same queue, keeping the page URL as source provenance. Native generation uses compact defaults, and `Apply template` reuses the templates already managed in full Flashcards. `Review due card` is a compact due-card loop; use full Flashcards for imports, cram, assistant support, analytics, and broader deck management. You can also use the browser context menu path: `tldw` -> `Save` -> `Save to Notes`, then choose `Generate flashcards` from the sidepanel review dialog when you want the fuller generation workflow in the full Flashcards workspace.

## Tips

- Due cards: Use the Study tab’s next-card flow to step through scheduled cards.
- Cloze cards: Toggle “Is Cloze” and select `cloze` model when creating/editing.
- Tags: Use the tags input in Create & Import or Manage edit flows; tags are stored as a JSON array.

> Note: Flashcards features are marked Experimental in the server API and may change.
