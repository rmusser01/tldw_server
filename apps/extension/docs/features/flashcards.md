---
title: Flashcards (Experimental)
---

# Flashcards (Experimental)

The Flashcards page lets you create, review, and manage spaced-repetition cards backed by your tldw_server. It also supports CSV/TSV import and export, plus optional .apkg export where supported by the server.

Access it from the Web UI header by clicking the Layers icon, or navigate to `/flashcards`. In the extension sidepanel, the Flashcards entry opens a compact bridge with actions for the full Flashcards workspace and selected-text generation.

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
3. Click `Generate from page selection`.
4. The full Web UI opens Flashcards on the `Create & Import` tab with the selected text prefilled in the generation workflow.
5. Generate, review/edit the drafts, choose a deck or create one, then save the cards.

You can also use the browser context menu path: `tldw` -> `Save` -> `Save to Notes`, then choose `Generate flashcards` from the sidepanel review dialog. The sidepanel Flashcards route does not save cards by itself; the save happens after you review and save generated drafts in the full Flashcards workspace.

## Tips

- Due cards: Use the Study tab’s next-card flow to step through scheduled cards.
- Cloze cards: Toggle “Is Cloze” and select `cloze` model when creating/editing.
- Tags: Use the tags input in Create & Import or Manage edit flows; tags are stored as a JSON array.

> Note: Flashcards features are marked Experimental in the server API and may change.
