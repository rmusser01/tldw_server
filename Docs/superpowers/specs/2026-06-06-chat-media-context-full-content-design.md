# Chat Media Context Full-Content Design

## Problem

Users report that adding a media item as context from `/chat` can insert or send only the media title and metadata, not the media content. The exact clicked control is unknown, so the fix should cover every `/chat` media-result action that turns a media search result into chat context.

## Root Cause

`/api/v1/media/search` returns list-shaped media items with `id`, `title`, `type`, and `url`. The backend search repository selects `m.content`, but the media search endpoint formats results as `MediaListItem`, which does not include content or `content_preview`.

The `/chat` Knowledge panel normalizes those search rows into RAG-style results. When no content is present, it creates a fallback snippet like `Library item: <title>`. Some action paths already fetch full media details before inserting context, but pinning/saving and ask-style paths can format and store the fallback snippet directly. When pinned results are later appended to a prompt, the chat receives only that fallback text.

## Goals

- Ensure media search results used as chat context resolve to full media text whenever the result has a media ID.
- Keep `/api/v1/media/search` lightweight for listing/search use.
- Make Insert, Attach, Ask, Preview Insert, and Pin/Save behavior consistent for media results.
- Preserve existing RAG chunk behavior: chunks returned by the RAG endpoint should still insert the retrieved chunk, not force whole-document expansion.

## Non-Goals

- Do not change the media search API response contract in this slice.
- Do not redesign the Knowledge panel UI.
- Do not force full media content into prompts when file retrieval is enabled and the user is relying on media-scoped RAG instead.

## Design

Use the existing full-media resolver as the central boundary for actions that convert a media library search result into context.

To avoid expanding RAG chunks that happen to include `media_id`, normalized media-library search results should carry an explicit origin marker: `metadata.origin = "media-library"`. Only results with that marker are eligible for whole-media expansion. RAG/QA source chunks should remain chunk-scoped even when their metadata includes `media_id`.

- Convert a result to `RagPinnedResult`.
- If it has `mediaId` and the media-library origin marker, fetch `/api/v1/media/{id}` with `include_content=true`, `include_versions=false`, and `include_version_content=false`.
- Extract content from the returned detail using the existing media-detail extraction helper.
- Replace the snippet with full text only when non-empty full text is available.
- Fall back to the existing snippet on fetch failure or missing content.

Apply this resolver before:

- Direct Knowledge Search Insert.
- File Search Attach.
- Preview modal Insert.
- Preview modal Ask.
- Knowledge Search Ask for a media result.
- Pin/Save of a media result.
- Clipboard copy of a media search result if it is exposed as a context-copy action.

For Pin/Save, store the resolved full-text snippet in `ragPinnedResults`. The existing submit path can keep using `formatPinnedResults(ragPinnedResults, "markdown")`; it will then append real content rather than `Library item: <title>`.

## Data Flow

1. User searches media from `/chat` Knowledge panel.
2. `/api/v1/media/search` returns lightweight media rows.
3. UI normalizes rows into `RagResult` with `metadata.media_id` and a media-library origin marker.
4. User chooses an action that creates context.
5. The action resolves full media content through the shared resolver.
6. The action inserts, sends, pins, or copies the formatted resolved context.

## Error Handling

If fetching full media details fails, keep the current fallback behavior and do not block the user action. The UI should not throw from context insertion. This preserves offline/degraded behavior while making successful connected flows include content.

## Testing

Add focused frontend tests around the Knowledge helpers/actions:

- A lightweight media search row normalizes with `metadata.media_id` and `metadata.origin = "media-library"`.
- A media result with only `id/title/type/url` resolves full content from `getMediaDetails`.
- Pin/Save stores full content when detail content is available.
- Ask formats full content for media results.
- Preview Ask formats full content for media results.
- Existing fallback snippet remains when detail fetch fails or content is empty.
- RAG chunk insertion remains chunk-scoped and is not expanded to full media content even if the chunk metadata contains `media_id`.

## Backlog

Tracked by `TASK-527`.
