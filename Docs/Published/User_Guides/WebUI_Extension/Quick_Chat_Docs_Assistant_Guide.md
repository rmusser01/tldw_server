# Quick Chat Docs Assistant Guide

This guide covers the Quick Chat Helper assistant modes that were added to improve discoverability:

- `Chat`: normal model chat
- `Docs Q&A`: asks the backend RAG endpoint for documentation-style answers with citations
- `Browse Guides`: per-page Tutorials + browsable pre-written Q/A workflow cards

Design rationale and architecture details are documented in
[`Docs/Design/Quick_Chat_Docs_Assistant.md`](../../Design/Quick_Chat_Docs_Assistant.md).

Primary implementation files:

- `apps/packages/ui/src/components/Common/QuickChatHelper/QuickChatHelperModal.tsx`
- `apps/packages/ui/src/routes/option-quick-chat-popout.tsx`
- `apps/packages/ui/src/hooks/useQuickChat.tsx`
- `apps/packages/ui/src/components/Common/QuickChatHelper/docs-rag-profile.ts`
- `apps/packages/ui/src/components/Common/QuickChatHelper/workflow-guides.ts`
- `apps/packages/ui/src/components/Common/QuickChatHelper/rag-response.ts`

## How To Use

1. Open Quick Chat Helper.
2. Pick one of the three modes using the segmented control.
3. For `Docs Q&A`, ask a question like:
   - "What page should I use to benchmark models?"
   - "What's the synopsis of this paper?"
   - "How do I do this on this page?"
4. For `Browse Guides`, search and either:
   - use `Tutorials for this page` to run a guided tour for the current route, or
   - click `Ask docs mode` to send the guide question into Docs Q&A, or
   - click `Open <page>` to navigate directly.

## What `Docs Q&A` Sends To RAG

Quick Chat uses `POST /api/v1/rag/search` via `tldwClient.ragSearch(...)`.

Current request profile behavior:

- Base retrieval:
  - `search_mode: "hybrid"`
  - `enable_generation: true`
  - `enable_citations: true`
  - `enable_reranking: true`
  - `reranking_strategy: "flashrank"`
  - strict mode defaults:
    - `sources: ["media_db"]`
    - `corpus: "media_db"`
    - `index_namespace: "project_docs"`
- Profile tuning:
  - synopsis-like queries increase recall/context (`top_k`, parent-document behavior, generation budget)
  - troubleshooting-like queries lower `min_score` to improve recall
- Route-aware query augmentation:
  - if user says "this page/current page/here", quick chat appends:
    - `Current page context: <label> (<route>)`

## What `Browse Guides` Uses

`Browse Guides` has two distinct layers:

1. `Tutorials for this page` (dynamic, route-scoped):
   - sourced from `src/tutorials/registry.ts`
   - shows only tutorials matching the active route
   - supports `Start`, `Replay`, and `Locked` (when prerequisites are not met)
2. Workflow cards (curated Q/A):
   - sourced from `QUICK_CHAT_WORKFLOW_GUIDES`
   - can be overridden in settings

Workflow card fields:

- title
- user-style question
- suggested answer
- page route
- tags

It is deterministic and does not call the backend by itself.

## Edit Per-Page Tutorials (Developer Workflow)

Per-page tutorial entries are not edited from settings JSON. Update them in code:

1. Add or edit a definition in `apps/packages/ui/src/tutorials/definitions/*.ts`.
2. Ensure the definition is included in `TUTORIAL_REGISTRY` in `apps/packages/ui/src/tutorials/registry.ts`.
3. Add/update i18n keys in:
   - `apps/packages/ui/src/assets/locale/en/tutorials.json`
   - `apps/packages/ui/src/public/_locales/en/tutorials.json`
4. Ensure target selectors exist and are stable (`data-testid` preferred).
5. Run tutorial/quick-chat tests.

## Edit Pre-Written Workflow Cards In UI

1. Open `Settings -> Chat behavior`.
2. Find `Quick Chat workflow cards`.
3. Edit the JSON in `Quick Chat workflow cards JSON`.
4. Click `Save workflow cards`.

Expected JSON shape:

```json
[
  {
    "id": "custom-guide-id",
    "title": "Guide title",
    "question": "User-style question",
    "answer": "Curated answer",
    "route": "/research-studio",
    "routeLabel": "Research Studio",
    "tags": ["workflow", "discovery"]
  }
]
```

Notes:

- `route` can be written as `research-studio` or `/research-studio`; it is normalized to start with `/`.
- Invalid or incomplete cards are ignored during validation.
- `Reset to built-in defaults` restores the shipped guide set.
- This setting only changes workflow cards; it does not change per-page Tutorials.

## Post-Processing Added To Docs Replies

Docs replies now include optional `Suggested Pages` in addition to normal answer/citations:

- recommendations are scored from:
  - user query
  - generated answer
  - citation titles/sources
  - current route
- each recommendation includes:
  - route label
  - route path
  - short reason
  - "(current page)" marker when applicable

## Important Scope Note

Current behavior is now **strict by default** for project docs:

- quick chat docs mode scopes to `media_db` only
- uses `index_namespace: "project_docs"` unless overridden

Results are still only as good as the corpus hygiene. If non-project files are indexed into the same strict namespace, they can still appear.

`Browse Guides` is the only strictly curated pre-written Q/A layer.

## Strict Scope Configuration Keys

Quick chat reads the following storage keys:

1. `quickChatStrictDocsOnly` (`boolean`, default `true`)
2. `quickChatDocsIndexNamespace` (`string`, default `"project_docs"`)
3. `quickChatDocsProjectMediaIds` (`number[]`, JSON string, or comma-separated list)

When `quickChatDocsProjectMediaIds` is set, it is passed as `include_media_ids` to hard-limit retrieval to those media IDs.

You can configure these from the UI in `Settings -> Chat behavior` under `Quick Chat Docs Q&A scope`.

## If You Need Even Tighter "Project Docs Only"

1. Keep strict mode enabled.
2. Use a dedicated namespace that only contains official project docs.
3. Set `quickChatDocsProjectMediaIds` to the authoritative docs set.
4. Keep `Browse Guides` as deterministic curated guidance.

## Testing

Feature-specific tests live in:

- `apps/packages/ui/src/components/Common/QuickChatHelper/__tests__/docs-rag-profile.test.ts`
- `apps/packages/ui/src/components/Common/QuickChatHelper/__tests__/workflow-guides.test.ts`
- `apps/packages/ui/src/components/Common/QuickChatHelper/__tests__/rag-response.test.ts`

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Common/QuickChatHelper/__tests__
```
