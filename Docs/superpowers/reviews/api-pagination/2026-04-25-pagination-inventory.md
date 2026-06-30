# API Pagination Inventory

- Date: 2026-04-25
- Scope: `tldw_Server_API/app/api/v1/endpoints`, `tldw_Server_API/app/api/v1/schemas`, and `tldw_Server_API/app/api/v1/endpoints/_pagination_utils.py`
- Related plan: `Docs/superpowers/plans/2026-04-25-phase3-2-pagination-standardization-implementation-plan.md`

## Summary

This inventory is the starting point for Phase 3.2. The API currently has several pagination dialects. The standardization work should first add shared helpers and compatibility aliases, then migrate route families in small slices.

Measured on 2026-04-25:

- `411` pagination-like query parameter declarations were found in endpoint files.
- `173` schema classes contain pagination-like fields.
- `_pagination_utils.py` currently only builds RFC5988 `Link` headers; it does not normalize request parameters or response metadata.

## Route-Family Styles

Route-family style counts:

- `56` families use offset-style parameters only.
- `16` families mix offset-style and page-style parameters.
- `9` families use page-style parameters only.
- `4` families mix cursor-style and offset-style parameters.
- `1` family mixes cursor-style and page-style parameters.

Offset-only examples:

- `admin/admin_acp_agents`
- `admin/admin_api_keys`
- `audio/audio_jobs`
- `chat`
- `claims`
- `data_tables`
- `flashcards`
- `notes`
- `quizzes`
- `slides`
- `storage`
- `writing`

Offset plus page examples:

- `admin/admin_budgets`
- `admin/admin_usage`
- `admin/admin_user`
- `character_chat_sessions`
- `characters_endpoint`
- `kanban/kanban_boards`
- `kanban/kanban_cards`
- `media/listing`
- `prompts`
- `reading`
- `watchlists`

Page-only examples:

- `collections_feeds`
- `connectors`
- `items`
- `privileges`
- `prompt_studio/prompt_studio_optimization`
- `prompt_studio/prompt_studio_projects`
- `prompt_studio/prompt_studio_prompts`
- `prompt_studio/prompt_studio_test_cases`

Cursor/hybrid examples:

- `audio/audio_history`: cursor plus offset.
- `evaluations/evaluations_crud`: `after` plus limit.
- `notifications`: `after` plus offset.
- `paper_search`: provider cursors plus page-style routes.
- `workflows`: cursor plus offset.

## Query Parameter Dialects

Canonical or near-canonical offset fields:

- `limit`
- `offset`
- `rows_limit`
- `rows_offset`

Page-based fields:

- `page`
- `per_page`
- `results_per_page`
- `page_size`
- `page_number`

Cursor-based fields:

- `cursor`
- `after`
- `before`

Some route families already mark `page`/`per_page` as legacy compatibility aliases, especially in kanban routes. Those are good models for a compatibility-first migration.

## Schema Dialects

Representative response schemas:

- `chat_conversation_schemas.ConversationListPagination`: `limit`, `offset`, `total`, `has_more`.
- `kanban_schemas.PaginationInfo`: `total`, `limit`, `offset`, `has_more`.
- `document_references.DocumentReferencesResponse`: `offset`, `limit`, `has_more`, `next_offset`.
- `media_response_models.PaginationInfo`: `page`, `results_per_page`, `total_pages`, `total_items`.
- `media_response_models.PaginationInfoSearch`: `page`, `per_page`, `total`, `total_pages`.
- `prompt_studio_base.PaginationMetadata`: `page`, `per_page`, `total`, `total_pages`.
- `openai_eval_schemas.ListResponse`: `has_more`.
- `openai_eval_schemas.ListQueryParams`: `limit`, `after`.
- `audio_schemas.TTSHistoryListResponse`: `total`, `limit`, `offset`, `next_cursor`.
- `workflows.WorkflowRunListResponse`: `next_offset`, `next_cursor`.

The main split is not just parameter naming. Some routes report `total`, some report `has_more`, some report `total_pages`, and some provide continuation fields without a complete nested `pagination` object.

## Canonical Target

Offset metadata should converge on:

```json
{
  "pagination": {
    "mode": "offset",
    "limit": 50,
    "offset": 0,
    "total": 123,
    "has_more": true,
    "next_offset": 50
  }
}
```

Cursor metadata should converge on:

```json
{
  "pagination": {
    "mode": "cursor",
    "limit": 50,
    "cursor": "input-cursor-or-null",
    "next_cursor": "opaque-token-or-null",
    "has_more": true
  }
}
```

Legacy fields should remain in migrated route families until frontend and extension callers are updated.

## Pilot Recommendations

Recommended offset pilot:

1. `skills`
   - Small, offset-style list endpoint.
   - Good first place to prove helper ergonomics.
2. `slides`
   - Multiple list/search endpoints with typed schemas.
   - Useful after `skills` because it exercises several list shapes.
3. `data_tables`
   - Exercises list pagination and row-window pagination through `rows_limit`/`rows_offset`.

Recommended cursor pilot:

1. `workflows`
   - Already has `cursor`, `next_cursor`, and `Link` header behavior.
   - Larger surface, but pagination semantics are central to the module.
2. `audio/audio_history`
   - Smaller cursor/offset hybrid.
   - Good fallback if workflows is too broad.

Avoid first:

- `media/listing`, because it mixes `page`, `per_page`, `results_per_page`, `total_pages`, and old media response contracts.
- `paper_search`, because provider compatibility and provider-specific cursor/page semantics are part of its public contract.
- `admin/*`, because auth/audit/tenant scoping raises migration risk.

## Next Action

Add shared pagination schemas and request-normalization helpers, then migrate `skills` as the first offset pilot. Treat cursor work as a separate PR after the offset helper is proven.
