# Knowledge, Media, And Sources

Use these pages when you want to add material, organize it, ask cited questions, review sources, or build research workspaces.

## Pages And Feature Sets

| Page or feature | Surface/status | What it lets you do | Common uses |
| --- | --- | --- | --- |
| `/knowledge` | WebUI, extension options | Ask cited questions over selected knowledge sources. | Library Q&A, evidence review, source-grounded answers. |
| `/search` | Legacy alias | Compatibility entry for Knowledge QA. | Old bookmarks and route compatibility. |
| `/research` | Advanced self-hosted | Work with research runs and provider-backed discovery. | Research sessions, source discovery, web/paper search. |
| `/workspaces` | Advanced self-hosted | Manage canonical workspaces. | Project organization, shared research contexts. |
| `/research-workspace` | Experimental/labs | Use a source-oriented research workspace. | Multi-source research, chat over curated material, sharing/export flows. |
| `/document-workspace` | Advanced self-hosted | Work in a document-centered workspace. | Deep reading, document chat, focused source analysis. |
| `/media` | WebUI, extension options | Browse and inspect ingested media. | Video/audio/document library review. |
| `/media-multi` | Advanced self-hosted | Select and operate on multiple media items. | Bulk review, multi-item curation. |
| `/media/[id]/view` | WebUI dynamic route | Inspect a single media item. | Transcript review, metadata inspection, source details. |
| `/review` | Advanced self-hosted | Use media review queue workflows. | Curation, review, cleanup. |
| `/media-trash` | Advanced self-hosted | Recover or clean deleted media. | Soft-delete recovery, library cleanup. |
| `/items` | Advanced self-hosted | Work with generic item records. | Library object management. |
| `/collections` | WebUI, extension options | Manage collections, reading items, highlights, templates, and import/export panels. | Reading workflows, feed-style collections, saved searches. |
| `/reading` | WebUI | Manage the reading-list workflow. | Saved articles, review queues. |
| `/notes` | WebUI, extension options | Create and manage notes. | Notebook workflows, source-linked notes, capture review. |
| `/sources`, `/sources/new`, `/sources/[sourceId]` | WebUI, extension options | Create, inspect, and manage ingestion sources. | Folder feeds, recurring imports, source status. |
| `/connectors`, `/connectors/browse`, `/connectors/jobs`, `/connectors/sources` | Advanced self-hosted | Browse connector placeholders, connector jobs, and connector source surfaces. | External systems, source adapters, connector status. |
| `/shared`, `/share/[token]` | Advanced self-hosted | View shared resources and public share links. | Collaboration, read-only links, shared workspaces. |

## Larger Systems

Knowledge workflows depend on source ownership. Media, notes, collections, sources, and workspaces create or organize material. Knowledge QA searches selected indexed material and produces cited answers. Research Workspace adds a higher-level project shell around source organization, chat, export, sharing, and recovery states.

If a Knowledge QA answer has weak or missing citations, narrow the source scope, check indexing, or inspect the source pages. If the backend is unreachable, use `/settings/health` before changing retrieval settings.

## Extension Differences

The extension can hand browser pages into source or quick-ingest workflows through context menu actions and sidepanel clipper flows. Extension failures can be caused by missing host permission, missing API key/login state, blocked background requests, or server URL mismatch.

## Related Docs

- [Knowledge QA guide](../WebUI_Extension/Knowledge_QA_Guide.md)
- [Web scraping and ingestion](../Server/Web_Scraping_Ingestion_Guide.md)
- [Ingestion sources API](../../API-related/Ingestion_Sources_API.md)
- [Reading list API](../../API-related/Reading_List_API.md)
- [Collections feeds API](../../API-related/Collections_Feeds_API.md)
- [RAG API guide](../../API-related/RAG-API-Guide.md)
- [Prototype workspaces](../Prototype_Workspaces.md)
