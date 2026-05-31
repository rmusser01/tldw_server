# RAG Functional Preset Pipeline Guide

This page is legacy/historical context. Earlier RAG work described small functional preset pipelines as the primary contributor surface. Current development uses the unified RAG pipeline and the active implementation notes in [RAG-Developer-Guide.md](RAG-Developer-Guide.md).

Do not use this page as an active API or extension guide. New contributor work should start with the canonical developer guide and the current unified pipeline modules.

## Current Active Entrypoints

The active programmatic entrypoints are:

- `unified_rag_pipeline(...)`
- `unified_batch_pipeline(...)`
- `simple_search(...)`
- `advanced_search(...)`

These live in `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`.

## Current HTTP Surface

The active router prefix is `/api/v1/rag`. For endpoint details, see:

- [RAG API Documentation](../API-related/RAG_API_Documentation.md)
- [RAG API Consumer Guide](../API-related/RAG-API-Guide.md)

## Historical Note

The older preset-pipeline material was useful during the transition to a composable RAG architecture, but it no longer describes the supported contributor path. It is retained only to explain why some archived files or older discussions mention preset-style pipeline names.
