# API Design

## Introduction
Design document to outline the design of the API for the tldw Project.

## API Design
The API will be designed to be RESTful.

## Overview
- First stab at a mapping of the API endpoints and their functionality

- Very much a WIP, not complete and not all functionality is defined yet.

HTTP Everywhere.
Passing JSON data.
Overly descriptive vs asbtract.
Complexity hides in payloads vs endpoints
Liberal in allowed input, strict in parsing/operations on parsed data. (If it isn't valid, its junk)
- **URLs**
    - Main page: http://tldwproject.com
    - API Documentation page: http://tldwproject.com/docs
    - API Redoc page: http://tldwproject.com/redoc
    - API url: http://tldwproject.com/api/v1/
- **Endpoints**
- 
---------------------------------
### Resources
- The following resources will be available in the API:
    - **Auth/Users**
        * Manages accounts and permissions.
    - **Media**
        * Central resource for all ingested content (video, audio, docs).
    - **Chats**
        * For RAG, LLM, or character-based conversations.
    - **Characters**
        * Manages character “cards” (metadata) used in specialized chats.
    - **Prompts**
        * Manages prompt templates, cloning, and versioning.
    - **Keywords**
        * Tags for classification.
    - **Import/Export**
        * Managing data flows in/out.
    - **Trash**
        * Soft-deleted items.
    - **Tools**
        * Utility endpoints (mind maps, web search, YouTube DL, etc.).
    - **LLM** 
        * Local/remote model endpoints for text generation.
    - **Evaluations**
        * Evals to test or benchmark models.
    - **Search**
        * (Optional) A single place to handle multi-resource search if desired, or rely on resource-specific endpoints.
    - **Third Party**
        * Endpoints for 3rd party services/integrations
        * e.g. Arxiv, BioRxiv, etc.

### Endpoints
- **Overview of Available Endpoints**
    - Auth/Users:
        * Manages accounts and permissions.
    - Media:
        * Central resource for all ingested content (video, audio, docs).
    - Chats: For RAG, LLM, or character-based conversations.
    - Characters: Manages character “cards” (metadata) used in specialized chats.
    - Prompts: Manages prompt templates, cloning, and versioning.
    - Keywords: Tags for classification.
    - Import/Export: Managing data flows in/out.
    - Trash: Soft-deleted items.
    - Tools: Utility endpoints (mind maps, web search, YouTube DL, etc.).
    - LLM: Local/remote model endpoints for text generation.
    - Evaluations: Evals to test or benchmark models.
    - Search: (Optional) A single place to handle multi-resource search if desired, or rely on resource-specific endpoints.
    - 3rd_party: Endpoints for 3rd party services/integrations



#### The following endpoints will be available in the API:
- `Auth / Users - /api/v1/auth`
    * What it covers: User registration, login, logout, password management, user profiles, etc.
    - Endpoints could include:
        * `POST /api/v1/auth/register`
        * `POST /api/v1/auth/login`
        * `POST /api/v1/auth/logout`
        * `GET /api/v1/auth/me` (fetching current user’s details)
        * etc.
- `Media - /api/v1/media`
    * What it covers: All ingested or processed content—videos, audio, documents, PDF, text, and so on.
    - Endpoints typically include:
        * `GET /api/v1/media` — list/search media
        * `GET /api/v1/media/{id}` — get a single media item
        * `POST /api/v1/media` — create/ingest a new media item
        * `PUT /api/v1/media/{id}` — update media metadata, summary, etc.
        * `DELETE /api/v1/media/{id}` — remove media item (potentially into trash first)
    - Sub-resources (examples):
        * `/api/v1/media/{id}/embeddings` (create, retrieve, or delete embeddings)
        * `/api/v1/media/search` (search with query params)
        * `/api/v1/media/{id}/process` (to re-process a file or generate transcripts, if separate from the main create)
- `Chats - /api/v1/chats`
    - What it covers: 
        * All chat sessions, including RAG QA Chat, LLM chat, character chat, and notes.
    - Endpoints typically include:
        * `GET /api/v1/chats` — list or search chats
        * `GET /api/v1/chats/{id}` — get a single chat’s history
        * `POST /api/v1/chats` — create a new chat (whether RAG, LLM, or character-based)
        * `POST /api/v1/chats/{id}/message` — add a message to an existing chat
        * `PUT /api/v1/chats/{id}/message/{messageId}` — edit a message
        * `DELETE /api/v1/chats/{id}/message/{messageId}` — remove a message
    - Possible sub-resources:
        * `/api/v1/chats/{id}/notes`
        * `/api/v1/chats/{id}/metadata`
- Characters
    - What it covers: 
        * Character “cards,” multi-character chat, creation, validation.
    - Endpoints typically include:
        * `GET /api/v1/characters` — list/search available characters
        * `GET /api/v1/characters/{id}` — get details for one character
        * `POST /api/v1/characters` — create a new character card
        * `PUT /api/v1/characters/{id}` — update character details
        * `DELETE /api/v1/characters/{id}` — delete a character 
    - Sub-resources or expansions:
        * `/api/v1/characters/{id}/chat` — might tie in with “Chats” if you want specialized conversation endpoints for a given character.
- `Prompts - /api/v1/prompts`
    - What it covers: 
        * A database of prompts that can be searched, edited, cloned, or exported.
    - Endpoints typically include:
        * `GET /api/v1/prompts` — list/search all prompts
        * `GET /api/v1/prompts/{id}` — get one prompt
        * `POST /api/v1/prompts` — create new prompt
        * `PUT /api/v1/prompts/{id}` — edit an existing prompt
        * `DELETE /api/v1/prompts/{id}` — remove a prompt
    - Extras:
        * Cloning or exporting might be separate calls, e.g. POST /api/v1/prompts/{id}/clone or POST /api/v1/prompts/export.
- `Keywords - /api/v1/keywords`
    - What it covers: 
        * The tags or keywords associated with media, prompts, or other data.
    - Endpoints might include:
        * `GET /api/v1/keywords` — get the full list
        * `POST /api/v1/keywords` — create a new keyword
        * `DELETE /api/v1/keywords/{keyword}` — delete a keyword
    - Possibly integrated with other resources:
        * Because keywords often belong with media, prompts, or other objects, you might treat them as a separate resource or handle them within each object’s payload (e.g., adding a keywords array to the POST /api/v1/media body).
- `Import/Export - /api/v1/import, /api/v1/export`
    - What it covers: 
        * Bringing data in (Markdown, text, Obsidian vaults, MediaWiki, etc.) or sending data out (prompts, DB entries, conversation logs).
    - Endpoints might include:
        * `POST /api/v1/import` — import content (payload or config-driven)
        * `POST /api/v1/export` — export content (specify formats, filters)
    - Sub-resources if needed:
        * `/api/v1/import/obsidian`
        * `/api/v1/export/media`
- `Trash - /api/v1/trash`
    - What it covers: 
        * Items (media, prompts, chats, etc.) marked as “deleted,” but not permanently removed.
    - Endpoints might include:
        * `GET /api/v1/trash` — view items in trash
        * `POST /api/v1/trash/{id}/restore` — restore from trash
        * `DELETE /api/v1/trash/{id}` — permanently delete
- `Tools - /api/v1/tools`
    - What it covers: 
        * Miscellaneous “utility” features like mind map generation, web search, YouTube downloading, etc.
    - Endpoints might include:
        * `/api/v1/tools/mindmap`
        * `/api/v1/tools/websearch`
        * `/api/v1/tools/anki`
        * `/api/v1/tools/youtube_dl`
    * Each sub-route is basically a specialized operation. If these grow too large or too domain-specific, you might break them out into separate resources (e.g., /api/v1/anki, /api/v1/mindmap).
- `LLM - /api/v1/llm`
    - What it covers:
        * Local LLM or Ollama-based model serving endpoints for generation and config.
    - Endpoints:
        * `GET /api/v1/llm` — list available local LLM models
        * `POST /api/v1/llm` — configure and load a model
        * `POST /api/v1/llm/generate` — produce text from a local model
    * Similarly for Ollama, e.g. `POST /api/v1/llm/ollama/generate`.
- `Evaluations - /api/v1/evals`
    - What it covers: 
        * Evals like G-Eval, Infinite Bench, etc.
    - Endpoints:
        * `GET /api/v1/evals` — list all evals or configurations
        * `POST /api/v1/evals` — run an eval against a model or piece of content
- `Search - /api/v1/search` (Optional as its own resource)
    * You may unify search under each resource (e.g., `GET /api/v1/media?search=foo`), or you can have a universal endpoint that queries multiple resources at once (e.g., `GET /api/v1/search?query=foo&type=media,chats,prompts`).
    * If you do unify, you might omit a “search” resource and just rely on query parameters. If you want a single endpoint searching across all data, you might define something like `GET /api/v1/search` with resource-type filters.
- `3rd_party - /api/v1/3rd_party` (Optional as its own resource)
    - What it covers: 
        * Endpoints for 3rd party services or integrations.
        * E.g. Arxiv, BioRxiv, etc.
    - Endpoints:
        - `GET /api/v1/3rd_party/arxiv` — search Arxiv
        - `GET /api/v1/3rd_party/biorxiv` — search BioRxiv
        - etc.
----------------------------------



## Links
https://levelup.gitconnected.com/great-api-design-comprehensive-guide-from-basics-to-best-practices-9b4e0b613a44
https://github.com/TypeError/secure
