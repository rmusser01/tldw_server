# API Plan

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
    - **Search**
        Each search endpoint accepts a JSON payload with the search query and returns a list of results.
        Media search supports RAG and FTS. 
        Chat supports FTS
        Prompts supports FTS
        Characters supports FTS
        Keywords supports FTS
        ```
            GET /api/v1/media/search - Search across all ingested media content
            GET /api/v1/chats/search - Search across all chat conversations
            GET /api/v1/prompts/search - Search across all prompts
            GET /api/v1/keywords/search - Search across all keywords
            GET /api/v1/characters/search_convos - Search across all character conversations
            GET /api/v1/characters/search - Search across all characters
        ```
- **`/api/v1/auth` endpoint**
    - **User Management**
        ```
            POST /api/v1/auth/register - Register a new user
            POST /api/v1/auth/login - Login a user
            POST /api/v1/auth/logout - Logout a user
            POST /api/v1/auth/forgot-password - Forgot password
            POST /api/v1/auth/reset-password - Reset password
            GET /api/v1/auth/user_details - Get current user details
            PUT /api/v1/auth/user_update - Update current user details
            DELETE /api/v1/auth/user_delete - Delete user
        ```
- **`/api/v1/media` endpoint**
    - **Media Processing & Ingestion**
        ```
            POST /api/v1/media/ingest - Ingest media content (Details specified in JSON payload) - persistent
            POST /api/v1/media/process - Process media content like you would for ingestion, but do not add the media content to DB - ephemeral
        ```
    - **Media Management**
        ```
            GET /api/v1/media - List all media items
            GET /api/v1/media/{id} - Get details of a specific media item      
            PUT /api/v1/media/{id} - Update an existing media item
            DELETE /api/v1/media/{id} - Delete a media item
        ```
    - **Embeddings**
        Subset of /media/ API endpoint.
        ```
            POST /api/v1/media/embeddings - Create embeddings for content
            GET /api/v1/media/embeddings/{id} - Get embeddings for a specific item
        ```
- **`/api/v1/analyze` endpoint**
    - **Analysis**
        ```
            POST /api/analyze - Analyze submitted content according to specified prompt. Not meant for chatting.
        ```
- **`/api/v1/keywords` endpoint**
    - **Keyword Management**
        ```
            GET /api/keywords - List all keywords
            POST /api/keywords - Add a new keyword
            DELETE /api/keywords/{keyword} - Delete a keyword
        ```
- **`/api/v1/chats` endpoint**
    - **Chat Management**
        ```
            GET /api/v1/chats - List all chat conversations for current user
            GET /api/v1/chats/{id} - Get details of a specific chat conversation
            POST /api/v1/chats - Create a new chat conversation
            POST /api/v1/chats/{id}/message - Add a message to a chat conversation
            PUT /api/v1/chats/{id}/message/{message_id} - Update a chat message
            DELETE /api/v1/chats/{id}/message/{message_id} - Delete a chat message
        ```
- **`/api/v1/prompts` endpoint**
    - **Prompts**
        ```
            GET /api/v1/prompts - List all prompts
            GET /api/v1/prompts/{id} - Get details of a specific prompt
            POST /api/v1/prompts - Create a new prompt
            PUT /api/v1/prompts/{id} - Update an existing prompt
            DELETE /api/v1/prompts/{id} - Delete a prompt
        ```
- **`/api/v1/audio` endpoint**
    - **TTS**
        ```
            GET /api/v1/audio/tts - List all available TTS options
            POST /api/v1/audio/speech - Generate TTS audio from text
        ```
    - **STT**
        ```
            GET /api/v1/audio/stt - List all available STT options
            POST /api/v1/audio/transcriptions - Convert audio to text
        ```
    - **S2S**
        ```
            GET /api/v1/audio/s2s - List all available Speech-2-Speech options
            POST /api/v1/audio/s2s - Submit speech for 'live' conversation
            POST /api/v1/audio/realtime - Carry on a real-time conversation
        ```
- **`/api/v1/import` & `/api/v1/export` endpoints**
    - **Import Management**
        ```
            GET /api/v1/import - List all import options
            POST /api/v1/import - Import content from an external source
        ```
    - **Export Management**
        ```
            GET /api/v1/export - List all export options
            POST /api/v1/export - Export content to an external source
        ```
- **`/api/v1/trash` endpoint**
    - **Trash Management**
        ```
            GET /api/v1/trash - List items in trash
            POST /api/v1/trash/{id} - Move an item to trash
            DELETE /api/v1/trash/{id} - Permanently delete an item from trash
            POST /api/v1/trash/{id}/restore - Restore an item from trash
        ```
- **`/api/v1/tools` endpoint**
    - **Anki Management**
        ```
            GET /api/v1/tools/anki - List all Anki options
            POST /api/v1/tools/anki - Generate Anki deck
        ```
    - **MindMap Generation**
        ```
            POST /api/v1/tools/mindmap - Generate a mindmap from content
        ```
    - **WebSearch**
        ```
            POST /api/v1/tools/websearch - Search the web for content
        ```
    - **DeepResearch**
        ```
            POST /api/v1/tools/deepresearch - Perform deep research on a topic
        ```
    - **Youtube Video DL**
        ```
            POST /api/v1/tools/youtube_dl - Download a Youtube video
        ```
    - **Youtube Audio DL**
        ```
            POST /api/v1/tools/youtube_audio - Download audio from a Youtube video
        ```
- **`/api/v1/llm` endpoint**
    - **Local LLM**
        ```
            GET /api/v1/llm - List all available Local LLM options
            POST /api/v1/llm - Provide configuration/parameters for Local LLM
            POST /api/v1/llm/generate - Generate content (text) from a Local LLM model
        ```
    - **Ollama Model Serving**
        ```
            GET /api/v1/llm/ollama - List all available Ollama options
            POST /api/v1/llm/ollama - Provide configuration options for Ollama
            POST /api/v1/llm/ollama/generate - Generate content (text) from an Ollama model
        ```
- **`/api/v1/evaluations` endpoint**
    - **Evaluation**
        ```
            GET /api/v1/evals - List all available evaluations
            POST /api/v1/evals - Evaluate a model
        ```

