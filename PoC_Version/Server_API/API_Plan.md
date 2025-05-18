# API Plan

## Overview
- First stab at a mapping of the API endpoints and their functionality

- **Media Management**
```
    GET /api/media - List all media items
    GET /api/media/{id} - Get details of a specific media item
    POST /api/media - Add a new media item
    PUT /api/media/{id} - Update an existing media item
    DELETE /api/media/{id} - Delete a media item
```

- **Transcription and Summarization**
```
    POST /api/transcribe - Transcribe audio/video
    POST /api/summarize - Summarize text content
```

- **Keyword Management**
```
    GET /api/keywords - List all keywords
    POST /api/keywords - Add a new keyword
    DELETE /api/keywords/{keyword} - Delete a keyword
```

- **Search**
```
    GET /api/search - Search across all content
```

- **PDF Operations**
```
    POST /api/pdf/parse - Parse a PDF file
    POST /api/pdf/ingest - Ingest a PDF into the database
```

- **Book Operations**
```
    POST /api/book/ingest - Ingest a book into the database
```

- **Chat Management**
```
    GET /api/chat - List all chat conversations
    GET /api/chat/{id} - Get details of a specific chat conversation
    POST /api/chat - Create a new chat conversation
    POST /api/chat/{id}/message - Add a message to a chat conversation
    PUT /api/chat/{id}/message/{message_id} - Update a chat message
    DELETE /api/chat/{id}/message/{message_id} - Delete a chat message
```

- **Document Versioning**
```
    GET /api/document/{id}/versions - List all versions of a document
    GET /api/document/{id}/versions/{version_number} - Get a specific version of a document
    POST /api/document/{id}/versions - Create a new version of a document
```

- **RAG (Retrieval-Augmented Generation)**
```
    POST /api/rag/search - Perform a RAG search
```

- **Embedding Management**
```
    POST /api/embeddings - Create embeddings for content
    GET /api/embeddings/{id} - Get embeddings for a specific item
```

- **Prompt Management**
```
    GET /api/prompts - List all prompts
    GET /api/prompts/{id} - Get details of a specific prompt
    POST /api/prompts - Create a new prompt
    PUT /api/prompts/{id} - Update an existing prompt
    DELETE /api/prompts/{id} - Delete a prompt
```

- **Import Management**
```
    POST /api/import - Import content from an external source
```

- **Trash Management**
```
    GET /api/trash - List items in trash
    POST /api/trash/{id} - Move an item to trash
    DELETE /api/trash/{id} - Permanently delete an item from trash
    POST /api/trash/{id}/restore - Restore an item from trash
```


