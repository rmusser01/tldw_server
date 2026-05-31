# API Tag Index

This index maps OpenAPI tags to the most relevant documentation pages in the docs site. Tags follow a consistent kebab-case convention to make navigation and grouping predictable in Swagger and ReDoc.

Notes:
- Some tags group multiple endpoints; the linked doc covers the primary usage and examples.
- If a tag shows “Coming soon”, endpoints exist but dedicated docs are not yet published here.

| Tag | Documentation |
|-----|---------------|
| `chat` | API-related/Chat_API_Documentation.md |
| `messages` | API-related/Anthropic_Messages_API.md |
| `chat-dictionaries` | API-related/Chatbook_Features_API_Documentation.md#chat-dictionary-api |
| `chat-documents` | API-related/Chatbook_Features_API_Documentation.md#document-generator-api |
| `rag-unified` | API-related/RAG-API-Guide.md |
| `rag-health` | API-related/RAG-API-Guide.md |
| `prompt-studio` | API-related/Prompt_Studio_API.md |
| `chatbooks` | API-related/Chatbook_API_Documentation.md |
| `embeddings` | API-related/Embeddings_API_Documentation.md |
| `vector-stores` | API-related/API_Design.md |
| `ocr` | API-related/OCR_API_Documentation.md |
| `providers` (`llm`) | API-related/Providers_API_Documentation.md |
| `chunking-templates` | API-related/Chunking_Templates_API_Documentation.md |
| `audio` | API-related/Audio_Transcription_API.md |
| `audio-jobs` | API-related/Audio_Jobs_API.md |
| `Media Ingestion Jobs` | API-related/Media_Ingest_Jobs_API.md |
| `evaluations` | API-related/Evaluations_API_Unified_Reference.md |
| `reading` | API-related/Reading_List_API.md |
| `collections-feeds` | API-related/Collections_Feeds_API.md |
| `benchmarks` | Coming soon |
| `characters` | CHARACTER_CHAT_API_DOCUMENTATION.md |
| `character-chat-sessions` | API-related/Character_Chat_Sessions_API.md |
| `character-messages` | API-related/Character_Messages_API.md |
| `flashcards` | Coming soon |
| `mcp-unified` | MCP/Unified/Developer_Guide.md |
| `workflows` | Coming soon |

The `chatbooks` tag includes chatbook export/import, OpenWebUI JSON and database import, and OpenWebUI attachment hydration preview/job endpoints.

If you spot a mismatch between tags and docs, please open an issue or PR.
