# Wave 5 Ownership Inventory

Date: 2026-04-17

## Scope decision

Wave 5 is intentionally bounded to the smallest reviewable set of overlapping slices that still proves the pre-mixin ownership guard pattern:

- In-scope slices: `admin`, `workspace-api`, `presentations`
- Deferred follow-on slices: `models-audio`, `characters`, `collections`, `media`, `chat-rag`

These three in-scope slices were chosen because they are materially smaller than the deferred domains, have clear method ownership boundaries, and let this wave lock the overlap baseline without changing the public mixed-client surface. The deferred domains still overlap heavily, but they carry much more surface area and are better handled in later tasks after the base-vs-facade boundary is explicit in code and tests.

## Live overlap inventory

This inventory reflects the current branch state after the Wave 5 bounded de-shadow work and matches `apps/packages/ui/src/services/tldw/client-ownership.ts`.

| Domain | Overlap count | Overlapping methods |
| --- | ---: | --- |
| `admin` | 0 | none; de-shadowed in Wave 5 |
| `workspace-api` | 0 | none; de-shadowed in Wave 5 |
| `presentations` | 0 | none; de-shadowed in Wave 5 |
| `models-audio` | 23 | `createTtsJob`, `generateImage`, `getEmbeddingModelsList`, `getEmbeddingProvidersConfig`, `getImageBackends`, `getLlamacppStatus`, `getLlmProviders`, `getMediaIngestionBudgetDiagnostics`, `getMlxStatus`, `getModels`, `getModelsMetadata`, `getProviders`, `getSystemStats`, `getTranscriptionModelHealth`, `getTranscriptionModels`, `getTtsJobArtifacts`, `listLlamacppModels`, `loadMlxModel`, `startLlamacppServer`, `stopLlamacppServer`, `synthesizeSpeech`, `transcribeAudio`, `unloadMlxModel` |
| `characters` | 23 | `createCharacter`, `createCharacterChatSession`, `deleteCharacter`, `deleteCharacterChatSession`, `diffCharacterVersions`, `exportCharacter`, `filterCharactersByTags`, `getCharacter`, `getCharacterListIdentity`, `importCharacterFile`, `listAllCharacters`, `listCharacterChatSessions`, `listCharacterMessages`, `listCharacterVersions`, `listCharacters`, `listCharactersPage`, `normalizeCharacterListResponse`, `restoreCharacter`, `revertCharacter`, `searchCharacters`, `sendCharacterMessage`, `streamCharacterMessage`, `updateCharacter` |
| `collections` | 45 | `addReadingItem`, `bulkUpdateItems`, `bulkUpdateReadingItems`, `createHighlight`, `createNote`, `createOutputTemplate`, `createPrompt`, `createReadingDigestSchedule`, `createReadingSavedSearch`, `deleteHighlight`, `deleteOutputTemplate`, `deleteReadingDigestSchedule`, `deleteReadingItem`, `deleteReadingSavedSearch`, `downloadOutput`, `exportReadingList`, `generateOutput`, `generateReadingItemTts`, `getHighlights`, `getItems`, `getOutputTemplates`, `getPrompts`, `getReadingDigestSchedule`, `getReadingImportJob`, `getReadingItem`, `getReadingList`, `importReadingList`, `linkReadingItemToNote`, `listNotes`, `listOutputs`, `listReadingDigestSchedules`, `listReadingImportJobs`, `listReadingItemNoteLinks`, `listReadingSavedSearches`, `previewTemplate`, `searchNotes`, `searchPrompts`, `summarizeReadingItem`, `unlinkReadingItemNote`, `updateHighlight`, `updateOutputTemplate`, `updatePrompt`, `updateReadingDigestSchedule`, `updateReadingItem`, `updateReadingSavedSearch` |
| `media` | 39 | `addMedia`, `addMediaForm`, `bulkUpdateMediaKeywords`, `createAnnotation`, `createImageArtifact`, `deleteAnnotation`, `deleteDataTable`, `deleteMedia`, `deleteReadingProgress`, `exportDataTable`, `generateDataTable`, `generateDocumentInsights`, `getDataTable`, `getDataTableJob`, `getDocumentFigures`, `getDocumentOutline`, `getDocumentReferences`, `getMediaDetails`, `getMediaIngestJob`, `getMediaStatistics`, `getReadingProgress`, `listAnnotations`, `listDataTables`, `listMedia`, `listMediaIngestJobs`, `permanentlyDeleteMedia`, `regenerateDataTable`, `reprocessMedia`, `restoreMedia`, `saveDataTableContent`, `searchMedia`, `submitMediaIngestJobs`, `syncAnnotations`, `translate`, `updateAnnotation`, `updateDataTable`, `updateMediaKeywords`, `updateReadingProgress`, `uploadMedia` |
| `chat-rag` | 101 | `addChatMessage`, `addDictionaryEntry`, `addWorldBookEntry`, `attachWorldBookToCharacter`, `bulkDictionaryEntries`, `bulkWorldBookEntries`, `cancelChatDocumentJob`, `cancelChatbookExportJob`, `cancelChatbookImportJob`, `chatDocumentStatistics`, `chatQueueActivity`, `chatQueueStatus`, `chatbooksHealth`, `cleanupChatbooks`, `completeCharacterChatTurn`, `completeChat`, `createChat`, `createChatCompletion`, `createConversationShareLink`, `createDictionary`, `createWorldBook`, `deleteChat`, `deleteChatDocument`, `deleteDictionary`, `deleteDictionaryEntry`, `deleteMessage`, `deleteWorldBook`, `deleteWorldBookEntry`, `detachWorldBookFromCharacter`, `dictionaryActivity`, `dictionaryStatistics`, `dictionaryVersionSnapshot`, `dictionaryVersions`, `downloadChatbookExport`, `editMessage`, `exportChatbook`, `exportDictionaryJSON`, `exportDictionaryMarkdown`, `exportWorldBook`, `generateChatDocument`, `getCharacterPromptPreview`, `getChat`, `getChatDocument`, `getChatDocumentJob`, `getChatDocumentPrompt`, `getChatLorebookDiagnostics`, `getChatSettings`, `getChatbookExportJob`, `getChatbookImportJob`, `getDictionary`, `getMessage`, `getWorldBookRuntimeConfig`, `importChatbook`, `importDictionaryJSON`, `importDictionaryMarkdown`, `importWorldBook`, `listCharacterWorldBooks`, `listChatCommands`, `listChatDocuments`, `listChatMessages`, `listChatbookExportJobs`, `listChatbookImportJobs`, `listChats`, `listChatsWithMeta`, `listConversationShareLinks`, `listDictionaries`, `listDictionaryEntries`, `listWorldBookEntries`, `listWorldBooks`, `normalizeChatSummary`, `persistCharacterCompletion`, `prepareCharacterCompletion`, `previewChatbook`, `processDictionary`, `processWorldBookContext`, `ragHealth`, `ragSearch`, `ragSearchStream`, `ragSimple`, `removeChatbookExportJob`, `removeChatbookImportJob`, `reorderDictionaryEntries`, `resolveConversationShareLink`, `restoreChat`, `revertDictionaryVersion`, `revokeConversationShareLink`, `saveChatDocumentPrompt`, `saveChatKnowledge`, `searchChatMessages`, `streamCharacterChatCompletion`, `streamChatCompletion`, `streamCompleteChat`, `updateChat`, `updateChatSettings`, `updateDictionary`, `updateDictionaryEntry`, `updateWorldBook`, `updateWorldBookEntry`, `validateDictionary`, `webSearch`, `worldBookStatistics` |

## Deferred follow-on handoff

Wave 5 completed the bounded de-shadow work for `admin`, `workspace-api`, and `presentations`. The remaining follow-on work should start from the deferred domains below:

Wave 5 also repaired the `verify:openapi` maintainer path so the check no longer assumes a tracked `apps/extension/openapi.json` snapshot. The command now prefers that snapshot when present and otherwise derives the OpenAPI spec from the checked-out backend.

During that repair, the command also exposed reviewed follow-on drift outside the Wave 5 slice:

- `/api/v1/billing/*` remains a hosted/legacy UI surface even though the OSS backend explicitly removes public billing routes.
- `/api/v1/media/bulk/keyword-update` is currently an optional optimization path because the client already falls back to per-item keyword updates.
- `/api/v1/media/statistics` remains a stale media-client surface and should be resolved in the deferred media cleanup wave rather than folded into Wave 5.

- `models-audio` (23 overlaps): split models/provider discovery from audio/TTS/transcription before removing the remaining base-class duplication.
- `characters` (23 overlaps): separate list/search/session transport from persona normalization and streaming helpers, then de-shadow in smaller groups.
- `collections` (45 overlaps): split notes/prompts/reading-list behavior into smaller ownership slices before overlap removal.
- `media` (39 overlaps): separate core media operations from data tables, annotations, and ingest flows before de-shadowing.
- `chat-rag` (101 overlaps): treat this as a multi-wave effort and split chat core, RAG, dictionaries, world books, chatbooks, and streaming into distinct cleanup passes.
