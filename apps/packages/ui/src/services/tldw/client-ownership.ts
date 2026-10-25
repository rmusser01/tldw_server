export const WAVE5_IN_SCOPE_SLICES = [
  "admin",
  "workspace-api",
  "presentations"
] as const

export const DEFERRED_OVERLAP_DOMAINS = [
  "models-audio",
  "characters",
  "collections",
  "media",
  "chat-rag"
] as const

export const TRANSITIONAL_DOMAIN_OVERLAPS = {
  "admin": [],
  "workspace-api": [],
  "presentations": [],
  "models-audio": [],
  "characters": [
    "exportCharacter",
    "filterCharactersByTags",
    "getCharacter",
    "importCharacterFile",
    "listCharacterMessages",
    "listCharacterVersions",
    "searchCharacters",
    "sendCharacterMessage",
    "streamCharacterMessage"
  ],
  "collections": [
    "generateReadingItemTts",
    "listNotes",
    "listReadingSavedSearches",
    "searchNotes",
    "summarizeReadingItem",
    "updateHighlight",
    "updateOutputTemplate",
    "updateReadingItem"
  ],
  "media": [
    "createAnnotation",
    "createImageArtifact",
    "deleteAnnotation",
    "deleteReadingProgress",
    "generateDocumentInsights",
    "getDataTable",
    "getDocumentFigures",
    "getDocumentOutline",
    "getDocumentReferences",
    "getMediaDetails",
    "getMediaIngestJob",
    "getReadingProgress",
    "listAnnotations",
    "listMedia",
    "listMediaIngestJobs",
    "regenerateDataTable",
    "reprocessMedia",
    "saveDataTableContent",
    "searchMedia",
    "syncAnnotations",
    "translate",
    "updateAnnotation",
    "updateDataTable",
    "updateMediaKeywords",
    "updateReadingProgress"
  ],
  "chat-rag": [
    "addChatMessage",
    "appendRoleplayResumeInput",
    "attachWorldBookToCharacter",
    "completeCharacterChatTurn",
    "createChat",
    "createChatCompletion",
    "createConversationShareLink",
    "deleteChat",
    "deleteMessage",
    "dictionaryActivity",
    "dictionaryVersions",
    "editMessage",
    "getChat",
    "getChatLorebookDiagnostics",
    "getChatSettings",
    "getMessage",
    "importChatbook",
    "listChatMessages",
    "listChats",
    "listChatsWithMeta",
    "listConversationShareLinks",
    "listDictionaryEntries",
    "persistCharacterCompletion",
    "ragSearchStream",
    "reorderDictionaryEntries",
    "restoreChat",
    "revokeConversationShareLink",
    "streamCharacterChatCompletion",
    "streamChatCompletion",
    "streamCompleteChat",
    "updateChat",
    "updateChatSettings",
    "updateWorldBook"
  ]
} as const
/**
 * Stage 3 (TASK-13526, 2026-10-25): the previously recorded ~190-entry
 * overlap inventory was deleted down to the surviving names below (the
 * runtime implementations are the domain mixins; see the
 * prototype-ownership guard). Survivors are either documented keeps
 * (createChatCompletion — deliberately re-pinned to the base by the
 * composition root; getCharacter — bisect-isolated as not safely
 * shadowed) or domain methods whose base copies post-date the stage-3
 * sweep (sibling programs kept landing). The wave-5 slices (admin,
 * workspace-api, presentations) and models-audio are fully clear.
 */


/**
 * Stage 3 (TASK-13526, 2026-10-25): the transitional overlap inventory is
 * terminal — every shadowed base copy recorded here has been deleted (the
 * runtime implementations were always the domain mixins; see the
 * prototype-ownership guard). The one surviving base copy whose name
 * collides with a mixin is getCharacter (characters), kept because its base
 * implementation is NOT safely shadowed (bisect-isolated; see the task
 * record).
 */

export type TransitionalOverlapDomain = keyof typeof TRANSITIONAL_DOMAIN_OVERLAPS
