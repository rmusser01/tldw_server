
import Dexie, { type Table } from 'dexie';
import {
  HistoryInfo,
  Message,
  Prompt,
  SessionFiles,
  UserSettings,
  Webshare,
  Model,
  ModelNickname,
  Folder,
  Keyword,
  FolderKeywordLink,
  ConversationKeywordLink,
  ProcessedMedia,
  CompareState,
  ContentDraft,
  DraftBatch,
  DraftAsset,
  AudiobookProject,
  AudiobookChapterAsset,
  TtsClip,
  SttRecordingRow,
  MediaReadAlongAudioCacheEntry
} from "./types"

export class PageAssistDexieDB extends Dexie {
  chatHistories!: Table<HistoryInfo>;
  messages!: Table<Message>;
  prompts!: Table<Prompt>;
  webshares!: Table<Webshare>;
  sessionFiles!: Table<SessionFiles>;
  userSettings!: Table<UserSettings>;

  customModels!: Table<Model>;
  modelNickname!: Table<ModelNickname>
  processedMedia!: Table<ProcessedMedia>
  compareStates!: Table<CompareState>
  contentDrafts!: Table<ContentDraft>
  draftBatches!: Table<DraftBatch>
  draftAssets!: Table<DraftAsset>

  // Folder system tables (cache from server)
  folders!: Table<Folder>
  keywords!: Table<Keyword>
  folderKeywordLinks!: Table<FolderKeywordLink>
  conversationKeywordLinks!: Table<ConversationKeywordLink>

  // Audiobook projects
  audiobookProjects!: Table<AudiobookProject>
  audiobookChapterAssets!: Table<AudiobookChapterAsset>
  ttsClips!: Table<TtsClip>

  // STT recordings
  sttRecordings!: Table<SttRecordingRow>

  // Media read-along generated audio cache
  mediaReadAlongAudioCache!: Table<MediaReadAlongAudioCacheEntry>

  constructor() {
    super('PageAssistDatabase');

    this.version(1).stores({
      chatHistories: 'id, title, is_rag, message_source, is_pinned, createdAt, doc_id, last_used_prompt, model_id',
      messages: 'id, history_id, name, role, content, createdAt, messageType, modelName',
      prompts: 'id, title, content, is_system, createdBy, createdAt',
      webshares: 'id, title, url, api_url, share_id, createdAt',
      sessionFiles: 'sessionId, retrievalEnabled, createdAt',
      userSettings: 'id, user_id',
      customModels: 'id, model_id, name, model_name, model_image, provider_id, lookup, model_type, db_type',
      modelNickname: 'id, model_id, model_name, model_avatar',
      processedMedia: 'id, url, createdAt'
    });

    // Version 2: Add timeline/branching fields for conversation tree visualization
    this.version(2).stores({
      chatHistories: 'id, title, is_rag, message_source, is_pinned, createdAt, doc_id, last_used_prompt, model_id, root_id, parent_conversation_id',
      messages: 'id, history_id, name, role, content, createdAt, messageType, modelName, parent_message_id',
      prompts: 'id, title, content, is_system, createdBy, createdAt',
      webshares: 'id, title, url, api_url, share_id, createdAt',
      sessionFiles: 'sessionId, retrievalEnabled, createdAt',
      userSettings: 'id, user_id',
      customModels: 'id, model_id, name, model_name, model_image, provider_id, lookup, model_type, db_type',
      modelNickname: 'id, model_id, model_name, model_avatar',
      processedMedia: 'id, url, createdAt'
    });

    // Version 3: Add folder system tables (cache from tldw_server)
    this.version(3).stores({
      chatHistories: 'id, title, is_rag, message_source, is_pinned, createdAt, doc_id, last_used_prompt, model_id, root_id, parent_conversation_id',
      messages: 'id, history_id, name, role, content, createdAt, messageType, modelName, parent_message_id',
      prompts: 'id, title, content, is_system, createdBy, createdAt',
      webshares: 'id, title, url, api_url, share_id, createdAt',
      sessionFiles: 'sessionId, retrievalEnabled, createdAt',
      userSettings: 'id, user_id',
      customModels: 'id, model_id, name, model_name, model_image, provider_id, lookup, model_type, db_type',
      modelNickname: 'id, model_id, model_name, model_avatar',
      processedMedia: 'id, url, createdAt',
      // Folder system: cache of server data
      folders: 'id, name, parent_id, deleted',
      keywords: 'id, keyword, deleted',
      folderKeywordLinks: '[folder_id+keyword_id], folder_id, keyword_id',
      conversationKeywordLinks: '[conversation_id+keyword_id], conversation_id, keyword_id'
    });

    // Version 4: Compare state + compare metadata on messages
    this.version(4).stores({
      chatHistories: 'id, title, is_rag, message_source, is_pinned, createdAt, doc_id, last_used_prompt, model_id, root_id, parent_conversation_id',
      messages: 'id, history_id, name, role, content, createdAt, messageType, modelName, clusterId, modelId, parent_message_id',
      prompts: 'id, title, content, is_system, createdBy, createdAt',
      webshares: 'id, title, url, api_url, share_id, createdAt',
      sessionFiles: 'sessionId, retrievalEnabled, createdAt',
      userSettings: 'id, user_id',
      customModels: 'id, model_id, name, model_name, model_image, provider_id, lookup, model_type, db_type',
      modelNickname: 'id, model_id, model_name, model_avatar',
      processedMedia: 'id, url, createdAt',
      folders: 'id, name, parent_id, deleted',
      keywords: 'id, keyword, deleted',
      folderKeywordLinks: '[folder_id+keyword_id], folder_id, keyword_id',
      conversationKeywordLinks: '[conversation_id+keyword_id], conversation_id, keyword_id',
      compareStates: 'history_id'
    });

    // Version 5: Content review drafts + batches + assets
    this.version(5).stores({
      chatHistories: 'id, title, is_rag, message_source, is_pinned, createdAt, doc_id, last_used_prompt, model_id, root_id, parent_conversation_id',
      messages: 'id, history_id, name, role, content, createdAt, messageType, modelName, clusterId, modelId, parent_message_id',
      prompts: 'id, title, content, is_system, createdBy, createdAt',
      webshares: 'id, title, url, api_url, share_id, createdAt',
      sessionFiles: 'sessionId, retrievalEnabled, createdAt',
      userSettings: 'id, user_id',
      customModels: 'id, model_id, name, model_name, model_image, provider_id, lookup, model_type, db_type',
      modelNickname: 'id, model_id, model_name, model_avatar',
      processedMedia: 'id, url, createdAt',
      folders: 'id, name, parent_id, deleted',
      keywords: 'id, keyword, deleted',
      folderKeywordLinks: '[folder_id+keyword_id], folder_id, keyword_id',
      conversationKeywordLinks: '[conversation_id+keyword_id], conversation_id, keyword_id',
      compareStates: 'history_id',
      contentDrafts: 'id, batchId, status, mediaType, createdAt, updatedAt, expiresAt',
      draftBatches: 'id, createdAt, updatedAt',
      draftAssets: 'id, draftId, createdAt'
    });

    // Version 6: add server chat mapping for local mirrors
    this.version(6).stores({
      chatHistories: 'id, title, is_rag, message_source, is_pinned, createdAt, doc_id, last_used_prompt, model_id, root_id, parent_conversation_id, server_chat_id',
      messages: 'id, history_id, name, role, content, createdAt, messageType, modelName, clusterId, modelId, parent_message_id',
      prompts: 'id, title, content, is_system, createdBy, createdAt',
      webshares: 'id, title, url, api_url, share_id, createdAt',
      sessionFiles: 'sessionId, retrievalEnabled, createdAt',
      userSettings: 'id, user_id',
      customModels: 'id, model_id, name, model_name, model_image, provider_id, lookup, model_type, db_type',
      modelNickname: 'id, model_id, model_name, model_avatar',
      processedMedia: 'id, url, createdAt',
      folders: 'id, name, parent_id, deleted',
      keywords: 'id, keyword, deleted',
      folderKeywordLinks: '[folder_id+keyword_id], folder_id, keyword_id',
      conversationKeywordLinks: '[conversation_id+keyword_id], conversation_id, keyword_id',
      compareStates: 'history_id',
      contentDrafts: 'id, batchId, status, mediaType, createdAt, updatedAt, expiresAt',
      draftBatches: 'id, createdAt, updatedAt',
      draftAssets: 'id, draftId, createdAt'
    });

    // Version 7: Audiobook projects
    this.version(7).stores({
      chatHistories: 'id, title, is_rag, message_source, is_pinned, createdAt, doc_id, last_used_prompt, model_id, root_id, parent_conversation_id, server_chat_id',
      messages: 'id, history_id, name, role, content, createdAt, messageType, modelName, clusterId, modelId, parent_message_id',
      prompts: 'id, title, content, is_system, createdBy, createdAt',
      webshares: 'id, title, url, api_url, share_id, createdAt',
      sessionFiles: 'sessionId, retrievalEnabled, createdAt',
      userSettings: 'id, user_id',
      customModels: 'id, model_id, name, model_name, model_image, provider_id, lookup, model_type, db_type',
      modelNickname: 'id, model_id, model_name, model_avatar',
      processedMedia: 'id, url, createdAt',
      folders: 'id, name, parent_id, deleted',
      keywords: 'id, keyword, deleted',
      folderKeywordLinks: '[folder_id+keyword_id], folder_id, keyword_id',
      conversationKeywordLinks: '[conversation_id+keyword_id], conversation_id, keyword_id',
      compareStates: 'history_id',
      contentDrafts: 'id, batchId, status, mediaType, createdAt, updatedAt, expiresAt',
      draftBatches: 'id, createdAt, updatedAt',
      draftAssets: 'id, draftId, createdAt',
      audiobookProjects: 'id, title, status, createdAt, updatedAt, lastOpenedAt',
      audiobookChapterAssets: 'id, projectId, chapterId, createdAt'
    });

    // Version 8: Chat TTS clip history
    this.version(8).stores({
      chatHistories: 'id, title, is_rag, message_source, is_pinned, createdAt, doc_id, last_used_prompt, model_id, root_id, parent_conversation_id, server_chat_id',
      messages: 'id, history_id, name, role, content, createdAt, messageType, modelName, clusterId, modelId, parent_message_id',
      prompts: 'id, title, content, is_system, createdBy, createdAt',
      webshares: 'id, title, url, api_url, share_id, createdAt',
      sessionFiles: 'sessionId, retrievalEnabled, createdAt',
      userSettings: 'id, user_id',
      customModels: 'id, model_id, name, model_name, model_image, provider_id, lookup, model_type, db_type',
      modelNickname: 'id, model_id, model_name, model_avatar',
      processedMedia: 'id, url, createdAt',
      folders: 'id, name, parent_id, deleted',
      keywords: 'id, keyword, deleted',
      folderKeywordLinks: '[folder_id+keyword_id], folder_id, keyword_id',
      conversationKeywordLinks: '[conversation_id+keyword_id], conversation_id, keyword_id',
      compareStates: 'history_id',
      contentDrafts: 'id, batchId, status, mediaType, createdAt, updatedAt, expiresAt',
      draftBatches: 'id, createdAt, updatedAt',
      draftAssets: 'id, draftId, createdAt',
      audiobookProjects: 'id, title, status, createdAt, updatedAt, lastOpenedAt',
      audiobookChapterAssets: 'id, projectId, chapterId, createdAt',
      ttsClips: 'id, createdAt, historyId, serverChatId, messageId, serverMessageId, provider'
    });

    // Version 9: Prompt soft delete support with deletedAt field
    this.version(9).stores({
      chatHistories: 'id, title, is_rag, message_source, is_pinned, createdAt, doc_id, last_used_prompt, model_id, root_id, parent_conversation_id, server_chat_id',
      messages: 'id, history_id, name, role, content, createdAt, messageType, modelName, clusterId, modelId, parent_message_id',
      prompts: 'id, title, content, is_system, createdBy, createdAt, deletedAt',
      webshares: 'id, title, url, api_url, share_id, createdAt',
      sessionFiles: 'sessionId, retrievalEnabled, createdAt',
      userSettings: 'id, user_id',
      customModels: 'id, model_id, name, model_name, model_image, provider_id, lookup, model_type, db_type',
      modelNickname: 'id, model_id, model_name, model_avatar',
      processedMedia: 'id, url, createdAt',
      folders: 'id, name, parent_id, deleted',
      keywords: 'id, keyword, deleted',
      folderKeywordLinks: '[folder_id+keyword_id], folder_id, keyword_id',
      conversationKeywordLinks: '[conversation_id+keyword_id], conversation_id, keyword_id',
      compareStates: 'history_id',
      contentDrafts: 'id, batchId, status, mediaType, createdAt, updatedAt, expiresAt',
      draftBatches: 'id, createdAt, updatedAt',
      draftAssets: 'id, draftId, createdAt',
      audiobookProjects: 'id, title, status, createdAt, updatedAt, lastOpenedAt',
      audiobookChapterAssets: 'id, projectId, chapterId, createdAt',
      ttsClips: 'id, createdAt, historyId, serverChatId, messageId, serverMessageId, provider'
    });

    // Version 10: Unified Prompt schema with server sync support
    this.version(10).stores({
      chatHistories: 'id, title, is_rag, message_source, is_pinned, createdAt, doc_id, last_used_prompt, model_id, root_id, parent_conversation_id, server_chat_id',
      messages: 'id, history_id, name, role, content, createdAt, messageType, modelName, clusterId, modelId, parent_message_id',
      prompts: 'id, title, content, is_system, createdBy, createdAt, deletedAt, serverId, studioProjectId, syncStatus, sourceSystem',
      webshares: 'id, title, url, api_url, share_id, createdAt',
      sessionFiles: 'sessionId, retrievalEnabled, createdAt',
      userSettings: 'id, user_id',
      customModels: 'id, model_id, name, model_name, model_image, provider_id, lookup, model_type, db_type',
      modelNickname: 'id, model_id, model_name, model_avatar',
      processedMedia: 'id, url, createdAt',
      folders: 'id, name, parent_id, deleted',
      keywords: 'id, keyword, deleted',
      folderKeywordLinks: '[folder_id+keyword_id], folder_id, keyword_id',
      conversationKeywordLinks: '[conversation_id+keyword_id], conversation_id, keyword_id',
      compareStates: 'history_id',
      contentDrafts: 'id, batchId, status, mediaType, createdAt, updatedAt, expiresAt',
      draftBatches: 'id, createdAt, updatedAt',
      draftAssets: 'id, draftId, createdAt',
      audiobookProjects: 'id, title, status, createdAt, updatedAt, lastOpenedAt',
      audiobookChapterAssets: 'id, projectId, chapterId, createdAt',
      ttsClips: 'id, createdAt, historyId, serverChatId, messageId, serverMessageId, provider'
    }).upgrade(tx => {
      // Migrate existing prompts to set default sync fields
      return tx.table('prompts').toCollection().modify(prompt => {
        if (prompt.syncStatus === undefined) {
          prompt.syncStatus = 'local';
        }
        if (prompt.sourceSystem === undefined) {
          prompt.sourceSystem = 'workspace';
        }
      });
    });

    // Version 11: Prompt usage tracking fields
    this.version(11).stores({
      chatHistories: 'id, title, is_rag, message_source, is_pinned, createdAt, doc_id, last_used_prompt, model_id, root_id, parent_conversation_id, server_chat_id',
      messages: 'id, history_id, name, role, content, createdAt, messageType, modelName, clusterId, modelId, parent_message_id',
      prompts: 'id, title, content, is_system, createdBy, createdAt, deletedAt, serverId, studioProjectId, syncStatus, sourceSystem, usageCount, lastUsedAt',
      webshares: 'id, title, url, api_url, share_id, createdAt',
      sessionFiles: 'sessionId, retrievalEnabled, createdAt',
      userSettings: 'id, user_id',
      customModels: 'id, model_id, name, model_name, model_image, provider_id, lookup, model_type, db_type',
      modelNickname: 'id, model_id, model_name, model_avatar',
      processedMedia: 'id, url, createdAt',
      folders: 'id, name, parent_id, deleted',
      keywords: 'id, keyword, deleted',
      folderKeywordLinks: '[folder_id+keyword_id], folder_id, keyword_id',
      conversationKeywordLinks: '[conversation_id+keyword_id], conversation_id, keyword_id',
      compareStates: 'history_id',
      contentDrafts: 'id, batchId, status, mediaType, createdAt, updatedAt, expiresAt',
      draftBatches: 'id, createdAt, updatedAt',
      draftAssets: 'id, draftId, createdAt',
      audiobookProjects: 'id, title, status, createdAt, updatedAt, lastOpenedAt',
      audiobookChapterAssets: 'id, projectId, chapterId, createdAt',
      ttsClips: 'id, createdAt, historyId, serverChatId, messageId, serverMessageId, provider'
    }).upgrade(tx => {
      return tx.table('prompts').toCollection().modify(prompt => {
        if (typeof prompt.usageCount !== 'number' || Number.isNaN(prompt.usageCount)) {
          prompt.usageCount = 0;
        }
        if (prompt.lastUsedAt === undefined) {
          prompt.lastUsedAt = null;
        }
      });
    });

    // Version 12: STT recording blob persistence
    this.version(12).stores({
      chatHistories: 'id, title, is_rag, message_source, is_pinned, createdAt, doc_id, last_used_prompt, model_id, root_id, parent_conversation_id, server_chat_id',
      messages: 'id, history_id, name, role, content, createdAt, messageType, modelName, clusterId, modelId, parent_message_id',
      prompts: 'id, title, content, is_system, createdBy, createdAt, deletedAt, serverId, studioProjectId, syncStatus, sourceSystem, usageCount, lastUsedAt',
      webshares: 'id, title, url, api_url, share_id, createdAt',
      sessionFiles: 'sessionId, retrievalEnabled, createdAt',
      userSettings: 'id, user_id',
      customModels: 'id, model_id, name, model_name, model_image, provider_id, lookup, model_type, db_type',
      modelNickname: 'id, model_id, model_name, model_avatar',
      processedMedia: 'id, url, createdAt',
      folders: 'id, name, parent_id, deleted',
      keywords: 'id, keyword, deleted',
      folderKeywordLinks: '[folder_id+keyword_id], folder_id, keyword_id',
      conversationKeywordLinks: '[conversation_id+keyword_id], conversation_id, keyword_id',
      compareStates: 'history_id',
      contentDrafts: 'id, batchId, status, mediaType, createdAt, updatedAt, expiresAt',
      draftBatches: 'id, createdAt, updatedAt',
      draftAssets: 'id, draftId, createdAt',
      audiobookProjects: 'id, title, status, createdAt, updatedAt, lastOpenedAt',
      audiobookChapterAssets: 'id, projectId, chapterId, createdAt',
      ttsClips: 'id, createdAt, historyId, serverChatId, messageId, serverMessageId, provider',
      sttRecordings: 'id, createdAt'
    });

    // Version 13: Structured prompt sync metadata defaults
    this.version(13).stores({
      chatHistories: 'id, title, is_rag, message_source, is_pinned, createdAt, doc_id, last_used_prompt, model_id, root_id, parent_conversation_id, server_chat_id',
      messages: 'id, history_id, name, role, content, createdAt, messageType, modelName, clusterId, modelId, parent_message_id',
      prompts: 'id, title, content, is_system, createdBy, createdAt, deletedAt, serverId, studioProjectId, syncStatus, sourceSystem, usageCount, lastUsedAt',
      webshares: 'id, title, url, api_url, share_id, createdAt',
      sessionFiles: 'sessionId, retrievalEnabled, createdAt',
      userSettings: 'id, user_id',
      customModels: 'id, model_id, name, model_name, model_image, provider_id, lookup, model_type, db_type',
      modelNickname: 'id, model_id, model_name, model_avatar',
      processedMedia: 'id, url, createdAt',
      folders: 'id, name, parent_id, deleted',
      keywords: 'id, keyword, deleted',
      folderKeywordLinks: '[folder_id+keyword_id], folder_id, keyword_id',
      conversationKeywordLinks: '[conversation_id+keyword_id], conversation_id, keyword_id',
      compareStates: 'history_id',
      contentDrafts: 'id, batchId, status, mediaType, createdAt, updatedAt, expiresAt',
      draftBatches: 'id, createdAt, updatedAt',
      draftAssets: 'id, draftId, createdAt',
      audiobookProjects: 'id, title, status, createdAt, updatedAt, lastOpenedAt',
      audiobookChapterAssets: 'id, projectId, chapterId, createdAt',
      ttsClips: 'id, createdAt, historyId, serverChatId, messageId, serverMessageId, provider',
      sttRecordings: 'id, createdAt'
    }).upgrade(tx => {
      return tx.table('prompts').toCollection().modify(prompt => {
        if (prompt.promptFormat === undefined) {
          prompt.promptFormat = 'legacy';
        }
        if (prompt.promptSchemaVersion === undefined) {
          prompt.promptSchemaVersion = null;
        }
        if (prompt.structuredPromptDefinition === undefined) {
          prompt.structuredPromptDefinition = null;
        }
        if (prompt.syncPayloadVersion === undefined) {
          prompt.syncPayloadVersion = 1;
        }
      });
    });

    // Version 14: Media read-along generated audio cache
    this.version(14).stores({
      chatHistories: 'id, title, is_rag, message_source, is_pinned, createdAt, doc_id, last_used_prompt, model_id, root_id, parent_conversation_id, server_chat_id',
      messages: 'id, history_id, name, role, content, createdAt, messageType, modelName, clusterId, modelId, parent_message_id',
      prompts: 'id, title, content, is_system, createdBy, createdAt, deletedAt, serverId, studioProjectId, syncStatus, sourceSystem, usageCount, lastUsedAt',
      webshares: 'id, title, url, api_url, share_id, createdAt',
      sessionFiles: 'sessionId, retrievalEnabled, createdAt',
      userSettings: 'id, user_id',
      customModels: 'id, model_id, name, model_name, model_image, provider_id, lookup, model_type, db_type',
      modelNickname: 'id, model_id, model_name, model_avatar',
      processedMedia: 'id, url, createdAt',
      folders: 'id, name, parent_id, deleted',
      keywords: 'id, keyword, deleted',
      folderKeywordLinks: '[folder_id+keyword_id], folder_id, keyword_id',
      conversationKeywordLinks: '[conversation_id+keyword_id], conversation_id, keyword_id',
      compareStates: 'history_id',
      contentDrafts: 'id, batchId, status, mediaType, createdAt, updatedAt, expiresAt',
      draftBatches: 'id, createdAt, updatedAt',
      draftAssets: 'id, draftId, createdAt',
      audiobookProjects: 'id, title, status, createdAt, updatedAt, lastOpenedAt',
      audiobookChapterAssets: 'id, projectId, chapterId, createdAt',
      ttsClips: 'id, createdAt, historyId, serverChatId, messageId, serverMessageId, provider',
      sttRecordings: 'id, createdAt',
      mediaReadAlongAudioCache: 'id, createdAt, lastUsedAt, mediaId, mediaKind, segmentId, settingsSignature, textHash'
    });
  }
}

export const db = new PageAssistDexieDB();
