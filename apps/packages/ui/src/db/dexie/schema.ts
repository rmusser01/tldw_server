import type { HistoryBookmark, ForkOperation } from "./types";
import type { LegacyHistoryProjectionV1 } from "@/types/history-selection";

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
  MediaReadAlongAudioCacheEntry,
  SearchTokenRow
} from "./types"
// Static by design: the v17 upgrade handler (below) runs inside the
// version-change transaction and uses buildSearchTokenRows synchronously
// from this import. search-tokenizer has no runtime dependencies, so this
// cannot create a module cycle.
import { buildSearchTokenRows } from "./search-tokenizer"

export class PageAssistDexieDB extends Dexie {
  forkOperations!: Table<ForkOperation>;
  historySelections!: Table<HistoryBookmark>;
  historyProjections!: Table<LegacyHistoryProjectionV1>;
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

  // Message-content search token side table (WP-23 / TASK-13523)
  searchTokens!: Table<SearchTokenRow>

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
      mediaReadAlongAudioCache: 'id, createdAt, lastUsedAt, [lastUsedAt+sizeBytes+id], mediaId, mediaKind, segmentId, settingsSignature, textHash'
    });
    // Structure only: existing mirrors remain unbound; no ancestry rewrite.
    this.version(15).stores({
      historySelections: "[profile_id+client_session_id+owner_key+conversation_id], owner_key, conversation_id",
      historyProjections: "[owner_key+conversation_id+projection_id], owner_key, conversation_id"
    });
    this.version(16).stores({
      forkOperations: "[owner_key+operation_id], source_key, candidate_key, &active_intent"
    });

    // Version 17: message-content search token side table (WP-23 / TASK-13523).
    // Additive-only: no existing table's spec changes. Fresh installs pick the
    // declaration up directly; existing databases get a one-time backfill so
    // content search keeps finding messages written before this version.
    // The tokenizer is imported statically: the upgrade runs inside the
    // version-change transaction, which must not await non-Dexie promises
    // (a dynamic import of an unfetched lazy chunk could auto-commit the
    // transaction before it starts).
    this.version(17).stores({
      searchTokens: '&[token+history_id], token'
    }).upgrade(async tx => {
      const rows: Array<{ token: string; history_id: string }> = [];
      const pairs = new Set<string>();
      await tx.table('messages').each(message => {
        for (const row of buildSearchTokenRows(message.history_id, message.content)) {
          const pair = `${row.token}|${row.history_id}`;
          if (pairs.has(pair)) continue;
          pairs.add(pair);
          rows.push(row);
        }
      });
      // Chunked writes bound each request batch on very large databases.
      for (let index = 0; index < rows.length; index += 5000) {
        await tx.table('searchTokens').bulkPut(rows.slice(index, index + 5000));
      }
    });

    // Version 18: metadata/pagination compound indexes (WP-26/WP-27 /
    // TASK-13523). `[history_id+createdAt]` lets history-metadata reads run
    // as index-only counts plus an O(log n) newest-message lookup instead of
    // materializing each history's messages; `[createdAt+id]` gives the
    // sidebar's keyset pagination an exact range cursor (no offset scan, no
    // stateful fast-forward predicate, tie-safe). Additive-only: pure index
    // additions, no data rewrite, empty upgrade.
    this.version(18).stores({
      messages: 'id, history_id, name, role, content, createdAt, messageType, modelName, clusterId, modelId, parent_message_id, [history_id+createdAt]',
      chatHistories: 'id, title, is_rag, message_source, is_pinned, createdAt, doc_id, last_used_prompt, model_id, root_id, parent_conversation_id, server_chat_id, [createdAt+id]'
    });
  }
}

export const db = new PageAssistDexieDB();
