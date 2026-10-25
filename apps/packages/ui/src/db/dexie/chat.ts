import { assertPromptOwnedBy, isPromptOwnedBy, resolvePromptOwnerId } from "@/services/prompt-local-owner"
import type { CapturedPromptOwnerScope } from "@/services/prompt-local-owner"
import {
  assertNoPendingLocalSettingsWrite,
  isReadableOnDeviceHistory,
  type OnDeviceHistoryReadOptions,
  type OnDeviceHistoryScope,
  withLocalMessageMutation,
  type LocalHistoryOwnerV1,
  type HistoryOperationOptions,
  sanitizeImportedHistory,
  sanitizeImportedMessage,
  stripHistoryAuthority
} from "./history-selection"
import {
  ChatHistory,
  HistoryInfo,
  LastUsedModelType,
  Message,
  MessageHistory,
  Prompt,
  Prompts,
  PromptSyncStatus,
  CompareState,
  SessionFiles,
  UploadedFile,
  Webshare,

} from "./types"
import Dexie, { type ObservabilitySet, type UpdateSpec } from "dexie"
import { db } from './schema';
import { getAllModelNicknames, invalidateModelNicknameCache } from "./nickname";
import { indexMessageSearchTokens, indexMessagesSearchTokens, searchTokenHistoryIds } from "./search-tokens";
const PAGE_SIZE = 30;

export { isOnDeviceHistory, isReadableOnDeviceHistory, isOwnedCachedServerHistory, type OnDeviceHistoryScope, type OnDeviceHistoryReadOptions, type VisibleServerChats } from "./history-selection";

async function assertOnDeviceReader(scope: OnDeviceHistoryScope): Promise<void> {
  if (!scope.isCurrent()) throw new Error("on_device_history_scope_changed");
  const settings = await db.userSettings.get("main");
  if (!scope.isCurrent() || (settings?.history_profile_id ?? null) !== scope.profileId) {
    throw new Error("on_device_history_scope_changed");
  }
}

function searchQueryInContent(content: string, query: string): boolean {
  if (!content || !query) {
    return false;
  }

  const normalizedContent = content.toLowerCase();
  const normalizedQuery = query.toLowerCase().trim();

  const wordBoundaryPattern = new RegExp(`\\b${normalizedQuery.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i');

  return wordBoundaryPattern.test(normalizedContent);
}



export class PageAssistDatabase {
  async getCompareState(historyId: string): Promise<CompareState | null> {
    const state = await db.compareStates.get(historyId);
    return state || null;
  }

  async setCompareState(state: CompareState) {
    await db.compareStates.put(state);
  }

  async deleteCompareState(historyId: string) {
    await db.compareStates.delete(historyId);
  }

  async getSessionFiles(sessionId: string): Promise<UploadedFile[]> {
    const sessionFiles = await db.sessionFiles.get(sessionId);
    return sessionFiles?.files || [];
  }

  async getSessionFilesInfo(sessionId: string): Promise<SessionFiles | null> {
    const sessionFiles = await db.sessionFiles.get(sessionId);
    return sessionFiles || null;
  }

  async addFileToSession(sessionId: string, file: UploadedFile) {
    const sessionFiles = await this.getSessionFilesInfo(sessionId);
    const updatedFiles = sessionFiles ? [...sessionFiles.files, file] : [file];
    const sessionData: SessionFiles = {
      sessionId,
      files: updatedFiles,
      retrievalEnabled: sessionFiles?.retrievalEnabled || false,
      createdAt: sessionFiles?.createdAt || Date.now()
    };
    await db.sessionFiles.put(sessionData);
  }

  async removeFileFromSession(sessionId: string, fileId: string) {
    const sessionFiles = await this.getSessionFilesInfo(sessionId);
    if (sessionFiles) {
      const updatedFiles = sessionFiles.files.filter(f => f.id !== fileId);
      const sessionData: SessionFiles = {
        ...sessionFiles,
        files: updatedFiles
      };
      await db.sessionFiles.put(sessionData);
    }
  }

  async updateFileInSession(sessionId: string, fileId: string, updates: Partial<UploadedFile>) {
    const sessionFiles = await this.getSessionFilesInfo(sessionId);
    if (sessionFiles) {
      const updatedFiles = sessionFiles.files.map(f =>
        f.id === fileId ? { ...f, ...updates } : f
      );
      const sessionData: SessionFiles = {
        ...sessionFiles,
        files: updatedFiles
      };
      await db.sessionFiles.put(sessionData);
    }
  }

  async setRetrievalEnabled(sessionId: string, enabled: boolean) {
    const sessionFiles = await this.getSessionFilesInfo(sessionId);
    const sessionData: SessionFiles = {
      sessionId,
      files: sessionFiles?.files || [],
      retrievalEnabled: enabled,
      createdAt: sessionFiles?.createdAt || Date.now()
    };
    await db.sessionFiles.put(sessionData);
  }

  async clearSessionFiles(sessionId: string) {
    await db.sessionFiles.delete(sessionId);
  }

  async getChatHistory(id: string): Promise<MessageHistory> {
    const modelNicknames = await getAllModelNicknames();
    const messages = await db.messages.where('history_id').equals(id).toArray();

    return messages.map((message) => {
      return {
        ...message,
        modelName: modelNicknames[message.name]?.model_name || message.name,
        modelImage: modelNicknames[message.name]?.model_avatar || undefined
      };
    });
  }

  async getChatHistories(): Promise<ChatHistory> {
    return await db.chatHistories.orderBy('createdAt').reverse().toArray();
  }

  /** The `limit` most recently active histories that pass `filter`, newest first. */
  async getRecentChatHistories(
    limit: number,
    filter: (history: HistoryInfo) => boolean = () => true
  ): Promise<ChatHistory> {
    return await db.chatHistories
      .orderBy('createdAt')
      .reverse()
      .filter(filter)
      .limit(limit)
      .toArray();
  }

  async fullTextSearchChatHistories(query: string): Promise<ChatHistory> {
    const normalizedQuery = query.toLowerCase().trim();
    if (!normalizedQuery) {
      return this.getChatHistories();
    }

    const titleMatches = await db.chatHistories
      .where('title')
      .startsWithIgnoreCase(normalizedQuery)
      .or('title')
      .anyOfIgnoreCase(normalizedQuery.split(' '))
      .toArray();

    // Content search (WP-23): the searchTokens index narrows to candidate
    // histories, then only candidate message rows are streamed through the
    // exact word-boundary check. The messages table is never materialized.
    const candidateHistoryIds = await searchTokenHistoryIds(normalizedQuery);
    const contentMatchedIds = new Set<string>();
    if (candidateHistoryIds === null) {
      // Punctuation-only query: no word tokens to prefilter with. Keep the
      // exact semantics via a streaming pass that never materializes the
      // table (rare path).
      await db.messages.each(message => {
        if (searchQueryInContent(message.content, normalizedQuery)) {
          contentMatchedIds.add(message.history_id);
        }
      });
    } else if (candidateHistoryIds.size > 0) {
      await db.messages
        .where('history_id')
        .anyOf([...candidateHistoryIds])
        .filter(message => searchQueryInContent(message.content, normalizedQuery))
        .each(message => contentMatchedIds.add(message.history_id));
    }

    const historiesFromMessages = contentMatchedIds.size === 0 ? [] : await db.chatHistories
      .where('id')
      .anyOf([...contentMatchedIds])
      .toArray();

    // Set dedupe: title matches win over content matches for the same id.
    const seenIds = new Set<string>();
    const uniqueHistories: ChatHistory = [];
    for (const history of [...titleMatches, ...historiesFromMessages]) {
      if (seenIds.has(history.id)) continue;
      seenIds.add(history.id);
      uniqueHistories.push(history);
    }

    return uniqueHistories.sort((a, b) => b.createdAt - a.createdAt);
  }

  async getChatHistoryTitleById(id: string): Promise<string> {
    const chatHistory = await db.chatHistories.get(id);
    return chatHistory?.title || '';
  }

  async getHistoryInfo(id: string): Promise<HistoryInfo> {
    return db.chatHistories.get(id);
  }

  async getHistoryByServerChatId(serverChatId: string): Promise<HistoryInfo | null> {
    if (!serverChatId) return null;
    const history = await db.chatHistories
      .where('server_chat_id')
      .equals(serverChatId)
      .first();
    return history || null;
  }

  async getHistoryByDocId(docId: string): Promise<HistoryInfo | null> {
    if (!docId) return null;
    // Get the most recent history for this document
    const history = await db.chatHistories
      .where('doc_id')
      .equals(docId)
      .reverse()
      .sortBy('createdAt');
    return history[0] || null;
  }

  async getAllHistoriesByDocId(docId: string): Promise<HistoryInfo[]> {
    if (!docId) return [];
    return await db.chatHistories
      .where('doc_id')
      .equals(docId)
      .reverse()
      .sortBy('createdAt');
  }

  /**
   * The newest message of a history via the `[history_id+createdAt]` compound
   * index (WP-26): two cursor seeks instead of materializing the history's
   * messages. The newest row is the index maximum; a tie on `createdAt`
   * resolves to the FIRST row in primary-key order, exactly matching the
   * previous reduce() over the PK-ordered `history_id` scan.
   */
  private async newestMessageByHistoryId(historyId: string): Promise<Message | undefined> {
    const newest = await db.messages
      .where('[history_id+createdAt]')
      .between([historyId, -Infinity], [historyId, Infinity])
      .last();
    if (!newest) return undefined;
    const firstOfTie = await db.messages
      .where('[history_id+createdAt]')
      .equals([historyId, newest.createdAt])
      .first();
    return firstOfTie ?? newest;
  }

  private async historyMetadataEntry(historyId: string): Promise<{
    messageCount: number;
    lastMessage?: { content: string; role: string; createdAt: number };
  }> {
    // Index-only count: no message values are materialized for the count.
    const messageCount = await db.messages.where('history_id').equals(historyId).count();
    if (messageCount === 0) return { messageCount };
    const lastMessage = await this.newestMessageByHistoryId(historyId);
    if (!lastMessage) return { messageCount };
    return {
      messageCount,
      lastMessage: {
        content: lastMessage.content,
        role: lastMessage.role,
        createdAt: lastMessage.createdAt
      }
    };
  }

  async getHistoryMetadata(historyId: string): Promise<{
    messageCount: number;
    lastMessage?: { content: string; role: string; createdAt: number };
  }> {
    return this.historyMetadataEntry(historyId);
  }

  async getHistoriesWithMetadata(historyIds: string[], scope?: OnDeviceHistoryScope, options?: OnDeviceHistoryReadOptions): Promise<Map<string, {
    messageCount: number;
    lastMessage?: { content: string; role: string; createdAt: number };
  }>> {
    if (scope) return db.transaction("r", [db.userSettings, db.chatHistories, db.messages], async () => {
      await assertOnDeviceReader(scope);
      const rows = await Promise.all(historyIds.map(id => db.chatHistories.get(id)));
      const eligible = rows.filter((row): row is HistoryInfo => !!row && isReadableOnDeviceHistory(row, scope, options));
      const metadata = await this.getHistoriesWithMetadata(eligible.map(row => row.id));
      await assertOnDeviceReader(scope);
      return metadata;
    });
    const result = new Map();
    if (historyIds.length === 0) return result;

    // Per-history index counts plus O(log n) newest-message lookups (WP-26):
    // opening the sidebar on hundreds of histories never loads the messages
    // table — only index entries and at most one message row per history.
    // One read transaction batches the per-history queries.
    await db.transaction('r', db.messages, async () => {
      for (const historyId of historyIds) {
        result.set(historyId, await this.historyMetadataEntry(historyId));
      }
    });

    return result;
  }

  async addChatHistory(history: HistoryInfo) {
    await db.chatHistories.add(sanitizeImportedHistory(history));
  }

  async updateChatHistoryCreatedAt(id: string, createdAt: number) {
    await db.chatHistories.update(id, { createdAt });
  }

  async setHistoryServerChatId(historyId: string, serverChatId: string) {
    await db.chatHistories.update(historyId, { server_chat_id: serverChatId });
  }

  async addMessage(message: Message) {
    await db.messages.add(stripHistoryAuthority(message));
    // Keep the content-search token index fresh. Best-effort: the message is
    // already committed, so an index failure degrades only that message's
    // searchability rather than failing the save.
    try {
      await indexMessageSearchTokens(message);
    } catch (error) {
      console.warn("[search-tokens] failed to index a saved message:", error);
    }
  }

  async updateMessage(
    history_id: string,
    message_id: string,
    content: string,
    owner?: LocalHistoryOwnerV1
  ) {
    await withLocalMessageMutation(history_id, owner, async () => {
      const row = await db.messages.get(message_id);
      if (!row) throw new Error("missing_message");
      if (row.history_id !== history_id)
        throw new Error("message_owner_mismatch");
      await db.messages.update(message_id, { content });
      // Re-index in the same transaction: stale tokens from the old content
      // are harmless (exact verification rejects them), the new content must
      // be findable.
      await indexMessageSearchTokens({ history_id, content });
    });
  }

  async updateMessageMedia(
    message_id: string,
    updates: { images?: string[]; generationInfo?: any }
  ) {
    const patch: UpdateSpec<Message> = {}
    if (Array.isArray(updates.images)) {
      patch.images = updates.images
    }
    if (updates.generationInfo !== undefined) {
      patch.generationInfo = updates.generationInfo
    }
    if (Object.keys(patch).length === 0) return
    await db.messages.where("id").equals(message_id).modify(patch)
  }

  async updateMessageDiscoSkillComment(
    message_id: string,
    discoSkillComment: Message["discoSkillComment"] | null
  ) {
    await db.messages
      .where('id')
      .equals(message_id)
      .modify({ discoSkillComment: discoSkillComment ?? null });
  }

  async removeChatHistory(id: string) {
    await db.transaction("rw", [db.chatHistories, db.compareStates], async () => {
      assertNoPendingLocalSettingsWrite(await db.chatHistories.get(id));
      await db.chatHistories.delete(id);
      await db.compareStates.delete(id);
    });
  }

  async removeMessage(
    history_id: string,
    message_id: string,
    owner?: LocalHistoryOwnerV1,
    options?: HistoryOperationOptions
  ) {
    return withLocalMessageMutation(
      history_id,
      owner,
      async () => {
        const row = await db.messages.get(message_id)
        if (!row) throw new Error("missing_message")
        if (row.history_id !== history_id)
          throw new Error("message_owner_mismatch")
        const rows = await db.messages
          .where("history_id")
          .equals(history_id)
          .toArray()
        if (rows.some((child) => child.parent_message_id === message_id))
          throw new Error("message_has_descendants")
        await db.messages.delete(message_id)
        return row
      },
      options
    )
  }

  async updateLastUsedModel(history_id: string, model_id: string) {
    await db.chatHistories.update(history_id, { model_id });
  }

  async updateLastUsedPrompt(history_id: string, usedPrompt: LastUsedModelType) {
    await db.chatHistories.update(history_id, { last_used_prompt: usedPrompt });
  }

  async clear() {
    await db.delete();
    await db.open();
    // db.delete() wipes every table, including nicknames.
    invalidateModelNicknameCache();
  }

  async deleteChatHistory(id: string) {
    await db.transaction('rw', [db.chatHistories, db.messages, db.compareStates], async () => {
      assertNoPendingLocalSettingsWrite(await db.chatHistories.get(id));
      await db.chatHistories.delete(id);
      await db.messages.where('history_id').equals(id).delete();
      await db.compareStates.delete(id);
    });
  }

  async deleteAllChatHistory() {
    await db.transaction('rw', [db.chatHistories, db.messages, db.compareStates], async () => {
      (await db.chatHistories.toArray()).forEach(assertNoPendingLocalSettingsWrite);
      await db.chatHistories.clear();
      await db.messages.clear();
      await db.compareStates.clear();
    });
  }

  async clearDB() {
    await db.delete();
    await db.open();
    // db.delete() wipes every table, including nicknames.
    invalidateModelNicknameCache();
  }

  async deleteMessage(history_id: string) {
    await db.messages.where('history_id').equals(history_id).delete();
  }
  async getChatHistoriesPaginated(page: number = 1, searchQuery?: string, scope?: OnDeviceHistoryScope, options?: OnDeviceHistoryReadOptions): Promise<{
    histories: ChatHistory;
    hasMore: boolean;
    totalCount: number;
  }> {
    const offset = (page - 1) * PAGE_SIZE;
    const query = searchQuery?.trim();
    if (scope) return db.transaction("r", [db.userSettings, db.chatHistories, db.messages, db.searchTokens], async () => {
      await assertOnDeviceReader(scope);
      const candidates = query ? await this.fullTextSearchChatHistories(query) : await this.getChatHistories();
      const eligible = candidates.filter(history => isReadableOnDeviceHistory(history, scope, options, query));
      await assertOnDeviceReader(scope);
      return {
        histories: eligible.slice(offset, offset + PAGE_SIZE),
        hasMore: offset + PAGE_SIZE < eligible.length,
        totalCount: eligible.length
      };
    });

    if (query) {
      const allResults = await this.fullTextSearchChatHistories(query);
      const paginatedResults = allResults.slice(offset, offset + PAGE_SIZE);
      return {
        histories: paginatedResults,
        hasMore: offset + PAGE_SIZE < allResults.length,
        totalCount: allResults.length
      };
    }

    if (page === 1) {
      const histories = await db.chatHistories
        .orderBy('createdAt')
        .reverse()
        .limit(PAGE_SIZE)
        .toArray();

      const totalCount = await db.chatHistories.count();

      return {
        histories,
        hasMore: offset + PAGE_SIZE < totalCount,
        totalCount
      };
    } else {
      const skipCount = offset;
      const histories = await db.chatHistories
        .orderBy('createdAt')
        .reverse()
        .offset(skipCount)
        .limit(PAGE_SIZE)
        .toArray();

      const totalCount = await db.chatHistories.count();

      return {
        histories,
        hasMore: offset + PAGE_SIZE < totalCount,
        totalCount
      };
    }
  }
  /** Stored rows for a labelled read-only preview; no H1 path or bookmark is adopted. */
  async getOnDeviceChatPreview(id: string, scope: OnDeviceHistoryScope, options?: OnDeviceHistoryReadOptions): Promise<{ historyInfo: HistoryInfo; messages: Message[] } | null> {
    return db.transaction("r", [db.userSettings, db.chatHistories, db.messages], async () => {
      await assertOnDeviceReader(scope);
      const historyInfo = await db.chatHistories.get(id);
      if (!historyInfo || !isReadableOnDeviceHistory(historyInfo, scope, options)) return null;
      const messages = await db.messages.where("history_id").equals(id).toArray();
      await assertOnDeviceReader(scope);
      return { historyInfo, messages: messages.sort((a, b) => a.createdAt - b.createdAt || a.id.localeCompare(b.id)) };
    });
  }

  /**
   * Keyset pagination over the `[createdAt+id]` compound index (WP-27): each
   * page is an exact range scan strictly below the previous page's last row,
   * so pages never duplicate or skip — including on createdAt ties — without
   * an offset scan or a stateful fast-forward predicate.
   */
  async getChatHistoriesPaginatedOptimized(lastEntry?: Pick<HistoryInfo, 'createdAt' | 'id'>, searchQuery?: string): Promise<{
    histories: ChatHistory;
    hasMore: boolean;
  }> {
    if (searchQuery) {
      // Search results are computed and sorted in memory; continue after the
      // previous page's last row by id. A cursor that no longer matches any
      // result (row deleted between pages) ends the walk cleanly instead of
      // re-serving page 1.
      const allResults = await this.fullTextSearchChatHistories(searchQuery);
      if (!lastEntry) {
        return {
          histories: allResults.slice(0, PAGE_SIZE),
          hasMore: allResults.length > PAGE_SIZE
        };
      }
      const cursor = allResults.findIndex(row => row.id === lastEntry.id);
      if (cursor === -1) return { histories: [], hasMore: false };
      const histories = allResults.slice(cursor + 1, cursor + 1 + PAGE_SIZE);
      return {
        histories,
        hasMore: cursor + 1 + PAGE_SIZE < allResults.length
      };
    }

    if (!lastEntry) {
      const histories = await db.chatHistories
        .orderBy('createdAt')
        .reverse()
        .limit(PAGE_SIZE)
        .toArray();
      return {
        histories,
        hasMore: histories.length === PAGE_SIZE
          ? await this.hasMoreHistoriesBefore(histories[histories.length - 1])
          : false
      };
    }

    const histories = await db.chatHistories
      .where('[createdAt+id]')
      .below([lastEntry.createdAt, lastEntry.id])
      .reverse()
      .limit(PAGE_SIZE)
      .toArray();
    return {
      histories,
      hasMore: histories.length === PAGE_SIZE
        ? await this.hasMoreHistoriesBefore(histories[histories.length - 1])
        : false
    };
  }

  /** One-row probe below the page cursor: exact hasMore without an offset or full count. */
  private async hasMoreHistoriesBefore(lastRow: HistoryInfo | undefined): Promise<boolean> {
    if (!lastRow) return false;
    const next = await db.chatHistories
      .where('[createdAt+id]')
      .below([lastRow.createdAt, lastRow.id])
      .limit(1)
      .first();
    return !!next;
  }

  // Prompts Methods
  /** Read shared, unlinked legacy data without probing authenticated ownership. */
  async getSharedLocalPrompts(deleted = false): Promise<Prompts> {
    // Only trashed prompts (numeric deletedAt, always a positive Date.now()
    // timestamp) have entries in the `deletedAt` index — IndexedDB never
    // indexes null/undefined — so only the deleted side can be an index
    // range. The non-deleted side keeps its JS filter: rows with
    // deletedAt:null or with the field absent (pre-v9 legacy rows) have no
    // index entries to query.
    if (!deleted) {
      return await db.prompts
        .filter(p => p.ownerId == null && isPromptOwnedBy(p, null) && !p.deletedAt)
        .reverse()
        .sortBy('createdAt');
    }
    return await db.prompts
      .where('deletedAt')
      .above(0)
      .filter(p => p.ownerId == null && isPromptOwnedBy(p, null))
      .reverse()
      .sortBy('deletedAt');
  }

  async getAllPrompts(expectedOwnerId?: string | null): Promise<Prompts> {
    const ownerId = await resolvePromptOwnerId(expectedOwnerId);
    // Only return non-deleted prompts. Non-deleted rows are exactly the ones
    // WITHOUT an index entry (see getSharedLocalPrompts), so the JS filter
    // over the table is unavoidable here — but its cost is proportional to
    // the returned set.
    return await db.prompts
      .filter(p => !p.deletedAt && isPromptOwnedBy(p, ownerId))
      .reverse()
      .sortBy('createdAt');
  }

  async getDeletedPrompts(expectedOwnerId?: string | null): Promise<Prompts> {
    const ownerId = await resolvePromptOwnerId(expectedOwnerId);
    // Return soft-deleted prompts (trash): the numeric-deletedAt range of
    // the index is exactly the trashed set.
    return await db.prompts
      .where('deletedAt')
      .above(0)
      .filter(p => isPromptOwnedBy(p, ownerId))
      .reverse()
      .sortBy('deletedAt');
  }

  /** Copy historical storage verbatim; migration has no authority to claim links. */
  async migrateLegacyPrompt(prompt: Prompt) {
    await db.prompts.add(prompt);
  }

  async countStoredPromptsForMigration(): Promise<number> {
    return db.prompts.count();
  }

  async addPrompt(prompt: Prompt, expectedOwnerId?: string | null) {
    const ownerId = await resolvePromptOwnerId(expectedOwnerId);
    if (prompt.ownerId && prompt.ownerId !== ownerId) throw new Error("Prompt account changed");
    const mergedKeywords = prompt.keywords ?? prompt.tags;
    const now = Date.now();
    const normalizedUsageCount =
      typeof prompt.usageCount === "number" && Number.isFinite(prompt.usageCount)
        ? Math.max(0, Math.floor(prompt.usageCount))
        : 0;
    const normalized: Prompt = {
      ...prompt,
      ownerId,
      title: prompt.title || prompt.name,
      name: prompt.name ?? prompt.title,
      tags: mergedKeywords ?? prompt.tags,
      keywords: mergedKeywords ?? prompt.keywords ?? prompt.tags,
      promptFormat: prompt.promptFormat ?? 'legacy',
      promptSchemaVersion: prompt.promptSchemaVersion ?? null,
      structuredPromptDefinition: prompt.structuredPromptDefinition ?? null,
      syncPayloadVersion:
        typeof prompt.syncPayloadVersion === "number" && Number.isFinite(prompt.syncPayloadVersion)
          ? prompt.syncPayloadVersion
          : 1,
      deletedAt: null,
      usageCount: normalizedUsageCount,
      lastUsedAt:
        typeof prompt.lastUsedAt === "number" && Number.isFinite(prompt.lastUsedAt)
          ? prompt.lastUsedAt
          : null,
      updatedAt: now
    };
    await db.prompts.add(normalized);
  }

  async deletePrompt(id: string) {
    assertPromptOwnedBy(await db.prompts.get(id), await resolvePromptOwnerId());
    // Soft delete: set deletedAt timestamp
    const now = Date.now();
    await db.prompts.update(id, { deletedAt: now, updatedAt: now });
  }

  async permanentlyDeletePrompt(id: string, expectedOwnerId?: string | null) {
    assertPromptOwnedBy(await db.prompts.get(id), await resolvePromptOwnerId(expectedOwnerId));
    // Hard delete: remove from database entirely
    await db.prompts.delete(id);
  }

  async restorePromptSnapshot(snapshot: Prompt, expectedOwnerId?: string | null) {
    const ownerId = await resolvePromptOwnerId(expectedOwnerId);
    const existing = await db.prompts.get(snapshot.id);
    assertPromptOwnedBy(existing, ownerId);
    if (snapshot.ownerId !== existing.ownerId) throw new Error("Prompt owner cannot change");
    if (!(await db.prompts.get(snapshot.id))) {
      throw new Error('prompt_snapshot_target_missing');
    }
    await db.prompts.put(structuredClone(snapshot));
  }

  async restorePrompt(id: string) {
    assertPromptOwnedBy(await db.prompts.get(id), await resolvePromptOwnerId());
    // Restore from trash: clear deletedAt
    const now = Date.now();
    await db.prompts.update(id, { deletedAt: null, updatedAt: now });
  }

  async emptyTrash(expectedOwnerId?: string | null): Promise<number> {
    // Permanently delete all trashed prompts: the numeric-deletedAt index
    // range selects the trash directly instead of JS-scanning every prompt.
    const ownerId = await resolvePromptOwnerId(expectedOwnerId);
    const deleted = await db.prompts
      .where('deletedAt')
      .above(0)
      .filter(p => isPromptOwnedBy(p, ownerId))
      .toArray();
    const ids = deleted.map(p => p.id);
    await db.prompts.bulkDelete(ids);
    return ids.length;
  }

  async autoCleanupTrash(maxAgeDays: number = 30, expectedOwnerId?: string | null): Promise<number> {
    // Auto-purge prompts deleted more than maxAgeDays ago (0 < deletedAt < cutoff).
    const ownerId = await resolvePromptOwnerId(expectedOwnerId);
    const cutoff = Date.now() - (maxAgeDays * 24 * 60 * 60 * 1000);
    const expired = await db.prompts
      .where('deletedAt')
      .between(0, cutoff, false, false)
      .filter(p => isPromptOwnedBy(p, ownerId))
      .toArray();
    const ids = expired.map(p => p.id);
    await db.prompts.bulkDelete(ids);
    return ids.length;
  }

  async updatePrompt(
    id: string,
    updates: Partial<Prompt> & {
      title?: string
      name?: string
      content?: string
      is_system?: boolean
      tags?: string[]
      keywords?: string[]
      favorite?: boolean
      usageCount?: number
      lastUsedAt?: number | null
    },
    expectedOwnerId?: string | null
  ) {
    const existing = await db.prompts.get(id);
    if (!existing) return;
    assertPromptOwnedBy(existing, await resolvePromptOwnerId(expectedOwnerId));

    const mergedKeywords = updates.keywords ?? updates.tags;
    const resolvedUsageCount =
      typeof updates.usageCount === "number" && Number.isFinite(updates.usageCount)
        ? Math.max(0, Math.floor(updates.usageCount))
        : existing.usageCount;
    const resolvedLastUsedAt =
      updates.lastUsedAt === undefined
        ? existing.lastUsedAt
        : typeof updates.lastUsedAt === "number" && Number.isFinite(updates.lastUsedAt)
          ? updates.lastUsedAt
          : null;
    const merged: Prompt = {
      ...existing,
      ...updates,
      ownerId: existing.ownerId,
      title: updates.name ?? updates.title ?? existing.title,
      name: updates.name ?? existing.name ?? existing.title,
      content: updates.content ?? existing.content,
      system_prompt: updates.system_prompt ?? existing.system_prompt,
      user_prompt: updates.user_prompt ?? existing.user_prompt,
      promptFormat: updates.promptFormat ?? existing.promptFormat ?? 'legacy',
      promptSchemaVersion:
        updates.promptSchemaVersion === undefined
          ? existing.promptSchemaVersion ?? null
          : updates.promptSchemaVersion ?? null,
      structuredPromptDefinition:
        updates.structuredPromptDefinition === undefined
          ? existing.structuredPromptDefinition ?? null
          : updates.structuredPromptDefinition ?? null,
      syncPayloadVersion:
        typeof updates.syncPayloadVersion === "number" && Number.isFinite(updates.syncPayloadVersion)
          ? updates.syncPayloadVersion
          : typeof existing.syncPayloadVersion === "number" && Number.isFinite(existing.syncPayloadVersion)
            ? existing.syncPayloadVersion
            : 1,
      tags: mergedKeywords ?? existing.tags,
      keywords: mergedKeywords ?? existing.keywords ?? existing.tags,
      favorite:
        typeof updates.favorite === "boolean" ? updates.favorite : existing.favorite,
      usageCount: resolvedUsageCount,
      lastUsedAt: resolvedLastUsedAt
    };

    await db.prompts.put(merged);
  }

  async incrementPromptUsage(id: string) {
    assertPromptOwnedBy(await db.prompts.get(id), await resolvePromptOwnerId());
    const existing = await db.prompts.get(id);
    if (!existing || existing.deletedAt) return null;

    const now = Date.now();
    const currentUsageCount =
      typeof existing.usageCount === "number" && Number.isFinite(existing.usageCount)
        ? Math.max(0, Math.floor(existing.usageCount))
        : 0;
    const usageCount = currentUsageCount + 1;
    const patch: Pick<Prompt, "usageCount" | "lastUsedAt" | "updatedAt"> = {
      usageCount,
      lastUsedAt: now,
      updatedAt: now
    };
    await db.prompts.update(id, patch);
    return {
      usageCount,
      lastUsedAt: now
    };
  }

  async getPromptById(id: string, expectedOwnerId?: string | null): Promise<Prompt | undefined> {
    const prompt = await db.prompts.get(id);
    const ownerId = await resolvePromptOwnerId(expectedOwnerId);
    return prompt && isPromptOwnedBy(prompt, ownerId) ? prompt : undefined;
  }

  // ─── Prompt Sync Methods ───

  async getPromptByServerId(serverId: number): Promise<Prompt | undefined> {
    return (await this.getAllPrompts()).find(p => p.serverId === serverId);
  }

  async getPromptsBySyncStatus(status: PromptSyncStatus): Promise<Prompts> {
    return (await this.getAllPrompts()).filter(p => p.syncStatus === status);
  }

  async getPromptsByStudioProject(projectId: number): Promise<Prompts> {
    return (await this.getAllPrompts()).filter(p => p.studioProjectId === projectId);
  }

  async getSyncedPrompts(): Promise<Prompts> {
    return (await this.getAllPrompts()).filter(p => !!p.serverId);
  }

  async getLocalOnlyPrompts(): Promise<Prompts> {
    return (await this.getAllPrompts()).filter(p => !p.serverId && p.syncStatus !== 'synced');
  }

  async updatePromptSyncStatus(id: string, updates: {
    syncStatus?: PromptSyncStatus;
    serverId?: number | null;
    studioProjectId?: number | null;
    studioPromptId?: number | null;
    lastSyncedAt?: number | null;
    serverUpdatedAt?: string | null;
  }, expectedOwnerId?: string | null) {
    const existing = await db.prompts.get(id);
    if (!existing) return;
    assertPromptOwnedBy(existing, await resolvePromptOwnerId(expectedOwnerId));

    await db.prompts.update(id, {
      ...updates,
      updatedAt: Date.now()
    });
  }

  // Webshare Methods
  async getWebshare(id: string) {
    return await db.webshares.get(id);
  }

  async getAllWebshares(): Promise<Webshare[]> {
    return await db.webshares.orderBy('createdAt').reverse().toArray();
  }

  async addWebshare(webshare: Webshare) {
    await db.webshares.add(webshare);
  }

  async deleteWebshare(id: string) {
    await db.webshares.delete(id);
  }

  // User Settings Methods
  async getUserID(): Promise<string> {
    const userSettings = await db.userSettings.get('main');
    return userSettings?.user_id || '';
  }

  async setUserID(id: string) {
    await db.transaction('rw', [db.userSettings], async () => {
      const previous = await db.userSettings.get('main');
      await db.userSettings.put({ ...previous, id: 'main', user_id: id });
    });
  }


  async importChatHistoryV2(data: any[], options: {
    replaceExisting?: boolean;
    mergeData?: boolean;
  } = {}) {
    const { replaceExisting = false, mergeData = true } = options;

    // Preserve this destination's guard, never imported authority, under the replacement lock.
    await db.transaction('rw', [db.chatHistories, db.messages, db.compareStates, db.searchTokens], async () => {
      const histories = data.filter(item => item.history).map(item => sanitizeImportedHistory(item.history));
      for (const history of histories) {
        const destination = await db.chatHistories.get(history.id);
        if (destination?.local_settings_guard)
          history.local_settings_guard = destination.local_settings_guard;
        if (destination && (destination.local_owner_key != null || destination.server_scope_key != null || destination.server_chat_id != null || destination.message_source === "server")) {
          // Restoring public data cannot clear or redirect an existing trusted owner.
          Object.assign(history, {
            local_owner_key: destination.local_owner_key,
            server_scope_key: destination.server_scope_key,
            server_chat_id: destination.server_chat_id,
            message_source: destination.message_source
          });
        }
      }
      if (!mergeData && !replaceExisting) {
        (await db.chatHistories.toArray()).forEach(assertNoPendingLocalSettingsWrite);
        await db.chatHistories.clear();
        await db.messages.clear();
        await db.compareStates.clear();
      }
      const allMessages = data.flatMap(item => (item.messages || []).map(sanitizeImportedMessage));

      // Bulk put histories
      if (histories.length > 0) {
        await db.chatHistories.bulkPut(histories);
      }

      // For messages, check existing if not replacing
      if (allMessages.length > 0) {
        if (replaceExisting) {
          await db.messages.bulkPut(allMessages);
          await indexMessagesSearchTokens(allMessages);
        } else {
          // Filter out existing messages
          const existingIds = new Set(
            (await db.messages.where('id').anyOf(allMessages.map(m => m.id)).toArray())
              .map(m => m.id)
          );
          const newMessages = allMessages.filter(m => !existingIds.has(m.id));
          if (newMessages.length > 0) {
            await db.messages.bulkPut(newMessages);
            await indexMessagesSearchTokens(newMessages);
          }
        }
      }
    });
  }

  async importPromptsV2(data: Prompt[], options: {
    replaceExisting?: boolean;
    mergeData?: boolean;
    expectedOwnerId?: string | null;
    ownerScope?: CapturedPromptOwnerScope;
  } = {}) {
    const { replaceExisting = false, mergeData = true } = options;
    options.ownerScope?.assertCurrent();
    const ownerId = options.ownerScope ? options.ownerScope.ownerId : await resolvePromptOwnerId(options.expectedOwnerId);

    // Normalize all prompts first
    const normalizedPrompts = data.map(prompt => {
      const mergedKeywords = prompt.keywords ?? prompt.tags;
      const usageCount =
        typeof prompt.usageCount === "number" && Number.isFinite(prompt.usageCount)
          ? Math.max(0, Math.floor(prompt.usageCount))
          : 0;
      const lastUsedAt =
        typeof prompt.lastUsedAt === "number" && Number.isFinite(prompt.lastUsedAt)
          ? prompt.lastUsedAt
          : null;
      return {
        ...prompt,
        // Imports are content copies, never authority to another account's link.
        ownerId,
        serverId: null,
        studioProjectId: null,
        studioPromptId: null,
        serverLibraryId: null,
        serverLibraryUuid: null,
        syncStatus: 'local',
        sourceSystem: 'workspace',
        lastSyncedAt: null,
        serverUpdatedAt: null,
        title: prompt.title || prompt.name,
        name: prompt.name ?? prompt.title,
        tags: mergedKeywords ?? prompt.tags,
        keywords: mergedKeywords ?? prompt.keywords ?? prompt.tags,
        usageCount,
        lastUsedAt
      } as Prompt;
    });

    let imported = 0;
    let skipped = 0;
    let failed = 0;

    // Use transaction for atomic batch operations
    await db.transaction('rw', db.prompts, async () => {
      const foreign = await db.prompts.filter(p => !isPromptOwnedBy(p, ownerId)).toArray();
      options.ownerScope?.assertCurrent();
      const foreignIds = new Set(foreign.map(p => p.id));
      const allowed = normalizedPrompts.filter(p => !foreignIds.has(p.id));
      if (replaceExisting) {
        const visible = await db.prompts.filter(p => isPromptOwnedBy(p, ownerId)).toArray();
        options.ownerScope?.assertCurrent();
        await db.prompts.bulkDelete(visible.map(p => p.id));
        options.ownerScope?.assertCurrent();
        await db.prompts.bulkPut(allowed);
        imported = allowed.length;
        skipped = normalizedPrompts.length - allowed.length;
        return;
      }

      if (!mergeData) {
        const visible = await db.prompts.filter(p => isPromptOwnedBy(p, ownerId)).toArray();
        options.ownerScope?.assertCurrent();
        await db.prompts.bulkDelete(visible.map(p => p.id));
      }

      // Filter out existing prompts
      const existingIds = new Set(
        (await db.prompts.where('id').anyOf(normalizedPrompts.map(p => p.id)).toArray())
          .map(p => p.id)
      );
      const newPrompts = allowed.filter(p => !existingIds.has(p.id));
      options.ownerScope?.assertCurrent();
      skipped = normalizedPrompts.length - newPrompts.length;
      if (newPrompts.length > 0) {
        await db.prompts.bulkPut(newPrompts);
      }
      imported = newPrompts.length;
    });

    return { imported, skipped, failed };
  }

  async importSessionFilesV2(data: SessionFiles[], options: {
    replaceExisting?: boolean;
    mergeData?: boolean;
  } = {}) {
    const { replaceExisting = false, mergeData = true } = options;

    if (!mergeData && !replaceExisting) {
      await db.sessionFiles.clear();
    }

    for (const sessionFile of data) {
      const existingSessionFile = await db.sessionFiles.get(sessionFile.sessionId);

      if (existingSessionFile && !replaceExisting) {
        if (mergeData) {
          // Merge files arrays
          const mergedFiles = [...existingSessionFile.files];
          for (const newFile of sessionFile.files) {
            if (!mergedFiles.find(f => f.id === newFile.id)) {
              mergedFiles.push(newFile);
            }
          }
          await db.sessionFiles.put({
            ...existingSessionFile,
            files: mergedFiles
          });
        }
        continue;
      }

      await db.sessionFiles.put(sessionFile);
    }
  }

}

export const pageAssistDatabase = new PageAssistDatabase()

/** Invalidate existing history queries without scanning stored message bodies. */
export function watchOnDeviceHistoryChanges(changed: (profileChanged: boolean) => void): () => void {
  const listener = (parts: ObservabilitySet) => {
    const keys = Object.keys(parts)
    const touches = (table: string) => keys.some(key => key === "all" || key.startsWith(`idb://${db.name}/${table}/`))
    const profileChanged = touches("userSettings")
    if (profileChanged || touches("chatHistories") || touches("messages")) changed(profileChanged)
  }
  Dexie.on("storagemutated", listener)
  return () => Dexie.on.storagemutated.unsubscribe(listener)
}
