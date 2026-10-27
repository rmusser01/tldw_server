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
import { getAllModelNicknames } from "./nickname";
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

function fastForward(lastRow: any, idProp: string, otherCriterion?: (item: any) => boolean) {
  let fastForwardComplete = false;
  return (item: any) => {
    if (fastForwardComplete) return otherCriterion ? otherCriterion(item) : true;
    if (item[idProp] === lastRow[idProp]) {
      fastForwardComplete = true;
    }
    return false;
  };
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

    const messageMatches = await db.messages
      .filter(message => searchQueryInContent(message.content, normalizedQuery))
      .toArray();

    const historyIdsFromMessages = [...new Set(messageMatches.map(msg => msg.history_id))];

    const historiesFromMessages = await db.chatHistories
      .where('id')
      .anyOf(historyIdsFromMessages)
      .toArray();

    const allMatches = [...titleMatches, ...historiesFromMessages];
    const uniqueHistories = allMatches.filter((history, index, self) =>
      index === self.findIndex(h => h.id === history.id)
    );

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

  async getHistoryMetadata(historyId: string): Promise<{
    messageCount: number;
    lastMessage?: { content: string; role: string; createdAt: number };
  }> {
    const messages = await db.messages
      .where('history_id')
      .equals(historyId)
      .toArray();

    const messageCount = messages.length;

    if (messageCount === 0) {
      return { messageCount };
    }

    // Find the most recent message
    const lastMessage = messages.reduce((latest, msg) =>
      msg.createdAt > latest.createdAt ? msg : latest
    , messages[0]);

    return {
      messageCount,
      lastMessage: {
        content: lastMessage.content,
        role: lastMessage.role,
        createdAt: lastMessage.createdAt
      }
    };
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

    // Batch fetch all messages for the given history IDs
    const allMessages = await db.messages
      .where('history_id')
      .anyOf(historyIds)
      .toArray();

    // Group messages by history_id
    const messagesByHistory = new Map<string, typeof allMessages>();
    for (const msg of allMessages) {
      const existing = messagesByHistory.get(msg.history_id) || [];
      existing.push(msg);
      messagesByHistory.set(msg.history_id, existing);
    }

    // Compute metadata for each history
    for (const historyId of historyIds) {
      const messages = messagesByHistory.get(historyId) || [];
      const messageCount = messages.length;

      if (messageCount === 0) {
        result.set(historyId, { messageCount });
      } else {
        const lastMessage = messages.reduce((latest, msg) =>
          msg.createdAt > latest.createdAt ? msg : latest
        , messages[0]);

        result.set(historyId, {
          messageCount,
          lastMessage: {
            content: lastMessage.content,
            role: lastMessage.role,
            createdAt: lastMessage.createdAt
          }
        });
      }
    }

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
    if (scope) return db.transaction("r", [db.userSettings, db.chatHistories, db.messages], async () => {
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

  async getChatHistoriesPaginatedOptimized(lastEntry?: any, searchQuery?: string): Promise<{
    histories: ChatHistory;
    hasMore: boolean;
  }> {
    if (searchQuery) {
      const allResults = await this.fullTextSearchChatHistories(searchQuery);
      return {
        histories: allResults.slice(0, PAGE_SIZE),
        hasMore: allResults.length > PAGE_SIZE
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
      };
    } else {
      const histories = await db.chatHistories
        .where('createdAt')
        .belowOrEqual(lastEntry.createdAt)
        .filter(fastForward(lastEntry, "id"))
        .limit(PAGE_SIZE)
        .reverse()
        .toArray();

      return {
        histories,
        hasMore: histories.length === PAGE_SIZE
      };
    }
  }

  // Prompts Methods
  /** Read shared, unlinked legacy data without probing authenticated ownership. */
  async getSharedLocalPrompts(deleted = false): Promise<Prompts> {
    return await db.prompts
      .filter(p => p.ownerId == null && isPromptOwnedBy(p, null) && !!p.deletedAt === deleted)
      .reverse()
      .sortBy(deleted ? 'deletedAt' : 'createdAt');
  }

  async getAllPrompts(expectedOwnerId?: string | null): Promise<Prompts> {
    const ownerId = await resolvePromptOwnerId(expectedOwnerId);
    // Only return non-deleted prompts
    return await db.prompts
      .filter(p => !p.deletedAt && isPromptOwnedBy(p, ownerId))
      .reverse()
      .sortBy('createdAt');
  }

  async getDeletedPrompts(expectedOwnerId?: string | null): Promise<Prompts> {
    const ownerId = await resolvePromptOwnerId(expectedOwnerId);
    // Return soft-deleted prompts (trash)
    return await db.prompts
      .filter(p => !!p.deletedAt && isPromptOwnedBy(p, ownerId))
      .reverse()
      .sortBy('deletedAt');
  }

  /** Copy historical storage verbatim; migration has no authority to claim links. */
  async migrateLegacyPrompt(prompt: Prompt) {
    await db.prompts.add(prompt);
  }

  async countStoredPromptsForMigration(): Promise<number> {
    return (await db.prompts.toArray()).length;
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
    // Permanently delete all trashed prompts
    const ownerId = await resolvePromptOwnerId(expectedOwnerId);
    const deleted = await db.prompts.filter(p => !!p.deletedAt && isPromptOwnedBy(p, ownerId)).toArray();
    const ids = deleted.map(p => p.id);
    await db.prompts.bulkDelete(ids);
    return ids.length;
  }

  async autoCleanupTrash(maxAgeDays: number = 30, expectedOwnerId?: string | null): Promise<number> {
    // Auto-purge prompts deleted more than maxAgeDays ago
    const ownerId = await resolvePromptOwnerId(expectedOwnerId);
    const cutoff = Date.now() - (maxAgeDays * 24 * 60 * 60 * 1000);
    const expired = await db.prompts
      .filter(p => !!p.deletedAt && p.deletedAt < cutoff && isPromptOwnedBy(p, ownerId))
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
    await db.transaction('rw', [db.chatHistories, db.messages, db.compareStates], async () => {
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
        } else {
          // Filter out existing messages
          const existingIds = new Set(
            (await db.messages.where('id').anyOf(allMessages.map(m => m.id)).toArray())
              .map(m => m.id)
          );
          const newMessages = allMessages.filter(m => !existingIds.has(m.id));
          if (newMessages.length > 0) {
            await db.messages.bulkPut(newMessages);
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
