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
import type { UpdateSpec } from "dexie"
import { db } from './schema';
import { getAllModelNicknames } from "./nickname";
const PAGE_SIZE = 30;

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

  async getHistoriesWithMetadata(historyIds: string[]): Promise<Map<string, {
    messageCount: number;
    lastMessage?: { content: string; role: string; createdAt: number };
  }>> {
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
    await db.chatHistories.add(history);
  }

  async updateChatHistoryCreatedAt(id: string, createdAt: number) {
    await db.chatHistories.update(id, { createdAt });
  }

  async setHistoryServerChatId(historyId: string, serverChatId: string) {
    await db.chatHistories.update(historyId, { server_chat_id: serverChatId });
  }

  async addMessage(message: Message) {
    await db.messages.add(message);
  }

  async updateMessage(history_id: string, message_id: string, content: string) {
    await db.messages.where('id').equals(message_id).modify({ content });
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
    await db.chatHistories.delete(id);
    await db.compareStates.delete(id);
  }

  async removeMessage(history_id: string, message_id: string) {
    await db.messages.delete(message_id);
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
      await db.chatHistories.delete(id);
      await db.messages.where('history_id').equals(id).delete();
      await db.compareStates.delete(id);
    });
  }

  async deleteAllChatHistory() {
    await db.transaction('rw', [db.chatHistories, db.messages, db.compareStates], async () => {
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
  async getChatHistoriesPaginated(page: number = 1, searchQuery?: string): Promise<{
    histories: ChatHistory;
    hasMore: boolean;
    totalCount: number;
  }> {
    const offset = (page - 1) * PAGE_SIZE;

    if (searchQuery) {
      console.log("Searching chat histories with query:", searchQuery);
      const allResults = await this.fullTextSearchChatHistories(searchQuery);
      const paginatedResults = allResults.slice(offset, offset + PAGE_SIZE)
      console.log("Paginated search results:", paginatedResults);
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
        hasMore: histories.length === PAGE_SIZE,
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
  async getAllPrompts(): Promise<Prompts> {
    // Only return non-deleted prompts
    return await db.prompts
      .filter(p => !p.deletedAt)
      .reverse()
      .sortBy('createdAt');
  }

  async getDeletedPrompts(): Promise<Prompts> {
    // Return soft-deleted prompts (trash)
    return await db.prompts
      .filter(p => !!p.deletedAt)
      .reverse()
      .sortBy('deletedAt');
  }

  async addPrompt(prompt: Prompt) {
    const mergedKeywords = prompt.keywords ?? prompt.tags;
    const now = Date.now();
    const normalizedUsageCount =
      typeof prompt.usageCount === "number" && Number.isFinite(prompt.usageCount)
        ? Math.max(0, Math.floor(prompt.usageCount))
        : 0;
    const normalized: Prompt = {
      ...prompt,
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
    // Soft delete: set deletedAt timestamp
    const now = Date.now();
    await db.prompts.update(id, { deletedAt: now, updatedAt: now });
  }

  async permanentlyDeletePrompt(id: string) {
    // Hard delete: remove from database entirely
    await db.prompts.delete(id);
  }

  async restorePromptSnapshot(snapshot: Prompt) {
    if (!(await db.prompts.get(snapshot.id))) {
      throw new Error('prompt_snapshot_target_missing');
    }
    await db.prompts.put(structuredClone(snapshot));
  }

  async restorePrompt(id: string) {
    // Restore from trash: clear deletedAt
    const now = Date.now();
    await db.prompts.update(id, { deletedAt: null, updatedAt: now });
  }

  async emptyTrash(): Promise<number> {
    // Permanently delete all trashed prompts
    const deleted = await db.prompts.filter(p => !!p.deletedAt).toArray();
    const ids = deleted.map(p => p.id);
    await db.prompts.bulkDelete(ids);
    return ids.length;
  }

  async autoCleanupTrash(maxAgeDays: number = 30): Promise<number> {
    // Auto-purge prompts deleted more than maxAgeDays ago
    const cutoff = Date.now() - (maxAgeDays * 24 * 60 * 60 * 1000);
    const expired = await db.prompts
      .filter(p => !!p.deletedAt && p.deletedAt < cutoff)
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
    }
  ) {
    const existing = await db.prompts.get(id);
    if (!existing) return;

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

  async getPromptById(id: string): Promise<Prompt | undefined> {
    return await db.prompts.get(id);
  }

  // ─── Prompt Sync Methods ───

  async getPromptByServerId(serverId: number): Promise<Prompt | undefined> {
    return await db.prompts
      .where('serverId')
      .equals(serverId)
      .first();
  }

  async getPromptsBySyncStatus(status: PromptSyncStatus): Promise<Prompts> {
    return await db.prompts
      .where('syncStatus')
      .equals(status)
      .filter(p => !p.deletedAt)
      .toArray();
  }

  async getPromptsByStudioProject(projectId: number): Promise<Prompts> {
    return await db.prompts
      .where('studioProjectId')
      .equals(projectId)
      .filter(p => !p.deletedAt)
      .toArray();
  }

  async getSyncedPrompts(): Promise<Prompts> {
    return await db.prompts
      .filter(p => !p.deletedAt && !!p.serverId)
      .toArray();
  }

  async getLocalOnlyPrompts(): Promise<Prompts> {
    return await db.prompts
      .filter(p => !p.deletedAt && !p.serverId && p.syncStatus !== 'synced')
      .toArray();
  }

  async updatePromptSyncStatus(id: string, updates: {
    syncStatus?: PromptSyncStatus;
    serverId?: number | null;
    studioProjectId?: number | null;
    studioPromptId?: number | null;
    lastSyncedAt?: number | null;
    serverUpdatedAt?: string | null;
  }) {
    const existing = await db.prompts.get(id);
    if (!existing) return;

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
    await db.userSettings.put({ id: 'main', user_id: id });
  }


  async importChatHistoryV2(data: any[], options: {
    replaceExisting?: boolean;
    mergeData?: boolean;
  } = {}) {
    const { replaceExisting = false, mergeData = true } = options;

    if (!mergeData && !replaceExisting) {
      // Clear existing data
      await this.deleteAllChatHistory();
    }

    // Use transaction for atomic batch operations
    await db.transaction('rw', [db.chatHistories, db.messages], async () => {
      // Collect all histories and messages for bulk operations
      const histories = data.filter(item => item.history).map(item => item.history);
      const allMessages = data.flatMap(item => item.messages || []);

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
  } = {}) {
    const { replaceExisting = false, mergeData = true } = options;

    if (!mergeData && !replaceExisting) {
      await db.prompts.clear();
    }

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
      if (replaceExisting) {
        // Replace mode should fully replace the existing prompt dataset.
        await db.prompts.clear();
        await db.prompts.bulkPut(normalizedPrompts);
        imported = normalizedPrompts.length;
        return;
      }

      // Filter out existing prompts
      const existingIds = new Set(
        (await db.prompts.where('id').anyOf(normalizedPrompts.map(p => p.id)).toArray())
          .map(p => p.id)
      );
      const newPrompts = normalizedPrompts.filter(p => !existingIds.has(p.id));
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
