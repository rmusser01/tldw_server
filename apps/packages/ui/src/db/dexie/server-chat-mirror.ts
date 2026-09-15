import type { Message as ChatMessage } from "@/store/option"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error"
import { db } from "./schema"
import { generateID } from "./helpers"
import { runChatPersistenceTransaction } from "./chat-persistence-transaction"
import type { Message } from "./types"

/** Full verified target/owner identity; never stores bearer tokens or API keys. */
export const serverChatMirrorOwnerKey = (snapshot: ServicePromptSnapshot): string => {
  const { config, userId } = snapshot.requestScope
  return JSON.stringify([
    config.serverUrl.trim().replace(/\/+$/, ""), config.authMode,
    config.authSource || "manual", config.orgId ?? null,
    userId == null ? null : String(userId),
    config.expectedSingleUserApiKeyScope ?? null
  ])
}

export const linkServerChatMirror = async ({
  chatId, title, ownerKey, currentHistoryId, legacyHistoryId, signal
}: {
  chatId: string; title: string; ownerKey: string
  currentHistoryId?: string | null; legacyHistoryId?: string | null; signal?: AbortSignal
}): Promise<string> => runChatPersistenceTransaction(signal, async () => {
  const candidates = await db.chatHistories.where("server_chat_id").equals(chatId).toArray()
  const owned = candidates.find(history => history.server_scope_key === ownerKey)
  const current = currentHistoryId ? await db.chatHistories.get(currentHistoryId) : undefined
  const canLinkCurrent = current && (!current.server_chat_id || current.server_chat_id === chatId) &&
    (current.server_scope_key === ownerKey ||
      (!current.server_scope_key && current.id === legacyHistoryId))
  const history = owned || (canLinkCurrent ? current : undefined)
  if (history) {
    await db.chatHistories.update(history.id, { title, server_chat_id: chatId, server_scope_key: ownerKey })
    return history.id
  }
  const id = generateID()
  await db.chatHistories.add({ id, title, createdAt: Date.now(), is_rag: false,
    message_source: "server", server_chat_id: chatId, server_scope_key: ownerKey })
  return id
})

const canonicalId = (message: ChatMessage) => message.serverMessageId?.trim() || null

/** Add a completed owned snapshot without dropping local work or rolling back newer rows. */
export const reconcileServerChatMessages = (
  current: ChatMessage[], incoming: ChatMessage[], beforeAwait?: ChatMessage[]
): ChatMessage[] => {
  const serverIds = new Set(incoming.map(canonicalId).filter(Boolean))
  const key = (message: ChatMessage) => {
    const serverId = canonicalId(message) ||
      (message.id && serverIds.has(String(message.id)) ? String(message.id) : null)
    return serverId ? `server:${serverId}` : message.id ? `local:${message.id}` : null
  }
  const incomingKeys = new Set(incoming.map(key).filter(Boolean))
  const beforeById = new Map((beforeAwait || []).map(message => [key(message), message]))
  const localById = new Map(current.flatMap(message => {
    const id = key(message)
    return id ? [[id, message] as const] : []
  }))
  const merged = incoming.map(remote => {
    const id = key(remote)
    const local = id ? localById.get(id) : undefined
    if (!local) return remote
    const localVersion = local.serverMessageVersion ?? 0
    const remoteVersion = remote.serverMessageVersion ?? 0
    // Unknown/equal revisions cannot prove that a differing local edit is stale.
    const changedDuringAwait = beforeAwait !== undefined && beforeById.get(id)?.message !== local.message
    const preserveContent = (changedDuringAwait || local.serverMessageVersion == null || localVersion >= remoteVersion) && local.message !== remote.message
    return { ...local, ...remote, ...(preserveContent ? local : {}),
      id: local.id || remote.id, serverMessageId: canonicalId(remote) || canonicalId(local) || undefined,
      serverMessageVersion: Math.max(localVersion, remoteVersion) || undefined }
  })
  for (const local of current) {
    const id = key(local)
    if (id && incomingKeys.has(id)) continue
    // A synthetic greeting is superseded only by an acknowledged server greeting.
    if (!canonicalId(local) && (local.messageType === "character:greeting" || local.messageType === "greeting") &&
      incoming.some(message => message.messageType === "character:greeting" || message.messageType === "greeting")) continue
    merged.push(local)
  }
  return merged
}

export const reconcileServerChatMirror = async ({
  historyId, chatId, ownerKey, messages, signal
}: {
  historyId: string; chatId: string; ownerKey: string; messages: ChatMessage[]; signal?: AbortSignal
}): Promise<{ localIds: Map<string, string>; rows: Message[] }> => runChatPersistenceTransaction(signal, async () => {
  const history = await db.chatHistories.get(historyId)
  if (history?.server_chat_id !== chatId || history.server_scope_key !== ownerKey) {
    throw createServicePromptScopeChangedError()
  }
  const rows = await db.messages.where("history_id").equals(historyId).toArray()
  const localIds = new Map<string, string>()
  for (const remote of messages) {
    const serverMessageId = canonicalId(remote)
    if (!serverMessageId) continue
    const local = rows.find(row => row.serverMessageId === serverMessageId ||
      (!row.serverMessageId && row.id === serverMessageId))
    const id = local?.id || `${historyId}:server:${encodeURIComponent(serverMessageId)}`
    const existing = await db.messages.get(id)
    if (existing && existing.history_id !== historyId) throw createServicePromptScopeChangedError()
    const localVersion = local?.serverMessageVersion ?? 0
    const remoteVersion = remote.serverMessageVersion ?? 0
    const preserveContent = local && (local.serverMessageVersion == null || localVersion >= remoteVersion) && local.content !== remote.message
    const next: Message = {
      ...local, id, history_id: historyId, name: remote.name || "Assistant",
      role: remote.role || (remote.isBot ? "assistant" : "user"), content: remote.message,
      images: remote.images || [], sources: remote.sources || [], createdAt: remote.createdAt ?? local?.createdAt ?? Date.now(),
      messageType: remote.messageType, generationInfo: remote.generationInfo,
      metadataExtra: remote.metadataExtra, clusterId: remote.clusterId, modelId: remote.modelId,
      modelName: remote.modelName, modelImage: remote.modelImage, parent_message_id: remote.parentMessageId ?? null,
      ...(preserveContent ? local : {}), serverMessageId,
      serverMessageVersion: Math.max(localVersion, remoteVersion) || undefined
    }
    if (existing) await db.messages.put(next)
    else await db.messages.add(next)
    localIds.set(serverMessageId, id)
  }
  return { localIds, rows: await db.messages.where("history_id").equals(historyId).toArray() }
})
