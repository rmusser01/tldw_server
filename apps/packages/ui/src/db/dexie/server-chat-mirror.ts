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

/** Persist a receipt only onto the captured source row; edits are never replaced. */
export const acknowledgePromotedChatMessage = async ({
  historyId, chatId, ownerKey, source, serverMessageId, version, signal, isCurrent
}: {
  historyId: string | null; chatId: string; ownerKey: string; source: ChatMessage
  serverMessageId: string; version?: number; signal: AbortSignal; isCurrent: () => boolean
}): Promise<void> => {
  if (!historyId || !source.id) return
  await runChatPersistenceTransaction(signal, async () => {
    const history = await db.chatHistories.get(historyId)
    if (!isCurrent()) throw createServicePromptScopeChangedError()
    if (!history) return
    if ((history.server_chat_id && history.server_chat_id !== chatId) ||
      (history.server_scope_key && history.server_scope_key !== ownerKey)) throw createServicePromptScopeChangedError()
    const row = await db.messages.get(source.id!)
    if (!isCurrent()) throw createServicePromptScopeChangedError()
    if (!row) return // A synthetic greeting has no durable row until the first mirror.
    if (row.history_id !== historyId || row.role !== (source.role || (source.isBot ? "assistant" : "user")) ||
      (row.serverMessageId && row.serverMessageId !== serverMessageId)) throw createServicePromptScopeChangedError()
    await db.messages.put({ ...row, serverMessageId,
      serverMessageVersion: row.serverMessageVersion ?? (row.content === source.message ? version : undefined) })
    if (!isCurrent()) throw createServicePromptScopeChangedError()
  })
}

const canonicalId = (message: ChatMessage) => message.serverMessageId?.trim() || null

/** A request's local user ID is an identity anchor even when the provider failed before any ACK. */
const recoverCorrelatedUsers = (current: ChatMessage[], incoming: ChatMessage[]): ChatMessage[] => {
  const claims = new Map<string, ChatMessage[]>()
  for (const remote of incoming) {
    const clientId = remote.metadataExtra?.client_message_id
    if (remote.isBot || (remote.role && remote.role !== "user") || !canonicalId(remote) ||
      typeof clientId !== "string" || !/^[A-Za-z0-9_-]{1,128}$/.test(clientId)) continue
    claims.set(clientId, [...(claims.get(clientId) || []), remote])
  }
  return current.map(local => {
    const matches = local.id ? claims.get(local.id) : undefined
    if (matches?.length !== 1 || local.isBot || (local.role && local.role !== "user") || canonicalId(local)) return local
    const remote = matches[0]
    const serverId = canonicalId(remote)!
    if (current.some(message => canonicalId(message) === serverId || message.id === serverId) ||
      current.filter(message => message.id === local.id).length !== 1 || local.message !== remote.message ||
      JSON.stringify((local.images || []).filter(Boolean)) !== JSON.stringify((remote.images || []).filter(Boolean))) return local
    return { ...local, serverMessageId: serverId }
  })
}

/** Recover only a local user paired to an acknowledged saved reply. Text alone is never identity. */
const recoverAnchoredUsers = (current: ChatMessage[], incoming: ChatMessage[]): ChatMessage[] => {
  current = recoverCorrelatedUsers(current, incoming)
  const claims = new Map<string, string[]>()
  for (let index = 1; index < incoming.length; index++) {
    const reply = incoming[index]
    const user = incoming[index - 1]
    const replyId = canonicalId(reply)
    const userId = canonicalId(user)
    if (!reply.isBot || (reply.role && reply.role !== "assistant") || user.isBot || (user.role && user.role !== "user") || !replyId || !userId) continue
    if (reply.parentMessageId && reply.parentMessageId !== userId && reply.parentMessageId !== user.id) continue
    if (current.some(message => canonicalId(message) === userId || message.id === userId)) continue
    const anchors = current.filter(message => message.isBot && canonicalId(message) === replyId)
    if (anchors.length !== 1 || !anchors[0].parentMessageId) continue
    const candidates = current.filter(message => message.id === anchors[0].parentMessageId)
    const local = candidates[0]
    if (candidates.length !== 1 || local.isBot || canonicalId(local) || local.message !== user.message ||
      // Text-only local saves historically retained the empty composer image.
      JSON.stringify((local.images || []).filter(image => image !== "")) !==
        JSON.stringify((user.images || []).filter(image => image !== ""))) continue
    claims.set(local.id!, [...(claims.get(local.id!) || []), userId])
  }
  return current.map(message => {
    const ids = message.id ? claims.get(message.id) : undefined
    return ids?.length === 1 ? { ...message, serverMessageId: ids[0] } : message
  })
}

/** Add a completed owned snapshot without dropping local work or rolling back newer rows. */
export const reconcileServerChatMessages = (
  current: ChatMessage[], incoming: ChatMessage[], beforeAwait?: ChatMessage[]
): ChatMessage[] => {
  current = recoverAnchoredUsers(current, incoming)
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
      parentMessageId: remote.parentMessageId ?? local.parentMessageId,
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
  historyId, chatId, ownerKey, messages, localMessages = [], signal
}: {
  historyId: string; chatId: string; ownerKey: string; messages: ChatMessage[]; localMessages?: ChatMessage[]; signal?: AbortSignal
}): Promise<{ localIds: Map<string, string>; rows: Message[] }> => runChatPersistenceTransaction(signal, async () => {
  const history = await db.chatHistories.get(historyId)
  if (history?.server_chat_id !== chatId || history.server_scope_key !== ownerKey) {
    throw createServicePromptScopeChangedError()
  }
  const rows = await db.messages.where("history_id").equals(historyId).toArray()
  const recovered = recoverAnchoredUsers(rows.map<ChatMessage>(row => ({
    id: row.id, serverMessageId: row.serverMessageId, message: row.content,
    name: row.name, isBot: row.role !== "user",
    role: row.role === "user" ? "user" : row.role === "assistant" ? "assistant" : "system",
    images: row.images, sources: row.sources || [], parentMessageId: row.parent_message_id
  })), messages)
  const acknowledgedRows = rows.map((row, index) => ({ ...row, serverMessageId: recovered[index].serverMessageId }))
  const localIds = new Map<string, string>()
  for (const remote of messages) {
    const serverMessageId = canonicalId(remote)
    if (!serverMessageId) continue
    const persisted = acknowledgedRows.find(row => row.serverMessageId === serverMessageId ||
      (!row.serverMessageId && row.id === serverMessageId))
    const confirmed = localMessages.filter(message => canonicalId(message) === serverMessageId && message.id)
    const captured = !persisted && confirmed.length === 1 ? confirmed[0] : undefined
    const local = persisted || (captured ? {
      id: captured.id!, history_id: historyId, role: captured.role || (captured.isBot ? "assistant" : "user"),
      content: captured.message, name: captured.name, images: captured.images || [],
      createdAt: captured.createdAt ?? Date.now(), messageType: captured.messageType,
      parent_message_id: captured.parentMessageId, serverMessageId, serverMessageVersion: captured.serverMessageVersion
    } as Message : undefined)
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
      modelName: remote.modelName, modelImage: remote.modelImage, parent_message_id: remote.parentMessageId ?? local?.parent_message_id ?? null,
      ...(preserveContent ? local : {}), serverMessageId,
      serverMessageVersion: Math.max(localVersion, remoteVersion) || undefined
    }
    if (existing) await db.messages.put(next)
    else await db.messages.add(next)
    localIds.set(serverMessageId, id)
  }
  return { localIds, rows: await db.messages.where("history_id").equals(historyId).toArray() }
})
