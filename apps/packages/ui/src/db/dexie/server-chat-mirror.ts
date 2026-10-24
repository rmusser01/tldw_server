import type { Message as ChatMessage } from "@/store/option"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error"
import { db } from "./schema"
import { generateID } from "./helpers"
import { indexMessagesSearchTokens } from "./search-tokens"
import { runChatPersistenceTransaction } from "./chat-persistence-transaction"
import type { Message } from "./types"
import type { ChatScope } from "@/types/chat-scope"
import { assertNoPendingLocalSettingsWrite, type HistoryOperationOptions } from "./history-selection"

/** Full verified target/owner identity; never stores bearer tokens or API keys. */
export const serverChatMirrorOwnerKey = (snapshot: Pick<ServicePromptSnapshot, "requestScope">): string => {
  const { config, userId } = snapshot.requestScope
  return JSON.stringify([
    config.serverUrl.trim().replace(/\/+$/, ""), config.authMode,
    config.authSource || "manual", config.orgId ?? null,
    userId == null ? null : String(userId),
    config.expectedSingleUserApiKeyScope ?? null
  ])
}

type MirrorBinding = { chatId: string; ownerKey: string; scope?: ChatScope }
const mirrorBindingKey = ({ chatId, ownerKey, scope }: MirrorBinding) => JSON.stringify([ownerKey, chatId, scope?.type ?? "global", scope?.type === "workspace" ? scope.workspaceId : null])
// Cache provenance only; callers must retain their original server ownership guards.
const acknowledgedMirrorIds = new Map<string, string>()
export const hasAcknowledgedServerChatMirror = ({ historyId, ...binding }: MirrorBinding & { historyId: string | null }) => historyId !== null && acknowledgedMirrorIds.get(mirrorBindingKey(binding)) === historyId

export const linkServerChatMirror = async ({
  chatId, title, ownerKey, currentHistoryId, legacyHistoryId, scope, signal, validate_lease
}: {
  chatId: string; title: string; ownerKey: string
  currentHistoryId?: string | null; legacyHistoryId?: string | null; scope?: ChatScope
} & HistoryOperationOptions): Promise<string> => {
  const namespace = { server_scope_type: scope?.type ?? "global" as const,
    server_workspace_id: scope?.type === "workspace" ? scope.workspaceId : undefined }
  const receiptKey = mirrorBindingKey({ chatId, ownerKey, scope })
  const localId = await runChatPersistenceTransaction(signal, async () => {
    const candidates = await db.chatHistories.where("server_chat_id").equals(chatId).toArray()
    const sameNamespace = (history: import("./types").HistoryInfo) =>
      !history.server_scope_type || (history.server_scope_type === namespace.server_scope_type &&
        (history.server_workspace_id ?? undefined) === namespace.server_workspace_id)
    const owned = candidates.find(history => history.server_scope_key === ownerKey && sameNamespace(history))
    const current = currentHistoryId ? await db.chatHistories.get(currentHistoryId) : undefined
    const canLinkCurrent = current && sameNamespace(current) && (!current.server_chat_id || current.server_chat_id === chatId) &&
      (current.server_scope_key === ownerKey ||
        (!current.server_scope_key && current.id === legacyHistoryId))
    const history = owned || (canLinkCurrent ? current : undefined)
    if (history) {
      await db.chatHistories.update(history.id, { title, server_chat_id: chatId, server_scope_key: ownerKey, ...namespace })
      return history.id
    }
    const id = generateID()
    await db.chatHistories.add({ id, title, createdAt: Date.now(), is_rag: false,
      message_source: "server", server_chat_id: chatId, server_scope_key: ownerKey, ...namespace })
    return id
  }, undefined, validate_lease)
  acknowledgedMirrorIds.set(receiptKey, localId)
  return localId
}

/**
 * Delete this owner's local copies of a server chat once the chat itself is
 * deleted (moved to Trash), so local history search stops offering it. Another
 * owner's copy is left alone. Returns the ids of the removed histories.
 */
export const removeServerChatMirror = async ({
  chatId,
  ownerKey,
  scope,
  signal,
  validate_lease
}: {
  chatId: string
  ownerKey: string
  scope?: ChatScope
} & HistoryOperationOptions): Promise<string[]> =>
  db.transaction(
    "rw",
    [db.chatHistories, db.messages, db.compareStates],
    async () => {
      const assertCurrent = () => {
        if (signal?.aborted || validate_lease?.() === false)
          throw createServicePromptScopeChangedError()
      }
      assertCurrent()
      const owned = (
        await db.chatHistories.where("server_chat_id").equals(chatId).toArray()
      ).filter(
        (history) =>
          history.server_scope_key === ownerKey &&
          (scope?.type === "workspace"
            ? history.server_scope_type === "workspace" &&
              history.server_workspace_id === scope.workspaceId
            : (!history.server_scope_type ||
                history.server_scope_type === "global") &&
              !history.server_workspace_id)
      )
      assertCurrent()
      owned.forEach(assertNoPendingLocalSettingsWrite)
      for (const history of owned) {
        assertCurrent()
        await db.chatHistories.delete(history.id)
        assertCurrent()
        await db.messages.where("history_id").equals(history.id).delete()
        assertCurrent()
        await db.compareStates.delete(history.id)
        assertCurrent()
      }
      return owned.map((history) => history.id)
    }
  )

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

const canonicalId = (message: Pick<ChatMessage, "serverMessageId">) => message.serverMessageId?.trim() || null

/** Content edits do not change the creation time of an acknowledged message. */
const canonicalCreatedAt = (createdAt: unknown): number | undefined =>
  typeof createdAt === "number" && Number.isFinite(createdAt) ? createdAt : undefined

/** A request's local user ID is an identity anchor even when the provider failed before any ACK. */
const recoverCorrelatedUsers = (current: ChatMessage[], incoming: ChatMessage[]): ChatMessage[] => {
  const claims = new Map<string, ChatMessage[]>()
  for (const remote of incoming) {
    const clientId = remote.metadataExtra?.client_message_id
    if (remote.isBot || (remote.role && remote.role !== "user") || !canonicalId(remote) ||
      typeof clientId !== "string" || !/^[A-Za-z0-9_-]{1,128}$/.test(clientId)) continue
    claims.set(clientId, [...(claims.get(clientId) || []), remote])
  }
  // One pass up front instead of a scan per local row: canonical ids and raw
  // ids are both ownership anchors a server id must not collide with, and a
  // local id may only claim when it is unambiguous.
  const knownIds = new Set<string>()
  const localIdCounts = new Map<string, number>()
  for (const message of current) {
    const canonical = canonicalId(message)
    if (canonical) knownIds.add(canonical)
    if (message.id) {
      knownIds.add(message.id)
      localIdCounts.set(message.id, (localIdCounts.get(message.id) || 0) + 1)
    }
  }
  return current.map(local => {
    const matches = local.id ? claims.get(local.id) : undefined
    if (matches?.length !== 1 || local.isBot || (local.role && local.role !== "user") || canonicalId(local)) return local
    const remote = matches[0]
    const serverId = canonicalId(remote)!
    if (knownIds.has(serverId) || localIdCounts.get(local.id) !== 1 || local.message !== remote.message ||
      JSON.stringify((local.images || []).filter(Boolean)) !== JSON.stringify((remote.images || []).filter(Boolean))) return local
    return { ...local, serverMessageId: serverId }
  })
}

/** Recover only through an acknowledged reply and its explicit canonical remote parent. */
const recoverAnchoredUsers = (current: ChatMessage[], incoming: ChatMessage[]): ChatMessage[] => {
  current = recoverCorrelatedUsers(current, incoming)
  // Index the post-correlation state once; multiplicity is load-bearing, so
  // every index keeps all matches in order.
  const incomingByCanonical = new Map<string, ChatMessage[]>()
  for (const message of incoming) {
    const id = canonicalId(message)
    if (id) incomingByCanonical.set(id, [...(incomingByCanonical.get(id) || []), message])
  }
  const knownIds = new Set<string>()
  const currentById = new Map<string, ChatMessage[]>()
  const botByCanonical = new Map<string, ChatMessage[]>()
  for (const message of current) {
    const canonical = canonicalId(message)
    if (canonical) {
      knownIds.add(canonical)
      if (message.isBot) botByCanonical.set(canonical, [...(botByCanonical.get(canonical) || []), message])
    }
    if (message.id) {
      knownIds.add(message.id)
      currentById.set(message.id, [...(currentById.get(message.id) || []), message])
    }
  }
  const claims = new Map<string, string[]>()
  for (const reply of incoming) {
    const replyId = canonicalId(reply)
    if (!reply.isBot || (reply.role && reply.role !== "assistant") || !replyId || !reply.parentMessageId) continue
    const parents = incomingByCanonical.get(reply.parentMessageId) || []
    if (parents.length !== 1) continue
    const user = parents[0]
    const userId = canonicalId(user)!
    if (user.isBot || (user.role && user.role !== "user")) continue
    if (knownIds.has(userId)) continue
    const anchors = botByCanonical.get(replyId) || []
    if (anchors.length !== 1 || !anchors[0].parentMessageId) continue
    const candidates = currentById.get(anchors[0].parentMessageId) || []
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
  const variants = incoming.filter(message => message.isBot).flatMap(message => message.variants || [])
  const serverIds = new Set([...incoming, ...variants].map(canonicalId).filter(Boolean))
  const key = (message: Pick<ChatMessage, "id" | "serverMessageId">) => {
    const serverId = canonicalId(message) ||
      (message.id && serverIds.has(String(message.id)) ? String(message.id) : null)
    return serverId ? `server:${serverId}` : message.id ? `local:${message.id}` : null
  }
  const incomingKeys = new Set(incoming.map(key).filter(Boolean))
  const variantsById = new Map(variants.map(variant => [key(variant), variant]))
  const isRepresentedVariant = (local: Pick<ChatMessage, "id" | "serverMessageId" | "message" | "images">) => {
    const id = key(local)
    const remote = id ? variantsById.get(id) : undefined
    return remote !== undefined && local.message === remote.message &&
      JSON.stringify(local.images || []) === JSON.stringify(remote.images || [])
  }
  const beforeById = new Map((beforeAwait || []).map(message => [key(message), message]))
  const localById = new Map(current.flatMap(message => {
    const id = key(message)
    return id ? [[id, message] as const] : []
  }))
  const hasServerGreeting = incoming.some(message => message.messageType === "character:greeting" || message.messageType === "greeting")
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
      createdAt: canonicalCreatedAt(remote.createdAt) ?? local.createdAt,
      parentMessageId: remote.parentMessageId ?? local.parentMessageId,
      id: local.id || remote.id, serverMessageId: canonicalId(remote) || canonicalId(local) || undefined,
      serverMessageVersion: Math.max(localVersion, remoteVersion) || undefined }
  })
  for (const local of current) {
    const id = key(local)
    if (id && incomingKeys.has(id)) continue
    // The mirror collapses saved alternatives into one visible answer. Do not
    // append an unchanged copy again; retain edits in any local variant.
    if (local.isBot && isRepresentedVariant(local) && (local.variants || []).every(isRepresentedVariant)) continue
    // A synthetic greeting is superseded only by an acknowledged server greeting.
    if (!canonicalId(local) && (local.messageType === "character:greeting" || local.messageType === "greeting") &&
      hasServerGreeting) continue
    merged.push(local)
  }
  return merged
}

export const reconcileServerChatMirror = async ({
  historyId, chatId, ownerKey, messages, localMessages = [], signal, validate_lease
}: {
  historyId: string; chatId: string; ownerKey: string; messages: ChatMessage[]; localMessages?: ChatMessage[]
} & HistoryOperationOptions): Promise<{ localIds: Map<string, string>; rows: Message[] }> => runChatPersistenceTransaction(signal, async () => {
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
  // First persisted row per server id — for unacknowledged rows the raw id is
  // the only alias a server id can match. One index instead of a find() scan
  // per incoming message; first write wins, like the first find() match.
  const persistedByKey = new Map<string, Message>()
  for (const row of acknowledgedRows) {
    const key = row.serverMessageId || row.id
    if (!persistedByKey.has(key)) persistedByKey.set(key, row)
  }
  // A captured source row must be confirmed by exactly one live message.
  const confirmedByServerId = new Map<string, ChatMessage[]>()
  for (const message of localMessages) {
    const serverId = canonicalId(message)
    if (!serverId || !message.id) continue
    confirmedByServerId.set(serverId, [...(confirmedByServerId.get(serverId) || []), message])
  }
  // Plan every write synchronously; reads and writes are batched below.
  const planned = messages.flatMap(remote => {
    const serverMessageId = canonicalId(remote)
    if (!serverMessageId) return []
    const persisted = persistedByKey.get(serverMessageId)
    const confirmed = confirmedByServerId.get(serverMessageId) || []
    const captured = !persisted && confirmed.length === 1 ? confirmed[0] : undefined
    const local = persisted || (captured ? {
      id: captured.id!, history_id: historyId, role: captured.role || (captured.isBot ? "assistant" : "user"),
      content: captured.message, name: captured.name, images: captured.images || [],
      createdAt: captured.createdAt ?? Date.now(), messageType: captured.messageType,
      parent_message_id: captured.parentMessageId, serverMessageId, serverMessageVersion: captured.serverMessageVersion
    } as Message : undefined)
    const id = local?.id || `${historyId}:server:${encodeURIComponent(serverMessageId)}`
    const localVersion = local?.serverMessageVersion ?? 0
    const remoteVersion = remote.serverMessageVersion ?? 0
    const preserveContent = local && (local.serverMessageVersion == null || localVersion >= remoteVersion) && local.content !== remote.message
    const next: Message = {
      ...local, id, history_id: historyId, name: remote.name || (remote.role === "user" || (!remote.role && !remote.isBot) ? "You" : remote.role === "system" ? "System" : "Assistant"),
      role: remote.role || (remote.isBot ? "assistant" : "user"), content: remote.message,
      images: remote.images || [], sources: remote.sources || [],
      messageType: remote.messageType, generationInfo: remote.generationInfo,
      metadataExtra: remote.metadataExtra, clusterId: remote.clusterId, modelId: remote.modelId,
      modelName: remote.modelName, modelImage: remote.modelImage, parent_message_id: remote.parentMessageId ?? local?.parent_message_id ?? null,
      ...(preserveContent ? local : {}), serverMessageId,
      createdAt: canonicalCreatedAt(remote.createdAt) ?? local?.createdAt ?? Date.now(),
      serverMessageVersion: Math.max(localVersion, remoteVersion) || undefined
    }
    return [{ serverMessageId, id, next }]
  })
  // One batched existence read; a qualified id owned by another history must
  // abort the transaction before any write lands.
  const existingRows = await db.messages.bulkGet(planned.map(message => message.id))
  for (const existing of existingRows)
    if (existing && existing.history_id !== historyId) throw createServicePromptScopeChangedError()
  const nextRows = planned.map(message => message.next)
  if (nextRows.length > 0) {
    await db.messages.bulkPut(nextRows)
    // Mirror content can change (remote edits); keep the search index fresh.
    // Stale tokens for previous content are harmless extras that exact
    // verification rejects.
    await indexMessagesSearchTokens(nextRows)
  }
  const localIds = new Map<string, string>(planned.map(message => [message.serverMessageId, message.id] as const))
  // Return exactly what a fresh index scan would: every row of this history
  // ordered by primary key (equal index keys order by primary key), with the
  // planned writes overlaid on the initial read.
  const rowsById = new Map(rows.map(row => [row.id, row] as const))
  for (const next of nextRows) rowsById.set(next.id, next)
  const mirrorRows = [...rowsById.values()].sort((left, right) => (left.id < right.id ? -1 : left.id > right.id ? 1 : 0))
  return { localIds, rows: mirrorRows }
}, undefined, validate_lease)

/** Apply a successful canonical delete to only its already-acknowledged local mirror. */
export const removeAcknowledgedServerMirrorMessage = async ({
  historyId,
  chatId,
  localMessageId,
  serverMessageId,
  ownerKey,
  scope,
  signal,
  validate_lease
}: {
  historyId: string
  chatId: string
  localMessageId?: string
  serverMessageId: string
  ownerKey?: string
  scope?: ChatScope
} & HistoryOperationOptions): Promise<void> =>
  runChatPersistenceTransaction(
    signal,
    async () => {
      const history = await db.chatHistories.get(historyId)
      if (
        history?.server_chat_id !== chatId ||
        !history.server_scope_key ||
        (ownerKey !== undefined && history.server_scope_key !== ownerKey) ||
        (scope?.type === "workspace"
          ? history.server_scope_type !== "workspace" ||
            history.server_workspace_id !== scope.workspaceId
          : scope &&
            ((history.server_scope_type &&
              history.server_scope_type !== "global") ||
              history.server_workspace_id))
      )
        throw createServicePromptScopeChangedError()
      const rows = await db.messages
        .where("history_id")
        .equals(historyId)
        .toArray()
      const matches = rows.filter(
        (candidate) => candidate.serverMessageId === serverMessageId
      )
      if (!matches.length && !localMessageId) return
      const row = localMessageId
        ? await db.messages.get(localMessageId)
        : matches[0]
      if (!row) return
      if (
        row.history_id !== historyId ||
        row.serverMessageId !== serverMessageId ||
        matches.length !== 1
      )
        throw createServicePromptScopeChangedError()
      await db.messages.delete(row.id)
    },
    undefined,
    validate_lease
  )

/** H1 translates a clicked local mirror ID before native capture. No text/position inference. */
export const resolveServerMirrorCursor = async ({
  historyId, chatId, ownerKey, cursor, signal
}: {
  historyId: string; chatId: string; ownerKey: string
  cursor: import('@/types/history-selection').HistoryCursorV1; signal?: AbortSignal
}): Promise<import('@/types/history-selection').HistoryCursorV1> => runChatPersistenceTransaction(signal, async () => {
  const history = await db.chatHistories.get(historyId)
  if (history?.server_chat_id !== chatId || history.server_scope_key !== ownerKey) throw createServicePromptScopeChangedError()
  if (cursor.kind === 'empty') return { kind: 'empty' }
  const row = await db.messages.get(cursor.message_id)
  if (!row || row.history_id !== historyId || !row.serverMessageId?.trim()) throw new Error('missing_canonical_message_id')
  const rows = await db.messages.where('history_id').equals(historyId).toArray()
  if (rows.filter(candidate => candidate.serverMessageId === row.serverMessageId).length !== 1) throw new Error('ambiguous_canonical_message_id')
  return { kind: cursor.kind, message_id: row.serverMessageId }
})
