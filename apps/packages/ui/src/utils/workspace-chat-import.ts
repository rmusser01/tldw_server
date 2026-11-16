import type { WorkspaceBundleChatSession, WorkspaceExportBundle } from "@/store/workspace-bundle"
import { canonicalHistoryJson } from "@/db/dexie/history-selection"
import { projectWorkspaceRagSources, readWorkspaceRagSourceProvenance, WORKSPACE_RAG_SOURCES_KEY } from "./workspace-rag-sources"

const enrolledChats = new WeakMap<WorkspaceBundleChatSession, { scopeKey: string; workspaceId: string; snapshot: string }>()

/** A parsed archive cannot mint this runtime-only canonical enrollment proof. */
export const bindImportedWorkspaceChat = (session: WorkspaceBundleChatSession, scopeKey: string, workspaceId: string) => {
  enrolledChats.set(session, { scopeKey, workspaceId, snapshot: canonicalHistoryJson(session) })
  return session
}
export const hasImportedWorkspaceChat = (session: WorkspaceBundleChatSession, scopeKey: string, workspaceId: string): boolean => {
  const proof = enrolledChats.get(session)
  return Boolean(proof && proof.scopeKey === scopeKey && proof.workspaceId === workspaceId && proof.snapshot === canonicalHistoryJson(session))
}

/** Source IDs are graph aliases, never destination mutation authority. */
export const remapImportedWorkspaceChat = (
  session: WorkspaceBundleChatSession,
  makeId: () => string = () => crypto.randomUUID(),
): WorkspaceBundleChatSession => {
  const aliases = new Map<string, number>()
  const ids = session.messages.map(() => makeId())
  if (new Set(ids).size !== ids.length) throw new Error("Imported chat message identities are ambiguous.")
  session.messages.forEach((row, index) => {
    if (!row.id || row.variants?.length) throw new Error("Imported chat requires a complete, unambiguous message graph.")
    for (const alias of [row.id, row.serverMessageId].filter(Boolean) as string[]) {
      if (aliases.has(alias) && aliases.get(alias) !== index) throw new Error("Imported chat message aliases are ambiguous.")
      aliases.set(alias, index)
    }
  })
  const parents = session.messages.map(row => {
    if (row.parentMessageId == null) return null
    const parent = aliases.get(row.parentMessageId)
    if (parent === undefined) throw new Error("Imported chat has an unresolved message parent.")
    return parent
  })
  const checked = new Set<number>()
  for (let index = 0; index < parents.length; index++) {
    const path = new Set<number>()
    let current: number | null = index
    while (current !== null && !checked.has(current)) {
      if (path.has(current)) throw new Error("Imported chat message parents contain a cycle.")
      path.add(current)
      current = parents[current]
    }
    path.forEach(value => checked.add(value))
  }
  const messages = session.messages.map((row, index) => {
    const copy = { ...row, id: ids[index], parentMessageId: parents[index] === null ? null : ids[parents[index]!] }
    delete copy.serverMessageId
    delete copy.serverMessageVersion
    return copy
  })
  const history = messages.filter(row => row.message).map(row => ({
    role: row.role || (row.isBot ? "assistant" : "user"), content: row.message,
  }))
  if (session.history.length && canonicalHistoryJson(session.history) !== canonicalHistoryJson(history))
    throw new Error("Imported chat request history differs from its visible messages.")
  return { ...session, messages, history, historyId: null, serverChatId: null }
}

export type WorkspaceChatImportMessage = {
  id: string; parent_message_id: string | null; role: "user" | "assistant" | "system"
  content: string; timestamp: string; images: string[]
  metadata_extra?: Record<string, unknown>
  metadata?: Record<string, unknown>
}
export type WorkspaceChatImport = {
  id: string; title: string; scope_type: "workspace"; workspace_id: string
  created_at: string; last_modified: string; messages: WorkspaceChatImportMessage[]
}

/** Stable receipt-derived identity makes an uncertain atomic import safe to replay. */
export const buildWorkspaceChatImport = async (
  bundle: WorkspaceExportBundle, workspaceId: string,
): Promise<WorkspaceChatImport | null> => {
  const session = bundle.workspace.chatSession
  if (!session?.messages.length) return null
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(`tldw:workspace-import-chat:v1:${workspaceId}`))
  const hex = [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, "0")).join("")
  const id = `${hex.slice(0, 8)}-${hex.slice(8, 12)}-4${hex.slice(13, 16)}-a${hex.slice(17, 20)}-${hex.slice(20, 32)}`
  const messages = session.messages.map((row): WorkspaceChatImportMessage => {
    const role = row.role || (row.isBot ? "assistant" : "user")
    if (!["user", "assistant", "system"].includes(role) || !row.id ||
        typeof row.createdAt !== "number" || !Number.isFinite(row.createdAt) || row.createdAt < 0)
      throw new Error("Imported chat requires supported roles and original message timestamps.")
    const images = (row.images || []).filter(Boolean)
    if (!row.message?.trim() && !images.length) throw new Error("Imported chat contains an empty message.")
    const projected = row.sources?.length ? projectWorkspaceRagSources(row.sources) : null
    const raw = row.metadataExtra?.[WORKSPACE_RAG_SOURCES_KEY]
    const stored = raw == null ? null : readWorkspaceRagSourceProvenance(raw)
    if ((row.sources?.length && !projected) || (raw != null && !stored) ||
        (stored && projected && canonicalHistoryJson(stored) !== canonicalHistoryJson(projected)))
      throw new Error("Imported chat citation evidence is incomplete or inconsistent.")
    const provenance = stored || projected
    if (provenance && role !== "assistant") throw new Error("Imported chat citations require an assistant message.")
    const generation = role === "assistant" ? Object.fromEntries(
      ["generation_status", "model_id", "provider", "finish_reason", "usage"].flatMap(key =>
        row.metadataExtra?.[key] == null ? [] : [[key, row.metadataExtra[key]]])
    ) : {}
    if (role === "assistant" && generation.model_id == null && row.modelId) generation.model_id = row.modelId
    return { id: row.id, parent_message_id: row.parentMessageId ?? null, role: role as WorkspaceChatImportMessage["role"],
      content: row.message || "", timestamp: new Date(row.createdAt).toISOString(), images,
      ...(Object.keys(generation).length ? { metadata: generation } : {}),
      ...(provenance ? { metadata_extra: { [WORKSPACE_RAG_SOURCES_KEY]: provenance } } : {}) }
  })
  const dates = messages.map(row => row.timestamp).sort()
  return { id, title: bundle.workspace.name, scope_type: "workspace", workspace_id: workspaceId,
    created_at: dates[0], last_modified: dates[dates.length - 1], messages }
}
