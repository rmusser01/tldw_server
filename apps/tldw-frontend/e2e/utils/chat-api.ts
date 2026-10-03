/**
 * Read and extend saved server chats through the real API, for live-backend
 * specs that check what a UI action did to the server's copy of a chat.
 * Seeding a chat lives in seed-api.ts (createChatWithMessages).
 */
import type { APIResponse } from "@playwright/test"
import type { SeedApi } from "./seed-api"

export type ServerChatMessage = {
  id: string
  parentId: string | null
  role: string
  content: string
}

export type ServerChatState = {
  /** HTTP status of GET /api/v1/chats/<id>: 404 once the chat is deleted or in the trash. */
  status: number
  title: string | null
  deleted: boolean
}

async function ensureOk(response: APIResponse, action: string): Promise<APIResponse> {
  if (!response.ok()) {
    const body = await response.text().catch(() => "")
    throw new Error(`${action} failed: ${response.status()} ${body.slice(0, 300)}`)
  }
  return response
}

/** A chat as GET /api/v1/chats/<id> reports it. A soft-deleted chat reads as 404. */
export async function readServerChat(api: SeedApi, chatId: string): Promise<ServerChatState> {
  const response = await api.get(`/api/v1/chats/${encodeURIComponent(chatId)}`)
  if (response.status() === 404) return { status: 404, title: null, deleted: true }
  await ensureOk(response, `Reading chat ${chatId}`)
  const payload = await response.json()
  return { status: response.status(), title: payload?.title ?? null, deleted: Boolean(payload?.deleted) }
}

/** The chat's messages in server order (oldest first). */
export async function listServerChatMessages(api: SeedApi, chatId: string): Promise<ServerChatMessage[]> {
  const response = await ensureOk(
    await api.get(`/api/v1/chats/${encodeURIComponent(chatId)}/messages?limit=200`),
    `Listing the messages of chat ${chatId}`
  )
  const payload = await response.json()
  const rows: Array<Record<string, unknown>> = Array.isArray(payload?.messages) ? payload.messages : []
  return rows.map((row) => ({
    id: String(row.id ?? ""),
    parentId: row.parent_message_id ? String(row.parent_message_id) : null,
    role: String(row.sender ?? row.role ?? ""),
    content: String(row.content ?? ""),
  }))
}

/**
 * Append a message to a saved chat as a reply to `parentId`, the way another
 * client continuing the conversation saves it. Returns the new message id.
 */
export async function addServerChatMessage(
  api: SeedApi,
  chatId: string,
  message: { role: "user" | "assistant"; content: string; parentId: string }
): Promise<string> {
  const response = await ensureOk(
    await api.post(`/api/v1/chats/${encodeURIComponent(chatId)}/messages`, {
      role: message.role,
      content: message.content,
      parent_message_id: message.parentId,
    }),
    `Adding a ${message.role} message to chat ${chatId}`
  )
  const id = String((await response.json())?.id ?? "")
  if (!id) throw new Error(`A ${message.role} message in chat ${chatId} was created without an id`)
  return id
}
