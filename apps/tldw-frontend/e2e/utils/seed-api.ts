/**
 * Seed data through the real tldw API for live-backend specs.
 *
 * Used by the ux-regression project, which always runs against an isolated
 * backend (bun run e2e:ux-regression). Requests retry on 429 so seeding a
 * library-sized data set does not trip the per-user rate limiters.
 */
import type { APIRequestContext, APIResponse } from "@playwright/test"
import { TEST_CONFIG } from "./helpers"

const MAX_ATTEMPTS = 5

export type SeedApi = {
  serverUrl: string
  get: (path: string, options?: { timeout?: number }) => Promise<APIResponse>
  post: (path: string, data: unknown) => Promise<APIResponse>
}

export type SeededNote = { id: string; title: string }

export type ServerNote = { id: string; title: string; content: string; version: number }

export type SeedChatMessage = { role: "user" | "assistant"; content: string }

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

const retryDelayMs = (response: APIResponse, attempt: number): number => {
  const retryAfter = Number(response.headers()["retry-after"])
  if (Number.isFinite(retryAfter) && retryAfter > 0) return Math.min(retryAfter, 10) * 1000
  return 250 * 2 ** attempt
}

async function withRetry(send: () => Promise<APIResponse>): Promise<APIResponse> {
  let response = await send()
  for (let attempt = 1; attempt < MAX_ATTEMPTS && response.status() === 429; attempt += 1) {
    await sleep(retryDelayMs(response, attempt))
    response = await send()
  }
  return response
}

async function expectOk(response: APIResponse, action: string): Promise<APIResponse> {
  if (!response.ok()) {
    const body = await response.text().catch(() => "")
    throw new Error(`${action} failed: ${response.status()} ${body.slice(0, 300)}`)
  }
  return response
}

export function createSeedApi(
  request: APIRequestContext,
  {
    serverUrl = TEST_CONFIG.serverUrl,
    apiKey = TEST_CONFIG.apiKey,
  }: { serverUrl?: string; apiKey?: string } = {}
): SeedApi {
  const base = serverUrl.replace(/\/$/, "")
  const headers = { "X-API-KEY": apiKey }
  return {
    serverUrl: base,
    get: (path, options = {}) =>
      withRetry(() => request.get(`${base}${path}`, { headers, timeout: options.timeout })),
    post: (path, data) => withRetry(() => request.post(`${base}${path}`, { headers, data })),
  }
}

let backendWarmed: Promise<void> | null = null

/**
 * Pay the backend's cold-start costs once per worker before a spec drives the
 * UI. The first /openapi.json build freezes a cold backend for ~30 s and makes
 * concurrent page requests fail (#3135), which would otherwise make whichever
 * spec runs first flaky.
 */
export function warmBackendOnce(api: SeedApi): Promise<void> {
  backendWarmed ??= (async () => {
    await expectOk(await api.get("/openapi.json", { timeout: 120_000 }), "Warming the OpenAPI schema")
    await expectOk(await api.get("/api/v1/llm/models/metadata", { timeout: 60_000 }), "Warming the model catalog")
  })()
  return backendWarmed
}

/**
 * Create `count` notes titled "<prefix> note 001" … with bounded concurrency.
 * The prefix should include a per-test token (generateTestId) so specs can
 * find their own rows in a shared library.
 */
export async function seedNotes(
  api: SeedApi,
  { count, prefix, keywords = [], concurrency = 4 }: {
    count: number
    prefix: string
    keywords?: string[]
    concurrency?: number
  }
): Promise<SeededNote[]> {
  const width = String(count).length
  const notes: SeededNote[] = new Array(count)
  let next = 0
  const worker = async () => {
    while (next < count) {
      const index = next++
      const title = `${prefix} note ${String(index + 1).padStart(width, "0")}`
      const response = await expectOk(
        await api.post("/api/v1/notes/", {
          title,
          content: `Seeded by the UX regression harness (${prefix}).`,
          keywords,
        }),
        `Seeding note ${title}`
      )
      const created = await response.json()
      notes[index] = { id: String(created?.id ?? ""), title }
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, count) }, worker))
  return notes
}

/** The server's true number of active notes (`pagination.total`). */
export async function notesTotal(api: SeedApi): Promise<number> {
  const response = await expectOk(await api.get("/api/v1/notes/?limit=1&offset=0"), "Counting notes")
  const payload = await response.json()
  const total = Number(payload?.pagination?.total ?? payload?.total)
  if (!Number.isFinite(total)) throw new Error("Notes list did not report a total")
  return total
}

/** Read one note as the server stores it (content and optimistic-lock version). */
export async function readNote(api: SeedApi, id: string): Promise<ServerNote> {
  const response = await expectOk(await api.get(`/api/v1/notes/${encodeURIComponent(id)}`), `Reading note ${id}`)
  const payload = await response.json()
  return {
    id: String(payload?.id ?? id),
    title: String(payload?.title ?? ""),
    content: String(payload?.content ?? ""),
    version: Number(payload?.version),
  }
}

/** Create a character card and return its numeric id. */
export async function createCharacter(api: SeedApi, name: string): Promise<number> {
  const greeting = `Hello from ${name}.`
  const response = await expectOk(
    await api.post("/api/v1/characters/", { name, greeting, first_message: greeting }),
    `Creating character ${name}`
  )
  const created = await response.json()
  const id = Number(created?.id)
  if (!Number.isInteger(id) || id <= 0) throw new Error(`Character ${name} was created without a numeric id`)
  return id
}

/**
 * Create a saved server chat with the given messages, in order, and return
 * its id. The chat is attached to `characterId`, like a chat started from the
 * character picker.
 *
 * Each message replies to the one before it (`parent_message_id`), as the
 * chat UI saves a conversation. Clients that walk the message tree, such as
 * the extension's history selection, cannot load a chat whose messages are
 * unlinked.
 */
export async function createChatWithMessages(
  api: SeedApi,
  { characterId, title, messages }: { characterId: number; title: string; messages: SeedChatMessage[] }
): Promise<string> {
  const response = await expectOk(
    await api.post("/api/v1/chats/", { title, character_id: characterId, state: "in-progress", source: "e2e" }),
    `Creating chat ${title}`
  )
  const created = await response.json()
  const chatId = String(created?.id ?? created?.chat_id ?? created?.conversation_id ?? "")
  if (!chatId) throw new Error(`Chat ${title} was created without an id`)
  let parentId: string | null = null
  for (const message of messages) {
    const added = await expectOk(
      await api.post(
        `/api/v1/chats/${encodeURIComponent(chatId)}/messages`,
        parentId ? { ...message, parent_message_id: parentId } : message
      ),
      `Adding a ${message.role} message to chat ${title}`
    )
    parentId = String((await added.json())?.id ?? "")
    if (!parentId) throw new Error(`A ${message.role} message in chat ${title} was created without an id`)
  }
  return chatId
}
