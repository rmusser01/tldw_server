/**
 * In-memory fake of the tldw HTTP API for Playground integration tests.
 *
 * It is installed as `globalThis.fetch`, so the real tldwClient, bgRequest and
 * bgStream code paths run unchanged and every request the UI makes is recorded
 * with its method, path, query and JSON body.
 */
import { resolveHistorySelection } from "@/utils/history-selection"

export const FAKE_SERVER_URL = "http://tldw.test"
export const FAKE_API_KEY = "harness-api-key"
export const FAKE_PROVIDER = "custom-openai-api"
export const FAKE_MODEL = "local-uat-chat"
/** The model id the Playground stores for the one configured provider. */
export const FAKE_SELECTED_MODEL = `${FAKE_PROVIDER}:${FAKE_MODEL}`

/** A parsed JSON request body. The fake reads arbitrary client payloads defensively. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any -- request bodies are schemaless JSON
export type JsonBody = Record<string, any>

export type RecordedRequest = {
  method: string
  path: string
  query: Record<string, string>
  /** Parsed JSON body; a non-JSON text body is recorded as `{ raw }`. */
  body: JsonBody | undefined
}

export type FakeChatMessage = {
  id: string
  role: "user" | "assistant" | "system"
  content: string
  parent_message_id?: string | null
  created_at: string
  version: number
}

export type FakeChat = {
  id: string
  title: string
  created_at: string
  last_modified: string
  version: number
  state: string
  source?: string | null
  messages: FakeChatMessage[]
}

/** One configured LLM provider and its chat models, in catalog order. */
export type FakeProvider = {
  name: string
  displayName?: string
  models: string[]
}

/** The provider catalog served by /llm/providers and /llm/models/metadata. */
export type FakeCatalog = {
  providers: FakeProvider[]
  defaultProvider: string
}

const DEFAULT_CATALOG: FakeCatalog = {
  providers: [{ name: FAKE_PROVIDER, displayName: "Custom OpenAI API", models: [FAKE_MODEL] }],
  defaultProvider: FAKE_PROVIDER
}

/** How the next chat completion answers; defaults to an immediate reply. */
export type CompletionPlan = {
  reply?: string
  /** Delay before the first streamed token (ms, uses timers so fake timers apply). */
  firstTokenDelayMs?: number
  /** Resolve the response only when this promise settles. */
  gate?: Promise<void>
  status?: number
  errorBody?: unknown
  /**
   * Stream the reply as these content deltas instead of one chunk. With
   * `pauseAfterChunks`, the stream stops after that many deltas until `resume`
   * settles (or the client aborts), which models a reply still being written.
   */
  chunks?: string[]
  pauseAfterChunks?: number
  resume?: Promise<void>
  /** After resuming: finish normally (default) or drop the connection. */
  end?: "finish" | "drop"
}

type Deferred = { promise: Promise<void>; resolve: () => void }
export const deferred = (): Deferred => {
  let resolve!: () => void
  const promise = new Promise<void>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" }
  })

/** Wait for `gate`, rejecting with an AbortError if the client aborts first. */
const waitUnlessAborted = (gate: Promise<void>, signal?: AbortSignal | null) =>
  new Promise<void>((resolve, reject) => {
    const abort = () => reject(new DOMException("The operation was aborted.", "AbortError"))
    if (signal?.aborted) return abort()
    signal?.addEventListener("abort", abort, { once: true })
    gate.then(resolve, reject)
  })

const sleep = (ms: number, signal?: AbortSignal | null) =>
  new Promise<void>((resolve, reject) => {
    const timer = setTimeout(resolve, ms)
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer)
        reject(new DOMException("The operation was aborted.", "AbortError"))
      },
      { once: true }
    )
  })

const readBody = async (
  input: RequestInfo | URL,
  init?: RequestInit
): Promise<JsonBody | undefined> => {
  const raw =
    init?.body ?? (input instanceof Request ? await input.clone().text() : undefined)
  if (raw == null) return undefined
  if (typeof raw !== "string") return { raw: "[non-text body]" }
  try {
    const parsed: unknown = JSON.parse(raw)
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as JsonBody)
      : { raw: parsed }
  } catch {
    return { raw }
  }
}

const nowIso = () => new Date().toISOString()

/** Optional capability probes the UI tolerates as unsupported (404). */
const OPTIONAL_UNSUPPORTED = new Set([
  "/api/v1/users/me/profile",
  "/api/v1/prompts/capabilities",
  "/api/v1/ingestion-sources/capabilities",
  "/api/v1/rag/health",
  "/api/v1/audio/voices",
  "/api/v1/audio/voices/catalog"
])

/** Minimal OpenAPI document advertising the chat surface the real server exposes. */
const OPENAPI_SPEC = {
  openapi: "3.1.0",
  info: { title: "tldw harness", version: "harness" },
  paths: {
    "/api/v1/chat/completions": {
      post: {
        requestBody: {
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  model: { type: "string" },
                  messages: { type: "array" },
                  stream: { type: "boolean" },
                  save_to_db: { type: "boolean" },
                  conversation_id: { type: "string" }
                }
              }
            }
          }
        }
      }
    },
    "/api/v1/chats/": { get: {}, post: {} },
    "/api/v1/chats/{chat_id}": { get: {}, put: {}, delete: {} },
    "/api/v1/chats/{chat_id}/messages": { get: {}, post: {} },
    "/api/v1/chats/{chat_id}/settings": { get: {}, put: {} },
    "/api/v1/chat/conversations/{conversation_id}/history/selection": { post: {} },
    "/api/v1/llm/providers": { get: {} },
    "/api/v1/llm/models/metadata": { get: {} },
    "/api/v1/health": { get: {} }
  }
}

export const createFakeTldwServer = ({ catalog = DEFAULT_CATALOG }: { catalog?: FakeCatalog } = {}) => {
  const requests: RecordedRequest[] = []
  const unhandled: RecordedRequest[] = []
  const chats = new Map<string, FakeChat>()
  const completionPlans: CompletionPlan[] = []
  let defaultReply = "Harness reply"
  let chatSequence = 0
  let messageSequence = 0

  const nextChatId = () => {
    chatSequence += 1
    return `server-chat-${chatSequence}`
  }
  const nextMessageId = () => {
    messageSequence += 1
    return `server-message-${messageSequence}`
  }

  const chatSummary = (chat: FakeChat) => ({
    id: chat.id,
    // The real ChatSessionResponse always carries its scope ("global" by
    // default); the server-chat loader refuses a chat whose scope is missing.
    scope_type: "global",
    workspace_id: null,
    title: chat.title,
    created_at: chat.created_at,
    last_modified: chat.last_modified,
    updated_at: chat.last_modified,
    version: chat.version,
    state: chat.state,
    source: chat.source ?? null,
    character_id: null,
    assistant_kind: null,
    assistant_id: null,
    message_count: chat.messages.length
  })

  const messageRow = (message: FakeChatMessage) => ({
    id: message.id,
    role: message.role,
    sender: message.role,
    content: message.content,
    created_at: message.created_at,
    version: message.version,
    parent_message_id: message.parent_message_id ?? null
  })

  const appendMessage = (
    chat: FakeChat,
    message: Omit<FakeChatMessage, "id" | "created_at" | "version"> & { id?: string }
  ) => {
    const row: FakeChatMessage = {
      ...message,
      id: String(message.id ?? nextMessageId()),
      created_at: nowIso(),
      version: 1
    }
    chat.messages.push(row)
    chat.version += 1
    chat.last_modified = row.created_at
    return row
  }

  const createChat = (title: string, extra: Partial<FakeChat> = {}) => {
    const stamp = nowIso()
    const chat: FakeChat = {
      id: nextChatId(),
      title,
      created_at: stamp,
      last_modified: stamp,
      version: 1,
      state: "in-progress",
      source: null,
      messages: [],
      ...extra
    }
    chats.set(chat.id, chat)
    return chat
  }

  /**
   * `save_to_db: true` on /chat/completions persists the turn server side, into
   * `conversation_id` or a new conversation, like the real endpoint.
   */
  const persistCompletionTurn = (body: JsonBody | undefined, reply: string) => {
    if (body?.save_to_db !== true) return null
    const existing = body.conversation_id ? chats.get(String(body.conversation_id)) : undefined
    const chat = existing ?? createChat("Chat", { source: "chat-completions" })
    const lastUser = [...(body?.messages ?? [])].reverse().find((m: JsonBody) => m?.role === "user")
    const user = appendMessage(chat, {
      role: "user",
      content:
        typeof lastUser?.content === "string"
          ? lastUser.content
          : JSON.stringify(lastUser?.content ?? ""),
      parent_message_id: chat.messages.at(-1)?.id ?? null
    })
    appendMessage(chat, { role: "assistant", content: reply, parent_message_id: user.id })
    return chat.id
  }

  /** Emulates the native history-selection capture with the client's own resolver. */
  const captureSelection = (chat: FakeChat, request: JsonBody) => {
    const ownerKey = "harness-native-owner"
    const view = { ...request.view, owner_key: ownerKey }
    const nodes = chat.messages.map((message, index) => ({
      id: message.id,
      revision: String(message.version),
      role: message.role,
      // A stored null parent is a root (e.g. a question reasked from the
      // start); only rows that never recorded a parent read as linear.
      parent_id:
        message.parent_message_id !== undefined
          ? message.parent_message_id
          : index
            ? chat.messages[index - 1].id
            : null,
      settled: true,
      preview: message.content.slice(0, 120)
    }))
    const snapshot = {
      version: 1 as const,
      owner_key: ownerKey,
      conversation_id: chat.id,
      fences: { conversation: String(chat.version), history: String(chat.messages.length), settings: "1" },
      nodes,
      source_digest: `source-${chat.messages.length}`,
      storage_context_digest: "storage",
      interpretation_status: { kind: "parent_graph_v1" as const }
    }
    const resolved = resolveHistorySelection(snapshot as never, view, request.purpose, "")
    if (resolved.status !== "ready") {
      return { status: resolved.status, code: resolved.code, snapshot, view }
    }
    const text = new Map(chat.messages.map((message) => [message.id, message.content]))
    return {
      status: "captured",
      snapshot,
      view,
      rows: resolved.rows,
      purpose: request.purpose,
      storage_context_digest: "storage",
      selected_content: resolved.rows.map((row: { id: string; revision: string }) => ({
        id: row.id,
        revision: row.revision,
        message: text.get(row.id) ?? "",
        images: []
      }))
    }
  }

  const completionResponse = async (body: JsonBody | undefined, signal?: AbortSignal | null) => {
    const plan = completionPlans.shift() ?? {}
    if (plan.gate) await plan.gate
    if (plan.status && plan.status >= 400) {
      return json(plan.errorBody ?? { detail: "Provider failed" }, plan.status)
    }
    const reply = plan.chunks ? plan.chunks.join("") : (plan.reply ?? defaultReply)
    const model = body?.model ?? FAKE_MODEL
    const conversationId = persistCompletionTurn(body, reply)
    if (!body?.stream) {
      return json({
        ...(conversationId ? { tldw_conversation_id: conversationId } : {}),
        id: "chatcmpl-harness",
        object: "chat.completion",
        model,
        choices: [
          { index: 0, finish_reason: "stop", message: { role: "assistant", content: reply } }
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 }
      })
    }
    const encoder = new TextEncoder()
    const chunk = (payload: unknown) => encoder.encode(`data: ${JSON.stringify(payload)}\n\n`)
    const stream = new ReadableStream<Uint8Array>({
      async start(controller) {
        try {
          if (plan.firstTokenDelayMs) await sleep(plan.firstTokenDelayMs, signal)
          const deltas = plan.chunks ?? [reply]
          const pause = async () => {
            // An unresolved `resume` models a page that never hears back.
            await waitUnlessAborted(plan.resume ?? new Promise<void>(() => {}), signal)
            if (plan.end === "drop") throw new TypeError("network error")
          }
          for (const [index, content] of deltas.entries()) {
            if (index === plan.pauseAfterChunks) await pause()
            controller.enqueue(
              chunk({
                id: "chatcmpl-harness",
                object: "chat.completion.chunk",
                model,
                choices: [
                  {
                    index: 0,
                    delta: index === 0 ? { role: "assistant", content } : { content },
                    finish_reason: null
                  }
                ]
              })
            )
          }
          if (plan.pauseAfterChunks === deltas.length) await pause()
          controller.enqueue(
            chunk({
              id: "chatcmpl-harness",
              object: "chat.completion.chunk",
              model,
              choices: [{ index: 0, delta: {}, finish_reason: "stop" }]
            })
          )
          controller.enqueue(encoder.encode("data: [DONE]\n\n"))
          controller.close()
        } catch (error) {
          controller.error(error)
        }
      }
    })
    return new Response(stream, {
      status: 200,
      headers: { "content-type": "text/event-stream" }
    })
  }

  const route = async (request: RecordedRequest, signal?: AbortSignal | null): Promise<Response> => {
    const { method, path, body } = request
    if (OPTIONAL_UNSUPPORTED.has(path)) {
      return json({ detail: "Not supported by the harness server" }, 404)
    }
    if (path === "/openapi.json") return json(OPENAPI_SPEC)
    if (path === "/api/v1/config/docs-info") {
      return json({ capabilities: {}, supported_features: {}, ffmpeg_available: false })
    }
    if (path === "/api/v1/config/providers") {
      return json({
        providers: catalog.providers.map((provider) => ({
          name: provider.name,
          configured: true,
          requires_api_key: false
        })),
        any_configured: true
      })
    }
    if (path === "/api/v1/health" || path.startsWith("/api/v1/health/")) {
      return json({ status: "healthy", ok: true })
    }
    if (path === "/api/v1/llm/providers") {
      return json({
        providers: catalog.providers.map((provider) => ({
          name: provider.name,
          display_name: provider.displayName ?? provider.name,
          type: "openai-compatible",
          is_configured: true,
          configured: true,
          default_model: provider.models[0],
          models: provider.models,
          models_info: provider.models.map((model) => ({
            id: model,
            name: model,
            type: "chat",
            capabilities: ["chat"]
          }))
        })),
        default_provider: catalog.defaultProvider,
        total_configured: catalog.providers.length
      })
    }
    if (path === "/api/v1/llm/models/metadata" || path === "/api/v1/llm/models") {
      const models = catalog.providers.flatMap((provider) =>
        provider.models.map((model) => ({
          id: model,
          name: model,
          provider: provider.name,
          type: "chat",
          modalities: { input: ["text"], output: ["text"] },
          capabilities: ["chat", "streaming"],
          context_window: 8192,
          is_configured: true
        }))
      )
      return json({ models, total: models.length })
    }
    if (path === "/api/v1/chat/completions" && method === "POST") {
      return completionResponse(body, signal)
    }
    const capture = path.match(/^\/api\/v1\/chat\/conversations\/([^/]+)\/history\/selection$/)
    if (capture && method === "POST") {
      const chat = chats.get(decodeURIComponent(capture[1]))
      return chat ? json(captureSelection(chat, body)) : json({ detail: "Chat not found" }, 404)
    }
    if ((path === "/api/v1/chats/" || path === "/api/v1/chats") && method === "GET") {
      const list = [...chats.values()].map(chatSummary)
      return json({ chats: list, total: list.length })
    }
    if ((path === "/api/v1/chats/" || path === "/api/v1/chats") && method === "POST") {
      const chat = createChat(String(body?.title ?? "Untitled"), {
        state: String(body?.state ?? "in-progress"),
        source: body?.source ?? null
      })
      return json(chatSummary(chat), 201)
    }
    const messages = path.match(/^\/api\/v1\/chats\/([^/]+)\/messages$/)
    if (messages) {
      const chat = chats.get(decodeURIComponent(messages[1]))
      if (!chat) return json({ detail: "Chat not found" }, 404)
      if (method === "GET") return json({ messages: chat.messages.map(messageRow) })
      if (method === "POST") {
        const selection = body?.tldw_history_selection_v1
        const admission = body?.tldw_history_admission_v1
        // Like append_selected_history_input: the input follows the last
        // selected message, whatever the cursor kind.
        const parent = selection
          ? (selection.messages?.at(-1)?.id ?? null)
          : admission
            ? admission.input_message_id
            : (body?.parent_message_id ?? chat.messages.at(-1)?.id ?? null)
        const message = appendMessage(chat, {
          id: body?.id,
          role: body?.role ?? "user",
          content: String(body?.content ?? ""),
          parent_message_id: parent
        })
        if (selection) {
          // Native history admission: the server accepts the user turn against
          // the selection it was sent with and echoes the selected path back.
          return json(
            {
              ...messageRow(message),
              tldw_history_admission_v1: {
                version: 1,
                owner_key: selection.owner_key,
                conversation_id: chat.id,
                input_message_id: message.id,
                input_message_revision: String(message.version),
                selection_digest: selection.selection_digest,
                messages: selection.messages,
                originating_selection_revision: selection.selection_revision
              }
            },
            201
          )
        }
        return json(messageRow(message), 201)
      }
    }
    if (/^\/api\/v1\/chats\/[^/]+\/research-runs$/.test(path) && method === "GET") {
      return json({ runs: [] })
    }
    const settings = path.match(/^\/api\/v1\/chats\/([^/]+)\/settings$/)
    if (settings) {
      return json({ conversation_id: decodeURIComponent(settings[1]), settings: {}, version: 1 })
    }
    const chatPath = path.match(/^\/api\/v1\/chats\/([^/]+)\/?$/)
    if (chatPath) {
      const chat = chats.get(decodeURIComponent(chatPath[1]))
      if (!chat) return json({ detail: "Chat not found" }, 404)
      if (method === "GET") return json(chatSummary(chat))
      if (method === "PUT" || method === "PATCH") {
        Object.assign(chat, { title: body?.title ?? chat.title, state: body?.state ?? chat.state })
        chat.version += 1
        return json(chatSummary(chat))
      }
      if (method === "DELETE") {
        chats.delete(chat.id)
        return new Response(null, { status: 204 })
      }
    }
    unhandled.push(request)
    return json({ detail: `Harness: no handler for ${method} ${path}` }, 404)
  }

  const fetchImpl = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(
      typeof input === "string" ? input : input instanceof URL ? input.href : input.url,
      FAKE_SERVER_URL
    )
    const method = String(init?.method ?? (input instanceof Request ? input.method : "GET")).toUpperCase()
    const request: RecordedRequest = {
      method,
      path: url.pathname,
      query: Object.fromEntries(url.searchParams.entries()),
      body: await readBody(input, init)
    }
    requests.push(request)
    if (init?.signal?.aborted) throw new DOMException("The operation was aborted.", "AbortError")
    return route(request, init?.signal)
  }

  return {
    url: FAKE_SERVER_URL,
    requests,
    unhandled,
    chats,
    fetch: fetchImpl,
    /** Every POST /api/v1/chat/completions body, in order. */
    completionRequests: () =>
      requests.filter((r) => r.method === "POST" && r.path === "/api/v1/chat/completions"),
    /** Requests matching a method and a path (string or pattern). */
    find: (method: string, path: string | RegExp) =>
      requests.filter(
        (r) => r.method === method && (typeof path === "string" ? r.path === path : path.test(r.path))
      ),
    setDefaultReply: (reply: string) => {
      defaultReply = reply
    },
    /** Queue the behaviour of the next chat completion. */
    planCompletion: (plan: CompletionPlan) => {
      completionPlans.push(plan)
    },
    /** Seed a saved server conversation (user/assistant turns in order). */
    seedChat: (seed: { id: string; title: string; turns: Array<{ role: "user" | "assistant"; content: string }> }) => {
      const stamp = nowIso()
      const chat: FakeChat = {
        id: seed.id,
        title: seed.title,
        created_at: stamp,
        last_modified: stamp,
        version: 1,
        state: "in-progress",
        source: null,
        messages: []
      }
      seed.turns.forEach((turn, index) => {
        chat.messages.push({
          id: `${seed.id}-m${index + 1}`,
          role: turn.role,
          content: turn.content,
          parent_message_id: index ? `${seed.id}-m${index}` : null,
          created_at: new Date(Date.parse(stamp) + index * 1000).toISOString(),
          version: 1
        })
      })
      chats.set(chat.id, chat)
      return chat
    }
  }
}

export type FakeTldwServer = ReturnType<typeof createFakeTldwServer>
