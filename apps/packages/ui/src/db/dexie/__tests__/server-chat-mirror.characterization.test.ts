import { beforeEach, describe, expect, it, vi } from "vitest"

/**
 * Stage 2 (WP-24 / TASK-13523) — characterization golden for the server-chat
 * mirror reconcile, captured from the pre-Maps/bulkPut implementation and
 * frozen byte-identical across the refactor.
 *
 * The fixture interleaves server-acknowledged rows with local-only work:
 * an unsynced draft, a client-correlated user (client_message_id claim), an
 * anchored legacy user (canonical parent recovery), a locally-edited
 * acknowledged row (version-guarded content preservation), a captured live
 * message (adoption without a durable row), a fresh assistant reply
 * (synthetic qualified id), and a greeting-shaped absent remote on the
 * follow-up call (local row must survive an absent server row).
 *
 * Dimensions pinned by the golden:
 * - message rows and their primary-key ordering: a Dexie index scan over
 *   `history_id` orders equal index keys by primary key, and the memory
 *   harness implements the same ordering — insertion order here is
 *   deliberately scrambled against primary-key order;
 * - server-id stamping through both recovery paths;
 * - the token side table (Stage 1): fresh tokens for every mirror-written
 *   row, stale tokens retained as harmless extras after a remote-wins edit;
 * - the steady state of a repeated reconcile and the evolution on a third
 *   call (remote-wins edit, addition, absence).
 *
 * Regenerate with CAPTURE_GOLDEN=1 — then verify the implementation still
 * produces exactly these bytes before committing any change here.
 */
const memory = vi.hoisted(() => ({ db: null as unknown }))

vi.mock("@/db/dexie/schema", async () => {
  const { createMemoryDexie } = await import(
    "@/components/Option/Playground/__tests__/harness/memory-dexie"
  )
  memory.db = memory.db ?? createMemoryDexie()
  return { db: memory.db }
})

import type { MemoryDexie } from "@/components/Option/Playground/__tests__/harness/memory-dexie"
import type { Message as ChatMessage } from "@/store/option"
import { reconcileServerChatMirror, reconcileServerChatMessages } from "../server-chat-mirror"
import { tokenizeSearchText } from "../search-tokenizer"
import type { Message } from "../types"

const db = () => memory.db as MemoryDexie

const HISTORY = "h1"
const KEEP_IMAGE = "data:image/png;base64,characterization"

const incoming = (over: Partial<ChatMessage> & Pick<ChatMessage, "id" | "message" | "isBot" | "role">): ChatMessage => ({
  name: over.isBot ? "Assistant" : "You",
  sources: [],
  images: [],
  createdAt: 1,
  ...over
})

/** The server snapshot for the first (and identical second) reconcile. */
const firstIncoming = (): ChatMessage[] => [
  incoming({ id: "u1", isBot: false, role: "user", message: "Repeat alpha", images: [""], createdAt: 1000, serverMessageId: "u1" }),
  // Unnamed remotes pin the role-appropriate fallback names.
  incoming({ id: "a1", isBot: true, role: "assistant", name: "", message: "Server answer one", createdAt: 1100, serverMessageId: "a1", parentMessageId: "u1" }),
  incoming({ id: "u2", isBot: false, role: "user", name: "", message: "Correlated beta", images: [KEEP_IMAGE], createdAt: 1200, serverMessageId: "u2", metadataExtra: { client_message_id: "mid-q" } }),
  incoming({ id: "q2", isBot: false, role: "user", message: "Server second question v3", createdAt: 1300, serverMessageId: "q2", serverMessageVersion: 3 }),
  incoming({ id: "a2", isBot: true, role: "assistant", name: "", message: "Server answer two words", createdAt: 1400, serverMessageId: "a2", parentMessageId: "q2" }),
  incoming({ id: "u3", isBot: false, role: "user", message: "Captured gamma text", createdAt: 1500, serverMessageId: "u3", serverMessageVersion: 2 })
]

/** Third call: q2 wins remotely (version 9), u4 is new, a1 is absent. */
const thirdIncoming = (): ChatMessage[] => [
  ...firstIncoming().filter(message => message.id !== "q2" && message.id !== "a1"),
  incoming({ id: "q2", isBot: false, role: "user", message: "Server second question v9", createdAt: 1300, serverMessageId: "q2", serverMessageVersion: 9 }),
  incoming({ id: "u4", isBot: false, role: "user", message: "Fresh delta words", createdAt: 1600, serverMessageId: "u4" })
]

/** Stored rows in an order deliberately scrambled against primary-key order. */
const seedRows = (): Message[] => [
  { id: "z-draft", history_id: HISTORY, name: "You", role: "user", content: "Unsent zebra draft", images: [], createdAt: 500 },
  { id: "mid-q", history_id: HISTORY, name: "You", role: "user", content: "Correlated beta", images: [KEEP_IMAGE], createdAt: 600 },
  { id: "q2-row", history_id: HISTORY, name: "You", role: "user", content: "Locally edited second question", images: [], createdAt: 700, serverMessageId: "q2", serverMessageVersion: 7 },
  { id: "local-a", history_id: HISTORY, name: "Assistant", role: "assistant", content: "Local answer text", images: [], createdAt: 800, serverMessageId: "a1", parent_message_id: "legacy-u" },
  { id: "legacy-u", history_id: HISTORY, name: "You", role: "user", content: "Repeat alpha", images: [""], createdAt: 400 }
]

/** Live in-memory store at call time: the draft plus an unacknowledged capture. */
const liveMessages = (): ChatMessage[] => [
  incoming({ id: "z-draft", isBot: false, role: "user", message: "Unsent zebra draft", createdAt: 500 }),
  incoming({ id: "live-gamma", isBot: false, role: "user", message: "Captured gamma text", createdAt: 1500, serverMessageId: "u3", serverMessageVersion: 2 })
]

const seed = async () => {
  await db().chatHistories.put({ id: HISTORY, title: "Characterization", createdAt: 1, is_rag: false, server_chat_id: "chat-1", server_scope_key: "A" })
  await db().messages.bulkPut(seedRows())
}

const reconcile = (messages: ChatMessage[]) =>
  reconcileServerChatMirror({ historyId: HISTORY, chatId: "chat-1", ownerKey: "A", messages, localMessages: liveMessages() })

/** Deterministic serialization: object keys sorted, Maps as entry pairs. */
const canonicalJson = (value: unknown): string => JSON.stringify(sortValue(value))
const sortValue = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(sortValue)
  if (value instanceof Map) return sortValue([...value.entries()])
  if (value && typeof value === "object") {
    const source = value as Record<string, unknown>
    return Object.fromEntries(Object.keys(source).sort().map(key => [key, sortValue(source[key])]))
  }
  return value
}

describe("server-chat mirror reconcile characterization", () => {
  beforeEach(() => {
    db().resetAll()
    vi.spyOn(Date, "now").mockReturnValue(1_700_000_000_000)
  })

  it("produces the frozen golden outcome across interleaved server/local work", async () => {
    await seed()
    const first = await reconcile(firstIncoming())
    const second = await reconcile(firstIncoming())
    const third = await reconcile(thirdIncoming())
    const outcome = {
      firstRows: first.rows,
      firstLocalIds: first.localIds,
      secondRows: second.rows,
      secondLocalIds: second.localIds,
      thirdRows: third.rows,
      thirdLocalIds: third.localIds,
      storedRows: await db().messages.where("history_id").equals(HISTORY).toArray(),
      tokens: await db().searchTokens.toArray()
    }
    if (process.env.CAPTURE_GOLDEN) console.log(`GOLDEN_START${canonicalJson(outcome)}GOLDEN_END`)
    expect(canonicalJson(outcome)).toBe("{\"firstLocalIds\":[[\"u1\",\"legacy-u\"],[\"a1\",\"local-a\"],[\"u2\",\"mid-q\"],[\"q2\",\"q2-row\"],[\"a2\",\"h1:server:a2\"],[\"u3\",\"live-gamma\"]],\"firstRows\":[{\"content\":\"Server answer two words\",\"createdAt\":1400,\"history_id\":\"h1\",\"id\":\"h1:server:a2\",\"images\":[],\"name\":\"Assistant\",\"parent_message_id\":\"q2\",\"role\":\"assistant\",\"serverMessageId\":\"a2\",\"sources\":[]},{\"content\":\"Repeat alpha\",\"createdAt\":1000,\"history_id\":\"h1\",\"id\":\"legacy-u\",\"images\":[\"\"],\"name\":\"You\",\"parent_message_id\":null,\"role\":\"user\",\"serverMessageId\":\"u1\",\"sources\":[]},{\"content\":\"Captured gamma text\",\"createdAt\":1500,\"history_id\":\"h1\",\"id\":\"live-gamma\",\"images\":[],\"name\":\"You\",\"parent_message_id\":null,\"role\":\"user\",\"serverMessageId\":\"u3\",\"serverMessageVersion\":2,\"sources\":[]},{\"content\":\"Local answer text\",\"createdAt\":1100,\"history_id\":\"h1\",\"id\":\"local-a\",\"images\":[],\"name\":\"Assistant\",\"parent_message_id\":\"legacy-u\",\"role\":\"assistant\",\"serverMessageId\":\"a1\",\"sources\":[]},{\"content\":\"Correlated beta\",\"createdAt\":1200,\"history_id\":\"h1\",\"id\":\"mid-q\",\"images\":[\"data:image/png;base64,characterization\"],\"metadataExtra\":{\"client_message_id\":\"mid-q\"},\"name\":\"You\",\"parent_message_id\":null,\"role\":\"user\",\"serverMessageId\":\"u2\",\"sources\":[]},{\"content\":\"Locally edited second question\",\"createdAt\":1300,\"history_id\":\"h1\",\"id\":\"q2-row\",\"images\":[],\"name\":\"You\",\"parent_message_id\":null,\"role\":\"user\",\"serverMessageId\":\"q2\",\"serverMessageVersion\":7,\"sources\":[]},{\"content\":\"Unsent zebra draft\",\"createdAt\":500,\"history_id\":\"h1\",\"id\":\"z-draft\",\"images\":[],\"name\":\"You\",\"role\":\"user\"}],\"secondLocalIds\":[[\"u1\",\"legacy-u\"],[\"a1\",\"local-a\"],[\"u2\",\"mid-q\"],[\"q2\",\"q2-row\"],[\"a2\",\"h1:server:a2\"],[\"u3\",\"live-gamma\"]],\"secondRows\":[{\"content\":\"Server answer two words\",\"createdAt\":1400,\"history_id\":\"h1\",\"id\":\"h1:server:a2\",\"images\":[],\"name\":\"Assistant\",\"parent_message_id\":\"q2\",\"role\":\"assistant\",\"serverMessageId\":\"a2\",\"sources\":[]},{\"content\":\"Repeat alpha\",\"createdAt\":1000,\"history_id\":\"h1\",\"id\":\"legacy-u\",\"images\":[\"\"],\"name\":\"You\",\"parent_message_id\":null,\"role\":\"user\",\"serverMessageId\":\"u1\",\"sources\":[]},{\"content\":\"Captured gamma text\",\"createdAt\":1500,\"history_id\":\"h1\",\"id\":\"live-gamma\",\"images\":[],\"name\":\"You\",\"parent_message_id\":null,\"role\":\"user\",\"serverMessageId\":\"u3\",\"serverMessageVersion\":2,\"sources\":[]},{\"content\":\"Local answer text\",\"createdAt\":1100,\"history_id\":\"h1\",\"id\":\"local-a\",\"images\":[],\"name\":\"Assistant\",\"parent_message_id\":\"legacy-u\",\"role\":\"assistant\",\"serverMessageId\":\"a1\",\"sources\":[]},{\"content\":\"Correlated beta\",\"createdAt\":1200,\"history_id\":\"h1\",\"id\":\"mid-q\",\"images\":[\"data:image/png;base64,characterization\"],\"metadataExtra\":{\"client_message_id\":\"mid-q\"},\"name\":\"You\",\"parent_message_id\":null,\"role\":\"user\",\"serverMessageId\":\"u2\",\"sources\":[]},{\"content\":\"Locally edited second question\",\"createdAt\":1300,\"history_id\":\"h1\",\"id\":\"q2-row\",\"images\":[],\"name\":\"You\",\"parent_message_id\":null,\"role\":\"user\",\"serverMessageId\":\"q2\",\"serverMessageVersion\":7,\"sources\":[]},{\"content\":\"Unsent zebra draft\",\"createdAt\":500,\"history_id\":\"h1\",\"id\":\"z-draft\",\"images\":[],\"name\":\"You\",\"role\":\"user\"}],\"storedRows\":[{\"content\":\"Server answer two words\",\"createdAt\":1400,\"history_id\":\"h1\",\"id\":\"h1:server:a2\",\"images\":[],\"name\":\"Assistant\",\"parent_message_id\":\"q2\",\"role\":\"assistant\",\"serverMessageId\":\"a2\",\"sources\":[]},{\"content\":\"Fresh delta words\",\"createdAt\":1600,\"history_id\":\"h1\",\"id\":\"h1:server:u4\",\"images\":[],\"name\":\"You\",\"parent_message_id\":null,\"role\":\"user\",\"serverMessageId\":\"u4\",\"sources\":[]},{\"content\":\"Repeat alpha\",\"createdAt\":1000,\"history_id\":\"h1\",\"id\":\"legacy-u\",\"images\":[\"\"],\"name\":\"You\",\"parent_message_id\":null,\"role\":\"user\",\"serverMessageId\":\"u1\",\"sources\":[]},{\"content\":\"Captured gamma text\",\"createdAt\":1500,\"history_id\":\"h1\",\"id\":\"live-gamma\",\"images\":[],\"name\":\"You\",\"parent_message_id\":null,\"role\":\"user\",\"serverMessageId\":\"u3\",\"serverMessageVersion\":2,\"sources\":[]},{\"content\":\"Local answer text\",\"createdAt\":1100,\"history_id\":\"h1\",\"id\":\"local-a\",\"images\":[],\"name\":\"Assistant\",\"parent_message_id\":\"legacy-u\",\"role\":\"assistant\",\"serverMessageId\":\"a1\",\"sources\":[]},{\"content\":\"Correlated beta\",\"createdAt\":1200,\"history_id\":\"h1\",\"id\":\"mid-q\",\"images\":[\"data:image/png;base64,characterization\"],\"metadataExtra\":{\"client_message_id\":\"mid-q\"},\"name\":\"You\",\"parent_message_id\":null,\"role\":\"user\",\"serverMessageId\":\"u2\",\"sources\":[]},{\"content\":\"Server second question v9\",\"createdAt\":1300,\"history_id\":\"h1\",\"id\":\"q2-row\",\"images\":[],\"name\":\"You\",\"parent_message_id\":null,\"role\":\"user\",\"serverMessageId\":\"q2\",\"serverMessageVersion\":9,\"sources\":[]},{\"content\":\"Unsent zebra draft\",\"createdAt\":500,\"history_id\":\"h1\",\"id\":\"z-draft\",\"images\":[],\"name\":\"You\",\"role\":\"user\"}],\"thirdLocalIds\":[[\"u1\",\"legacy-u\"],[\"u2\",\"mid-q\"],[\"a2\",\"h1:server:a2\"],[\"u3\",\"live-gamma\"],[\"q2\",\"q2-row\"],[\"u4\",\"h1:server:u4\"]],\"thirdRows\":[{\"content\":\"Server answer two words\",\"createdAt\":1400,\"history_id\":\"h1\",\"id\":\"h1:server:a2\",\"images\":[],\"name\":\"Assistant\",\"parent_message_id\":\"q2\",\"role\":\"assistant\",\"serverMessageId\":\"a2\",\"sources\":[]},{\"content\":\"Fresh delta words\",\"createdAt\":1600,\"history_id\":\"h1\",\"id\":\"h1:server:u4\",\"images\":[],\"name\":\"You\",\"parent_message_id\":null,\"role\":\"user\",\"serverMessageId\":\"u4\",\"sources\":[]},{\"content\":\"Repeat alpha\",\"createdAt\":1000,\"history_id\":\"h1\",\"id\":\"legacy-u\",\"images\":[\"\"],\"name\":\"You\",\"parent_message_id\":null,\"role\":\"user\",\"serverMessageId\":\"u1\",\"sources\":[]},{\"content\":\"Captured gamma text\",\"createdAt\":1500,\"history_id\":\"h1\",\"id\":\"live-gamma\",\"images\":[],\"name\":\"You\",\"parent_message_id\":null,\"role\":\"user\",\"serverMessageId\":\"u3\",\"serverMessageVersion\":2,\"sources\":[]},{\"content\":\"Local answer text\",\"createdAt\":1100,\"history_id\":\"h1\",\"id\":\"local-a\",\"images\":[],\"name\":\"Assistant\",\"parent_message_id\":\"legacy-u\",\"role\":\"assistant\",\"serverMessageId\":\"a1\",\"sources\":[]},{\"content\":\"Correlated beta\",\"createdAt\":1200,\"history_id\":\"h1\",\"id\":\"mid-q\",\"images\":[\"data:image/png;base64,characterization\"],\"metadataExtra\":{\"client_message_id\":\"mid-q\"},\"name\":\"You\",\"parent_message_id\":null,\"role\":\"user\",\"serverMessageId\":\"u2\",\"sources\":[]},{\"content\":\"Server second question v9\",\"createdAt\":1300,\"history_id\":\"h1\",\"id\":\"q2-row\",\"images\":[],\"name\":\"You\",\"parent_message_id\":null,\"role\":\"user\",\"serverMessageId\":\"q2\",\"serverMessageVersion\":9,\"sources\":[]},{\"content\":\"Unsent zebra draft\",\"createdAt\":500,\"history_id\":\"h1\",\"id\":\"z-draft\",\"images\":[],\"name\":\"You\",\"role\":\"user\"}],\"tokens\":[{\"history_id\":\"h1\",\"token\":\"alpha\"},{\"history_id\":\"h1\",\"token\":\"answer\"},{\"history_id\":\"h1\",\"token\":\"beta\"},{\"history_id\":\"h1\",\"token\":\"captured\"},{\"history_id\":\"h1\",\"token\":\"correlated\"},{\"history_id\":\"h1\",\"token\":\"delta\"},{\"history_id\":\"h1\",\"token\":\"edited\"},{\"history_id\":\"h1\",\"token\":\"fresh\"},{\"history_id\":\"h1\",\"token\":\"gamma\"},{\"history_id\":\"h1\",\"token\":\"local\"},{\"history_id\":\"h1\",\"token\":\"locally\"},{\"history_id\":\"h1\",\"token\":\"question\"},{\"history_id\":\"h1\",\"token\":\"repeat\"},{\"history_id\":\"h1\",\"token\":\"second\"},{\"history_id\":\"h1\",\"token\":\"server\"},{\"history_id\":\"h1\",\"token\":\"text\"},{\"history_id\":\"h1\",\"token\":\"two\"},{\"history_id\":\"h1\",\"token\":\"v9\"},{\"history_id\":\"h1\",\"token\":\"words\"}]}")
  })

  it("indexes every word of every mirror-written row, keeping stale extras (Stage 1 contract)", async () => {
    await seed()
    const first = await reconcile(firstIncoming())
    const third = await reconcile(thirdIncoming())
    const pairs = new Set((await db().searchTokens.toArray()).map(row => `${row.token}|${row.history_id}`))
    expect(pairs.size).toBeGreaterThan(0)
    for (const localId of [...first.localIds.values(), ...third.localIds.values()]) {
      const row = await db().messages.get(localId)
      expect(row, `mirror-written row ${localId}`).toBeDefined()
      for (const token of tokenizeSearchText(row!.content)) expect(pairs.has(`${token}|${HISTORY}`), `token ${token}`).toBe(true)
    }
    // The remote-wins q2 edit leaves the pre-edit words as harmless extras.
    expect(pairs.has(`edited|${HISTORY}`)).toBe(true)
    // The untouched local draft is never written by the mirror, so it may stay unindexed.
    const draftTokens = tokenizeSearchText("Unsent zebra draft")
    expect(draftTokens.every(token => pairs.has(`${token}|${HISTORY}`))).toBe(false)
  })

  it("characterizes the pure reconcile stamping and merge over the interleaved store", () => {
    const current: ChatMessage[] = [
      incoming({ id: "z-draft", isBot: false, role: "user", message: "Unsent zebra draft", createdAt: 500 }),
      incoming({ id: "mid-q", isBot: false, role: "user", message: "Correlated beta", images: [KEEP_IMAGE], createdAt: 600 }),
      incoming({ id: "local-a", isBot: true, role: "assistant", message: "Local answer text", createdAt: 800, serverMessageId: "a1", parentMessageId: "legacy-u" }),
      incoming({ id: "legacy-u", isBot: false, role: "user", message: "Repeat alpha", images: [""], createdAt: 400 }),
      incoming({ id: "q2-row", isBot: false, role: "user", message: "Locally edited second question", createdAt: 700, serverMessageId: "q2", serverMessageVersion: 7 })
    ]
    const merged = reconcileServerChatMessages(current, firstIncoming())
    const outcome = merged.map(message => ({ id: message.id, serverMessageId: message.serverMessageId ?? null, message: message.message, parent: message.parentMessageId ?? null }))
    if (process.env.CAPTURE_GOLDEN) console.log(`PURE_START${canonicalJson(outcome)}PURE_END`)
    expect(canonicalJson(outcome)).toBe("[{\"id\":\"legacy-u\",\"message\":\"Repeat alpha\",\"parent\":null,\"serverMessageId\":\"u1\"},{\"id\":\"local-a\",\"message\":\"Local answer text\",\"parent\":\"u1\",\"serverMessageId\":\"a1\"},{\"id\":\"mid-q\",\"message\":\"Correlated beta\",\"parent\":null,\"serverMessageId\":\"u2\"},{\"id\":\"q2-row\",\"message\":\"Locally edited second question\",\"parent\":null,\"serverMessageId\":\"q2\"},{\"id\":\"a2\",\"message\":\"Server answer two words\",\"parent\":\"q2\",\"serverMessageId\":\"a2\"},{\"id\":\"u3\",\"message\":\"Captured gamma text\",\"parent\":null,\"serverMessageId\":\"u3\"},{\"id\":\"z-draft\",\"message\":\"Unsent zebra draft\",\"parent\":null,\"serverMessageId\":null}]")
  })
})

describe("server-chat mirror batched writes (WP-24)", () => {
  beforeEach(() => {
    db().resetAll()
    vi.spyOn(Date, "now").mockReturnValue(1_700_000_000_000)
  })

  it("persists the planned rows through one bulkGet and one bulkPut, never per-message writes", async () => {
    await seed()
    const bulkGet = vi.spyOn(db().messages, "bulkGet")
    const put = vi.spyOn(db().messages, "put")
    const add = vi.spyOn(db().messages, "add")
    const bulkPut = vi.spyOn(db().messages, "bulkPut")
    const tokenBulkPut = vi.spyOn(db().searchTokens, "bulkPut")
    const result = await reconcile(firstIncoming())
    expect(bulkGet).toHaveBeenCalledTimes(1)
    expect(bulkPut).toHaveBeenCalledTimes(1)
    expect(put).not.toHaveBeenCalled()
    expect(add).not.toHaveBeenCalled()
    expect(tokenBulkPut).toHaveBeenCalledTimes(1)
    // One batched write carrying every planned row, in incoming order.
    expect(bulkPut.mock.calls[0][0].map((row: Message) => row.id)).toEqual(
      ["legacy-u", "local-a", "mid-q", "q2-row", "h1:server:a2", "live-gamma"])
    expect([...result.localIds.values()]).toEqual(
      ["legacy-u", "local-a", "mid-q", "q2-row", "h1:server:a2", "live-gamma"])
  })
})
