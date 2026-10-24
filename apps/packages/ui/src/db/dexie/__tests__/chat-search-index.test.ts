import { beforeEach, describe, expect, it, vi } from "vitest"
import type { MemoryDexie } from "@/components/Option/Playground/__tests__/harness/memory-dexie"

/**
 * WP-23 / TASK-13523 — chat content search must not materialize the messages
 * table. The token side-table (`searchTokens`) narrows a content query to
 * candidate histories; only candidate message rows are then read for the exact
 * word-boundary check. Dedupe of title + content matches is O(n) via Set.
 *
 * The production PageAssistDatabase runs against the in-memory Dexie stand-in
 * (jsdom has no IndexedDB and the repo ships no fake-indexeddb).
 */
const memory = vi.hoisted(() => ({ db: null as unknown }))

vi.mock("@/db/dexie/schema", async () => {
  const { createMemoryDexie } = await import(
    "@/components/Option/Playground/__tests__/harness/memory-dexie"
  )
  memory.db = memory.db ?? createMemoryDexie()
  return { db: memory.db }
})

import { pageAssistDatabase } from "../chat"
import { indexMessagesSearchTokens } from "../search-tokens"
import type { HistoryInfo, Message } from "../types"

const db = () => memory.db as MemoryDexie

// Deterministic seed per W0 bench conventions: fixed ids and timestamps.
const HISTORY_COUNT = 100
const MESSAGES_PER_HISTORY = 100
const BASE_TS = 1_700_000_000_000
const CONTENT_NEEDLE = "needleword"
/** Histories whose message content contains the needle (3 also matches by title). */
const CONTENT_NEEDLE_HISTORIES = new Set([3, 7, 42, 99])

const historyId = (index: number) => `hist-${String(index).padStart(3, "0")}`
const messageId = (h: number, m: number) =>
  `msg-${String(h).padStart(3, "0")}-${String(m).padStart(3, "0")}`

const history = (index: number): HistoryInfo => ({
  id: historyId(index),
  title: index === 3 ? "needleword report" : `Conversation ${index}`,
  is_rag: false,
  createdAt: BASE_TS + index
})

const NEEDLE_MESSAGE_INDEX: Record<number, number> = { 3: 10, 7: 50, 42: 0, 99: 99 }

const message = (h: number, m: number): Message => ({
  id: messageId(h, m),
  history_id: historyId(h),
  name: m % 2 === 0 ? "You" : "Assistant",
  role: m % 2 === 0 ? "user" : "assistant",
  content:
    h in NEEDLE_MESSAGE_INDEX && m === NEEDLE_MESSAGE_INDEX[h]
      ? `filler text ${CONTENT_NEEDLE} number ${m}`
      : h === 5 && m === 5
        ? "the alpha beta gamma phrase"
        : h === 6 && m === 1
          ? "alpha standing alone"
          : h === 6 && m === 2
            ? "beta standing alone"
            : h === 10 && m === 4
              ? "bonus x+++y today"
              : `filler text ${h} ${m}`,
  createdAt: BASE_TS + h * MESSAGES_PER_HISTORY + m
})

const seedMessages = () =>
  Array.from({ length: HISTORY_COUNT }, (_, h) =>
    Array.from({ length: MESSAGES_PER_HISTORY }, (_, m) => message(h, m))
  ).flat()

/** Mirrors a database that has been through the v17 token backfill. */
const seed = async () => {
  const messages = seedMessages()
  await db().chatHistories.bulkPut(
    Array.from({ length: HISTORY_COUNT }, (_, index) => history(index))
  )
  await db().messages.bulkPut(messages)
  await indexMessagesSearchTokens(messages)
}

describe("fullTextSearchChatHistories content search (WP-23)", () => {
  beforeEach(() => {
    db().resetAll()
  })

  it("returns only the histories whose messages contain the query, unique and newest first", async () => {
    await seed()
    const results = await pageAssistDatabase.fullTextSearchChatHistories(CONTENT_NEEDLE)
    expect(results.map((item) => item.id)).toEqual([
      "hist-099",
      "hist-042",
      "hist-007",
      "hist-003"
    ])
    expect(new Set(results.map((item) => item.id)).size).toBe(results.length)
  })

  it("matches case-insensitively", async () => {
    await seed()
    const results = await pageAssistDatabase.fullTextSearchChatHistories("NeEdLeWoRd")
    expect(results.map((item) => item.id)).toEqual([
      "hist-099",
      "hist-042",
      "hist-007",
      "hist-003"
    ])
  })

  it("keeps contiguous word-boundary semantics for multi-word queries", async () => {
    await seed()
    // hist-005 contains "alpha beta" contiguously; hist-006 holds the words
    // in separate messages and must not match.
    const results = await pageAssistDatabase.fullTextSearchChatHistories("alpha beta")
    expect(results.map((item) => item.id)).toEqual(["hist-005"])
  })

  it("still returns title prefix matches", async () => {
    await seed()
    await db().chatHistories.put({ ...history(50), title: "Zebras on parade" })
    const results = await pageAssistDatabase.fullTextSearchChatHistories("zebras")
    expect(results.map((item) => item.id)).toEqual(["hist-050"])
  })

  it("does not materialize the messages table during a content search", async () => {
    await seed()
    const messagesToArray = vi.spyOn(db().messages, "toArray")
    const messagesFilter = vi.spyOn(db().messages, "filter")
    const messagesEach = vi.spyOn(db().messages, "each")
    const messagesWhere = vi.spyOn(db().messages, "where")
    const tokensWhere = vi.spyOn(db().searchTokens, "where")

    const results = await pageAssistDatabase.fullTextSearchChatHistories(CONTENT_NEEDLE)

    expect(results.map((item) => item.id)).toEqual([
      "hist-099",
      "hist-042",
      "hist-007",
      "hist-003"
    ])
    expect(messagesToArray).not.toHaveBeenCalled()
    expect(messagesFilter).not.toHaveBeenCalled()
    expect(messagesEach).not.toHaveBeenCalled()
    // Candidate verification goes through the history_id index, not a scan.
    expect(messagesWhere).toHaveBeenCalledWith("history_id")
    // The candidate prefilter consults the token index.
    expect(tokensWhere).toHaveBeenCalledWith("token")
  })

  it("streams punctuation-only queries without materializing the messages table", async () => {
    await seed()
    const messagesToArray = vi.spyOn(db().messages, "toArray")
    // "+++" tokenizes to no word runs, so the token index cannot prefilter;
    // the exact semantics must still hold via a streaming pass.
    const results = await pageAssistDatabase.fullTextSearchChatHistories("+++")
    expect(results.map((item) => item.id)).toEqual(["hist-010"])
    expect(messagesToArray).not.toHaveBeenCalled()
  })
})

describe("message write paths keep the token index fresh", () => {
  beforeEach(() => {
    db().resetAll()
  })

  it("addMessage indexes the new message content", async () => {
    await db().chatHistories.put(history(0))
    await pageAssistDatabase.addMessage({
      ...message(0, 0),
      content: "the quick brown fox"
    })
    const results = await pageAssistDatabase.fullTextSearchChatHistories("quick")
    expect(results.map((item) => item.id)).toEqual(["hist-000"])
  })

  it("updateMessage re-indexes edited content", async () => {
    await db().userSettings.put({ id: "main", user_id: "u", history_profile_id: "P" })
    await db().chatHistories.put({ ...history(1), local_owner_key: "local-history-v1:P" })
    await pageAssistDatabase.addMessage({ ...message(1, 0), content: "old text" })
    expect(
      (await pageAssistDatabase.fullTextSearchChatHistories("old")).map((item) => item.id)
    ).toEqual(["hist-001"])

    await pageAssistDatabase.updateMessage(historyId(1), messageId(1, 0), "brand new text")

    expect(
      (await pageAssistDatabase.fullTextSearchChatHistories("brand")).map((item) => item.id)
    ).toEqual(["hist-001"])
    // The stale token may still make the history a candidate, but exact
    // verification must keep it out of the results.
    expect(
      (await pageAssistDatabase.fullTextSearchChatHistories("old")).map((item) => item.id)
    ).toEqual([])
  })

  it("importChatHistoryV2 indexes imported messages", async () => {
    await pageAssistDatabase.importChatHistoryV2([
      {
        history: { ...history(2), title: "Imported chat" },
        messages: [{ ...message(2, 0), content: "an imported trinket" }]
      }
    ])
    const results = await pageAssistDatabase.fullTextSearchChatHistories("trinket")
    expect(results.map((item) => item.id)).toEqual(["hist-002"])
  })

  it("indexMessagesSearchTokens is idempotent per (token, history) pair", async () => {
    const messages = [message(4, 0), message(4, 1), message(4, 0)]
    await indexMessagesSearchTokens(messages)
    const afterFirst = await db().searchTokens.count()
    await indexMessagesSearchTokens(messages)
    expect(await db().searchTokens.count()).toBe(afterFirst)
  })
})

describe("scoped paginated search still qualifies ownership (characterization)", () => {
  beforeEach(() => {
    db().resetAll()
  })

  it("searches content inside the scope read and then filters foreign owners", async () => {
    await db().userSettings.put({ id: "main", user_id: "u", history_profile_id: "P" })
    await db().chatHistories.bulkPut([
      { ...history(0), local_owner_key: "local-history-v1:P" },
      { ...history(1), local_owner_key: "local-history-v1:Q", title: "Foreign chat" }
    ])
    const messages = [
      { ...message(0, 0), content: "scopedneedle owned body" },
      { ...message(1, 0), content: "scopedneedle foreign body" }
    ]
    await db().messages.bulkPut(messages)
    await indexMessagesSearchTokens(messages)

    const page = await pageAssistDatabase.getChatHistoriesPaginated(1, "scopedneedle", {
      profileId: "P",
      serverScopeKey: "A",
      isCurrent: () => true
    })

    expect(page.histories.map((item) => item.id)).toEqual(["hist-000"])
    expect(page.totalCount).toBe(1)
  })
})
