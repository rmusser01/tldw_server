import { afterEach, describe, expect, it, vi } from "vitest"

import {
  PageAssitDatabase,
  deleteHistoriesByDateRange,
  getPromptById,
  getSessionFiles,
  getSessionFilesInfo
} from ".."

/**
 * WP-18 / TASK-13522 Stage 4 quarantine suite for the legacy
 * chrome.storage blob DB. Pins the Firefox-private-mode fallback surface:
 * batched deleteHistoriesByDateRange, titles-only searchChatHistories, and
 * the legacy getPromptById / session-files shapes that the Dexie
 * characterization suite (db/dexie/__tests__/helpers.repoint-characterization)
 * must mirror.
 */

type HistoryRecord = {
  id: string
  title: string
  is_rag: boolean
  createdAt: number
  is_pinned?: boolean
}

const installLegacyStorage = (seed: Record<string, unknown> = {}) => {
  const store = new Map<string, unknown>(
    Object.entries(structuredClone(seed))
  )
  const calls = {
    getKeys: [] as string[][],
    setKeys: [] as string[][],
    removedKeys: [] as Array<string | string[]>
  }
  const area: any = {
    __calls: calls,
    __store: store,
    get(keys: any, callback: (items: any) => void) {
      const requested = Array.isArray(keys) ? keys : [keys]
      calls.getKeys.push(requested)
      const out: Record<string, unknown> = {}
      for (const key of requested) {
        if (store.has(key)) out[key] = structuredClone(store.get(key))
      }
      setTimeout(() => callback(out), 0)
    },
    set(items: Record<string, unknown>, callback?: () => void) {
      calls.setKeys.push(Object.keys(items))
      Object.entries(items).forEach(([key, value]) =>
        store.set(key, structuredClone(value))
      )
      setTimeout(() => callback?.(), 0)
    },
    remove(keys: any, callback?: () => void) {
      calls.removedKeys.push(keys)
      for (const key of Array.isArray(keys) ? keys : [keys]) {
        store.delete(key)
      }
      setTimeout(() => callback?.(), 0)
    }
  }
  vi.stubGlobal("chrome", { storage: { local: area } })
  return area
}

const history = (
  id: string,
  title: string,
  createdAt: number,
  is_pinned = false
): HistoryRecord => ({
  id,
  title,
  is_rag: false,
  createdAt,
  is_pinned
})

describe("legacy deleteHistoriesByDateRange batching (WP-18)", () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  const seedRanges = () => {
    const now = new Date()
    const todayStart = new Date(now)
    todayStart.setHours(0, 0, 0, 0)
    const minute = 60_000
    const hour = 60 * minute
    const day = 24 * hour
    const todayIds = ["today-1", "today-2", "today-3"]
    const pinnedId = "today-pinned"
    const yesterdayId = "yesterday-1"
    const olderId = "older-1"
    const histories = [
      ...todayIds.map((id, i) =>
        history(id, `Today ${i}`, todayStart.getTime() + (i + 1) * minute)
      ),
      history(
        pinnedId,
        "Pinned today",
        todayStart.getTime() + 10 * minute,
        true
      ),
      history(yesterdayId, "Yesterday", todayStart.getTime() - 3 * hour),
      history(olderId, "Older", todayStart.getTime() - 8 * day)
    ]
    const seed: Record<string, unknown> = { chatHistories: histories }
    for (const h of histories) {
      seed[h.id] = [
        {
          id: `${h.id}-m1`,
          history_id: h.id,
          name: "You",
          role: "user",
          content: `message of ${h.title}`,
          createdAt: h.createdAt
        }
      ]
    }
    return { seed, todayIds, pinnedId, yesterdayId, olderId, histories }
  }

  it("deletes the today range with exactly ONE batched remove and ONE index rewrite", async () => {
    const { seed, todayIds, pinnedId, yesterdayId, olderId } = seedRanges()
    const area = installLegacyStorage(seed)

    const deleted = await deleteHistoriesByDateRange("today")

    expect(deleted.sort()).toEqual([...todayIds].sort())
    // The whole range collapses into one storage remove carrying every id...
    expect(area.__calls.removedKeys).toHaveLength(1)
    expect([...(area.__calls.removedKeys[0] as string[])]).toEqual(
      [...todayIds].sort()
    )
    // ...and one rewrite of the chatHistories index — never per-history.
    expect(area.__calls.setKeys).toHaveLength(1)
    expect(area.__calls.setKeys[0]).toEqual(["chatHistories"])

    const remaining = area.__store.get("chatHistories") as HistoryRecord[]
    expect(remaining.map((h) => h.id).sort()).toEqual(
      [pinnedId, yesterdayId, olderId].sort()
    )
    for (const id of todayIds) {
      expect(area.__store.has(id)).toBe(false)
    }
    expect(area.__store.has(pinnedId)).toBe(true)
  })

  it("keeps the batched shape for other ranges (pinned band)", async () => {
    const { seed, pinnedId } = seedRanges()
    const area = installLegacyStorage(seed)

    const deleted = await deleteHistoriesByDateRange("pinned")

    expect(deleted).toEqual([pinnedId])
    expect(area.__calls.removedKeys).toHaveLength(1)
    expect(area.__calls.removedKeys[0]).toEqual([pinnedId])
    expect(area.__calls.setKeys).toHaveLength(1)
  })

  it("writes nothing when the range selects no histories", async () => {
    const { seed } = seedRanges()
    const area = installLegacyStorage(seed)

    expect(await deleteHistoriesByDateRange("last7Days")).toEqual([])
    expect(area.__calls.removedKeys).toHaveLength(0)
    expect(area.__calls.setKeys).toHaveLength(0)

    expect(await deleteHistoriesByDateRange("bogus-range")).toEqual([])
    expect(area.__calls.removedKeys).toHaveLength(0)
    expect(area.__calls.setKeys).toHaveLength(0)
  })
})

describe("legacy searchChatHistories is titles-only (WP-18)", () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  const seedSearch = () => {
    const histories = [
      history("alpha", "Quarterly Report", 2),
      history("beta", "Beta Chat", 1)
    ]
    const seed: Record<string, unknown> = {
      chatHistories: histories,
      // Content that would match a message-body search, but no title match.
      alpha: [
        {
          id: "alpha-m1",
          history_id: "alpha",
          name: "You",
          role: "user",
          content: "totally unrelated body text",
          createdAt: 3
        }
      ],
      beta: [
        {
          id: "beta-m1",
          history_id: "beta",
          name: "You",
          role: "user",
          content: "the secret sauce recipe lives here",
          createdAt: 4
        }
      ]
    }
    return { histories, seed }
  }

  it("matches titles and never deserializes message blobs", async () => {
    const { seed } = seedSearch()
    const area = installLegacyStorage(seed)
    const db = new PageAssitDatabase()
    const deserialize = vi.spyOn(db, "getChatHistory")

    const matched = await db.searchChatHistories("report")

    expect(matched.map((h) => h.id)).toEqual(["alpha"])
    // Titles-only: no per-history message reads were issued at all.
    expect(deserialize).not.toHaveBeenCalled()
    expect(area.__calls.getKeys).toEqual([["chatHistories"]])
  })

  it("no longer returns message-content-only matches", async () => {
    const { seed } = seedSearch()
    installLegacyStorage(seed)
    const db = new PageAssitDatabase()
    const deserialize = vi.spyOn(db, "getChatHistory")

    const matched = await db.searchChatHistories("secret sauce recipe")

    expect(matched).toEqual([])
    expect(deserialize).not.toHaveBeenCalled()
  })

  it("returns every history for an empty query without deserializing", async () => {
    const { histories, seed } = seedSearch()
    installLegacyStorage(seed)
    const db = new PageAssitDatabase()
    const deserialize = vi.spyOn(db, "getChatHistory")

    expect(await db.searchChatHistories("   ")).toEqual(histories)
    expect(deserialize).not.toHaveBeenCalled()
  })
})

describe("legacy getPromptById oracle (characterization reference)", () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  const prompt = (id: string, extra: Record<string, unknown> = {}) => ({
    id,
    title: `Title ${id}`,
    name: `Title ${id}`,
    content: `Content ${id}`,
    is_system: false,
    keywords: ["k"],
    createdAt: 1234,
    ...extra
  })

  it("returns the stored record for an existing prompt", async () => {
    const existing = prompt("p1")
    installLegacyStorage({ prompts: [existing, prompt("p2")] })

    await expect(getPromptById("p1")).resolves.toEqual(existing)
  })

  it("returns undefined for a missing prompt and null for an empty id", async () => {
    installLegacyStorage({ prompts: [prompt("p1")] })

    await expect(getPromptById("nope")).resolves.toBeUndefined()
    await expect(getPromptById("")).resolves.toBeNull()
    await expect(getPromptById("   ")).resolves.toBeNull()
  })

  it("still surfaces soft-deleted prompts (deletedAt is visible)", async () => {
    const softDeleted = prompt("p1", { deletedAt: 9999 })
    installLegacyStorage({ prompts: [softDeleted] })

    await expect(getPromptById("p1")).resolves.toEqual(softDeleted)
  })
})

describe("legacy session-files shape oracle (characterization reference)", () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  const file = {
    id: "f1",
    filename: "a.txt",
    type: "text/plain",
    content: "hello",
    size: 5,
    uploadedAt: 10,
    processed: true
  }

  it("returns files for a known session and [] for an unknown one", async () => {
    installLegacyStorage({
      "session_files_s1": {
        sessionId: "s1",
        files: [file],
        retrievalEnabled: true,
        createdAt: 1
      }
    })

    await expect(getSessionFiles("s1")).resolves.toEqual([file])
    await expect(getSessionFiles("missing")).resolves.toEqual([])
    await expect(getSessionFilesInfo("missing")).resolves.toBeNull()
  })
})
