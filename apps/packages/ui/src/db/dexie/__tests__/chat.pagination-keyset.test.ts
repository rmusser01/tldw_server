import { beforeEach, describe, expect, it, vi } from "vitest"
import type { MemoryDexie } from "@/components/Option/Playground/__tests__/harness/memory-dexie"

/**
 * WP-27 / TASK-13523 — the sidebar's history list must use keyset
 * pagination (getChatHistoriesPaginatedOptimized) instead of offset scans.
 *
 * Outcome tests were captured against the keyset implementation as it exists
 * at HEAD (dead code: the sidebar still used the offset variant). The search
 * continuation test pins the NEW contract: page 2 of a search must continue
 * after the previous page's last row instead of re-serving page 1.
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
import type { HistoryInfo } from "../types"

const db = () => memory.db as MemoryDexie
const PAGE_SIZE = 30
const BASE_TS = 1_700_000_000_000

const history = (index: number, title = `Conversation ${index}`): HistoryInfo => ({
  id: `hist-${String(index).padStart(3, "0")}`,
  title,
  is_rag: false,
  createdAt: BASE_TS + index
})

const seed = (count: number) =>
  db().chatHistories.bulkPut(Array.from({ length: count }, (_, index) => history(index)))

const ids = (rows: HistoryInfo[]) => rows.map(row => row.id)

describe("keyset history pagination (WP-27, getChatHistoriesPaginatedOptimized)", () => {
  beforeEach(() => {
    db().resetAll()
  })

  it("serves the newest PAGE_SIZE histories first and reports hasMore", async () => {
    seed(70)
    const page = await pageAssistDatabase.getChatHistoriesPaginatedOptimized()
    expect(page.histories).toHaveLength(PAGE_SIZE)
    expect(page.hasMore).toBe(true)
    expect(ids(page.histories)).toEqual(
      Array.from({ length: PAGE_SIZE }, (_, i) => `hist-${String(69 - i).padStart(3, "0")}`)
    )
  })

  it("continues after the last entry of the previous page without duplicates or skips", async () => {
    seed(70)
    const first = await pageAssistDatabase.getChatHistoriesPaginatedOptimized()
    const second = await pageAssistDatabase.getChatHistoriesPaginatedOptimized(first.histories.at(-1))
    const third = await pageAssistDatabase.getChatHistoriesPaginatedOptimized(second.histories.at(-1))

    expect(second.histories).toHaveLength(PAGE_SIZE)
    expect(third.histories).toHaveLength(10)
    expect(third.hasMore).toBe(false)

    const all = [...first.histories, ...second.histories, ...third.histories]
    expect(new Set(all.map(row => row.id)).size).toBe(70)
    expect(ids(all)).toEqual(
      Array.from({ length: 70 }, (_, i) => `hist-${String(69 - i).padStart(3, "0")}`)
    )
  })

  it("stays stable across inserts: a newer history never duplicates or skips earlier rows", async () => {
    seed(70)
    const first = await pageAssistDatabase.getChatHistoriesPaginatedOptimized()
    // Insert a brand-new history (and one mid-history timestamp) between page fetches.
    db().chatHistories.put(history(500, "brand new"))
    const second = await pageAssistDatabase.getChatHistoriesPaginatedOptimized(first.histories.at(-1))
    const third = await pageAssistDatabase.getChatHistoriesPaginatedOptimized(second.histories.at(-1))

    const all = [...first.histories, ...second.histories, ...third.histories]
    // No duplicates, and every pre-existing history appears exactly once
    // (the new row is newer than the whole first page, so it is not expected
    // until a fresh first page is fetched).
    expect(new Set(all.map(row => row.id)).size).toBe(all.length)
    const expected = Array.from({ length: 70 }, (_, i) => `hist-${String(69 - i).padStart(3, "0")}`)
    expect(ids(all)).toEqual(expected)
  })

  it("reports hasMore false when the table holds exactly one page", async () => {
    seed(PAGE_SIZE)
    const page = await pageAssistDatabase.getChatHistoriesPaginatedOptimized()
    expect(page.histories).toHaveLength(PAGE_SIZE)
    expect(page.hasMore).toBe(false)
  })

  it("serves the first page of a search and continues past it without duplicates (new contract)", async () => {
    seed(70)
    const first = await pageAssistDatabase.getChatHistoriesPaginatedOptimized(undefined, "Conversation")
    expect(first.histories).toHaveLength(PAGE_SIZE)
    expect(first.hasMore).toBe(true)

    const second = await pageAssistDatabase.getChatHistoriesPaginatedOptimized(first.histories.at(-1), "Conversation")
    expect(second.histories).toHaveLength(PAGE_SIZE)

    const third = await pageAssistDatabase.getChatHistoriesPaginatedOptimized(second.histories.at(-1), "Conversation")
    expect(third.histories).toHaveLength(10)
    expect(third.hasMore).toBe(false)

    const all = [...first.histories, ...second.histories, ...third.histories]
    expect(new Set(all.map(row => row.id)).size).toBe(70)
  })

  it("returns an empty page when the search cursor no longer matches any result", async () => {
    seed(70)
    const first = await pageAssistDatabase.getChatHistoriesPaginatedOptimized(undefined, "Conversation")
    // The row the cursor points at was deleted between pages.
    db().chatHistories.delete(first.histories.at(-1)!.id)
    const second = await pageAssistDatabase.getChatHistoriesPaginatedOptimized(first.histories.at(-1), "Conversation")
    expect(second.histories).toEqual([])
    expect(second.hasMore).toBe(false)
  })

  it("serves the exact next-30 rows for a keyset cursor (no offset drift)", async () => {
    seed(70)
    const first = await pageAssistDatabase.getChatHistoriesPaginatedOptimized()
    const second = await pageAssistDatabase.getChatHistoriesPaginatedOptimized(first.histories.at(-1))
    expect(ids(second.histories)).toEqual(
      Array.from({ length: PAGE_SIZE }, (_, i) => `hist-${String(39 - i).padStart(3, "0")}`)
    )
    expect(second.hasMore).toBe(true)
  })
})
