import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { BoundedTtlCache } from "../bounded-ttl-cache"

// WP-31 / TASK-13523 Stage 4: the shared payload caches are LRU-bounded Maps
// (~100 entries, insertion-order eviction, recency refreshed on re-set) whose
// expired entries are reclaimed by one module-level 60s sweep instead of the
// read path, with an optional per-group (per-chat query shape) cap.

const CAP = 100

const entry = (ttlMs = 60_000) => ({
  value: { marker: `v-${Math.random()}` },
  expiresAt: Date.now() + ttlMs
})

const created: Array<BoundedTtlCache<any>> = []
const makeCache = <T>(options?: ConstructorParameters<typeof BoundedTtlCache<T>>[0]) => {
  const cache = new BoundedTtlCache<T>(options)
  created.push(cache)
  return cache
}

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  // clear() also releases the module-level sweep registration so the next
  // test re-arms the interval on its own fake clock.
  for (const cache of created.splice(0)) cache.clear()
  vi.useRealTimers()
})

it("evicts the oldest entry once the cap is exceeded", () => {
  const cache = makeCache<any>()
  for (let i = 0; i < CAP + 5; i++) cache.set(`k${i}`, entry())
  expect(cache.size).toBe(CAP)
  expect(cache.has("k0")).toBe(false)
  expect(cache.has("k4")).toBe(false)
  expect(cache.has("k5")).toBe(true)
  expect(cache.has(`k${CAP + 4}`)).toBe(true)
})

it("refreshes recency when an existing key is re-set", () => {
  const cache = makeCache<any>()
  for (let i = 0; i < CAP; i++) cache.set(`k${i}`, entry())
  // Re-set the oldest key: it must no longer be the eviction candidate.
  cache.set("k0", entry())
  cache.set("k-new", entry())
  expect(cache.size).toBe(CAP)
  expect(cache.has("k0")).toBe(true)
  expect(cache.has("k1")).toBe(false)
  expect(cache.has("k-new")).toBe(true)
})

it("evicts before insertion so the size never exceeds the cap", () => {
  const cache = makeCache<any>()
  for (let i = 0; i < CAP * 2; i++) {
    cache.set(`k${i}`, entry())
    expect(cache.size).toBeLessThanOrEqual(CAP)
  }
})

it("does not return expired entries even before the sweep fires", () => {
  const cache = makeCache<any>()
  cache.set("stale", entry(-1_000))
  expect(cache.get("stale")).toBeUndefined()
  expect(cache.size).toBe(0)

  cache.set("fresh", entry(5_000))
  expect(cache.get("fresh")).toBeDefined()
  vi.advanceTimersByTime(5_001)
  expect(cache.get("fresh")).toBeUndefined()
})

it("reclaims expired entries on the 60s module sweep while idle", async () => {
  const cache = makeCache<any>()
  cache.set("gone", entry(60_000))
  cache.set("kept", entry(10 * 60_000))
  expect(cache.size).toBe(2)

  await vi.advanceTimersByTimeAsync(60_001)

  expect(cache.has("gone")).toBe(false)
  expect(cache.has("kept")).toBe(true)
})

it("re-arms the sweep after clear()", async () => {
  const first = makeCache<any>()
  first.set("a", entry())
  first.clear()

  const second = makeCache<any>()
  second.set("b", entry(60_000))
  await vi.advanceTimersByTimeAsync(60_001)
  expect(second.has("b")).toBe(false)
})

it("caps entries per group (per-chat query shapes)", () => {
  const cache = makeCache<any>({
    maxPerGroup: 5,
    groupOf: (key) => key.split("?")[0]!
  })
  for (let i = 0; i < 6; i++) cache.set(`chat-1?shape=${i}`, entry())
  cache.set("chat-2?shape=0", entry())

  // chat-1 keeps only its 5 newest query shapes; other chats are unaffected.
  expect([...cache.keys()].filter((key) => key.startsWith("chat-1")).length).toBe(5)
  expect(cache.has("chat-1?shape=0")).toBe(false)
  expect(cache.has("chat-1?shape=1")).toBe(true)
  expect(cache.has("chat-2?shape=0")).toBe(true)
})

it("still enforces the byte budget alongside the entry cap", () => {
  const cache = makeCache<any>()
  cache.set("huge", { value: "x".repeat(5 * 1024 * 1024), expiresAt: Date.now() + 60_000 })
  expect(cache.size).toBe(0)
})
