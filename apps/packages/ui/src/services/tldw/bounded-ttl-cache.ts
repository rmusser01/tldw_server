type Entry<T> = { value: T; expiresAt: number }

export type BoundedTtlCacheOptions = {
  /** Entry cap per cache (~100 per the perf plan; TASK-13523 Stage 4). */
  maxEntries?: number
  /** Optional per-group cap, e.g. ~5 query shapes per chat. */
  maxPerGroup?: number
  /** Group extractor for the per-group cap; keys without a group are ungrouped. */
  groupOf?: (key: string) => string
}

const MAX_ENTRIES = 100
// Conservative UTF-16 payload budget retained from TASK-13450; it bounds
// memory below the entry cap when a few responses are very large.
const MAX_BYTES = 8 * 1024 * 1024
const SWEEP_INTERVAL_MS = 60 * 1000

// One module-level sweep for every live cache (WP-31): expired entries are
// reclaimed off the read path by a single 60s interval instead of per-entry
// timers or per-get sweeps. TldwApiClient is a long-lived singleton, so the
// interval is intentionally never torn down in production; it is guarded
// against multiple registrations and released whenever no cache holds
// entries (cache.clear() and an empty sweep both disarm it).
const liveCaches = new Set<BoundedTtlCache<any>>()
let sweepTimer: ReturnType<typeof setInterval> | null = null

function ensureSweepTimer(): void {
  if (sweepTimer !== null) return
  sweepTimer = setInterval(() => {
    const now = Date.now()
    for (const cache of [...liveCaches]) cache.sweepExpired(now)
    releaseSweepIfIdle()
  }, SWEEP_INTERVAL_MS)
  sweepTimer.unref?.()
}

function releaseSweepIfIdle(): void {
  if (liveCaches.size === 0 && sweepTimer !== null) {
    clearInterval(sweepTimer)
    sweepTimer = null
  }
}

/** Map-compatible LRU ownership for the two shared API payload caches. */
export class BoundedTtlCache<T> extends Map<string, Entry<T>> {
  private sizes = new Map<string, number>()
  private bytes = 0
  private readonly maxEntries: number
  private readonly maxPerGroup: number | undefined
  private readonly groupOf: ((key: string) => string) | undefined

  constructor(options: BoundedTtlCacheOptions = {}) {
    super()
    this.maxEntries = options.maxEntries ?? MAX_ENTRIES
    this.maxPerGroup = options.maxPerGroup
    this.groupOf = options.groupOf
  }

  /** O(1): never returns an expired entry; reclamation is left to the sweep. */
  override get(key: string): Entry<T> | undefined {
    const entry = super.get(key)
    if (entry && entry.expiresAt <= Date.now()) {
      this.delete(key)
      return undefined
    }
    return entry
  }

  override set(key: string, entry: Entry<T>): this {
    this.delete(key)
    let bytes: number
    try {
      bytes = (JSON.stringify(entry.value)?.length ?? 0) * 2
    } catch {
      return this
    }
    if (bytes > MAX_BYTES || entry.expiresAt <= Date.now()) return this
    // Evict before insertion while at any bound; insertion order is the LRU
    // order and delete+set above refreshes recency for re-set keys.
    while (this.size >= this.maxEntries || this.bytes + bytes > MAX_BYTES) {
      this.delete(this.keys().next().value!)
    }
    super.set(key, entry)
    this.sizes.set(key, bytes)
    this.bytes += bytes
    this.trimGroup(key)
    liveCaches.add(this)
    ensureSweepTimer()
    return this
  }

  override delete(key: string): boolean {
    this.bytes -= this.sizes.get(key) ?? 0
    this.sizes.delete(key)
    return super.delete(key)
  }

  override clear(): void {
    for (const key of this.keys()) this.delete(key)
    liveCaches.delete(this)
    releaseSweepIfIdle()
  }

  sweepExpired(now: number): void {
    for (const [key, entry] of this) {
      if (entry.expiresAt <= now) this.delete(key)
    }
    if (this.size === 0) {
      liveCaches.delete(this)
      releaseSweepIfIdle()
    }
  }

  private trimGroup(key: string): void {
    if (!this.groupOf || !this.maxPerGroup) return
    const group = this.groupOf(key)
    const groupKeys: string[] = []
    for (const existing of this.keys()) {
      if (this.groupOf(existing) === group) groupKeys.push(existing)
    }
    let excess = groupKeys.length - this.maxPerGroup
    for (let i = 0; excess > 0 && i < groupKeys.length; i++, excess--) {
      if (groupKeys[i] !== key) this.delete(groupKeys[i])
    }
  }
}
