// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest"

type Lifecycle =
  | "draft"
  | "processing"
  | "completed"
  | "partial_failure"
  | "cancelled"
  | "interrupted"

type TestRow = {
  id: string
  lifecycle: Lifecycle
  updatedAt: number
  expiresAt: number
  value: string
  submissionLeaseOwnerId?: string
  submissionLeaseExpiresAt?: number
}

class MemoryQuickIngestTable {
  readonly rows = new Map<string, TestRow>()
  readonly put = vi.fn(async (row: TestRow) => {
    if (this.putFailure) throw this.putFailure
    if (this.beforePut) await this.beforePut(row)
    this.rows.set(row.id, structuredClone(row))
    return row.id
  })
  readonly get = vi.fn(async (id: string) => {
    const row = this.rows.get(id)
    return row ? structuredClone(row) : undefined
  })
  readonly delete = vi.fn(async (id: string) => {
    if (this.deleteFailure) throw this.deleteFailure
    this.rows.delete(id)
  })
  readonly toArray = vi.fn(async () => {
    if (this.toArrayFailure) throw this.toArrayFailure
    return Array.from(this.rows.values(), (row) => structuredClone(row))
  })
  readonly bulkDelete = vi.fn(async (ids: string[]) => {
    for (const id of ids) this.rows.delete(id)
  })

  putFailure: unknown = null
  deleteFailure: unknown = null
  toArrayFailure: unknown = null
  beforePut: ((row: TestRow) => Promise<void>) | null = null
}

class MemoryQuickIngestDatabase {
  readonly quickIngestSessions = new MemoryQuickIngestTable()
  private transactionTail: Promise<void> = Promise.resolve()

  transaction = async <T>(
    _mode: "rw",
    _table: MemoryQuickIngestTable,
    operation: () => Promise<T>
  ): Promise<T> => {
    const previous = this.transactionTail
    let release!: () => void
    this.transactionTail = new Promise<void>((resolve) => {
      release = resolve
    })
    await previous
    try {
      return await operation()
    } finally {
      release()
    }
  }
}

class MemoryLegacyStorage {
  private readonly values = new Map<string, string>()
  getFailure: unknown = null
  removeFailure: unknown = null

  getItem = (key: string): string | null => {
    if (this.getFailure) throw this.getFailure
    return this.values.get(key) ?? null
  }
  peek = (key: string): string | null => this.values.get(key) ?? null
  setItem = (key: string, value: string): void => {
    this.values.set(key, value)
  }
  removeItem = (key: string): void => {
    if (this.removeFailure) throw this.removeFailure
    this.values.delete(key)
  }
}

const STORAGE_KEY = "tldw-quick-ingest-session"

const envelope = (
  id: string,
  lifecycle: Lifecycle,
  updatedAt: number,
  overrides: Record<string, unknown> = {}
): string =>
  JSON.stringify({
    state: {
      session: {
        id,
        lifecycle,
        visibility: "visible",
        currentStep: lifecycle === "processing" ? 4 : 3,
        queueItems: [],
        createdAt: updatedAt - 1,
        updatedAt,
        ...overrides,
      },
    },
    version: 0,
  })

const row = (
  id: string,
  lifecycle: Lifecycle,
  updatedAt: number,
  expiresAt: number,
  overrides: Partial<TestRow> = {}
): TestRow => ({
  id,
  lifecycle,
  updatedAt,
  expiresAt,
  value: envelope(id, lifecycle, updatedAt),
  ...overrides,
})

const loadSubject = async () =>
  vi.importActual<any>("@/db/dexie/quick-ingest")

const loadSchema = async () =>
  vi.importActual<typeof import("@/db/dexie/schema")>("@/db/dexie/schema")

const createPersistence = async (options: {
  database?: MemoryQuickIngestDatabase
  legacyStorage?: MemoryLegacyStorage
  ownerId?: string
  now?: () => number
  normalizeValue?: (value: string) => string | null
} = {}) => {
  const subject = await loadSubject()
  const database = options.database ?? new MemoryQuickIngestDatabase()
  const legacyStorage = options.legacyStorage ?? new MemoryLegacyStorage()
  const persistence = subject.createQuickIngestIndexedDbStorage({
    database,
    legacyStorage,
    ownerId: options.ownerId ?? "owner-a",
    now: options.now ?? (() => 1_000),
    normalizeValue: options.normalizeValue,
  })
  return { subject, database, legacyStorage, persistence }
}

describe("quick ingest IndexedDB persistence", () => {
  it("declares the Dexie v15 quick ingest table and required indexes", async () => {
    const { PageAssistDexieDB } = await loadSchema()
    const database = new PageAssistDexieDB()
    const table = database.tables.find(({ name }) => name === "quickIngestSessions")

    expect(database.verno).toBe(15)
    expect(table?.schema.primKey.name).toBe("id")
    expect(table?.schema.indexes.map(({ name }) => name)).toEqual([
      "lifecycle",
      "updatedAt",
      "expiresAt",
    ])
    database.close()
  })

  it("treats an absent legacy key as a successful empty migration", async () => {
    const { database, persistence } = await createPersistence()

    await expect(persistence.initialize()).resolves.toBeUndefined()
    expect(persistence.getStatus()).toBe("ready")
    expect(database.quickIngestSessions.rows.size).toBe(0)
  })

  it("fails closed when reading the legacy source throws", async () => {
    const database = new MemoryQuickIngestDatabase()
    const authority = row("durable-authority", "processing", 300, 10_000)
    database.quickIngestSessions.rows.set(authority.id, authority)
    const legacyStorage = new MemoryLegacyStorage()
    legacyStorage.setItem(STORAGE_KEY, envelope("legacy-source", "draft", 100))
    const failure = new DOMException("blocked", "SecurityError")
    legacyStorage.getFailure = failure
    const { persistence } = await createPersistence({ database, legacyStorage })

    await expect(persistence.initialize()).rejects.toBe(failure)
    expect(persistence.getStatus()).toBe("unavailable")
    expect(legacyStorage.peek(STORAGE_KEY)).not.toBeNull()
    expect(database.quickIngestSessions.rows.get(authority.id)).toEqual(authority)
  })

  it("fails closed when the default sessionStorage getter throws", async () => {
    const subject = await loadSubject()
    const database = new MemoryQuickIngestDatabase()
    const authority = row("default-storage-authority", "processing", 300, 10_000)
    database.quickIngestSessions.rows.set(authority.id, authority)
    const failure = new DOMException("blocked", "SecurityError")
    const sessionStorageGetter = vi
      .spyOn(window, "sessionStorage", "get")
      .mockImplementation(() => {
        throw failure
      })

    try {
      const persistence = subject.createQuickIngestIndexedDbStorage({
        database,
        ownerId: "owner-a",
        now: () => 1_000,
      })

      await expect(persistence.initialize()).rejects.toBe(failure)
      expect(persistence.getStatus()).toBe("unavailable")
      expect(database.quickIngestSessions.rows.get(authority.id)).toEqual(authority)
    } finally {
      sessionStorageGetter.mockRestore()
    }
  })

  it.each([
    ["malformed JSON", "{"],
    [
      "structurally malformed envelope",
      JSON.stringify({ state: { session: { id: "", lifecycle: "draft" } } }),
    ],
  ])("fails closed for a %s legacy value", async (_label, legacyValue) => {
    const database = new MemoryQuickIngestDatabase()
    const authority = row("malformed-authority", "processing", 300, 10_000)
    database.quickIngestSessions.rows.set(authority.id, authority)
    const legacyStorage = new MemoryLegacyStorage()
    legacyStorage.setItem(STORAGE_KEY, legacyValue)
    const { persistence } = await createPersistence({ database, legacyStorage })

    await expect(persistence.initialize()).rejects.toBeInstanceOf(Error)
    expect(persistence.getStatus()).toBe("unavailable")
    expect(legacyStorage.peek(STORAGE_KEY)).toBe(legacyValue)
    expect(database.quickIngestSessions.rows.get(authority.id)).toEqual(authority)
  })

  it("rejects a malformed write without deleting current authority and later accepts a valid write", async () => {
    const database = new MemoryQuickIngestDatabase()
    const authority = row("current-authority", "draft", 100, 10_000)
    database.quickIngestSessions.rows.set(authority.id, authority)
    const { persistence } = await createPersistence({ database })
    expect(await persistence.storage.getItem(STORAGE_KEY)).toBe(authority.value)

    await expect(persistence.storage.setItem(STORAGE_KEY, "{")).rejects.toBeInstanceOf(
      Error
    )
    expect(persistence.getStatus()).toBe("unavailable")
    expect(database.quickIngestSessions.rows.get(authority.id)).toEqual(authority)

    await expect(
      persistence.storage.setItem(
        STORAGE_KEY,
        envelope(authority.id, "draft", 200)
      )
    ).resolves.toBeUndefined()
    expect(database.quickIngestSessions.rows.get(authority.id)?.updatedAt).toBe(200)
    expect(persistence.getStatus()).toBe("ready")
  })

  it("treats a valid null-session envelope as an intentional clear", async () => {
    const database = new MemoryQuickIngestDatabase()
    const authority = row("clear-authority", "draft", 100, 10_000)
    database.quickIngestSessions.rows.set(authority.id, authority)
    const { persistence } = await createPersistence({ database })
    await persistence.storage.getItem(STORAGE_KEY)

    await expect(
      persistence.storage.setItem(
        STORAGE_KEY,
        JSON.stringify({ state: { session: null }, version: 0 })
      )
    ).resolves.toBeUndefined()
    expect(database.quickIngestSessions.rows.has(authority.id)).toBe(false)
    expect(persistence.getStatus()).toBe("ready")
  })

  it("applies the injected envelope normalizer to legacy and regular writes", async () => {
    const marker = "data:image/png;base64,unsafe-payload"
    const normalizeValue = vi.fn((value: string) => {
      const parsed = JSON.parse(value)
      if (parsed.state?.session?.results?.[0]) {
        delete parsed.state.session.results[0].data
      }
      return JSON.stringify(parsed)
    })
    const legacyStorage = new MemoryLegacyStorage()
    legacyStorage.setItem(
      STORAGE_KEY,
      envelope("normalized-legacy", "draft", 100, {
        results: [{ id: "legacy-result", status: "ok", type: "video", data: marker }],
      })
    )
    const migrated = await createPersistence({ legacyStorage, normalizeValue })

    await migrated.persistence.initialize()
    expect(normalizeValue).toHaveBeenCalledTimes(1)
    expect(
      migrated.database.quickIngestSessions.rows.get("normalized-legacy")?.value
    ).not.toContain(marker)

    await migrated.persistence.storage.setItem(
      STORAGE_KEY,
      envelope("normalized-write", "draft", 200, {
        results: [{ id: "write-result", status: "ok", type: "video", data: marker }],
      })
    )
    expect(normalizeValue).toHaveBeenCalledTimes(2)
    expect(
      migrated.database.quickIngestSessions.rows.get("normalized-write")?.value
    ).not.toContain(marker)
  })

  it("keeps the legacy source until its imported row is durably written", async () => {
    const database = new MemoryQuickIngestDatabase()
    const legacyStorage = new MemoryLegacyStorage()
    legacyStorage.setItem(STORAGE_KEY, envelope("legacy-session", "draft", 100))
    let releasePut!: () => void
    database.quickIngestSessions.beforePut = () =>
      new Promise<void>((resolve) => {
        releasePut = resolve
      })
    const { persistence } = await createPersistence({ database, legacyStorage })

    const migration = persistence.initialize()
    await vi.waitFor(() => expect(releasePut).toBeTypeOf("function"))
    expect(legacyStorage.getItem(STORAGE_KEY)).not.toBeNull()

    releasePut()
    await migration
    expect(database.quickIngestSessions.rows.get("legacy-session")?.value).toContain(
      '"id":"legacy-session"'
    )
    expect(legacyStorage.getItem(STORAGE_KEY)).toBeNull()
  })

  it("repeats an interrupted legacy import idempotently", async () => {
    const database = new MemoryQuickIngestDatabase()
    const legacyStorage = new MemoryLegacyStorage()
    const legacyEnvelope = envelope("legacy-retry", "interrupted", 200)
    legacyStorage.setItem(STORAGE_KEY, legacyEnvelope)
    const first = await createPersistence({ database, legacyStorage })

    await first.persistence.initialize()
    legacyStorage.setItem(STORAGE_KEY, legacyEnvelope)
    const second = await createPersistence({ database, legacyStorage })
    await second.persistence.initialize()

    expect(database.quickIngestSessions.rows.size).toBe(1)
    expect(database.quickIngestSessions.rows.get("legacy-retry")?.value).toBe(
      legacyEnvelope
    )
    expect(legacyStorage.getItem(STORAGE_KEY)).toBeNull()
  })

  it("does not let an older legacy envelope overwrite newer IndexedDB authority", async () => {
    const database = new MemoryQuickIngestDatabase()
    const legacyStorage = new MemoryLegacyStorage()
    const currentEnvelope = envelope("authority", "processing", 300)
    database.quickIngestSessions.rows.set(
      "authority",
      row("authority", "processing", 300, 10_000, { value: currentEnvelope })
    )
    legacyStorage.setItem(STORAGE_KEY, envelope("authority", "draft", 200))
    const { persistence } = await createPersistence({ database, legacyStorage })

    await persistence.initialize()

    expect(await persistence.storage.getItem(STORAGE_KEY)).toBe(currentEnvelope)
    expect(database.quickIngestSessions.rows.get("authority")?.lifecycle).toBe(
      "processing"
    )
    expect(legacyStorage.getItem(STORAGE_KEY)).toBeNull()
  })

  it("returns the most recently updated retained record", async () => {
    const database = new MemoryQuickIngestDatabase()
    database.quickIngestSessions.rows.set(
      "older",
      row("older", "interrupted", 300, 10_000)
    )
    const newest = row("newest", "completed", 400, 10_000)
    database.quickIngestSessions.rows.set("newest", newest)
    const { persistence } = await createPersistence({ database })

    expect(await persistence.storage.getItem(STORAGE_KEY)).toBe(newest.value)
  })

  it("uses named bounded retention windows for each lifecycle", async () => {
    const { subject } = await createPersistence()

    expect(subject.QUICK_INGEST_DRAFT_RETENTION_MS).toBeGreaterThan(0)
    expect(subject.QUICK_INGEST_TERMINAL_RETENTION_MS).toBeGreaterThan(0)
    expect(subject.QUICK_INGEST_INTERRUPTED_RETENTION_MS).toBeGreaterThan(
      subject.QUICK_INGEST_TERMINAL_RETENTION_MS
    )
    expect(subject.QUICK_INGEST_PROCESSING_RETENTION_MS).toBeGreaterThan(
      subject.QUICK_INGEST_INTERRUPTED_RETENTION_MS
    )
  })

  it("keeps unexpired interrupted and terminal rows while removing expired remnants", async () => {
    let now = 1_000
    const database = new MemoryQuickIngestDatabase()
    database.quickIngestSessions.rows.set(
      "active",
      row("active", "processing", 100, now + 1)
    )
    database.quickIngestSessions.rows.set(
      "interrupted-recent",
      row("interrupted-recent", "interrupted", 200, now + 1)
    )
    database.quickIngestSessions.rows.set(
      "terminal-recent",
      row("terminal-recent", "completed", 300, now + 1)
    )
    database.quickIngestSessions.rows.set(
      "terminal-expired",
      row("terminal-expired", "partial_failure", 400, now)
    )
    database.quickIngestSessions.rows.set(
      "draft-expired",
      row("draft-expired", "draft", 500, now - 1)
    )
    const { persistence } = await createPersistence({ database, now: () => now })

    await persistence.cleanupExpired()

    expect(Array.from(database.quickIngestSessions.rows.keys()).sort()).toEqual([
      "active",
      "interrupted-recent",
      "terminal-recent",
    ])
  })

  it("removes processing authority only at its explicit long-retention boundary", async () => {
    let now = 1_000
    const database = new MemoryQuickIngestDatabase()
    database.quickIngestSessions.rows.set(
      "processing",
      row("processing", "processing", 100, now + 1)
    )
    const { persistence } = await createPersistence({ database, now: () => now })

    await persistence.cleanupExpired()
    expect(database.quickIngestSessions.rows.has("processing")).toBe(true)

    now += 1
    await persistence.cleanupExpired()
    expect(database.quickIngestSessions.rows.has("processing")).toBe(false)
  })

  it.each([
    [new DOMException("full", "QuotaExceededError"), "quota_error"],
    [new DOMException("blocked", "SecurityError"), "unavailable"],
    [new Error("write failed"), "unavailable"],
  ])("classifies visible persistence failures", async (failure, expectedStatus) => {
    const database = new MemoryQuickIngestDatabase()
    database.quickIngestSessions.putFailure = failure
    const { persistence } = await createPersistence({ database })
    const statuses: string[] = []
    persistence.subscribeStatus((status: string) => statuses.push(status))

    await expect(
      persistence.storage.setItem(
        STORAGE_KEY,
        envelope("write-failure", "completed", 500)
      )
    ).rejects.toBe(failure)
    expect(statuses.at(-1)).toBe(expectedStatus)
  })

  it("publishes and rethrows a read failure after initialization", async () => {
    const database = new MemoryQuickIngestDatabase()
    const { persistence } = await createPersistence({ database })
    await persistence.initialize()
    const failure = new DOMException("blocked", "SecurityError")
    database.quickIngestSessions.toArrayFailure = failure

    await expect(persistence.storage.getItem(STORAGE_KEY)).rejects.toBe(failure)
    expect(persistence.getStatus()).toBe("unavailable")
  })

  it("publishes and rethrows a delete failure", async () => {
    const database = new MemoryQuickIngestDatabase()
    database.quickIngestSessions.rows.set(
      "delete-failure",
      row("delete-failure", "draft", 100, 10_000)
    )
    const { persistence } = await createPersistence({ database })
    await persistence.storage.getItem(STORAGE_KEY)
    const failure = new DOMException("blocked", "SecurityError")
    database.quickIngestSessions.deleteFailure = failure

    await expect(persistence.storage.removeItem(STORAGE_KEY)).rejects.toBe(
      failure
    )
    expect(persistence.getStatus()).toBe("unavailable")
  })

  it("never evicts processing authority while handling a quota failure", async () => {
    const database = new MemoryQuickIngestDatabase()
    database.quickIngestSessions.rows.set(
      "active-authority",
      row("active-authority", "processing", 100, 101)
    )
    database.quickIngestSessions.rows.set(
      "expired-terminal",
      row("expired-terminal", "completed", 50, 99)
    )
    database.quickIngestSessions.putFailure = new DOMException(
      "full",
      "QuotaExceededError"
    )
    const { persistence } = await createPersistence({ database, now: () => 100 })

    await expect(
      persistence.storage.setItem(
        STORAGE_KEY,
        envelope("new-terminal", "completed", 500)
      )
    ).rejects.toMatchObject({ name: "QuotaExceededError" })
    expect(database.quickIngestSessions.rows.has("active-authority")).toBe(true)
  })

  it("preserves an acquired lease across ordinary session writes", async () => {
    const database = new MemoryQuickIngestDatabase()
    database.quickIngestSessions.rows.set(
      "lease-write",
      row("lease-write", "draft", 100, 100_000)
    )
    const ownerA = await createPersistence({
      database,
      ownerId: "owner-a",
      now: () => 1_000,
    })
    const ownerB = await createPersistence({
      database,
      ownerId: "owner-b",
      now: () => 1_000,
    })
    expect(
      await ownerA.persistence.acquireSubmissionLease("lease-write", 100)
    ).toBe(true)

    await ownerA.persistence.storage.setItem(
      STORAGE_KEY,
      envelope("lease-write", "draft", 200)
    )

    expect(
      database.quickIngestSessions.rows.get("lease-write")
        ?.submissionLeaseOwnerId
    ).toBe("owner-a")
    expect(
      await ownerB.persistence.acquireSubmissionLease("lease-write", 100)
    ).toBe(false)
  })

  it("does not let a stale background draft demote non-draft authority or reopen its expired lease", async () => {
    const database = new MemoryQuickIngestDatabase()
    database.quickIngestSessions.rows.set(
      "stale-draft-authority",
      row("stale-draft-authority", "processing", 300, 100_000, {
        submissionLeaseOwnerId: "expired-owner",
        submissionLeaseExpiresAt: 999,
      })
    )
    const staleWriter = await createPersistence({
      database,
      ownerId: "stale-writer",
      now: () => 1_000,
    })
    const contender = await createPersistence({
      database,
      ownerId: "contender",
      now: () => 1_000,
    })
    await staleWriter.persistence.storage.getItem(STORAGE_KEY)

    await staleWriter.persistence.storage.setItem(
      STORAGE_KEY,
      envelope("stale-draft-authority", "draft", 100)
    )
    const acquired = await contender.persistence.acquireSubmissionLease(
      "stale-draft-authority",
      100
    )

    expect({
      lifecycle:
        database.quickIngestSessions.rows.get("stale-draft-authority")
          ?.lifecycle,
      acquired,
    }).toEqual({ lifecycle: "processing", acquired: false })
  })

  it("does not let a stale background clear delete non-draft authority", async () => {
    const database = new MemoryQuickIngestDatabase()
    database.quickIngestSessions.rows.set(
      "stale-clear-authority",
      row("stale-clear-authority", "completed", 300, 100_000)
    )
    const staleWriter = await createPersistence({
      database,
      ownerId: "stale-writer",
    })
    await createPersistence({ database, ownerId: "authority-owner" })
    await staleWriter.persistence.storage.getItem(STORAGE_KEY)

    await staleWriter.persistence.storage.removeItem(STORAGE_KEY)

    expect(
      database.quickIngestSessions.rows.get("stale-clear-authority")?.lifecycle
    ).toBe("completed")
  })

  it("allows an exact confirmed Review CAS to replace processing authority with a draft", async () => {
    const database = new MemoryQuickIngestDatabase()
    const expected = envelope("confirmed-review-handoff", "processing", 200)
    database.quickIngestSessions.rows.set(
      "confirmed-review-handoff",
      {
        ...row("confirmed-review-handoff", "processing", 200, 100_000),
        value: expected,
      }
    )
    const { persistence } = await createPersistence({ database })

    await expect(
      persistence.commitReviewHandoff(
        expected,
        envelope("confirmed-review-handoff", "draft", 300)
      )
    ).resolves.toBe(true)

    expect(
      database.quickIngestSessions.rows.get("confirmed-review-handoff")
        ?.lifecycle
    ).toBe("draft")
  })

  it("rejects a captured Review CAS after the durable envelope advances", async () => {
    const database = new MemoryQuickIngestDatabase()
    const captured = envelope("captured-review-handoff", "processing", 200, {
      tracking: { mode: "extension-runtime", generation: "old" },
    })
    const advanced = envelope("captured-review-handoff", "processing", 300, {
      tracking: { mode: "extension-runtime", generation: "new" },
    })
    database.quickIngestSessions.rows.set("captured-review-handoff", {
      ...row("captured-review-handoff", "processing", 300, 100_000),
      value: advanced,
    })
    const { persistence } = await createPersistence({ database })

    await expect(
      persistence.commitReviewHandoff(
        captured,
        envelope("captured-review-handoff", "draft", 400)
      )
    ).resolves.toBe(false)
    expect(database.quickIngestSessions.rows.get("captured-review-handoff")).toMatchObject({
      lifecycle: "processing",
      updatedAt: 300,
      value: advanced,
    })
  })

  it("clears terminal authority only when its lifecycle and revision still match", async () => {
    const database = new MemoryQuickIngestDatabase()
    database.quickIngestSessions.rows.set(
      "terminal-clear-cas",
      row("terminal-clear-cas", "completed", 300, 100_000)
    )
    const { persistence } = await createPersistence({ database })

    await expect(
      persistence.clearAuthoritativeSession({
        id: "terminal-clear-cas",
        lifecycle: "completed",
        updatedAt: 200,
      })
    ).resolves.toBe(false)
    expect(database.quickIngestSessions.rows.has("terminal-clear-cas")).toBe(
      true
    )

    await expect(
      persistence.clearAuthoritativeSession({
        id: "terminal-clear-cas",
        lifecycle: "completed",
        updatedAt: 300,
      })
    ).resolves.toBe(true)
    expect(database.quickIngestSessions.rows.has("terminal-clear-cas")).toBe(
      false
    )
  })

  it("persists Start over as a fresh acquirable draft that stale terminal clear cannot delete", async () => {
    const database = new MemoryQuickIngestDatabase()
    database.quickIngestSessions.rows.set(
      "terminal-start-over",
      row("terminal-start-over", "completed", 300, 100_000)
    )
    const { persistence } = await createPersistence({
      database,
      ownerId: "fresh-draft-owner",
    })
    const { createQuickIngestSessionStore } =
      await vi.importActual<any>("@/store/quick-ingest-session")
    const store = createQuickIngestSessionStore({ persistence })
    await store.persist.rehydrate()
    const terminal = store.getState().session
    expect(terminal).toMatchObject({
      id: "terminal-start-over",
      lifecycle: "completed",
    })

    const freshDraft = store.getState().replaceWithNewDraft()
    await persistence.storage.getItem(STORAGE_KEY)

    expect(freshDraft).toMatchObject({ lifecycle: "draft", currentStep: 1 })
    expect(freshDraft.id).not.toBe("terminal-start-over")
    expect(database.quickIngestSessions.rows.get(freshDraft.id)?.lifecycle).toBe(
      "draft"
    )
    await expect(
      persistence.acquireSubmissionLease(freshDraft.id, 100)
    ).resolves.toBe(true)
    await expect(
      persistence.clearAuthoritativeSession({
        id: "terminal-start-over",
        lifecycle: "completed",
        updatedAt: 300,
      })
    ).resolves.toBe(false)
    expect(database.quickIngestSessions.rows.get(freshDraft.id)).toMatchObject({
      lifecycle: "draft",
      submissionLeaseOwnerId: "fresh-draft-owner",
    })
  })

  it("allows only one of two stores sharing the database to reach submission", async () => {
    const database = new MemoryQuickIngestDatabase()
    database.quickIngestSessions.rows.set(
      "shared-store-race",
      row("shared-store-race", "draft", 100, 100_000)
    )
    const ownerA = await createPersistence({
      database,
      ownerId: "owner-a",
      now: () => 1_000,
    })
    const ownerB = await createPersistence({
      database,
      ownerId: "owner-b",
      now: () => 1_000,
    })
    await Promise.all([
      ownerA.persistence.initialize(),
      ownerB.persistence.initialize(),
    ])
    const { createEmptyQuickIngestSession, createQuickIngestSessionStore } =
      await vi.importActual<any>("@/store/quick-ingest-session")
    const draft = {
      ...createEmptyQuickIngestSession(),
      id: "shared-store-race",
      updatedAt: 100,
    }
    const storeA = createQuickIngestSessionStore({
      persistence: ownerA.persistence,
    })
    const storeB = createQuickIngestSessionStore({
      persistence: ownerB.persistence,
    })
    storeA.setState({ session: draft, persistenceStatus: "ready" })
    storeB.setState({ session: draft, persistenceStatus: "ready" })
    const submit = vi.fn()
    const startIfOwner = async (store: typeof storeA, label: string) => {
      if (await store.getState().acquireSubmissionLease()) submit(label)
    }

    await Promise.all([
      startIfOwner(storeA, "owner-a"),
      startIfOwner(storeB, "owner-b"),
    ])

    expect(submit).toHaveBeenCalledTimes(1)
  })

  it("allows an expired draft lease to be taken over", async () => {
    const database = new MemoryQuickIngestDatabase()
    database.quickIngestSessions.rows.set(
      "expired-draft",
      row("expired-draft", "draft", 100, 100_000, {
        submissionLeaseOwnerId: "stale-owner",
        submissionLeaseExpiresAt: 999,
      })
    )
    const { persistence } = await createPersistence({
      database,
      ownerId: "new-owner",
      now: () => 1_000,
    })

    expect(
      await persistence.acquireSubmissionLease("expired-draft", 100)
    ).toBe(true)
    expect(
      database.quickIngestSessions.rows.get("expired-draft")
        ?.submissionLeaseOwnerId
    ).toBe("new-owner")
  })

  it("renews the same owner's lease when Start immediately reacquires", async () => {
    const database = new MemoryQuickIngestDatabase()
    database.quickIngestSessions.rows.set(
      "same-owner",
      row("same-owner", "draft", 100, 100_000, {
        submissionLeaseOwnerId: "owner-a",
        submissionLeaseExpiresAt: 1_050,
      })
    )
    const { persistence } = await createPersistence({
      database,
      ownerId: "owner-a",
      now: () => 1_000,
    })

    expect(await persistence.acquireSubmissionLease("same-owner", 100)).toBe(
      true
    )
    expect(
      database.quickIngestSessions.rows.get("same-owner")
        ?.submissionLeaseExpiresAt
    ).toBe(1_100)
  })

  it("fences a queued processing CAS after lease takeover and renews drafts only", async () => {
    let now = 1_000
    const database = new MemoryQuickIngestDatabase()
    const expectedDraft = envelope("lease-cas-takeover", "draft", 100)
    database.quickIngestSessions.rows.set("lease-cas-takeover", {
      ...row("lease-cas-takeover", "draft", 100, 100_000),
      value: expectedDraft,
      submissionLeaseOwnerId: "owner-a",
      submissionLeaseExpiresAt: 1_100,
    })
    const ownerA = await createPersistence({
      database,
      ownerId: "owner-a",
      now: () => now,
    })
    const ownerB = await createPersistence({
      database,
      ownerId: "owner-b",
      now: () => now,
    })

    now = 1_050
    expect(
      await ownerA.persistence.renewSubmissionLease("lease-cas-takeover", 100)
    ).toBe(true)
    now = 1_151
    expect(
      await ownerB.persistence.acquireSubmissionLease("lease-cas-takeover", 100)
    ).toBe(true)

    await expect(
      ownerA.persistence.commitProcessingHandoff(
        expectedDraft,
        envelope("lease-cas-takeover", "processing", 200)
      )
    ).resolves.toBe(false)
    expect(database.quickIngestSessions.rows.get("lease-cas-takeover")).toMatchObject({
      lifecycle: "draft",
      value: expectedDraft,
      submissionLeaseOwnerId: "owner-b",
      submissionLeaseExpiresAt: 1_251,
    })

    const takenOver = database.quickIngestSessions.rows.get("lease-cas-takeover")!
    database.quickIngestSessions.rows.set("lease-cas-takeover", {
      ...takenOver,
      lifecycle: "processing",
      value: envelope("lease-cas-takeover", "processing", 200),
    })
    const leaseExpiry = takenOver.submissionLeaseExpiresAt
    expect(
      await ownerB.persistence.renewSubmissionLease("lease-cas-takeover", 100)
    ).toBe(false)
    expect(
      database.quickIngestSessions.rows.get("lease-cas-takeover")
        ?.submissionLeaseExpiresAt
    ).toBe(leaseExpiry)
  })

  it.each([
    "processing",
    "completed",
    "partial_failure",
    "cancelled",
    "interrupted",
  ] as const)(
    "rejects lease acquisition for an authoritative %s row even after expiry",
    async (lifecycle) => {
      const database = new MemoryQuickIngestDatabase()
      database.quickIngestSessions.rows.set(
        `non-draft-${lifecycle}`,
        row(`non-draft-${lifecycle}`, lifecycle, 200, 100_000, {
          submissionLeaseOwnerId: "stale-owner",
          submissionLeaseExpiresAt: 999,
        })
      )
      const { persistence } = await createPersistence({
        database,
        ownerId: "new-owner",
        now: () => 1_000,
      })

      expect(
        await persistence.acquireSubmissionLease(
          `non-draft-${lifecycle}`,
          100
        )
      ).toBe(false)
      expect(
        database.quickIngestSessions.rows.get(`non-draft-${lifecycle}`)
          ?.submissionLeaseOwnerId
      ).toBe("stale-owner")
    }
  )

  it("allows exactly one owner to acquire, renew, expire, take over, and release a lease", async () => {
    let now = 1_000
    const database = new MemoryQuickIngestDatabase()
    database.quickIngestSessions.rows.set(
      "lease-session",
      row("lease-session", "draft", now, 100_000)
    )
    const ownerA = await createPersistence({
      database,
      ownerId: "owner-a",
      now: () => now,
    })
    const ownerB = await createPersistence({
      database,
      ownerId: "owner-b",
      now: () => now,
    })

    const [aWon, bWon] = await Promise.all([
      ownerA.persistence.acquireSubmissionLease("lease-session", 100),
      ownerB.persistence.acquireSubmissionLease("lease-session", 100),
    ])
    expect([aWon, bWon].filter(Boolean)).toHaveLength(1)

    const winner = aWon ? ownerA.persistence : ownerB.persistence
    const loser = aWon ? ownerB.persistence : ownerA.persistence
    expect(await loser.acquireSubmissionLease("lease-session", 100)).toBe(false)

    now = 1_050
    expect(await winner.renewSubmissionLease("lease-session", 100)).toBe(true)
    now = 1_101
    expect(await loser.acquireSubmissionLease("lease-session", 100)).toBe(false)
    now = 1_151
    expect(await loser.acquireSubmissionLease("lease-session", 100)).toBe(true)

    await winner.releaseSubmissionLease("lease-session")
    expect(await winner.acquireSubmissionLease("lease-session", 100)).toBe(false)
    await loser.releaseSubmissionLease("lease-session")
    expect(await winner.acquireSubmissionLease("lease-session", 100)).toBe(true)
  })
})
