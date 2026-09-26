import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import {
  CLONE_RECOVERY_KEY,
  CLONE_RECOVERY_TTL,
  CloneRecoveryFullError,
  clearCloneRecovery,
  readCloneRecovery,
  writeCloneRecovery
} from "../shared-clone-recovery"
import { operationId } from "../tldw/domains/__tests__/shared-workspace-clone.fixture"

const scope = "https://server.example|user:42"
const now = 1_800_000_000_000
const record = (share_id = 42) => ({
  share_id,
  expires_at: now + CLONE_RECOVERY_TTL,
  idempotency_key: "clone-key-0000000001",
  name: "Research (Copy)"
})

describe("bounded scoped clone recovery", () => {
  beforeEach(() => {
    window.localStorage.clear()
    const tails = new Map<string, Promise<unknown>>()
    vi.stubGlobal("navigator", {
      locks: {
        request: (name: string, options: unknown, callback?: () => unknown) => {
          const run = (tails.get(name) ?? Promise.resolve()).then(
            callback ?? (options as () => unknown)
          )
          tails.set(
            name,
            run.catch(() => undefined)
          )
          return run
        }
      }
    })
  })
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })
  it("round trips the exact pre-submit command without storing workspace content", async () => {
    await writeCloneRecovery(window.localStorage, scope, record(), now)
    expect(await readCloneRecovery(window.localStorage, scope, now)).toEqual([
      record()
    ])
  })
  it("retains only the operation pointer at terminal completion", async () => {
    await writeCloneRecovery(window.localStorage, scope, record(), now)
    const terminal = {
      share_id: 42,
      expires_at: record().expires_at,
      operation_id: operationId
    }
    await writeCloneRecovery(window.localStorage, scope, terminal, now)
    expect(await readCloneRecovery(window.localStorage, scope, now)).toEqual([
      terminal
    ])
    expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).not.toContain(
      "Research"
    )
  })
  it("merges independent shares and removes a definitely rejected command", async () => {
    await writeCloneRecovery(window.localStorage, scope, record(1), now)
    await writeCloneRecovery(window.localStorage, scope, record(2), now)
    await writeCloneRecovery(window.localStorage, scope, null, now, 1)
    expect(await readCloneRecovery(window.localStorage, scope, now)).toEqual([
      record(2)
    ])
  })
  it("preserves foreign recovery and rejects the scoped read", async () => {
    await writeCloneRecovery(window.localStorage, scope, record(), now)
    const raw = window.localStorage.getItem(CLONE_RECOVERY_KEY)
    await expect(
      readCloneRecovery(
        window.localStorage,
        "https://other.example|user:42",
        now
      )
    ).rejects.toMatchObject({ name: "CloneRecoveryScopeMismatchError" })
    expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBe(raw)
  })

  it.each(["stale update", "new admission", "removal"])(
    "rejects a foreign-scope %s before mutation under the lock",
    async (kind) => {
      let release!: () => void
      const held = navigator.locks.request(
        CLONE_RECOVERY_KEY,
        () =>
          new Promise<void>((done) => {
            release = done
          })
      )
      await Promise.resolve()
      const write = writeCloneRecovery(
        window.localStorage,
        scope,
        kind === "removal" ? null : record(),
        now,
        42,
        { expected: kind === "new admission" ? null : record() }
      )
      const rejected = expect(write).rejects.toMatchObject({
        name: "CloneRecoveryScopeMismatchError"
      })
      const raw = JSON.stringify({
        version: 1,
        scope: "B",
        records: [record(43)]
      })
      window.localStorage.setItem(CLONE_RECOVERY_KEY, raw)
      release()
      await held
      await rejected
      expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBe(raw)
    }
  )

  it.each(
    [
      [],
      [{ ...record(), expires_at: now - 1 }],
      [{ ...record(), expires_at: now + CLONE_RECOVERY_TTL * 2 }],
      [{ ...record(), secret: "foreign" }]
    ].map((records) => ({ records }))
  )(
    "checks foreign scope before empty, expiry or schema cleanup: %j",
    async ({ records }) => {
      const raw = JSON.stringify({ version: 1, scope: "B", records })
      window.localStorage.setItem(CLONE_RECOVERY_KEY, raw)
      await expect(
        readCloneRecovery(window.localStorage, scope, now)
      ).rejects.toMatchObject({ name: "CloneRecoveryScopeMismatchError" })
      expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBe(raw)
    }
  )

  it.each([{ version: 99 }, { future_field: "new format" }])(
    "preserves foreign envelopes before schema validation: %j",
    async (extra) => {
      const raw = JSON.stringify({
        version: 1,
        scope: "B",
        records: [record()],
        ...extra
      })
      window.localStorage.setItem(CLONE_RECOVERY_KEY, raw)
      await expect(
        readCloneRecovery(window.localStorage, scope, now)
      ).rejects.toMatchObject({ name: "CloneRecoveryScopeMismatchError" })
      await expect(
        writeCloneRecovery(window.localStorage, scope, record(), now)
      ).rejects.toMatchObject({ name: "CloneRecoveryScopeMismatchError" })
      expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBe(raw)
    }
  )
  it("expires records after seven days", async () => {
    await writeCloneRecovery(window.localStorage, scope, record(), now)
    expect(
      await readCloneRecovery(
        window.localStorage,
        scope,
        now + CLONE_RECOVERY_TTL
      )
    ).toEqual([])
  })
  it.each([
    "{",
    "x".repeat(33 * 1024),
    JSON.stringify({ version: 99, scope, records: [] })
  ])("discards corrupt or oversized input", async (raw) => {
    window.localStorage.setItem(CLONE_RECOVERY_KEY, raw)
    expect(await readCloneRecovery(window.localStorage, scope, now)).toEqual([])
    expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBeNull()
  })
  it("rejects unknown fields and an unbounded expiry", async () => {
    for (const extra of [
      { secret: "content" },
      { expires_at: now + CLONE_RECOVERY_TTL * 2 }
    ]) {
      window.localStorage.setItem(
        CLONE_RECOVERY_KEY,
        JSON.stringify({
          version: 1,
          scope,
          records: [{ ...record(), ...extra }]
        })
      )
      expect(await readCloneRecovery(window.localStorage, scope, now)).toEqual(
        []
      )
    }
  })
  it("refuses to evict recoverable operations when the map is full", async () => {
    for (let share = 1; share <= 32; share++)
      await writeCloneRecovery(window.localStorage, scope, record(share), now)
    await expect(async () =>
      writeCloneRecovery(window.localStorage, scope, record(33), now)
    ).rejects.toThrow()
    expect(
      await readCloneRecovery(window.localStorage, scope, now)
    ).toHaveLength(32)
  })

  it("does not rewrite unchanged records or reorder shares during polling", async () => {
    await writeCloneRecovery(window.localStorage, scope, record(1), now)
    await writeCloneRecovery(window.localStorage, scope, record(2), now)
    const before = window.localStorage.getItem(CLONE_RECOVERY_KEY)
    const set = vi.spyOn(window.localStorage, "setItem")
    await writeCloneRecovery(window.localStorage, scope, record(1), now)
    expect(set).not.toHaveBeenCalled()
    expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBe(before)
  })

  it("waits for the cross-tab lock before merging independent commands", async () => {
    let release!: () => void
    const held = navigator.locks.request(
      CLONE_RECOVERY_KEY,
      () =>
        new Promise<void>((resolve) => {
          release = resolve
        })
    )
    await Promise.resolve()
    const first = writeCloneRecovery(window.localStorage, scope, record(1), now)
    const second = writeCloneRecovery(
      window.localStorage,
      scope,
      record(2),
      now
    )
    const before = window.localStorage.getItem(CLONE_RECOVERY_KEY)
    release()
    await held
    await Promise.all([first, second])
    expect(before).toBeNull()
    expect(await readCloneRecovery(window.localStorage, scope, now)).toEqual([
      record(1),
      record(2)
    ])
  })

  it("serializes read cleanup with a new command instead of deleting it", async () => {
    window.localStorage.setItem(CLONE_RECOVERY_KEY, "{")
    let release!: () => void
    const held = navigator.locks.request(
      CLONE_RECOVERY_KEY,
      () =>
        new Promise<void>((resolve) => {
          release = resolve
        })
    )
    await Promise.resolve()
    const cleanup = readCloneRecovery(window.localStorage, scope, now)
    const before = window.localStorage.getItem(CLONE_RECOVERY_KEY)
    const write = writeCloneRecovery(window.localStorage, scope, record(), now)
    release()
    await held
    await Promise.all([cleanup, write])
    expect(before).toBe("{")
    expect(await readCloneRecovery(window.localStorage, scope, now)).toEqual([
      record()
    ])
  })

  it("returns the canonical attempt when a stale writer tries to replace or remove a retry", async () => {
    const original = record()
    const retry = { ...original, idempotency_key: "clone-key-0000000002" }
    await writeCloneRecovery(window.localStorage, scope, retry, now)
    for (const update of [{ ...original, operation_id: operationId }, null]) {
      expect(
        await writeCloneRecovery(window.localStorage, scope, update, now, 42, {
          expected: original
        })
      ).toEqual(retry)
      expect(await readCloneRecovery(window.localStorage, scope, now)).toEqual([
        retry
      ])
    }
  })

  it("invalidates queued writes before logout cleanup obtains the lock", async () => {
    let release!: () => void
    const held = navigator.locks.request(
      CLONE_RECOVERY_KEY,
      () =>
        new Promise<void>((resolve) => {
          release = resolve
        })
    )
    await Promise.resolve()
    const write = Promise.resolve().then(() =>
      writeCloneRecovery(window.localStorage, scope, record(), now)
    )
    await Promise.resolve()
    const rejected = expect(write).rejects.toMatchObject({ name: "AbortError" })
    const clear = clearCloneRecovery()
    const before = window.localStorage.getItem(CLONE_RECOVERY_KEY)
    release()
    await held
    await rejected
    await clear
    expect(before).toBeNull()
    expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBeNull()
  })

  it("checks the departing scope inside the lock before clearing a newer tab's recovery", async () => {
    await writeCloneRecovery(window.localStorage, scope, record(), now)
    let release!: () => void
    const held = navigator.locks.request(
      CLONE_RECOVERY_KEY,
      () =>
        new Promise<void>((resolve) => {
          release = resolve
        })
    )
    await Promise.resolve()
    const clear = clearCloneRecovery(scope)
    const raw = JSON.stringify({
      version: 1,
      scope: "other-principal",
      records: [record(43)]
    })
    // Another tab owns the lock and replaces A's envelope before its cleanup runs.
    window.localStorage.setItem(CLONE_RECOVERY_KEY, raw)
    release()
    await held
    await clear
    expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBe(raw)
    await clearCloneRecovery("other-principal")
    expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBeNull()
  })

  it("refuses durable storage without Web Locks but clears logout data synchronously", async () => {
    vi.stubGlobal("navigator", {})
    window.localStorage.setItem(CLONE_RECOVERY_KEY, "{")
    await expect(async () =>
      readCloneRecovery(window.localStorage, scope, now)
    ).rejects.toThrow()
    await expect(async () =>
      writeCloneRecovery(window.localStorage, scope, record(), now)
    ).rejects.toThrow()
    expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBe("{")
    const clear = clearCloneRecovery()
    expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBeNull()
    await clear
  })

  it("accepts a 255-codepoint supplementary Unicode command name", async () => {
    const entry = { ...record(), name: "\u{1f4da}".repeat(248) + " (Copy)" }
    await writeCloneRecovery(window.localStorage, scope, entry, now)
    expect(await readCloneRecovery(window.localStorage, scope, now)).toEqual([
      entry
    ])
  })

  it("does not discard live recovery when persisting expiry cleanup fails", async () => {
    const live = record(1)
    const expired = { ...record(2), expires_at: now }
    const raw = JSON.stringify({ version: 1, scope, records: [live, expired] })
    window.localStorage.setItem(CLONE_RECOVERY_KEY, raw)
    const storage = {
      getItem: (key: string) => window.localStorage.getItem(key),
      setItem: () => {
        throw new Error("quota exceeded")
      },
      removeItem: (key: string) => window.localStorage.removeItem(key)
    }
    await expect(readCloneRecovery(storage, scope, now)).rejects.toThrow(
      "quota exceeded"
    )
    expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBe(raw)
  })

  it("bounds lock acquisition and prevents a timed-out queued write from running later", async () => {
    vi.useFakeTimers()
    let release!: () => void
    const held = navigator.locks.request(
      CLONE_RECOVERY_KEY,
      () =>
        new Promise<void>((done) => {
          release = done
        })
    )
    await Promise.resolve()
    let outcome: string | undefined
    const write = writeCloneRecovery(
      window.localStorage,
      scope,
      record(),
      now
    ).then(
      () => {
        outcome = "saved"
      },
      (error) => {
        outcome = error.name
      }
    )
    await vi.advanceTimersByTimeAsync(5000)
    const atDeadline = outcome
    release()
    await held
    await write
    expect(atDeadline).toBe("TimeoutError")
    expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBeNull()
  })

  it("reserves enough bytes for every admitted Unicode command to gain an operation pointer", async () => {
    const admitted = []
    for (let id = 1; id <= 32; id++) {
      const entry = {
        ...record(id),
        idempotency_key: "a".repeat(36),
        name: "\u{1f4da}".repeat(248) + " (Copy)"
      }
      try {
        await writeCloneRecovery(window.localStorage, scope, entry, now)
        admitted.push(entry)
      } catch (error) {
        expect(error).toBeInstanceOf(CloneRecoveryFullError)
        break
      }
    }
    expect(admitted.length).toBeGreaterThan(0)
    expect(admitted.length).toBeLessThan(32)
    for (const entry of admitted) {
      await expect(
        writeCloneRecovery(
          window.localStorage,
          scope,
          { ...entry, operation_id: operationId },
          now
        )
      ).resolves.toMatchObject({ operation_id: operationId })
    }
    const restored = await readCloneRecovery(window.localStorage, scope, now)
    expect(restored).toHaveLength(admitted.length)
    expect(restored.every((entry) => entry.operation_id === operationId)).toBe(
      true
    )
    expect(
      new TextEncoder().encode(window.localStorage.getItem(CLONE_RECOVERY_KEY)!)
        .byteLength
    ).toBeLessThanOrEqual(32 * 1024)
  })
})
