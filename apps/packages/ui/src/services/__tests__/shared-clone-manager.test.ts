import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { SharedCloneManager } from "../shared-clone-manager"
import {
  CLONE_RECOVERY_KEY,
  CLONE_RECOVERY_TTL,
  readCloneRecovery,
  writeCloneRecovery
} from "../shared-clone-recovery"
import { TldwApiError } from "../tldw/api-error"
import {
  clonePayload,
  operationId
} from "../tldw/domains/__tests__/shared-workspace-clone.fixture"
import { sharedCloneOperationSchema } from "@/types/shared-workspace-clone"

const scope = "https://server.example|user:42"
const receipt = (status = "queued", shareId = 42) =>
  sharedCloneOperationSchema.parse(clonePayload(status, shareId))
const flush = async () => {
  await vi.advanceTimersByTimeAsync(0)
}
let manager: SharedCloneManager
const api = { clone: vi.fn(), cloneStatus: vi.fn() }
const create = (storage = window.localStorage) => {
  manager = new SharedCloneManager(scope, storage, api, () => undefined)
  manager.resume()
  return manager
}

describe("shared clone operation lifecycle", () => {
  beforeEach(() => {
    vi.useFakeTimers()
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
    api.clone.mockReset().mockResolvedValue(receipt())
    api.cloneStatus.mockReset().mockResolvedValue(receipt("running"))
  })
  afterEach(() => {
    manager?.dispose()
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it.each(["delayed sync", "admission read", "admission write"])(
    "suspends on foreign recovery during %s without volatile dispatch",
    async (phase) => {
      const blocked = vi.fn()
      manager = new SharedCloneManager(
        scope,
        window.localStorage,
        api,
        () => undefined,
        blocked
      )
      await manager.sync()
      let release!: () => void
      let held = navigator.locks.request(
        CLONE_RECOVERY_KEY,
        () =>
          new Promise<void>((done) => {
            release = done
          })
      )
      await Promise.resolve()
      const work =
        phase === "delayed sync"
          ? manager.sync()
          : manager.begin(42, "Research")
      if (phase === "admission write") {
        const first = held
        const releaseFirst = release
        held = navigator.locks.request(
          CLONE_RECOVERY_KEY,
          () =>
            new Promise<void>((done) => {
              release = done
            })
        )
        releaseFirst()
        await first
        await flush()
      }
      const raw = JSON.stringify({
        version: 1,
        scope: "B",
        records: [
          {
            share_id: 43,
            operation_id: operationId,
            expires_at: Date.now() + CLONE_RECOVERY_TTL
          }
        ]
      })
      window.localStorage.setItem(CLONE_RECOVERY_KEY, raw)
      release()
      await held
      await work
      expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBe(raw)
      expect(blocked).toHaveBeenCalledWith("recovery_conflict")
      await manager.begin(44, "Blocked")
      manager.setVisible(true)
      manager.refresh()
      await vi.advanceTimersByTimeAsync(40_000)
      expect(api.clone).not.toHaveBeenCalled()
      expect(api.cloneStatus).not.toHaveBeenCalled()
    }
  )

  it("suspends all requests when a late receipt encounters foreign recovery", async () => {
    const complete: Array<(value: unknown) => void> = []
    api.clone.mockImplementation(
      () => new Promise((done) => complete.push(done))
    )
    const blocked = vi.fn()
    manager = new SharedCloneManager(
      scope,
      window.localStorage,
      api,
      () => undefined,
      blocked
    )
    await manager.begin(42, "First")
    await manager.begin(43, "Second")
    await flush()
    const raw = JSON.stringify({ version: 1, scope: "B", records: [] })
    window.localStorage.setItem(CLONE_RECOVERY_KEY, raw)
    complete[0](receipt())
    await flush()
    expect(blocked).toHaveBeenCalledWith("recovery_conflict")
    expect(api.clone.mock.calls.every((call) => call[3].aborted)).toBe(true)
    complete[1](receipt("succeeded", 43))
    await manager.begin(44, "Blocked")
    manager.refresh()
    await vi.advanceTimersByTimeAsync(40_000)
    expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBe(raw)
    expect(api.clone).toHaveBeenCalledTimes(2)
    expect(api.cloneStatus).not.toHaveBeenCalled()
    expect(
      manager.rows().every((row) => !row.operation && row.recoveryAvailable)
    ).toBe(true)
  })

  it("persists before dispatch, then polls and strips the command at completion", async () => {
    api.clone.mockImplementation(async () => {
      expect(
        (await readCloneRecovery(window.localStorage, scope))[0].idempotency_key
      ).toBeTruthy()
      return receipt()
    })
    create().begin(42, "Research")
    await flush()
    expect(manager.rows()[0].operation?.status).toBe("queued")
    api.cloneStatus.mockResolvedValue(receipt("succeeded"))
    await vi.advanceTimersByTimeAsync(2000)
    expect(manager.rows()[0].operation?.status).toBe("succeeded")
    expect((await readCloneRecovery(window.localStorage, scope))[0]).toEqual({
      share_id: 42,
      operation_id: operationId,
      expires_at: expect.any(Number)
    })
    await vi.advanceTimersByTimeAsync(20_000)
    expect(api.cloneStatus).toHaveBeenCalledTimes(1)
  })

  it("reloads an ambiguous admission using exactly the same key and name", async () => {
    api.clone.mockRejectedValue(new TypeError("offline"))
    create().begin(42, "Research")
    await flush()
    const [, originalName, originalKey] = api.clone.mock.calls[0]
    manager.dispose()
    api.clone.mockResolvedValue(receipt())
    create()
    await flush()
    expect(api.clone.mock.calls[1].slice(0, 3)).toEqual([
      42,
      originalName,
      originalKey
    ])
  })

  it("reloads an accepted pointer with GET, not another POST", async () => {
    await writeCloneRecovery(window.localStorage, scope, {
      share_id: 42,
      operation_id: operationId,
      expires_at: Date.now() + CLONE_RECOVERY_TTL
    })
    create()
    await flush()
    expect(api.clone).not.toHaveBeenCalled()
    expect(api.cloneStatus.mock.calls[0].slice(0, 2)).toEqual([42, operationId])
  })

  it("follows an in-progress conflict's validated operation reference", async () => {
    api.clone.mockRejectedValue(
      new TldwApiError("busy", 409, {
        code: "clone_already_in_progress",
        operation_id: operationId
      })
    )
    create().begin(42, "Research")
    await flush()
    expect(api.cloneStatus).toHaveBeenCalled()
    expect(manager.rows()[0].operation?.status).toBe("running")
  })

  it.each([422])(
    "clears a definitely rejected %s admission without retrying",
    async (status) => {
      api.clone.mockRejectedValue(
        new TldwApiError("private server detail", status, {})
      )
      create().begin(42, "Research")
      await flush()
      expect(await readCloneRecovery(window.localStorage, scope)).toEqual([])
      expect(manager.rows()[0].issue).toBe("rejected")
      await vi.advanceTimersByTimeAsync(20_000)
      expect(api.clone).toHaveBeenCalledTimes(1)
    }
  )

  it.each(["lost response", "timeout with late receipt", "reload"])(
    "retains an ambiguous admission after replay 401 and rotates credentials without a new key: %s",
    async (phase) => {
      let late!: (value: unknown) => void
      const blocked = vi.fn()
      if (phase === "timeout with late receipt") {
        api.clone.mockImplementationOnce(
          () =>
            new Promise((done) => {
              late = done
            })
        )
      } else
        api.clone.mockRejectedValueOnce(new TypeError("Accepted response lost"))
      manager = new SharedCloneManager(
        scope,
        window.localStorage,
        api,
        () => undefined,
        blocked
      )
      manager.begin(42, "Research")
      await flush()
      const original = api.clone.mock.calls[0].slice(0, 3)
      if (phase === "timeout with late receipt")
        await vi.advanceTimersByTimeAsync(30_000)
      api.clone.mockRejectedValueOnce(
        new TldwApiError("Expired token", 401, {})
      )
      if (phase === "reload") {
        manager.dispose()
        manager = new SharedCloneManager(
          scope,
          window.localStorage,
          api,
          () => undefined,
          blocked
        )
        manager.resume()
        await flush()
      } else {
        await vi.advanceTimersByTimeAsync(5000)
      }
      const saved = await readCloneRecovery(window.localStorage, scope)
      expect(saved[0]?.idempotency_key).toBe(original[2])
      expect(manager.rows()[0].issue).not.toBe("rejected")
      await manager.begin(42, "Wrong new attempt")
      manager.setVisible(true)
      manager.refresh()
      await vi.advanceTimersByTimeAsync(40_000)
      expect(api.clone).toHaveBeenCalledTimes(2)
      expect(blocked).toHaveBeenCalledWith("auth_required")
      const rotated = {
        clone: vi.fn().mockResolvedValue(receipt()),
        cloneStatus: vi.fn().mockResolvedValue(receipt("running"))
      }
      manager.setVisible(true)
      await manager.resume(rotated)
      await flush()
      expect(rotated.clone.mock.calls[0].slice(0, 3)).toEqual(original)
      const after = window.localStorage.getItem(CLONE_RECOVERY_KEY)
      if (late) late(receipt("succeeded"))
      await flush()
      expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBe(after)
      expect(manager.rows()[0].operation?.status).toBe("queued")
    }
  )

  it.each([401, 403])(
    "requires verification on first-attempt %s while preserving its key",
    async (status) => {
      const blocked = vi.fn()
      api.clone.mockRejectedValueOnce(
        new TldwApiError("Not authorized", status, {})
      )
      manager = new SharedCloneManager(
        scope,
        window.localStorage,
        api,
        () => undefined,
        blocked
      )
      await manager.begin(42, "Research")
      await flush()
      expect(blocked).toHaveBeenCalledWith("auth_required")
      expect(
        (await readCloneRecovery(window.localStorage, scope))[0]
          ?.idempotency_key
      ).toBe(api.clone.mock.calls[0][2])
    }
  )

  it.each([
    { status: 403, detail: {}, label: "CSRF token validation failed" },
    {
      status: 403,
      detail: { code: "sharing_permission_required" },
      label: "Permission revoked"
    },
    { status: 422, detail: {}, label: "Validation failed" }
  ])(
    "preserves ambiguous admission after $label and reuses its key on verified resume",
    async ({ status, detail, label }) => {
      const blocked = vi.fn()
      api.clone.mockRejectedValueOnce(new TypeError("Accepted response lost"))
      manager = new SharedCloneManager(
        scope,
        window.localStorage,
        api,
        () => undefined,
        blocked
      )
      await manager.begin(42, "Research")
      await flush()
      const originalKey = api.clone.mock.calls[0][2]
      api.clone.mockRejectedValueOnce(new TldwApiError(label, status, detail))
      await vi.advanceTimersByTimeAsync(5000)
      expect(
        (await readCloneRecovery(window.localStorage, scope))[0]
          ?.idempotency_key
      ).toBe(originalKey)
      await manager.begin(42, "Wrong new attempt")
      await vi.advanceTimersByTimeAsync(40_000)
      expect(api.clone).toHaveBeenCalledTimes(2)
      if (status === 403) expect(blocked).toHaveBeenCalledWith("auth_required")
      else expect(blocked).not.toHaveBeenCalled()
      expect(manager.rows()[0].issue).toBe("unavailable")
      manager.suspend()
      const rotated = {
        clone: vi.fn().mockResolvedValue(receipt()),
        cloneStatus: vi.fn()
      }
      manager.setVisible(true)
      await manager.resume(rotated)
      await flush()
      expect(rotated.clone.mock.calls[0].slice(0, 3)).toEqual(
        api.clone.mock.calls[0].slice(0, 3)
      )
      expect(manager.rows()[0].operation?.status).toBe("queued")
    }
  )

  it("keeps a shared key when the first request gets 403 after another manager's ambiguous replay", async () => {
    let rejectFirst!: (error: unknown) => void
    api.clone.mockImplementationOnce(
      () =>
        new Promise((_resolve, reject) => {
          rejectFirst = reject
        })
    )
    api.clone.mockRejectedValueOnce(
      new TypeError("Replay accepted, response lost")
    )
    create().begin(42, "Research")
    await flush()
    const original = api.clone.mock.calls[0].slice(0, 3)
    const other = new SharedCloneManager(
      scope,
      window.localStorage,
      api,
      () => undefined
    )
    try {
      await other.resume()
      await flush()
      expect(api.clone.mock.calls[1].slice(0, 3)).toEqual(original)
      expect(other.rows()[0].issue).toBe("uncertain")
      const raw = window.localStorage.getItem(CLONE_RECOVERY_KEY)
      rejectFirst(
        new TldwApiError("Permission revoked", 403, {
          code: "sharing_permission_required"
        })
      )
      await flush()
      expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBe(raw)
      other.suspend()
      await manager.begin(42, "Wrong new attempt")
      manager.refresh()
      await vi.advanceTimersByTimeAsync(40_000)
      expect(api.clone).toHaveBeenCalledTimes(2)
      const rotated = {
        clone: vi.fn().mockResolvedValue(receipt()),
        cloneStatus: vi.fn()
      }
      manager.setVisible(true)
      await manager.resume(rotated)
      await flush()
      expect(rotated.clone.mock.calls[0].slice(0, 3)).toEqual(original)
    } finally {
      other.dispose()
    }
  })

  it("honors throttling even when focus requests an immediate check", async () => {
    api.clone.mockRejectedValue(
      new TldwApiError("busy", 429, { retry_after_ms: 20_000 })
    )
    create().begin(42, "Research")
    await flush()
    manager.refresh()
    await vi.advanceTimersByTimeAsync(19_999)
    expect(api.clone).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(1)
    expect(api.clone).toHaveBeenCalledTimes(2)
  })

  it("uses a fresh key only after a retryable terminal failure", async () => {
    api.clone.mockResolvedValue(receipt("failed"))
    create().begin(42, "Research")
    await flush()
    const firstKey = api.clone.mock.calls[0][2]
    manager.begin(42, "Research")
    await flush()
    expect(api.clone.mock.calls[1][2]).not.toBe(firstKey)
    api.clone.mockResolvedValue({ ...receipt("failed"), retryable: false })
    manager.begin(42, "Research")
    await flush()
    manager.begin(42, "Research")
    await flush()
    expect(api.clone).toHaveBeenCalledTimes(3)
  })

  it("caps requests at four and pauses while hidden", async () => {
    const resolve: Array<() => void> = []
    api.clone.mockImplementation(
      (shareId: number) =>
        new Promise((done) =>
          resolve.push(() => done(receipt("queued", shareId)))
        )
    )
    create()
    for (let id = 1; id <= 6; id++) manager.begin(id, "Research")
    await flush()
    expect(api.clone).toHaveBeenCalledTimes(4)
    manager.setVisible(false)
    resolve.forEach((done) => done())
    await vi.advanceTimersByTimeAsync(10_000)
    expect(api.clone).toHaveBeenCalledTimes(4)
    manager.setVisible(true)
    await flush()
    expect(api.clone).toHaveBeenCalledTimes(6)
  })

  it("recovers another tab's records without resetting terminal rows", async () => {
    create()
    await writeCloneRecovery(window.localStorage, scope, {
      share_id: 42,
      operation_id: operationId,
      expires_at: Date.now() + CLONE_RECOVERY_TTL
    })
    api.cloneStatus.mockResolvedValue(receipt("succeeded"))
    manager.sync()
    await flush()
    manager.sync()
    await flush()
    expect(api.cloneStatus).toHaveBeenCalledTimes(1)
  })

  it("keeps in-memory progress and exposes unavailable reload recovery", async () => {
    const storage = {
      getItem: () => {
        throw new Error("denied")
      },
      setItem: () => {
        throw new Error("denied")
      },
      removeItem: vi.fn()
    } as unknown as Storage
    create(storage).begin(42, "Research")
    await flush()
    expect(manager.rows()[0]).toMatchObject({
      recoveryAvailable: false,
      operation: { status: "queued" }
    })
  })

  it.each(["sync", "receipt"])(
    "still recognizes foreign recovery after temporary storage failure during %s",
    async (phase) => {
      let unavailable = true
      const storage = {
        getItem: (key: string) => {
          if (unavailable) throw new Error("temporarily denied")
          return window.localStorage.getItem(key)
        },
        setItem: (key: string, value: string) =>
          window.localStorage.setItem(key, value),
        removeItem: (key: string) => window.localStorage.removeItem(key)
      }
      let complete!: (value: unknown) => void
      api.clone.mockImplementationOnce(
        () =>
          new Promise((done) => {
            complete = done
          })
      )
      const blocked = vi.fn()
      manager = new SharedCloneManager(
        scope,
        storage,
        api,
        () => undefined,
        blocked
      )
      await manager.begin(42, "Research")
      await flush()
      unavailable = false
      const raw = JSON.stringify({ version: 1, scope: "B", records: [] })
      window.localStorage.setItem(CLONE_RECOVERY_KEY, raw)
      if (phase === "sync") await manager.sync()
      else complete(receipt())
      await flush()
      expect(blocked).toHaveBeenCalledWith("recovery_conflict")
      await manager.begin(43, "Blocked")
      manager.refresh()
      await vi.advanceTimersByTimeAsync(40_000)
      expect(api.clone).toHaveBeenCalledTimes(1)
      expect(api.cloneStatus).not.toHaveBeenCalled()
      expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBe(raw)
    }
  )

  it("drops late results and cancels network calls when disposed", async () => {
    let resolve!: (value: ReturnType<typeof receipt>) => void
    api.clone.mockImplementation(
      () =>
        new Promise((done) => {
          resolve = done
        })
    )
    create().begin(42, "Research")
    await flush()
    const signal = api.clone.mock.calls[0][3] as AbortSignal
    manager.dispose()
    resolve(receipt("succeeded"))
    await flush()
    expect(signal.aborted).toBe(true)
    expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toContain(
      "idempotency_key"
    )
  })

  it("bounds and normalizes long Unicode names before storing the command", async () => {
    create().begin(42, "  " + "\u{1f4da}".repeat(255) + "  notes ")
    await flush()
    const request = api.clone.mock.calls[0][1]
    expect(Array.from(request.name)).toHaveLength(255)
    expect(request.name).toBe("\u{1f4da}".repeat(248) + " (Copy)")
    expect(request.name).toMatch(/ \(Copy\)$/)
    expect(request.name).not.toMatch(/[\uD800-\uDBFF] \(Copy\)$/)
  })

  it("retains the original key for an in-progress conflict with an invalid receipt ID", async () => {
    api.clone.mockRejectedValue(
      new TldwApiError("busy", 409, {
        code: "clone_already_in_progress",
        operation_id: "invalid"
      })
    )
    create().begin(42, "Research")
    await flush()
    expect(manager.rows()[0].issue).toBe("uncertain")
    expect(
      (await readCloneRecovery(window.localStorage, scope))[0].idempotency_key
    ).toBe(api.clone.mock.calls[0][2])
  })

  it("allows an explicit terminal cleanup check without resuming automatic polling", async () => {
    const failed = receipt("failed")
    if (failed.status === "failed") failed.error.cleanup_state = "pending"
    failed.retryable = false
    api.clone.mockResolvedValue(failed)
    create().begin(42, "Research")
    await flush()
    api.cloneStatus.mockResolvedValue(receipt("failed"))
    manager.refresh(42)
    await flush()
    expect(manager.rows()[0].operation?.retryable).toBe(true)
    await vi.advanceTimersByTimeAsync(20_000)
    expect(api.cloneStatus).toHaveBeenCalledTimes(1)
  })

  it("releases all stalled slots at a deadline and retries with the same command", async () => {
    api.clone.mockImplementation(() => new Promise(() => undefined))
    create()
    for (let id = 1; id <= 5; id++) manager.begin(id, "Research")
    await flush()
    const original = api.clone.mock.calls[0].slice(0, 3)
    const signal = api.clone.mock.calls[0][3] as AbortSignal
    await vi.advanceTimersByTimeAsync(30_000)
    expect(
      manager.rows().find((row) => row.entry.share_id === 1)
    ).toMatchObject({ pending: false, issue: "uncertain" })
    expect(signal.aborted).toBe(true)
    expect(api.clone.mock.calls.some(([id]) => id === 5)).toBe(true)
    await vi.advanceTimersByTimeAsync(5000)
    expect(
      api.clone.mock.calls.filter(([id]) => id === 1)[1].slice(0, 3)
    ).toEqual(original)
  })

  it("expires rows independently of stalled slots and hidden-page polling", async () => {
    for (let id = 1; id <= 5; id++) {
      await writeCloneRecovery(window.localStorage, scope, {
        share_id: id,
        operation_id: operationId,
        expires_at: Date.now() + 1000
      })
    }
    api.cloneStatus.mockImplementation(() => new Promise(() => undefined))
    create()
    await flush()
    manager.setVisible(false)
    await vi.advanceTimersByTimeAsync(1000)
    expect(manager.rows()).toEqual([])
    expect(
      JSON.parse(window.localStorage.getItem(CLONE_RECOVERY_KEY)!).records
    ).toEqual([])
  })

  it("ignores a late success after an abort-ignoring admission times out", async () => {
    let resolve!: (value: ReturnType<typeof receipt>) => void
    api.clone.mockImplementation(
      () =>
        new Promise((done) => {
          resolve = done
        })
    )
    create().begin(42, "Research")
    await flush()
    await vi.advanceTimersByTimeAsync(30_000)
    resolve(receipt("succeeded"))
    await flush()
    expect(manager.rows()[0]).toMatchObject({
      issue: "uncertain",
      pending: false
    })
    expect(manager.rows()[0].operation).toBeUndefined()
    expect(
      (await readCloneRecovery(window.localStorage, scope))[0].idempotency_key
    ).toBeTruthy()
  })

  it.each(["admission", "poll"])(
    "replaces a suspended transport without adopting a late prior %s receipt",
    async (phase) => {
      let resolve!: (value: ReturnType<typeof receipt>) => void
      const oldRequest = new Promise<ReturnType<typeof receipt>>((done) => {
        resolve = done
      })
      if (phase === "admission") api.clone.mockReturnValueOnce(oldRequest)
      else api.cloneStatus.mockReturnValueOnce(oldRequest)
      create().begin(42, "Research")
      await flush()
      if (phase === "poll") await vi.advanceTimersByTimeAsync(2000)
      const entry = { ...manager.rows()[0].entry }
      const previousSignal =
        phase === "admission"
          ? api.clone.mock.calls[0][3]
          : api.cloneStatus.mock.calls[0][2]
      manager.suspend()
      const replacement = {
        clone: vi.fn().mockResolvedValue(receipt("running")),
        cloneStatus: vi.fn().mockResolvedValue(receipt("running"))
      }
      manager.setVisible(true)
      manager.resume(replacement)
      await flush()
      manager.refresh(42)
      await flush()
      expect(previousSignal.aborted).toBe(true)
      const request =
        phase === "admission" ? replacement.clone : replacement.cloneStatus
      expect(request).toHaveBeenCalled()
      if (phase === "admission") {
        expect(request.mock.calls[0].slice(0, 3)).toEqual([
          42,
          { name: entry.name },
          entry.idempotency_key
        ])
      } else {
        expect(request.mock.calls[0].slice(0, 2)).toEqual([
          42,
          entry.operation_id
        ])
        expect(replacement.clone).not.toHaveBeenCalled()
      }
      resolve(receipt("succeeded"))
      await flush()
      expect(manager.rows()[0].operation?.status).toBe("running")
      expect(
        (await readCloneRecovery(window.localStorage, scope))[0].operation_id
      ).toBe(operationId)
    }
  )

  it.each(["admission", "poll"])(
    "suspends a scope-change 412 during %s without deleting recovery or automatically replaying",
    async (phase) => {
      const rejected = phase === "admission" ? api.clone : api.cloneStatus
      rejected.mockRejectedValue(
        new TldwApiError("Account changed", 412, {
          code: "request_config_scope_changed",
          retryable: false,
          recovery_action: "refresh"
        })
      )
      const scopeChanged = vi.fn()
      manager = new SharedCloneManager(
        scope,
        window.localStorage,
        api,
        () => undefined,
        scopeChanged
      )
      await manager.begin(42, "Research")
      await flush()
      if (phase === "poll") await vi.advanceTimersByTimeAsync(2000)
      const original = (await readCloneRecovery(window.localStorage, scope))[0]
      expect(
        phase === "admission"
          ? original?.idempotency_key
          : original?.operation_id
      ).toBeTruthy()
      expect(scopeChanged).toHaveBeenCalledTimes(1)
      manager.refresh(42)
      await manager.begin(42, "Stale retry")
      await vi.advanceTimersByTimeAsync(60_000)
      expect(api.clone).toHaveBeenCalledTimes(1)
      expect(api.cloneStatus).toHaveBeenCalledTimes(phase === "poll" ? 1 : 0)
      expect((await readCloneRecovery(window.localStorage, scope))[0]).toEqual(
        original
      )
    }
  )

  it("adopts a concurrent same-share attempt before submitting instead of overwriting it", async () => {
    create()
    await flush()
    const concurrent = {
      share_id: 42,
      idempotency_key: "concurrent-key-00000001",
      name: "Concurrent (Copy)",
      expires_at: Date.now() + CLONE_RECOVERY_TTL
    }
    const otherTab = writeCloneRecovery(window.localStorage, scope, concurrent)
    manager.begin(42, "Research")
    await otherTab
    await flush()
    expect(api.clone.mock.calls[0].slice(0, 3)).toEqual([
      42,
      { name: "Concurrent (Copy)" },
      concurrent.idempotency_key
    ])
    expect(manager.rows()[0].entry.idempotency_key).toBe(
      concurrent.idempotency_key
    )
  })

  it("does not overwrite a concurrent retry with an old terminal response", async () => {
    let resolve!: (value: ReturnType<typeof receipt>) => void
    api.clone.mockImplementationOnce(
      () =>
        new Promise((done) => {
          resolve = done
        })
    )
    create().begin(42, "Research")
    await flush()
    const retry = {
      ...manager.rows()[0].entry,
      idempotency_key: "concurrent-key-00000002"
    }
    await writeCloneRecovery(window.localStorage, scope, retry)
    resolve(receipt("succeeded"))
    await flush()
    expect(
      (await readCloneRecovery(window.localStorage, scope))[0].idempotency_key
    ).toBe(retry.idempotency_key)
    expect(manager.rows()[0].entry.idempotency_key).toBe(retry.idempotency_key)
    expect(manager.rows()[0].operation?.status).not.toBe("succeeded")
  })

  it("does not persist or dispatch a command queued across disposal", async () => {
    create()
    await flush()
    let release!: () => void
    const held = navigator.locks.request(
      CLONE_RECOVERY_KEY,
      () =>
        new Promise<void>((done) => {
          release = done
        })
    )
    await Promise.resolve()
    manager.begin(42, "Research")
    manager.dispose()
    release()
    await held
    await flush()
    expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBeNull()
    expect(api.clone).not.toHaveBeenCalled()
  })

  it("explicitly marks unsupported browsers volatile without touching recovery storage", async () => {
    vi.stubGlobal("navigator", {})
    create().begin(42, "Research")
    await flush()
    expect(manager.rows()[0]).toMatchObject({
      recoveryAvailable: false,
      operation: { status: "queued" }
    })
    expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBeNull()
  })

  it("arbitrates simultaneous new attempts and terminal retries with the same expiry", async () => {
    const other = new SharedCloneManager(
      scope,
      window.localStorage,
      api,
      () => undefined
    )
    try {
      api.clone.mockResolvedValue(receipt("failed"))
      create()
      await Promise.all([manager.begin(42, "First"), other.begin(42, "Second")])
      await flush()
      const firstKey = api.clone.mock.calls[0][2]
      expect(new Set(api.clone.mock.calls.map((call) => call[2]))).toEqual(
        new Set([firstKey])
      )
      api.clone.mockImplementation(() => new Promise(() => undefined))
      api.clone.mockClear()
      await Promise.all([
        manager.begin(42, "First retry"),
        other.begin(42, "Second retry")
      ])
      await flush()
      const retryKey = (await readCloneRecovery(window.localStorage, scope))[0]
        .idempotency_key
      expect(retryKey).not.toBe(firstKey)
      expect(api.clone).toHaveBeenCalled()
      expect(new Set(api.clone.mock.calls.map((call) => call[2]))).toEqual(
        new Set([retryKey])
      )
      expect(manager.rows()[0].entry.idempotency_key).toBe(retryKey)
      expect(other.rows()[0].entry.idempotency_key).toBe(retryKey)
    } finally {
      other.dispose()
    }
  })

  it("cancels an actual pre-submit write waiting behind the lock on suspension", async () => {
    create()
    await flush()
    let releaseFirst!: () => void
    let releaseWrite!: () => void
    const first = navigator.locks.request(
      CLONE_RECOVERY_KEY,
      () =>
        new Promise<void>((done) => {
          releaseFirst = done
        })
    )
    await Promise.resolve()
    const begin = manager.begin(42, "Research")
    const held = navigator.locks.request(
      CLONE_RECOVERY_KEY,
      () =>
        new Promise<void>((done) => {
          releaseWrite = done
        })
    )
    releaseFirst()
    await first
    await flush()
    manager.suspend()
    releaseWrite()
    await held
    await begin
    expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBeNull()
    expect(api.clone).not.toHaveBeenCalled()
  })

  it("releases stalled status reads without losing their operation pointers", async () => {
    for (let id = 1; id <= 5; id++) {
      await writeCloneRecovery(window.localStorage, scope, {
        share_id: id,
        operation_id: operationId,
        expires_at: Date.now() + CLONE_RECOVERY_TTL
      })
    }
    api.cloneStatus.mockImplementation(() => new Promise(() => undefined))
    create()
    await flush()
    await vi.advanceTimersByTimeAsync(30_000)
    expect(manager.rows()[0]).toMatchObject({
      pending: false,
      issue: "uncertain",
      entry: { operation_id: operationId }
    })
    expect(api.cloneStatus.mock.calls.some(([id]) => id === 5)).toBe(true)
    await vi.advanceTimersByTimeAsync(5000)
    expect(
      api.cloneStatus.mock.calls.filter(([id]) => id === 1)[1].slice(0, 2)
    ).toEqual([1, operationId])
    expect(api.clone).not.toHaveBeenCalled()
  })

  it("falls back to explicit volatile admission when the pre-submit lock cannot be acquired", async () => {
    create()
    await flush()
    let releaseFirst!: () => void
    let releaseWrite!: () => void
    const first = navigator.locks.request(
      CLONE_RECOVERY_KEY,
      () =>
        new Promise<void>((done) => {
          releaseFirst = done
        })
    )
    await Promise.resolve()
    const begin = manager.begin(42, "Research")
    const held = navigator.locks.request(
      CLONE_RECOVERY_KEY,
      () =>
        new Promise<void>((done) => {
          releaseWrite = done
        })
    )
    releaseFirst()
    await first
    await flush()
    await vi.advanceTimersByTimeAsync(5000)
    const atDeadline = manager.rows()
    releaseWrite()
    await held
    await begin
    await flush()
    expect(atDeadline[0]).toMatchObject({
      recoveryAvailable: false,
      pending: true
    })
    // Receipt acceptance waits for the scope recheck, even in volatile mode.
    expect(manager.rows()[0]).toMatchObject({
      recoveryAvailable: false,
      operation: { status: "queued" }
    })
    expect(window.localStorage.getItem(CLONE_RECOVERY_KEY)).toBeNull()
  })

  it("retains accepted receipts and keeps polling when a legacy Unicode recovery map has no pointer capacity", async () => {
    const records = Array.from({ length: 29 }, (_, index) => ({
      share_id: index + 1,
      expires_at: Date.now() + CLONE_RECOVERY_TTL,
      idempotency_key: `legacy-key-${String(index).padStart(25, "0")}`,
      name: "\u{1f4da}".repeat(248) + " (Copy)"
    }))
    const raw = JSON.stringify({ version: 1, scope, records })
    expect(new TextEncoder().encode(raw).byteLength).toBeLessThanOrEqual(
      32 * 1024
    )
    window.localStorage.setItem(CLONE_RECOVERY_KEY, raw)
    api.clone.mockImplementation(async (id: number) => receipt("queued", id))
    api.cloneStatus.mockImplementation(async (id: number) =>
      receipt("succeeded", id)
    )
    create()
    await flush()
    expect(manager.rows()).toHaveLength(29)
    expect(
      manager
        .rows()
        .every(
          (row) => row.operation?.status === "queued" && row.issue === undefined
        )
    ).toBe(true)
    expect(manager.rows().some((row) => !row.recoveryAvailable)).toBe(true)
    await manager.sync()
    await flush()
    await vi.advanceTimersByTimeAsync(2000)
    expect(
      manager.rows().every((row) => row.operation?.status === "succeeded")
    ).toBe(true)
    expect(api.cloneStatus).toHaveBeenCalledTimes(29)
    expect(api.clone).toHaveBeenCalledTimes(29)
    expect(manager.rows().some((row) => !row.recoveryAvailable)).toBe(true)
  })
})
